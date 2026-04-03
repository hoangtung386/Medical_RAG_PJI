"""Ingestion Pipeline using Google Gemini Flash API for PDF parsing.

Gemini Flash is free (generous rate limits) and excellent at
understanding complex PDF layouts, tables, and medical documents.

Usage:
    1. Add GEMINI_API_KEY to .env file
    2. Run: uv run python3 ingest_gemini.py
"""

import glob
import logging
import os
import re
import shutil
import tempfile
import time

from dotenv import load_dotenv
from google import genai
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.getLogger(
    "langchain_milvus.vectorstores.milvus"
).setLevel(logging.ERROR)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COLLECTION_NAME = "medical_rag_docs"
GEMINI_MODEL = "gemini-2.0-flash"

# Rate limit: chờ giữa mỗi file (giây)
DELAY_BETWEEN_FILES = 6
# Số lần retry tối đa khi bị rate limit
MAX_RETRIES = 3


def _safe_copy_for_upload(file_path: str) -> str:
    """Copy file sang tên ASCII tạm nếu tên gốc có ký tự Unicode.

    Gemini upload API không xử lý được filename non-ASCII.
    Trả về đường dẫn file tạm (cần xóa sau khi dùng).
    """
    filename = os.path.basename(file_path)
    try:
        filename.encode("ascii")
        return file_path  # Tên đã là ASCII, không cần copy
    except UnicodeEncodeError:
        pass

    # Tạo tên ASCII tạm giữ nguyên extension
    ext = os.path.splitext(filename)[1]
    tmp_dir = tempfile.mkdtemp(prefix="gemini_upload_")
    safe_name = f"doc_{abs(hash(filename)) % 10**8}{ext}"
    tmp_path = os.path.join(tmp_dir, safe_name)
    shutil.copy2(file_path, tmp_path)
    return tmp_path


def parse_pdf_with_gemini(client, file_path: str) -> str:
    """Upload PDF to Gemini and extract text content.

    Gemini Flash understands tables, charts, and complex layouts.
    Tự động copy file sang tên ASCII nếu tên gốc có tiếng Việt.
    """
    upload_path = _safe_copy_for_upload(file_path)
    tmp_created = upload_path != file_path

    try:
        uploaded_file = client.files.upload(file=upload_path)

        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(
                name=uploaded_file.name,
            )

        if uploaded_file.state.name == "FAILED":
            raise Exception(
                f"File processing failed: {uploaded_file.state.name}"
            )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                uploaded_file,
                "Extract ALL text content from this PDF document. "
                "Preserve the document structure including headings, "
                "paragraphs, tables, and lists. "
                "For tables, convert them to a readable text format. "
                "Output the full text content in the original "
                "language of the document. "
                "Do NOT summarize - extract the complete text.",
            ],
        )

        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        return response.text

    finally:
        # Xóa file tạm nếu đã copy
        if tmp_created:
            try:
                os.remove(upload_path)
                os.rmdir(os.path.dirname(upload_path))
            except Exception:
                pass


def _parse_retry_delay(error_msg: str) -> int:
    """Trích xuất thời gian chờ từ error message của Gemini 429."""
    match = re.search(r"retry in (\d+)", error_msg, re.IGNORECASE)
    if match:
        return int(match.group(1)) + 5  # Thêm 5s buffer
    return 60  # Mặc định chờ 60s


def parse_with_retry(client, file_path: str, max_retries: int = MAX_RETRIES) -> str:
    """Parse PDF với auto-retry khi bị rate limit (429)."""
    for attempt in range(1, max_retries + 1):
        try:
            return parse_pdf_with_gemini(client, file_path)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = _parse_retry_delay(error_msg)
                print(
                    f"    -> Rate limited (attempt {attempt}/{max_retries}). "
                    f"Cho {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise  # Lỗi khác, không retry
    raise Exception(f"Van bi rate limit sau {max_retries} lan thu lai")


def ingest_documents():
    print(
        "Starting document ingestion with "
        "Gemini Flash API..."
    )

    pdf_files = glob.glob(
        os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True,
    )
    pdf_files.extend(
        glob.glob(
            os.path.join(DATA_DIR, "**", "*.PDF"),
            recursive=True,
        )
    )
    pdf_files = list(set(pdf_files))

    print(f"Found {len(pdf_files)} PDF files.")
    if not pdf_files:
        print("No PDF files found. Exiting.")
        return

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    zilliz_uri = os.getenv("ZILLIZ_URI")
    zilliz_api_key = os.getenv("ZILLIZ_API_KEY")

    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        print(
            "Get your free API key at: "
            "https://aistudio.google.com/apikey"
        )
        return
    if not all([cohere_api_key, zilliz_uri, zilliz_api_key]):
        print(
            "Error: Missing COHERE or ZILLIZ API keys "
            "in .env file."
        )
        return

    client = genai.Client(api_key=gemini_api_key)

    documents = []
    failed_files = []
    print(
        f"\n[1/4] Parsing PDFs with Gemini Flash "
        f"({GEMINI_MODEL})..."
    )

    for i, file_path in enumerate(pdf_files, 1):
        filename = os.path.basename(file_path)
        print(f"  [{i}/{len(pdf_files)}] Parsing: {filename}")

        try:
            text = parse_with_retry(client, file_path)

            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": filename},
                )
                documents.append(doc)
                print(f"    -> OK ({len(text)} chars)")
            else:
                print("    -> Empty result, skipped")
                failed_files.append(filename)

            if i < len(pdf_files):
                time.sleep(DELAY_BETWEEN_FILES)

        except Exception as e:
            print(f"    -> Error: {e}")
            failed_files.append(filename)
            time.sleep(DELAY_BETWEEN_FILES)

    print(f"\nSuccessfully parsed {len(documents)}/{len(pdf_files)} documents.")
    if failed_files:
        print(f"Failed ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")

    if not documents:
        print("No documents were successfully parsed. Exiting.")
        return

    print("\n[2/4] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print(
        "\n[3/4] Initializing Cohere "
        "embed-multilingual-v3.0 model..."
    )
    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-multilingual-v3.0",
    )

    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    print(
        f"\n[4/4] Inserting {len(chunks)} chunks into Zilliz "
        f"Cloud (Collection: {COLLECTION_NAME})..."
    )
    print(
        f"  Processing in {total_batches} batches "
        f"of {batch_size}..."
    )

    vector_db = None
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        print(
            f"  Batch {batch_idx + 1}/{total_batches} "
            f"({start}-{end})..."
        )

        try:
            if batch_idx == 0:
                vector_db = Milvus.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    connection_args={
                        "uri": zilliz_uri,
                        "token": zilliz_api_key,
                        "secure": True,
                    },
                    collection_name=COLLECTION_NAME,
                    drop_old=True,
                )
            else:
                vector_db.add_documents(batch)
        except Exception as e:
            print(f"    Error in batch {batch_idx + 1}: {e}")
            continue

    print(
        f"\nIngestion Complete! {len(chunks)} chunks "
        "indexed into Zilliz."
    )
    print("Data is ready for retrieval.")
    print("Run: uv run python3 api.py")


if __name__ == "__main__":
    ingest_documents()
