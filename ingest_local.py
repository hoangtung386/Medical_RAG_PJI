"""Ingestion Pipeline using PyPDFLoader (local, mien phi, khong can API).

Doc PDF bang thu vien local, khong can Gemini hay HuggingFace.
Phu hop khi het quota hoac can chay nhanh.

Usage:
    uv run python3 ingest_local.py
"""

import glob
import logging
import os

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.getLogger(
    "langchain_milvus.vectorstores.milvus"
).setLevel(logging.ERROR)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COLLECTION_NAME = "medical_rag_docs"


def ingest_documents():
    print("Starting document ingestion with PyPDFLoader (local)...")

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
        print("No PDF files found in data/ folder. Exiting.")
        return

    cohere_api_key = os.getenv("COHERE_API_KEY")
    zilliz_uri = os.getenv("ZILLIZ_URI")
    zilliz_api_key = os.getenv("ZILLIZ_API_KEY")

    if not all([cohere_api_key, zilliz_uri, zilliz_api_key]):
        print("Error: Missing COHERE or ZILLIZ API keys in .env file.")
        return

    # [1/4] Parse PDFs locally
    documents = []
    failed_files = []
    print(f"\n[1/4] Parsing {len(pdf_files)} PDFs with PyPDFLoader (local)...")

    for i, file_path in enumerate(pdf_files, 1):
        filename = os.path.basename(file_path)
        print(f"  [{i}/{len(pdf_files)}] {filename}", end="")

        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            if pages:
                # Gop tat ca trang thanh 1 document, giu metadata
                full_text = "\n\n".join(p.page_content for p in pages if p.page_content.strip())
                if full_text.strip():
                    doc = Document(
                        page_content=full_text,
                        metadata={"source": filename},
                    )
                    documents.append(doc)
                    print(f" -> OK ({len(full_text)} chars, {len(pages)} pages)")
                else:
                    print(" -> Empty (scanned PDF?), skipped")
                    failed_files.append(filename)
            else:
                print(" -> No pages extracted, skipped")
                failed_files.append(filename)

        except Exception as e:
            print(f" -> Error: {e}")
            failed_files.append(filename)

    print(f"\nParsed {len(documents)}/{len(pdf_files)} documents.")
    if failed_files:
        print(f"Failed/empty ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")

    if not documents:
        print("No documents were successfully parsed. Exiting.")
        return

    # [2/4] Chunk
    print("\n[2/4] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # [3/4] Embed
    print("\n[3/4] Initializing Cohere embed-multilingual-v3.0...")
    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-multilingual-v3.0",
    )

    # [4/4] Insert into Zilliz
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    print(
        f"\n[4/4] Inserting {len(chunks)} chunks into Zilliz Cloud "
        f"(Collection: {COLLECTION_NAME})..."
    )
    print(f"  {total_batches} batches of {batch_size}...")

    vector_db = None
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches} ({start}-{end})...")

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
        f"\nIngestion Complete! {len(chunks)} chunks indexed into Zilliz."
    )
    print("Run: uv run python3 api.py")


if __name__ == "__main__":
    ingest_documents()
