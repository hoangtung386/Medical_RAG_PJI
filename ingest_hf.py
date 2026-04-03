"""Ingestion Pipeline using HuggingFace Vision Model for PDF OCR.

Converts each PDF page to an image, then uses a free HuggingFace
vision model to extract text (including tables, charts, figures).

This handles complex layouts much better than PyPDFLoader because
the model "sees" the page visually, just like a human would.

Usage:
    1. Add HF_TOKEN to .env
       (get free at https://huggingface.co/settings/tokens)
    2. Run: python ingest_hf.py

Free tier: ~1000 requests/day on HF Inference API
"""

import glob
import io
import logging
import os
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path

load_dotenv()

logging.getLogger(
    "langchain_milvus.vectorstores.milvus"
).setLevel(logging.ERROR)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COLLECTION_NAME = "medical_rag_docs"

HF_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

MAX_PAGES_PER_PDF = 0

PDF_DPI = 150


def image_to_bytes(image) -> bytes:
    """Convert PIL Image to bytes for HF API."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def extract_text_from_page(
    client: InferenceClient, image,
) -> str:
    """Send a page image to HuggingFace Vision model.

    The model "sees" the page visually, so it can understand
    tables, charts, and complex layouts.
    """
    import base64

    image_bytes = image_to_bytes(image)
    encoded = base64.b64encode(image_bytes).decode()

    response = client.chat_completion(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                f"{encoded}"
                            ),
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text content from this "
                            "document page. Preserve the "
                            "structure: headings, paragraphs, "
                            "bullet points, and numbered lists. "
                            "For tables, convert them to a "
                            "readable text format with columns "
                            "separated by ' | '. "
                            "For figures/charts, describe the "
                            "key data points. "
                            "Output the complete text in the "
                            "original language. Do NOT summarize."
                        ),
                    },
                ],
            }
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def parse_pdf_with_hf(
    client: InferenceClient, file_path: str,
) -> str:
    """Parse a PDF by converting pages to images and OCR."""
    try:
        images = convert_from_path(file_path, dpi=PDF_DPI)
    except Exception as e:
        print(f"    Error converting PDF to images: {e}")
        return ""

    if MAX_PAGES_PER_PDF > 0:
        images = images[:MAX_PAGES_PER_PDF]

    all_text = []
    total_pages = len(images)

    for page_num, image in enumerate(images, 1):
        try:
            text = extract_text_from_page(client, image)
            if text and text.strip():
                all_text.append(
                    f"--- Page {page_num} ---\n{text}"
                )

            if page_num < total_pages:
                time.sleep(2)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                print(
                    f"    Rate limited at page "
                    f"{page_num}/{total_pages}. Waiting 30s..."
                )
                time.sleep(30)
                try:
                    text = extract_text_from_page(client, image)
                    if text and text.strip():
                        all_text.append(
                            f"--- Page {page_num} ---\n{text}"
                        )
                except Exception:
                    print(
                        f"    Skipping page {page_num} "
                        "after retry"
                    )
            else:
                print(
                    f"    Error on page {page_num}: "
                    f"{error_msg[:100]}"
                )

    return "\n\n".join(all_text)


def ingest_documents():
    print(
        "Starting document ingestion with "
        "HuggingFace Vision OCR..."
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

    hf_token = os.getenv("HF_TOKEN")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    zilliz_uri = os.getenv("ZILLIZ_URI")
    zilliz_api_key = os.getenv("ZILLIZ_API_KEY")

    if not hf_token:
        print("Error: HF_TOKEN not found in .env file.")
        print(
            "Get your free token at: "
            "https://huggingface.co/settings/tokens"
        )
        return
    if not all([cohere_api_key, zilliz_uri, zilliz_api_key]):
        print(
            "Error: Missing COHERE or ZILLIZ API keys "
            "in .env file."
        )
        return

    client = InferenceClient(token=hf_token)
    print(f"Using HuggingFace model: {HF_MODEL}")

    documents = []
    print(
        "\n[1/4] Parsing PDFs with "
        "HuggingFace Vision OCR..."
    )

    for i, file_path in enumerate(pdf_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n  [{i}/{len(pdf_files)}] Parsing: {filename}")

        text = parse_pdf_with_hf(client, file_path)

        if text and text.strip():
            doc = Document(
                page_content=text,
                metadata={"source": file_path},
            )
            documents.append(doc)
            print(f"    -> OK ({len(text)} chars extracted)")
        else:
            print("    -> No text extracted, skipped")

    print(f"\nSuccessfully parsed {len(documents)} documents.")

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
    print("Data is ready for retrieval. Run: python api.py")


if __name__ == "__main__":
    ingest_documents()
