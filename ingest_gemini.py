"""Ingestion Pipeline using Google Gemini Flash API for PDF parsing.

Gemini Flash is free (generous rate limits) and excellent at
understanding complex PDF layouts, tables, and medical documents.

Usage:
    1. Add GEMINI_API_KEY to .env file
    2. Run: python ingest_gemini.py
"""

import glob
import logging
import os
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

DATA_DIR = "/home/sotatek/Downloads/RAG_medical/Data"
COLLECTION_NAME = "medical_rag_docs"
GEMINI_MODEL = "gemini-2.0-flash"


def parse_pdf_with_gemini(client, file_path: str) -> str:
    """Upload PDF to Gemini and extract text content.

    Gemini Flash understands tables, charts, and complex layouts.
    """
    uploaded_file = client.files.upload(file=file_path)

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
    print(
        f"\n[1/4] Parsing PDFs with Gemini Flash "
        f"({GEMINI_MODEL})..."
    )

    for i, file_path in enumerate(pdf_files, 1):
        filename = os.path.basename(file_path)
        print(f"  [{i}/{len(pdf_files)}] Parsing: {filename}")

        try:
            text = parse_pdf_with_gemini(client, file_path)

            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path},
                )
                documents.append(doc)
                print(f"    -> OK ({len(text)} chars)")
            else:
                print("    -> Empty result, skipped")

            if i < len(pdf_files):
                time.sleep(4)

        except Exception as e:
            print(f"    -> Error: {e}")
            time.sleep(5)

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
    print("Data is ready for retrieval. Run: python app.py")


if __name__ == "__main__":
    ingest_documents()
