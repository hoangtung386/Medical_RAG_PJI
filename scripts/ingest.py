"""Ingestion pipeline — PDF to Zilliz Cloud vector DB.

Reads PDFs locally with PyPDFLoader, chunks them, embeds with Cohere,
and inserts into Zilliz Cloud.

Usage:
    uv run python -m scripts.ingest
"""

import glob
import logging
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

from app.config import settings  # noqa: E402
from app.llm.providers import get_cohere_embeddings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


def _find_pdf_files() -> list[str]:
    """Return deduplicated list of PDF paths under ``DATA_DIR``."""
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    files.extend(glob.glob(os.path.join(DATA_DIR, "**", "*.PDF"), recursive=True))
    return list(set(files))


def _parse_pdfs(pdf_files: list[str]) -> list[Document]:
    """Parse PDFs into single-document-per-file representation."""
    documents: list[Document] = []
    failed_files: list[str] = []

    logger.info("[1/4] Parsing %d PDFs with PyPDFLoader...", len(pdf_files))

    for i, file_path in enumerate(pdf_files, 1):
        filename = os.path.basename(file_path)
        logger.info("  [%d/%d] %s", i, len(pdf_files), filename)

        try:
            pages = PyPDFLoader(file_path).load()
            if pages:
                full_text = "\n\n".join(
                    p.page_content for p in pages if p.page_content.strip()
                )
                if full_text.strip():
                    documents.append(
                        Document(
                            page_content=full_text,
                            metadata={"source": filename},
                        ),
                    )
                    logger.info(
                        "    -> OK (%d chars, %d pages)",
                        len(full_text),
                        len(pages),
                    )
                else:
                    logger.warning("    -> Empty (scanned PDF?), skipped")
                    failed_files.append(filename)
            else:
                logger.warning("    -> No pages extracted, skipped")
                failed_files.append(filename)
        except Exception:
            logger.error("    -> Error processing %s", filename, exc_info=True)
            failed_files.append(filename)

    logger.info("Parsed %d/%d documents.", len(documents), len(pdf_files))
    if failed_files:
        logger.warning("Failed/empty (%d): %s", len(failed_files), failed_files)

    return documents


def ingest_documents() -> None:
    """Run the full ingestion pipeline."""
    logger.info("Starting document ingestion...")

    pdf_files = _find_pdf_files()
    logger.info("Found %d PDF files.", len(pdf_files))
    if not pdf_files:
        logger.error("No PDF files found in %s. Exiting.", DATA_DIR)
        return

    if not all([settings.cohere_api_key, settings.zilliz_uri, settings.zilliz_api_key]):
        logger.error("Missing COHERE or ZILLIZ API keys in .env file.")
        return

    # [1/4] Parse
    documents = _parse_pdfs(pdf_files)
    if not documents:
        logger.error("No documents were successfully parsed. Exiting.")
        return

    # [2/4] Chunk
    logger.info("[2/4] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Created %d text chunks.", len(chunks))

    # [3/4] Embed
    logger.info("[3/4] Initializing Cohere embeddings...")
    embeddings = get_cohere_embeddings()

    # [4/4] Insert into Zilliz
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    logger.info(
        "[4/4] Inserting %d chunks into Zilliz Cloud "
        "(collection: %s) — %d batches...",
        len(chunks),
        settings.collection_name,
        total_batches,
    )

    vector_db = None
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        logger.info("  Batch %d/%d (%d-%d)...", batch_idx + 1, total_batches, start, end)

        try:
            if batch_idx == 0:
                vector_db = Milvus.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    connection_args={
                        "uri": settings.zilliz_uri,
                        "token": settings.zilliz_api_key,
                        "secure": True,
                    },
                    collection_name=settings.collection_name,
                    drop_old=True,
                )
            else:
                vector_db.add_documents(batch)
        except Exception:
            logger.error("Error in batch %d", batch_idx + 1, exc_info=True)
            continue

    logger.info(
        "Ingestion complete! %d chunks indexed into Zilliz.", len(chunks),
    )
    logger.info("Run: uv run python -m app.main")


if __name__ == "__main__":
    ingest_documents()
