"""Base retrieval strategy with shared resources."""

import logging
from typing import Optional

from langchain_core.documents import Document

from app.core.shared import SharedResources

logger = logging.getLogger(__name__)


class BaseRetrievalStrategy:
    """Base class for all retrieval strategies.

    Subclasses override :meth:`retrieve` to implement their own
    query-expansion / search logic while reusing the shared vector
    store, reranker, and LLM instances.
    """

    def __init__(self, resources: SharedResources) -> None:
        self.llm = resources.llm
        self.fast_llm = resources.fast_llm
        self.vector_store = resources.vector_store
        self.reranker = resources.reranker
        self.compression_retriever = resources.compression_retriever
        self.cfg = resources.cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dual_search(
        self,
        original_query: str,
        enhanced_query: str,
        k_each: int | None = None,
    ) -> list[Document]:
        """Search with both original and enhanced queries, then deduplicate."""
        k_each = k_each or self.cfg.dual_search_k

        docs_original = self.vector_store.similarity_search(
            original_query, k=k_each,
        )
        logger.debug(
            "Original query returned %d docs", len(docs_original),
        )

        docs_enhanced = self.vector_store.similarity_search(
            enhanced_query, k=k_each,
        )
        logger.debug(
            "Enhanced query returned %d docs", len(docs_enhanced),
        )

        all_docs = docs_original + docs_enhanced
        unique = list({doc.page_content: doc for doc in all_docs}.values())
        logger.debug("Merged & deduplicated: %d unique docs", len(unique))
        return unique

    # ------------------------------------------------------------------
    # Default retrieve (fallback)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 4,
        user_context: Optional[str] = None,
    ) -> list[Document]:
        """Fallback retrieval using compression retriever."""
        self.reranker.top_n = k
        return self.compression_retriever.invoke(query)
