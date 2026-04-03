"""Factual retrieval strategy — enhances query for precise fact lookup."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.core.shared import SharedResources
from app.prompts import load_prompt

from .base import BaseRetrievalStrategy

logger = logging.getLogger(__name__)


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    """Translate & enhance query to English, dual-search, then rerank."""

    def __init__(self, resources: SharedResources) -> None:
        super().__init__(resources)
        prompt = PromptTemplate(
            input_variables=["query"],
            template=load_prompt("factual_enhance"),
        )
        self._enhance_chain = prompt | self.llm

    def retrieve(
        self,
        query: str,
        k: int = 4,
        user_context: Optional[str] = None,
    ) -> list[Document]:
        logger.info("[Factual] Retrieving facts for: %s", query)

        enhanced_query = self._enhance_chain.invoke(
            {"query": query},
        ).content.strip()
        logger.info("Enhanced query: %s", enhanced_query)

        unique_docs = self._dual_search(query, enhanced_query)

        if not unique_docs:
            return []

        reranked = self.reranker.compress_documents(
            documents=unique_docs, query=query,
        )
        return reranked[:k]
