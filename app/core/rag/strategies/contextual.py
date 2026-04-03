"""Contextual retrieval strategy — reformulates query with patient context."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.core.shared import SharedResources
from app.prompts import load_prompt

from .base import BaseRetrievalStrategy

logger = logging.getLogger(__name__)


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    """Reformulate the query using patient context, then dual-search."""

    def __init__(self, resources: SharedResources) -> None:
        super().__init__(resources)
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=load_prompt("contextual_reformulate"),
        )
        self._contextualize_chain = prompt | self.llm

    def retrieve(
        self,
        query: str,
        k: int = 4,
        user_context: Optional[str] = None,
    ) -> list[Document]:
        logger.info("[Contextual] Contextualizing query...")

        if not user_context:
            user_context = "No specific patient context provided."

        contextualized_query = self._contextualize_chain.invoke(
            {"query": query, "context": user_context},
        ).content.strip()
        logger.info("Contextualized query (EN): %s", contextualized_query)

        unique_docs = self._dual_search(query, contextualized_query)

        if not unique_docs:
            return []

        reranked = self.reranker.compress_documents(
            documents=unique_docs, query=contextualized_query,
        )
        return reranked[:k]
