"""Analytical retrieval strategy — decomposes complex queries."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.core.shared import SharedResources
from app.prompts import load_prompt

from .base import BaseRetrievalStrategy

logger = logging.getLogger(__name__)


class SubQueries(BaseModel):
    """Structured output for sub-query decomposition."""

    sub_queries: list[str] = Field(
        description="List of sub-queries for comprehensive analysis",
    )


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    """Break a complex query into sub-queries, search each, then rerank."""

    def __init__(self, resources: SharedResources) -> None:
        super().__init__(resources)
        parser = PydanticOutputParser(pydantic_object=SubQueries)
        prompt = PromptTemplate(
            input_variables=["query", "k"],
            template=load_prompt("analytical_decompose"),
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )
        self._decompose_chain = prompt | self.fast_llm | parser

    def retrieve(
        self,
        query: str,
        k: int = 5,
        user_context: Optional[str] = None,
    ) -> list[Document]:
        logger.info("[Analytical] Breaking down query: %s", query)

        try:
            sub_queries = self._decompose_chain.invoke(
                {"query": query, "k": 3},
            ).sub_queries
            logger.info("Sub-queries: %s", sub_queries)
        except Exception:
            logger.warning(
                "Failed to generate sub-queries, using original.",
                exc_info=True,
            )
            sub_queries = [query]

        all_docs: list[Document] = []
        for sq in sub_queries:
            all_docs.extend(self.vector_store.similarity_search(sq, k=5))

        all_docs.extend(self.vector_store.similarity_search(query, k=10))
        logger.debug("Total docs before dedup: %d", len(all_docs))

        unique_docs = list(
            {doc.page_content: doc for doc in all_docs}.values(),
        )
        logger.debug("After dedup: %d unique docs", len(unique_docs))

        if not unique_docs:
            return []

        reranked = self.reranker.compress_documents(
            documents=unique_docs, query=query,
        )
        return reranked[:k]
