"""Opinion retrieval strategy — gathers diverse medical viewpoints."""

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


class SelectedIndices(BaseModel):
    """Structured output for diversity selection."""

    indices: list[int] = Field(
        description="Indices of selected diverse documents",
    )


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    """Identify viewpoints, search each, then select diverse subset."""

    def __init__(self, resources: SharedResources) -> None:
        super().__init__(resources)

        # Viewpoint identification chain
        vp_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template=load_prompt("opinion_viewpoints"),
        )
        self._viewpoint_chain = vp_prompt | self.llm

        # Diversity selection chain
        parser = PydanticOutputParser(pydantic_object=SelectedIndices)
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=load_prompt("opinion_diversity"),
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )
        self._diversity_chain = diversity_prompt | self.fast_llm | parser

    def retrieve(
        self,
        query: str,
        k: int = 4,
        user_context: Optional[str] = None,
    ) -> list[Document]:
        logger.info("[Opinion] Looking for diverse perspectives on: %s", query)

        # Step 1 — identify viewpoints
        viewpoints_text = self._viewpoint_chain.invoke(
            {"query": query, "k": 3},
        ).content
        viewpoints = [v.strip() for v in viewpoints_text.split("\n") if v.strip()]
        logger.info("Identified viewpoints: %s", viewpoints)

        # Step 2 — search per viewpoint + original
        all_docs: list[Document] = []
        for vp in viewpoints:
            all_docs.extend(self.vector_store.similarity_search(vp, k=3))

        all_docs.extend(self.vector_store.similarity_search(query, k=10))

        unique_docs = list(
            {doc.page_content: doc for doc in all_docs}.values(),
        )
        logger.debug("%d unique docs after dedup", len(unique_docs))

        if not unique_docs:
            return []

        # Step 3 — diversity selection via LLM
        docs_text = "\n".join(
            f"[{i}]: {doc.page_content[:200]}..."
            for i, doc in enumerate(unique_docs)
        )

        try:
            selected_indices = self._diversity_chain.invoke(
                {"query": query, "docs": docs_text, "k": k},
            ).indices
            logger.info("Selected diverse indices: %s", selected_indices)
            return [
                unique_docs[i]
                for i in selected_indices
                if i < len(unique_docs)
            ]
        except Exception:
            logger.warning(
                "Diversity selection failed, falling back to reranker.",
                exc_info=True,
            )
            return self.reranker.compress_documents(
                documents=unique_docs, query=query,
            )[:k]
