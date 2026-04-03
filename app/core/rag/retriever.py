"""Adaptive retriever and RAG engine.

``AdaptiveRetriever`` classifies a query and dispatches to the
appropriate retrieval strategy.  ``AdaptiveRAG`` wraps the retriever
with an LLM answer-generation step.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from tavily import TavilyClient

from app.core.rag.classifier import QueryClassifier
from app.core.rag.strategies import (
    AnalyticalRetrievalStrategy,
    ContextualRetrievalStrategy,
    FactualRetrievalStrategy,
    OpinionRetrievalStrategy,
)
from app.core.shared import SharedResources
from app.prompts import load_prompt

logger = logging.getLogger(__name__)


class AdaptiveRetriever:
    """Select a retrieval strategy based on query intent."""

    def __init__(self, resources: SharedResources) -> None:
        self.classifier = QueryClassifier(resources)

        self.strategies = {
            "Factual": FactualRetrievalStrategy(resources),
            "Analytical": AnalyticalRetrievalStrategy(resources),
            "Opinion": OpinionRetrievalStrategy(resources),
            "Contextual": ContextualRetrievalStrategy(resources),
        }

        tavily_key = resources.cfg.tavily_api_key
        self.tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

    def get_relevant_documents(
        self,
        query: str,
        user_context: Optional[str] = None,
        use_web_search: bool = True,
    ) -> tuple[list[Document], str]:
        """Retrieve documents using the best-matched strategy.

        Returns:
            A tuple of ``(documents, category)``.
        """
        category = self.classifier.classify(query)
        logger.info("Strategy selected: %s", category)

        strategy = self.strategies.get(category, self.strategies["Factual"])
        docs = strategy.retrieve(query, user_context=user_context)

        # Fallback: web search when local results are sparse
        if use_web_search and self.tavily_client and len(docs) < 2:
            logger.info("Few local results — triggering Tavily web search.")
            try:
                search_res = self.tavily_client.search(
                    query, search_depth="advanced", max_results=3,
                )
                for res in search_res.get("results", []):
                    docs.append(
                        Document(
                            page_content=res["content"],
                            metadata={"source": res["url"], "type": "web"},
                        ),
                    )
            except Exception:
                logger.warning("Tavily search failed.", exc_info=True)

        return docs, category


class AdaptiveRAG:
    """End-to-end RAG: retrieve → generate answer."""

    def __init__(self, resources: SharedResources) -> None:
        self.retriever = AdaptiveRetriever(resources)
        self.llm = resources.llm

        self.prompt_template = PromptTemplate(
            template=load_prompt("rag_answer"),
            input_variables=["context", "question"],
        )
        self.qa_chain = self.prompt_template | self.llm

    def answer(
        self,
        query: str,
        user_context: Optional[str] = None,
        use_web: bool = True,
    ) -> dict:
        """Retrieve context and generate an answer for *query*."""
        docs, category = self.retriever.get_relevant_documents(
            query=query,
            user_context=user_context,
            use_web_search=use_web,
        )

        logger.info("Retrieved %d documents.", len(docs))

        context_str = "\n\n".join(
            f"Source [{doc.metadata.get('source', 'Unknown')}]:\n"
            f"{doc.page_content}"
            for doc in docs
        )

        logger.info("Generating final answer...")
        result = self.qa_chain.invoke(
            {"context": context_str, "question": query},
        )

        return {
            "answer": result.content,
            "category": category,
            "sources": [d.metadata.get("source") for d in docs],
        }
