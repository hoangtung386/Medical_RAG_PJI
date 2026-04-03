"""Shared resources that are expensive to create.

All retrieval strategies and the RAG engine share a single instance
of this class so that we only create one Milvus connection, one
reranker, one embedding model, etc.
"""

import logging

from langchain.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langchain_milvus import Milvus

from app.config import Settings, settings
from app.llm.providers import get_cohere_embeddings, get_groq_llm

logger = logging.getLogger(__name__)


class SharedResources:
    """Singleton-like container for expensive, reusable objects."""

    def __init__(self, cfg: Settings | None = None) -> None:
        cfg = cfg or settings
        logger.info("Initializing shared resources...")

        # LLMs
        self.llm = get_groq_llm(cfg.main_model, cfg.main_model_temperature)
        self.fast_llm = get_groq_llm(
            cfg.fast_model, cfg.fast_model_temperature,
        )

        # Embeddings
        self.embeddings = get_cohere_embeddings()

        # Vector store (Milvus / Zilliz)
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=cfg.collection_name,
            connection_args={
                "uri": cfg.zilliz_uri,
                "token": cfg.zilliz_api_key,
                "secure": True,
            },
        )

        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": cfg.retrieval_k},
        )

        # Reranker
        self.reranker = CohereRerank(
            cohere_api_key=cfg.cohere_api_key,
            model=cfg.rerank_model,
            top_n=cfg.rerank_top_n,
        )

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.base_retriever,
        )

        # Keep config reference for downstream consumers
        self.cfg = cfg

        logger.info("Shared resources initialized.")
