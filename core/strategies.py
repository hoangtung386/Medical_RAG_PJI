import os
from typing import List, Optional

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus
from pydantic import BaseModel, Field

from .llm_config import get_cohere_embeddings, get_groq_llm


class BaseRetrievalStrategy:

    def __init__(self, collection_name="medical_rag_docs"):
        self.llm = get_groq_llm(
            "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
        )
        self.embeddings = get_cohere_embeddings()

        zilliz_uri = os.getenv("ZILLIZ_URI")
        zilliz_api_key = os.getenv("ZILLIZ_API_KEY")

        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            connection_args={
                "uri": zilliz_uri,
                "token": zilliz_api_key,
                "secure": True,
            },
        )

        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 15},
        )

        cohere_api_key = os.getenv("COHERE_API_KEY")
        self.reranker = CohereRerank(
            cohere_api_key=cohere_api_key,
            model="rerank-multilingual-v3.0",
            top_n=5,
        )
        self.compression_retriever = (
            ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=self.base_retriever,
            )
        )

    def _dual_search(
        self,
        original_query: str,
        enhanced_query: str,
        k_each: int = 10,
    ):
        """Search with both original and enhanced queries.

        Uses cross-lingual search via Cohere multilingual
        embeddings, then deduplicates results.
        """
        docs_original = self.vector_store.similarity_search(
            original_query, k=k_each,
        )
        print(
            f"  -> Original query found {len(docs_original)} docs"
        )

        docs_enhanced = self.vector_store.similarity_search(
            enhanced_query, k=k_each,
        )
        print(
            f"  -> Enhanced query found {len(docs_enhanced)} docs"
        )

        all_docs = docs_original + docs_enhanced
        unique_docs = list(
            {doc.page_content: doc for doc in all_docs}.values()
        )
        print(
            f"  -> Merged & deduplicated: "
            f"{len(unique_docs)} unique docs"
        )

        return unique_docs

    def retrieve(self, query: str, k: int = 4):
        """Base fallback retrieval."""
        self.reranker.top_n = k
        return self.compression_retriever.invoke(query)


class FactualRetrievalStrategy(BaseRetrievalStrategy):

    def retrieve(self, query: str, k: int = 4):
        print(
            f"[Strategy: Factual] Retrieving facts for: {query}"
        )

        prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are a medical search expert. Translate and "
                "enhance the following medical query into English "
                "for searching an English medical document "
                "database. Keep all medical terminology accurate. "
                "Output ONLY the enhanced query text in English.\n"
                "Query: {query}\n"
                "Enhanced Query (English):"
            ),
        )
        chain = prompt | self.llm
        enhanced_query = chain.invoke(
            {"query": query},
        ).content.strip()
        print(f"Enhanced Query: {enhanced_query}")

        unique_docs = self._dual_search(
            query, enhanced_query, k_each=10,
        )

        if not unique_docs:
            return []
        reranked = self.reranker.compress_documents(
            documents=unique_docs, query=query,
        )
        return reranked[:k]


class SubQueries(BaseModel):
    sub_queries: List[str] = Field(
        description="List of sub-queries for comprehensive analysis",
    )


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):

    def retrieve(self, query: str, k: int = 5):
        print(
            "[Strategy: Analytical] Breaking down query: "
            f"{query}"
        )

        parser = PydanticOutputParser(
            pydantic_object=SubQueries,
        )
        prompt = PromptTemplate(
            input_variables=["query", "k"],
            template=(
                "Break down the following complex medical query "
                "into {k} different sub-questions for "
                "comprehensive analysis. Output all sub-questions "
                "in English regardless of input language.\n"
                "{format_instructions}\n"
                "Query: {query}"
            ),
            partial_variables={
                "format_instructions": (
                    parser.get_format_instructions()
                ),
            },
        )
        chain = (
            prompt
            | get_groq_llm("llama-3.1-8b-instant", temperature=0.2)
            | parser
        )

        try:
            sub_queries = chain.invoke(
                {"query": query, "k": 3},
            ).sub_queries
            print(f"Sub-queries: {sub_queries}")
        except Exception:
            print(
                "Failed to generate sub-queries, "
                "using original query."
            )
            sub_queries = [query]

        all_docs = []
        for sq in sub_queries:
            docs_en = self.vector_store.similarity_search(
                sq, k=5,
            )
            all_docs.extend(docs_en)

        docs_original = self.vector_store.similarity_search(
            query, k=10,
        )
        all_docs.extend(docs_original)
        print(
            f"  -> Total docs before dedup: {len(all_docs)}"
        )

        unique_docs = list(
            {doc.page_content: doc for doc in all_docs}.values()
        )
        print(f"  -> After dedup: {len(unique_docs)} unique docs")

        if not unique_docs:
            return []

        reranked_docs = self.reranker.compress_documents(
            documents=unique_docs, query=query,
        )
        return reranked_docs[:k]


class SelectedIndices(BaseModel):
    indices: List[int] = Field(
        description="Indices of selected diverse documents",
    )


class OpinionRetrievalStrategy(BaseRetrievalStrategy):

    def retrieve(self, query: str, k: int = 4):
        print(
            "[Strategy: Opinion] Looking for diverse "
            f"perspectives on: {query}"
        )

        prompt = PromptTemplate(
            input_variables=["query", "k"],
            template=(
                "Identify {k} distinctly different viewpoints, "
                "theories, or medical opinions regarding the "
                "following topic. Output as a numbered list in "
                "English.\nTopic: {query}"
            ),
        )
        chain = prompt | self.llm
        viewpoints_text = chain.invoke(
            {"query": query, "k": 3},
        ).content
        viewpoints = [
            v.strip()
            for v in viewpoints_text.split("\n")
            if v.strip()
        ]
        print(f"Identified viewpoints: {viewpoints}")

        all_docs = []
        for vp in viewpoints:
            docs = self.vector_store.similarity_search(vp, k=3)
            all_docs.extend(docs)

        docs_original = self.vector_store.similarity_search(
            query, k=10,
        )
        all_docs.extend(docs_original)

        unique_docs = list(
            {doc.page_content: doc for doc in all_docs}.values()
        )
        print(
            f"  -> {len(unique_docs)} unique docs after dedup"
        )

        if not unique_docs:
            return []

        parser = PydanticOutputParser(
            pydantic_object=SelectedIndices,
        )
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=(
                "Classify these medical excerpts into distinct "
                "opinions on '{query}'. Select the {k} most "
                "representative AND diverse viewpoints.\n"
                "Documents:\n{docs}\n"
                "{format_instructions}\n"
                "Selected indices (0-indexed):"
            ),
            partial_variables={
                "format_instructions": (
                    parser.get_format_instructions()
                ),
            },
        )

        docs_text = "\n".join(
            f"[{i}]: {doc.page_content[:200]}..."
            for i, doc in enumerate(unique_docs)
        )
        diversity_chain = (
            diversity_prompt
            | get_groq_llm("llama-3.1-8b-instant")
            | parser
        )

        try:
            selected_indices = diversity_chain.invoke(
                {"query": query, "docs": docs_text, "k": k},
            ).indices
            print(f"Selected diverse indices: {selected_indices}")
            return [
                unique_docs[i]
                for i in selected_indices
                if i < len(unique_docs)
            ]
        except Exception:
            return self.reranker.compress_documents(
                documents=unique_docs, query=query,
            )[:k]


class ContextualRetrievalStrategy(BaseRetrievalStrategy):

    def retrieve(
        self, query: str, user_context: Optional[str] = None,
        k: int = 4,
    ):
        print("[Strategy: Contextual] Contextualizing query...")
        if not user_context:
            user_context = "No specific patient context provided."

        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Given the patient/user context: '{context}'\n"
                "Reformulate this medical query to specifically "
                "address the user's needs: '{query}'\n"
                "Output the reformulated query in English.\n"
                "Output ONLY the Reformulated Query (English):"
            ),
        )
        chain = prompt | self.llm
        contextualized_query = chain.invoke(
            {"query": query, "context": user_context},
        ).content.strip()
        print(
            "Contextualized Query (EN): "
            f"{contextualized_query}"
        )

        unique_docs = self._dual_search(
            query, contextualized_query, k_each=10,
        )

        if not unique_docs:
            return []

        reranked = self.reranker.compress_documents(
            documents=unique_docs, query=contextualized_query,
        )
        return reranked[:k]
