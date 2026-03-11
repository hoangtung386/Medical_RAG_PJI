import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from tavily import TavilyClient

from .classifier import QueryClassifier
from .llm_config import get_groq_llm
from .strategies import (
    AnalyticalRetrievalStrategy,
    ContextualRetrievalStrategy,
    FactualRetrievalStrategy,
    OpinionRetrievalStrategy,
)


class AdaptiveRetriever:

    def __init__(self, collection_name="medical_rag_docs"):
        self.classifier = QueryClassifier()

        self.strategies = {
            "Factual": FactualRetrievalStrategy(
                collection_name,
            ),
            "Analytical": AnalyticalRetrievalStrategy(
                collection_name,
            ),
            "Opinion": OpinionRetrievalStrategy(
                collection_name,
            ),
            "Contextual": ContextualRetrievalStrategy(
                collection_name,
            ),
        }

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tavily_client = TavilyClient(
                api_key=tavily_api_key,
            )
        else:
            self.tavily_client = None

    def get_relevant_documents(
        self,
        query: str,
        user_context: Optional[str] = None,
        use_web_search: bool = True,
    ) -> tuple[List[Document], str]:
        category = self.classifier.classify(query)
        print(f"--> Strategy Selected: {category}")

        strategy = self.strategies.get(
            category, self.strategies["Factual"],
        )

        if category == "Contextual":
            docs = strategy.retrieve(
                query, user_context=user_context,
            )
        else:
            docs = strategy.retrieve(query)

        if (
            use_web_search
            and self.tavily_client
            and len(docs) < 2
        ):
            print(
                "--> Few local results found. "
                "Triggering Tavily Web Search Grounding..."
            )
            try:
                search_res = self.tavily_client.search(
                    query,
                    search_depth="advanced",
                    max_results=3,
                )
                for res in search_res.get("results", []):
                    web_doc = Document(
                        page_content=res["content"],
                        metadata={
                            "source": res["url"],
                            "type": "web",
                        },
                    )
                    docs.append(web_doc)
            except Exception as e:
                print(f"Error executing Tavily search: {e}")

        return docs, category


class AdaptiveRAG:

    def __init__(self, collection_name="medical_rag_docs"):
        self.retriever = AdaptiveRetriever(collection_name)

        self.llm = get_groq_llm(
            model_name=(
                "meta-llama/llama-4-scout-17b-16e-instruct"
            ),
            temperature=0.1,
        )

        self.prompt_template = PromptTemplate(
            template=(
                "Ban la mot tro ly y te AI chuyen gia. "
                "Su dung cac doan ngu canh sau day de tra loi "
                "cau hoi o cuoi.\n"
                "Neu ban danh gia rang ngu canh khong chua du "
                "thong tin de tra loi chinh xac, hay noi ro "
                "rang ban khong biet dua tren tai lieu duoc "
                "cung cap. Khong duoc doan hoac bia dat "
                "cau tra loi.\n"
                "Trong truong hop ngu canh co nhieu quan diem "
                "khac nhau, hay tom tat tat ca cac quan diem "
                "mot cach khach quan.\n"
                "QUAN TRONG: Luon tra loi bang tieng Viet.\n\n"
                "Ngu canh duoc cung cap:\n"
                "-----------------\n"
                "{context}\n"
                "-----------------\n\n"
                "Cau hoi: {question}\n"
                "Cau tra loi chuyen gia (dinh dang Markdown):"
            ),
            input_variables=["context", "question"],
        )

        self.qa_chain = self.prompt_template | self.llm

    def answer(
        self,
        query: str,
        user_context: Optional[str] = None,
        use_web: bool = True,
    ) -> dict:
        docs, category = self.retriever.get_relevant_documents(
            query=query,
            user_context=user_context,
            use_web_search=use_web,
        )

        print(f"--> Retrieved {len(docs)} documents.")

        context_str = "\n\n".join(
            f"Source [{doc.metadata.get('source', 'Unknown')}]:\n"
            f"{doc.page_content}"
            for doc in docs
        )

        print("--> Generating Final Answer...")

        result = self.qa_chain.invoke(
            {"context": context_str, "question": query},
        )

        return {
            "answer": result.content,
            "category": category,
            "sources": [
                d.metadata.get("source") for d in docs
            ],
        }
