"""Centralized application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    main_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    main_model_temperature: float = 0.1
    fast_model: str = "llama-3.1-8b-instant"
    fast_model_temperature: float = 0.0

    # --- Embedding & Reranker (Cohere) ---
    cohere_api_key: str = ""
    embedding_model: str = "embed-multilingual-v3.0"
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_top_n: int = 5

    # --- Vector DB (Zilliz / Milvus) ---
    zilliz_uri: str = ""
    zilliz_api_key: str = ""
    collection_name: str = "medical_rag_docs"

    # --- Retrieval ---
    retrieval_k: int = 15
    dual_search_k: int = 10

    # --- Web search (Tavily) ---
    tavily_api_key: str = ""

    # --- RabbitMQ ---
    rabbitmq_url: str = ""
    rabbitmq_exchange: str = "pji.ai.exchange"
    rabbitmq_recommendation_queue: str = "pji.ai.recommendation.queue"
    rabbitmq_recommendation_result_routing_key: str = "ai.recommendation.result"
    rabbitmq_prefetch_count: int = 1

    # --- CORS ---
    cors_origins: list[str] = ["*"]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
