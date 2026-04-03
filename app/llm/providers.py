"""LLM and embedding provider wrappers.

Centralizes creation of Groq LLM and Cohere Embedding instances
so that configuration lives in one place.
"""

from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

from app.config import settings


def get_groq_llm(
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatGroq:
    """Return a ChatGroq instance.

    Args:
        model_name: Override model; defaults to ``settings.main_model``.
        temperature: Override temperature; defaults to
            ``settings.main_model_temperature``.
    """
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=model_name or settings.main_model,
        temperature=(
            temperature if temperature is not None
            else settings.main_model_temperature
        ),
    )


def get_cohere_embeddings() -> CohereEmbeddings:
    """Return a CohereEmbeddings instance for multilingual texts."""
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY not found in environment variables.")

    return CohereEmbeddings(
        cohere_api_key=settings.cohere_api_key,
        model=settings.embedding_model,
    )
