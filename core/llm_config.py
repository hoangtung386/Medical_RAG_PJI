import os

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

load_dotenv()


def get_groq_llm(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
):
    """Initialize Groq LLM.

    Use `meta-llama/llama-4-scout-17b-16e-instruct` for heavy tasks.
    Use `llama-3.1-8b-instant` for fast routing/classification.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )


def get_cohere_embeddings():
    """Initialize Cohere Embeddings.

    embed-multilingual-v3.0 is recommended for Vietnamese texts.
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY not found in environment variables."
        )

    return CohereEmbeddings(
        cohere_api_key=api_key,
        model="embed-multilingual-v3.0",
    )
