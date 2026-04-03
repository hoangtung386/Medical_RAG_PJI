"""FastAPI dependency injection helpers."""

from fastapi import HTTPException, Request

from app.core.rag.retriever import AdaptiveRAG
from app.core.recommendation import PJIRecommendationEngine


def get_rag_system(request: Request) -> AdaptiveRAG:
    """Return the ``AdaptiveRAG`` instance from application state."""
    rag: AdaptiveRAG | None = getattr(request.app.state, "rag_system", None)
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="He thong AI chua san sang.",
        )
    return rag


def get_pji_engine(request: Request) -> PJIRecommendationEngine:
    """Return the ``PJIRecommendationEngine`` from application state."""
    engine: PJIRecommendationEngine | None = getattr(
        request.app.state, "pji_engine", None,
    )
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="He thong AI chua san sang.",
        )
    return engine
