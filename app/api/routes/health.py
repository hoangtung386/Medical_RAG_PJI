"""Health-check endpoint."""

from fastapi import APIRouter, Request

from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return server status and whether the RAG system is initialized."""
    rag_ready = getattr(request.app.state, "rag_system", None) is not None
    return HealthResponse(status="ok", rag_initialized=rag_ready)
