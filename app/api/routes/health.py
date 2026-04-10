"""Health-check endpoint."""

from fastapi import APIRouter, Request

from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return server status and whether the RAG system is initialized."""
    rag_ready = getattr(request.app.state, "rag_system", None) is not None

    worker = getattr(request.app.state, "worker", None)
    rabbitmq_ok = (
        worker is not None
        and worker._connection is not None
        and not worker._connection.is_closed
    )

    return HealthResponse(
        status="ok",
        rag_initialized=rag_ready,
        rabbitmq_connected=rabbitmq_ok,
    )
