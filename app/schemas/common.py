"""Shared / common Pydantic models."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    status: str
    rag_initialized: bool


class ModelInfo(BaseModel):
    """Metadata about the AI model used."""

    name: str
    version: str
