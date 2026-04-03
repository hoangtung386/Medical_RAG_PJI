"""Request schemas for the API."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ProcessSnapshotRequest(BaseModel):
    """Request body for ``POST /api/v1/process-snapshot``."""

    request_id: str
    episode_id: int
    snapshot_id: int
    snapshot_data_json: dict[str, Any]
    options: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Options: language (str), include_citations (bool), top_k (int)"
        ),
    )


class ChatMessage(BaseModel):
    """A single message in the chat history."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for ``POST /api/v1/chat``."""

    question: str
    episode_summary: Optional[dict[str, Any]] = None
    recommendation_context: Optional[dict[str, Any]] = None
    chat_history: Optional[list[ChatMessage]] = None
