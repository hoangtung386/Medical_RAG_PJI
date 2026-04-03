"""Response schemas for the API."""

from typing import Any, Optional

from pydantic import BaseModel

from .common import ModelInfo
from .completeness import DataCompleteness


class RecommendationItemResponse(BaseModel):
    """One row in the ``ai_recommendation_items`` table."""

    id: str
    category: str  # DIAGNOSTIC_TEST | LOCAL_ANTIBIOTIC | SYSTEMIC_ANTIBIOTIC | SURGERY_PROCEDURE
    title: str
    item_json: dict[str, Any]


class CitationResponse(BaseModel):
    """One row in the ``ai_rag_citations`` table."""

    id: str
    run_id: str
    item_id: Optional[str] = None
    source_type: str  # GUIDELINE | META_ANALYSIS | JOURNAL_ARTICLE | ...
    source_title: str
    source_uri: Optional[str] = None
    snippet: str
    relevance_score: float
    cited_for: Optional[str] = None


class ProcessSnapshotResponse(BaseModel):
    """Full response for ``POST /api/v1/process-snapshot``."""

    request_id: str
    status: str  # SUCCESS | ERROR
    model: ModelInfo
    latency_ms: int
    run_id: str
    data_completeness: DataCompleteness
    ai_recommendation_items: list[RecommendationItemResponse]
    ai_rag_citations: list[CitationResponse]


class ChatResponse(BaseModel):
    """Response for ``POST /api/v1/chat``."""

    answer: str
    latency_ms: int
    tokens_used: Optional[int] = None
    references: Optional[list[dict[str, Any]]] = None
