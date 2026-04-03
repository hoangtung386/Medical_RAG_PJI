"""Pydantic models for data-completeness results."""

from pydantic import BaseModel


class MissingItem(BaseModel):
    """A single missing data field."""

    field: str
    category: str  # ICM_MAJOR | ICM_MINOR | CLINICAL
    importance: str  # CRITICAL | HIGH | MEDIUM
    message: str


class DataCompleteness(BaseModel):
    """Deterministic data-completeness check result."""

    is_complete: bool
    missing_items: list[MissingItem]
    completeness_score: str
    impact_note: str
