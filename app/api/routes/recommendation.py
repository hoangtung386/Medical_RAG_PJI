"""Process-snapshot endpoint — main recommendation pipeline."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from app.core.completeness import check_data_completeness
from app.core.recommendation import PJIRecommendationEngine
from app.dependencies import get_pji_engine
from app.schemas.common import ModelInfo
from app.schemas.completeness import DataCompleteness
from app.schemas.request import ProcessSnapshotRequest
from app.schemas.response import (
    CitationResponse,
    ProcessSnapshotResponse,
    RecommendationItemResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendation"])


@router.post(
    "/process-snapshot",
    response_model=ProcessSnapshotResponse,
)
async def process_snapshot(
    request: ProcessSnapshotRequest,
    engine: PJIRecommendationEngine = Depends(get_pji_engine),
) -> ProcessSnapshotResponse:
    """Process clinical snapshot and return recommendations + citations."""
    try:
        start_time = time.time()
        snapshot_data = request.snapshot_data_json

        # Step 1: deterministic completeness check (no LLM)
        completeness = check_data_completeness(snapshot_data)

        # Step 2: generate recommendations + citations via RAG
        options = request.options or {
            "language": "vi",
            "include_citations": True,
            "top_k": 5,
        }
        result = engine.generate_recommendation(
            snapshot_data=snapshot_data,
            options=options,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        recommendation_items = [
            RecommendationItemResponse(
                id=item["id"],
                category=item["category"],
                title=item["title"],
                item_json=item["item_json"],
            )
            for item in result.get("recommendation_items", [])
        ]

        citations = [
            CitationResponse(
                id=cit["id"],
                run_id=cit["run_id"],
                item_id=cit.get("item_id"),
                source_type=cit["source_type"],
                source_title=cit["source_title"],
                source_uri=cit.get("source_uri"),
                snippet=cit["snippet"],
                relevance_score=cit["relevance_score"],
                cited_for=cit.get("cited_for"),
            )
            for cit in result.get("citations", [])
        ]

        return ProcessSnapshotResponse(
            request_id=request.request_id,
            status="SUCCESS",
            model=ModelInfo(
                name=engine.model_name,
                version=engine.model_version,
            ),
            latency_ms=latency_ms,
            run_id=result.get("run_id", ""),
            data_completeness=DataCompleteness(**completeness),
            assessment_json=result.get("assessment_json"),
            explanation_json=result.get("explanation_json"),
            warnings_json=result.get("warnings_json"),
            ai_recommendation_items=recommendation_items,
            ai_rag_citations=citations,
        )

    except Exception as exc:
        logger.error("Process snapshot failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Loi xu ly snapshot: {exc}",
        ) from exc
