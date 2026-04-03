"""Chat endpoint — conversational Q&A about PJI clinical cases."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from app.core.recommendation import PJIRecommendationEngine
from app.dependencies import get_pji_engine
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    engine: PJIRecommendationEngine = Depends(get_pji_engine),
) -> ChatResponse:
    """Chat with AI about PJI clinical decisions."""
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Cau hoi khong duoc de trong.",
        )

    try:
        start_time = time.time()
        result = engine.chat(
            question=request.question,
            episode_summary=request.episode_summary,
            recommendation_context=request.recommendation_context,
            chat_history=(
                [m.model_dump() for m in request.chat_history]
                if request.chat_history
                else None
            ),
        )
        latency_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            answer=result.get("answer", ""),
            latency_ms=latency_ms,
            tokens_used=result.get("tokens_used"),
            references=result.get("references"),
        )

    except Exception as exc:
        logger.error("Chat failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Loi chat: {exc}",
        ) from exc
