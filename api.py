"""FastAPI server for Medical Adaptive RAG.

Provides REST API endpoints for web app integration.

Usage:
    python api.py

    Or with uvicorn:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
logging.getLogger(
    "langchain_milvus.vectorstores.milvus"
).setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from core.engine import AdaptiveRAG  # noqa: E402
from core.pji_recommendation import PJIRecommendationEngine  # noqa: E402

app = FastAPI(
    title="Medical Adaptive RAG API",
    description=(
        "API tro ly y te thong minh voi kha nang tu dong "
        "lua chon chien luoc truy xuat"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = None
pji_engine = None


@app.on_event("startup")
async def startup():
    global rag_system, pji_engine
    print("Dang khoi tao He thong Medical Adaptive RAG...")
    try:
        rag_system = AdaptiveRAG()
        pji_engine = PJIRecommendationEngine(rag_system)
        print("Khoi tao thanh cong!")
    except Exception as e:
        print(f"Loi khoi tao: {e}")


# ==================== Existing Models ====================

class QueryRequest(BaseModel):
    question: str
    user_context: Optional[str] = None
    use_web_search: bool = True


class QueryResponse(BaseModel):
    answer: str
    category: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool


# ==================== Recommendation Models ====================

class RecommendationOptions(BaseModel):
    language: str = "vi"
    include_citations: bool = True
    top_k: int = 5


class RecommendationRequest(BaseModel):
    request_id: str
    trigger_type: str
    episode_id: int
    snapshot_id: int
    snapshot_data_json: dict[str, Any]
    options: RecommendationOptions = RecommendationOptions()


class ModelInfo(BaseModel):
    name: str
    version: str


class RecommendationItem(BaseModel):
    client_item_key: Optional[str] = None
    category: str
    title: str
    priority_order: int
    is_primary: bool
    item_json: dict[str, Any]


class CitationItem(BaseModel):
    client_item_key: Optional[str] = None
    source_type: str
    source_title: str
    source_uri: Optional[str] = None
    snippet: str
    relevance_score: float
    cited_for: Optional[str] = None


class RecommendationResponse(BaseModel):
    request_id: str
    status: str
    model: ModelInfo
    latency_ms: int
    assessment_json: Optional[dict[str, Any]] = None
    explanation_json: Optional[dict[str, Any]] = None
    warnings_json: Optional[list[dict[str, Any]]] = None
    items: list[RecommendationItem]
    citations: list[CitationItem] = []


# ==================== Chat Models ====================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    episode_summary: Optional[dict[str, Any]] = None
    recommendation_context: Optional[dict[str, Any]] = None
    chat_history: Optional[list[ChatMessage]] = None


class ChatResponse(BaseModel):
    answer: str
    latency_ms: int
    tokens_used: Optional[int] = None
    references: Optional[list[dict[str, Any]]] = None


# ==================== Existing Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kiem tra trang thai server."""
    return HealthResponse(
        status="ok",
        rag_initialized=rag_system is not None,
    )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Endpoint chinh - Hoi cau hoi y te."""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="He thong chua duoc khoi tao. Kiem tra API keys.",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Cau hoi khong duoc de trong.",
        )

    try:
        result = rag_system.answer(
            query=request.question,
            user_context=request.user_context,
            use_web=request.use_web_search,
        )

        return QueryResponse(
            answer=result["answer"],
            category=result["category"],
            sources=result["sources"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Loi xu ly cau hoi: {str(e)}",
        )


# ==================== Recommendation Endpoint ====================

@app.post("/api/v1/recommendation/generate", response_model=RecommendationResponse)
async def generate_recommendation(request: RecommendationRequest):
    """Generate PJI recommendation based on clinical snapshot data.

    Receives normalized clinical data, runs RAG pipeline,
    and returns structured recommendation items with citations.
    """
    if not rag_system or not pji_engine:
        raise HTTPException(
            status_code=503,
            detail="He thong AI chua san sang.",
        )

    try:
        start_time = time.time()
        result = pji_engine.generate_recommendation(
            snapshot_data=request.snapshot_data_json,
            options=request.options.model_dump(),
        )
        latency_ms = int((time.time() - start_time) * 1000)

        return RecommendationResponse(
            request_id=request.request_id,
            status="SUCCESS",
            model=ModelInfo(
                name=pji_engine.model_name,
                version=pji_engine.model_version,
            ),
            latency_ms=latency_ms,
            assessment_json=result.get("assessment_json"),
            explanation_json=result.get("explanation_json"),
            warnings_json=result.get("warnings_json"),
            items=result.get("items", []),
            citations=result.get("citations", []),
        )

    except Exception as e:
        logging.error(f"Recommendation generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Loi tao recommendation: {str(e)}",
        )


# ==================== Chat Endpoint ====================

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with AI about PJI clinical decisions.

    Supports episode context, recommendation context,
    and chat history for contextual conversations.
    """
    if not rag_system or not pji_engine:
        raise HTTPException(
            status_code=503,
            detail="He thong AI chua san sang.",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Cau hoi khong duoc de trong.",
        )

    try:
        start_time = time.time()
        result = pji_engine.chat(
            question=request.question,
            episode_summary=request.episode_summary,
            recommendation_context=request.recommendation_context,
            chat_history=[m.model_dump() for m in request.chat_history] if request.chat_history else None,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            answer=result.get("answer", ""),
            latency_ms=latency_ms,
            tokens_used=result.get("tokens_used"),
            references=result.get("references"),
        )

    except Exception as e:
        logging.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Loi chat: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
