"""FastAPI server for PJI Clinical Decision Support.

Backend API that receives clinical snapshot data from the web backend,
processes it through RAG pipeline, and returns:
1. ai_recommendation_items (DIAGNOSTIC_TEST, SYSTEMIC_ANTIBIOTIC, LOCAL_ANTIBIOTIC, SURGERY_PROCEDURE)
2. ai_rag_citations (evidence citations linked to items)
3. data_completeness (deterministic completeness check)

Usage:
    python api.py

    Or with uvicorn:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
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

from core.data_completeness import check_data_completeness  # noqa: E402
from core.engine import AdaptiveRAG  # noqa: E402
from core.pji_recommendation import PJIRecommendationEngine  # noqa: E402

app = FastAPI(
    title="PJI Clinical Decision Support API",
    description="API ho tro quyet dinh lam sang nhiem trung khop gia (PJI)",
    version="2.0.0",
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
    print("Dang khoi tao He thong PJI Clinical Decision Support...")
    try:
        rag_system = AdaptiveRAG()
        pji_engine = PJIRecommendationEngine(rag_system)
        print("Khoi tao thanh cong!")
    except Exception as e:
        print(f"Loi khoi tao: {e}")


# ==================== Pydantic Models ====================

class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool


class ProcessSnapshotRequest(BaseModel):
    """Request matching section_1_input of the API contract."""
    request_id: str
    episode_id: int
    snapshot_id: int
    snapshot_data_json: dict[str, Any]
    options: Optional[dict[str, Any]] = Field(
        default=None,
        description="Options: language (str), include_citations (bool), top_k (int)",
    )


class ModelInfo(BaseModel):
    name: str
    version: str


class RecommendationItemResponse(BaseModel):
    """One row in ai_recommendation_items table."""
    id: str
    category: str  # DIAGNOSTIC_TEST | LOCAL_ANTIBIOTIC | SYSTEMIC_ANTIBIOTIC | SURGERY_PROCEDURE
    title: str
    item_json: dict[str, Any]


class CitationResponse(BaseModel):
    """One row in ai_rag_citations table."""
    id: str
    run_id: str
    item_id: Optional[str] = None
    source_type: str  # GUIDELINE | META_ANALYSIS | JOURNAL_ARTICLE | CONSENSUS_STATEMENT | SYSTEMATIC_REVIEW
    source_title: str
    source_uri: Optional[str] = None
    snippet: str
    relevance_score: float
    cited_for: Optional[str] = None


class MissingItem(BaseModel):
    field: str
    category: str  # ICM_MAJOR | ICM_MINOR | CLINICAL
    importance: str  # CRITICAL | HIGH | MEDIUM
    message: str


class DataCompleteness(BaseModel):
    """Deterministic data completeness check result."""
    is_complete: bool
    missing_items: list[MissingItem]
    completeness_score: str
    impact_note: str


class ProcessSnapshotResponse(BaseModel):
    """Full response for /api/v1/process-snapshot."""
    request_id: str
    status: str  # SUCCESS | ERROR
    model: ModelInfo
    latency_ms: int
    run_id: str
    data_completeness: DataCompleteness
    ai_recommendation_items: list[RecommendationItemResponse]
    ai_rag_citations: list[CitationResponse]


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


# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kiem tra trang thai server."""
    return HealthResponse(
        status="ok",
        rag_initialized=rag_system is not None,
    )


@app.post("/api/v1/process-snapshot", response_model=ProcessSnapshotResponse)
async def process_snapshot(request: ProcessSnapshotRequest):
    """Main endpoint - Process clinical snapshot and return recommendations + citations + completeness.

    Input: snapshot_data_json (section_1_input from API contract)
    Output:
        - data_completeness: deterministic check of missing data
        - ai_recommendation_items: 4 categories (DIAGNOSTIC_TEST, LOCAL_ANTIBIOTIC, SYSTEMIC_ANTIBIOTIC, SURGERY_PROCEDURE)
        - ai_rag_citations: evidence citations linked to recommendation items
    """
    if not rag_system or not pji_engine:
        raise HTTPException(
            status_code=503,
            detail="He thong AI chua san sang.",
        )

    try:
        start_time = time.time()
        snapshot_data = request.snapshot_data_json

        # Step 1: Deterministic data completeness check (no LLM)
        completeness = check_data_completeness(snapshot_data)

        # Step 2: Generate recommendations + citations via RAG + LLM
        options = request.options or {"language": "vi", "include_citations": True, "top_k": 5}
        result = pji_engine.generate_recommendation(
            snapshot_data=snapshot_data,
            options=options,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Build response matching contract
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
                name=pji_engine.model_name,
                version=pji_engine.model_version,
            ),
            latency_ms=latency_ms,
            run_id=result.get("run_id", ""),
            data_completeness=DataCompleteness(**completeness),
            ai_recommendation_items=recommendation_items,
            ai_rag_citations=citations,
        )

    except Exception as e:
        logging.error(f"Process snapshot failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Loi xu ly snapshot: {str(e)}",
        )


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
