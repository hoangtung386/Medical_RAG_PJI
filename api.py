"""FastAPI server for Medical Adaptive RAG.

Provides REST API endpoints for web app integration.

Usage:
    python api.py

    Or with uvicorn:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logging.getLogger(
    "langchain_milvus.vectorstores.milvus"
).setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.engine import AdaptiveRAG  # noqa: E402

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


@app.on_event("startup")
async def startup():
    global rag_system
    print("Dang khoi tao He thong Medical Adaptive RAG...")
    try:
        rag_system = AdaptiveRAG()
        print("Khoi tao thanh cong!")
    except Exception as e:
        print(f"Loi khoi tao: {e}")


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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kiem tra trang thai server."""
    return HealthResponse(
        status="ok",
        rag_initialized=rag_system is not None,
    )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Endpoint chinh - Hoi cau hoi y te.

    Request body:
    - question: Cau hoi (tieng Viet hoac tieng Anh)
    - user_context: Thong tin benh nhan (tuy chon)
    - use_web_search: Bat/tat tim kiem web (mac dinh: true)

    Response:
    - answer: Cau tra loi
    - category: Chien luoc duoc su dung
    - sources: Danh sach nguon tham khao
    """
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail=(
                "He thong chua duoc khoi tao. "
                "Kiem tra API keys."
            ),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
