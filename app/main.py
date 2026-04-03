"""FastAPI application entry-point for PJI Clinical Decision Support."""

import logging
import logging.config
import time
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import chat, health, recommendation
from app.config import settings
from app.core.rag.retriever import AdaptiveRAG
from app.core.recommendation import PJIRecommendationEngine
from app.core.shared import SharedResources

# Suppress pymilvus pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "langchain_milvus.vectorstores.milvus": {"level": "ERROR"},
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 — required signature
    """Initialize shared resources on startup; clean up on shutdown."""
    logger.info("Starting PJI Clinical Decision Support system...")
    t0 = time.time()
    try:
        resources = SharedResources()
        rag_system = AdaptiveRAG(resources)
        pji_engine = PJIRecommendationEngine(rag_system, resources)

        app.state.resources = resources
        app.state.rag_system = rag_system
        app.state.pji_engine = pji_engine

        elapsed = time.time() - t0
        logger.info(
            "Initialization complete in %.1fs. "
            "Collection: %s",
            elapsed,
            resources.cfg.collection_name,
        )
    except Exception:
        logger.error("Initialization failed.", exc_info=True)

    yield  # application runs here

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PJI Clinical Decision Support API",
    description="API ho tro quyet dinh lam sang nhiem trung khop gia (PJI)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(health.router)
app.include_router(recommendation.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


# --- Root redirect ---
@app.get("/", include_in_schema=False)
async def root():
    """Redirect ``/`` to Swagger UI for convenience."""
    return RedirectResponse(url="/docs")


# --- Request timing middleware ---
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add ``X-Process-Time`` header to every response."""
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.3f}s"
    return response


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
