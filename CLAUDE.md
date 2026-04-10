# CLAUDE.md — PJI Clinical Decision Support (Backend AI)

## Project Overview

A **FastAPI-based RAG backend** for clinical decision support in **Periprosthetic Joint Infection (PJI)** diagnosis and treatment planning. It receives clinical snapshot data from a web backend, processes it through an adaptive retrieval pipeline, and returns structured treatment recommendations with evidence citations.

- **Author:** hoangtung386
- **License:** MIT
- **Python:** >=3.10 (Docker uses 3.11)
- **Package Manager:** [uv](https://astral.sh/uv)

## Project Structure

```
.
├── api.py                      # FastAPI server — main endpoints
├── ingest_local.py             # PDF ingestion pipeline → Zilliz vector DB
├── api_contract.json           # JSON contract spec (input/output schemas)
├── core/
│   ├── engine.py               # AdaptiveRetriever + AdaptiveRAG orchestration
│   ├── classifier.py           # Query intent classifier (4 categories)
│   ├── strategies.py           # 4 adaptive retrieval strategies
│   ├── llm_config.py           # LLM & embedding initialization (Groq, Cohere)
│   ├── pji_recommendation.py   # PJI recommendation engine with citations
│   └── data_completeness.py    # Deterministic data quality check (ICM 2018)
├── Dockerfile                  # Production image (python:3.11-slim)
├── docker-compose.yml          # Single-service orchestration
├── pyproject.toml              # uv project config
├── requirements.txt            # pip dependencies
└── .env                        # API keys (not committed)
```

## Tech Stack

| Component        | Technology                              | Notes                                    |
| ---------------- | --------------------------------------- | ---------------------------------------- |
| LLM (generation) | Groq — `llama-4-scout-17b-16e-instruct` | Temperature 0.1, Vietnamese system prompt |
| LLM (routing)    | Groq — `llama-3.1-8b-instant`          | Temperature 0.0, fast classification     |
| Embeddings       | Cohere `embed-multilingual-v3.0`        | Vietnamese + 100 languages               |
| Reranker         | Cohere `rerank-multilingual-v3.0`       | top_n=5                                  |
| Vector DB        | Zilliz Cloud (managed Milvus)           | Collection: `medical_rag_docs`           |
| Web search       | Tavily                                  | Fallback when <2 vector results          |
| Framework        | LangChain v0.3                          | Orchestration layer                      |
| Web server       | FastAPI + Uvicorn                       | Port 8000                                |
| PDF parsing      | PyPDF (PyPDFLoader)                     | Local extraction, no API                 |

## Environment Variables (.env)

```
GROQ_API_KEY=       # Groq LLM access
COHERE_API_KEY=     # Embeddings & reranking
ZILLIZ_URI=         # Zilliz Cloud endpoint
ZILLIZ_API_KEY=     # Zilliz Cloud auth
TAVILY_API_KEY=     # (Optional) Web search fallback
```

## Common Commands

```bash
# Install dependencies
uv sync

# Run dev server
uv run python3 api.py
# OR with hot reload:
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Ingest PDFs into vector DB (reads from data/ directory)
uv run python3 ingest_local.py

# Docker build & run
docker compose build
docker compose up -d
docker compose logs -f ai-backend
```

## Architecture & Data Flow

### Request Pipeline (POST /api/v1/process-snapshot)

1. **Data Completeness Check** — Deterministic (no LLM). Validates ICM 2018 major/minor criteria presence. Returns missing items with severity levels (CRITICAL/HIGH/MEDIUM).
2. **RAG Retrieval** — Builds query from snapshot → classifies intent → retrieves from Zilliz (k=15) → reranks with Cohere (top-5) → falls back to Tavily web search if needed.
3. **LLM Generation** — LLaMA 4 Scout 17B generates structured recommendations with citations.
4. **Output Formatting** — Returns `ai_recommendation_items` + `ai_rag_citations` + `data_completeness`.

### Adaptive Retrieval (4 strategies, auto-selected by classifier)

- **Factual** — Translates to English, dual search, deduplicates
- **Analytical** — Decomposes into 3 sub-questions, searches each, combines
- **Opinion** — Identifies 3 viewpoints, diversity selection
- **Contextual** — Reformulates query with patient context, dual search

### Recommendation Categories (4 types)

- `DIAGNOSTIC_TEST` — ICM 2018 scoring, infection classification
- `LOCAL_ANTIBIOTIC` — Cement spacer regimen, mixing ratios
- `SYSTEMIC_ANTIBIOTIC` — Multi-phase regimen (IV → oral), dosing
- `SURGERY_PROCEDURE` — Strategy (DAIR/1-stage/2-stage), staging, risks

## API Endpoints

### GET /health
Health check. Returns `{ "status": "ok", "rag_initialized": true }`.

### POST /api/v1/process-snapshot
Primary endpoint. Accepts clinical snapshot, returns structured recommendations + citations + data completeness assessment. See `api_contract.json` for full schema.

### POST /api/v1/chat
Interactive Q&A about PJI clinical decisions. Accepts question + optional episode summary, recommendation context, and chat history. Returns answer + references.

## Ingestion Pipeline (ingest_local.py)

1. Reads all PDFs from `data/` directory (recursive)
2. Splits text: `RecursiveCharacterTextSplitter` (chunk=1000, overlap=200)
3. Embeds with Cohere multilingual
4. Upserts to Zilliz Cloud (batch=50, drops old collection first — idempotent)

## Key Design Decisions

- **Dual-model approach**: Fast 8B model for routing/classification, powerful 17B for generation
- **Deterministic completeness check**: No LLM involved — pure rule-based ICM 2018 criteria validation
- **Structured JSON output**: All recommendations are strict JSON (not free text) for downstream integration
- **Citation traceability**: Every citation is linked to a specific recommendation item via `item_id`
- **Idempotent ingestion**: Drop-and-recreate collection on each run
- **CORS**: `allow_origins=["*"]` for web backend integration
- **Vietnamese-first**: System prompts and completeness messages in Vietnamese

## Testing

No test suite configured yet.

## Code Style

- Python with type hints
- Pydantic v2 for data validation
- LangChain abstractions for LLM/retriever orchestration
- Logging: LangChain Milvus suppressed to ERROR level; app logs to stdout
