# PJI Clinical Decision Support — Backend AI

Backend AI cho he thong ho tro quyet dinh lam sang **Nhiem trung Khop gia (Periprosthetic Joint Infection — PJI)**.

Nhan du lieu lam sang tu web backend, chay qua pipeline RAG, tra ve phac do dieu tri + trich dan bang chung + kiem tra du lieu thieu.

---

## Muc luc

- [Tong quan he thong](#tong-quan-he-thong)
- [Tech Stack](#tech-stack)
- [Cau truc du an](#cau-truc-du-an)
- [Huong dan cai dat (Development)](#huong-dan-cai-dat-development)
- [Nap tai lieu y khoa (PDF → Vector DB)](#nap-tai-lieu-y-khoa-pdf--vector-db)
- [Chay server (Development)](#chay-server-development)
- [Deploy bang Docker (Production)](#deploy-bang-docker-production)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Pipeline xu ly chi tiet](#pipeline-xu-ly-chi-tiet)

---

## Tong quan he thong

```
Web Backend                          AI Backend (repo nay)
───────────                          ────────────────────
snapshot_data_json ──── POST ────▶  /api/v1/process-snapshot
(du lieu lam sang)                         │
                                           ├─ [1] Data Completeness Check (deterministic, khong qua LLM)
                                           ├─ [2] RAG Retrieval (Adaptive Strategy + Cohere Reranker)
                                           └─ [3] LLM Generation (Llama 4 Scout 17B via Groq)
                                           │
                   ◀── Response ───────────┘
                   │
                   ├─ data_completeness        (du lieu thieu gi?)
                   ├─ ai_recommendation_items  (4 loai phac do)
                   └─ ai_rag_citations         (trich dan bang chung)
```

### Output tra ve cho web backend

| # | Response | Mo ta |
|---|----------|-------|
| 1 | `ai_recommendation_items` | 4 loai phac do: `DIAGNOSTIC_TEST`, `SYSTEMIC_ANTIBIOTIC`, `LOCAL_ANTIBIOTIC`, `SURGERY_PROCEDURE` |
| 2 | `ai_rag_citations` | Trich dan tai lieu bang chung lien ket den tung item qua `item_id` |
| 3 | `data_completeness` | Kiem tra deterministic du lieu dau vao thieu gi (`CRITICAL` / `HIGH` / `MEDIUM`) |

Chi tiet cau truc JSON xem file [`docs/api_contract.json`](docs/api_contract.json).

---

## Tech Stack

| Thanh phan | Cong nghe |
|---|---|
| LLM | Groq (Llama 3.1-8B cho routing, Llama 4 Scout 17B cho generation) |
| Embedding | Cohere embed-multilingual-v3.0 |
| Reranker | Cohere rerank-multilingual-v3.0 |
| Vector DB | Zilliz Cloud (managed Milvus) |
| Web Search | Tavily (fallback khi it tai lieu noi bo) |
| Framework | LangChain, FastAPI |
| Package Manager | uv |
| Containerization | Docker |

---

## Cau truc du an

```
Medical_RAG_PJI/
│
├── app/                               # Package chinh
│   ├── main.py                        # FastAPI app + lifespan (entry-point)
│   ├── config.py                      # Pydantic Settings tap trung
│   ├── dependencies.py                # FastAPI dependency injection
│   │
│   ├── api/routes/                    # API endpoints
│   │   ├── health.py                  # GET /health
│   │   ├── recommendation.py          # POST /api/v1/process-snapshot
│   │   └── chat.py                    # POST /api/v1/chat
│   │
│   ├── schemas/                       # Pydantic models (request/response)
│   │   ├── common.py                  # HealthResponse, ModelInfo
│   │   ├── request.py                 # ProcessSnapshotRequest, ChatRequest
│   │   ├── response.py                # ProcessSnapshotResponse, ChatResponse
│   │   └── completeness.py            # DataCompleteness, MissingItem
│   │
│   ├── core/                          # Business logic
│   │   ├── shared.py                  # SharedResources (LLM, VectorDB, Reranker)
│   │   ├── recommendation.py          # PJIRecommendationEngine
│   │   ├── completeness.py            # Kiem tra du lieu thieu (deterministic)
│   │   └── rag/
│   │       ├── retriever.py           # AdaptiveRetriever + AdaptiveRAG
│   │       ├── classifier.py          # QueryClassifier (4 loai)
│   │       └── strategies/            # 4 chien luoc truy xuat
│   │           ├── base.py
│   │           ├── factual.py
│   │           ├── analytical.py
│   │           ├── opinion.py
│   │           └── contextual.py
│   │
│   ├── llm/                           # LLM & Embedding wrappers
│   │   └── providers.py               # get_groq_llm, get_cohere_embeddings
│   │
│   └── prompts/                       # Prompt templates (text files)
│       ├── recommendation_system.txt
│       ├── chat_system.txt
│       ├── query_classifier.txt
│       └── ...
│
├── scripts/
│   └── ingest.py                      # Pipeline nap PDF vao Vector DB
│
├── tests/                             # Tests
│   ├── conftest.py
│   ├── test_completeness.py
│   ├── test_api.py
│   └── fixtures/
│       └── sample_snapshot.json
│
├── docs/
│   └── api_contract.json              # Mau JSON giao tiep voi web backend
│
├── data/                              # PDF y khoa (gitignored)
│
├── pyproject.toml                     # Dependencies (source of truth duy nhat)
├── uv.lock                            # Lock file
├── Dockerfile
├── docker-compose.yml
├── .env.example                       # Template API keys
└── .gitignore
```

---

## Huong dan cai dat (Development)

### Buoc 1 — Cai uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Buoc 2 — Clone repo & cai dependencies

```bash
git clone https://github.com/hoangtung386/Medical_RAG_PJI.git
cd Medical_RAG_PJI
uv sync
```

### Buoc 3 — Tao file `.env`

```bash
cp .env.example .env
# Sau do dien API keys vao file .env
```

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
ZILLIZ_URI=your_zilliz_cloud_uri
ZILLIZ_API_KEY=your_zilliz_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Nap tai lieu y khoa (PDF → Vector DB)

> **Buoc nay chi can lam 1 lan** (hoac khi them tai lieu moi).

### 1. Chuan bi PDF

Dat cac file PDF y khoa vao thu muc `data/`.

### 2. Chay pipeline ingest

```bash
uv run python -m scripts.ingest
```

Pipeline chay qua 4 buoc:
1. Doc PDF bang PyPDFLoader (local, mien phi)
2. Chia nho text thanh chunk 1000 ky tu (overlap 200)
3. Embedding bang Cohere `embed-multilingual-v3.0`
4. Insert batch vao Zilliz Cloud

---

## Chay server (Development)

```bash
# Chay truc tiep
uv run python -m app.main

# Hoac dung uvicorn (hot reload)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

---

## Deploy bang Docker (Production)

```bash
# Build
docker compose build

# Chay
docker compose up -d

# Xem logs
docker compose logs -f ai-backend

# Health check
curl http://localhost:8000/health

# Run
docker compose up -d

# Donwn:
docker compose down

# View logs
docker compose logs -f ai-backend
```

---

## API Endpoints

### `GET /health`

```json
{ "status": "ok", "rag_initialized": true }
```

### `POST /api/v1/process-snapshot`

Nhan `snapshot_data_json`, tra ve phac do + trich dan + kiem tra du lieu.

```bash
curl -X POST http://localhost:8000/api/v1/process-snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-001",
    "episode_id": 1001,
    "snapshot_id": 123,
    "snapshot_data_json": { ... }
  }'
```

### `POST /api/v1/chat`

Chat hoi dap voi AI ve ca benh PJI.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{ "question": "Co nen dung Vancomycin cho case nay?" }'
```

---

## Testing

```bash
# Chay toan bo tests
uv run pytest

# Chay voi verbose
uv run pytest -v

# Chay 1 file cu the
uv run pytest tests/test_completeness.py -v
```

---

## Pipeline xu ly chi tiet

```
snapshot_data_json (tu web backend)
       │
       ├──▶ [Data Completeness] ─── Deterministic check (khong qua LLM)
       │    Kiem tra thieu: sinus_tract, culture, CRP, ESR, WBC dich khop,
       │    Alpha-Defensin, histology, infection_type, implant_stability...
       │
       └──▶ [RAG Pipeline]
            │
            ├─ Build query tu snapshot (organism, joint, infection type, resistance)
            ├─ Query Classifier → chon strategy (Factual/Analytical/Opinion/Contextual)
            ├─ Adaptive Retrieval → tim tai lieu tu Zilliz
            ├─ Cohere Reranker → top 5 tai lieu lien quan nhat
            ├─ (Fallback) Tavily Web Search neu < 2 tai lieu
            │
            └─ LLM Generation (Llama 4 Scout 17B via Groq)
               ├─ DIAGNOSTIC_TEST:      ICM 2018 scoring, major/minor criteria
               ├─ LOCAL_ANTIBIOTIC:     Spacer khang sinh, ti le tron, monitoring
               ├─ SYSTEMIC_ANTIBIOTIC:  Phases (IV tan cong → uong duy tri)
               ├─ SURGERY_PROCEDURE:    Strategy (DAIR/1-stage/2-stage), risks
               └─ Citations:            Nguon tai lieu bang chung cho tung item
```

---

## License

Du an duoc phan phoi theo giay phep [MIT](LICENSE).
