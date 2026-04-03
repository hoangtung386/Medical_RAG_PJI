# PJI Clinical Decision Support — Backend AI

Backend AI cho hệ thống hỗ trợ quyết định lâm sàng **Nhiễm trùng Khớp giả (Periprosthetic Joint Infection — PJI)**.

Nhận dữ liệu lâm sàng từ web backend, chạy qua pipeline RAG, trả về phác đồ điều trị + trích dẫn bằng chứng + kiểm tra dữ liệu thiếu.

---

## Mục lục

- [Tổng quan hệ thống](#tổng-quan-hệ-thống)
- [Tech Stack](#tech-stack)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Hướng dẫn cài đặt (Development)](#hướng-dẫn-cài-đặt-development)
- [Nạp tài liệu y khoa (PDF → Vector DB)](#nạp-tài-liệu-y-khoa-pdf--vector-db)
- [Chạy server (Development)](#chạy-server-development)
- [Deploy bằng Docker (Production)](#deploy-bằng-docker-production)
- [API Endpoints](#api-endpoints)
- [Pipeline xử lý chi tiết](#pipeline-xử-lý-chi-tiết)

---

## Tổng quan hệ thống

```
Web Backend                          AI Backend (repo này)
───────────                          ────────────────────
snapshot_data_json ──── POST ────▶  /api/v1/process-snapshot
(dữ liệu lâm sàng)                        │
                                           ├─ [1] Data Completeness Check (deterministic, không qua LLM)
                                           ├─ [2] RAG Retrieval (Adaptive Strategy + Cohere Reranker)
                                           └─ [3] LLM Generation (Llama 4 Scout 17B via Groq)
                                           │
                   ◀── Response ───────────┘
                   │
                   ├─ data_completeness        (dữ liệu thiếu gì?)
                   ├─ ai_recommendation_items  (4 loại phác đồ)
                   └─ ai_rag_citations         (trích dẫn bằng chứng)
```

### Output trả về cho web backend

| # | Response | Mô tả |
|---|----------|-------|
| 1 | `ai_recommendation_items` | 4 loại phác đồ: `DIAGNOSTIC_TEST` (chẩn đoán ICM 2018), `SYSTEMIC_ANTIBIOTIC` (kháng sinh toàn thân), `LOCAL_ANTIBIOTIC` (kháng sinh tại chỗ), `SURGERY_PROCEDURE` (phẫu thuật) |
| 2 | `ai_rag_citations` | Trích dẫn tài liệu bằng chứng (guideline, meta-analysis, journal article...) liên kết đến từng item qua `item_id` |
| 3 | `data_completeness` | Kiểm tra deterministic (không qua LLM) dữ liệu đầu vào thiếu gì, phân loại theo `CRITICAL` / `HIGH` / `MEDIUM` |

Chi tiết cấu trúc JSON xem file [`api_contract.json`](api_contract.json).

---

## Tech Stack

| Thành phần | Công nghệ |
|---|---|
| LLM | Groq (Llama 3.1-8B cho routing, Llama 4 Scout 17B cho generation) |
| Embedding | Cohere embed-multilingual-v3.0 |
| Reranker | Cohere rerank-multilingual-v3.0 |
| Vector DB | Zilliz Cloud (managed Milvus) |
| Web Search | Tavily (fallback khi ít tài liệu nội bộ) |
| Framework | LangChain, FastAPI |
| Package Manager | uv |
| Containerization | Docker |

---

## Cấu trúc dự án

```
Medical_RAG_PJI/
├── api.py                         # FastAPI server — endpoint chính
├── api_contract.json              # Mẫu JSON giao tiếp với web backend
├── ingest_local.py                # Pipeline nạp PDF vào Vector DB
│
├── core/
│   ├── engine.py                  # AdaptiveRetriever + AdaptiveRAG
│   ├── classifier.py              # Phân loại ý định câu hỏi (4 loại)
│   ├── strategies.py              # 4 chiến lược truy xuất thích ứng
│   ├── llm_config.py              # Khởi tạo LLM (Groq) & Embedding (Cohere)
│   ├── pji_recommendation.py      # Engine sinh phác đồ PJI + citations
│   └── data_completeness.py       # Kiểm tra dữ liệu thiếu (deterministic)
│
├── Dockerfile                     # Build image cho production
├── docker-compose.yml             # Orchestration (chạy chung với web backend)
├── .dockerignore                  # Loại file không cần khi build image
│
├── pyproject.toml                 # Cấu hình dự án & dependencies (dùng với uv)
├── uv.lock                       # Lock file (đảm bảo reproducible)
├── requirements.txt               # Dependencies cho pip / Docker
│
├── data/                          # Thư mục chứa PDF y khoa (không push git)
├── .env                           # API keys (không push git)
└── .gitignore
```

---

## Hướng dẫn cài đặt (Development)

### Bước 1 — Cài uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Bước 2 — Clone repo & cài dependencies

```bash
git clone https://github.com/hoangtung386/Medical_RAG_PJI.git
cd Medical_RAG_PJI
uv sync
```

`uv sync` sẽ tự động tạo `.venv/`, cài đúng Python và toàn bộ dependencies.

### Bước 3 — Tạo file `.env`

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
ZILLIZ_URI=your_zilliz_cloud_uri
ZILLIZ_API_KEY=your_zilliz_api_key
```

---

## Nạp tài liệu y khoa (PDF → Vector DB)

> **Bước này chỉ cần làm 1 lần** (hoặc khi thêm tài liệu mới). Sau khi nạp xong, dữ liệu nằm trên Zilliz Cloud — server API chỉ đọc từ đó.

### 1. Tạo tài khoản Zilliz Cloud

### 2. Chuẩn bị tài liệu PDF

```
Medical_RAG_PJI/
└── data/
    ├── ICM_2018_PJI_Criteria.pdf
    ├── IDSA_PJI_Guidelines_2013.pdf
    ├── Vancomycin_Dosing_ASHP_2020.pdf
    └── ... (các tài liệu y khoa khác)
```

### 3. Chạy pipeline ingest

```bash
uv run python3 ingest_local.py
```

Pipeline chạy qua 4 bước:
1. Đọc từng PDF bằng PyPDFLoader (local, miễn phí)
2. Chia nhỏ text thành chunk 1000 ký tự (overlap 200)
3. Embedding mỗi chunk bằng Cohere `embed-multilingual-v3.0`
4. Insert batch vào Zilliz Cloud (collection: `medical_rag_docs`)

### 4. Xác nhận

Vào Zilliz Cloud Console → chọn cluster → collection `medical_rag_docs` → kiểm tra **Loaded Entities** > 0.

---

## Chạy server (Development)

```bash
# Chạy trực tiếp
uv run python3 api.py

# Hoặc dùng uvicorn (hot reload khi sửa code)
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Server sẵn sàng khi in: `Khoi tao thanh cong!`

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Deploy bằng Docker (Production)

### Yêu cầu
- Docker & Docker Compose đã cài trên server

### Bước 1 — Chuẩn bị file `.env`

Tạo file `.env` tại thư mục gốc dự án (nội dung giống phần [Tạo file .env](#bước-3--tạo-file-env) ở trên). File này **không được push lên git** — cần tạo thủ công trên server.

### Bước 2 — Build Docker image

```bash
docker compose build
```

Image bao gồm: Python 3.11 + tất cả dependencies + source code. Không chứa `.env`, `data/`, hay `.venv/`.

### Bước 3 — Chạy container

```bash
# Chạy nền (detached)
docker compose up -d

# Xem logs
docker compose logs -f ai-backend

# Kiểm tra trạng thái
docker compose ps
```

### Bước 4 — Kiểm tra hoạt động

```bash
# Health check
curl http://localhost:8000/health
# → {"status":"ok","rag_initialized":true}
```

### Các lệnh Docker thường dùng

```bash
# Dừng container
docker compose down

# Rebuild khi có thay đổi code
docker compose build && docker compose up -d

# Xem logs realtime
docker compose logs -f ai-backend

# Restart container
docker compose restart ai-backend
```

### Tích hợp với web backend

Nếu web backend cũng chạy Docker, thêm service `ai-backend` vào file `docker-compose.yml` chung:

```yaml
services:
  # ... các service web backend khác ...

  ai-backend:
    build: ./Medical_RAG_PJI      # đường dẫn đến repo này
    container_name: pji-ai-backend
    ports:
      - "8000:8000"
    env_file:
      - ./Medical_RAG_PJI/.env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
```

Web backend gọi AI qua: `http://ai-backend:8000/api/v1/process-snapshot` (trong cùng Docker network) hoặc `http://localhost:8000/api/v1/process-snapshot` (từ host).

---

## API Endpoints

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "rag_initialized": true }
```

---

### `POST /api/v1/process-snapshot` — Xử lý dữ liệu lâm sàng (endpoint chính)

Nhận `snapshot_data_json` từ web backend, trả về phác đồ + trích dẫn + kiểm tra dữ liệu.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/process-snapshot \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-001",
    "episode_id": 1001,
    "snapshot_id": 123,
    "snapshot_data_json": {
      "snapshot_metadata": { "episode_id": 1001 },
      "patient_demographics": { "gender": "MALE" },
      "medical_history": {
        "medical_history": "Thay khop hang phai 3 nam truoc",
        "allergies": { "is_allergy": true, "allergy_note": "Di ung Penicillin" }
      },
      "clinical_records": {
        "symptoms": { "fever": true, "pain": true, "sinus_tract": false },
        "infection_assessment": {
          "suspected_infection_type": "CHRONIC",
          "implant_stability": "UNSTABLE",
          "prosthesis_joint": "HIP_RIGHT"
        }
      },
      "lab_results": {
        "latest": {
          "inflammatory_markers_blood": { "crp": 95.3, "esr": 85, "alpha_defensin": "POSITIVE" },
          "synovial_fluid": { "synovial_wbc": 52000, "synovial_pmn": 92.0 },
          "biochemical_data": { "creatinine": 95, "alt": 28, "ast": 32 }
        }
      },
      "culture_results": {
        "items": [
          { "name": "Staphylococcus aureus", "result_status": "POSITIVE", "sensitivities": [] },
          { "organism_name": "Staphylococcus aureus", "result_status": "POSITIVE", "sensitivities": [] }
        ]
      }
    }
  }'
```

**Response:**

```json
{
  "request_id": "req-001",
  "status": "SUCCESS",
  "model": { "name": "rag-llm", "version": "v1" },
  "latency_ms": 5432,
  "run_id": "run-uuid-xxx",
  "data_completeness": {
    "is_complete": false,
    "missing_items": [
      { "field": "synovial_LE", "category": "ICM_MINOR", "importance": "MEDIUM", "message": "..." }
    ],
    "completeness_score": "7/9 ICM minor criteria co du lieu",
    "impact_note": "..."
  },
  "ai_recommendation_items": [
    {
      "id": "item-uuid-xxx",
      "category": "DIAGNOSTIC_TEST",
      "title": "...",
      "item_json": { "scoring_system": {}, "major_criteria": {}, "minor_criteria_scoring": {}, "ai_reasoning": {} }
    },
    { "id": "...", "category": "LOCAL_ANTIBIOTIC", "title": "...", "item_json": {} },
    { "id": "...", "category": "SYSTEMIC_ANTIBIOTIC", "title": "...", "item_json": {} },
    { "id": "...", "category": "SURGERY_PROCEDURE", "title": "...", "item_json": {} }
  ],
  "ai_rag_citations": [
    {
      "id": "cit-uuid-xxx",
      "run_id": "run-uuid-xxx",
      "item_id": "item-uuid-xxx",
      "source_type": "GUIDELINE",
      "source_title": "ICM 2018 Definition of PJI...",
      "source_uri": "https://doi.org/...",
      "snippet": "...",
      "relevance_score": 0.98,
      "cited_for": "..."
    }
  ]
}
```

---

### `POST /api/v1/chat` — Chat hỏi đáp với AI

Dùng cho bác sĩ hỏi thêm về ca bệnh sau khi đã có recommendation.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Co nen dung Vancomycin cho case nay?",
    "episode_summary": {},
    "recommendation_context": {},
    "chat_history": []
  }'
```

---

## Pipeline xử lý chi tiết

```
snapshot_data_json (từ web backend)
       │
       ├──▶ [Data Completeness] ─── Deterministic check (không qua LLM)
       │    Kiểm tra thiếu: sinus_tract, culture, CRP, ESR, WBC dịch khớp,
       │    Alpha-Defensin, histology, infection_type, implant_stability...
       │
       └──▶ [RAG Pipeline]
            │
            ├─ Build query từ snapshot (organism, joint, infection type, resistance)
            ├─ Query Classifier → chọn strategy (Factual/Analytical/Opinion/Contextual)
            ├─ Adaptive Retrieval → tìm tài liệu từ Zilliz
            ├─ Cohere Reranker → top 5 tài liệu liên quan nhất
            ├─ (Fallback) Tavily Web Search nếu < 2 tài liệu
            │
            └─ LLM Generation (Llama 4 Scout 17B via Groq)
               ├─ DIAGNOSTIC_TEST:      ICM 2018 scoring, major/minor criteria, reasoning
               ├─ LOCAL_ANTIBIOTIC:     Spacer kháng sinh, tỉ lệ trộn, monitoring
               ├─ SYSTEMIC_ANTIBIOTIC:  Phases (IV tấn công → uống duy trì), monitoring
               ├─ SURGERY_PROCEDURE:    Strategy (DAIR/1-stage/2-stage), stages, risks
               └─ Citations:            Nguồn tài liệu bằng chứng cho từng item
```

---

## License

Dự án được phân phối theo giấy phép [MIT](LICENSE).
