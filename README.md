# PJI Clinical Decision Support — Backend API

Backend AI cho hệ thống hỗ trợ quyết định lâm sàng **Nhiễm trùng Khớp giả (Periprosthetic Joint Infection — PJI)**. Nhận dữ liệu lâm sàng từ web backend, chạy qua pipeline RAG, trả về phác đồ điều trị + trích dẫn bằng chứng + kiểm tra dữ liệu thiếu.

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
| 2 | `ai_rag_citations` | Trích dẫn tài liệu bằng chứng (guideline, meta-analysis, journal article...) liên kết đến từng recommendation item qua `item_id` |
| 3 | `data_completeness` | Kiểm tra deterministic (không qua LLM) dữ liệu đầu vào thiếu gì, phân loại theo `CRITICAL` / `HIGH` / `MEDIUM` |

Chi tiết cấu trúc JSON xem file [`api_contract.json`](api_contract.json).

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

## Cài đặt

### 1. Cài uv (package manager)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone repo

```bash
git clone https://github.com/hoangtung386/Medical_RAG_PJI.git
cd Medical_RAG_PJI
```

### 3. Cài đặt dependencies

```bash
uv sync
```

`uv sync` sẽ tự động:
- Tạo virtual environment `.venv/`
- Cài đặt đúng phiên bản Python (nếu chưa có)
- Cài toàn bộ dependencies từ `pyproject.toml` + `uv.lock`

### 4. Cấu hình environment variables

Tạo file `.env` tại thư mục gốc dự án:

```env
# === BẮT BUỘC ===
GROQ_API_KEY=your_groq_api_key           # LLM inference — https://console.groq.com/keys
COHERE_API_KEY=your_cohere_api_key       # Embedding + Reranker — https://dashboard.cohere.com/api-keys
ZILLIZ_URI=your_zilliz_cloud_uri         # Vector DB endpoint — https://cloud.zilliz.com
ZILLIZ_API_KEY=your_zilliz_api_key       # Vector DB auth token

# === TÙY CHỌN ===
TAVILY_API_KEY=your_tavily_api_key       # Web search fallback — https://tavily.com
```

## Nạp tài liệu y khoa (PDF → Vector DB)

Trước khi hệ thống AI có thể trả lời, cần nạp (ingest) các tài liệu y khoa PDF vào **Zilliz Cloud** (dịch vụ Vector Database trên cloud, dựa trên Milvus).

### Bước 1 — Tạo tài khoản Zilliz Cloud

1. Truy cập [https://cloud.zilliz.com](https://cloud.zilliz.com) và đăng ký tài khoản miễn phí
2. Tạo một **Cluster** mới (Free Tier đủ dùng cho dự án này)
3. Sau khi cluster khởi tạo xong, vào **Connect** để lấy:
   - **URI** (dạng `https://in03-xxxxxxx.api.gcp-us-west1.zillizcloud.com`) → điền vào `ZILLIZ_URI`
   - **API Key** → điền vào `ZILLIZ_API_KEY`
4. Không cần tạo collection thủ công — script ingest sẽ tự tạo collection `medical_rag_docs`

### Bước 2 — Chuẩn bị tài liệu PDF

Tạo thư mục `data/` và đặt các file PDF hướng dẫn y khoa (guidelines PJI, IDSA, ICM 2018...) vào trong:

```
Medical_RAG_PJI/
└── data/
    ├── ICM_2018_PJI_Criteria.pdf
    ├── IDSA_PJI_Guidelines_2013.pdf
    ├── Vancomycin_Dosing_ASHP_2020.pdf
    └── ... (các tài liệu y khoa khác)
```

### Bước 3 — Chạy pipeline ingest

```bash
uv run python3 ingest_local.py
```

**Pipeline chạy qua 4 bước:**
1. Đọc từng PDF bằng PyPDFLoader (local, miễn phí, không cần API key)
2. Chia nhỏ text thành chunk 1000 ký tự (overlap 200)
3. Embedding mỗi chunk bằng Cohere `embed-multilingual-v3.0`
4. Insert batch vào Zilliz Cloud (collection: `medical_rag_docs`)

### Bước 4 — Xác nhận dữ liệu đã được nạp

Sau khi ingest xong, vào Zilliz Cloud Console → chọn cluster → chọn collection `medical_rag_docs` → kiểm tra số lượng entities đã insert.

## Chạy server

```bash
# Chạy trực tiếp
uv run python3 api.py

# Hoặc dùng uvicorn (hot reload)
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Server khởi tạo xong sẽ in: `Khoi tao thanh cong!`

Truy cập tài liệu API tự động tại: [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "rag_initialized": true }
```

### `POST /api/v1/process-snapshot` — Xử lý dữ liệu lâm sàng (endpoint chính)

Nhận `snapshot_data_json` từ web backend, trả về phác đồ + trích dẫn + kiểm tra dữ liệu.

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

**Response chứa 3 phần chính:**

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

## Cấu trúc dự án

```
Medical_RAG_PJI/
├── api.py                         # FastAPI server — endpoint chính
├── api_contract.json              # Mẫu JSON giao tiếp với web backend
├── pyproject.toml                 # Cấu hình dự án & dependencies (dùng với uv)
├── uv.lock                       # Lock file (uv tự sinh, đảm bảo reproducible)
├── ingest_local.py                # Pipeline nạp PDF vào Vector DB
├── core/
│   ├── engine.py                  # AdaptiveRetriever + AdaptiveRAG
│   ├── classifier.py              # Phân loại ý định câu hỏi (4 loại)
│   ├── strategies.py              # 4 chiến lược truy xuất thích ứng
│   ├── llm_config.py              # Khởi tạo LLM (Groq) & Embedding (Cohere)
│   ├── pji_recommendation.py      # Engine sinh phác đồ PJI + citations
│   └── data_completeness.py       # Kiểm tra dữ liệu thiếu (deterministic)
├── data/                          # Thư mục chứa PDF y khoa (không push git)
├── .venv/                         # Virtual environment (uv tự tạo, không push git)
├── .env                           # API keys (không push git)
└── .gitignore
```

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

## License

Dự án được phân phối theo giấy phép [MIT](LICENSE).
