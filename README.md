# Medical Adaptive RAG

Hệ thống **Retrieval-Augmented Generation (RAG)** thông minh cho lĩnh vực y khoa, tự động lựa chọn chiến lược truy xuất phù hợp dựa trên ý định của câu hỏi. Hỗ trợ cả tiếng Việt và tiếng Anh.

![RAG Architecture](Medical_RAG_PJI/rag_architecture.png)

## Tính năng chính

- **Adaptive Retrieval** — Tự động phân loại câu hỏi thành 4 loại (Factual, Analytical, Opinion, Contextual) và áp dụng chiến lược truy xuất riêng cho từng loại
- **Đa ngôn ngữ** — Hỗ trợ tiếng Việt & tiếng Anh nhờ Cohere Multilingual Embeddings
- **Web Grounding** — Tích hợp Tavily Search khi tài liệu nội bộ không đủ
- **Reranking** — Sử dụng Cohere Reranker để chọn tài liệu liên quan nhất
- **2 giao diện** — Gradio Web UI + FastAPI REST API

## Kiến trúc

```
User Query
  │
  ▼
Query Classifier (Llama 3.1-8B) → Phân loại ý định
  │
  ├─ Factual    → Tăng cường query + tìm kiếm kép
  ├─ Analytical → Tách thành 3 sub-query, tìm đa chiều
  ├─ Opinion    → Xác định 3 góc nhìn, truy xuất đa dạng
  └─ Contextual → Kết hợp thông tin bệnh nhân vào query
  │
  ▼
Cohere Reranker → Top 5 tài liệu
  │
  ▼ (nếu < 2 docs & web search ON)
Tavily Web Search
  │
  ▼
Groq LLM (Llama 4 Scout) → Sinh câu trả lời tiếng Việt + nguồn trích dẫn
```

## Tech Stack

| Thành phần | Công nghệ |
|---|---|
| LLM | Groq (Llama 3.1-8B, Llama 4 Scout) |
| Embedding | Cohere embed-multilingual-v3.0 |
| Reranker | Cohere rerank-multilingual-v3.0 |
| Vector DB | Milvus / Zilliz Cloud |
| Web Search | Tavily |
| PDF Parsing | Google Gemini Flash / HuggingFace Vision |
| Framework | LangChain |
| UI | Gradio, FastAPI |

## Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/hoangtung386/Medical_RAG_PJI.git
cd Medical_RAG_PJI
```

### 2. Tạo virtual environment

```bash
cd Medical_RAG_PJI
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình environment variables

Tạo file `Medical_RAG_PJI/.env`:

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
ZILLIZ_URI=your_zilliz_cloud_uri
ZILLIZ_API_KEY=your_zilliz_api_key
TAVILY_API_KEY=your_tavily_api_key        # Optional, cho web search
GEMINI_API_KEY=your_gemini_api_key        # Cho ingestion bằng Gemini
HF_TOKEN=your_huggingface_token           # Cho ingestion bằng HF Vision
```

## Sử dụng

### Ingest dữ liệu (PDF → Vector DB)

```bash
# Option A: Dùng Gemini Flash (khuyến nghị)
python ingest_gemini.py

# Option B: Dùng HuggingFace Vision OCR
python ingest_hf.py
```

Đặt file PDF y khoa vào folder `Data/` trước khi chạy.

### Chạy ứng dụng

**Gradio Web UI:**

```bash
python app.py
# Mở trình duyệt tại http://localhost:7860
```

**FastAPI REST API:**

```bash
python api.py
# API docs tại http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ask` | Gửi câu hỏi y khoa |

**Ví dụ request:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Chỉ số CRP bao nhiêu thì nghi ngờ nhiễm khuẩn khớp?",
    "user_context": "Bệnh nhân nam 60 tuổi, sau phẫu thuật thay khớp",
    "use_web_search": true
  }'
```

## Cấu trúc dự án

```
Medical_RAG_PJI/
├── Data/                      # Tài liệu PDF y khoa (không được push lên git)
├── medical_rag/
│   ├── core/
│   │   ├── llm_config.py     # Khởi tạo LLM, Embedding, Reranker
│   │   ├── classifier.py     # Phân loại ý định câu hỏi
│   │   ├── strategies.py     # 4 chiến lược truy xuất
│   │   └── engine.py         # AdaptiveRetriever & AdaptiveRAG
│   ├── app.py                # Gradio chatbot UI
│   ├── api.py                # FastAPI REST server
│   ├── ingest_gemini.py      # Pipeline ingest bằng Gemini
│   ├── ingest_hf.py          # Pipeline ingest bằng HF Vision
│   └── requirements.txt      # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## License

Dự án được phân phối theo giấy phép [MIT](LICENSE).
