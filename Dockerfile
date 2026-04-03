FROM python:3.11-slim

WORKDIR /app

# System dependencies for pymilvus/grpcio
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api.py .
COPY ingest_local.py .
COPY core/ core/
COPY api_contract.json .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
