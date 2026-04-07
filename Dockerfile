FROM python:3.11-slim

WORKDIR /app

# System dependencies for pymilvus/grpcio
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

RUN uv lock && uv sync --frozen --no-dev --no-editable

# Copy source code
COPY app/ app/
COPY scripts/ scripts/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
