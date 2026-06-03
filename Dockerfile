# Before It Breaks — Predictive Maintenance API (FastAPI + LSTM, CPU).
# Single-stage, non-root. Image tag: before-it-breaks:latest (rule: docker.image_name).
FROM python:3.12-slim

WORKDIR /app

# build-essential covers any transitive dep without a manylinux wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install the CPU-only torch wheel first so the PyPI resolver does not pull the
# CUDA build via requirements.txt (torch>=2.12.0 is satisfied by 2.12.0+cpu).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Runtime code + artefacts. ModelServer reads reports/results.json at startup
# (GET /api/v1/model_info), so it must be present or the container crashes on
# the first request — copy just the JSON, not the heavy figures.
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY reports/results.json ./reports/results.json

# Drop root before runtime.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
