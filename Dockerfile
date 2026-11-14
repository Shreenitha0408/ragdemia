# syntax=docker/dockerfile:1.7
# RAGDemia application image (web app + API). The LLM runs in a separate Ollama container.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller; embeddings and reranking are light enough.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[api]"

COPY config.yaml ./
COPY eval ./eval
COPY scripts ./scripts

# Bake the embedding model and reranker into the image so containers start offline.
ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"; fi

RUN useradd --create-home ragdemia && mkdir -p /app/data && chown -R ragdemia /app /models
USER ragdemia

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "ragdemia.api:app", "--host", "0.0.0.0", "--port", "8000"]
