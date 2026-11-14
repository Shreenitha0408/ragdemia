# syntax=docker/dockerfile:1.7
# RAGDemia application image (UI + API). The LLM runs in a separate Ollama container.
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
RUN pip install -e ".[ui,api]"

COPY app ./app
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

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
