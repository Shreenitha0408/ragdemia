# Self-contained PUBLIC DEMO image: code + 7 open-access sample papers + prebuilt index + cached
# embedding/reranker models, configured for public mode with an OpenAI-compatible LLM API.
# Built by .github/workflows/demo-image.yml and published as ghcr.io/<owner>/ragdemia-demo.
# Runs on any container host (Azure Container Apps, Google Cloud Run, a VM with Docker).
#
# Runtime configuration (environment variables; keys are injected by the host as secrets):
#   LLM_API_KEY             API key for the LLM endpoint (Groq by default)
#   RAGDEMIA_ADMIN_TOKEN    admin password for managing the shared library
#   optional: RAGDEMIA_LLM__BASE_URL, RAGDEMIA_LLM__MODEL to use another provider/model
# Admin uploads persist via AZURE_STORAGE_CONNECTION_STRING (Blob Storage), or a volume
# mounted at /app/data/raw/uploads.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf TOKENIZERS_PARALLELISM=false
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md config.yaml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install -e ".[api,azure]"

# Bake the corpus, the index and the models so a cold start only has to load them.
ENV RAGDEMIA_EMBEDDING__DEVICE=cpu
RUN python scripts/download_sample_corpus.py \
    && ragdemia ingest \
    && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')" \
    && mkdir -p data/raw/uploads

ENV RAGDEMIA_SERVER__PUBLIC=true \
    RAGDEMIA_LLM__BACKEND=openai \
    RAGDEMIA_LLM__BASE_URL=https://api.groq.com/openai/v1 \
    RAGDEMIA_LLM__MODEL=llama-3.1-8b-instant \
    RAGDEMIA_LLM__API_KEY_ENV=LLM_API_KEY \
    HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
# Runs as root so a mounted cloud file share (SMB, root-owned) is writable for admin uploads.
CMD ["uvicorn", "ragdemia.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
