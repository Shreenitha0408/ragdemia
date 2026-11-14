# Hugging Face Space (Docker SDK) for the public RAGDemia demo.
#
# The image clones the GitHub repository, downloads the 7 open-access sample papers, builds the
# index and caches the embedding + reranker models at BUILD time, so the Space starts ready.
# Answers are generated through an OpenAI-compatible API (Groq's free tier by default).
#
# Space settings -> Variables and secrets:
#   secret   GROQ_API_KEY                   LLM API key (console.groq.com, free)
#   secret   RAGDEMIA_ADMIN_TOKEN           your admin password for the shared library
#   secret   HF_TOKEN                       HF write token (stores admin uploads permanently)
#   variable RAGDEMIA_SERVER__PERSIST_REPO  e.g. <your-user>/ragdemia-library (private dataset)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /home/user/app

# CPU-only PyTorch: embeddings and reranking only; the LLM runs behind the API.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

ARG RAGDEMIA_REPO=https://github.com/Shreenitha0408/ragdemia.git
ARG RAGDEMIA_REF=main
# Changes whenever the branch gets a new commit, so a rebuild picks up the latest code.
ADD https://api.github.com/repos/Shreenitha0408/ragdemia/commits/${RAGDEMIA_REF} /tmp/ragdemia-version.json
RUN git clone --depth 1 --branch ${RAGDEMIA_REF} ${RAGDEMIA_REPO} . \
    && pip install --user -e ".[api]"

ENV RAGDEMIA_EMBEDDING__DEVICE=cpu
RUN python scripts/download_sample_corpus.py \
    && ragdemia ingest \
    && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

ENV RAGDEMIA_SERVER__PUBLIC=true \
    RAGDEMIA_LLM__BACKEND=openai \
    RAGDEMIA_LLM__BASE_URL=https://api.groq.com/openai/v1 \
    RAGDEMIA_LLM__MODEL=llama-3.1-8b-instant \
    RAGDEMIA_LLM__API_KEY_ENV=GROQ_API_KEY

EXPOSE 7860
CMD ["uvicorn", "ragdemia.api:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
