.PHONY: install sample ingest ui api eval test lint docker-up docker-gpu

install:      ## editable install with all extras
	pip install -e ".[api,dev]"
sample:       ## download the 7-paper sample corpus
	python scripts/download_sample_corpus.py
ingest:       ## build the index from data/raw
	ragdemia ingest
ui:           ## web app + API on :8000, opens the browser
	ragdemia ui
api:          ## web app + API on :8000
	ragdemia serve
eval:         ## retrieval ablation + RAG vs closed-book report
	ragdemia eval
test:
	pytest -q
lint:
	ruff check src tests app scripts && ruff format --check src tests app scripts
docker-up:
	docker compose up -d --build
docker-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
