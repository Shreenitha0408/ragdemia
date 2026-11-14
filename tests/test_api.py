"""API + web app, wired to hermetic fakes (no model downloads)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ragdemia.api import create_app
from ragdemia.chunking import whitespace_token_counter
from ragdemia.config import Config
from ragdemia.embeddings import HashingEmbedder
from ragdemia.ingest import ingest
from ragdemia.library import Library
from ragdemia.llm import EchoLLM
from ragdemia.pipeline import RAGPipeline
from ragdemia.retrieval import Retriever
from ragdemia.store import IndexStore

from .conftest import make_paper


@pytest.fixture()
def client(tmp_path):
    raw, idx = tmp_path / "raw", tmp_path / "index"
    (raw / "private").mkdir(parents=True)
    make_paper(raw / "private" / "paper.pdf")
    cfg = Config()
    cfg.paths.raw_dir, cfg.paths.index_dir = str(raw), str(idx)
    cfg.chunking.max_tokens, cfg.chunking.overlap_tokens, cfg.chunking.min_tokens = 80, 20, 10
    emb = HashingEmbedder()
    ingest(cfg, emb, whitespace_token_counter)
    llm = EchoLLM("The encoder used a batch size of 128 [1].")
    pipe = RAGPipeline(Retriever(IndexStore.load(idx), emb), llm, top_k=3, index_dir=str(idx))
    app = create_app(pipe, Library(cfg, pipe, whitespace_token_counter))
    with TestClient(app) as c:
        yield c


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_web_app_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "RAGDemia" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_health_and_documents(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["n_documents"] == 1 and h["defaults"]["top_k"] == 3
    docs = client.get("/documents").json()
    assert docs[0]["doc_id"] == "private/paper" and docs[0]["deletable"] is False


def test_ask_stream_event_order(client):
    r = client.post("/ask/stream", json={"question": "What batch size was used?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and "token" in kinds
    meta, done = events[0][1], events[-1][1]
    assert len(meta["sources"]) == 3
    assert "".join(d["t"] for e, d in events if e == "token").strip() == done["answer"]
    assert done["citations"]["cited"] == [1]


def test_upload_and_delete_roundtrip(client, tmp_path):
    make_paper(tmp_path / "new.pdf")
    import pymupdf

    doc = pymupdf.open(tmp_path / "new.pdf")
    doc.new_page().insert_text((72, 72), "A different page so the checksum differs", fontsize=10)
    doc.saveIncr()
    doc.close()
    with open(tmp_path / "new.pdf", "rb") as fh:
        r = client.post("/documents", files=[("files", ("Lecture 9.pdf", fh, "application/pdf"))])
    assert r.status_code == 201 and r.json()["added"] == ["uploads/Lecture_9"]
    assert client.get("/health").json()["n_documents"] == 2
    assert client.delete("/documents/private/paper").status_code == 403
    r = client.delete("/documents/uploads/Lecture_9")
    assert r.status_code == 200 and r.json()["removed"] == ["uploads/Lecture_9"]


def test_upload_rejects_non_pdf(client):
    r = client.post("/documents", files=[("files", ("x.pdf", b"not a pdf", "application/pdf"))])
    assert r.status_code == 400 and "not a PDF" in r.json()["detail"]
