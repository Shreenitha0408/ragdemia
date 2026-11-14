"""Public-deployment mode: private visitor workspaces, admin-only shared library, rate limits."""

from __future__ import annotations

import uuid

import pymupdf
import pytest
from fastapi.testclient import TestClient

from ragdemia.api import create_app
from ragdemia.chunking import whitespace_token_counter
from ragdemia.config import Config, ServerConfig
from ragdemia.embeddings import HashingEmbedder
from ragdemia.ingest import ingest
from ragdemia.library import Library
from ragdemia.llm import EchoLLM
from ragdemia.pipeline import RAGPipeline
from ragdemia.retrieval import Retriever
from ragdemia.store import IndexStore
from ragdemia.workspaces import WorkspaceManager

from .conftest import make_paper

ADMIN = "correct horse battery staple"


def visitor_pdf(path, text="Quantum annealing schedules for zebra migration."):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Visitor Notes on Zebras", fontsize=18, fontname="hebo")
    page.insert_text((72, 110), "1 Findings", fontsize=12, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 120, 520, 400), (text + " ") * 8, fontsize=10)
    doc.save(path)
    return path.read_bytes()


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    def _make(rate=100, max_docs=2, admin=ADMIN):
        if admin:
            monkeypatch.setenv("RAGDEMIA_ADMIN_TOKEN", admin)
        else:
            monkeypatch.delenv("RAGDEMIA_ADMIN_TOKEN", raising=False)
        raw, idx = tmp_path / "raw", tmp_path / "index"
        (raw / "sample").mkdir(parents=True, exist_ok=True)
        make_paper(raw / "sample" / "paper.pdf")
        cfg = Config()
        cfg.paths.raw_dir, cfg.paths.index_dir = str(raw), str(idx)
        cfg.chunking.max_tokens, cfg.chunking.overlap_tokens, cfg.chunking.min_tokens = 80, 20, 10
        emb = HashingEmbedder()
        ingest(cfg, emb, whitespace_token_counter)
        pipe = RAGPipeline(
            Retriever(IndexStore.load(idx), emb), EchoLLM("Zebras [1]."), index_dir=str(idx)
        )
        server = ServerConfig(public=True, rate_limit_per_minute=rate, workspace_max_docs=max_docs)
        ws = WorkspaceManager(cfg, emb, whitespace_token_counter, max_docs=max_docs)
        app = create_app(
            pipe, Library(cfg, pipe, whitespace_token_counter), server=server, workspaces=ws
        )
        return TestClient(app)

    return _make


def wsh():
    return {"X-Workspace": str(uuid.uuid4())}


def test_visitor_upload_is_private_to_its_workspace(make_client, tmp_path):
    with make_client() as c:
        alice, bob = wsh(), wsh()
        data = visitor_pdf(tmp_path / "z.pdf")
        r = c.post(
            "/documents", files=[("files", ("zebra.pdf", data, "application/pdf"))], headers=alice
        )
        assert r.status_code == 201 and r.json()["scope"] == "workspace"
        doc_id = r.json()["added"][0]
        assert doc_id.startswith("workspace/")

        alice_docs = {d["doc_id"]: d for d in c.get("/documents", headers=alice).json()}
        bob_docs = {d["doc_id"] for d in c.get("/documents", headers=bob).json()}
        assert doc_id in alice_docs and alice_docs[doc_id]["deletable"]
        assert doc_id not in bob_docs and "sample/paper" in bob_docs

        q = {"query": "quantum annealing zebra migration", "top_k": 3}
        assert c.post("/retrieve", json=q, headers=alice).json()[0]["doc_id"] == doc_id
        assert all(r["doc_id"] != doc_id for r in c.post("/retrieve", json=q, headers=bob).json())

        assert c.delete(f"/documents/{doc_id}", headers=bob).status_code == 404
        assert c.delete(f"/documents/{doc_id}", headers=alice).status_code == 200
        assert c.get("/documents", headers=alice).json()[0]["doc_id"] == "sample/paper"


def test_fresh_visitor_gets_workspace_limits(make_client):
    with make_client() as c:  # no uploads yet: the manager is empty but must still report
        w = c.get("/health", headers=wsh()).json()["workspace"]
        assert w == {"ttl_hours": 6.0, "max_docs": 2, "max_mb": 20, "n_documents": 0}


def test_shared_library_is_admin_only(make_client, tmp_path):
    with make_client() as c:
        h = c.get("/health").json()
        assert h["public"] and not h["is_admin"] and not h["can_edit_shared"]
        assert c.get("/health", headers={"X-Admin-Token": ADMIN}).json()["is_admin"]
        assert not c.get("/health", headers={"X-Admin-Token": "wrong"}).json()["is_admin"]

        data = visitor_pdf(tmp_path / "a.pdf", "Shared lecture on attention heads.")
        r = c.post(
            "/documents",
            files=[("files", ("lec.pdf", data, "application/pdf"))],
            headers={"X-Admin-Token": ADMIN, **wsh()},
        )
        assert r.status_code == 201 and r.json()["scope"] == "shared"
        assert r.json()["added"] == ["uploads/lec"]
        # a visitor sees it, but cannot delete it
        docs = {d["doc_id"]: d for d in c.get("/documents", headers=wsh()).json()}
        assert docs["uploads/lec"]["scope"] == "shared" and not docs["uploads/lec"]["deletable"]
        assert c.delete("/documents/uploads/lec").status_code == 403
        assert (
            c.delete("/documents/uploads/lec", headers={"X-Admin-Token": ADMIN}).status_code == 200
        )


def test_upload_without_workspace_or_admin_is_rejected(make_client, tmp_path):
    with make_client(admin="") as c:
        assert not c.get("/health").json()["admin_enabled"]
        data = visitor_pdf(tmp_path / "z.pdf")
        r = c.post("/documents", files=[("files", ("z.pdf", data, "application/pdf"))])
        assert r.status_code == 400 and "workspace" in r.json()["detail"]
        # an empty admin token must never grant admin
        r = c.post(
            "/documents",
            files=[("files", ("z.pdf", data, "application/pdf"))],
            headers={"X-Admin-Token": ""},
        )
        assert r.status_code == 400


def test_workspace_document_limit(make_client, tmp_path):
    with make_client(max_docs=1) as c:
        h = wsh()
        a = visitor_pdf(tmp_path / "a.pdf", "First upload about lions.")
        b = visitor_pdf(tmp_path / "b.pdf", "Second upload about tigers.")
        assert (
            c.post(
                "/documents", files=[("files", ("a.pdf", a, "application/pdf"))], headers=h
            ).status_code
            == 201
        )
        r = c.post("/documents", files=[("files", ("b.pdf", b, "application/pdf"))], headers=h)
        assert r.status_code == 400 and "already holds 1" in r.json()["detail"]


def test_rate_limit(make_client):
    with make_client(rate=2) as c:
        q = {"question": "What batch size?"}
        assert c.post("/ask", json=q).status_code == 200
        assert c.post("/ask", json=q).status_code == 200
        r = c.post("/ask", json=q)
        assert r.status_code == 429 and "rate-limited" in r.json()["detail"]
