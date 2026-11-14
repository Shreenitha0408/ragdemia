"""Incremental ingestion: only new/changed files are embedded; deletions are dropped."""

from __future__ import annotations

import shutil

import pytest

from ragdemia.chunking import whitespace_token_counter
from ragdemia.config import Config
from ragdemia.embeddings import HashingEmbedder
from ragdemia.ingest import ingest
from ragdemia.store import IndexStore

from .conftest import make_paper


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.embedded = 0

    def encode_documents(self, texts, show_progress=False):
        self.embedded += len(texts)
        return super().encode_documents(texts)


@pytest.fixture()
def setup(tmp_path):
    raw, idx = tmp_path / "raw", tmp_path / "index"
    raw.mkdir()
    make_paper(raw / "a.pdf")
    cfg = Config()
    cfg.chunking.max_tokens, cfg.chunking.overlap_tokens, cfg.chunking.min_tokens = 80, 20, 10
    emb = CountingEmbedder()

    def run(**kw):
        return ingest(cfg, emb, whitespace_token_counter, raw_dir=raw, index_dir=idx, **kw)

    return raw, idx, cfg, emb, run


def test_first_run_is_full_build(setup):
    raw, idx, _, emb, run = setup
    rep = run()
    assert rep.full_rebuild and rep.added == ["a"]
    assert emb.embedded == rep.n_chunks > 0
    assert IndexStore.load(idx).vectors.ntotal == rep.n_chunks


def test_unchanged_rerun_embeds_nothing(setup):
    _, _, _, emb, run = setup
    run()
    before = emb.embedded
    rep = run()
    assert rep.unchanged == ["a"] and not rep.changed
    assert emb.embedded == before


def test_add_update_remove(setup, tmp_path):
    raw, idx, _, emb, run = setup
    first = run()
    shutil.copy(raw / "a.pdf", raw / "b.pdf")
    rep = run()
    assert rep.added == ["b"] and rep.unchanged == ["a"]
    assert rep.n_embedded == first.n_chunks  # only b was embedded

    # modify a: append a page -> new checksum -> re-embedded
    import pymupdf

    doc = pymupdf.open(raw / "a.pdf")
    doc.new_page().insert_text((72, 72), "1 Appendix", fontsize=12, fontname="hebo")
    doc.saveIncr()
    doc.close()
    rep = run()
    assert rep.updated == ["a"] and rep.unchanged == ["b"]

    (raw / "b.pdf").unlink()
    rep = run()
    assert rep.removed == ["b"]
    store = IndexStore.load(idx)
    assert {c.doc_id for c in store.chunks} == {"a"}
    assert store.vectors.ntotal == len(store.chunks)


def test_reused_vectors_match_fresh_ones(setup):
    raw, idx, _, emb, run = setup
    run()
    shutil.copy(raw / "a.pdf", raw / "b.pdf")
    run()  # a's vectors come back out of FAISS
    store = IndexStore.load(idx)
    a_chunks = [i for i, c in enumerate(store.chunks) if c.doc_id == "a"]
    fresh = emb.encode_documents([store.chunks[i].embed_text for i in a_chunks])
    reused = store.vectors.reconstruct_n(0, store.vectors.ntotal)[a_chunks]
    assert abs(fresh - reused).max() < 1e-6


def test_settings_change_forces_full_rebuild(setup):
    _, _, cfg, _, run = setup
    run()
    cfg.chunking.max_tokens = 60
    rep = run()
    assert rep.full_rebuild and "settings" in rep.rebuild_reason
