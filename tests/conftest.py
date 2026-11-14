"""Shared fixtures. Everything here is hermetic: no model downloads, no network."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ragdemia.chunking import chunk_document
from ragdemia.embeddings import HashingEmbedder
from ragdemia.parsing import parse_pdf
from ragdemia.retrieval import Retriever
from ragdemia.store import IndexStore

BODY = (
    "Retrieval augmented generation combines a retriever with a generator. "
    "The retriever finds relevant passages for a question. "
)


def make_paper(path: Path) -> Path:
    """Write a small two-section 'paper' with a bold numbered heading per section."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    page.insert_text((72, y), "A Tiny Paper About Retrieval", fontsize=18, fontname="hebo")
    y += 40
    sections = [
        ("1 Introduction", "Dense retrievers embed questions and passages. " * 6),
        ("2 Method", "We train the encoder with in-batch negatives and a batch size of 128. " * 6),
        ("3 Results", "Our model reaches 78.4 percent top-20 accuracy on Natural Questions. " * 6),
        ("References", "[1] Some cited paper. [2] Another cited paper. " * 4),
    ]
    for heading, text in sections:
        page.insert_text((72, y), heading, fontsize=12, fontname="hebo")
        y += 20
        rect = pymupdf.Rect(72, y, 520, y + 120)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv")
        y += 130
        if y > 650:
            page = doc.new_page()
            y = 60
    doc.save(path)
    return path


@pytest.fixture()
def paper_pdf(tmp_path: Path) -> Path:
    return make_paper(tmp_path / "tiny.pdf")


@pytest.fixture()
def parsed(paper_pdf: Path):
    return parse_pdf(paper_pdf, skip_sections=["references"])


@pytest.fixture()
def store(parsed):
    chunks = chunk_document(parsed, max_tokens=80, overlap_tokens=20, min_tokens=10)
    emb = HashingEmbedder()
    vecs = emb.encode_documents([c.embed_text for c in chunks])
    return IndexStore.build(chunks, vecs, {"embedding_model": emb.model_name})


@pytest.fixture()
def retriever(store):
    return Retriever(store, HashingEmbedder(), mode="hybrid", candidates=10)
