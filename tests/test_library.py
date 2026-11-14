import pytest

from ragdemia.chunking import whitespace_token_counter
from ragdemia.config import Config
from ragdemia.embeddings import HashingEmbedder
from ragdemia.ingest import ingest
from ragdemia.library import Library, LibraryError, safe_filename
from ragdemia.llm import EchoLLM
from ragdemia.pipeline import RAGPipeline
from ragdemia.retrieval import Retriever
from ragdemia.store import IndexStore

from .conftest import make_paper


@pytest.fixture()
def lib(tmp_path):
    raw, idx = tmp_path / "raw", tmp_path / "index"
    (raw / "private").mkdir(parents=True)
    make_paper(raw / "private" / "mine.pdf")
    cfg = Config()
    cfg.paths.raw_dir, cfg.paths.index_dir = str(raw), str(idx)
    cfg.chunking.max_tokens, cfg.chunking.overlap_tokens, cfg.chunking.min_tokens = 80, 20, 10
    emb = HashingEmbedder()
    ingest(cfg, emb, whitespace_token_counter)
    pipe = RAGPipeline(Retriever(IndexStore.load(idx), emb), EchoLLM("ok [1]."), index_dir=str(idx))
    return Library(cfg, pipe, whitespace_token_counter), tmp_path


def test_safe_filename_blocks_traversal():
    assert safe_filename("../../etc/passwd") == "passwd.pdf"
    assert safe_filename("My Notes (v2).PDF") == "My_Notes_v2.pdf"


def test_upload_index_and_delete(lib):
    library, tmp = lib
    data = (tmp / "raw" / "private" / "mine.pdf").read_bytes()
    with pytest.raises(LibraryError, match="already in the library"):
        library.add("copy.pdf", data)

    make_paper(tmp / "other.pdf")
    import pymupdf

    doc = pymupdf.open(tmp / "other.pdf")
    doc.new_page().insert_text((72, 72), "Extra page about tokenizers", fontsize=10)
    doc.saveIncr()
    doc.close()
    path = library.add("Lecture 3.pdf", (tmp / "other.pdf").read_bytes())
    report = library.sync()
    assert report.added == ["uploads/Lecture_3"]
    assert {d["doc_id"] for d in library.documents} == {"private/mine", "uploads/Lecture_3"}
    # the live pipeline sees the new document without being rebuilt
    assert any(c.doc_id == "uploads/Lecture_3" for c in library.pipeline.retriever.store.chunks)

    with pytest.raises(LibraryError, match="added on disk"):
        library.remove("private/mine")
    library.remove("uploads/Lecture_3")
    assert not path.exists()
    assert library.sync().removed == ["uploads/Lecture_3"]


def test_rejects_non_pdf(lib):
    library, _ = lib
    with pytest.raises(LibraryError, match="not a PDF"):
        library.add("evil.pdf", b"MZ\x90\x00 not a pdf")
