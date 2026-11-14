"""Ingestion: PDFs -> parsed sections -> chunks -> embeddings -> on-disk index.

Indexing is **incremental**.  The folder of PDFs is the source of truth; every run
reconciles the index with it:

* a file whose SHA-256 checksum matches the manifest is *unchanged* - its chunks and vectors
  are reused (vectors are read back out of FAISS, so nothing is re-embedded);
* a new or modified file is parsed, chunked and embedded;
* a file that disappeared from the folder is dropped from the index.

Reuse is only valid if the old chunks would be produced identically today, so the manifest
stores a **build signature**: a hash of the parser version, the parsing and chunking settings
and the embedding model.  If any of these changed, the run silently becomes a full rebuild.

FAISS and BM25 are always rebuilt from the final chunk list; with a few thousand chunks that
takes milliseconds, and it keeps row ``i`` of every structure pointing at the same chunk.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunking import Chunk, TokenCounter, chunk_document
from .config import Config
from .embeddings import Embedder
from .parsing import PARSER_VERSION, discover_pdfs, file_sha256, make_doc_id, parse_pdf
from .store import IndexStore


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)  # doc_ids
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    full_rebuild: bool = False
    rebuild_reason: str = ""
    documents: list[dict] = field(default_factory=list)  # manifest entries after the run
    n_chunks: int = 0
    n_embedded: int = 0  # chunks that actually went through the embedding model
    seconds: float = 0.0
    index_dir: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.full_rebuild)


def build_signature(cfg: Config, embedding_model: str) -> str:
    """Everything that determines chunk text and vectors. A change forces a full rebuild."""
    relevant = {
        "parser_version": PARSER_VERSION,
        "parsing": cfg.to_dict()["parsing"],
        "chunking": cfg.to_dict()["chunking"],
        "embedding_model": embedding_model,
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def _load_existing(index_dir: Path) -> IndexStore | None:
    try:
        return IndexStore.load(index_dir)
    except FileNotFoundError:
        return None


def ingest(
    cfg: Config,
    embedder: Embedder,
    count_tokens: TokenCounter,
    *,
    raw_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    full: bool = False,
    allow_empty: bool = False,
) -> IngestReport:
    """Reconcile the index with the PDFs under ``raw_dir``. ``full=True`` re-processes all."""
    t0 = time.perf_counter()
    raw = Path(raw_dir or cfg.paths.raw_dir)
    out = Path(index_dir or cfg.paths.index_dir)
    say = progress or (lambda _m: None)
    report = IngestReport(index_dir=str(out))

    pdfs = discover_pdfs(raw)
    if not pdfs and not allow_empty:
        raise FileNotFoundError(f"No PDFs found under '{raw}'.")

    signature = build_signature(cfg, embedder.model_name)
    existing = None if full else _load_existing(out)
    if full:
        report.full_rebuild, report.rebuild_reason = True, "requested"
    elif existing is None:
        report.full_rebuild, report.rebuild_reason = True, "no existing index"
    elif existing.manifest.get("build_signature") != signature:
        report.full_rebuild, report.rebuild_reason = (
            True,
            "parser/chunking/embedding settings changed",
        )
        existing = None

    # ---- what does the index currently hold?
    old_docs: dict[str, dict] = {}
    old_chunks: dict[str, list[tuple[Chunk, np.ndarray]]] = {}
    if existing is not None:
        old_docs = {d["doc_id"]: d for d in existing.manifest.get("documents", [])}
        vectors = (
            existing.vectors.reconstruct_n(0, existing.vectors.ntotal)
            if existing.vectors.ntotal
            else np.zeros((0, embedder.dim), "float32")
        )
        for chunk, vec in zip(existing.chunks, vectors, strict=True):
            old_chunks.setdefault(chunk.doc_id, []).append((chunk, vec))

    # ---- reconcile with the folder
    kept: list[tuple[Chunk, np.ndarray]] = []
    new_chunks: list[Chunk] = []
    doc_entries: list[dict] = []
    seen: set[str] = set()
    for pdf in pdfs:
        doc_id = make_doc_id(pdf, raw)
        seen.add(doc_id)
        sha = file_sha256(pdf)
        prev = old_docs.get(doc_id)
        if prev and prev["sha256"] == sha and doc_id in old_chunks:
            kept.extend(old_chunks[doc_id])
            doc_entries.append(prev)
            report.unchanged.append(doc_id)
            continue

        say(f"parsing {pdf.name}")
        try:
            doc = parse_pdf(
                pdf,
                root=raw,
                min_font_ratio=cfg.parsing.min_font_ratio,
                skip_sections=cfg.parsing.skip_sections,
            )
        except Exception as exc:  # a corrupt PDF must not kill the whole ingest
            report.failures.append((str(pdf), repr(exc)))
            continue
        if doc.n_chars == 0:
            report.failures.append((str(pdf), "no extractable text (scanned PDF? needs OCR)"))
            continue
        chunks = chunk_document(
            doc,
            max_tokens=cfg.chunking.max_tokens,
            overlap_tokens=cfg.chunking.overlap_tokens,
            min_tokens=cfg.chunking.min_tokens,
            count_tokens=count_tokens,
        )
        new_chunks.extend(chunks)
        (report.updated if prev else report.added).append(doc_id)
        doc_entries.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source_path": doc.source_path,
                "sha256": doc.sha256,
                "num_pages": doc.num_pages,
                "n_sections": len(doc.sections),
                "n_chunks": len(chunks),
                "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    report.removed = sorted(set(old_docs) - seen)

    # ---- embed only what is new
    if new_chunks:
        say(f"embedding {len(new_chunks)} chunks with {embedder.model_name}")
        new_vecs = embedder.encode_documents(
            [c.embed_text for c in new_chunks], show_progress=progress is not None
        )
    else:
        new_vecs = np.zeros((0, embedder.dim), "float32")
    report.n_embedded = len(new_chunks)

    all_chunks = [c for c, _ in kept] + new_chunks
    parts = [np.stack([v for _, v in kept])] if kept else []
    all_vecs = np.concatenate(parts + [new_vecs]) if parts else new_vecs

    if existing is not None and not report.changed:
        say("index is up to date")
        report.documents, report.n_chunks = doc_entries, len(all_chunks)
        report.seconds = time.perf_counter() - t0
        return report

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_signature": signature,
        "embedding_model": embedder.model_name,
        "embedding_dim": int(embedder.dim),
        "n_chunks": len(all_chunks),
        "config": cfg.to_dict(),
        "documents": sorted(doc_entries, key=lambda d: d["doc_id"]),
    }
    IndexStore.build(all_chunks, all_vecs.reshape(-1, embedder.dim), manifest).save(out)
    report.documents, report.n_chunks = manifest["documents"], len(all_chunks)
    report.seconds = time.perf_counter() - t0
    return report
