"""Ingestion pipeline: PDFs -> parsed sections -> chunks -> embeddings -> on-disk index."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunk, TokenCounter, chunk_document
from .config import Config
from .embeddings import Embedder
from .parsing import ParsedDocument, discover_pdfs, parse_pdf
from .store import IndexStore


@dataclass
class IngestReport:
    documents: list[ParsedDocument]
    chunks: list[Chunk]
    failures: list[tuple[str, str]]
    seconds: float
    index_dir: str


def ingest(
    cfg: Config,
    embedder: Embedder,
    count_tokens: TokenCounter,
    *,
    raw_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    progress=None,
) -> IngestReport:
    t0 = time.perf_counter()
    raw = Path(raw_dir or cfg.paths.raw_dir)
    out = Path(index_dir or cfg.paths.index_dir)
    pdfs = discover_pdfs(raw)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under '{raw}'.")

    docs: list[ParsedDocument] = []
    chunks: list[Chunk] = []
    failures: list[tuple[str, str]] = []
    for pdf in pdfs:
        if progress:
            progress(f"parsing {pdf.name}")
        try:
            doc = parse_pdf(
                pdf,
                root=raw,
                min_font_ratio=cfg.parsing.min_font_ratio,
                skip_sections=cfg.parsing.skip_sections,
            )
        except Exception as exc:  # a corrupt PDF must not kill the whole ingest
            failures.append((str(pdf), repr(exc)))
            continue
        if doc.n_chars == 0:
            failures.append((str(pdf), "no extractable text (scanned PDF? needs OCR)"))
            continue
        docs.append(doc)
        chunks.extend(
            chunk_document(
                doc,
                max_tokens=cfg.chunking.max_tokens,
                overlap_tokens=cfg.chunking.overlap_tokens,
                min_tokens=cfg.chunking.min_tokens,
                count_tokens=count_tokens,
            )
        )

    if progress:
        progress(f"embedding {len(chunks)} chunks with {embedder.model_name}")
    vectors = embedder.encode_documents(
        [c.embed_text for c in chunks], show_progress=bool(progress)
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": embedder.model_name,
        "embedding_dim": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "config": cfg.to_dict(),
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "source_path": d.source_path,
                "sha256": d.sha256,
                "num_pages": d.num_pages,
                "n_sections": len(d.sections),
                "n_chunks": sum(1 for c in chunks if c.doc_id == d.doc_id),
            }
            for d in docs
        ],
    }
    store = IndexStore.build(chunks, vectors, manifest)
    store.save(out)
    return IngestReport(docs, chunks, failures, time.perf_counter() - t0, str(out))
