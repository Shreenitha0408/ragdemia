"""On-disk index: FAISS vectors + BM25 statistics + chunk metadata + a manifest.

Layout of ``data/index/``::

    vectors.faiss   FAISS IndexFlatIP over L2-normalised embeddings (cosine similarity)
    chunks.jsonl    one chunk per line, row i <-> vector i
    bm25.json       inverted index for lexical search
    manifest.json   embedding model, config, per-document checksums, build time

Why ``IndexFlatIP`` (exact, brute-force search) rather than HNSW / IVF?  A personal corpus
is a few thousand chunks; exact search over 5k x 768 floats is a ~4M multiply-add matrix
product - under a millisecond.  Approximate indexes only pay off at ~10^6 vectors, and they
add recall loss and tuning knobs.  Choosing the simplest structure that meets the latency
budget is the point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk


@dataclass
class IndexStore:
    chunks: list[Chunk]
    vectors: faiss.Index
    bm25: BM25Index
    manifest: dict = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: np.ndarray, manifest: dict) -> IndexStore:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must align")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        bm25 = BM25Index().fit([c.embed_text for c in chunks])
        return cls(chunks=chunks, vectors=index, bm25=bm25, manifest=manifest)

    def dense_search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.vectors.ntotal == 0:
            return []
        scores, ids = self.vectors.search(query_vec.reshape(1, -1).astype("float32"), top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def save(self, index_dir: str | Path) -> None:
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.vectors, str(d / "vectors.faiss"))
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (d / "bm25.json").write_text(json.dumps(self.bm25.to_dict()), encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> IndexStore:
        d = Path(index_dir)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"No index found in '{d}'. Run `ragdemia ingest` first.")
        with open(d / "chunks.jsonl", encoding="utf-8") as fh:
            chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
        return cls(
            chunks=chunks,
            vectors=faiss.read_index(str(d / "vectors.faiss")),
            bm25=BM25Index.from_dict(json.loads((d / "bm25.json").read_text(encoding="utf-8"))),
            manifest=json.loads((d / "manifest.json").read_text(encoding="utf-8")),
        )
