"""Private, temporary per-visitor workspaces for the public deployment.

A visitor's browser generates a random workspace id (a UUID, kept in its local storage) and
sends it with every request.  PDFs uploaded under that id are parsed, chunked and embedded
into a small **in-memory** index that only requests carrying the same id can search.  At
question time it is searched *together with* the shared library (see
``Retriever.retrieve(extra_stores=...)``), so a reviewer can upload their own paper and ask
about it next to the sample corpus.

Privacy and abuse limits, all enforced server-side:

* the uploaded PDF is deleted from disk right after parsing - only chunks and vectors are
  kept, in RAM, and never written to the shared index or the persistent store;
* workspaces idle for longer than ``ttl`` are forgotten; at most ``max_workspaces`` exist
  (least-recently-used ones are evicted first);
* per workspace: at most ``max_docs`` PDFs of at most ``max_mb`` each;
* workspace ids must be well-formed UUIDs, so they cannot be used to address files.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import Chunk, TokenCounter, chunk_document
from .config import Config
from .embeddings import Embedder
from .library import LibraryError, safe_filename
from .parsing import parse_pdf
from .store import IndexStore

WORKSPACE_PREFIX = "workspace/"


def valid_workspace_id(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


@dataclass
class Workspace:
    id: str
    docs: list[dict] = field(default_factory=list)
    chunks: list[tuple[Chunk, np.ndarray]] = field(default_factory=list)
    store: IndexStore | None = None
    last_used: float = field(default_factory=time.time)


class WorkspaceManager:
    def __init__(
        self,
        cfg: Config,
        embedder: Embedder,
        count_tokens: TokenCounter,
        *,
        ttl_hours: float = 6.0,
        max_docs: int = 5,
        max_mb: int = 20,
        max_workspaces: int = 200,
    ) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.count_tokens = count_tokens
        self.ttl = ttl_hours * 3600
        self.max_docs = max_docs
        self.max_bytes = max_mb * 1024 * 1024
        self.max_workspaces = max_workspaces
        self._spaces: dict[str, Workspace] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Config, embedder: Embedder, count_tokens: TokenCounter):
        s = cfg.server
        return cls(
            cfg,
            embedder,
            count_tokens,
            ttl_hours=s.workspace_ttl_hours,
            max_docs=s.workspace_max_docs,
            max_mb=s.workspace_max_mb,
            max_workspaces=s.max_workspaces,
        )

    @property
    def limits(self) -> dict:
        return {
            "ttl_hours": self.ttl / 3600,
            "max_docs": self.max_docs,
            "max_mb": self.max_bytes // (1024 * 1024),
        }

    # ------------------------------------------------------------------ housekeeping
    def _gc(self) -> None:
        now = time.time()
        for wid in [w for w, ws in self._spaces.items() if now - ws.last_used > self.ttl]:
            del self._spaces[wid]
        while len(self._spaces) > self.max_workspaces:
            oldest = min(self._spaces.values(), key=lambda w: w.last_used)
            del self._spaces[oldest.id]

    def get(self, workspace_id: str | None) -> Workspace | None:
        wid = valid_workspace_id(workspace_id)
        if wid is None:
            return None
        with self._lock:
            self._gc()
            ws = self._spaces.get(wid)
            if ws:
                ws.last_used = time.time()
            return ws

    def stores(self, workspace_id: str | None) -> list[IndexStore]:
        ws = self.get(workspace_id)
        return [ws.store] if ws and ws.store and ws.chunks else []

    def documents(self, workspace_id: str | None) -> list[dict]:
        ws = self.get(workspace_id)
        return list(ws.docs) if ws else []

    @property
    def count(self) -> int:
        # Deliberately not __len__: an empty manager must still be truthy in `if ws:` checks.
        return len(self._spaces)

    # ------------------------------------------------------------------ changes
    def _rebuild(self, ws: Workspace) -> None:
        chunks = [c for c, _ in ws.chunks]
        vecs = (
            np.stack([v for _, v in ws.chunks])
            if ws.chunks
            else np.zeros((0, self.embedder.dim), "float32")
        )
        ws.store = IndexStore.build(chunks, vecs, {"embedding_model": self.embedder.model_name})

    def add(self, workspace_id: str | None, filename: str, data: bytes) -> dict:
        wid = valid_workspace_id(workspace_id)
        if wid is None:
            raise LibraryError("Missing or invalid workspace id")
        if len(data) > self.max_bytes:
            raise LibraryError(f"{filename}: larger than {self.max_bytes // 2**20} MB")
        if not data.startswith(b"%PDF"):
            raise LibraryError(f"{filename}: not a PDF file")
        sha = hashlib.sha256(data).hexdigest()

        with self._lock:
            self._gc()
            ws = self._spaces.get(wid) or Workspace(id=wid)
            if len(ws.docs) >= self.max_docs:
                raise LibraryError(
                    f"Your workspace already holds {self.max_docs} documents; remove one first."
                )
            if any(d["sha256"] == sha for d in ws.docs):
                raise LibraryError(f"{filename}: already in your workspace")

        # Parse outside the lock (it is the slow part); the PDF never outlives this block.
        stem = re.sub(r"\.pdf$", "", safe_filename(filename))
        with tempfile.TemporaryDirectory(prefix="ragdemia-ws-") as tmp:
            path = Path(tmp) / f"{stem}.pdf"
            path.write_bytes(data)
            doc = parse_pdf(
                path,
                root=Path(tmp),
                min_font_ratio=self.cfg.parsing.min_font_ratio,
                skip_sections=self.cfg.parsing.skip_sections,
            )
        if doc.n_chars == 0:
            raise LibraryError(f"{filename}: no extractable text (scanned PDF? needs OCR)")
        doc.doc_id = WORKSPACE_PREFIX + doc.doc_id
        doc.source_path = f"(your upload) {filename}"
        chunks = chunk_document(
            doc,
            max_tokens=self.cfg.chunking.max_tokens,
            overlap_tokens=self.cfg.chunking.overlap_tokens,
            min_tokens=self.cfg.chunking.min_tokens,
            count_tokens=self.count_tokens,
        )
        vecs = self.embedder.encode_documents([c.embed_text for c in chunks])

        entry = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source_path": doc.source_path,
            "sha256": sha,
            "num_pages": doc.num_pages,
            "n_sections": len(doc.sections),
            "n_chunks": len(chunks),
            "scope": "workspace",
            "deletable": True,
        }
        with self._lock:
            ws = self._spaces.setdefault(wid, ws)
            if any(d["doc_id"] == entry["doc_id"] for d in ws.docs):
                n = 2
                while any(d["doc_id"] == f"{entry['doc_id']}-{n}" for d in ws.docs):
                    n += 1
                new_id = f"{entry['doc_id']}-{n}"
                for c in chunks:
                    c.doc_id = new_id
                    c.chunk_id = c.chunk_id.replace(entry["doc_id"] + "::", new_id + "::")
                entry["doc_id"] = new_id
            ws.docs.append(entry)
            ws.chunks.extend(zip(chunks, vecs, strict=True))
            ws.last_used = time.time()
            self._rebuild(ws)
        return entry

    def remove(self, workspace_id: str | None, doc_id: str) -> None:
        ws = self.get(workspace_id)
        if ws is None or not any(d["doc_id"] == doc_id for d in ws.docs):
            raise LibraryError(f"No document '{doc_id}' in your workspace")
        with self._lock:
            ws.docs = [d for d in ws.docs if d["doc_id"] != doc_id]
            ws.chunks = [(c, v) for c, v in ws.chunks if c.doc_id != doc_id]
            self._rebuild(ws)
