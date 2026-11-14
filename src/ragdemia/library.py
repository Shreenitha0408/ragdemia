"""Document library: add and remove PDFs at runtime, then sync the index incrementally.

Used by both the web UI and the REST API so the rules live in one place:

* uploads are written to ``<raw_dir>/uploads/`` under a sanitised name (no path traversal),
  must start with the ``%PDF`` magic bytes, and are capped in size;
* a file whose content is already in the library (same SHA-256) is rejected;
* only documents that were uploaded through the app can be deleted through it - files you
  placed on disk yourself (``sample/``, ``private/``) are never removed by the UI or API;
* after any change, the index is synced incrementally (only the new file is embedded) and the
  live pipeline is pointed at the new index - models stay loaded.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

from .chunking import TokenCounter
from .config import Config
from .ingest import IngestReport, ingest
from .pipeline import RAGPipeline

UPLOAD_SUBDIR = "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class LibraryError(ValueError):
    """A user-facing problem with an upload or deletion (bad file, duplicate, not allowed)."""


def safe_filename(name: str) -> str:
    stem = Path(name).name  # drop any directory components ("../../etc/passwd" -> "passwd")
    stem = re.sub(r"\.pdf$", "", stem, flags=re.I)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
    return f"{stem[:100]}.pdf"


class Library:
    def __init__(
        self,
        cfg: Config,
        pipeline: RAGPipeline,
        count_tokens: TokenCounter,
        storage=None,  # optional HubStorage: mirrors uploads so they survive restarts
    ) -> None:
        self.cfg = cfg
        self.storage = storage
        self.pipeline = pipeline
        self.count_tokens = count_tokens
        self.raw_dir = Path(cfg.paths.raw_dir)
        self.upload_dir = self.raw_dir / UPLOAD_SUBDIR
        self._lock = threading.Lock()  # one index rebuild at a time

    # ------------------------------------------------------------------ queries
    @property
    def documents(self) -> list[dict]:
        docs = self.pipeline.retriever.store.manifest.get("documents", [])
        return [{**d, "deletable": self.is_deletable(d)} for d in docs]

    def is_deletable(self, doc: dict) -> bool:
        try:
            Path(doc["source_path"]).resolve().relative_to(self.upload_dir.resolve())
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------ changes
    def add(self, filename: str, data: bytes) -> Path:
        """Validate and store an uploaded PDF (does not re-index; call ``sync``)."""
        if len(data) > MAX_UPLOAD_BYTES:
            raise LibraryError(f"{filename}: larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        if not data.startswith(b"%PDF"):
            raise LibraryError(f"{filename}: not a PDF file")
        sha = hashlib.sha256(data).hexdigest()
        for d in self.pipeline.retriever.store.manifest.get("documents", []):
            if d.get("sha256") == sha:
                raise LibraryError(f"{filename}: already in the library as '{d['title']}'")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        target = self.upload_dir / safe_filename(filename)
        n = 2
        while target.exists():
            target = target.with_name(f"{target.stem.rsplit('-', 1)[0]}-{n}.pdf")
            n += 1
        target.write_bytes(data)
        if self.storage is not None:
            try:
                self.storage.push(target)
            except Exception as exc:
                target.unlink(missing_ok=True)
                raise LibraryError(
                    f"{filename}: could not save to persistent storage ({exc})"
                ) from exc
        return target

    def remove(self, doc_id: str) -> Path:
        doc = next((d for d in self.documents if d["doc_id"] == doc_id), None)
        if doc is None:
            raise LibraryError(f"No document with id '{doc_id}'")
        if not doc["deletable"]:
            raise LibraryError(
                f"'{doc['title']}' was added on disk, not through the app; remove the file "
                f"({doc['source_path']}) and run `ragdemia ingest`."
            )
        path = Path(doc["source_path"])
        if self.storage is not None:
            try:
                self.storage.delete(path.name)
            except Exception as exc:
                raise LibraryError(f"Could not remove from persistent storage ({exc})") from exc
        path.unlink(missing_ok=True)
        return path

    def sync(self, progress=None) -> IngestReport:
        """Bring the index in line with the folder and hot-swap it into the pipeline."""
        with self._lock:
            report = ingest(
                self.cfg,
                self.pipeline.retriever.embedder,
                self.count_tokens,
                progress=progress,
                allow_empty=True,
            )
            if report.changed:
                self.pipeline.reload_index()
            return report
