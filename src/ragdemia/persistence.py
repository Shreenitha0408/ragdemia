"""Persist the shared library's uploaded PDFs to a private Hugging Face *dataset* repository.

Free Hugging Face Spaces have an ephemeral disk: every restart or rebuild starts from the
container image.  Documents the admin uploads after deployment would be lost, so each upload
is mirrored to a private dataset repo and pulled back at startup, after which incremental
ingestion re-indexes only those files.  (The sample corpus is baked into the image and the
index is rebuilt, so only the PDFs themselves need to be stored.)
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class HubStorage:
    def __init__(self, repo_id: str, token: str | None, local_dir: str | Path) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.api = HfApi(token=token)

    def ensure_repo(self) -> None:
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)

    def pull(self) -> list[Path]:
        """Download every stored PDF into ``local_dir``; returns the local paths."""
        self.local_dir.mkdir(parents=True, exist_ok=True)
        names = [
            f
            for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
            if f.lower().endswith(".pdf") and "/" not in f
        ]
        paths = []
        for name in names:
            path = self.api.hf_hub_download(
                self.repo_id, name, repo_type="dataset", local_dir=self.local_dir
            )
            paths.append(Path(path))
        log.info("pulled %d document(s) from %s", len(paths), self.repo_id)
        return paths

    def push(self, path: Path) -> None:
        self.api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add {path.name}",
        )

    def delete(self, name: str) -> None:
        self.api.delete_file(
            name, repo_id=self.repo_id, repo_type="dataset", commit_message=f"Remove {name}"
        )
