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


class BlobStorage:
    """Same interface as ``HubStorage``, backed by an Azure Blob Storage container.

    Used on Azure Container Apps: it needs no volume mount (which some environment types do
    not support), only a connection string supplied as a secret.
    """

    def __init__(
        self,
        connection_string: str | None,
        container: str,
        local_dir: str | Path,
        container_client=None,  # injectable for tests
    ) -> None:
        self.local_dir = Path(local_dir)
        if container_client is None:
            from azure.storage.blob import BlobServiceClient

            container_client = BlobServiceClient.from_connection_string(
                connection_string
            ).get_container_client(container)
        self.container = container_client

    def ensure_repo(self) -> None:
        if not self.container.exists():
            self.container.create_container()

    def pull(self) -> list[Path]:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for blob in self.container.list_blobs():
            name = blob.name
            if not name.lower().endswith(".pdf") or "/" in name:
                continue
            target = self.local_dir / name
            target.write_bytes(self.container.download_blob(name).readall())
            paths.append(target)
        log.info("pulled %d document(s) from blob storage", len(paths))
        return paths

    def push(self, path: Path) -> None:
        with open(path, "rb") as fh:
            self.container.upload_blob(path.name, fh, overwrite=True)

    def delete(self, name: str) -> None:
        try:
            self.container.delete_blob(name)
        except Exception as exc:  # already gone is fine
            if "BlobNotFound" not in str(exc) and "not exist" not in str(exc).lower():
                raise


def storage_from_config(cfg) -> HubStorage | BlobStorage | None:
    """Pick the persistence backend for admin uploads (public mode only).

    * ``AZURE_STORAGE_CONNECTION_STRING`` set -> Azure Blob Storage (container ``uploads``)
    * ``server.persist_repo`` set             -> private Hugging Face dataset
    * neither                                 -> local disk only (fine for a VM with a volume)
    """
    import os

    uploads = Path(cfg.paths.raw_dir) / "uploads"
    if conn := os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        return BlobStorage(conn, os.environ.get("RAGDEMIA_BLOB_CONTAINER", "uploads"), uploads)
    if cfg.server.persist_repo:
        return HubStorage(
            cfg.server.persist_repo, os.environ.get(cfg.server.persist_token_env), uploads
        )
    return None
