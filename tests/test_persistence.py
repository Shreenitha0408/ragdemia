"""Blob persistence: admin uploads survive a restart (fake container client, no Azure)."""

from __future__ import annotations

from ragdemia.persistence import BlobStorage


class FakeBlob:
    def __init__(self, name):
        self.name = name


class FakeDownload:
    def __init__(self, data):
        self.data = data

    def readall(self):
        return self.data


class FakeContainer:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.created = False

    def exists(self):
        return self.created

    def create_container(self):
        self.created = True

    def list_blobs(self):
        return [FakeBlob(n) for n in self.blobs]

    def download_blob(self, name):
        return FakeDownload(self.blobs[name])

    def upload_blob(self, name, fh, overwrite=False):
        self.blobs[name] = fh.read()

    def delete_blob(self, name):
        if name not in self.blobs:
            raise RuntimeError("BlobNotFound")
        del self.blobs[name]


def test_push_pull_delete_roundtrip(tmp_path):
    remote = FakeContainer()
    a = BlobStorage(None, "uploads", tmp_path / "server1", container_client=remote)
    a.ensure_repo()
    pdf = tmp_path / "lecture.pdf"
    pdf.write_bytes(b"%PDF-1.7 test")
    a.push(pdf)
    remote.blobs["notes.txt"] = b"ignored"  # only top-level PDFs are restored

    # a fresh container instance (after a restart) gets the file back
    b = BlobStorage(None, "uploads", tmp_path / "server2", container_client=remote)
    restored = b.pull()
    assert [p.name for p in restored] == ["lecture.pdf"]
    assert restored[0].read_bytes() == b"%PDF-1.7 test"

    b.delete("lecture.pdf")
    b.delete("lecture.pdf")  # deleting twice is harmless
    assert "lecture.pdf" not in remote.blobs
