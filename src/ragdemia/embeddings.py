"""Dense text embeddings with an open model (default: BAAI/bge-base-en-v1.5, 110M params).

What happens underneath ``encode``:

1. The text is split into WordPiece sub-tokens and truncated to 512 tokens.
2. A 12-layer BERT encoder produces one 768-d vector per token.
3. BGE uses the **[CLS] token's vector** as the sentence representation (sentence-transformers
   reads the pooling strategy from the model's config, so this is not hard-coded here).
4. We **L2-normalise** every vector.  For unit vectors, inner product == cosine similarity,
   which lets FAISS use a plain inner-product index.

BGE was trained asymmetrically: queries are prefixed with an instruction
("Represent this sentence for searching relevant passages: "), passages are not.  Forgetting
the prefix measurably hurts retrieval, so ``encode_query`` adds it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        query_instruction: str = "",
        batch_size: int = 32,
        device: str = "auto",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._model = SentenceTransformer(model_name, device=self.device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            [self.query_instruction + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec[0].astype("float32")


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests and CI (no model download).

    Each token is hashed into one of ``dim`` buckets; the vector is L2-normalised.  It is a
    poor semantic model but has the right *interface* and makes tests hermetic.
    """

    def __init__(self, dim: int = 256) -> None:
        self.model_name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        import hashlib
        import re

        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return (
            np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)
