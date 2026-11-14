"""Typed configuration loaded from ``config.yaml`` with environment-variable overrides.

Keeping every tunable in one file (and echoing it into the index manifest) means an
experiment is reproducible: the manifest records exactly which chunk size, embedding
model and retrieval mode produced a given set of results.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "RAGDEMIA_"


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    index_dir: str = "data/index"


@dataclass
class ParsingConfig:
    min_font_ratio: float = 0.7
    skip_sections: list[str] = field(
        default_factory=lambda: [
            "references",
            "bibliography",
            "acknowledgments",
            "acknowledgements",
        ]
    )


@dataclass
class ChunkingConfig:
    max_tokens: int = 320
    overlap_tokens: int = 64
    min_tokens: int = 40


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-base-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 32
    device: str = "auto"


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"
    top_k: int = 5
    candidates: int = 30
    rrf_k: int = 60
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"


@dataclass
class LLMConfig:
    backend: str = "ollama"
    model: str = "qwen2.5:7b"
    hf_model: str = "Qwen/Qwen2.5-3B-Instruct"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_new_tokens: int = 512
    num_ctx: int = 8192


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(value: str, target: Any) -> Any:
    """Convert an environment-variable string to the type of the default value."""
    if isinstance(target, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(target, int):
        return int(value)
    if isinstance(target, float):
        return float(value)
    if isinstance(target, list):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def load_config(path: str | Path | None = None) -> Config:
    """Load config: dataclass defaults <- YAML file <- ``RAGDEMIA_SECTION__KEY`` env vars."""
    cfg = Config()
    path = Path(path or os.environ.get(f"{ENV_PREFIX}CONFIG", "config.yaml"))
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    for section in fields(cfg):
        section_obj = getattr(cfg, section.name)
        overrides = raw.get(section.name) or {}
        for f in fields(section_obj):
            if f.name in overrides:
                setattr(section_obj, f.name, overrides[f.name])
            env_key = f"{ENV_PREFIX}{section.name.upper()}__{f.name.upper()}"
            if env_key in os.environ:
                current = getattr(section_obj, f.name)
                setattr(section_obj, f.name, _coerce(os.environ[env_key], current))
    return cfg
