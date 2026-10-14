"""集中式配置：全部来自环境变量，带安全默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_db() -> Path:
    base = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return Path(base) / "memory-mcp" / "memory.db"


@dataclass(frozen=True)
class Config:
    db_path: Path
    host: str
    port: int
    token: str | None          # REST 侧车鉴权；None 表示仅本机回环保护
    model_path: Path | None    # Qwen3-Embedding-0.6B GGUF；None → 纯 BM25 降级
    embed_dim: int
    device: str                # cuda（默认）| cpu
    recall_limit_default: int
    rrf_k: int                 # RRF 常数，经典值 60

    @classmethod
    def from_env(cls) -> "Config":
        model = os.environ.get("MEMORY_MODEL_PATH")
        return cls(
            db_path=Path(os.environ.get("MEMORY_DB_PATH", _default_db())),
            host=os.environ.get("MEMORY_HOST", "127.0.0.1"),
            port=int(os.environ.get("MEMORY_PORT", "8765")),
            token=os.environ.get("MEMORY_TOKEN") or None,
            model_path=Path(model) if model else None,
            embed_dim=int(os.environ.get("MEMORY_EMBED_DIM", "1024")),
            device=os.environ.get("MEMORY_DEVICE", "cuda").lower(),
            recall_limit_default=int(os.environ.get("MEMORY_RECALL_LIMIT", "5")),
            rrf_k=int(os.environ.get("MEMORY_RRF_K", "60")),
        )
