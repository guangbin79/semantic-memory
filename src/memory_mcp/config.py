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
    inject_boost: float        # 遥测反哺强度：score × (1 + α·ln(1+注入次数))；0 = 关闭
    recall_min_cosine: float   # 纯向量命中（无 BM25 支持）的余弦门限；0 = 关闭
    # ---- 生命周期维护（调用面：启动钩子 + CLI，刻意不在 MCP 工具面）----
    ttl_artifact_days: int     # artifact 类 TTL；0 = 不过期
    ttl_injection_days: int    # 注入遥测 TTL；0 = 不过期
    maintenance_on_start: bool # 启动时跑维护例程
    dedup_sim: float           # 向量近似去重阈值（同 project 内）
    # ---- 服务发现（agent 自举：MEMORY_PORT=0 随机端口时靠它找到服务）----
    discovery_file: Path | None  # 启动后写入 {url,port,pid,...}；None = 不写

    @classmethod
    def from_env(cls) -> "Config":
        model = os.environ.get("MEMORY_MODEL_PATH")
        db_path = Path(os.environ.get("MEMORY_DB_PATH", _default_db()))
        discovery = os.environ.get("MEMORY_DISCOVERY_FILE")
        if discovery is None:
            discovery_file: Path | None = db_path.with_name(db_path.name + ".server.json")
        elif discovery.strip().lower() in ("", "off", "none"):
            discovery_file = None
        else:
            discovery_file = Path(discovery)
        return cls(
            db_path=db_path,
            host=os.environ.get("MEMORY_HOST", "127.0.0.1"),
            port=int(os.environ.get("MEMORY_PORT", "8765")),
            token=os.environ.get("MEMORY_TOKEN") or None,
            model_path=Path(model) if model else None,
            embed_dim=int(os.environ.get("MEMORY_EMBED_DIM", "1024")),
            device=os.environ.get("MEMORY_DEVICE", "cuda").lower(),
            recall_limit_default=int(os.environ.get("MEMORY_RECALL_LIMIT", "5")),
            rrf_k=int(os.environ.get("MEMORY_RRF_K", "60")),
            inject_boost=float(os.environ.get("MEMORY_INJECT_BOOST", "0.15")),
            recall_min_cosine=float(os.environ.get("MEMORY_RECALL_MIN_COSINE", "0.45")),
            ttl_artifact_days=int(os.environ.get("MEMORY_TTL_ARTIFACT_DAYS", "90")),
            ttl_injection_days=int(os.environ.get("MEMORY_TTL_INJECTION_DAYS", "90")),
            maintenance_on_start=os.environ.get("MEMORY_MAINTENANCE_ON_START", "1") == "1",
            dedup_sim=float(os.environ.get("MEMORY_DEDUP_SIM", "0.98")),
            discovery_file=discovery_file,
        )
