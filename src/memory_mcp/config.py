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
    recall_max_cosine: float   # 冗余上限：纯向量命中 cos 高于此值出局（过近=冗余）；1.0 = 关闭
    # ---- 生命周期维护（调用面：启动钩子 + CLI，刻意不在 MCP 工具面）----
    ttl_artifact_days: int     # artifact 类 TTL；0 = 不过期
    ttl_injection_days: int    # 注入遥测 TTL；0 = 不过期
    ttl_stale_days: int        # 使用度留存：fact/artifact 未召回超此天数清除；0 = 关闭
    maintenance_on_start: bool # 启动时跑维护例程
    dedup_sim: float           # 向量近似去重阈值（同 project 内）
    # ---- 重排（completions logprobs 法；调用面接线在 hybrid 召回）----
    rerank_model: Path | None  # Qwen3-Reranker GGUF；None → 重排关闭（RRF 序直出）
    rerank_candidates: int     # 送入重排的候选条数（RRF 融合后截取）
    rerank_budget_ms: int      # 单次 rerank 总预算（毫秒）；超时整体 fail-open
    # ---- 蒸馏（Qwen3-1.7B 非思考；调用面：flush_distill_queue + 三路触发）----
    distill_model: Path | None  # Qwen3-1.7B GGUF；None → 蒸馏关闭（原文直存）
    distill_batch: int          # distill_queue pending ≥ 此数即触发 flush；0 = 关闭批量触发
    # ---- 云端整合通道（OpenAI 兼容 /chat/completions；默认开 + 三道安全带）----
    cloud_base_url: str         # 端点；空 = 通道关闭
    cloud_api_key: str          # 空 = 关闭
    cloud_model: str            # 空 = 关闭
    cloud_enabled: bool         # MEMORY_CLOUD_ENABLED=0 显式 kill switch（优先于 env 存在性）
    cloud_daily_max_calls: int  # 每日调用硬顶；进程内存计数，服务重启即重置
    cloud_daily_max_tokens: int # 每日 token（估算）硬顶
    cloud_batch_n: int          # 待整合候选 ≥ 此数自动触发整合

    # ---- 服务发现（agent 自举：MEMORY_PORT=0 随机端口时靠它找到服务）----
    discovery_file: Path | None  # 启动后写入 {url,port,pid,...}；None = 不写

    @classmethod
    def from_env(cls) -> "Config":
        model = os.environ.get("MEMORY_MODEL_PATH")
        rerank_model = os.environ.get("MEMORY_RERANK_MODEL")
        distill_model = os.environ.get("MEMORY_DISTILL_MODEL")
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
            recall_max_cosine=float(os.environ.get("MEMORY_RECALL_MAX_COSINE", "0.92")),
            ttl_artifact_days=int(os.environ.get("MEMORY_TTL_ARTIFACT_DAYS", "90")),
            ttl_injection_days=int(os.environ.get("MEMORY_TTL_INJECTION_DAYS", "90")),
            ttl_stale_days=int(os.environ.get("MEMORY_TTL_STALE_DAYS", "180")),
            maintenance_on_start=os.environ.get("MEMORY_MAINTENANCE_ON_START", "1") == "1",
            dedup_sim=float(os.environ.get("MEMORY_DEDUP_SIM", "0.98")),
            rerank_model=Path(rerank_model) if rerank_model else None,
            rerank_candidates=int(os.environ.get("MEMORY_RERANK_CANDIDATES", "20")),
            rerank_budget_ms=int(os.environ.get("MEMORY_RERANK_BUDGET_MS", "500")),
            distill_model=Path(distill_model) if distill_model else None,
            distill_batch=int(os.environ.get("MEMORY_DISTILL_BATCH", "20")),
            cloud_enabled=os.environ.get("MEMORY_CLOUD_ENABLED", "1") != "0",
            cloud_base_url=os.environ.get("MEMORY_CLOUD_BASE_URL", "").strip(),
            cloud_api_key=os.environ.get("MEMORY_CLOUD_API_KEY", "").strip(),
            cloud_model=os.environ.get("MEMORY_CLOUD_MODEL", "").strip(),
            cloud_daily_max_calls=int(os.environ.get("MEMORY_CLOUD_DAILY_MAX_CALLS", "20")),
            cloud_daily_max_tokens=int(os.environ.get("MEMORY_CLOUD_DAILY_MAX_TOKENS", "200000")),
            cloud_batch_n=int(os.environ.get("MEMORY_CLOUD_BATCH_N", "30")),
            discovery_file=discovery_file,
        )
