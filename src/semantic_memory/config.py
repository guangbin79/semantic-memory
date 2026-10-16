"""集中式配置：env > <data_dir>/config.toml > 内置默认（D4）。

文件键 = env 同名蛇形全集（Config 全字段，如 model_path/rerank_model/
distill_model/device/token/db_path/host/port/cloud_*）；tomllib 标准库读取，
零新依赖；不存在 → 空；语法错 → ValueError 带文件路径与行号（tomllib
消息内含"at line X"）→ 启动失败指明文件。write_config 供 setup 子命令
写文件（任务 8 接线）：含密钥字段（token/cloud_api_key）→ chmod 600。
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .paths import default_db_path, default_model_path, resolve_data_dir

CONFIG_FILENAME = "config.toml"
_SECRET_KEYS = ("token", "cloud_api_key")  # 写文件时触发 chmod 600 的密钥字段


def config_file_path() -> Path:
    return resolve_data_dir() / CONFIG_FILENAME


def load_config_file(path: Path | None = None) -> dict:
    """读 config.toml：不存在 → {}；语法错/编码错 → ValueError（含文件与行号）。"""
    p = path or config_file_path()
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return {}
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"config.toml 无法解析（{p}）: {e}") from e


def write_config(data_dir: Path, values: dict) -> Path:
    """setup 写 config.toml：平铺标量 k=v；含密钥字段 → chmod 600。

    值限标量（str/int/float/bool/Path）；None 跳过（未配置不落盘）。
    整文件重写（setup 一次性成文，无增量合并语义）。
    """
    lines = []
    for k in sorted(values):
        v = values[k]
        if v is None:
            continue
        if isinstance(v, bool):
            rendered = "true" if v else "false"
        elif isinstance(v, (int, float)):
            rendered = repr(v)
        else:
            rendered = json.dumps(str(v))  # JSON 字符串转义是合法 TOML 基本串
        lines.append(f"{k} = {rendered}")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / CONFIG_FILENAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if any(k in values for k in _SECRET_KEYS):
        os.chmod(path, 0o600)
    return path


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
        file_cfg = load_config_file()  # 语法错在此抛出 → 启动失败指明文件

        def s(env: str, key: str, default: str = "") -> str:
            """str 档：env > 文件 > 默认；env/文件值均剥首尾空白（检查/取值一致）。"""
            v = os.environ.get(env)
            if v is not None:
                return v.strip()
            fv = file_cfg.get(key)
            return default if fv is None else str(fv).strip()

        def opt(env: str, key: str) -> str | None:
            """可选档：env 在场（含空串，原样不剥）→ 否则文件键；全无 → None。"""
            v = os.environ.get(env)
            if v is not None:
                return v
            fv = file_cfg.get(key)
            return None if fv is None else str(fv)

        def _bool(env: str, key: str, default: bool, env_literal) -> bool:
            """布尔档：env 按既有字面语义（env_literal）；文件 TOML 原生 bool。"""
            v = os.environ.get(env)
            if v is not None:
                return env_literal(v)
            fv = file_cfg.get(key)
            if fv is None:
                return default
            if isinstance(fv, bool):
                return fv
            if isinstance(fv, str):  # 宽容：字符串 "1"/"true"
                return env_literal(fv.strip())
            raise ValueError(f"config.toml 字段 {key} 应为布尔，得到 {type(fv).__name__}")

        # ---- 模型路径三件（空串=显式禁用的既有语义逐字段保留） ----
        model = opt("MEMORY_MODEL_PATH", "model_path")
        # 门禁 PASSED（.omo/evidence/rerank-gate.md 2026-10-16：recall@5 0.939≥基线 0.909）
        # 后默认启用：全未配置 → <data_dir>/models/ 常规路径（文件存在即生效，缺失优雅关闭）
        rerank_raw = opt("MEMORY_RERANK_MODEL", "rerank_model")
        rerank_model = (
            str(default_model_path("qwen3-reranker-0.6b-q8_0.gguf"))
            if rerank_raw is None else rerank_raw.strip()
        )
        distill_model = opt("MEMORY_DISTILL_MODEL", "distill_model")

        db_path = Path(s("MEMORY_DB_PATH", "db_path") or default_db_path())
        discovery = opt("MEMORY_DISCOVERY_FILE", "discovery_file")
        if discovery is None:
            discovery_file: Path | None = db_path.with_name(db_path.name + ".server.json")
        elif discovery.strip().lower() in ("", "off", "none"):
            discovery_file = None
        else:
            discovery_file = Path(discovery)
        return cls(
            db_path=db_path,
            host=s("MEMORY_HOST", "host", "127.0.0.1"),
            port=int(s("MEMORY_PORT", "port", "8765")),
            token=s("MEMORY_TOKEN", "token") or None,
            model_path=Path(model) if model else None,
            embed_dim=int(s("MEMORY_EMBED_DIM", "embed_dim", "1024")),
            device=s("MEMORY_DEVICE", "device", "cuda").lower(),
            recall_limit_default=int(s("MEMORY_RECALL_LIMIT", "recall_limit_default", "5")),
            rrf_k=int(s("MEMORY_RRF_K", "rrf_k", "60")),
            inject_boost=float(s("MEMORY_INJECT_BOOST", "inject_boost", "0.15")),
            recall_min_cosine=float(s("MEMORY_RECALL_MIN_COSINE", "recall_min_cosine", "0.45")),
            recall_max_cosine=float(s("MEMORY_RECALL_MAX_COSINE", "recall_max_cosine", "0.92")),
            ttl_artifact_days=int(s("MEMORY_TTL_ARTIFACT_DAYS", "ttl_artifact_days", "90")),
            ttl_injection_days=int(s("MEMORY_TTL_INJECTION_DAYS", "ttl_injection_days", "90")),
            ttl_stale_days=int(s("MEMORY_TTL_STALE_DAYS", "ttl_stale_days", "180")),
            maintenance_on_start=_bool(
                "MEMORY_MAINTENANCE_ON_START", "maintenance_on_start",
                True, lambda v: v == "1"),
            dedup_sim=float(s("MEMORY_DEDUP_SIM", "dedup_sim", "0.98")),
            rerank_model=Path(rerank_model) if rerank_model.strip() else None,
            rerank_candidates=int(s("MEMORY_RERANK_CANDIDATES", "rerank_candidates", "20")),
            rerank_budget_ms=int(s("MEMORY_RERANK_BUDGET_MS", "rerank_budget_ms", "500")),
            distill_model=Path(distill_model) if distill_model else None,
            distill_batch=int(s("MEMORY_DISTILL_BATCH", "distill_batch", "20")),
            cloud_enabled=_bool(
                "MEMORY_CLOUD_ENABLED", "cloud_enabled",
                True, lambda v: v != "0"),
            cloud_base_url=s("MEMORY_CLOUD_BASE_URL", "cloud_base_url"),
            cloud_api_key=s("MEMORY_CLOUD_API_KEY", "cloud_api_key"),
            cloud_model=s("MEMORY_CLOUD_MODEL", "cloud_model"),
            cloud_daily_max_calls=int(s("MEMORY_CLOUD_DAILY_MAX_CALLS", "cloud_daily_max_calls", "20")),
            cloud_daily_max_tokens=int(s("MEMORY_CLOUD_DAILY_MAX_TOKENS", "cloud_daily_max_tokens", "200000")),
            cloud_batch_n=int(s("MEMORY_CLOUD_BATCH_N", "cloud_batch_n", "30")),
            discovery_file=discovery_file,
        )
