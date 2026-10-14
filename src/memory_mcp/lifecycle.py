"""生命周期维护：摄取去重、TTL/软删除清理、向量补嵌。

调用面原则（勿破坏）：本模块的能力**不通过 MCP 工具暴露**——批量破坏性操作
必须留在 LLM 够不到的地方。触发入口只有两个：服务启动维护钩子、CLI 子命令。
agent 只能 memory_forget（单条）和 memory_stats（观察），不能 bulk。
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .db import MemoryStore, content_hash
from .embedder import Embedder

log = logging.getLogger("memory_mcp.lifecycle")


def ingest_with_dedup(
    store: MemoryStore,
    embedder: Embedder,
    content: str,
    kind: str,
    source: str,
    project: str | None,
    tags: list[str] | None,
    dedup_sim: float = 0.98,
) -> tuple[str, bool]:
    """写入路径内嵌去重，返回 (uid, 是否命中去重)。

    两级判重：
    1. 精确内容哈希（全局）：命中则刷新时间戳——重复出现视为"再次被证实相关"；
    2. 向量近似（同 project，cosine ≥ dedup_sim）：压缩托管类来源容易反复
       写入措辞略变的同义内容，近似命中同样刷新而非新增。
    向量通道关闭时只有第一级。
    """
    hit = store.find_by_hash(content_hash(content))
    if hit:
        store.touch(hit["uid"])
        return hit["uid"], True

    vec = embedder.embed_batch([content])
    if vec is not None:
        uids, mat = store.all_vectors(project=project)
        if len(uids):
            sims = mat @ vec[0]
            i = int(np.argmax(sims))
            if float(sims[i]) >= dedup_sim:
                store.touch(uids[i])
                return uids[i], True

    uid = store.store(
        content=content, kind=kind, source=source, project=project, tags=tags,
        embedding=vec[0] if vec is not None else None,
    )
    return uid, False


def run_maintenance(
    store: MemoryStore,
    ttl_artifact_days: int = 90,
    ttl_injection_days: int = 90,
) -> dict:
    """维护例程：硬删软删除 → TTL 清理（artifact + 注入遥测）→ 有删除才 VACUUM。

    设计取舍：fact/decision/preference 永久保留（去重抑制其增长），
    TTL 只作用于时效性最强的 artifact 类与遥测流水。VACUUM 重写整个
    库文件，只有真正删了东西才值得跑。
    """
    report: dict = {"soft_deleted_purged": store.purge_soft_deleted()}
    if ttl_artifact_days > 0:
        cutoff = time.time() - ttl_artifact_days * 86400
        report["ttl_purged"] = store.purge_older_than("artifact", cutoff)
    if ttl_injection_days > 0:
        cutoff = time.time() - ttl_injection_days * 86400
        report["injections_purged"] = store.purge_injections_before(cutoff)
    removed = sum(v for v in report.values())
    if removed:
        store.vacuum()
        report["vacuumed"] = True
    report["total_removed"] = removed
    return report


def backfill_vectors(
    store: MemoryStore,
    embedder: Embedder,
    batch_size: int = 64,
) -> dict:
    """向量补嵌：为没有向量的活条目补算嵌入（装模型后的一次性追平）。

    手动 CLI 触发——重操作不该自动化。embedder 复用服务同款惰性加载。
    """
    if not embedder.available:
        return {"vectorized": 0, "error": "embedder unavailable (check MEMORY_MODEL_PATH)"}
    total = 0
    while True:
        rows = store.entries_without_vectors(batch_size)
        if not rows:
            break
        vecs = embedder.embed_batch([r["content"] for r in rows])
        for r, v in zip(rows, vecs):
            store.insert_vector(r["uid"], v)
        total += len(rows)
        log.info("backfill progress: %d", total)
    return {"vectorized": total, "device": embedder.device}
