"""生命周期维护：摄取去重、TTL/软删除清理、向量补嵌。

调用面原则（勿破坏）：本模块的能力**不通过 MCP 工具暴露**——批量破坏性操作
必须留在 LLM 够不到的地方。触发入口只有两个：服务启动维护钩子、CLI 子命令。
agent 只能 memory_forget（单条）和 memory_stats（观察），不能 bulk。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from typing import Final

from ._compat import np

from .db import INSERT_MEMORY_SQL, MemoryStore, content_hash
from .distiller import Distiller
from .embedder import Embedder

log = logging.getLogger("semantic_memory.lifecycle")


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


# ---------------- 蒸馏缓冲（task5：入队 + 三路触发 + flush 落库） ----------------

# 进程内单飞：/idle 与批量阈值可能密集触发，已有 flush 在跑则本次跳过，
# 防两路并发读同一批 pending 行造成重复落库。
_flush_lock = threading.Lock()

# 近邻介入带（task6）：≥NEAR_HI 近重复（同 MEMORY_DEDUP_SIM）→ touch；
# NEAR_LO~NEAR_HI 模糊带 → 随蒸馏提示词问"是否取代"；<NEAR_LO 无关。
NEAR_HI: Final = 0.98
NEAR_LO: Final = 0.60


def enqueue_raw(
    store: MemoryStore,
    content: str,
    source: str,
    project: str | None,
    session_id: str | None = None,
) -> int:
    """蒸馏缓冲入队：原文落盘 distill_queue（pending），返回当前 pending 总数。

    chat/compaction 类来源由 rest_ingest 路由至此（热路径零 LLM）；显式
    remember / tool capture 保持直存语义，不经此路。flush 前持久，崩溃不丢。
    """
    return store.distill_enqueue(content, source, project, session_id)


def _dedup_uid(
    store: MemoryStore,
    embedder: Embedder | None,
    content: str,
    project: str | None,
    pending_hash: dict[str, str],
) -> tuple[str | None, np.ndarray | None]:
    """typed 条目落库判定：返回 (应 touch 的既有 uid 或 None, 内容向量或 None)。

    一级精确哈希（全局，同 ingest_with_dedup）；二级同域向量近邻
    cos≥NEAR_HI → touch。project=None 为全局域（project IS NULL），与
    project 域互不判重。embedder 未配 → 近邻级关闭，退化为仅哈希。
    pending_hash 收录本 flush 未提交插入（flush 独立连接未 commit，
    store 连接不可见），保证同批重复内容同样命中一级判重。
    """
    h = content_hash(content)
    if h in pending_hash:
        return pending_hash[h], None
    hit = store.find_by_hash(h)
    if hit:
        return hit["uid"], None
    vec = embedder.embed_batch([content]) if embedder else None
    if vec is not None:
        uids, mat = store.all_vectors(
            project=project, scope="global" if project is None else "all"
        )
        if len(uids):
            sims = mat @ vec[0]
            i = int(np.argmax(sims))
            if float(sims[i]) >= NEAR_HI:
                return uids[i], None
    return None, vec


def _band_neighbors(
    store: MemoryStore,
    embedder: Embedder | None,
    texts: list[str],
    project: str | None,
) -> list[tuple[str, str]]:
    """蒸馏组级近邻上下文：各原文同域 NEAR_LO~NEAR_HI 模糊带近邻（去重）。

    该带既非重复（≥NEAR_HI 会 touch）也非无关（<NEAR_LO），是"可能被取代"
    的候选——随提示词一次性带入，由同一次蒸馏调用裁决 supersedes。
    """
    if embedder is None:
        return []
    vecs = embedder.embed_batch(texts)
    if vecs is None:
        return []
    uids, mat = store.all_vectors(
        project=project, scope="global" if project is None else "all"
    )
    if not len(uids):
        return []
    found: list[str] = []
    for row in vecs @ mat.T:
        for i in np.argsort(row)[::-1][:3]:
            cos = float(row[i])
            if cos < NEAR_LO:
                break
            if cos < NEAR_HI and uids[i] not in found:
                found.append(uids[i])
    return [(r["uid"], r["content"]) for r in store.get_many(found)]


def flush_distill_queue(
    store: MemoryStore,
    distiller: Distiller | None,
    commit_batch: int = 5,
    embedder: Embedder | None = None,
) -> dict:
    """蒸馏队列落库：取全部 pending → Distiller → typed 条目；失败原文直存。

    连接纪律：本函数会在 daemon 线程运行（/idle、批量阈值触发），sqlite3 连接
    禁止跨线程复用——在此新建独立连接（WAL 是库级持久开关，新连接自动继承），
    分批 ≤commit_batch 条一次 commit（限 WAL 增长与写者串行），用后即关。
    每批原子：memories 插入/向量、touch、supersede 标记与队列标记同事务，
    中途崩溃该批回滚留待下次 flush。

    typed 条目三级行为（task6）：①精确哈希（全局）或同域向量近邻
    cos≥NEAR_HI → touch 既有不插入；②模糊带近邻随提示词一次性带入，
    蒸馏输出 supersedes → 新条先插入，旧条 superseded_by=新 uid；③其余 ADD。
    scope=global → project 置 NULL（全局域），与 project 域互不判重；
    新条向量随插写入（embedder 未配则缺者经 backfill 追平）。蒸馏按
    project 分组（提示词需要单一 project 上下文）；Distiller 未配置/
    不可用/失败/返回空 → 该组原文按 kind=fact 直存（status=stored_raw，
    现状行为不变差）。session_id 沿用队列行。
    """
    if not _flush_lock.acquire(blocking=False):
        return {"skipped": "flush already running"}
    try:
        conn = sqlite3.connect(str(store.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, content, source, project, session_id"
                " FROM distill_queue WHERE status='pending' ORDER BY id"
            ).fetchall()
            if not rows:
                return {"typed": 0, "stored_raw": 0}

            # 蒸馏结果按 project 分组缓存；任何失败路径收敛为 None（原文直存）
            entries_by_project: dict = {}
            if distiller is not None and distiller.is_available():
                for project in {r["project"] for r in rows}:
                    texts = [r["content"] for r in rows if r["project"] == project]
                    neighbors = _band_neighbors(store, embedder, texts, project)
                    try:
                        result = distiller.distill(texts, project or "",
                                                   neighbors or None)
                        if result is not None and len(result) != len(texts):
                            log.warning("distill 输出 %d 条 ≠ 输入 %d 条，无逐条对应，整组转直存",
                                        len(result), len(texts))
                            result = None
                        entries_by_project[project] = result
                    except Exception:
                        log.exception("distill 失败，%d 条原文转直存", len(texts))
                        entries_by_project[project] = None

            now = time.time()
            report = {"typed": 0, "stored_raw": 0}
            batch: list[tuple[str, int]] = []        # (队列 status, 队列 id)
            inserts: list[tuple] = []                # memories 行参数
            vecs: list[tuple[str, np.ndarray]] = []  # (uid, 向量)
            touches: list[str] = []                  # touch 的既有 uid
            supersedes: list[tuple[str, str]] = []   # (新 uid, 被取代旧 uid)
            pending_hash: dict[str, str] = {}        # 本批未提交插入（判重跨连接不可见）

            def commit() -> None:
                with conn:  # 插入/向量/touch/取代标记/队列标记同一事务
                    if inserts:
                        conn.executemany(INSERT_MEMORY_SQL, inserts)
                    if vecs:
                        conn.executemany(
                            "INSERT OR REPLACE INTO memories_vec(uid, embedding)"
                            " VALUES (?,?)",
                            [(u, v.tobytes()) for u, v in vecs],
                        )
                    if touches:
                        conn.executemany(
                            "UPDATE memories SET updated_at=?, last_recalled_at=?"
                            " WHERE uid=?",
                            [(now, now, u) for u in touches],
                        )
                    if supersedes:
                        conn.executemany(
                            "UPDATE memories SET superseded_by=?, superseded_at=?"
                            " WHERE uid=? AND deleted=0 AND superseded_by IS NULL",
                            [(new, now, old) for new, old in supersedes],
                        )
                    conn.executemany(
                        "UPDATE distill_queue SET status=? WHERE id=?", batch
                    )
                inserts.clear()
                vecs.clear()
                touches.clear()
                supersedes.clear()
                batch.clear()

            for r in rows:
                entries = entries_by_project.get(r["project"])
                entry = entries.pop(0) if entries else None
                if entry is None:
                    dup, vec = _dedup_uid(store, embedder, r["content"], r["project"],
                                           pending_hash)
                    if dup is not None:  # 直存路径同样判重：近重复/同文 → 刷新既有
                        touches.append(dup)
                        batch.append(("stored_raw", r["id"]))
                        continue
                    raw_uid = uuid.uuid4().hex[:16]
                    inserts.append((
                        raw_uid, r["content"], "fact", r["source"],
                        r["project"], "[]", now, now, content_hash(r["content"]),
                        r["session_id"],
                    ))
                    if vec is not None:  # 直存行同步落向量，免去事后 backfill
                        vecs.append((raw_uid, vec[0]))
                    pending_hash[content_hash(r["content"])] = raw_uid
                    batch.append(("stored_raw", r["id"]))
                    report["stored_raw"] += 1
                    continue
                report["typed"] += 1
                proj = None if entry["scope"] == "global" else r["project"]
                dup, vec = _dedup_uid(store, embedder, entry["content"], proj,
                                      pending_hash)
                if dup is not None:  # 近重复/同文 → 刷新既有（再次被证实相关）
                    touches.append(dup)
                else:
                    uid = uuid.uuid4().hex[:16]
                    pending_hash[content_hash(entry["content"])] = uid
                    inserts.append((
                        uid, entry["content"], entry["kind"], r["source"], proj,
                        json.dumps(entry["tags"], ensure_ascii=False),
                        now, now, content_hash(entry["content"]), r["session_id"],
                    ))
                    if vec is not None:
                        vecs.append((uid, vec[0]))
                    for old in entry.get("supersedes") or []:
                        supersedes.append((uid, old))
                batch.append(("done", r["id"]))
                if len(batch) >= commit_batch:
                    commit()
            if batch:
                commit()
            return report
        finally:
            conn.close()
    finally:
        _flush_lock.release()


def stamp_recalled(store: MemoryStore, uids: list[str]) -> int:
    """注入上报路径的召回打点：被注入=被召回，刷新 last_recalled_at。"""
    return store.stamp_recalled(uids)


def run_maintenance(
    store: MemoryStore,
    ttl_artifact_days: int = 90,
    ttl_injection_days: int = 90,
    ttl_stale_days: int = 180,
    dry_run: bool = False,
) -> dict:
    """维护例程：软删除硬清 → artifact TTL → 遥测 TTL → 使用度留存。

    dry_run=True 只统计各级将删条数与预计回收字节，不动库不 VACUUM。
    使用度留存：fact/artifact 超 ttl_stale_days 未被召回（NULL 视为
    created_at）或已被取代 → 软删除（下轮 soft_deleted 步硬清）；
    preference/decision 永久豁免；ttl_stale_days=0 关闭该步。
    fact/decision/preference 的增长由摄取去重抑制，TTL 只作用时效性
    最强的 artifact 类与遥测流水。VACUUM 只有真删了东西才值得跑。
    """
    now = time.time()
    soft = store.purge_soft_deleted(dry_run=dry_run)
    report: dict = {"soft_deleted_purged": soft["count"]}
    if ttl_artifact_days > 0:
        ttl = store.purge_older_than("artifact", now - ttl_artifact_days * 86400,
                                     dry_run=dry_run)
        report["ttl_purged"] = ttl["count"]
    else:
        ttl = {"count": 0, "bytes": 0}
    if ttl_injection_days > 0:
        inj = store.purge_injections_before(now - ttl_injection_days * 86400,
                                            dry_run=dry_run)
        report["injections_purged"] = inj["count"]
    else:
        inj = {"count": 0, "bytes": 0}
    stale_rows: list = []
    if ttl_stale_days > 0:
        stale_rows = store.stale_candidates(now - ttl_stale_days * 86400)
        by_kind: dict[str, int] = {}
        for r in stale_rows:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        if not dry_run:
            store.soft_delete_uids([r["uid"] for r in stale_rows])
        report["stale_purged"] = len(stale_rows)
        report["stale_by_kind"] = by_kind
    else:
        report["stale_purged"] = 0
    removed = soft["count"] + ttl["count"] + inj["count"] + report["stale_purged"]
    report["reclaim_bytes"] = (soft["bytes"] + ttl["bytes"]
                               + sum(r["nbytes"] or 0 for r in stale_rows))
    if dry_run:
        report["dry_run"] = True
    if removed and not dry_run:
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
