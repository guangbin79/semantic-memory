"""生命周期维护：摄取去重、TTL/软删除清理、向量补嵌。

调用面原则（勿破坏）：本模块的能力**不通过 MCP 工具暴露**——批量破坏性操作
必须留在 LLM 够不到的地方。触发入口只有两个：服务启动维护钩子、CLI 子命令。
agent 只能 memory_forget（单条）和 memory_stats（观察），不能 bulk。
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

from . import gpu_breaker

from ._compat import np

from .db import INSERT_MEMORY_SQL, MemoryStore, content_hash
from .distiller import DistillParseError, Distiller
from .embedder import Embedder
from .noise import _is_credential, is_task_event_noise, strip_noise_lines
from .paths import resolve_data_dir

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
    跨项目捕获去重：同 content_hash 已存在于任一项目则跳过——自动捕获
    非项目权威知识，先捕先得，其余项目经全局桶召回（2026-10-21 六胞胎
    噪音实证：同 harness 横幅被 6 个并发会话按 project 各存一份）。
    """
    h = content_hash(content)
    if store.content_exists(h) or store.pending_queue_has(h):
        return store.distill_pending()
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
    project 分组（提示词需要单一 project 上下文）；Distiller 未配置 →
    该组原文按 kind=fact 直存（无裁判时的唯一兜底，否则队列成黑洞）；
    **已配置但不可用/失败/输出异常 → fail-closed：整组留 pending 待下次
    flush（2026-10-21 实证：fail-open raw 直存在裁判失联期制造噪音洪水，
    distill_queue_len 经 /health 可观测，不会静默黑洞）；**输出条数≠输入
    条数不 fail：超产 WARN+截断 result[:len(texts)] 按序绑定（超出条目丢
    弃），短缺 WARN+缺额行走 entry=None 原文直存——模型本无逐条 1:1 契
    约（2026-10-23 实证：条数守卫 fail 整组曾致队列积压 112 条）。
    噪声闸（noise.py）：队列行先经行级滤洗，滤洗后为空的行（纯任务状态
    回执/全噪声线）整行丢弃不蒸馏不落库；混合内容只送知识行，蒸馏输入
    与非噪声队列行保持按序 1:1 绑定。raw 直存兜底（无裁判/蒸馏 None/短缺
    缺额）仍写原文不滤洗——行级误杀对落库不可逆，宁漏勿误（有意取舍）。
    session_id 沿用队列行。
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
                return {"typed": 0, "stored_raw": 0, "left_pending": 0, "dropped": 0}
            if gpu_breaker.tripped():
                # 已熔断：无裁判可依（触碰蒸馏可能 abort）——原文直存排空，
                # fail-open 唯一兜底（捕获过滤器上游已挡 harness 噪音）
                distiller = None
            if distiller is not None and not distiller.is_available() and distiller.configured:
                # fail-closed：已配置但暂不可用（模型加载失败等）→ 留 pending 重试，
                # 裁判失联期的 raw 直存正是 2026-10-20 噪音洪水的放大器。
                # 未配置（configured=False）走下方 fail-open 原文直存（无裁判时的唯一兜底）。
                return {"typed": 0, "stored_raw": 0, "dropped": 0,
                        "left_pending": len(rows), "skipped": "distiller unavailable"}

            # 蒸馏结果按 project 分组缓存；失败项目收敛进 failed_projects（留 pending）
            entries_by_project: dict = {}
            failed_projects: set = set()
            dropped_projects: set = set()  # SYSTEM_PROMPT 契约：[]＝该组全部不值得记
            # 噪声闸（noise.py）：蒸馏输入按行级滤洗；滤洗后为空的行（纯状态
            # 回执/全噪声线行）无知识可馏——空槽送蒸馏只会诱出垃圾释义并打破
            # 按序绑定，故整行随行循环丢弃（dropped 计数可观测）
            stripped = {r["id"]: strip_noise_lines(r["content"]) for r in rows}
            noise_rows = {rid for rid, s in stripped.items() if not s}
            if distiller is not None:
                for project in {r["project"] for r in rows}:
                    texts = [stripped[r["id"]] for r in rows
                             if r["project"] == project
                             and r["id"] not in noise_rows]
                    if not texts:
                        continue  # 整组全噪声：无蒸馏必要（空文本组会误触熔断）
                    neighbors = _band_neighbors(store, embedder, texts, project)
                    try:
                        result = distiller.distill(texts, project or "",
                                                   neighbors or None)
                        if result is None:
                            # None＝硬件类失败（加载失败/响应异常；解析抖动走
                            # DistillParseError 不经此路）→ 已配置即熔断；未配置
                            # （configured=False）属于无裁判模式，raw 直存不熔断。
                            if getattr(distiller, "configured", False):
                                gpu_breaker.trip(f"distill returned None: project={project!r}")
                            entries_by_project[project] = None
                        elif result == []:
                            # SYSTEM_PROMPT 契约："Output [] when nothing is worth
                            # remembering"——合法的整组丢弃裁决，行标记 done
                            dropped_projects.add(project)
                        elif len(result) > len(texts):
                            # 超产：模型无逐条 1:1 契约 → 截断按序绑定，超出条目
                            # 丢弃（2026-10-23 实证：条数守卫 fail 整组曾致队列
                            # 积压 112 条，同 prompt 重试确定性复现等待无意义）
                            log.warning("distill 输出 %d 条 > 输入 %d 条，截断超出 %d 条",
                                        len(result), len(texts),
                                        len(result) - len(texts))
                            entries_by_project[project] = result[:len(texts)]
                        elif len(result) < len(texts):
                            # 短缺：前 len(result) 行按序绑定，缺额行经行循环
                            # 既有 entry=None 路径原文直存（kind=fact、判重照旧）
                            log.warning("distill 输出 %d 条 < 输入 %d 条，缺额 %d 行原文直存",
                                        len(result), len(texts),
                                        len(texts) - len(result))
                            entries_by_project[project] = result
                        else:
                            entries_by_project[project] = result
                    except DistillParseError as exc:
                        # 解析抖动（输出无 JSON）：硬件无损 → 不熔断。fail-closed
                        # 留 pending 待下次 flush（2026-10-22 实证：一次抖动曾
                        # trip 全局熔断器，停注入 16h 直到手动重启）
                        log.warning("蒸馏输出解析失败（%s），%d 条留待下次 flush",
                                    exc, len(texts))
                        failed_projects.add(project)
                    except Exception:
                        log.exception("distill 失败，%d 条留待下次 flush", len(texts))
                        gpu_breaker.trip(f"distill exception: project={project!r}")
                        failed_projects.add(project)

            now = time.time()
            report = {"typed": 0, "stored_raw": 0, "left_pending": 0}
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

            dropped = 0
            left_pending = 0
            for r in rows:
                if r["project"] in failed_projects:
                    left_pending += 1  # fail-closed：留 pending 待下次 flush
                    continue
                if r["id"] in noise_rows:
                    # 噪声闸：纯状态回执/全噪声线行不蒸馏不落库——滤洗后无
                    # 知识行即整行丢弃（宁漏勿误见 noise.py，dropped 可观测）
                    batch.append(("done", r["id"]))
                    dropped += 1
                    continue
                if r["project"] in dropped_projects:
                    batch.append(("done", r["id"]))  # 蒸馏裁决：全部不值得记
                    dropped += 1
                    continue
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
                if is_task_event_noise(entry["content"]):
                    # 噪声闸：蒸馏对事件行再创作的释义仍是噪声 → 同行丢弃
                    batch.append(("done", r["id"]))
                    dropped += 1
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
            report["left_pending"] = left_pending
            report["dropped"] = dropped
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


# ---------------- 夜扫（噪声兜底清扫：定时全库扫＋疑似晨报，ag-bb3029） ----------------
# 噪声闸（noise.py）拦新垃圾于入库时，历史漏网与未知变体由本清扫兜底：
# 确定性谓词命中→自动 forget 软删（安全：谓词确定性强＋软删留恢复窗口）；
# 疑似（谓词未命中但弱信号齐）→只进报告不动。黑名单规则升级不在此路
# （过拟合静默误杀风险，仍走派发评审）。

_noise_sweep_lock = threading.Lock()

# 凭据独立兜底（不依赖 noise.py 凭据短路是否随基线合并，单独实现于此）
_SWEEP_CRED_TOKEN = re.compile(r"token[:：]\s*[0-9a-zA-Z_\-]{40,}", re.IGNORECASE)
# 疑似弱信号三件套：[agora] task 前缀＋裸 40hex＋状态词
_BARE_HEX40 = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
_WEAK_STATUS = re.compile(
    r"done|completed|closed|merged|blocked|已 done|已 closed|已 blocked|已完成|已关闭|已取消",
    re.IGNORECASE)


def _credential_hit(content: str) -> bool:
    return ("[agora] credential" in content
            or "pypi-AgEI" in content
            or _SWEEP_CRED_TOKEN.search(content) is not None)


def _weak_signals(content: str) -> list[str]:
    sigs = []
    if content.lstrip().startswith("[agora] task"):
        sigs.append("[agora] task 前缀")
    if _BARE_HEX40.search(content):
        sigs.append("裸40hex")
    if _WEAK_STATUS.search(content):
        sigs.append("状态词")
    return sigs


def _brief(content: str, cap: int = 80) -> str:
    flat = " ".join(content.split())
    return flat if len(flat) <= cap else flat[: cap - 1] + "…"


def noise_sweep(store: MemoryStore, dry_run: bool = False, page_size: int = 200) -> dict:
    """夜扫一次：全库活条目过噪声谓词，命中→forget 软删；疑似→只进晨报。

    单飞与 flush 同款：已有实例在跑则本次跳过。凭据族独立兜底直接命中。
    dry_run=True 零写入，报告照常带 forgotten 候选。
    分页自补偿：软删行使 deleted=0 窗口前移，offset 回退等量防跳页；
    seen 防并发插入致窗口重扫时重复报告。
    """
    if not _noise_sweep_lock.acquire(blocking=False):
        return {"skipped": "noise sweep already running"}
    try:
        forgotten: list[dict] = []
        suspicious: list[dict] = []
        seen: set[str] = set()
        scanned = 0
        offset = 0
        while True:
            total, rows = store.list_page(limit=page_size, offset=offset)
            if not rows:
                break
            removed = 0
            for row in rows:
                uid = row["uid"]
                if uid in seen:
                    continue
                seen.add(uid)
                scanned += 1
                content = row["content"]
                if _credential_hit(content) or is_task_event_noise(content):
                    if not dry_run:
                        store.forget(uid)
                        removed += 1
                    # 摘要省略判定与删除谓词取并集（_is_credential 覆盖引号
                    # 包装等 noise.py 全形态）：报告随日志落盘，不得携带凭据原文
                    forgotten.append({"uid": uid, "kind": row["kind"],
                                      "summary": "[credential elided]"
                                      if _is_credential(content) else _brief(content)})
                else:
                    weak = _weak_signals(content)
                    if len(weak) == 3:  # 三弱信号齐＝疑似，报告不删
                        suspicious.append({"uid": uid, "summary": _brief(content),
                                           "weak_signals": weak})
            offset += len(rows) - removed
            # 并发取舍：他人于此间删除已扫活行时窗口仍可能滑过未扫行——
            # 单行天级延迟、次日定点扫自愈（软删无损，不为竞态加锁复杂度）
            if offset >= total:
                break
        report: dict = {"scanned": scanned, "forgotten": forgotten,
                        "suspicious": suspicious}
        if dry_run:
            report["dry_run"] = True
        return report
    finally:
        _noise_sweep_lock.release()


def _parse_sweep_at(at: str) -> tuple[int, int]:
    """MEMORY_NOISE_SWEEP_AT "HH:MM" → (hour, minute)；非法即抛（启动 fail-fast）。"""
    parts = at.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"MEMORY_NOISE_SWEEP_AT 应为 HH:MM，得到 {at!r}")
    hh, mm = int(parts[0]), int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"MEMORY_NOISE_SWEEP_AT 超出 0-23:0-59，得到 {at!r}")
    return hh, mm


def _next_sweep_run(hh: int, mm: int, now: datetime) -> datetime:
    nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return nxt


def write_noise_sweep_log(report: dict) -> Path:
    """报告落 <data_dir>/logs/noise-sweep-YYYYMMDD.log（日志不进记忆库自身）。"""
    log_dir = resolve_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"noise-sweep-{datetime.now():%Y%m%d}.log"
    body = json.dumps({"at": datetime.now().isoformat(timespec="seconds"), **report},
                      ensure_ascii=False)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def _noise_sweep_daemon(store: MemoryStore, at: str) -> None:
    hh, mm = _parse_sweep_at(at)
    while True:
        now = datetime.now()
        time.sleep(max(0.0, (_next_sweep_run(hh, mm, now) - now).total_seconds()))
        try:
            report = noise_sweep(store)
            if "skipped" not in report:  # 手动端点在跑则让位
                # 先 log.info 后落盘：写文件失败（目录占用/盘满）报告仍留服务日志
                log.info("noise sweep: %s", json.dumps(report, ensure_ascii=False))
                write_noise_sweep_log(report)
        except Exception:
            log.exception("noise sweep failed")


def start_noise_sweep_daemon(store: MemoryStore, at: str) -> None:
    _parse_sweep_at(at)  # 线程前校验：非法即在 serve() 同步抛出，兑现启动 fail-fast
    threading.Thread(target=_noise_sweep_daemon, args=(store, at),
                     daemon=True, name="noise-sweep").start()
