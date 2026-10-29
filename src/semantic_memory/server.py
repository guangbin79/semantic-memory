"""语义记忆服务：stdlib http.server 单进程 REST 面。

ThreadingHTTPServer + BaseHTTPRequestHandler 分发：12 条端点（11 静态 +
DELETE /api/memories/{uid}），函数体与鉴权规则与旧外壳逐条等价。

HTTP 硬性属性（评审钉死）：
- protocol_version="HTTP/1.1"，所有响应带准确 Content-Length（204 无体且
  按 RFC 7230 不带 Content-Length）；
- handler timeout=30：慢连接不得钉死线程；
- POST 缺 Content-Length 或 chunked → 411；坏 JSON → 400；
- 分发层 try/except → 500 JSON 兜底（异常不裸露为连接重置）；
- Host 白名单（DNS rebinding 防护）+ 写端点 Origin 白名单与强制
  application/json（浏览器 drive-by 双重防护；curl/插件不带 Origin 零摩擦）。
"""
from __future__ import annotations

import argparse
import hmac
import http.server
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .web import INDEX_HTML

from ._compat import HAS_NUMPY, INSTALL_HINT
from . import gpu_breaker
from .cloud import Consolidator
from .config import (AUTO_MARGIN_MIB, AUTO_RESIDENCY_MIB, RERANK_GGUF_NAME,
                     Config, auto_rerank_enabled)
from .consistency import run_sweep
from .db import MemoryStore
from .distiller import Distiller
from .embedder import Embedder, _card_free_mib
from .graph import cross_project_extras, graph_edges
from .hybrid import hybrid_recall
from .lifecycle import (
    NEAR_LO,
    backfill_vectors,
    enqueue_raw,
    flush_distill_queue,
    ingest_with_dedup,
    run_maintenance,
    stamp_recalled,
)
from .project import backfill_project_keys, resolve_project
from .paths import (MigrationError, MigrationRefused, default_model_path,
                    migrate_legacy, resolve_data_dir)
from .reranker import Reranker
from .startup_gate import check_startup

log = logging.getLogger("semantic_memory")

# ---------------- 观测面（task14）：召回延迟分位数 + 阈值告警 ----------------
# 提醒只住观测面（stats/日志/网页）——禁向会话/注入流写入任何告警。

recall_latencies: deque[float] = deque(maxlen=100)   # /recall 最近延迟样本（ms）
_alert_last: dict[str, float] = {}                   # level → 上次告警的 monotonic 时刻
ALERT_WINDOW_S = 600.0      # 同级告警节流窗口：10 分钟内只打一次
WARN_P95_MS = 800.0
ERROR_P95_MS = 1200.0
ENTRIES_CEILING = 40000     # 暴力扫性能天花板预警线
_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


def percentile(xs: list[float], p: float) -> float:
    """最近秩分位数：排序后取第 ceil(n·p) 个；空序列返回 0.0。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[max(0, math.ceil(len(s) * p / 100) - 1)]


def plan_alerts(p95_ms: float, entries: int, last: dict[str, float],
                now: float) -> list[tuple[str, str]]:
    """阈值判定（纯函数，不改动 last）：p95>1200 ERROR、>800 WARN、
    entries>40000 INFO（接近暴力扫天花板）；同级受 10 分钟节流。"""
    candidates: list[tuple[str, str]] = []
    if p95_ms > ERROR_P95_MS:
        candidates.append(("error", f"recall p95={p95_ms:.0f}ms > {ERROR_P95_MS:.0f}ms"))
    elif p95_ms > WARN_P95_MS:
        candidates.append(("warning", f"recall p95={p95_ms:.0f}ms > {WARN_P95_MS:.0f}ms"))
    if entries > ENTRIES_CEILING:
        candidates.append(("info",
                           f"entries={entries} > {ENTRIES_CEILING}，接近暴力扫性能天花板"))
    return [(lv, msg) for lv, msg in candidates
            if now - last.get(lv, float("-inf")) >= ALERT_WINDOW_S]


def _obs_stats() -> dict:
    """观测面字段：/api/stats 共用同一份。"""
    lats = list(recall_latencies)
    return {
        "recall_p50_ms": round(percentile(lats, 50), 1),
        "recall_p95_ms": round(percentile(lats, 95), 1),
        "distill_queue_len": store.distill_pending(),
        # task12 语义：进程内存计数，服务重启即重置
        "cloud_calls_24h": consolidator.stats()["calls"],
        "entries": store.stats()["total"],
    }


def _record_recall(ms: float) -> None:
    """/recall 计时入窗口并评估阈值告警（仅观测面动作，零注入流影响）。"""
    recall_latencies.append(ms)
    now = time.monotonic()
    p95 = percentile(list(recall_latencies), 95)
    for lv, msg in plan_alerts(p95, store.stats()["total"], _alert_last, now):
        _alert_last[lv] = now
        log.log(_LEVELS[lv], "recall alert: %s", msg)

# ---------------- 惰性核心单例（v0.3：import 零副作用） ----------------
# import 本模块不触发迁移/开库/建模型（--help、setup、测试收集零成本）；
# 首次属性访问（server.cfg/store/...）或 main() 才物化，物化后写入模块 globals。
_CORE_SINGLETONS = ("cfg", "store", "embedder", "reranker", "distiller", "consolidator")
_core_lock = threading.Lock()
_core_built = False


def _build_core() -> dict:
    try:  # 旧目录自动迁移（D3）：仅一次，禁入热路径；拒迁不阻塞启动
        migrate_legacy()
    except MigrationRefused as e:
        log.error("legacy migration skipped: %s", e)
    except MigrationError as e:
        log.critical("旧目录数据无法自动迁移（%s）——拒绝以空库启动", e)
        raise SystemExit(2) from e
    cfg = Config.from_env()
    store = MemoryStore(cfg.db_path)
    # rerank 三态落地：显式路径/"" 直接用；auto → 任何模型加载前探测整卡空闲，
    # ≥ 三模型常驻+余量 → 3 常驻启用重排，否则 2 常驻关闭（RRF 直出）。
    rerank_path = cfg.rerank_model
    if rerank_path is None and cfg.rerank_auto:
        free_mib = _card_free_mib()
        rerank_on = auto_rerank_enabled(free_mib, cfg.device)
        if rerank_on:
            rerank_path = default_model_path(RERANK_GGUF_NAME)
        log.info("rerank auto: device=%s free=%sMiB 阈值=%dMiB → %s",
                 cfg.device, free_mib, AUTO_RESIDENCY_MIB + AUTO_MARGIN_MIB,
                 "3 模型常驻，重排启用" if rerank_on else "2 模型常驻，重排关闭")
    embedder = Embedder(cfg.model_path, dim=cfg.embed_dim, device=cfg.device)
    return {
        "cfg": cfg, "store": store, "embedder": embedder,
        # 三模型各自常驻（2026-10-29 废弃单槽分时，换载开销真机不可接受）：
        # Reranker/Distiller 各持私有槽，单 occupant 永不换载。
        "reranker": Reranker(rerank_path, device=cfg.device,
                              budget_ms=cfg.rerank_budget_ms),
        "distiller": Distiller(cfg.distill_model, cfg.device),
        "consolidator": Consolidator(
            cfg.cloud_base_url, cfg.cloud_api_key, cfg.cloud_model,
            enabled=cfg.cloud_enabled, daily_max_calls=cfg.cloud_daily_max_calls,
            daily_max_tokens=cfg.cloud_daily_max_tokens, batch_n=cfg.cloud_batch_n),
    }


def init_core() -> None:
    """物化全部核心单例（幂等、线程安全）；main() 与首次属性访问都会触发。"""
    global _core_built
    if _core_built:
        return
    with _core_lock:
        if _core_built:
            return
        globals().update(_build_core())
        _core_built = True


def __getattr__(name: str):
    """PEP 562：外部属性访问触发物化；模块内函数走 globals（init 后同名同源）。"""
    if name in _CORE_SINGLETONS:
        init_core()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------- 检索核心（REST 端点共用） ----------------

def _recall_hits(query: str, limit: int, project: str | None, global_limit: int) -> list[dict]:
    """混合召回；global_limit>0 且指定 project 时两路合并（REST 共用）。

    project 路（scope='project'）在前，global 路（scope='global'，配额
    global_limit）补足其后；去重后按 limit 截断——全局条目不挤占项目配额。
    global_limit=0 或未指定 project 时单路调用，与既有行为逐字节一致。
    """
    if global_limit > 0 and project:
        q = embedder.embed_query(query)  # 双路共用同一次查询嵌入（免重复推理）
        proj_hits = hybrid_recall(
            store, embedder, query=query, limit=limit, project=project,
            rrf_k=cfg.rrf_k, inject_boost=cfg.inject_boost,
            min_cosine=cfg.recall_min_cosine, max_cosine=cfg.recall_max_cosine,
            reranker=reranker, rerank_candidates=cfg.rerank_candidates,
            query_vec=q,
        )
        glob_hits = hybrid_recall(
            store, embedder, query=query, limit=global_limit, project=None,
            rrf_k=cfg.rrf_k, inject_boost=cfg.inject_boost,
            min_cosine=cfg.recall_min_cosine, max_cosine=cfg.recall_max_cosine,
            reranker=reranker, rerank_candidates=cfg.rerank_candidates,
            scope="global", query_vec=q,
        )
        seen = {h["uid"] for h in proj_hits}
        merged = (proj_hits + [h for h in glob_hits if h["uid"] not in seen])[:limit]
    else:
        merged = hybrid_recall(
            store, embedder, query=query, limit=limit, project=project,
            rrf_k=cfg.rrf_k, inject_boost=cfg.inject_boost,
            min_cosine=cfg.recall_min_cosine, max_cosine=cfg.recall_max_cosine,
            reranker=reranker, rerank_candidates=cfg.rerank_candidates,
        )
    # 跨项目图谱：主结果后追加同源解法（不占主配额；全局域走上面的两路召回）
    if project and cfg.cross_project and cfg.cross_limit > 0 and merged:
        seen = {h["uid"] for h in merged}
        merged += [e for e in cross_project_extras(
            store, merged, project=project,
            cross_sim=cfg.cross_sim, limit=cfg.cross_limit)
            if e["uid"] not in seen]
    _mark_decision_conflicts(merged)
    return merged


def _mark_decision_conflicts(hits: list[dict]) -> None:
    """C 冲突聚簇标注：同 project 的 decision 条目两两余弦 ≥ NEAR_LO → conflict=True。

    机器只做候选发现（向量与 kind 都在库内，零新模型）；裁决交给当场有上下文
    的 agent（注入块尾 supersede 指引，插件面渲染）。无 numpy / 无向量 / 不足
    两条 → 静默跳过。阈值复用 lifecycle.NEAR_LO（近邻介入带下沿，语义同源）。
    """
    if not HAS_NUMPY:
        return
    groups: dict[str, list[dict]] = {}
    for h in hits:
        if h.get("kind") == "decision" and h.get("project"):
            groups.setdefault(h["project"], []).append(h)
    for proj, group in groups.items():
        if len(group) < 2:
            continue
        uids, mat = store.all_vectors(project=proj)
        row = {u: i for i, u in enumerate(uids)}
        pair = [(h, row[h["uid"]]) for h in group if h["uid"] in row]
        if len(pair) < 2:
            continue
        m = mat[[i for _, i in pair]]
        sims = m @ m.T
        for a in range(len(pair)):
            for b in range(a + 1, len(pair)):
                if float(sims[a, b]) >= NEAR_LO:
                    pair[a][0]["conflict"] = True
                    pair[b][0]["conflict"] = True


# ---------------- REST 端点（鉴权与传输门由 _Handler 负责） ----------------

def rest_health() -> dict:
    s = store.stats()
    out = {"status": "ok", "embedder": embedder.available, "device": embedder.device, **s}
    if gpu_breaker.tripped():
        # CUDA 熔断：模型路径全部 fail-open（BM25 召回），重启恢复
        out["gpu_degraded"] = gpu_breaker.reason()
    if not HAS_NUMPY:
        out["install_hint"] = INSTALL_HINT  # 无 numpy 时出现，有 numpy 时绝不出现
    return out


# ---------------- 人用读/管接口 + 内置浏览页 ----------------
# 读接口 + 单条遗忘：与 CLI 同级的人工信道。批量销毁依旧只走启动钩子/CLI。

def api_stats() -> dict:
    inj = store.injection_stats()
    # 热门条目附带内容摘要（遗忘过的 uid 解析为 None，前端自行跳过）
    top = []
    for t in inj.pop("top_uids"):
        rows = store.get_many([t["uid"]])
        top.append({"uid": t["uid"], "times": t["times"],
                    "content": rows[0]["content"][:120] if rows else None,
                    "kind": rows[0]["kind"] if rows else None})
    out = {**store.stats(), "embedder": embedder.available,
           "device": embedder.device,
           "injections": {**inj, "top": top}, **_obs_stats()}
    if not HAS_NUMPY:
        out["install_hint"] = INSTALL_HINT
    return out


def _qint(q: dict, key: str, default: int, cap: int | None = None) -> int | None:
    """GET 查询参数整数解析：非法 → None（调用方回 400 而非 500）。"""
    raw = q.get(key)
    if raw is None:
        return default
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None  # 负数拒绝：SQLite LIMIT 负值＝无上界，会绕开 200 封顶
    return min(v, cap) if cap is not None else v


def api_injections(q: dict) -> dict | tuple[int, dict]:
    limit = _qint(q, "limit", 50, 200)
    offset = _qint(q, "offset", 0)
    if limit is None or offset is None:
        return 400, {"error": "limit/offset must be integers"}
    rows = store.list_injections(
        harness=q.get("harness") or None,
        project=q.get("project") or None,
        limit=limit,
        offset=offset,
        session_id=q.get("session_id") or None,
    )
    if q.get("expand") == "1":
        # 侧边栏/面板预览：uids→记忆摘要批量展开（cap 500 防 IN 参数越界）
        for row in rows:
            if isinstance(row["uids"], str):
                row["uids"] = json.loads(row["uids"])
        uid_list = sorted({uid for row in rows for uid in row["uids"][:50]})[:500]
        previews = store.memories_by_uids(uid_list)
        for row in rows:
            row["memories"] = [previews[uid] for uid in row["uids"] if uid in previews]
    return {"injections": rows}


def api_memories(q: dict) -> dict | tuple[int, dict]:
    limit = _qint(q, "limit", 50, 200)
    offset = _qint(q, "offset", 0)
    if limit is None or offset is None:
        return 400, {"error": "limit/offset must be integers"}
    rows = store.list_recent(
        project=q.get("project") or None,
        kind=q.get("kind") or None,
        limit=limit,
        offset=offset,
    )
    return {"memories": [
        {"uid": r["uid"], "content": r["content"], "kind": r["kind"],
         "source": r["source"], "project": r["project"], "tags": r["tags"],
         "created_at": r["created_at"], "updated_at": r["updated_at"]}
        for r in rows
    ]}


def api_graph(q: dict) -> dict:
    """GET /api/graph?uid=：单条目 1-hop 边（三种；以 to_uid 为种子再查即链式遍历）。

    取代链前代经管理面读出（召回面不可见，此处仅观测）；superseded=True 标记
    被取代邻居。cross_sim 可选参数覆盖默认阈值。未知 uid 返回空边集。
    """
    uid = (q.get("uid") or "").strip()
    row = store.get_one(uid) if uid else None
    if row is None:
        return {"uid": uid, "project": None, "edges": []}
    try:
        sim = float(q.get("cross_sim", cfg.cross_sim))
    except (TypeError, ValueError):
        sim = cfg.cross_sim
    if not math.isfinite(sim):
        sim = cfg.cross_sim  # nan/inf 能过 float() 却令阈值比较恒 False，静默关掉阈值——回落默认
    return {
        "uid": uid, "project": row["project"],
        "edges": [
            {"kind": e["kind"], "from_uid": e["from_uid"], "to_uid": e["to_uid"],
             "to_project": e["to_project"], "weight": e["weight"],
             "to_kind": e["to_row"]["kind"], "to_content": e["to_row"]["content"][:120],
             "superseded": e["kind"] == "supersedes"}
            for e in graph_edges(store, [uid], cross_sim=sim)
        ],
    }


def api_forget(uid: str) -> dict:
    return {"uid": uid, "forgotten": store.forget(uid)}


# ---------------- 批量维护（双闸门：必须配置 token + 请求鉴权） ----------------
# 默认无 token 的回环安装下，HTTP 面对本机进程（含 agent shell）是开放的，
# 因此批量破坏性操作此时仅 CLI 可达（403）；配置 token 后才对持票人开放。

def _bulk_guard(headers) -> tuple[int, dict] | None:
    if cfg.token is None:
        return 403, {"error": "bulk maintenance disabled without MEMORY_TOKEN; use the CLI"}
    if not _authorized(headers):
        return 401, {"error": "unauthorized"}
    return None


def api_maintenance_purge() -> dict:
    return run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days,
                           ttl_injection_days=cfg.ttl_injection_days,
                           ttl_stale_days=cfg.ttl_stale_days)


def api_maintenance_backfill() -> dict:
    return backfill_vectors(store, embedder)


def rest_recall(body: dict) -> dict | tuple[int, dict]:
    q = body.get("query")
    if not isinstance(q, str):
        return 400, {"error": "query must be a string"}
    if not q.strip():
        return {"hits": []}  # 既有契约：空查询 → 空 hits（见 test_recall_empty_query）
    raw_project = body.get("project")
    if raw_project is not None and not isinstance(raw_project, str):
        return 400, {"error": "project must be a string"}
    try:
        limit = int(body.get("limit", cfg.recall_limit_default))
        global_limit = int(body.get("global_limit", 0))
    except (TypeError, ValueError):
        return 400, {"error": "limit/global_limit must be integers"}
    if limit < 0 or global_limit < 0:
        return 400, {"error": "limit/global_limit must be non-negative"}
    t0 = time.perf_counter()
    hits = _recall_hits(
        query=q,
        limit=limit,
        project=resolve_project(raw_project),
        global_limit=global_limit,
    )
    _record_recall((time.perf_counter() - t0) * 1000.0)
    out = {"hits": hits}
    if gpu_breaker.tripped():
        # 熔断期：注入是干扰不是帮助（用户裁定：达不到准确不如不用）——
        # 响应带降级标记，插件据此停止注入；BM25 召回本体照常（工具显式调用可用）
        out["gpu_degraded"] = gpu_breaker.reason()
    return out


# 入队路由（r3 评审）：chat/compaction 是蒸馏队列的投料源——落盘缓冲、热路径零 LLM；
# 显式 remember / tool capture 保持直存（即时可见，也避免队列无人投料的死管道）。
QUEUED_SOURCES = frozenset({"compaction", "chat"})
# W 写侧门的自动捕获源＝队列源＋工具产物捕获（插件 execute.after 白名单制，也是
# 钩子自动发出而非用户显式——终闸 B1：漏它则 /tmp 噪声经全局桶跨项目注入复活）。
AUTO_SOURCES = QUEUED_SOURCES | {"tool"}


def _spawn_flush() -> None:
    # ponytail: 服务首批后台线程——独立 SQLite 连接、进程内单飞锁都在 flush_distill_queue；
    # 此处仅兜底线程异常不让其无声死；线程死于进程死（daemon）。
    def _run() -> None:
        try:
            report = flush_distill_queue(store, distiller, embedder=embedder)
            log.info("distill flush: %s", report)
            # task12：flush 完成后的云整合触发判定——沿用本 daemon 线程（禁入同步请求路径）
            log.info("cloud consolidate: %s", consolidator.maybe_run(
                store, embedder,
                new_entries=report.get("typed", 0) + report.get("stored_raw", 0)))
        except Exception:
            log.exception("distill flush failed")
    threading.Thread(target=_run, daemon=True, name="distill-flush").start()


def rest_ingest(body: dict) -> dict | tuple[int, dict]:
    source = body.get("source")
    if source is None:
        source = "chat"  # 显式 null 也回落 chat（dict.get 带默认不接 null）
    if not isinstance(source, str):
        return 400, {"error": "source must be a string"}  # 不可哈希源会在 frozenset 成员判定上 TypeError→500
    raw_project = body.get("project")
    if raw_project is not None and not isinstance(raw_project, str):
        return 400, {"error": "project must be a string"}
    project = resolve_project(raw_project)
    # W 写侧门（2026-10-26 /tmp fixture 噪声实证；终闸 R2 补 tool 源）：绝对路径
    # 归一失败 ⇒ 非 git/无 origin 工作区——自动捕获（chat/compaction/tool，皆插件
    # 钩子自动发出）拒收；显式调用（remember 等）不拒收，降级全局桶并在响应回显。
    # 归一化键（如 github.com/a/b）与相对串不以 / 开头，天然不受门影响。
    if raw_project and project == raw_project and raw_project.startswith("/"):
        if source in AUTO_SOURCES:
            return {"queued": False, "dropped": "unresolvable_project"}
        project = None
    content = body.get("content")
    if not isinstance(content, str):
        return 400, {"error": "content must be a string"}
    if source in QUEUED_SOURCES:
        pending = enqueue_raw(store, content, source, project,
                              body.get("session_id"))
        if cfg.distill_batch > 0 and pending >= cfg.distill_batch:
            _spawn_flush()  # 触发器一路：队列积压达阈值（flush 在 daemon 线程）
        return {"queued": True, "pending": pending}
    uid, deduped = ingest_with_dedup(
        store, embedder,
        content=content,
        kind=body.get("kind", "fact"),
        source=source,
        project=project,
        tags=body.get("tags"),
        dedup_sim=cfg.dedup_sim,
    )
    # 终闸 B2：W 门可能把显式条目降级全局——回显实际落库 project，调用方可观测
    return {"uid": uid, "deduped": deduped, "project": project}


def inject_report(body: dict) -> tuple[int, dict]:
    """注入遥测上报（插件/适配器信道，只追加不删除）。

    只有注入方知道"这次召回真的注进了上下文"——服务端只见召回。
    记录供给侧事实：谁(harness)、哪个钩子、触发的查询、注入了哪些
    条目、多大块。模型是否采纳不可知，此处不做伪精确度量。
    """
    uids = body.get("uids")
    if not isinstance(uids, list) or not all(isinstance(u, str) for u in uids):
        return 400, {"error": "uids must be a string array"}
    # 遥测侧 record_injection 自带 uids[:50]；stamp 走分批不截断（见循环处注释）
    try:
        chars = int(body.get("chars", 0))
    except (TypeError, ValueError):
        return 400, {"error": "chars must be an integer"}
    rid = store.record_injection(
        harness=str(body.get("harness", "")),
        hook=str(body.get("hook", "")),
        query=str(body.get("query", "")),
        uids=uids,
        chars=chars,
        project=body.get("project"),
        # 可选扩展：会话归属（面板按当前会话过滤）＋注入块全文（截 2000，record_injection 再兜底）
        session_id=body.get("session_id") if isinstance(body.get("session_id"), str) else None,
        block=body.get("block") if isinstance(body.get("block"), str) else None,
    )
    # 落入 stale_candidates 被留存误清（评审 R1-C2 修正）
    for i in range(0, len(uids), 50):
        stamp_recalled(store, uids[i:i + 50])
    return 200, {"recorded": True, "id": rid}


# ---------------- HTTP 外壳（stdlib http.server） ----------------

def _authorized(headers) -> bool:
    if cfg.token is None:
        return True  # 未配置 token：依赖 127.0.0.1 回环绑定 + Host/Origin 白名单
    auth = headers.get("authorization", "")
    return hmac.compare_digest(auth.encode("latin-1"), f"Bearer {cfg.token}".encode())  # 常量时间比较（bytes 防 non-ASCII TypeError）


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_MAX_BODY_BYTES = 8 * 1024 * 1024  # 回环个人服务；嵌入文本量级远小于此
_JSON_POST_PATHS = frozenset({
    "/recall", "/ingest", "/inject/report",
    "/api/maintenance/purge", "/api/maintenance/backfill",
})


def _hostname_of(host_header: str) -> str:
    """Host/Origin 值 → 主机名（剥端口；IPv6 字面量 [::1]:8765 → ::1）。"""
    h = host_header.strip().lower()
    if h.startswith("["):
        return h.split("]")[0].lstrip("[")
    return h.split(":")[0]


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "semantic-memory"
    timeout = 30  # 慢连接读超时：防单线程钉死（BaseHTTPRequestHandler 捕获后关连接）

    # ---- 响应发送（每响应恰一次；业务函数先算后发，异常兜底 500 不会撞半途响应） ----

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_204(self) -> None:
        # RFC 7230：204 必无体且禁带 Content-Length；HTTP/1.1 下无体即消息完整，连接可复用
        self.send_response(204)
        self.end_headers()

    # ---- 传输安全门 ----

    def _host_ok(self) -> bool:
        """Host 白名单：绑定地址/localhost/127.0.0.1（带端口变体）——DNS rebinding 防护。"""
        host = self.headers.get("Host")
        if host is None:
            return False
        h = _hostname_of(host)
        return h in _LOOPBACK_HOSTS or h == cfg.host.lower()

    def _origin_ok(self) -> bool:
        """写端点 Origin 门：头存在且非回环白名单 → 拒（浏览器跨源必带；curl/插件不带→零摩擦）。"""
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        try:
            hostname = urlsplit(origin).hostname or ""
        except ValueError:
            return False
        hostname = hostname.lower()
        return hostname in _LOOPBACK_HOSTS or hostname == cfg.host.lower()

    def _json_content_type(self) -> bool:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        return ctype == "application/json"

    def _read_body(self) -> bytes | None:
        """按 Content-Length 读 POST body；缺 CL / chunked → 411（回错前标记断连，防 keep-alive 错位）。"""
        te = (self.headers.get("Transfer-Encoding") or "").strip().lower()
        cl_raw = self.headers.get("Content-Length")
        if (te and te != "identity") or cl_raw is None:
            self.close_connection = True
            self._send_json({"error": "Content-Length required (chunked not supported)"}, 411)
            return None
        try:
            n = int(cl_raw)
        except ValueError:
            self.close_connection = True
            self._send_json({"error": "invalid Content-Length"}, 400)
            return None
        if n < 0 or n > _MAX_BODY_BYTES:
            self.close_connection = True
            self._send_json({"error": "request body too large"}, 413)
            return None
        return self.rfile.read(n) if n else b""

    # ---- 分发 ----

    def do_GET(self) -> None:  # noqa: N802 (http.server 约定命名)
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _send_h(self, r: dict | tuple[int, dict]) -> None:
        """handler 结果发送：元组 (status, payload) 或裸 dict（200）。"""
        status, payload = r if isinstance(r, tuple) else (200, r)
        self._send_json(payload, status)

    def _dispatch(self, method: str) -> None:
        try:
            init_core()  # 首请求即物化单例：单文件/独立运行测试无需先触模块属性
            self._route(method)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # 客户端已走：异常不得外泄为噪音
        except Exception:
            log.exception("unhandled error: %s %s", method, self.path)
            try:
                self._send_json({"error": "internal server error"}, 500)
            except OSError:
                self.close_connection = True

    def _route(self, method: str) -> None:
        if not self._host_ok():
            self.close_connection = True
            self._send_json({"error": "host not allowed"}, 403)
            return
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}

        if method == "GET":
            if path == "/":
                return self._send_html(INDEX_HTML)
            if path == "/health":
                return self._send_json(rest_health())
            if path == "/api/stats":
                if not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                return self._send_json(api_stats())
            if path == "/api/injections":
                if not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                return self._send_h(api_injections(q))
            if path == "/api/memories":
                if not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                return self._send_h(api_memories(q))
            if path == "/api/graph":
                if not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                return self._send_h(api_graph(q))
            return self._send_json({"error": "not found"}, 404)

        if method == "DELETE":
            if not self._origin_ok():
                return self._send_json({"error": "cross-origin write rejected"}, 403)
            if path.startswith("/api/memories/"):
                if not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                return self._send_json(api_forget(path[len("/api/memories/"):]))
            return self._send_json({"error": "not found"}, 404)

        if method == "POST":
            body = self._read_body()  # 最先读体：任何提前响应后连接内不留未读字节（keep-alive 防串扰）
            if body is None:
                return
            if not self._origin_ok():
                return self._send_json({"error": "cross-origin write rejected"}, 403)
            if path == "/idle":
                if not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                _spawn_flush()  # 204 立即回，flush 在 daemon 线程执行（热路径零 LLM）
                return self._send_204()
            if path in _JSON_POST_PATHS:
                if path.startswith("/api/maintenance/"):
                    guard = _bulk_guard(self.headers)
                    if guard is not None:
                        return self._send_json(guard[1], guard[0])
                elif not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                if not self._json_content_type():
                    # 封死 no-cors text/plain 盲写：非 application/json 一律拒
                    return self._send_json({"error": "Content-Type must be application/json"}, 415)
                try:
                    parsed = json.loads(body.decode("utf-8")) if body else None
                    if not isinstance(parsed, dict):
                        raise ValueError("body must be a JSON object")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    return self._send_json({"error": "invalid JSON body"}, 400)
                if path == "/recall":
                    return self._send_h(rest_recall(parsed))
                if path == "/ingest":
                    return self._send_h(rest_ingest(parsed))
                if path == "/inject/report":
                    status, payload = inject_report(parsed)
                    return self._send_json(payload, status)
                if path == "/api/maintenance/purge":
                    return self._send_json(api_maintenance_purge())
                return self._send_json(api_maintenance_backfill())
            return self._send_json({"error": "not found"}, 404)

        return self._send_json({"error": "method not allowed"}, 405)

    def log_message(self, fmt: str, *args) -> None:  # 默认每请求写 stderr 一行 → 降为 debug
        log.debug("%s - %s", self.address_string(), fmt % args)


# ---------------- 服务发现（随机端口自举） ----------------

def _announce(port: int) -> dict:
    """绑定完成后对外发布真实地址：发现文件 + stdout 机器可读行。

    agent 自举流程：spawn 本进程（MEMORY_PORT=0）→ 读 discovery_file
    或解析 stdout 的 MEMORY_LISTENING 行 → 得到 url。
    """
    url = f"http://{cfg.host}:{port}"
    info = {
        "url": url,
        "port": port,
        "host": cfg.host,
        "pid": os.getpid(),
        "db_path": str(cfg.db_path),
        "token_required": cfg.token is not None,
        "started_at": time.time(),
    }
    f = cfg.discovery_file
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"MEMORY_LISTENING {json.dumps(info, ensure_ascii=False)}", flush=True)
    log.info("listening on %s", url)
    return info


def _retract() -> None:
    """退出时撤下发现文件，避免 agent 读到僵尸地址（文件内含 pid 可二次校验）。"""
    f = cfg.discovery_file
    if f is not None:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass


def _install_exit_handler() -> None:
    """SIGINT/SIGTERM → SystemExit：让 finally（撤下发现文件/关服）能展开，
    退出码保持 128+sig 语义。"""

    def _handler(sig, _frame):
        raise SystemExit(128 + sig)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def serve() -> None:
    """HTTP 主循环：ThreadingHTTPServer 构造即绑定（port=0 时读回真实端口）→ 发布 → 常驻。"""
    _install_exit_handler()
    httpd = http.server.ThreadingHTTPServer((cfg.host, cfg.port), _Handler)
    httpd.daemon_threads = True
    _announce(httpd.server_address[1])
    try:
        httpd.serve_forever()
    finally:
        _retract()
        httpd.server_close()


# ---------------- 离线评测与整合 CLI ----------------

def _load_golden(path: Path) -> list[dict]:
    """读 golden jsonl：每行 {query, expect_uids[], project?}，跳过空行。"""
    rows: list[dict] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError(f"第 {i} 行不是 JSON 对象")
        if not obj.get("query") or not isinstance(obj.get("expect_uids"), list) \
                or not obj["expect_uids"]:
            raise ValueError(f"第 {i} 行缺少 query 或 expect_uids")
        rows.append({"query": obj["query"], "expect_uids": list(obj["expect_uids"]),
                     "project": obj.get("project")})
    return rows


def run_eval(store: MemoryStore, embedder: Embedder, rows: list[dict], ks: list[int]) -> dict:
    """基线测量核心：对 golden 行逐条跑 hybrid_recall，聚合 recall@k 与 MRR。

    recall@k：expect_uids 与 top-k 交集非空即该行命中；MRR 取首个命中排名
    的倒数（未命中记 0）；两者都在全部 golden 行上取平均。检索窗口取
    max(ks)——recall 与 MRR 共用同一次召回。调参取 cfg（与服务路径同源），
    保证基线量的是“正在服役的检索器”（先测量后加装）。
    """
    per_row: list[dict] = []
    for row in rows:
        hits = hybrid_recall(store, embedder, row["query"], max(ks), row.get("project"),
                             rrf_k=cfg.rrf_k, inject_boost=cfg.inject_boost,
                             min_cosine=cfg.recall_min_cosine,
                             max_cosine=cfg.recall_max_cosine,
                             reranker=reranker, rerank_candidates=cfg.rerank_candidates)
        expect = set(row["expect_uids"])
        first = next((i for i, h in enumerate(hits, 1) if h["uid"] in expect), 0)
        per_row.append({
            "query": row["query"],
            "hit_ks": {k: 0 < first <= k for k in ks},
            "mrr": 1.0 / first if first else 0.0,
        })
    n = len(per_row)
    return {
        "n": n,
        "recall_at_k": {str(k): sum(r["hit_ks"][k] for r in per_row) / n for k in ks},
        "mrr": sum(r["mrr"] for r in per_row) / n,
        "rows": per_row,
    }


def _judge_misses(missed: list[dict], golden_rows: list[dict]) -> None:
    """--judge 通道（v0.3 接线）：对未命中行裁决「检索漏召回 vs golden 标签可疑」。

    未命中 = 期望 uid 未进召回面。取期望条目内容交 Reranker 打 query/doc 相关分：
    top 分 ≥0.5 → 标签可信，是检索真 miss；<0.5 → golden 标签本身可疑（伪 miss）。
    Reranker 不可用/超预算 → 打印跳过，退出码不变（fail-open）。
    """
    if not missed:
        return
    if reranker is None or not reranker.available:
        print(f"[eval] judge: 重排模型不可用（MEMORY_RERANK_MODEL），跳过 {len(missed)} 条未命中评注")
        return
    by_query = {r["query"]: r for r in golden_rows}
    for row in missed:
        g = by_query.get(row["query"])
        if not g:
            continue
        cands = [(r["uid"], r["content"]) for r in store.get_many(list(g["expect_uids"]))]
        if not cands:
            print(f"[judge] {row['query'][:40]!r}: 期望条目均已不在库（遗忘/被取代）——无法裁决")
            continue
        ranked = reranker.rerank(row["query"], cands)
        if not ranked:
            print(f"[judge] {row['query'][:40]!r}: 重排失败（超预算/异常）——无法裁决")
            continue
        top_uid, top_score = ranked[0]
        verdict = ("检索漏召回（golden 标签可信）" if top_score >= 0.5
                   else "golden 标签可疑（内容与查询相关性低）")
        print(f"[judge] {row['query'][:40]!r}: top相关分={top_score:.2f} → {verdict}（uid={top_uid}）")


def _cmd_eval(argv: list[str]) -> int:
    """eval 子命令：golden 集上的 recall@k / MRR 基线（纯 CLI，禁入任何服务端点）。

    检索侧每次改动都用同一命令对照基线——先测量后加装。
    """
    ap = argparse.ArgumentParser(
        prog="semantic-memory eval",
        description="对 golden jsonl 逐行跑混合检索，输出 recall@k 与 MRR 基线报告",
    )
    ap.add_argument("--golden", required=True,
                    help="golden jsonl 路径，每行 {query, expect_uids[], project?}")
    ap.add_argument("--k", default="1,3,5", help="逗号分隔的 k 值列表（默认 1,3,5）")
    ap.add_argument("--min-recall", dest="min_recall", default=None, metavar="K:FLOAT",
                    help="门禁：如 5:0.85 表示 recall@5 < 0.85 时退出码 1（CI/回归门用）")
    ap.add_argument("--judge", action="store_true",
                    help="对未命中行裁决检索 miss vs golden 标签可疑（需 Reranker）")
    ns = ap.parse_args(argv)

    golden = Path(ns.golden)
    if not golden.is_file():
        print(f"eval: golden 文件不存在: {golden}", file=sys.stderr)
        return 2
    try:
        ks = sorted({int(x) for x in ns.k.split(",") if x.strip()})
    except ValueError:
        ks = []
    if not ks or ks[0] <= 0:
        print(f"eval: --k 须为逗号分隔的正整数: {ns.k!r}", file=sys.stderr)
        return 2
    try:
        rows = _load_golden(golden)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"eval: golden 格式错误: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"eval: golden 无有效行: {golden}", file=sys.stderr)
        return 2

    report = run_eval(store, embedder, rows, ks)

    for row in report["rows"]:
        marks = "  ".join(f"@{k}={'Y' if row['hit_ks'][k] else '-'}" for k in ks)
        print(f"{row['query'][:48]:<48}  {marks}  mrr={row['mrr']:.3f}")
    summary = "  ".join(f"recall@{k}={report['recall_at_k'][str(k)]:.3f}" for k in ks)
    print(f"n={report['n']}  {summary}  mrr={report['mrr']:.3f}")
    print(json.dumps(report, ensure_ascii=False))

    if ns.min_recall:
        try:
            k_s, v_s = ns.min_recall.split(":", 1)
            k_gate, v_gate = int(k_s), float(v_s)
        except ValueError:
            print(f"eval: --min-recall 须为 K:FLOAT 形式（如 5:0.85）: {ns.min_recall!r}", file=sys.stderr)
            return 2
        actual = report["recall_at_k"].get(str(k_gate))
        if actual is None:
            print(f"eval: --min-recall 的 k={k_gate} 不在 --k 列表 {ks}", file=sys.stderr)
            return 2
        if actual < v_gate:
            print(f"eval: recall@{k_gate}={actual:.3f} < 门槛 {v_gate:.3f} —— 门禁未过", file=sys.stderr)
            return 1

    if ns.judge:
        _judge_misses([r for r in report["rows"] if r["mrr"] == 0.0], rows)
    return 0


def _cmd_golden(argv: list[str]) -> int:
    """golden 子命令（评测飞轮）：注入遥测 → golden jsonl，供 eval 消费。

    供给侧语义：expect_uids 取实际注入过的 uid——注入≠模型采纳，但作为
    recall@k 期望集自洽（检索目标就是复现真实注入面）。按 (project, query)
    去重、取最近注入；真实使用即标注数据。
    """
    ap = argparse.ArgumentParser(
        prog="semantic-memory golden",
        description="从注入遥测构建 golden jsonl（每行 {query, expect_uids[], project?}）")
    ap.add_argument("--out", default="-", help="输出路径（默认 - = stdout）")
    ap.add_argument("--limit", type=int, default=200, help="最多导出行数")
    ns = ap.parse_args(argv)

    seen: set = set()
    out_rows: list[dict] = []
    for inj in store.list_injections(limit=ns.limit * 4):
        q = (inj["query"] or "").strip()
        try:
            uids = [u for u in json.loads(inj["uids"]) if isinstance(u, str)]
        except (json.JSONDecodeError, TypeError):
            continue
        if not q or not uids:
            continue
        key = (inj.get("project") or "", q)
        if key in seen:
            continue
        seen.add(key)
        out_rows.append({"query": q[:2000], "expect_uids": uids,
                         "project": inj.get("project") or None})
        if len(out_rows) >= ns.limit:
            break
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    if ns.out == "-":
        sys.stdout.write(payload)
    else:
        Path(ns.out).write_text(payload, encoding="utf-8")
    print(f"golden: {len(out_rows)} 行 → {ns.out}", file=sys.stderr)
    return 0


def _cmd_consolidate() -> None:
    """consolidate 子命令：云端整合手动入口（同步执行一次，打印报告 JSON）。"""
    print(json.dumps(consolidator.run(store, embedder), ensure_ascii=False))


def _cmd_supersede(argv: list[str]) -> int:
    """supersede 子命令：人工温层退役入口（写 superseded_by/at，召回面即刻滤除）。

    UPDATE 与 cloud.Consolidator 同款：只接未退役（superseded_by IS NULL）且未删除的条目。
    """
    ap = argparse.ArgumentParser(
        prog="supersede",
        description="人工温层退役：标记旧记忆被同 project 的新记忆接替",
    )
    ap.add_argument("old", help="待退役 uid")
    ap.add_argument("--by", dest="new", required=True, help="接替 uid")
    ns = ap.parse_args(argv)
    old, new = ns.old, ns.new
    if old == new:
        print("refused: old == new", file=sys.stderr)
        return 2
    rows = store.conn.execute(
        "SELECT uid, project FROM memories WHERE uid IN (?, ?) AND deleted = 0",
        (old, new)).fetchall()
    by_uid = {r["uid"]: r["project"] for r in rows}
    missing = [u for u in (old, new) if u not in by_uid]
    for u in missing:
        print(json.dumps({"error": "not found", "uid": u}, ensure_ascii=False),
              file=sys.stderr)
    if missing:
        return 1
    if by_uid[old] != by_uid[new]:
        print(json.dumps({"error": "cross-project supersede refused",
                          "old": old, "new": new,
                          "old_project": by_uid[old], "new_project": by_uid[new]},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    now = time.time()
    with store.conn:
        cur = store.conn.execute(
            "UPDATE memories SET superseded_by=?, superseded_at=?"
            " WHERE uid=? AND deleted=0 AND superseded_by IS NULL",
            (new, now, old),
        )
    print(json.dumps({"superseded": cur.rowcount, "old": old, "new": new},
                     ensure_ascii=False))
    return 0


# ---------------- 入口 ----------------

_USAGE = """usage: semantic-memory [-h] [command]

语义记忆服务（单进程）。默认命令 serve：启动 HTTP 服务（REST 面）。
管理子命令：setup / purge / backfill / eval / golden / consolidate / distill / sweep / supersede（详见各 --help）。
"""


def _setup_logging() -> None:
    """stderr 之外强制落盘：autostart spawn 用 stdio:"ignore" 丢弃 stderr——
    熔断原因等诊断信息必须进文件，否则熔断不可诊断（2026-10-22 实证 16h 盲态）。"""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    raw = os.environ.get("MEMORY_LOG_FILE")
    log_file = Path(raw) if raw else resolve_data_dir() / "logs" / "server.log"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            log_file, maxBytes=5 * 2**20, backupCount=2))
    except OSError:
        log.warning("日志文件不可写（%s）——仅 stderr", log_file)
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main() -> None:
    _setup_logging()
    args = sys.argv[1:]
    cmd = next((a for a in args if not a.startswith("-")), "serve")
    if cmd == "serve" and ("--help" in args or "-h" in args):
        print(_USAGE)
        return

    # ---- 管理子命令（维护面：LLM 够不到的地方）----
    if cmd == "setup":  # 安装面（任务 8）：探测→校验下载→写 config→冒烟；零核心单例成本
        from .setup import run_setup
        raise SystemExit(run_setup(args[1:]))
    init_core()  # 其余子命令与 serve 都需要核心单例（惰性物化点）
    if cmd == "purge":
        report = run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days,
                          ttl_injection_days=cfg.ttl_injection_days,
                          ttl_stale_days=cfg.ttl_stale_days,
                          dry_run="--dry-run" in args)
        print(json.dumps(report, ensure_ascii=False))
        return
    if cmd == "backfill":
        print(json.dumps(backfill_vectors(store, embedder), ensure_ascii=False))
        return
    if cmd == "eval":
        raise SystemExit(_cmd_eval(args[1:]))
    if cmd == "golden":
        raise SystemExit(_cmd_golden(args[1:]))
    if cmd == "consolidate":
        _cmd_consolidate()
        return
    if cmd == "distill":  # 触发器三路：手动 flush（同步跑完打印报告）
        print(json.dumps(flush_distill_queue(store, distiller, embedder=embedder), ensure_ascii=False))
        return
    if cmd == "sweep":  # 一致性巡检：悬空路径引用候选报告，只报告不删除（悬空≠无效）
        print(json.dumps(run_sweep(store), ensure_ascii=False))
        return
    if cmd == "supersede":  # 人工温层退役入口（cloud.Consolidator 同款 UPDATE）
        raise SystemExit(_cmd_supersede(args[1:]))

    # ---- 分层严格启动门（任务 9/R2）：拒启必须先于端口绑定 ----
    gate = check_startup(cfg)
    if gate:
        raise SystemExit(gate)

    # ---- serve：启动维护钩子（频率=服务重启，天级粒度足够）----
    if cfg.maintenance_on_start:
        log.info("startup maintenance: %s",
                 run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days,
                          ttl_injection_days=cfg.ttl_injection_days,
                          ttl_stale_days=cfg.ttl_stale_days))
    # 存量回填（Metis M-3）：目录路径型 project 改写为归一化键；幂等，唯一旧值批量不逐行 spawn
    log.info("project key backfill: %s", backfill_project_keys(store))

    log.info("db=%s embedder=%s prefer=%s", cfg.db_path, embedder.available, cfg.device)
    if not HAS_NUMPY:
        log.info("numpy 未安装——向量通道关闭（纯 BM25）。%s", INSTALL_HINT)
    # 非回环无 token 的拒启在 startup_gate（check_startup 先于此执行）；此处不再重复告警
    # MEMORY_PORT=0 → OS 分配临时端口，绑定后通过发现文件/stdout 发布真实地址
    serve()


if __name__ == "__main__":
    main()
