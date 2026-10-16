"""FastMCP 服务：MCP 工具（agent 主动调用）+ REST 侧车（opencode v2 插件调用）。

单进程双接口：HTTP transport 下 MCP 端点与 REST 端点同端口，
保证插件和 MCP 客户端操作同一份索引、同一个嵌入模型。
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import math
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import uvicorn
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from .web import INDEX_HTML

from .cloud import Consolidator
from .config import Config
from .db import MemoryStore
from .distiller import Distiller
from .embedder import Embedder
from .hybrid import hybrid_recall
from .lifecycle import (
    backfill_vectors,
    enqueue_raw,
    flush_distill_queue,
    ingest_with_dedup,
    run_maintenance,
    stamp_recalled,
)
from .project import backfill_project_keys, resolve_project
from .modelslot import ModelSlot
from .reranker import Reranker

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
    """观测面字段：/api/stats 与 memory_stats 共用同一份。"""
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

cfg = Config.from_env()
store = MemoryStore(cfg.db_path)
embedder = Embedder(cfg.model_path, dim=cfg.embed_dim, device=cfg.device)
_slot = ModelSlot()  # 单槽分时：rerank 与 distill 互斥换载（embedder 常驻槽外）
reranker = Reranker(cfg.rerank_model, device=cfg.device, budget_ms=cfg.rerank_budget_ms, slot=_slot)
distiller = Distiller(cfg.distill_model, cfg.device, slot=_slot)  # 惰性加载：首次 distill 才经槽加载模型
consolidator = Consolidator(
    cfg.cloud_base_url, cfg.cloud_api_key, cfg.cloud_model,
    enabled=cfg.cloud_enabled, daily_max_calls=cfg.cloud_daily_max_calls,
    daily_max_tokens=cfg.cloud_daily_max_tokens, batch_n=cfg.cloud_batch_n,
)

mcp = FastMCP(
    name="memory",
    instructions=(
        "语义记忆库。memory_store 存入事实/决策/偏好/产物元数据；"
        "memory_recall 混合检索（BM25+向量 RRF）；memory_forget 软删除。"
        "长事实请浓缩为单句再存入，tags 用于横向归类。"
    ),
)


# ---------------- MCP 工具 ----------------

@mcp.tool
def memory_store(
    content: str,
    kind: str = "fact",
    tags: list[str] | None = None,
    project: str | None = None,
) -> dict:
    """存入一条记忆。kind: fact|decision|preference|artifact。返回 uid。

    写入路径内嵌去重：内容相同（或同项目内向量近似）时刷新已有条目
    时间戳而非新增——deduped=true 表示命中去重。
    """
    uid, deduped = ingest_with_dedup(
        store, embedder, content, kind, "manual", resolve_project(project), tags, cfg.dedup_sim
    )
    return {"uid": uid, "deduped": deduped, "vectorized": embedder.available}


@mcp.tool
def memory_recall(
    query: str,
    limit: int = 5,
    project: str | None = None,
    global_limit: int = 0,
) -> list[dict]:
    """混合检索相关记忆，按 RRF 分数降序。

    global_limit>0 且指定 project 时走 project+global 两路合并：
    项目条目在前，全局（project=NULL）条目补足其后，总数 ≤ limit。
    """
    return _recall_hits(query, limit, resolve_project(project), global_limit)


def _recall_hits(query: str, limit: int, project: str | None, global_limit: int) -> list[dict]:
    """混合召回；global_limit>0 且指定 project 时两路合并（REST/MCP 共用）。

    project 路（scope='project'）在前，global 路（scope='global'，配额
    global_limit）补足其后；去重后按 limit 截断——全局条目不挤占项目配额。
    global_limit=0 或未指定 project 时单路调用，与既有行为逐字节一致。
    """
    if global_limit > 0 and project:
        proj_hits = hybrid_recall(
            store, embedder, query=query, limit=limit, project=project,
            rrf_k=cfg.rrf_k, inject_boost=cfg.inject_boost,
            min_cosine=cfg.recall_min_cosine, max_cosine=cfg.recall_max_cosine,
            reranker=reranker, rerank_candidates=cfg.rerank_candidates,
        )
        glob_hits = hybrid_recall(
            store, embedder, query=query, limit=global_limit, project=None,
            rrf_k=cfg.rrf_k, inject_boost=cfg.inject_boost,
            min_cosine=cfg.recall_min_cosine, max_cosine=cfg.recall_max_cosine,
            reranker=reranker, rerank_candidates=cfg.rerank_candidates,
            scope="global",
        )
        seen = {h["uid"] for h in proj_hits}
        return (proj_hits + [h for h in glob_hits if h["uid"] not in seen])[:limit]
    return hybrid_recall(
        store, embedder, query=query, limit=limit, project=project,
        rrf_k=cfg.rrf_k, inject_boost=cfg.inject_boost,
        min_cosine=cfg.recall_min_cosine, max_cosine=cfg.recall_max_cosine,
        reranker=reranker, rerank_candidates=cfg.rerank_candidates,
    )


@mcp.tool
def memory_forget(uid: str) -> dict:
    """按 uid 软删除一条记忆。"""
    return {"uid": uid, "forgotten": store.forget(uid)}


@mcp.tool
def memory_list(project: str | None = None, limit: int = 20) -> list[dict]:
    """按时间倒序列出记忆。"""
    rows = store.list_recent(project=project, limit=limit)
    return [
        {
            "uid": r["uid"], "content": r["content"], "kind": r["kind"],
            "project": r["project"], "tags": r["tags"], "created_at": r["created_at"],
        }
        for r in rows
    ]


@mcp.tool
def memory_stats() -> dict:
    """库统计：总量、向量化比例、按 kind 分布、嵌入通道状态、注入总量、观测面。"""
    return {**store.stats(), "embedder": embedder.available, "device": embedder.device,
            "injections": store.injection_stats()["total"],
            "cloud": consolidator.stats(), **_obs_stats()}


# ---------------- REST 侧车（插件专用） ----------------

def _authorized(request: Request) -> bool:
    if cfg.token is None:
        return True  # 未配置 token：依赖 127.0.0.1 回环绑定
    auth = request.headers.get("authorization", "")
    return hmac.compare_digest(auth.encode("latin-1"), f"Bearer {cfg.token}".encode())  # 常量时间比较（bytes 防 non-ASCII TypeError）


class _McpAuthMiddleware:
    """把 REST 侧车的 Bearer 门同样罩到 /mcp：FastMCP 未配 auth provider 时 /mcp 本为裸奔。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/mcp" \
                and cfg.token is not None:
            headers = {k.lower(): v for k, v in scope.get("headers") or []}
            auth = headers.get(b"authorization", b"")  # 保持 bytes：latin-1 str 会触发 compare_digest TypeError（500）
            if not hmac.compare_digest(auth, f"Bearer {cfg.token}".encode("latin-1")):
                resp = JSONResponse({"error": "unauthorized"}, status_code=401)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)

@mcp.custom_route("/health", methods=["GET"])
async def rest_health(_: Request) -> JSONResponse:
    s = store.stats()
    return JSONResponse({"status": "ok", "embedder": embedder.available, "device": embedder.device, **s})


# ---------------- 人用读/管接口 + 内置浏览页 ----------------
# 读接口 + 单条遗忘：与 CLI 同级的人工信道。批量销毁依旧只走启动钩子/CLI。

@mcp.custom_route("/", methods=["GET"])
async def web_index(_: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@mcp.custom_route("/api/stats", methods=["GET"])
async def api_stats(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    inj = store.injection_stats()
    # 热门条目附带内容摘要（遗忘过的 uid 解析为 None，前端自行跳过）
    top = []
    for t in inj.pop("top_uids"):
        rows = store.get_many([t["uid"]])
        top.append({"uid": t["uid"], "times": t["times"],
                    "content": rows[0]["content"][:120] if rows else None,
                    "kind": rows[0]["kind"] if rows else None})
    return JSONResponse({**store.stats(), "embedder": embedder.available,
                         "device": embedder.device,
                         "injections": {**inj, "top": top}, **_obs_stats()})


@mcp.custom_route("/api/injections", methods=["GET"])
async def api_injections(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    q = request.query_params
    rows = store.list_injections(
        harness=q.get("harness") or None,
        project=q.get("project") or None,
        limit=min(int(q.get("limit", 50)), 200),
        offset=int(q.get("offset", 0)),
    )
    return JSONResponse({"injections": rows})


@mcp.custom_route("/api/memories", methods=["GET"])
async def api_memories(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    q = request.query_params
    rows = store.list_recent(
        project=q.get("project") or None,
        kind=q.get("kind") or None,
        limit=min(int(q.get("limit", 50)), 200),
        offset=int(q.get("offset", 0)),
    )
    return JSONResponse({"memories": [
        {"uid": r["uid"], "content": r["content"], "kind": r["kind"],
         "source": r["source"], "project": r["project"], "tags": r["tags"],
         "created_at": r["created_at"], "updated_at": r["updated_at"]}
        for r in rows
    ]})


@mcp.custom_route("/api/memories/{uid}", methods=["DELETE"])
async def api_forget(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    uid = request.path_params["uid"]
    return JSONResponse({"uid": uid, "forgotten": store.forget(uid)})


# ---------------- 批量维护（双闸门：必须配置 token + 请求鉴权） ----------------
# 默认无 token 的回环安装下，HTTP 面对本机进程（含 agent shell）是开放的，
# 因此批量破坏性操作此时仅 CLI 可达（403）；配置 token 后才对持票人开放。

def _bulk_guard(request: Request) -> JSONResponse | None:
    if cfg.token is None:
        return JSONResponse(
            {"error": "bulk maintenance disabled without MEMORY_TOKEN; use the CLI"},
            status_code=403,
        )
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


@mcp.custom_route("/api/maintenance/purge", methods=["POST"])
async def api_maintenance_purge(request: Request) -> JSONResponse:
    guard = _bulk_guard(request)
    if guard:
        return guard
    return JSONResponse(run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days,
                          ttl_injection_days=cfg.ttl_injection_days,
                          ttl_stale_days=cfg.ttl_stale_days))


@mcp.custom_route("/api/maintenance/backfill", methods=["POST"])
async def api_maintenance_backfill(request: Request) -> JSONResponse:
    guard = _bulk_guard(request)
    if guard:
        return guard
    return JSONResponse(backfill_vectors(store, embedder))


@mcp.custom_route("/recall", methods=["POST"])
async def rest_recall(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    t0 = time.perf_counter()
    hits = _recall_hits(
        query=body["query"],
        limit=int(body.get("limit", cfg.recall_limit_default)),
        project=resolve_project(body.get("project")),
        global_limit=int(body.get("global_limit", 0)),
    )
    _record_recall((time.perf_counter() - t0) * 1000.0)
    return JSONResponse({"hits": hits})


# 入队路由（r3 评审）：chat/compaction 是蒸馏队列的投料源——落盘缓冲、热路径零 LLM；
# 显式 remember / tool capture 保持直存（即时可见，也避免队列无人投料的死管道）。
QUEUED_SOURCES = frozenset({"compaction", "chat"})


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


@mcp.custom_route("/idle", methods=["POST"])
async def rest_idle(request: Request) -> Response:
    """空闲触发蒸馏 flush：先回 204，flush 在 daemon 线程执行（热路径零 LLM）。"""
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    _spawn_flush()
    return Response(status_code=204)


@mcp.custom_route("/ingest", methods=["POST"])
async def rest_ingest(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    source = body.get("source", "chat")
    project = resolve_project(body.get("project"))
    if source in QUEUED_SOURCES:
        pending = enqueue_raw(store, body["content"], source, project,
                              body.get("session_id"))
        if cfg.distill_batch > 0 and pending >= cfg.distill_batch:
            _spawn_flush()  # 触发器一路：队列积压达阈值（flush 在 daemon 线程）
        return JSONResponse({"queued": True, "pending": pending})
    uid, deduped = ingest_with_dedup(
        store, embedder,
        content=body["content"],
        kind=body.get("kind", "fact"),
        source=source,
        project=project,
        tags=body.get("tags"),
        dedup_sim=cfg.dedup_sim,
    )
    return JSONResponse({"uid": uid, "deduped": deduped})


@mcp.custom_route("/inject/report", methods=["POST"])
async def inject_report(request: Request) -> JSONResponse:
    """注入遥测上报（插件/适配器信道，只追加不删除）。

    只有注入方知道"这次召回真的注进了上下文"——服务端只见召回。
    记录供给侧事实：谁(harness)、哪个钩子、触发的查询、注入了哪些
    条目、多大块。模型是否采纳不可知，此处不做伪精确度量。
    """
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    uids = body.get("uids")
    if not isinstance(uids, list) or not all(isinstance(u, str) for u in uids):
        return JSONResponse({"error": "uids must be a string array"}, status_code=400)
    try:
        chars = int(body.get("chars", 0))
    except (TypeError, ValueError):
        return JSONResponse({"error": "chars must be an integer"}, status_code=400)
    rid = store.record_injection(
        harness=str(body.get("harness", "")),
        hook=str(body.get("hook", "")),
        query=str(body.get("query", "")),
        uids=uids,
        chars=chars,
        project=body.get("project"),
    )
    stamp_recalled(store, uids)
    return JSONResponse({"recorded": True, "id": rid})


# ---------------- 服务发现（随机端口自举） ----------------

def _announce(port: int) -> dict:
    """绑定完成后对外发布真实地址：发现文件 + stdout 机器可读行。

    agent 自举流程：spawn 本进程（MEMORY_PORT=0）→ 读 discovery_file
    或解析 stdout 的 MEMORY_MCP_LISTENING 行 → 得到 url。
    """
    url = f"http://{cfg.host}:{port}"
    info = {
        "url": url,
        "mcp_url": f"{url}/mcp",
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
    print(f"MEMORY_MCP_LISTENING {json.dumps(info, ensure_ascii=False)}", flush=True)
    log.info("listening on %s (mcp: %s/mcp)", url, url)
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
    """SIGINT/SIGTERM → SystemExit：uvicorn 优雅停机后会以默认处置重放信号
    （rc=143，Python 栈不再展开）；换成 SystemExit 让 finally/atexit 能撤下
    发现文件，退出码仍保持 128+sig 语义。"""

    def _handler(sig, _frame):
        raise SystemExit(128 + sig)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


async def _serve_http_async() -> None:
    """程序化驱动 uvicorn：只有拿到 server 对象，才能在 port=0 时读回真实端口。"""
    _install_exit_handler()
    app = mcp.http_app(path="/mcp", host_origin_protection="auto")  # 回环绑定默认开 Host/Origin 校验（DNS rebinding 防护）
    app = _McpAuthMiddleware(app)  # /mcp 与 REST 同一 Bearer 门（轮 2 安全评审：MCP 端点此前绕过 MEMORY_TOKEN）
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port,
                                           log_level="warning"))
    task = asyncio.create_task(server.serve())
    for _ in range(600):  # 启动完成探测，最长 30s
        if server.started:
            break
        if task.done():
            task.result()  # 抛出真实启动错误
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("uvicorn startup timeout (30s)")
    port = server.servers[0].sockets[0].getsockname()[1]
    _announce(port)
    try:
        await task
    finally:
        _retract()


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


def _judge_misses(missed: list[dict]) -> None:
    """--judge 通道：对未命中样本做相关性抽样评注（Wave 3 Distiller 接线点）。

    环境未配或模块未实装都只打印跳过提示——不报错，退出码保持 0。
    """
    if not missed:
        return
    if not os.environ.get("MEMORY_DISTILL_MODEL"):
        print(f"[eval] judge: MEMORY_DISTILL_MODEL 未配置，跳过 {len(missed)} 条未命中样本评注")
        return
    try:
        from .distiller import Distiller
    except ImportError:
        print(f"[eval] judge: Distiller 未实装，跳过 {len(missed)} 条未命中样本评注")
        return
    # 评注接线未实装：Distiller 模块已交付，逐条评注实现留待后续任务
    print(f"[eval] judge: Distiller 已可用，评注接线未实装（{len(missed)} 条未命中）")


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
    ap.add_argument("--judge", action="store_true",
                    help="对未命中样本做相关性抽样评注（需 Distiller，Wave 3）")
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

    if ns.judge:
        _judge_misses([r for r in report["rows"] if r["mrr"] == 0.0])
    return 0


def _cmd_consolidate() -> None:
    """consolidate 子命令：云端整合手动入口（同步执行一次，打印报告 JSON）。"""
    print(json.dumps(consolidator.run(store, embedder), ensure_ascii=False))


_USAGE = """usage: semantic-memory [-h] [--stdio] [command]

语义记忆服务（单进程）。默认命令 serve：启动 HTTP 服务（MCP + REST 同端口）。
管理子命令：purge / backfill / eval / consolidate / distill（详见各 --help）。
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = sys.argv[1:]
    cmd = next((a for a in args if not a.startswith("-")), "serve")
    if cmd == "serve" and ("--help" in args or "-h" in args):
        print(_USAGE)
        return

    # ---- 管理子命令（维护面：LLM 够不到的地方）----
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
    if cmd == "consolidate":
        _cmd_consolidate()
        return
    if cmd == "distill":  # 触发器三路：手动 flush（同步跑完打印报告）
        print(json.dumps(flush_distill_queue(store, distiller, embedder=embedder), ensure_ascii=False))
        return

    # ---- serve：启动维护钩子（频率=服务重启，天级粒度足够）----
    if cfg.maintenance_on_start:
        log.info("startup maintenance: %s",
                 run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days,
                          ttl_injection_days=cfg.ttl_injection_days,
                          ttl_stale_days=cfg.ttl_stale_days))
    # 存量回填（Metis M-3）：目录路径型 project 改写为归一化键；幂等，唯一旧值批量不逐行 spawn
    log.info("project key backfill: %s", backfill_project_keys(store))

    transport = "stdio" if "--stdio" in args else "http"
    log.info("db=%s embedder=%s prefer=%s transport=%s", cfg.db_path, embedder.available, cfg.device, transport)
    if cfg.token is None and cfg.host != "127.0.0.1":
        log.warning("MEMORY_TOKEN 未设置且绑定非回环地址 %s，REST 侧车无鉴权！", cfg.host)
    if transport == "stdio":
        # stdio 模式仅暴露 MCP 工具（REST 侧车不可用，无端口与发现文件）
        mcp.run(transport="stdio")
    else:
        # MEMORY_PORT=0 → OS 分配临时端口，绑定后通过发现文件/stdout 发布真实地址
        asyncio.run(_serve_http_async())


if __name__ == "__main__":
    main()
