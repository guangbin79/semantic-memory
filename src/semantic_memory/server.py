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
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

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


# ---------------- 检索核心（REST 端点共用） ----------------

def _recall_hits(query: str, limit: int, project: str | None, global_limit: int) -> list[dict]:
    """混合召回；global_limit>0 且指定 project 时两路合并（REST 共用）。

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


# ---------------- REST 端点（鉴权与传输门由 _Handler 负责） ----------------

def rest_health() -> dict:
    s = store.stats()
    return {"status": "ok", "embedder": embedder.available, "device": embedder.device, **s}


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
    return {**store.stats(), "embedder": embedder.available,
            "device": embedder.device,
            "injections": {**inj, "top": top}, **_obs_stats()}


def api_injections(q: dict) -> dict:
    rows = store.list_injections(
        harness=q.get("harness") or None,
        project=q.get("project") or None,
        limit=min(int(q.get("limit", 50)), 200),
        offset=int(q.get("offset", 0)),
    )
    return {"injections": rows}


def api_memories(q: dict) -> dict:
    rows = store.list_recent(
        project=q.get("project") or None,
        kind=q.get("kind") or None,
        limit=min(int(q.get("limit", 50)), 200),
        offset=int(q.get("offset", 0)),
    )
    return {"memories": [
        {"uid": r["uid"], "content": r["content"], "kind": r["kind"],
         "source": r["source"], "project": r["project"], "tags": r["tags"],
         "created_at": r["created_at"], "updated_at": r["updated_at"]}
        for r in rows
    ]}


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


def rest_recall(body: dict) -> dict:
    t0 = time.perf_counter()
    hits = _recall_hits(
        query=body["query"],
        limit=int(body.get("limit", cfg.recall_limit_default)),
        project=resolve_project(body.get("project")),
        global_limit=int(body.get("global_limit", 0)),
    )
    _record_recall((time.perf_counter() - t0) * 1000.0)
    return {"hits": hits}


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


def rest_ingest(body: dict) -> dict:
    source = body.get("source", "chat")
    project = resolve_project(body.get("project"))
    if source in QUEUED_SOURCES:
        pending = enqueue_raw(store, body["content"], source, project,
                              body.get("session_id"))
        if cfg.distill_batch > 0 and pending >= cfg.distill_batch:
            _spawn_flush()  # 触发器一路：队列积压达阈值（flush 在 daemon 线程）
        return {"queued": True, "pending": pending}
    uid, deduped = ingest_with_dedup(
        store, embedder,
        content=body["content"],
        kind=body.get("kind", "fact"),
        source=source,
        project=project,
        tags=body.get("tags"),
        dedup_sim=cfg.dedup_sim,
    )
    return {"uid": uid, "deduped": deduped}


def inject_report(body: dict) -> tuple[int, dict]:
    """注入遥测上报（插件/适配器信道，只追加不删除）。

    只有注入方知道"这次召回真的注进了上下文"——服务端只见召回。
    记录供给侧事实：谁(harness)、哪个钩子、触发的查询、注入了哪些
    条目、多大块。模型是否采纳不可知，此处不做伪精确度量。
    """
    uids = body.get("uids")
    if not isinstance(uids, list) or not all(isinstance(u, str) for u in uids):
        return 400, {"error": "uids must be a string array"}
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
    )
    stamp_recalled(store, uids)
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

    def _dispatch(self, method: str) -> None:
        try:
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
                return self._send_json(api_injections(q))
            if path == "/api/memories":
                if not _authorized(self.headers):
                    return self._send_json({"error": "unauthorized"}, 401)
                return self._send_json(api_memories(q))
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
            if not self._origin_ok():
                return self._send_json({"error": "cross-origin write rejected"}, 403)
            body = self._read_body()  # 先读后鉴权：任何错误响应后连接内不留未读字节
            if body is None:
                return
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
                    return self._send_json(rest_recall(parsed))
                if path == "/ingest":
                    return self._send_json(rest_ingest(parsed))
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


# ---------------- 入口 ----------------

_USAGE = """usage: semantic-memory [-h] [command]

语义记忆服务（单进程）。默认命令 serve：启动 HTTP 服务（REST 面）。
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

    log.info("db=%s embedder=%s prefer=%s", cfg.db_path, embedder.available, cfg.device)
    if cfg.token is None and cfg.host != "127.0.0.1":
        log.warning("MEMORY_TOKEN 未设置且绑定非回环地址 %s，REST 侧车无鉴权！", cfg.host)
    # MEMORY_PORT=0 → OS 分配临时端口，绑定后通过发现文件/stdout 发布真实地址
    serve()


if __name__ == "__main__":
    main()
