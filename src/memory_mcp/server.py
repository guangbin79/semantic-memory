"""FastMCP 服务：MCP 工具（agent 主动调用）+ REST 侧车（opencode v2 插件调用）。

单进程双接口：HTTP transport 下 MCP 端点与 REST 端点同端口，
保证插件和 MCP 客户端操作同一份索引、同一个嵌入模型。
"""
from __future__ import annotations

import json
import logging
import sys

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from .web import INDEX_HTML

from .config import Config
from .db import MemoryStore
from .embedder import Embedder
from .hybrid import hybrid_recall
from .lifecycle import backfill_vectors, ingest_with_dedup, run_maintenance

log = logging.getLogger("memory_mcp")

cfg = Config.from_env()
store = MemoryStore(cfg.db_path)
embedder = Embedder(cfg.model_path, dim=cfg.embed_dim, device=cfg.device)

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
        store, embedder, content, kind, "manual", project, tags, cfg.dedup_sim
    )
    return {"uid": uid, "deduped": deduped, "vectorized": embedder.available}


@mcp.tool
def memory_recall(
    query: str,
    limit: int = 5,
    project: str | None = None,
) -> list[dict]:
    """混合检索相关记忆，按 RRF 分数降序。"""
    return hybrid_recall(store, embedder, query, limit, project, rrf_k=cfg.rrf_k)


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
    """库统计：总量、向量化比例、按 kind 分布、嵌入通道状态。"""
    return {**store.stats(), "embedder": embedder.available, "device": embedder.device}


# ---------------- REST 侧车（插件专用） ----------------

def _authorized(request: Request) -> bool:
    if cfg.token is None:
        return True  # 未配置 token：依赖 127.0.0.1 回环绑定
    auth = request.headers.get("authorization", "")
    return auth == f"Bearer {cfg.token}"


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
    return JSONResponse({**store.stats(), "embedder": embedder.available,
                         "device": embedder.device})


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
    return JSONResponse(run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days))


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
    hits = hybrid_recall(
        store, embedder,
        query=body["query"],
        limit=int(body.get("limit", cfg.recall_limit_default)),
        project=body.get("project"),
        rrf_k=cfg.rrf_k,
    )
    return JSONResponse({"hits": hits})


@mcp.custom_route("/ingest", methods=["POST"])
async def rest_ingest(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    uid, deduped = ingest_with_dedup(
        store, embedder,
        content=body["content"],
        kind=body.get("kind", "fact"),
        source=body.get("source", "chat"),
        project=body.get("project"),
        tags=body.get("tags"),
        dedup_sim=cfg.dedup_sim,
    )
    return JSONResponse({"uid": uid, "deduped": deduped})


# ---------------- 入口 ----------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = sys.argv[1:]
    cmd = next((a for a in args if not a.startswith("-")), "serve")

    # ---- 管理子命令（维护面：LLM 够不到的地方）----
    if cmd == "purge":
        report = run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days)
        print(json.dumps(report, ensure_ascii=False))
        return
    if cmd == "backfill":
        print(json.dumps(backfill_vectors(store, embedder), ensure_ascii=False))
        return

    # ---- serve：启动维护钩子（频率=服务重启，天级粒度足够）----
    if cfg.maintenance_on_start:
        log.info("startup maintenance: %s",
                 run_maintenance(store, ttl_artifact_days=cfg.ttl_artifact_days))

    transport = "stdio" if "--stdio" in args else "http"
    log.info("db=%s embedder=%s prefer=%s transport=%s", cfg.db_path, embedder.available, cfg.device, transport)
    if cfg.token is None and cfg.host != "127.0.0.1":
        log.warning("MEMORY_TOKEN 未设置且绑定非回环地址 %s，REST 侧车无鉴权！", cfg.host)
    if transport == "stdio":
        # stdio 模式仅暴露 MCP 工具（REST 侧车不可用）
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=cfg.host, port=cfg.port, path="/mcp")


if __name__ == "__main__":
    main()
