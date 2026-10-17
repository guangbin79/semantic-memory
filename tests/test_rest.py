"""REST 面与 API 路由测试：进程内 ThreadingHTTPServer(port 0) + http.client。

覆盖：health / ingest / recall / api stats / api memories(过滤分页) /
单条删除 / 批量维护双闸门（无 token 403、错票 401、对票 200）/
传输硬性属性（HTTP/1.1+Content-Length、411/415/400、500 JSON 兜底）/
Host 与 Origin 白名单 / 旧 MCP 工具面语义迁移（去重、kind 校验、
空查询、stats、forget——MCP 面已删，契约并入对应端点用例）。
server 模块在导入时按环境建库，本模块导入前指向临时库。
"""
import http.client
import os
import subprocess
import tempfile
import threading
from types import SimpleNamespace

import pytest

from conftest import raw_request

os.environ["MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="rest-"), "rest.db")
os.environ.pop("MEMORY_TOKEN", None)

from semantic_memory import server


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "db_bytes" in body


def test_ingest_and_recall_roundtrip(client):
    r = client.post("/ingest", json={
        "content": "REST 路由级测试：混合检索往返验证",
        "kind": "fact", "source": "test", "project": "rest-proj",
    })
    assert r.status_code == 200 and r.json()["uid"]
    r = client.post("/recall", json={"query": "往返验证", "limit": 5, "project": "rest-proj"})
    hits = r.json()["hits"]
    assert hits and "往返验证" in hits[0]["content"]


def test_ingest_dedup_via_rest(client):
    """旧 MCP memory_store 去重契约：重复内容刷新既有条目而非新增。"""
    body = {"content": "重复写入的 REST 条目", "kind": "fact", "source": "test"}
    u1 = client.post("/ingest", json=body).json()["uid"]
    r2 = client.post("/ingest", json=body).json()
    assert r2["uid"] == u1 and r2["deduped"] is True


def test_ingest_path_project_stored_normalized(client, monkeypatch):
    """/ingest 传工作区目录路径 → 库里存的是 git origin 归一化键。"""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="git@github.com:acme/restrepo.git\n"))
    r = client.post("/ingest", json={
        "content": "归一化键存取验证 QX9norm", "kind": "fact", "source": "test",
        "project": "/home/me/restrepo"})
    assert r.status_code == 200
    # 检索/管理面按归一化键命中——路径字符串已不再是库内键
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=128, stdout=""))
    rows = client.get("/api/memories",
                      params={"project": "github.com/acme/restrepo"}).json()["memories"]
    assert any(m["content"] == "归一化键存取验证 QX9norm" for m in rows)


def test_api_stats_and_memories_filter(client):
    s = client.get("/api/stats").json()
    assert s["total"] >= 2
    assert s["by_kind"].get("fact", 0) >= 1  # 旧 MCP memory_stats 契约：按 kind 分布
    r = client.get("/api/memories", params={"project": "rest-proj", "limit": 10})
    assert all(m["project"] == "rest-proj" for m in r.json()["memories"])
    r = client.get("/api/memories", params={"kind": "decision"})
    assert all(m["kind"] == "decision" for m in r.json()["memories"])


def test_api_memories_pagination(client):
    for i in range(5):
        client.post("/ingest", json={"content": f"分页条目 {i}", "kind": "fact",
                                     "source": "test", "project": "page-proj"})
    page1 = client.get("/api/memories",
                       params={"project": "page-proj", "limit": 3, "offset": 0}).json()["memories"]
    page2 = client.get("/api/memories",
                       params={"project": "page-proj", "limit": 3, "offset": 3}).json()["memories"]
    assert len(page1) == 3 and len(page2) == 2
    assert {m["uid"] for m in page1}.isdisjoint({m["uid"] for m in page2})


def test_api_forget_single(client):
    """旧 MCP memory_forget/memory_list 契约：单条软删除 + 列表同步消失。"""
    uid = client.post("/ingest", json={"content": "将被删除的唯一条目 qx7delete",
                                       "kind": "fact", "source": "test"}).json()["uid"]
    assert client.delete(f"/api/memories/{uid}").json()["forgotten"] is True
    assert client.delete(f"/api/memories/{uid}").json()["forgotten"] is False
    # 遗忘后召回为空（FTS 触发器同步删除；查询词对本库唯一）
    assert client.post("/recall", json={"query": "qx7delete"}).json()["hits"] == []


def test_web_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "semantic-memory" in r.text


# ---------- 鉴权与双闸门（运行时切换 cfg.token） ----------

@pytest.fixture()
def with_token():
    object.__setattr__(server.cfg, "token", "secret")
    yield
    object.__setattr__(server.cfg, "token", None)


def test_bulk_guard_403_without_token(client):
    r = client.post("/api/maintenance/purge", json={})
    assert r.status_code == 403 and "MEMORY_TOKEN" in r.json()["error"]


def test_bulk_guard_401_and_200_with_token(client, with_token):
    assert client.post("/api/maintenance/purge", json={}).status_code == 401
    assert client.post("/api/maintenance/purge", json={},
                       headers={"authorization": "Bearer bad"}).status_code == 401
    r = client.post("/api/maintenance/purge", json={},
                    headers={"authorization": "Bearer secret"})
    assert r.status_code == 200 and "total_removed" in r.json()


def test_read_routes_require_auth_when_token_set(client, with_token):
    assert client.get("/api/stats").status_code == 401
    assert client.get("/api/stats",
                      headers={"authorization": "Bearer secret"}).status_code == 200
    # 写入信道同样受护
    assert client.post("/ingest", json={"content": "x"}).status_code == 401
    assert client.post("/idle").status_code == 401


# ---------- 传输硬性属性（stdlib http.server 评审钉死项） ----------

def test_http11_content_length_and_keepalive(client):
    """HTTP/1.1 + 每响应带准确 Content-Length + 同连接可复用。"""
    conn = http.client.HTTPConnection(client.host, client.port, timeout=10)
    try:
        for _ in range(2):
            conn.request("GET", "/health")
            r = conn.getresponse()
            body = r.read()
            assert r.status == 200 and r.version == 11
            assert int(r.headers["Content-Length"]) == len(body)
            assert r.headers["Content-Type"] == "application/json"
    finally:
        conn.close()


def test_204_no_body_no_content_length(client):
    r = client.post("/idle", json={})
    assert r.status_code == 204
    assert r._body == b""
    assert "Content-Length" not in r.headers  # RFC 7230：204 禁带


def test_unknown_path_404_including_mcp(client):
    """MCP 面已删：/mcp 及任意未知路径一律 404 JSON。"""
    assert client.get("/mcp").status_code == 404
    assert client.post("/mcp", json={}).status_code == 404
    assert client.get("/nope").status_code == 404


def test_host_allowlist_403(client):
    """DNS rebinding 防护：非回环 Host → 403（GET /health 亦不豁免）。"""
    out = raw_request(client.port,
                      b"GET /health HTTP/1.1\r\nHost: evil.example\r\n"
                      b"Connection: close\r\n\r\n")
    assert b" 403 " in out.split(b"\r\n", 1)[0] and b"host not allowed" in out


def test_missing_content_length_411(client):
    out = raw_request(client.port,
                      b"POST /ingest HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
    assert b" 411 " in out.split(b"\r\n", 1)[0]


def test_chunked_body_411(client):
    out = raw_request(client.port,
                      b"POST /ingest HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                      b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
    assert b" 411 " in out.split(b"\r\n", 1)[0]


def test_text_plain_rejected_415(client):
    """no-cors 盲写封堵：非 application/json 的 POST 一律 415（未落库）。"""
    before = server.store.stats()["total"]
    r = client.post("/ingest", data=b'{"content": "smuggled"}',
                    headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    assert server.store.stats()["total"] == before


def test_bad_json_400(client):
    r = client.post("/ingest", data=b"not-json{",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_dispatcher_500_json_fallback(client, monkeypatch):
    def boom(body):
        raise RuntimeError("boom")
    monkeypatch.setattr(server, "rest_recall", boom)
    r = client.post("/recall", json={"query": "x"})
    assert r.status_code == 500 and r.json()["error"] == "internal server error"
    assert r.headers["Content-Type"] == "application/json"


# ---------- 写端点双重防 drive-by（Origin 白名单） ----------

def test_origin_evil_403(client):
    """浏览器跨源必带 Origin：非回环 Origin 的写请求 → 403（curl/插件不带→零摩擦）。"""
    r = client.post("/ingest", json={"content": "drive-by"},
                    headers={"Origin": "https://evil.com"})
    assert r.status_code == 403


def test_origin_loopback_allowed(client):
    """同源（Web UI 自身）写请求零摩擦：回环 Origin 放行。"""
    r = client.post("/ingest", json={"content": "同源写入验证 QXorg", "source": "test"},
                    headers={"Origin": f"http://127.0.0.1:{client.port}"})
    assert r.status_code == 200


# ---------- task10：project+global 两路召回 ----------

def _ingest(client, content, project=None):
    body = {"content": content, "kind": "fact", "source": "test"}
    if project:
        body["project"] = project
    return client.post("/ingest", json=body).json()["uid"]


def test_web_index_has_global_toggle(client):
    html = client.get("/").text
    assert 'id="global"' in html and "global_limit" in html


def test_global_merge_order(client):
    """global_limit=2：项目条目在前、全局补足其后，各占配额，is_global 如实标注。"""
    p1 = _ingest(client, "两路召回项目条目甲 QX10A", project="two-path")
    p2 = _ingest(client, "两路召回项目条目乙 QX10B", project="two-path")
    g1 = _ingest(client, "两路召回全局条目丙 QX10C")
    g2 = _ingest(client, "两路召回全局条目丁 QX10D")
    r = client.post("/recall", json={
        "query": "两路召回", "limit": 4, "project": "two-path", "global_limit": 2})
    hits = r.json()["hits"]
    assert set(h["uid"] for h in hits[:2]) == {p1, p2}
    assert set(h["uid"] for h in hits[2:]) == {g1, g2}
    assert len(hits) == 4
    assert [h["is_global"] for h in hits] == [False, False, True, True]
    # 全局条目不得挤占项目配额：limit=2 时结果全为项目条目
    r = client.post("/recall", json={
        "query": "两路召回", "limit": 2, "project": "two-path", "global_limit": 2})
    assert {h["uid"] for h in r.json()["hits"]} == {p1, p2}
    # 不传 global_limit（=0）：现状语义（BM25 通道本就不按 project 过滤）
    r0 = client.post("/recall", json={
        "query": "两路召回", "limit": 4, "project": "two-path"})
    assert {p1, p2} <= {h["uid"] for h in r0.json()["hits"]}
    # 旧 MCP 契约迁移：不传 project 时 global_limit>0 退化为全局检索（两路开关不改语义）
    r_all = client.post("/recall", json={
        "query": "两路召回", "limit": 4, "global_limit": 2})
    assert {p1, p2, g1, g2} <= {h["uid"] for h in r_all.json()["hits"]}


def test_global_fills_project_remainder(client):
    """项目命中不足 limit 时全局条目补足，但全局侧只出自己的 global_limit 配额。"""
    p = _ingest(client, "补足验证项目条目 QX10E", project="two-path")
    g1 = _ingest(client, "补足验证全局条目一 QX10F")
    g2 = _ingest(client, "补足验证全局条目二 QX10G")
    _ingest(client, "补足验证全局条目三 QX10H")
    hits = client.post("/recall", json={
        "query": "补足验证", "limit": 3, "project": "two-path",
        "global_limit": 2}).json()["hits"]
    assert [h["uid"] for h in hits[:1]] == [p]
    assert {h["uid"] for h in hits[1:]} == {g1, g2}
    assert len(hits) == 3


def test_global_zero_byte_identical(client):
    """global_limit=0 与改动前单路 hybrid_recall 输出完全一致（uid 序列 + 全字段）。"""
    old = server.hybrid_recall(
        server.store, server.embedder, query="两路召回", limit=4,
        project=server.resolve_project("two-path"),
        rrf_k=server.cfg.rrf_k, inject_boost=server.cfg.inject_boost,
        min_cosine=server.cfg.recall_min_cosine,
        max_cosine=server.cfg.recall_max_cosine,
        reranker=server.reranker,
        rerank_candidates=server.cfg.rerank_candidates,
    )
    assert client.post("/recall", json={
        "query": "两路召回", "limit": 4, "project": "two-path",
        "global_limit": 0}).json() == {"hits": old}
    assert client.post("/recall", json={
        "query": "两路召回", "limit": 4, "project": "two-path",
        "global_limit": 0}).json() == {"hits": old}


def test_recall_empty_query(client):
    """旧 MCP 契约迁移：空查询返回空集（不炸、不凑数）。"""
    assert client.post("/recall", json={"query": ""}).json()["hits"] == []


def test_store_rejects_invalid_kind(client):
    """旧 MCP 契约迁移：非法 kind 拒收——存储层 ValueError 仍在，端点面 500 兜底且不落库。"""
    with pytest.raises(ValueError):
        server.store.store(content="bad kind", kind="nonsense")
    before = server.store.stats()["total"]
    r = client.post("/ingest", json={"content": "bad kind entry", "kind": "nonsense", "source": "test"})
    assert r.status_code == 500 and r.json()["error"] == "internal server error"
    assert server.store.stats()["total"] == before


# ---------- task5：蒸馏缓冲路由 + 批量阈值 + /idle ----------

def _pending_count():
    return server.store.conn.execute(
        "SELECT COUNT(*) c FROM distill_queue WHERE status='pending'"
    ).fetchone()["c"]


def test_ingest_routing_queue_vs_direct(client):
    """路由断言：compaction/chat 入队不直存；tool/remember 直存不入队。"""
    mem_before = server.store.stats()["total"]
    r = client.post("/ingest", json={"content": "压缩入队验证 QX5q", "source": "compaction",
                                     "project": "queue-proj", "session_id": "s5"})
    assert r.status_code == 200 and r.json()["queued"] is True
    r = client.post("/ingest", json={"content": "聊天入队验证 QX5q", "source": "chat"})
    assert r.json()["queued"] is True
    r = client.post("/ingest", json={"content": "工具直存验证 QX5q", "source": "tool"})
    assert r.json()["uid"]
    r = client.post("/ingest", json={"content": "显式记住直存验证 QX5q", "source": "remember"})
    assert r.json()["uid"]
    rows = server.store.conn.execute(
        "SELECT source FROM distill_queue WHERE status='pending' AND content LIKE '%QX5q%'"
    ).fetchall()
    assert sorted(r["source"] for r in rows) == ["chat", "compaction"]
    assert server.store.stats()["total"] == mem_before + 2


def test_batch_threshold_triggers_flush(client, monkeypatch):
    """触发器一路：入队使 pending 达 MEMORY_DISTILL_BATCH → _spawn_flush 恰一次。"""
    pending0 = _pending_count()
    calls = []
    monkeypatch.setattr(server, "_spawn_flush", lambda: calls.append(1))
    object.__setattr__(server.cfg, "distill_batch", pending0 + 3)
    try:
        for i in range(2):
            client.post("/ingest", json={"content": f"批量触发测试 {i} QX5b",
                                         "source": "compaction", "project": "batch-proj"})
        assert calls == []
        client.post("/ingest", json={"content": "批量触发第三条 QX5b",
                                     "source": "compaction", "project": "batch-proj"})
        assert len(calls) == 1
    finally:
        object.__setattr__(server.cfg, "distill_batch", 20)


def test_idle_returns_204_and_flushes_async(client, monkeypatch):
    """触发器二路：/idle 先回 204，flush 在 daemon 线程执行。"""
    done = threading.Event()
    calls = []

    def fake_flush(*args, **kwargs):
        calls.append(1)
        done.set()
        return {"typed": 0, "stored_raw": 0}

    monkeypatch.setattr(server, "flush_distill_queue", fake_flush)
    r = client.post("/idle", json={})
    assert r.status_code == 204
    assert done.wait(timeout=5)
    assert len(calls) == 1


# ---------- task14：观测面 ----------

def test_recall_latency_recorded(client):
    r = client.post("/recall", json={"query": "观测面延迟验证"})
    assert r.status_code == 200
    assert len(server.recall_latencies) >= 1
    s = client.get("/api/stats").json()
    assert s["recall_p95_ms"] >= s["recall_p50_ms"] >= 0


def test_distill_queue_len_in_stats(client):
    before = client.get("/api/stats").json()["distill_queue_len"]
    r = client.post("/ingest", json={"content": "观测面队列验证 QX14", "source": "chat"})
    assert r.json()["queued"] is True
    after = client.get("/api/stats").json()["distill_queue_len"]
    assert after == before + 1


# ---------- 零依赖基座：install_hint 门控 ----------

def test_install_hint_gating(client, monkeypatch):
    """无 numpy 时 /health 与 /api/stats 出现 install_hint；有 numpy 时绝不出现。"""
    monkeypatch.setattr(server, "HAS_NUMPY", False)
    assert "install_hint" in client.get("/health").json()
    assert "install_hint" in client.get("/api/stats").json()
    monkeypatch.setattr(server, "HAS_NUMPY", True)
    assert "install_hint" not in client.get("/health").json()
    assert "install_hint" not in client.get("/api/stats").json()


# ---------- 跨项目记忆图谱（第四期） ----------

def test_recall_cross_project_co_injection_extra(client):
    """查 A 项目能带出 B 项目同源解法：共注入过的 B 条目以 cross_kind 标记追加。"""
    ua = client.post("/ingest", json={"content": "跨项目图谱种子条目 XG7", "kind": "fact",
                                "source": "test", "project": "cross-a"}).json()["uid"]
    ub = client.post("/ingest", json={"content": "跨项目同源解法条目 XG8", "kind": "fact",
                                "source": "test", "project": "cross-b"}).json()["uid"]
    client.post("/inject/report", json={"harness": "test", "hook": "context",
                                       "query": "xg", "uids": [ua, ub], "chars": 64,
                                       "project": "cross-a"})
    hits = client.post("/recall", json={"query": "图谱种子条目", "limit": 3,
                                       "project": "cross-a"}).json()["hits"]
    cross = [h for h in hits if h.get("cross_kind")]
    assert any(h["uid"] == ub and h["cross_kind"] == "co_injection"
               and h["via_uid"] == ua for h in cross)
    # 主结果在前、附加条目在后，不占用主配额
    assert [h for h in hits if not h.get("cross_kind")][0]["uid"] == ua


def test_api_graph_endpoint(client):
    """GET /api/graph?uid= 可遍历三种边（观测面，管理面含被取代前代）。"""
    r = client.get("/api/graph", params={"uid": "nonexistent"})
    assert r.status_code == 200 and r.json()["edges"] == []
    # 取代链：旧→新，从新条目可查到前代
    old = client.post("/ingest", json={"content": "图谱旧版条目 XG9", "source": "test", "project": "cross-a"}).json()["uid"]
    new = client.post("/ingest", json={"content": "图谱新版条目 XG10", "source": "test", "project": "cross-a"}).json()["uid"]
    with server.store.conn:
        server.store.conn.execute(
            "UPDATE memories SET superseded_by=?, superseded_at=? WHERE uid=?",
            (new, 1234.5, old))
    edges = client.get("/api/graph", params={"uid": new}).json()["edges"]
    assert any(e["kind"] == "supersedes" and e["to_uid"] == old
               and e["superseded"] is True for e in edges)
