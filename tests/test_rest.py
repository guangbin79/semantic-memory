"""REST 侧车与 API 路由测试：真实 HTTP 打 FastMCP ASGI app。

覆盖：health / ingest / recall / api stats / api memories(过滤分页) /
单条删除 / 批量维护双闸门（无 token 403、错票 401、对票 200）。
server 模块在导入时按环境建库，本模块导入前指向临时库。
"""
import os
import tempfile

os.environ["MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="rest-"), "rest.db")
os.environ.pop("MEMORY_TOKEN", None)

import pytest
from starlette.testclient import TestClient

from memory_mcp import server


@pytest.fixture(scope="module")
def client():
    app = server.mcp.http_app()
    with TestClient(app) as c:
        yield c


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
    body = {"content": "重复写入的 REST 条目", "kind": "fact", "source": "test"}
    u1 = client.post("/ingest", json=body).json()["uid"]
    r2 = client.post("/ingest", json=body).json()
    assert r2["uid"] == u1 and r2["deduped"] is True


def test_api_stats_and_memories_filter(client):
    assert client.get("/api/stats").json()["total"] >= 2
    r = client.get("/api/memories", params={"project": "rest-proj", "limit": 10})
    assert all(m["project"] == "rest-proj" for m in r.json()["memories"])
    r = client.get("/api/memories", params={"kind": "decision"})
    assert all(m["kind"] == "decision" for m in r.json()["memories"])


def test_api_memories_pagination(client):
    for i in range(5):
        client.post("/ingest", json={"content": f"分页条目 {i}", "kind": "fact",
                                     "project": "page-proj"})
    page1 = client.get("/api/memories",
                       params={"project": "page-proj", "limit": 3, "offset": 0}).json()["memories"]
    page2 = client.get("/api/memories",
                       params={"project": "page-proj", "limit": 3, "offset": 3}).json()["memories"]
    assert len(page1) == 3 and len(page2) == 2
    assert {m["uid"] for m in page1}.isdisjoint({m["uid"] for m in page2})


def test_api_forget_single(client):
    uid = client.post("/ingest", json={"content": "将被删除的唯一条目 qx7delete",
                                       "kind": "fact"}).json()["uid"]
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
    r = client.post("/api/maintenance/purge")
    assert r.status_code == 403 and "MEMORY_TOKEN" in r.json()["error"]


def test_bulk_guard_401_and_200_with_token(client, with_token):
    assert client.post("/api/maintenance/purge").status_code == 401
    assert client.post("/api/maintenance/purge",
                       headers={"authorization": "Bearer bad"}).status_code == 401
    r = client.post("/api/maintenance/purge",
                    headers={"authorization": "Bearer secret"})
    assert r.status_code == 200 and "total_removed" in r.json()


def test_read_routes_require_auth_when_token_set(client, with_token):
    assert client.get("/api/stats").status_code == 401
    assert client.get("/api/stats",
                      headers={"authorization": "Bearer secret"}).status_code == 200
    # 写入信道同样受护
    assert client.post("/ingest", json={"content": "x"}).status_code == 401
