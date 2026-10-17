"""账上问题全清（硬化批次，2026-10-17）。

① BM25 项目等值过滤——项目域召回不再漏他项目条目（与向量腿对齐）；
② /inject/report uids 截断——超长列表不再打爆 SQLite 变量上限（500）；
③ 畸形参数 400 化——query 缺失/limit 非法/project 非字符串返回 400 而非 500；
④ project.py 缓存封顶（实现侧 lru，行为由现有测试面覆盖）。
"""
import pytest

from semantic_memory.db import MemoryStore
from semantic_memory.embedder import Embedder
@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "hard.db")


# ---------------- ① BM25 项目等值过滤 ----------------

def test_fts_search_project_equality(store):
    a1 = store.store("过滤器验证条目 ftspf 甲", project="pa")
    b1 = store.store("过滤器验证条目 ftspf 乙同词", project="pb")
    # 项目域：只回本项目条目（此前他项目含查询 token 会漏入主命中）
    assert [u for u, _ in store.fts_search("ftspf", 10, project="pa")] == [a1]
    assert [u for u, _ in store.fts_search("ftspf", 10, project="pb")] == [b1]
    # 无 project：全局面不变（全库检索契约）
    assert {u for u, _ in store.fts_search("ftspf", 10)} == {a1, b1}


def test_hybrid_project_recall_excludes_other_projects(store):
    """项目域召回：他项目条目即使 token 命中也不再作主命中。"""
    from semantic_memory.hybrid import hybrid_recall

    store.store("共享 token 条目 hpf yes", project="pa")
    store.store("共享 token 条目 hpf 同词", project="pb")
    hits = hybrid_recall(store, embedder=Embedder(None), query="hpf", limit=5, project="pa")
    assert hits and all(h["project"] == "pa" for h in hits)


# ---------------- ② inject_report uids 截断 ----------------

def test_inject_report_oversized_uids_trimmed(client):
    """6 万 uid 的上报：200 而非 500（截断至 50，与 record_injection 同语义）。"""
    uids = [f"{i:016x}" for i in range(60_000)]
    r = client.post("/inject/report", json={
        "harness": "t", "hook": "context", "query": "q", "uids": uids, "chars": 1})
    assert r.status_code == 200 and r.json().get("recorded") is True


# ---------------- ③ 畸形参数 400 化 ----------------

def test_recall_missing_query_400(client):
    r = client.post("/recall", json={"limit": 3})
    assert r.status_code == 400 and "error" in r.json()


def test_recall_nonstring_limit_400(client):
    r = client.post("/recall", json={"query": "x", "limit": "abc"})
    assert r.status_code == 400 and "error" in r.json()


def test_api_memories_bad_limit_400(client):
    r = client.get("/api/memories", params={"limit": "abc"})
    assert r.status_code == 400 and "error" in r.json()

def test_ingest_nonstring_project_400(client):
    r = client.post("/ingest", json={"content": "x", "project": {"a": 1}})
    assert r.status_code == 400 and "error" in r.json()


def test_recall_nonstring_project_400(client):
    r = client.post("/recall", json={"query": "x", "project": ["p"]})
    assert r.status_code == 400 and "error" in r.json()


def test_negative_limits_400(client):
    """负数 limit：/recall 防切片语义静默错果，GET 防绕 200 封顶（SQLite 负 LIMIT＝无界）。"""
    assert client.post("/recall", json={"query": "x", "limit": -1}).status_code == 400
    assert client.get("/api/memories", params={"limit": -1}).status_code == 400
    assert client.get("/api/injections", params={"limit": -5}).status_code == 400


def test_ingest_missing_content_400(client):
    r = client.post("/ingest", json={"kind": "fact"})
    assert r.status_code == 400 and "error" in r.json()


def test_ingest_bad_or_null_source(client):
    """不可哈希 source → 400 非 500；显式 null → 回落 chat 正常落库。"""
    r = client.post("/ingest", json={"content": "硬化验证 nullsrc", "source": ["chat"]})
    assert r.status_code == 400 and "error" in r.json()
    r2 = client.post("/ingest", json={"content": "硬化验证 nullsrc 正常体", "source": None, "kind": "fact"})
    assert r2.status_code == 200 and r2.json().get("queued") is True  # null→chat 走入队路径
