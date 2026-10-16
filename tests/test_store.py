"""BM25 降级路径（无嵌入模型）的核心链路测试。"""
import numpy as np
import pytest

from semantic_memory.db import MemoryStore
from semantic_memory.embedder import Embedder
from semantic_memory.hybrid import hybrid_recall, rrf_fuse


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture()
def no_embedder():
    return Embedder(None)  # 纯 BM25 模式


def test_store_and_fts_recall(store, no_embedder):
    store.store("GDAL 私有编译版本位于 /opt/gdal-private，含私有格式驱动", kind="fact")
    store.store("PostGIS 数据库使用只读账号连接", kind="decision")
    store.store("unrelated note about cooking pasta", kind="fact")

    hits = hybrid_recall(store, no_embedder, "GDAL 驱动", limit=5)
    assert hits and "GDAL" in hits[0]["content"]


def test_forget_excludes_from_recall(store, no_embedder):
    uid = store.store("temporary secret token abc123", kind="fact")
    assert hybrid_recall(store, no_embedder, "secret token", limit=5)
    assert store.forget(uid) is True
    assert hybrid_recall(store, no_embedder, "secret token", limit=5) == []
    assert store.forget(uid) is False  # 重复删除


def test_vector_channel_rrf(store):
    """手工注入向量，验证向量通道参与 RRF 融合。"""
    dim = 8
    emb = np.zeros(dim, dtype=np.float32)
    emb[0] = 1.0
    uid = store.store("vectorized memory about rasters", kind="fact", embedding=emb)
    store.store("keyword only note", kind="fact")

    class FakeEmbedder:
        available = True
        device = "cpu"
        def embed_batch(self, texts):
            return np.stack([np.eye(dim, dtype=np.float32)[0] for _ in texts])
        def embed_query(self, text):
            return np.eye(dim, dtype=np.float32)[0]

    hits = hybrid_recall(store, FakeEmbedder(), "rasters", limit=5)
    assert hits and hits[0]["uid"] == uid


def test_rrf_math():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60, limit=3)
    # a: 1/61 + 1/62 ≈ 0.0325；c: 1/63 + 1/61 ≈ 0.0323；b: 1/62
    assert fused[0][0] == "a"
    assert fused[1][0] == "c"
    assert fused[2][0] == "b"


def test_stats_and_list(store):
    store.store("a fact", kind="fact", project="p1")
    store.store("a decision", kind="decision", project="p1")
    store.store("other project note", kind="fact", project="p2")
    s = store.stats()
    assert s["total"] == 3 and s["by_kind"]["fact"] == 2
    assert len(store.list_recent(project="p1")) == 2


def test_cjk_substring_recall(store, no_embedder):
    """trigram 通道：中文连续文本的中间子串（≥3 字符）可召回。"""
    uid = store.store("私有编译的GDAL包含内部格式驱动模块", kind="fact")
    store.store("完全不相关的英文条目 completely different", kind="fact")
    hits = hybrid_recall(store, no_embedder, "格式驱动", limit=5)  # 4 字中文子串
    assert hits and hits[0]["uid"] == uid


def test_query_instruct_prefix():
    """锁定与 ctx-semantic 一致的非对称嵌入约定：查询带 Instruct 前缀，文档不带。"""
    from semantic_memory.embedder import format_query
    q = format_query("GDAL 驱动")
    assert q.startswith("Instruct: Given a web search query")
    assert q.endswith("Query: GDAL 驱动")


def test_list_recent_kind_and_offset(store):
    for i in range(5):
        store.store(f"fact number {i}", kind="fact", project="p1")
    store.store("a preference", kind="preference", project="p1")
    facts = store.list_recent(kind="fact", limit=10)
    assert len(facts) == 5
    page1 = store.list_recent(project="p1", limit=4, offset=0)
    page2 = store.list_recent(project="p1", limit=4, offset=4)
    assert len(page1) == 4 and len(page2) == 2
    assert {r["uid"] for r in page1}.isdisjoint({r["uid"] for r in page2})


def test_fts_special_chars_no_crash(store, no_embedder):
    store.store("C++ Qt compile_commands.json 必须生成", kind="preference")
    # 特殊字符查询不应抛异常
    hits = hybrid_recall(store, no_embedder, 'C++ "Qt" (compile)', limit=5)
    assert isinstance(hits, list)

def test_superseded_filtered_in_recall_but_visible_in_admin(store, no_embedder):
    """被取代条目：召回面（fts/向量/get_many）不可见，管理面（list_page）仍可见。"""
    emb = np.eye(8, dtype=np.float32)[0]
    old = store.store("prefer pip for installing deps", kind="preference", embedding=emb)
    other = store.store("GDAL private build lives in /opt", kind="fact")
    new = store.store("unrelated pasta recipe", kind="fact")
    with store.conn:  # supersede 标记的落库属后续任务，此处直写 SQL 验证过滤语义
        store.conn.execute(
            "UPDATE memories SET superseded_by=? WHERE uid=?", (new, old)
        )

    assert old not in {u for u, _ in store.fts_search("pip", limit=10)}
    vec_uids, _ = store.all_vectors()
    assert old not in vec_uids
    assert [r["uid"] for r in store.get_many([old, other])] == [other]

    total, rows = store.list_page()
    assert total == 3
    assert old in {r["uid"] for r in rows}


def test_migrate_idempotent(tmp_path):
    """旧库（缺新列）跑两遍迁移：schema 收敛到同一形态，不报错不重复加列。"""
    import sqlite3

    db = tmp_path / "old.db"
    raw = sqlite3.connect(db)
    raw.executescript("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'fact',
            source TEXT NOT NULL DEFAULT 'manual',
            project TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            deleted INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT
        );
        INSERT INTO memories(uid, content, created_at, updated_at)
        VALUES ('legacy1', 'old memory', 1.0, 1.0);
    """)
    raw.commit()
    raw.close()

    def snapshot():
        c = sqlite3.connect(db)
        cols = {r[1] for r in c.execute("PRAGMA table_info(memories)")}
        has_queue = c.execute(
            "SELECT 1 FROM sqlite_master WHERE name='distill_queue'"
        ).fetchone()
        c.close()
        return cols, has_queue

    s1 = MemoryStore(db)  # 第一遍迁移
    s1.close()
    after_first = snapshot()

    s2 = MemoryStore(db)  # 第二遍：__init__ 再跑一次 _migrate，另显式补跑一遍
    s2._migrate()
    after_second = snapshot()
    s2.close()

    assert after_first == after_second
    expected = {"content_hash", "session_id", "last_recalled_at", "superseded_by", "superseded_at"}
    assert expected <= after_second[0]
    assert after_second[1]  # distill_queue 已建
