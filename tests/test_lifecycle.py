"""生命周期维护测试：去重、purge、TTL、backfill、既有库迁移。"""
import time

import numpy as np
import pytest

from memory_mcp.db import MemoryStore
from memory_mcp.embedder import Embedder
from memory_mcp.lifecycle import backfill_vectors, ingest_with_dedup, run_maintenance


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture()
def no_embedder():
    return Embedder(None)


class FakeEmbedder:
    """恒等向量：所有文本嵌入相同 → 向量近似判重必命中。"""
    available = True
    device = "cpu"
    def embed_batch(self, texts):
        return np.stack([np.ones(8, dtype=np.float32) / np.sqrt(8) for _ in texts])
    def embed_query(self, text):
        return np.ones(8, dtype=np.float32) / np.sqrt(8)


def test_dedup_exact_hash(store, no_embedder):
    uid1, d1 = ingest_with_dedup(store, no_embedder, "EPSG:3857 为默认输出", "fact", "manual", None, None)
    uid2, d2 = ingest_with_dedup(store, no_embedder, "EPSG:3857 为默认输出", "fact", "manual", None, None)
    assert not d1 and d2 and uid1 == uid2
    assert store.stats()["total"] == 1


def test_dedup_refreshes_timestamp(store, no_embedder):
    uid, _ = ingest_with_dedup(store, no_embedder, "GDAL 私有版本路径 /opt/gdal-private", "fact", "manual", None, None)
    before = store.find_by_hash.__self__.conn.execute(
        "SELECT updated_at FROM memories WHERE uid=?", (uid,)).fetchone()[0]
    time.sleep(0.01)
    ingest_with_dedup(store, no_embedder, "GDAL 私有版本路径 /opt/gdal-private", "fact", "manual", None, None)
    after = store.find_by_hash.__self__.conn.execute(
        "SELECT updated_at FROM memories WHERE uid=?", (uid,)).fetchone()[0]
    assert after > before


def test_dedup_vector_similarity(store):
    uid1, _ = ingest_with_dedup(store, FakeEmbedder(), "原始表述：瓦片统一 EPSG:3857", "fact", "compaction", "p1", None)
    uid2, d2 = ingest_with_dedup(store, FakeEmbedder(), "措辞略变的同义内容", "fact", "compaction", "p1", None)
    assert d2 and uid1 == uid2
    assert store.stats()["total"] == 1


def test_purge_soft_deleted(store, no_embedder):
    uid, _ = ingest_with_dedup(store, no_embedder, "待删除的条目", "fact", "manual", None, None)
    store.forget(uid)
    report = run_maintenance(store, ttl_artifact_days=0)
    assert report["soft_deleted_purged"] == 1 and report["vacuumed"]
    assert store.stats()["total"] == 0


def test_ttl_purges_artifact_only(store, no_embedder):
    old = time.time() - 100 * 86400
    a_uid, _ = ingest_with_dedup(store, no_embedder, "过期的工具产物", "artifact", "tool", None, None)
    f_uid, _ = ingest_with_dedup(store, no_embedder, "永久保留的事实", "fact", "manual", None, None)
    store.conn.execute("UPDATE memories SET created_at=? WHERE uid=?", (old, a_uid))
    store.conn.execute("UPDATE memories SET created_at=? WHERE uid=?", (old, f_uid))
    store.conn.commit()
    report = run_maintenance(store, ttl_artifact_days=90)
    assert report["ttl_purged"] == 1
    assert store.stats()["total"] == 1 and store.stats()["by_kind"] == {"fact": 1}


def test_backfill_vectors(store, no_embedder):
    ingest_with_dedup(store, no_embedder, "无向量时期写入的记忆", "fact", "manual", None, None)
    assert store.stats()["vectorized"] == 0
    result = backfill_vectors(store, FakeEmbedder())
    assert result["vectorized"] == 1
    assert store.stats()["vectorized"] == 1


def test_backfill_without_model(store, no_embedder):
    result = backfill_vectors(store, no_embedder)
    assert result["vectorized"] == 0 and "error" in result


def test_migration_backfills_hash(tmp_path):
    """模拟旧库（无 content_hash 列）打开后自动迁移。"""
    db = tmp_path / "old.db"
    s = MemoryStore(db)
    # 拆掉列模拟旧 schema（先删依赖它的索引，否则 SQLite 拒绝 DROP COLUMN）
    s.conn.execute("DROP INDEX idx_memories_hash")
    s.conn.execute("ALTER TABLE memories DROP COLUMN content_hash")
    s.conn.execute("INSERT INTO memories(uid, content, kind, source, project, tags, created_at, updated_at)"
                   " VALUES ('legacy1', '旧库遗留内容', 'fact', 'manual', NULL, '[]', 1, 1)")
    s.conn.commit()
    s.close()
    s2 = MemoryStore(db)  # 重开触发迁移
    from memory_mcp.db import content_hash
    hit = s2.find_by_hash(content_hash("旧库遗留内容"))
    assert hit and hit["uid"] == "legacy1"
    s2.close()
