"""生命周期维护测试：去重、purge、TTL、使用度留存、backfill、既有库迁移。"""
import time

import numpy as np
import pytest

from semantic_memory.db import MemoryStore
from semantic_memory.embedder import Embedder
from semantic_memory.distiller import Distiller
from semantic_memory.lifecycle import (
    backfill_vectors,
    enqueue_raw,
    flush_distill_queue,
    ingest_with_dedup,
    run_maintenance,
    stamp_recalled,
)


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
    from semantic_memory.db import content_hash
    hit = s2.find_by_hash(content_hash("旧库遗留内容"))
    assert hit and hit["uid"] == "legacy1"
    s2.close()


# ---------------- 使用度留存（task8） ----------------


def test_touch_and_stamp_refresh_last_recalled(store, no_embedder):
    uid, _ = ingest_with_dedup(store, no_embedder, "会被再次证实的内容", "fact", "manual", None, None)
    t0 = store.conn.execute(
        "SELECT last_recalled_at FROM memories WHERE uid=?", (uid,)).fetchone()[0]
    assert t0 is None
    time.sleep(0.01)
    store.touch(uid)
    t1 = store.conn.execute(
        "SELECT last_recalled_at FROM memories WHERE uid=?", (uid,)).fetchone()[0]
    assert t1 is not None
    time.sleep(0.01)
    assert stamp_recalled(store, [uid, "ghost-uid"]) == 1
    t2 = store.conn.execute(
        "SELECT last_recalled_at FROM memories WHERE uid=?", (uid,)).fetchone()[0]
    assert t2 > t1


def test_stale_fact_purged(store, no_embedder):
    stale_uid, _ = ingest_with_dedup(store, no_embedder, "两百天无人问津的事实", "fact", "manual", None, None)
    fresh_uid, _ = ingest_with_dedup(store, no_embedder, "昨天刚被召回的事实", "fact", "manual", None, None)
    store.conn.execute("UPDATE memories SET created_at=? WHERE uid=?",
                       (time.time() - 200 * 86400, stale_uid))
    store.conn.execute("UPDATE memories SET last_recalled_at=? WHERE uid=?",
                       (time.time() - 86400, fresh_uid))
    store.conn.commit()
    report = run_maintenance(store, ttl_artifact_days=0, ttl_stale_days=180)
    assert report["stale_purged"] == 1 and report["stale_by_kind"] == {"fact": 1}
    assert report["reclaim_bytes"] > 0
    assert store.get_one(stale_uid) is None
    assert store.get_one(fresh_uid) is not None


def test_preference_decision_exempt(store, no_embedder):
    p_uid, _ = ingest_with_dedup(store, no_embedder, "永远偏好中文回复", "preference", "manual", None, None)
    d_uid, _ = ingest_with_dedup(store, no_embedder, "决策：单库单文件存储", "decision", "manual", None, None)
    for uid in (p_uid, d_uid):
        store.conn.execute("UPDATE memories SET created_at=? WHERE uid=?",
                           (time.time() - 365 * 86400, uid))
    store.conn.commit()
    report = run_maintenance(store, ttl_artifact_days=0, ttl_stale_days=180)
    assert report["stale_purged"] == 0
    assert store.get_one(p_uid) is not None and store.get_one(d_uid) is not None


def test_superseded_fact_swept(store, no_embedder):
    sup_uid, _ = ingest_with_dedup(store, no_embedder, "旧表述：服务端口 8765", "fact", "manual", None, None)
    store.conn.execute("UPDATE memories SET created_at=?, superseded_by='newer' WHERE uid=?",
                       (time.time() - 3 * 86400, sup_uid))
    store.conn.commit()
    report = run_maintenance(store, ttl_artifact_days=0, ttl_stale_days=180)
    assert report["stale_purged"] == 1 and report["stale_by_kind"] == {"fact": 1}
    assert store.get_one(sup_uid) is None


def test_stale_disabled_when_zero(store, no_embedder):
    uid, _ = ingest_with_dedup(store, no_embedder, "超期但留存已关闭", "fact", "manual", None, None)
    store.conn.execute("UPDATE memories SET created_at=? WHERE uid=?",
                       (time.time() - 400 * 86400, uid))
    store.conn.commit()
    report = run_maintenance(store, ttl_artifact_days=0, ttl_stale_days=0)
    assert report["stale_purged"] == 0
    assert store.get_one(uid) is not None


def test_dry_run_deletes_nothing(store, no_embedder):
    stale_uid, _ = ingest_with_dedup(store, no_embedder, "干跑目标：过期事实", "fact", "manual", None, None)
    a_uid, _ = ingest_with_dedup(store, no_embedder, "干跑目标：过期产物", "artifact", "tool", None, None)
    del_uid, _ = ingest_with_dedup(store, no_embedder, "已软删除的条目", "fact", "manual", None, None)
    store.forget(del_uid)
    old = time.time() - 400 * 86400
    for uid in (stale_uid, a_uid):
        store.conn.execute("UPDATE memories SET created_at=? WHERE uid=?", (old, uid))
    store.conn.commit()
    before = store.stats()
    report = run_maintenance(store, ttl_artifact_days=90, ttl_injection_days=90,
                             ttl_stale_days=180, dry_run=True)
    assert report["dry_run"] is True
    assert report["stale_purged"] == 2
    assert report["stale_by_kind"] == {"fact": 1, "artifact": 1}
    assert report["ttl_purged"] == 1
    assert report["soft_deleted_purged"] == 1 and report["total_removed"] == 4
    assert report["reclaim_bytes"] > 0 and "vacuumed" not in report
    assert store.stats() == before


def test_cli_purge_dry_run(tmp_path):
    """验收原文：dry-run 零删除且 stdout 含计数（真实 CLI 入口）。"""
    import json
    import os
    import subprocess
    import sys
    db = tmp_path / "cli.db"
    s = MemoryStore(db)
    uid, _ = ingest_with_dedup(s, Embedder(None), "CLI 干跑：两百天未召回的事实", "fact", "manual", None, None)
    s.conn.execute("UPDATE memories SET created_at=? WHERE uid=?",
                   (time.time() - 200 * 86400, uid))
    s.conn.commit()
    s.close()
    env = {**os.environ, "MEMORY_DB_PATH": str(db), "MEMORY_MODEL_PATH": ""}
    proc = subprocess.run(
        [sys.executable, "-m", "semantic_memory.server", "purge", "--dry-run"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["dry_run"] is True and report["stale_purged"] == 1
    s2 = MemoryStore(db)
    assert s2.get_one(uid) is not None
    s2.close()
    s2 = MemoryStore(db)
    assert s2.get_one(uid) is not None
    s2.close()


# ---------------- 蒸馏缓冲与 flush（task5） ----------------


class FakeDistiller:
    """可编程假蒸馏器：available 可控、distill 返回预设 typed 条目并记录调用。"""

    def __init__(self, entries=None, available=True):
        self._entries = entries
        self.available = available
        self.calls = []

    def is_available(self):
        return self.available

    def distill(self, texts, project, neighbors=None):
        self.calls.append((list(texts), project, neighbors))
        return self._entries


def test_enqueue_returns_pending_count(store):
    assert enqueue_raw(store, "入队一", "chat", "p1", "s1") == 1
    assert enqueue_raw(store, "入队二", "compaction", "p1", "s1") == 2


def test_idle_flush_stores_typed_facts(store):
    """验收链路 enqueue→flush：typed 条目落库（session_id/scope 定档）且队列清空。"""
    enqueue_raw(store, "压缩丢弃的原文：偏好 uv", "compaction", "proj-a", "sess-1")
    enqueue_raw(store, "压缩丢弃的原文：团队用中文", "chat", "proj-a", "sess-1")
    fake = FakeDistiller(entries=[
        {"content": "用户偏好 uv 包管理器", "kind": "preference", "scope": "project",
         "rationale": "影响后续所有安装命令", "tags": ["uv"]},
        {"content": "团队交流用中文", "kind": "preference", "scope": "global",
         "rationale": "跨项目有效", "tags": []},
    ])
    report = flush_distill_queue(store, fake)
    assert report == {"typed": 2, "stored_raw": 0}
    rows = store.conn.execute(
        "SELECT content, kind, project, session_id, source FROM memories ORDER BY id"
    ).fetchall()
    assert [r["content"] for r in rows] == ["用户偏好 uv 包管理器", "团队交流用中文"]
    assert rows[0]["kind"] == "preference" and rows[0]["source"] == "compaction"
    assert rows[0]["project"] == "proj-a" and rows[0]["session_id"] == "sess-1"
    assert rows[1]["project"] is None  # scope=global → project 置 NULL（全局域）
    # 队列清空：无 pending，全部标 done
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM distill_queue WHERE status='pending'"
    ).fetchone()["c"] == 0
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM distill_queue WHERE status='done'"
    ).fetchone()["c"] == 2
    # 同 project 分组的两条原文一次蒸馏；project=None 时传空串
    assert fake.calls == [(["压缩丢弃的原文：偏好 uv", "压缩丢弃的原文：团队用中文"], "proj-a", None)]


def test_distiller_unavailable_falls_back_raw(store):
    """验收：Distiller 不可用（未配模型）→ 原文直存 stored_raw，现状行为不变差。"""
    enqueue_raw(store, "无模型时期的原始文本", "compaction", "proj-b", "sess-2")
    report = flush_distill_queue(store, Distiller(None))
    assert report == {"typed": 0, "stored_raw": 1}
    row = store.conn.execute(
        "SELECT content, kind, source, project, session_id FROM memories"
    ).fetchone()
    assert row["content"] == "无模型时期的原始文本" and row["kind"] == "fact"
    assert row["source"] == "compaction" and row["project"] == "proj-b"
    assert row["session_id"] == "sess-2"
    assert store.conn.execute(
        "SELECT status FROM distill_queue"
    ).fetchone()["status"] == "stored_raw"


def test_distill_failure_falls_back_raw(store):
    """蒸馏返回 None（解析失败 fail-open）→ 该组原文直存。"""
    enqueue_raw(store, "蒸馏失败的原始文本", "chat", "p", None)
    report = flush_distill_queue(store, FakeDistiller(entries=None))
    assert report == {"typed": 0, "stored_raw": 1}
    assert store.stats()["total"] == 1


def test_distill_exception_falls_back_raw(store):
    """蒸馏抛异常 → flush 不崩，该组原文直存。"""
    class BoomDistiller:
        available = True

        def is_available(self):
            return True

        def distill(self, texts, project, neighbors=None):
            raise RuntimeError("boom")

    enqueue_raw(store, "异常路径原文", "chat", "p", None)
    report = flush_distill_queue(store, BoomDistiller())
    assert report == {"typed": 0, "stored_raw": 1}


def test_flush_empty_queue(store):
    assert flush_distill_queue(store, None) == {"typed": 0, "stored_raw": 0}


def test_flush_commits_in_batches(store):
    """分批提交语义：> commit_batch 条也全部落库，队列清零。"""
    for i in range(7):
        enqueue_raw(store, f"批量原文 {i}", "chat", "p", None)
    report = flush_distill_queue(store, FakeDistiller(entries=None), commit_batch=3)
    assert report["stored_raw"] == 7
    assert store.stats()["total"] == 7
    assert store.conn.execute(
        "SELECT COUNT(*) c FROM distill_queue WHERE status='pending'"
    ).fetchone()["c"] == 0


def test_cli_distill_falls_back_raw(tmp_path):
    """触发器三路之 CLI：无模型环境手动 flush，原文直存并输出报告。"""
    import json
    import os
    import subprocess
    import sys
    db = tmp_path / "cli.db"
    s = MemoryStore(db)
    enqueue_raw(s, "CLI 蒸馏缓冲原文", "compaction", "proj-c", "sess-3")
    s.close()
    env = {**os.environ, "MEMORY_DB_PATH": str(db),
           "MEMORY_MODEL_PATH": "", "MEMORY_DISTILL_MODEL": ""}
    proc = subprocess.run(
        [sys.executable, "-m", "semantic_memory.server", "distill"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report == {"typed": 0, "stored_raw": 1}
    s2 = MemoryStore(db)
    assert s2.stats()["total"] == 1
    assert s2.conn.execute(
        "SELECT status FROM distill_queue"
    ).fetchone()["status"] == "stored_raw"
    s2.close()


# ---------------- 蒸馏管线行为（task6：三级行为 / supersede / scope 判重域） ----------------


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(8).astype(np.float32)
    return v / np.linalg.norm(v)


def _cos_to(base, seed, cos):
    """构造与 base 余弦恰为 cos 的单位向量（正交分量来自 seed）。"""
    ortho = _unit(seed) - base * float(_unit(seed) @ base)
    ortho /= np.linalg.norm(ortho)
    v = base * cos + ortho * np.sqrt(1 - cos * cos)
    return v.astype(np.float32) / np.linalg.norm(v)


class DictEmbedder:
    """查表向量器：测试对每条文本的余弦有精确控制。"""
    available = True
    device = "cpu"

    def __init__(self, table):
        self.table = table

    def embed_batch(self, texts):
        return np.stack([self.table[t] for t in texts])


def test_contradiction_supersedes_old(store):
    """验收：旧'偏好 pip' + 新'偏好 uv' → 旧条 superseded_by 非空且召回不见旧条。"""
    v_pip = _unit(1)
    v_uv = _cos_to(v_pip, 2, 0.7)  # 落在 0.60~0.98 模糊带
    old = store.store("偏好 pip 装依赖", "preference", "manual", "p", embedding=v_pip)
    emb = DictEmbedder({"用户改口：偏好 uv": v_uv, "偏好 uv": v_uv})
    enqueue_raw(store, "用户改口：偏好 uv", "chat", "p", None)
    fake = FakeDistiller(entries=[
        {"content": "偏好 uv", "kind": "preference", "scope": "project",
         "rationale": "新选择", "tags": [], "supersedes": [old]},
    ])
    report = flush_distill_queue(store, fake, embedder=emb)
    assert report == {"typed": 1, "stored_raw": 0}
    # 近邻带上下文随同一次蒸馏调用进入提示词（无二次 LLM）
    assert fake.calls == [(["用户改口：偏好 uv"], "p", [(old, "偏好 pip 装依赖")])]
    rows = store.conn.execute(
        "SELECT uid, superseded_by FROM memories ORDER BY id").fetchall()
    assert rows[0]["uid"] == old and rows[0]["superseded_by"] == rows[1]["uid"]
    # 召回路径（get_many）不见被取代的旧条
    assert [r["uid"] for r in store.get_many([old, rows[1]["uid"]])] == [rows[1]["uid"]]


def test_near_duplicate_touches(store):
    """验收：同域近重复（cos≥0.98）→ touch 既有，无新行。"""
    v = _unit(3)
    old = store.store("团队统一用 pnpm", "preference", "manual", "p", embedding=v)
    emb = DictEmbedder({"团队决定改用 pnpm 管理依赖": v, "团队统一用 pnpm 管理依赖": v})
    enqueue_raw(store, "团队决定改用 pnpm 管理依赖", "chat", "p", None)
    fake = FakeDistiller(entries=[
        {"content": "团队统一用 pnpm 管理依赖", "kind": "preference",
         "scope": "project", "rationale": "一致性", "tags": []},
    ])
    assert flush_distill_queue(store, fake, embedder=emb) == {"typed": 1, "stored_raw": 0}
    assert store.stats()["total"] == 1  # 无新行
    row = store.conn.execute(
        "SELECT last_recalled_at FROM memories WHERE uid=?", (old,)).fetchone()
    assert row[0] is not None  # touch 语义（同 store.touch：刷新召回打点）
    assert fake.calls[0][2] is None  # ≥0.98 不进模糊带，提示词无近邻


def test_global_entries_dedup_in_null_domain(store):
    """验收：两条近重复 global 偏好 → 后者 touch 而非双插；project 置 NULL。"""
    v1 = _unit(5)
    v2 = _cos_to(v1, 6, 0.99)  # ≥0.98 近重复
    emb = DictEmbedder({"raw-a": v1, "用户偏好深色主题": v1,
                        "raw-b": v2, "用户偏好深色主题配色": v2})
    enqueue_raw(store, "raw-a", "chat", "proj-a", None)
    f1 = FakeDistiller(entries=[
        {"content": "用户偏好深色主题", "kind": "preference",
         "scope": "global", "rationale": "跨项目", "tags": []},
    ])
    assert flush_distill_queue(store, f1, embedder=emb) == {"typed": 1, "stored_raw": 0}
    row = store.conn.execute("SELECT uid, project FROM memories").fetchone()
    assert row["project"] is None  # scope=global → project IS NULL（全局域）
    enqueue_raw(store, "raw-b", "chat", "proj-b", None)
    f2 = FakeDistiller(entries=[
        {"content": "用户偏好深色主题配色", "kind": "preference",
         "scope": "global", "rationale": "跨项目", "tags": []},
    ])
    assert flush_distill_queue(store, f2, embedder=emb) == {"typed": 1, "stored_raw": 0}
    assert store.stats()["total"] == 1  # 后者 touch，无双插
    assert f2.calls[0][2] is None  # 原文域 proj-b 无近邻——两域互不判重


def test_band_neighbor_without_supersedes_adds(store):
    """模糊带近邻 + 蒸馏未判取代 → ADD 双行，旧条保持可见。"""
    v_old = _unit(7)
    v_new = _cos_to(v_old, 8, 0.8)
    old = store.store("旧事实：服务端口 8765", "fact", "manual", "p", embedding=v_old)
    emb = DictEmbedder({"新原文": v_new, "新事实：服务端口 9000": v_new})
    enqueue_raw(store, "新原文", "chat", "p", None)
    fake = FakeDistiller(entries=[
        {"content": "新事实：服务端口 9000", "kind": "fact",
         "scope": "project", "rationale": "", "tags": []},
    ])
    assert flush_distill_queue(store, fake, embedder=emb) == {"typed": 1, "stored_raw": 0}
    assert store.stats()["total"] == 2  # ADD：不取代也不 touch
    assert fake.calls[0][2] == [(old, "旧事实：服务端口 8765")]
    assert store.get_many([old])[0]["uid"] == old  # 旧条未被取代


def test_no_embedder_degrades_to_hash_dedup(store):
    """降级：Embedder 未配 → 近邻级关闭，退化为仅哈希精确判重。"""
    enqueue_raw(store, "原文 A", "chat", "p", None)
    enqueue_raw(store, "原文 B", "chat", "p", None)
    dup_entry = {"content": "同一句结论", "kind": "fact", "scope": "project",
                 "rationale": "", "tags": []}
    fake = FakeDistiller(entries=[dup_entry, dict(dup_entry)])
    assert flush_distill_queue(store, fake, embedder=None) == {"typed": 2, "stored_raw": 0}
    assert store.stats()["total"] == 1  # 第二条哈希命中 → touch 不插入
    assert fake.calls[0][2] is None
