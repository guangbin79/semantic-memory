"""v0.3 演进套件：向量缓存 / 计数列 / embed 复用 / 加载锁 / 惰性单例 / 评测飞轮。

覆盖验收标准 #2-#10（.omo/req-gate/ses_eba6b6b8*.md）：
- all_vectors 进程内缓存双轨失效（write_gen + data_version）
- injection_count 计数列（迁移回填 + record_injection 自增，热路径免扫遥测表）
- hybrid_recall query_vec 复用（双路召回 embed×1）
- Embedder._ensure_loaded 并发单载
- import semantic_memory.server 零副作用（迁移/开库延后到属性访问）
- golden CLI round-trip / judge 接线（fake reranker）/ eval --min-recall 门禁
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from semantic_memory import server
from semantic_memory.db import MemoryStore
from semantic_memory.hybrid import hybrid_recall

REPO = Path(__file__).resolve().parent.parent


# ---------------- 共用替身 ----------------

class CountingEmbedder:
    """向量通道替身：记录 embed_query 次数，返回固定单位向量。"""

    available = True

    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return np.ones(4, dtype=np.float32)

    def embed_batch(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def _legacy_db(tmp_path, with_telemetry=True):
    """构造 v0.2 形态旧库（缺扩展列 + 遥测表），返回 db 路径。"""
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript("""
CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'fact', source TEXT NOT NULL DEFAULT 'manual',
    project TEXT, tags TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL, updated_at REAL NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0, content_hash TEXT);
CREATE TABLE injections (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    harness TEXT NOT NULL DEFAULT '', hook TEXT NOT NULL DEFAULT '', query TEXT NOT NULL DEFAULT '',
    project TEXT, chars INTEGER NOT NULL DEFAULT 0, uids TEXT NOT NULL DEFAULT '[]');
INSERT INTO memories(uid, content, created_at, updated_at) VALUES ('u1','legacy alpha',1,1),('u2','legacy beta',1,1);
""")
    if with_telemetry:
        raw.executescript("""
INSERT INTO injections(ts, uids) VALUES (1,'["u1"]'),(2,'["u1","u2"]'),(3,'["ghost"]');
""")
    raw.commit()
    raw.close()
    return db


# ---------------- 向量缓存（验收 #2） ----------------

def test_all_vectors_cache_hit_and_invalidation(tmp_path):
    store = MemoryStore(_legacy_db(tmp_path, with_telemetry=False))
    store.insert_vector("u1", np.ones(4, dtype=np.float32))
    a = store.all_vectors()
    b = store.all_vectors()
    assert a[0] is b[0] and a[1] is b[1], "无写时必须命中缓存（同一对象）"
    # 他连接提交 → data_version 推进 → 失效
    other = sqlite3.connect(store.db_path)
    other.execute("INSERT INTO memories(uid, content, created_at, updated_at) VALUES ('u9','z',1,1)")
    other.commit()
    other.close()
    c = store.all_vectors()
    assert a[0] is not c[0], "他连接写必须失效缓存"
    # 本连接写 → write_gen 推进 → 失效
    store.insert_vector("u2", np.ones(4, dtype=np.float32))
    d = store.all_vectors()
    assert len(d[0]) == len(a[0]) + 1, "本连接写必须失效缓存"
    store.close()


# ---------------- 计数列（验收 #4） ----------------

def test_injection_count_backfill_and_increment(tmp_path):
    store = MemoryStore(_legacy_db(tmp_path))
    # 回填口径 == 旧 json_each 实时口径（ghost 不在 memories → 不出现）
    assert store.injection_counts(["u1", "u2"]) == {"u1": 2, "u2": 1}
    store.record_injection("t", "h", "q", ["u1"], 10)
    assert store.injection_counts(["u1"]) == {"u1": 3}, "record_injection 同事务自增"
    # 热路径不再扫遥测表：清空 injections 后计数仍在（列维护，非派生查询）
    with store.conn:
        store.conn.execute("DELETE FROM injections")
    assert store.injection_counts(["u1"]) == {"u1": 3}
    store.close()


def test_legacy_rows_survive_trigger_paths(tmp_path):
    """FTS rebuild 修复验收：legacy 行的 touch/forget/检索在迁移后必须安全。"""
    store = MemoryStore(_legacy_db(tmp_path, with_telemetry=False))
    store.touch("u1")
    assert store.forget("u2") is True
    assert "u1" in {u for u, _ in store.fts_search("legacy", 10)}, "legacy 行经 rebuild 可检索"
    assert store.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    store.close()


# ---------------- embed 复用（验收 #3） ----------------

def test_hybrid_query_vec_reuse():
    store = MemoryStore(":memory:")
    uid = store.store("hello world", embedding=np.ones(4, dtype=np.float32))
    emb = CountingEmbedder()
    q = emb.embed_query("hello")
    hybrid_recall(store, emb, "hello", 5, query_vec=q)
    hybrid_recall(store, emb, "hello", 5, scope="global", query_vec=q)
    assert emb.calls == 1, "传入 query_vec 的两路调用只应嵌入一次"
    # 不传时各自嵌入（既有语义）
    hybrid_recall(store, emb, "hello", 5)
    hybrid_recall(store, emb, "hello", 5)
    assert emb.calls == 3
    assert uid


# ---------------- 加载锁（验收 #5） ----------------

def test_embedder_concurrent_load_once(tmp_path, monkeypatch):
    from semantic_memory import embedder as emb_mod

    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"x" * 16)
    loads = []
    lock = threading.Lock()

    def fake_new_model(self, use_gpu):
        with lock:
            loads.append(use_gpu)
        time.sleep(0.05)  # 放大竞争窗口
        class M:
            def embed(self, texts):
                return [list(np.ones(1024, dtype=np.float32)) for _ in texts]
        return M()

    monkeypatch.setattr(emb_mod.Embedder, "_new_model", fake_new_model)
    e = emb_mod.Embedder(gguf, device="cpu")
    assert e.available
    threads = [threading.Thread(target=e._ensure_loaded) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(loads) == 1, f"并发首嵌必须只加载一次（实际 {len(loads)}）"


# ---------------- import 零副作用（验收 #6） ----------------

def test_import_server_no_side_effects(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(REPO / "src"),
        "HOME": str(home),
        "MEMORY_DATA_DIR": "", "XDG_DATA_HOME": "",
        "MEMORY_DB_PATH": str(tmp_path / "iso.db"),
    })
    proc = subprocess.run([sys.executable, "-c", "import semantic_memory.server"],
                          env=env, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert not (home / "semantic-memory").exists(), "import 不得触发迁移/建目录"
    proc = subprocess.run([sys.executable, "-c", "import semantic_memory.server as s; s.cfg"],
                          env=env, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "iso.db").exists(), "首次属性访问才物化（开库落盘）"


# ---------------- 评测飞轮（验收 #8/#9/#10） ----------------

@pytest.fixture
def flywheel(tmp_path, monkeypatch):
    """独立库 + 替换单例：golden/judge/min-recall 不污染共享库。"""
    store = MemoryStore(tmp_path / "fly.db")
    u1 = store.store("deploy the service with semantic-memory", kind="fact", project="p1")
    u2 = store.store("banana bread recipe with walnuts", kind="fact", project="p1")
    store.record_injection("opencode", "generate", "how to deploy", [u1], 120, project="p1")
    store.record_injection("opencode", "generate", "banana bread", [u2], 90, project="p1")
    store.record_injection("opencode", "generate", "how to deploy", [u1], 120, project="p1")
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "embedder", CountingEmbedder())
    yield store, u1, u2
    store.close()


def test_golden_roundtrip(flywheel, tmp_path, capsys, monkeypatch):
    store, u1, u2 = flywheel
    out = tmp_path / "g.jsonl"
    assert server._cmd_golden(["--out", str(out)]) == 0
    rows = server._load_golden(out)
    assert len(rows) == 2, "按 (project, query) 去重后两条"
    by_q = {r["query"]: r for r in rows}
    assert by_q["how to deploy"]["expect_uids"] == [u1]
    assert by_q["banana bread"]["expect_uids"] == [u2]
    assert by_q["how to deploy"]["project"] == "p1"
    # golden 可被 eval 直接消费（round-trip 闭环）——命中路径走 BM25
    rc = server._cmd_eval(["--golden", str(out), "--k", "1,5"])
    assert rc == 0
    assert "recall@1=1.000" in capsys.readouterr().out


def test_min_recall_gate(flywheel, tmp_path, capsys):
    store, u1, _ = flywheel
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({"query": "deploy", "expect_uids": [u1]}) + "\n",
                      encoding="utf-8")
    assert server._cmd_eval(["--golden", str(golden), "--min-recall", "1:0.5"]) == 0
    assert server._cmd_eval(["--golden", str(golden), "--min-recall", "1:1.5"]) == 1
    assert server._cmd_eval(["--golden", str(golden), "--min-recall", "bad"]) == 2
    assert server._cmd_eval(["--golden", str(golden), "--k", "3", "--min-recall", "5:0.9"]) == 2


class FakeReranker:
    available = True

    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, candidates):
        return sorted(((uid, self._scores.get(uid, 0.1)) for uid, _ in candidates),
                      key=lambda kv: kv[1], reverse=True)


def test_judge_verdicts(flywheel, capsys, monkeypatch):
    store, u1, u2 = flywheel
    # deploy 期望高分（真 miss）；banana 期望低分（golden 标签可疑）
    monkeypatch.setattr(server, "reranker", FakeReranker({u1: 0.9, u2: 0.05}))
    server._judge_misses(
        [{"query": "deploy"}, {"query": "banana bread"}],
        [{"query": "deploy", "expect_uids": [u1]},
         {"query": "banana bread", "expect_uids": [u2]}])
    out = capsys.readouterr().out
    assert "检索漏召回" in out and "golden 标签可疑" in out


def test_judge_skips_without_reranker(flywheel, capsys):
    class Off:
        available = False
    server.reranker, saved = Off(), server.reranker
    try:
        server._judge_misses([{"query": "x"}], [{"query": "x", "expect_uids": ["y"]}])
        assert "重排模型不可用" in capsys.readouterr().out
    finally:
        server.reranker = saved
