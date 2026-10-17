"""跨项目记忆图谱：三种边（supersedes / co_injection / synonym）与召回附加条目。

契约：
- graph_edges(store, seed_uids, *, cross_sim, per_kind_limit) → 边列表，每边
  {kind, from_uid(种子), to_uid(邻居), weight, to_project, to_row}；
  supersedes 边取管理面前代（被取代条目，召回面不可见）；co_injection 权重=共注入
  次数；synonym 权重=余弦（无 numpy/无向量时该边缺席，不崩）。
- cross_project_extras(store, hits, *, project, cross_sim, limit) → 召回命中集的
  跨项目附加条目（hit 形状 + cross_kind/via_uid），严格排除当前项目与全局域，
  synonym 优先于 co_injection，被取代/已删除邻居绝不进入。
"""
import time

import pytest

from semantic_memory.db import MemoryStore

try:
    import numpy as np
except ImportError:  # pragma: no cover - dev 依赖必含 numpy，此处兜底
    np = None

from semantic_memory import graph


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "graph.db")


def _supersede(st, old_uid, new_uid):
    with st.conn:
        st.conn.execute(
            "UPDATE memories SET superseded_by=?, superseded_at=? WHERE uid=?",
            (new_uid, time.time(), old_uid),
        )


def _unit(vs):
    v = np.asarray(vs, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------- supersedes 边 ----------------

def test_supersedes_edge_from_current_to_predecessor(store):
    old = store.store("旧版解法（将被取代）", project="pa")
    new = store.store("新版解法（取代旧版）", project="pa")
    _supersede(store, old, new)

    edges = graph.graph_edges(store, [new])
    se = [e for e in edges if e["kind"] == "supersedes"]
    assert [(e["from_uid"], e["to_uid"]) for e in se] == [(new, old)]
    assert se[0]["to_row"]["content"] == "旧版解法（将被取代）"

    # 链可遍历：以前代为种子再查一步，得到更旧的前代
    oldest = store.store("最旧版解法", project="pa")
    _supersede(store, oldest, old)
    chain = graph.graph_edges(store, [old])
    assert any(e["kind"] == "supersedes" and e["to_uid"] == oldest for e in chain)


# ---------------- co_injection 边 ----------------

def test_co_injection_edge_with_count_and_via(store):
    ua = store.store("A 项目条目", project="pa")
    ub = store.store("B 项目条目", project="pb")
    noise = store.store("噪声条目", project="pz")
    store.record_injection("t", "context", "q", [ua, ub], 10, project="pa")
    store.record_injection("t", "context", "q", [ua, ub], 10, project="pa")

    edges = graph.graph_edges(store, [ua])
    co = [e for e in edges if e["kind"] == "co_injection"]
    assert [(e["to_uid"], e["from_uid"], e["weight"]) for e in co] == [(ub, ua, 2.0)]
    assert all(e["to_uid"] != noise for e in co)


def test_co_injection_dead_neighbor_dropped(store):
    ua = store.store("A 项目条目", project="pa")
    ub = store.store("将被取代的 B 条目", project="pb")
    store.record_injection("t", "context", "q", [ua, ub], 10, project="pa")
    _supersede(store, ub, store.store("B 条目新版", project="pb"))

    edges = graph.graph_edges(store, [ua])
    assert all(e["kind"] != "co_injection" or e["to_uid"] != ub for e in edges)


# ---------------- synonym 边（向量近邻） ----------------

@pytest.mark.skipif(np is None, reason="numpy 缺席时 synonym 边按设计缺席")
def test_synonym_edge_cross_project_only_for_extras(store):
    ua = store.store("A 项目解法：分块重试", project="pa", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    ub = store.store("B 项目同源解法：分块重试变体", project="pb",
                     embedding=_unit([0.95, 0.3123, 0.0, 0.0]))  # cos≈0.95
    store.store("无关条目", project="pc", embedding=_unit([0.0, 1.0, 0.0, 0.0]))  # cos=0

    edges = graph.graph_edges(store, [ua], cross_sim=0.90)
    syn = [e for e in edges if e["kind"] == "synonym"]
    assert {e["to_uid"] for e in syn} == {ub}
    assert syn[0]["weight"] == pytest.approx(0.95, abs=1e-2)
    assert syn[0]["to_project"] == "pb"

    hits = [{"uid": ua, "content": "A 项目解法：分块重试"}]
    extras = graph.cross_project_extras(store, hits, project="pa", cross_sim=0.90, limit=2)
    assert [e["uid"] for e in extras] == [ub]
    e0 = extras[0]
    assert e0["cross_kind"] == "synonym" and e0["via_uid"] == ua
    assert e0["project"] == "pb" and e0["content"] == "B 项目同源解法：分块重试变体"


@pytest.mark.skipif(np is None, reason="numpy 缺席时 synonym 边按设计缺席")
def test_extras_limit_and_priority_synonym_first(store):
    ua = store.store("A 项目种子", project="pa", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    ub = store.store("B 同源", project="pb", embedding=_unit([0.95, 0.3123, 0.0, 0.0]))
    uc = store.store("C 共注入", project="pc")
    store.record_injection("t", "context", "q", [ua, uc], 10, project="pa")

    extras = graph.cross_project_extras(
        store, [{"uid": ua}], project="pa", cross_sim=0.90, limit=1)
    # limit=1 时 synonym（同源信号最强）优先于 co_injection
    assert len(extras) == 1 and extras[0]["cross_kind"] == "synonym"


@pytest.mark.skipif(np is None, reason="numpy 缺席时 synonym 边按设计缺席")
def test_extras_exclude_same_project_and_global(store):
    ua = store.store("A 项目种子", project="pa", embedding=_unit([1.0, 0.0, 0.0, 0.0]))
    same = store.store("同项目近邻", project="pa", embedding=_unit([0.95, 0.3123, 0.0, 0.0]))
    glob = store.store("全局近邻", project=None, embedding=_unit([0.96, 0.28, 0.0, 0.0]))

    extras = graph.cross_project_extras(
        store, [{"uid": ua}], project="pa", cross_sim=0.90, limit=5)
    assert all(e["uid"] not in (same, glob) for e in extras) and extras == []


# ---------------- 纯 BM25 降级 ----------------

def test_no_numpy_degrades_to_non_vector_edges(store, monkeypatch):
    ua = store.store("A 项目条目", project="pa")
    ub = store.store("B 项目条目", project="pb")
    store.record_injection("t", "context", "q", [ua, ub], 10, project="pa")
    monkeypatch.setattr(graph, "np", None)

    edges = graph.graph_edges(store, [ua], cross_sim=0.90)
    assert all(e["kind"] != "synonym" for e in edges)
    assert any(e["kind"] == "co_injection" and e["to_uid"] == ub for e in edges)


# ---------------- 配置三变量 ----------------

def test_config_cross_env_defaults_and_overrides(monkeypatch):
    from semantic_memory.config import Config

    c = Config.from_env()
    assert c.cross_project is True and c.cross_limit == 2 and c.cross_sim == pytest.approx(0.90)

    monkeypatch.setenv("MEMORY_CROSS_PROJECT", "0")
    monkeypatch.setenv("MEMORY_CROSS_LIMIT", "3")
    monkeypatch.setenv("MEMORY_CROSS_SIM", "0.85")
    c2 = Config.from_env()
    assert c2.cross_project is False and c2.cross_limit == 3 and c2.cross_sim == pytest.approx(0.85)
