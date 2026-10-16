"""排序后处理：遥测反哺、向量门限、冗余上限、重排与 scope 过滤。"""
import numpy as np
import pytest

from memory_mcp.db import MemoryStore
from memory_mcp.embedder import Embedder
from memory_mcp.hybrid import hybrid_recall
import numpy as np
import pytest

from memory_mcp.db import MemoryStore
from memory_mcp.embedder import Embedder
from memory_mcp.hybrid import hybrid_recall


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture()
def no_embedder():
    return Embedder(None)  # 纯 BM25 模式


def test_inject_boost_reorders(store, no_embedder):
    """BM25 第二名被注入 2 次后，开启反哺应翻转到第一；关闭则不变。"""
    uid_a = store.store("共同词 甲独有", kind="fact")
    uid_b = store.store("共同词 乙独有词", kind="fact")
    query = "共同词 乙独有词"  # B 命中两个检索词，BM25 排第一

    base = hybrid_recall(store, no_embedder, query, limit=5, inject_boost=0.0)
    assert [h["uid"] for h in base][:2] == [uid_b, uid_a]

    for _ in range(2):
        store.record_injection("opencode", "context", query, [uid_a], chars=80)

    boosted = hybrid_recall(store, no_embedder, query, limit=5, inject_boost=0.15)
    assert boosted[0]["uid"] == uid_a  # ×(1+0.15·ln3)≈1.165 足以翻转相邻名次
    # 关闭反哺不受影响
    still = hybrid_recall(store, no_embedder, query, limit=5, inject_boost=0.0)
    assert still[0]["uid"] == uid_b


class _StubEmbedder:
    """固定查询向量 e0，便于精确控制余弦。"""

    available = True
    device = "cpu"

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed_batch(self, texts):
        return np.stack([np.eye(self.dim, dtype=np.float32)[0] for _ in texts])

    def embed_query(self, text):
        return np.eye(self.dim, dtype=np.float32)[0]


def test_min_cosine_gates_vector_only_hits(store):
    """纯向量命中（无 BM25 证据）低于门限出局；BM25 命中不受门限影响。"""
    dim = 8
    emb = _StubEmbedder(dim)
    near = np.eye(dim, dtype=np.float32)[0]                       # cos = 1.0
    far = np.zeros(dim, dtype=np.float32)
    far[0], far[1] = 0.3, float(np.sqrt(1 - 0.3**2))              # cos = 0.3

    uid_x = store.store("memory alpha", kind="fact", embedding=near)
    uid_y = store.store("memory beta", kind="fact", embedding=far)
    uid_z = store.store("zzznomatch keyword note", kind="fact", embedding=far)

    query = "zzznomatch"  # FTS 只命中 Z；X/Y 为纯向量命中

    gated = hybrid_recall(store, emb, query, limit=5, min_cosine=0.45, max_cosine=1.0)
    gated_uids = {h["uid"] for h in gated}
    assert uid_x in gated_uids          # cos 1.0 过门限
    assert uid_y not in gated_uids      # cos 0.3 < 0.45，纯向量出局
    assert uid_z in gated_uids          # BM25 命中是 token 级证据，放行

    open_ = hybrid_recall(store, emb, query, limit=5, min_cosine=0.0, max_cosine=1.0)
    open_uids = {h["uid"] for h in open_}
    assert {uid_x, uid_y, uid_z} <= open_uids  # 门限为 0 时全部保留


class _StubReranker:
    """可编程重排桩：返回预设 (uid, score) 序，记录调用入参。"""

    available = True

    def __init__(self, ranked: list[tuple[str, float]] | None):
        self.ranked = ranked
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def rerank(self, query, candidates):
        self.calls.append((query, list(candidates)))
        return self.ranked


def _pair_store(store):
    """两条 BM25 可分序的条目：query='共同词 乙独有词' 时 B 在前。"""
    uid_a = store.store("共同词 甲独有", kind="fact")
    uid_b = store.store("共同词 乙独有词", kind="fact")
    return uid_a, uid_b


def test_rerank_reorders_and_receives_contents(store, no_embedder):
    """重排序覆盖 RRF 序；候选以 (uid, content) 对传入。"""
    uid_a, uid_b = _pair_store(store)
    query = "共同词 乙独有词"

    base = hybrid_recall(store, no_embedder, query, limit=5)
    assert [h["uid"] for h in base][:2] == [uid_b, uid_a]

    rr = _StubReranker([(uid_a, 0.9), (uid_b, 0.4)])
    out = hybrid_recall(store, no_embedder, query, limit=5, reranker=rr)
    assert [h["uid"] for h in out] == [uid_a, uid_b]
    (got_query, got_cands), = rr.calls
    assert got_query == query
    assert got_cands == [(uid_b, "共同词 乙独有词"), (uid_a, "共同词 甲独有")]


def test_rerank_tie_breaks_by_updated_at(store, no_embedder):
    """重排同分时 updated_at 新者在前。"""
    uid_a, uid_b = _pair_store(store)
    store.conn.execute("UPDATE memories SET updated_at=1000 WHERE uid=?", (uid_a,))
    store.conn.execute("UPDATE memories SET updated_at=2000 WHERE uid=?", (uid_b,))

    rr = _StubReranker([(uid_a, 0.5), (uid_b, 0.5)])  # 分数相同
    out = hybrid_recall(store, no_embedder, "共同词 乙独有词", limit=5, reranker=rr)
    assert [h["uid"] for h in out][:2] == [uid_b, uid_a]


def test_rerank_fail_open(store, no_embedder):
    """rerank 返回 None（禁用/超预算/异常）→ 保 RRF 序；reranker=None 同。"""
    uid_a, uid_b = _pair_store(store)
    query = "共同词 乙独有词"

    rr = _StubReranker(None)
    failed = hybrid_recall(store, no_embedder, query, limit=5, reranker=rr)
    assert [h["uid"] for h in failed] == [uid_b, uid_a]
    assert [h["uid"] for h in hybrid_recall(store, no_embedder, query, limit=5)] == [
        uid_b, uid_a,
    ]  # reranker=None → RRF 序


def test_redundancy_ceiling_filters(store):
    """cos 超上限的纯向量命中出局；同 cos 的 BM25 命中保留；1.0 = 关闭。"""
    dim = 8
    emb = _StubEmbedder(dim)
    high = np.zeros(dim, dtype=np.float32)
    high[0] = 0.95
    high[1] = float(np.sqrt(1 - 0.95**2))                     # cos = 0.95

    uid_vec = store.store("unrelated tokens xyz", kind="fact", embedding=high)
    uid_bm25 = store.store("probe keyword note", kind="fact", embedding=high)
    query = "probe"  # FTS 只命中 uid_bm25

    capped = hybrid_recall(store, emb, query, limit=5, max_cosine=0.92)
    assert {h["uid"] for h in capped} == {uid_bm25}

    off = hybrid_recall(store, emb, query, limit=5, max_cosine=1.0)
    assert {h["uid"] for h in off} == {uid_vec, uid_bm25}


def test_superseded_uid_not_recalled(store, no_embedder):
    """superseded_by 非空的条目不出现在召回结果（SQL 层过滤，task1）。"""
    old = store.store("旧版结论 superseded-demo", kind="decision")
    new = store.store("新版结论 superseded-demo", kind="decision")
    store.conn.execute("UPDATE memories SET superseded_by=? WHERE uid=?", (new, old))

    uids = [h["uid"] for h in hybrid_recall(store, no_embedder, "superseded-demo", 10)]
    assert old not in uids
    assert new in uids


def test_scope_filters_bm25_channel(store, no_embedder):
    """scope='global' 只返回 project IS NULL；'project' 排除全局域；非法值报错。"""
    uid_g = store.store("scope-demo 全局知识", kind="fact", project=None)
    uid_p = store.store("scope-demo 项目知识", kind="fact", project="proj-a")
    query = "scope-demo"

    g = [h["uid"] for h in hybrid_recall(store, no_embedder, query, 5, scope="global")]
    assert g == [uid_g]
    p = [h["uid"] for h in hybrid_recall(store, no_embedder, query, 5, scope="project")]
    assert p == [uid_p]
    all_ = {h["uid"] for h in hybrid_recall(store, no_embedder, query, 5, scope="all")}
    assert all_ == {uid_g, uid_p}
    with pytest.raises(ValueError):
        hybrid_recall(store, no_embedder, query, 5, scope="bogus")


def test_scope_filters_vector_channel(store):
    """向量通道同样按 scope 过滤（all_vectors 下探）。"""
    dim = 8
    emb = _StubEmbedder(dim)
    vec = np.eye(dim, dtype=np.float32)[0]
    uid_g = store.store("全局条目 vscope", kind="fact", project=None, embedding=vec)
    uid_p = store.store("项目条目 vscope", kind="fact", project="proj-b", embedding=vec)
    query = "qqqzzz"  # 无 BM25 token 证据，命中全来自向量通道

    g = {h["uid"] for h in
         hybrid_recall(store, emb, query, 5, max_cosine=1.0, scope="global")}
    assert g == {uid_g}
    p = {h["uid"] for h in
         hybrid_recall(store, emb, query, 5, max_cosine=1.0, scope="project")}
    assert p == {uid_p}


def test_hit_payload_fields(store, no_embedder):
    """hit dict 含 updated_at / injections / is_global（task11 头块排序依赖）。"""
    uid_g = store.store("payload 全局条目", kind="fact", project=None)
    uid_p = store.store("payload 项目条目", kind="fact", project="proj-x")
    query = "payload"

    hits = {h["uid"]: h for h in hybrid_recall(store, no_embedder, query, 5)}
    assert hits[uid_g]["updated_at"] == store.get_one(uid_g)["updated_at"]
    assert hits[uid_g]["is_global"] is True
    assert hits[uid_p]["is_global"] is False
    assert hits[uid_g]["injections"] == 0  # 反哺未启用 → 0

    for _ in range(3):
        store.record_injection("opencode", "context", query, [uid_g], chars=80)
    boosted = {h["uid"]: h for h in
               hybrid_recall(store, no_embedder, query, 5, inject_boost=0.15)}
    assert boosted[uid_g]["injections"] == 3
