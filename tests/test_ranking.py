"""排序后处理：遥测反哺（inject_boost）与向量门限（min_cosine）。"""
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

    gated = hybrid_recall(store, emb, query, limit=5, min_cosine=0.45)
    gated_uids = {h["uid"] for h in gated}
    assert uid_x in gated_uids          # cos 1.0 过门限
    assert uid_y not in gated_uids      # cos 0.3 < 0.45，纯向量出局
    assert uid_z in gated_uids          # BM25 命中是 token 级证据，放行

    open_ = hybrid_recall(store, emb, query, limit=5, min_cosine=0.0)
    open_uids = {h["uid"] for h in open_}
    assert {uid_x, uid_y, uid_z} <= open_uids  # 门限为 0 时全部保留
