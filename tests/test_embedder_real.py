"""真实模型端到端验证（需 MEMORY_MODEL_PATH 指向 GGUF；无模型自动跳过）。

CPU 模式即可运行（0.6B Q8，首轮加载约 10-20s）。覆盖：
维度/归一化/查询前缀约定、trigram 零重叠查询的向量兜底召回。
"""
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # 零依赖基座：无 numpy 时整模块跳过

MODEL = os.environ.get("MEMORY_MODEL_PATH", "")
pytestmark = pytest.mark.skipif(
    not MODEL or not Path(MODEL).exists(),
    reason="MEMORY_MODEL_PATH 未指向 GGUF 模型，跳过真实模型验证",
)


@pytest.fixture(scope="module")
def emb():
    from semantic_memory.embedder import Embedder
    e = Embedder(model_path=MODEL, device="cpu")
    assert e.available, "模型存在但加载失败"
    return e


def test_dim_norm_and_query_prefix(emb):
    v = emb.embed_batch(["部署分支固定为 release/2026"])
    assert v.shape == (1, 1024) and v.dtype == np.float32
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-4  # L2 归一化
    raw = emb.embed_batch(["发布用哪个分支"])[0]
    pref = emb.embed_query("发布用哪个分支")
    assert float(raw @ pref) < 0.95  # instruct 前缀生效（非对称嵌入）


def test_semantic_recall_backstops_bm25(tmp_path, emb):
    """trigram 零重叠的查询：BM25 漏召回，向量通道命中。"""
    from semantic_memory.db import MemoryStore
    from semantic_memory.embedder import Embedder
    from semantic_memory.hybrid import hybrid_recall
    from semantic_memory.lifecycle import ingest_with_dedup

    store = MemoryStore(tmp_path / "m.db")
    ingest_with_dedup(store, emb,
                      "部署分支固定为 release/2026，禁止直接从 main 发布",
                      "preference", "chat", None, [])
    ingest_with_dedup(store, emb, "投影矩阵计算已迁到 worker 线程",
                      "fact", "chat", None, [])

    no_embedder = Embedder(model_path=None)  # available=False → 纯 BM25
    bm25_hits = hybrid_recall(store, no_embedder, "发布用哪个分支", 3)
    assert all("release/2026" not in h["content"] for h in bm25_hits)

    hits = hybrid_recall(store, emb, "发布用哪个分支", 3)
    assert hits and "release/2026" in hits[0]["content"]
    store.close()
