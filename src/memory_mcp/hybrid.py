"""混合检索：BM25 候选 + 向量候选，RRF 融合。与 ctx-semantic 同架构，但自包含。"""
from __future__ import annotations

import numpy as np

from .db import MemoryStore
from .embedder import Embedder


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)，rank 从 1 起。"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, uid in enumerate(ranked, start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def hybrid_recall(
    store: MemoryStore,
    embedder: Embedder,
    query: str,
    limit: int,
    project: str | None = None,
    rrf_k: int = 60,
) -> list[dict]:
    """主召回入口。向量通道可用时双路融合，否则纯 BM25。"""
    candidate_n = max(limit * 4, 20)

    # 通道 1：BM25
    bm25_hits = store.fts_search(query, candidate_n)
    bm25_ranked = [uid for uid, _ in bm25_hits]

    # 通道 2：向量余弦（暴力，~50k 条内亚毫秒；查询侧自动带 Instruct 前缀）
    vec_ranked: list[str] = []
    q = embedder.embed_query(query)
    if q is not None:
        uids, mat = store.all_vectors(project=project)
        if len(uids):
            sims = mat @ q                           # 已归一化，点积即余弦
            top = np.argsort(-sims)[:candidate_n]
            vec_ranked = [uids[i] for i in top if sims[i] > 0.0]

    fused = rrf_fuse([bm25_ranked, vec_ranked], k=rrf_k, limit=limit)
    rows = store.get_many([uid for uid, _ in fused])
    score_map = dict(fused)
    return [
        {
            "uid": r["uid"],
            "content": r["content"],
            "kind": r["kind"],
            "source": r["source"],
            "project": r["project"],
            "tags": r["tags"],
            "created_at": r["created_at"],
            "score": round(score_map[r["uid"]], 6),
        }
        for r in rows
    ]
