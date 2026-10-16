"""混合检索：BM25 候选 + 向量候选，RRF 融合 + 后处理管线。与 ctx-semantic 同架构，但自包含。"""
from __future__ import annotations

import math

import numpy as np

from .db import MemoryStore
from .embedder import Embedder
from .reranker import Reranker

VALID_SCOPES = ("all", "project", "global")


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
    inject_boost: float = 0.0,
    min_cosine: float = 0.0,
    max_cosine: float = 0.92,
    reranker: Reranker | None = None,
    rerank_candidates: int = 20,
    scope: str = "all",
) -> list[dict]:
    """主召回入口。向量通道可用时双路融合，否则纯 BM25。

    后处理（按执行序）：
    - max_cosine（融合前）：冗余上限——纯向量命中 cos 超过上限出局（过近 =
      冗余/近重复）；BM25 命中不受影响。1.0 = 关闭，默认 0.92。
    - min_cosine：RRF 只编码名次不编码相关强度——"无相关里的最近邻"也会
      上榜。纯向量命中（无 BM25 token 证据）且余弦低于门限时丢弃。
      实测：相关 ~0.68，无关 ~0.28-0.31，门限默认 0.45。
    - inject_boost：遥测反哺——被真实注入过的记忆更可能再次相关，
      score × (1 + α·ln(1+注入次数))。对数抑制富者愈富，α 默认 0.15。
    - rerank：RRF 序 top rerank_candidates 交 Reranker 判别相关分；返回
      None（禁用/超预算/异常）→ 保 RRF 序 fail-open；重排后同分按
      updated_at desc 平决。
    - scope：'all'=现状；'global'=只查 project IS NULL（全局域）；
      'project'=排除全局域。
    """
    if scope not in VALID_SCOPES:
        raise ValueError(f"invalid scope {scope!r}, expected one of {VALID_SCOPES}")
    candidate_n = max(limit * 4, 20)

    # 通道 1：BM25
    bm25_hits = store.fts_search(query, candidate_n, scope=scope)
    bm25_ranked = [uid for uid, _ in bm25_hits]
    bm25_set = set(bm25_ranked)

    # 通道 2：向量余弦（暴力，~50k 条内亚毫秒；查询侧自动带 Instruct 前缀）
    vec_ranked: list[str] = []
    cos_map: dict[str, float] = {}
    q = embedder.embed_query(query)
    if q is not None:
        uids, mat = store.all_vectors(project=project, scope=scope)
        if len(uids):
            sims = mat @ q                           # 已归一化，点积即余弦
            top = np.argsort(-sims)[:candidate_n]
            vec_ranked = [uids[i] for i in top if sims[i] > 0.0]
            cos_map = {uids[i]: float(sims[i]) for i in top}

    # 冗余上限：纯向量命中过近出局（BM25 命中不受上限影响）
    if 0.0 < max_cosine < 1.0:
        vec_ranked = [u for u in vec_ranked if cos_map[u] <= max_cosine]

    # 候选级融合（多于 limit；有重排时窗口扩到 rerank_candidates）
    fuse_limit = max(limit * 2, 10)
    if reranker is not None and reranker.available:
        fuse_limit = max(fuse_limit, rerank_candidates)
    fused = rrf_fuse([bm25_ranked, vec_ranked], k=rrf_k, limit=fuse_limit)

    # 门限：纯向量弱相关出局（BM25 命中本身已是 token 级证据，放行）
    if min_cosine > 0 and cos_map:
        fused = [(u, s) for u, s in fused
                 if u in bm25_set or cos_map.get(u, 0.0) >= min_cosine]

    # 遥测反哺：注入次数弱先验（counts 顺带透出为 hit 的 injections 字段）
    inj_counts: dict[str, int] = {}
    if inject_boost > 0 and fused:
        inj_counts = store.injection_counts([u for u, _ in fused])
        if inj_counts:
            fused = sorted(
                ((u, s * (1.0 + inject_boost * math.log1p(inj_counts.get(u, 0))))
                 for u, s in fused),
                key=lambda kv: kv[1], reverse=True,
            )

    # 重排：RRF 序 top-N 交 Reranker；None/失败 → 保 RRF 序 fail-open。
    # 重排后同分按 updated_at desc 平决；候选窗口外的条目按 RRF 序殿后。
    if reranker is not None and fused:
        cands = fused[:rerank_candidates]
        cand_rows = store.get_many([u for u, _ in cands])
        contents = {r["uid"]: r["content"] for r in cand_rows}
        updated_at = {r["uid"]: r["updated_at"] for r in cand_rows}
        ranked = reranker.rerank(
            query, [(u, contents[u]) for u, _ in cands if u in contents]
        )
        if ranked:
            head = sorted(
                (kv for kv in ranked if kv[0] in contents),
                key=lambda kv: (-kv[1], -updated_at.get(kv[0], 0.0)),
            )
            head_uids = {u for u, _ in head}
            fused = head + [(u, s) for u, s in fused if u not in head_uids]

    fused = fused[:limit]
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
            "updated_at": r["updated_at"],
            "injections": inj_counts.get(r["uid"], 0),
            "is_global": r["project"] is None,
            "score": round(score_map[r["uid"]], 6),
        }
        for r in rows
    ]
