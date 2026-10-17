"""跨项目记忆图谱：取代链 / 共注入 / 同义跨域三种边，纯查询侧派生。

零写路径新状态、零新表——supersedes 边来自 memories.superseded_by（管理面前代），
co_injection 边来自 injections.uids 共现（权重=共注入次数），synonym 边来自全矩阵
暴力余弦的跨项目近邻（权重=余弦；无 numpy/无向量时该边按设计缺席）。
召回集成见 server._recall_hits：项目查询在主结果后追加跨项目附加条目（不占主配额），
可遍历观测面见 GET /api/graph（以 to_uid 为种子再查即链式遍历）。
"""
from __future__ import annotations

from ._compat import np
from .db import MemoryStore

EDGE_KIND_ORDER = ("supersedes", "co_injection", "synonym")
_LIVE_EDGE_KINDS = ("co_injection", "synonym")  # 可进注入面的边；前代已被取代，仅观测面
# 种子封顶（三轮通用）：/recall 的 limit 客户端可控且未设上限，不封顶则逐种子
# matvec+argsort 放大热路径成本（~50k 规模单种子≈4ms，16≈64ms 内）；
# ponytail: 超过 16 个种子的边信号边际价值≈0（extras 本就被 cross_limit 截断），
# 真需要更大种子面时改 argpartition 或调大此值
_MAX_SEEDS = 16


def graph_edges(store: MemoryStore, seed_uids: list[str], *,
                cross_sim: float = 0.90, per_kind_limit: int = 10) -> list[dict]:
    """种子集的 1-hop 边（kind 序 supersedes → co_injection → synonym，同序内权重降序）。

    边形状：{kind, from_uid(种子), to_uid(邻居), weight, to_project, to_row}。
    supersedes 的 to_row 为管理面行（被取代者仍可读内容/项目）；co_injection/synonym
    邻居仅取活面（get_many），死亡（被取代/已删除/无行）邻居直接落边。
    """
    seeds = list(dict.fromkeys(u for u in seed_uids if u))[:_MAX_SEEDS]
    if not seeds:
        return []

    edges: list[dict] = []
    # 取代链（后向）：被种子直接取代的前代
    for row in store.supersede_predecessors(seeds)[:per_kind_limit]:
        edges.append({"kind": "supersedes", "from_uid": row["superseded_by"],
                      "to_uid": row["uid"], "weight": 1.0,
                      "to_project": row["project"], "to_row": dict(row)})

    # 共注入：与种子同 injections 行共现过的其他 uid（via=字典序最小共现种子）
    for row in store.co_injection_counts(seeds, per_kind_limit):
        edges.append({"kind": "co_injection", "from_uid": row["via_uid"],
                      "to_uid": row["to_uid"], "weight": float(row["c"]),
                      "to_project": None, "to_row": None})

    # 同义跨域：全矩阵暴力余弦近邻（已归一化，点积即余弦；无 numpy → 缺席）
    if np is not None:
        uids, mat = store.all_vectors()
        idx = {u: i for i, u in enumerate(uids)}
        seed_set = set(seeds)
        for s in seeds:
            i = idx.get(s)
            if i is None:
                continue
            sims = mat @ mat[i]
            picked = 0
            for j in np.argsort(-sims):
                if picked >= per_kind_limit:
                    break
                u = uids[j]
                if u == s or u in seed_set:
                    continue
                if float(sims[j]) < cross_sim:
                    break  # 降序遍历，首个低于阈值即全余皆低
                edges.append({"kind": "synonym", "from_uid": s, "to_uid": u,
                              "weight": round(float(sims[j]), 6),
                              "to_project": None, "to_row": None})
                picked += 1

    # 活面富化：co/syn 邻居取行与项目（死邻居落边出局）
    pending = [e for e in edges if e["kind"] in _LIVE_EDGE_KINDS]
    rows = {r["uid"]: dict(r) for r in store.get_many([e["to_uid"] for e in pending])}
    live = []
    for e in pending:
        r = rows.get(e["to_uid"])
        if r is not None:
            e["to_project"] = r["project"]
            e["to_row"] = r
            live.append(e)

    order = {k: i for i, k in enumerate(EDGE_KIND_ORDER)}
    return sorted(
        [e for e in edges if e["kind"] == "supersedes"] + live,
        key=lambda e: (order[e["kind"]], -e["weight"]),
    )


def cross_project_extras(store: MemoryStore, hits: list[dict], *,
                         project: str, cross_sim: float = 0.90, limit: int = 2) -> list[dict]:
    """召回命中的跨项目附加条目——查 A 项目带出 B 项目同源解法。

    仅消费活边（synonym 优先于 co_injection——同源信号最强），严格排除当前项目
    与全局域（全局条目走 global_limit 两路召回，职责不重叠）；被取代/已删除邻居
    已在 graph_edges 落边。返回 hybrid hit 形状 + cross_kind/via_uid，供主配额外追加。
    """
    seeds = [h["uid"] for h in hits if isinstance(h, dict) and h.get("uid")]
    if not seeds or not project:
        return []
    edges = graph_edges(store, seeds, cross_sim=cross_sim,
                        per_kind_limit=max(limit * 3, 6))
    prio = {"synonym": 0, "co_injection": 1}
    cand = sorted(
        (e for e in edges
         if e["kind"] in prio and e["to_row"] is not None
         and e["to_row"]["project"] and e["to_row"]["project"] != project),
        key=lambda e: (prio[e["kind"]], -e["weight"]),
    )
    seen = set(seeds)
    picked: list[dict] = []
    for e in cand:
        if len(picked) >= limit:
            break
        if e["to_uid"] in seen:
            continue
        seen.add(e["to_uid"])
        r = e["to_row"]
        picked.append({
            "uid": r["uid"], "content": r["content"], "kind": r["kind"],
            "source": r["source"], "project": r["project"], "tags": r["tags"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "injections": 0, "is_global": False,
            "score": round(e["weight"], 6),
            "cross_kind": e["kind"], "via_uid": e["from_uid"],
        })
    if picked:
        counts = store.injection_counts([h["uid"] for h in picked])
        for h in picked:
            h["injections"] = counts.get(h["uid"], 0)
    return picked
