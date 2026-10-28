# hybrid_recall 门限语义（hybrid-recall）

> 相关性地板与通道豁免：RRF 只编码名次不编码相关强度，融合后统一过余弦地板。

## 地板作用域（2026-10-28 gt-bdffea）
- min_cosine 地板作用于**全部命中**：BM25 词面证据只影响排序、不再豁免门限——词面过、语义不过的条目出局（历史多次"词面过、语义不过"噪声注入的根治）。
- shortcut: 混合场景（cos_map 非空）下无向量条目必出局——`cos_map.get(u, 0.0)` 必低于正门限；修复路径＝embedding backfill，未做。
- 边界（fail-open）：cos_map 为空（查询向量缺失或嵌入失败 → 纯 BM25 回退）时门限块整体短路不生效，BM25 命中照常放行——与 reranker fail-open 同哲学的有意取舍，降级期"词面过、语义不过"噪声在此路径原样回归，无日志标记（已知缺口，记录未修）。
- 实测标定：相关 ~0.68，无关 ~0.28-0.31，门限默认 0.45。

## 实证
- 2026-10-28 gt-bdffea：门限块 `if min_cosine > 0 and cos_map: fused = [... if cos_map.get(u, 0.0) >= min_cosine]`（hybrid.py）；`tests/test_ranking.py::test_min_cosine_gates_all_hits` 断言 BM25 命中（cos 0.3 < 0.45）出局。
