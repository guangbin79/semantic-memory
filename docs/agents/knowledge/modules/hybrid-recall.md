# hybrid_recall 门限语义（hybrid-recall）

> 相关性地板与通道豁免：RRF 只编码名次不编码相关强度，融合后统一过余弦地板。

## 地板作用域（2026-10-28 gt-bdffea）
- min_cosine 地板作用于**全部命中**：BM25 词面证据只影响排序、不再豁免门限——词面过、语义不过的条目出局（历史多次"词面过、语义不过"噪声注入的根治）。
- shortcut: 无向量条目（嵌入缺失）在地板开启时不可经 BM25 召回——`cos_map.get(u, 0.0)` 必低于正门限；修复路径＝embedding backfill，未做。
- 实测标定：相关 ~0.68，无关 ~0.28-0.31，门限默认 0.45。

## 实证
- 2026-10-28 gt-bdffea：门限块 `if min_cosine > 0 and cos_map: fused = [... if cos_map.get(u, 0.0) >= min_cosine]`（hybrid.py）；`tests/test_ranking.py::test_min_cosine_gates_all_hits` 断言 BM25 命中（cos 0.3 < 0.45）出局。
