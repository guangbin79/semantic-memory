# 知识索引（knowledge-index）

> semantic-memory 项目知识库目录（Layer 1）。详细模块页在 `./docs/agents/knowledge/modules/`。

## 模块列表

| 模块 | 一句话描述 | 详细路径 |
|------|-----------|----------|
| opencode-plugin | opencode 集成插件：安装正道（plugins/ 目录）、静默加载失败的根因/修复/存在性探针 | `./docs/agents/knowledge/modules/opencode-plugin.md` |
| gpu-breaker | CUDA 熔断器与蒸馏失败语义：解析抖动不熔断、trip 原因字串诊断、日志强制落盘 | `./docs/agents/knowledge/modules/gpu-breaker.md` |
| verify-env | src/ 改动验证语义：pytest 走 pythonpath=src，venv 裸 -m 走 site-packages 拷贝——CLI 试跑须 PYTHONPATH=src | `./docs/agents/knowledge/modules/verify-env.md` |
| hybrid-recall | hybrid_recall 门限语义：min_cosine 地板作用于全部命中，BM25 词面证据只影响排序不再豁免；无向量条目在地板开启时不可经 BM25 召回（backfill 待做） | `./docs/agents/knowledge/modules/hybrid-recall.md` |
