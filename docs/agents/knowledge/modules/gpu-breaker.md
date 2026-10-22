# GPU 熔断器与蒸馏失败语义（gpu-breaker）

> Last updated: 2026-10-22 · 核心件 `src/semantic_memory/gpu_breaker.py`（进程内全局态）、`distiller.py`、`lifecycle.py`；观测面＝`/health` 与 `/recall` 的 `gpu_degraded` 字段（停注入判定走 /recall），`/api/stats` 的 `distill_queue_len` 看积压。

## 架构要点

- 熔断器语义：首个硬件类失败 `trip(reason)`（幂等，首因定案）→ embedder/reranker/distiller 三路 fail-open（BM25 降级）＋注入停止＋面板显示熔断态；**进程内状态，重启即清**
- trip 原因字串区分类别（诊断第一入口）：`embed|rerank|distill CUDA error`（GPU 推理炸）、`distill returned None`（硬件类加载失败，仅 distill 路径）、`distill exception`（异常路径，仅 distill）

## Decisions

### 解析抖动与硬件失败分流（2026-10-22）
- **Id:** gpu-breaker/parse-vs-hardware
- **Source:** session ses_e9ee7f9f2ffeE2BozQxVU86o（2026-10-22）
- **Decision:** 蒸馏输出无法解析（无 JSON 数组）→ `DistillParseError`，flush 捕获后该组 fail-closed 留 pending、**不 trip**；只有硬件类失败（模型加载失败返 None / GPU 推理炸）才熔断
- **Reason:** 2026-10-21 一次解析抖动 trip 全局熔断器，停注入 16h 直到手动重启；`_parse` 失败与 CUDA 损坏共用「返 None」通道，lifecycle 无法区分。诊断受阻 16h 的直接原因是 spawn 端 `stdio:"ignore"` 丢弃 stderr——同日修复：`_setup_logging()` 强制落盘 `<data_dir>/logs/server.log`（`MEMORY_LOG_FILE` 可覆盖），对任何 spawner 生效
- **Consequences:** 解析抖动组永久 pending 重试（无退避上限）——与 fail-closed 可观测设计一致；`/api/stats` 的 `distill_queue_len` 监控积压

## 陷阱（Bug Experiences）

### 评测会话直灌真实库（2026-10-21/22）
- **Id:** gpu-breaker/eval-noise-capture
- **Symptom:** `/tmp/rl-e2e-s*`（review-loop e2e 的 tmux 会话）向真实服务捕获 22 条垃圾条目，评测高峰恰是熔断误触发时段
- **Root cause:** `review-loop-e2e.sh` spawn `opencode --standalone` 未关 `MEMORY_PLUGIN_AUTOSTART`，评测会话插件连到共享服务（8765）
- **Fix:** 派生会话 env 加 `MEMORY_PLUGIN_AUTOSTART=0`（dotfiles 仓，经 agora 委派）
