# 知识库日志（log）

> 追加式；每次 ingest 一节。格式：`## [{date}] ingest | {module-name}`

## [2026-10-20] ingest | opencode-plugin
- 建库首录：opencode 插件静默加载失败三层根因（file:// 配置不支持／@opencode/plugin 依赖无处解析／失败无日志）与修复安装路径（plugins/ 目录＋config 目录依赖）；附存在性探针与 autostart 语义

## [2026-10-21] ingest | tui-panel
- 状态断言类 UI 的门控语义：探活≠数据可得——「本会话无相关记忆注入」断言必须以 /api/injections 拉取成功为前提（rowsOk 信号），仅凭 /health 探活（online）会在 401/瞬时失败/首挂载竞态下渲染假断言；失败时沉默（隐藏零态）优于虚假断言（gt-6888 评审 R1 两处 medium 的根因）
- 面板三态互斥：熔断态（degraded）／有注入态（latest）／零态（!degraded && online && rowsOk && !latest）；拉取失败时 rows/degraded 保持旧态（静默重试语义）
- 已知缺口（记录未修）：MEMORY_TOKEN 配置时 TUI fetch 不带 Authorization → /api/injections 401 → rowsOk 恒 false 面板空白（含注入态不可见），是否带凭证拉取待产品决策

## [2026-10-22] ingest | gpu-breaker
- 解析抖动与硬件失败分流：蒸馏输出无 JSON 数组 → DistillParseError（fail-closed 留 pending，不熔断）；硬件类失败（返 None / CUDA 炸）才 trip——一次解析抖动曾 trip 全局熔断器致停注入 16h（2026-10-21 20:43 起，rl-e2e 评测高峰时段）
- 服务日志强制落盘：autostart spawn `stdio:"ignore"` 丢弃 stderr 使熔断不可诊断——`_setup_logging()` 落盘 `<data_dir>/logs/server.log`（MEMORY_LOG_FILE 覆盖），trip 原因字串（distill CUDA error / distill returned None / distill exception）是诊断第一入口
- 评测噪音直灌真实库：review-loop-e2e.sh spawn 会话未关 MEMORY_PLUGIN_AUTOSTART，/tmp/rl-e2e-s* 向共享服务捕获 22 条垃圾——修复走 dotfiles 委派（env 前缀关闭 autostart）

## [2026-10-24] ingest | verify-env
- 新模块页 verify-env：pytest `pythonpath=["src"]` 只在 pytest 生效；venv 裸 `-m` 走 site-packages 安装拷贝（非 editable），CLI 验证须 `PYTHONPATH=src`（gt-61cf sweep 试跑实证：裸跑落 serve 分支＋端口占用误报）
- consistency sweep 语义要点（gt-61cf）：只报告不删除；相对指针仅 project=现存绝对路径目录才校验，否则 unverifiable 不计悬空；遍历为 WAL 读快照（显式 BEGIN..finally COMMIT，防 OFFSET 窗口并发跳页/重扫）；file:// 非 localhost authority→unverifiable。已知缺口（记录未修）：_URLSPACE 含 v1/v2＋尾斜杠丢弃会漏检 `v2/config.yaml` 类指针（评审 A1，收窄 stoplist 属产品决策）
