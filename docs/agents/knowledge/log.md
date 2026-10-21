# 知识库日志（log）

> 追加式；每次 ingest 一节。格式：`## [{date}] ingest | {module-name}`

## [2026-10-20] ingest | opencode-plugin
- 建库首录：opencode 插件静默加载失败三层根因（file:// 配置不支持／@opencode/plugin 依赖无处解析／失败无日志）与修复安装路径（plugins/ 目录＋config 目录依赖）；附存在性探针与 autostart 语义

## [2026-10-21] ingest | tui-panel
- 状态断言类 UI 的门控语义：探活≠数据可得——「本会话无相关记忆注入」断言必须以 /api/injections 拉取成功为前提（rowsOk 信号），仅凭 /health 探活（online）会在 401/瞬时失败/首挂载竞态下渲染假断言；失败时沉默（隐藏零态）优于虚假断言（gt-6888 评审 R1 两处 medium 的根因）
- 面板三态互斥：熔断态（degraded）／有注入态（latest）／零态（!degraded && online && rowsOk && !latest）；拉取失败时 rows/degraded 保持旧态（静默重试语义）
- 已知缺口（记录未修）：MEMORY_TOKEN 配置时 TUI fetch 不带 Authorization → /api/injections 401 → rowsOk 恒 false 面板空白（含注入态不可见），是否带凭证拉取待产品决策
