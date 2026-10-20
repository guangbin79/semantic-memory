# 知识库日志（log）

> 追加式；每次 ingest 一节。格式：`## [{date}] ingest | {module-name}`

## [2026-10-20] ingest | opencode-plugin
- 建库首录：opencode 插件静默加载失败三层根因（file:// 配置不支持／@opencode/plugin 依赖无处解析／失败无日志）与修复安装路径（plugins/ 目录＋config 目录依赖）；附存在性探针与 autostart 语义
