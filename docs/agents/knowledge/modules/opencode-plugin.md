# opencode 集成插件（opencode-plugin）

> Last updated: 2026-10-20 · 插件本体 `integrations/opencode/memory.ts`，装于 `~/.config/opencode/plugins/`；服务为独立 Python 进程（127.0.0.1:8765），插件仅是它的无状态 HTTP 客户端。

## 架构要点

- 插件＝遥控器（钩子＋工具注册），服务＝记忆库（模型＋SQLite）；OpenCode 重启重载插件但不带走服务（spawn 时 detached+unref）
- 插件随 serve 进程加载：改插件代码/配置必须 `opencode service restart`，重启 TUI 无效
- autostart（`MEMORY_PLUGIN_AUTOSTART=1`）＝插件发现 /health 不通时顺手 spawn 服务；不设则服务死了永远静默躺平

## Decisions

### 插件安装走 plugins/ 目录＋config 目录依赖（2026-10-20）
- **Id:** opencode-plugin/install-path
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-20）
- **Decision:** 一律 `cp integrations/opencode/memory.ts ~/.config/opencode/plugins/`；`@opencode/plugin` 依赖装进 `~/.config/opencode/package.json`（`bun add @opencode/plugin@^2.0.28`，OpenCode 启动时自动 bun install 的官方机制）
- **Reason:** opencode.json 的 file:// 对象式 plugins 条目未列入官方文档（仅支持 npm 包名字符串），且该形态下依赖无处解析——实测从未加载成功。README 中该指引已证实为坏路径，待修

## Bug Experiences

### 插件从未加载→服务死后无人拉起（2026-10-20）
- **Id:** opencode-plugin/silent-load-failure
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-20）
- **Symptom:** 10-16→10-19 密集开发期 `injections` 表 0 行、库内仅 setup 冒烟金丝雀；10-19 系统重启杀掉服务后，4 个 OpenCode 会话无一拉起，静默持续 1 天
- **Root cause:** 三层叠加——① file:// 对象式配置不被支持；② 本地插件顶层 `import { Plugin } from "@opencode/plugin"` 无处解析（node_modules 缺失＋config 目录无 package.json）；③ 加载失败静默无日志，autostart spawn 逻辑随之从未存在
- **Fix:** cp 至 plugins/ ＋ config 目录 bun add ＋ 删 file:// 条目 ＋ autostart 环境变量 ＋ service restart
- **Prevention:** 插件加载失败全程静默——存在性探针＝当前会话工具列表是否含 `memory_store`/`memory_recall`；npm test（mock harness）全绿不代表真机加载路径通；改动后三步验证：service restart → 看工具列表 → 服务死后看 injections 是否恢复增长

## Open Questions

- 系统重启后 `.zshrc` 的 `MEMORY_PLUGIN_AUTOSTART=1` 能否进 serve 进程 env 未实证——下次重启后 `curl -m2 http://127.0.0.1:8765/health` 验证 autostart 自愈
- README 双语"file:// 声明式安装"指引为坏路径，尚未修正
