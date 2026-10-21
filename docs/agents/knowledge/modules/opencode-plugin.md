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
- **Supplements（2026-10-20 晚）:** TUI 侧边栏面板走**目录形态**：`ln -s …/integrations/opencode/tui-plugin ~/.config/opencode/plugins/semantic-memory-tui`——TUI 插件发现（packages/tui/src/plugin/discovery.ts）只认目录/软链，单文件永不被 TUI 加载；入口约定＝目录下 `tui.tsx`（Host.resolve 解析 ./tui），槽位＝`context.ui.slot({ append: "sidebar.content" })`，热重载生效无需 service restart

## 环境陷阱（Bug Experiences 补遗）

### 蒸馏生成路径 sampler 断言崩进程（2026-10-21，第 4 类）
- **Id:** opencode-plugin/distill-sampler-abort
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-21）；serve.log `llama-sampler.cpp:940: GGML_ASSERT(logits != nullptr)`
- **Symptom:** 服务第 4 次静默死亡；同日 09:53 flush 蒸馏返回 None（raw 直存 7 条）为同路径前兆
- **Root cause:** Distiller（1.7B 生成，ModelSlot 换载）generate 路径 sampler 断言——与 embedder 并发互斥修复正交（互斥只护 embedder）；换载后 CUDA 上下文状态疑似腐坏
- **Fix（临时止损）:** config.toml 注释 distill_model → Distiller(None) → flush 走 raw 直存兜底（捕获过滤器已上游挡噪音，raw 内容为真实用户消息）；蒸馏 typed 功能待架构审计后恢复
- **Prevention:** 架构任务（ModelSlot/CUDA 上下文管理）必须覆盖全部三条模型路径（embedder/reranker/distiller），逐路径修锁不 enough；蒸馏 fail-open（None→raw）与崩溃（abort）的边界要在架构层消除

### venv 非可编辑安装：测试绿而服务行为旧（2026-10-20）
- **Id:** opencode-plugin/venv-non-editable
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-20）
- **Symptom:** pytest 全绿（pythonpath=["src"] 跑新代码）但 `~/.local/bin/semantic-memory` 行为仍旧——API 新参数被静默忽略
- **Root cause:** venv 里 `pip install .` 非可编辑副本，console 入口跑 site-packages 旧代码；tests 与服务跑的是两份代码
- **Fix:** 改 src 后 `venv/bin/pip install .` 重装再重启服务；或开发期 `PYTHONPATH=src python3 -m semantic_memory`
- **Prevention:** 改 src 后凡经 console 入口验证，先重装；「测试绿」不能为「服务行为」作证——两边跑的代码可能不同源

## Bug Experiences

### 插件从未加载→服务死后无人拉起（2026-10-20）
- **Id:** opencode-plugin/silent-load-failure
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-20）
- **Symptom:** 10-16→10-19 密集开发期 `injections` 表 0 行、库内仅 setup 冒烟金丝雀；10-19 系统重启杀掉服务后，4 个 OpenCode 会话无一拉起，静默持续 1 天
- **Root cause:** 三层叠加——① file:// 对象式配置不被支持；② 本地插件顶层 `import { Plugin } from "@opencode/plugin"` 无处解析（node_modules 缺失＋config 目录无 package.json）；③ 加载失败静默无日志，autostart spawn 逻辑随之从未存在
- **Fix:** cp 至 plugins/ ＋ config 目录 bun add ＋ 删 file:// 条目 ＋ autostart 环境变量 ＋ service restart
- **Prevention:** 插件加载失败全程静默——存在性探针＝当前会话工具列表是否含 `memory_store`/`memory_recall`；npm test（mock harness）全绿不代表真机加载路径通；改动后三步验证：service restart → 看工具列表 → 服务死后看 injections 是否恢复增长

### 服务反复静默死亡：并发推理无互斥（GGML abort，2026-10-20）
- **Id:** opencode-plugin/ggml-concurrent-inference
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-20）；serve.log `GGML_ASSERT(out_ids==n_outputs)`
- **Symptom:** 服务数日内 3 次静默消失（19:38 起）；每次都发生在一次 recall 完成打点后瞬间
- **Root cause:** ThreadingHTTPServer 多请求线程并发调用同一非线程安全 `llama_cpp.Llama`（llama_decode 释放 GIL；Embedder 独居 ModelSlot 槽外，`_load_lock` 只护首载不护推理）→ C 层数据竞争 abort。**首修走了弯路：误判为超窗输入**——库 `embed()` 默认 `truncate=True` 截 n_batch=512 token，超长输入直调库存活；零共享终闸 R2 以并发复现（3 线程 exit 135）推翻
- **Fix:** `Embedder._infer_lock` 全程互斥（_ensure_loaded＋embed＋CPU 重建）；字符钳制 6000 保留为纵深防御
- **Prevention:** 涉及非线程安全 C 库的请求路径，首载锁≠推理锁；并发回归测试必须防「空转绿」——桩测试经 `__new__` 绕过 `__init__` 时锁属性缺失会让线程崩在锁前、桩从未执行、收集断言空转 PASS（R3 教训：须补锁属性＋进入计数 `len(calls)==N` 断言，并以摘锁红/装锁绿双向实证）

### 捕获过滤器 ^ 锚定失配（2026-10-21）
- **Id:** opencode-plugin/harness-noise-anchor
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-21）；R2 终闸 2/7 漏检实证
- **Symptom:** HARNESS_NOISE_RE 上线后 09:53 仍有 "Work finished…" 横幅入库（该项目 7 条代理提示词噪音漏 2）
- **Root cause:** 过滤器 `^` 锚定 trim 后首字符——横幅前常缀注入块/引号等前导内容，锚定必失配
- **Fix:** 去 `^` 改子串匹配（模式特异足够，误伤面＝讨论这些字面量的会话被保守跳过捕获，可接受）
- **Prevention:** 内容过滤器禁用行首锚定——注入文本的“位置”不可假设（前缀可以是任意上下文）

### injectTail 裸字符串 content 崩会话（2026-10-21）
- **Id:** opencode-plugin/injecttail-content-shape
- **Source:** session ses_ea6ef6d57ffewSnefsn5aJbOWb（2026-10-21）
- **Symptom:** 真实记忆入库后所有会话随机 "Session failed"（core A.content.map）；headless 评审通道全灭
- **Root cause:** `injectTail` 注入 `{role:"user", content: block}`——v2 消息契约 content 须为 part 数组；触发条件＝服务在线＋召回有命中（服务长期离线时 bug 休眠，探针假阴）
- **Fix:** content 改 `[{type:"text",text:block}]`；3 连跑 standalone＋真机会话实证
- **Prevention:** 钩子 mutate `event.messages/system` 必须走 part 数组形状；「测试绿」要先问「bug 触发条件在测试环境是否成立」——休眠型 bug 的绿灯是假灯；配套采集黑名单 `HARNESS_NOISE_RE` 挡 harness 横幅入库（六胞胎噪音实证）

## Open Questions

- README 双语"file:// 声明式安装"指引为坏路径，尚未修正（2026-10-20 晚：对象式实测失败；元组式未测，修正前需先实证）
- ~~系统重启后 `.zshrc` 的 `MEMORY_PLUGIN_AUTOSTART=1` 能否进 serve 进程 env~~（2026-10-20 晚已答：`opencode service restart` 自非交互 shell 跑时不可达；但用户正常流 zsh→TUI→serve 继承链成立——仅手动 service restart 后需手动拉起服务）
