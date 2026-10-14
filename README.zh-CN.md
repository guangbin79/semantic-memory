# semantic-memory-mcp

**[English](README.md) | [中文](README.zh-CN.md)**

coding agent 上下文管理的**侧车（sidecar）**：提供**模型语义级**的上下文检索，
确保上下文相关性不丢失。

**它的目标不是节省 token、不是压缩上下文。** opencode v2 原生 compaction 和 DCP
负责上下文"数量"（压缩、修剪、丢弃以适配窗口）；本组件负责上下文"相关性"——
被压缩掉的、被剪掉的、上一个会话里的一切，都进入语义可检索的仓库，当它与
**当前任务**相关时被自动找回注入。它花一笔受预算约束的 token（注入块），
买回本会永久丢失的相关性。压缩让窗口够用，侧车让压缩安全。

- MCP 服务（Python FastMCP）：BM25（FTS5 双通道 porter + trigram）+ 向量（Qwen3-Embedding-0.6B GGUF，可选）+ RRF 融合——检索发生在嵌入空间，措辞不同也能命中
- opencode v2 插件（`opencode-plugin/memory.ts`）：钩住 context / compaction / generate / chat.message / tool execute.after，让"记"和"忆"自动发生
- 独立进程、独立模型：不占用主 agent 的窗口与模型，服务宕机会话无感（fail-open）

## 架构

```
┌─ opencode v2 插件（memory.ts，零依赖，fail-open）────────┐
│  context/compaction/generate → /recall → 尾部注入         │
│  compaction 前 → /ingest（压缩丢弃 ≠ 记忆永存）            │
│  chat.message "记住…" → /ingest                          │
│  tool execute.after（白名单）→ /ingest                    │
└──────────────────────┬────────────────────────────────────┘
                       │ HTTP 127.0.0.1:8765（Bearer token 可选）
┌──────────────────────▼────────────────────────────────────┐
│  memory-mcp 服务（单进程双接口）                            │
│  MCP /mcp：memory_store / recall / forget / list / stats  │
│  REST 侧车：/recall /ingest /health（插件专用）            │
│  SQLite + FTS5(porter|trigram) + 向量BLOB，RRF(k=60)      │
└───────────────────────────────────────────────────────────┘
```

REST 侧车是为**多宿主支持**准备的：任何带生命周期钩子的 agent harness
（Claude Code、Codex、Kimi Code……）都可以把控制面接成"钩子事件 → curl
localhost:8765"——只有 opencode v2 需要进程内插件，因为它的钩子在 JS 运行时里。
各家适配见 `adapters/`——Claude Code 与 Kimi Code 为完整适配，Codex 为模板，ZCode/DeepSeek 仅 MCP 说明。

设计约束：插件零智能（检索/嵌入/存储全在服务端）；服务挂了插件静默跳过；注入集合未变不重复注入（保护 prefix cache）；MCP 脱离插件独立可用。
度量标准不是"省了多少 token"，而是"当前任务所需相关性的召回率"（recall@k）。

## 快速开始

```bash
# 1. 安装服务（建议 uv 或 venv）
pip install -e .                # 纯 BM25 模式即可用
pip install -e ".[embed]"       # 需要向量通道时（编译 llama-cpp-python）

# 2.（可选）下载嵌入模型，启用向量通道
bash scripts/download_model.sh  # 输出 MEMORY_MODEL_PATH 导出语句

# 3. 启动服务（HTTP 模式，MCP 与 REST 同端口）
memory-mcp                      # http://127.0.0.1:8765/mcp
memory-mcp --stdio              # stdio 模式（仅 MCP 工具，无 REST 侧车）

# 4. opencode v2：注册 MCP 服务器（~/.config/opencode/opencode.jsonc）
{
  "mcp": { "servers": {
    "memory": { "type": "remote", "url": "http://127.0.0.1:8765/mcp" }
  } }
}

# 5. opencode v2：安装控制面插件
cp opencode-plugin/memory.ts ~/.config/opencode/plugins/
# 插件目录的 package.json 需有 @opencode/plugin（仅类型）：
#   cd ~/.config/opencode && npm install @opencode/plugin@beta
```

## 配置（环境变量）

服务端：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_DB_PATH` | `~/.local/share/memory-mcp/memory.db` | 数据库路径 |
| `MEMORY_MODEL_PATH` | 空 | GGUF 模型路径；空 = 纯 BM25 降级 |
| `MEMORY_DEVICE` | `cuda` | 嵌入设备：`cuda`（默认，失败自动回退 CPU）/ `cpu` |
| `MEMORY_HOST` / `MEMORY_PORT` | `127.0.0.1` / `8765` | 绑定地址；端口 `0` = OS 随机分配 |
| `MEMORY_DISCOVERY_FILE` | `<db>.server.json` | 服务发现文件；`off` = 关闭 |
| `MEMORY_TOKEN` | 空 | REST 侧车 Bearer 鉴权；绑非回环地址时**必配** |
| `MEMORY_RRF_K` | `60` | RRF 融合常数 |
| `MEMORY_TTL_ARTIFACT_DAYS` | `90` | `artifact` 类 TTL 天数；`0` = 不过期 |
| `MEMORY_MAINTENANCE_ON_START` | `1` | 服务启动时运行维护例程 |
| `MEMORY_DEDUP_SIM` | `0.98` | 向量近似去重阈值（同 project 内） |

插件：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_PLUGIN_URL` | `http://127.0.0.1:8765` | 服务地址 |
| `MEMORY_PLUGIN_TOKEN` | 空 | 与服务端 `MEMORY_TOKEN` 对应 |
| `MEMORY_PLUGIN_LIMIT` | `5` | 每次召回条数 |
| `MEMORY_PLUGIN_BUDGET_CHARS` | `2000` | 注入块字符预算 |
| `MEMORY_PLUGIN_TOOL_CAPTURE` | 空 | 工具捕获白名单（逗号分隔），如 `gis,bash` |
| `MEMORY_PLUGIN_AUTOSTART` | `0` | `=1` 时服务未启动则自动 spawn |
| `MEMORY_PLUGIN_DEBUG` | `0` | `=1` 打印 ctx 形态与解析后配置 |

每个 `MEMORY_PLUGIN_*` 变量都有一个插件 options 孪生键（`url`、`token`、
`limit`、`budget_chars`、`tool_capture`、`autostart`、`debug`），因此也可以
直接在 `opencode.json` 的插件声明处配置——优先级：**环境变量 > 插件
options > 默认值**：

```jsonc
{
  "plugin": [
    ["file:///path/to/semantic-memory-mcp/opencode-plugin",
     { "url": "http://127.0.0.1:9000", "limit": 8, "tool_capture": ["bash", "gis"] }]
  ]
}
```

dev 运行时（`0.0.0-dev-*`）把元组第二元素透传给 `server(ctx, options)`
（已端到端实测）；stable 2.x 运行时则以 `ctx.options` 暴露同一张表。

## 随机端口与 agent 自举

agent 自己拉起服务时可以完全绕开端口占用问题——`MEMORY_PORT=0` 由 OS 分配
空闲端口，绑定完成的瞬间通过两条通道发布真实地址：

```bash
MEMORY_PORT=0 memory-mcp
# stdout（机器可读，单行）：
# MEMORY_MCP_LISTENING {"url":"http://127.0.0.1:41697","mcp_url":".../mcp","port":41697,"pid":...}
# 同一份 JSON 同时写入发现文件：<db>.server.json（可用 MEMORY_DISCOVERY_FILE 覆盖）
```

自举流程：spawn → 解析 stdout 行**或**读发现文件 → 把 `url` 填进
`MEMORY_PLUGIN_URL` / MCP 注册配置。优雅退出（SIGINT/SIGTERM）时撤下文件；
被 kill -9 时读取方可凭文件内 `pid` 判活。`stdio` 模式无端口，不发布任何信息。

## 索引：接近实时，无重建窗口

- **FTS（BM25）**：FTS5 外部内容表由触发器在**与写入相同的事务**内更新——
  `memory_store` 返回的那一刻即可被关键词检索。
- **向量**：嵌入在**写入请求内同步计算**并随行落库。检索是查询时的暴力余弦
  全扫——不存在需要构建、滞后或重建的 ANN 结构。
- **冷启动**：嵌入模型在首次嵌入调用时惰性加载（秒级，每进程一次）。
  该请求会同步等待，但不会丢数据。
- **捕获是至多一次**：钩子 fire-and-forget（1.5s 预算）。服务宕机期间到达的
  记忆会被静默丢弃——阻塞会话比漏一条记忆更糟。持久性是 best-effort，
  不是 exactly-once。
- **向量空洞**：嵌入器未配置期间写入的条目只有 BM25 可检索；装好模型后用
  `memory-mcp backfill` CLI 一次性追平（见"维护与生命周期"）。

## 存储布局

一切收在**一个 SQLite 文件**里（`MEMORY_DB_PATH`，默认
`~/.local/share/memory-mcp/memory.db`）：

| 对象 | 内容 |
|---|---|
| `memories` | 原文 + 元数据（kind / source / project / tags / 时间戳） |
| `memories_fts`、`memories_fts_tri` | FTS5 倒排索引（porter / trigram），外部内容表——正文不重复存储 |
| `memories_vec` | 向量 BLOB：每条 1024 维 float32 = 4KB |

运行期有 WAL 伴生文件（`-wal`、`-shm`）；GGUF 模型（~600MB）是独立文件。
删除 db 文件即干净的全量重置——没有其他状态。备份用
`sqlite3 memory.db ".backup '/path/to/backup.db'"`。与 ctx-semantic 不同，
向量矩阵每次查询从 SQLite 现读（page cache 可撑到 ~50k 条）——
没有任何缓存失效逻辑。

## 共享模型

- **跨会话**：索引在磁盘上，会话是进程。配合压缩前托管，连被丢弃的上下文
  也能活过会话边界。
- **跨 harness**：opencode v2、Claude Code、Kimi Code、Codex 都连同一个服务，
  即**同一份索引**——Claude Code 存的记忆在 opencode 里可召回。换 harness
  不丢记忆；没有第二份拷贝需要同步。
- **project 是过滤器，不是隔离墙**：索引全局唯一；插件注入时按当前项目过滤，
  但 `memory_recall` 不传 project 就是全库检索。
- **单用户单机**：回环绑定 + 可选 Bearer token，无多租户。
- **归因缺口**：`source` 记录来源类型（chat / compaction / tool），但不记录
  由哪个 harness 写入——出现该查询需求时加 `harness:<name>` 标签即可。

### 指定存储位置

`MEMORY_DB_PATH` 在服务启动时读取（父目录自动创建）。一个库文件对应一个
服务实例——需要库级硬隔离（如涉密项目）时，跑多个实例、各自端口，用
`MEMORY_PLUGIN_URL` 指向对应实例。**不要**把活跃库放在 NFS/SMB 或同步盘上
（WAL 锁语义不可靠）；跨机迁移走备份。在出现硬隔离需求之前，优先用
单实例 + `project` 过滤。

## HTTP 读/管接口 + 内置浏览页

同一进程还提供**人面信道**（与钩子用的 `/recall` `/ingest` 区分，
鉴权规则一致：配置了 `MEMORY_TOKEN` 即需 Bearer）：

| 端点 | 用途 |
|---|---|
| `GET /` | 自包含浏览页（零构建、零框架、离线可用） |
| `GET /api/stats` | 总量、向量化数、库体积、嵌入通道/设备 |
| `GET /api/memories?kind=&project=&limit=&offset=` | 分页浏览，按更新时间倒序 |
| `DELETE /api/memories/{uid}` | 单条遗忘（与 CLI 同信任级） |
| `POST /api/maintenance/purge` | 批量维护（软删除硬清 + TTL + VACUUM） |
| `POST /api/maintenance/backfill` | 向量补嵌 |

批量维护端点走**双闸门**：未配置 `MEMORY_TOKEN` 一律 `403`（默认无 token
的回环安装下，HTTP 面对本机所有进程——包括 agent 的 shell——都是开放的，
此时该能力仅 CLI 可达）；持错票/无票 `401`。浏览页把两个操作做成按钮
（确认 → 报告 → 统计刷新）。

浏览器打开 `http://127.0.0.1:8765/` 即可查看统计总览、RRF 检索、
kind/project 过滤、单条遗忘，以及（在安全安装下）一键维护。
仍然没有批量编辑——可观测性让"相关性不丢失"可被验证。

## 维护与生命周期

保留策略是**配置，不是 LLM 的自由裁量**。批量破坏性操作**刻意不作为 MCP
工具暴露**——它们待在 prompt 注入够不到的地方：启动钩子和 CLI。agent 只能
`memory_forget`（单条、用户发起）和 `memory_stats`（观察和建议）。

- **摄取去重**（写入路径内嵌，无人"调用"）：精确内容哈希（全局）或向量
  近似（同 project，余弦 ≥ `MEMORY_DEDUP_SIM`）命中时刷新已有条目时间戳，
  不新增——重复出现视为"再次被证实相关"。这直接掐住压缩托管类来源反复
  存入近似内容导致的增长斜率。
- **保留分层**：`preference` / `decision` / `fact` 永久；`artifact`
  （工具捕获）按 `MEMORY_TTL_ARTIFACT_DAYS` 过期。
- **`memory-mcp purge`**：硬删软删除条目 → TTL 清理 → 有实际删除才 `VACUUM`。
  同一例程在每次服务启动时运行（`MEMORY_MAINTENANCE_ON_START=1`）——
  重启频率对天级粒度的保留策略足够；无 cron、无后台线程。
- **`memory-mcp backfill`**：为所有缺向量的活条目补算嵌入（装模型后的
  一次性追平）。重操作，保持手动。

## 向量通道（与 ctx-semantic 同源）

嵌入层逐项对齐 [ctx-semantic](https://github.com/guangbin79/ctx-semantic) 的向量与方式，
同一模型产出同一向量（llama-cpp-python 也锁定同一版本 0.3.35）：

- 模型：Qwen3-Embedding-0.6B **Q8_0** GGUF，dim 1024，pooling **LAST**（显式指定）
- 非对称嵌入：**查询侧**自动加 `Instruct: ...\nQuery: {text}` 前缀，**文档侧**原文
- n_ctx=8192，批大小 32 子批，float32 行 L2 归一化（1e-12 保护）
- 向量检索：暴力余弦（~50k 条内亚毫秒，无需 ANN）

**设备策略（本项目规格，与 ctx-semantic 相反）**：`MEMORY_DEVICE=cuda` 为**默认**，
`cpu` 可选。GPU 路径带确定性回退：模型加载 → 探针嵌入 → **nvidia-smi PID 验证**
（识别"声称 GPU 实际 CPU"的静默回退）→ 任何失败重建于 CPU 并记日志；
GPU 运行期嵌入失败（如 OOM）自动 CPU 重建重试一次。无 NVIDIA 环境设
`MEMORY_DEVICE=cpu` 可跳过 GPU 尝试。`memory_stats` / `/health` 返回实际运行设备。

## 多语言全文检索

- **porter 通道**（unicode61 + 英文词干）：索引所有有词边界的文字——英、法、德、西、俄、希腊、阿拉伯语等；
- **trigram 通道**（≥3 字符子串）：覆盖无词边界的文字——中、日、韩、泰、老挝、高棉、缅语，并为任意语言提供子串兜底（包括德语复合词）。

已知缺口：两字中文词（如"坐标""投影"）低于 trigram 下限。向量通道（Qwen3-Embedding
支持 100+ 语言）已兜底；若纯 BM25 模式出现真实漏召回，修复方案是应用层 bigram
通道（约 15 行，零依赖）——见分期路线。

## 测试

```bash
pip install -e ".[dev]"
pytest            # 33 项：存储/双通道召回/软删除/RRF 数学/CJK 子串/查询前缀约定/
                  # 容错/生命周期去重与 TTL/迁移/REST 与 /api 路由/
                  # 批量双闸门鉴权/MCP 工具层/随机端口与服务发现

cd opencode-plugin
npm install       # 然后：
npm run typecheck # tsc --strict，类型基线 @opencode/plugin 2.0.28
npm test          # 构建 + 16 项 mock-ctx 钩子断言（打真实服务；
                  # stable 线：注入/去重/显式记忆/压缩托管/工具捕获；
                  # legacy 线：v1 钩子映射、synthetic part、compacting 重置）
```

插件另经真实 opencode dev 加载器（`0.0.0-dev-202610130450`）验证：双形态导出
被接受、自动选 legacy 线、项目标识正确解析为工作区目录，无加载错误。

## 分期路线

- [x] 第一期：MCP 工具 + 混合检索 + REST 侧车（本仓库当前状态）
- [x] 第二期：opencode v2 插件四族钩子（注入 / 压缩前托管 / 显式记忆 / 工具白名单）
- [x] 第三期a：多宿主适配器（`adapters/`——Claude Code、Kimi Code、Codex 模板）
- [x] 第三期b：生命周期维护——摄取去重、TTL/purge/VACUUM、向量补嵌
      （启动钩子 + CLI，刻意不在 MCP 工具面）
- [ ] 第三期c：chat.message 语义级自动捕获、两字中文词 bigram 通道
      （触发条件：纯 BM25 模式出现真实漏召回）、跨项目记忆图谱、冷文件归档

## 致谢

架构参考：[Basic Memory](https://github.com/basicmachines-co/basic-memory)（Markdown+SQLite FTS）、
[ctx-semantic](https://github.com/guangbin79/ctx-semantic)（BM25+Qwen3-Embedding+RRF）、
[context-mode](https://github.com/mksglu/context-mode)（钩子拦截范式）。
