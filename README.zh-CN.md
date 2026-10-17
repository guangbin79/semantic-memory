# semantic-memory

**[English](README.md) | [中文](README.zh-CN.md)**

> 旧家目录（`~/.local/share/memory-mcp/` 与 v0.2 的 `~/semantic-memory-mcp/`）数据
> 会在首次启动时自动迁移（见下文「统一家目录」）。

coding agent 上下文管理的**侧车（sidecar）**：提供**模型语义级**的上下文检索，
确保上下文相关性不丢失。

**它的目标不是节省 token、不是压缩上下文。** opencode v2 原生 compaction 和 DCP
负责上下文"数量"（压缩、修剪、丢弃以适配窗口）；本组件负责上下文"相关性"——
被压缩掉的、被剪掉的、上一个会话里的一切，都进入语义可检索的仓库，当它与
**当前任务**相关时被自动找回注入。它花一笔受预算约束的 token（注入块），
买回本会永久丢失的相关性。压缩让窗口够用，侧车让压缩安全。

- 自包含 HTTP 服务（纯 Python 标准库，零第三方依赖）：BM25（FTS5 双通道 porter + trigram）+ 向量（Qwen3-Embedding GGUF，可选）+ RRF 融合 + 可选本地重排——检索发生在嵌入空间，措辞不同也能命中；异步蒸馏把原文捕获消化成 typed facts
- opencode v2 插件（`integrations/opencode/memory.ts`）：钩住 context / compaction / generate / chat.message / tool execute.after，让"记"和"忆"自动发生
- 独立进程、独立模型：不占用主 agent 的窗口与模型，服务宕机会话无感（fail-open）

## 架构

```
┌─ opencode v2 插件（memory.ts，零依赖，fail-open）────────┐
│  context/compaction/generate → /recall → 注入块           │
│    拼在末条 user 消息之前（次末位）+ 每会话冻结头块        │
│    （偏好/决定，字节冻结进 event.system）                 │
│  session.idle → POST /idle（排空蒸馏队列）                │
│  compaction 前 → /ingest（压缩丢弃 ≠ 记忆永存）            │
│  chat.message "记住…" / 语义自动捕获 → /ingest            │
│  tool execute.after（白名单）→ /ingest                    │
└──────────────────────┬────────────────────────────────────┘
                       │ HTTP 127.0.0.1:8765（Bearer token 可选）
┌──────────────────────▼────────────────────────────────────┐
│  semantic-memory 服务（单进程，标准库 http.server）        │
│  REST API：/recall /ingest /idle /health + 浏览页         │
│  opencode 原生工具：memory_store / memory_recall          │
│  SQLite + FTS5(porter|trigram) + 向量BLOB，RRF(k=60)      │
│                                                          │
│  检索侧：RRF → 冗余上限 → 本地重排                        │
│  写入侧：蒸馏队列 → 异步 flush → typed facts              │
│  云通道（配置生效）：整合只返回元数据操作计划——            │
│    云端内容永不入库                                       │
│  共享模型槽：嵌入常驻；重排与蒸馏分时复用单槽（LRU=1）      │
└───────────────────────────────────────────────────────────┘
```

HTTP API 为**多宿主支持**而生：任何带生命周期钩子的 agent harness
（Claude Code、Codex、Kimi Code……）都可以把控制面接成"钩子事件 → curl
localhost:8765"——只有 opencode v2 需要进程内插件，因为它的钩子在 JS 运行时里。
各家接线见 `integrations/` 与快速开始里的支持矩阵。

设计约束：插件零智能（检索/嵌入/存储全在服务端）；服务挂了插件静默跳过；注入集合未变不重复注入（保护 prefix cache）；服务可独立运行——不装任何插件，HTTP API 与浏览页照常可用。
度量标准不是"省了多少 token"，而是"当前任务所需相关性的召回率"（recall@k）。

## 快速开始

### 零仪式安装（三档任选）

| 档位 | 命令 | 得到什么 |
|---|---|---|
| 0 —— 源码直跑，零安装 | `PYTHONPATH=src python3 -m semantic_memory serve` | 裸 python3 ≥ 3.11 直接跑，零第三方依赖（纯 BM25；缺 numpy → 向量腿自关，附自解释安装提示） |
| 1 —— pip 用户安装 | `pip install --user .` | `semantic-memory` 命令行入口，仍是零依赖 |
| 2 —— 向量 / 全栈 | `pip install ".[vector]"` 或 `pip install ".[embed]"` | +numpy（向量腿）/ +llama-cpp-python（本地 GGUF 推理） |

第 2 档的模型按通道可选：`bash scripts/download_model.sh`（嵌入 →
`MEMORY_MODEL_PATH`）与 `bash scripts/download_models.sh`（蒸馏 + 重排 →
`MEMORY_DISTILL_MODEL` / `MEMORY_RERANK_MODEL`）——也可以完全跳过手动步骤，
用下面的一键安装。

### 一键安装（探测 → 校验下载 → 写配置 → 冒烟）

人，一条 curl（钉 release tag v1.0）：

```bash
curl -fsSL \
  https://raw.githubusercontent.com/guangbin79/semantic-memory/v1.0/scripts/setup.sh \
  | bash -s -- --profile full --json --yes
```

agent（仓库已签出）：

```bash
bash scripts/setup.sh --profile full --json --yes
```

两者都会建出完全自立的 `~/semantic-memory/`（venv + sha256 校验下载模型 +
`config.toml` + 冒烟）。`--profile` 选档：`full`（嵌入+蒸馏+重排，默认）/
`embed` / `bm25`（零下载）。`--json` 报告为机器可读单行；之后一切命令都是
`~/semantic-memory/venv/bin/semantic-memory …`。

### 起服

```bash
semantic-memory serve    # 或：~/semantic-memory/venv/bin/semantic-memory serve
# → http://127.0.0.1:8765  （REST API + 浏览页；MEMORY_PORT=0 = OS 随机端口）
```

### 接线（生命周期钩子，而非工具协议）

| 宿主 | 支持 | 接线方式 |
|---|---|---|
| opencode v2 | ✅ | 进程内插件——hooks **加** 原生 `memory_store` / `memory_recall` 工具（`integrations/opencode/`） |
| Claude Code | ✅ | hooks：`UserPromptSubmit` + `PreCompact`（`integrations/claude-code/`） |
| Kimi Code | ✅ | hooks：`[[hooks]]` 或插件清单（`integrations/kimi-code/`） |
| Codex | ⚠️ | 仅模板——hook 插件需逐 hook 用户信任（`integrations/codex/`） |
| ZCode / DeepSeek Harness | ❌ | 无生命周期钩子；MCP 无法承接（hooks 是进程/HTTP 生命周期事件，不是可调用工具）——待其官方提供 hooks 后重评 |

opencode v2（唯一进程内集成）：

```bash
cp integrations/opencode/memory.ts ~/.config/opencode/plugins/
# 插件目录的 package.json 需有 @opencode/plugin（仅类型）：
#   cd ~/.config/opencode && npm install @opencode/plugin@^2.0.28
# ——或不复制，直接在 opencode.jsonc 的 plugin 数组声明，可附带插件
#   options（见下文"配置"一节）：
#   "plugin": [["file:///abs/path/to/semantic-memory/integrations/opencode", {"url": "http://127.0.0.1:9000"}]]
```

其余宿主都是"钩子事件 → curl localhost:8765"的薄适配器——见
`integrations/README.md`。

### 统一家目录：`~/semantic-memory/`

| 路径 | 内容 |
|---|---|
| `~/semantic-memory/venv/` | 自带虚拟环境（一键安装） |
| `~/semantic-memory/models/` | GGUF 模型（嵌入 / 蒸馏 / 重排） |
| `~/semantic-memory/data/memory.db` | SQLite 记忆库（`MEMORY_DB_PATH` 默认） |
| `~/semantic-memory/config.toml` | 文件配置层：env > 此文件 > 内置默认 |

解析序：`MEMORY_DATA_DIR` > 显式 `XDG_DATA_HOME`（→
`$XDG_DATA_HOME/memory-mcp/`）> `~/semantic-memory/`。**旧目录自动迁移**：
首次启动时，历代旧家目录（`~/.local/share/memory-mcp/`、v0.2 的
`~/semantic-memory-mcp/`）会逐个整体平移进来——先
wal_checkpoint，逐文件 sha256 校验，旧服务还在跑则拒迁，旧目录最终只留一个
`MIGRATED` 指针文件。幂等；未校验通过绝不删数据。

### 分层严格启动

服务拒绝半配置启动（宁可大声失败，不要静默降级）：

- **非回环绑定且未配 `MEMORY_TOKEN`** → 拒启 `exit 2`（契约 enforced：无 token 的
  REST 面会把读/写/单条遗忘裸暴露给网络；`MEMORY_ALLOW_BM25_ONLY` 不豁免——
  那只管模型档）。
- **全未配置、无 `config.toml`** → 招募模式：纯 BM25 合法运行，附指向
  `semantic-memory setup` 的安装提示。
- **已配置（env 或 `config.toml`）但模型文件缺失** → 拒启 `exit 2`，指明缺失
  路径与修复用的 setup 命令。
- **`config.toml` 在场** → 按 setup 写入的 profile 严格生效。
- **`MEMORY_ALLOW_BM25_ONLY=1`**（或文件内 `allow_bm25_only = true`）→ 恒放行，
  显式的纯 BM25 逃生门。

## 配置（环境变量）

服务端：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_DATA_DIR` | `~/semantic-memory/`（解析序：此 env > 显式 `XDG_DATA_HOME` → `$XDG_DATA_HOME/memory-mcp/` > 默认家目录） | 数据家目录覆写：models/ data/ config.toml 都在其下 |
| `MEMORY_DB_PATH` | `~/semantic-memory/data/memory.db` | 数据库路径（只覆 db 本身；家目录其余部分不受影响） |
| `MEMORY_MODEL_PATH` | 空 | 嵌入 GGUF 模型路径；空 = 纯 BM25 降级 |
| `MEMORY_DEVICE` | `cuda` | 嵌入设备：`cuda`（默认，失败自动回退 CPU）/ `cpu` |
| `MEMORY_EMBED_DIM` | `1024` | 嵌入维度 |
| `MEMORY_HOST` / `MEMORY_PORT` | `127.0.0.1` / `8765` | 绑定地址；端口 `0` = OS 随机分配 |
| `MEMORY_DISCOVERY_FILE` | `<db>.server.json` | 服务发现文件；`off` = 关闭 |
| `MEMORY_TOKEN` | 空 | HTTP API 的 Bearer 鉴权；绑非回环地址时**必配**（否则启动即拒 exit 2） |
| `MEMORY_RECALL_LIMIT` | `5` | 请求未带 limit 时的默认召回条数 |
| `MEMORY_RRF_K` | `60` | RRF 融合常数 |
| `MEMORY_RECALL_MIN_COSINE` | `0.45` | 纯向量命中的余弦门限（BM25 命中恒放行）；`0` = 关闭 |
| `MEMORY_RECALL_MAX_COSINE` | `0.92` | 冗余上限：纯向量命中超过此值按冗余出局（BM25 不受影响）；`1.0` = 关闭 |
| `MEMORY_CROSS_PROJECT` | `1` | 跨项目图谱：项目召回后追加 B 项目同源解法；`=0` 关闭 |
| `MEMORY_CROSS_LIMIT` | `2` | 跨项目附加条目上限（不占主配额） |
| `MEMORY_CROSS_SIM` | `0.90` | synonym 边的跨项目近邻余弦阈值 |
| `MEMORY_INJECT_BOOST` | `0.15` | 遥测反哺：score × (1 + α·ln(1+注入次数))；`0` = 关闭 |
| `MEMORY_RERANK_MODEL` | `~/semantic-memory/models/qwen3-reranker-0.6b-q8_0.gguf` | 重排 GGUF（**仅 ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF**——官方转换，社区镜像可能缺 rank 头）；默认指向常规路径（门禁 2026-10-16 PASSED：recall@5 0.939 vs 基线 0.909）——文件在即启用，缺失优雅关闭并告警；设空禁用 |

**云通道隐私。** 配置 `MEMORY_CLOUD_BASE_URL` + `MEMORY_CLOUD_API_KEY` + `MEMORY_CLOUD_MODEL`（且 `MEMORY_CLOUD_ENABLED` ≠ `0`，默认）即意味着**记忆内容会发送至该端点**——经确定性预过滤的近邻簇成员及其直接邻域，绝非全库。云端只返回**元数据操作计划**（按 uid 的 `supersede`/`touch`）：云生成的任何内容都不入库，计划中出现疑似注入模式即整批丢弃。每日硬顶（`MEMORY_CLOUD_DAILY_MAX_CALLS` / `MEMORY_CLOUD_DAILY_MAX_TOKENS`）是进程内存计数——**服务重启即重置**。`MEMORY_CLOUD_ENABLED=0` 是优先级最高的 kill switch。
| `MEMORY_RERANK_CANDIDATES` | `20` | RRF 融合后送入重排的候选条数 |
| `MEMORY_RERANK_BUDGET_MS` | `500` | 单次召回的重排总预算（含模型槽加载）；超时整体 fail-open 回 RRF 序 |
| `MEMORY_DISTILL_MODEL` | 空 | 蒸馏 GGUF（Qwen3-1.7B 非思考）；空 = 蒸馏关闭（flush 时原文直存） |
| `MEMORY_DISTILL_BATCH` | `20` | 队列 pending ≥ N 触发 flush；`0` = 关闭批量触发 |
| `MEMORY_CLOUD_ENABLED` | `1` | 云整合 kill switch：`=0` 强制关闭（即使配置了 `MEMORY_CLOUD_*`） |
| `MEMORY_CLOUD_BASE_URL` | 空 | OpenAI 兼容 `/chat/completions` 端点；空 = 云通道关闭 |
| `MEMORY_CLOUD_API_KEY` | 空 | 云端 API key；空 = 关闭 |
| `MEMORY_CLOUD_MODEL` | 空 | 云端模型名；空 = 关闭 |
| `MEMORY_CLOUD_BATCH_N` | `30` | 待整合候选 ≥ N 自动触发整合 |
| `MEMORY_CLOUD_DAILY_MAX_CALLS` | `20` | 每日云端调用硬顶；超顶当日停发 |
| `MEMORY_CLOUD_DAILY_MAX_TOKENS` | `200000` | 每日 token（估算）硬顶 |
| `MEMORY_TTL_ARTIFACT_DAYS` | `90` | `artifact` 类 TTL 天数；`0` = 不过期 |
| `MEMORY_TTL_INJECTION_DAYS` | `90` | 注入遥测流水 TTL 天数；`0` = 不过期 |
| `MEMORY_TTL_STALE_DAYS` | `180` | 使用度留存：`fact`/`artifact` 超此天数未被召回，下次 purge 时软删（NULL 按 created_at；`preference`/`decision` 豁免）；`0` = 关闭 |
| `MEMORY_MAINTENANCE_ON_START` | `1` | 服务启动时运行维护例程 |
| `MEMORY_DEDUP_SIM` | `0.98` | 向量近似去重阈值（同 project 内） |
| `MEMORY_ALLOW_BM25_ONLY` | `0` | 第 0 档逃生门：`=1` 时即使 `config.toml` 在场也允许纯 BM25 启动 |

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
| `MEMORY_PLUGIN_INJECT_POSITION` | `tail` | `tail` = 注入块拼在末条 user 消息之前；`system` = 旧行为 `event.system.push` 回退 |
| `MEMORY_PLUGIN_HEAD_BLOCK` | `1` | 每会话一次的冻结头块（偏好/决定，字节冻结推入 `event.system`）；`=0` 关闭 |
| `MEMORY_PLUGIN_GLOBAL_MEMORIES` | `1` | 全局记忆两路召回（`/recall` 带 `global_limit=2`）；`=0` 关闭 |

每个 `MEMORY_PLUGIN_*` 变量都有一个插件 options 孪生键（`url`、`token`、
`limit`、`budget_chars`、`tool_capture`、`autostart`、`debug`、`inject_position`、`head_block`、`global_memories`），因此也可以
直接在 `opencode.json` 的插件声明处配置——优先级：**环境变量 > 插件
options > 默认值**：

```jsonc
{
  "plugin": [
    ["file:///path/to/semantic-memory/integrations/opencode",
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
MEMORY_PORT=0 semantic-memory
# stdout（机器可读，单行）：
# MEMORY_LISTENING {"url":"http://127.0.0.1:41697","port":41697,"pid":...}
# 同一份 JSON 同时写入发现文件：<db>.server.json（可用 MEMORY_DISCOVERY_FILE 覆盖）
```

自举流程：spawn → 解析 stdout 行**或**读发现文件 → 把 `url` 填进
`MEMORY_PLUGIN_URL`。优雅退出（SIGINT/SIGTERM）时撤下文件；被 kill -9 时
读取方可凭文件内 `pid` 判活。

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
  `semantic-memory backfill` CLI 一次性追平（见"维护与生命周期"）。

## 存储布局

一切收在**一个 SQLite 文件**里（`MEMORY_DB_PATH`，默认
`~/semantic-memory/data/memory.db`）：

| 对象 | 内容 |
|---|---|
| `memories` | 原文 + 元数据（kind / source / project / tags / 时间戳） |
| `memories_fts`、`memories_fts_tri` | FTS5 倒排索引（porter / trigram），外部内容表——正文不重复存储 |
| `memories_vec` | 向量 BLOB：每条 1024 维 float32 = 4KB |
| `injections` | 注入遥测流水（harness/钩子/查询/uids/字符数） |

运行期有 WAL 伴生文件（`-wal`、`-shm`）；GGUF 模型（~600MB）是独立文件。
删除 db 文件即干净的全量重置——没有其他状态。备份用
`sqlite3 memory.db ".backup '/path/to/backup.db'"`。与 ctx-semantic 不同，
没有 ANN 需要构建或重建——且自 v0.3 起向量矩阵按 `(project, scope)` 键进程内
缓存，由 `PRAGMA data_version`（任何连接提交即推进，含 distill flush 的独立
连接）+ 本连接写计数双轨失效。无写入交错的召回不再付 SQLite 向量读。

## 共享模型

- **跨会话**：索引在磁盘上，会话是进程。配合压缩前托管，连被丢弃的上下文
  也能活过会话边界。
- **跨 harness**：opencode v2、Claude Code、Kimi Code、Codex 都连同一个服务，
  即**同一份索引**——Claude Code 存的记忆在 opencode 里可召回。换 harness
  不丢记忆；没有第二份拷贝需要同步。
- **project 是过滤器，不是隔离墙**：索引全局唯一；项目域召回在服务端两腿（BM25 与向量）
  都做 project 等值过滤（2026-10-17 起 BM25 腿对齐向量腿），但 `memory_recall` 不传 project
  就是全库检索；跨项目同源解法另走图谱附加条目（显式标记，不冒充本项目命中）。
- **单用户单机**：回环绑定 + 可选 Bearer token，无多租户。
- **归因缺口**：`source` 记录来源类型（chat / compaction / tool），但不记录
  由哪个 harness 写入——出现该查询需求时加 `harness:<name>` 标签即可。

### 跨项目记忆图谱

查 A 项目能带出 B 项目的同源解法。三种边全部从既有数据查询侧派生（零新表、零写路径状态）：

- **supersedes（取代链）**：`memories.superseded_by` 指针；前代经管理面读出，仅观测
  不注入（被取代 = 过时）；`GET /api/graph` 链式遍历全链。
- **co_injection（共注入）**：注入遥测里同块共现过的条目（权重 = 共注入次数；只看近 30 天遥测，骑 idx_injections_ts 有界扫）。
- **synonym（同义跨域）**：全矩阵暴力余弦的跨项目近邻（权重 = 余弦；纯 BM25 无向量时缺席）。

`/recall` 带项目时在主结果后追加附加条目（`cross_kind`/`via_uid` 标记，不占主配额，
`MEMORY_CROSS_LIMIT` 截断，synonym 优先）；插件注入块渲染 `(cross-project <kind>)` 标记。
附加条目被上报后自然加固 co_injection 边——飞轮。观测面：`GET /api/graph?uid=` 返回
单条目 1-hop 边（以 `to_uid` 为种子再查即遍历）。

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
| `GET /api/stats` | 总量、向量化数、库体积、嵌入通道/设备、注入统计 |
| `GET /api/memories?kind=&project=&limit=&offset=` | 分页浏览，按更新时间倒序 |
| `GET /api/injections?harness=&project=&limit=&offset=` | 注入遥测，按时间倒序 |
| `GET /api/graph?uid=&cross_sim=` | 单条目 1-hop 图谱边（取代链/共注入/同义跨域，可遍历观测面） |
| `POST /inject/report` | 注入遥测上报（插件/适配器信道，只追加） |
| `POST /idle` | 通知空闲：daemon 线程排空蒸馏队列（立即 204，鉴权同 /ingest） |
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

**注入遥测。** 召回 ≠ 注入——只有注入方知道这次召回真的进了上下文，
因此 opencode 插件在每完成一次真实注入后向 `POST /inject/report`
上报（fire-and-forget）：harness、钩子、触发查询、注入的记忆 uid、
块大小。浏览页据此呈现供给侧图景：累计/近 7 天注入次数、累计字符
（附 ÷4 的 token 粗估）、按 harness/钩子分布、**反复被注入的记忆
Top10**、最近 20 次注入流水。shell 适配器（Claude Code / Kimi Code）
同样上报——`recall.sh` 在输出注入块后 fire-and-forget 调 `/inject/report`，
用 `MEMORY_HARNESS` / `MEMORY_HOOK` 标记来源。这里度量的是"供给了
什么"——模型是否采纳不可度量，也刻意不做伪精确。遥测流水按
`MEMORY_TTL_INJECTION_DAYS`（默认 90 天）过期清理。

**上下文卫生。** 记忆内容是不可信数据，最终会进入 prompt。每个注入块都
带有明确的"参考资料而非指令"声明，且内容里出现的 `<semantic-memory>`
边界标签会被剥除——已存条目无法逃逸包裹。写入侧，工具产物捕获会跳过
命中典型 prompt-injection 模式（"ignore all previous instructions"、
"忽略…指令"、`<system>` 等）的产物，避免被污染的工具结果经记忆库
回流到后续会话。

## 排序后处理（实测取值，不是拍的）

RRF 只编码名次，不编码相关强度——"无相关里的最近邻"也会得分。融合后处理步骤：

- **余弦门限**（`MEMORY_RECALL_MIN_COSINE=0.45`）：纯向量命中低于门限
  丢弃；BM25 命中恒放行（token 命中本身就是证据）。门限来自真实模型
  实测：相关命中 ~0.68，无关 ~0.28-0.31。
- **遥测反哺**（`MEMORY_INJECT_BOOST=0.15`）：被真实注入过的记忆
  score × (1 + α·ln(1+注入次数))。对数抑制富者愈富；注入 2 次即可
  翻转相邻名次。
- **冗余上限**（`MEMORY_RECALL_MAX_COSINE=0.92`）：纯向量命中余弦超过上限即按
  冗余出局（"过近 = 已知"）；BM25 命中不受影响；`1.0` = 关闭。
- **本地重排**（`MEMORY_RERANK_MODEL`，**门禁 2026-10-16 PASSED 后默认开**——
  33 条 golden 集 recall@5 0.939 vs 基线 0.909，recall@1 +21pp）：RRF 融合后
  top `MEMORY_RERANK_CANDIDATES=20` 条由 Qwen3-Reranker-0.6B 重打分（贪心补全
  + 采样分布行 yes/no softmax，GPU 实测 ~13ms/候选），硬预算
  `MEMORY_RERANK_BUDGET_MS=500`（含模型槽加载）；超预算或任何错误 → 保 RRF 序
  （fail-open）。同分按 `updated_at` 倒序平决。默认指向常规模型路径——设
  `MEMORY_RERANK_MODEL=`（空）可禁用。

## 容量与告警（观测面，不打扰会话）

告警只出现在服务日志与浏览页徽章——绝不注入会话流。`/api/stats` 暴露
`recall_p50_ms`、`recall_p95_ms`、`distill_queue_len`、
`cloud_calls_24h`、`entries`（延迟取滚动 100 样本窗口）：

| 信号 | 阈值 | 级别 |
|---|---|---|
| recall p95 | > 800 ms | WARNING |
| recall p95 | > 1200 ms | ERROR——超过插件 1.5s 硬超时，注入将静默消失 |
| 条目数 | > 40000 | INFO——接近暴力扫性能天花板（~50k） |

同级告警 10 分钟内只打一次。

## 离线评测（先测量，后加装）

```bash
semantic-memory eval --golden FILE [--k 1,3,5] [--judge]
```

golden jsonl 每行 `{query, expect_uids[], project?}`；逐行输出 @k 命中，末尾
recall@k / MRR 汇总与 JSON 报告。纯 CLI——绝不进入任何服务端点。

**评测飞轮（v0.3）**：`semantic-memory golden [--out FILE]` 从真实注入遥测构建
golden 集（按 `(project, query)` 去重、取最近注入——供给侧标签：注入≠采纳，
但作为 recall@k 期望集自洽）。`--judge` 以 **Reranker** 评注未命中：期望内容
相关分高 = 检索真 miss，分低 = golden 标签本身可疑（无重排模型则优雅跳过）。
`--min-recall K:FLOAT`（如 `5:0.85`）把 eval 变成门禁——低于门槛退出码 1
（CI/回归用；`.github/workflows/ci.yml` 经 `MEMORY_GOLDEN`+`MEMORY_MIN_RECALL` 接线）。
`tests/fixtures/golden.sample.jsonl` 仅为 3 行格式示例；闸门用的真实集
（从你自己的记忆库构建 ≥30 条真实查询）决定重排默认：**在真实集上 eval
显示 recall@5 ≥ RRF 基线之前，重排保持关闭。**

## 维护与生命周期

保留策略是**配置，不是 LLM 的自由裁量**。批量破坏性操作**刻意不向 agent 暴露**
——它们待在 prompt 注入够不到的地方：启动钩子和 CLI。agent 面只有
`memory_store` / `memory_recall`（opencode 原生工具，fail-open）；单条遗忘在
浏览页 / `DELETE /api/memories/{uid}`。

- **摄取去重**（写入路径内嵌，无人"调用"）：精确内容哈希（全局）或向量
  近似（同 project，余弦 ≥ `MEMORY_DEDUP_SIM`）命中时刷新已有条目时间戳，
  不新增——重复出现视为"再次被证实相关"。这直接掐住压缩托管类来源反复
  存入近似内容导致的增长斜率。
- **写入侧蒸馏缓冲**（配置 `MEMORY_DISTILL_MODEL` 后）：`/ingest` 的 `source`
  为 `compaction` 或 `chat` 时落入持久化的 `distill_queue` 表——热路径零 LLM；
  显式 `remember` / `tool` 捕获保持直存。flush 触发器（三路，仅此三路——
  **无启动排空、无 cron**）：下次 `/ingest` 入队时 pending ≥ `MEMORY_DISTILL_BATCH`、
  `POST /idle` 通知、`semantic-memory distill`（手动）。flush 在 daemon 线程内用
  独立短连接 SQLite 跑（每批 ≤5 行提交，进程内单飞）；蒸馏输出按 typed facts
  落库（`scope=global` → `project` NULL，保留 session id）；蒸馏关闭/失败/
  不可解析 → 原文直存（`stored_raw`）——不差于蒸馏前行为；向量经 `backfill`
  追平。**仅装适配器（shell adapters，无 v2 插件）时无 `/idle` 投递者**：
  队列内容在批量阈值触发或手动 `semantic-memory distill` 之前不可检索——
  按设计的有界延迟。
- **并发契约**：服务仅有的后台工作是**蒸馏 flush 的 daemon 线程**（由 `/idle`、
  批量阈值或 CLI 触发），每次 flush 完成后在同一线程内评估云整合触发。
  两级都 fail-open、各自使用独立短连接、绝不在同步请求路径内跑。
  仍然没有 cron。
- **保留分层**：`preference` / `decision` 永久；`fact` / `artifact` 按使用度
  存活——超过 `MEMORY_TTL_STALE_DAYS`（默认 180；NULL 按 created_at）未被召回，
  下次 purge 时软删，被矛盾取代（superseded）的条目一并软删；注入遥测流水按
  `MEMORY_TTL_INJECTION_DAYS` 过期。
- **`semantic-memory purge`**：硬删软删除条目 → TTL 清理 → 有实际删除才 `VACUUM`。
  同一例程在每次服务启动时运行（`MEMORY_MAINTENANCE_ON_START=1`）——
  重启频率对天级粒度的保留策略足够。
- **`semantic-memory backfill`**：为所有缺向量的活条目补算嵌入（装模型后的
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
`MEMORY_DEVICE=cpu` 可跳过 GPU 尝试。`/api/stats` / `/health` 返回实际运行设备。

## 多语言全文检索

- **porter 通道**（unicode61 + 英文词干）：索引所有有词边界的文字——英、法、德、西、俄、希腊、阿拉伯语等；
- **trigram 通道**（≥3 字符子串）：覆盖无词边界的文字——中、日、韩、泰、老挝、高棉、缅语，并为任意语言提供子串兜底（包括德语复合词）。

已知缺口：两字中文词（如"坐标""投影"）低于 trigram 下限。向量通道（Qwen3-Embedding
支持 100+ 语言）已兜底；若纯 BM25 模式出现真实漏召回，修复方案是应用层 bigram
通道（约 15 行，零依赖）——见分期路线。

## 测试

```bash
pip install -e ".[dev]"
pytest            # 基线 43 项：存储/双通道召回/软删除/RRF 数学/CJK 子串/
                  # 查询前缀约定/容错/生命周期去重与 TTL/迁移/REST 与 /api 路由/
                  # 批量双闸门鉴权/
                  # 注入遥测（上报/聚合/TTL/鉴权）/排序后处理（余弦门限、遥测反哺）/
                  # 真实模型向量端到端（MEMORY_MODEL_PATH 可选启用）
                  # + 演进套件：蒸馏队列与 typed-fact 取代 / 重排器与共享模型槽 /
                  # git-origin 项目键 / 使用度留存 / 云整合（mock 传输）/
                  # 离线评测 / 召回延迟遥测与节流告警 / 跨项目图谱（三种边+召回附加）
                  # HTTP 硬化（项目域 BM25 等值过滤/畸形参数 400/uids 截断/单文件可跑）

cd integrations/opencode
npm install       # 然后：
npm run typecheck # tsc --strict，类型基线 @opencode/plugin 2.0.28
npm test          # 构建 + mock-ctx 钩子断言（打真实服务；
                  # 基线 37 项 + 演进：次末位注入、冻结头块、按会话键控状态、
                  #  chat 语义自动捕获、idle 通知；
                  #  stable 线：注入/去重/显式记忆/压缩托管/工具捕获/遥测上报/
                  #  捕获卫生/边界标签消毒；
                  #  legacy 线：v1 钩子映射、synthetic part、compacting 重置、遥测上报；
                  #  options：env 优先、缺省补位）
```

插件另经真实 opencode dev 加载器（`0.0.0-dev-202610130450`）验证：双形态导出
被接受、自动选 legacy 线、项目标识正确解析为工作区目录，无加载错误。

## 分期路线

- [x] 第一期：混合检索 + REST 控制面（早期 MCP 工具面已在 v0.2 移除——现在是 hooks-only）
- [x] 第二期：opencode v2 插件四族钩子（注入 / 压缩前托管 / 显式记忆 / 工具白名单）
- [x] 第三期a：多宿主适配器（`integrations/`——Claude Code、Kimi Code、Codex 模板）
- [x] 第三期b：生命周期维护——摄取去重、TTL/purge/VACUUM、向量补嵌
      （启动钩子 + CLI，刻意不在 agent 工具面）
- [x] 第三期c：上下文补全演进——写入侧蒸馏、本地重排 + 共享模型槽、
      次末位注入 + 冻结头块、云整合、使用度留存、离线评测、观测面
- [x] 第四期a：跨项目记忆图谱——取代链/共注入/同义跨域可遍历边（`/recall` 同源附加 + `GET /api/graph`）
- [ ] 第四期b：两字中文词 bigram 通道（触发条件：纯 BM25 模式出现真实漏召回；全配有向量兜底，不急）
- [ ] 第四期c：冷文件归档（缓建——preference/decision 永不删但 SQLite TTL/purge 留存已管增长；
      热集接近 ~50k 暴力余弦上限时再启动）

