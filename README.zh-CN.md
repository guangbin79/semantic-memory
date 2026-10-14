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
| `MEMORY_HOST` / `MEMORY_PORT` | `127.0.0.1` / `8765` | 绑定地址 |
| `MEMORY_TOKEN` | 空 | REST 侧车 Bearer 鉴权；绑非回环地址时**必配** |
| `MEMORY_RRF_K` | `60` | RRF 融合常数 |

插件：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_PLUGIN_URL` | `http://127.0.0.1:8765` | 服务地址 |
| `MEMORY_PLUGIN_TOKEN` | 空 | 与服务端 `MEMORY_TOKEN` 对应 |
| `MEMORY_PLUGIN_LIMIT` | `5` | 每次召回条数 |
| `MEMORY_PLUGIN_BUDGET_CHARS` | `2000` | 注入块字符预算 |
| `MEMORY_PLUGIN_TOOL_CAPTURE` | 空 | 工具捕获白名单（逗号分隔），如 `gis,bash` |
| `MEMORY_PLUGIN_AUTOSTART` | `0` | `=1` 时服务未启动则自动 spawn |

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
pytest            # 8 项：存储/双通道召回/软删除/RRF 数学/CJK 子串/查询前缀约定/容错
```

## 分期路线

- [x] 第一期：MCP 工具 + 混合检索 + REST 侧车（本仓库当前状态）
- [x] 第二期：opencode v2 插件四族钩子（注入 / 压缩前托管 / 显式记忆 / 工具白名单）
- [x] 第三期a：多宿主适配器（`adapters/`——Claude Code、Kimi Code、Codex 模板）
- [ ] 第三期b：chat.message 语义级
      自动捕获、两字中文词 bigram 通道（触发条件：纯 BM25 模式出现真实漏召回）、
      跨项目记忆图谱

## 致谢

架构参考：[Basic Memory](https://github.com/basicmachines-co/basic-memory)（Markdown+SQLite FTS）、
[ctx-semantic](https://github.com/guangbin79/ctx-semantic)（BM25+Qwen3-Embedding+RRF）、
[context-mode](https://github.com/mksglu/context-mode)（钩子拦截范式）。
