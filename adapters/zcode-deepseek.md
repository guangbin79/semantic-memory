# ZCode / DeepSeek Harness：仅 MCP 数据面

这两家当前没有可依赖的生命周期钩子能力（ZCode 插件清单无 hooks 字段；
DeepSeek Harness 核心未内置 MCP、生态处 developer preview），因此只接入
**Tier 1：MCP 工具**。控制面（自动召回注入）用 AGENTS.md 纪律代替：

> 开始涉及历史决策/既往配置的任务前，先调用 `memory_recall`；
> 形成决策或发现关键事实后，调用 `memory_store`。

## ZCode（智谱）

`setting.json` 的 `mcp.servers` 增加：

```json
{
  "mcp": {
    "servers": {
      "memory": { "type": "remote", "url": "http://127.0.0.1:8765/mcp" }
    }
  }
}
```

## DeepSeek Harness

核心无 MCP 支持，需社区插件 [`dsh-mcp-manager`](https://github.com/hyqhyq3/dsh-mcp-manager)
（读取 `.dsh/dshmm/mcp.json`，支持 remote HTTP / local stdio）：

```json
// .dsh/dshmm/mcp.json
{
  "mcpServers": {
    "memory": { "type": "remote", "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

待两家官方钩子能力落地后，可直接复用 `adapters/common/` 的两个脚本升级
为完整控制面适配。
