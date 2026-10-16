# Harness adapters

The REST sidecar (`/recall` `/ingest` `/health`) makes every control-plane
adapter a thin "hook event → curl localhost:8765" client. Only opencode v2 needs
an in-process plugin (`opencode-plugin/memory.ts`); everyone else reuses the
shared scripts here.

| Harness | Tier 1: MCP tools | Tier 2: hook automation | Status |
|---|---|---|---|
| opencode v2 | ✅ | ✅ in-process plugin | done (`opencode-plugin/`) |
| Claude Code | ✅ `mcpServers` | ✅ `UserPromptSubmit` + `PreCompact` | done (`claude-code/`) |
| Kimi Code | ✅ `mcp_servers` | ✅ `[[hooks]]` or plugin manifest | done (`kimi-code/`) |
| Codex | ✅ `mcp_servers` | ✅ hooks.json (v0.150+, needs user trust) | template (`codex/`) |
| ZCode | ✅ `mcp.servers` | ❌ no hooks yet | MCP-only (`zcode-deepseek.md`) |
| DeepSeek Harness | ⚠️ via community `dsh-mcp-manager` | ❌ preview ecosystem | MCP-only (`zcode-deepseek.md`) |

## Shared scripts (`common/`)

- `recall.sh "query" [limit]` — prints a `<semantic-memory>` block on stdout
  (empty when no hits). Fail-open, 1.5s budget.
- `ingest.sh "content" [kind] [source] [project]` — stores one memory. Fail-open.
- Idle flush notification (`POST /idle`) is opencode-plugin-only (v2 `session.idle`
  event). Shell adapters have no idle counterpart — their queued entries (chat/
  compaction sources) flush via the batch threshold (pending ≥ 20) or `memory-mcp
  distill` manually.

Environment: `MEMORY_PLUGIN_URL` (default `http://127.0.0.1:8765`),
`MEMORY_PLUGIN_TOKEN`, `MEMORY_PLUGIN_LIMIT`, `MEMORY_PLUGIN_BUDGET_CHARS` —
same names as the opencode plugin.

## Claude Code

```bash
# 1. make scripts executable
chmod +x adapters/common/*.sh adapters/claude-code/*.sh
# 2. merge settings.snippet.json into ~/.claude/settings.json
#    (absolute paths!), or run:
claude mcp add memory --transport http http://127.0.0.1:8765/mcp
```

`on-prompt.sh` captures explicit "记住…/remember …" signals AND injects recalled
memories via `hookSpecificOutput.additionalContext`; `pre-compact.sh` escrows the
latest user content before compaction.

## Kimi Code

Merge `kimi-code/config.snippet.toml` into `~/.kimi/config.toml`, or package the
repo as a plugin using `kimi-code/kimi-plugin.example.json`
(`.kimi-plugin/plugin.json` at repo root, then `/plugins install <repo>`).
First session: verify the injection actually lands — hook event names/payloads
follow the Kimi Code hooks documentation.

## Codex

Template only: plugins carrying hooks are excluded from the public directory and
must be trusted per-hook. Copy `codex/hooks.template.json`, point it at the
Claude Code scripts (identical payloads), merge `codex/config.snippet.toml`.
