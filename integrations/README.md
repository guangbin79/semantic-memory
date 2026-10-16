# Harness integrations

The HTTP API (`/recall` `/ingest` `/health`) makes every control-plane adapter a
thin "hook event → curl localhost:8765" client. Only opencode v2 needs an
in-process plugin (`integrations/opencode/memory.ts`); everyone else reuses the
shared scripts here.

| Harness | Support | Wiring |
|---|---|---|
| opencode v2 | ✅ | in-process plugin: lifecycle hooks + native `memory_store` / `memory_recall` tools (`opencode/`) |
| Claude Code | ✅ | hooks: `UserPromptSubmit` + `PreCompact` (`claude-code/`) |
| Kimi Code | ✅ | hooks: `[[hooks]]` or plugin manifest (`kimi-code/`) |
| Codex | ⚠️ | template only — hook plugins need per-hook user trust (`codex/`) |
| ZCode / DeepSeek Harness | ❌ | no lifecycle hooks available; MCP cannot deliver them (hooks are process/HTTP lifecycle events, not invocable tools) — revisit when those harnesses ship hooks |

## Shared scripts (`common/`)

- `recall.sh "query" [limit]` — prints a `<semantic-memory>` block on stdout
  (empty when no hits). Fail-open, 1.5s budget.
- `ingest.sh "content" [kind] [source] [project]` — stores one memory. Fail-open.
- Idle flush notification (`POST /idle`) is opencode-only (v2 `session.idle`
  event). Shell adapters have no idle counterpart — their queued entries (chat/
  compaction sources) flush via the batch threshold (pending ≥ 20) or
  `semantic-memory distill` manually.

Environment: `MEMORY_PLUGIN_URL` (default `http://127.0.0.1:8765`),
`MEMORY_PLUGIN_TOKEN`, `MEMORY_PLUGIN_LIMIT`, `MEMORY_PLUGIN_BUDGET_CHARS` —
same names as the opencode plugin.

## Claude Code

```bash
# 1. make scripts executable
chmod +x integrations/common/*.sh integrations/claude-code/*.sh
# 2. merge settings.snippet.json into ~/.claude/settings.json (absolute paths!)
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
Claude Code scripts (identical payloads).
