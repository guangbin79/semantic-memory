# semantic-memory-mcp

**[English](README.md) | [中文](README.zh-CN.md)**

A **sidecar** for coding-agent context management: **model-semantic-level** context
retrieval that keeps context relevance alive.

**Its goal is not saving tokens or compressing context.** opencode v2's native
compaction and DCP manage context *quantity* (compress, prune, discard to fit the
window); this component manages context *relevance* — everything that gets
compressed away, pruned, or left behind in a previous session lands in a
semantically searchable store, and is automatically recalled and injected when it
becomes relevant to the **current task**. It spends a small, budgeted amount of
tokens (the injection block) to buy back relevance that would otherwise be lost
forever. Compaction makes the window fit; the sidecar makes compression safe.

- MCP server (Python FastMCP): BM25 (dual-channel FTS5: porter + trigram) + vectors
  (Qwen3-Embedding-0.6B GGUF, optional) + RRF fusion — retrieval happens in
  embedding space, so differently-phrased memories still match
- opencode v2 plugin (`opencode-plugin/memory.ts`): hooks into
  context / compaction / generate / chat.message / tool execute.after, making
  remembering and recalling automatic
- Separate process, separate model: never touches the main agent's window or
  model; if the service is down the session is unaffected (fail-open)

## Architecture

```
┌─ opencode v2 plugin (memory.ts, zero deps, fail-open) ───┐
│  context/compaction/generate → /recall → tail injection  │
│  pre-compaction → /ingest (discarded ≠ lost)             │
│  chat.message "remember ..." → /ingest                   │
│  tool execute.after (whitelist) → /ingest                │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP 127.0.0.1:8765 (optional Bearer token)
┌──────────────────────▼───────────────────────────────────┐
│  memory-mcp service (single process, dual interface)     │
│  MCP /mcp: memory_store / recall / forget / list / stats │
│  REST sidecar: /recall /ingest /health (for plugins)     │
│  SQLite + FTS5(porter|trigram) + vector BLOBs, RRF(k=60) │
└──────────────────────────────────────────────────────────┘
```

The REST sidecar exists for **multi-harness support**: any agent harness with
lifecycle hooks (Claude Code, Codex, Kimi Code, …) can wire the control plane as
"hook event → curl localhost:8765" — only opencode v2 needs an in-process plugin,
because its hooks live inside the JS runtime. See `adapters/` for per-harness
integrations (planned).

Design constraints: the plugin holds zero intelligence (retrieval, embedding and
storage all live in the service); a dead service means silent skips, never a
blocked session; no re-injection when the recalled set is unchanged (protects the
prefix cache); the MCP server is fully usable without the plugin. The metric that
matters is not "tokens saved" but **recall@k of task-relevant context**.

## Quick start

```bash
# 1. Install the service (uv or venv recommended)
pip install -e .                # BM25-only mode works out of the box
pip install -e ".[embed]"       # add the vector channel (builds llama-cpp-python)

# 2. (Optional) download the embedding model to enable the vector channel
bash scripts/download_model.sh  # prints the MEMORY_MODEL_PATH export line

# 3. Start the service (HTTP mode: MCP and REST share one port)
memory-mcp                      # http://127.0.0.1:8765/mcp
memory-mcp --stdio              # stdio mode (MCP tools only, no REST sidecar)

# 4. Register the MCP server in opencode v2 (~/.config/opencode/opencode.jsonc)
{
  "mcp": { "servers": {
    "memory": { "type": "remote", "url": "http://127.0.0.1:8765/mcp" }
  } }
}

# 5. Install the control-plane plugin
cp opencode-plugin/memory.ts ~/.config/opencode/plugins/
# the plugin directory needs @opencode/plugin (types only):
#   cd ~/.config/opencode && npm install @opencode/plugin@beta
```

## Configuration (environment variables)

Service:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DB_PATH` | `~/.local/share/memory-mcp/memory.db` | Database path |
| `MEMORY_MODEL_PATH` | empty | GGUF model path; empty = BM25-only fallback |
| `MEMORY_DEVICE` | `cuda` | Embed device: `cuda` (default, auto-falls back to CPU) / `cpu` |
| `MEMORY_HOST` / `MEMORY_PORT` | `127.0.0.1` / `8765` | Bind address |
| `MEMORY_TOKEN` | empty | REST sidecar Bearer auth; **required** when binding non-loopback |
| `MEMORY_RRF_K` | `60` | RRF fusion constant |

Plugin:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_PLUGIN_URL` | `http://127.0.0.1:8765` | Service URL |
| `MEMORY_PLUGIN_TOKEN` | empty | Matches the service-side `MEMORY_TOKEN` |
| `MEMORY_PLUGIN_LIMIT` | `5` | Recalled items per injection |
| `MEMORY_PLUGIN_BUDGET_CHARS` | `2000` | Injection block character budget |
| `MEMORY_PLUGIN_TOOL_CAPTURE` | empty | Tool capture whitelist (comma-separated), e.g. `gis,bash` |
| `MEMORY_PLUGIN_AUTOSTART` | `0` | `=1` spawns the service if unreachable |

## Vector channel (same vectors as ctx-semantic)

The embedding layer mirrors [ctx-semantic](https://github.com/guangbin79/ctx-semantic)
item by item — same model file, same llama-cpp-python pin (0.3.35), same output:

- Model: Qwen3-Embedding-0.6B **Q8_0** GGUF, dim 1024, pooling **LAST** (explicit)
- Asymmetric embedding: **queries** get an `Instruct: ...\nQuery: {text}` prefix,
  **documents** are embedded raw
- n_ctx=8192, sub-batches of 32, float32 rows L2-normalized (1e-12 guard)
- Vector search: brute-force cosine (sub-millisecond up to ~50k entries, no ANN needed)

**Device policy (deliberately inverted vs. ctx-semantic)**: `MEMORY_DEVICE=cuda` is the
**default**, `cpu` is opt-in. The GPU path falls back deterministically: load → probe
embed → **nvidia-smi PID verification** (detects silent CPU fallback) → any failure
rebuilds on CPU with a log line; a runtime GPU embed failure (e.g. OOM) rebuilds on
CPU and retries once. On machines without NVIDIA, set `MEMORY_DEVICE=cpu` to skip the
GPU attempt. `memory_stats` / `/health` report the actual device in use.

## Multilingual full-text search

- The **porter** channel (unicode61 + English stemming) indexes every script with
  word boundaries: English, French, German, Spanish, Russian, Greek, Arabic, …
- The **trigram** channel (≥3-char substrings) covers scripts without word
  boundaries: Chinese, Japanese, Korean, Thai, Lao, Khmer, Burmese — and provides
  substring fallback for any language (even German compounds).

Known gap: 2-character CJK words (e.g. 坐标, 投影) fall below the trigram minimum.
The vector channel backstops this (Qwen3-Embedding is multilingual, 100+ languages);
if BM25-only mode ever shows a real miss, the fix is an application-level bigram
channel (~15 lines, zero dependencies) — see the roadmap.

## Testing

```bash
pip install -e ".[dev]"
pytest            # 8 tests: store / dual-channel recall / soft delete / RRF math /
                  # CJK substring / query-prefix contract / fault tolerance
```

## Roadmap

- [x] Phase 1: MCP tools + hybrid retrieval + REST sidecar (current state)
- [x] Phase 2: opencode v2 plugin, four hook families (inject / pre-compaction flush
      / explicit remember / tool whitelist)
- [ ] Phase 3: Claude Code & Kimi Code adapters (`adapters/`), semantic-level
      auto-capture for chat.message, bigram channel for 2-char CJK terms (trigger:
      a real recall miss in BM25-only mode), cross-project memory graph

## Acknowledgments

Architecture inspired by [Basic Memory](https://github.com/basicmachines-co/basic-memory)
(Markdown + SQLite FTS), [ctx-semantic](https://github.com/guangbin79/ctx-semantic)
(BM25 + Qwen3-Embedding + RRF), and [context-mode](https://github.com/mksglu/context-mode)
(hook-interception paradigm).
