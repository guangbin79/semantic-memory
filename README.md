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
| `MEMORY_HOST` / `MEMORY_PORT` | `127.0.0.1` / `8765` | Bind address; port `0` = OS-assigned random port |
| `MEMORY_DISCOVERY_FILE` | `<db>.server.json` | Service-discovery file; `off` = disabled |
| `MEMORY_TOKEN` | empty | REST sidecar Bearer auth; **required** when binding non-loopback |
| `MEMORY_RRF_K` | `60` | RRF fusion constant |
| `MEMORY_TTL_ARTIFACT_DAYS` | `90` | TTL for `artifact` kind; `0` = never expire |
| `MEMORY_MAINTENANCE_ON_START` | `1` | Run the maintenance routine at service startup |
| `MEMORY_DEDUP_SIM` | `0.98` | Cosine threshold for vector near-duplicate ingest dedup |

Plugin:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_PLUGIN_URL` | `http://127.0.0.1:8765` | Service URL |
| `MEMORY_PLUGIN_TOKEN` | empty | Matches the service-side `MEMORY_TOKEN` |
| `MEMORY_PLUGIN_LIMIT` | `5` | Recalled items per injection |
| `MEMORY_PLUGIN_BUDGET_CHARS` | `2000` | Injection block character budget |
| `MEMORY_PLUGIN_TOOL_CAPTURE` | empty | Tool capture whitelist (comma-separated), e.g. `gis,bash` |
| `MEMORY_PLUGIN_AUTOSTART` | `0` | `=1` spawns the service if unreachable |

## Random port & agent bootstrap

An agent that spawns the service itself can avoid port hunting and collisions
entirely — `MEMORY_PORT=0` lets the OS assign a free port, and the real address
is published through two channels the moment the bind completes:

```bash
MEMORY_PORT=0 memory-mcp
# stdout (machine-readable, one line):
# MEMORY_MCP_LISTENING {"url":"http://127.0.0.1:41697","mcp_url":".../mcp","port":41697,"pid":...}
# and the same JSON in the discovery file: <db>.server.json (override: MEMORY_DISCOVERY_FILE)
```

So the bootstrap flow is: spawn → parse the stdout line **or** read the
discovery file → point `MEMORY_PLUGIN_URL` / MCP registration at `url`.
The file is retracted on graceful shutdown (SIGINT/SIGTERM); after a hard kill
readers can still detect staleness via the embedded `pid`. `stdio` mode has no
port and publishes nothing.

## Indexing: near-real-time, no rebuild windows

- **FTS (BM25)**: the FTS5 external-content tables are updated by triggers in the
  **same transaction** as the insert — a memory is keyword-searchable the moment
  `memory_store` returns.
- **Vectors**: the embedding is computed **synchronously inside the write request**
  and stored alongside the row. Retrieval is brute-force cosine over all vectors at
  query time — there is no ANN structure to build, lag, or ever rebuild.
- **Cold start**: the embedding model loads lazily on the first embed call
  (seconds, once per process). That first request blocks on it; nothing is dropped.
- **Capture is at-most-once**: hooks fire and forget (1.5s budget). Memories that
  arrive while the service is down are silently dropped — a blocked session is
  worse than a missed memory. Durability is best-effort, not exactly-once.
- **Vector gap**: entries written while the embedder is unconfigured stay
  BM25-searchable only; after installing the model, reconcile once with the
  `memory-mcp backfill` CLI (see Maintenance & lifecycle).

## Storage layout

Everything lives in **one SQLite file** (`MEMORY_DB_PATH`, default
`~/.local/share/memory-mcp/memory.db`):

| Object | Contents |
|---|---|
| `memories` | content + metadata (kind / source / project / tags / timestamps) |
| `memories_fts`, `memories_fts_tri` | FTS5 inverted indexes (porter / trigram), external-content — text is NOT duplicated |
| `memories_vec` | vector BLOBs, 1024-dim float32 = 4 KB per entry |

WAL sidecar files (`-wal`, `-shm`) exist at runtime; the GGUF model (~600 MB) is a
separate file. Deleting the db is a clean full reset — there is no other state.
Back up with `sqlite3 memory.db ".backup '/path/to/backup.db'"`. Unlike
ctx-semantic, the vector matrix is read from SQLite per query (the page cache
carries this to ~50k entries) — zero cache-invalidation logic.

## Sharing model

- **Cross-session**: the index is on disk; sessions are processes. Combined with
  the pre-compaction flush, even discarded context outlives its session.
- **Cross-harness**: opencode v2, Claude Code, Kimi Code and Codex all connect to
  the same service and therefore the **same index** — a memory stored by Claude
  Code is recallable from opencode. Switching harnesses loses nothing; there is
  no second copy to sync.
- **`project` is a filter, not a partition**: the index is global; the plugin
  filters by current project at injection time, but omitting `project` in
  `memory_recall` searches everything.
- **Single user, single machine**: loopback bind + optional Bearer token. No
  multi-tenancy.
- **Attribution gap**: `source` records the origin type (chat / compaction /
  tool) but not which harness wrote the entry — add a `harness:<name>` tag when
  that query need actually arises.

### Choosing the storage location

`MEMORY_DB_PATH` is read at service startup (parent directories are
auto-created). One db file per service instance — for hard db-level isolation
(e.g. a classified project), run multiple instances on different ports and point
`MEMORY_PLUGIN_URL` at the right one. Do NOT put the live db on NFS/SMB or
synced folders (WAL locking semantics); use backups for cross-machine movement.
Until a hard isolation need exists, prefer one instance + `project` filtering.

## HTTP read/manage API + built-in web UI

The same process also serves a **human-facing** channel (distinct from the
hook-facing `/recall` `/ingest`, same auth rule: Bearer when `MEMORY_TOKEN` is set):

| Endpoint | Purpose |
|---|---|
| `GET /` | Self-contained browse page (zero build, zero framework, offline-capable) |
| `GET /api/stats` | Totals, vectorized count, db size, embedder/device |
| `GET /api/memories?kind=&project=&limit=&offset=` | Paginated browse, newest-updated first |
| `DELETE /api/memories/{uid}` | Single-entry forget (same trust level as the CLI) |
| `POST /api/maintenance/purge` | Bulk maintenance (soft-delete purge + TTL + VACUUM) |
| `POST /api/maintenance/backfill` | Vector backfill for vectorless entries |

Bulk-maintenance endpoints sit behind a **double gate**: they return `403`
unless `MEMORY_TOKEN` is configured (on a default no-token loopback install the
HTTP surface is open to every local process — including the agent's shell — so
there the capability stays CLI-only), and `401` without a valid Bearer token.
The browse page exposes both as buttons (confirm → report → stats refresh).

Open `http://127.0.0.1:8765/` for stats overview, RRF search, kind/project
filtering, single-entry forgetting, and (on token-secured installs) one-click
maintenance. There is still no bulk editing; observability is what makes
"relevance is never lost" believable.

## Maintenance & lifecycle

Retention policy is **configuration, not LLM discretion**. Bulk destructive
operations are deliberately **not exposed as MCP tools** — they live where
prompt injection cannot reach: the startup hook and the CLI. The agent can only
`memory_forget` (single entry, user-initiated) and `memory_stats` (observe and
recommend).

- **Ingest dedup** (in the write path, nobody "calls" it): exact content-hash
  match (global) or vector near-duplicate (same project, cosine ≥
  `MEMORY_DEDUP_SIM`) refreshes the existing entry's timestamp instead of
  inserting — recurrence is treated as "proven relevant again". This caps the
  growth slope of compaction-flush sources, which love re-storing near-identical
  content.
- **Retention tiers**: `preference` / `decision` / `fact` are permanent;
  `artifact` (tool captures) expires after `MEMORY_TTL_ARTIFACT_DAYS`.
- **`memory-mcp purge`**: hard-deletes soft-deleted entries, applies the TTL,
  then `VACUUM` (only when something was actually removed). Same routine runs at
  every service start (`MEMORY_MAINTENANCE_ON_START=1`) — restart frequency is
  enough for day-granularity retention; no cron, no background threads.
- **`memory-mcp backfill`**: embeds every live entry that has no vector yet
  (one-shot reconcile after installing the model). Manual because it is heavy.

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
pytest            # 33 tests: store / dual-channel recall / soft delete / RRF math /
                  # CJK substring / query-prefix contract / fault tolerance /
                  # lifecycle dedup & TTL / migration / REST & /api routes /
                  # bulk double-gate auth / MCP tool layer /
                  # random-port serve & service discovery

cd opencode-plugin
npm install       # then:
npm run typecheck # tsc --strict against @opencode/plugin 2.0.28 types
npm test          # build + 16 mock-ctx hook assertions against a live service
                  # (stable line: inject/dedup/remember/compaction/tool-capture;
                  #  legacy line: v1 hook map, synthetic parts, compacting reset)
```

The plugin is additionally verified against the real opencode dev loader
(`0.0.0-dev-202610130450`): it loads the dual-form export, selects the legacy
line, and resolves the workspace as project — no load errors.

## Roadmap

- [x] Phase 1: MCP tools + hybrid retrieval + REST sidecar (current state)
- [x] Phase 2: opencode v2 plugin, four hook families (inject / pre-compaction flush
      / explicit remember / tool whitelist)
- [x] Phase 3a: multi-harness adapters (`adapters/` — Claude Code, Kimi Code, Codex template)
- [x] Phase 3b: lifecycle maintenance — ingest dedup, TTL/purge/VACUUM, vector
      backfill (startup hook + CLI, deliberately off the MCP tool surface)
- [ ] Phase 3c: semantic-level auto-capture for chat.message, bigram channel for
      2-char CJK terms (trigger: a real recall miss in BM25-only mode),
      cross-project memory graph, memory archive (cold file)

## Acknowledgments

Architecture inspired by [Basic Memory](https://github.com/basicmachines-co/basic-memory)
(Markdown + SQLite FTS), [ctx-semantic](https://github.com/guangbin79/ctx-semantic)
(BM25 + Qwen3-Embedding + RRF), and [context-mode](https://github.com/mksglu/context-mode)
(hook-interception paradigm).
