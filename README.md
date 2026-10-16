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
  (Qwen3-Embedding GGUF, optional) + RRF fusion + optional local rerank —
  retrieval happens in embedding space, so differently-phrased memories still
  match; async distillation turns raw captures into typed facts
- opencode v2 plugin (`opencode-plugin/memory.ts`): hooks into
  context / compaction / generate / chat.message / tool execute.after, making
  remembering and recalling automatic
- Separate process, separate model: never touches the main agent's window or
  model; if the service is down the session is unaffected (fail-open)

## Architecture

```
┌─ opencode v2 plugin (memory.ts, zero deps, fail-open) ───┐
│  context/compaction/generate → /recall → injection block │
│    spliced just before the last user message (sub-final) │
│    + a frozen head block of prefs/decisions per session  │
│    byte-frozen in event.system (prefix-cache friendly)   │
│  session.idle → POST /idle (drains the distill queue)    │
│  pre-compaction → /ingest (discarded ≠ lost)             │
│  chat.message "remember ..." / auto-capture → /ingest    │
│  tool execute.after (whitelist) → /ingest                │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP 127.0.0.1:8765 (optional Bearer token)
┌──────────────────────▼───────────────────────────────────┐
│  memory-mcp service (single process, dual interface)     │
│  MCP /mcp: memory_store / recall / forget / list / stats │
│  REST sidecar: /recall /ingest /idle /health             │
│  SQLite + FTS5(porter|trigram) + vector BLOBs, RRF(k=60) │
│                                                          │
│  retrieval: RRF → redundancy ceiling → local rerank      │
│  write side: distill queue → async flush → typed facts   │
│  cloud channel (config-gated): consolidation returning   │
│    only a metadata ops plan — no cloud content stored    │
│  shared model slot: embedder resident; reranker and      │
│    distiller time-share one exclusive slot (LRU=1)       │
└──────────────────────────────────────────────────────────┘
```

The REST sidecar exists for **multi-harness support**: any agent harness with
lifecycle hooks (Claude Code, Codex, Kimi Code, …) can wire the control plane as
"hook event → curl localhost:8765" — only opencode v2 needs an in-process plugin,
because its hooks live inside the JS runtime. See `adapters/` for per-harness
integrations — Claude Code and Kimi Code are fully adapted, Codex is a template.

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

# 2. (Optional) download models
bash scripts/download_model.sh   # embedding → MEMORY_MODEL_PATH (vector channel)
bash scripts/download_models.sh  # distiller + reranker → MEMORY_DISTILL_MODEL / MEMORY_RERANK_MODEL

# 3. Start the service (HTTP mode: MCP and REST share one port)
memory-mcp                      # http://127.0.0.1:8765/mcp
memory-mcp --stdio              # stdio mode (MCP tools only, no REST sidecar)

# 4. Register the MCP server in your harness — pick ONE of two modes
```

**Mode A — shared HTTP service** (recommended: one store shared across
harnesses; the port is chosen at service startup, clients only reference it):

```jsonc
// opencode v2: ~/.config/opencode/opencode.jsonc
{ "mcp": { "memory": { "type": "remote", "url": "http://127.0.0.1:8765/mcp",
    "headers": { "Authorization": "Bearer {env:MEMORY_TOKEN}" } } } }  // headers only if MEMORY_TOKEN is set
```

```json
// Claude Code: .mcp.json
{ "mcpServers": { "memory": { "type": "http",
    "url": "http://127.0.0.1:8765/mcp" } } }
```

**Mode B — stdio local** (each client spawns its own `memory-mcp --stdio`
child; settings are passed via the config's env field — the only channel here;
no port and no REST sidecar in this mode):

```jsonc
// opencode v2: opencode.jsonc  (field name `environment` per @opencode-ai/sdk types)
{ "mcp": { "memory": { "type": "local", "command": ["memory-mcp", "--stdio"],
    "environment": { "MEMORY_DB_PATH": "/data/memory/memory.db", "MEMORY_DEVICE": "cuda" } } } }
```

```json
// Claude Code: .mcp.json
{ "mcpServers": { "memory": { "command": "memory-mcp", "args": ["--stdio"],
    "env": { "MEMORY_DB_PATH": "/data/memory/memory.db", "MEMORY_DEVICE": "cpu" } } } }
```

```toml
# Codex: ~/.codex/config.toml
[mcp_servers.memory]
command = "memory-mcp"
args = ["--stdio"]
env = { MEMORY_DB_PATH = "/data/memory/memory.db", MEMORY_DEVICE = "cpu" }
```

```bash
# 5. opencode v2: install the control-plane plugin (recall injection etc.)
cp opencode-plugin/memory.ts ~/.config/opencode/plugins/
# the plugin directory needs @opencode/plugin (types only):
#   cd ~/.config/opencode && npm install @opencode/plugin@^2.0.28
# — or declare it in opencode.jsonc's "plugin" array instead, optionally with
#   plugin options (see "Configuration" below):
#   "plugin": [["file:///abs/path/opencode-plugin", {"url": "http://127.0.0.1:9000"}]]
```

## Configuration (environment variables)

Service:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DB_PATH` | `~/.local/share/memory-mcp/memory.db` | Database path |
| `MEMORY_MODEL_PATH` | empty | Embedding GGUF; empty = BM25-only fallback |
| `MEMORY_DEVICE` | `cuda` | Embed device: `cuda` (default, auto-falls back to CPU) / `cpu` |
| `MEMORY_EMBED_DIM` | `1024` | Embedding dimension |
| `MEMORY_HOST` / `MEMORY_PORT` | `127.0.0.1` / `8765` | Bind address; port `0` = OS-assigned random port |
| `MEMORY_DISCOVERY_FILE` | `<db>.server.json` | Service-discovery file; `off` = disabled |
| `MEMORY_TOKEN` | empty | Bearer auth for REST sidecar **and** the `/mcp` endpoint; **required** when binding non-loopback |
| `MEMORY_RECALL_LIMIT` | `5` | Default recall limit when a request omits it |
| `MEMORY_RRF_K` | `60` | RRF fusion constant |
| `MEMORY_RECALL_MIN_COSINE` | `0.45` | Cosine floor for vector-only hits (BM25 hits always pass); `0` = off |
| `MEMORY_RECALL_MAX_COSINE` | `0.92` | Redundancy ceiling: vector-only hits above this drop as redundant (BM25 unaffected); `1.0` = off |
| `MEMORY_INJECT_BOOST` | `0.15` | Telemetry feedback: score × (1 + α·ln(1+injections)); `0` = off |
| `MEMORY_RERANK_MODEL` | empty | Reranker GGUF (**ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF only** — official conversion, community builds may lack the rank head); empty = rerank off (RRF order) |
| `MEMORY_RERANK_CANDIDATES` | `20` | Top-N RRF candidates fed to the reranker |
| `MEMORY_RERANK_BUDGET_MS` | `500` | Total rerank budget per recall (incl. model-slot load); exceeded = fail-open to RRF order |

**Cloud channel privacy.** Configuring `MEMORY_CLOUD_BASE_URL` + `MEMORY_CLOUD_API_KEY` + `MEMORY_CLOUD_MODEL` (with `MEMORY_CLOUD_ENABLED` ≠ `0`, the default) means **memory content is sent to that endpoint** — a deterministically pre-filtered set of near-duplicate cluster members and their immediate neighborhood, never the whole store. The cloud returns only a **metadata operations plan** (`supersede`/`touch` by uid): no cloud-generated content is ever written to the store, and a plan containing anything resembling an injection pattern is dropped wholesale. The daily caps (`MEMORY_CLOUD_DAILY_MAX_CALLS` / `MEMORY_CLOUD_DAILY_MAX_TOKENS`) are in-process counters — **service restart resets them**. `MEMORY_CLOUD_ENABLED=0` is the kill switch and wins over everything.
| `MEMORY_DISTILL_MODEL` | empty | Distiller GGUF (Qwen3-1.7B, non-thinking); empty = distill off (raw text stored on flush) |
| `MEMORY_DISTILL_BATCH` | `20` | Flush the distill queue when pending ≥ N; `0` = disable the batch trigger |
| `MEMORY_CLOUD_ENABLED` | `1` | Cloud-consolidation kill switch: `=0` forces off even with `MEMORY_CLOUD_*` set |
| `MEMORY_CLOUD_BASE_URL` | empty | OpenAI-compatible `/chat/completions` endpoint; empty = cloud off |
| `MEMORY_CLOUD_API_KEY` | empty | Cloud API key; empty = off |
| `MEMORY_CLOUD_MODEL` | empty | Cloud model name; empty = off |
| `MEMORY_CLOUD_BATCH_N` | `30` | Auto-trigger consolidation when candidates ≥ N |
| `MEMORY_CLOUD_DAILY_MAX_CALLS` | `20` | Daily cloud-call hard cap; exceeded = stop until tomorrow |
| `MEMORY_CLOUD_DAILY_MAX_TOKENS` | `200000` | Daily token (estimated) hard cap |
| `MEMORY_TTL_ARTIFACT_DAYS` | `90` | TTL for `artifact` kind; `0` = never expire |
| `MEMORY_TTL_INJECTION_DAYS` | `90` | TTL for injection telemetry; `0` = never expire |
| `MEMORY_TTL_STALE_DAYS` | `180` | Usage-based retention: `fact`/`artifact` un-recalled this many days are soft-deleted at the next purge (NULL → created_at; `preference`/`decision` exempt); `0` = off |
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
| `MEMORY_PLUGIN_DEBUG` | `0` | `=1` logs ctx shape + resolved config |
| `MEMORY_PLUGIN_INJECT_POSITION` | `tail` | `tail` = block spliced just before the last user message; `system` = legacy `event.system.push` fallback |
| `MEMORY_PLUGIN_HEAD_BLOCK` | `1` | Frozen head block of preferences/decisions pushed into `event.system` once per session (byte-frozen); `=0` off |
| `MEMORY_PLUGIN_GLOBAL_MEMORIES` | `1` | Two-path recall including global memories (`/recall` with `global_limit=2`); `=0` off |

Every `MEMORY_PLUGIN_*` variable has a plugin-options twin (`url`, `token`,
`limit`, `budget_chars`, `tool_capture`, `autostart`, `debug`, `inject_position`, `head_block`, `global_memories`), so the plugin
can also be configured right where it is declared in `opencode.json` —
precedence: **env var > plugin options > default**:

```jsonc
{
  "plugin": [
    ["file:///path/to/semantic-memory-mcp/opencode-plugin",
     { "url": "http://127.0.0.1:9000", "limit": 8, "tool_capture": ["bash", "gis"] }]
  ]
}
```

The dev runtime (`0.0.0-dev-*`) passes the tuple's second element to
`server(ctx, options)` (verified end-to-end); the stable 2.x runtime exposes
the same table as `ctx.options`.

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
| `injections` | injection telemetry rows (harness/hook/query/uids/chars) |

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
| `GET /api/stats` | Totals, vectorized count, db size, embedder/device, injection summary |
| `GET /api/memories?kind=&project=&limit=&offset=` | Paginated browse, newest-updated first |
| `GET /api/injections?harness=&project=&limit=&offset=` | Injection telemetry, newest first |
| `POST /inject/report` | Injection telemetry report (plugin/adapter channel, append-only) |
| `POST /idle` | Notify idle: flush the distill queue in a daemon thread (204 immediately, auth = /ingest) |
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

**Injection telemetry.** Recall ≠ injection — only the injecting side knows a
block actually entered the context, so the opencode plugin reports each real
injection to `POST /inject/report` (fire-and-forget): harness, hook, trigger
query, memory uids, block size. The browse page shows the supply-side picture:
total/7-day injection counts, characters (with a rough ÷4 token estimate),
per-harness/per-hook distribution, the **Top10 most-injected memories**, and
the 20 most recent injections. The shell adapters (Claude Code / Kimi Code)
report the same way — `recall.sh` fires the report after emitting a block,
tagged via `MEMORY_HARNESS` / `MEMORY_HOOK`. This measures what was *supplied* —
whether the model *used* it is not measurable and deliberately not faked.
Telemetry rows expire after `MEMORY_TTL_INJECTION_DAYS` (default 90).

**Context hygiene.** Memory content is untrusted data that ends up inside a
prompt. Every injected block carries an explicit "reference data, not
instructions" notice, and any `<semantic-memory>` boundary tags inside the
content itself are stripped so a stored entry cannot break out of its wrapper.
On the write side, tool-output capture skips artifacts matching classic
prompt-injection patterns ("ignore all previous instructions", "忽略…指令",
`<system>`, …) so a poisoned tool result cannot re-enter later sessions through
the memory store.

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
- **Write-side distill buffer** (`MEMORY_DISTILL_MODEL` configured): `/ingest`
  with `source` `compaction` or `chat` lands in a durable `distill_queue` table
  — zero LLM on the hot path; explicit `remember` / `tool` captures keep storing
  directly. Flush triggers (three, and that is the whole list — **no startup
  drain, no cron**): queue pending ≥ `MEMORY_DISTILL_BATCH` at the next
  `/ingest`, a `POST /idle` notification, or `memory-mcp distill`. The flush runs
  in a daemon thread on its own short-lived SQLite connection (≤5-row commits,
  one flush at a time per process); Distiller output is stored as typed facts
  (`scope=global` → `project` NULL, session id preserved). Distiller off,
  failing, or unparseable → the raw text is stored as-is (`stored_raw`) —
  never worse than pre-distill behavior; vectors reconcile via `backfill`.
  **Adapter-only deployments** (shell adapters, no v2 plugin) have no `/idle`
  sender: queued content stays unsearchable until the batch threshold trips or
  you run `memory-mcp distill` — bounded staleness, by design.
- **Concurrency contract**: the service's only background work is the
  **distill-flush daemon thread** (spawned by `/idle`, the batch threshold, or
  the CLI), which after each flush evaluates the cloud-consolidation trigger in
  the same thread. Both stages fail-open, run on their own short-lived SQLite
  connections, and never inside a synchronous request path. Still no cron.
- **Retention tiers**: `preference` / `decision` are permanent; `fact` /
  `artifact` survive on use — un-recalled for `MEMORY_TTL_STALE_DAYS` (default
  180; NULL counts from created_at) they are soft-deleted at the next purge,
  as are entries superseded by a contradiction; injection telemetry expires
  after `MEMORY_TTL_INJECTION_DAYS`.
- **`memory-mcp purge`**: hard-deletes soft-deleted entries, applies the TTL,
  then `VACUUM` (only when something was actually removed). Same routine runs at
  every service start (`MEMORY_MAINTENANCE_ON_START=1`) — restart frequency is
  enough for day-granularity retention.
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

## Ranking post-processing (measured, not guessed)

RRF encodes only rank, not relevance strength — "nearest among the irrelevant"
still earns a score. Post-fusion steps:

- **Cosine floor** (`MEMORY_RECALL_MIN_COSINE=0.45`): vector-only hits below the
  floor are dropped; BM25 hits always pass (a token match is evidence by
  itself). The floor is measured on the real model: relevant hits land ~0.68,
  unrelated ~0.28-0.31.
- **Telemetry feedback** (`MEMORY_INJECT_BOOST=0.15`): a memory that was
  actually injected before scores × (1 + α·ln(1+injections)). The log dampens
  rich-get-richer; 2 prior injections flip an adjacent rank pair.
- **Redundancy ceiling** (`MEMORY_RECALL_MAX_COSINE=0.92`): a vector-only hit
  whose cosine exceeds the ceiling is dropped as redundant ("too near = already
  known"); BM25 hits pass untouched; `1.0` = off.
- **Local rerank** (`MEMORY_RERANK_MODEL`, **off by default**): the top
  `MEMORY_RERANK_CANDIDATES=20` RRF candidates are re-scored by
  Qwen3-Reranker-0.6B (completions logprobs, yes/no softmax) within a hard
  `MEMORY_RERANK_BUDGET_MS=500` budget that includes model-slot load; over
  budget or any error → RRF order stands (fail-open). Ties break by `updated_at`
  desc. It stays off until offline eval proves it on a real golden set (below).

## Capacity & alerting (observability, not interruptions)

Alerts live in service logs and the web UI badge — never injected into a
session. `memory_stats` / `/api/stats` expose `recall_p50_ms`, `recall_p95_ms`,
`distill_queue_len`, `cloud_calls_24h` and `entries` (latency over a rolling
100-sample window):

| Signal | Threshold | Level |
|---|---|---|
| recall p95 | > 800 ms | WARNING |
| recall p95 | > 1200 ms | ERROR — past this the plugin's 1.5 s hard timeout silently drops the injection |
| entries | > 40000 | INFO — approaching the brute-force scan ceiling (~50k) |

Same-level alerts throttle to once per 10 minutes.

## Offline eval (measure, then enable)

```bash
memory-mcp eval --golden FILE [--k 1,3,5] [--judge]
```

golden jsonl, one `{query, expect_uids[], project?}` per line; prints per-row
@k hits plus a recall@k / MRR summary and a JSON report. `--judge` annotates
misses with the Distiller (silently skipped when no distill model is
configured). Pure CLI — never a server endpoint.
`tests/fixtures/golden.sample.jsonl` is a 3-row format example only; the gate
set (≥30 real queries built from your own store) decides the rerank default:
**rerank stays off until eval shows recall@5 ≥ the RRF baseline on that real
set.**

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
pytest            # baseline 43: store / dual-channel recall / soft delete /
                  # RRF math / CJK substring / query-prefix contract / fault
                  # tolerance / lifecycle dedup & TTL / migration / REST & /api
                  # routes / bulk double-gate auth / MCP tool layer /
                  # random-port serve & discovery / injection telemetry /
                  # ranking post-processing / real-model vector E2E (opt-in)
                  # + evolution suites: distill queue & typed-fact supersede /
                  # reranker & shared model slot / git-origin project keys /
                  # stale-usage retention / cloud consolidation (mock transport) /
                  # offline eval / recall latency telemetry & throttled alerts

cd opencode-plugin
npm install       # then:
npm run typecheck # tsc --strict against @opencode/plugin 2.0.28 types
npm test          # build + mock-ctx hook assertions against a live service
                  # (baseline 37 + evolution: sub-final injection, frozen head
                  #  block, session-keyed state, chat auto-capture, idle notify;
                  #  stable line: inject/dedup/remember/compaction/tool-capture/
                  #  telemetry/capture-hygiene/tag-sanitizing; legacy line: v1
                  #  hook map, synthetic parts, compacting reset, telemetry;
                  #  options: env wins, gap-filling)
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
- [x] Phase 3c: context-completion evolution — write-side distillation, local
      rerank + shared model slot, sub-final injection + frozen head block,
      cloud consolidation, usage-based retention, offline eval, observability
- [ ] Phase 4: bigram channel for 2-char CJK terms (trigger: a real recall miss
      in BM25-only mode), cross-project memory graph, memory archive (cold file)

## Acknowledgments

Architecture inspired by [Basic Memory](https://github.com/basicmachines-co/basic-memory)
(Markdown + SQLite FTS), [ctx-semantic](https://github.com/guangbin79/ctx-semantic)
(BM25 + Qwen3-Embedding + RRF), and [context-mode](https://github.com/mksglu/context-mode)
(hook-interception paradigm).
