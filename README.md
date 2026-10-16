> Repository rename pending: **semantic-memory-mcp → semantic-memory** — rename it in GitHub Settings, then run `git remote set-url origin git@github.com:guangbin79/semantic-memory.git`; old links redirect automatically.

# semantic-memory

**[English](README.md) | [中文](README.zh-CN.md)**

> **Renamed** — this project was previously published as `semantic-memory-mcp`;
> data from the old default home `~/.local/share/memory-mcp/` migrates automatically
> on first start (see "The unified home" below).

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

- Self-contained HTTP service (pure Python stdlib, zero third-party dependencies):
  BM25 (dual-channel FTS5: porter + trigram) + vectors (Qwen3-Embedding GGUF,
  optional) + RRF fusion + optional local rerank — retrieval happens in embedding
  space, so differently-phrased memories still match; async distillation turns raw
  captures into typed facts
- opencode v2 plugin (`integrations/opencode/memory.ts`): hooks into
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
│  semantic-memory service (single process, stdlib http)   │
│  REST API: /recall /ingest /idle /health + web UI        │
│  opencode native tools: memory_store / memory_recall     │
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

The HTTP API is designed for **multi-harness support**: any agent harness with
lifecycle hooks (Claude Code, Codex, Kimi Code, …) can wire the control plane as
"hook event → curl localhost:8765" — only opencode v2 needs an in-process plugin,
because its hooks live inside the JS runtime. See `integrations/` for per-harness
wiring and the support matrix in Quick start.

Design constraints: the plugin holds zero intelligence (retrieval, embedding and
storage all live in the service); a dead service means silent skips, never a
blocked session; no re-injection when the recalled set is unchanged (protects the
prefix cache); the service runs standalone — HTTP API and web UI work with no
plugin attached. The metric that matters is not "tokens saved" but **recall@k of
task-relevant context**.

## Quick start

### Zero-ceremony install (pick one tier)

| Tier | Command | What you get |
|---|---|---|
| 0 — source run, zero install | `PYTHONPATH=src python3 -m semantic_memory serve` | works immediately on a bare python3 ≥ 3.11, zero third-party dependencies (BM25-only; numpy absent → vector leg off with a self-explaining install hint) |
| 1 — pip user install | `pip install --user .` | the `semantic-memory` console script, still zero dependencies |
| 2 — vector / full stack | `pip install ".[vector]"` or `pip install ".[embed]"` | +numpy (vector leg) / +llama-cpp-python (local GGUF inference) |

Models for tier 2 are optional per channel: `bash scripts/download_model.sh`
(embedding → `MEMORY_MODEL_PATH`) and `bash scripts/download_models.sh` (distiller +
reranker → `MEMORY_DISTILL_MODEL` / `MEMORY_RERANK_MODEL`) — or skip the manual steps
entirely and use the one-shot setup below.

### One-shot setup (probe → verified download → config → smoke)

Humans, one curl (pinned to release tag v0.2.0):

```bash
curl -fsSL \
  https://raw.githubusercontent.com/guangbin79/semantic-memory-mcp/v0.2.0/scripts/setup.sh \
  | bash -s -- --profile full --json --yes
```

Agents with the repo checked out:

```bash
bash scripts/setup.sh --profile full --json --yes
```

Both build a fully self-contained `~/semantic-memory-mcp/` (venv + sha256-verified
model downloads + `config.toml` + smoke test). `--profile` picks the tier:
`full` (embed + distill + rerank, default) / `embed` / `bm25` (zero download). With
`--json` the report is one machine-readable line; afterwards everything runs via
`~/semantic-memory-mcp/venv/bin/semantic-memory …`.

### Start the service

```bash
semantic-memory serve    # or: ~/semantic-memory-mcp/venv/bin/semantic-memory serve
# → http://127.0.0.1:8765  (REST API + web UI; MEMORY_PORT=0 = OS-assigned port)
```

### Wire your harness (lifecycle hooks, not tool protocols)

| Harness | Support | Wiring |
|---|---|---|
| opencode v2 | ✅ | in-process plugin — hooks **and** native `memory_store` / `memory_recall` tools (`integrations/opencode/`) |
| Claude Code | ✅ | hooks: `UserPromptSubmit` + `PreCompact` (`integrations/claude-code/`) |
| Kimi Code | ✅ | hooks: `[[hooks]]` or plugin manifest (`integrations/kimi-code/`) |
| Codex | ⚠️ | template only — hook plugins need per-hook user trust (`integrations/codex/`) |
| ZCode / DeepSeek Harness | ❌ | no lifecycle hooks available; MCP cannot deliver them (hooks are process/HTTP lifecycle events, not invocable tools) — revisit when those harnesses ship hooks |

opencode v2 (the only in-process integration):

```bash
cp integrations/opencode/memory.ts ~/.config/opencode/plugins/
# the plugin directory needs @opencode/plugin (types only):
#   cd ~/.config/opencode && npm install @opencode/plugin@^2.0.28
# — or declare it in opencode.jsonc's "plugin" array instead, optionally with
#   plugin options (see "Configuration" below):
#   "plugin": [["file:///abs/path/to/semantic-memory/integrations/opencode", {"url": "http://127.0.0.1:9000"}]]
```

Every other harness is a thin "hook event → curl localhost:8765" shell adapter —
see `integrations/README.md`.

### The unified home: `~/semantic-memory-mcp/`

| Path | Contents |
|---|---|
| `~/semantic-memory-mcp/venv/` | self-contained virtualenv (one-shot setup) |
| `~/semantic-memory-mcp/models/` | GGUF models (embedding / distiller / reranker) |
| `~/semantic-memory-mcp/data/memory.db` | the SQLite store (default `MEMORY_DB_PATH`) |
| `~/semantic-memory-mcp/config.toml` | file-config tier: env > this file > built-in defaults |

Resolution order: `MEMORY_DATA_DIR` > explicit `XDG_DATA_HOME` (→
`$XDG_DATA_HOME/memory-mcp/`) > `~/semantic-memory-mcp/`. **Legacy migration is
automatic**: on first start, a pre-rename `~/.local/share/memory-mcp/` home is
moved in — WAL-checkpointed first, sha256-verified per file, refused while a live
old instance holds it, and the old directory is left with a single `MIGRATED`
pointer file. Idempotent; nothing is ever deleted unverified.

### Tiered strict startup

The service refuses to start half-configured (fail loudly instead of silently
degraded):

- **Nothing configured, no `config.toml`** → recruit mode: BM25-only runs legally,
  with an install hint pointing at `semantic-memory setup`.
- **Configured (env or `config.toml`) but the model file is missing** → refused with
  `exit 2`, the offending path, and the setup command that fixes it.
- **`config.toml` present** → strict per the profile setup wrote.
- **`MEMORY_ALLOW_BM25_ONLY=1`** (or `allow_bm25_only = true` in the file) → always
  allowed; the explicit BM25-only escape hatch.

## Configuration (environment variables)

Service:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DATA_DIR` | `~/semantic-memory-mcp/` (resolution: this env > explicit `XDG_DATA_HOME` → `$XDG_DATA_HOME/memory-mcp/` > default home) | Data home override: models/ data/ config.toml all live under it |
| `MEMORY_DB_PATH` | `~/semantic-memory-mcp/data/memory.db` | Database path (only overrides the db; the rest of the home stays put) |
| `MEMORY_MODEL_PATH` | empty | Embedding GGUF; empty = BM25-only fallback |
| `MEMORY_DEVICE` | `cuda` | Embed device: `cuda` (default, auto-falls back to CPU) / `cpu` |
| `MEMORY_EMBED_DIM` | `1024` | Embedding dimension |
| `MEMORY_HOST` / `MEMORY_PORT` | `127.0.0.1` / `8765` | Bind address; port `0` = OS-assigned random port |
| `MEMORY_DISCOVERY_FILE` | `<db>.server.json` | Service-discovery file; `off` = disabled |
| `MEMORY_TOKEN` | empty | Bearer auth for the HTTP API; **required** when binding non-loopback |
| `MEMORY_RECALL_LIMIT` | `5` | Default recall limit when a request omits it |
| `MEMORY_RRF_K` | `60` | RRF fusion constant |
| `MEMORY_RECALL_MIN_COSINE` | `0.45` | Cosine floor for vector-only hits (BM25 hits always pass); `0` = off |
| `MEMORY_RECALL_MAX_COSINE` | `0.92` | Redundancy ceiling: vector-only hits above this drop as redundant (BM25 unaffected); `1.0` = off |
| `MEMORY_INJECT_BOOST` | `0.15` | Telemetry feedback: score × (1 + α·ln(1+injections)); `0` = off |
| `MEMORY_RERANK_MODEL` | `~/semantic-memory-mcp/models/qwen3-reranker-0.6b-q8_0.gguf` | Reranker GGUF (**ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF only** — official conversion, community builds may lack the rank head); default = conventional path (gate PASSED 2026-10-16: recall@5 0.939 vs 0.909 baseline) — on when the file exists, graceful off + warning when missing; set empty to disable |
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
| `MEMORY_ALLOW_BM25_ONLY` | `0` | Tier-0 escape hatch: `=1` always allows BM25-only startup even when `config.toml` is present |

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
    ["file:///path/to/semantic-memory/integrations/opencode",
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
MEMORY_PORT=0 semantic-memory
# stdout (machine-readable, one line):
# MEMORY_LISTENING {"url":"http://127.0.0.1:41697","port":41697,"pid":...}
# and the same JSON in the discovery file: <db>.server.json (override: MEMORY_DISCOVERY_FILE)
```

So the bootstrap flow is: spawn → parse the stdout line **or** read the
discovery file → point `MEMORY_PLUGIN_URL` at `url`.
The file is retracted on graceful shutdown (SIGINT/SIGTERM); after a hard kill
readers can still detect staleness via the embedded `pid`.

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
  `semantic-memory backfill` CLI (see Maintenance & lifecycle).

## Storage layout

Everything lives in **one SQLite file** (`MEMORY_DB_PATH`, default
`~/semantic-memory-mcp/data/memory.db`):

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
operations are deliberately **not exposed to the agent** — they live where
prompt injection cannot reach: the startup hook and the CLI. The agent surface is
exactly `memory_store` / `memory_recall` (opencode native tools, fail-open);
single-entry forget lives in the web UI / `DELETE /api/memories/{uid}`.

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
  `/ingest`, a `POST /idle` notification, or `semantic-memory distill`. The flush runs
  in a daemon thread on its own short-lived SQLite connection (≤5-row commits,
  one flush at a time per process); Distiller output is stored as typed facts
  (`scope=global` → `project` NULL, session id preserved). Distiller off,
  failing, or unparseable → the raw text is stored as-is (`stored_raw`) —
  never worse than pre-distill behavior; vectors reconcile via `backfill`.
  **Adapter-only deployments** (shell adapters, no v2 plugin) have no `/idle`
  sender: queued content stays unsearchable until the batch threshold trips or
  you run `semantic-memory distill` — bounded staleness, by design.
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
- **`semantic-memory purge`**: hard-deletes soft-deleted entries, applies the TTL,
  then `VACUUM` (only when something was actually removed). Same routine runs at
  every service start (`MEMORY_MAINTENANCE_ON_START=1`) — restart frequency is
  enough for day-granularity retention.
- **`semantic-memory backfill`**: embeds every live entry that has no vector yet
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
GPU attempt. `/api/stats` / `/health` report the actual device in use.

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
- **Local rerank** (`MEMORY_RERANK_MODEL`, **on by default** since the gate
  PASSED 2026-10-16 — recall@5 0.939 vs 0.909 RRF baseline, recall@1 +21pp on a
  33-query golden set): the top `MEMORY_RERANK_CANDIDATES=20` RRF candidates
  are re-scored by Qwen3-Reranker-0.6B (greedy completion + sampling-row
  yes/no softmax, ~13ms/candidate on GPU) within a hard
  `MEMORY_RERANK_BUDGET_MS=500` budget that includes model-slot load; over
  budget or any error → RRF order stands (fail-open). Ties break by `updated_at`
  desc. The default points at the conventional model path — set `MEMORY_RERANK_MODEL=`
  (empty) to disable.

## Capacity & alerting (observability, not interruptions)

Alerts live in service logs and the web UI badge — never injected into a
session. `/api/stats` expose `recall_p50_ms`, `recall_p95_ms`,
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
semantic-memory eval --golden FILE [--k 1,3,5] [--judge]
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
                  # routes / bulk double-gate auth /
                  # random-port serve & discovery / injection telemetry /
                  # ranking post-processing / real-model vector E2E (opt-in)
                  # + evolution suites: distill queue & typed-fact supersede /
                  # reranker & shared model slot / git-origin project keys /
                  # stale-usage retention / cloud consolidation (mock transport) /
                  # offline eval / recall latency telemetry & throttled alerts

cd integrations/opencode
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

- [x] Phase 1: hybrid retrieval + REST control plane (an earlier MCP tool surface was removed in v0.2 — hooks-only)
- [x] Phase 2: opencode v2 plugin, four hook families (inject / pre-compaction flush
      / explicit remember / tool whitelist)
- [x] Phase 3a: multi-harness adapters (`integrations/` — Claude Code, Kimi Code, Codex template)
- [x] Phase 3b: lifecycle maintenance — ingest dedup, TTL/purge/VACUUM, vector
      backfill (startup hook + CLI, deliberately off the agent-facing tool surface)
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
