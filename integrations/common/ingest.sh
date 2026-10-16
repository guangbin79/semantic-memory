#!/usr/bin/env bash
# ingest.sh "content" [kind] [source] [project]
# 向 semantic-memory REST API 写入一条记忆。fail-open：失败静默退出 0。
# project 缺省时取 MEMORY_PROJECT（第 4 参优先）。
set -u

BASE="${MEMORY_PLUGIN_URL:-http://127.0.0.1:8765}"
TOKEN="${MEMORY_PLUGIN_TOKEN:-}"
CONTENT="${1:-}"
KIND="${2:-fact}"
SOURCE="${3:-hook}"
PROJECT="${4:-${MEMORY_PROJECT:-}}"

[ -z "$CONTENT" ] && exit 0

AUTH=()
[ -n "$TOKEN" ] && AUTH=(-H "authorization: Bearer $TOKEN")

BODY=$(jq -nc --arg c "$CONTENT" --arg k "$KIND" --arg s "$SOURCE" --arg p "$PROJECT" \
  '{content:$c, kind:$k, source:$s} + (if $p != "" then {project:$p} else {} end)')

curl -fsS --max-time 1.5 -X POST "$BASE/ingest" \
  -H 'content-type: application/json' "${AUTH[@]}" -d "$BODY" >/dev/null 2>&1 || true
exit 0
