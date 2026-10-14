#!/usr/bin/env bash
# recall.sh "query text" [limit]
# 从 memory-mcp REST 侧车召回相关记忆，格式化块输出到 stdout（无命中则无输出）。
# 所有 harness 的控制面适配共用此脚本。fail-open：服务不可达静默退出 0。
set -u

BASE="${MEMORY_PLUGIN_URL:-http://127.0.0.1:8765}"
TOKEN="${MEMORY_PLUGIN_TOKEN:-}"
BUDGET="${MEMORY_PLUGIN_BUDGET_CHARS:-2000}"
QUERY="${1:-}"
LIMIT="${2:-${MEMORY_PLUGIN_LIMIT:-5}}"

[ -z "$QUERY" ] && exit 0

AUTH=()
[ -n "$TOKEN" ] && AUTH=(-H "authorization: Bearer $TOKEN")

BODY=$(jq -nc --arg q "$QUERY" --argjson l "$LIMIT" '{query:$q, limit:$l}')
RESP=$(curl -fsS --max-time 1.5 -X POST "$BASE/recall" \
        -H 'content-type: application/json' "${AUTH[@]}" -d "$BODY" 2>/dev/null) || exit 0

BLOCK=$(echo "$RESP" | jq -r '
  .hits // [] | if length == 0 then "" else
    (["<semantic-memory>"] +
     [.[] | "- [\(.kind)] \(.content)" + (if .project then " (project: \(.project))" else "" end)] +
     ["</semantic-memory>"]) | join("\n")
  end')

[ -z "$BLOCK" ] && exit 0
echo "$BLOCK" | head -c "$BUDGET"
