#!/usr/bin/env bash
# Kimi Code 用户提交钩子：显式记忆捕获 + 召回注入。
# 与 Claude Code 版逻辑相同，仅事件载荷字段做了多路径兜底
# （.prompt // .message // .text——以 Kimi Code 当前文档的事件载荷为准）。fail-open。
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$DIR/../common"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // .message // .text // empty')
[ -z "$PROMPT" ] && exit 0

MEM=$(echo "$PROMPT" | grep -oE '(请记住|记住|remember)[:：]?[^。]{10,500}' | head -1 || true)
if [ -n "$MEM" ]; then
  CLEAN=$(echo "$MEM" | sed -E 's/^(请记住|记住|remember)[:：]?[[:space:]]*//')
  bash "$COMMON/ingest.sh" "$CLEAN" "preference" "chat" &
fi

MEMORY_HARNESS="${MEMORY_HARNESS:-kimi-code}" MEMORY_HOOK="prompt" \
BLOCK=$(bash "$COMMON/recall.sh" "$PROMPT")
[ -z "$BLOCK" ] && exit 0

# Kimi Code 的上下文注入回包格式以其 hooks 文档为准；
# 若与 Claude Code 相同（additionalContext），本输出直接可用。
jq -nc --arg ctx "$BLOCK" '{
  hookSpecificOutput: {
    hookEventName: "UserPromptSubmit",
    additionalContext: $ctx
  }
}'
