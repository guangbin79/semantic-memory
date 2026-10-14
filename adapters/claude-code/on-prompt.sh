#!/usr/bin/env bash
# Claude Code UserPromptSubmit 钩子：
#   1. 检测显式记忆信号（"记住…" / "remember …"）→ /ingest
#   2. 用用户 prompt 做 /recall → additionalContext 注入
# stdin 为钩子事件 JSON（含 .prompt）。fail-open。
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$DIR/../common"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')
[ -z "$PROMPT" ] && exit 0

# 1. 显式记忆：记住… / remember …（best-effort 模式匹配）
MEM=$(echo "$PROMPT" | grep -oE '(请记住|记住|remember)[:：]?[^。]{10,500}' | head -1 || true)
if [ -n "$MEM" ]; then
  CLEAN=$(echo "$MEM" | sed -E 's/^(请记住|记住|remember)[:：]?[[:space:]]*//')
  bash "$COMMON/ingest.sh" "$CLEAN" "preference" "chat" &
fi

# 2. 召回注入（带上 harness/hook 标识用于注入遥测）
MEMORY_HARNESS="${MEMORY_HARNESS:-claude-code}" MEMORY_HOOK="prompt" \
BLOCK=$(bash "$COMMON/recall.sh" "$PROMPT")
[ -z "$BLOCK" ] && exit 0

jq -nc --arg ctx "$BLOCK" '{
  hookSpecificOutput: {
    hookEventName: "UserPromptSubmit",
    additionalContext: $ctx
  }
}'
