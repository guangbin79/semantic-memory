#!/usr/bin/env bash
# hook-on-prompt.sh <harness> <prompt-jq表达式>
# 共用 UserPromptSubmit 钩子：显式记忆捕获（记住…/remember …）+ /recall 注入
# → hookSpecificOutput.additionalContext 回包。stdin 为钩子事件 JSON。fail-open。
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_NAME="${1:?usage: hook-on-prompt.sh <harness> <prompt-jq>}"
PROMPT_JQ="${2:?usage: hook-on-prompt.sh <harness> <prompt-jq>}"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r "$PROMPT_JQ")
[ -z "$PROMPT" ] && exit 0

# 会话标识（recall.sh 同会话去重）与项目作用域（服务端归一化为 git-origin key）
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
[ -n "$SESSION_ID" ] && export MEMORY_SESSION_ID="$SESSION_ID"
export MEMORY_PROJECT="${MEMORY_PROJECT:-$PWD}"

# 1. 显式记忆：记住… / remember …（best-effort 模式匹配）
MEM=$(echo "$PROMPT" | grep -oE '(请记住|记住|remember)[:：]?[^。]{10,500}' | head -1 || true)
if [ -n "$MEM" ]; then
  CLEAN=$(echo "$MEM" | sed -E 's/^(请记住|记住|remember)[:：]?[[:space:]]*//')
  bash "$DIR/ingest.sh" "$CLEAN" "preference" "remember" &
fi

# 2. 召回注入（带上 harness/hook 标识用于注入遥测；前缀赋值须置于 $() 内才会传给子进程）
BLOCK=$(MEMORY_HARNESS="${MEMORY_HARNESS:-$HARNESS_NAME}" MEMORY_HOOK="prompt" \
  bash "$DIR/recall.sh" "$PROMPT")
[ -z "$BLOCK" ] && exit 0

jq -nc --arg ctx "$BLOCK" '{
  hookSpecificOutput: {
    hookEventName: "UserPromptSubmit",
    additionalContext: $ctx
  }
}'
