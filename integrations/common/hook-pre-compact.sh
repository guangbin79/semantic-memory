#!/usr/bin/env bash
# hook-pre-compact.sh <transcript-jq表达式>
# 共用 PreCompact 钩子：压缩前把最近一条用户文本托管进记忆库。
# 压缩丢弃 ≠ 永久丢失——侧车让压缩安全。stdin 为钩子事件 JSON。fail-open。
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIPT_JQ="${1:?usage: hook-pre-compact.sh <transcript-jq>}"

export MEMORY_PROJECT="${MEMORY_PROJECT:-$PWD}"

INPUT=$(cat)
TRANSCRIPT=$(echo "$INPUT" | jq -r "$TRANSCRIPT_JQ")
[ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] || exit 0

# transcript 为 JSONL；取最近一条用户文本消息（best-effort，结构以 harness 版本为准）
TEXT=$(tail -50 "$TRANSCRIPT" \
  | jq -r 'select(.message.role=="user") | .message.content[]? | select(.type=="text") | .text' \
  2>/dev/null | tail -1 | head -c 1200)

[ ${#TEXT} -lt 40 ] && exit 0
bash "$DIR/ingest.sh" "$TEXT" "fact" "compaction"
