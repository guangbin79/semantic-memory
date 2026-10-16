#!/usr/bin/env bash
# Claude Code PreCompact 钩子：压缩前把最近用户内容托管进记忆库。
# 压缩丢弃 ≠ 永久丢失——侧车让压缩安全。fail-open。
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$DIR/../common"

INPUT=$(cat)
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
[ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] || exit 0

# transcript 为 JSONL；取最近一条用户文本消息（best-effort，结构以 CC 版本为准）
TEXT=$(tail -50 "$TRANSCRIPT" \
  | jq -r 'select(.message.role=="user") | .message.content[]? | select(.type=="text") | .text' \
  2>/dev/null | tail -1 | head -c 1200)

[ ${#TEXT} -lt 40 ] && exit 0
bash "$COMMON/ingest.sh" "$TEXT" "fact" "compaction"
