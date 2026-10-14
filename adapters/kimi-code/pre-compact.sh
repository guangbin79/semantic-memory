#!/usr/bin/env bash
# Kimi Code 压缩前钩子：把最近用户内容托管进记忆库。fail-open。
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$DIR/../common"

INPUT=$(cat)
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // .transcript // empty')
[ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] || exit 0

TEXT=$(tail -50 "$TRANSCRIPT" \
  | jq -r 'select(.message.role=="user") | .message.content[]? | select(.type=="text") | .text' \
  2>/dev/null | tail -1 | head -c 1200)

[ ${#TEXT} -lt 40 ] && exit 0
bash "$COMMON/ingest.sh" "$TEXT" "fact" "compaction"
