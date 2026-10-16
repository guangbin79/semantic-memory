#!/usr/bin/env bash
# Kimi Code 压缩前钩子：薄壳。载荷多路径兜底（.transcript_path // .transcript）
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/../common/hook-pre-compact.sh" '.transcript_path // .transcript // empty'
