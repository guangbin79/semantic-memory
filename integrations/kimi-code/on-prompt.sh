#!/usr/bin/env bash
# Kimi Code 用户提交钩子：薄壳。载荷多路径兜底（.prompt // .message // .text）
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/../common/hook-on-prompt.sh" kimi-code '.prompt // .message // .text // empty'
