#!/usr/bin/env bash
# Claude Code PreCompact 钩子：薄壳——共用逻辑见 ../common/hook-pre-compact.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/../common/hook-pre-compact.sh" '.transcript_path // empty'
