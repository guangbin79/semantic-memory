#!/usr/bin/env bash
# Claude Code UserPromptSubmit 钩子：薄壳——共用逻辑见 ../common/hook-on-prompt.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/../common/hook-on-prompt.sh" claude-code '.prompt // empty'
