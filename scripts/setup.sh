#!/usr/bin/env bash
# semantic-memory 一键自举（任务 10）：独立 venv → 装包 → 透传 setup 子命令。
#
# 人：curl -fsSL \
#   https://raw.githubusercontent.com/guangbin79/semantic-memory/v1.0/scripts/setup.sh \
#   | bash -s -- --profile full --json --yes
# agent：bash scripts/setup.sh --profile full --json --yes
#
# 数据目录（TARGET）：--data-dir > MEMORY_DATA_DIR > ~/semantic-memory；
# venv 建在 TARGET/venv，与服务数据同居、完全自立。
# MEMORY_MCP_SRC=<仓库路径> 覆写为本地源安装（开发/测试）；默认钉 TAG 安装，禁 HEAD。
# stdout 纪律：本脚本一切杂讯走 stderr——--json 时 stdout 恰为 setup 的单行 JSON。
set -euo pipefail

TAG="v1.0"   # 钉 release tag；与 README curl 示例同源，升级 = 改这一处
REMOTE_SPEC="git+https://github.com/guangbin79/semantic-memory.git@${TAG}#egg=semantic-memory[embed]"

# ---- TARGET 解析：--data-dir（两种写法）> MEMORY_DATA_DIR > XDG > ~/semantic-memory（与 paths.py 同构；~ 展开）----
data_dir_arg=""
prev=""
for a in "$@"; do
  if [[ $prev == "--data-dir" ]]; then data_dir_arg=$a; fi
  if [[ $a == --data-dir=* ]]; then data_dir_arg=${a#--data-dir=}; fi
  prev=$a
done
if [[ -n $data_dir_arg ]]; then
  TARGET="${data_dir_arg/#\~/$HOME}"
elif [[ -n ${MEMORY_DATA_DIR:-} ]]; then
  TARGET="${MEMORY_DATA_DIR/#\~/$HOME}"
elif [[ -n ${XDG_DATA_HOME:-} ]]; then
  TARGET="$XDG_DATA_HOME/memory-mcp"
else
  TARGET="$HOME/semantic-memory"
fi

# ---- 前置：选一个 >= 3.11 的解释器（tomllib 是硬依赖；Ubuntu 等系统 python3 常偏旧）----
PY=""
for cand in python3 python3.13 python3.12 python3.11; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY=$cand; break; fi
done
if [[ -z $PY ]]; then
  echo "setup.sh: 未找到 python3 >= 3.11（python3 当前为 $(python3 --version 2>&1)；请安装 3.11+ 后重试）" >&2
  exit 1
fi
echo "setup.sh: 目标目录 $TARGET（解释器 $PY）" >&2
mkdir -p "$TARGET"
"$PY" -m venv "$TARGET/venv"

if [[ -n ${MEMORY_MCP_SRC:-} ]]; then
  echo "setup.sh: 从本地源安装 $MEMORY_MCP_SRC（extras 由该环境自理）" >&2
  "$TARGET/venv/bin/pip" install "$MEMORY_MCP_SRC" >&2
else
  echo "setup.sh: 安装 semantic-memory $REMOTE_SPEC（含 embed extras，llama-cpp-python 需编译，请耐心）" >&2
  "$TARGET/venv/bin/pip" install "$REMOTE_SPEC" >&2
fi

set +e
"$TARGET/venv/bin/semantic-memory" setup "$@"
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  REPO_TREE="https://github.com/guangbin79/semantic-memory/tree/${TAG}/integrations"
  cat >&2 <<EOF

下一步：
  起服：$TARGET/venv/bin/semantic-memory serve
        （或把 $TARGET/venv/bin 加入 PATH 后：semantic-memory serve）
  接线（各宿主钩子见仓库 integrations/，$REPO_TREE）：
    opencode     → integrations/opencode/    （进程内插件：hooks + memory_store/memory_recall 原生工具）
    Claude Code  → integrations/claude-code/ （settings.snippet.json：UserPromptSubmit + PreCompact hooks）
    Kimi Code    → integrations/kimi-code/   （config.snippet.toml：[[hooks]] 或插件清单）
    Codex        → integrations/codex/       （hooks.template.json：模板，需逐 hook 信任）
EOF
fi
exit "$rc"
