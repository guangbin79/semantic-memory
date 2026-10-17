#!/usr/bin/env bash
# 下载 Qwen3-Embedding-0.6B GGUF（约 0.6GB，Q8_0）
set -euo pipefail

# 模型目录随数据目录三档解析（与 paths.py / download_models.sh 同语义）
if [ -n "${MEMORY_DATA_DIR:-}" ]; then
  MODEL_DIR="${MEMORY_DATA_DIR/#\~/$HOME}/models"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
  MODEL_DIR="${XDG_DATA_HOME/#\~/$HOME}/memory-mcp/models"
else
  MODEL_DIR="$HOME/semantic-memory/models"
  # 旧家目录未迁移时先起一次服务（自动迁移），否则本次下载会把新家灌非空、迁移被护栏跳过
  for legacy in "$HOME/.local/share/memory-mcp" "$HOME/semantic-memory-mcp"; do
    if [ -d "$legacy" ] && [ ! -f "$legacy/MIGRATED" ]; then
      echo "警告：检测到未迁移的旧家目录 $legacy——建议先运行服务完成自动迁移再下载模型（忽略则旧数据迁移将被跳过）" >&2
    fi
  done
fi
mkdir -p "$MODEL_DIR"

FILE="Qwen3-Embedding-0.6B-Q8_0.gguf"
URL="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/${FILE}"
# 与 setup.py 固定常量表同源（2026-10-17 硬化：手动路径补供应链钉验）
SHA256="06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"

if [ -f "$MODEL_DIR/$FILE" ]; then
  if ! echo "$SHA256  $MODEL_DIR/$FILE" | sha256sum -c --status -; then
    echo "已存在但 sha256 不符：$MODEL_DIR/$FILE（损坏或被替换）——请删除后重下" >&2
    exit 1
  fi
  echo "已存在: $MODEL_DIR/$FILE"
else
  echo "下载 $FILE ..."
  curl -fSL --progress-bar "$URL" -o "$MODEL_DIR/$FILE"
  if ! echo "$SHA256  $MODEL_DIR/$FILE" | sha256sum -c --status -; then
    rm -f "$MODEL_DIR/$FILE"
    echo "sha256 校验失败：已删除 $FILE（供应链防护）——请重试或改用 semantic-memory setup" >&2
    exit 1
  fi
fi

cat <<EOF

完成。把下面这行加入 shell 配置（~/.bashrc / ~/.zshrc）：

  export MEMORY_MODEL_PATH="$MODEL_DIR/$FILE"

不加则服务以纯 BM25 模式运行（中文召回仍可用，靠 trigram 通道）。
EOF
