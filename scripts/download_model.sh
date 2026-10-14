#!/usr/bin/env bash
# 下载 Qwen3-Embedding-0.6B GGUF（约 0.6GB，Q8_0）
set -euo pipefail

MODEL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/memory-mcp/models"
mkdir -p "$MODEL_DIR"

FILE="Qwen3-Embedding-0.6B-Q8_0.gguf"
URL="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/${FILE}"

if [ -f "$MODEL_DIR/$FILE" ]; then
  echo "已存在: $MODEL_DIR/$FILE"
else
  echo "下载 $FILE ..."
  curl -fSL --progress-bar "$URL" -o "$MODEL_DIR/$FILE"
fi

cat <<EOF

完成。把下面这行加入 shell 配置（~/.bashrc / ~/.zshrc）：

  export MEMORY_MODEL_PATH="$MODEL_DIR/$FILE"

不加则服务以纯 BM25 模式运行（中文召回仍可用，靠 trigram 通道）。
EOF
