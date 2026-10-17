#!/usr/bin/env bash
# 下载 Qwen3-Embedding-0.6B GGUF（约 0.6GB，Q8_0）
set -euo pipefail

# 缺 sha256sum 直接快败：否则缺工具会被当作校验失败误删/误拒（macOS 裸机无 coreutils）
command -v sha256sum >/dev/null 2>&1 || {
  echo "缺少 sha256sum——macOS 可 brew install coreutils，或改用 semantic-memory setup" >&2
  exit 1
}

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
# 与 setup.py 固定常量表同源（锚测试见 tests/test_hardening.py，漂移即红）
SHA256="06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439"

verify() {  # $1=路径 $2=sha —— 摘要比对（不用 -c --status：非零退出混入不可读/缺工具等非匹配失败）
  local path="$1" sha="$2" sum=""
  sum=$(sha256sum "$path" 2>/dev/null | cut -d' ' -f1) || sum=""
  [ "$sum" = "$sha" ]
}

if [ -f "$MODEL_DIR/$FILE" ]; then
  # 既存件不符只拒用不删：用户数据不被脚本销毁（换量化/新版本交用户裁量）
  if ! verify "$MODEL_DIR/$FILE" "$SHA256"; then
    echo "已存在但 sha256 不符：$MODEL_DIR/$FILE（损坏/被替换/换量化）——请确认后手动删除重下" >&2
    exit 1
  fi
  echo "已存在: $MODEL_DIR/$FILE"
else
  echo "下载 $FILE ..."
  # 直落 .part 再原子改名：断传残件不冒充既存件（与 setup.py 同构）
  curl -fSL --progress-bar "$URL" -o "$MODEL_DIR/$FILE.part"
  if ! verify "$MODEL_DIR/$FILE.part" "$SHA256"; then
    rm -f "$MODEL_DIR/$FILE.part"
    echo "sha256 校验失败：已删除 $FILE.part（供应链防护）——请重试或改用 semantic-memory setup" >&2
    exit 1
  fi
  mv -f "$MODEL_DIR/$FILE.part" "$MODEL_DIR/$FILE"
fi

cat <<EOF

完成。把下面这行加入 shell 配置（~/.bashrc / ~/.zshrc）：

  export MEMORY_MODEL_PATH="$MODEL_DIR/$FILE"

不加则服务以纯 BM25 模式运行（中文召回仍可用，靠 trigram 通道）。
EOF
