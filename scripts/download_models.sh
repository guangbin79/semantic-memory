#!/usr/bin/env bash
# 下载蒸馏 + 重排双模型（嵌入模型见 download_model.sh）。
#   蒸馏: unsloth/Qwen3-1.7B-GGUF Q8_0（非思考小模型，原文消化成 typed facts）
#   重排: ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF
#     重排模型必须用 ggml-org 官方转换：社区镜像常缺 cls/rank 头，
#     打分全为垃圾（llama.cpp PR #9510/#15824 记录的转换陷阱）。
# 用法:
#   download_models.sh          下载两个模型并打印 export 行
#   download_models.sh --check  只验 URL 可达（HEAD），不下载
#   download_models.sh --audit  README env 表 vs 代码默认值逐一比对，不一致退出非零
set -euo pipefail

# 缺 sha256sum 直接快败：否则缺工具会被当作校验失败误删/误拒（macOS 裸机无 coreutils）
command -v sha256sum >/dev/null 2>&1 || {
  echo "缺少 sha256sum——macOS 可 brew install coreutils，或改用 semantic-memory setup" >&2
  exit 1
}

# 模型目录随数据目录三档解析（与 paths.py 同语义，v0.3 起默认 ~/semantic-memory）
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

DISTILL_FILE="Qwen3-1.7B-Q8_0.gguf"
RERANK_FILE="qwen3-reranker-0.6b-q8_0.gguf"   # 该仓库内文件名全小写
DISTILL_URL="https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/${DISTILL_FILE}"
RERANK_URL="https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/${RERANK_FILE}"
# 与 setup.py 固定常量表同源（2026-10-17 硬化：手动路径补供应链钉验）
DISTILL_SHA="0becaa825564295d82e9af4d008bca5f8b7f5f73bf1c6a0b58f7c53ef26b47fd"
RERANK_SHA="22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48"

if [ "${1:-}" = "--check" ]; then
  rc=0
  for url in "$DISTILL_URL" "$RERANK_URL"; do
    code=$(curl -sIL -o /dev/null -w '%{http_code}' "$url")
    echo "$code $url"
    [ "$code" = "200" ] || { echo "FAIL: $url" >&2; rc=1; }
  done
  [ "$rc" = "0" ] && echo "check: 两个模型 URL 均可达"
  exit "$rc"
fi

if [ "${1:-}" = "--audit" ]; then
  # 文档 env 表校验统一走权威校对器（旧内联抽取在 config 重构后失配，已废弃）
  exec python3 "$(cd "$(dirname "$0")" && pwd)/check_docs_env.py"
fi

verify() {  # $1=路径 $2=sha —— 摘要比对（不用 -c --status：非零退出混入不可读/缺工具等非匹配失败）
  local path="$1" sha="$2" sum=""
  sum=$(sha256sum "$path" 2>/dev/null | cut -d' ' -f1) || sum=""
  if [ "$sum" != "$sha" ]; then
    return 1
  fi
}

refuse() {  # $1=file $2=信息 —— 既存件不符：只拒用不删（用户数据不被脚本销毁）
  echo "$2" >&2
  echo "已存在但 sha256 不符：$MODEL_DIR/$1（损坏/被替换/换量化）——请确认后手动删除重下" >&2
  exit 1
}

download() {
  local file="$1" url="$2" sha="$3"
  if [ -f "$MODEL_DIR/$file" ]; then
    verify "$MODEL_DIR/$file" "$sha" \
      || refuse "$file" "既存件校验失败——不自动删除，交用户裁量"
    echo "已存在: $MODEL_DIR/$file"
  else
    echo "下载 $file ..."
    # 直落 .part 再原子改名：断传残件不冒充既存件（与 setup.py 同构）
    curl -fSL --progress-bar "$url" -o "$MODEL_DIR/$file.part"
    if ! verify "$MODEL_DIR/$file.part" "$sha"; then
      rm -f "$MODEL_DIR/$file.part"
      echo "sha256 校验失败：已删除 $file.part（供应链防护）——请重试或改用 semantic-memory setup" >&2
      exit 1
    fi
    mv -f "$MODEL_DIR/$file.part" "$MODEL_DIR/$file"
  fi
}

download "$DISTILL_FILE" "$DISTILL_URL" "$DISTILL_SHA"
download "$RERANK_FILE" "$RERANK_URL" "$RERANK_SHA"

cat <<EOF

完成。把下面两行加入 shell 配置（~/.bashrc / ~/.zshrc）：

  export MEMORY_DISTILL_MODEL="$MODEL_DIR/$DISTILL_FILE"
  export MEMORY_RERANK_MODEL="$MODEL_DIR/$RERANK_FILE"

不配则蒸馏关闭（原文直存）、重排关闭（RRF 序直出）——均 fail-open，服务照常。
EOF
