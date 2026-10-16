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

MODEL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/memory-mcp/models"
mkdir -p "$MODEL_DIR"

DISTILL_FILE="Qwen3-1.7B-Q8_0.gguf"
RERANK_FILE="qwen3-reranker-0.6b-q8_0.gguf"   # 该仓库内文件名全小写
DISTILL_URL="https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/${DISTILL_FILE}"
RERANK_URL="https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/${RERANK_FILE}"

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
  exec python3 - "$PWD" <<'PYEOF'
"""校对 README.md / README.zh-CN.md env 表的每个 MEMORY_* 默认值 == 代码默认值。

双树取值：服务变量读 src/memory_mcp/config.py，插件变量读 opencode-plugin/memory.ts；
再反向断言代码里的每个 MEMORY_* env 都出现在文档表中（防文档漂移）。
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
cfg_src = (root / "src/memory_mcp/config.py").read_text(encoding="utf-8")
ts_src = (root / "opencode-plugin/memory.ts").read_text(encoding="utf-8")

EMPTIES = {"", "empty", "空"}


def doc_rows(name: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `MEMORY_"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        var_list = re.findall(r"MEMORY_[A-Z_0-9]+", cells[0])
        defaults = [d.strip().strip("`") for d in cells[1].split(" / ")]
        if len(var_list) != len(defaults):
            raise SystemExit(f"{name}: 表行默认列数与变量数不符: {line}")
        rows.update(zip(var_list, defaults))
    return rows


def cfg_default(var: str) -> str | None:
    """从 config.py 取 os.environ.get("VAR", "literal") 的字面默认；取不到回 SPECIAL。"""
    special = {
        "MEMORY_DB_PATH": "~/.local/share/memory-mcp/memory.db",   # _default_db()
        "MEMORY_DISCOVERY_FILE": "<db>.server.json",               # 派生默认
    }
    if var in special:
        return special[var]
    m = re.search(r'os\.environ\.get\("' + var + r'"(?:,\s*"([^"]*)")?\)', cfg_src)
    if m:
        return m.group(1) if m.group(1) is not None else ""
    return None


def ts_default(var: str) -> str | None:
    m = re.search(r"process\.env\." + var + r' \?\? "?([^"\s,)]*)"?\s*[,)\n]', ts_src)
    if m:
        return m.group(1)
    if re.search(r'process\.env\.' + var + r' === "1"', ts_src):
        return "0"   # 仅显式 =1 开 → 默认关
    if re.search(r'process\.env\.' + var + r' !== "0"', ts_src):
        return "1"   # 仅显式 =0 关 → 默认开
    m = re.search(r'process\.env\.' + var + r' === "system" \? "system" : "(\w+)"', ts_src)
    return m.group(1) if m else None


fails: list[str] = []
for name in ("README.md", "README.zh-CN.md"):
    rows = doc_rows(name)
    for var, doc in sorted(rows.items()):
        code = ts_default(var) if var.startswith("MEMORY_PLUGIN_") else cfg_default(var)
        if code is None:
            fails.append(f"{name}: {var} 在代码中找不到（文档编造？）")
        elif doc in EMPTIES and code == "":
            continue
        elif doc != code:
            fails.append(f"{name}: {var} 文档默认 {doc!r} != 代码默认 {code!r}")
    # 反向：代码里每个 env 都要出现在文档表（防代码新增/改名后文档漏更）
    for var in set(re.findall(r'os\.environ\.get\("(MEMORY_[A-Z_0-9]+)"', cfg_src)) \
            | set(re.findall(r"process\.env\.(MEMORY_PLUGIN_[A-Z_0-9]+)", ts_src)):
        if var not in rows:
            fails.append(f"{name}: {var} 代码存在但文档表缺失")

if fails:
    print("\n".join(f"FAIL {f}" for f in fails))
    sys.exit(1)
print("audit: 双语 README env 表与代码默认值一致")
PYEOF
fi

download() {
  local file="$1" url="$2"
  if [ -f "$MODEL_DIR/$file" ]; then
    echo "已存在: $MODEL_DIR/$file"
  else
    echo "下载 $file ..."
    curl -fSL --progress-bar "$url" -o "$MODEL_DIR/$file"
  fi
}

download "$DISTILL_FILE" "$DISTILL_URL"
download "$RERANK_FILE" "$RERANK_URL"

cat <<EOF

完成。把下面两行加入 shell 配置（~/.bashrc / ~/.zshrc）：

  export MEMORY_DISTILL_MODEL="$MODEL_DIR/$DISTILL_FILE"
  export MEMORY_RERANK_MODEL="$MODEL_DIR/$RERANK_FILE"

不配则蒸馏关闭（原文直存）、重排关闭（RRF 序直出）——均 fail-open，服务照常。
EOF
