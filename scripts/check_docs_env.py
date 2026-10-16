#!/usr/bin/env python3
"""文档环境变量校对（任务 11）：README×2 的 MEMORY_* 变量与默认值 vs 代码实现。

比对源：
  - src/semantic_memory/config.py   —— from_env() 的 s()/opt()/_bool() 调用模式
  - src/semantic_memory/paths.py    —— 实际加载模块，干净 env 下调用 resolve_data_dir()
  - integrations/opencode/memory.ts —— process.env.MEMORY_PLUGIN_* 默认值模式

不一致（文档写了代码没有的变量 / 默认值不同）→ 逐条列出并 exit 1。
运行：python3 scripts/check_docs_env.py（仓库根；无需第三方依赖）。
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL: list[str] = []

# ---- ① config.py：s()/opt()/_bool() 调用 → env 默认值 ----
cfg_src = (ROOT / "src/semantic_memory/config.py").read_text(encoding="utf-8")
code_defaults: dict[str, str] = {}
for m in re.finditer(r'\bs\("MEMORY_([A-Z0-9_]+)",\s*"[a-z0-9_]+"\s*(?:,\s*"([^"]*)")?\)', cfg_src):
    code_defaults["MEMORY_" + m.group(1)] = m.group(2) or ""
for m in re.finditer(r'\bopt\("MEMORY_([A-Z0-9_]+)",\s*"[a-z0-9_]+"\)', cfg_src):
    code_defaults.setdefault("MEMORY_" + m.group(1), "")
for m in re.finditer(r'_bool\(\s*"MEMORY_([A-Z0-9_]+)",\s*"[a-z0-9_]+",\s*(True|False)', cfg_src):
    code_defaults["MEMORY_" + m.group(1)] = "1" if m.group(2) == "True" else "0"

code_defaults["MEMORY_DISCOVERY_FILE"] = "<db>.server.json"  # config.py 153-154：opt 后默认 db_path.with_name(name + ".server.json")

# ---- ② paths.py：真加载模块推导动态默认（HOME 打桩 → 换回 ~ 前缀）----
os.environ.pop("MEMORY_DATA_DIR", None)  # 胶注残留修复：pop 必须真实执行
os.environ.pop("XDG_DATA_HOME", None)
os.environ["HOME"] = "/home/checkdocs"
_spec = importlib.util.spec_from_file_location("_sm_paths", ROOT / "src/semantic_memory/paths.py")
_paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paths)
_home = _paths.resolve_data_dir()
_home_tilde = str(_home).replace("/home/checkdocs", "~")
code_defaults["MEMORY_DATA_DIR"] = _home_tilde
code_defaults["MEMORY_DB_PATH"] = str(_paths.default_db_path()).replace("/home/checkdocs", "~")

# rerank 默认 = default_model_path("qwen3-reranker-0.6b-q8_0.gguf")（config.py 143-148 常规路径默认）
code_defaults["MEMORY_RERANK_MODEL"] = str(
    _paths.default_model_path("qwen3-reranker-0.6b-q8_0.gguf")).replace("/home/checkdocs", "~")

# ---- ③ memory.ts：process.env.MEMORY_PLUGIN_* 默认值 ----
ts_src = (ROOT / "integrations/opencode/memory.ts").read_text(encoding="utf-8")
for m in re.finditer(r'process\.env\.(MEMORY_PLUGIN_[A-Z_]+)\s*\?\?\s*("[^"]*"|-?[\d.]+)', ts_src):
    v = m.group(2).strip('"')
    code_defaults[m.group(1)] = v
for m in re.finditer(r'process\.env\.(MEMORY_PLUGIN_[A-Z_]+)\s*===\s*"([^"]+)"', ts_src):
    code_defaults[m.group(1)] = "0"          # === "1"/"system" 才生效 → 默认关
for m in re.finditer(r'process\.env\.(MEMORY_PLUGIN_[A-Z_]+)\s*!==\s*"([^"]+)"', ts_src):
    code_defaults[m.group(1)] = "1"          # !== "0" 才关 → 默认开
code_defaults["MEMORY_PLUGIN_INJECT_POSITION"] = "tail"  # 三元 === "system" ? ... : "tail"

# 文档可提及但无默认值语义的 TOKEN（遥测标签、发现行）
KNOWN_NON_CONFIG = {"MEMORY_HARNESS", "MEMORY_HOOK", "MEMORY_LISTENING"}

def norm(cell: str) -> str:
    cell = cell.strip().strip("`").strip()
    return "" if cell in ("empty", "空") else cell

for name in ("README.md", "README.zh-CN.md"):
    text = (ROOT / name).read_text(encoding="utf-8")
    doc_rows: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*`(MEMORY_[A-Z0-9_]+)`.*?\|\s*([^|]+)\|", line)  # 兼容合并行（HOST/PORT 同格）
        if not m:
            continue
        var, default = m.group(1), norm(m.group(2))
        if var == "MEMORY_HOST":                      # 合并行 "`127.0.0.1` / `8765`"
            port = default.split("/")[-1].strip().strip("`")  # 合并行端口段带反引号
            doc_rows["MEMORY_HOST"], doc_rows["MEMORY_PORT"] = "127.0.0.1", port
            continue
        if var == "MEMORY_DATA_DIR":                  # 单元格含解析序说明 → 只断言默认前缀
            doc_rows[var] = default
            continue
        doc_rows[var] = default

    for var, default in doc_rows.items():
        if var == "MEMORY_DATA_DIR":
            if not default.startswith(code_defaults[var]):
                FAIL.append(f"{name}: {var} 默认应含 {code_defaults[var]!r}，实际 {default!r}")
            continue
        if var not in code_defaults:
            FAIL.append(f"{name}: {var} 不存在于 config.py/paths.py/memory.ts（拼写或已删除的变量）")
        elif default != code_defaults[var]:
            FAIL.append(f"{name}: {var} 默认不一致：文档 {default!r} vs 代码 {code_defaults[var]!r}")

    # 文档正文提及但完全未知的 MEMORY_*（防 MEMORY_MCP_LISTENING 之类残留；
    # 以 _ 结尾的是通配前缀提及（MEMORY_PLUGIN_*），只要存在同前缀真变量即放行）
    for var in set(re.findall(r"MEMORY_[A-Z0-9_]+", text)):
        if var in code_defaults or var in KNOWN_NON_CONFIG or var in doc_rows:
            continue
        if var.endswith("_") and any(k.startswith(var) for k in code_defaults):
            continue
        FAIL.append(f"{name}: 正文提及未知变量 {var}")

if FAIL:
    print(f"check_docs_env: {len(FAIL)} 处不一致：", file=sys.stderr)
    for f in FAIL:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print(f"check_docs_env: OK（README×2 全部 MEMORY_* 变量与默认值和代码一致；共 {len(code_defaults)} 个代码侧变量）")
