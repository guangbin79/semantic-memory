#!/usr/bin/env python3
"""task16 spike①: Qwen3-Reranker logprobs scoring path via llama-cpp-python.

Verifies three things, each independently falsifiable:
  1. llama-cpp-python (0.3.35 pinned in pyproject.toml) is importable;
  2. a Qwen3-Reranker GGUF is present locally (no downloads here);
  3. create_completion(max_tokens=1, logprobs=20) exposes yes/no token
     logprobs on the official reranker template, and the relevant
     (query, doc) pair outscores the irrelevant one.

Exit codes: 0 = PASS or SKIPPED_NO_MODEL (reason printed); 1 = FAIL.
Never starts a server, never downloads anything.
"""

import glob
import math
import os
import sys

# 官方 Qwen3-Reranker 判别模板（Qwen3-Reranker model card，无附加指令时
# instruction 默认为 "Given a user query, retrieve relevant passages..."）。
SYSTEM_TMPL = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".'
)
USER_TMPL = (
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: {doc}"
)
# 生成侧后缀：非思考模式（think 块置空后直接进入回答）。
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a user query, retrieve relevant passages that answer the query"

MODEL_CANDIDATE_DIRS = [
    os.path.expanduser("~/ctx-semantic/models"),
    os.path.expanduser("~/.local/share/memory-mcp/models"),
    os.path.expanduser("~/.cache/llama.cpp/models"),
    os.path.expanduser("~/models"),
]


def find_model() -> str | None:
    """MEMORY_RERANK_MODEL 显式指定优先，否则在常见目录下找 *Rerank*.gguf。"""
    env = os.environ.get("MEMORY_RERANK_MODEL", "").strip()
    if env and os.path.isfile(env):
        return env
    for d in MODEL_CANDIDATE_DIRS:
        for pat in ("*Rerank*.gguf", "*rerank*.gguf"):
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0]
    return None


def pick_logprob(top: dict, token: str) -> float | None:
    """top_logprobs 的 key 可能带前导空格/变体归一化，取 yes/no 的 logprob。"""
    for k, v in top.items():
        if k.strip().lower() == token:
            return v
    return None


def score(llm, query: str, doc: str, instruction: str = DEFAULT_INSTRUCTION) -> tuple[float | None, dict]:
    """构造 (query, doc) 判别 prompt，取首 token top-20 logprobs，softmax(yes,no) 为分。"""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_TMPL}<|im_end|>\n"
        f"<|im_start|>user\n{USER_TMPL.format(instruction=instruction, query=query, doc=doc[:512])}"
        f"{ASSISTANT_SUFFIX}"
    )
    out = llm.create_completion(prompt, max_tokens=1, logprobs=20, temperature=0.0)
    choice = out["choices"][0]
    lps = choice.get("logprobs") or {}
    tops = lps.get("top_logprobs") or []
    top = tops[0] if tops else {}
    lp_yes = pick_logprob(top, "yes")
    lp_no = pick_logprob(top, "no")
    if lp_yes is None or lp_no is None:
        return None, {"top10": dict(sorted(top.items(), key=lambda kv: -kv[1])[:10])}
    m = max(lp_yes, lp_no)
    p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))
    return p_yes, {"lp_yes": lp_yes, "lp_no": lp_no, "top10": dict(sorted(top.items(), key=lambda kv: -kv[1])[:10])}


def main() -> int:
    # 检查 1：依赖可导入（不可导入 → SKIPPED_NO_MODEL，回退已写计划）。
    try:
        import llama_cpp
    except ImportError as e:
        print(f"SKIPPED_NO_MODEL: llama-cpp-python not importable ({e}); pinned 0.3.35 in pyproject.toml [embed] extra")
        return 0
    print(f"llama-cpp-python {getattr(llama_cpp, '__version__', '?')} importable")

    # 检查 2：模型真件存在（缺失 → SKIPPED_NO_MODEL，禁止现场下载）。
    model_path = find_model()
    if not model_path:
        print("SKIPPED_NO_MODEL: no Qwen3-Reranker GGUF found (searched MEMORY_RERANK_MODEL, "
              + ", ".join(MODEL_CANDIDATE_DIRS) + "); rerank defaults stay off until F3")
        return 0
    print(f"model: {model_path} ({os.path.getsize(model_path) / 1e6:.0f} MB)")

    llm = llama_cpp.Llama(
        model_path=model_path,
        n_gpu_layers=-1 if os.environ.get("MEMORY_DEVICE", "cuda") != "cpu" else 0,
        n_ctx=4096,
        verbose=False,
    )

    # 检查 3：相关对得分 > 无关对，且 yes/no logprob 均可取得。
    query = "how do I configure the embedding model for the memory service?"
    relevant = "Set MEMORY_MODEL_PATH to the Qwen3-Embedding GGUF file to enable the vector channel; empty means BM25-only fallback."
    irrelevant = "The Paris Metro line 12 was opened in 1910 and runs between Invalides and Porte de Versailles."

    p_rel, info_rel = score(llm, query, relevant)
    p_irr, info_irr = score(llm, query, irrelevant)
    print(f"relevant:   p_yes={p_rel} details={info_rel}")
    print(f"irrelevant: p_yes={p_irr} details={info_irr}")

    if p_rel is None or p_irr is None:
        print("FAIL: yes/no logprobs not retrievable from top-20 (rerank path unusable)")
        return 1
    if p_rel <= p_irr:
        print(f"FAIL: relevant pair does not outscore irrelevant ({p_rel:.4f} <= {p_irr:.4f})")
        return 1
    print(f"PASS: relevant p_yes={p_rel:.4f} > irrelevant p_yes={p_irr:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
