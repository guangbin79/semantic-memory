"""Qwen3-Reranker-0.6B 重排器：采样分布行直读判别法（spike① 方案的提速实现）。

官方判别模板（ggml-org/Qwen3-Reranker-0.6B-GGUF）：system 固定判决指令，
user 携带 <Instruct>/<Query>/<Document>（文档截 512 字符），助手侧空 think
引导；create_completion(max_tokens=1, 无 logprobs) 走 generate 快路径（含
提示词前缀 KV 复用），事后按 yes/no 判别 token id 直读 scores 采样分布行，
两类边际 softmax 得 0~1 相关分。避开 logprobs=N 后处理病灶（0.3.35 对每
个提示词 token 做 top-K+逐项 detokenize，实测 ~270ms/候选；本路径 GPU
实测 13ms/候选、20 候选 263ms——预算 500ms 内）。词表缺判别 token、预算
超时、任何异常 → 整体返回 None（fail-open，调用方保持 RRF 序直出）。

加载生命周期与 embedder.py 同规格：懒加载（经 ModelSlot，与 Distiller
分时互斥）、cuda 默认 → 探针推理 → nvidia-smi PID 校验（识别静默 CPU
回退）→ 失败重建 CPU；运行期 GPU 推理失败（如 OOM）重建 CPU 重试一次。
设备策略沿用 MEMORY_DEVICE（cuda 默认/cpu 可选）。
"""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Final

from .embedder import _pid_vram_on_gpu, _preload_cuda_libs
from . import gpu_breaker
from .modelslot import ModelSlot

log = logging.getLogger("semantic_memory.reranker")

MAX_DOC_CHARS: Final = 512  # 官方建议的文档截断长度（字符）
RERANK_N_CTX: Final = 1024  # system+instruct+query+512字符文档 ≯400 token；logits_all 的 scores 矩阵=n_ctx×n_vocab（1024×151936×4B≈622MB）
DEFAULT_INSTRUCT: Final = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
SYSTEM_PROMPT: Final = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no"'
)
ASSISTANT_SUFFIX: Final = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_YES_FORMS: Final = ("yes", " yes", "Yes", " Yes", "YES", " YES")  # 真机 F3 变体
_NO_FORMS: Final = ("no", " no", "No", " No", "NO", " NO")
MAX_QUERY_CHARS: Final = 256  # query 侧截断：doc 已截 512 字符，CJK 下不截可超 n_ctx=1024


def format_prompt(instruction: str, query: str, doc: str) -> str:
    """官方 Qwen3-Reranker 判别 prompt；文档截 MAX_DOC_CHARS、query 截 MAX_QUERY_CHARS 字符。"""
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT
        + "<|im_end|>\n<|im_start|>user\n"
        + f"<Instruct>: {instruction}\n<Query>: {query[:MAX_QUERY_CHARS]}\n<Document>: {doc[:MAX_DOC_CHARS]}"
        + ASSISTANT_SUFFIX
    )


class _GpuInferenceError(Exception):
    """GPU 推理失败（触发 CPU 重建、整次调用重试一次）。"""


class _BudgetExceeded(Exception):
    """超预算信号（fail-open）。"""


class Reranker:
    """惰性加载。available=False 时调用方跳过重排（保持 RRF 序）。"""

    def __init__(
        self,
        model_path: Path | None,
        device: str = "cuda",
        budget_ms: int = 500,
        slot: ModelSlot | None = None,
    ):
        self.prefer = device          # 用户意图：cuda|cpu
        self.device: str | None = None  # 实际运行设备：cuda|cpu|None
        self.budget_ms = budget_ms
        self._slot = slot if slot is not None else ModelSlot()
        self._gguf_path = Path(model_path) if model_path else None
        self._force_cpu = False       # 运行期 GPU 推理失败后置位（OOM 重建路径）
        self._yes_ids: tuple[int, ...] | None = None  # 判别 token id，首次打分前解析
        self._no_ids: tuple[int, ...] | None = None
        self.available = False
        if self._gguf_path is None:
            log.info("MEMORY_RERANK_MODEL 未设置，重排关闭（RRF 序直出）")
        elif not self._gguf_path.exists():
            log.warning("重排模型文件不存在: %s，重排关闭", self._gguf_path)
        else:
            self.available = True  # 惰性：首次 rerank 才经槽加载

    # ---------- 加载与回退（镜像 embedder.py 的链路） ----------

    def _new_model(self, use_gpu: bool) -> Any:
        import llama_cpp  # 延迟导入：非重排路径不付 CUDA dlopen 成本

        return llama_cpp.Llama(
            model_path=str(self._gguf_path),
            embedding=False,
            n_gpu_layers=-1 if use_gpu else 0,
            n_ctx=RERANK_N_CTX,
            logits_all=True,  # eval() 仅在 logits_all 下填充 scores（直接 logits 法依赖）
            verbose=False,
        )

    @staticmethod
    def _probe(model: Any) -> None:
        model.create_completion("device probe", max_tokens=1, temperature=0.0)

    def _load(self) -> Any:
        """槽加载器：GPU 尝试 → 探针补全 → nvidia-smi PID 校验 → 失败重建 CPU。"""
        if self._force_cpu or self.prefer == "cpu":
            model = self._new_model(False)
            self._probe(model)
            self.device = "cpu"
            return model
        _preload_cuda_libs()
        try:
            model = self._new_model(True)
            self._probe(model)
            vram = _pid_vram_on_gpu(os.getpid())
            if vram is None:
                raise RuntimeError(
                    "CUDA 会话已建立但 nvidia-smi 无此 pid 显存——"
                    "判定为静默 CPU 回退，拒绝声称 GPU"
                )
            self.device = "cuda"
            log.info("重排器运行于 GPU: pid=%s vram=%s", os.getpid(), vram)
            return model
        except Exception as exc:
            log.warning("GPU 初始化失败（%s: %s）——回退 CPU 重排器",
                        type(exc).__name__, exc)
            model = self._new_model(False)
            self._probe(model)
            self.device = "cpu"
            return model

    # ---------- 打分 ----------

    def _resolve_ids(self, model: Any) -> None:
        """词表内解析 yes/no 判别 token id（单 token 表面形；多 token 形剔除）。"""
        def one_token_ids(forms: tuple[str, ...]) -> tuple[int, ...]:
            out: set[int] = set()
            for form in forms:
                toks = model.tokenize(form.encode("utf-8"), add_bos=False)
                if len(toks) == 1:
                    out.add(int(toks[0]))
            return tuple(out)

        yes, no = one_token_ids(_YES_FORMS), one_token_ids(_NO_FORMS)
        if not yes or not no:
            raise ValueError("词表缺 yes/no 判别 token——无法打分")
        self._yes_ids, self._no_ids = yes, no

    @staticmethod
    def _infer(model: Any, prompt: str) -> Any:
        """快路径补全：必须走 create_completion（内部 generate 含前缀 KV 复用，
        system+query 前缀跨候选共享）；直连 eval 无此优化且 KV 跨调用累积爆 n_ctx。"""
        return model.create_completion(prompt, max_tokens=1, temperature=0)

    def _score_from(self, model: Any, resp: Any) -> float:
        """直读采样分布行，两类边际 softmax。

        create_completion 在 max_tokens=1 的首个 yield 处挂起，生成 token 未
        计入 n_tokens——sample() 实际读的行即 scores[n_tokens-1]（末位提示词
        位置的下一 token 分布）。类内多表面形 exp 求和作互斥形边际，等价于
        旧 top-logprobs 二类 softmax且免去“无关侧掉出 top-K”的钳底特判。
        """
        row = model.scores[model.n_tokens - 1]
        # 廉价自检：temperature=0 贪心 token 应是 yes/no 判别形（模型契约/行语义错位在此暴露）
        ans = str(resp["choices"][0]["text"]).strip().lower()
        if ans not in ("yes", "no"):
            raise ValueError(f"采样 token {ans!r} ∉ yes/no 判别形——契约破坏或行语义错位")
        e_yes = sum(math.exp(float(row[i])) for i in self._yes_ids)
        e_no = sum(math.exp(float(row[i])) for i in self._no_ids)
        if not (e_yes + e_no > 0.0):
            raise ValueError("yes/no logits 退化")
        return e_yes / (e_yes + e_no)

    def _score_all(
        self,
        model: Any,
        query: str,
        candidates: list[tuple[str, str]],
        deadline: float,
    ) -> list[tuple[str, float]]:
        if self._yes_ids is None:
            self._resolve_ids(model)  # 词表解析失败 → 下方通用 except → fail-open
        scored: list[tuple[str, float]] = []
        # INFER_LOCK：跨模型 GPU 推理全局互斥（embedder 常驻 × 槽内模型并发
        # 解码即 CUDA error/GGML abort——2026-10-21 四类崩溃审计的机制结论）
        with gpu_breaker.INFER_LOCK:
            for uid, content in candidates:
                if time.monotonic() > deadline:  # 每次推理前检查，超时整体 fail-open
                    raise _BudgetExceeded
                try:
                    # 仅推理纳入 GPU 重建边界：纯 Python 打分错误不应触发永久 CPU 降级
                    resp = self._infer(model, format_prompt(DEFAULT_INSTRUCT, query, content))
                except Exception as exc:
                    if self.device != "cuda":
                        raise
                    raise _GpuInferenceError(str(exc)) from exc
                scored.append((uid, self._score_from(model, resp)))  # 行读取/softmax 出错 → 直接 fail-open
        return sorted(scored, key=lambda kv: kv[1], reverse=True)

    # ---------- 对外接口 ----------

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]] | None:
        """按相关分降序重排候选；任何失败路径 → None（fail-open）。

        预算覆盖整次调用（含首叫时的模型槽加载耗时）；deadline 在每次推理
        前检查，超预算整体返回 None，模型留在槽内供后续调用命中。
        """
        if not self.available or not candidates:
            return None
        if gpu_breaker.tripped():
            return None  # CUDA 熔断：不再触碰 llama（BM25 召回保活）
        deadline = time.monotonic() + self.budget_ms / 1000.0
        for _attempt in range(2):  # 第二轮仅用于 GPU 推理失败后的 CPU 重建重试
            try:
                # 无 trip 后复检：acquire 后的复检会杀死 in-call CPU 重试
                # （trip 于第 1 轮置位→复检命中→重试永不发生，2026-10-21 实测）——
                # 竞态窗口有意接受，勿加回复检
                model = self._slot.acquire("rerank", self._load)
            except Exception as exc:
                log.warning("重排模型加载失败（%s: %s）", type(exc).__name__, exc)
                return None
            try:
                return self._score_all(model, query, candidates, deadline)
            except _GpuInferenceError as exc:
                # CUDA 熔断：损坏模型禁入 GC（free 即 abort，2026-10-21 实证
                # evict 解引用后的任意 GC 时刻延迟引爆）——墓场保活＋熔断，
                # 第二轮 CPU 重建重试一次（designed 降级链保留）
                gpu_breaker.quarantine(model)
                gpu_breaker.trip(f"rerank CUDA error: {exc}")
                self._force_cpu = True
                log.warning("GPU 重排失败（%s）——熔断＋CPU 重建重试", exc)
                self._slot.evict("rerank")
            except Exception as exc:
                log.warning("重排失败（%s: %s）——保 RRF 序 fail-open", type(exc).__name__, exc)
                return None
            finally:
                self._slot.release()
        return None
