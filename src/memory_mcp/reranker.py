"""Qwen3-Reranker-0.6B 重排器：completions logprobs 判别法（spike① 方案）。

官方判别模板（ggml-org/Qwen3-Reranker-0.6B-GGUF）：system 固定判决指令，
user 携带 <Instruct>/<Query>/<Document>（文档截 512 字符），助手侧空 think
引导；对 max_tokens=1、logprobs=20 的补全取首 token top-logprobs 中
yes/no 两项做二类 softmax 得 0~1 相关分。logprobs 缺 yes/no、预算超时、
任何异常 → 整体返回 None（fail-open，调用方保持 RRF 序直出）。

加载生命周期与 embedder.py 同规格：懒加载（经 ModelSlot，与 Distiller
分时互斥）、cuda 默认 → 探针补全 → nvidia-smi PID 校验（识别静默 CPU
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
from .modelslot import ModelSlot

log = logging.getLogger("memory_mcp.reranker")

MAX_DOC_CHARS: Final = 512  # 官方建议的文档截断长度（字符）
LOGPROBS_K: Final = 20
RERANK_N_CTX: Final = 4096  # system + instruct + query + 512 字符文档足够
DEFAULT_INSTRUCT: Final = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
SYSTEM_PROMPT: Final = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no"'
)
ASSISTANT_SUFFIX: Final = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_prompt(instruction: str, query: str, doc: str) -> str:
    """官方 Qwen3-Reranker 判别 prompt；文档截 MAX_DOC_CHARS 字符。"""
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT
        + "<|im_end|>\n<|im_start|>user\n"
        + f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc[:MAX_DOC_CHARS]}"
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
            logits_all=True,  # create_completion(logprobs=N) 依赖此构造参数（0.3.35 否则必抛 ValueError）
            verbose=False,
        )

    @staticmethod
    def _probe(model: Any) -> None:
        model.create_completion("device probe", max_tokens=1, logprobs=1)

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

    @staticmethod
    def _yes_no_score(resp: dict) -> float:
        """首 token top-logprobs 中 yes/no 的二类 softmax。

        键做 strip/lower 归一（真机实测变体：" No"/"NO"/"No"）；无关候选的
        一侧常整掉出 top-K（真机 F3 复现）——缺失侧钳到 floor（最小 logprob-1，
        必低于在场侧）保序得分；两侧皆缺才是响应契约破坏，fail-open。
        """
        try:
            top = resp["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("completion 响应缺 top_logprobs") from exc
        lp: dict[str, float] = {}
        for k, v in top.items():
            kk = k.strip().lower()
            if kk in ("yes", "no") and kk not in lp:
                lp[kk] = float(v)
        if not lp:
            raise ValueError(f"logprobs 缺 yes/no（实际键: {sorted(top)}）")
        floor = min(float(v) for v in top.values()) - 1.0
        exp_yes = math.exp(lp.get("yes", floor))
        return exp_yes / (exp_yes + math.exp(lp.get("no", floor)))

    def _score_all(
        self,
        model: Any,
        query: str,
        candidates: list[tuple[str, str]],
        deadline: float,
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for uid, content in candidates:
            if time.monotonic() > deadline:  # 每次推理前检查，超时整体 fail-open
                raise _BudgetExceeded
            try:
                resp = model.create_completion(
                    format_prompt(DEFAULT_INSTRUCT, query, content),
                    max_tokens=1,
                    logprobs=LOGPROBS_K,
                )
            except Exception as exc:
                if self.device != "cuda":
                    raise
                raise _GpuInferenceError(str(exc)) from exc
            scored.append((uid, self._yes_no_score(resp)))
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
        deadline = time.monotonic() + self.budget_ms / 1000.0
        for _attempt in range(2):  # 第二轮仅用于 GPU 推理失败后的 CPU 重建重试
            try:
                model = self._slot.acquire("rerank", self._load)
            except Exception as exc:
                log.warning("重排模型加载失败（%s: %s）", type(exc).__name__, exc)
                return None
            try:
                return self._score_all(model, query, candidates, deadline)
            except _GpuInferenceError as exc:
                self._force_cpu = True
                log.warning("GPU 重排失败（%s）——CPU 重建重试", exc)
                self._slot.evict("rerank")  # 解引用损坏的 GPU 槽模型（不销毁）
            except Exception as exc:
                log.warning("重排失败（%s: %s）——保 RRF 序 fail-open", type(exc).__name__, exc)
                return None
            finally:
                self._slot.release()
        return None
