"""Qwen3-1.7B 非思考蒸馏器：把压缩丢弃的原文消化成 typed 记忆条目。

Qwen3 官方 chat 模板的非思考形态：<|im_start|> system/user/assistant，
助手侧无 <think> 引导；要求模型只输出 JSON 数组
[{content, kind∈fact|decision|preference|artifact, scope∈project|global,
rationale, tags}]；带 neighbors 时条目可另含 supersedes: [uid]——取代裁决
与内容提取共用同一次调用。解析容错（截取首 [ 至末 ]，兼容 markdown 围栏与
前后闲话），解析失败返回 None（fail-open，调用方原文直存）；
decision/preference 缺 rationale 降级 kind=fact 并 WARN。

加载生命周期与 reranker.py 同规格：懒加载（经 ModelSlot，与 Reranker
分时互斥）、cuda 默认 → 探针补全 → nvidia-smi PID 校验（识别静默 CPU
回退）→ 失败重建 CPU；运行期 GPU 推理失败（如 OOM）重建 CPU 重试一次。
设备策略沿用 MEMORY_DEVICE（cuda 默认/cpu 可选）。MEMORY_DISTILL_MODEL
空 = 蒸馏关闭。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Final

from . import gpu_breaker


from .embedder import _pid_vram_on_gpu, _preload_cuda_libs
from .modelslot import ModelSlot

log = logging.getLogger("semantic_memory.distiller")

DISTILL_N_CTX: Final = 8192   # 输入可含整段压缩 transcript（~8000 字符）+ 输出预算
MAX_NEW_TOKENS: Final = 1024  # 一批 ≤20 条的 JSON 数组输出预算
MAX_INPUT_CHARS: Final = 5000  # 输入截断：中文 ~1 token/字，n_ctx 8192 需留 system+近邻+输出预算
IM_END: Final = "<|im_end|>"
VALID_KINDS: Final = ("fact", "decision", "preference", "artifact")
VALID_SCOPES: Final = ("project", "global")
SYSTEM_PROMPT: Final = (
    "You extract durable memories from raw conversation text. Respond with "
    "ONLY a JSON array - no markdown fences, no commentary. Each element has "
    'keys "content" (concise self-contained memory), "kind" (one of "fact", '
    '"decision", "preference", "artifact"), "scope" ("project" or "global"), '
    '"rationale" (why it matters; REQUIRED when kind is decision or '
    'preference), "tags" (short keywords). Use scope "project" for '
    'project-specific knowledge and "global" for universally applicable '
    "knowledge. Output [] when nothing is worth remembering."
)


def format_prompt(
    texts: list[str],
    project: str,
    neighbors: list[tuple[str, str]] | None = None,
) -> str:
    """Qwen3 非思考 chat 模板；助手侧无 <think> 引导。

    neighbors 为 (uid, content) 近邻对：非空时附 <<<MEMORIES>>> 段并列出
    取代条目的 supersedes 输出契约——取代裁决与内容提取共用同一次调用。
    """
    joined = "\n---\n".join(texts)[:MAX_INPUT_CHARS]
    user = f"Project: {project}\n<<<TEXT>>>\n{joined}\n<<<END>>>"
    if neighbors:
        listed = "\n".join(f"[{uid}] {content}" for uid, content in neighbors)
        user += (
            "\n<<<MEMORIES>>>\n" + listed + "\n<<<END>>>"
            '\nIf a new entry contradicts one of the memories above or makes it'
            ' outdated, add a "supersedes" key to that entry: the array of'
            ' replaced memory uids, e.g. "supersedes": ["<uid>"]. Omit'
            ' "supersedes" when nothing is replaced.'
        )
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT
        + IM_END
        + "\n<|im_start|>user\n"
        + user
        + IM_END
        + "\n<|im_start|>assistant\n"
    )


def _normalize(entry: Any) -> dict | None:
    """边界归一化：非字典/空 content 丢弃；越界 kind/scope 收敛到合法词表。"""
    if not isinstance(entry, dict):
        return None
    content = entry.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    kind = entry.get("kind")
    rationale = entry.get("rationale")
    if kind not in VALID_KINDS:
        kind = "fact"
    if kind in ("decision", "preference") and not (
        isinstance(rationale, str) and rationale.strip()
    ):
        log.warning("decision/preference 条目缺 rationale，降级为 kind=fact")
        kind = "fact"
    scope = entry.get("scope")
    if scope not in VALID_SCOPES:
        scope = "project"
    tags = entry.get("tags")
    sup = entry.get("supersedes")
    out = {
        "content": content,
        "kind": kind,
        "scope": scope,
        "rationale": rationale if isinstance(rationale, str) else "",
        "tags": [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else [],
    }
    supersedes = (
        [s for s in sup if isinstance(s, str) and s.strip()]
        if isinstance(sup, list) else []
    )
    if supersedes:  # 仅在模型确实给出时才带键：基础输出契约保持不变
        out["supersedes"] = supersedes
    return out


def _parse(raw: str) -> list[dict] | None:
    """容错解析：截取首 [ 至末 ]（兼容 markdown 围栏/前后闲话）；失败 None。"""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [e for e in (_normalize(x) for x in data) if e is not None]


class _GpuInferenceError(Exception):
    """GPU 推理失败（触发 CPU 重建、整次调用重试一次）。"""


class Distiller:
    """惰性加载。available=False 时调用方跳过蒸馏（原文直存）。"""

    def __init__(
        self,
        model_path: Path | None,
        device: str = "cuda",
        slot: ModelSlot | None = None,
    ):
        self.prefer = device          # 用户意图：cuda|cpu
        self.device: str | None = None  # 实际运行设备：cuda|cpu|None
        self._slot = slot if slot is not None else ModelSlot()
        self._gguf_path = Path(model_path) if model_path else None
        self._force_cpu = False       # 运行期 GPU 推理失败后置位（OOM 重建路径）
        self.available = False
        if self._gguf_path is None:
            log.info("MEMORY_DISTILL_MODEL 未设置，蒸馏关闭（原文直存）")
        elif not self._gguf_path.exists():
            log.warning("蒸馏模型文件不存在: %s，蒸馏关闭", self._gguf_path)
        else:
            self.available = True  # 惰性：首次 distill 才经槽加载

    def is_available(self) -> bool:
        return self.available

    @property
    def configured(self) -> bool:
        """是否配置了裁判模型（路径级；与加载可用性无关——用于区分未配置与暂不可用）。"""
        return self._gguf_path is not None

    # ---------- 加载与回退（镜像 reranker.py 的链路） ----------

    def _new_model(self, use_gpu: bool) -> Any:
        import llama_cpp  # 延迟导入：非蒸馏路径不付 CUDA dlopen 成本

        return llama_cpp.Llama(
            model_path=str(self._gguf_path),
            embedding=False,
            n_gpu_layers=-1 if use_gpu else 0,
            n_ctx=DISTILL_N_CTX,
            verbose=False,
        )

    @staticmethod
    def _probe(model: Any) -> None:
        model.create_completion("device probe", max_tokens=1)

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
            log.info("蒸馏器运行于 GPU: pid=%s vram=%s", os.getpid(), vram)
            return model
        except Exception as exc:
            log.warning("GPU 初始化失败（%s: %s）——回退 CPU 蒸馏器",
                        type(exc).__name__, exc)
            model = self._new_model(False)
            self._probe(model)
            self.device = "cpu"
            return model

    # ---------- 蒸馏 ----------

    def _generate(
        self, model: Any, texts: list[str], project: str,
        neighbors: list[tuple[str, str]] | None = None,
    ) -> str:
        try:
            # INFER_LOCK：跨模型 GPU 推理全局互斥（同 gpu_breaker.INFER_LOCK 注释）
            with gpu_breaker.INFER_LOCK:
                resp = model.create_completion(
                    format_prompt(texts, project, neighbors),
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=0.1,
                    stop=[IM_END],
                )
        except Exception as exc:
            if self.device != "cuda":
                raise
            raise _GpuInferenceError(str(exc)) from exc
        try:
            return resp["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("completion 响应缺 text") from exc

    def distill(
        self, texts: list[str], project: str,
        neighbors: list[tuple[str, str]] | None = None,
    ) -> list[dict] | None:
        """一批原文 → typed 条目数组；任何失败路径 → None（fail-open）。

        neighbors 为同域近邻 (uid, content) 上下文：非空时提示词附带
        "是否取代"问答，取代关系经条目 supersedes 字段一并返回。"""
        if not self.available or not texts:
            return None
        for _attempt in range(2):  # 第二轮仅用于 GPU 推理失败后的 CPU 重建重试
            try:
                model = self._slot.acquire("distill", self._load)
            except Exception as exc:
                log.warning("蒸馏模型加载失败（%s: %s）", type(exc).__name__, exc)
                return None
            try:
                entries = _parse(self._generate(model, texts, project, neighbors))
                if entries:  # supersedes 白名单无条件生效：无近邻时 allowed 为空集全部剥离（防模型对任意 uid 发 supersede）
                    allowed = {uid for uid, _ in (neighbors or [])}
                    for e in entries:
                        if "supersedes" in e:
                            kept = [u for u in e["supersedes"] if u in allowed]
                            if kept:
                                e["supersedes"] = kept
                            else:
                                del e["supersedes"]
                return entries
            except _GpuInferenceError as exc:
                # CUDA 熔断：损坏模型禁入 GC（free 即 abort，2026-10-21 实证
                # evict 解引用后的任意 GC 时刻延迟引爆）——墓场保活＋熔断，
                # 第二轮 CPU 重建重试一次（designed 降级链保留）
                gpu_breaker.quarantine(model)
                gpu_breaker.trip(f"distill CUDA error: {exc}")
                self._force_cpu = True
                log.warning("GPU 蒸馏失败（%s）——熔断＋CPU 重建重试", exc)
                self._slot.evict("distill")  # 解引用损坏的 GPU 槽模型（不销毁）
            except Exception as exc:
                log.warning("蒸馏失败（%s: %s）——原文转直存", type(exc).__name__, exc)
                return None
            finally:
                self._slot.release()
        return None
