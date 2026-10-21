"""CUDA 熔断器：进程内 llama.cpp 触发 CUDA 错误后，全模型路径 fail-open。

2026-10-21 实证（第 4 类服务死亡族）：CUDA 上下文一旦报错即不可逆损坏——
此后任何触碰 llama 的操作（推理/GPU 模型销毁/CPU 重建时的上下文清理）
都可能 GGML abort 直接杀进程，Python 层不可捕获。embedder 的 _infer_lock
互斥只防并发竞态，防不了损坏上下文的后续 abort。

熔断语义：首个 CUDA 类错误 → trip(reason) → 三条模型路径全部短路
fail-open：embed 返回 None（召回退 BM25）、rerank 返回 None（保 RRF 序）、
蒸馏 flush 原文直存排空（fail-open 不堆积）。服务以纯 BM25 存活直到重启，
状态经 /health 的 gpu_degraded 暴露（可观测，不静默）。
"""
from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
INFER_LOCK = threading.Lock()
"""进程级 GPU 推理全局互斥：三条模型路径（embedder/reranker/distiller）的
加载与推理都先持此锁。背景（2026-10-21 四类崩溃审计）：llama.cpp 的 CUDA
后端状态跨 context 共享且非线程安全——embedder（常驻）与槽内模型并发解码
即触发 CUDA error/GGML abort（第 4 类死亡的机制）。锁顺序恒定：
embedder 为 _infer_lock→INFER_LOCK，槽路径为 INFER_LOCK（槽锁已先行释放），
无嵌套交叉，无死锁面。单机单用户下串行化的延迟代价可忽略。"""
_tripped = False
_reason = ""


def trip(reason: str) -> None:
    """熔断（幂等）：首次调用记录原因，后续调用不覆盖。"""
    global _tripped, _reason
    with _lock:
        if not _tripped:
            _tripped = True
            _reason = reason


def tripped() -> bool:
    return _tripped


def reason() -> str:
    return _reason


def reset_for_test() -> None:
    """仅供测试：清空熔断态与墓场。"""
    global _tripped, _reason
    with _lock:
        _tripped = False
        _reason = ""
        _graveyard.clear()


_graveyard: list = []


def quarantine(model: Any) -> None:
    """损坏模型墓场：强引用保活、禁入 GC。

    llama free 损坏 CUDA 上下文的模型会 abort 杀进程（2026-10-21 实证：
    evict 解引用后的任意 GC 时刻延迟引爆）。墓场用内存泄漏换进程存活，
    重启即释。
    """
    with _lock:
        _graveyard.append(model)


def quarantined_count() -> int:
    with _lock:
        return len(_graveyard)
