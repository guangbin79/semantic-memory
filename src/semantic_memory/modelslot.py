"""线程安全共享模型槽：Reranker 与 Distiller 分时复用同一槽位（LRU=1）。

并发语义（.omo/plans/context-completion-evolution.md task3，r3 评审钉死）：
- 单一 threading.Lock 守护槽状态；换载（加载）在锁内完成（单槽本就串行）；
- call-scoped hold：acquire() 返回的模型引用由调用方在单次推理期间持有，
  推理本身在锁外进行，用毕 release()；
- 引用持有式驱逐：换载/驱逐只解引用槽内旧模型，绝不 close/销毁——
  仍被持有中的模型由调用方引用保活至 release，随后交 GC。双线程
  （请求线程 rerank + daemon flush 蒸馏）下无 use-after-free；
- 嵌入器常驻、不经本槽（Embedder 自持模型，独占显存/内存）。
"""
from __future__ import annotations

import threading
from typing import Any, Callable


class ModelSlot:
    """任意时刻至多一个模型 resident 的共享槽。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kind: str | None = None  # 当前 resident 种类："rerank" | "distill" | ...
        self._model: Any = None
        self._holds = 0                # 未 release 的 acquire 数（跨代诊断计数）

    def acquire(self, kind: str, loader: Callable[[], Any]) -> Any:
        """取 kind 模型引用，必要时经 loader 换载；调用方用毕必须 release()。

        同 kind 已 resident → 直接复用；异 kind → 先解引用旧模型再换载。
        旧模型即使仍被他线程推理中也只解引用、不销毁（引用持有式驱逐）。
        """
        with self._lock:
            if self._model is not None and self._kind == kind:
                self._holds += 1
                return self._model
            self._model = None  # 引用持有式驱逐：仅解引用，持有方引用照常保活
            self._kind = None
            self._model = loader()
            self._kind = kind
            self._holds += 1
            return self._model

    def release(self) -> None:
        """归还一次 acquire 持有。不销毁任何模型——生命周期归 GC。"""
        with self._lock:
            self._holds = max(0, self._holds - 1)

    def is_loaded(self, kind: str) -> bool:
        with self._lock:
            return self._kind == kind and self._model is not None

    @property
    def in_use(self) -> int:
        """当前未 release 的持有数（诊断/测试用）。"""
        with self._lock:
            return self._holds

    def evict(self, kind: str | None = None) -> None:
        """解引用槽内模型（kind 给定时仅当种类匹配才动手）。"""
        with self._lock:
            if kind is None or self._kind == kind:
                self._model = None
                self._kind = None
