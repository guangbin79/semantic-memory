"""CUDA 熔断器测试：trip/墓场语义＋三条模型路径的 fail-open 短路。

每条对应 2026-10-21 第 4 类服务死亡的防线：CUDA 上下文损坏后
任何 llama 触碰（推理/GPU 模型 free/CPU 重建）都可能 GGML abort 杀进程——
损坏模型入墓场（禁入 GC）＋熔断后全路径 fail-open，服务以 BM25 存活。
"""
import threading

import pytest

from semantic_memory import gpu_breaker
from semantic_memory.db import MemoryStore
from semantic_memory.embedder import Embedder
from semantic_memory.lifecycle import enqueue_raw, flush_distill_queue
from semantic_memory.reranker import Reranker


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _clean_breaker():
    gpu_breaker.reset_for_test()
    yield
    gpu_breaker.reset_for_test()


def _embedder_stub(device="cuda"):
    """最小可用桩：_ensure_loaded 直通、_embed_all 可注入、GPU 模型可观测。"""
    e = Embedder.__new__(Embedder)
    e.available = True
    e.device = device
    e._model = object()  # 桩模型对象（墓场 quarantine 的观测目标）
    e._infer_lock = threading.Lock()
    return e


def test_trip_is_idempotent_and_keeps_first_reason():
    gpu_breaker.trip("first")
    gpu_breaker.trip("second")
    assert gpu_breaker.tripped() is True
    assert gpu_breaker.reason() == "first"


def test_embedder_tripped_returns_none_without_inference():
    gpu_breaker.trip("pre-tripped")
    e = _embedder_stub()

    def boom():
        raise AssertionError("熔断后不得触碰推理")

    e._embed_all = boom
    assert e.embed_batch(["x"]) is None


def test_embedder_cuda_error_quarantines_rebuilds_then_fails_open():
    # 回归：CUDA 上下文损坏后，模型 free（GC）与 CPU 重建路径都会 abort 杀进程
    # → 损坏模型入墓场（禁入 GC）＋CPU 重建重试一次；重试再败 → trip＋fail-open
    e = _embedder_stub()

    calls: list[str] = []

    def embed_all(texts):
        calls.append("embed")
        raise RuntimeError("CUDA error")

    e._embed_all = embed_all
    rebuilds: list[Exception] = []
    e._rebuild_on_cpu = lambda exc: rebuilds.append(exc)

    assert e.embed_batch(["x"]) is None
    assert gpu_breaker.tripped() is True
    assert "embed" in gpu_breaker.reason()
    assert gpu_breaker.quarantined_count() == 1  # 损坏模型入墓场
    assert len(rebuilds) == 1  # CPU 重建重试发生过一次（墓场已消除 GC 引爆）
    assert calls.count("embed") == 2  # GPU 失败＋CPU 重试各一次


def test_embedder_cpu_error_still_raises_no_trip():
    # CPU 设备的 Python 异常不属于 CUDA 熔断语义：原样抛出、不熔断
    e = _embedder_stub(device="cpu")

    def boom(texts):
        raise RuntimeError("cpu bug")

    e._embed_all = boom
    with pytest.raises(RuntimeError):
        e.embed_batch(["x"])
    assert gpu_breaker.tripped() is False
    assert gpu_breaker.quarantined_count() == 0


def test_reranker_tripped_returns_none():
    gpu_breaker.trip("pre-tripped")
    r = Reranker.__new__(Reranker)
    r.available = True

    def boom():
        raise AssertionError("熔断后不得触碰槽/推理")

    r._slot = type("_S", (), {"acquire": staticmethod(boom)})()
    assert r.rerank("q", [("u", "c")]) is None


def test_flush_distill_exception_trips_breaker(store):
    class BoomDistiller:
        configured = True

        def is_available(self):
            return True

        def distill(self, texts, project, neighbors=None):
            raise RuntimeError("CUDA error during distill")

    enqueue_raw(store, "蒸馏异常原文", "chat", "p", None)
    report = flush_distill_queue(store, BoomDistiller())
    assert report["left_pending"] == 1  # fail-closed：留 pending
    assert gpu_breaker.tripped() is True  # 并熔断全模型路径


def test_health_reports_gpu_degraded(client):
    # 熔断态经 /health 暴露（可观测，不静默）——R1 INFO 项的覆盖补齐
    gpu_breaker.trip("unit-test")
    try:
        data = client.get("/health").json()
        assert data["gpu_degraded"] == "unit-test"
    finally:
        gpu_breaker.reset_for_test()


def test_flush_tripped_drains_raw(store):
    # 已熔断：无裁判可依 → pending 原文直存排空（fail-open 唯一兜底，不堆积）
    gpu_breaker.trip("pre-tripped")

    class _NoopDistiller:
        configured = True

        def is_available(self):
            return True

        def distill(self, texts, project, neighbors=None):  # 熔断期不得被调用
            raise AssertionError("已熔断时蒸馏不得运行")

    enqueue_raw(store, "熔断期捕获的真实用户消息", "chat", "p", None)
    report = flush_distill_queue(store, _NoopDistiller())
    assert report["typed"] == 0 and report["stored_raw"] == 1 and report["left_pending"] == 0
    assert store.stats()["total"] == 1


def test_recall_reports_gpu_degraded_but_bm25_works(client):
    """熔断期：/recall 响应带 gpu_degraded 标记（插件据此停止注入），
    但 BM25 召回本体照常（memory_recall 工具显式调用仍可用）。"""
    gpu_breaker.trip("unit-test")
    try:
        r = client.post("/recall", json={"query": "anything", "limit": 3})
        data = r.json()
        assert data["gpu_degraded"] == "unit-test"
        assert "hits" in data  # BM25 召回照常（本例空库 → 空 hits）
    finally:
        gpu_breaker.reset_for_test()


def test_global_infer_lock_serializes_cross_model(store):
    # 四类崩溃审计结论：embedder（常驻）× 槽内模型并发解码 → CUDA abort。
    # INFER_LOCK 必须让 embed 与 rerank 的 GPU 推理窗口互斥（跨模型）。
    import time as time_mod

    from test_reranker import MockRerankModel

    gpu_breaker.reset_for_test()
    active: list[str] = []
    overlaps: list[str] = []
    errors: list[Exception] = []

    def tracker(name, fn):
        def w(*a, **k):
            active.append(name)
            others = [x for x in active if x != name]
            if others:
                overlaps.append(f"{name} vs {others}")
            try:
                time_mod.sleep(0.02)
                return fn(*a, **k)
            except Exception as exc:
                errors.append(exc)
                return None
            finally:
                active.remove(name)
        return w

    e = _embedder_stub()
    import numpy as np
    e._embed_all = tracker("embed", lambda texts: np.zeros((len(texts), 4), dtype=np.float32))

    r = Reranker.__new__(Reranker)
    r.available = True
    r.device = "cuda"
    r.budget_ms = 5000
    r._yes_ids = ["yes", "no"]
    r._resolve_ids = lambda model: None
    r._score_from = lambda model, resp: 0.5
    from semantic_memory.modelslot import ModelSlot
    r._slot = ModelSlot()
    r._force_cpu = False
    dummy = object()
    r._infer = tracker("rerank", lambda model, prompt: {"choices": [{"text": "yes"}]})
    r._load = lambda: dummy

    threads = []
    for k in range(3):
        threads.append(threading.Thread(target=e.embed_batch, args=([f"x{k}"],)))
        threads.append(threading.Thread(target=r.rerank, args=("q", [("u", "doc")])))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"推理错误: {errors}"
    assert not overlaps, f"检测到跨模型并发推理: {overlaps[:3]}"
