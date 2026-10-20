"""嵌入输入防线（纯函数＋桩互斥，无需 GGUF 模型）。

clamp＝超窗纵深防御；serialized＝并发推理互斥（真根因回归，2026-10-20 GGML abort）。
"""
import numpy as np
import pytest


def test_clamp_embed_text():
    from semantic_memory.embedder import EMBED_TEXT_CHAR_BUDGET, clamp_embed_text

    assert clamp_embed_text("a" * (EMBED_TEXT_CHAR_BUDGET + 99)) == "a" * EMBED_TEXT_CHAR_BUDGET
    assert clamp_embed_text("短文本") == "短文本"


def test_embed_batch_serialized():
    # 并发推理互斥回归：llama_cpp.Llama 非线程安全，embed_batch 全程必须串行
    # （对无锁实现必红——sleep 窗口放大竞态；见 embedder._infer_lock）
    import threading
    import time

    from semantic_memory.embedder import Embedder

    e = Embedder.__new__(Embedder)  # 绕过 __init__：不加载真实模型
    e._infer_lock = threading.Lock()  # __new__ 不跑 __init__，锁需手动补齐（缺失=线程崩死=空转绿）
    e.available = True
    e.device = "cpu"
    e._ensure_loaded = lambda: None
    inside: list[int] = []
    calls: list[int] = []
    errors: list[Exception] = []

    def fake_embed_all(texts):
        calls.append(1)
        try:
            assert not inside, "检测到并发 _embed_all——推理互斥失效"
            inside.append(1)
            time.sleep(0.05)
            inside.pop()
            return np.zeros((len(texts), 4), dtype=np.float32)
        except Exception as exc:  # 线程内异常传不回主线程，手工收集
            errors.append(exc)
            return np.zeros((len(texts), 4), dtype=np.float32)

    e._embed_all = fake_embed_all
    threads = [threading.Thread(target=e.embed_batch, args=([f"t{k}"],)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 进入计数防空转：线程若在锁处崩死，fake 从未执行，errors=[] 的断言会空转绿
    assert len(calls) == 6, f"fake_embed_all 仅进入 {len(calls)}/6——线程在锁前崩死，测试无效"
    assert not errors, f"并发互斥失效: {errors}"
