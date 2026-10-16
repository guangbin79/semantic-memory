"""ModelSlot 并发契约：引用持有式驱逐（in-use 模型绝不被销毁）+ LRU=1。

真实双线程（threading.Thread + Event 同步）验证：一边 mock 推理中，
另一边换载驱逐——断言 in-use 模型仍可用、close 从未被槽调用、
任意时刻至多一个 resident。
"""
import threading

from memory_mcp.modelslot import ModelSlot


class MockSlotModel:
    def __init__(self, name: str):
        self.name = name
        self.close_calls = 0

    def infer(self) -> str:
        return f"infer:{self.name}"

    def close(self) -> None:
        self.close_calls += 1


def _run_threaded(targets, timeout: float = 5.0) -> list[Exception]:
    """并发跑 targets，收集线程内异常到主线程断言（线程内 assert 不外抛）。"""
    errors: list[Exception] = []

    def guard(fn):
        def run():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - 测试收集点
                errors.append(exc)
        return run

    threads = [threading.Thread(target=guard(t)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
        assert not t.is_alive(), "线程未在超时内完成（疑似死锁）"
    return errors


def test_eviction_keeps_inuse_model_alive():
    """双线程：A 持 rerank 推理中，B acquire distill 换载驱逐——A 的模型不被销毁。"""
    slot = ModelSlot()
    model_x = MockSlotModel("rerank")
    model_y = MockSlotModel("distill")
    a_acquired = threading.Event()
    evict_done = threading.Event()
    results: dict[str, str] = {}

    def thread_a():
        m = slot.acquire("rerank", lambda: model_x)
        assert m is model_x
        a_acquired.set()
        assert evict_done.wait(5), "未等到 B 完成换载驱逐"
        results["a_infer"] = m.infer()  # 驱逐后 in-use 模型必须仍然可用
        slot.release()

    def thread_b():
        assert a_acquired.wait(5), "未等到 A 持有模型"
        m = slot.acquire("distill", lambda: model_y)  # 换载：解引用 X 换 Y
        assert m is model_y
        evict_done.set()
        assert not slot.is_loaded("rerank")  # 任意时刻至多一个 resident
        slot.release()

    errors = _run_threaded([thread_a, thread_b])
    assert not errors, errors
    assert results["a_infer"] == "infer:rerank"  # in-use 模型未 use-after-free
    assert model_x.close_calls == 0 and model_y.close_calls == 0  # 槽从不销毁
    assert slot.is_loaded("distill") and not slot.is_loaded("rerank")
    assert slot.in_use == 0


def test_same_kind_reuses_resident_model():
    slot = ModelSlot()
    calls = []

    def loader():
        calls.append(1)
        return MockSlotModel("rerank")

    m1 = slot.acquire("rerank", loader)
    m2 = slot.acquire("rerank", loader)
    assert m1 is m2 and len(calls) == 1  # 同 kind 快路径复用，loader 只跑一次
    slot.release()
    slot.release()
    assert slot.in_use == 0


def test_evict_kind_scoped():
    slot = ModelSlot()
    slot.acquire("rerank", lambda: MockSlotModel("rerank"))
    slot.release()
    slot.acquire("distill", lambda: MockSlotModel("distill"))
    slot.evict("rerank")  # 种类不匹配 → no-op
    assert slot.is_loaded("distill")
    slot.evict("distill")
    assert not slot.is_loaded("distill")
    slot.release()
    slot.evict()  # 空槽 evict 安全
    assert slot.in_use == 0


def test_evict_does_not_destroy_inuse_model():
    slot = ModelSlot()
    model = MockSlotModel("rerank")
    m = slot.acquire("rerank", lambda: model)
    slot.evict()
    assert m.infer() == "infer:rerank"  # evict 只解引用，持有方引用照常保活
    assert model.close_calls == 0
    slot.release()


def test_concurrent_storm_never_destroys_models():
    """4 线程交替争抢两种 kind：全程无销毁、结束后恰好一个 resident。"""
    slot = ModelSlot()
    created: list[MockSlotModel] = []
    lock = threading.Lock()

    def loader(name):
        def make():
            m = MockSlotModel(name)
            with lock:
                created.append(m)
            return m
        return make

    def worker(kind: str, other: str):
        for i in range(25):
            k = kind if i % 2 == 0 else other
            m = slot.acquire(k, loader(k))
            assert m.infer() == f"infer:{k}"
            slot.release()

    errors = _run_threaded([lambda: worker("rerank", "distill"),
                            lambda: worker("distill", "rerank"),
                            lambda: worker("rerank", "distill"),
                            lambda: worker("distill", "rerank")])
    assert not errors, errors
    assert all(m.close_calls == 0 for m in created)  # 槽从不销毁任何模型
    assert slot.is_loaded("rerank") != slot.is_loaded("distill")  # 恰一个 resident
    assert slot.in_use == 0
