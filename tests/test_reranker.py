"""Reranker mock 单测：模板形状 / 排序 / fail-open / 预算守卫 / 设备回退。

spike① 结论为 SKIPPED_NO_MODEL（按 FAIL 裁定默认值）：本文件全部走 mock，
MEMORY_RERANK_* 默认保持空，真件验证顺延 F3（llama-cpp-python 本环境
不可导入，mock 路径经类级 monkeypatch 不触发其导入）。
"""
import time
from pathlib import Path

import pytest

from memory_mcp.reranker import Reranker, format_prompt

SYS_SENTENCE = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no"'
)


def _lp(top: dict) -> dict:
    return {"choices": [{"logprobs": {"top_logprobs": [top]}}]}


class MockRerankModel:
    """create_completion 按序弹出 scores；加载探针（无 <Document>:）固定应答。"""

    def __init__(self, scores=None, delay=0.0, fail_docs=False):
        self.scores = list(scores or [])
        self.delay = delay
        self.fail_docs = fail_docs
        self.prompts: list[str] = []
        self.close_calls = 0

    def create_completion(self, prompt, max_tokens=1, logprobs=20, **_):
        self.prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        if "<Document>:" not in prompt:  # 加载探针
            return _lp({"yes": -0.7, "no": -0.7})
        if self.fail_docs:
            raise RuntimeError("CUDA error: out of memory")
        return _lp(self.scores.pop(0) if self.scores else {"yes": -0.1, "no": -3.0})

    def close(self):
        self.close_calls += 1


@pytest.fixture
def gguf(tmp_path) -> Path:
    p = tmp_path / "reranker.gguf"
    p.write_bytes(b"mock")
    return p


def make_reranker(monkeypatch, gguf, mock, device="cpu", budget_ms=500) -> Reranker:
    monkeypatch.setattr(Reranker, "_new_model", lambda self, use_gpu: mock)
    return Reranker(gguf, device=device, budget_ms=budget_ms)


def test_prompt_template_official_shape(monkeypatch, gguf):
    """官方判别模板：system 句 / Instruct-Query-Document / 空 think 后缀 / 512 截断。"""
    mock = MockRerankModel(scores=[{"yes": -0.1, "no": -3.0}])
    rr = make_reranker(monkeypatch, gguf, mock)
    long_doc = "A" * 600 + "TAIL"
    result = rr.rerank("部署用哪个分支", [("u1", long_doc)])

    assert result == [("u1", pytest.approx(0.9478, abs=1e-3))]
    prompt = mock.prompts[-1]
    assert prompt.startswith("<|im_start|>system\n")
    assert SYS_SENTENCE in prompt
    assert "<|im_end|>\n<|im_start|>user\n" in prompt
    assert "<Instruct>: Given a web search query" in prompt
    assert "<Query>: 部署用哪个分支" in prompt
    assert "<Document>: " + "A" * 512 in prompt
    assert "TAIL" not in prompt  # 文档截 512 字符
    assert prompt.endswith("</think>\n\n")  # 空 think 引导后缀


def test_format_prompt_short_doc_untouched():
    assert format_prompt("ins", "q", "doc") == (
        "<|im_start|>system\n" + SYS_SENTENCE
        + "<|im_end|>\n<|im_start|>user\n"
        + "<Instruct>: ins\n<Query>: q\n<Document>: doc"
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_rerank_order_by_yes_logprob(monkeypatch, gguf):
    mock = MockRerankModel(scores=[
        {"no": -0.1, "yes": -3.0},   # 候选顺序在前，相关分低
        {"no": -3.0, "yes": -0.1},   # 相关候选应被排到第一
    ])
    rr = make_reranker(monkeypatch, gguf, mock)
    result = rr.rerank("部署用哪个分支",
                       [("u_noise", "今天天气不错"), ("u_rel", "部署用 release 分支")])
    assert result is not None
    assert [uid for uid, _ in result] == ["u_rel", "u_noise"]
    assert 0.9 < result[0][1] < 0.99
    assert 0.0 < result[1][1] < 0.1


@pytest.mark.parametrize("top", [
    {"maybe": -0.1, "nope": -2.0},  # 两侧皆缺：响应契约破坏，fail-open
])
def test_missing_logprobs_fail_open(monkeypatch, gguf, top):
    mock = MockRerankModel(scores=[top])
    rr = make_reranker(monkeypatch, gguf, mock)
    assert rr.rerank("q", [("u1", "doc")]) is None


def test_missing_side_clamped_orders_by_present_side(monkeypatch, gguf):
    """真机 F3：无关候选的一侧常整掉出 top-K——缺失侧钳底保序，不 fail-open。"""
    mock = MockRerankModel(scores=[{"yes": -0.1}, {"no": -0.1}])
    rr = make_reranker(monkeypatch, gguf, mock)
    result = rr.rerank("q", [("u_yes", "doc rel"), ("u_no", "doc irr")])
    assert result is not None
    scores = dict(result)
    assert scores["u_yes"] > 0.5 > scores["u_no"]  # 在场侧占优



def test_budget_timeout_returns_none(monkeypatch, gguf):
    mock = MockRerankModel(delay=0.3)  # 单次推理就吃满预算
    rr = make_reranker(monkeypatch, gguf, mock, budget_ms=50)
    result = rr.rerank("q", [("u1", "a"), ("u2", "b")])
    assert result is None  # 超预算整体 fail-open


def test_budget_exhausted_by_load_skips_inference(monkeypatch, gguf):
    mock = MockRerankModel()
    rr = make_reranker(monkeypatch, gguf, mock, budget_ms=0)  # 含加载的预算即耗尽
    assert rr.rerank("q", [("u1", "a")]) is None
    assert all("<Document>:" not in p for p in mock.prompts)  # 一次推理都没跑


def test_disabled_without_model():
    assert Reranker(None).available is False
    assert Reranker(None).rerank("q", [("u1", "a")]) is None
    assert Reranker(Path("/nonexistent/a.gguf")).available is False


def test_gpu_init_fallback_pid_check(monkeypatch, gguf):
    """加载链：GPU 探针通过但 nvidia-smi 无 pid → 判静默回退 → CPU 重建。"""
    import memory_mcp.reranker as mod

    mock = MockRerankModel(scores=[{"yes": -0.1, "no": -3.0}])
    monkeypatch.setattr(Reranker, "_new_model", lambda self, use_gpu: mock)
    monkeypatch.setattr(mod, "_pid_vram_on_gpu", lambda pid: None)
    rr = Reranker(gguf, device="cuda")
    result = rr.rerank("q", [("u1", "doc")])
    assert rr.device == "cpu"
    assert result is not None and result[0][1] > 0.9


def test_gpu_oom_rebuilds_cpu_and_retries_once(monkeypatch, gguf):
    """运行期 GPU 推理 OOM → 强制 CPU 重建、整次调用重试一次成功。"""
    import memory_mcp.reranker as mod

    gpu_mock = MockRerankModel(fail_docs=True)
    cpu_mock = MockRerankModel(scores=[{"yes": -0.1, "no": -3.0}])
    monkeypatch.setattr(
        Reranker, "_new_model",
        lambda self, use_gpu: gpu_mock if use_gpu else cpu_mock,
    )
    monkeypatch.setattr(mod, "_pid_vram_on_gpu", lambda pid: "100 MiB")
    rr = Reranker(gguf, device="cuda")
    result = rr.rerank("q", [("u1", "doc")])

    assert rr.device == "cpu"
    assert result == [("u1", pytest.approx(0.9478, abs=1e-3))]
    assert len(gpu_mock.prompts) == 2  # 探针 + 1 次失败推理（只重试一轮）
    assert gpu_mock.close_calls == 0 and cpu_mock.close_calls == 0  # 槽从不销毁
    assert cpu_mock.prompts[-1].startswith("<|im_start|>system\n")  # 重建后模板不变


def test_modelslot_shared_with_distiller_kind(monkeypatch, gguf):
    """槽位契约：rerank 走 "rerank" kind，与 distill 互斥分时（LRU=1）。"""
    from memory_mcp.modelslot import ModelSlot

    slot = ModelSlot()
    mock = MockRerankModel(scores=[{"yes": -0.1, "no": -3.0}])
    rr = make_reranker(monkeypatch, gguf, mock)
    rr._slot = slot
    distill = MockRerankModel()
    slot.acquire("distill", lambda: distill)
    slot.release()

    assert rr.rerank("q", [("u1", "doc")]) is not None  # 自动驱逐 distill 换 rerank
    assert slot.is_loaded("rerank") and not slot.is_loaded("distill")
    assert distill.close_calls == 0 and mock.close_calls == 0
