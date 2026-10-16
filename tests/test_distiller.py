"""Distiller mock 单测：非思考模板形状 / JSON 解析容错 / rationale 降级 /
fail-open / 设备回退 / 槽位契约。

全部走 mock（llama-cpp-python 本环境不可导入，mock 经类级 monkeypatch 不
触发其导入）；真件冒烟 opt-in：MEMORY_DISTILL_MODEL 指向真 GGUF 才跑，
否则 skip（tests/test_embedder_real.py 先例）。
"""
import logging
import os
from pathlib import Path

import pytest

from semantic_memory.distiller import SYSTEM_PROMPT, Distiller, format_prompt


class MockDistillModel:
    """create_completion 按序弹出 outputs；加载探针（无 <<<TEXT>>>）固定应答。"""

    def __init__(self, outputs=None, fail_generation=False):
        self.outputs = list(outputs or [])
        self.fail_generation = fail_generation
        self.prompts: list[str] = []
        self.close_calls = 0

    def create_completion(self, prompt, max_tokens=1, **_):
        self.prompts.append(prompt)
        if "<<<TEXT>>>" not in prompt:  # 加载探针
            return {"choices": [{"text": ""}]}
        if self.fail_generation:
            raise RuntimeError("CUDA error: out of memory")
        return {"choices": [{"text": self.outputs.pop(0) if self.outputs else "[]"}]}

    def close(self):
        self.close_calls += 1


@pytest.fixture
def gguf(tmp_path) -> Path:
    p = tmp_path / "distiller.gguf"
    p.write_bytes(b"mock")
    return p


def make_distiller(monkeypatch, gguf, mock, device="cpu") -> Distiller:
    monkeypatch.setattr(Distiller, "_new_model", lambda self, use_gpu: mock)
    return Distiller(gguf, device=device)


def test_prompt_shape_non_thinking(monkeypatch, gguf):
    """官方非思考模板：system/user/assistant 三段、无 <think> 引导。"""
    mock = MockDistillModel(outputs=["[]"])
    d = make_distiller(monkeypatch, gguf, mock)
    assert d.distill(["第一条原文", "第二条原文"], "github.com/a/b") == []

    prompt = mock.prompts[-1]
    assert prompt.startswith("<|im_start|>system\n")
    assert SYSTEM_PROMPT in prompt
    assert '"kind"' in prompt and '"rationale"' in prompt  # 输出契约进提示词
    assert "<|im_end|>\n<|im_start|>user\n" in prompt
    assert "Project: github.com/a/b" in prompt
    assert "第一条原文" in prompt and "第二条原文" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<think>" not in prompt  # 非思考：无 think 引导


def test_format_prompt_joins_and_wraps():
    prompt = format_prompt(["a", "b"], "proj")
    assert "a\n---\nb" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<think>" not in prompt


def test_distill_parses_valid_array(monkeypatch, gguf):
    raw = (
        '[{"content": "部署分支固定为 release/2026", "kind": "decision", '
        '"scope": "project", "rationale": "回滚方便", "tags": ["deploy", "分支"]}]'
    )
    mock = MockDistillModel(outputs=[raw])
    d = make_distiller(monkeypatch, gguf, mock)
    assert d.distill(["原文"], "github.com/a/b") == [{
        "content": "部署分支固定为 release/2026",
        "kind": "decision",
        "scope": "project",
        "rationale": "回滚方便",
        "tags": ["deploy", "分支"],
    }]


def test_markdown_fence_tolerated(monkeypatch, gguf):
    """LLM 输出裹 markdown 围栏/前后闲话：截首 [ 至末 ] 仍可解析。"""
    raw = (
        "好的，提取结果如下：\n```json\n"
        '[{"content": "c", "kind": "fact", "scope": "global", "tags": []}]\n'
        "```\n如需调整请告知。"
    )
    mock = MockDistillModel(outputs=[raw])
    d = make_distiller(monkeypatch, gguf, mock)
    assert d.distill(["原文"], "p") == [{
        "content": "c", "kind": "fact", "scope": "global",
        "rationale": "", "tags": [],
    }]


def test_decision_without_rationale_downgrades_to_fact(monkeypatch, gguf, caplog):
    raw = ('[{"content": "偏好 uv", "kind": "preference", "scope": "global", '
           '"rationale": "", "tags": []}]')
    mock = MockDistillModel(outputs=[raw])
    d = make_distiller(monkeypatch, gguf, mock)
    with caplog.at_level(logging.WARNING, logger="semantic_memory.distiller"):
        out = d.distill(["原文"], "p")
    assert out[0]["kind"] == "fact"
    assert any("rationale" in r.message for r in caplog.records)


@pytest.mark.parametrize("raw", [
    "抱歉，我无法处理这段文本。",   # 无方括号
    '[{"content": "x", "kind": "decision"',  # 截断 JSON
    "```",                          # 围栏无内容
])
def test_bad_json_fail_open(monkeypatch, gguf, raw):
    mock = MockDistillModel(outputs=[raw])
    d = make_distiller(monkeypatch, gguf, mock)
    assert d.distill(["原文"], "p") is None  # fail-open：调用方原文直存


def test_non_dict_entries_dropped(monkeypatch, gguf):
    mock = MockDistillModel(outputs=['[1, {"content": "ok", "kind": "fact"}]'])
    d = make_distiller(monkeypatch, gguf, mock)
    out = d.distill(["原文"], "p")
    assert out == [{"content": "ok", "kind": "fact", "scope": "project",
                    "rationale": "", "tags": []}]


def test_empty_texts_returns_none(monkeypatch, gguf):
    mock = MockDistillModel()
    d = make_distiller(monkeypatch, gguf, mock)
    assert d.distill([], "p") is None
    assert mock.prompts == []  # 未触发任何推理


def test_disabled_without_model():
    d = Distiller(None)
    assert d.available is False
    assert d.is_available() is False
    assert d.distill(["原文"], "p") is None
    assert Distiller(Path("/nonexistent/a.gguf")).available is False


def test_gpu_init_fallback_pid_check(monkeypatch, gguf):
    """加载链：GPU 探针通过但 nvidia-smi 无 pid → 判静默回退 → CPU 重建。"""
    mock = MockDistillModel(outputs=["[]"])
    monkeypatch.setattr(Distiller, "_new_model", lambda self, use_gpu: mock)
    monkeypatch.setattr("semantic_memory.distiller._pid_vram_on_gpu", lambda pid: None)
    d = Distiller(gguf, device="cuda")
    assert d.distill(["原文"], "p") == []
    assert d.device == "cpu"


def test_gpu_oom_rebuilds_cpu_and_retries_once(monkeypatch, gguf):
    """运行期 GPU 推理 OOM → 强制 CPU 重建、整次调用重试一次成功。"""
    gpu_mock = MockDistillModel(fail_generation=True)
    cpu_mock = MockDistillModel(outputs=["[]"])
    monkeypatch.setattr(
        Distiller, "_new_model",
        lambda self, use_gpu: gpu_mock if use_gpu else cpu_mock,
    )
    monkeypatch.setattr("semantic_memory.distiller._pid_vram_on_gpu", lambda pid: "100 MiB")
    d = Distiller(gguf, device="cuda")
    out = d.distill(["原文"], "p")

    assert d.device == "cpu" and out == []
    assert len(gpu_mock.prompts) == 2  # 探针 + 1 次失败推理（只重试一轮）
    assert gpu_mock.close_calls == 0 and cpu_mock.close_calls == 0  # 槽从不销毁
    assert cpu_mock.prompts[-1].startswith("<|im_start|>system\n")  # 重建后模板不变


def test_modelslot_shared_with_reranker_kind(monkeypatch, gguf):
    """槽位契约：distill 走 "distill" kind，与 rerank 互斥分时（LRU=1）。"""
    from semantic_memory.modelslot import ModelSlot

    slot = ModelSlot()
    mock = MockDistillModel(outputs=["[]"])
    d = make_distiller(monkeypatch, gguf, mock)
    d._slot = slot
    rerank_model = MockDistillModel()
    slot.acquire("rerank", lambda: rerank_model)
    slot.release()

    assert d.distill(["原文"], "p") == []  # 自动驱逐 rerank 换 distill
    assert slot.is_loaded("distill") and not slot.is_loaded("rerank")
    assert rerank_model.close_calls == 0 and mock.close_calls == 0


def test_config_distill_model_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DISTILL_MODEL", str(tmp_path / "m.gguf"))
    from semantic_memory.config import Config

    assert Config.from_env().distill_model == tmp_path / "m.gguf"
    monkeypatch.setenv("MEMORY_DISTILL_MODEL", "")
    assert Config.from_env().distill_model is None  # 空 = 蒸馏禁用


def test_config_rerank_model_default(monkeypatch, tmp_path):
    """三态语义：未设→常规 XDG 路径；显式空→关闭；空白填充→取值与判空一致。"""
    from semantic_memory.config import Config

    monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)  # conftest 隔离值不参与 tier 判定

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("MEMORY_RERANK_MODEL", raising=False)
    assert Config.from_env().rerank_model == (
        tmp_path / "memory-mcp" / "models" / "qwen3-reranker-0.6b-q8_0.gguf")
    monkeypatch.setenv("MEMORY_RERANK_MODEL", "")
    assert Config.from_env().rerank_model is None  # 空 = 重排禁用
    monkeypatch.setenv("MEMORY_RERANK_MODEL", f" {tmp_path / 'r.gguf'} ")
    assert Config.from_env().rerank_model == tmp_path / "r.gguf"  # 剥首尾空白
    monkeypatch.setenv("XDG_DATA_HOME", "")
    monkeypatch.delenv("MEMORY_RERANK_MODEL", raising=False)
    assert Config.from_env().rerank_model.is_absolute()  # XDG 设但空串也回退 $HOME


# ---------- 真件冒烟（opt-in：MEMORY_DISTILL_MODEL 指向真 GGUF 才跑） ----------

REAL_MODEL = os.environ.get("MEMORY_DISTILL_MODEL", "")


@pytest.mark.skipif(
    not REAL_MODEL or not Path(REAL_MODEL).exists(),
    reason="MEMORY_DISTILL_MODEL 未指向 GGUF 模型，跳过真实模型冒烟",
)
def test_real_model_smoke():
    """Qwen3-1.7B 非思考真蒸馏：一条带理由的决定应被提取为合法条目。"""
    d = Distiller(Path(REAL_MODEL), device="cpu")
    assert d.available
    out = d.distill(
        ["用户决定：部署分支固定为 release/2026，理由是发布出问题回滚更快"],
        "github.com/guangbin79/semantic-memory-mcp",
    )
    assert out is not None and out, "真模型应至少提取出一条"
    entry = out[0]
    assert set(entry) == {"content", "kind", "scope", "rationale", "tags"}
    assert entry["kind"] in ("fact", "decision", "preference", "artifact")
    assert entry["scope"] in ("project", "global")
    assert isinstance(entry["tags"], list)


def test_format_prompt_neighbors_section():
    """r5 接口扩展：neighbors 上下文进 user 段，uid 随内容列出供 supersedes 引用。"""
    prompt = format_prompt(["新原文"], "p", neighbors=[("abc123", "旧内容一")])
    assert "<<<MEMORIES>>>" in prompt
    assert "[abc123] 旧内容一" in prompt
    assert '"supersedes"' in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<think>" not in prompt


def test_format_prompt_without_neighbors_identical():
    """无 neighbors 时提示词与基础形态逐字节一致（基础契约回归锚）。"""
    assert format_prompt(["a"], "p", neighbors=None) == format_prompt(["a"], "p")


def test_supersedes_whitelist_unconditional(monkeypatch, gguf):
    """supersedes 白名单无条件生效：无 neighbors 时 allowed 为空集全部剥离；
    有 neighbors 时仅保留白名单内合法 uid（轮3 终闸 MAJOR 修复后契约）。"""
    raw = (
        '[{"content": "偏好 uv", "kind": "preference", "scope": "project",'
        ' "rationale": "r", "tags": [], "supersedes": ["nb1", 5, ""]},'
        ' {"content": "x", "kind": "fact", "supersedes": "nb1"}]'
    )
    mock = MockDistillModel(outputs=[raw])
    d = make_distiller(monkeypatch, gguf, mock)
    out = d.distill(["原文"], "p")  # 无 neighbors：模型 uid 无凭据，全剥
    assert "supersedes" not in out[0]
    assert "supersedes" not in out[1]
    d2 = make_distiller(monkeypatch, gguf, MockDistillModel(outputs=[raw]))
    out2 = d2.distill(["原文"], "p", neighbors=[("nb1", "旧内容")])  # 白名单内合法 uid 保留
    assert out2[0]["supersedes"] == ["nb1"]
    assert "supersedes" not in out2[1]
