"""rerank_model=auto（缺省）：按启动时空闲显存决定 2/3 模型常驻。

阈值依据 2026-10-29 真机 spike（RTX 3060 Laptop 6GB，ModelSlot 分时废弃、
三模型各自常驻拓扑）：per-pid 常驻实测 4802 MiB（embed 1956 / rerank +1026 /
distill +1820），取整 4800 + 余量 512 = 5312 MiB。
"""
from pathlib import Path

from semantic_memory.config import Config, auto_rerank_enabled

THRESHOLD = 4800 + 512
_ABSENT = object()


def _cfg(monkeypatch, tmp_path, rerank_env):
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
    if rerank_env is _ABSENT:
        monkeypatch.delenv("MEMORY_RERANK_MODEL", raising=False)
    else:
        monkeypatch.setenv("MEMORY_RERANK_MODEL", rerank_env)
    return Config.from_env()


# ---- auto 判定纯函数 ----

def test_auto_on_when_free_above_threshold():
    assert auto_rerank_enabled(5792, "cuda") is True


def test_auto_boundary_exactly_threshold():
    assert auto_rerank_enabled(THRESHOLD, "cuda") is True


def test_auto_off_when_free_below_threshold():
    assert auto_rerank_enabled(THRESHOLD - 1, "cuda") is False


def test_auto_off_when_free_unknown():
    # nvidia-smi 缺失/查询失败：宁可降级也不赌 OOM
    assert auto_rerank_enabled(None, "cuda") is False


def test_auto_cpu_has_no_vram_constraint():
    assert auto_rerank_enabled(None, "cpu") is True


# ---- rerank_model 三态解析（env 档；文件档同一 opt 通道）----

def test_default_is_auto(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, _ABSENT)
    assert cfg.rerank_auto is True
    assert cfg.rerank_model is None


def test_explicit_auto(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, "auto")
    assert cfg.rerank_auto is True
    assert cfg.rerank_model is None


def test_empty_string_disables(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, "")
    assert cfg.rerank_auto is False
    assert cfg.rerank_model is None


def test_path_forces_on(monkeypatch, tmp_path):
    cfg = _cfg(monkeypatch, tmp_path, "/x/r.gguf")
    assert cfg.rerank_auto is False
    assert cfg.rerank_model == Path("/x/r.gguf")
