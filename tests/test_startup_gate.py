"""分层严格启动契约测试（任务 9/R2）：门函数独立单测 + main() 接线。"""
import io
import logging
import sys
from dataclasses import replace

import pytest

from semantic_memory.config import Config
from semantic_memory.startup_gate import check_startup


def clean_env(monkeypatch, tmp_path):
    """env 清场 + 数据目录隔离（config.toml 落点随 tmp）。"""
    for var in ("MEMORY_MODEL_PATH", "MEMORY_DISTILL_MODEL", "MEMORY_ALLOW_BM25_ONLY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
    return tmp_path


def gate(cfg: Config) -> tuple[int, str]:
    err = io.StringIO()
    return check_startup(cfg, err=err), err.getvalue()


# ---------------- 判定序（R2） ----------------

def test_allow_bm25_only_always_passes(tmp_path):
    code, err = gate(Config.from_env())  # 基线：无配置 → 放行（下方复用作对照）
    assert code == 0
    cfg = replace(Config.from_env(), allow_bm25_only=True,
                  model_path=tmp_path / "missing.gguf")  # 缺失也放行
    code, err = gate(cfg)
    assert code == 0 and err == ""


def test_env_configured_present_runs_as_usual(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    model = tmp_path / "embed.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setenv("MEMORY_MODEL_PATH", str(model))
    code, err = gate(Config.from_env())
    assert code == 0 and err == ""


def test_env_configured_missing_refuses(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    missing = tmp_path / "nope.gguf"
    monkeypatch.setenv("MEMORY_MODEL_PATH", str(missing))
    code, err = gate(Config.from_env())
    assert code == 2
    assert str(missing) in err                          # stderr 指明路径
    assert "semantic-memory setup --profile embed" in err  # 修复提示


def test_distill_configured_missing_refuses(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    missing = tmp_path / "nope-distill.gguf"
    monkeypatch.setenv("MEMORY_DISTILL_MODEL", str(missing))
    code, err = gate(Config.from_env())
    assert code == 2
    assert str(missing) in err
    assert "semantic-memory setup --profile full" in err


def test_all_unconfigured_recruitment_mode(monkeypatch, tmp_path, caplog):
    clean_env(monkeypatch, tmp_path)
    with caplog.at_level(logging.INFO, logger="semantic_memory.startup_gate"):
        code, err = gate(Config.from_env())
    assert code == 0 and err == ""
    assert "招募模式" in caplog.text and "semantic-memory setup" in caplog.text


def test_config_file_counts_as_configured(monkeypatch, tmp_path):
    data_dir = clean_env(monkeypatch, tmp_path)
    (data_dir / "config.toml").write_text(
        f'model_path = "{tmp_path / "gone.gguf"}"\n', encoding="utf-8")
    code, err = gate(Config.from_env())
    assert code == 2 and "gone.gguf" in err              # 文件档已配置缺失同样拒启


def test_allow_bm25_only_via_env(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMORY_MODEL_PATH", str(tmp_path / "missing.gguf"))
    monkeypatch.setenv("MEMORY_ALLOW_BM25_ONLY", "1")
    code, err = gate(Config.from_env())
    assert code == 0 and err == ""


def test_empty_env_means_disabled_not_configured(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMORY_MODEL_PATH", "")          # 空串=显式禁用（既有语义）
    code, err = gate(Config.from_env())
    assert code == 0                                     # 未配置 → 招募放行


# ---------------- main() 接线：拒启先于端口绑定 ----------------

def test_main_refuses_before_serve(monkeypatch, tmp_path):
    from semantic_memory import server

    bad = replace(server.cfg, model_path=tmp_path / "missing.gguf",
                  distill_model=None, allow_bm25_only=False)
    monkeypatch.setattr(server, "cfg", bad)

    def must_not_bind(*a, **kw):
        raise AssertionError("拒启后不得绑定端口/进入 serve")

    monkeypatch.setattr(server, "serve", must_not_bind)
    monkeypatch.setattr(sys, "argv", ["semantic-memory"])
    with pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 2


def test_main_recruitment_reaches_serve(monkeypatch, tmp_path):
    from semantic_memory import server

    ok = replace(server.cfg, model_path=None, distill_model=None,
                 allow_bm25_only=False, maintenance_on_start=False)
    monkeypatch.setattr(server, "cfg", ok)
    monkeypatch.setattr(server, "serve", lambda: None)   # 起服被替换为无操作标记
    monkeypatch.setattr(sys, "argv", ["semantic-memory"])
    server.main()                                        # 不抛 SystemExit 即放行
