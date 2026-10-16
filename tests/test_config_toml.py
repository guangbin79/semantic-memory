"""config.toml 加载（D4）：env > 文件 > 内置默认；write_config 密钥 chmod 600。"""
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from semantic_memory.config import Config, load_config_file, write_config

REPO = Path(__file__).resolve().parent.parent


def _data_dir(monkeypatch, tmp_path):
    dd = tmp_path / "data-home"
    monkeypatch.setenv("MEMORY_DATA_DIR", str(dd))
    for leak in ("MEMORY_DB_PATH", "MEMORY_DISCOVERY_FILE", "MEMORY_RERANK_MODEL"):
        monkeypatch.delenv(leak, raising=False)  # 其它测试模块 import 期设过的全局 env（如 test_telemetry 的 db 路径）
    return dd


def _write_toml(dd: Path, text: str) -> Path:
    dd.mkdir(parents=True, exist_ok=True)
    p = dd / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_priority_env_over_file_over_default(monkeypatch, tmp_path):
    dd = _data_dir(monkeypatch, tmp_path)
    _write_toml(dd, 'host = "127.0.0.2"\nport = 9000\ndevice = "cpu"\n')
    cfg = Config.from_env()
    assert cfg.host == "127.0.0.2"   # 文件值
    assert cfg.port == 9000
    assert cfg.device == "cpu"
    assert cfg.recall_limit_default == 5  # 全未配置 → 内置默认
    monkeypatch.setenv("MEMORY_PORT", "9100")
    cfg = Config.from_env()
    assert cfg.port == 9100          # env 压过文件
    assert cfg.host == "127.0.0.2"   # 未设 env 的字段仍取文件
    monkeypatch.setenv("MEMORY_HOST", "127.0.0.3")
    assert Config.from_env().host == "127.0.0.3"


def test_syntax_error_names_file_and_line(monkeypatch, tmp_path):
    dd = _data_dir(monkeypatch, tmp_path)
    p = _write_toml(dd, 'port = [unclosed\nhost = "x"\n')
    with pytest.raises(ValueError) as ei:
        Config.from_env()
    assert str(p) in str(ei.value)   # 指明文件
    assert "line" in str(ei.value)   # 带行号（tomllib 消息内含 at line X）


def test_missing_file_and_missing_keys_fall_back(monkeypatch, tmp_path):
    dd = _data_dir(monkeypatch, tmp_path)
    cfg = Config.from_env()  # 无 config.toml → 全默认
    assert cfg.port == 8765 and cfg.token is None
    assert cfg.db_path == dd / "data" / "memory.db"
    assert cfg.rerank_model == dd / "models" / "qwen3-reranker-0.6b-q8_0.gguf"
    _write_toml(dd, 'device = "cpu"\n')  # 缺键回落：其余字段仍默认
    cfg = Config.from_env()
    assert cfg.device == "cpu" and cfg.port == 8765 and cfg.token is None


def test_env_single_field_and_empty_string_semantics(monkeypatch, tmp_path):
    dd = _data_dir(monkeypatch, tmp_path)
    _write_toml(dd, f'device = "cpu"\nrerank_model = "{tmp_path / "r.gguf"}"\n')
    monkeypatch.setenv("MEMORY_DEVICE", "CUDA")
    assert Config.from_env().device == "cuda"  # env 单项覆盖 + 既有 lower()
    monkeypatch.setenv("MEMORY_RERANK_MODEL", "")  # env 显式空 = 禁用，压过文件值
    assert Config.from_env().rerank_model is None


def test_write_config_secret_chmod_600_and_roundtrip(monkeypatch, tmp_path):
    dd = _data_dir(monkeypatch, tmp_path)
    path = write_config(dd, {"token": "s3cret", "port": 9000, "cloud_enabled": False})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600  # 含密钥 → 600
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed == {"token": "s3cret", "port": 9000, "cloud_enabled": False}

    cfg = Config.from_env()  # 写出的文件即刻可被加载（roundtrip）
    assert cfg.token == "s3cret" and cfg.port == 9000 and cfg.cloud_enabled is False
    plain = write_config(tmp_path / "plain", {"port": 9001, "device": "cpu"})
    assert stat.S_IMODE(plain.stat().st_mode) != 0o600  # 无密钥不强制 600（新文件走默认 umask）


def test_config_file_follows_data_dir(monkeypatch, tmp_path):
    dd_a = tmp_path / "a"
    dd_b = tmp_path / "b"
    monkeypatch.setenv("MEMORY_DATA_DIR", str(dd_a))
    _write_toml(dd_a, "port = 9000\n")
    assert Config.from_env().port == 9000
    monkeypatch.setenv("MEMORY_DATA_DIR", str(dd_b))  # 换 data_dir → 不见 a 的文件
    assert Config.from_env().port == 8765
    assert load_config_file(dd_b / "config.toml") == {}  # 不存在 → 空


def test_file_paths_and_discovery_off(monkeypatch, tmp_path):
    dd = _data_dir(monkeypatch, tmp_path)
    _write_toml(dd, (
        f'db_path = "{tmp_path / "x.db"}"\n'
        f'model_path = "{tmp_path / "e.gguf"}"\n'
        'discovery_file = "off"\n'
        'cloud_batch_n = 42\n'
        "maintenance_on_start = false\n"
    ))
    cfg = Config.from_env()
    assert cfg.db_path == tmp_path / "x.db"
    assert cfg.model_path == tmp_path / "e.gguf"
    assert cfg.discovery_file is None        # off 语义与 env 一致
    assert cfg.cloud_batch_n == 42           # TOML 原生 int
    assert cfg.maintenance_on_start is False # TOML 原生 bool


def test_server_startup_fails_naming_bad_config_file(tmp_path):
    """语法错 → 物化即失败（属性访问触发 from_env）且错误指明文件。"""
    dd = tmp_path / "bad"
    dd.mkdir()
    (dd / "config.toml").write_text("not valid toml ][\n", encoding="utf-8")
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(REPO / "src"),
        "MEMORY_DATA_DIR": str(dd),
        "XDG_DATA_HOME": str(tmp_path / "xdg"),
        "MEMORY_DB_PATH": str(tmp_path / "i.db"),
        "MEMORY_DEVICE": "cpu",
    })
    proc = subprocess.run([sys.executable, "-c", "import semantic_memory.server as s; s.cfg"],
                          env=env, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "config.toml" in proc.stderr and str(dd / "config.toml") in proc.stderr
