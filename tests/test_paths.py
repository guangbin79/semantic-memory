"""paths.py：data_dir 三档解析（D2）+ 旧目录自动迁移（D3 硬约束）。

全部用例在 monkeypatch 的 HOME/XDG/MEMORY_DATA_DIR 沙盒里跑，不碰开发机
真实目录（conftest 已另行钉住会话级隔离根）。
"""
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from semantic_memory import paths
from semantic_memory.paths import (
    MigrationError,
    MigrationRefused,
    legacy_data_dir,
    migrate_legacy,
    resolve_data_dir,
)

REPO = Path(__file__).resolve().parent.parent


def _sandbox(monkeypatch, tmp_path):
    """HOME 沙盒：tier-3 落 tmp/home，旧目录落 tmp/home/.local/share/memory-mcp。"""
    monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path / "home"


def _make_legacy(home: Path, *, with_db=True, with_models=True) -> Path:
    legacy = home / ".local" / "share" / "memory-mcp"
    legacy.mkdir(parents=True)
    if with_db:
        conn = sqlite3.connect(legacy / "memory.db")
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('hello-migration')")
        conn.commit()
        conn.close()
    if with_models:
        (legacy / "models").mkdir()
        (legacy / "models" / "qwen3-reranker-0.6b-q8_0.gguf").write_bytes(b"fake-gguf-bytes")
    return legacy


# ---------- 三档解析矩阵（D2） ----------

def test_resolve_data_dir_tier_matrix(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    # tier 3：全未设 → ~/semantic-memory-mcp
    assert resolve_data_dir() == home / "semantic-memory"
    # tier 1：MEMORY_DATA_DIR 显式（含 ~ 展开）压过一切
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "custom"))
    assert resolve_data_dir() == tmp_path / "custom"
    monkeypatch.setenv("MEMORY_DATA_DIR", "~/data-dir")
    assert resolve_data_dir() == home / "data-dir"
    # tier 1 空串视同未设 → 落 tier 2
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MEMORY_DATA_DIR", "")
    assert resolve_data_dir() == tmp_path / "xdg" / "memory-mcp"
    # tier 2：XDG 显式 → <xdg>/memory-mcp
    monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)
    assert resolve_data_dir() == tmp_path / "xdg" / "memory-mcp"
    # XDG 设但空串也回退 $HOME 档（or 写法，同 config.py）
    monkeypatch.setenv("XDG_DATA_HOME", "")
    assert resolve_data_dir() == home / "semantic-memory"


def test_legacy_data_dir_follows_xdg_or_home(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    assert legacy_data_dir() == home / ".local" / "share" / "memory-mcp"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert legacy_data_dir() == tmp_path / "xdg" / "memory-mcp"


# ---------- 迁移（D3 硬约束） ----------

def test_migrate_moves_db_and_models_writes_migrated(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    legacy = _make_legacy(home)
    assert migrate_legacy() is True
    new = home / "semantic-memory"
    # 旧库 → <data>/data/memory.db，行还在；模型目录原位落新根
    conn = sqlite3.connect(new / "data" / "memory.db")
    assert conn.execute("SELECT v FROM t").fetchall() == [("hello-migration",)]
    conn.close()
    assert (new / "models" / "qwen3-reranker-0.6b-q8_0.gguf").read_bytes() == b"fake-gguf-bytes"
    # 旧目录只剩 MIGRATED 指针，内容钉新目录
    assert sorted(p.name for p in legacy.iterdir()) == ["MIGRATED"]
    assert legacy.joinpath("MIGRATED").read_text(encoding="utf-8").strip() == str(new)


def test_migrate_idempotent_second_call_noop(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    _make_legacy(home)
    assert migrate_legacy() is True
    new = home / "semantic-memory"
    before = sorted(str(p.relative_to(new)) for p in new.rglob("*"))
    assert migrate_legacy() is False  # MIGRATED 已在 → 跳过
    after = sorted(str(p.relative_to(new)) for p in new.rglob("*"))
    assert before == after


def test_migrate_skips_when_new_dir_nonempty(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    legacy = _make_legacy(home)
    new = home / "semantic-memory"
    new.mkdir(parents=True)
    (new / "config.toml").write_text('device = "cpu"\n', encoding="utf-8")
    assert migrate_legacy() is False
    assert (legacy / "memory.db").exists()      # 旧数据原封不动
    assert not (legacy / "MIGRATED").exists()   # 未迁不写指针
    assert (new / "data").exists() is False


def test_xdg_explicit_tier_uses_same_dir_no_migration(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    xdg = tmp_path / "xdg"
    legacy = _make_legacy(home)
    # 把整个旧目录挪到 XDG 档位下（新 tier-2 与旧默认同址）
    target = xdg / "memory-mcp"
    target.parent.mkdir(parents=True)
    os.rename(legacy, target)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    assert resolve_data_dir() == target
    assert migrate_legacy() is False               # 新旧同址直用
    assert (target / "memory.db").exists()         # 内容原地不动
    assert not (target / "MIGRATED").exists()


def test_memory_db_path_only_overrides_db(monkeypatch, tmp_path):
    """MEMORY_DB_PATH 只覆 db 路径，不动 data_dir（模型默认仍随 data_dir）。"""
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "dd"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "x" / "y.db"))
    monkeypatch.delenv("MEMORY_RERANK_MODEL", raising=False)  # conftest 置空会压掉默认值
    from semantic_memory.config import Config

    cfg = Config.from_env()
    assert cfg.db_path == tmp_path / "x" / "y.db"
    assert cfg.rerank_model == tmp_path / "dd" / "models" / "qwen3-reranker-0.6b-q8_0.gguf"
    monkeypatch.delenv("MEMORY_DB_PATH", raising=False)
    assert Config.from_env().db_path == tmp_path / "dd" / "data" / "memory.db"


def test_migrate_refused_on_live_pid(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    legacy = _make_legacy(home)
    (legacy / "memory.db.server.json").write_text(
        json.dumps({"pid": os.getpid(), "port": 8765}), encoding="utf-8")  # 活实例 = 自己
    with pytest.raises(MigrationRefused):
        migrate_legacy()
    assert (legacy / "memory.db").exists()       # 拒迁：数据不动
    assert not (legacy / "MIGRATED").exists()
    assert not (home / "semantic-memory").exists()


def test_migrate_refused_when_flock_held(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    legacy = _make_legacy(home)
    fd = os.open(legacy / ".migrate.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # 模拟另一迁移进程持锁
        with pytest.raises(MigrationRefused):
            migrate_legacy()
    finally:
        os.close(fd)
    assert (legacy / "memory.db").exists()
    assert not (home / "semantic-memory" / "data").exists()


def test_migrate_wal_checkpoint_front_loaded(monkeypatch, tmp_path):
    """约束①：wal 帧先折进主库再搬——数据只住 -wal 时迁移后仍可读。"""
    home = _sandbox(monkeypatch, tmp_path)
    legacy = _make_legacy(home, with_models=False)
    conn = sqlite3.connect(legacy / "memory.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("INSERT INTO t VALUES ('only-in-wal')")
    conn.commit()
    try:
        assert (legacy / "memory.db-wal").stat().st_size > 0  # 前提：帧确实滞留 wal
        assert migrate_legacy() is True
        probe = sqlite3.connect(home / "semantic-memory" / "data" / "memory.db")
        rows = probe.execute("SELECT v FROM t ORDER BY v").fetchall()
        probe.close()
        assert rows == [("hello-migration",), ("only-in-wal",)]
    finally:
        conn.close()


def test_migrate_crossfs_copy_sha256_verified(monkeypatch, tmp_path):
    """约束③：rename 报 EXDEV 时走 copy→sha256 校验→通过才清旧。"""
    home = _sandbox(monkeypatch, tmp_path)
    legacy = _make_legacy(home)
    src_digest = (legacy / "models" / "qwen3-reranker-0.6b-q8_0.gguf").read_bytes()

    real_rename = os.rename

    def _exdev(src, dst):
        raise OSError(18, "Invalid cross-device link")  # errno.EXDEV

    monkeypatch.setattr(paths.os, "rename", _exdev)
    assert migrate_legacy() is True
    monkeypatch.setattr(paths.os, "rename", real_rename)
    new = home / "semantic-memory"
    assert (new / "models" / "qwen3-reranker-0.6b-q8_0.gguf").read_bytes() == src_digest
    assert not (legacy / "models").exists()  # 校验通过 → 源已清
    assert (legacy / "MIGRATED").exists()


def test_migrate_sha256_mismatch_keeps_both_sides(monkeypatch, tmp_path):
    """校验不过：双份保留 + MigrationError，绝不删源。"""
    home = _sandbox(monkeypatch, tmp_path)
    legacy = _make_legacy(home, with_models=False)

    def _exdev(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(paths.os, "rename", _exdev)
    real_sha = paths._sha256
    monkeypatch.setattr(paths, "_sha256",
                        lambda p: real_sha(p) + "x" if "/semantic-memory/" in str(p) else real_sha(p))
    with pytest.raises(MigrationError):
        migrate_legacy()
    monkeypatch.setattr(paths.os, "rename", os.rename)
    assert (legacy / "memory.db").exists()  # 源在
    assert (home / "semantic-memory" / "data" / "memory.db").exists()  # 目标也在
    assert not (legacy / "MIGRATED").exists()


def test_migrate_no_legacy_noop(monkeypatch, tmp_path):
    home = _sandbox(monkeypatch, tmp_path)
    assert migrate_legacy() is False
    assert not (home / "semantic-memory").exists()  # 跳过时不创建新目录



def test_migrate_v023_home_legacy(monkeypatch, tmp_path):
    """v0.2 家目录 ~/semantic-memory-mcp → ~/semantic-memory 自动迁移（v0.3 改名）。"""
    home = _sandbox(monkeypatch, tmp_path)
    old_home = home / "semantic-memory-mcp"
    old_home.joinpath("data").mkdir(parents=True)
    conn = sqlite3.connect(old_home / "data" / "memory.db")
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.execute("INSERT INTO t VALUES ('v023-home')")
    conn.commit()
    conn.close()
    (old_home / "models").mkdir()
    (old_home / "models" / "m.gguf").write_bytes(b"gguf")

    assert migrate_legacy() is True
    new = home / "semantic-memory"
    probe = sqlite3.connect(new / "data" / "memory.db")
    assert probe.execute("SELECT v FROM t").fetchall() == [("v023-home",)]
    probe.close()
    assert (new / "models" / "m.gguf").read_bytes() == b"gguf"
    assert old_home.joinpath("MIGRATED").read_text(encoding="utf-8").strip() == str(new)
    assert sorted(p.name for p in old_home.iterdir()) == ["MIGRATED"]


def test_migrate_both_legacies_first_wins_second_skipped(monkeypatch, tmp_path):
    """两代旧目录并存：老档先迁，v0.2 档因新家非空按护栏告警跳过（数据不动）。"""
    home = _sandbox(monkeypatch, tmp_path)
    xdg_legacy = _make_legacy(home)
    v023 = home / "semantic-memory-mcp"
    v023.mkdir(parents=True)
    (v023 / "config.toml").write_text('device = "cpu"\n', encoding="utf-8")

    assert migrate_legacy() is True  # xdg 档迁成
    new = home / "semantic-memory"
    assert (new / "models").exists()
    # v0.2 档被跳过：内容原封、无指针
    assert (v023 / "config.toml").exists()
    assert not (v023 / "MIGRATED").exists()
    assert xdg_legacy.joinpath("MIGRATED").exists()

def test_server_startup_triggers_migration(tmp_path):
    """核心单例物化即迁移（v0.3 起 import 零副作用，属性访问触发；serve 同链）。"""
    home = tmp_path / "home"
    home.mkdir()
    legacy = _make_legacy(home)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(REPO / "src"),
        "HOME": str(home),
        "MEMORY_DATA_DIR": "",
        "XDG_DATA_HOME": "",
        "MEMORY_DB_PATH": str(tmp_path / "isolate.db"),  # 不占用默认 db 位
        "MEMORY_DEVICE": "cpu",
    })
    proc = subprocess.run([sys.executable, "-c", "import semantic_memory.server as s; s.store"],
                          env=env, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    new = home / "semantic-memory"
    assert (new / "models" / "qwen3-reranker-0.6b-q8_0.gguf").exists()
    assert (legacy / "MIGRATED").read_text(encoding="utf-8").strip() == str(new)
