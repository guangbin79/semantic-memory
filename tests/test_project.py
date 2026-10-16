"""项目身份归一化测试：git origin URL 归一化、回退语义、存量回填迁移。"""
import subprocess
from types import SimpleNamespace

import pytest

from semantic_memory import project
from semantic_memory.db import MemoryStore
from semantic_memory.project import backfill_project_keys, normalize_git_url, resolve_project


@pytest.fixture(autouse=True)
def _clear_cache():
    project._CACHE.clear()
    yield
    project._CACHE.clear()


# ---------- normalize：纯函数 ----------

def test_normalize_scp_and_https_same_key():
    assert normalize_git_url("git@github.com:a/b.git") == "github.com/a/b"
    assert normalize_git_url("https://github.com/a/b.git") == "github.com/a/b"


def test_normalize_host_lowercase_and_variants():
    assert normalize_git_url("https://GitHub.com/A/B") == "github.com/A/B"  # 仅 host 小写
    assert normalize_git_url("ssh://git@gitlab.com/x/y.git") == "gitlab.com/x/y"
    assert normalize_git_url("git@github.com:a/b") == "github.com/a/b"


# ---------- resolve：subprocess 边界 ----------

def test_resolve_origin_normalized(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        assert kw["timeout"] == 2
        return SimpleNamespace(returncode=0, stdout="git@github.com:acme/widget.git\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_project("/home/me/widget") == "github.com/acme/widget"
    assert calls and calls[0][:4] == ["git", "-C", "/home/me/widget", "remote"]


def test_resolve_caches_per_path(monkeypatch):
    n = 0

    def fake_run(cmd, **kw):
        nonlocal n
        n += 1
        return SimpleNamespace(returncode=0, stdout="https://github.com/acme/widget.git")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_project("/w") == "github.com/acme/widget"
    assert resolve_project("/w") == "github.com/acme/widget"
    assert n == 1


def test_resolve_non_git_falls_back(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=128, stdout=""))
    assert resolve_project("/not/a/repo") == "/not/a/repo"


def test_resolve_timeout_falls_back(monkeypatch):
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd="git", timeout=2)

    monkeypatch.setattr(subprocess, "run", slow)
    assert resolve_project("/slow/repo") == "/slow/repo"


def test_resolve_git_missing_falls_back(monkeypatch):
    def no_git(cmd, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert resolve_project("/no/git") == "/no/git"


def test_resolve_empty_project_passthrough(monkeypatch):
    def boom(cmd, **kw):
        raise AssertionError("空 project 不该触发 git 调用")

    monkeypatch.setattr(subprocess, "run", boom)
    assert resolve_project(None) is None
    assert resolve_project("") == ""


# ---------- 存量回填 ----------

def test_backfill_migrates_existing_paths(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "old.db")
    u1 = store.store(content="旧路径事实甲", kind="fact", source="test",
                     project="/home/me/oldrepo")
    u2 = store.store(content="旧路径事实乙", kind="fact", source="test",
                     project="/home/me/oldrepo")
    store.record_injection(harness="t", hook="context", query="q", uids=[u1], chars=8,
                           project="/home/me/oldrepo")
    store.store(content="归一化键既有条目", kind="fact", source="test",
                project="github.com/acme/oldrepo")

    def fake_run(cmd, **kw):
        path = cmd[cmd.index("-C") + 1]
        if path == "/home/me/oldrepo":
            return SimpleNamespace(returncode=0, stdout="git@github.com:acme/oldrepo.git\n")
        return SimpleNamespace(returncode=128, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = backfill_project_keys(store)
    assert report["projects_renamed"] == 3  # 2 memories + 1 injection
    assert [r["project"] for r in store.get_many([u1, u2])] == ["github.com/acme/oldrepo"] * 2
    assert len(store.list_injections(project="github.com/acme/oldrepo")) == 1
    assert store.list_injections(project="/home/me/oldrepo") == []

    # 幂等：第二遍零改写，键不再变化
    assert backfill_project_keys(store)["projects_renamed"] == 0
    assert {r["project"] for r in store.get_many([u1, u2])} == {"github.com/acme/oldrepo"}


def test_backfill_keeps_unresolvable_paths(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "b.db")
    u = store.store(content="无 origin 目录", kind="fact", source="test",
                    project="/plain/dir")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=128, stdout=""))
    assert backfill_project_keys(store)["projects_renamed"] == 0
    assert store.get_many([u])[0]["project"] == "/plain/dir"
