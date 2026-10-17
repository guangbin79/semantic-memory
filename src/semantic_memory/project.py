"""项目身份归一化：工作区目录路径 → git origin 归一化键（服务端单点）。

插件/适配器只上报工作区目录；同一仓库经不同检出路径访问时目录字符串
互不相同，而 origin URL 是稳定身份键。归一化在此单点完成后，shell 适配
器与插件自动同受益（插件侧零智能不变）。任何失败都回退原路径——与现
状行为唯一的差异是"能归一化时归一化"。
"""
from __future__ import annotations

import re
import subprocess

from .db import MemoryStore

# 进程内缓存：含回退结果——非 git 目录反复 spawn git 是纯浪费。
_CACHE: dict[str, str] = {}
_CACHE_MAX = 512  # 封顶：防畸形路径洪灌内存（2026-10-17 硬化；含回退结果）


def normalize_git_url(url: str) -> str:
    """git remote URL → 归一化项目键。

    git@github.com:a/b.git 与 https://github.com/a/b.git 同归一为
    github.com/a/b：剥 scheme 与 user@、scp 风格冒号转斜杠、剥 .git
    后缀、host 小写。
    """
    s = url.strip().rstrip("/")
    m = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", s)  # https:// ssh:// ...
    had_scheme = m is not None
    if m:
        s = s[m.end():]
        s = re.sub(r":\d+(?=/|$)", "", s, count=1)     # scheme 形式可带显式 :port
    s = re.sub(r"^[^/@]+@", "", s)                     # git@ user@
    if not had_scheme and ":" in s and "/" not in s.split(":", 1)[0]:  # scp 风格 host:path
        s = s.replace(":", "/", 1)
    if s.endswith(".git"):
        s = s[:-4]
    host, sep, rest = s.partition("/")
    return f"{host.lower()}/{rest}" if sep else host.lower()


def resolve_project(path: str | None) -> str | None:
    """目录路径 → origin 归一化键；无 origin/非 git/失败/超时 → 原 path。

    subprocess 2s 超时 + 结果（含回退）缓存，入口路径零阻塞累积。
    """
    if not path:
        return path
    hit = _CACHE.get(path)
    if hit is not None:
        return hit
    resolved = path
    try:
        r = subprocess.run(
            ["git", "-C", path, "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip():
            resolved = normalize_git_url(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass  # git 不存在 / 目录不存在 / 超时 → 原 path
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()  # ponytail: 粗粒度全清——键是路径、收益是防无界增长，不值得 LRU 复杂度
    _CACHE[path] = resolved
    return resolved


def backfill_project_keys(store: MemoryStore) -> dict:
    """存量回填：把 project=目录路径的既有行批量改写为归一化键（幂等）。

    按唯一旧值批量 UPDATE——每个旧值至多一次 git 调用，绝不逐行 spawn。
    resolve 失败保持原值；归一化键（非 git 路径）resolve 回自身，天然幂等。
    """
    renamed = 0
    for old in store.distinct_projects():
        new = resolve_project(old)
        if new != old:
            renamed += store.rename_project(old, new)
    return {"projects_renamed": renamed}
