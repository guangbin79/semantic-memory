"""数据目录三档解析（D2）+ 旧目录自动迁移（D3）。

resolve_data_dir：MEMORY_DATA_DIR env > XDG_DATA_HOME 显式（→<xdg>/memory-mcp）
> ~/semantic-memory-mcp/；XDG 设但空串回退 $HOME 档（or 写法，同 config.py）。

migrate_legacy：启动时一次（server 模块导入处调用，禁入热路径）。硬约束：
① 旧库 wal_checkpoint(TRUNCATE) 前置；② 旧目录 flock 独占 + 发现文件 pid
存活检查——活实例拒迁（MigrationRefused，不阻塞启动，数据原封不动）；
③ 跨 fs copy→sha256 校验→通过后才清旧；④ MIGRATED 指针钉在旧目录。
幂等：新目录非空或旧目录已有 MIGRATED → 跳过。禁删用户数据。
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

MIGRATED_MARKER = "MIGRATED"
_LOCK_NAME = ".migrate.lock"
_DISCOVERY_GLOB = "*.server.json"


class MigrationRefused(RuntimeError):
    """活实例在跑/并发迁移中——拒迁。数据不动，调用面记录后照常启动。"""


class MigrationError(RuntimeError):
    """迁移自身失败（如 sha256 校验不通过）——数据保持双份或原样，需人工介入。"""


def resolve_data_dir() -> Path:
    """三档解析：env > XDG 显式 > ~/semantic-memory-mcp（空串均视同未设）。"""
    env = os.environ.get("MEMORY_DATA_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "memory-mcp"
    return Path.home() / "semantic-memory-mcp"


def legacy_data_dir() -> Path:
    """改名前的旧默认数据目录（XDG 设但空串回退 ~/.local/share，or 写法）。"""
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base).expanduser() / "memory-mcp"


def default_db_path() -> Path:
    return resolve_data_dir() / "data" / "memory.db"


def default_model_path(name: str) -> Path:
    return resolve_data_dir() / "models" / name


def migrate_legacy() -> bool:
    """旧目录 → resolve_data_dir() 自动迁移；返回是否真的搬了。

    布局映射：旧根平铺的 memory.db(+sidecars) → <data>/data/；models/、
    config.toml 等其余条目原位落新根。旧目录最终只留 MIGRATED 指针。
    """
    data_dir = resolve_data_dir()
    legacy = legacy_data_dir()
    if _skip(data_dir, legacy):
        return False
    lock_path = legacy / _LOCK_NAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 并发迁移互斥
        except OSError:
            raise MigrationRefused("另一进程正在迁移旧目录——请稍后重启重试") from None
        acquired = True
        if _skip(data_dir, legacy):  # 等锁窗口内别的实例可能已迁完，复查
            return False
        live = _live_instance_pid(legacy)
        if live is not None:
            raise MigrationRefused(
                f"旧服务仍在运行（pid={live}）——请先停止旧服务，再重启本服务完成自动迁移")
        _wal_checkpoint_all(legacy)
        data_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(legacy.iterdir()):
            if entry.name in (_LOCK_NAME, MIGRATED_MARKER):
                continue
            _move_verified(entry, data_dir / _new_rel(entry.name))
        (legacy / MIGRATED_MARKER).write_text(f"{data_dir}\n", encoding="utf-8")
        log.info("legacy data migrated: %s -> %s", legacy, data_dir)
        return True
    finally:
        os.close(fd)
        if acquired:  # 未获锁（他方持有）不得删除他方的锁文件
            lock_path.unlink(missing_ok=True)


def _skip(data_dir: Path, legacy: Path) -> bool:
    """迁移跳过判定（幂等护栏）：同址直用 / 无旧目录 / 已迁 / 旧空 / 新非空。"""
    if data_dir.expanduser().resolve() == legacy.expanduser().resolve():
        return True  # XDG 显式档：新旧同址，直用不迁
    if not legacy.is_dir() or (legacy / MIGRATED_MARKER).exists():
        return True
    if not any(p.name not in (_LOCK_NAME, MIGRATED_MARKER) for p in legacy.iterdir()):
        return True  # 空旧目录（或只剩指针残留）
    if data_dir.exists() and any(data_dir.iterdir()):
        log.warning("旧目录数据未迁移（新目录已非空），如需合并请手动处理: %s", legacy)
        return True
    return False


def _new_rel(name: str) -> Path:
    """旧根平铺的 memory.db（含 -wal/-shm sidecars）落 data/ 子目录，其余原位。"""
    if name == "memory.db" or name.startswith("memory.db-"):
        return Path("data") / name
    return Path(name)


def _live_instance_pid(legacy: Path) -> int | None:
    """发现文件（*.server.json）里的 pid 存活 → 视为旧服务在跑。"""
    for f in legacy.glob(_DISCOVERY_GLOB):
        try:
            pid = int(json.loads(f.read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError):
            continue  # 撤下一半/坏文件：按无实例处理
        if _pid_alive(pid):
            return pid
    return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 别人的进程但确实活着
    return True


def _wal_checkpoint_all(legacy: Path) -> None:
    """约束①：把 -wal 折进主库文件再搬（TRUNCATE）；坏库告警跳过（sidecar 随搬兜底）。"""
    for db in legacy.glob("*.db"):
        try:
            conn = sqlite3.connect(db)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
        except sqlite3.Error as e:
            log.warning("wal_checkpoint 跳过 %s: %s", db, e)


def _move_verified(src: Path, dst: Path) -> None:
    """约束③：同 fs 原子改名；跨 fs copy→sha256 校验→通过后才清源。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
        return
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
    if src.is_dir():
        shutil.copytree(src, dst)
        for s in sorted(src.rglob("*")):
            d = dst / s.relative_to(src)
            if s.is_file() and _sha256(s) != _sha256(d):
                raise MigrationError(f"sha256 校验失败，源未清除: {s} vs {d}")
        shutil.rmtree(src)
    else:
        shutil.copy2(src, dst)
        if _sha256(src) != _sha256(dst):
            raise MigrationError(f"sha256 校验失败，源未清除: {src} vs {dst}")
        src.unlink()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
