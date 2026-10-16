"""serve 启动路径集成测试：随机端口 + 服务发现。

子进程真实拉起服务（MEMORY_PORT=0），验证：
发现文件字段、stdout 机器可读行、真实端口可服务、退出后文件撤下。
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _spawn(tmp_path, extra_env=None):
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(REPO / "src"),
        "MEMORY_DB_PATH": str(tmp_path / "serve.db"),
        "MEMORY_PORT": "0",
        "MEMORY_HOST": "127.0.0.1",
        "MEMORY_DISCOVERY_FILE": str(tmp_path / "discover.json"),
        "MEMORY_MAINTENANCE_ON_START": "0",
        "MEMORY_DEVICE": "cpu",
    })
    env.pop("MEMORY_TOKEN", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-m", "semantic_memory.server"],
        env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait_discovery(path: Path, proc: subprocess.Popen, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(f"服务提前退出 rc={proc.returncode}\n{out}\n{err}")
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass  # 写入中途
        time.sleep(0.1)
    raise AssertionError("发现文件未在 30s 内出现")


def test_random_port_discovery(tmp_path):
    disc = tmp_path / "discover.json"
    proc = _spawn(tmp_path)
    try:
        info = _wait_discovery(disc, proc)
        assert info["port"] > 0 and info["pid"] == proc.pid
        assert info["url"] == f"http://127.0.0.1:{info['port']}"
        assert info["token_required"] is False

        # 真实端口可服务：health + web 首页
        with urllib.request.urlopen(f"{info['url']}/health", timeout=5) as r:
            assert json.loads(r.read())["status"] == "ok"
        with urllib.request.urlopen(info["url"], timeout=5) as r:
            assert "semantic-memory" in r.read().decode()
    finally:
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=30)

    # stdout 机器可读行（agent spawn 场景的第二条发现通道）
    line = next(l for l in out.splitlines() if l.startswith("MEMORY_LISTENING "))
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["port"] == info["port"]
    assert "mcp_url" not in payload
    # 优雅退出后撤下发现文件
    assert not disc.exists()


def test_fixed_port_still_works(tmp_path):
    import socket
    with socket.socket() as s:  # 先探一个空闲端口作为"固定"端口（避免与环境的占用冲突）
        s.bind(("127.0.0.1", 0))
        fixed = s.getsockname()[1]
    disc = tmp_path / "discover.json"
    proc = _spawn(tmp_path, {"MEMORY_PORT": str(fixed)})
    try:
        info = _wait_discovery(disc, proc)
        assert info["port"] == fixed
        with urllib.request.urlopen(f"http://127.0.0.1:{fixed}/health", timeout=5) as r:
            assert r.status == 200
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=30)
