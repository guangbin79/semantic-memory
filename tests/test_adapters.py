"""shell 适配器冒烟测试（评审 [08]/[09]）：stub-curl 驱动真实脚本。

覆盖：
- 同会话 uid 集合不变 → 第二次不注入、不发遥测（前缀缓存保护）
- uid 变化 → 重新注入
- 无 MEMORY_SESSION_ID → 旧行为（每次注入）
- 无 sha256sum（PATH 剥离）→ 哈希为空跳过去重，仍然注入（宁多注不误抑）
- project 透传进 /recall 请求体
- 优先 CLAUDE 缺席 → fail-open 退出 0
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ADAPTERS = REPO / "integrations"

CANNED = '{"hits":[{"uid":"a1","content":"memo","kind":"fact","score":1.0}]}'


@pytest.fixture
def stub_env(tmp_path):
    """stub curl（/recall 回 canned JSON 并记录请求体）+ 隔离状态目录。"""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    canned = tmp_path / "canned.json"
    canned.write_text(CANNED, encoding="utf-8")
    stub = bindir / "curl"
    stub.write_text(f"""#!/usr/bin/env bash
url=""; data=""
while [ $# -gt 0 ]; do case "$1" in http*) url="$1";; -d) data="$2";; esac; shift; done
printf '%s\\t%s\\n' "$url" "$data" >> {log}
case "$url" in
  */recall) cat {canned};;
esac
exit 0
""", encoding="utf-8")
    os.chmod(stub, 0o700)

    def run(script, stdin_json, env_extra=None, strip_sha=False):
        env = dict(os.environ)
        if strip_sha:
            # 真隔离：PATH 仅含受限目录（无 sha256sum/shasum，其余必需件软链在位）
            (bindir / "shaless").mkdir(exist_ok=True)
            import shutil
            for tool in ("jq", "cat", "grep", "sed", "head", "tail", "mkdir",
                         "tr", "find", "cut", "bash", "sh", "dirname"):
                dst = bindir / "shaless" / tool
                if not dst.exists():
                    w = shutil.which(tool)
                    if w:
                        dst.symlink_to(w)
            curl_link = bindir / "shaless" / "curl"
            if not curl_link.exists():
                curl_link.symlink_to(bindir / "curl")
            env["PATH"] = str(bindir / "shaless")
        else:
            env["PATH"] = str(bindir) + ":" + env["PATH"]
        env.update({
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "MEMORY_PROJECT": str(tmp_path),  # 固定值，避免 $PWD 漂移
        })
        env.update(env_extra or {})
        return subprocess.run(
            ["bash", str(script)], input=stdin_json, capture_output=True,
            text=True, env=env, cwd=REPO, timeout=30)

    return run, log, canned


def _reports(log):
    return [l for l in log.read_text().splitlines() if "/inject/report" in l]


def test_suppression_same_uid_set(stub_env):
    run, log, _ = stub_env
    stdin = '{"session_id":"s1","prompt":"hello world"}'
    r1 = run(ADAPTERS / "claude-code/on-prompt.sh", stdin)
    assert r1.returncode == 0
    assert "semantic-memory" in r1.stdout and len(r1.stdout) > 0, "首次必须注入"
    r2 = run(ADAPTERS / "claude-code/on-prompt.sh", stdin)
    assert r2.returncode == 0 and r2.stdout == "", "同 uid 集合必须抑制"
    assert len(_reports(log)) == 1, "抑制轮不得发遥测"


def test_reemit_on_uid_change(stub_env):
    run, log, canned = stub_env
    stdin = '{"session_id":"s2","prompt":"hello"}'
    assert run(ADAPTERS / "claude-code/on-prompt.sh", stdin).stdout
    canned.write_text(CANNED.replace("a1", "b2"), encoding="utf-8")
    r3 = run(ADAPTERS / "claude-code/on-prompt.sh", stdin)
    assert "semantic-memory" in r3.stdout, "uid 变化必须重注入"


def test_no_session_legacy_behavior(stub_env):
    run, log, _ = stub_env
    stdin = '{"prompt":"hello no session"}'
    assert run(ADAPTERS / "claude-code/on-prompt.sh", stdin).stdout
    assert run(ADAPTERS / "claude-code/on-prompt.sh", stdin).stdout, "无会话 ID 恒注入（旧行为）"


def test_no_sha256sum_still_injects(stub_env, tmp_path):
    """[05] 回归：sha256sum 缺席（macOS 形态）不得永久抑制。"""
    run, log, _ = stub_env
    # 受限目录已由 stub_env 的 strip_sha 分支统一构建（无 sha256sum/shasum）
    stdin = '{"session_id":"s3","prompt":"no sha world"}'
    r1 = run(ADAPTERS / "claude-code/on-prompt.sh", stdin, strip_sha=True)
    assert "semantic-memory" in r1.stdout, "无摘要工具时必须照常注入（宁多注不误抑）"
    r2 = run(ADAPTERS / "claude-code/on-prompt.sh", stdin, strip_sha=True)
    assert "semantic-memory" in r2.stdout, "第二轮同样不得误抑制"


def test_project_passthrough(stub_env, tmp_path):
    run, log, _ = stub_env
    run(ADAPTERS / "claude-code/on-prompt.sh", '{"session_id":"s4","prompt":"proj"}')
    recall_lines = [l for l in log.read_text().splitlines() if "/recall" in l]
    assert recall_lines and f'"{tmp_path}"' in recall_lines[0], "project 必须进 /recall 体"


def test_service_down_fail_open(stub_env, tmp_path):
    """stub curl 改为对 /recall 报错（模拟服务死）→ 钩子退出 0 零输出。"""
    run, log, _ = stub_env
    stub = tmp_path / "bin" / "curl"
    stub.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    r = run(ADAPTERS / "claude-code/on-prompt.sh", '{"session_id":"s5","prompt":"x"}')
    assert r.returncode == 0 and r.stdout == "", "fail-open：退出 0 且无输出"


def test_kimi_wrapper_payload_fallback(stub_env):
    run, log, _ = stub_env
    # kimi 路径兜底：.prompt 缺席时取 .message
    r = run(ADAPTERS / "kimi-code/on-prompt.sh",
            '{"session_id":"k1","message":"from message field"}')
    assert r.returncode == 0
    assert "semantic-memory" in r.stdout, ".message 兜底路径必须同样注入"
