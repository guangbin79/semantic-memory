"""夜扫噪声维护测试：确定性命中清除/软删可逆/疑似只报告/dry_run 零写入/单飞/调度时间。

历史漏网与未来未知变体的兜底清扫（ag-bb3029）：谓词确定性强→自动 forget 软删；
弱信号疑似→只产晨报不动。端点鉴权与 purge 同款，形态见 test_rest.py。
"""
import json
from datetime import datetime

import pytest

from semantic_memory import lifecycle
from semantic_memory.db import MemoryStore
from semantic_memory.lifecycle import (
    _next_sweep_run,
    _noise_sweep_lock,
    _parse_sweep_at,
    noise_sweep,
    write_noise_sweep_log,
)

# 三族真样本（同 test_noise.py 口径：done+hash 回执/merged@环境已回收/凭据）
DONE_HASH = ("task ag-5fbe44 done from ses_e6e0b04a9ffey3LhI83J5i0mGT\n"
             "88c9662ed9005f50f826d8a11df04aea75251749")
MERGED_RECYCLED = "merged @5d5164d6bf2f179a66d95c90d6bad1951——环境已回收"
CREDENTIAL = ("[agora] credential ag-76ad62\n"
              "token: 5b8b071a0123456789abcdef0123456789abcdef0123456789abcdef5f6a9c0d")
TOKEN_BARE = "token: " + "ab" * 20  # token:+≥40 位独立兜底族
# 疑似：谓词未命中（词表外正文放行，宁漏勿误）但三弱信号齐
SUSPICIOUS = ("[agora] task ag-1 note from ses_y\n"
              "部署完成 done，commit 88c9662ed9005f50f826d8a11df04aea75251749，"
              "根因是证书过期")
KNOWLEDGE = "构建失败，根因是缺 submodule 初始化"

NOISE = {"done": DONE_HASH, "merged": MERGED_RECYCLED,
         "cred": CREDENTIAL, "token": TOKEN_BARE}


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


def _seed(store: MemoryStore, contents: dict) -> dict:
    return {k: store.store(content=c, kind="fact", source="chat")
            for k, c in contents.items()}


def _row(store: MemoryStore, uid: str):
    return store.conn.execute(
        "SELECT deleted FROM memories WHERE uid=?", (uid,)).fetchone()


def test_sweep_forgets_three_families_keeps_knowledge(store):
    uids = _seed(store, {**NOISE, "knowledge": KNOWLEDGE})
    report = noise_sweep(store)
    noise_uids = set(uids.values()) - {uids["knowledge"]}
    assert {f["uid"] for f in report["forgotten"]} == noise_uids
    assert all(f["kind"] == "fact" and f["summary"] for f in report["forgotten"])
    assert report["scanned"] == 5 and report["suspicious"] == []
    assert store.get_one(uids["knowledge"]) is not None
    for uid in noise_uids:
        assert store.get_one(uid) is None


def test_soft_delete_reversible_before_purge(store):
    uid = _seed(store, NOISE)["done"]
    noise_sweep(store)
    row = _row(store, uid)  # 软删：数据仍在（deleted=1），purge 前可查可恢复
    assert row is not None and row["deleted"] == 1


def test_suspicious_reported_not_deleted(store):
    uid = _seed(store, {"susp": SUSPICIOUS, "knowledge": KNOWLEDGE})["susp"]
    report = noise_sweep(store)
    assert report["forgotten"] == []
    assert len(report["suspicious"]) == 1
    item = report["suspicious"][0]
    assert item["uid"] == uid and len(item["weak_signals"]) == 3
    assert store.get_one(uid) is not None  # 疑似不动，只进晨报


def test_dry_run_zero_writes(store):
    uids = _seed(store, NOISE)
    report = noise_sweep(store, dry_run=True)
    assert report["dry_run"] is True
    assert {f["uid"] for f in report["forgotten"]} == set(uids.values())
    assert all(_row(store, uid)["deleted"] == 0 for uid in uids.values())  # 零写入
    assert {f["uid"] for f in noise_sweep(store)["forgotten"]} == set(uids.values())


def test_single_flight_and_idempotent(store):
    uids = _seed(store, NOISE)
    with _noise_sweep_lock:  # 已有实例在跑 → 本次跳过（与 flush 同款锁模式）
        assert "skipped" in noise_sweep(store)
        assert all(_row(store, uid)["deleted"] == 0 for uid in uids.values())
    noise_sweep(store)
    rerun = noise_sweep(store)  # 幂等：已清条目退出活集，重扫不再命中
    assert rerun["forgotten"] == [] and rerun["suspicious"] == []


def test_credential_summary_elided_in_report(store):
    uid = _seed(store, NOISE)["cred"]
    report = noise_sweep(store)
    entry = next(f for f in report["forgotten"] if f["uid"] == uid)
    assert entry["summary"] == "[credential elided]"  # 摘要不携带凭据原文进日志
    assert "5b8b071a" not in json.dumps(report)


def test_start_daemon_fail_fast_on_bad_at(store):
    with pytest.raises(ValueError):
        lifecycle.start_noise_sweep_daemon(store, "25:00")  # 线程 spawn 前同步抛


def test_parse_sweep_at():
    assert _parse_sweep_at("03:30") == (3, 30)
    assert _parse_sweep_at("23:59") == (23, 59)
    for bad in ("25:00", "3:60", "abc", "0330", ""):
        with pytest.raises(ValueError):
            _parse_sweep_at(bad)


def test_next_sweep_run():
    now = datetime(2026, 11, 3, 2, 0)
    assert _next_sweep_run(3, 30, now) == datetime(2026, 11, 3, 3, 30)  # 今日未到→今日
    now = datetime(2026, 11, 3, 4, 0)
    assert _next_sweep_run(3, 30, now) == datetime(2026, 11, 4, 3, 30)  # 已过→明日


def test_write_noise_sweep_log(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
    report = {"scanned": 2, "forgotten": [], "suspicious": []}
    path = write_noise_sweep_log(report)
    assert path.parent == tmp_path / "logs"
    assert path.name.startswith("noise-sweep-") and path.name.endswith(".log")
    assert json.loads(path.read_text(encoding="utf-8"))["scanned"] == 2


def test_sweep_offset_no_skip_on_mass_noise(tmp_path):
    # 软删使 deleted=0 分页窗口前移：整页命中时 offset 回退不跳页
    s = MemoryStore(tmp_path / "bulk.db")
    try:
        for i in range(7):
            s.store(content=f"task ag-{i} done from ses_x", kind="fact", source="chat")
        s.store(content=KNOWLEDGE, kind="fact", source="chat")
        report = noise_sweep(s, page_size=3)
        assert len(report["forgotten"]) == 7 and report["scanned"] == 8
        assert s.stats()["total"] == 1  # 只剩知识条目
    finally:
        s.close()
