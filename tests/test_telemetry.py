"""注入遥测：上报端点、参数校验、统计聚合、Top 解析、TTL 清理、鉴权、Web 区块。

server 模块在导入时按环境建库，本模块导入前指向临时库。
"""
import os
import tempfile
import time

os.environ["MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="telemetry-"), "t.db")
os.environ.pop("MEMORY_TOKEN", None)

import pytest

from semantic_memory import server
from semantic_memory.lifecycle import run_maintenance


# client fixture（进程内 ThreadingHTTPServer + http.client）由 conftest 提供；
# server 模块单例沿用首个导入模块（test_rest）的临时库。


def test_report_and_list(client):
    r = client.post("/inject/report", json={
        "harness": "opencode", "hook": "context", "query": "部署分支",
        "uids": ["u1", "u2"], "chars": 320, "project": "p1"})
    assert r.status_code == 200 and r.json()["recorded"] is True and r.json()["id"] >= 1
    rows = client.get("/api/injections", params={"harness": "opencode"}).json()["injections"]
    assert rows and rows[0]["hook"] == "context" and rows[0]["chars"] == 320
    assert rows[0]["project"] == "p1"


def test_report_validation(client):
    assert client.post("/inject/report", json={"uids": "notalist"}).status_code == 400
    assert client.post("/inject/report", json={"uids": [1, 2]}).status_code == 400
    assert client.post("/inject/report", json={"uids": [], "chars": "abc"}).status_code == 400
    # 空 uids 是合法边界（注入块极简时），默认字段补齐
    r = client.post("/inject/report", json={"uids": []})
    assert r.json()["recorded"] is True


def test_stats_aggregation_and_top(client):
    u1 = client.post("/ingest", json={"content": "遥测热门条目甲", "kind": "fact", "source": "test"}).json()["uid"]
    u2 = client.post("/ingest", json={"content": "遥测热门条目乙", "kind": "decision", "source": "test"}).json()["uid"]
    client.post("/inject/report", json={"harness": "opencode", "hook": "generate",
                                        "query": "q", "uids": [u1, u2], "chars": 100})
    client.post("/inject/report", json={"harness": "claude-code", "hook": "prompt",
                                        "query": "q", "uids": [u1], "chars": 60})
    s = client.get("/api/stats").json()["injections"]
    assert s["total"] >= 4 and s["total_chars"] >= 480 and s["recent_7d"] >= 4
    assert s["by_harness"].get("opencode") >= 2 and s["by_harness"].get("claude-code") == 1
    assert s["by_hook"].get("generate") >= 1 and s["by_hook"].get("prompt") >= 1
    top = {t["uid"]: t for t in s["top"]}
    assert top[u1]["times"] >= 2 and "热门条目甲" in top[u1]["content"]
    assert top[u1]["kind"] == "fact"
    # 存储层注入总量与 /api/stats 聚合一致（旧 MCP 工具面已删，契约并入存储/HTTP 双面）
    assert server.store.injection_stats()["total"] == s["total"]


def test_ttl_purge(client):
    old = time.time() - 100 * 86400
    server.store.record_injection("old-harness", "context", "q", ["x"], 10, ts=old)
    before = client.get("/api/stats").json()["injections"]["total"]
    report = run_maintenance(server.store, ttl_artifact_days=0, ttl_injection_days=90)
    assert report["injections_purged"] >= 1
    after = client.get("/api/stats").json()["injections"]["total"]
    assert after == before - report["injections_purged"]


def test_token_guard(client):
    object.__setattr__(server.cfg, "token", "secret")
    try:
        assert client.post("/inject/report", json={"uids": []}).status_code == 401
        assert client.get("/api/injections").status_code == 401
        ok = client.post("/inject/report", json={"uids": []},
                         headers={"authorization": "Bearer secret"})
        assert ok.status_code == 200
    finally:
        object.__setattr__(server.cfg, "token", None)


def test_web_page_has_telemetry_section(client):
    assert "注入统计" in client.get("/").text


# ---------- task14：观测面（召回延迟分位数 + 节流告警 + 队列/云调用） ----------


def test_percentiles():
    from semantic_memory.server import percentile

    xs = [float(i) for i in range(1, 101)]  # 已知序列（最近秩）：p50=50, p95=95
    assert percentile(xs, 50) == 50.0
    assert percentile(xs, 95) == 95.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([5.0, 1.0], 95) == 5.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([], 95) == 0.0  # 空窗口：0，不出 NaN


def test_throttle_no_spam():
    from semantic_memory.server import ALERT_WINDOW_S, plan_alerts

    last: dict[str, float] = {}
    now = 1000.0
    fired = plan_alerts(900.0, 41000, last, now)
    assert [lv for lv, _ in fired] == ["warning", "info"]
    for lv, _ in fired:
        last[lv] = now
    # 10 分钟窗口内同级告警只打一次
    assert plan_alerts(900.0, 41000, last, now + ALERT_WINDOW_S - 1) == []
    # 窗口过后恢复
    assert [lv for lv, _ in plan_alerts(900.0, 41000, last, now + ALERT_WINDOW_S)] == [
        "warning", "info"]
    # WARN→ERROR 升级：不同级，不受 WARN 节流压住
    assert [lv for lv, _ in plan_alerts(1500.0, 10, last, now + 1)] == ["error"]
    # 未超阈值：静默
    assert plan_alerts(100.0, 10, last, now) == []


def test_api_stats_observability_fields(client):
    s = client.get("/api/stats").json()
    for k in ("recall_p50_ms", "recall_p95_ms", "distill_queue_len",
              "cloud_calls_24h", "entries"):
        assert k in s, k
    assert s["entries"] == s["total"]
    assert isinstance(s["cloud_calls_24h"], int) and s["cloud_calls_24h"] >= 0


def test_memory_stats_observability_fields(client):
    s = client.get("/api/stats").json()
    for k in ("recall_p50_ms", "recall_p95_ms", "distill_queue_len",
              "cloud_calls_24h", "entries"):
        assert k in s, k


def test_web_page_has_observability_section(client):
    html = client.get("/").text
    assert "蒸馏队列" in html and "云调用" in html
    assert "recall_p95_ms" in html and "recall_p50_ms" in html
