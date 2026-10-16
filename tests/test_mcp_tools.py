"""MCP 工具层测试：直接调用 server 模块的工具函数。

覆盖 agent 信道（/mcp）上的五个工具：store（含去重标志与 kind 校验）、
recall（含空查询）、list、stats、forget。HTTP 面测试见 test_rest.py。
server 模块在导入时按环境建库，本模块导入前指向临时库。
"""
import os
import tempfile

os.environ["MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="mcp-tools-"), "mcp.db")
os.environ.pop("MEMORY_TOKEN", None)

import pytest

from semantic_memory import server


def test_store_and_dedup_flag():
    r = server.memory_store(content="opencode v2 插件接口为 Plugin.define",
                            kind="decision", tags=["opencode"], project="p1")
    assert r["deduped"] is False and r["uid"]
    r2 = server.memory_store(content="opencode v2 插件接口为 Plugin.define",
                             kind="decision", project="p1")
    assert r2["deduped"] is True and r2["uid"] == r["uid"]


def test_recall_list_stats_forget():
    r = server.memory_store(content="投影矩阵计算已迁到 worker 线程",
                            kind="fact", project="gis")
    hits = server.memory_recall(query="投影矩阵", limit=3)
    assert hits and hits[0]["uid"] == r["uid"]

    listing = server.memory_list(project="gis")
    assert len(listing) == 1 and listing[0]["uid"] == r["uid"]
    stats = server.memory_stats()
    assert stats["by_kind"].get("fact", 0) >= 1

    assert server.memory_forget(r["uid"])["forgotten"] is True
    assert server.memory_list(project="gis") == []


def test_store_rejects_invalid_kind():
    with pytest.raises(ValueError):
        server.memory_store(content="bad kind", kind="nonsense")


def test_recall_empty_query():
    assert server.memory_recall(query="", limit=5) == []


def test_memory_recall_global_two_path():
    """global_limit>0 + project：项目条目在前、全局补足；默认 0 保持现状。"""
    p = server.memory_store(content="MCP 两路项目条目 QY10M",
                            kind="fact", project="two-path-mcp")["uid"]
    g = server.memory_store(content="MCP 两路全局条目 QY10N", kind="fact")["uid"]
    hits = server.memory_recall(query="QY10", limit=4,
                                project="two-path-mcp", global_limit=2)
    assert [h["uid"] for h in hits] == [p, g]
    # 默认（global_limit=0）：现状兼容——项目过滤下项目条目仍命中
    hits0 = server.memory_recall(query="QY10", limit=4, project="two-path-mcp")
    assert p in [h["uid"] for h in hits0]
    # 不传 project：退化为现状全局检索（两路开关不改变该语义）
    hits_all = server.memory_recall(query="QY10", limit=4, global_limit=2)
    assert {p, g} <= {h["uid"] for h in hits_all}
