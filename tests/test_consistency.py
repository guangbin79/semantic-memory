"""一致性巡检测试：路径引用抽取、悬空判定、只报告不删除边界。"""
from pathlib import Path

from semantic_memory.consistency import check_ref, extract_path_refs, run_sweep
from semantic_memory.db import MemoryStore

import pytest


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


def test_extract_hits():
    content = (
        "配置在 file:///home/a/b%20c/conf.toml；脚本 ~/bin/run.sh；"
        "日志落 /var/log/app.log。文档见 docs/guide.md。"
    )
    assert extract_path_refs(content) == [
        "file:///home/a/b%20c/conf.toml", "~/bin/run.sh",
        "/var/log/app.log", "docs/guide.md",
    ]


def test_extract_noise_rejected():
    content = (
        "POST /api/maintenance/purge 走 token；参考 https://example.com/docs/x.md；"
        "这是纯语义陈述，无任何路径。DELETE /api/memories/<uid> 闭环。"
    )
    assert extract_path_refs(content) == []


def test_extract_trailing_punct_and_dedup():
    assert extract_path_refs("见 /tmp/a.log，再看 /tmp/a.log。") == ["/tmp/a.log"]


def test_check_ref_absolute(tmp_path):
    existing = tmp_path / "f.txt"
    existing.write_text("x")
    assert check_ref(str(existing), None)[0] == "ok"
    assert check_ref(str(tmp_path / "nope"), None)[0] == "missing"


def test_check_ref_relative_needs_abs_dir_base(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x")
    git_key = "github.com/a/b"
    assert check_ref("docs/a.md", git_key) == ("unverifiable", "docs/a.md")
    assert check_ref("docs/a.md", str(tmp_path))[0] == "ok"
    assert check_ref("docs/ghost.md", str(tmp_path))[0] == "missing"
    assert check_ref("docs/a.md", None)[0] == "unverifiable"


def test_sweep_reports_dangling_without_deletion(store, tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("x")
    u1 = store.store(content=f"真文件在 {real}", project=None)
    u2 = store.store(content="红线清单见 ~/nonexistent-gt61cf/ghost-checklist.md",
                     kind="decision", project="/nonexistent/dir")
    u3 = store.store(content="无从属路径的纯语义陈述")
    before = store.stats()["total"]
    report = run_sweep(store)
    # 只报告不删除
    assert store.stats()["total"] == before
    assert report["scanned"] == 3
    assert report["entries_with_refs"] == 2
    assert report["refs_ok"] == 1
    dangling = report["dangling"]
    assert [d["uid"] for d in dangling] == [u2]
    assert dangling[0]["ref"] == "~/nonexistent-gt61cf/ghost-checklist.md"
    assert dangling[0]["project"] == "/nonexistent/dir"
    assert dangling[0]["kind"] == "decision"
    assert report["dangling_by_project"] == {"/nonexistent/dir": 1}
    assert u1 and u3  # 存在引用与纯语义条目均不入悬空清单


def test_sweep_pagination(store):
    for i in range(5):
        store.store(content=f"条目 {i} 指向 /nonexistent/{i}.log")
    report = run_sweep(store, page_size=2)  # 强制多页遍历
    assert report["scanned"] == 5
    assert len(report["dangling"]) == 5


def test_sweep_skips_soft_deleted(store, tmp_path):
    missing = tmp_path / "gone.txt"
    store.store(content=f"指向 {missing}")
    uid = store.store(content=f"已软删 {missing}")
    store.forget(uid)
    report = run_sweep(store)
    assert report["scanned"] == 1
