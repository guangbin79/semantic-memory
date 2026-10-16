"""eval / consolidate CLI 测试：临时库 + 已知条目上的 recall@k / MRR 精确断言。

server 模块在导入时按环境建库，本模块导入前指向临时库（禁触真实用户库）；
run_eval 系列另建独立临时库并注入禁用态 Embedder（embed=None → 纯 BM25，
即 Wave 0 基线语义，参照 test_embedder_real.py 的 skip 先例）。
语料设计（BM25-only 下排名确定性）：
- kappa/wavelength 仅 doc3 含 → 期望 doc3 时必排第 1；
- zeta/throughput 为 doc1、doc2 共有同频词，BM25 长度归一使更短的 doc1
  稳排 doc2 之前 → 期望 doc2 时首中排名恰为 2；
- pneumonic/ultraviolet 全库无匹配 → 零命中。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="mcp-eval-"), "mcp.db")
os.environ.pop("MEMORY_DISTILL_MODEL", None)

import pytest

from semantic_memory import server
from semantic_memory.db import MemoryStore
from semantic_memory.embedder import Embedder

FIXTURE = Path(__file__).parent / "fixtures" / "golden.sample.jsonl"


def _bm25_embedder() -> Embedder:
    """禁用态嵌入器：hybrid_recall 退化为纯 BM25（评测基线形态）。"""
    return Embedder(None, device="cpu")


@pytest.fixture()
def eval_env(tmp_path, monkeypatch):
    """独立临时库 + 禁用态嵌入器注入模块单例（_cmd_eval 走同一入口）。"""
    s = MemoryStore(tmp_path / "eval.db")
    monkeypatch.setattr(server, "store", s)
    monkeypatch.setattr(server, "embedder", _bm25_embedder())
    uids = [
        s.store("zeta throughput calibration report"),        # doc1（短）
        s.store("zeta eta throughput calibration summary"),   # doc2（长）
        s.store("kappa wavelength measurement log"),          # doc3
        s.store("filler passage number one"),
        s.store("filler passage number two"),
        s.store("filler passage number three"),
    ]
    return s, uids, tmp_path


def test_recall_at_k_exact_values(eval_env):
    """三行 golden：recall@1/3/5 与 MRR 全部手算对照。"""
    s, (u1, u2, u3, *_fill), _tmp = eval_env
    rows = [
        {"query": "kappa wavelength", "expect_uids": [u3], "project": None},
        {"query": "zeta throughput", "expect_uids": [u2], "project": None},
        {"query": "pneumonic ultraviolet", "expect_uids": [u3], "project": None},
    ]
    report = server.run_eval(s, _bm25_embedder(), rows, [1, 3, 5])
    assert report["recall_at_k"]["1"] == pytest.approx(1 / 3)
    assert report["recall_at_k"]["3"] == pytest.approx(2 / 3)
    assert report["recall_at_k"]["5"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert report["rows"][0]["hit_ks"] == {1: True, 3: True, 5: True}
    assert report["rows"][1]["hit_ks"] == {1: False, 3: True, 5: True}
    assert report["rows"][2]["hit_ks"] == {1: False, 3: False, 5: False}
    assert report["rows"][1]["mrr"] == pytest.approx(0.5)
    assert report["rows"][2]["mrr"] == 0.0


def test_mrr_first_hit_among_multiple_expects(eval_env):
    """多 expect 取最早命中名次；命中与否只看与 top-k 的交集。"""
    s, (u1, u2, u3, f1, _f2, _f3), _tmp = eval_env
    rows = [
        {"query": "kappa wavelength", "expect_uids": [f1, u3]},
        {"query": "zeta throughput", "expect_uids": [u2]},
    ]
    report = server.run_eval(s, _bm25_embedder(), rows, [1, 5])
    assert report["rows"][0]["mrr"] == pytest.approx(1.0)   # u3 排第 1，先于 f1 命中
    assert report["rows"][1]["mrr"] == pytest.approx(0.5)   # u1 第 1、u2 第 2
    assert report["mrr"] == pytest.approx(0.75)
    assert report["recall_at_k"]["1"] == pytest.approx(0.5)
    assert report["recall_at_k"]["5"] == pytest.approx(1.0)
def test_run_eval_passes_service_retrieval_params(eval_env, monkeypatch):
    """F3 发现③：eval 必须传与服务路径同源的 reranker/max_cosine/rerank_candidates。"""
    captured: dict = {}
    orig = server.hybrid_recall

    def spy(store, embedder, query, limit, project=None, **kw):
        captured.update(kw)
        return orig(store, embedder, query, limit, project, **kw)

    monkeypatch.setattr(server, "hybrid_recall", spy)
    s, (_u1, u2, _u3, *_fill), _tmp = eval_env
    server.run_eval(s, _bm25_embedder(), [{"query": "zeta throughput",
                                           "expect_uids": [u2]}], [1, 5])
    assert captured["reranker"] is server.reranker
    assert captured["max_cosine"] == server.cfg.recall_max_cosine
    assert captured["rerank_candidates"] == server.cfg.rerank_candidates




def test_cmd_eval_reports_table_and_json(eval_env, capsys):
    """--judge 关：stdout 出表格 + 末行 JSON 报告；Distiller 全程未被触碰。"""
    _s, (u1, u2, u3, *_fill), tmp = eval_env
    golden = tmp / "golden.jsonl"
    golden.write_text("\n".join([
        json.dumps({"query": "kappa wavelength", "expect_uids": [u3]}),
        json.dumps({"query": "zeta throughput", "expect_uids": [u2]}),
        json.dumps({"query": "pneumonic ultraviolet", "expect_uids": [u3]}),
    ]) + "\n", encoding="utf-8")
    assert server._cmd_eval(["--golden", str(golden), "--k", "1,5"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out.strip().splitlines()[-1])
    assert report["n"] == 3
    assert report["recall_at_k"]["5"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx(0.5)
    assert "recall@1=" in out and "recall@5=" in out
    assert "semantic_memory.distill" not in sys.modules


def test_golden_missing_file_exits_nonzero(tmp_path, capsys):
    assert server._cmd_eval(["--golden", str(tmp_path / "missing.jsonl")]) != 0
    assert "不存在" in capsys.readouterr().err


def test_golden_malformed_line_exits_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query": "缺 expect_uids"}\n"bare string"\n', encoding="utf-8")
    assert server._cmd_eval(["--golden", str(bad)]) != 0


def test_judge_without_env_skips(eval_env, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MEMORY_DISTILL_MODEL", raising=False)
    _s, (_u1, _u2, u3, *_fill), tmp = eval_env
    golden = tmp / "g.jsonl"
    golden.write_text(json.dumps({"query": "nomatch xyz", "expect_uids": [u3]}) + "\n",
                      encoding="utf-8")
    assert server._cmd_eval(["--golden", str(golden), "--judge"]) == 0
    assert "跳过" in capsys.readouterr().out
    assert "semantic_memory.distill" not in sys.modules


def test_judge_without_wiring_exits_zero(eval_env, tmp_path, capsys, monkeypatch):
    """env 已配且 Distiller 已实装但评注接线未实现：明示未实装、退出码保持 0。"""
    monkeypatch.setenv("MEMORY_DISTILL_MODEL", "/tmp/whatever.gguf")
    _s, (_u1, _u2, u3, *_fill), tmp = eval_env
    golden = tmp / "g.jsonl"
    golden.write_text(json.dumps({"query": "nomatch xyz", "expect_uids": [u3]}) + "\n",
                      encoding="utf-8")
    assert server._cmd_eval(["--golden", str(golden), "--judge"]) == 0
    assert "评注接线未实装" in capsys.readouterr().out


def test_reranker_distiller_share_one_modelslot():
    """组合根单槽分时（F2 B-1）：reranker 与 distiller 必须共用同一 ModelSlot。"""
    assert server.reranker._slot is server.distiller._slot


def test_sample_fixture_loads_three_rows():
    rows = server._load_golden(FIXTURE)
    assert len(rows) == 3
    assert all(r["query"] and r["expect_uids"] for r in rows)


def test_consolidate_cli_reports_json(capsys):
    server._cmd_consolidate()  # 测试环境无 MEMORY_CLOUD_* 配置 → 安全带生效不发调用
    out = json.loads(capsys.readouterr().out)
    assert out["skipped"] == "disabled"
