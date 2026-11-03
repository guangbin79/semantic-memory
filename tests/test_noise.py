"""任务事件噪声闸测试：真样本噪声/放行两侧钉死＋flush 三挂点接线。

入口闸（rest_ingest → task_event_noise 丢弃）的判定核即本文件覆盖的
纯函数，REST 形态由 test_rest.py 的既有门面保证。
"""
import time

import pytest

from semantic_memory.db import MemoryStore
from semantic_memory.lifecycle import enqueue_raw, flush_distill_queue
from semantic_memory.noise import is_task_event_noise, strip_noise_lines

# 2026-11 热mem 污染事故的真实回执样本（三层形态）
NOISE_SAMPLES = [
    # raw hook 直写：done + 裸 40 位 hash
    "task ag-5fbe44 done from ses_e6e0b04a9ffey3LhI83J5i0mGT\n"
    "88c9662ed9005f50f826d8a11df04aea75251749",
    "Task ag-c91e5d completed from ses_x",
    # 改写层：中英混排 closed + commit=hash
    "任务 ag-aa9a74 已 closed；commit=356c6c3b040d66e3a383a81336930c94b8361efa",
    # 蒸馏释义层：对事件行再创作
    "The task was merged with commit "
    "9c929dce4cc20ada4a74e502ccb7020ecffa59dd, and the environment was "
    "recycled.",
    "merged @5d5164d6bf2f179a66d95c90d6bad1951——环境已回收",
    "任务 ag-fa17e4 已 blocked，需重派或取消",
    # 门禁样板剥除路径
    "任务 ag-22c3b1 已 done；non-ff 拒并：目标分支已推进——同步主干后重新 "
    "post(done, commit)",
]

KEEP_SAMPLES = [
    # peer 附言：事件头是噪声行，但正文含词表外 CJK → 整条放行
    "[agora] task X note from ses_Y\n开工：先读任务页与现状，再实施。",
    # 词表外知识陈述
    "构建失败，根因是缺 submodule 初始化",
    # 无状态关键词的短句
    "每天站会 10 点",
    # 混合：一行噪声救不走词表外知识行（宁漏勿误）
    "task ag-1 done from ses_1\n构建失败，根因是缺 submodule 初始化",
    # len 闸：行全噪声＋关键词命中，仅超长即放行
    "task done merged " * 40,
    # 裸英文 fixture 词：字符集命中但词表外 → 不判噪声（误杀回归钉）
    "raw-a",
]


@pytest.mark.parametrize("content", NOISE_SAMPLES)
def test_noise_samples(content):
    assert is_task_event_noise(content) is True


@pytest.mark.parametrize("content", KEEP_SAMPLES)
def test_keep_samples(content):
    assert is_task_event_noise(content) is False


def test_bare_recycled_line_only_noise_at_line_level():
    # 裸「环境已回收」无状态关键词（宁漏勿误）→ 整条不判噪声，但行级是噪声
    assert is_task_event_noise("环境已回收") is False
    assert strip_noise_lines("环境已回收") == ""


def test_strip_keeps_knowledge_lines():
    assert strip_noise_lines("task ag-x done from ses_y\n保留的知识") == "保留的知识"
    assert strip_noise_lines("verify 失败——修复后重新 post(done, commit)\n真知识") == "真知识"
    assert strip_noise_lines("无噪声") == "无噪声"


class _StubDistiller:
    configured = True

    def __init__(self, result=None, capture=None, per_text=None):
        self.result = result
        self.capture = capture  # list：distill 收到的 texts
        self.per_text = per_text  # dict：按文本 1:1 产出释义（混合组绑定钉）
        self.calls = 0

    def is_available(self):
        return True

    def distill(self, texts, project, neighbors=None):
        self.calls += 1
        if self.capture is not None:
            self.capture.extend(texts)
        if self.per_text is not None:
            return [self.per_text[t] for t in texts]
        return self.result


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


def _mem_count(store):
    return store.conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]


def test_flush_drops_noise_rows(store):
    """挂点①：全噪声行 → done+dropped，不蒸馏不落库。"""
    enqueue_raw(store, "task ag-1 done from ses_1", "chat", "p")
    enqueue_raw(store, "构建失败，根因是缺 submodule 初始化", "chat", "p")
    report = flush_distill_queue(store, None)  # 无裁判：知识行走原文直存
    assert report["dropped"] == 1
    assert report["stored_raw"] == 1
    assert _mem_count(store) == 1


def test_flush_strips_noise_lines_from_distill_input(store):
    """挂点②：蒸馏输入先滤噪声行（混合内容只送知识部分）。"""
    captured = []
    entry = {"content": "真实知识", "kind": "fact", "scope": "project", "tags": []}
    enqueue_raw(store, "task ag-1 done from ses_9\n真实知识", "chat", "p")
    report = flush_distill_queue(store, _StubDistiller([entry], captured))
    assert captured == ["真实知识"]
    assert report["typed"] == 1


def test_flush_drops_noise_typed_entry(store):
    """挂点③：蒸馏释义仍是噪声 → 同行丢弃，零落库。"""
    entry = {"content": "The task was merged with commit "
             "9c929dce4cc20ada4a74e502ccb7020ecffa59dd, and the environment "
             "was recycled.", "kind": "fact", "scope": "project", "tags": []}
    enqueue_raw(store, "知识A", "chat", "p")
    report = flush_distill_queue(store, _StubDistiller([entry]))
    assert report["typed"] == 0
    assert report["dropped"] == 1
    assert _mem_count(store) == 0


def test_flush_mixed_group_binding(store):
    """错绑回归钉（评审三路同报）：纯噪声行不占蒸馏槽位，绑定不位移。"""
    captured = []
    noise_para = {"content": "PARA::EMPTY", "kind": "fact",
                  "scope": "project", "tags": []}
    real = {"content": "真实知识B", "kind": "fact",
            "scope": "project", "tags": []}
    enqueue_raw(store, "task ag-1 done from ses_1", "chat", "p")
    enqueue_raw(store, "真实知识B", "chat", "p")
    report = flush_distill_queue(
        store, _StubDistiller(per_text={"真实知识B": real,
                                        "": noise_para}, capture=captured))
    assert captured == ["真实知识B"]  # 纯噪声行不进蒸馏输入
    assert report == {"typed": 1, "stored_raw": 0,
                      "left_pending": 0, "dropped": 1}
    rows = store.conn.execute(
        "SELECT content FROM memories").fetchall()
    assert [r["content"] for r in rows] == ["真实知识B"]


def test_flush_drops_vocab_only_row_without_distill(store):
    """评审 B-F2 钉：词表 CJK 无状态关键词的行——滤洗后为空 → 整行丢弃，
    空串不进蒸馏（不诱出垃圾释义）。"""
    stub = _StubDistiller(result=[])
    enqueue_raw(store, "评审环境已设置，任务状态已标记完成", "chat", "p")
    report = flush_distill_queue(store, stub)
    assert stub.calls == 0
    assert report["typed"] == 0
    assert report["dropped"] == 1
    assert _mem_count(store) == 0


def test_long_line_pinned_no_regex_blowup():
    """评审 C-F1 钉：超长行免判定（_PATH 二次回溯），strip 原样通过。"""
    long_line = "z" * 100_000
    start = time.monotonic()
    assert strip_noise_lines(long_line) == long_line
    assert is_task_event_noise("task done\n" + long_line) is False
    assert time.monotonic() - start < 2
