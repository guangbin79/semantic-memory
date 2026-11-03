"""任务事件噪声闸：agora/hook 状态回执的行级判定与滤除。

三层污染路共用同一判定核（raw hook 直写 / 显式 memory_store / 蒸馏对
transcript 事件行再创作）：先剥标识 token（哈希/session/task id/路径）与
门禁样板子串，残余对照词表。宁漏勿误——任何一行含词表外正文即放行整条
（误杀知识不可逆），故整条判噪声还须叠加长度与状态关键词双条件。
挂点：rest_ingest 入口丢弃；flush_distill_queue 行丢弃、蒸馏输入滤洗、
typed 释义丢弃（详见 server.py / lifecycle.py）。
"""
from __future__ import annotations

import re

_HEX = re.compile(r"[0-9a-f]{40}")
_SHORT_HEX = re.compile(r"\b[0-9a-f]{7,39}\b")
_SES = re.compile(r"ses_[A-Za-z0-9]+")
# CJK 与 ASCII 同属 \w，\b 在 "任务gt-xxx" 处不成立——必须负向后行
_TASKID = re.compile(r"(?<![A-Za-z0-9])(?:ag|gt|task)-[A-Za-z0-9]+")
_PATH = re.compile(r"[\w.-]+/[\w.-]+(?:/[\w.-]+)*")

# 门禁样板子串：状态回执固定短语，剥除后残余才做词表判定
_BOILERPLATE = (
    "non-ff 拒并：目标分支已推进——同步主干后重新 post(done, commit)",
    "verify 失败——修复后重新 post(done, commit)",
    "输出尾部:",
    "merged @",
    "——环境已回收",
    "environment recycled",
    "environment recovered",
    "Merged commit",
)

_CJK_VOCAB = set(
    "任务已完成关闭提交的哈希为其分别和评审环锚设置状态取取消环境回收"
    "并标记里程碑合并成功需重派或阻因连三失败质次量请原因"
)
# blocked 不在派发词表原文里，但关键词表含 "已 blocked" 且噪声真样本
# "任务 ag-x 已 blocked，需重派或取消" 依赖它——按样本收编
_EN_VOCAB = frozenset((
    "task tasks closed completed merged done blocked commit commits hash with"
    " for project the a was and environment recycled recovered verify from"
    " agorad agora result into main branch ci cd ag gt sha head master"
    " specific needs to post milestone t p s dependency_failed"
).split())
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _tokens_ok(residual: str) -> bool:
    """英文 token 全落词表（lower 后）；含数字 token 视为中性放行。"""
    return all(t.lower() in _EN_VOCAB or any(c.isdigit() for c in t)
               for t in _TOKEN.findall(residual))
# 无 CJK 残余的白名单字符集（出集字符即非噪声行，宁漏勿误）
_PLAIN = re.compile(r'^[\w\s\[\]@:：,，.()（）/+\-—#\'"]*$')
_KEYWORDS = re.compile(
    r"done|completed|closed|merged|cancel|milestone|已完成|已关闭|已取消"
    r"|已 closed|已 blocked|已 done|non-ff|verify 失败",
    re.IGNORECASE,
)


def _is_noise_line(line: str) -> bool:
    """行级判定：剥 token/样板后残余为空/纯 #，或 CJK、英文 token 全落词表
    （无 CJK 时残余须全为白名单字符）。超长行直接非噪声——与整条闸同口径，
    防 _PATH 在长无斜线段上的二次回溯拖垮 flush（评审实测 10⁵ 字符 20s）。"""
    if len(line) > 400:
        return False
    residual = line
    for pat in (_HEX, _SHORT_HEX, _SES, _TASKID, _PATH):
        residual = pat.sub("", residual)
    for s in _BOILERPLATE:
        residual = residual.replace(s, "")
    if not residual.strip(" #"):
        return True
    if _CJK.search(residual):
        if not all(ch in _CJK_VOCAB for ch in _CJK.findall(residual)):
            return False
        return _tokens_ok(residual)
    # 字符集是必要条件不充分：裸英文词（如测试 fixture "raw-a"）须同样
    # 落词表，否则蒸馏输入滤洗会误杀纯英文知识行
    return _PLAIN.match(residual) is not None and _tokens_ok(residual)


def is_task_event_noise(content: str) -> bool:
    """整条判定：≤400 字符且非空行全为噪声行且命中任务状态关键词。

    三条件缺一不可：词表放行了太多正常英文短句，状态关键词是最后一道
    「这确实是任务回执」的证据门。
    """
    if len(content) > 400 or not content.strip():
        return False
    if not all(_is_noise_line(ln) for ln in content.split("\n")):
        return False
    return _KEYWORDS.search(content) is not None


def strip_noise_lines(content: str) -> str:
    """逐行滤除噪声行，其余按原序 \\n 连接（混合内容的蒸馏输入滤洗）。"""
    return "\n".join(ln for ln in content.split("\n")
                     if not _is_noise_line(ln))
