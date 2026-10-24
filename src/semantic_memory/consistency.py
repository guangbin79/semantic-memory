"""热层一致性巡检：悬空路径引用报告器（只报告不删除）。

热层条目由会话自动蒸馏产生，无机制校验条目指向的事实存在——悬空引用
（指向已不存在的路径）会躺进召回面反复注入打断工作。巡检遍历活条目，
抽取路径型引用（file: URI、~/ 展开、绝对路径、相对文档指针）核对文件
系统存在性，产出人工裁决用的候选清单。硬边界：悬空≠无效（历史陈述
可为真时态），本模块零删除零写入；只查路径型引用，不碰语义陈述。
"""
from __future__ import annotations

import os
import re
import urllib.parse

from .db import MemoryStore

# 路径主体字符集：不含空白/引号/尖括号——`<uid>` 等占位符在边界自然截断；
# % 收进以覆盖 file: URI 的 %XX 转义
_BODY = r"[A-Za-z0-9._~/@+%-]+"
# file:// URI 与 ~/ 展开（先于绝对路径匹配，避免重复计入）
_FILE_URI_RE = re.compile(rf"file://{_BODY}")
_TILDE_RE = re.compile(rf"(?<!\w)~/{_BODY}")
# 绝对路径：lookbehind 排除 URL 内层与 ~/ 展开——前邻 : / ~ 的斜杠非路径起点
_ABS_RE = re.compile(rf"(?<![:/~\w])/{_BODY}")
# 相对文档指针：≥1 目录层＋扩展名尾（docs/x.md、src/a/b.py）；
# 纯文件名（AGENTS.md）与无扩展名片段噪声太大不收；前邻 : 排端口、% 排 URI 转义尾随
_REL_RE = re.compile(r"(?<![\w./~:%\-])(?:\./)?[\w.\-]+(?:/[\w.\-]+)+\.[A-Za-z0-9]{1,8}")
# 域名式首段（example.com/x.md）：是 host 不是文档指针，常见 TLD 停用词表兜底
_HOSTY_RE = re.compile(
    r"(?i)^[\w\-]+\.(?:com|net|org|io|dev|cn|edu|gov|co|ai|app|info|xyz|me)(?:/|$)")
# URL 空间首段停用词（/api/... 这类 REST 片段不是文件系统引用）
_URLSPACE = frozenset({"api", "http", "https", "ftp", "v1", "v2"})

_TRAILING_PUNCT = ".,;:!?)]}》」』\"'`，。；：！？、）"


def _clean(m: str) -> str:
    return m.rstrip(_TRAILING_PUNCT)


def extract_path_refs(content: str) -> list[str]:
    """顺序保序去重抽取路径型引用；以 / 结尾的命名空间片段、URL 空间
    首段（/api/...）与域名式首段（example.com/x.md）丢弃——它们不是
    文件系统引用。"""
    refs: list[str] = []
    for pat in (_FILE_URI_RE, _TILDE_RE, _ABS_RE, _REL_RE):
        for m in (_clean(x) for x in pat.findall(content)):
            if not m or m.endswith("/") or m in refs:
                continue
            first = m.lstrip("./").split("/", 1)[0]
            if first in _URLSPACE or _HOSTY_RE.match(m):
                continue
            refs.append(m)
    return refs


def check_ref(ref: str, base: str | None) -> tuple[str, str]:
    """单引用核验，返回 (status, resolved)。

    status：ok（存在）| missing（悬空候选）| unverifiable（无 base 可依，
    不计悬空）。相对路径仅在 project 字段本身是绝对路径形式的现存目录
    （非 git 回退形态）时以其为 base——归一化 git 键无法定位检出目录。
    file:// 带非本机 authority（RFC 8089）不是本地路径，unverifiable。
    """
    if ref.startswith("file://"):
        parsed = urllib.parse.urlparse(ref)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            return "unverifiable", ref
        path = urllib.parse.unquote(parsed.path)
    elif ref.startswith("~/"):
        path = os.path.expanduser(ref)
    elif ref.startswith("/"):
        path = ref
    else:
        if base and os.path.isabs(base) and os.path.isdir(base):
            path = os.path.join(base, ref)
        else:
            return "unverifiable", ref
    return ("ok" if os.path.exists(path) else "missing"), path


def run_sweep(store: MemoryStore, page_size: int = 200) -> dict:
    """全量遍历活条目（deleted=0），产出悬空引用候选报告（零写入）。

    WAL 读快照事务包住遍历：并发软删/新插不致 OFFSET 窗口跳页/重扫，
    报告语义＝sweep 起始时刻的活条目快照。报告悬空明细带 uid/project/
    kind/ref，供人工核对后自行 forget——本函数不删。
    """
    dangling: list[dict] = []
    by_project: dict[str, int] = {}
    scanned = with_refs = ok = unverifiable = 0
    offset = 0
    store.conn.execute("BEGIN")  # 延迟事务：首读取快照，遍历期一致
    try:
        while True:
            total, rows = store.list_page(limit=page_size, offset=offset)
            if not rows:
                break
            for row in rows:
                scanned += 1
                refs = extract_path_refs(row["content"])
                if not refs:
                    continue
                with_refs += 1
                for ref in refs:
                    status, _ = check_ref(ref, row["project"])
                    if status == "ok":
                        ok += 1
                    elif status == "missing":
                        dangling.append({"uid": row["uid"], "project": row["project"],
                                         "kind": row["kind"], "ref": ref})
                        key = row["project"] or "(none)"
                        by_project[key] = by_project.get(key, 0) + 1
                    else:
                        unverifiable += 1
            offset += len(rows)
            if offset >= total:
                break
    finally:
        store.conn.execute("COMMIT")
    return {"scanned": scanned, "entries_with_refs": with_refs, "refs_ok": ok,
            "dangling": dangling, "unverifiable_refs": unverifiable,
            "dangling_by_project": by_project}
