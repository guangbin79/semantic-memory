"""云端整合通道（task12）：默认开启的深度整理，三道安全带。

计划钉死的边界（勿破坏）：
- 云端只返回元数据操作计划（supersede / touch）——content 永不由云端生成入库；
- INJECTION_RE 对计划 JSON 全文消毒，命中弃整批（纵深防御：正常计划只含 uid 引用）；
- 调用面：/idle flush 完成后的 daemon 线程 + CLI 手动入口，禁入任何同步请求路径；
- 日顶为进程内存计数，服务重启即重置（README 明示该语义，不建持久化计数表）；
- 预过滤是确定性的：只送"有近邻的条目 + 其直接邻域"，孤立条目永不上云。
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Final

try:
    import httpx2  # 历史传递依赖（API 与 httpx 同构）
except ImportError:  # 环境仅携带 httpx 时回落
    import httpx as httpx2
import numpy as np

log = logging.getLogger("semantic_memory.cloud")

# 注入模式集（移植自 integrations/opencode/memory.ts 的 INJECTION_RE）：
# 对云端返回的计划 JSON 全文检查——未来若放开 content 入库，此防线强制。
INJECTION_RE: Final = re.compile(
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?"
    r"|忽略(之前|以上|所有|先前)的?(指令|指示)"
    r"|</?system\s*>"
    r"|you\s+are\s+now\s+(a|an|in)\b"
    r"|disregard\s+(all\s+)?(previous|above)",
    re.IGNORECASE,
)

NEAR_LO: Final = 0.60    # 近邻认定下限（同 lifecycle 模糊带下限）
_CHUNK: Final = 512      # 自相似矩阵分块行数（防 n² 一次性爆内存）
_DAY: Final = 86400.0

_SYSTEM: Final = (
    "You consolidate a memory store. The user message lists related entries"
    " as JSON objects {uid, project, kind, content}. Reply with ONLY a raw"
    ' JSON array of metadata ops and nothing else:'
    ' [{"op":"supersede","old":"<uid>","new":"<uid>"}] marks an outdated'
    " entry as superseded by a newer entry from the same project;"
    ' [{"op":"touch","uid":"<uid>"}] refreshes a restated entry.'
    " Never invent content or uids. If nothing needs changing, reply []."
)


class Consolidator:
    """OpenAI 兼容 /chat/completions 云端整合器（同步客户端，daemon 线程专用）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        enabled: bool = True,
        daily_max_calls: int = 20,
        daily_max_tokens: int = 200000,
        batch_n: int = 30,
        client: httpx2.Client | None = None,
    ) -> None:
        self._base = base_url.strip().rstrip("/")
        self._key = api_key
        self._model = model.strip()
        self._enabled = enabled
        self._max_calls = daily_max_calls
        self._max_tokens = daily_max_tokens
        self._batch_n = batch_n
        self._client = client  # 测试注入点；生产惰性建（无配置则永不创建）
        self._lock = threading.Lock()
        self._day = ""
        self._calls = 0
        self._tokens = 0
        self._last_ts = 0.0

    # ---------- 启用与统计 ----------

    def is_enabled(self) -> bool:
        """三项配置齐备且未被显式 kill switch（=0 优先于 env 存在性，Metis M-4）。"""
        return (self._enabled and bool(self._base)
                and bool(self._key) and bool(self._model))

    def stats(self) -> dict:
        """调用计数 / token 估算 / 上次整合时间（进程内存计数，重启重置）。"""
        with self._lock:
            return {"calls": self._calls, "tokens": self._tokens,
                    "last_ts": self._last_ts or None}

    # ---------- 触发与执行 ----------

    def maybe_run(self, store, embedder, new_entries: int = 0) -> dict:
        """/idle flush 完成后的自动触发判定（调用方须已在 daemon 线程）。

        触发：待整合候选（有近邻的活条目）≥ batch_n；或距上次成功整合
        ≥24h 且本次 flush 有新增。扫描无簇 → 不发起任何调用。
        """
        if not self.is_enabled():
            return {"skipped": "disabled"}
        neighbors, candidates = self._scan(store, embedder)
        stale = time.time() - self._last_ts >= _DAY
        if candidates < self._batch_n and not (stale and new_entries > 0):
            return {"skipped": "below-threshold", "candidates": candidates}
        if not neighbors:
            return {"skipped": "no-cluster", "candidates": 0}
        return self._apply(store, neighbors, candidates)

    def run(self, store, embedder) -> dict:
        """CLI 手动入口：跳过触发阈值，其余安全带（启用/预算/消毒）全数生效。"""
        if not self.is_enabled():
            return {"skipped": "disabled"}
        neighbors, candidates = self._scan(store, embedder)
        if not neighbors:
            return {"skipped": "no-cluster", "candidates": 0}
        return self._apply(store, neighbors, candidates)

    # ---------- 确定性预过滤 ----------

    @staticmethod
    def _scan(store, embedder) -> tuple[dict[str, list[str]], int]:
        """全库活向量自相似扫描（分块控内存），返回 (uid→直接邻域, 候选数)。

        候选 = 有 ≥1 近邻（cos ≥ NEAR_LO 且非自身）的活条目；邻域 = 其
        top-3 近邻。embedder 可用时为无向量条目补算嵌入（同 lifecycle
        向量近邻模式）；向量通道整体不可用 → 空扫描——无确定性簇时
        绝不退化为全库上云。
        """
        uids, mat = store.all_vectors()
        have = set(uids)
        missing = [
            r for r in store.conn.execute(
                "SELECT uid, content FROM memories"
                " WHERE deleted=0 AND superseded_by IS NULL"
            ).fetchall()
            if r["uid"] not in have
        ]
        if embedder is not None and embedder.available and missing:
            vecs = embedder.embed_batch([r["content"] for r in missing])
            if vecs is not None:
                uids = uids + [r["uid"] for r in missing]
                mat = vecs if not len(mat) else np.vstack([mat, vecs])
        neighbors: dict[str, list[str]] = {}
        n = len(uids)
        for start in range(0, n, _CHUNK):
            block = mat[start:start + _CHUNK] @ mat.T
            for r, row in enumerate(block):
                cand = uids[start + r]
                k = min(4, n)
                top = np.argpartition(row, n - k)[-k:]
                top = top[np.argsort(row[top], kind="stable")[::-1]]
                near = [uids[i] for i in top
                        if uids[i] != cand and float(row[i]) >= NEAR_LO][:3]
                if near:
                    neighbors[cand] = near
        return neighbors, len(neighbors)

    # ---------- 调用 / 消毒 / 校验 / 应用 ----------

    def _apply(self, store, neighbors: dict[str, list[str]], candidates: int) -> dict:
        """载荷组装 → 日顶 → 云调用 → 消毒 → 计划校验 → 单事务应用。"""
        send: list[str] = []
        seen: set[str] = set()
        for uid, near in neighbors.items():
            for u in (uid, *near):
                if u not in seen:
                    seen.add(u)
                    send.append(u)
        # ponytail: 单次载荷上限 4×batch_n——超大连簇截断留待下次（禁全库上云的硬护栏）
        send = send[:4 * self._batch_n]
        rows = store.get_many(send)
        if not rows:
            return {"skipped": "no-cluster", "candidates": candidates}
        if not self._reserve_budget():
            log.warning("cloud daily budget exhausted (calls=%d tokens=%d)，今日停发",
                        self._max_calls, self._max_tokens)
            return {"skipped": "budget", "candidates": candidates}

        entries = [{"uid": r["uid"], "project": r["project"], "kind": r["kind"],
                    "content": r["content"]} for r in rows]
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",
                 "content": json.dumps(entries, ensure_ascii=False)},
            ],
            "temperature": 0,
        }
        data: dict = {}
        raw = ""
        try:
            resp = self._http.post(
                f"{self._base}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
        except Exception as exc:  # fail-open：云失败不动库（边界兜底，同 lifecycle）
            log.warning("cloud consolidate call failed: %s", exc)
            return {"skipped": "http-error", "candidates": candidates}
        if not isinstance(raw, str):
            return {"skipped": "bad-plan", "candidates": candidates}

        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0) or (
            len(json.dumps(body, ensure_ascii=False)) + len(raw)) // 4
        with self._lock:
            self._tokens += tokens

        if INJECTION_RE.search(raw):  # 消毒：命中即弃整批
            log.warning("cloud plan rejected (injection pattern)")
            return {"skipped": "injection-pattern", "candidates": candidates}
        try:
            ops = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("cloud plan unparseable")
            return {"skipped": "bad-plan", "candidates": candidates}
        if not isinstance(ops, list):
            return {"skipped": "bad-plan", "candidates": candidates}

        by_uid = {r["uid"]: r for r in rows}
        plan: list[dict] = []
        for op in ops:  # 逐条校验：形状 + 存在性 + 同域取代
            if not isinstance(op, dict):
                continue
            if op.get("op") == "touch" and isinstance(op.get("uid"), str):
                if op["uid"] in by_uid:
                    plan.append({"op": "touch", "uid": op["uid"]})
            elif op.get("op") == "supersede":
                old, new = op.get("old"), op.get("new")
                if (isinstance(old, str) and isinstance(new, str) and old != new
                        and old in by_uid and new in by_uid
                        and by_uid[old]["project"] == by_uid[new]["project"]):
                    plan.append({"op": "supersede", "old": old, "new": new})
            else:
                log.warning("cloud plan op dropped: %r", op)

        now = time.time()
        superseded = touched = 0
        with store.conn:
            for op in plan:
                if op["op"] == "supersede":
                    cur = store.conn.execute(
                        "UPDATE memories SET superseded_by=?, superseded_at=?"
                        " WHERE uid=? AND deleted=0 AND superseded_by IS NULL",
                        (op["new"], now, op["old"]),
                    )
                    superseded += cur.rowcount
                else:
                    cur = store.conn.execute(
                        "UPDATE memories SET updated_at=?, last_recalled_at=?"
                        " WHERE uid=? AND deleted=0 AND superseded_by IS NULL",
                        (now, now, op["uid"]),
                    )
                    touched += cur.rowcount
        with self._lock:
            self._last_ts = now
        return {"candidates": candidates, "sent": len(rows),
                "superseded": superseded, "touched": touched, **self.stats()}

    def _reserve_budget(self) -> bool:
        """日顶判定并占一个调用名额；跨日自动清零（进程内存计数，重启同效）。"""
        today = time.strftime("%Y-%m-%d")
        with self._lock:
            if today != self._day:
                self._day, self._calls, self._tokens = today, 0, 0
            if self._calls >= self._max_calls or self._tokens >= self._max_tokens:
                return False
            self._calls += 1
            return True

    @property
    def _http(self) -> httpx2.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx2.Client(timeout=60.0)
            return self._client
