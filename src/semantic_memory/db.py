"""存储层：SQLite + FTS5(外部内容表) + 向量 BLOB。单写者，线程内单连接。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from ._compat import np


def content_hash(text: str) -> str:
    """去重键：去首尾空白后的 sha256 截断。精确去重与迁移回填共用。"""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:32]

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    uid        TEXT NOT NULL UNIQUE,
    content    TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'fact',      -- fact|decision|preference|artifact
    source     TEXT NOT NULL DEFAULT 'manual',    -- manual|chat|compaction|tool
    project    TEXT,
    tags       TEXT NOT NULL DEFAULT '[]',        -- JSON array
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    deleted    INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT,                           -- 摄取去重键（sha256[:32]）
    injection_count INTEGER NOT NULL DEFAULT 0   -- 被注入次数（反哺热路径计数，record_injection 同事务自增；老库经 _migrate 回填）
);
-- 双 FTS 通道：porter 处理英文词干/标签，trigram 处理 CJK 子串（≥3 字符）。
-- 中文连续文本在 porter 下会塌缩成单个大 token 导致不可检索，trigram 通道是中文召回的关键。
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags,
    content='memories', content_rowid='id',
    tokenize='porter'
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts_tri USING fts5(
    content,
    content='memories', content_rowid='id',
    tokenize='trigram'
);
-- 外部内容表同步触发器（两表各一组）
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
    INSERT INTO memories_fts_tri(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts_tri(memories_fts_tri, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.id, new.content, new.tags);
    INSERT INTO memories_fts_tri(memories_fts_tri, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts_tri(rowid, content) VALUES (new.id, new.content);
END;
CREATE TABLE IF NOT EXISTS memories_vec (
    uid       TEXT PRIMARY KEY REFERENCES memories(uid),
    embedding BLOB NOT NULL                        -- float32 little-endian
);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project) WHERE deleted = 0;
CREATE INDEX IF NOT EXISTS idx_memories_kind    ON memories(kind)    WHERE deleted = 0;
-- 注入遥测：插件/适配器实际完成注入后上报（召回≠注入，只有插件知道注没注）。
-- 记录的是"供给侧"事实——模型是否采纳不可知，不做伪精确。
CREATE TABLE IF NOT EXISTS injections (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    harness TEXT NOT NULL DEFAULT '',   -- opencode / claude-code / kimi-code / ...
    hook    TEXT NOT NULL DEFAULT '',   -- context / compaction / generate / chat.message / ...
    query   TEXT NOT NULL DEFAULT '',
    project TEXT,
    chars   INTEGER NOT NULL DEFAULT 0, -- 注入块字符数
    uids    TEXT NOT NULL DEFAULT '[]'  -- JSON 数组：本次注入的记忆 uid
);
CREATE INDEX IF NOT EXISTS idx_injections_ts ON injections(ts);
-- 蒸馏队列：写入侧异步蒸馏的落盘缓冲（flush 前持久，崩溃不丢）。
CREATE TABLE IF NOT EXISTS distill_queue (
    id         INTEGER PRIMARY KEY,
    content    TEXT,
    source     TEXT,
    project    TEXT,
    session_id TEXT,
    created_at REAL,
    status     TEXT DEFAULT 'pending'
);
"""

VALID_KINDS = {"fact", "decision", "preference", "artifact"}

# memories 写入 SQL：store() 与 flush_distill_queue（独立连接）共用，FTS 由触发器同步
INSERT_MEMORY_SQL = (
    "INSERT INTO memories(uid, content, kind, source, project, tags,"
    " created_at, updated_at, content_hash, session_id)"
    " VALUES (?,?,?,?,?,?,?,?,?,?)"
)


class _SerializedCursor:
    """急切物化的游标：execute 时在锁内 fetchall，之后无锁读取也安全。

    本项目的全部调用点只使用 fetchone（单次）/fetchall/迭代/rowcount，
    因此物化包装即可覆盖，不追求 DB-API 完整语义。
    """

    def __init__(self, cur: sqlite3.Cursor):
        self._rows = cur.fetchall()
        self.rowcount = cur.rowcount
        cur.close()

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list:
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class _SerializedConn:
    """单连接的跨线程序列化门面。

    线程化 HTTP 服务器会在工作线程里执行路由，
    而 sqlite3 连接默认绑定创建线程（check_same_thread）。门面用 RLock
    保证：单条语句原子执行并急切物化结果；`with conn:` 事务块整体持锁，
    多语句事务不会与别的线程交织。
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()
        # 本连接写代数：PRAGMA data_version 不反映同连接提交，向量缓存须自计。
        # 多计只导致过度失效（多查一次），永不返旧数据——安全方向。
        self.write_gen = 0


    def execute(self, *args, **kwargs) -> _SerializedCursor:
        with self._lock:
            return _SerializedCursor(self._conn.execute(*args, **kwargs))

    def executemany(self, sql, params) -> None:
        # 约束：调用方必须在 `with self.conn:` 事务内使用——裸用不 commit 也不递增
        # write_gen，向量缓存将看不到该写（现仓库唯一调用点 _migrate 在事务内）。
        with self._lock:
            cur = self._conn.executemany(sql, params)
            cur.close()

    def executescript(self, *args, **kwargs) -> None:
        with self._lock:
            self._conn.executescript(*args, **kwargs)
            self.write_gen += 1

    def __enter__(self) -> "_SerializedConn":
        self._lock.acquire()
        self._conn.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        try:
            return self._conn.__exit__(*exc)
        finally:
            self._lock.release()
            self.write_gen += 1  # 无条件多计：rollback 也失效缓存，只多查不返旧

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()
            self.write_gen += 1

    def rollback(self) -> None:
        with self._lock:
            self._conn.rollback()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class MemoryStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(str(self.db_path), check_same_thread=False)
        raw.row_factory = sqlite3.Row
        self.conn = _SerializedConn(raw)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        had_fts = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone() is not None
        self.conn.executescript(SCHEMA)
        if not had_fts:
            self._rebuild_fts()  # legacy 行补进索引：外部内容表行不在索引内时，任何 UPDATE 触发器的 'delete' 都会损坏库
        self._migrate()
        # 向量矩阵进程内缓存：key=(project,scope) → (uids, mat, data_version, write_gen)。
        # ponytail: 个人规模（≤50k×4KB≈200MB/键）直接留驻内存，键数≤4；超规模再换 LRU。
        self._vec_cache: dict[tuple, tuple] = {}

    def _rebuild_fts(self) -> None:
        """两通道 FTS 重建（新库无行时是零成本空扫）。

        根因：外部内容表的 legacy 行从未入索引，UPDATE 触发器对其发 'delete' 命令
        → sqlite 报 database disk image is malformed（3.37 实测）。建表后重建一次即根治。
        """
        self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        self.conn.execute("INSERT INTO memories_fts_tri(memories_fts_tri) VALUES('rebuild')")

    def _migrate(self) -> None:
        """轻量迁移：为既有库补缺失列（content_hash + 会话/取代/使用度扩展列），索引始终确保。

        索引不在 SCHEMA 里建——旧库缺列时 CREATE INDEX 会在列添加之前失败。
        """
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(memories)")}
        with self.conn:
            if "content_hash" not in cols:
                self.conn.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")
                for rid, content in self.conn.execute("SELECT id, content FROM memories"):
                    self.conn.execute(
                        "UPDATE memories SET content_hash=? WHERE id=?",
                        (content_hash(content), rid),
                    )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_hash"
                " ON memories(content_hash) WHERE deleted=0"
            )
            # 扩展列：会话身份 / 召回打点 / 取代链（老库补列，已有列跳过保证幂等）
            for col, decl in (
                ("session_id", "TEXT"),
                ("last_recalled_at", "REAL"),
                ("superseded_by", "TEXT"),
                ("superseded_at", "REAL"),
            ):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")
            # 反哺计数列：老库补列；回填拆独立事务（同事务 ALTER+触发器+json_each 子查询实测 sqlite3.37 报 malformed）
            need_count_backfill = "injection_count" not in cols
            if need_count_backfill:
                self.conn.execute(
                    "ALTER TABLE memories ADD COLUMN injection_count INTEGER NOT NULL DEFAULT 0")
            # injections 扩展列：会话归属（面板按当前会话过滤）＋注入块全文预览（老库补列，幂等）
            icols = {r[1] for r in self.conn.execute("PRAGMA table_info(injections)")}
            for col, decl in (("session_id", "TEXT"), ("block", "TEXT")):
                if col not in icols:
                    self.conn.execute(f"ALTER TABLE injections ADD COLUMN {col} {decl}")
        if need_count_backfill:
            # 反哺计数回填：聚合 json_each 后逐行 UPDATE（legacy 行已由 _rebuild_fts 入索引，触发器安全）
            counts = self.conn.execute(
                "SELECT j.value AS uid, COUNT(*) AS c FROM injections i, json_each(i.uids) j"
                " GROUP BY j.value").fetchall()
            with self.conn:
                self.conn.executemany(
                    "UPDATE memories SET injection_count=? WHERE uid=?",
                    [(r["c"], r["uid"]) for r in counts],
                )

    # ---------- 写入 ----------

    def store(
        self,
        content: str,
        kind: str = "fact",
        source: str = "manual",
        project: str | None = None,
        tags: list[str] | None = None,
        embedding: np.ndarray | None = None,
        session_id: str | None = None,
    ) -> str:
        if kind not in VALID_KINDS:
            raise ValueError(f"invalid kind {kind!r}, expected one of {sorted(VALID_KINDS)}")
        uid = uuid.uuid4().hex[:16]
        now = time.time()
        with self.conn:
            self.conn.execute(
                INSERT_MEMORY_SQL,
                (uid, content, kind, source, project, json.dumps(tags or [], ensure_ascii=False),
                 now, now, content_hash(content), session_id),
            )
            if embedding is not None:
                vec = np.asarray(embedding, dtype=np.float32)
                self.conn.execute(
                    "INSERT INTO memories_vec(uid, embedding) VALUES (?,?)",
                    (uid, vec.tobytes()),
                )
        return uid

    def forget(self, uid: str) -> bool:
        """软删除：FTS 通过 UPDATE 触发器同步，向量直接移除。"""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE memories SET deleted=1, updated_at=? WHERE uid=? AND deleted=0",
                (time.time(), uid),
            )
            self.conn.execute("DELETE FROM memories_vec WHERE uid=?", (uid,))
        return cur.rowcount > 0

    # ---------- 蒸馏队列（写入侧缓冲；flush 见 lifecycle.flush_distill_queue） ----------

    def distill_enqueue(self, content: str, source: str, project: str | None,
                        session_id: str | None = None) -> int:
        """入队 pending 行（flush 前持久，崩溃不丢），返回当前 pending 总数。"""
        with self.conn:
            self.conn.execute(
                "INSERT INTO distill_queue(content, source, project, session_id, created_at, status)"
                " VALUES (?,?,?,?,?,'pending')",
                (content, source, project, session_id, time.time()),
            )
            return self.distill_pending()

    # ---------- 去重与维护（维护面：启动钩子/CLI，不经 MCP 工具暴露） ----------

    def distill_pending(self) -> int:
        """当前 pending 行数（观测面只读）。"""
        return self.conn.execute(
            "SELECT COUNT(*) c FROM distill_queue WHERE status='pending'"
        ).fetchone()["c"]

    def find_by_hash(self, h: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT uid, updated_at FROM memories WHERE content_hash=? AND deleted=0",
            (h,),
        ).fetchone()

    def touch(self, uid: str) -> None:
        """去重命中：刷新时间戳（等价于'这条记忆再次被证实相关'）。"""
        now = time.time()
        with self.conn:
            self.conn.execute(
                "UPDATE memories SET updated_at=?, last_recalled_at=? WHERE uid=?",
                (now, now, uid),
            )

    def distinct_projects(self) -> list[str]:
        """memories 与 injections 两表的全部非空 project 值（回填迁移用）。"""
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT project FROM memories WHERE project IS NOT NULL"
                " UNION SELECT DISTINCT project FROM injections WHERE project IS NOT NULL"
            )
        ]

    def rename_project(self, old: str, new: str) -> int:
        """project 键批量改写：memories 与 injections 同事务（回填迁移用）。"""
        with self.conn:
            c1 = self.conn.execute(
                "UPDATE memories SET project=? WHERE project=?", (new, old)).rowcount
            c2 = self.conn.execute(
                "UPDATE injections SET project=? WHERE project=?", (new, old)).rowcount
        return c1 + c2

    def purge_soft_deleted(self, dry_run: bool = False) -> dict:
        """硬删软删除条目，真正回收空间（dry_run 只统计不动库）。返回 {count, bytes}。"""
        rows = self.conn.execute(
            "SELECT uid, LENGTH(CAST(content AS BLOB)) AS nbytes FROM memories WHERE deleted=1"
        ).fetchall()
        stats = {"count": len(rows), "bytes": sum(r["nbytes"] or 0 for r in rows)}
        if stats["count"] and not dry_run:
            with self.conn:
                self.conn.execute("DELETE FROM memories WHERE deleted=1")
        return stats

    def purge_older_than(self, kind: str, cutoff_ts: float, dry_run: bool = False) -> dict:
        """TTL 硬清：指定 kind 且创建时间早于 cutoff 的活条目（dry_run 只统计）。

        DELETE 触发器同步 FTS。返回 {count, bytes}。
        """
        rows = self.conn.execute(
            "SELECT uid, LENGTH(CAST(content AS BLOB)) AS nbytes FROM memories"
            " WHERE kind=? AND deleted=0 AND created_at<?",
            (kind, cutoff_ts),
        ).fetchall()
        stats = {"count": len(rows), "bytes": sum(r["nbytes"] or 0 for r in rows)}
        if not rows or dry_run:
            return stats
        uids = [r["uid"] for r in rows]
        marks = ",".join("?" * len(uids))
        with self.conn:
            self.conn.execute(f"DELETE FROM memories_vec WHERE uid IN ({marks})", uids)
            self.conn.execute(f"DELETE FROM memories WHERE uid IN ({marks})", uids)
        return stats

    def stamp_recalled(self, uids: list[str]) -> int:
        """注入上报：被注入=被召回，刷新 last_recalled_at（留存计时的豁免凭证）。"""
        if not uids:
            return 0
        marks = ",".join("?" * len(uids))
        with self.conn:
            cur = self.conn.execute(
                f"UPDATE memories SET last_recalled_at=?"
                f" WHERE uid IN ({marks}) AND deleted=0",
                (time.time(), *uids),
            )
        return cur.rowcount

    def stale_candidates(self, cutoff_ts: float) -> list[sqlite3.Row]:
        """使用度留存候选：fact/artifact 超期未召回（NULL 视为 created_at）或已被取代。

        preference/decision 永久豁免（即使超期未召回）。
        """
        return self.conn.execute(
            "SELECT uid, kind, LENGTH(CAST(content AS BLOB)) AS nbytes FROM memories"
            " WHERE deleted=0 AND kind IN ('fact','artifact')"
            " AND (superseded_by IS NOT NULL OR COALESCE(last_recalled_at, created_at) < ?)",
            (cutoff_ts,),
        ).fetchall()

    def soft_delete_uids(self, uids: list[str]) -> int:
        """按 uid 批量软删除（FTS 走 UPDATE 触发器同步，向量直接移除）。"""
        if not uids:
            return 0
        marks = ",".join("?" * len(uids))
        with self.conn:
            self.conn.execute(f"DELETE FROM memories_vec WHERE uid IN ({marks})", uids)
            cur = self.conn.execute(
                f"UPDATE memories SET deleted=1, updated_at=?"
                f" WHERE uid IN ({marks}) AND deleted=0",
                (time.time(), *uids),
            )
        return cur.rowcount

    def entries_without_vectors(self, limit: int = 500) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT m.uid, m.content FROM memories m"
            " LEFT JOIN memories_vec v ON m.uid = v.uid"
            " WHERE m.deleted = 0 AND v.uid IS NULL LIMIT ?",
            (limit,),
        ).fetchall()

    def insert_vector(self, uid: str, vec) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO memories_vec(uid, embedding) VALUES (?,?)",
                (uid, np.asarray(vec, dtype=np.float32).tobytes()),
            )

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

    # ---------- 读取 ----------

    @staticmethod
    def _scope_cond(scope: str) -> str:
        """scope 过滤片段（'project'|'global'|'all'）：拼进 WHERE，不带参数。"""
        if scope == "global":
            return " AND m.project IS NULL"
        if scope == "project":
            return " AND m.project IS NOT NULL"
        if scope == "all":
            return ""
        raise ValueError(f"invalid scope {scope!r}, expected 'all'|'project'|'global'")

    def fts_search(self, query: str, limit: int, scope: str = "all",
                 project: str | None = None) -> list[tuple[str, float]]:
        """双通道 BM25 检索，返回 [(uid, best_rank)]，rank 越小越好。

        porter 通道：英文词干 + 标签；trigram 通道：CJK/子串（仅 ≥3 字符的词）。
        两路结果按 uid 合并取最优 rank，顺序即 BM25 升序。
        project 在场（且非 global 域）时两腿都加等值过滤——与向量腿对齐，
        项目域召回不再漏入他项目条目（2026-10-17 硬化）。
        """
        terms = [t for t in query.replace('"', " ").split() if t]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        best: dict[str, float] = {}

        # 与向量腿 all_vectors 同条件：仅 scope='all' 时启用（'project' 档语义＝排全域跨项目，两腿一致不窄化）
        proj_cond = " AND m.project = ?" if (project and scope == "all") else ""
        proj_params: tuple = (project,) if proj_cond else ()
        rows = self.conn.execute(
            f"""
            SELECT m.uid, bm25(memories_fts) AS r
            FROM memories_fts f JOIN memories m ON m.id = f.rowid
            WHERE memories_fts MATCH ? AND m.deleted = 0 AND m.superseded_by IS NULL
            {self._scope_cond(scope)}{proj_cond}
            ORDER BY r LIMIT ?
            """,
            (fts_query, *proj_params, limit),
        ).fetchall()
        for r in rows:
            best[r["uid"]] = float(r["r"])

        tri_terms = [t for t in terms if len(t) >= 3]  # trigram 不支持 <3 字符
        if tri_terms:
            tri_query = " OR ".join(f'"{t}"' for t in tri_terms)
            rows = self.conn.execute(
                f"""
                SELECT m.uid, bm25(memories_fts_tri) AS r
                FROM memories_fts_tri f JOIN memories m ON m.id = f.rowid
                WHERE memories_fts_tri MATCH ? AND m.deleted = 0 AND m.superseded_by IS NULL
                {self._scope_cond(scope)}{proj_cond}
                ORDER BY r LIMIT ?
                """,
                (tri_query, *proj_params, limit),
            ).fetchall()
            for r in rows:
                uid = r["uid"]
                if uid not in best or float(r["r"]) < best[uid]:
                    best[uid] = float(r["r"])

        return sorted(best.items(), key=lambda kv: kv[1])[:limit]

    def all_vectors(
        self, project: str | None = None, scope: str = "all"
    ) -> tuple[list[str], np.ndarray]:
        """取出全部活向量用于暴力余弦；带进程内缓存（读多写少的热路径主优化）。

        失效双轨：本连接写代数 write_gen + PRAGMA data_version（任何其它连接
        提交都会推进，如 distill flush 的独立连接）。两代数都没变 → 直接返回
        上次矩阵，免掉 fetchall+物化（实测 N=5k 约 58ms/次的主导成本）。
        代数在查询前采样（时序正确性）：若查询期间发生写，存下的是旧代数，
        下一次调用必然失效重查——宁可多查一次，绝不把旧数据配新代数驻缓存。
        已知窗口（设计取舍）：同连接事务内先写后读本缓存可能读到事务前快照
        （write_gen 在 __exit__ 才递增）——现无此调用序，新增调用方勿在
        `with conn:` 写事务中途调 all_vectors。
        scope='global' 只取 project IS NULL，'project' 排除 NULL；
        这两档下 project 参数被忽略。
        """
        key = (project, scope)
        try:
            dv = self.conn.execute("PRAGMA data_version").fetchone()[0]
        except sqlite3.Error:
            dv = None
        gen = self.conn.write_gen
        hit = self._vec_cache.get(key)
        if hit is not None and hit[2] == dv and hit[3] == gen:
            return hit[0], hit[1]

        sql = (
            "SELECT v.uid, v.embedding FROM memories_vec v"
            " JOIN memories m ON m.uid = v.uid WHERE m.deleted = 0 AND m.superseded_by IS NULL"
            + self._scope_cond(scope)
        )
        params: tuple = ()
        if project and scope == "all":
            sql += " AND m.project = ?"
            params = (project,)
        rows = self.conn.execute(sql, params).fetchall()
        if np is None:
            # 无 numpy 基座：不物化矩阵（缓存该降级结果亦无害）
            self._vec_cache[key] = ([], None, dv, gen)
            return [], None
        if not rows:
            mat = np.empty((0, 0), dtype=np.float32)
            self._vec_cache[key] = ([], mat, dv, gen)
            return [], mat
        uids = [r["uid"] for r in rows]
        mat = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
        dim = len(rows[0]["embedding"]) // 4
        mat = mat.reshape(len(rows), dim)
        self._vec_cache[key] = (uids, mat, dv, gen)
        return uids, mat

    # ---------- 跨项目图谱边源（查询侧派生边；编排见 graph.py） ----------

    def supersede_predecessors(self, uids: list[str]) -> list[sqlite3.Row]:
        """被这些 uid 直接取代的前代（管理面读：含被取代条目，召回面不可见）。"""
        if not uids:
            return []
        marks = ",".join("?" * len(uids))
        return self.conn.execute(
            f"SELECT uid, content, kind, project, superseded_by FROM memories"
            f" WHERE superseded_by IN ({marks}) AND deleted = 0 ORDER BY uid",
            uids,
        ).fetchall()

    def co_injection_counts(self, seed_uids: list[str], limit: int = 20) -> list[sqlite3.Row]:
        """与种子同 injections 行共现过的其他 uid：返回 (to_uid, via_uid, c=共现行数)。

        只看近 30 天遥测（骑 idx_injections_ts 有界扫）——全表双重 json_each 在
        召回热路径不可接受，与 injection_counts 的计数列化同一取舍（见其注释）。
        """
        if not seed_uids:
            return []
        marks = ",".join("?" * len(seed_uids))
        return self.conn.execute(
            f"SELECT j2.value AS to_uid, MIN(j1.value) AS via_uid, COUNT(DISTINCT i.id) AS c"
            f" FROM injections i, json_each(i.uids) j1, json_each(i.uids) j2"
            f" WHERE i.ts > ? AND j1.value IN ({marks}) AND j2.value NOT IN ({marks})"
            f" GROUP BY j2.value ORDER BY c DESC, to_uid LIMIT ?",
            (time.time() - 30 * 86400, *seed_uids, *seed_uids, limit),
        ).fetchall()

    def get_many(self, uids: list[str]) -> list[sqlite3.Row]:
        if not uids:
            return []
        marks = ",".join("?" * len(uids))
        rows = self.conn.execute(
            f"SELECT uid, content, kind, source, project, tags, created_at, updated_at FROM memories"
            f" WHERE uid IN ({marks}) AND deleted = 0 AND superseded_by IS NULL",
            uids,
        ).fetchall()
        order = {u: i for i, u in enumerate(uids)}
        return sorted(rows, key=lambda r: order[r["uid"]])

    _ROW_COLS = "uid, content, kind, source, project, tags, created_at, updated_at"

    def get_one(self, uid: str) -> sqlite3.Row | None:
        return self.conn.execute(
            f"SELECT {self._ROW_COLS} FROM memories WHERE uid=? AND deleted=0", (uid,)
        ).fetchone()

    def list_page(
        self,
        kind: str | None = None,
        source: str | None = None,
        project: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[sqlite3.Row]]:
        """分页 + 过滤（编排历史浏览用）。返回 (total, rows)，按时间倒序。"""
        conds, params = ["deleted = 0"], []
        for col, val in (("kind", kind), ("source", source), ("project", project)):
            if val:
                conds.append(f"{col} = ?")
                params.append(val)
        where = " AND ".join(conds)
        total = self.conn.execute(
            f"SELECT COUNT(*) c FROM memories WHERE {where}", params
        ).fetchone()["c"]
        rows = self.conn.execute(
            f"SELECT {self._ROW_COLS} FROM memories WHERE {where}"
            " ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, min(limit, 200), offset),
        ).fetchall()
        return total, rows

    def projects(self) -> list[str]:
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT project FROM memories"
                " WHERE deleted=0 AND project IS NOT NULL ORDER BY 1"
            )
        ]

    def list_recent(
        self,
        project: str | None = None,
        kind: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        sql = (
            "SELECT uid, content, kind, source, project, tags, created_at, updated_at"
            " FROM memories WHERE deleted = 0"
        )
        params: list = []
        if project:
            sql += " AND project = ?"
            params.append(project)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return self.conn.execute(sql, params).fetchall()

    def stats(self) -> dict:
        total = self.conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE deleted=0"
        ).fetchone()["c"]
        vecs = self.conn.execute("SELECT COUNT(*) c FROM memories_vec").fetchone()["c"]
        by_kind = {
            r["kind"]: r["c"]
            for r in self.conn.execute(
                "SELECT kind, COUNT(*) c FROM memories WHERE deleted=0 GROUP BY kind"
            )
        }
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"total": total, "vectorized": vecs, "by_kind": by_kind, "db_bytes": size}

    # ---------------- 注入遥测 ----------------

    def record_injection(
        self,
        harness: str,
        hook: str,
        query: str,
        uids: list[str],
        chars: int,
        project: str | None = None,
        ts: float | None = None,
        session_id: str | None = None,
        block: str | None = None,
    ) -> int:
        with self.conn:  # 事务块整体持锁：last_insert_rowid 不会与别的线程交错
            trimmed = uids[:50]
            self.conn.execute(
                "INSERT INTO injections(ts, harness, hook, query, project, chars, uids, session_id, block)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts if ts is not None else time.time(),
                 harness[:64], hook[:64], query[:2000], project, max(0, int(chars)),
                 json.dumps(trimmed), session_id[:128] if session_id else None,
                 block[:2000] if block else None),
            )
            if trimmed:  # 反哺计数同事务自增：injection_counts 热路径免扫遥测表
                marks = ",".join("?" * len(trimmed))
                self.conn.execute(
                    f"UPDATE memories SET injection_count = injection_count + 1"
                    f" WHERE uid IN ({marks})", trimmed)
            return self.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    def list_injections(
        self,
        harness: str | None = None,
        project: str | None = None,
        limit: int = 50,
        offset: int = 0,
        session_id: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM injections"
        cond, args = [], []
        if harness:
            cond.append("harness = ?")
            args.append(harness)
        if project:
            cond.append("project = ?")
            args.append(project)
        if session_id:
            cond.append("session_id = ?")
            args.append(session_id)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [dict(r) for r in self.conn.execute(sql, args)]

    def memories_by_uids(self, uids: list[str]) -> dict[str, dict]:
        """uid→记忆摘要（注入展开/面板预览用）；未知与已删 uid 静默缺失。"""
        if not uids:
            return {}
        marks = ",".join("?" * len(uids))
        rows = self.conn.execute(
            f"SELECT uid, kind, project, content FROM memories WHERE uid IN ({marks}) AND deleted=0",
            uids,
        ).fetchall()
        return {
            r["uid"]: {"kind": r["kind"], "project": r["project"], "preview": r["content"][:120]}
            for r in rows
        }

    def content_exists(self, h: str) -> bool:
        """全局内容哈希存在性（跨项目捕获去重用）。"""
        return self.conn.execute(
            "SELECT 1 FROM memories WHERE content_hash=? AND deleted=0 LIMIT 1", (h,)
        ).fetchone() is not None

    def pending_queue_has(self, h: str) -> bool:
        """pending 蒸馏队列同 hash 内容检查（队列无哈希列，逐条现算；量小可接受）。"""
        return any(
            content_hash(r["content"]) == h
            for r in self.conn.execute(
                "SELECT content FROM distill_queue WHERE status='pending'")
        )

    def injection_counts(self, uids: list[str]) -> dict[str, int]:
        """给定 uid 集合的历史注入次数（遥测反哺排序用）。

        读 memories.injection_count 计数列（record_injection 同事务维护，
        老库迁移时自遥测回填）——每次召回不再对 injections 表做 json_each
        全表 GROUP BY（遥测 TTL 90 天可达数万行，O(injections)/召回）。
        """
        if not uids:
            return {}
        marks = ",".join("?" * len(uids))
        return {
            r["uid"]: r["injection_count"]
            for r in self.conn.execute(
                f"SELECT uid, injection_count FROM memories"
                f" WHERE uid IN ({marks}) AND injection_count > 0",
                uids,
            )
        }

    def purge_injections_before(self, cutoff_ts: float, dry_run: bool = False) -> dict:
        """注入遥测 TTL（dry_run 只统计）。返回 {count, bytes=0}（遥测行不计内容字节）。"""
        count = self.conn.execute(
            "SELECT COUNT(*) c FROM injections WHERE ts < ?", (cutoff_ts,)
        ).fetchone()["c"]
        if count and not dry_run:
            self.conn.execute("DELETE FROM injections WHERE ts < ?", (cutoff_ts,))
            self.conn.commit()
        return {"count": count, "bytes": 0}

    def injection_stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) c FROM injections").fetchone()["c"]
        chars = self.conn.execute(
            "SELECT COALESCE(SUM(chars),0) s FROM injections").fetchone()["s"]
        by_harness = {
            r["harness"]: r["c"]
            for r in self.conn.execute(
                "SELECT harness, COUNT(*) c FROM injections GROUP BY harness")
        }
        by_hook = {
            r["hook"]: r["c"]
            for r in self.conn.execute(
                "SELECT hook, COUNT(*) c FROM injections GROUP BY hook")
        }
        recent7 = self.conn.execute(
            "SELECT COUNT(*) c FROM injections WHERE ts > ?",
            (time.time() - 7 * 86400,),
        ).fetchone()["c"]
        # 热门条目：json_each 展开 uids 统计注入频次 Top10
        top = [
            {"uid": r["value"], "times": r["c"]}
            for r in self.conn.execute(
                "SELECT j.value, COUNT(*) c FROM injections,"
                " json_each(injections.uids) j GROUP BY j.value"
                " ORDER BY c DESC LIMIT 10")
        ]
        return {"total": total, "total_chars": chars, "recent_7d": recent7,
                "by_harness": by_harness, "by_hook": by_hook, "top_uids": top}

    def close(self) -> None:
        self.conn.close()
