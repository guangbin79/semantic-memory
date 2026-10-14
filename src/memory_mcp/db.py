"""存储层：SQLite + FTS5(外部内容表) + 向量 BLOB。单写者，线程内单连接。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np


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
    content_hash TEXT                            -- 摄取去重键（sha256[:32]）
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
"""

VALID_KINDS = {"fact", "decision", "preference", "artifact"}


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

    ASGI 服务器（uvicorn / TestClient）会在工作线程里执行同步路由，
    而 sqlite3 连接默认绑定创建线程（check_same_thread）。门面用 RLock
    保证：单条语句原子执行并急切物化结果；`with conn:` 事务块整体持锁，
    多语句事务不会与别的线程交织。
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, *args, **kwargs) -> _SerializedCursor:
        with self._lock:
            return _SerializedCursor(self._conn.execute(*args, **kwargs))

    def executescript(self, *args, **kwargs) -> None:
        with self._lock:
            self._conn.executescript(*args, **kwargs)

    def __enter__(self) -> "_SerializedConn":
        self._lock.acquire()
        self._conn.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        try:
            return self._conn.__exit__(*exc)
        finally:
            self._lock.release()

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

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
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """轻量迁移：为既有库补 content_hash 列（摄取去重键），索引始终确保。

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

    # ---------- 写入 ----------

    def store(
        self,
        content: str,
        kind: str = "fact",
        source: str = "manual",
        project: str | None = None,
        tags: list[str] | None = None,
        embedding: np.ndarray | None = None,
    ) -> str:
        if kind not in VALID_KINDS:
            raise ValueError(f"invalid kind {kind!r}, expected one of {sorted(VALID_KINDS)}")
        uid = uuid.uuid4().hex[:16]
        now = time.time()
        with self.conn:
            self.conn.execute(
                "INSERT INTO memories(uid, content, kind, source, project, tags,"
                " created_at, updated_at, content_hash)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (uid, content, kind, source, project, json.dumps(tags or [], ensure_ascii=False),
                 now, now, content_hash(content)),
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

    # ---------- 去重与维护（维护面：启动钩子/CLI，不经 MCP 工具暴露） ----------

    def find_by_hash(self, h: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT uid, updated_at FROM memories WHERE content_hash=? AND deleted=0",
            (h,),
        ).fetchone()

    def touch(self, uid: str) -> None:
        """去重命中：刷新时间戳（等价于'这条记忆再次被证实相关'）。"""
        with self.conn:
            self.conn.execute(
                "UPDATE memories SET updated_at=? WHERE uid=?", (time.time(), uid)
            )

    def purge_soft_deleted(self) -> int:
        """硬删软删除条目，真正回收空间（向量在 forget 时已移除）。"""
        with self.conn:
            cur = self.conn.execute("DELETE FROM memories WHERE deleted=1")
        return cur.rowcount

    def purge_older_than(self, kind: str, cutoff_ts: float) -> int:
        """TTL 硬清：指定 kind 且创建时间早于 cutoff 的活条目。DELETE 触发器同步 FTS。"""
        uids = [
            r["uid"]
            for r in self.conn.execute(
                "SELECT uid FROM memories WHERE kind=? AND deleted=0 AND created_at<?",
                (kind, cutoff_ts),
            )
        ]
        if not uids:
            return 0
        marks = ",".join("?" * len(uids))
        with self.conn:
            self.conn.execute(f"DELETE FROM memories_vec WHERE uid IN ({marks})", uids)
            cur = self.conn.execute(f"DELETE FROM memories WHERE uid IN ({marks})", uids)
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

    def fts_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """双通道 BM25 检索，返回 [(uid, best_rank)]，rank 越小越好。

        porter 通道：英文词干 + 标签；trigram 通道：CJK/子串（仅 ≥3 字符的词）。
        两路结果按 uid 合并取最优 rank，顺序即 BM25 升序。
        """
        terms = [t for t in query.replace('"', " ").split() if t]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        best: dict[str, float] = {}

        rows = self.conn.execute(
            """
            SELECT m.uid, bm25(memories_fts) AS r
            FROM memories_fts f JOIN memories m ON m.id = f.rowid
            WHERE memories_fts MATCH ? AND m.deleted = 0
            ORDER BY r LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        for r in rows:
            best[r["uid"]] = float(r["r"])

        tri_terms = [t for t in terms if len(t) >= 3]  # trigram 不支持 <3 字符
        if tri_terms:
            tri_query = " OR ".join(f'"{t}"' for t in tri_terms)
            rows = self.conn.execute(
                """
                SELECT m.uid, bm25(memories_fts_tri) AS r
                FROM memories_fts_tri f JOIN memories m ON m.id = f.rowid
                WHERE memories_fts_tri MATCH ? AND m.deleted = 0
                ORDER BY r LIMIT ?
                """,
                (tri_query, limit),
            ).fetchall()
            for r in rows:
                uid = r["uid"]
                if uid not in best or float(r["r"]) < best[uid]:
                    best[uid] = float(r["r"])

        return sorted(best.items(), key=lambda kv: kv[1])[:limit]

    def all_vectors(self, project: str | None = None) -> tuple[list[str], np.ndarray]:
        """取出全部活向量用于暴力余弦（个人规模 <10k 条时亚毫秒级，无需 ANN）。"""
        sql = (
            "SELECT v.uid, v.embedding FROM memories_vec v"
            " JOIN memories m ON m.uid = v.uid WHERE m.deleted = 0"
        )
        params: tuple = ()
        if project:
            sql += " AND m.project = ?"
            params = (project,)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        uids = [r["uid"] for r in rows]
        mat = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
        dim = len(rows[0]["embedding"]) // 4
        return uids, mat.reshape(len(rows), dim)

    def get_many(self, uids: list[str]) -> list[sqlite3.Row]:
        if not uids:
            return []
        marks = ",".join("?" * len(uids))
        rows = self.conn.execute(
            f"SELECT uid, content, kind, source, project, tags, created_at FROM memories"
            f" WHERE uid IN ({marks}) AND deleted = 0",
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
    ) -> int:
        with self.conn:  # 事务块整体持锁：last_insert_rowid 不会与别的线程交错
            self.conn.execute(
                "INSERT INTO injections(ts, harness, hook, query, project, chars, uids)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts if ts is not None else time.time(),
                 harness[:64], hook[:64], query[:2000], project, max(0, int(chars)),
                 json.dumps(uids[:50])),
            )
            return self.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    def list_injections(
        self,
        harness: str | None = None,
        project: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        sql = "SELECT * FROM injections"
        cond, args = [], []
        if harness:
            cond.append("harness = ?")
            args.append(harness)
        if project:
            cond.append("project = ?")
            args.append(project)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [dict(r) for r in self.conn.execute(sql, args)]

    def injection_counts(self, uids: list[str]) -> dict[str, int]:
        """给定 uid 集合的历史注入次数（遥测反哺排序用）。"""
        if not uids:
            return {}
        marks = ",".join("?" * len(uids))
        return {
            r["value"]: r["c"]
            for r in self.conn.execute(
                f"SELECT j.value, COUNT(*) c FROM injections,"
                f" json_each(injections.uids) j WHERE j.value IN ({marks})"
                f" GROUP BY j.value",
                uids,
            )
        }

    def purge_injections_before(self, cutoff_ts: float) -> int:
        cur = self.conn.execute("DELETE FROM injections WHERE ts < ?", (cutoff_ts,))
        self.conn.commit()
        return cur.rowcount

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
