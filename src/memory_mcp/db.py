"""存储层：SQLite + FTS5(外部内容表) + 向量 BLOB。单写者，线程内单连接。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
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
"""

VALID_KINDS = {"fact", "decision", "preference", "artifact"}


class MemoryStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
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

    def list_recent(self, project: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
        sql = (
            "SELECT uid, content, kind, source, project, tags, created_at FROM memories"
            " WHERE deleted = 0"
        )
        params: list = []
        if project:
            sql += " AND project = ?"
            params.append(project)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
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

    def close(self) -> None:
        self.conn.close()
