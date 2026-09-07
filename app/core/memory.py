from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.core.sqlite_utils import connect as sqlite_connect


@dataclass
class MemoryRecord:
    id: int
    user_id: str
    text: str
    kind: str
    importance: float
    created_at: float


class MemoryStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite_connect(self.path)
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = sqlite_connect(self.path)
        try:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(user_id, conversation_id, id);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user
                    ON memories(user_id, id);
                """
            )
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories', content_rowid='id')"
                )
                conn.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                      INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                      INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                      INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
                      INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                    """
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    def add_message(self, user_id: str, conversation_id: str, role: str, content: str) -> None:
        self._conn().execute(
            "INSERT INTO messages(user_id, conversation_id, role, content, created_at) VALUES(?,?,?,?,?)",
            (user_id, conversation_id, role, content, time.time()),
        )
        self._conn().commit()

    def recent_messages(self, user_id: str, conversation_id: str, limit: int = 16) -> list[dict[str, str]]:
        rows = self._conn().execute(
            """
            SELECT role, content FROM messages
            WHERE user_id=? AND conversation_id=?
            ORDER BY id DESC LIMIT ?
            """,
            (user_id, conversation_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def add_memory(self, user_id: str, text: str, kind: str = "fact", importance: float = 0.5) -> int:
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty")
        importance = max(0.0, min(1.0, float(importance)))
        cur = self._conn().execute(
            "INSERT INTO memories(user_id, text, kind, importance, created_at) VALUES(?,?,?,?,?)",
            (user_id, text, kind, importance, time.time()),
        )
        self._conn().commit()
        return int(cur.lastrowid)

    def search_memories(self, user_id: str, query: str, limit: int = 6) -> list[MemoryRecord]:
        query = query.strip()
        if not query:
            rows = self._conn().execute(
                "SELECT * FROM memories WHERE user_id=? ORDER BY importance DESC, id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            return [self._to_record(r) for r in rows]

        tokens = re.findall(r"[\wáéíóúñüÁÉÍÓÚÑÜ-]+", query, flags=re.UNICODE)
        fts_q = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in tokens[:12])
        if fts_q:
            try:
                rows = self._conn().execute(
                    """
                    SELECT m.* FROM memories_fts f
                    JOIN memories m ON m.id=f.rowid
                    WHERE m.user_id=? AND memories_fts MATCH ?
                    ORDER BY bm25(memories_fts), m.importance DESC
                    LIMIT ?
                    """,
                    (user_id, fts_q, limit),
                ).fetchall()
                if rows:
                    return [self._to_record(r) for r in rows]
            except sqlite3.OperationalError:
                pass

        like = "%" + "%".join(tokens[:4] or [query]) + "%"
        rows = self._conn().execute(
            """
            SELECT * FROM memories WHERE user_id=? AND text LIKE ?
            ORDER BY importance DESC, id DESC LIMIT ?
            """,
            (user_id, like, limit),
        ).fetchall()
        return [self._to_record(r) for r in rows]

    def list_memories(self, user_id: str, limit: int = 50) -> list[MemoryRecord]:
        rows = self._conn().execute(
            "SELECT * FROM memories WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [self._to_record(r) for r in rows]

    def delete_memory(self, user_id: str, memory_id: int) -> bool:
        cur = self._conn().execute(
            "DELETE FROM memories WHERE user_id=? AND id=?", (user_id, memory_id)
        )
        self._conn().commit()
        return cur.rowcount > 0

    @staticmethod
    def _to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            user_id=row["user_id"],
            text=row["text"],
            kind=row["kind"],
            importance=float(row["importance"]),
            created_at=float(row["created_at"]),
        )
