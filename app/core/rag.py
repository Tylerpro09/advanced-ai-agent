from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.core.sqlite_utils import connect as sqlite_connect


@dataclass
class RAGHit:
    id: int
    user_id: str
    source: str
    chunk_index: int
    text: str
    score: float


_WORD_RE = re.compile(r"[\wáéíóúñüÁÉÍÓÚÑÜ-]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [x.lower() for x in _WORD_RE.findall(text)]


def _chunk_text(text: str, chunk_chars: int = 1400, overlap: int = 180) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + chunk_chars // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [c for c in chunks if c]


def extract_text(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".toml", ".ini", ".log"}:
        return p.read_text(encoding="utf-8", errors="replace")
    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    if ext == ".csv":
        rows: list[str] = []
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(f):
                rows.append(" | ".join(row))
        return "\n".join(rows)
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires: pip install pypdf") from exc
        reader = PdfReader(str(p))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported document type: {ext or 'no extension'}")


class RAGStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite_connect(self.db_path)
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = sqlite_connect(self.db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rag_user_source ON rag_chunks(user_id, source, id);
                """
            )
            try:
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(text, content='rag_chunks', content_rowid='id')")
                conn.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS rag_ai AFTER INSERT ON rag_chunks BEGIN
                      INSERT INTO rag_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS rag_ad AFTER DELETE ON rag_chunks BEGIN
                      INSERT INTO rag_fts(rag_fts, rowid, text) VALUES('delete', old.id, old.text);
                    END;
                    """
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    def ingest_text(self, user_id: str, source: str, text: str, replace: bool = True) -> int:
        chunks = _chunk_text(text)
        conn = self._conn()
        if replace:
            conn.execute("DELETE FROM rag_chunks WHERE user_id=? AND source=?", (user_id, source))
        for i, chunk in enumerate(chunks):
            counts = Counter(_tokens(chunk))
            conn.execute(
                "INSERT INTO rag_chunks(user_id,source,chunk_index,text,token_json,created_at) VALUES(?,?,?,?,?,?)",
                (user_id, source, i, chunk, json.dumps(counts, ensure_ascii=False), time.time()),
            )
        conn.commit()
        return len(chunks)

    def ingest_file(self, user_id: str, path: str, source: str | None = None) -> int:
        return self.ingest_text(user_id, source or os.path.basename(path), extract_text(path), replace=True)

    def list_sources(self, user_id: str) -> list[dict]:
        rows = self._conn().execute(
            "SELECT source, COUNT(*) chunks, MIN(created_at) created_at FROM rag_chunks WHERE user_id=? GROUP BY source ORDER BY source",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_source(self, user_id: str, source: str) -> bool:
        cur = self._conn().execute("DELETE FROM rag_chunks WHERE user_id=? AND source=?", (user_id, source))
        self._conn().commit()
        return cur.rowcount > 0

    def search(self, user_id: str, query: str, limit: int = 6) -> list[RAGHit]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return []

        candidates: list[sqlite3.Row] = []
        fts_q = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in q_tokens[:16])
        if fts_q:
            try:
                candidates = self._conn().execute(
                    """
                    SELECT c.* FROM rag_fts f
                    JOIN rag_chunks c ON c.id=f.rowid
                    WHERE c.user_id=? AND rag_fts MATCH ?
                    ORDER BY bm25(rag_fts) LIMIT ?
                    """,
                    (user_id, fts_q, max(limit * 6, 24)),
                ).fetchall()
            except sqlite3.OperationalError:
                candidates = []
        if not candidates:
            like = "%" + "%".join(q_tokens[:3]) + "%"
            candidates = self._conn().execute(
                "SELECT * FROM rag_chunks WHERE user_id=? AND text LIKE ? LIMIT ?",
                (user_id, like, max(limit * 6, 24)),
            ).fetchall()

        q = Counter(q_tokens)
        q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored: list[RAGHit] = []
        for row in candidates:
            counts = Counter(json.loads(row["token_json"]))
            dot = sum(q[t] * counts.get(t, 0) for t in q)
            d_norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
            score = dot / (q_norm * d_norm)
            scored.append(RAGHit(int(row["id"]), row["user_id"], row["source"], int(row["chunk_index"]), row["text"], score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]
