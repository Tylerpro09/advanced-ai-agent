from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import settings


def connect(path: str, *, foreign_keys: bool = True) -> sqlite3.Connection:
    """Open a resilient SQLite connection with consistent pragmas across all stores."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    timeout_s = max(1.0, float(settings.sqlite_busy_timeout_ms) / 1000.0)
    conn = sqlite3.connect(path, timeout=timeout_s, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={max(1000, int(settings.sqlite_busy_timeout_ms))}")
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys=ON")
    # WAL improves coexistence between chat, feedback and indexing threads.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError:
        pass
    return conn
