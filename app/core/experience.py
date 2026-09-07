from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.sqlite_utils import connect as sqlite_connect
from app.core.memory import MemoryStore
from app.core.neural_memory import NeuralEmbedder


@dataclass
class ExperienceHit:
    id: int
    user_id: str
    conversation_id: str
    situation: str
    action: str
    tool_log: list[dict[str, Any]]
    outcome: str
    reward: float
    lesson: str
    created_at: float
    updated_at: float
    semantic_score: float = 0.0
    score: float = 0.0


class ExperienceStore:
    """Episodic neural memory: situation -> action -> outcome -> reward -> lesson."""

    def __init__(self, memory: MemoryStore, embedder: NeuralEmbedder | None = None):
        self.memory = memory
        self.db_path = memory.path
        self.embedder = embedder or NeuralEmbedder()
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
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    situation TEXT NOT NULL,
                    action TEXT NOT NULL,
                    tool_log TEXT NOT NULL DEFAULT '[]',
                    outcome TEXT NOT NULL DEFAULT 'unknown',
                    reward REAL NOT NULL DEFAULT 0.0,
                    lesson TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experiences_user
                    ON experiences(user_id, id);

                CREATE TABLE IF NOT EXISTS experience_vectors (
                    experience_id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(experience_id) REFERENCES experiences(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_experience_vectors_user
                    ON experience_vectors(user_id, experience_id);
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _pack(vector: list[float]) -> bytes:
        return struct.pack(f"<{len(vector)}f", *vector)

    @staticmethod
    def _unpack(blob: bytes, dimension: int) -> tuple[float, ...]:
        return struct.unpack(f"<{dimension}f", blob)

    @staticmethod
    def _vector_text(situation: str, lesson: str = "", outcome: str = "unknown") -> str:
        extra = f"\nLesson: {lesson}" if lesson.strip() else ""
        return f"Situation: {situation}\nOutcome: {outcome}{extra}"

    async def add(
        self,
        user_id: str,
        conversation_id: str,
        situation: str,
        action: str,
        tool_log: list[dict[str, Any]] | None = None,
        outcome: str = "unknown",
        reward: float = 0.0,
        lesson: str = "",
    ) -> int:
        now = time.time()
        reward = max(-1.0, min(1.0, float(reward)))
        cur = self._conn().execute(
            """
            INSERT INTO experiences(user_id,conversation_id,situation,action,tool_log,outcome,reward,lesson,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                user_id,
                conversation_id,
                situation.strip(),
                action.strip(),
                json.dumps(tool_log or [], ensure_ascii=False),
                outcome.strip() or "unknown",
                reward,
                lesson.strip(),
                now,
                now,
            ),
        )
        self._conn().commit()
        experience_id = int(cur.lastrowid)
        await self._index(experience_id, user_id, self._vector_text(situation, lesson, outcome))
        return experience_id

    async def _index(self, experience_id: int, user_id: str, text: str) -> bool:
        if not self.embedder.enabled:
            return False
        try:
            vectors = await self.embedder.embed([text])
        except Exception:
            return False
        if not vectors:
            return False
        vector = vectors[0]
        self._conn().execute(
            """
            INSERT INTO experience_vectors(experience_id,user_id,model,dimension,vector,updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(experience_id) DO UPDATE SET
                user_id=excluded.user_id,
                model=excluded.model,
                dimension=excluded.dimension,
                vector=excluded.vector,
                updated_at=excluded.updated_at
            """,
            (experience_id, user_id, self.embedder.model, len(vector), self._pack(vector), time.time()),
        )
        self._conn().commit()
        return True

    async def feedback(
        self,
        user_id: str,
        experience_id: int,
        reward: float,
        outcome: str = "",
        lesson: str = "",
    ) -> bool:
        row = self._conn().execute(
            "SELECT * FROM experiences WHERE user_id=? AND id=?", (user_id, experience_id)
        ).fetchone()
        if row is None:
            return False
        reward = max(-1.0, min(1.0, float(reward)))
        new_outcome = outcome.strip() or row["outcome"]
        new_lesson = lesson.strip() or row["lesson"]
        self._conn().execute(
            """
            UPDATE experiences SET reward=?, outcome=?, lesson=?, updated_at=?
            WHERE user_id=? AND id=?
            """,
            (reward, new_outcome, new_lesson, time.time(), user_id, experience_id),
        )
        self._conn().commit()
        await self._index(
            experience_id,
            user_id,
            self._vector_text(row["situation"], new_lesson, new_outcome),
        )
        return True

    async def search(self, user_id: str, query: str, limit: int = 6) -> list[ExperienceHit]:
        query = query.strip()
        if not query:
            return self.list(user_id, limit)
        if not self.embedder.enabled:
            return self._lexical_search(user_id, query, limit)
        try:
            q_vectors = await self.embedder.embed([query])
        except Exception:
            return self._lexical_search(user_id, query, limit)
        if not q_vectors:
            return self._lexical_search(user_id, query, limit)
        q = q_vectors[0]
        rows = self._conn().execute(
            """
            SELECT e.*, v.dimension, v.vector
            FROM experience_vectors v
            JOIN experiences e ON e.id=v.experience_id
            WHERE e.user_id=? AND v.model=?
            """,
            (user_id, self.embedder.model),
        ).fetchall()
        hits: list[ExperienceHit] = []
        for row in rows:
            dim = int(row["dimension"])
            if dim != len(q):
                continue
            vec = self._unpack(row["vector"], dim)
            semantic = sum(a * b for a, b in zip(q, vec))
            if semantic < settings.experience_min_similarity:
                continue
            age_days = max(0.0, (time.time() - float(row["updated_at"])) / 86400.0)
            recency = 1.0 / (1.0 + age_days / max(1.0, settings.experience_recency_half_life_days))
            # Reward does not dominate relevance. Negative experiences remain useful as warnings.
            confidence = min(1.0, abs(float(row["reward"])))
            score = semantic * 0.84 + recency * 0.08 + confidence * 0.08
            hit = self._to_hit(row)
            hit.semantic_score = float(semantic)
            hit.score = float(score)
            hits.append(hit)
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:limit]

    def _lexical_search(self, user_id: str, query: str, limit: int) -> list[ExperienceHit]:
        terms = [x for x in query.lower().split() if len(x) > 2][:6]
        rows = self._conn().execute(
            "SELECT * FROM experiences WHERE user_id=? ORDER BY updated_at DESC LIMIT 500",
            (user_id,),
        ).fetchall()
        ranked: list[tuple[int, ExperienceHit]] = []
        for row in rows:
            text = (row["situation"] + " " + row["lesson"] + " " + row["outcome"]).lower()
            score = sum(1 for term in terms if term in text)
            if score or not terms:
                hit = self._to_hit(row)
                hit.score = float(score)
                ranked.append((score, hit))
        ranked.sort(key=lambda x: (x[0], x[1].updated_at), reverse=True)
        return [item[1] for item in ranked[:limit]]

    def list(self, user_id: str, limit: int = 50) -> list[ExperienceHit]:
        rows = self._conn().execute(
            "SELECT * FROM experiences WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall()
        return [self._to_hit(r) for r in rows]

    def stats(self, user_id: str | None = None) -> dict[str, Any]:
        conn = self._conn()
        if user_id:
            total = conn.execute("SELECT COUNT(*) c FROM experiences WHERE user_id=?", (user_id,)).fetchone()["c"]
            rated = conn.execute("SELECT COUNT(*) c FROM experiences WHERE user_id=? AND reward != 0", (user_id,)).fetchone()["c"]
            indexed = conn.execute(
                "SELECT COUNT(*) c FROM experience_vectors WHERE user_id=? AND model=?",
                (user_id, self.embedder.model),
            ).fetchone()["c"]
        else:
            total = conn.execute("SELECT COUNT(*) c FROM experiences").fetchone()["c"]
            rated = conn.execute("SELECT COUNT(*) c FROM experiences WHERE reward != 0").fetchone()["c"]
            indexed = conn.execute(
                "SELECT COUNT(*) c FROM experience_vectors WHERE model=?", (self.embedder.model,)
            ).fetchone()["c"]
        return {
            "enabled": settings.experience_memory_enabled,
            "experiences": int(total),
            "rated": int(rated),
            "indexed": int(indexed),
            "embedding_model": self.embedder.model,
        }


    def rated_vector_samples(self, user_id: str, limit: int = 1000) -> list[tuple[list[float], float]]:
        """Return indexed rated experiences for the small neural policy network."""
        rows = self._conn().execute(
            """
            SELECT e.reward, v.dimension, v.vector
            FROM experience_vectors v
            JOIN experiences e ON e.id=v.experience_id
            WHERE e.user_id=? AND e.reward != 0 AND v.model=?
            ORDER BY e.updated_at DESC LIMIT ?
            """,
            (user_id, self.embedder.model, max(1, int(limit))),
        ).fetchall()
        out: list[tuple[list[float], float]] = []
        for row in rows:
            dim = int(row["dimension"])
            out.append((list(self._unpack(row["vector"], dim)), float(row["reward"])))
        return out

    def training_examples(self, user_id: str, min_reward: float = 0.5, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """
            SELECT id, situation, action, outcome, reward, lesson, created_at, updated_at
            FROM experiences
            WHERE user_id=? AND reward>=?
            ORDER BY updated_at DESC LIMIT ?
            """,
            (user_id, max(-1.0, min(1.0, float(min_reward))), max(1, int(limit))),
        ).fetchall()
        return [dict(r) for r in rows]

    def export_training_jsonl(self, user_id: str, output_path: str, min_reward: float = 0.5) -> int:
        """Export positively-rated experiences as chat examples for optional future LoRA/SFT training."""
        rows = self._conn().execute(
            """
            SELECT * FROM experiences
            WHERE user_id=? AND reward>=?
            ORDER BY id ASC
            """,
            (user_id, max(-1.0, min(1.0, float(min_reward)))),
        ).fetchall()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                item = {
                    "messages": [
                        {"role": "user", "content": row["situation"]},
                        {"role": "assistant", "content": row["action"]},
                    ],
                    "metadata": {
                        "outcome": row["outcome"],
                        "reward": float(row["reward"]),
                        "lesson": row["lesson"],
                    },
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def _to_hit(row: sqlite3.Row) -> ExperienceHit:
        try:
            tools = json.loads(row["tool_log"] or "[]")
            if not isinstance(tools, list):
                tools = []
        except Exception:
            tools = []
        return ExperienceHit(
            id=int(row["id"]),
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            situation=row["situation"],
            action=row["action"],
            tool_log=tools,
            outcome=row["outcome"],
            reward=float(row["reward"]),
            lesson=row["lesson"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
