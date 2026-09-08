from __future__ import annotations

import math
import re
import sqlite3
import struct
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
from app.core.experience import ExperienceHit, ExperienceStore
from app.core.neural_memory import NeuralEmbedder
from app.core.sqlite_utils import connect as sqlite_connect


_SENSITIVE = re.compile(
    r"(?i)(password|passwd|api[_ -]?key|secret[_ -]?key|bearer\s+[a-z0-9._-]{12,}|"
    r"BEGIN [A-Z ]*PRIVATE KEY|credit card|cvv|access[_ -]?token|refresh[_ -]?token)"
)


@dataclass
class CognitiveMemory:
    id: int
    user_id: str
    memory_type: str
    title: str
    trigger: str
    content: str
    confidence: float
    strength: float
    successes: int
    failures: int
    reward_mean: float
    salience: float
    source_experience_id: int | None
    created_at: float
    updated_at: float
    last_access: float
    semantic_score: float = 0.0
    activation_score: float = 0.0


class HumanLearningSystem:
    """Human-inspired learning layered on episodic memory.

    It models useful cognitive mechanisms—consolidation, procedures, avoidance,
    reinforcement, forgetting, retrieval practice, novelty and metacognition—
    without claiming consciousness or feelings.
    """

    def __init__(self, experiences: ExperienceStore, embedder: NeuralEmbedder | None = None):
        self.experiences = experiences
        self.db_path = experiences.db_path
        self.embedder = embedder or experiences.embedder
        self._local = threading.local()
        self._write_lock = threading.RLock()
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
                CREATE TABLE IF NOT EXISTS cognitive_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    trigger_text TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    strength REAL NOT NULL DEFAULT 1.0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    reward_mean REAL NOT NULL DEFAULT 0.0,
                    salience REAL NOT NULL DEFAULT 0.5,
                    source_experience_id INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_access REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cognitive_user_type
                    ON cognitive_memories(user_id, memory_type, updated_at);
                CREATE TABLE IF NOT EXISTS cognitive_vectors (
                    memory_id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES cognitive_memories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_cognitive_vectors_user
                    ON cognitive_vectors(user_id, memory_id);
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
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _safe_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        if _SENSITIVE.search(text):
            return ""
        return text[:limit]

    async def _embed_one(self, text: str) -> list[float] | None:
        if not self.embedder.enabled:
            return None
        try:
            vectors = await self.embedder.embed([text])
        except Exception:
            return None
        return vectors[0] if vectors else None

    def _row(self, row: sqlite3.Row) -> CognitiveMemory:
        return CognitiveMemory(
            id=int(row["id"]), user_id=str(row["user_id"]), memory_type=str(row["memory_type"]),
            title=str(row["title"]), trigger=str(row["trigger_text"]), content=str(row["content"]),
            confidence=float(row["confidence"]), strength=float(row["strength"]),
            successes=int(row["successes"]), failures=int(row["failures"]), reward_mean=float(row["reward_mean"]),
            salience=float(row["salience"]), source_experience_id=int(row["source_experience_id"]) if row["source_experience_id"] is not None else None,
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]), last_access=float(row["last_access"]),
        )

    async def reinforce(self, user_id: str, memory_type: str, title: str, trigger: str, content: str,
                        reward: float, source_experience_id: int | None, salience: float) -> int | None:
        title = self._safe_text(title, 160)
        trigger = self._safe_text(trigger, 2000)
        content = self._safe_text(content, 5000)
        if not trigger or not content:
            return None
        reward = max(-1.0, min(1.0, float(reward)))
        salience = self._clip(salience)
        text = f"Type: {memory_type}\nTrigger: {trigger}\nKnowledge: {content}"
        vector = await self._embed_one(text)
        existing = None
        if vector:
            rows = self._conn().execute(
                """SELECT c.*,v.dimension,v.vector FROM cognitive_vectors v
                   JOIN cognitive_memories c ON c.id=v.memory_id
                   WHERE c.user_id=? AND c.memory_type=? AND v.model=?""",
                (user_id, memory_type, self.embedder.model),
            ).fetchall()
            best_score = -1.0
            for row in rows:
                dim = int(row["dimension"])
                if dim != len(vector):
                    continue
                score = sum(a * b for a, b in zip(vector, self._unpack(row["vector"], dim)))
                if score > best_score:
                    best_score, existing = float(score), row
            if best_score < float(settings.human_consolidation_similarity):
                existing = None
        if existing is None:
            terms = {x for x in re.findall(r"[\wáéíóúñü]+", trigger.lower()) if len(x) > 3}
            for row in self._conn().execute(
                "SELECT * FROM cognitive_memories WHERE user_id=? AND memory_type=? ORDER BY updated_at DESC LIMIT 300",
                (user_id, memory_type),
            ).fetchall():
                candidate = set(re.findall(r"[\wáéíóúñü]+", (row["trigger_text"] + " " + row["content"]).lower()))
                if len(terms & candidate) >= 2:
                    existing = row
                    break

        now = time.time()
        with self._write_lock:
            if existing is None:
                successes, failures = int(reward > 0.15), int(reward < -0.15)
                cur = self._conn().execute(
                    """INSERT INTO cognitive_memories(user_id,memory_type,title,trigger_text,content,confidence,strength,
                       successes,failures,reward_mean,salience,source_experience_id,created_at,updated_at,last_access)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (user_id, memory_type, title, trigger, content, self._clip(0.45 + abs(reward) * 0.25),
                     1.0 + abs(reward) * salience, successes, failures, reward, salience,
                     source_experience_id, now, now, now),
                )
                memory_id = int(cur.lastrowid)
            else:
                memory_id = int(existing["id"])
                successes = int(existing["successes"]) + int(reward > 0.15)
                failures = int(existing["failures"]) + int(reward < -0.15)
                n = max(1, int(existing["successes"]) + int(existing["failures"]))
                mean = (float(existing["reward_mean"]) * n + reward) / (n + 1)
                confidence = ((failures if memory_type == "avoidance" else successes) + 1.0) / (successes + failures + 2.0)
                strength = min(float(settings.human_max_memory_strength), float(existing["strength"]) +
                               float(settings.human_reinforcement_rate) * (0.25 + abs(reward)) * (0.5 + salience))
                richer = content if len(content) >= len(str(existing["content"])) else str(existing["content"])
                self._conn().execute(
                    """UPDATE cognitive_memories SET title=?,trigger_text=?,content=?,confidence=?,strength=?,successes=?,
                       failures=?,reward_mean=?,salience=?,source_experience_id=?,updated_at=?,last_access=?
                       WHERE id=? AND user_id=?""",
                    (title or str(existing["title"]), trigger, richer, self._clip(confidence), strength, successes, failures,
                     mean, max(salience, float(existing["salience"])), source_experience_id or existing["source_experience_id"],
                     now, now, memory_id, user_id),
                )
            self._conn().commit()
            if vector:
                self._conn().execute(
                    """INSERT INTO cognitive_vectors(memory_id,user_id,model,dimension,vector,updated_at) VALUES(?,?,?,?,?,?)
                       ON CONFLICT(memory_id) DO UPDATE SET user_id=excluded.user_id,model=excluded.model,
                       dimension=excluded.dimension,vector=excluded.vector,updated_at=excluded.updated_at""",
                    (memory_id, user_id, self.embedder.model, len(vector), self._pack(vector), now),
                )
                self._conn().commit()
        return memory_id

    async def learn_from_experience(self, user_id: str, experience: ExperienceHit,
                                    reflection: dict[str, Any] | None = None) -> dict[str, Any]:
        reflection = reflection if isinstance(reflection, dict) else {}
        reward = float(experience.reward)
        novelty = self._clip(float(reflection.get("novelty", 0.5)))
        salience = self._clip(0.35 + abs(reward) * 0.45 + novelty * 0.20)
        principle = self._safe_text(reflection.get("principle") or experience.lesson, 2500)
        title = self._safe_text(reflection.get("title") or (principle[:120] if principle else "Learned pattern"), 160)
        trigger = self._safe_text(reflection.get("trigger") or experience.situation, 2000)
        procedure_value = reflection.get("procedure")
        if isinstance(procedure_value, list):
            procedure = "\n".join(f"{i+1}. {str(x).strip()}" for i, x in enumerate(procedure_value[:12]) if str(x).strip())
        else:
            procedure = self._safe_text(procedure_value, 4500)
        procedure = procedure or self._safe_text(experience.action, 4500)
        mistake = self._safe_text(reflection.get("mistake") or experience.lesson or experience.action, 4000)
        ids: list[int] = []
        if principle:
            mid = await self.reinforce(user_id, "concept", title, trigger, principle, reward, experience.id, salience)
            if mid:
                ids.append(mid)
        if reward >= float(settings.human_positive_reward_threshold):
            mid = await self.reinforce(user_id, "procedure", title, trigger, procedure, reward, experience.id, salience)
            if mid:
                ids.append(mid)
        elif reward <= float(settings.human_negative_reward_threshold):
            avoidance = f"Avoid repeating this pattern: {mistake}"
            if experience.lesson:
                avoidance += f"\nLesson: {experience.lesson}"
            mid = await self.reinforce(user_id, "avoidance", title, trigger, avoidance, reward, experience.id, salience)
            if mid:
                ids.append(mid)
        return {"learned": bool(ids), "memory_ids": ids, "salience": salience, "reflection": reflection}

    async def retrieve(self, user_id: str, query: str, limit: int | None = None) -> list[CognitiveMemory]:
        query = query.strip()
        if not query:
            return []
        limit = max(1, min(20, int(limit or settings.human_learning_results)))
        vector = await self._embed_one(query)
        now = time.time()
        hits: list[CognitiveMemory] = []
        if vector:
            rows = self._conn().execute(
                """SELECT c.*,v.dimension,v.vector FROM cognitive_vectors v
                   JOIN cognitive_memories c ON c.id=v.memory_id WHERE c.user_id=? AND v.model=?""",
                (user_id, self.embedder.model),
            ).fetchall()
            for row in rows:
                dim = int(row["dimension"])
                if dim != len(vector):
                    continue
                semantic = float(sum(a * b for a, b in zip(vector, self._unpack(row["vector"], dim))))
                if semantic < float(settings.human_retrieval_min_similarity):
                    continue
                memory = self._row(row)
                age_days = max(0.0, (now - memory.last_access) / 86400.0)
                half_life = max(1.0, float(settings.human_forgetting_half_life_days) * max(0.5, memory.strength))
                retention = math.pow(0.5, age_days / half_life)
                memory.semantic_score = semantic
                memory.activation_score = semantic * 0.68 + memory.confidence * 0.12 + min(1.0, memory.strength / max(1.0, float(settings.human_max_memory_strength))) * 0.08 + memory.salience * 0.07 + retention * 0.05
                hits.append(memory)
        else:
            terms = [x for x in query.lower().split() if len(x) > 3][:8]
            for row in self._conn().execute(
                "SELECT * FROM cognitive_memories WHERE user_id=? ORDER BY updated_at DESC LIMIT 500", (user_id,)
            ).fetchall():
                overlap = sum(1 for term in terms if term in (row["trigger_text"] + " " + row["content"]).lower())
                if overlap:
                    memory = self._row(row)
                    memory.semantic_score = min(1.0, overlap / max(1, len(terms)))
                    memory.activation_score = memory.semantic_score * 0.8 + memory.confidence * 0.2
                    hits.append(memory)
        hits.sort(key=lambda x: x.activation_score, reverse=True)
        selected = hits[:limit]
        if selected:
            with self._write_lock:
                for item in selected:
                    item.strength = min(float(settings.human_max_memory_strength), item.strength + float(settings.human_retrieval_reinforcement))
                    item.last_access = now
                    self._conn().execute("UPDATE cognitive_memories SET strength=?,last_access=? WHERE id=? AND user_id=?",
                                         (item.strength, now, item.id, user_id))
                self._conn().commit()
        return selected

    async def cognitive_context(self, user_id: str, query: str,
                                policy_signal: dict[str, Any] | None = None) -> dict[str, Any]:
        memories = await self.retrieve(user_id, query)
        top_similarity = max((m.semantic_score for m in memories), default=0.0)
        novelty = self._clip(1.0 - top_similarity)
        evidence = 0.0
        if memories:
            weights = [max(0.05, m.activation_score) for m in memories[:5]]
            evidence = sum(m.confidence * w for m, w in zip(memories[:5], weights)) / sum(weights)
        policy_conf = abs(float(policy_signal.get("predicted_reward", 0.0))) if isinstance(policy_signal, dict) else 0.0
        confidence = self._clip(evidence * 0.62 + policy_conf * 0.23 + (1.0 - novelty) * 0.15)
        has_avoidance = any(m.memory_type == "avoidance" and m.activation_score >= 0.45 for m in memories)
        has_procedure = any(m.memory_type == "procedure" and m.activation_score >= 0.45 for m in memories)
        if has_avoidance:
            mode = "caution"
        elif novelty >= float(settings.human_novelty_threshold) or confidence < float(settings.human_low_confidence_threshold):
            mode = "explore"
        elif has_procedure:
            mode = "apply_skill"
        else:
            mode = "reason"
        return {"memories": [asdict(m) for m in memories], "novelty": novelty, "confidence": confidence,
                "mode": mode, "known_patterns": len(memories)}

    async def consolidate_existing(self, user_id: str, limit: int = 200) -> dict[str, Any]:
        processed = learned = 0
        for exp in reversed(self.experiences.list(user_id, max(1, min(2000, int(limit))))):
            if abs(exp.reward) < 0.15:
                continue
            processed += 1
            learned += int(bool((await self.learn_from_experience(user_id, exp)).get("learned")))
        return {"processed": processed, "learned": learned, **self.stats(user_id)}

    def stats(self, user_id: str | None = None) -> dict[str, Any]:
        if user_id:
            rows = self._conn().execute("SELECT memory_type,COUNT(*) c FROM cognitive_memories WHERE user_id=? GROUP BY memory_type", (user_id,)).fetchall()
        else:
            rows = self._conn().execute("SELECT memory_type,COUNT(*) c FROM cognitive_memories GROUP BY memory_type").fetchall()
        counts = {str(r["memory_type"]): int(r["c"]) for r in rows}
        return {"enabled": bool(settings.human_like_learning_enabled), "concepts": counts.get("concept", 0),
                "procedures": counts.get("procedure", 0), "avoidances": counts.get("avoidance", 0),
                "total": sum(counts.values()), "embedding_model": self.embedder.model}
