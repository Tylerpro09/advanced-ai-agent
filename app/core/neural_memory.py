from __future__ import annotations

import asyncio
import importlib.util
import math
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import httpx

from app.config import settings
from app.core.sqlite_utils import connect as sqlite_connect
from app.core.memory import MemoryRecord, MemoryStore


@dataclass
class NeuralMemoryHit:
    id: int
    user_id: str
    text: str
    kind: str
    importance: float
    created_at: float
    semantic_score: float
    score: float


class NeuralEmbedder:
    """Creates dense neural embeddings using a local Transformer or an OpenAI-compatible /embeddings endpoint."""

    def __init__(self) -> None:
        self.provider = settings.embedding_provider.lower().strip()
        self.model = settings.embedding_model.strip()
        self.base_url = (settings.embedding_base_url or settings.ai_base_url).rstrip("/")
        self.api_key = settings.embedding_api_key or settings.ai_api_key
        self.timeout = settings.ai_timeout_seconds
        self._local_model = None
        self._local_lock = threading.Lock()
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return settings.neural_memory_enabled and self.provider not in {"off", "disabled", "none"}

    def status(self) -> dict:
        dependency_error = ""
        if self.enabled and self.provider == "local" and importlib.util.find_spec("sentence_transformers") is None:
            dependency_error = "sentence-transformers is not installed; run: pip install -r requirements-neural.txt"
        error = self.last_error or dependency_error
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "ready": self.enabled and not bool(error),
            "last_error": error,
        }

    def _load_local_model(self):
        if self._local_model is not None:
            return self._local_model
        with self._local_lock:
            if self._local_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "Local neural memory requires sentence-transformers. Install: pip install -r requirements-neural.txt"
                    ) from exc
                self._local_model = SentenceTransformer(self.model)
        return self._local_model

    @staticmethod
    def _normalize(vector: Iterable[float]) -> list[float]:
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def _embed_local_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_local_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row.tolist()] for row in vectors]

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        ordered = sorted(data.get("data", []), key=lambda x: int(x.get("index", 0)))
        if len(ordered) != len(texts):
            raise RuntimeError("Embedding endpoint returned an unexpected number of vectors")
        return [self._normalize(item["embedding"]) for item in ordered]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            return []
        if not texts:
            return []
        try:
            if self.provider == "local":
                vectors = await asyncio.to_thread(self._embed_local_sync, texts)
            elif self.provider in {"openai", "remote", "openai-compatible"}:
                vectors = await self._embed_remote(texts)
            else:
                raise ValueError(f"Unknown EMBEDDING_PROVIDER: {self.provider}")
            self.last_error = ""
            return vectors
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise


class NeuralMemoryStore:
    """Dense-vector memory layer backed by the same SQLite database as MemoryStore."""

    def __init__(self, memory: MemoryStore):
        self.memory = memory
        self.db_path = memory.path
        self.embedder = NeuralEmbedder()
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
                CREATE TABLE IF NOT EXISTS neural_memory_vectors (
                    memory_id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_neural_memory_user
                    ON neural_memory_vectors(user_id, memory_id);
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

    async def index_memory(self, memory_id: int, user_id: str, text: str) -> bool:
        if not self.embedder.enabled:
            return False
        vectors = await self.embedder.embed([text])
        if not vectors:
            return False
        vector = vectors[0]
        self._conn().execute(
            """
            INSERT INTO neural_memory_vectors(memory_id,user_id,model,dimension,vector,updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id=excluded.user_id,
                model=excluded.model,
                dimension=excluded.dimension,
                vector=excluded.vector,
                updated_at=excluded.updated_at
            """,
            (memory_id, user_id, self.embedder.model, len(vector), self._pack(vector), time.time()),
        )
        self._conn().commit()
        return True

    async def add_memory(self, user_id: str, text: str, kind: str = "fact", importance: float = 0.5) -> int:
        memory_id = self.memory.add_memory(user_id, text, kind, importance)
        try:
            await self.index_memory(memory_id, user_id, text)
        except Exception:
            # Text memory must remain usable even if the neural encoder is unavailable.
            pass
        return memory_id

    async def sync_user(self, user_id: str, limit: int = 5000) -> dict:
        if not self.embedder.enabled:
            return {"indexed": 0, "skipped": 0, "enabled": False}
        rows = self._conn().execute(
            """
            SELECT m.* FROM memories m
            LEFT JOIN neural_memory_vectors v ON v.memory_id=m.id AND v.model=?
            WHERE m.user_id=? AND v.memory_id IS NULL
            ORDER BY m.id ASC LIMIT ?
            """,
            (self.embedder.model, user_id, limit),
        ).fetchall()
        indexed = 0
        # Small batches avoid large VRAM/RAM spikes while still using the neural model efficiently.
        batch_size = max(1, min(64, settings.embedding_batch_size))
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [r["text"] for r in batch]
            vectors = await self.embedder.embed(texts)
            conn = self._conn()
            for row, vector in zip(batch, vectors):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO neural_memory_vectors(memory_id,user_id,model,dimension,vector,updated_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (int(row["id"]), user_id, self.embedder.model, len(vector), self._pack(vector), time.time()),
                )
                indexed += 1
            conn.commit()
        return {"indexed": indexed, "skipped": 0, "enabled": True, "model": self.embedder.model}

    async def search(self, user_id: str, query: str, limit: int = 8) -> list[NeuralMemoryHit]:
        if not self.embedder.enabled or not query.strip():
            return []
        # New or manually-created memories are indexed lazily.
        await self.sync_user(user_id, limit=settings.neural_sync_limit)
        q_vectors = await self.embedder.embed([query])
        if not q_vectors:
            return []
        q = q_vectors[0]
        rows = self._conn().execute(
            """
            SELECT m.*, v.dimension, v.vector
            FROM neural_memory_vectors v
            JOIN memories m ON m.id=v.memory_id
            WHERE v.user_id=? AND v.model=?
            """,
            (user_id, self.embedder.model),
        ).fetchall()
        hits: list[NeuralMemoryHit] = []
        for row in rows:
            dim = int(row["dimension"])
            if dim != len(q):
                continue
            vec = self._unpack(row["vector"], dim)
            semantic = sum(a * b for a, b in zip(q, vec))
            # Blend semantic similarity with user-defined importance and a small recency bonus.
            importance = float(row["importance"])
            age_days = max(0.0, (time.time() - float(row["created_at"])) / 86400.0)
            recency = 1.0 / (1.0 + age_days / max(1.0, settings.neural_recency_half_life_days))
            score = (
                semantic * settings.neural_semantic_weight
                + importance * settings.neural_importance_weight
                + recency * settings.neural_recency_weight
            )
            if semantic >= settings.neural_min_similarity:
                hits.append(
                    NeuralMemoryHit(
                        id=int(row["id"]),
                        user_id=row["user_id"],
                        text=row["text"],
                        kind=row["kind"],
                        importance=importance,
                        created_at=float(row["created_at"]),
                        semantic_score=float(semantic),
                        score=float(score),
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    async def hybrid_search(self, user_id: str, query: str, limit: int = 8) -> list[NeuralMemoryHit]:
        lexical = self.memory.search_memories(user_id, query, limit * 2)
        try:
            neural = await self.search(user_id, query, limit * 2)
        except Exception:
            neural = []

        merged: dict[int, NeuralMemoryHit] = {}
        for rank, item in enumerate(lexical):
            lexical_bonus = max(0.0, 1.0 - rank / max(1, len(lexical)))
            merged[item.id] = NeuralMemoryHit(
                id=item.id,
                user_id=item.user_id,
                text=item.text,
                kind=item.kind,
                importance=item.importance,
                created_at=item.created_at,
                semantic_score=0.0,
                score=0.55 * lexical_bonus + 0.30 * item.importance,
            )
        for hit in neural:
            current = merged.get(hit.id)
            if current:
                current.semantic_score = max(current.semantic_score, hit.semantic_score)
                current.score = max(current.score, hit.score) + 0.15
            else:
                merged[hit.id] = hit
        out = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return out[:limit]

    def stats(self, user_id: str | None = None) -> dict:
        if user_id:
            total_memories = self._conn().execute("SELECT COUNT(*) c FROM memories WHERE user_id=?", (user_id,)).fetchone()["c"]
            indexed = self._conn().execute(
                "SELECT COUNT(*) c FROM neural_memory_vectors WHERE user_id=? AND model=?",
                (user_id, self.embedder.model),
            ).fetchone()["c"]
        else:
            total_memories = self._conn().execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
            indexed = self._conn().execute(
                "SELECT COUNT(*) c FROM neural_memory_vectors WHERE model=?", (self.embedder.model,)
            ).fetchone()["c"]
        return {
            **self.embedder.status(),
            "memories": int(total_memories),
            "indexed": int(indexed),
            "coverage": (float(indexed) / float(total_memories)) if total_memories else 1.0,
        }
