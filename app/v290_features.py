from __future__ import annotations

"""Advanced AI Agent v2.9 feature pack.

Adds user-visible, inspectable learning controls without changing the existing chat API:
- explicit answer correction (strong teaching signal);
- brain dashboard;
- manual policy consolidation + dataset export;
- conversation reset.

Slow learning work is always queued after durable SQLite writes so the UI remains responsive.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("advanced_ai_agent.v290")

_INSTALLED = False
_ORIGINAL_FASTAPI_INIT = None
_TASKS: set[asyncio.Task[Any]] = set()

# Intentionally targets actual-looking secrets rather than ordinary words like "password" in documentation/code.
_SENSITIVE = re.compile(
    r"(?i)(bearer\s+[A-Za-z0-9._-]{16,}|BEGIN [A-Z ]*PRIVATE KEY|"
    r"(?:password|passwd|api[_ -]?key|access[_ -]?token|refresh[_ -]?token)\s*[:=]\s*[^\s]{8,})"
)


class _CorrectionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    corrected_answer: str = Field(min_length=1, max_length=12000)
    reason: str = Field(default="", max_length=2000)


class _UserAction(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)


def _retain(task: asyncio.Task[Any]) -> None:
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:100] or "user"


async def _learn_correction(
    user_id: str,
    original_id: int,
    correction_id: int,
    situation: str,
    original_lesson: str,
    corrected_lesson: str,
) -> None:
    try:
        from app import main as main_module

        store = main_module.experiences
        # Index both the warning and the preferred answer.
        try:
            await store._index(  # type: ignore[attr-defined]
                original_id,
                user_id,
                store._vector_text(situation, original_lesson, "corrected_by_user"),  # type: ignore[attr-defined]
            )
            await store._index(  # type: ignore[attr-defined]
                correction_id,
                user_id,
                store._vector_text(situation, corrected_lesson, "explicit_user_correction"),  # type: ignore[attr-defined]
            )
        except Exception:
            logger.exception("Could not index explicit correction %s", correction_id)

        # Learn the old response as a caution and the correction as a positive procedure/example.
        for experience_id in (original_id, correction_id):
            try:
                await main_module.agent.learn_from_feedback(user_id, experience_id)
            except Exception:
                logger.exception("Cognitive correction learning failed for %s", experience_id)

        if settings.experience_policy_enabled:
            try:
                await asyncio.to_thread(main_module.experience_policy.train_user, user_id)
            except Exception:
                logger.exception("Policy retraining after correction failed for %s", user_id)
    except Exception:
        logger.exception("Unexpected correction-learning failure")


def _install_routes(app: FastAPI) -> None:
    @app.post("/v1/experiences/{experience_id}/correction")
    async def correct_experience(experience_id: int, req: _CorrectionRequest) -> dict[str, Any]:
        from app import main as main_module

        corrected = " ".join(req.corrected_answer.split()) if "\n" not in req.corrected_answer else req.corrected_answer.strip()
        reason = req.reason.strip()
        if _SENSITIVE.search(corrected) or (reason and _SENSITIVE.search(reason)):
            raise HTTPException(400, "Correction looks like it contains a credential or private key and was not stored.")

        store = main_module.experiences
        conn = store._conn()  # type: ignore[attr-defined]
        row = conn.execute(
            "SELECT * FROM experiences WHERE user_id=? AND id=?",
            (req.user_id, int(experience_id)),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Experience not found")

        now = time.time()
        original_lesson = (
            "The user explicitly corrected this response. Avoid repeating the rejected answer pattern."
            + (f" Reason: {reason}" if reason else "")
        )[:4000]
        corrected_lesson = (
            "Explicit user correction: prefer this answer/content pattern when the same intent applies."
            + (f" User explanation: {reason}" if reason else "")
        )[:4000]

        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE experiences
                SET reward=-1.0, outcome='corrected_by_user', lesson=?, updated_at=?
                WHERE user_id=? AND id=?
                """,
                (original_lesson, now, req.user_id, int(experience_id)),
            )
            cur = conn.execute(
                """
                INSERT INTO experiences(
                    user_id,conversation_id,situation,action,tool_log,outcome,reward,lesson,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    req.user_id,
                    str(row["conversation_id"]),
                    str(row["situation"]),
                    corrected,
                    json.dumps(
                        [{"source": "explicit_user_correction", "original_experience_id": int(experience_id)}],
                        ensure_ascii=False,
                    ),
                    "explicit_user_correction",
                    1.0,
                    corrected_lesson,
                    now,
                    now,
                ),
            )
            correction_id = int(cur.lastrowid)
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            raise HTTPException(500, f"Could not save correction: {type(exc).__name__}: {exc}") from exc

        task = asyncio.create_task(
            _learn_correction(
                req.user_id,
                int(experience_id),
                correction_id,
                str(row["situation"]),
                original_lesson,
                corrected_lesson,
            )
        )
        _retain(task)
        return {
            "ok": True,
            "original_experience_id": int(experience_id),
            "correction_experience_id": correction_id,
            "saved": True,
            "reward": 1.0,
            "learning_queued": True,
        }

    @app.get("/v1/brain/dashboard")
    def brain_dashboard(user_id: str) -> dict[str, Any]:
        from app import main as main_module
        from app.version import APP_VERSION

        provider = main_module.agent.provider
        if hasattr(provider, "status"):
            try:
                model_status: dict[str, Any] = provider.status()
            except Exception:
                model_status = {}
        else:
            model_status = {
                "backend": settings.model_backend,
                "configured_model": getattr(provider, "model", settings.ai_model),
                "resolved_model": getattr(provider, "_resolved_model", None),
                "base_url": getattr(provider, "base_url", settings.ai_base_url),
            }

        memories = main_module.memory.list_memories(user_id, 10)
        recent = main_module.experiences.list(user_id, 10)
        goals = main_module.agent.development.goals(user_id, "open", 10)
        return {
            "version": APP_VERSION,
            "model": model_status,
            "memory": {
                "neural": main_module.neural_memory.stats(user_id),
                "recent": [
                    {"id": x.id, "kind": x.kind, "importance": x.importance, "text": x.text}
                    for x in memories
                ],
            },
            "experiences": {
                **main_module.experiences.stats(user_id),
                "recent": [
                    {
                        "id": x.id,
                        "situation": x.situation[:240],
                        "reward": x.reward,
                        "outcome": x.outcome,
                        "lesson": x.lesson[:400],
                    }
                    for x in recent
                ],
            },
            "policy": main_module.experience_policy.status(user_id),
            "human": main_module.agent.human_learning.stats(user_id),
            "developmental": {
                "profile": main_module.agent.development.profile(user_id),
                "goals": [
                    {
                        "id": g.id,
                        "title": g.title,
                        "priority": g.priority,
                        "confidence": g.confidence,
                        "attempts": g.attempts,
                        "successes": g.successes,
                    }
                    for g in goals
                ],
            },
            "training": main_module.lora_trainer.status(user_id),
        }

    @app.post("/v1/brain/consolidate")
    async def consolidate_brain(req: _UserAction) -> dict[str, Any]:
        from app import main as main_module

        policy = await asyncio.to_thread(main_module.experience_policy.train_user, req.user_id)
        export_dir = Path(settings.project_root) / "data" / "training"
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / f"{_safe_name(req.user_id)}_latest.jsonl"
        exported = await asyncio.to_thread(
            main_module.experiences.export_training_jsonl,
            req.user_id,
            str(path),
            0.5,
        )
        return {
            "ok": True,
            "policy": policy,
            "exported_examples": int(exported),
            "dataset": str(path),
            "experience_stats": main_module.experiences.stats(req.user_id),
            "human": main_module.agent.human_learning.stats(req.user_id),
            "developmental": main_module.agent.development.profile(req.user_id),
        }

    @app.post("/v1/conversations/{conversation_id}/clear")
    def clear_conversation(conversation_id: str, req: _UserAction) -> dict[str, Any]:
        from app import main as main_module

        conversation_id = conversation_id.strip()[:120]
        if not conversation_id:
            raise HTTPException(400, "conversation_id is empty")
        conn = main_module.memory._conn()  # type: ignore[attr-defined]
        cur = conn.execute(
            "DELETE FROM messages WHERE user_id=? AND conversation_id=?",
            (req.user_id, conversation_id),
        )
        conn.commit()
        return {"ok": True, "deleted_messages": int(cur.rowcount)}


def install() -> None:
    global _INSTALLED, _ORIGINAL_FASTAPI_INIT
    if _INSTALLED:
        return

    _ORIGINAL_FASTAPI_INIT = FastAPI.__init__

    def init(self: FastAPI, *args: Any, **kwargs: Any) -> None:
        assert _ORIGINAL_FASTAPI_INIT is not None
        _ORIGINAL_FASTAPI_INIT(self, *args, **kwargs)
        _install_routes(self)

    FastAPI.__init__ = init  # type: ignore[method-assign]
    FastAPI._aaa_v290_features = True  # type: ignore[attr-defined]
    _INSTALLED = True
