from __future__ import annotations

"""Fast, reliable feedback path for v2.8.1.

The original feedback endpoint waited for reflection/model calls and policy training
before returning to the browser. On local models this could make a click look dead.
This module installs the same route *before* app.main registers its compatibility
route, commits the rating immediately, then queues expensive learning work.
"""

import asyncio
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("advanced_ai_agent.feedback")

_INSTALLED = False
_ORIGINAL_FASTAPI_INIT = None
_TASKS: set[asyncio.Task[Any]] = set()


class _FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    reward: float = Field(ge=-1.0, le=1.0)
    outcome: str = Field(default="", max_length=1000)
    lesson: str = Field(default="", max_length=2000)


def _retain(task: asyncio.Task[Any]) -> None:
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def _learn_after_feedback(
    user_id: str,
    experience_id: int,
    reward: float,
    situation: str,
    lesson: str,
    outcome: str,
) -> None:
    """Run all slow post-feedback work without delaying the UI response."""
    try:
        from app import main as main_module

        # Reindex the updated episode. A failure here must never undo the rating.
        try:
            await main_module.experiences._index(  # type: ignore[attr-defined]
                experience_id,
                user_id,
                main_module.experiences._vector_text(situation, lesson, outcome),  # type: ignore[attr-defined]
            )
        except Exception:
            logger.exception("Experience reindex failed for feedback %s", experience_id)

        # Human/developmental reflection may call the local LLM, so it belongs here.
        if settings.human_like_learning_enabled or settings.developmental_learning_enabled:
            try:
                await main_module.agent.learn_from_feedback(user_id, experience_id)
            except Exception:
                logger.exception("Background cognitive learning failed for feedback %s", experience_id)

        # Tiny policy network training can still take noticeable time once the history grows.
        if settings.experience_policy_enabled:
            try:
                await asyncio.to_thread(main_module.experience_policy.train_user, user_id)
            except Exception:
                logger.exception("Background experience-policy training failed for %s", user_id)

        # Preserve the existing optional automatic LoRA behavior, but never block feedback.
        every = max(0, int(settings.lora_auto_train_every))
        if settings.continual_learning_enabled and every > 0 and reward >= float(settings.lora_min_reward):
            try:
                positive_count = len(
                    main_module.experiences.training_examples(
                        user_id, settings.lora_min_reward, 1_000_000
                    )
                )
                if (
                    positive_count >= int(settings.lora_min_examples)
                    and positive_count % every == 0
                    and user_id not in main_module.auto_lora_users
                ):
                    main_module.auto_lora_users.add(user_id)
                    await main_module._auto_train_lora(user_id)
            except Exception:
                logger.exception("Background automatic LoRA check failed for %s", user_id)
    except Exception:
        logger.exception("Unexpected background feedback failure for %s", experience_id)


def _install_route(app: FastAPI) -> None:
    @app.post("/v1/experiences/{experience_id}/feedback")
    async def fast_experience_feedback(experience_id: int, req: _FeedbackRequest) -> dict[str, Any]:
        from app import main as main_module

        store = main_module.experiences
        conn = store._conn()  # type: ignore[attr-defined]
        row = conn.execute(
            "SELECT * FROM experiences WHERE user_id=? AND id=?",
            (req.user_id, int(experience_id)),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Experience not found")

        reward = max(-1.0, min(1.0, float(req.reward)))
        outcome = req.outcome.strip() or str(row["outcome"])
        lesson = req.lesson.strip() or str(row["lesson"])
        now = time.time()

        try:
            conn.execute(
                """
                UPDATE experiences
                SET reward=?, outcome=?, lesson=?, updated_at=?
                WHERE user_id=? AND id=?
                """,
                (reward, outcome, lesson, now, req.user_id, int(experience_id)),
            )
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            raise HTTPException(500, f"Could not save feedback: {type(exc).__name__}: {exc}") from exc

        task = asyncio.create_task(
            _learn_after_feedback(
                req.user_id,
                int(experience_id),
                reward,
                str(row["situation"]),
                lesson,
                outcome,
            )
        )
        _retain(task)

        return {
            "ok": True,
            "experience_id": int(experience_id),
            "reward": reward,
            "saved": True,
            "learning_queued": True,
        }


def install() -> None:
    global _INSTALLED, _ORIGINAL_FASTAPI_INIT
    if _INSTALLED:
        return

    _ORIGINAL_FASTAPI_INIT = FastAPI.__init__

    def init(self: FastAPI, *args: Any, **kwargs: Any) -> None:
        assert _ORIGINAL_FASTAPI_INIT is not None
        _ORIGINAL_FASTAPI_INIT(self, *args, **kwargs)
        _install_route(self)

    FastAPI.__init__ = init  # type: ignore[method-assign]
    FastAPI._aaa_v281_feedback = True  # type: ignore[attr-defined]
    _INSTALLED = True
