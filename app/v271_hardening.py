from __future__ import annotations

"""Compatibility hardening for the v2.7.1 developmental-learning release.

The module is intentionally small and idempotent.  It closes security/concurrency
edge cases and installs the developmental API on older v2.7 main.py layouts.
It can be removed once all deployments have moved to a main module that contains
these guards natively.
"""

import ast
import asyncio
import threading
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.core.developmental_learning import DevelopmentalLearningSystem, SafePracticeSandbox

_INSTALLED = False
_PATCH_LOCK = threading.RLock()
_PRACTICE_LOCKS: dict[tuple[int, str, int], asyncio.Lock] = {}
_PRACTICE_LOCKS_GUARD = threading.Lock()


class _DevelopmentGoalCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=2000)
    title: str = Field(default="", max_length=160)
    curiosity: float = Field(default=0.75, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _DevelopmentSandboxValidate(BaseModel):
    code: str = Field(min_length=1, max_length=20000)


def _practice_lock(system: DevelopmentalLearningSystem, user_id: str, goal_id: int) -> asyncio.Lock:
    key = (id(system), user_id, int(goal_id))
    with _PRACTICE_LOCKS_GUARD:
        lock = _PRACTICE_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _PRACTICE_LOCKS[key] = lock
        return lock


def _patch_sandbox() -> None:
    if getattr(SafePracticeSandbox, "_v271_hardened", False):
        return

    original_validate = SafePracticeSandbox.validate

    def validate(self: SafePracticeSandbox, code: str) -> tuple[bool, str]:
        code = str(code or "")
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            return original_validate(self, code)

        banned = set(self.BANNED_CALLS)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned:
                # Prevent alias escapes such as: f = open; f("x")
                return False, f"name_not_allowed:{node.id}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                aliases = node.names
                for alias in aliases:
                    if alias.name.startswith("__") or (alias.asname and alias.asname.startswith("__")):
                        return False, "dunder_import_alias_not_allowed"
        return original_validate(self, code)

    SafePracticeSandbox.validate = validate  # type: ignore[method-assign]
    SafePracticeSandbox._v271_hardened = True  # type: ignore[attr-defined]


def _patch_developmental_methods() -> None:
    cls = DevelopmentalLearningSystem
    if getattr(cls, "_v271_hardened", False):
        return

    original_ensure_profile = cls._ensure_profile
    original_add_xp = cls._add_xp
    original_create_goal = cls.create_goal
    original_record_analogy = cls.record_analogy
    original_update_feedback = cls.update_from_feedback
    original_practice_goal = cls.practice_goal

    def ensure_profile(self: DevelopmentalLearningSystem, user_id: str) -> None:
        with self._write_lock:
            original_ensure_profile(self, user_id)

    def add_xp(self: DevelopmentalLearningSystem, user_id: str, amount: float) -> dict[str, Any]:
        with self._write_lock:
            return original_add_xp(self, user_id, amount)

    def create_goal(
        self: DevelopmentalLearningSystem,
        user_id: str,
        question: str,
        curiosity: float,
        confidence: float,
        title: str = "",
    ) -> int | None:
        with self._write_lock:
            # Preserve duplicate merging even when the cap is already reached.
            cleaned = " ".join(str(question or "").split())[:2000]
            terms = self._terms(cleaned)
            rows = self._conn().execute(
                "SELECT * FROM learning_goals WHERE user_id=? AND status='open' ORDER BY updated_at DESC LIMIT 100",
                (user_id,),
            ).fetchall()
            duplicate = False
            for row in rows:
                existing_terms = self._terms(str(row["question"]))
                union = terms | existing_terms
                if union and len(terms & existing_terms) / len(union) >= 0.55:
                    duplicate = True
                    break
            if not duplicate and len(rows) >= max(1, int(settings.developmental_max_open_goals)):
                return None
            return original_create_goal(self, user_id, cleaned, curiosity, confidence, title)

    def record_analogy(
        self: DevelopmentalLearningSystem,
        user_id: str,
        source_title: str,
        target_query: str,
        mapping: str,
        limitation: str = "",
        usefulness: float = 0.5,
    ) -> int:
        with self._write_lock:
            self._ensure_profile(user_id)
            return original_record_analogy(
                self, user_id, source_title, target_query, mapping, limitation, usefulness
            )

    def update_from_feedback(
        self: DevelopmentalLearningSystem, user_id: str, query: str, reward: float
    ) -> dict[str, Any]:
        with self._write_lock:
            return original_update_feedback(self, user_id, query, reward)

    async def practice_goal(
        self: DevelopmentalLearningSystem, user_id: str, goal_id: int, provider: Any
    ) -> dict[str, Any]:
        # A per-goal async lock prevents two long model calls from later overwriting
        # each other's attempt/success counters with stale snapshots.
        async with _practice_lock(self, user_id, int(goal_id)):
            return await original_practice_goal(self, user_id, goal_id, provider)

    cls._ensure_profile = ensure_profile  # type: ignore[method-assign]
    cls._add_xp = add_xp  # type: ignore[method-assign]
    cls.create_goal = create_goal  # type: ignore[method-assign]
    cls.record_analogy = record_analogy  # type: ignore[method-assign]
    cls.update_from_feedback = update_from_feedback  # type: ignore[method-assign]
    cls.practice_goal = practice_goal  # type: ignore[method-assign]
    cls._v271_hardened = True  # type: ignore[attr-defined]


def _install_routes_on(app: FastAPI) -> None:
    existing = {getattr(route, "path", "") for route in app.routes}

    if "/v1/development/status" not in existing:
        @app.get("/v1/development/status")
        def development_status(user_id: str) -> dict:
            from app import main as main_module
            return {
                "profile": main_module.agent.development.profile(user_id),
                "goals": [asdict(x) for x in main_module.agent.development.goals(user_id, "open", 20)],
            }

    if "/v1/development/goals" not in existing:
        @app.get("/v1/development/goals")
        def development_goals(
            user_id: str,
            status: str = Query(default="open", pattern="^(open|mastered|all)$"),
            limit: int = Query(default=20, ge=1, le=100),
        ) -> dict:
            from app import main as main_module
            return {"items": [asdict(x) for x in main_module.agent.development.goals(user_id, status, limit)]}

        @app.post("/v1/development/goals")
        def development_goal_create(req: _DevelopmentGoalCreate) -> dict:
            from app import main as main_module
            goal_id = main_module.agent.development.create_goal(
                req.user_id, req.question, req.curiosity, req.confidence, req.title
            )
            if goal_id is None:
                raise HTTPException(400, "Learning goal rejected, duplicated at capacity, or contains sensitive data")
            return {"ok": True, "goal_id": goal_id, "id": goal_id}

    if "/v1/development/goals/{goal_id}/practice" not in existing:
        @app.post("/v1/development/goals/{goal_id}/practice")
        async def development_practice(goal_id: int, user_id: str) -> dict:
            from app import main as main_module
            try:
                return await main_module.agent.development.practice_goal(
                    user_id, goal_id, main_module.agent.provider
                )
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
            except Exception as exc:
                raise HTTPException(500, f"Development practice failed: {exc}") from exc

    if "/v1/development/practice" not in existing:
        @app.get("/v1/development/practice")
        def development_practice_history(
            user_id: str,
            goal_id: int | None = None,
            limit: int = Query(default=50, ge=1, le=200),
        ) -> dict:
            from app import main as main_module
            return {
                "items": [
                    asdict(x)
                    for x in main_module.agent.development.practice_history(user_id, goal_id, limit)
                ]
            }

    if "/v1/development/sandbox/validate" not in existing:
        @app.post("/v1/development/sandbox/validate")
        def development_sandbox_validate(req: _DevelopmentSandboxValidate) -> dict:
            from app import main as main_module
            ok, reason = main_module.agent.development.sandbox.validate(req.code)
            return {"ok": ok, "reason": reason, "execution_enabled": bool(settings.developmental_sandbox_execution_enabled)}


def _patch_fastapi_constructor() -> None:
    if getattr(FastAPI, "_aaa_v271_hardened", False):
        return
    original_init = FastAPI.__init__

    def init(self: FastAPI, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _install_routes_on(self)

    FastAPI.__init__ = init  # type: ignore[method-assign]
    FastAPI._aaa_v271_hardened = True  # type: ignore[attr-defined]


def install() -> None:
    global _INSTALLED
    with _PATCH_LOCK:
        if _INSTALLED:
            return
        _patch_sandbox()
        _patch_developmental_methods()
        _patch_fastapi_constructor()
        _INSTALLED = True
