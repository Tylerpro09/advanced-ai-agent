from __future__ import annotations

import ast
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
from app.core.human_learning import HumanLearningSystem
from app.core.sqlite_utils import connect as sqlite_connect


@dataclass
class LearningGoal:
    id: int
    user_id: str
    title: str
    question: str
    status: str
    priority: float
    curiosity: float
    confidence: float
    attempts: int
    successes: int
    created_at: float
    updated_at: float
    last_attempt: float | None


@dataclass
class PracticeAttempt:
    id: int
    goal_id: int
    user_id: str
    mode: str
    prompt: str
    artifact: str
    score: float
    success: bool
    feedback: str
    created_at: float


class SafePracticeSandbox:
    """Restricted micro-practice runner.

    This is intentionally a soft Python sandbox, not an OS security boundary.
    It rejects filesystem/process/network-capable modules and dangerous builtins,
    runs isolated Python (-I) in a temporary directory, strips the environment,
    and applies time/output plus optional POSIX resource limits.
    """

    ALLOWED_IMPORTS = {
        "math", "statistics", "fractions", "decimal", "collections",
        "itertools", "functools", "re", "json",
    }
    BANNED_CALLS = {
        "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
        "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
        "help", "memoryview",
    }

    def __init__(self) -> None:
        self.timeout = max(1.0, min(15.0, float(settings.developmental_sandbox_timeout_seconds)))
        self.max_output = max(1000, min(100_000, int(settings.developmental_sandbox_max_output_chars)))
        self.memory_mb = max(64, min(2048, int(settings.developmental_sandbox_memory_mb)))

    def validate(self, code: str) -> tuple[bool, str]:
        code = str(code or "")
        if not code.strip():
            return False, "empty_code"
        if len(code) > int(settings.developmental_sandbox_max_code_chars):
            return False, "code_too_large"
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return False, f"syntax_error:{exc.msg}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in self.ALLOWED_IMPORTS:
                        return False, f"import_not_allowed:{root}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root not in self.ALLOWED_IMPORTS:
                    return False, f"import_not_allowed:{root}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.BANNED_CALLS:
                    return False, f"call_not_allowed:{node.func.id}"
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return False, "dunder_attribute_not_allowed"
            elif isinstance(node, ast.Name) and node.id.startswith("__"):
                return False, "dunder_name_not_allowed"
        return True, "ok"

    @staticmethod
    def _preexec(memory_mb: int):
        def apply_limits():
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
                mem = int(memory_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except Exception:
                pass
        return apply_limits

    def run(self, code: str) -> dict[str, Any]:
        ok, reason = self.validate(code)
        if not ok:
            return {"ok": False, "executed": False, "reason": reason, "stdout": "", "stderr": ""}
        if not settings.developmental_sandbox_execution_enabled:
            return {"ok": True, "executed": False, "reason": "execution_disabled", "stdout": "", "stderr": ""}
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PATH": os.environ.get("PATH", "")}
        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = self._preexec(self.memory_mb)
        with tempfile.TemporaryDirectory(prefix="aaa_practice_") as tmp:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", code], cwd=tmp, env=env, text=True,
                    capture_output=True, timeout=self.timeout, **kwargs,
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "executed": True, "reason": "timeout", "stdout": "", "stderr": ""}
            except Exception as exc:
                return {"ok": False, "executed": True, "reason": f"{type(exc).__name__}:{exc}", "stdout": "", "stderr": ""}
        return {
            "ok": proc.returncode == 0,
            "executed": True,
            "reason": "ok" if proc.returncode == 0 else f"exit_code:{proc.returncode}",
            "stdout": (proc.stdout or "")[: self.max_output],
            "stderr": (proc.stderr or "")[: self.max_output],
        }


class DevelopmentalLearningSystem:
    """Developmental learning layered on human-inspired cognitive memory."""

    STAGES = ((0.0, "seed"), (5.0, "explorer"), (20.0, "apprentice"), (60.0, "practitioner"), (150.0, "specialist"))

    def __init__(self, human_learning: HumanLearningSystem):
        self.human_learning = human_learning
        self.db_path = human_learning.db_path
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.sandbox = SafePracticeSandbox()
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
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS developmental_profiles (
                    user_id TEXT PRIMARY KEY, xp REAL NOT NULL DEFAULT 0.0,
                    stage TEXT NOT NULL DEFAULT 'seed', observations INTEGER NOT NULL DEFAULT 0,
                    practice_attempts INTEGER NOT NULL DEFAULT 0, practice_successes INTEGER NOT NULL DEFAULT 0,
                    analogies INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS learning_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, title TEXT NOT NULL,
                    question TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', priority REAL NOT NULL DEFAULT 0.5,
                    curiosity REAL NOT NULL DEFAULT 0.5, confidence REAL NOT NULL DEFAULT 0.0,
                    attempts INTEGER NOT NULL DEFAULT 0, successes INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL, last_attempt REAL
                );
                CREATE INDEX IF NOT EXISTS idx_learning_goals_user_status ON learning_goals(user_id,status,priority,updated_at);
                CREATE TABLE IF NOT EXISTS developmental_practice (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, goal_id INTEGER NOT NULL, user_id TEXT NOT NULL,
                    mode TEXT NOT NULL, prompt TEXT NOT NULL, artifact TEXT NOT NULL DEFAULT '',
                    score REAL NOT NULL DEFAULT 0.0, success INTEGER NOT NULL DEFAULT 0,
                    feedback TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
                    FOREIGN KEY(goal_id) REFERENCES learning_goals(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_dev_practice_goal ON developmental_practice(user_id,goal_id,created_at);
                CREATE TABLE IF NOT EXISTS developmental_analogies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, source_title TEXT NOT NULL,
                    target_query TEXT NOT NULL, mapping TEXT NOT NULL, limitation TEXT NOT NULL DEFAULT '',
                    usefulness REAL NOT NULL DEFAULT 0.5, created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dev_analogies_user ON developmental_analogies(user_id,created_at);
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def stage_for_xp(cls, xp: float) -> str:
        stage = cls.STAGES[0][1]
        for threshold, candidate in cls.STAGES:
            if float(xp) >= threshold:
                stage = candidate
        return stage

    def _ensure_profile(self, user_id: str) -> None:
        now = time.time()
        self._conn().execute(
            "INSERT INTO developmental_profiles(user_id,xp,stage,updated_at) VALUES(?,0.0,'seed',?) ON CONFLICT(user_id) DO NOTHING",
            (user_id, now),
        )
        self._conn().commit()

    def _add_xp(self, user_id: str, amount: float) -> dict[str, Any]:
        self._ensure_profile(user_id)
        row = self._conn().execute("SELECT * FROM developmental_profiles WHERE user_id=?", (user_id,)).fetchone()
        old_stage = str(row["stage"])
        xp = max(0.0, float(row["xp"]) + float(amount))
        stage = self.stage_for_xp(xp)
        self._conn().execute("UPDATE developmental_profiles SET xp=?,stage=?,updated_at=? WHERE user_id=?", (xp, stage, time.time(), user_id))
        self._conn().commit()
        return {"xp": xp, "stage": stage, "stage_changed": stage != old_stage, "previous_stage": old_stage}

    def profile(self, user_id: str) -> dict[str, Any]:
        self._ensure_profile(user_id)
        row = self._conn().execute("SELECT * FROM developmental_profiles WHERE user_id=?", (user_id,)).fetchone()
        open_goals = self._conn().execute("SELECT COUNT(*) c FROM learning_goals WHERE user_id=? AND status='open'", (user_id,)).fetchone()["c"]
        return {"enabled": bool(settings.developmental_learning_enabled), "user_id": user_id, "xp": float(row["xp"]),
                "stage": str(row["stage"]), "observations": int(row["observations"]),
                "practice_attempts": int(row["practice_attempts"]), "practice_successes": int(row["practice_successes"]),
                "analogies": int(row["analogies"]), "open_goals": int(open_goals)}

    @staticmethod
    def _goal_row(row: sqlite3.Row) -> LearningGoal:
        return LearningGoal(id=int(row["id"]), user_id=str(row["user_id"]), title=str(row["title"]), question=str(row["question"]),
                            status=str(row["status"]), priority=float(row["priority"]), curiosity=float(row["curiosity"]),
                            confidence=float(row["confidence"]), attempts=int(row["attempts"]), successes=int(row["successes"]),
                            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
                            last_attempt=float(row["last_attempt"]) if row["last_attempt"] is not None else None)

    def goals(self, user_id: str, status: str = "open", limit: int = 20) -> list[LearningGoal]:
        limit = max(1, min(100, int(limit)))
        if status == "all":
            rows = self._conn().execute("SELECT * FROM learning_goals WHERE user_id=? ORDER BY priority DESC,updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM learning_goals WHERE user_id=? AND status=? ORDER BY priority DESC,updated_at DESC LIMIT ?", (user_id, status, limit)).fetchall()
        return [self._goal_row(row) for row in rows]

    @staticmethod
    def _terms(text: str) -> set[str]:
        import re
        return {x for x in re.findall(r"[\wáéíóúñü]+", text.lower()) if len(x) > 3}

    def create_goal(self, user_id: str, question: str, curiosity: float, confidence: float, title: str = "") -> int | None:
        question = " ".join(str(question or "").split())[:2000]
        if not question:
            return None
        terms = self._terms(question)
        for row in self._conn().execute("SELECT * FROM learning_goals WHERE user_id=? AND status='open' ORDER BY updated_at DESC LIMIT 100", (user_id,)).fetchall():
            existing_terms = self._terms(str(row["question"]))
            union = terms | existing_terms
            if union and len(terms & existing_terms) / len(union) >= 0.55:
                self._conn().execute(
                    "UPDATE learning_goals SET curiosity=?,confidence=?,priority=?,updated_at=? WHERE id=?",
                    (max(float(row["curiosity"]), self._clip(curiosity)), max(float(row["confidence"]), self._clip(confidence)),
                     max(float(row["priority"]), self._clip(0.55 * curiosity + 0.45 * (1.0 - confidence))), time.time(), int(row["id"])),
                )
                self._conn().commit()
                return int(row["id"])
        now = time.time()
        priority = self._clip(0.60 * curiosity + 0.40 * (1.0 - confidence))
        cur = self._conn().execute(
            "INSERT INTO learning_goals(user_id,title,question,status,priority,curiosity,confidence,created_at,updated_at) VALUES(?,?,?,'open',?,?,?,?,?)",
            (user_id, (title or question[:120])[:160], question, priority, self._clip(curiosity), self._clip(confidence), now, now),
        )
        self._conn().commit()
        return int(cur.lastrowid)

    def observe(self, user_id: str, query: str, cognition: dict[str, Any]) -> dict[str, Any]:
        self._ensure_profile(user_id)
        novelty = self._clip(float(cognition.get("novelty", 1.0)))
        confidence = self._clip(float(cognition.get("confidence", 0.0)))
        known_patterns = max(0, int(cognition.get("known_patterns", 0)))
        gap = 1.0 / (1.0 + known_patterns)
        curiosity = self._clip(float(settings.developmental_curiosity_novelty_weight) * novelty
                               + float(settings.developmental_curiosity_uncertainty_weight) * (1.0 - confidence)
                               + float(settings.developmental_curiosity_gap_weight) * gap)
        self._conn().execute("UPDATE developmental_profiles SET observations=observations+1,updated_at=? WHERE user_id=?", (time.time(), user_id))
        self._conn().commit()
        goal_id = None
        if settings.developmental_auto_goals and curiosity >= float(settings.developmental_goal_curiosity_threshold) and len(self.goals(user_id, "open", 100)) < int(settings.developmental_max_open_goals):
            goal_id = self.create_goal(user_id, query, curiosity, confidence)
        stage = self.profile(user_id)["stage"]
        learning_mode = {"seed": "observe_and_compare", "explorer": "observe_and_compare", "apprentice": "practice_and_analogy",
                         "practitioner": "generalize_and_test", "specialist": "teach_back_and_refine"}.get(stage, "observe_and_compare")
        return {"stage": stage, "learning_mode": learning_mode, "curiosity": curiosity, "novelty": novelty,
                "confidence": confidence, "goal_created": goal_id, "active_goals": [asdict(g) for g in self.goals(user_id, "open", 5)]}

    def record_analogy(self, user_id: str, source_title: str, target_query: str, mapping: str, limitation: str = "", usefulness: float = 0.5) -> int:
        now = time.time()
        with self._write_lock:
            cur = self._conn().execute(
                "INSERT INTO developmental_analogies(user_id,source_title,target_query,mapping,limitation,usefulness,created_at) VALUES(?,?,?,?,?,?,?)",
                (user_id, str(source_title)[:200], str(target_query)[:2000], str(mapping)[:5000], str(limitation)[:2500], self._clip(usefulness), now),
            )
            self._conn().execute("UPDATE developmental_profiles SET analogies=analogies+1,updated_at=? WHERE user_id=?", (now, user_id))
            self._conn().commit()
        return int(cur.lastrowid)

    def update_from_feedback(self, user_id: str, query: str, reward: float) -> dict[str, Any]:
        reward = max(-1.0, min(1.0, float(reward)))
        terms = self._terms(query)
        best, best_score = None, 0.0
        for row in self._conn().execute("SELECT * FROM learning_goals WHERE user_id=? AND status='open' ORDER BY priority DESC,updated_at DESC LIMIT 100", (user_id,)).fetchall():
            gt = self._terms(str(row["question"])); union = terms | gt
            score = (len(terms & gt) / len(union)) if union else 0.0
            if score > best_score:
                best_score, best = score, row
        goal_id = None
        if best is not None and best_score >= 0.20:
            goal_id = int(best["id"]); attempts = int(best["attempts"]) + 1
            successes = int(best["successes"]) + int(reward >= 0.35)
            confidence = self._clip(float(best["confidence"]) * 0.75 + max(0.0, reward) * 0.25)
            status = "mastered" if successes >= int(settings.developmental_goal_mastery_successes) and confidence >= 0.65 else "open"
            self._conn().execute("UPDATE learning_goals SET attempts=?,successes=?,confidence=?,status=?,updated_at=?,last_attempt=? WHERE id=? AND user_id=?",
                                 (attempts, successes, confidence, status, time.time(), time.time(), goal_id, user_id))
            self._conn().commit()
        xp_gain = max(0.0, reward) * float(settings.developmental_feedback_xp)
        if reward < 0:
            xp_gain = float(settings.developmental_failure_learning_xp)
        progress = self._add_xp(user_id, xp_gain)
        return {"goal_id": goal_id, "matched": best_score, "xp_gain": xp_gain, **progress}

    async def practice_goal(self, user_id: str, goal_id: int, provider: Any) -> dict[str, Any]:
        row = self._conn().execute("SELECT * FROM learning_goals WHERE id=? AND user_id=?", (int(goal_id), user_id)).fetchone()
        if row is None:
            raise ValueError("learning_goal_not_found")
        goal = self._goal_row(row)
        prompt = [
            {"role": "system", "content": "Create one small deliberate-practice exercise for an AI learning goal. Return ONLY JSON with keys: mode ('concept' or 'python'), exercise, answer, code, expected. Use python only for deterministic algorithmic practice that needs no files, network, subprocesses or external packages. For concept mode leave code and expected empty. Keep it compact and safe."},
            {"role": "user", "content": f"Learning goal: {goal.title}\nQuestion/gap: {goal.question}\nCurrent confidence: {goal.confidence:.2f}"},
        ]
        msg = await provider.chat(prompt, temperature=0.2, tools=None, tool_choice=None)
        raw = str(msg.get("content") or "") if isinstance(msg, dict) else ""; start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("practice_generator_invalid_json")
        item = json.loads(raw[start:end + 1])
        if not isinstance(item, dict):
            raise RuntimeError("practice_generator_invalid_payload")
        mode = str(item.get("mode") or "concept").strip().lower(); exercise = str(item.get("exercise") or goal.question).strip()[:5000]
        answer = str(item.get("answer") or "").strip()[:5000]; code = str(item.get("code") or "").strip()[: int(settings.developmental_sandbox_max_code_chars)]
        expected = str(item.get("expected") or "").strip()[:2000]; artifact = answer; sandbox_result = None
        if mode == "python" and code:
            sandbox_result = self.sandbox.run(code); artifact = code
            if sandbox_result.get("executed"):
                stdout = str(sandbox_result.get("stdout") or "").strip(); score = 1.0 if sandbox_result.get("ok") and (not expected or expected in stdout) else 0.0
                feedback = "Sandbox execution matched the expected result." if score >= 1.0 else f"Sandbox result did not meet expectation. reason={sandbox_result.get('reason')}"
            else:
                score = 0.5 if sandbox_result.get("ok") else 0.0
                feedback = "Code passed the restricted sandbox validator; execution is disabled." if sandbox_result.get("ok") else f"Code rejected by sandbox: {sandbox_result.get('reason')}"
        else:
            judge_prompt = [
                {"role": "system", "content": "Judge a self-practice answer against its exercise. Return ONLY JSON: {\"score\":0.0,\"feedback\":\"...\"} where score is 0..1. Be strict, concise and focus on correctness."},
                {"role": "user", "content": f"Exercise: {exercise}\nCandidate answer: {answer}"},
            ]
            judged = await provider.chat(judge_prompt, temperature=0.0, tools=None, tool_choice=None); judge_raw = str(judged.get("content") or "") if isinstance(judged, dict) else ""
            js, je = judge_raw.find("{"), judge_raw.rfind("}"); score, feedback = 0.0, "Could not grade practice."
            if js >= 0 and je >= js:
                try:
                    parsed = json.loads(judge_raw[js:je + 1]); score = self._clip(float(parsed.get("score", 0.0))); feedback = str(parsed.get("feedback") or feedback)[:3000]
                except Exception:
                    pass
        success = score >= float(settings.developmental_practice_success_score); now = time.time()
        with self._write_lock:
            cur = self._conn().execute("INSERT INTO developmental_practice(goal_id,user_id,mode,prompt,artifact,score,success,feedback,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                                       (goal.id, user_id, mode, exercise, artifact, score, int(success), feedback, now))
            attempts = goal.attempts + 1; successes = goal.successes + int(success); confidence = self._clip(goal.confidence * 0.75 + score * 0.25)
            status = "mastered" if successes >= int(settings.developmental_goal_mastery_successes) and confidence >= 0.65 else "open"
            self._conn().execute("UPDATE learning_goals SET attempts=?,successes=?,confidence=?,status=?,updated_at=?,last_attempt=? WHERE id=? AND user_id=?",
                                 (attempts, successes, confidence, status, now, now, goal.id, user_id))
            self._conn().execute("UPDATE developmental_profiles SET practice_attempts=practice_attempts+1,practice_successes=practice_successes+?,updated_at=? WHERE user_id=?",
                                 (int(success), now, user_id)); self._conn().commit()
        xp_gain = float(settings.developmental_practice_xp) * (0.25 + 0.75 * score); progress = self._add_xp(user_id, xp_gain)
        return {"ok": True, "practice_id": int(cur.lastrowid), "goal_id": goal.id, "mode": mode, "exercise": exercise, "artifact": artifact,
                "score": score, "success": success, "feedback": feedback, "sandbox": sandbox_result, "xp_gain": xp_gain, "progress": progress}

    def practice_history(self, user_id: str, goal_id: int | None = None, limit: int = 50) -> list[PracticeAttempt]:
        limit = max(1, min(200, int(limit)))
        if goal_id is None:
            rows = self._conn().execute("SELECT * FROM developmental_practice WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM developmental_practice WHERE user_id=? AND goal_id=? ORDER BY id DESC LIMIT ?", (user_id, int(goal_id), limit)).fetchall()
        return [PracticeAttempt(id=int(r["id"]), goal_id=int(r["goal_id"]), user_id=str(r["user_id"]), mode=str(r["mode"]), prompt=str(r["prompt"]), artifact=str(r["artifact"]), score=float(r["score"]), success=bool(r["success"]), feedback=str(r["feedback"]), created_at=float(r["created_at"])) for r in rows]

    def context_text(self, state: dict[str, Any]) -> str:
        goals = state.get("active_goals") or []
        goal_text = "; ".join(f"{g.get('title','goal')} (priority={float(g.get('priority',0.0)):.2f}, confidence={float(g.get('confidence',0.0)):.2f})" for g in goals[:5] if isinstance(g, dict)) or "none"
        return (f"Development stage={state.get('stage','seed')}; learning mode={state.get('learning_mode','observe_and_compare')}; "
                f"curiosity={float(state.get('curiosity',0.0)):.3f}; novelty={float(state.get('novelty',1.0)):.3f}; confidence={float(state.get('confidence',0.0)):.3f}. "
                f"Active learning goals: {goal_text}. Curiosity is a prioritization signal, not an instruction to take unsafe or unauthorized actions. "
                "Prefer evidence, analogies with explicit limits, and deliberate practice over pretending to know.")
