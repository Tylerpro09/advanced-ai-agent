from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app.config import settings
from app.core.developmental_learning import DevelopmentalLearningSystem
from app.core.experience import ExperienceStore
from app.core.human_learning import HumanLearningSystem
from app.core.memory import MemoryStore


class FakeEmbedder:
    enabled = False
    model = "test-off"

    async def embed(self, texts):
        return []


class FakeProvider:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, temperature=0.0, tools=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": (
                    '{"mode":"concept","exercise":"Explain why a binary search halves the search space.",'
                    '"answer":"Because each comparison discards one ordered half of the remaining interval.",'
                    '"code":"","expected":""}'
                ),
            }
        return {
            "role": "assistant",
            "content": '{"score":0.95,"feedback":"Correct explanation of the invariant and halving step."}',
        }


async def main():
    old_exec = settings.developmental_sandbox_execution_enabled
    old_auto_goals = settings.developmental_auto_goals
    try:
        settings.developmental_sandbox_execution_enabled = False
        settings.developmental_auto_goals = True

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(str(Path(tmp) / "agent.db"))
            embedder = FakeEmbedder()
            experiences = ExperienceStore(memory, embedder)
            human = HumanLearningSystem(experiences, embedder)
            dev = DevelopmentalLearningSystem(human)

            assert dev.stage_for_xp(0) == "seed"
            assert dev.stage_for_xp(5) == "explorer"
            assert dev.stage_for_xp(20) == "apprentice"
            assert dev.stage_for_xp(60) == "practitioner"
            assert dev.stage_for_xp(150) == "specialist"

            state = dev.observe(
                "u1",
                "How do I debug a difficult Android build problem?",
                {"novelty": 0.98, "confidence": 0.08, "known_patterns": 0},
            )
            assert state["curiosity"] >= settings.developmental_goal_curiosity_threshold
            assert state["goal_created"] is not None
            assert len(dev.goals("u1")) == 1

            dev.create_goal(
                "u1",
                "How do I debug a difficult Android build problem?",
                0.9,
                0.1,
            )
            assert len(dev.goals("u1")) == 1

            query = "How do I debug a difficult Android build problem?"
            for _ in range(settings.developmental_goal_mastery_successes):
                progress = dev.update_from_feedback("u1", query, 1.0)
                assert progress["xp_gain"] > 0
            all_goals = dev.goals("u1", "all")
            assert all_goals[0].status == "mastered"
            assert dev.profile("u1")["xp"] > 0

            safe, reason = dev.sandbox.validate("print(sum(range(10)))")
            assert safe and reason == "ok"
            safe, reason = dev.sandbox.validate("import os\nprint(os.getcwd())")
            assert not safe and "import_not_allowed" in reason
            safe, reason = dev.sandbox.validate("open('x.txt','w')")
            assert not safe and "call_not_allowed" in reason
            sandbox_result = dev.sandbox.run("print(42)")
            assert sandbox_result["ok"] is True
            assert sandbox_result["executed"] is False

            practice_goal = dev.create_goal(
                "u1",
                "Understand binary search invariants",
                0.85,
                0.2,
                "Binary search",
            )
            result = await dev.practice_goal("u1", int(practice_goal), FakeProvider())
            assert result["success"] is True
            assert result["score"] >= 0.9
            assert dev.profile("u1")["practice_attempts"] >= 1
            assert len(dev.practice_history("u1", practice_goal, 10)) == 1

        print("Developmental learning tests passed")
    finally:
        settings.developmental_sandbox_execution_enabled = old_exec
        settings.developmental_auto_goals = old_auto_goals


if __name__ == "__main__":
    asyncio.run(main())
