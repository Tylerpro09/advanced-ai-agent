from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.core.experience import ExperienceStore
from app.core.human_learning import HumanLearningSystem
from app.core.memory import MemoryStore
from app.core.developmental_learning import DevelopmentalLearningSystem, SafePracticeSandbox


class NoEmbedder:
    enabled = False
    model = "off"

    async def embed(self, texts):
        return []


def build_system(path: str) -> DevelopmentalLearningSystem:
    memory = MemoryStore(path)
    experiences = ExperienceStore(memory, NoEmbedder())
    return DevelopmentalLearningSystem(HumanLearningSystem(experiences, NoEmbedder()))


def main() -> None:
    # v2.7.1 compatibility hook is installed by app.version during app.main import.
    from app.main import app

    client = TestClient(app)
    paths = {getattr(route, "path", "") for route in app.routes}
    required = {
        "/v1/development/status",
        "/v1/development/goals",
        "/v1/development/goals/{goal_id}/practice",
        "/v1/development/practice",
        "/v1/development/sandbox/validate",
    }
    assert required <= paths, sorted(required - paths)

    sandbox = SafePracticeSandbox()
    assert sandbox.validate("f = open\nf('x.txt', 'w')")[0] is False
    assert sandbox.validate("from math import __dict__")[0] is False
    assert sandbox.validate("import math\nprint(math.sqrt(9))")[0] is True

    with tempfile.TemporaryDirectory() as td:
        dev = build_system(str(Path(td) / "agent.db"))
        uid = "v271-concurrency"
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: dev._add_xp(uid, 0.5), range(40)))
        assert abs(dev.profile(uid)["xp"] - 20.0) < 1e-9

        fresh = "fresh-analogy-user"
        assert dev.record_analogy(fresh, "source", "target", "mapping") > 0
        assert dev.profile(fresh)["analogies"] == 1

        old_cap = settings.developmental_max_open_goals
        settings.developmental_max_open_goals = 2
        try:
            assert dev.create_goal(uid, "Learn alpha invariant", 0.9, 0.1) is not None
            assert dev.create_goal(uid, "Learn beta protocol", 0.9, 0.1) is not None
            assert dev.create_goal(uid, "Learn gamma topology", 0.9, 0.1) is None
        finally:
            settings.developmental_max_open_goals = old_cap

    print("v2.7.1 hardening tests passed")


if __name__ == "__main__":
    main()
