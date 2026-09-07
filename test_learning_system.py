from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path


class FakeEmbedder:
    enabled = True
    model = "fake-3d"

    async def embed(self, texts):
        out = []
        for text in texts:
            t = text.lower()
            v = [
                1.0 if any(x in t for x in ("build", "gradle", "android", "compile")) else 0.1,
                1.0 if any(x in t for x in ("error", "fall", "failed")) else 0.1,
                1.0 if any(x in t for x in ("success", "sirvió", "worked", "fix")) else 0.1,
            ]
            norm = sum(x*x for x in v) ** 0.5
            out.append([x / norm for x in v])
        return out


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        os.environ["DATABASE_PATH"] = str(Path(td) / "test.db")
        os.environ["EXPERIENCE_POLICY_DIR"] = str(Path(td) / "policy")
        os.environ["LORA_OUTPUT_DIR"] = str(Path(td) / "adapters")
        os.environ["EMBEDDING_PROVIDER"] = "off"

        from app.core.memory import MemoryStore
        from app.core.experience import ExperienceStore
        from app.core.continual_learning import AdapterRegistry, AdapterVersion
        from app.core.experience_policy import ExperiencePolicyNetwork

        memory = MemoryStore(os.environ["DATABASE_PATH"])
        experiences = ExperienceStore(memory, FakeEmbedder())
        policy = ExperiencePolicyNetwork(experiences)

        async def seed():
            rows = [
                ("Android Gradle build error", "update Gradle wrapper", 1.0, "success"),
                ("Android compile failed", "check AGP compatibility", 0.9, "success"),
                ("Build error after dependency update", "align dependency versions", 0.8, "success"),
                ("Gradle error", "delete random system files", -1.0, "failed"),
            ]
            for i, (situation, action, reward, outcome) in enumerate(rows):
                eid = await experiences.add("tester", f"c{i}", situation, action, outcome=outcome, reward=reward)
                assert eid > 0

        asyncio.run(seed())
        trained = policy.train_user("tester")
        # PyTorch is installed with the neural-memory stack in the normal installation.
        if trained.get("available"):
            assert trained["ok"], trained
            pred = asyncio.run(policy.predict("tester", "My Android build has a Gradle compile error"))
            assert pred is not None and -1.0 <= pred["predicted_reward"] <= 1.0

        registry = AdapterRegistry(os.environ["LORA_OUTPUT_DIR"])
        a1 = Path(td) / "a1"; a1.mkdir()
        a2 = Path(td) / "a2"; a2.mkdir()
        registry.register(AdapterVersion("v1", "tester", str(a1), "base", 1.0, 10, .7, 1.0, .0002))
        registry.register(AdapterVersion("v2", "tester", str(a2), "base", 2.0, 12, .7, 1.0, .0002))
        registry.activate("tester", "v1")
        registry.activate("tester", "v2")
        assert registry.active("tester")["id"] == "v2"
        rolled = registry.rollback("tester")
        assert rolled and rolled["id"] == "v1"

    print("Learning system tests passed")


if __name__ == "__main__":
    main()
