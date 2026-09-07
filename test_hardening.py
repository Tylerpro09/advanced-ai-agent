from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

from app.config import settings
from app.core.agent import Agent
from app.core.experience import ExperienceStore
from app.core.experience_policy import ExperiencePolicyNetwork
from app.core.memory import MemoryStore
from app.core.neural_memory import NeuralMemoryStore
from app.core.plugins import PluginManager
from app.core.rag import RAGStore
from app.tools.registry import ToolRegistry


class FakeEmbedder:
    enabled = True
    model = "hardening-fake-v1"

    async def embed(self, texts):
        out = []
        for text in texts:
            t = text.lower()
            v = [
                1.0 if "android" in t or "gradle" in t else 0.2,
                1.0 if "funcion" in t or "success" in t else 0.2,
                1.0 if "fall" in t or "error" in t else 0.2,
            ]
            norm = sum(x * x for x in v) ** 0.5
            out.append([x / norm for x in v])
        return out


class LoopingProvider:
    model = "fake-loop"

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, temperature=0.0, tools=None, tool_choice=None):
        self.calls += 1
        if tools is not None:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call-{self.calls}",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'},
                }],
            }
        return {"role": "assistant", "content": "Resultado final: 4"}


def test_sqlite_concurrency() -> None:
    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(str(Path(td) / "concurrent.db"))
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                for i in range(40):
                    store.add_memory("u", f"worker={n} item={i}")
            except Exception as exc:  # pragma: no cover - failure collector
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert len(store.list_memories("u", 1000)) == 160


def test_private_http_is_blocked() -> None:
    old_allowlist = settings.http_allowlist
    old_private = settings.http_allow_private_networks
    try:
        settings.http_allowlist = "localhost,127.0.0.1"
        settings.http_allow_private_networks = False
        with tempfile.TemporaryDirectory() as td:
            memory = MemoryStore(str(Path(td) / "http.db"))
            rag = RAGStore(memory.path)
            tools = ToolRegistry(memory, rag)
            try:
                tools._assert_network_target_allowed("127.0.0.1")
            except ValueError:
                pass
            else:
                raise AssertionError("private HTTP target should have been rejected")
    finally:
        settings.http_allowlist = old_allowlist
        settings.http_allow_private_networks = old_private


def test_numpy_policy_fallback() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = MemoryStore(str(Path(td) / "policy.db"))
        experiences = ExperienceStore(memory, FakeEmbedder())

        async def seed():
            rows = [
                ("Android Gradle error", "fix versions", 1.0, "success"),
                ("Android build error", "align AGP", 0.9, "success"),
                ("Gradle failed", "delete random files", -1.0, "failed"),
                ("Android compile failed", "inspect dependencies", 0.8, "success"),
            ]
            for idx, (s, a, r, o) in enumerate(rows):
                await experiences.add("u", str(idx), s, a, outcome=o, reward=r)

        asyncio.run(seed())
        policy = ExperiencePolicyNetwork(experiences)
        policy.root = Path(td) / "policy"
        policy.root.mkdir()
        policy._torch = lambda: (None, None)  # type: ignore[method-assign]
        result = policy.train_user("u")
        assert result["ok"], result
        assert result["backend"] == "numpy", result
        pred = asyncio.run(policy.predict("u", "Android has a Gradle error"))
        assert pred is not None
        assert -1.0 <= pred["predicted_reward"] <= 1.0


def test_project_paths_and_plugins() -> None:
    assert Path(settings.database_path).is_absolute()
    assert Path(settings.uploads_dir).is_absolute()
    assert Path(settings.local_model_path).is_absolute()
    manager = PluginManager()
    assert manager.directory.is_absolute()
    assert manager.directory.parent == settings.project_root


def test_tool_loop_is_forced_to_finish() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "agent.db")
        memory = MemoryStore(db)
        neural = NeuralMemoryStore(memory)
        neural.embedder.provider = "off"
        rag = RAGStore(db)
        experiences = ExperienceStore(memory, neural.embedder)
        agent = Agent(memory, rag, neural, experiences)
        provider = LoopingProvider()
        agent.provider = provider
        old_rounds = settings.max_tool_rounds
        old_auto_memory = settings.auto_memory
        old_exp = settings.experience_auto_store
        try:
            settings.max_tool_rounds = 1
            settings.auto_memory = False
            settings.experience_auto_store = False
            result = asyncio.run(agent.chat("u", "c", "cuanto es 2+2"))
            assert result["answer"] == "Resultado final: 4", result
            assert result["tool_rounds"] == 1
            assert provider.calls == 3
        finally:
            settings.max_tool_rounds = old_rounds
            settings.auto_memory = old_auto_memory
            settings.experience_auto_store = old_exp


def main() -> None:
    test_sqlite_concurrency()
    test_private_http_is_blocked()
    test_numpy_policy_fallback()
    test_project_paths_and_plugins()
    test_tool_loop_is_forced_to_finish()
    print("Hardening tests passed")


if __name__ == "__main__":
    main()
