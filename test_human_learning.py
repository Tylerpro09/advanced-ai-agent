from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app.core.experience import ExperienceStore
from app.core.human_learning import HumanLearningSystem
from app.core.memory import MemoryStore
from app.core.neural_memory import NeuralEmbedder


def fake_vector(text: str) -> list[float]:
    t = text.lower()
    if any(x in t for x in ("gradle", "android", "build", "compilar")):
        return [1.0, 0.0, 0.0, 0.0]
    if any(x in t for x in ("plugin", "spigot", "minecraft")):
        return [0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0, 0.0]


async def main_async() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = MemoryStore(str(Path(td) / "human.db"))
        embedder = NeuralEmbedder()
        embedder.provider = "fake-test"
        embedder.model = "fake-human-v1"

        async def fake_embed(texts: list[str]) -> list[list[float]]:
            return [fake_vector(t) for t in texts]

        embedder.embed = fake_embed  # type: ignore[method-assign]
        experiences = ExperienceStore(memory, embedder)
        learning = HumanLearningSystem(experiences, embedder)

        eid = await experiences.add(
            "u", "c", "Gradle falla al compilar Android", "Revisé Gradle wrapper y AGP", [],
            "success", 1.0, "Comprobar compatibilidad entre Gradle y AGP antes de cambiar código",
        )
        exp = next(x for x in experiences.list("u", 20) if x.id == eid)
        learned = await learning.learn_from_experience(
            "u", exp,
            {"title": "Compatibilidad de build Android", "trigger": "Errores de compilación Gradle en Android",
             "principle": "Verificar primero la matriz de compatibilidad Gradle/AGP.",
             "procedure": ["Leer versiones", "Comparar compatibilidad", "Actualizar solo lo necesario"],
             "mistake": "Cambiar código antes de revisar toolchain", "novelty": 0.8},
        )
        assert learned["learned"]
        stats = learning.stats("u")
        assert stats["concepts"] >= 1 and stats["procedures"] >= 1, stats

        ctx = await learning.cognitive_context("u", "¿Cómo arreglo otro error de build Android?", None)
        assert ctx["known_patterns"] >= 1, ctx
        assert any(x["memory_type"] == "procedure" for x in ctx["memories"]), ctx
        assert ctx["novelty"] < 0.5, ctx

        bad_id = await experiences.add(
            "u", "c", "Plugin Spigot no carga", "Forcé el inicio sin revisar dependencias", [],
            "failed", -1.0, "No forzar carga sin comprobar dependencias",
        )
        bad = next(x for x in experiences.list("u", 20) if x.id == bad_id)
        await learning.learn_from_experience(
            "u", bad,
            {"title": "Dependencias Spigot", "trigger": "Plugin de Minecraft no carga",
             "principle": "Revisar dependencias antes de forzar el arranque.", "procedure": [],
             "mistake": "Forzar la carga sin dependencias", "novelty": 0.6},
        )
        ctx2 = await learning.cognitive_context("u", "Mi plugin Spigot falla al cargar", None)
        assert any(x["memory_type"] == "avoidance" for x in ctx2["memories"]), ctx2
        assert ctx2["mode"] == "caution", ctx2
        print("Human-like learning tests passed")


if __name__ == "__main__":
    asyncio.run(main_async())
