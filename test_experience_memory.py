from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app.core.experience import ExperienceStore
from app.core.memory import MemoryStore
from app.core.neural_memory import NeuralEmbedder


def fake_vector(text: str) -> list[float]:
    t = text.lower()
    if any(x in t for x in ("gradle", "android", "compilar", "build")):
        return [1.0, 0.0, 0.0, 0.0]
    if any(x in t for x in ("minecraft", "plugin", "spigot")):
        return [0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0, 0.0]


async def main_async() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = MemoryStore(str(Path(td) / "exp.db"))
        embedder = NeuralEmbedder()
        embedder.provider = "fake-test"
        embedder.model = "fake-exp-v1"

        async def fake_embed(texts: list[str]) -> list[list[float]]:
            return [fake_vector(t) for t in texts]

        embedder.embed = fake_embed  # type: ignore[method-assign]
        experiences = ExperienceStore(memory, embedder)

        good = await experiences.add(
            "u", "c", "Gradle falla al compilar mi app Android", "Actualizar Gradle wrapper", [], "unrated", 0.0, ""
        )
        await experiences.feedback("u", good, 1.0, "La compilación terminó correctamente", "Revisar compatibilidad de Gradle antes de cambiar el proyecto")
        await experiences.add(
            "u", "c", "Un plugin Spigot no carga", "Revisar dependencias", [], "failed", -1.0, "No asumir que Vault está instalado"
        )

        hits = await experiences.search("u", "¿Cómo arreglo el build de Android?", 2)
        assert hits and hits[0].id == good, hits
        assert hits[0].reward == 1.0
        assert "Gradle" in hits[0].situation

        out = Path(td) / "training.jsonl"
        count = experiences.export_training_jsonl("u", str(out), min_reward=0.5)
        assert count == 1
        assert out.exists() and '"messages"' in out.read_text(encoding="utf-8")
        print("Experience memory tests passed")


if __name__ == "__main__":
    asyncio.run(main_async())
