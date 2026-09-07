from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app.core.memory import MemoryStore
from app.core.neural_memory import NeuralMemoryStore


def fake_vector(text: str) -> list[float]:
    t = text.lower()
    # Semantic groups for an offline deterministic test of vector persistence/search.
    if any(x in t for x in ("gpu", "gráfica", "rtx", "3050")):
        return [1.0, 0.0, 0.0, 0.0]
    if any(x in t for x in ("comida", "pizza", "hamburguesa")):
        return [0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0, 0.0]


async def main_async() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = MemoryStore(str(Path(td) / "neural.db"))
        neural = NeuralMemoryStore(memory)
        neural.embedder.provider = "fake-test"
        neural.embedder.model = "fake-test-v1"

        async def fake_embed(texts: list[str]) -> list[list[float]]:
            return [fake_vector(t) for t in texts]

        neural.embedder.embed = fake_embed  # type: ignore[method-assign]

        a = memory.add_memory("u", "Mi computadora tiene una NVIDIA RTX 3050", "fact", 0.9)
        b = memory.add_memory("u", "Me gusta la pizza", "preference", 0.7)
        await neural.index_memory(a, "u", "Mi computadora tiene una NVIDIA RTX 3050")
        await neural.index_memory(b, "u", "Me gusta la pizza")

        hits = await neural.search("u", "¿Qué tarjeta gráfica tengo?", 2)
        assert hits, "No neural hits"
        assert hits[0].id == a, hits
        assert hits[0].semantic_score > 0.99
        print("Neural memory tests passed")


if __name__ == "__main__":
    asyncio.run(main_async())
