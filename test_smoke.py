from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app.core.memory import MemoryStore
from app.core.plugins import PluginManager
from app.core.rag import RAGStore
from app.tools.registry import ToolRegistry, safe_calculator


def main():
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "test.db")
        memory = MemoryStore(db)
        rag = RAGStore(db)
        memory.add_memory("u", "El proyecto principal usa Python", "project", 0.9)
        assert memory.search_memories("u", "Python")
        assert rag.ingest_text("u", "manual.txt", "La nave Aurora utiliza un reactor de fusión. El sistema auxiliar usa baterías.") >= 1
        hits = rag.search("u", "¿Qué reactor usa Aurora?")
        assert hits and "fusión" in hits[0].text
        assert safe_calculator("sqrt(81)+3") == "12.0"
        plugins = PluginManager("plugins")
        tools = ToolRegistry(memory, rag, plugins)
        assert any(x["function"]["name"] == "knowledge_search" for x in tools.schemas())
        result = asyncio.run(tools.execute("knowledge_search", {"query": "Aurora"}, "u"))
        assert "Aurora" in result
    print("Smoke tests passed")


if __name__ == "__main__":
    main()
