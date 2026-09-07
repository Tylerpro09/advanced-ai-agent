from __future__ import annotations

import asyncio
import sys
import tempfile
import types
from pathlib import Path

from app.providers.embedded_gguf import EmbeddedGGUFProvider


class FakeLlama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_chat_completion(self, **kwargs):
        assert kwargs["messages"][-1]["content"] == "hola"
        return {"choices": [{"message": {"role": "assistant", "content": "hola desde GGUF"}}]}


async def main():
    with tempfile.TemporaryDirectory() as d:
        model = Path(d) / "model.gguf"
        model.write_bytes(b"fake-gguf")
        sys.modules["llama_cpp"] = types.SimpleNamespace(Llama=FakeLlama)
        provider = EmbeddedGGUFProvider(str(model), n_ctx=4096)
        msg = await provider.chat([{"role": "user", "content": "hola"}], tools=None)
        assert msg["content"] == "hola desde GGUF"
        assert provider.status()["loaded"] is True
    print("Embedded GGUF provider tests passed")


if __name__ == "__main__":
    asyncio.run(main())
