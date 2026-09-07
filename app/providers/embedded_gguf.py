from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock
from typing import Any


class EmbeddedGGUFProvider:
    """In-process GGUF LLM provider powered by llama-cpp-python.

    The model is loaded lazily on first chat call so the API can start and
    expose diagnostics even when the GGUF path is not configured yet.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "embedded-gguf",
        timeout: float = 120.0,
        n_ctx: int = 8192,
        n_gpu_layers: int = 0,
        n_threads: int = 0,
        chat_format: str = "",
        verbose: bool = False,
        lora_path: str = "",
        lora_scale: float = 1.0,
    ):
        self.model_path = model_path
        self.model = model_name or Path(model_path).stem or "embedded-gguf"
        self.timeout = timeout
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.chat_format = chat_format.strip()
        self.verbose = verbose
        self.lora_path = lora_path.strip()
        self.lora_scale = float(lora_scale)
        self._llm = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def status(self) -> dict[str, Any]:
        path = Path(self.model_path).expanduser() if self.model_path else None
        return {
            "backend": "embedded_gguf",
            "model": self.model,
            "model_path": str(path) if path else "",
            "model_exists": bool(path and path.is_file()),
            "loaded": self.loaded,
            "context_length": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
            "threads": self.n_threads,
            "chat_format": self.chat_format or "auto",
            "lora_path": self.lora_path,
            "lora_exists": bool(self.lora_path and Path(self.lora_path).expanduser().is_file()),
            "lora_scale": self.lora_scale,
        }

    def _load(self):
        if self._llm is not None:
            return self._llm
        with self._load_lock:
            if self._llm is not None:
                return self._llm
            if not self.model_path:
                raise RuntimeError(
                    "LOCAL_MODEL_PATH is empty. Put a .gguf model in models/ and set LOCAL_MODEL_PATH in .env."
                )
            path = Path(self.model_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"GGUF model not found: {path}")
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "Embedded GGUF backend requires llama-cpp-python. Install requirements-local-model.txt."
                ) from exc

            kwargs: dict[str, Any] = {
                "model_path": str(path),
                "n_ctx": int(self.n_ctx),
                "n_gpu_layers": int(self.n_gpu_layers),
                "verbose": bool(self.verbose),
            }
            if self.n_threads > 0:
                kwargs["n_threads"] = int(self.n_threads)
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            if self.lora_path:
                lora = Path(self.lora_path).expanduser().resolve()
                if not lora.is_file():
                    raise FileNotFoundError(f"GGUF LoRA adapter not found: {lora}")
                kwargs["lora_path"] = str(lora)
                kwargs["lora_scale"] = float(self.lora_scale)
            self._llm = Llama(**kwargs)
            return self._llm

    def _chat_sync(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        llm = self._load()
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        # llama.cpp model objects are not guaranteed to be safe for overlapping generations.
        # Serialize inference while still allowing FastAPI to serve non-model endpoints concurrently.
        with self._inference_lock:
            response = llm.create_chat_completion(**kwargs)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Embedded model returned no choices")
        message = choices[0].get("message") or {}
        # Normalise to the same OpenAI-style dictionary used by the agent.
        return dict(message)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
    ) -> dict[str, Any]:
        task = asyncio.to_thread(self._chat_sync, messages, temperature, tools, tool_choice)
        return await asyncio.wait_for(task, timeout=self.timeout)
