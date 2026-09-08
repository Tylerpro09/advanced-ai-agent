from __future__ import annotations

import asyncio
from typing import Any

import httpx


class OpenAICompatibleProvider:
    """OpenAI-compatible chat provider with LM Studio-friendly model discovery.

    Set AI_MODEL=auto to select the first model exposed by GET /v1/models.
    The resolved id is cached for subsequent requests. This works with LM Studio
    and other OpenAI-compatible servers that implement the models endpoint.
    """

    AUTO_MODEL_VALUES = {"", "auto", "automatic"}

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = (model or "auto").strip()
        self.timeout = timeout
        self._resolved_model: str | None = None
        self._model_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _resolve_model(self, client: httpx.AsyncClient) -> str:
        if self.model.lower() not in self.AUTO_MODEL_VALUES:
            return self.model
        if self._resolved_model:
            return self._resolved_model

        async with self._model_lock:
            if self._resolved_model:
                return self._resolved_model
            try:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                response.raise_for_status()
                payload = response.json()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot connect to the OpenAI-compatible server at {self.base_url}. "
                    "If using LM Studio, start Local Server in the Developer tab."
                ) from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Could not list models from {self.base_url}/models: {exc}") from exc
            except ValueError as exc:
                raise RuntimeError("The model-list endpoint returned invalid JSON.") from exc

            rows = payload.get("data", []) if isinstance(payload, dict) else []
            model_ids = [str(row.get("id", "")).strip() for row in rows if isinstance(row, dict)]
            model_ids = [value for value in model_ids if value]
            if not model_ids:
                raise RuntimeError(
                    "The OpenAI-compatible server is reachable but exposes no models. "
                    "Load a model in LM Studio, then try again."
                )
            self._resolved_model = model_ids[0]
            return self._resolved_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            model = await self._resolve_model(client)
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice

            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot connect to the OpenAI-compatible server at {self.base_url}. "
                    "If using LM Studio, start Local Server in the Developer tab."
                ) from exc
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:1000] if exc.response is not None else str(exc)
                raise RuntimeError(f"AI server returned HTTP {exc.response.status_code}: {detail}") from exc
            except (httpx.HTTPError, ValueError) as exc:
                raise RuntimeError(f"AI server request failed: {exc}") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices or not isinstance(choices, list) or not isinstance(choices[0], dict):
            raise RuntimeError("AI server returned no valid choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("AI server returned an invalid assistant message.")
        self.model = model
        return message

    def status(self) -> dict[str, Any]:
        return {
            "backend": "openai_compatible",
            "base_url": self.base_url,
            "configured_model": self.model,
            "resolved_model": self._resolved_model or (self.model if self.model.lower() not in self.AUTO_MODEL_VALUES else ""),
            "auto_model": self.model.lower() in self.AUTO_MODEL_VALUES,
        }
