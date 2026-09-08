from __future__ import annotations

import re
from typing import Any

import httpx


_CASUAL_GREETING = re.compile(
    r"^\s*(hola+|holi+|hey+|ey+|buenas+|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|hello+|hi+)\s*[!.?¡¿,]*\s*$",
    re.IGNORECASE,
)
_GENERIC_GREETING_REPLY = re.compile(
    r"(?i)(en\s+qu[eé]\s+puedo\s+ayudarte|c[oó]mo\s+puedo\s+ayudarte|how\s+can\s+i\s+help|what\s+can\s+i\s+help\s+you\s+with)"
)


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._resolved_model: str | None = None

    def _is_primary_conversation(self, messages: list[dict[str, Any]]) -> bool:
        systems = "\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        return "advanced, local-first AI assistant" in systems or "Developmental learning state" in systems

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages):
            if item.get("role") == "user":
                return str(item.get("content") or "").strip()
        return ""

    @staticmethod
    def _has_prior_substantive_turn(messages: list[dict[str, Any]]) -> bool:
        user_messages = [
            str(m.get("content") or "").strip()
            for m in messages
            if m.get("role") == "user" and str(m.get("content") or "").strip()
        ]
        if len(user_messages) < 2:
            return False
        return any(len(x) > 12 and not _CASUAL_GREETING.match(x) for x in user_messages[:-1])

    def _presence_instruction(self, messages: list[dict[str, Any]]) -> str | None:
        if not self._is_primary_conversation(messages):
            return None
        user_text = self._last_user_text(messages)
        greeting = bool(_CASUAL_GREETING.match(user_text))
        has_context = self._has_prior_substantive_turn(messages)

        rules = [
            "Sound like a natural ongoing conversation, not customer support or a help-desk bot.",
            "Match the user's language, brevity, vocabulary and energy without caricaturing them.",
            "Do not default to phrases like '¿En qué puedo ayudarte hoy?', '¿Cómo puedo ayudarte?' or their English equivalents.",
            "Do not repeatedly announce capabilities, identity, memory, tools or that you are an AI unless it is relevant.",
            "Use the conversation's concrete context naturally instead of restarting the relationship every turn.",
            "Acknowledge frustration by referring to the specific thing that went wrong; avoid canned empathy.",
            "Do not pretend to have human feelings, a body, childhood, private life or experiences you do not have.",
            "Keep casual turns casual. A one-word message usually does not need a paragraph.",
            "Vary phrasing naturally; do not use the same greeting or closing formula every time.",
        ]
        if greeting and has_context:
            rules.append(
                "This message is only a greeting inside an existing conversation. Reply briefly and warmly, and naturally signal continuity with the prior topic instead of asking a generic help question."
            )
        elif greeting:
            rules.append(
                "This message is only a greeting. Reply with a short natural greeting or conversational check-in; do not turn it into a support prompt."
            )
        return "CONVERSATIONAL PRESENCE:\n- " + "\n- ".join(rules)

    async def _resolve_model(self, client: httpx.AsyncClient, headers: dict[str, str]) -> str:
        configured = (self.model or "").strip()
        if configured and configured.lower() not in {"auto", "local-model", "lm-studio"}:
            return configured
        if self._resolved_model:
            return self._resolved_model
        try:
            response = await client.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach the OpenAI-compatible model list at {self.base_url}/models. "
                "If you use LM Studio, load a model and start Developer > Local Server."
            ) from exc
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            raise RuntimeError(
                "The OpenAI-compatible server is reachable but exposes no models. "
                "Load a model in LM Studio and keep Local Server running."
            )
        for item in items:
            if isinstance(item, dict) and str(item.get("id") or "").strip():
                self._resolved_model = str(item["id"]).strip()
                return self._resolved_model
        raise RuntimeError("The model server returned /v1/models without a usable model id.")

    def _humanize_canned_greeting(
        self,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._is_primary_conversation(messages):
            return assistant_message
        user_text = self._last_user_text(messages)
        if not _CASUAL_GREETING.match(user_text):
            return assistant_message
        content = str(assistant_message.get("content") or "").strip()
        if not _GENERIC_GREETING_REPLY.search(content):
            return assistant_message
        replacement = (
            "¡Hey! 👋 Aquí estoy. ¿Seguimos con lo de antes o cambiaste de tema?"
            if self._has_prior_substantive_turn(messages)
            else "¡Hey! 👋 ¿Qué tal?"
        )
        out = dict(assistant_message)
        out["content"] = replacement
        return out

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
    ) -> dict[str, Any]:
        outgoing = [dict(m) for m in messages]
        presence = self._presence_instruction(outgoing)
        if presence:
            outgoing.insert(1 if outgoing and outgoing[0].get("role") == "system" else 0, {"role": "system", "content": presence})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            model = await self._resolve_model(client, headers)
            payload: dict[str, Any] = {
                "model": model,
                "messages": outgoing,
                "temperature": temperature,
            }
            if tools:
                payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice

            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:1000]
                raise RuntimeError(
                    f"OpenAI-compatible server returned HTTP {exc.response.status_code}: {detail}"
                ) from exc
            except httpx.RequestError as exc:
                raise RuntimeError(
                    f"Could not connect to {self.base_url}. If you use LM Studio, start Developer > Local Server."
                ) from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("OpenAI-compatible server returned no valid choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI-compatible server returned an invalid assistant message.")
        return self._humanize_canned_greeting(messages, dict(message))
