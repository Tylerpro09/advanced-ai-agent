from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx


class MediaClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def vision(self, image_path: str, prompt: str) -> str:
        p = Path(image_path)
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/chat/completions", headers={**self._headers(), "Content-Type": "application/json"}, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content", "")

    async def transcribe(self, audio_path: str) -> str:
        p = Path(audio_path)
        with p.open("rb") as f:
            files = {"file": (p.name, f, mimetypes.guess_type(p.name)[0] or "application/octet-stream")}
            data = {"model": self.model}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f"{self.base_url}/audio/transcriptions", headers=self._headers(), data=data, files=files)
                r.raise_for_status()
                result = r.json()
        return str(result.get("text", ""))

    async def speech(self, text: str, voice: str = "alloy", response_format: str = "mp3") -> bytes:
        payload = {"model": self.model, "voice": voice, "input": text, "response_format": response_format}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/audio/speech", headers={**self._headers(), "Content-Type": "application/json"}, json=payload)
            r.raise_for_status()
            return r.content
