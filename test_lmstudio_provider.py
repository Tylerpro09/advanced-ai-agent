from __future__ import annotations

import asyncio

import app.providers.openai_compatible as provider_module
from app.providers.openai_compatible import OpenAICompatibleProvider


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            request = provider_module.httpx.Request("GET", "http://test")
            response = provider_module.httpx.Response(self.status_code, request=request, text=self.text)
            raise provider_module.httpx.HTTPStatusError("fake", request=request, response=response)

    def json(self):
        return self._payload


class FakeClient:
    model_payload = {"data": [{"id": "lm-test-model"}]}
    posts = []
    gets = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, headers=None):
        type(self).gets += 1
        return FakeResponse(type(self).model_payload)

    async def post(self, url, headers=None, json=None):
        type(self).posts.append(json)
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})


async def main():
    original = provider_module.httpx.AsyncClient
    provider_module.httpx.AsyncClient = FakeClient
    try:
        FakeClient.gets = 0
        FakeClient.posts = []
        FakeClient.model_payload = {"data": [{"id": "lm-test-model"}]}
        p = OpenAICompatibleProvider("http://127.0.0.1:1234/v1", "lm-studio", "auto")
        message = await p.chat([{"role": "user", "content": "hi"}])
        assert message["content"] == "ok"
        assert FakeClient.posts[-1]["model"] == "lm-test-model"
        assert p.model == "lm-test-model"
        assert FakeClient.gets == 1

        # Once resolved, a second request must not query /models again.
        await p.chat([{"role": "user", "content": "again"}])
        assert FakeClient.gets == 1

        # Fixed model ids skip discovery completely.
        FakeClient.gets = 0
        p2 = OpenAICompatibleProvider("http://127.0.0.1:1234/v1", "", "fixed-model")
        await p2.chat([{"role": "user", "content": "hi"}])
        assert FakeClient.posts[-1]["model"] == "fixed-model"
        assert FakeClient.gets == 0

        # Reachable server but no visible/loaded model gives an actionable error.
        FakeClient.model_payload = {"data": []}
        p3 = OpenAICompatibleProvider("http://127.0.0.1:1234/v1", "", "auto")
        try:
            await p3.chat([{"role": "user", "content": "hi"}])
            raise AssertionError("Expected no-model error")
        except RuntimeError as exc:
            assert "exposes no models" in str(exc)
    finally:
        provider_module.httpx.AsyncClient = original

    print("LM Studio provider tests passed")


if __name__ == "__main__":
    asyncio.run(main())
