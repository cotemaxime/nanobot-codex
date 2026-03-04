from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.providers.codex_sdk_provider import CodexSDKProvider


class _DummyTransport:
    def __init__(self, *args, **kwargs):
        self.last_model = None

    def validate_session(self):
        return True, "ok"

    async def chat(self, messages, tools, model, max_tokens, temperature):
        self.last_model = model
        return SimpleNamespace(
            content="ok",
            tool_calls=[],
            finish_reason="stop",
            usage={},
            reasoning_content=None,
        )


@pytest.mark.asyncio
async def test_codex_sdk_provider_normalizes_prefixed_model_override(monkeypatch):
    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexTransport", _DummyTransport)

    provider = CodexSDKProvider(
        default_model="openai-codex/gpt-5.3-codex",
        strict_auth=False,
    )

    response = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="openai-codex/gpt-5.3-codex",
    )

    assert response.content == "ok"
    assert provider.get_default_model() == "gpt-5.3-codex"
    assert provider.transport.last_model == "gpt-5.3-codex"
