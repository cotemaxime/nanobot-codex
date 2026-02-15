from types import SimpleNamespace

import pytest

from nanobot.config.schema import Config
from nanobot.providers.factory import create_provider
from nanobot.providers.litellm_provider import LiteLLMProvider


def test_create_provider_uses_codex_when_enabled(monkeypatch):
    cfg = Config()
    cfg.providers.codex.enabled = True

    sentinel = object()

    def _fake_ctor(**kwargs):
        return SimpleNamespace(kind="codex", kwargs=kwargs, sentinel=sentinel)

    monkeypatch.setattr("nanobot.providers.factory.CodexSDKProvider", _fake_ctor)

    provider = create_provider(cfg)
    assert provider.kind == "codex"
    assert provider.sentinel is sentinel
    assert provider.kwargs["default_model"] == cfg.agents.defaults.model


def test_create_provider_uses_litellm_when_codex_disabled():
    cfg = Config()
    cfg.providers.openrouter.api_key = "sk-or-test"
    cfg.providers.codex.enabled = False

    provider = create_provider(cfg)
    assert isinstance(provider, LiteLLMProvider)


def test_create_provider_codex_missing_session_raises(monkeypatch):
    cfg = Config()
    cfg.providers.codex.enabled = True

    class _FakeSDK:
        @staticmethod
        def chat(**kwargs):
            return {"content": "ok"}

    monkeypatch.setattr("nanobot.providers.codex_transport.CodexTransport._load_sdk", lambda self: _FakeSDK())
    monkeypatch.setattr(
        "nanobot.providers.codex_transport.CodexTransport.validate_session",
        lambda self: (False, "No Codex session detected"),
    )

    with pytest.raises(RuntimeError, match="Codex session is unavailable"):
        create_provider(cfg)
