from nanobot.config.schema import Config
from nanobot.providers.codex_sdk_provider import CodexSDKProvider
from nanobot.providers.custom_provider import CustomProvider
from nanobot.providers.factory import create_provider
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import OpenAICodexProvider


def test_create_provider_prefers_codex_sdk_for_oauth_model(monkeypatch):
    monkeypatch.setattr(
        "nanobot.providers.codex_sdk_provider.CodexTransport.validate_session",
        lambda self: (True, "ok"),
    )
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5-codex"

    provider = create_provider(cfg)
    assert isinstance(provider, CodexSDKProvider)


def test_create_provider_falls_back_to_openai_codex_when_sdk_unavailable(monkeypatch):
    monkeypatch.setattr(
        "nanobot.providers.factory.CodexSDKProvider",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sdk unavailable")),
    )
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5-codex"

    provider = create_provider(cfg)
    assert isinstance(provider, OpenAICodexProvider)


def test_create_provider_routes_gpt52_to_openai_codex_provider(monkeypatch):
    monkeypatch.setattr(
        "nanobot.providers.factory.CodexSDKProvider",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5.2"

    provider = create_provider(cfg)
    assert isinstance(provider, OpenAICodexProvider)


def test_create_provider_uses_litellm_for_api_key_provider():
    cfg = Config()
    cfg.providers.openrouter.api_key = "sk-or-test"

    provider = create_provider(cfg)
    assert isinstance(provider, LiteLLMProvider)


def test_create_provider_uses_custom_for_custom_model():
    cfg = Config()
    cfg.agents.defaults.model = "custom/my-model"
    cfg.providers.custom.api_key = "placeholder"
    cfg.providers.custom.api_base = "http://localhost:9000/v1"

    provider = create_provider(cfg)
    assert isinstance(provider, CustomProvider)
