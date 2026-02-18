from nanobot.config.schema import Config
from nanobot.providers.custom_provider import CustomProvider
from nanobot.providers.factory import create_provider
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import OpenAICodexProvider


def test_create_provider_uses_openai_codex_for_oauth_model():
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5-codex"

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
