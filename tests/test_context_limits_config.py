from nanobot.cli.commands import _build_context_limits
from nanobot.config.schema import Config


def test_build_context_limits_main_and_cron_models():
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5-codex"
    cfg.agents.defaults.cron_model = "openai-codex/gpt-5.1-codex"
    cfg.agents.defaults.context_warning_threshold = 0.8
    cfg.agents.defaults.model_context_limit_tokens = 200000
    cfg.agents.defaults.cron_context_limit_tokens = 64000

    threshold, default_limit, limits = _build_context_limits(cfg)

    assert threshold == 0.8
    assert default_limit == 200000
    assert limits["openai-codex/gpt-5-codex"] == 200000
    assert limits["openai-codex/gpt-5.1-codex"] == 64000


def test_build_context_limits_defaults_cron_to_main_limit():
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5-codex"
    cfg.agents.defaults.cron_model = "openai-codex/gpt-5.1-codex"
    cfg.agents.defaults.model_context_limit_tokens = 150000
    cfg.agents.defaults.cron_context_limit_tokens = None

    threshold, default_limit, limits = _build_context_limits(cfg)

    assert threshold == 0.75
    assert default_limit == 150000
    assert limits["openai-codex/gpt-5-codex"] == 150000
    assert limits["openai-codex/gpt-5.1-codex"] == 150000
