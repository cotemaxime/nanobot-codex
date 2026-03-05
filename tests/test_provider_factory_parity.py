from nanobot.config.schema import Config
from nanobot.providers.factory import create_provider


def test_factory_claude_provider_uses_claude_worker_config(monkeypatch, tmp_path):
    captured = {}

    class StubClaudeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "nanobot.providers.claude_agent_sdk_provider.ClaudeAgentSDKProvider",
        StubClaudeProvider,
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.ClaudeAgentSDKProvider",
        StubClaudeProvider,
    )

    config = Config()
    config.agents.defaults.workspace = str(tmp_path)
    config.agents.defaults.model = "claude-agent/claude-sonnet-4-5"
    config.agents.defaults.max_tool_iterations = 17
    config.agents.claude_worker.model = "claude-opus-4-1"
    config.agents.claude_worker.timeout_seconds = 789
    config.agents.claude_worker.max_turns = 6
    config.agents.claude_worker.max_internal_native_steps = 4
    config.agents.claude_worker.permission_mode = "bypassPermissions"
    config.agents.claude_worker.strict_auth = True
    config.agents.claude_worker.diagnostic_logging = True

    provider = create_provider(config)

    assert isinstance(provider, StubClaudeProvider)
    assert captured["default_model"] == "claude-sonnet-4-5"
    assert captured["timeout_seconds"] == 789
    assert captured["max_turns"] == 6
    assert captured["max_internal_native_steps"] == 4
    assert captured["permission_mode"] == "bypassPermissions"
    assert captured["strict_auth"] is True
    assert captured["diagnostic_logging"] is True
