import pytest

from nanobot.providers.codex_sdk_provider import CodexSDKProvider
from nanobot.providers.codex_transport import TransportResponse, TransportToolCall


class FakeTransport:
    def __init__(
        self,
        profile=None,
        workspace=None,
        timeout_seconds=180,
        skip_git_repo_check=True,
        sandbox_mode="workspace-write",
        approval_policy="never",
        network_access_enabled=True,
        web_search_enabled=True,
    ):
        self.calls = []
        self.responses = []

    def validate_session(self):
        return True, "ok"

    async def chat(self, messages, tools, model, max_tokens, temperature):
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        if not self.responses:
            return TransportResponse(content="done")
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_native_tools_are_executed_internally(monkeypatch):
    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexTransport", FakeTransport)

    provider = CodexSDKProvider(default_model="codex/default")
    provider.transport.responses = [
        TransportResponse(
            content=None,
            tool_calls=[
                TransportToolCall(id="1", name="exec", arguments={"command": "printf codex"}),
            ],
            finish_reason="tool_calls",
        ),
        TransportResponse(content="final answer", finish_reason="stop"),
    ]

    response = await provider.chat(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "run cmd",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
                },
            }
        ],
    )

    assert response.content == "final answer"
    assert response.tool_calls == []
    assert len(provider.transport.calls) == 2


@pytest.mark.asyncio
async def test_mixed_batches_delegate_all_tools_to_nanobot(monkeypatch):
    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexTransport", FakeTransport)

    provider = CodexSDKProvider(default_model="codex/default")
    provider.transport.responses = [
        TransportResponse(
            content="Need external tool",
            tool_calls=[
                TransportToolCall(id="1", name="exec", arguments={"command": "printf ignored"}),
                TransportToolCall(id="2", name="message", arguments={"content": "hello"}),
            ],
            finish_reason="tool_calls",
        )
    ]

    response = await provider.chat(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=[
            {"type": "function", "function": {"name": "exec", "description": "", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "message", "description": "", "parameters": {"type": "object"}}},
        ],
    )

    assert response.content == "Need external tool"
    assert [tc.name for tc in response.tool_calls] == ["message", "exec"]
    assert len(provider.transport.calls) == 1


@pytest.mark.asyncio
async def test_internal_native_step_limit(monkeypatch):
    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexTransport", FakeTransport)

    provider = CodexSDKProvider(default_model="codex/default", max_internal_native_steps=2)
    provider.transport.responses = [
        TransportResponse(
            content=None,
            tool_calls=[TransportToolCall(id="1", name="exec", arguments={"command": "printf x"})],
            finish_reason="tool_calls",
        ),
        TransportResponse(
            content=None,
            tool_calls=[TransportToolCall(id="2", name="exec", arguments={"command": "printf y"})],
            finish_reason="tool_calls",
        ),
        TransportResponse(
            content=None,
            tool_calls=[TransportToolCall(id="3", name="exec", arguments={"command": "printf z"})],
            finish_reason="tool_calls",
        ),
    ]

    response = await provider.chat(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "exec", "description": "", "parameters": {"type": "object"}}}],
    )

    assert response.finish_reason == "error"
    assert "exceeded internal native-tool step limit" in (response.content or "")
