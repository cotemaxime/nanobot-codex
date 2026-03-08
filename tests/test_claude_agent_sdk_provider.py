from __future__ import annotations

import types
import importlib

import pytest

from nanobot.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider
from nanobot.providers.base import LLMResponse, ToolCallRequest


def test_claude_agent_normalize_model_name():
    assert ClaudeAgentSDKProvider._normalize_model_name("claude-agent/claude-sonnet-4-5") == "claude-sonnet-4-5"
    assert ClaudeAgentSDKProvider._normalize_model_name("claude-sonnet-4-5") == "claude-sonnet-4-5"


def test_claude_agent_parse_response_payload_with_tool_calls():
    raw = (
        '{"content":null,"tool_calls":[{"id":"c1","name":"read_file",'
        '"arguments":"{\\"path\\":\\"/tmp/x.txt\\"}"}],'
        '"finish_reason":"tool_calls","reasoning_content":null}'
    )
    resp = ClaudeAgentSDKProvider._parse_response_payload(raw)
    assert resp.has_tool_calls
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments["path"] == "/tmp/x.txt"


def test_claude_agent_parse_response_strips_markdown_fences():
    raw = (
        '```json\n'
        '{"content":"hello","tool_calls":[],"finish_reason":"stop","reasoning_content":null}\n'
        '```'
    )
    resp = ClaudeAgentSDKProvider._parse_response_payload(raw)
    assert resp.content == "hello"
    assert not resp.has_tool_calls


def test_claude_agent_parse_response_extracts_json_from_prose():
    raw = (
        'Here is my response:\n'
        '{"content":null,"tool_calls":[{"id":"t1","name":"exec",'
        '"arguments":"{\\"command\\":\\"ls\\"}"}],'
        '"finish_reason":"tool_calls","reasoning_content":null}\n'
        'Hope that helps!'
    )
    resp = ClaudeAgentSDKProvider._parse_response_payload(raw)
    assert resp.has_tool_calls
    assert resp.tool_calls[0].name == "exec"
    assert resp.tool_calls[0].arguments["command"] == "ls"


def test_claude_agent_parse_response_tool_calls_with_dict_arguments():
    raw = (
        '{"content":null,"tool_calls":[{"id":"c1","name":"exec",'
        '"arguments":{"command":"echo hi"}}],'
        '"finish_reason":"tool_calls","reasoning_content":null}'
    )
    resp = ClaudeAgentSDKProvider._parse_response_payload(raw)
    assert resp.has_tool_calls
    assert resp.tool_calls[0].arguments["command"] == "echo hi"


def test_claude_agent_load_sdk_uses_claude_agent_sdk_module(monkeypatch):
    called = {}

    def _fake_import(name: str):
        called["name"] = name
        return types.SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    sdk = ClaudeAgentSDKProvider._load_sdk()

    assert isinstance(sdk, types.SimpleNamespace)
    assert called["name"] == "claude_agent_sdk"


@pytest.mark.asyncio
async def test_claude_provider_delegates_non_native_tool_calls(monkeypatch):
    monkeypatch.setattr(ClaudeAgentSDKProvider, "_load_sdk", staticmethod(lambda: types.SimpleNamespace()))
    monkeypatch.setattr(ClaudeAgentSDKProvider, "_probe_session", staticmethod(lambda: (True, "ok")))

    provider = ClaudeAgentSDKProvider(default_model="claude-agent/claude-sonnet-4-5")

    async def _query_once(*_args, **_kwargs):
        return LLMResponse(
            content="need tool",
            tool_calls=[ToolCallRequest(id="c1", name="read_file", arguments={"path": "/tmp/a"})],
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(provider, "_query_once", _query_once)

    response = await provider.chat(messages=[{"role": "user", "content": "read it"}], tools=[])
    assert response.has_tool_calls
    assert response.tool_calls[0].name == "read_file"


@pytest.mark.asyncio
async def test_claude_provider_executes_native_tools_internally(monkeypatch):
    monkeypatch.setattr(ClaudeAgentSDKProvider, "_load_sdk", staticmethod(lambda: types.SimpleNamespace()))
    monkeypatch.setattr(ClaudeAgentSDKProvider, "_probe_session", staticmethod(lambda: (True, "ok")))

    provider = ClaudeAgentSDKProvider(default_model="claude-agent/claude-sonnet-4-5")
    calls = {"query": 0, "native": 0}

    async def _query_once(*_args, **_kwargs):
        calls["query"] += 1
        if calls["query"] == 1:
            return LLMResponse(
                content="run exec",
                tool_calls=[ToolCallRequest(id="n1", name="exec", arguments={"command": "echo hi"})],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="done", finish_reason="stop")

    async def _run_native(_tc):
        calls["native"] += 1
        return "ok"

    monkeypatch.setattr(provider, "_query_once", _query_once)
    monkeypatch.setattr(provider, "_run_native_tool", _run_native)

    response = await provider.chat(
        messages=[{"role": "user", "content": "run"}],
        tools=[{"type": "function", "function": {"name": "exec", "parameters": {"type": "object"}}}],
    )
    assert response.content == "done"
    assert not response.has_tool_calls
    assert calls["query"] == 2
    assert calls["native"] == 1
