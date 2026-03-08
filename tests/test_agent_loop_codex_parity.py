import asyncio

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import Session


class ScriptedProvider(LLMProvider):
    def __init__(self, default_model: str = "test/default", responses=None):
        super().__init__()
        self.default_model = default_model
        self.responses = list(responses or [])

    def get_default_model(self) -> str:
        return self.default_model

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, reasoning_effort=None):
        if self.responses:
            nxt = self.responses.pop(0)
            if callable(nxt):
                result = nxt(messages, tools, model)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            return nxt
        return LLMResponse(content="ok")


class InMemorySessionManager:
    def __init__(self):
        self._sessions = {}

    def get_or_create(self, key: str):
        if key not in self._sessions:
            self._sessions[key] = Session(key=key)
        return self._sessions[key]

    def save(self, session: Session):
        self._sessions[session.key] = session

    def invalidate(self, key: str):
        self._sessions.pop(key, None)


class SlowFastProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "test/default"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, reasoning_effort=None):
        prompt = (messages[-1].get("content") if messages else "") or ""
        if "slow" in prompt:
            await asyncio.sleep(0.2)
            return LLMResponse(content="slow done")
        await asyncio.sleep(0.01)
        return LLMResponse(content="fast done")


class ReplayProvider(LLMProvider):
    def __init__(self, reply: str):
        super().__init__()
        self.reply = reply

    def get_default_model(self) -> str:
        return "test/default"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7, reasoning_effort=None):
        return LLMResponse(content=self.reply)


def test_codex_model_does_not_register_nanobot_web_tools(tmp_path):
    provider = ScriptedProvider(default_model="openai-codex/gpt-5.2")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="openai-codex/gpt-5.2",
    )

    assert not loop.tools.has("web_search")
    assert loop.tools.has("web_fetch")


def test_non_codex_model_registers_nanobot_web_tools(tmp_path):
    provider = ScriptedProvider(default_model="anthropic/claude-3-5-haiku")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="anthropic/claude-3-5-haiku",
    )

    assert loop.tools.has("web_search")
    assert loop.tools.has("web_fetch")


@pytest.mark.asyncio
async def test_agent_loop_standard_response(tmp_path):
    provider = ScriptedProvider(
        default_model="openai-codex/gpt-5.1-codex",
        responses=[LLMResponse(content="hello from provider")],
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="openai-codex/gpt-5.1-codex",
    )

    response = await loop.process_direct("hi", session_key="cli:test", channel="cli", chat_id="test")
    assert "hello from provider" in response


@pytest.mark.asyncio
async def test_agent_loop_delegated_tool_roundtrip(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("tool output", encoding="utf-8")

    provider = ScriptedProvider(
        responses=[
            LLMResponse(
                content="Reading file",
                tool_calls=[ToolCallRequest(id="r1", name="read_file", arguments={"path": str(f)})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done with delegated tool"),
        ]
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="test/default",
    )

    response = await loop.process_direct("read the file", session_key="cli:test2", channel="cli", chat_id="test2")
    assert "done with delegated tool" in response


@pytest.mark.asyncio
async def test_progress_callback_uses_tool_hint_not_model_narration(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("tool output", encoding="utf-8")

    provider = ScriptedProvider(
        responses=[
            LLMResponse(
                content="I'm retrying now and committing everything.",
                tool_calls=[ToolCallRequest(id="r1", name="read_file", arguments={"path": str(f)})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="test/default",
    )

    progress_updates: list[str] = []

    async def _progress(text: str, **kwargs) -> None:
        progress_updates.append(text)

    response = await loop.process_direct(
        "read the file",
        session_key="cli:test-progress",
        channel="cli",
        chat_id="test-progress",
        on_progress=_progress,
    )

    assert response == "done"
    assert progress_updates
    # The tool hint is now emitted directly (e.g. 'read_file("path")') without "Running:" prefix
    tool_hint_updates = [u for u in progress_updates if "read_file(" in u]
    assert tool_hint_updates
    assert "retrying now" not in tool_hint_updates[0].lower()
    assert loop.bus.outbound_size == 0


@pytest.mark.asyncio
async def test_codex_sdk_progress_callback_emits_thinking_status(tmp_path):
    class CodexSDKProvider(ScriptedProvider):
        pass

    provider = CodexSDKProvider(
        default_model="gpt-5.3-codex",
        responses=[LLMResponse(content="done")],
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="openai-codex/gpt-5.3-codex",
    )

    progress_updates: list[str] = []

    async def _progress(text: str, **kwargs) -> None:
        progress_updates.append(text)

    response = await loop.process_direct(
        "status check",
        session_key="cli:test-codex-progress",
        channel="cli",
        chat_id="test-codex-progress",
        on_progress=_progress,
    )

    assert response == "done"
    assert progress_updates
    assert progress_updates[0].startswith("Thinking with Codex SDK")


@pytest.mark.asyncio
async def test_claude_sdk_progress_callback_emits_thinking_status(tmp_path):
    class ClaudeAgentSDKProvider(ScriptedProvider):
        pass

    provider = ClaudeAgentSDKProvider(
        default_model="claude-agent/claude-sonnet-4-5",
        responses=[LLMResponse(content="done")],
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="claude-agent/claude-sonnet-4-5",
    )

    progress_updates: list[str] = []

    async def _progress(text: str, **kwargs) -> None:
        progress_updates.append(text)

    response = await loop.process_direct(
        "status check",
        session_key="cli:test-claude-progress",
        channel="cli",
        chat_id="test-claude-progress",
        on_progress=_progress,
    )

    assert response == "done"
    assert progress_updates
    assert progress_updates[0].startswith("Thinking with Claude Agent SDK")


@pytest.mark.asyncio
async def test_default_bus_progress_streams_with_explicit_prefix(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("tool output", encoding="utf-8")

    provider = ScriptedProvider(
        responses=[
            LLMResponse(
                content="Working on it",
                tool_calls=[ToolCallRequest(id="r1", name="read_file", arguments={"path": str(f)})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done"),
        ]
    )
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="test/default",
    )

    inbound = InboundMessage(
        channel="telegram",
        sender_id="1",
        chat_id="chat",
        content="read file",
        metadata={"telegram": {"message_thread_id": 99}},
    )
    response = await loop._process_message(inbound)
    assert response is not None
    assert response.content == "done"

    # The bus progress uses _progress and _tool_hint metadata keys
    # and emits the thought first, then the tool hint directly
    progress = await bus.consume_outbound()
    assert progress.channel == "telegram"
    assert progress.chat_id == "chat"
    assert progress.metadata.get("_progress") is True


@pytest.mark.asyncio
async def test_agent_loop_does_not_inject_reflection_user_turn(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("tool output", encoding="utf-8")

    def _second_turn(messages, _tools, _model):
        assert messages[-1]["role"] == "tool"
        assert "Reflect on the results and decide next steps." not in str(messages)
        return LLMResponse(content="final")

    provider = ScriptedProvider(
        responses=[
            LLMResponse(
                content="Reading file",
                tool_calls=[ToolCallRequest(id="r1", name="read_file", arguments={"path": str(f)})],
                finish_reason="tool_calls",
            ),
            _second_turn,
        ]
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="test/default",
    )

    response = await loop.process_direct("read the file", session_key="cli:test-reflect", channel="cli", chat_id="test-reflect")
    assert response == "final"


def test_codex_progress_intervals_ramp_then_hold():
    intervals = AgentLoop._codex_progress_intervals_seconds()
    first_twelve = [next(intervals) for _ in range(12)]
    assert first_twelve == [60, 60, 120, 120, 240, 240, 360, 360, 480, 480, 600, 600]
    assert next(intervals) == 600
    assert next(intervals) == 600


@pytest.mark.asyncio
async def test_memory_consolidation_json_contract_still_works(tmp_path):
    provider = ScriptedProvider(
        responses=[
            LLMResponse(
                content="Consolidating memory",
                tool_calls=[
                    ToolCallRequest(
                        id="mem1",
                        name="save_memory",
                        arguments={
                            "history_entry": "[2026-02-15 10:00] Discussed provider setup.",
                            "memory_update": "Prefers OAuth Codex mode.",
                        },
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )

    session = Session(key="cli:mem")
    session.add_message("user", "Please remember I prefer codex mode")
    session.add_message("assistant", "Noted")

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="test/default",
    )

    await loop._consolidate_memory(session, archive_all=True)

    memory_file = tmp_path / "memory" / "MEMORY.md"
    history_file = tmp_path / "memory" / "HISTORY.md"

    assert memory_file.exists()
    assert history_file.exists()
    assert "Prefers OAuth Codex mode." in memory_file.read_text(encoding="utf-8")
    assert "Discussed provider setup" in history_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_agent_run_processes_sessions_concurrently(tmp_path):
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=SlowFastProvider(),
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="test/default",
    )
    run_task = asyncio.create_task(loop.run())
    try:
        await bus.publish_inbound(InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="slow-chat",
            content="please do slow work",
        ))
        await asyncio.sleep(0.02)
        await bus.publish_inbound(InboundMessage(
            channel="telegram",
            sender_id="u2",
            chat_id="fast-chat",
            content="quick status",
        ))

        first = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        second = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)

        results = {first.chat_id: first.content, second.chat_id: second.content}
        assert results["fast-chat"] == "fast done"
        assert results["slow-chat"] == "slow done"
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=3.0)
