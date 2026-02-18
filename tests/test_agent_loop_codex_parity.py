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

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
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

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
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

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content=self.reply)


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


@pytest.mark.asyncio
async def test_memory_consolidation_json_contract_still_works(tmp_path):
    provider = ScriptedProvider(
        responses=[
            LLMResponse(
                content='{"history_entry":"[2026-02-15 10:00] Discussed provider setup.",' \
                        '"memory_update":"Prefers OAuth Codex mode."}'
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
async def test_agent_can_switch_model_via_set_model_tool(tmp_path):
    provider = ScriptedProvider(
        default_model="openai-codex/gpt-5-codex-mini",
        responses=[
            LLMResponse(
                content="Switching model",
                tool_calls=[
                    ToolCallRequest(
                        id="m1",
                        name="set_model",
                        arguments={
                            "action": "set",
                            "model": "openai-codex/gpt-5.1-codex",
                            "persist": True,
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            lambda _messages, _tools, model: LLMResponse(content=f"final via {model}"),
            lambda _messages, _tools, model: LLMResponse(content=f"next turn via {model}"),
        ],
    )

    manager = InMemorySessionManager()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=manager,
        model="openai-codex/gpt-5-codex-mini",
    )

    first = await loop.process_direct("handle complex coding task", session_key="cli:model", channel="cli", chat_id="model")
    assert "final via openai-codex/gpt-5.1-codex" in first

    session = manager.get_or_create("cli:model")
    assert session.metadata.get("model_override") == "openai-codex/gpt-5.1-codex"

    second = await loop.process_direct("follow-up", session_key="cli:model", channel="cli", chat_id="model")
    assert "next turn via openai-codex/gpt-5.1-codex" in second


@pytest.mark.asyncio
async def test_last_command_resends_previous_assistant_message(tmp_path):
    manager = InMemorySessionManager()
    session = manager.get_or_create("cli:last")
    session.add_message("user", "first question")
    session.add_message("assistant", "first answer")
    session.add_message("user", "second question")
    session.add_message("assistant", "second answer")
    manager.save(session)

    provider = ScriptedProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )

    result = await loop.process_direct("/last", session_key="cli:last", channel="cli", chat_id="last")
    assert result == "second answer"


@pytest.mark.asyncio
async def test_new_command_with_bot_mention_resets_without_model_call(tmp_path):
    manager = InMemorySessionManager()
    session = manager.get_or_create("telegram:chat")
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    manager.save(session)

    provider = ScriptedProvider()

    async def _should_not_call_model(*args, **kwargs):
        raise AssertionError("model should not be called for /new command")

    provider.chat = _should_not_call_model  # type: ignore[assignment]

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )

    result = await loop.process_direct("/new@nanobot", session_key="telegram:chat", channel="telegram", chat_id="chat")
    assert "New session started" in result
    assert manager.get_or_create("telegram:chat").messages == []


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

        assert first.chat_id == "fast-chat"
        assert first.content == "fast done"
        assert second.chat_id == "slow-chat"
        assert second.content == "slow done"
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=3.0)


@pytest.mark.asyncio
async def test_telegram_reaction_thumbs_up_marks_completed(tmp_path):
    manager = InMemorySessionManager()
    session = manager.get_or_create("telegram:chat")
    session.add_message("user", "finish item A")
    session.add_message("assistant", "Done item A")
    manager.save(session)

    loop = AgentLoop(
        bus=MessageBus(),
        provider=ReplayProvider("unused"),
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )

    response = await loop._process_message(InboundMessage(
        channel="telegram",
        sender_id="1",
        chat_id="chat",
        content="[telegram_reaction] user reacted on message 10: (none) -> 👍",
        metadata={
            "event_type": "telegram_message_reaction",
            "message_id": 10,
            "telegram": {"reaction_new": ["👍"], "reaction_old": []},
        },
    ))

    assert response is not None
    assert "Marked this as completed" in response.content
    updated = manager.get_or_create("telegram:chat")
    approvals = updated.metadata.get("approved_events")
    assert isinstance(approvals, list) and approvals
    assert approvals[-1]["message_id"] == 10


@pytest.mark.asyncio
async def test_telegram_reaction_redo_resends_last_assistant(tmp_path):
    manager = InMemorySessionManager()
    session = manager.get_or_create("telegram:chat")
    session.add_message("user", "question")
    session.add_message("assistant", "last answer")
    manager.save(session)

    loop = AgentLoop(
        bus=MessageBus(),
        provider=ReplayProvider("unused"),
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )

    response = await loop._process_message(InboundMessage(
        channel="telegram",
        sender_id="1",
        chat_id="chat",
        content="[telegram_reaction] user reacted on message 11: (none) -> ♻️",
        metadata={
            "event_type": "telegram_message_reaction",
            "message_id": 11,
            "telegram": {"reaction_new": ["♻️"], "reaction_old": []},
        },
    ))

    assert response is not None
    assert response.content == "last answer"


@pytest.mark.asyncio
async def test_telegram_reaction_retry_replays_last_user_request(tmp_path):
    manager = InMemorySessionManager()
    session = manager.get_or_create("telegram:chat")
    session.add_message("user", "please do it again")
    session.add_message("assistant", "old answer")
    manager.save(session)

    loop = AgentLoop(
        bus=MessageBus(),
        provider=ReplayProvider("new retried answer"),
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )

    response = await loop._process_message(InboundMessage(
        channel="telegram",
        sender_id="1",
        chat_id="chat",
        content="[telegram_reaction] user reacted on message 12: (none) -> 🔁",
        metadata={
            "event_type": "telegram_message_reaction",
            "message_id": 12,
            "telegram": {"reaction_new": ["🔁"], "reaction_old": []},
        },
    ))

    assert response is not None
    assert response.content == "new retried answer"
