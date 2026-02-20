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


def test_codex_model_with_worker_registers_codex_web_search_tool(tmp_path):
    class CodexSDKProvider(ScriptedProvider):
        pass

    provider = ScriptedProvider(default_model="openai-codex/gpt-5.2")
    worker = CodexSDKProvider(default_model="gpt-5.3-codex")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="openai-codex/gpt-5.2",
        subagent_provider=worker,
    )

    assert loop.tools.has("web_search")
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


def test_spawn_bridge_mode_hides_direct_tools_for_planner(tmp_path):
    provider = ScriptedProvider(default_model="openai-codex/gpt-5.2")
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="openai-codex/gpt-5.2",
        spawn_bridge_mode=True,
    )

    assert loop.tools.has("spawn")
    assert loop.tools.has("message")
    assert loop.tools.has("set_model")
    assert not loop.tools.has("read_file")
    assert not loop.tools.has("write_file")
    assert not loop.tools.has("edit_file")
    assert not loop.tools.has("list_dir")
    assert not loop.tools.has("exec")
    assert not loop.tools.has("web_search")
    assert not loop.tools.has("web_fetch")
    assert not loop.tools.has("cron")


@pytest.mark.asyncio
async def test_spawn_bridge_mode_retries_when_first_reply_is_refusal(tmp_path):
    provider = ScriptedProvider(
        default_model="openai-codex/gpt-5.2",
        responses=[
            LLMResponse(content="I can't browse from this environment."),
            LLMResponse(
                content="Delegating now",
                tool_calls=[
                    ToolCallRequest(
                        id="s1",
                        name="spawn",
                        arguments={"task": "research quick terminal size", "label": "ghostty-search"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="Spawned a worker and will report back."),
        ],
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="openai-codex/gpt-5.2",
        spawn_bridge_mode=True,
        subagent_provider=ReplayProvider("subagent ok"),
    )

    response = await loop.process_direct(
        "Please search ghostty docs for quick terminal persistence.",
        session_key="cli:spawn-retry",
        channel="cli",
        chat_id="spawn-retry",
    )

    assert "Spawned a worker" in response


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

    async def _progress(text: str) -> None:
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
    assert progress_updates[0].startswith("Running: read_file(")
    assert "retrying now" not in progress_updates[0].lower()
    assert loop.bus.outbound_size == 0


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

    progress = await bus.consume_outbound()
    assert progress.channel == "telegram"
    assert progress.chat_id == "chat"
    assert progress.content.startswith("[progress] Running: read_file(")
    assert progress.metadata.get("progress") is True


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


def test_system_prompt_includes_execution_honesty_guardrails(tmp_path):
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider(),
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="test/default",
    )

    prompt = loop.context.build_system_prompt()
    assert "Never claim you executed commands" in prompt
    assert "Use the spawn tool only when the user explicitly asks" in prompt


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
async def test_topic_command_response_preserves_telegram_thread_metadata(tmp_path):
    manager = InMemorySessionManager()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider(),
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )

    inbound = InboundMessage(
        channel="telegram",
        sender_id="1|alice",
        chat_id="1234",
        content="/model",
        metadata={
            "telegram": {"message_thread_id": 99},
            "session_key": "telegram:1234:99",
        },
    )

    response = await loop._process_message(inbound)

    assert response is not None
    assert response.chat_id == "1234"
    assert response.metadata.get("telegram", {}).get("message_thread_id") == 99
    assert "Current model for this chat/topic" in response.content


@pytest.mark.asyncio
async def test_model_selection_rejects_non_numeric_input(tmp_path):
    manager = InMemorySessionManager()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider(),
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )

    await loop.process_direct("/model", session_key="telegram:chat", channel="telegram", chat_id="chat")
    result = await loop.process_direct(
        "is it working?",
        session_key="telegram:chat",
        channel="telegram",
        chat_id="chat",
    )

    assert "Invalid selection. Send a model number only" in result
    session = manager.get_or_create("telegram:chat")
    assert session.metadata.get("model_override") is None
    assert session.metadata.get("pending_action") == "set_model"


@pytest.mark.asyncio
async def test_skill_selection_rejects_non_numeric_input(tmp_path):
    manager = InMemorySessionManager()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=ScriptedProvider(),
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
    )
    loop.context.skills.list_skills = lambda: [{"name": "search"}, {"name": "git"}]  # type: ignore[method-assign]

    await loop.process_direct("/skill", session_key="telegram:chat", channel="telegram", chat_id="chat")
    result = await loop.process_direct(
        "search",
        session_key="telegram:chat",
        channel="telegram",
        chat_id="chat",
    )

    assert "Invalid selection. Send skill numbers only" in result
    session = manager.get_or_create("telegram:chat")
    assert session.metadata.get("active_skills") is None
    assert session.metadata.get("pending_action") == "set_skills"


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
    consolidate_calls = []

    async def _fake_consolidate(s, archive_all=False, force=False):
        consolidate_calls.append((s.key, archive_all, force))

    loop._consolidate_memory = _fake_consolidate  # type: ignore[method-assign]

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
    await asyncio.sleep(0)
    assert consolidate_calls
    assert consolidate_calls[-1] == ("telegram:chat", False, True)
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
