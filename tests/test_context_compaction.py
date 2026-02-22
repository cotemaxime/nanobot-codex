import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
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


@pytest.mark.asyncio
async def test_compact_command_reduces_session_messages(tmp_path):
    manager = InMemorySessionManager()
    key = "cli:compact"
    session = manager.get_or_create(key)
    for i in range(30):
        role = "user" if i % 2 == 0 else "assistant"
        session.add_message(role, f"message-{i}")
    manager.save(session)

    provider = ScriptedProvider(
        default_model="test/default",
        responses=[LLMResponse(content="compact summary")],
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
        memory_window=20,
    )

    async def _fake_consolidate(_session, archive_all=False, force=False):
        return None

    loop._consolidate_memory = _fake_consolidate  # type: ignore[method-assign]

    response = await loop.process_direct("/compact", session_key=key, channel="cli", chat_id="compact")

    assert "Context compacted successfully" in response
    updated = manager.get_or_create(key)
    assert len(updated.messages) == 13  # summary + keep_recent (max(12, memory_window//2))
    assert updated.messages[0].get("compaction_summary") is True


@pytest.mark.asyncio
async def test_context_warning_sets_pending_compact_action(tmp_path):
    manager = InMemorySessionManager()
    provider = ScriptedProvider(
        default_model="test/default",
        responses=[LLMResponse(content="normal reply")],
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=manager,
        model="test/default",
        default_context_limit_tokens=100,
        context_warning_threshold=0.75,
    )

    text = "x" * 50000
    response = await loop.process_direct(text, session_key="cli:warn", channel="cli", chat_id="warn")

    assert "[Context warning]" in response
    session = manager.get_or_create("cli:warn")
    assert session.metadata.get("pending_action") == "confirm_compact"
    assert session.metadata.get("context_compact_warning") == "pending"
