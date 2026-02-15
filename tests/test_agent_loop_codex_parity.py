import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.codex_sdk_provider import CodexSDKProvider
from nanobot.providers.codex_transport import TransportResponse, TransportToolCall
from nanobot.session.manager import Session


class FakeTransport:
    def __init__(self, profile=None, workspace=None, timeout_seconds=180, skip_git_repo_check=True):
        self.responses = []

    def validate_session(self):
        return True, "ok"

    async def chat(self, messages, tools, model, max_tokens, temperature):
        if self.responses:
            nxt = self.responses.pop(0)
            if callable(nxt):
                return nxt(messages, tools, model)
            return nxt
        return TransportResponse(content="ok")


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
async def test_agent_loop_standard_response_with_codex_provider(monkeypatch, tmp_path):
    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexTransport", FakeTransport)

    provider = CodexSDKProvider(default_model="codex/default", workspace=str(tmp_path))
    provider.transport.responses = [TransportResponse(content="hello from codex")]

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="codex/default",
    )

    response = await loop.process_direct("hi", session_key="cli:test", channel="cli", chat_id="test")
    assert "hello from codex" in response


@pytest.mark.asyncio
async def test_agent_loop_delegated_tool_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexTransport", FakeTransport)

    f = tmp_path / "note.txt"
    f.write_text("tool output", encoding="utf-8")

    provider = CodexSDKProvider(default_model="codex/default", workspace=str(tmp_path))

    provider.transport.responses = [
        TransportResponse(
            content="Reading file",
            tool_calls=[TransportToolCall(id="r1", name="read_file", arguments={"path": str(f)})],
            finish_reason="tool_calls",
        ),
        TransportResponse(content="done with delegated tool"),
    ]

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="codex/default",
    )

    response = await loop.process_direct("read the file", session_key="cli:test2", channel="cli", chat_id="test2")
    assert "done with delegated tool" in response


@pytest.mark.asyncio
async def test_memory_consolidation_json_contract_still_works(monkeypatch, tmp_path):
    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexTransport", FakeTransport)

    provider = CodexSDKProvider(default_model="codex/default", workspace=str(tmp_path))
    provider.transport.responses = [
        TransportResponse(
            content='{"history_entry":"[2026-02-15 10:00] Discussed codex provider.",' \
                    '"memory_update":"Prefers Codex subscription mode."}'
        )
    ]

    session = Session(key="cli:mem")
    session.add_message("user", "Please remember I prefer codex mode")
    session.add_message("assistant", "Noted")

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        session_manager=InMemorySessionManager(),
        model="codex/default",
    )

    await loop._consolidate_memory(session, archive_all=True)

    memory_file = tmp_path / "memory" / "MEMORY.md"
    history_file = tmp_path / "memory" / "HISTORY.md"

    assert memory_file.exists()
    assert history_file.exists()
    assert "Prefers Codex subscription mode." in memory_file.read_text(encoding="utf-8")
    assert "Discussed codex provider" in history_file.read_text(encoding="utf-8")
