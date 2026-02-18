import asyncio
from types import SimpleNamespace

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.session.manager import Session


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


class EchoModelProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "openai-codex/gpt-5.1-codex"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content=f"model={model}")


class DummySpawnManager:
    def __init__(self):
        self.calls = []

    async def spawn(self, task: str, label: str | None, origin_channel: str, origin_chat_id: str) -> str:
        self.calls.append(
            {
                "task": task,
                "label": label,
                "origin_channel": origin_channel,
                "origin_chat_id": origin_chat_id,
            }
        )
        return "spawned"


class DummyCronService:
    def __init__(self):
        self.calls = []

    def add_job(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(name=kwargs["name"], id="job-1")

    def list_jobs(self):
        return []

    def remove_job(self, _job_id: str):
        return True


class DummyChannel:
    def __init__(self):
        self.sent: list[OutboundMessage] = []

    async def send(self, message: OutboundMessage) -> None:
        self.sent.append(message)


@pytest.mark.asyncio
async def test_model_override_is_scoped_by_topic_session_key(tmp_path):
    manager = InMemorySessionManager()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=EchoModelProvider(),
        workspace=tmp_path,
        session_manager=manager,
        model="openai-codex/gpt-5.1-codex",
    )

    topic_a = "telegram:42:topic-a"
    topic_b = "telegram:42:topic-b"

    await loop.process_direct("/model", session_key=topic_a, channel="telegram", chat_id="42")
    await loop.process_direct(
        "openai-codex/gpt-5-codex-mini",
        session_key=topic_a,
        channel="telegram",
        chat_id="42",
    )

    await loop.process_direct("/model", session_key=topic_b, channel="telegram", chat_id="42")
    await loop.process_direct(
        "openai-codex/gpt-5-codex",
        session_key=topic_b,
        channel="telegram",
        chat_id="42",
    )

    resp_a = await loop.process_direct("run", session_key=topic_a, channel="telegram", chat_id="42")
    resp_b = await loop.process_direct("run", session_key=topic_b, channel="telegram", chat_id="42")

    assert "openai-codex/gpt-5-codex-mini" in resp_a
    assert "openai-codex/gpt-5-codex" in resp_b
    assert manager.get_or_create(topic_a).metadata.get("model_override") == "openai-codex/gpt-5-codex-mini"
    assert manager.get_or_create(topic_b).metadata.get("model_override") == "openai-codex/gpt-5-codex"


@pytest.mark.asyncio
async def test_help_command_lists_kept_chat_commands(tmp_path):
    manager = InMemorySessionManager()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=EchoModelProvider(),
        workspace=tmp_path,
        session_manager=manager,
        model="openai-codex/gpt-5.1-codex",
    )

    result = await loop.process_direct("/help", session_key="telegram:42", channel="telegram", chat_id="42")

    assert "/new" in result
    assert "/last" in result
    assert "/skills" in result
    assert "/skill" in result
    assert "/model" in result


@pytest.mark.asyncio
async def test_skill_selection_is_scoped_by_topic_session_key(tmp_path):
    manager = InMemorySessionManager()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=EchoModelProvider(),
        workspace=tmp_path,
        session_manager=manager,
        model="openai-codex/gpt-5.1-codex",
    )
    loop.context.skills.list_skills = lambda: [{"name": "search"}, {"name": "git"}]  # type: ignore[method-assign]

    topic_a = "telegram:42:topic-a"
    topic_b = "telegram:42:topic-b"

    await loop.process_direct("/skill", session_key=topic_a, channel="telegram", chat_id="42")
    await loop.process_direct("1", session_key=topic_a, channel="telegram", chat_id="42")
    await loop.process_direct("/skill", session_key=topic_b, channel="telegram", chat_id="42")
    await loop.process_direct("2", session_key=topic_b, channel="telegram", chat_id="42")

    assert manager.get_or_create(topic_a).metadata.get("active_skills") == ["git"]
    assert manager.get_or_create(topic_b).metadata.get("active_skills") == ["search"]


@pytest.mark.asyncio
async def test_message_tool_uses_runtime_context_when_no_explicit_target():
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(
        send_callback=_send,
        context_getter=lambda: ("telegram", "chat-ctx"),
    )

    result = await tool.execute("hello")

    assert result == "Message sent to telegram:chat-ctx"
    assert len(sent) == 1
    assert sent[0].channel == "telegram"
    assert sent[0].chat_id == "chat-ctx"
    assert sent[0].content == "hello"


@pytest.mark.asyncio
async def test_spawn_tool_uses_runtime_context_when_no_explicit_target():
    manager = DummySpawnManager()
    tool = SpawnTool(manager=manager, context_getter=lambda: ("telegram", "chat-ctx"))

    result = await tool.execute(task="analyze", label="analysis")

    assert result == "spawned"
    assert len(manager.calls) == 1
    assert manager.calls[0]["origin_channel"] == "telegram"
    assert manager.calls[0]["origin_chat_id"] == "chat-ctx"


@pytest.mark.asyncio
async def test_cron_tool_uses_runtime_context_when_no_explicit_target():
    cron = DummyCronService()
    tool = CronTool(cron_service=cron, context_getter=lambda: ("telegram", "chat-ctx"))

    result = await tool.execute(action="add", message="Reminder", every_seconds=60)

    assert "Created job" in result
    assert len(cron.calls) == 1
    assert cron.calls[0]["channel"] == "telegram"
    assert cron.calls[0]["to"] == "chat-ctx"


@pytest.mark.asyncio
async def test_channel_manager_uses_per_channel_outbound_workers():
    manager = ChannelManager(config=Config(), bus=MessageBus())
    telegram = DummyChannel()
    discord = DummyChannel()
    manager.channels = {
        "telegram": telegram,
        "discord": discord,
    }

    queue_tg = manager._outbound_queues.setdefault("telegram", asyncio.Queue())
    queue_ds = manager._outbound_queues.setdefault("discord", asyncio.Queue())
    await queue_tg.put(OutboundMessage(channel="telegram", chat_id="1", content="t1"))
    await queue_tg.put(OutboundMessage(channel="telegram", chat_id="1", content="t2"))
    await queue_ds.put(OutboundMessage(channel="discord", chat_id="2", content="d1"))

    manager._ensure_outbound_worker("telegram")
    manager._ensure_outbound_worker("discord")
    await asyncio.gather(*manager._outbound_workers.values())

    assert [m.content for m in telegram.sent] == ["t1", "t2"]
    assert [m.content for m in discord.sent] == ["d1"]
    assert manager._outbound_workers == {}
