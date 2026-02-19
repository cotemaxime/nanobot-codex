import asyncio

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


class FakeProvider:
    def get_default_model(self) -> str:
        return "openai-codex/gpt-5.1-codex"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content="done")


class GenericProvider:
    def get_default_model(self) -> str:
        return "anthropic/claude-3-5-haiku"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content="done")


class SlowProvider:
    def get_default_model(self) -> str:
        return "anthropic/claude-3-5-haiku"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        import asyncio
        await asyncio.sleep(2.0)
        return LLMResponse(content="done")


def test_subagent_codex_model_does_not_register_nanobot_web_tools(tmp_path):
    manager = SubagentManager(
        provider=FakeProvider(),
        workspace=tmp_path,
        bus=MessageBus(),
    )
    assert manager._should_register_nanobot_web_tools() is False


def test_subagent_non_codex_model_registers_nanobot_web_tools(tmp_path):
    manager = SubagentManager(
        provider=GenericProvider(),
        workspace=tmp_path,
        bus=MessageBus(),
    )
    assert manager._should_register_nanobot_web_tools() is True


@pytest.mark.asyncio
async def test_spawn_sends_immediate_background_notice(tmp_path):
    bus = MessageBus()
    manager = SubagentManager(
        provider=FakeProvider(),
        workspace=tmp_path,
        bus=bus,
    )

    result = await manager.spawn(
        task="long task",
        label="Long Task",
        origin_channel="telegram",
        origin_chat_id="chat-1",
    )

    assert "Background task queued" in result

    outbound = await bus.consume_outbound()
    assert outbound.channel == "telegram"
    assert outbound.chat_id == "chat-1"
    assert "Background task 'Long Task' started" in outbound.content
    assert outbound.metadata.get("progress") is True


@pytest.mark.asyncio
async def test_spawn_preserves_origin_metadata_for_topic_routing(tmp_path):
    bus = MessageBus()
    manager = SubagentManager(
        provider=FakeProvider(),
        workspace=tmp_path,
        bus=bus,
    )

    metadata = {"telegram": {"message_thread_id": 41}, "session_key": "telegram:group_41"}
    await manager.spawn(
        task="long task",
        label="Long Task",
        origin_channel="telegram",
        origin_chat_id="-100123",
        origin_metadata=metadata,
        origin_session_key="telegram:-100123_41",
    )

    outbound = await bus.consume_outbound()
    assert outbound.metadata.get("telegram", {}).get("message_thread_id") == 41
    assert outbound.metadata.get("progress") is True


@pytest.mark.asyncio
async def test_spawn_emits_heartbeat_progress(tmp_path):
    bus = MessageBus()
    manager = SubagentManager(
        provider=SlowProvider(),
        workspace=tmp_path,
        bus=bus,
        heartbeat_interval_seconds=1,
    )

    await manager.spawn(
        task="long task",
        label="Long Task",
        origin_channel="telegram",
        origin_chat_id="chat-1",
    )

    started = await bus.consume_outbound()
    assert "started" in started.content.lower()

    heartbeat = await asyncio.wait_for(bus.consume_outbound(), timeout=3.0)
    assert heartbeat.metadata.get("progress") is True
    assert "still running" in heartbeat.content


def test_heartbeat_delay_schedule_caps_at_ten_minutes(tmp_path):
    manager = SubagentManager(
        provider=GenericProvider(),
        workspace=tmp_path,
        bus=MessageBus(),
    )
    observed = [manager._heartbeat_delay_seconds(i) for i in range(12)]
    assert observed == [60, 60, 120, 120, 240, 240, 360, 360, 480, 480, 600, 600]
