import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


class FakeProvider:
    def get_default_model(self) -> str:
        return "codex/default"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content="done")


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
    assert "Started background task" in outbound.content
    assert "final result" in outbound.content

