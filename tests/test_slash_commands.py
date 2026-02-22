import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.slash_commands import SlashCommandsLoader
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.session.manager import Session


class _InMemorySessionManager:
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


class _CaptureProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key="test", api_base=None)
        self.last_messages = None

    def get_default_model(self) -> str:
        return "test/default"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.last_messages = messages
        return LLMResponse(content="ok")


def test_slash_loader_parses_frontmatter_and_body(tmp_path):
    workspace = tmp_path / "workspace"
    slash = workspace / "slash"
    slash.mkdir(parents=True, exist_ok=True)
    (slash / "retrospective.md").write_text(
        "---\n"
        "name: retro\n"
        "description: Run daily retrospective\n"
        "---\n"
        "Summarize wins/losses and suggest next actions.\n",
        encoding="utf-8",
    )

    loader = SlashCommandsLoader(workspace)
    commands = loader.list_commands()

    assert len(commands) == 1
    assert commands[0]["name"] == "retro"
    assert commands[0]["description"] == "Run daily retrospective"
    assert "Summarize wins/losses" in commands[0]["prompt"]


@pytest.mark.asyncio
async def test_agent_injects_custom_slash_prompt(tmp_path):
    workspace = tmp_path / "workspace"
    slash = workspace / "slash"
    slash.mkdir(parents=True, exist_ok=True)
    (slash / "review.md").write_text(
        "---\n"
        "name: review\n"
        "description: Review conversations\n"
        "---\n"
        "Read the latest conversation logs and produce 3 action items.\n",
        encoding="utf-8",
    )

    provider = _CaptureProvider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        session_manager=_InMemorySessionManager(),
        model="test/default",
    )
    inbound = InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="chat-1",
        content="/review focus mobile blockers",
    )

    await loop._process_message(inbound)

    assert provider.last_messages is not None
    user_msg = provider.last_messages[-1]["content"]
    assert isinstance(user_msg, str)
    assert "[Slash command invoked: /review]" in user_msg
    assert "Arguments: focus mobile blockers" in user_msg
    assert "Read the latest conversation logs" in user_msg
