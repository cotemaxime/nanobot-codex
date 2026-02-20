from types import SimpleNamespace

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.session.manager import Session


class DummyApp:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)


def _make_channel():
    channel = TelegramChannel(config=TelegramConfig(), bus=MessageBus())
    channel._app = DummyApp()
    return channel


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


class _NoopProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "test/default"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content="ok")


def test_register_reaction_handlers_uses_filter_constants(monkeypatch):
    channel = _make_channel()

    reaction_filter = object()
    reaction_count_filter = object()
    fake_filters = SimpleNamespace(
        UpdateType=SimpleNamespace(
            MESSAGE_REACTION=reaction_filter,
            MESSAGE_REACTION_COUNT=reaction_count_filter,
        ),
        StatusUpdate=None,
    )

    class DummyMessageHandler:
        def __init__(self, flt, callback):
            self.filter = flt
            self.callback = callback

    import nanobot.channels.telegram as telegram_module

    monkeypatch.setattr(telegram_module, "filters", fake_filters)
    monkeypatch.setattr(telegram_module, "MessageHandler", DummyMessageHandler)

    channel._register_reaction_handlers()

    assert len(channel._app.handlers) == 2
    assert channel._app.handlers[0].filter is reaction_filter
    assert channel._app.handlers[0].callback == channel._on_reaction
    assert channel._app.handlers[1].filter is reaction_count_filter
    assert channel._app.handlers[1].callback == channel._on_reaction_count


def test_register_reaction_handlers_falls_back_to_type_handler(monkeypatch):
    channel = _make_channel()

    fake_filters = SimpleNamespace(UpdateType=None, StatusUpdate=None)

    class DummyTypeHandler:
        def __init__(self, update_type, callback):
            self.update_type = update_type
            self.callback = callback

    import nanobot.channels.telegram as telegram_module
    import telegram.ext as telegram_ext

    monkeypatch.setattr(telegram_module, "filters", fake_filters)
    monkeypatch.setattr(telegram_ext, "TypeHandler", DummyTypeHandler)

    channel._register_reaction_handlers()

    assert len(channel._app.handlers) == 1
    assert isinstance(channel._app.handlers[0], DummyTypeHandler)
    assert channel._app.handlers[0].callback == channel._on_reaction_update_fallback


@pytest.mark.asyncio
async def test_reaction_metadata_uses_tracked_thread_id():
    channel = _make_channel()
    channel._remember_message_thread(chat_id="123", message_id=42, thread_id=99)
    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)

    channel._handle_message = _capture  # type: ignore[method-assign]

    reaction = SimpleNamespace(
        user=SimpleNamespace(id=7, username="alice", first_name="Alice"),
        actor_chat=None,
        chat=SimpleNamespace(id=123),
        old_reaction=[],
        new_reaction=[SimpleNamespace(emoji="👍")],
        message_id=42,
    )
    update = SimpleNamespace(message_reaction=reaction, effective_user=None)
    await channel._on_reaction(update, None)

    assert captured["chat_id"] == "123"
    assert captured["metadata"]["telegram"]["message_thread_id"] == 99
    assert captured["metadata"]["session_key"] == "telegram:123:99"


@pytest.mark.asyncio
async def test_forward_command_uses_reply_thread_when_direct_thread_missing():
    channel = _make_channel()
    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)

    channel._handle_message = _capture  # type: ignore[method-assign]
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=7, username="alice"),
        message=SimpleNamespace(
            chat_id=123,
            message_id=55,
            message_thread_id=None,
            reply_to_message=SimpleNamespace(message_thread_id=99),
            text="/model",
        ),
    )

    await channel._forward_command(update, None)

    assert captured["chat_id"] == "123"
    assert captured["metadata"]["telegram"]["message_thread_id"] == 99
    assert captured["metadata"]["session_key"] == "telegram:123:99"


@pytest.mark.asyncio
async def test_forward_command_keeps_zero_thread_id_in_session_key():
    channel = _make_channel()
    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)

    channel._handle_message = _capture  # type: ignore[method-assign]
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=7, username="alice"),
        message=SimpleNamespace(
            chat_id=123,
            message_id=56,
            message_thread_id=0,
            reply_to_message=None,
            text="/help",
        ),
    )

    await channel._forward_command(update, None)

    assert captured["metadata"]["telegram"]["message_thread_id"] == 0
    assert captured["metadata"]["session_key"] == "telegram:123:0"


class DummyBot:
    def __init__(self):
        self.sent = []
        self.edited = []
        self._next_id = 100

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        msg = SimpleNamespace(message_id=self._next_id, message_thread_id=kwargs.get("message_thread_id"))
        self._next_id += 1
        return msg

    async def edit_message_text(self, **kwargs):
        self.edited.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id"))


@pytest.mark.asyncio
async def test_topic_command_roundtrip_sends_reply_to_same_thread(tmp_path):
    bus = MessageBus()
    channel = TelegramChannel(config=TelegramConfig(), bus=bus)
    channel._app = SimpleNamespace(bot=DummyBot())
    loop = AgentLoop(
        bus=bus,
        provider=_NoopProvider(),
        workspace=tmp_path,
        session_manager=_InMemorySessionManager(),
        model="test/default",
    )

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=7, username="alice"),
        message=SimpleNamespace(
            chat_id=123,
            message_id=57,
            message_thread_id=99,
            reply_to_message=None,
            text="/model",
        ),
    )

    await channel._forward_command(update, None)
    inbound = await bus.consume_inbound()

    assert isinstance(inbound, InboundMessage)
    assert inbound.metadata.get("telegram", {}).get("message_thread_id") == 99
    assert inbound.metadata.get("session_key") == "telegram:123:99"

    response = await loop._process_message(inbound)

    assert response is not None
    assert "Current model for this chat/topic" in response.content
    assert response.metadata.get("telegram", {}).get("message_thread_id") == 99

    await channel.send(response)

    assert channel._app.bot.sent
    sent = channel._app.bot.sent[-1]
    assert sent["chat_id"] == 123
    assert sent["message_thread_id"] == 99


@pytest.mark.asyncio
async def test_progress_messages_are_silent_and_styled():
    channel = TelegramChannel(config=TelegramConfig(), bus=MessageBus())
    channel._app = SimpleNamespace(bot=DummyBot())

    await channel.send(OutboundMessage(
        channel="telegram",
        chat_id="123",
        content='[progress] Running: read_file("todo.txt")',
        metadata={"progress": True, "telegram": {"message_thread_id": 99}},
    ))

    assert len(channel._app.bot.sent) == 1
    call = channel._app.bot.sent[0]
    assert call["disable_notification"] is True
    assert call["parse_mode"] == "HTML"
    assert call["text"].startswith("<i>⏳ Running: read_file")
    assert channel._progress_message_ids[(123, 99)] == 100


@pytest.mark.asyncio
async def test_progress_messages_edit_single_status_bubble():
    channel = TelegramChannel(config=TelegramConfig(), bus=MessageBus())
    channel._app = SimpleNamespace(bot=DummyBot())

    first = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content='[progress] Running: read_file("todo.txt")',
        metadata={"progress": True, "telegram": {"message_thread_id": 99}},
    )
    second = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content='[progress] Running: exec("git status")',
        metadata={"progress": True, "telegram": {"message_thread_id": 99}},
    )
    await channel.send(first)
    await channel.send(second)

    assert len(channel._app.bot.sent) == 1
    assert len(channel._app.bot.edited) == 1
    assert channel._app.bot.edited[0]["message_id"] == 100
    assert "⏳ Running: exec" in channel._app.bot.edited[0]["text"]


@pytest.mark.asyncio
async def test_normal_reply_clears_progress_tracking():
    channel = TelegramChannel(config=TelegramConfig(), bus=MessageBus())
    channel._app = SimpleNamespace(bot=DummyBot())

    await channel.send(OutboundMessage(
        channel="telegram",
        chat_id="123",
        content='[progress] Running: read_file("todo.txt")',
        metadata={"progress": True, "telegram": {"message_thread_id": 99}},
    ))
    await channel.send(OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="Final answer",
        metadata={"telegram": {"message_thread_id": 99}},
    ))

    assert (123, 99) not in channel._progress_message_ids
