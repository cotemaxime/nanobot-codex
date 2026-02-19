from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig


class _DummyBot:
    def __init__(self, fail_voice: bool = False):
        self.fail_voice = fail_voice
        self.voice_calls = []
        self.audio_calls = []
        self.message_calls = []

    async def send_voice(self, **kwargs):
        self.voice_calls.append(kwargs)
        if self.fail_voice:
            raise RuntimeError("voice send failed")
        return SimpleNamespace(message_id=101, message_thread_id=kwargs.get("message_thread_id"))

    async def send_audio(self, **kwargs):
        self.audio_calls.append(kwargs)
        return SimpleNamespace(message_id=102, message_thread_id=kwargs.get("message_thread_id"))

    async def send_message(self, **kwargs):
        self.message_calls.append(kwargs)
        return SimpleNamespace(message_id=103, message_thread_id=kwargs.get("message_thread_id"))

    async def send_photo(self, **kwargs):
        return SimpleNamespace(message_id=104, message_thread_id=kwargs.get("message_thread_id"))

    async def send_document(self, **kwargs):
        return SimpleNamespace(message_id=105, message_thread_id=kwargs.get("message_thread_id"))


class _FakeTTSProvider:
    calls: list[str] = []
    fail_all: bool = False

    def __init__(self, **kwargs):
        del kwargs

    async def synthesize(self, text: str, out_format: str):
        del text
        self.calls.append(out_format)
        if self.fail_all:
            return None
        path = Path("/tmp") / f"test_tts_{out_format}.{out_format}"
        path.write_bytes(b"fake-audio")
        return path


def _channel(tts_reply_mode: str = "voice_input_only", tts_send_text_also: bool = True) -> TelegramChannel:
    config = TelegramConfig(
        tts_enabled=True,
        tts_reply_mode=tts_reply_mode,
        tts_send_text_also=tts_send_text_also,
    )
    ch = TelegramChannel(config=config, bus=MessageBus(), openai_api_key="sk-test")
    ch._app = SimpleNamespace(bot=_DummyBot())
    return ch


@pytest.mark.asyncio
async def test_voice_input_metadata_triggers_tts(monkeypatch):
    import nanobot.providers.tts as tts_module

    _FakeTTSProvider.calls = []
    _FakeTTSProvider.fail_all = False
    monkeypatch.setattr(tts_module, "OpenAITTSProvider", _FakeTTSProvider)

    channel = _channel()
    msg = OutboundMessage(
        channel="telegram",
        chat_id="42",
        content="hello",
        metadata={"telegram": {"voice_input": True}},
    )
    await channel.send(msg)

    assert _FakeTTSProvider.calls == ["opus"]
    assert len(channel._app.bot.voice_calls) == 1
    assert len(channel._app.bot.message_calls) == 1


@pytest.mark.asyncio
async def test_text_only_does_not_trigger_tts_in_voice_input_mode(monkeypatch):
    import nanobot.providers.tts as tts_module

    _FakeTTSProvider.calls = []
    monkeypatch.setattr(tts_module, "OpenAITTSProvider", _FakeTTSProvider)

    channel = _channel()
    msg = OutboundMessage(channel="telegram", chat_id="42", content="hello", metadata={"telegram": {}})
    await channel.send(msg)

    assert _FakeTTSProvider.calls == []
    assert len(channel._app.bot.voice_calls) == 0
    assert len(channel._app.bot.audio_calls) == 0
    assert len(channel._app.bot.message_calls) == 1


@pytest.mark.asyncio
async def test_opus_send_failure_falls_back_to_mp3(monkeypatch):
    import nanobot.providers.tts as tts_module

    _FakeTTSProvider.calls = []
    _FakeTTSProvider.fail_all = False
    monkeypatch.setattr(tts_module, "OpenAITTSProvider", _FakeTTSProvider)

    channel = _channel()
    channel._app = SimpleNamespace(bot=_DummyBot(fail_voice=True))
    msg = OutboundMessage(
        channel="telegram",
        chat_id="42",
        content="hello",
        metadata={"telegram": {"voice_input": True}},
    )
    await channel.send(msg)

    assert _FakeTTSProvider.calls == ["opus", "mp3"]
    assert len(channel._app.bot.voice_calls) == 1
    assert len(channel._app.bot.audio_calls) == 1
    assert len(channel._app.bot.message_calls) == 1


@pytest.mark.asyncio
async def test_tts_failure_still_sends_text(monkeypatch):
    import nanobot.providers.tts as tts_module

    _FakeTTSProvider.calls = []
    _FakeTTSProvider.fail_all = True
    monkeypatch.setattr(tts_module, "OpenAITTSProvider", _FakeTTSProvider)

    channel = _channel()
    msg = OutboundMessage(
        channel="telegram",
        chat_id="42",
        content="hello",
        metadata={"telegram": {"voice_input": True}},
    )
    await channel.send(msg)

    assert _FakeTTSProvider.calls == ["opus", "mp3"]
    assert len(channel._app.bot.voice_calls) == 0
    assert len(channel._app.bot.audio_calls) == 0
    assert len(channel._app.bot.message_calls) == 1
