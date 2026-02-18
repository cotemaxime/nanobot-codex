from types import SimpleNamespace

from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig


class DummyApp:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)


def _make_channel():
    channel = TelegramChannel(config=TelegramConfig(), bus=MessageBus())
    channel._app = DummyApp()
    return channel


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
