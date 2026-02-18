from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


def test_has_pending_inbound_for_session_uses_metadata_session_key():
    bus = MessageBus()
    bus.inbound.put_nowait(
        InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="100",
            content="hello",
            metadata={"session_key": "telegram:100:thread-1"},
        )
    )

    assert bus.has_pending_inbound_for_session("telegram:100:thread-1") is True
    assert bus.has_pending_inbound_for_session("telegram:100:thread-2") is False


def test_has_pending_inbound_for_session_falls_back_to_default_key():
    bus = MessageBus()
    bus.inbound.put_nowait(
        InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="100",
            content="hello",
        )
    )

    assert bus.has_pending_inbound_for_session("telegram:100") is True
    assert bus.has_pending_inbound_for_session("telegram:999") is False

