import pytest
from pydantic import ValidationError

from nanobot.config.schema import Config, TelegramConfig


def test_telegram_tts_defaults():
    cfg = Config()
    tg = cfg.channels.telegram
    assert tg.tts_enabled is False
    assert tg.tts_reply_mode == "voice_input_only"
    assert tg.tts_send_text_also is True
    assert tg.tts_voice == "alloy"
    assert tg.tts_model == "gpt-4o-mini-tts"
    assert cfg.providers.openai.tts_api_base is None


def test_telegram_tts_mode_validation_rejects_invalid_value():
    with pytest.raises(ValidationError):
        TelegramConfig(tts_reply_mode="invalid-mode")
