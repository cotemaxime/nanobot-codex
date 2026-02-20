from nanobot.providers.openai_codex_provider import _convert_messages


def test_convert_messages_concatenates_multiple_system_messages():
    system_prompt, input_items = _convert_messages(
        [
            {"role": "system", "content": "First system instruction."},
            {"role": "system", "content": "Second system instruction."},
            {"role": "user", "content": "hello"},
        ]
    )

    assert system_prompt == "First system instruction.\n\nSecond system instruction."
    assert len(input_items) == 1
    assert input_items[0]["role"] == "user"

