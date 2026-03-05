from nanobot.providers.claude_agent_sdk_provider import ClaudeAgentSDKProvider


def test_claude_agent_normalize_model_name():
    assert ClaudeAgentSDKProvider._normalize_model_name("claude-agent/claude-sonnet-4-5") == "claude-sonnet-4-5"
    assert ClaudeAgentSDKProvider._normalize_model_name("claude-sonnet-4-5") == "claude-sonnet-4-5"


def test_claude_agent_parse_response_payload_with_tool_calls():
    raw = (
        '{"content":null,"tool_calls":[{"id":"c1","name":"read_file",'
        '"arguments":"{\\"path\\":\\"/tmp/x.txt\\"}"}],'
        '"finish_reason":"tool_calls","reasoning_content":null}'
    )
    resp = ClaudeAgentSDKProvider._parse_response_payload(raw)
    assert resp.has_tool_calls
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments["path"] == "/tmp/x.txt"
