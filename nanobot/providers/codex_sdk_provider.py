"""Codex SDK-backed provider with hybrid native/delegated tool handling."""

from __future__ import annotations

import json
import os
from typing import Any

from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.codex_transport import CodexTransport, TransportToolCall


class CodexSDKProvider(LLMProvider):
    """LLM provider that routes requests through Codex SDK session auth."""

    def __init__(
        self,
        default_model: str,
        profile: str | None = None,
        workspace: str | None = None,
        timeout_seconds: int = 180,
        max_internal_native_steps: int = 6,
        strict_auth: bool = True,
        skip_git_repo_check: bool = True,
        native_tools: list[str] | None = None,
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.profile = profile
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_internal_native_steps = max(1, max_internal_native_steps)
        self.skip_git_repo_check = skip_git_repo_check
        self.native_tools = set(native_tools or ["exec", "web_search", "web_fetch"])
        self.transport = CodexTransport(
            profile=profile,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
            skip_git_repo_check=skip_git_repo_check,
        )

        self._native_executors: dict[str, Any] = {
            "exec": ExecTool(working_dir=workspace or os.getcwd()),
            "web_search": WebSearchTool(api_key=os.environ.get("BRAVE_API_KEY")),
            "web_fetch": WebFetchTool(),
        }

        ok, detail = self.transport.validate_session()
        self.session_check = detail
        if strict_auth and not ok:
            raise RuntimeError(
                "Codex session is unavailable. "
                f"Detail: {detail}. "
                "Run Codex login/auth outside nanobot first, then retry."
            )

    def get_default_model(self) -> str:
        return self.default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        try:
            run_messages = list(messages)
            tools = tools or []
            native_defs, delegated_defs = self._split_tool_definitions(tools)
            all_defs = native_defs + delegated_defs

            steps = 0
            while steps < self.max_internal_native_steps:
                steps += 1
                response = await self.transport.chat(
                    messages=run_messages,
                    tools=all_defs,
                    model=model or self.default_model,
                    max_tokens=max(1, max_tokens),
                    temperature=temperature,
                )

                if not response.tool_calls:
                    return LLMResponse(
                        content=response.content,
                        finish_reason=response.finish_reason,
                        usage=response.usage,
                        reasoning_content=response.reasoning_content,
                    )

                native_calls: list[TransportToolCall] = []
                delegated_calls: list[TransportToolCall] = []
                for tc in response.tool_calls:
                    if tc.name in self.native_tools:
                        native_calls.append(tc)
                    else:
                        delegated_calls.append(tc)

                # Mixed batches cannot safely preserve native results across provider
                # boundaries, so delegate all calls to nanobot for correctness.
                if delegated_calls and native_calls:
                    all_calls = delegated_calls + native_calls
                    return LLMResponse(
                        content=response.content,
                        tool_calls=[
                            ToolCallRequest(id=tc.id, name=tc.name, arguments=tc.arguments)
                            for tc in all_calls
                        ],
                        finish_reason=response.finish_reason,
                        usage=response.usage,
                        reasoning_content=response.reasoning_content,
                    )

                if delegated_calls:
                    return LLMResponse(
                        content=response.content,
                        tool_calls=[
                            ToolCallRequest(id=tc.id, name=tc.name, arguments=tc.arguments)
                            for tc in delegated_calls
                        ],
                        finish_reason=response.finish_reason,
                        usage=response.usage,
                        reasoning_content=response.reasoning_content,
                    )

                run_messages = self._append_assistant_tool_call_message(run_messages, response)

                for tc in native_calls:
                    result = await self._run_native_tool(tc)
                    run_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        }
                    )

            return LLMResponse(
                content=(
                    "Error calling Codex: exceeded internal native-tool step limit "
                    f"({self.max_internal_native_steps})."
                ),
                finish_reason="error",
            )

        except Exception as e:
            return LLMResponse(content=f"Error calling Codex: {e}", finish_reason="error")

    def _split_tool_definitions(
        self, tools: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        native_defs: list[dict[str, Any]] = []
        delegated_defs: list[dict[str, Any]] = []

        for tool in tools:
            name = self._extract_tool_name(tool)
            if name and name in self.native_tools:
                native_defs.append(tool)
            else:
                delegated_defs.append(tool)

        return native_defs, delegated_defs

    @staticmethod
    def _extract_tool_name(tool: dict[str, Any]) -> str | None:
        if tool.get("type") == "function":
            fn = tool.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                if isinstance(name, str):
                    return name
        name = tool.get("name")
        return name if isinstance(name, str) else None

    @staticmethod
    def _append_assistant_tool_call_message(
        messages: list[dict[str, Any]], response: Any
    ) -> list[dict[str, Any]]:
        tool_call_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ]

        out = list(messages)
        out.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": tool_call_dicts,
            }
        )
        return out

    async def _run_native_tool(self, tool_call: TransportToolCall) -> str:
        tool = self._native_executors.get(tool_call.name)
        if not tool:
            return f"Error: Native tool '{tool_call.name}' is not configured"
        try:
            return await tool.execute(**tool_call.arguments)
        except Exception as e:
            return f"Error executing native tool '{tool_call.name}': {e}"
