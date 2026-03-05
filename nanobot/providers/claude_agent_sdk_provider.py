"""Claude Agent SDK-backed provider for OAuth/session-style local auth flows."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import time
from typing import Any
from uuid import uuid4

from loguru import logger

from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class ClaudeAgentSDKProvider(LLMProvider):
    """LLM provider that routes requests through Anthropic Agent SDK."""

    def __init__(
        self,
        default_model: str,
        workspace: str | None = None,
        timeout_seconds: int = 180,
        max_turns: int = 12,
        max_internal_native_steps: int = 6,
        permission_mode: str = "acceptEdits",
        native_tools: list[str] | None = None,
        strict_auth: bool = False,
        diagnostic_logging: bool = False,
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = self._normalize_model_name(default_model)
        self.workspace = workspace or os.getcwd()
        self.timeout_seconds = max(30, int(timeout_seconds))
        self.max_turns = max(1, int(max_turns))
        self.max_internal_native_steps = max(1, int(max_internal_native_steps))
        self.permission_mode = permission_mode
        self.native_tools = set(native_tools or ["exec", "web_search", "web_fetch"])
        self.diagnostic_logging = diagnostic_logging
        self._sdk = self._load_sdk()
        self._native_executors: dict[str, Any] = {
            "exec": ExecTool(working_dir=self.workspace),
            "web_search": WebSearchTool(api_key=os.environ.get("BRAVE_API_KEY")),
            "web_fetch": WebFetchTool(),
        }
        ok, detail = self._probe_session()
        self.session_check = detail
        if self.diagnostic_logging:
            logger.info(f"[claude-agent-provider] session_check ok={ok} detail={detail}")
        if strict_auth and not ok:
            raise RuntimeError(
                "Claude Agent SDK auth is unavailable. "
                f"Detail: {detail}. Configure ANTHROPIC_API_KEY or a valid local Claude SDK auth session."
            )

    @staticmethod
    def _load_sdk() -> Any:
        try:
            return importlib.import_module("anthropic_agent_sdk")
        except Exception as e:
            raise RuntimeError(
                "Anthropic Agent SDK is not available. Install with "
                "`pip install anthropic-agent-sdk` and retry."
            ) from e

    @staticmethod
    def _probe_session() -> tuple[bool, str]:
        # Anthropic Agent SDK officially supports API key and cloud credentials;
        # local session/oauth-style flows are environment/CLI dependent.
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True, "ANTHROPIC_API_KEY detected"
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return True, "CLAUDE_CODE_OAUTH_TOKEN detected"
        if shutil.which("claude"):
            return True, "Claude CLI detected on PATH"
        if os.path.exists(os.path.expanduser("~/.claude")):
            return True, "Potential Claude local auth directory detected"
        return False, "No explicit auth env var detected"

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        cleaned = (model or "").strip()
        lowered = cleaned.lower()
        if lowered.startswith("claude-agent/") or lowered.startswith("claude_agent/"):
            return cleaned.split("/", 1)[1]
        return cleaned

    def get_default_model(self) -> str:
        return self.default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        trace_id = uuid4().hex[:8]
        started = time.monotonic()
        model_name = self._normalize_model_name(model) if model else self.default_model
        if self.diagnostic_logging:
            logger.info(
                f"[claude-agent-provider:{trace_id}] chat start model={model_name} "
                f"messages={len(messages)} tools={len(tools or [])}"
            )
        try:
            run_messages = list(messages)
            tools = tools or []
            native_defs, delegated_defs = self._split_tool_definitions(tools)
            all_defs = native_defs + delegated_defs

            steps = 0
            while steps < self.max_internal_native_steps:
                steps += 1
                if self.diagnostic_logging:
                    logger.debug(
                        f"[claude-agent-provider:{trace_id}] sdk step={steps}/{self.max_internal_native_steps} "
                        f"message_count={len(run_messages)} native_defs={len(native_defs)} delegated_defs={len(delegated_defs)}"
                    )
                response = await self._query_once(run_messages, all_defs, model_name)

                if not response.tool_calls:
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    if self.diagnostic_logging:
                        logger.info(
                            f"[claude-agent-provider:{trace_id}] chat complete finish_reason={response.finish_reason} "
                            f"elapsed_ms={elapsed_ms}"
                        )
                    return response

                native_calls: list[ToolCallRequest] = []
                delegated_calls: list[ToolCallRequest] = []
                for tc in response.tool_calls:
                    if tc.name in self.native_tools:
                        native_calls.append(tc)
                    else:
                        delegated_calls.append(tc)
                if self.diagnostic_logging:
                    logger.info(
                        f"[claude-agent-provider:{trace_id}] tool_calls total={len(response.tool_calls)} "
                        f"native={len(native_calls)} delegated={len(delegated_calls)}"
                    )

                if delegated_calls and native_calls:
                    all_calls = delegated_calls + native_calls
                    return LLMResponse(
                        content=response.content,
                        tool_calls=all_calls,
                        finish_reason=response.finish_reason,
                        usage=response.usage,
                        reasoning_content=response.reasoning_content,
                    )

                if delegated_calls:
                    return LLMResponse(
                        content=response.content,
                        tool_calls=delegated_calls,
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
                    "Error calling Claude Agent SDK: exceeded internal native-tool step limit "
                    f"({self.max_internal_native_steps})."
                ),
                finish_reason="error",
            )
        except Exception as e:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if self.diagnostic_logging:
                logger.exception(
                    f"[claude-agent-provider:{trace_id}] chat failed model={model_name} "
                    f"elapsed_ms={elapsed_ms}: {e}"
                )
            return LLMResponse(content=f"Error calling Claude Agent SDK: {e}", finish_reason="error")

    async def _query_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
    ) -> LLMResponse:
        query = getattr(self._sdk, "query", None)
        options_cls = getattr(self._sdk, "ClaudeAgentOptions", None)
        if not callable(query) or options_cls is None:
            raise RuntimeError("anthropic_agent_sdk.query / ClaudeAgentOptions not found")

        prompt = self._build_prompt(messages, tools)
        options = options_cls(
            model=model_name,
            cwd=self.workspace,
            permission_mode=self.permission_mode,
            max_turns=self.max_turns,
        )

        async def _run_query() -> str:
            final_text = ""
            async for item in query(prompt=prompt, options=options):
                text = self._extract_stream_text(item)
                if text:
                    final_text = text
            return final_text

        raw = await asyncio.wait_for(_run_query(), timeout=self.timeout_seconds)
        return self._parse_response_payload(raw)

    @staticmethod
    def _extract_stream_text(item: Any) -> str:
        for attr in ("result", "content", "text", "message"):
            val = getattr(item, attr, None)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list):
                joined = " ".join(str(x) for x in val if x)
                if joined.strip():
                    return joined
        if isinstance(item, str) and item.strip():
            return item
        return ""

    @classmethod
    def _build_prompt(cls, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        schema = cls._output_contract_schema()
        lines = [
            "You are executing one assistant turn for an external orchestrator.",
            "Return only valid JSON matching this schema:",
            json.dumps(schema, ensure_ascii=False),
            "",
            "Conversation (latest last):",
        ]

        for m in messages:
            role = str(m.get("role", "unknown"))
            content = m.get("content")
            if isinstance(content, (dict, list)):
                content_txt = json.dumps(content, ensure_ascii=False)
            else:
                content_txt = str(content or "")
            if role == "tool":
                lines.append(
                    f"[{role}] name={m.get('name')} id={m.get('tool_call_id')}: {content_txt}"
                )
            else:
                lines.append(f"[{role}] {content_txt}")
            if "tool_calls" in m:
                lines.append(f"[assistant_tool_calls] {json.dumps(m['tool_calls'], ensure_ascii=False)}")

        if tools:
            lines.append("")
            lines.append("Delegatable tools (emit tool_calls using names from this list):")
            lines.append(json.dumps(tools, ensure_ascii=False))

        lines.append("")
        lines.append("Rules:")
        lines.append('- Put direct answer text in "content" when no tool call is needed.')
        lines.append('- For tool calls, set arguments as a JSON string object.')
        lines.append('- Never output markdown fences or prose outside the JSON object.')
        return "\n".join(lines)

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
    def _output_contract_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": ["string", "null"]},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "arguments": {"type": "string"},
                        },
                        "required": ["id", "name", "arguments"],
                    },
                },
                "finish_reason": {"type": "string"},
                "reasoning_content": {"type": ["string", "null"]},
            },
            "required": ["content", "tool_calls", "finish_reason", "reasoning_content"],
        }

    @staticmethod
    def _parse_response_payload(raw: str) -> LLMResponse:
        text = (raw or "").strip()
        if not text:
            return LLMResponse(content="", finish_reason="stop")
        try:
            payload = json.loads(text)
        except Exception:
            return LLMResponse(content=text, finish_reason="stop")

        content = payload.get("content")
        finish_reason = payload.get("finish_reason") or "stop"
        reasoning_content = payload.get("reasoning_content")
        tool_calls: list[ToolCallRequest] = []
        for idx, tc in enumerate(payload.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            if not isinstance(name, str) or not name:
                continue
            raw_args = tc.get("arguments")
            args: dict[str, Any] = {}
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            call_id = tc.get("id") if isinstance(tc.get("id"), str) and tc.get("id") else f"claude_call_{idx}"
            tool_calls.append(ToolCallRequest(id=call_id, name=name, arguments=args))

        return LLMResponse(
            content=content if isinstance(content, str) or content is None else str(content),
            tool_calls=tool_calls,
            finish_reason=str(finish_reason),
            reasoning_content=reasoning_content if isinstance(reasoning_content, str) or reasoning_content is None else str(reasoning_content),
        )

    @staticmethod
    def _append_assistant_tool_call_message(
        messages: list[dict[str, Any]], response: LLMResponse
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
        out.append({"role": "assistant", "content": response.content or "", "tool_calls": tool_call_dicts})
        return out

    async def _run_native_tool(self, tool_call: ToolCallRequest) -> str:
        tool = self._native_executors.get(tool_call.name)
        if not tool:
            return f"Error: Native tool '{tool_call.name}' is not configured"
        try:
            return await tool.execute(**tool_call.arguments)
        except Exception as e:
            return f"Error executing native tool '{tool_call.name}': {e}"
