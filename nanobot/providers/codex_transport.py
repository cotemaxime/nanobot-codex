"""Codex SDK transport adapter and session checks.

This module isolates Codex SDK interaction from provider logic so future SDK
changes are localized here.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


@dataclass
class TransportToolCall:
    """Normalized tool call from Codex transport."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TransportResponse:
    """Normalized response from Codex transport."""

    content: str | None
    tool_calls: list[TransportToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None


class CodexTransport:
    """Thin adapter around Codex SDK runtime."""

    def __init__(
        self,
        profile: str | None = None,
        workspace: str | None = None,
        timeout_seconds: int = 180,
        skip_git_repo_check: bool = True,
        sandbox_mode: str = "workspace-write",
        approval_policy: str = "never",
        network_access_enabled: bool = True,
        web_search_enabled: bool = True,
    ):
        self.profile = profile
        self.workspace = str(Path(workspace).expanduser().resolve()) if workspace else None
        self.timeout_seconds = timeout_seconds
        self.diagnostic_logging = False
        self.skip_git_repo_check = skip_git_repo_check
        self.sandbox_mode = sandbox_mode
        self.approval_policy = approval_policy
        self.network_access_enabled = network_access_enabled
        self.web_search_enabled = web_search_enabled
        self._sdk = self._load_sdk()

    @staticmethod
    def _load_sdk() -> Any:
        """Import Codex SDK module, or raise actionable error."""
        try:
            return __import__("openai_codex_sdk")
        except Exception as e:
            raise RuntimeError(
                "Codex SDK is not available. Install the official wrapper with "
                "`pip install openai-codex-sdk`, authenticate with Codex, then retry. "
                f"Last error: {e}"
            ) from e

    @staticmethod
    def probe_session(profile: str | None = None) -> tuple[bool, str]:
        """Best-effort local session availability probe without SDK dependency."""
        env_token = os.environ.get("CODEX_OAUTH_TOKEN") or os.environ.get("OPENAI_OAUTH_TOKEN")
        if env_token:
            return True, "OAuth token found in environment"

        auth_dir = Path.home() / ".codex"
        auth_file = auth_dir / "auth.json"
        if auth_file.exists():
            if profile:
                try:
                    data = json.loads(auth_file.read_text(encoding="utf-8"))
                    profiles = data.get("profiles") if isinstance(data, dict) else None
                    if isinstance(profiles, dict) and profile in profiles:
                        return True, f"Auth profile '{profile}' found"
                    return False, f"Auth profile '{profile}' not found"
                except Exception:
                    return True, "Auth file found (profile check unreadable)"
            return True, "Auth file found"

        return False, "No Codex session detected"

    def validate_session(self) -> tuple[bool, str]:
        """Validate session and optionally perform SDK-level check when available."""
        ok, detail = self.probe_session(self.profile)
        if not ok:
            return ok, detail

        # Optional SDK-level check. Keep resilient to SDK shape changes.
        try:  # pragma: no cover - depends on installed SDK API
            checker = getattr(self._sdk, "auth_status", None)
            if callable(checker):
                status = checker(profile=self.profile) if self.profile else checker()
                if isinstance(status, dict):
                    if status.get("authenticated") is False:
                        return False, "Codex SDK reports unauthenticated session"
                    msg = status.get("message") or detail
                    return True, str(msg)
            return True, detail
        except Exception as e:
            return False, f"Codex SDK session check failed: {e}"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
    ) -> TransportResponse:
        """Call Codex SDK and normalize result."""
        trace_id = uuid4().hex[:8]
        timeout_s = max(1, int(self.timeout_seconds or 1))
        started = time.monotonic()
        model_name = model or "codex/default"
        tool_defs = tools or []

        if self.diagnostic_logging:
            logger.info(
                f"[codex-sdk:{trace_id}] chat start model={model_name} "
                f"messages={len(messages)} tools={len(tool_defs)} timeout_s={timeout_s}"
            )
        if self.diagnostic_logging:
            logger.debug(
                f"[codex-sdk:{trace_id}] opts profile={self.profile!r} workspace={self.workspace!r} "
                f"sandbox={self.sandbox_mode} approval={self.approval_policy} "
                f"network={self.network_access_enabled} web_search={self.web_search_enabled}"
            )

        try:
            # Legacy guessed SDK API support.
            run_chat = getattr(self._sdk, "chat", None)
            if callable(run_chat):  # pragma: no cover - legacy path
                kwargs: dict[str, Any] = {
                    "messages": messages,
                    "tools": tools or [],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "timeout_seconds": self.timeout_seconds,
                }
                if model:
                    kwargs["model"] = model
                if self.profile:
                    kwargs["profile"] = self.profile
                if self.workspace:
                    kwargs["workspace"] = self.workspace

                raw = await self._invoke_with_timeout(
                    run_chat,
                    timeout_s=timeout_s,
                    **kwargs,
                )
                response = self._normalize_response(raw)
                elapsed_ms = int((time.monotonic() - started) * 1000)
                if self.diagnostic_logging:
                    logger.info(
                        f"[codex-sdk:{trace_id}] chat ok path=legacy finish_reason={response.finish_reason} "
                        f"tool_calls={len(response.tool_calls)} elapsed_ms={elapsed_ms}"
                    )
                return response

            # Official openai-codex-sdk path.
            if hasattr(self._sdk, "Codex"):
                response = await self._chat_via_official_sdk(messages, tool_defs, model)
                elapsed_ms = int((time.monotonic() - started) * 1000)
                if self.diagnostic_logging:
                    logger.info(
                        f"[codex-sdk:{trace_id}] chat ok path=official finish_reason={response.finish_reason} "
                        f"tool_calls={len(response.tool_calls)} elapsed_ms={elapsed_ms}"
                    )
                return response

            raise RuntimeError("Codex SDK does not expose a supported runtime API")
        except asyncio.TimeoutError as e:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if self.diagnostic_logging:
                logger.error(
                    f"[codex-sdk:{trace_id}] chat timeout model={model_name} "
                    f"elapsed_ms={elapsed_ms} timeout_s={timeout_s}"
                )
            raise RuntimeError(
                f"Codex SDK chat call timed out after {timeout_s}s"
            ) from e
        except Exception as e:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if self.diagnostic_logging:
                logger.exception(
                    f"[codex-sdk:{trace_id}] chat failed model={model_name} elapsed_ms={elapsed_ms}: {e}"
                )
            raise RuntimeError(f"Codex SDK chat call failed: {e}") from e

    async def _chat_via_official_sdk(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
    ) -> TransportResponse:
        """Adapt nanobot chat/tools into the official openai-codex-sdk turn model."""
        sdk = self._sdk
        timeout_s = max(1, int(self.timeout_seconds or 1))

        env = dict(os.environ)
        if self.profile:
            env.setdefault("CODEX_PROFILE", self.profile)

        codex = sdk.Codex(options={"env": env})
        thread_opts: dict[str, Any] = {
            "sandboxMode": self.sandbox_mode,
            "approvalPolicy": self.approval_policy,
            "networkAccessEnabled": self.network_access_enabled,
            "webSearchEnabled": self.web_search_enabled,
            "skipGitRepoCheck": self.skip_git_repo_check,
        }
        if model:
            thread_opts["model"] = model
        if self.workspace:
            thread_opts["workingDirectory"] = self.workspace

        thread = codex.start_thread(thread_opts)

        output_schema = self._build_output_schema(tools or [])
        prompt = self._build_prompt(messages, tools or [])
        if self.diagnostic_logging:
            logger.debug(
                f"[codex-sdk] official thread.run prompt_chars={len(prompt)} schema_keys={list(output_schema.keys())}"
            )
        turn = await self._invoke_with_timeout(
            thread.run,
            prompt,
            {"outputSchema": output_schema},
            timeout_s=timeout_s,
        )

        usage: dict[str, int] = {}
        if getattr(turn, "usage", None):
            usage = {
                "prompt_tokens": int(getattr(turn.usage, "input_tokens", 0)),
                "completion_tokens": int(getattr(turn.usage, "output_tokens", 0)),
                "total_tokens": int(getattr(turn.usage, "input_tokens", 0))
                + int(getattr(turn.usage, "output_tokens", 0)),
            }

        raw_final = getattr(turn, "final_response", None)
        output_text = getattr(turn, "output_text", None) or getattr(turn, "text", None)
        if not raw_final and isinstance(output_text, str) and output_text.strip():
            raw_final = {
                "content": output_text,
                "tool_calls": [],
                "finish_reason": "stop",
                "reasoning_content": None,
            }

        response = self._normalize_response(raw_final if raw_final is not None else "")
        if (not response.content or not str(response.content).strip()) and not response.tool_calls:
            fallback = self._extract_turn_fallback_text(turn)
            if fallback:
                response.content = fallback
            elif self.diagnostic_logging:
                logger.warning(
                    "[codex-sdk] official run returned empty content/tool_calls; "
                    "turn_type={} raw_final_type={}",
                    type(turn).__name__,
                    type(raw_final).__name__ if raw_final is not None else "None",
                )

        return response

    @staticmethod
    def _extract_turn_fallback_text(turn: Any) -> str | None:
        """Best-effort extraction of text from SDK turn object variants."""
        for attr in ("output_text", "text", "final_text", "result"):
            val = getattr(turn, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        final_response = getattr(turn, "final_response", None)
        if isinstance(final_response, dict):
            for key in ("text", "output_text", "content"):
                val = final_response.get(key)
                if isinstance(val, str) and val.strip():
                    return val
        return None

    @staticmethod
    async def _invoke_with_timeout(
        fn: Any,
        *args: Any,
        timeout_s: int,
        **kwargs: Any,
    ) -> Any:
        """Invoke SDK call that may be sync or async with enforced timeout."""
        if inspect.iscoroutinefunction(fn):
            return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_s)

        # If fn is synchronous, run it off the event loop so timeout can fire.
        result = await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=timeout_s,
        )
        if inspect.isawaitable(result):
            return await asyncio.wait_for(result, timeout=timeout_s)
        return result

    @staticmethod
    def _build_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        """Build deterministic prompt for Codex turn execution."""
        lines = [
            "You are executing one assistant turn for an external orchestrator.",
            "Return only valid JSON matching the provided schema.",
            "If you need orchestrator-delegated tools, emit tool_calls.",
            "For each tool call, set arguments to a JSON string object, e.g. \"{\\\"path\\\":\\\"/tmp/a.txt\\\"}\".",
            "Use direct answer text in content when no tool call is needed.",
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
            lines.append("Delegatable tools (call via tool_calls when needed):")
            lines.append(json.dumps(tools, ensure_ascii=False))

        return "\n".join(lines)

    @staticmethod
    def _build_output_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Schema constraining Codex output to nanobot-compatible shape."""
        tool_names: list[str] = []
        for t in tools:
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                name = t["function"].get("name")
                if isinstance(name, str):
                    tool_names.append(name)

        tool_call_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                # Codex response_format schema requires additionalProperties=false
                # for object types; use JSON string here to keep arbitrary args.
                "arguments": {"type": "string"},
            },
            "required": ["id", "name", "arguments"],
            "additionalProperties": False,
        }
        if tool_names:
            tool_call_schema["properties"]["name"] = {
                "type": "string",
                "enum": tool_names,
            }

        return {
            "type": "object",
            "properties": {
                "content": {"type": ["string", "null"]},
                "tool_calls": {
                    "type": "array",
                    "items": tool_call_schema,
                },
                "finish_reason": {"type": "string"},
                "reasoning_content": {"type": ["string", "null"]},
            },
            "required": ["content", "tool_calls", "finish_reason", "reasoning_content"],
            "additionalProperties": False,
        }

    def _normalize_response(self, raw: Any) -> TransportResponse:
        """Normalize various SDK response shapes."""
        if isinstance(raw, TransportResponse):
            return raw

        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("{"):
                try:
                    raw = json.loads(text)
                except Exception:
                    return TransportResponse(content=raw)
            else:
                return TransportResponse(content=raw)

        if isinstance(raw, dict):
            content = raw.get("content")
            finish_reason = raw.get("finish_reason") or raw.get("finishReason") or "stop"
            usage = raw.get("usage") or {}
            reasoning_content = raw.get("reasoning_content") or raw.get("reasoningContent")

            tool_calls: list[TransportToolCall] = []
            raw_tool_calls = raw.get("tool_calls") or raw.get("toolCalls") or []
            for i, tc in enumerate(raw_tool_calls):
                if not isinstance(tc, dict):
                    raise RuntimeError(f"Unsupported tool call type: {type(tc).__name__}")
                tool_id = str(tc.get("id") or f"tc_{i}")
                name = tc.get("name") or tc.get("function", {}).get("name")
                if not name:
                    raise RuntimeError("Unsupported tool call shape: missing name")

                args = tc.get("arguments")
                if args is None:
                    args = tc.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    raise RuntimeError(f"Unsupported tool call arguments type: {type(args).__name__}")

                tool_calls.append(TransportToolCall(id=tool_id, name=str(name), arguments=args))

            return TransportResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=str(finish_reason),
                usage=usage if isinstance(usage, dict) else {},
                reasoning_content=reasoning_content,
            )

        raise RuntimeError(f"Unsupported Codex SDK response type: {type(raw).__name__}")
