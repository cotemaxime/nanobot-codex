"""Subagent manager for background task execution."""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig
from nanobot.providers.base import LLMProvider

_NATIVE_SDK_PROVIDER_CLASS_NAMES = {"OpenAICodexProvider", "CodexSDKProvider", "ClaudeAgentSDKProvider"}


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        disabled_skills: list[str] | None = None,
        fallback_models: list[str] | None = None,
        heartbeat_interval_seconds: int = 30,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.disabled_skills = {
            s.strip().lower()
            for s in (disabled_skills or [])
            if isinstance(s, str) and s.strip()
        }
        self.fallback_models = [
            m.strip()
            for m in (fallback_models or [])
            if isinstance(m, str) and m.strip()
        ]
        self.heartbeat_interval_seconds = max(1, int(heartbeat_interval_seconds or 30))
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        origin_metadata: dict[str, Any] | None = None,
        origin_session_key: str | None = None,
        session_key: str | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        # Merge session_key from either parameter name for compatibility
        effective_session_key = session_key or origin_session_key

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "metadata": dict(origin_metadata or {}),
            "session_key": effective_session_key,
        }

        # Create background task
        started_monotonic = time.monotonic()
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, started_monotonic)
        )
        self._running_tasks[task_id] = bg_task
        if effective_session_key:
            self._session_tasks.setdefault(effective_session_key, set()).add(task_id)
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(task_id, display_label, origin, started_monotonic, bg_task)
        )

        # Cleanup when done
        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            heartbeat_task.cancel()
            if effective_session_key and (ids := self._session_tasks.get(effective_session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[effective_session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content=(
                    f"[progress] Background task '{display_label}' started "
                    f"(id: {task_id}). I will report progress here."
                ),
                metadata={
                    **dict(origin_metadata or {}),
                    "progress": True,
                },
            )
        )
        # Keep a tool result for the orchestrator; user-facing notice is sent above.
        return f"Background task queued (id: {task_id})."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, Any],
        started_monotonic: float,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
            if self._should_register_nanobot_web_tools():
                tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
                tools.register(WebFetchTool(proxy=self.web_proxy))

            system_prompt = self._build_subagent_prompt()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations) with model fallback support.
            final_result: str | None = None
            model_candidates = self._model_candidates()
            last_model_error = ""
            for model_name in model_candidates:
                logger.info("Subagent [{}] using worker model: {}", task_id, model_name)
                max_iterations = 15
                iteration = 0
                run_messages = list(messages)
                while iteration < max_iterations:
                    iteration += 1

                    response = await self.provider.chat(
                        messages=run_messages,
                        tools=tools.get_definitions(),
                        model=model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        reasoning_effort=self.reasoning_effort,
                    )
                    logger.info(
                        "Subagent [{}] model={} iter={} finish_reason={} tool_calls={} content_len={}",
                        task_id,
                        model_name,
                        iteration,
                        response.finish_reason,
                        len(response.tool_calls),
                        len(response.content or ""),
                    )

                    if response.finish_reason == "error":
                        last_model_error = response.content or "Worker provider returned error"
                        logger.warning(
                            "Subagent [{}] worker model error on {}: {}; trying fallback",
                            task_id,
                            model_name,
                            (response.content or "").strip()[:200],
                        )
                        break

                    if response.has_tool_calls:
                        # Add assistant message with tool calls
                        tool_call_dicts = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                                },
                            }
                            for tc in response.tool_calls
                        ]
                        run_messages.append({
                            "role": "assistant",
                            "content": response.content or "",
                            "tool_calls": tool_call_dicts,
                        })

                        # Execute tools
                        for tool_call in response.tool_calls:
                            args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                            logger.debug("Subagent [{}] executing: {} with arguments: {}", task_id, tool_call.name, args_str)
                            result = await tools.execute(tool_call.name, tool_call.arguments)
                            run_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            })
                    else:
                        final_result = response.content
                        break

                if final_result is not None:
                    break

            if final_result is None and last_model_error:
                raise RuntimeError(last_model_error)

            if not isinstance(final_result, str) or not final_result.strip():
                final_result = "Task completed but no final response was generated."

            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(
                task_id,
                label,
                task,
                final_result,
                origin,
                "ok",
                elapsed_seconds=time.monotonic() - started_monotonic,
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(
                task_id,
                label,
                task,
                error_msg,
                origin,
                "error",
                elapsed_seconds=time.monotonic() - started_monotonic,
            )

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, Any],
        status: str,
        elapsed_seconds: float = 0.0,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"
        elapsed_min = max(1, int(round(elapsed_seconds / 60.0)))

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}
Elapsed: about {elapsed_min} minute(s)

Result:
{result}

This is a delayed background-task callback, not the user's latest request.
Reply in exactly this format:
Background update (earlier task '{label}', {elapsed_min} min): <one short sentence summary>
Then add one optional next step sentence only if action is still needed."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            metadata={
                "origin_metadata": origin.get("metadata") or {},
                "origin_session_key": origin.get("session_key"),
            },
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    async def _heartbeat_loop(
        self,
        task_id: str,
        label: str,
        origin: dict[str, Any],
        started_monotonic: float,
        bg_task: asyncio.Task[None],
    ) -> None:
        """Emit periodic progress heartbeats while a background task is running."""
        try:
            heartbeat_index = 0
            while not bg_task.done():
                # Allow shorter configured intervals for the first few pings while
                # keeping the capped backoff schedule in _heartbeat_delay_seconds.
                delay = min(
                    self.heartbeat_interval_seconds,
                    self._heartbeat_delay_seconds(heartbeat_index),
                )
                await asyncio.sleep(delay)
                if bg_task.done():
                    break
                elapsed_min = max(1, int((time.monotonic() - started_monotonic) // 60))
                metadata = dict(origin.get("metadata") or {})
                metadata["progress"] = True
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=origin["channel"],
                        chat_id=origin["chat_id"],
                        content=(
                            f"[progress] Background task '{label}' is still running "
                            f"(id: {task_id}, ~{elapsed_min} min)."
                        ),
                        metadata=metadata,
                    )
                )
                heartbeat_index += 1
        except asyncio.CancelledError:
            return

    def _build_subagent_prompt(self) -> str:
        """Build a focused system prompt for the subagent."""
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        parts = [f"""# Subagent

{time_ctx}

You are a subagent spawned by the main agent to complete a specific task.
Stay focused on the assigned task. Your final response will be reported back to the main agent.

## Workspace
{self.workspace}"""]

        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        if skills_summary:
            parts.append(f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}")

        parts.append("When you have completed the task, provide a clear summary of your findings or actions.")

        return "\n\n".join(parts)

    def _should_register_nanobot_web_tools(self) -> bool:
        """Return whether nanobot web tools should be registered."""
        model_name = (self.model or "").strip().lower()
        provider_name = self.provider.__class__.__name__
        if model_name.startswith("openai-codex/") or model_name.startswith("claude-agent/"):
            return False
        if provider_name in _NATIVE_SDK_PROVIDER_CLASS_NAMES:
            return False
        return True

    def _model_candidates(self) -> list[str]:
        """Get ordered worker model candidates with deduplication."""
        ordered = [self.model] + self.fallback_models
        seen: set[str] = set()
        result: list[str] = []
        for raw in ordered:
            name = (raw or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
        return result

    def _heartbeat_delay_seconds(self, heartbeat_index: int) -> int:
        """Heartbeat delay schedule: 1m x2, 2m x2, 4m x2, ... cap at 10m."""
        if heartbeat_index < 0:
            heartbeat_index = 0
        stage = heartbeat_index // 2
        # Stage minutes sequence: 1,2,4,6,8,10,10,...
        stage_minutes = [1, 2, 4, 6, 8, 10]
        minutes = stage_minutes[min(stage, len(stage_minutes) - 1)]
        return max(5, minutes * 60, self.heartbeat_interval_seconds)

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
