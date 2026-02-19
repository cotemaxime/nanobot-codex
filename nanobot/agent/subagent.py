"""Subagent manager for background task execution."""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool

_CODEX_PROVIDER_CLASS_NAMES = {"OpenAICodexProvider", "CodexSDKProvider"}


class SubagentManager:
    """
    Manages background subagent execution.
    
    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
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
        self.brave_api_key = brave_api_key
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
        self.heartbeat_interval_seconds = max(5, int(heartbeat_interval_seconds or 30))
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
    
    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        origin_metadata: dict[str, Any] | None = None,
        origin_session_key: str | None = None,
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.
        
        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.
        
        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        
        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "metadata": dict(origin_metadata or {}),
            "session_key": origin_session_key,
        }
        
        # Create background task
        started_monotonic = time.monotonic()
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, started_monotonic)
        )
        self._running_tasks[task_id] = bg_task
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(task_id, display_label, origin, started_monotonic, bg_task)
        )
        
        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))
        bg_task.add_done_callback(lambda _: heartbeat_task.cancel())

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
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
        logger.info(f"Subagent [{task_id}] starting task: {label}")
        
        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            skill_roots = [self.workspace / "skills", BUILTIN_SKILLS_DIR]
            tools.register(
                ReadFileTool(
                    allowed_dir=allowed_dir,
                    blocked_skill_names=self.disabled_skills,
                    skill_roots=skill_roots,
                )
            )
            tools.register(
                WriteFileTool(
                    allowed_dir=allowed_dir,
                    blocked_skill_names=self.disabled_skills,
                    skill_roots=skill_roots,
                )
            )
            tools.register(
                EditFileTool(
                    allowed_dir=allowed_dir,
                    blocked_skill_names=self.disabled_skills,
                    skill_roots=skill_roots,
                )
            )
            tools.register(
                ListDirTool(
                    allowed_dir=allowed_dir,
                    blocked_skill_names=self.disabled_skills,
                    skill_roots=skill_roots,
                )
            )
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ))
            if self._should_register_nanobot_web_tools():
                tools.register(WebSearchTool(api_key=self.brave_api_key))
                tools.register(WebFetchTool())
            
            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]
            
            # Run agent loop (limited iterations) with model fallback support.
            final_result: str | None = None
            model_candidates = self._model_candidates()
            last_model_error = ""
            for model_name in model_candidates:
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
                    )

                    if (
                        response.finish_reason == "error"
                        and self._is_unsupported_model_error(response.content)
                    ):
                        last_model_error = response.content or ""
                        logger.warning(
                            f"Subagent [{task_id}] worker model unsupported: {model_name}; trying fallback"
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
                                    "arguments": json.dumps(tc.arguments),
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
                            args_str = json.dumps(tool_call.arguments)
                            logger.debug(f"Subagent [{task_id}] executing: {tool_call.name} with arguments: {args_str}")
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
            
            logger.info(f"Subagent [{task_id}] completed successfully")
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
            logger.error(f"Subagent [{task_id}] failed: {e}")
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
        elapsed_seconds: float,
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
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")

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
                await asyncio.sleep(self._heartbeat_delay_seconds(heartbeat_index))
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
    
    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""

    def _should_register_nanobot_web_tools(self) -> bool:
        """Return whether nanobot web tools should be registered."""
        model_name = (self.model or "").strip().lower()
        provider_name = self.provider.__class__.__name__
        if model_name.startswith("openai-codex/"):
            return False
        if provider_name in _CODEX_PROVIDER_CLASS_NAMES:
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

    @staticmethod
    def _is_unsupported_model_error(content: str | None) -> bool:
        text = (content or "").lower()
        return "model is not supported" in text and "codex" in text

    def _heartbeat_delay_seconds(self, heartbeat_index: int) -> int:
        """Heartbeat delay schedule: 1m x2, 2m x2, 4m x2, ... cap at 10m."""
        if heartbeat_index < 0:
            heartbeat_index = 0
        stage = heartbeat_index // 2
        # Stage minutes sequence: 1,2,4,6,8,10,10,...
        stage_minutes = [1, 2, 4, 6, 8, 10]
        minutes = stage_minutes[min(stage, len(stage_minutes) - 1)]
        return max(5, minutes * 60, self.heartbeat_interval_seconds)
    
    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
