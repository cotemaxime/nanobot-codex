"""Agent loop: the core processing engine."""

import asyncio
from contextvars import ContextVar, Token
from contextlib import suppress
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.model import SetModelTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    MODEL_CHOICES = [
        "openai-codex/gpt-5.1-codex",
        "openai-codex/gpt-5-codex",
        "openai-codex/gpt-5-codex-mini",
    ]
    PROGRESS_PING_STEP_INTERVAL_S = 120
    PROGRESS_PING_MAX_INTERVAL_S = 10 * 60
    PROGRESS_PING_REPEATS_PER_STEP = 2
    REACTION_APPROVE = {"👍", "✅", "☑️", "👌"}
    REACTION_RETRY = {"🔁", "🔄", "⟳"}
    REACTION_REDO = {"♻️", "↩️", "↪️"}

    @staticmethod
    def _reaction_matches(tokens: set[str], candidates: set[str]) -> bool:
        """Match reaction tokens, allowing emoji variation/skin-tone suffixes."""
        for token in tokens:
            if any(token == c or token.startswith(c) for c in candidates):
                return True
        return False

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._active_session_ctx: ContextVar[Session | None] = ContextVar(
            "active_session",
            default=None,
        )
        self._active_model_ctx: ContextVar[str | None] = ContextVar(
            "active_model",
            default=None,
        )
        self._active_route_ctx: ContextVar[tuple[str, str] | None] = ContextVar(
            "active_route",
            default=None,
        )
        self._session_queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        self._session_workers: dict[str, asyncio.Task[None]] = {}
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(
            send_callback=self.bus.publish_outbound,
            context_getter=self._get_active_route,
        )
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(
            manager=self.subagents,
            context_getter=self._get_active_route,
        )
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(
                self.cron_service,
                context_getter=self._get_active_route,
            ))

        # Model routing tool (allows agent-driven per-request / per-session model switch)
        self.tools.register(SetModelTool(set_model_callback=self._set_model_from_tool))
    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    def _get_active_route(self) -> tuple[str, str] | None:
        """Get current routing context for context-aware tools."""
        return self._active_route_ctx.get()

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        model: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1
            current_model = self._active_model_ctx.get() or model or self.model

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=current_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break

        return final_content, tools_used

    def _set_model_from_tool(self, action: str, model: str | None, persist: bool) -> str:
        """Apply model switch requested via set_model tool."""
        action = (action or "").strip().lower()
        session = self._active_session_ctx.get()
        current = self._active_model_ctx.get() or self.model

        if action == "show":
            session_model = session.metadata.get("model_override") if session else None
            if session_model:
                return f"Current model: {current} (session override: {session_model})"
            return f"Current model: {current} (using default)"

        if action == "clear":
            self._active_model_ctx.set(self.model)
            if session:
                session.metadata.pop("model_override", None)
                self.sessions.save(session)
            return f"Model reset to default: {self.model}"

        if action != "set":
            return "Error: action must be one of: set, show, clear"

        target = (model or "").strip()
        if not target:
            return "Error: model is required when action='set'"

        self._active_model_ctx.set(target)

        if persist and session:
            session.metadata["model_override"] = target
            self.sessions.save(session)
            return (
                f"Model switched from {current} to {target} for this request "
                "and saved for this chat/topic."
            )

        return f"Model switched from {current} to {target} for this request."

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(
                        self.bus.consume_inbound(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                session_key = self._resolve_session_key(msg, None)
                queue = self._session_queues.setdefault(session_key, asyncio.Queue())
                await queue.put(msg)
                self._ensure_session_worker(session_key)
        finally:
            await self._shutdown_session_workers()

    def _ensure_session_worker(self, session_key: str) -> None:
        """Ensure a per-session worker exists so sessions can run in parallel."""
        worker = self._session_workers.get(session_key)
        if worker is None or worker.done():
            queue = self._session_queues[session_key]
            self._session_workers[session_key] = asyncio.create_task(
                self._session_worker(session_key, queue)
            )

    async def _session_worker(
        self,
        session_key: str,
        queue: asyncio.Queue[InboundMessage],
    ) -> None:
        """Process one session queue serially, while other sessions run concurrently."""
        try:
            while self._running or not queue.empty():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                await self._process_inbound_message(msg)
                if queue.empty():
                    break
        finally:
            if self._session_workers.get(session_key) is asyncio.current_task():
                self._session_workers.pop(session_key, None)
            if queue.empty():
                self._session_queues.pop(session_key, None)
            elif self._running:
                self._session_workers[session_key] = asyncio.create_task(
                    self._session_worker(session_key, queue)
                )

    async def _process_inbound_message(self, msg: InboundMessage) -> None:
        """Process one inbound message and publish outbound responses/errors."""
        ping_task: asyncio.Task | None = None
        try:
            if msg.channel != "system":
                ping_task = asyncio.create_task(self._progress_ping_loop(msg))
            response = await self._process_message(msg)
            if response:
                session_key = self._resolve_session_key(msg, None)
                if (
                    msg.channel != "system"
                    and self.bus.has_pending_inbound_for_session(session_key)
                ):
                    response.content = (
                        "Result for your previous message:\n\n"
                        f"{response.content}\n\n"
                        "I already received a newer message from you and will process it next."
                    )
                await self.bus.publish_outbound(response)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Sorry, I encountered an error: {str(e)}",
                metadata=msg.metadata or {},
            ))
        finally:
            if ping_task:
                ping_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ping_task

    async def _shutdown_session_workers(self) -> None:
        """Cancel and await all active session workers."""
        workers = list(self._session_workers.values())
        self._session_workers.clear()
        self._session_queues.clear()
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _progress_ping_loop(self, msg: InboundMessage) -> None:
        """Send periodic progress pings while a request is still running."""

        current_interval = self.PROGRESS_PING_STEP_INTERVAL_S
        repeats = 0

        while True:
            await asyncio.sleep(current_interval)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Still working on this. I will send the result when it is ready.",
                    metadata=msg.metadata or {},
                )
            )

            repeats += 1
            if repeats >= self.PROGRESS_PING_REPEATS_PER_STEP:
                repeats = 0
                if current_interval < self.PROGRESS_PING_MAX_INTERVAL_S:
                    current_interval = min(
                        current_interval + self.PROGRESS_PING_STEP_INTERVAL_S,
                        self.PROGRESS_PING_MAX_INTERVAL_S,
                    )
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        response_metadata = msg.metadata or {}

        def _reply(content: str) -> OutboundMessage:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=response_metadata,
            )

        key = self._resolve_session_key(msg, session_key)
        session = self.sessions.get_or_create(key)

        # Telegram reaction events: persist event and trigger deterministic actions.
        event_type = ""
        if isinstance(msg.metadata, dict):
            raw_event_type = msg.metadata.get("event_type")
            if isinstance(raw_event_type, str):
                event_type = raw_event_type
        if msg.channel == "telegram" and event_type in {
            "telegram_message_reaction",
            "telegram_message_reaction_count",
        }:
            return await self._handle_telegram_reaction_event(msg, session, key, event_type)
        
        # Handle slash commands
        raw_cmd = msg.content.strip()
        cmd = raw_cmd.lower()
        cmd_token = cmd.split()[0] if cmd else ""
        cmd_base = cmd_token.split("@", 1)[0] if cmd_token.startswith("/") else cmd_token
        if cmd_base == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return _reply("New session started. Memory consolidation in progress.")
        if cmd_base == "/help":
            return _reply(
                "🐈 nanobot commands:\n"
                "/new — Start a new conversation\n"
                "/help — Show available commands\n"
                "/last — Resend the last assistant response\n"
                "/skills — List available skills\n"
                "/skill — Configure active skills for this chat/topic\n"
                "/model — Configure model override for this chat/topic"
            )
        if cmd_base == "/last":
            last_assistant = next(
                (m.get("content", "") for m in reversed(session.messages) if m.get("role") == "assistant"),
                "",
            )
            if not last_assistant:
                return _reply("No previous assistant response found for this chat yet.")
            return _reply(last_assistant)
        if cmd_base == "/skills":
            available = sorted([s["name"] for s in self.context.skills.list_skills()])
            if not available:
                return _reply("No skills are available.")
            listing = "\n".join(f"{i}. {name}" for i, name in enumerate(available, 1))
            return _reply("Please choose a skill:\n" + listing)
        if cmd_base == "/skill":
            available = sorted([s["name"] for s in self.context.skills.list_skills()])
            session.metadata["pending_action"] = "set_skills"
            session.metadata["pending_skill_choices"] = available
            self.sessions.save(session)
            active = session.metadata.get("active_skills", [])
            active_line = ", ".join(active) if active else "(none)"
            listing = "\n".join(f"{i}. {name}" for i, name in enumerate(available, 1))
            return _reply(
                f"Current active skills: {active_line}\n"
                "Please choose skills by number (comma-separated for multiple):\n"
                f"{listing}\n"
                "Send `0` or `cancel` to cancel.\n"
                "Send `clear` to disable skills in this chat/topic.\n"
                "Send `list` to see available skills."
            )
        if cmd_base == "/model":
            model_choices = list(self.MODEL_CHOICES)
            current_model = session.metadata.get("model_override") or self.model
            if current_model not in model_choices:
                model_choices.insert(0, current_model)
            session.metadata["pending_action"] = "set_model"
            session.metadata["pending_model_choices"] = model_choices
            self.sessions.save(session)
            listing = "\n".join(f"{i}. {name}" for i, name in enumerate(model_choices, 1))
            return _reply(
                f"Current model for this chat/topic: {current_model}\n"
                "Please choose a model:\n"
                f"{listing}\n"
                "Send a number only.\n"
                "Send `0` or `cancel` to cancel.\n"
                "Send `clear` to use global default again.\n"
                "Send `show` to view current model."
            )

        pending_action = session.metadata.get("pending_action")
        if pending_action == "set_skills":
            available_list = session.metadata.get("pending_skill_choices") or sorted(
                [s["name"] for s in self.context.skills.list_skills()]
            )
            text = raw_cmd.strip()
            if text.lower() in {"0", "cancel"}:
                session.metadata.pop("pending_action", None)
                session.metadata.pop("pending_skill_choices", None)
                self.sessions.save(session)
                return _reply("Skill update canceled.")
            if text.lower() == "list":
                listing = "\n".join(f"{i}. {name}" for i, name in enumerate(available_list, 1))
                return _reply("Please choose skills:\n" + listing)
            if text.lower() in {"clear", "none"}:
                session.metadata.pop("active_skills", None)
                session.metadata.pop("pending_action", None)
                session.metadata.pop("pending_skill_choices", None)
                self.sessions.save(session)
                return _reply("Active skills cleared.")

            requested: list[str] = []
            for part in [s.strip() for s in text.split(",") if s.strip()]:
                if not part.isdigit():
                    return _reply(
                        "Invalid selection. Send skill numbers only (for example: `1` or `1,3`)."
                    )
                idx = int(part)
                if 1 <= idx <= len(available_list):
                    requested.append(available_list[idx - 1])
                else:
                    return _reply(
                        f"Invalid skill number: {part}. Please choose 1-{len(available_list)}."
                    )

            deduped: list[str] = []
            for name in requested:
                if name not in deduped:
                    deduped.append(name)

            requested = deduped
            session.metadata["active_skills"] = requested
            session.metadata.pop("pending_action", None)
            session.metadata.pop("pending_skill_choices", None)
            self.sessions.save(session)
            return _reply(f"Active skills set: {', '.join(requested) if requested else '(none)'}")

        if pending_action == "set_model":
            text = raw_cmd.strip()
            model_choices = session.metadata.get("pending_model_choices") or []
            if text.lower() in {"0", "cancel"}:
                session.metadata.pop("pending_action", None)
                session.metadata.pop("pending_model_choices", None)
                self.sessions.save(session)
                return _reply("Model update canceled.")
            if text.lower() == "show":
                current = session.metadata.get("model_override") or self.model
                return _reply(f"Current model: {current}")
            if text.lower() in {"clear", "default"}:
                session.metadata.pop("model_override", None)
                session.metadata.pop("pending_action", None)
                session.metadata.pop("pending_model_choices", None)
                self.sessions.save(session)
                return _reply(f"Model reset to default: {self.model}")

            if not text.isdigit():
                return _reply(
                    "Invalid selection. Send a model number only (for example: `1` or `2`)."
                )

            selected = text
            if model_choices:
                idx = int(text)
                if 1 <= idx <= len(model_choices):
                    selected = model_choices[idx - 1]
                else:
                    return _reply(f"Invalid model number: {text}. Please choose 1-{len(model_choices)}.")

            session.metadata["model_override"] = selected
            session.metadata.pop("pending_action", None)
            session.metadata.pop("pending_model_choices", None)
            self.sessions.save(session)
            return _reply(f"Model override set to: {selected}")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        active_skills = session.metadata.get("active_skills")
        model_for_session = session.metadata.get("model_override") or self.model
        session_token: Token = self._active_session_ctx.set(session)
        model_token: Token = self._active_model_ctx.set(model_for_session)
        route_token: Token = self._active_route_ctx.set((msg.channel, msg.chat_id))
        try:
            initial_messages = self.context.build_messages(
                history=session.get_history(max_messages=self.memory_window),
                current_message=msg.content,
                skill_names=active_skills if isinstance(active_skills, list) else None,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            final_content, tools_used = await self._run_agent_loop(initial_messages, model=model_for_session)
        finally:
            self._active_route_ctx.reset(route_token)
            self._active_model_ctx.reset(model_token)
            self._active_session_ctx.reset(session_token)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )

    async def _handle_telegram_reaction_event(
        self,
        msg: InboundMessage,
        session: Session,
        session_key: str,
        event_type: str,
    ) -> OutboundMessage | None:
        """Persist telegram reaction events and execute reaction-based controls."""
        session.add_message("user", msg.content)

        # Aggregate count updates are persisted only (no bot response).
        if event_type == "telegram_message_reaction_count":
            self.sessions.save(session)
            return None

        telegram_meta = (msg.metadata or {}).get("telegram", {}) if isinstance(msg.metadata, dict) else {}
        reaction_new = telegram_meta.get("reaction_new")
        if not isinstance(reaction_new, list):
            reaction_new = []
        reactions = {str(item) for item in reaction_new if item}

        # 👍 = mark as finished/approved for follow-up parsing.
        if self._reaction_matches(reactions, self.REACTION_APPROVE):
            approved = session.metadata.get("approved_events")
            if not isinstance(approved, list):
                approved = []
            approved.append({
                "message_id": (msg.metadata or {}).get("message_id") if isinstance(msg.metadata, dict) else None,
                "reaction": sorted(reactions),
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            })
            # Keep metadata bounded.
            session.metadata["approved_events"] = approved[-200:]
            text = "Acknowledged. Marked this as completed."
            session.add_message("assistant", text, reaction_control=True)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=text,
                metadata=msg.metadata or {},
            )

        # ♻️ / ↩️ = resend last assistant response.
        if self._reaction_matches(reactions, self.REACTION_REDO):
            last_assistant = next(
                (
                    m.get("content", "")
                    for m in reversed(session.messages[:-1])
                    if m.get("role") == "assistant" and not m.get("reaction_control")
                ),
                "",
            )
            text = last_assistant or "No previous assistant response found to redo."
            if last_assistant:
                session.add_message("assistant", text)
            else:
                session.add_message("assistant", text, reaction_control=True)
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=text,
                metadata=msg.metadata or {},
            )

        # 🔁 / 🔄 = retry last actionable user request with a fresh run.
        if self._reaction_matches(reactions, self.REACTION_RETRY):
            last_user = ""
            for m in reversed(session.messages[:-1]):
                if m.get("role") != "user":
                    continue
                content = str(m.get("content", ""))
                if content.startswith("[telegram_reaction"):
                    continue
                last_user = content
                break

            if not last_user:
                text = "No previous user request found to retry."
                session.add_message("assistant", text, reaction_control=True)
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=text,
                    metadata=msg.metadata or {},
                )

            retry_metadata = dict(msg.metadata or {})
            retry_metadata.pop("event_type", None)
            retry_msg = InboundMessage(
                channel=msg.channel,
                sender_id=msg.sender_id,
                chat_id=msg.chat_id,
                content=last_user,
                metadata=retry_metadata,
            )
            self.sessions.save(session)
            return await self._process_message(retry_msg, session_key=session_key)

        self.sessions.save(session)
        return None

    @staticmethod
    def _resolve_session_key(msg: InboundMessage, session_key: str | None) -> str:
        """Resolve effective session key with explicit override support."""
        if session_key:
            return session_key
        metadata_session_key = None
        if isinstance(msg.metadata, dict):
            metadata_session_key = msg.metadata.get("session_key")
        if isinstance(metadata_session_key, str) and metadata_session_key:
            return metadata_session_key
        return msg.session_key
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        model_for_session = session.metadata.get("model_override") or self.model
        session_token: Token = self._active_session_ctx.set(session)
        model_token: Token = self._active_model_ctx.set(model_for_session)
        route_token: Token = self._active_route_ctx.set((origin_channel, origin_chat_id))
        try:
            initial_messages = self.context.build_messages(
                history=session.get_history(max_messages=self.memory_window),
                current_message=msg.content,
                channel=origin_channel,
                chat_id=origin_chat_id,
            )
            final_content, _ = await self._run_agent_loop(initial_messages, model=model_for_session)
        finally:
            self._active_route_ctx.reset(route_token)
            self._active_model_ctx.reset(model_token)
            self._active_session_ctx.reset(session_token)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
