"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from loguru import logger
from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import TelegramConfig

TELEGRAM_MAX_MESSAGE_CHARS = 3800


def _split_message(text: str, max_chars: int = TELEGRAM_MAX_MESSAGE_CHARS) -> list[str]:
    """Split long text into Telegram-safe chunks, preferring paragraph/newline boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n\n", 0, max_chars)
        if split_at < 0:
            split_at = remaining.rfind("\n", 0, max_chars)
        if split_at < 0:
            split_at = max_chars
        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:max_chars]
            split_at = max_chars
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""
    
    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []
    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"
    
    text = re.sub(r'```[\w]*\n?([\s\S]*?)```', save_code_block, text)
    
    # 2. Extract and protect inline code
    inline_codes: list[str] = []
    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"
    
    text = re.sub(r'`([^`]+)`', save_inline_code, text)
    
    # 3. Headers # Title -> just the title text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)
    
    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    
    # 7. Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    
    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'<i>\1</i>', text)
    
    # 9. Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    
    # 10. Bullet lists - item -> • item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")
    
    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")
    
    return text


def _split_message(content: str, max_len: int = 4000) -> list[str]:
    """Split content into chunks within max_len, preferring line breaks."""
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        pos = cut.rfind('\n')
        if pos == -1:
            pos = cut.rfind(' ')
        if pos == -1:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.
    
    Simple and reliable - no webhook/public IP needed.
    """
    
    name = "telegram"
    
    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("help", "Show available commands"),
        BotCommand("last", "Resend last assistant response"),
        BotCommand("skills", "List available skills"),
        BotCommand("skill", "Configure active skills for this chat/topic"),
        BotCommand("model", "Configure model override for this chat/topic"),
    ]
    
    def __init__(
        self,
        config: TelegramConfig,
        bus: MessageBus,
        groq_api_key: str = "",
    ):
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self.groq_api_key = groq_api_key
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing_tasks: dict[str, asyncio.Task] = {}  # chat_id -> typing loop task
        self._reaction_handler_mode: str = "none"  # none | filters | fallback
        self._message_thread_ids: dict[tuple[str, int], int] = {}  # (chat_id, message_id) -> thread_id
        self._progress_message_ids: dict[tuple[int, int | None], int] = {}  # (chat_id, thread_id) -> message_id
    
    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return
        
        self._running = True
        
        # Build the application with larger connection pool to avoid pool-timeout on long runs
        req = HTTPXRequest(connection_pool_size=16, pool_timeout=5.0, connect_timeout=30.0, read_timeout=30.0)
        builder = Application.builder().token(self.config.token).request(req).get_updates_request(req)
        if self.config.proxy:
            builder = builder.proxy(self.config.proxy).get_updates_proxy(self.config.proxy)
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)
        
        # Add command handlers
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("new", self._forward_command))
        self._app.add_handler(CommandHandler("help", self._forward_command))
        self._app.add_handler(CommandHandler("last", self._forward_command))
        self._app.add_handler(CommandHandler("skills", self._forward_command))
        self._app.add_handler(CommandHandler("skill", self._forward_command))
        self._app.add_handler(CommandHandler("model", self._forward_command))
        
        # Add message handler for text, photos, voice, documents
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO | filters.Document.ALL) 
                & ~filters.COMMAND, 
                self._on_message
            )
        )
        # Reaction updates (Telegram Bot API update types: message_reaction / message_reaction_count)
        self._register_reaction_handlers()
        
        logger.info("Starting Telegram bot (polling mode)...")
        
        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()
        
        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        logger.info(f"Telegram bot @{bot_info.username} connected")
        
        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            logger.debug("Telegram bot commands registered")
        except Exception as e:
            logger.warning(f"Failed to register bot commands: {e}")
        
        # Start polling (this runs until stopped)
        await self._app.updater.start_polling(
            allowed_updates=self._allowed_updates(),
            drop_pending_updates=True  # Ignore old messages on startup
        )
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        
        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)
        
        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None
    
    @staticmethod
    def _get_media_type(path: str) -> str:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp"):
            return "photo"
        if ext == "ogg":
            return "voice"
        if ext in ("mp3", "m4a", "wav", "aac"):
            return "audio"
        return "document"

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        if not self._app:
            logger.warning("Telegram bot not running")
            return

        self._stop_typing(msg.chat_id)

        try:
            chat_id = int(msg.chat_id)
            telegram_meta = (msg.metadata or {}).get("telegram", {})
            thread_id = telegram_meta.get("message_thread_id")
        except ValueError:
            logger.error(f"Invalid chat_id: {msg.chat_id}")
            return
        progress_key = (chat_id, thread_id)
        is_progress = bool((msg.metadata or {}).get("progress"))

        # Stream progress as a single silent, editable status bubble.
        if is_progress and msg.content and msg.content != "[empty message]":
            text = msg.content
            if text.startswith("[progress]"):
                text = text[len("[progress]"):].strip()
            styled = f"⏳ {text or 'Working...'}"
            html = f"<i>{_markdown_to_telegram_html(styled)}</i>"

            existing_message_id = self._progress_message_ids.get(progress_key)
            if existing_message_id is not None:
                try:
                    await self._app.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=existing_message_id,
                        text=html,
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    # Fall back to sending a fresh progress message.
                    self._progress_message_ids.pop(progress_key, None)

            try:
                sent = await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=html,
                    parse_mode="HTML",
                    message_thread_id=thread_id,
                    disable_notification=True,
                )
                mid = getattr(sent, "message_id", None)
                if isinstance(mid, int):
                    self._progress_message_ids[progress_key] = mid
                self._remember_message_thread(
                    chat_id=chat_id,
                    message_id=mid,
                    thread_id=getattr(sent, "message_thread_id", None) or thread_id,
                )
            except Exception as e:
                logger.error(f"Error sending Telegram progress message: {e}")
            return

        # Non-progress reply: stop reusing prior progress message for this thread.
        self._progress_message_ids.pop(progress_key, None)

        # Send media files
        for media_path in (msg.media or []):
            try:
                media_type = self._get_media_type(media_path)
                sender = {
                    "photo": self._app.bot.send_photo,
                    "voice": self._app.bot.send_voice,
                    "audio": self._app.bot.send_audio,
                }.get(media_type, self._app.bot.send_document)
                param = "photo" if media_type == "photo" else media_type if media_type in ("voice", "audio") else "document"
                with open(media_path, 'rb') as f:
                    sent = await sender(chat_id=chat_id, message_thread_id=thread_id, **{param: f})
                    self._remember_message_thread(
                        chat_id=chat_id,
                        message_id=getattr(sent, "message_id", None),
                        thread_id=getattr(sent, "message_thread_id", None) or thread_id,
                    )
            except Exception as e:
                filename = media_path.rsplit("/", 1)[-1]
                logger.error(f"Failed to send media {media_path}: {e}")
                sent = await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=f"[Failed to send: {filename}]",
                    message_thread_id=thread_id,
                )
                self._remember_message_thread(
                    chat_id=chat_id,
                    message_id=getattr(sent, "message_id", None),
                    thread_id=getattr(sent, "message_thread_id", None) or thread_id,
                )

        # Send text content
        if msg.content and msg.content != "[empty message]":
            for chunk in _split_message(msg.content):
                try:
                    html = _markdown_to_telegram_html(chunk)
                    sent = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=html,
                        parse_mode="HTML",
                        message_thread_id=thread_id,
                    )
                    self._remember_message_thread(
                        chat_id=chat_id,
                        message_id=getattr(sent, "message_id", None),
                        thread_id=getattr(sent, "message_thread_id", None) or thread_id,
                    )
                except Exception as e:
                    logger.warning(f"HTML parse failed, falling back to plain text: {e}")
                    try:
                        sent = await self._app.bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            message_thread_id=thread_id,
                        )
                        self._remember_message_thread(
                            chat_id=chat_id,
                            message_id=getattr(sent, "message_id", None),
                            thread_id=getattr(sent, "message_thread_id", None) or thread_id,
                        )
                    except Exception as e2:
                        logger.error(f"Error sending Telegram message: {e2}")
    
    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return
        
        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help to see available commands."
        )
    
    @staticmethod
    def _sender_id(user) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        thread_id = update.message.message_thread_id
        self._remember_message_thread(
            chat_id=update.message.chat_id,
            message_id=update.message.message_id,
            thread_id=thread_id,
        )
        session_key = f"{self.name}:{update.message.chat_id}:{thread_id}" if thread_id else None
        await self._handle_message(
            sender_id=self._sender_id(update.effective_user),
            chat_id=str(update.message.chat_id),
            content=update.message.text,
            metadata={
                "telegram": {
                    "message_thread_id": thread_id,
                },
                "session_key": session_key,
            },
        )
    
    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return
        
        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        sender_id = self._sender_id(user)
        
        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id
        self._remember_message_thread(
            chat_id=chat_id,
            message_id=message.message_id,
            thread_id=message.message_thread_id,
        )
        
        # Build content from text and/or media
        content_parts = []
        media_paths = []
        
        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)
        
        # Handle media files
        media_file = None
        media_type = None
        
        if message.photo:
            media_file = message.photo[-1]  # Largest photo
            media_type = "image"
        elif message.voice:
            media_file = message.voice
            media_type = "voice"
        elif message.audio:
            media_file = message.audio
            media_type = "audio"
        elif message.document:
            media_file = message.document
            media_type = "file"
        
        # Download media if present
        if media_file and self._app:
            try:
                file = await self._app.bot.get_file(media_file.file_id)
                ext = self._get_extension(media_type, getattr(media_file, 'mime_type', None))
                
                # Save to workspace/media/
                from pathlib import Path
                media_dir = Path.home() / ".nanobot" / "media"
                media_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = media_dir / f"{media_file.file_id[:16]}{ext}"
                await file.download_to_drive(str(file_path))
                
                media_paths.append(str(file_path))
                
                # Handle voice transcription
                if media_type == "voice" or media_type == "audio":
                    from nanobot.providers.transcription import GroqTranscriptionProvider
                    transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                    transcription = await transcriber.transcribe(file_path)
                    if transcription:
                        logger.info(f"Transcribed {media_type}: {transcription[:50]}...")
                        content_parts.append(f"[transcription: {transcription}]")
                    else:
                        content_parts.append(f"[{media_type}: {file_path}]")
                else:
                    content_parts.append(f"[{media_type}: {file_path}]")
                    
                logger.debug(f"Downloaded {media_type} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to download media: {e}")
                content_parts.append(f"[{media_type}: download failed]")
        
        content = "\n".join(content_parts) if content_parts else "[empty message]"
        
        logger.debug(f"Telegram message from {sender_id}: {content[:50]}...")
        
        str_chat_id = str(chat_id)
        
        # Start typing indicator before processing
        self._start_typing(str_chat_id)
        
        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.message_id,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_group": message.chat.type != "private",
                "telegram": {
                    "message_thread_id": message.message_thread_id,
                },
                "session_key": f"{self.name}:{str_chat_id}:{message.message_thread_id}" if message.message_thread_id else None,
            }
        )

    @staticmethod
    def _normalize_reaction_item(item: Any) -> str:
        """Normalize Telegram reaction objects to compact text."""
        if item is None:
            return ""
        emoji = getattr(item, "emoji", None)
        if isinstance(emoji, str) and emoji:
            return emoji
        custom_id = getattr(item, "custom_emoji_id", None)
        if isinstance(custom_id, str) and custom_id:
            return f"custom:{custom_id}"
        reaction_type = getattr(item, "type", None)
        if isinstance(reaction_type, str) and reaction_type:
            return reaction_type
        return str(item)

    @staticmethod
    def _as_int(value: Any) -> int | None:
        """Best-effort integer conversion."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _remember_message_thread(self, chat_id: str | int, message_id: Any, thread_id: Any) -> None:
        """Track thread id by message id so reaction updates can be topic-routed."""
        mid = self._as_int(message_id)
        tid = self._as_int(thread_id)
        if mid is None or tid is None:
            return
        self._message_thread_ids[(str(chat_id), mid)] = tid
        # Bound memory usage in long-running bots.
        if len(self._message_thread_ids) > 5000:
            oldest = next(iter(self._message_thread_ids))
            self._message_thread_ids.pop(oldest, None)

    def _resolve_message_thread(self, chat_id: str, message_id: Any, explicit_thread_id: Any) -> int | None:
        """Resolve reaction thread id from update fields or local message map."""
        tid = self._as_int(explicit_thread_id)
        if tid is not None:
            return tid
        mid = self._as_int(message_id)
        if mid is None:
            return None
        return self._message_thread_ids.get((chat_id, mid))

    async def _on_reaction(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Telegram message reaction updates."""
        reaction = update.message_reaction
        if not reaction:
            return

        user = getattr(reaction, "user", None) or update.effective_user
        actor_chat = getattr(reaction, "actor_chat", None)
        sender_id = ""
        username = None
        first_name = None
        if user:
            sender_id = str(getattr(user, "id", ""))
            username = getattr(user, "username", None)
            first_name = getattr(user, "first_name", None)
        elif actor_chat:
            sender_id = str(getattr(actor_chat, "id", ""))
            first_name = getattr(actor_chat, "title", None)
        if username:
            sender_id = f"{sender_id}|{username}"
        if not sender_id:
            return

        chat = getattr(reaction, "chat", None)
        chat_id = str(getattr(chat, "id", "")) if chat else ""
        if not chat_id:
            return

        old_reaction = [self._normalize_reaction_item(i) for i in getattr(reaction, "old_reaction", [])]
        new_reaction = [self._normalize_reaction_item(i) for i in getattr(reaction, "new_reaction", [])]
        old_clean = [r for r in old_reaction if r]
        new_clean = [r for r in new_reaction if r]

        message_id = getattr(reaction, "message_id", None)
        thread_id = self._resolve_message_thread(
            chat_id=chat_id,
            message_id=message_id,
            explicit_thread_id=getattr(reaction, "message_thread_id", None),
        )
        session_key = f"{self.name}:{chat_id}:{thread_id}" if thread_id is not None else None
        actor = first_name or username or sender_id.split("|", 1)[0]
        old_text = ", ".join(old_clean) if old_clean else "(none)"
        new_text = ", ".join(new_clean) if new_clean else "(none)"
        content = (
            f"[telegram_reaction] {actor} reacted on message {message_id}: "
            f"{old_text} -> {new_text}"
        )

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            metadata={
                "event_type": "telegram_message_reaction",
                "message_id": message_id,
                "user_id": getattr(user, "id", None),
                "username": username,
                "first_name": first_name,
                "telegram": {
                    "message_thread_id": thread_id,
                    "reaction_old": old_clean,
                    "reaction_new": new_clean,
                },
                "session_key": session_key,
            },
        )

    async def _on_reaction_count(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Telegram anonymous reaction-count updates."""
        reaction_count = update.message_reaction_count
        if not reaction_count:
            return

        chat = getattr(reaction_count, "chat", None)
        chat_id = str(getattr(chat, "id", "")) if chat else ""
        if not chat_id:
            return

        message_id = getattr(reaction_count, "message_id", None)
        thread_id = self._resolve_message_thread(
            chat_id=chat_id,
            message_id=message_id,
            explicit_thread_id=getattr(reaction_count, "message_thread_id", None),
        )
        session_key = f"{self.name}:{chat_id}:{thread_id}" if thread_id is not None else None
        counts = getattr(reaction_count, "reactions", []) or []
        normalized_counts: list[dict[str, Any]] = []
        parts: list[str] = []
        for item in counts:
            reaction_key = self._normalize_reaction_item(getattr(item, "type", None))
            count = int(getattr(item, "total_count", 0) or 0)
            normalized_counts.append({"reaction": reaction_key, "count": count})
            if reaction_key:
                parts.append(f"{reaction_key} x{count}")

        count_text = ", ".join(parts) if parts else "(none)"
        content = f"[telegram_reaction_count] message {message_id}: {count_text}"

        # Telegram reaction-count updates are anonymous; use synthetic sender id.
        await self._handle_message(
            sender_id=f"telegram_reaction_count:{chat_id}",
            chat_id=chat_id,
            content=content,
            metadata={
                "event_type": "telegram_message_reaction_count",
                "message_id": message_id,
                "telegram": {
                    "message_thread_id": thread_id,
                    "reaction_counts": normalized_counts,
                },
                "session_key": session_key,
            },
        )

    async def _on_reaction_update_fallback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Fallback reaction router for PTB variants without reaction-specific filters.
        """
        if getattr(update, "message_reaction", None):
            await self._on_reaction(update, context)
            return
        if getattr(update, "message_reaction_count", None):
            await self._on_reaction_count(update, context)

    def _register_reaction_handlers(self) -> None:
        """Register reaction handlers only when current PTB version supports them."""
        if not self._app:
            return

        update_type = getattr(filters, "UpdateType", None)
        status_update = getattr(filters, "StatusUpdate", None)
        reaction_filter = None
        reaction_count_filter = None

        if update_type:
            reaction_filter = getattr(update_type, "MESSAGE_REACTION", None)
            reaction_count_filter = getattr(update_type, "MESSAGE_REACTION_COUNT", None)

        # Compatibility fallback for PTB versions that expose reaction filters under StatusUpdate.
        if reaction_filter is None and status_update:
            reaction_filter = getattr(status_update, "MESSAGE_REACTION", None)
        if reaction_count_filter is None and status_update:
            reaction_count_filter = getattr(status_update, "MESSAGE_REACTION_COUNT", None)

        if reaction_filter is not None:
            self._app.add_handler(MessageHandler(reaction_filter, self._on_reaction))
        if reaction_count_filter is not None:
            self._app.add_handler(MessageHandler(reaction_count_filter, self._on_reaction_count))
        if reaction_filter is not None or reaction_count_filter is not None:
            self._reaction_handler_mode = "filters"

        if reaction_filter is None and reaction_count_filter is None:
            # Some PTB versions expose reaction fields on Update but not filter constants.
            try:
                from telegram.ext import TypeHandler
                self._app.add_handler(TypeHandler(Update, self._on_reaction_update_fallback))
                self._reaction_handler_mode = "fallback"
                logger.info(
                    "Telegram reaction filters are unavailable in this python-telegram-bot version; "
                    "using Update-level fallback handler."
                )
            except Exception:
                self._reaction_handler_mode = "none"
                logger.warning(
                    "Telegram reaction updates are not supported by this python-telegram-bot version; "
                    "message_reaction handlers were not registered."
                )

    def _allowed_updates(self) -> list[str]:
        """Build allowed updates list compatible with installed PTB version."""
        updates = ["message"]
        has_message_reaction = hasattr(Update, "message_reaction")
        has_message_reaction_count = hasattr(Update, "message_reaction_count")
        if has_message_reaction or self._reaction_handler_mode in {"filters", "fallback"}:
            updates.append("message_reaction")
        if has_message_reaction_count or self._reaction_handler_mode in {"filters", "fallback"}:
            updates.append("message_reaction_count")
        return updates
    
    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        # Cancel any existing typing task for this chat
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))
    
    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
    
    async def _typing_loop(self, chat_id: str) -> None:
        """Repeatedly send 'typing' action until cancelled."""
        try:
            while self._app:
                await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Typing indicator stopped for {chat_id}: {e}")
    
    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        logger.error(f"Telegram error: {context.error}")

    def _get_extension(self, media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]
        
        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        return type_map.get(media_type, "")
