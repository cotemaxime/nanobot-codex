"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    @staticmethod
    def _normalize_history_entry(entry: str) -> str:
        """Normalize one history entry for storage and dedupe checks."""
        text = (entry or "").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _canonicalize_history_entry(entry: str) -> str:
        """Build a fuzzy dedupe key that ignores timestamp and punctuation noise."""
        text = MemoryStore._normalize_history_entry(entry).lower()
        text = re.sub(r"^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\]\s*", "", text)
        text = re.sub(r"[^a-z0-9]+", " ", text).strip()
        return text

    @staticmethod
    def _is_low_signal_history_entry(entry: str) -> bool:
        """Drop repetitive low-information operational summaries from history."""
        text = MemoryStore._normalize_history_entry(entry)
        if not text:
            return True
        lower = text.lower()
        low_signal_markers = (
            "heartbeat_ok",
            "read heartbeat.md",
            "nothing needs attention",
            "nothing required attention",
            "no new conversation content",
            "no additional tasks",
            "no additional task outcomes",
            "no additional user preferences",
            "no further task outcomes",
        )
        return any(marker in lower for marker in low_signal_markers)

    @staticmethod
    def sanitize_history_content(content: str) -> tuple[str, dict[str, int]]:
        """Remove low-signal and duplicate entries from full HISTORY.md content."""
        raw = (content or "").strip()
        if not raw:
            return "", {"total": 0, "kept": 0, "dropped_low_signal": 0, "dropped_duplicate": 0}

        chunks = [c.strip() for c in re.split(r"\n\s*\n+", raw) if c.strip()]
        kept: list[str] = []
        seen: set[str] = set()
        dropped_low_signal = 0
        dropped_duplicate = 0

        for chunk in chunks:
            entry = MemoryStore._normalize_history_entry(chunk)
            if MemoryStore._is_low_signal_history_entry(entry):
                dropped_low_signal += 1
                continue
            key = MemoryStore._canonicalize_history_entry(entry)
            if key and key in seen:
                dropped_duplicate += 1
                continue
            if key:
                seen.add(key)
            kept.append(entry)

        cleaned = "\n\n".join(kept).strip()
        if cleaned:
            cleaned += "\n\n"
        stats = {
            "total": len(chunks),
            "kept": len(kept),
            "dropped_low_signal": dropped_low_signal,
            "dropped_duplicate": dropped_duplicate,
        }
        return cleaned, stats

    def append_history(self, entry: str) -> None:
        cleaned = self._normalize_history_entry(entry)
        if self._is_low_signal_history_entry(cleaned):
            logger.debug("History append skipped: low-signal entry")
            return

        key = self._canonicalize_history_entry(cleaned)
        if self.history_file.exists():
            try:
                existing = self.history_file.read_text(encoding="utf-8")
                recent = [c.strip() for c in re.split(r"\n\s*\n+", existing) if c.strip()][-50:]
                recent_keys = {self._canonicalize_history_entry(item) for item in recent}
                if key and key in recent_keys:
                    logger.debug("History append skipped: duplicate of recent entry")
                    return
            except Exception:
                logger.exception("History dedupe check failed; continuing append")

        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(cleaned + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _is_volatile_memory_heading(heading: str) -> bool:
        """Return whether a markdown section heading looks session/recency-specific."""
        h = (heading or "").strip().lower()
        if not h:
            return False
        if h.startswith(("recent ", "current ", "new ")):
            return True
        markers = (
            "project context",
            "context updates",
            "investigation context",
            "task context",
            "ops context",
            "workflow notes",
            "technical context",
            "todoapp investigation",
            "heartbeat",
        )
        return any(m in h for m in markers)

    @staticmethod
    def _is_volatile_memory_line(line: str) -> bool:
        """Filter obvious volatile bullets/log lines from long-term memory."""
        text = (line or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if re.search(r"^\[20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}\]", text):
            return True
        if re.search(r"^-\s*\(20\d{2}-\d{2}-\d{2}\)", text):
            return True
        if re.search(r"\b(todo|task|job)\s+id\b", lowered):
            return True
        if "in progress" in lowered:
            return True
        return False

    def sanitize_long_term_content(self, content: str, fallback: str = "") -> str:
        """Keep durable memory only; drop volatile status/investigation sections."""
        raw = (content or "").strip()
        if not raw:
            return fallback

        kept: list[str] = []
        drop_block = False
        for line in raw.splitlines():
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                level = len(m.group(1))
                heading = m.group(2).strip()
                if level <= 2:
                    drop_block = self._is_volatile_memory_heading(heading)
            if drop_block:
                continue
            if self._is_volatile_memory_line(line):
                continue
            kept.append(line)

        sanitized = "\n".join(kept).strip()
        if not sanitized:
            return fallback
        return sanitized

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                sanitized = self.sanitize_long_term_content(update, fallback=current_memory)
                if sanitized != current_memory:
                    self.write_long_term(sanitized)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
