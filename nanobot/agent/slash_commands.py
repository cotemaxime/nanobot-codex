"""Slash command loader for workspace-driven Telegram commands."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger


class SlashCommandsLoader:
    """Load slash command definitions from workspace/slash/*.md files."""

    VALID_COMMAND_RE = re.compile(r"^[a-z0-9_]{1,32}$")

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).expanduser()
        self.slash_dir = self.workspace / "slash"

    def list_commands(self) -> list[dict[str, str]]:
        """Return valid command definitions sorted by command name."""
        commands: list[dict[str, str]] = []
        if not self.slash_dir.exists():
            return commands

        for path in sorted(self.slash_dir.glob("*.md")):
            parsed = self._parse_file(path)
            if parsed is None:
                continue
            commands.append(parsed)
        return sorted(commands, key=lambda item: item["name"])

    def get_command(self, raw_name: str) -> dict[str, str] | None:
        """Get a command by name (with or without leading slash)."""
        name = self._normalize_command_name(raw_name)
        if not name:
            return None
        for command in self.list_commands():
            if command["name"] == name:
                return command
        return None

    @classmethod
    def _normalize_command_name(cls, raw_name: str) -> str | None:
        name = (raw_name or "").strip().lower()
        if name.startswith("/"):
            name = name[1:]
        if not name:
            return None
        if not cls.VALID_COMMAND_RE.fullmatch(name):
            return None
        return name

    def _parse_file(self, path: Path) -> dict[str, str] | None:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed reading slash command file {path}: {e}")
            return None

        metadata, body = self._split_frontmatter(content)
        raw_name = metadata.get("name") or path.stem
        name = self._normalize_command_name(raw_name)
        if not name:
            logger.warning(
                f"Skipping slash command {path.name}: invalid name '{raw_name}' "
                "(must match [a-z0-9_]{1,32})"
            )
            return None

        description = (metadata.get("description") or f"Run /{name} prompt").strip()
        prompt = body.strip()
        if not prompt:
            logger.warning(f"Skipping slash command {path.name}: empty prompt body")
            return None

        return {
            "name": name,
            "description": description,
            "path": str(path),
            "prompt": prompt,
        }

    @staticmethod
    def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
        text = content or ""
        if not text.startswith("---"):
            return {}, text

        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip() != "---":
            return {}, text

        end_idx = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            return {}, text

        metadata: dict[str, str] = {}
        for line in lines[1:end_idx]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip().strip("\"'")

        body = "\n".join(lines[end_idx + 1 :])
        return metadata, body

