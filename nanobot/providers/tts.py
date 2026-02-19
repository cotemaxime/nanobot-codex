"""Text-to-speech providers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import httpx
from loguru import logger


class OpenAITTSProvider:
    """OpenAI speech synthesis provider."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.voice = voice
        self.output_dir = Path.home() / ".nanobot" / "media" / "tts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text: str, out_format: Literal["opus", "mp3"]) -> Path | None:
        """Generate speech audio file and return local path."""
        if not text or not text.strip():
            logger.debug("TTS synthesize skipped: empty text")
            return None
        if not self.api_key:
            logger.warning("OpenAI API key not configured for TTS")
            return None

        self._cleanup_old_files()
        output_path = self.output_dir / f"tts_{int(time.time() * 1000)}.{out_format}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/audio/speech",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "voice": self.voice,
                        "input": text,
                        "response_format": out_format,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                output_path.write_bytes(response.content)
                return output_path
        except Exception as e:
            logger.error(f"OpenAI TTS error ({out_format}): {e}")
            return None

    def _cleanup_old_files(self, max_files: int = 100) -> None:
        """Keep output folder bounded in long-running bots."""
        try:
            files = sorted(
                self.output_dir.glob("tts_*.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for stale in files[max_files:]:
                stale.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"TTS cleanup skipped: {e}")
