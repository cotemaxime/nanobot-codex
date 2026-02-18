"""Tool for changing the active model during agent execution."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool


class SetModelTool(Tool):
    """Allow the agent to switch model for the current request/session."""

    def __init__(
        self,
        set_model_callback: Callable[[str, str | None, bool], str | Awaitable[str]],
    ):
        self._set_model_callback = set_model_callback

    @property
    def name(self) -> str:
        return "set_model"

    @property
    def description(self) -> str:
        return (
            "Change model selection. Use action='set' to switch model for this request. "
            "Set persist=true to save as chat/topic override. "
            "Use action='show' to inspect current model or action='clear' to remove override."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "show", "clear"],
                    "description": "set=change model, show=current model, clear=remove override",
                },
                "model": {
                    "type": "string",
                    "description": "Target model id when action is 'set' (e.g. gpt-5.3-codex)",
                },
                "persist": {
                    "type": "boolean",
                    "description": "When true with action='set', saves model override for this chat/topic",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        model: str | None = None,
        persist: bool = False,
        **kwargs: Any,
    ) -> str:
        result = self._set_model_callback(action, model, persist)
        if hasattr(result, "__await__"):
            return await result
        return str(result)

