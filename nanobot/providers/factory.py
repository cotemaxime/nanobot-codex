"""Provider factory to keep provider selection isolated from CLI logic."""

from __future__ import annotations

from loguru import logger

from nanobot.providers.codex_sdk_provider import CodexSDKProvider
from nanobot.providers.custom_provider import CustomProvider
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import OpenAICodexProvider


def _normalize_sdk_model_name(model: str) -> str:
    """Normalize model id for Codex SDK (uses raw ids like gpt-5-codex)."""
    cleaned = (model or "").strip()
    if cleaned.lower().startswith("openai-codex/"):
        return cleaned.split("/", 1)[1]
    return cleaned


def create_provider(config):
    """Create an LLM provider from config."""
    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    # OpenAI Codex (OAuth)
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        worker_cfg = config.agents.codex_worker
        sdk_model = _normalize_sdk_model_name(model) or _normalize_sdk_model_name(worker_cfg.model) or "gpt-5.3-codex"
        try:
            return CodexSDKProvider(
                default_model=sdk_model,
                workspace=str(config.workspace_path),
                timeout_seconds=worker_cfg.timeout_seconds,
                sandbox_mode=worker_cfg.sandbox_mode,
                approval_policy=worker_cfg.approval_policy,
                network_access_enabled=worker_cfg.network_access_enabled,
                web_search_enabled=worker_cfg.web_search_enabled,
                stream_reader_limit_bytes=worker_cfg.stream_reader_limit_bytes,
                diagnostic_logging=worker_cfg.diagnostic_logging,
            )
        except Exception as e:
            logger.warning(f"Codex SDK provider unavailable, falling back to HTTP bridge: {e}")
            return OpenAICodexProvider(default_model=model)

    # Custom: direct OpenAI-compatible endpoint, bypasses LiteLLM
    if provider_name == "custom":
        return CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    from nanobot.providers.registry import find_by_name

    spec = find_by_name(provider_name)
    if not model.startswith("bedrock/") and not (p and p.api_key) and not (spec and spec.is_oauth):
        raise RuntimeError(
            "No API key configured. Set one in ~/.nanobot/config.json under providers section"
        )

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )
