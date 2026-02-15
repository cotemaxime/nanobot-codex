"""Provider factory to keep provider selection isolated from CLI logic."""

from __future__ import annotations

from nanobot.providers.codex_sdk_provider import CodexSDKProvider
from nanobot.providers.litellm_provider import LiteLLMProvider


def is_codex_model(model: str) -> bool:
    return model.lower().startswith("codex/")


def use_codex_provider(config) -> bool:
    model = config.agents.defaults.model or ""
    codex_cfg = getattr(config.providers, "codex", None)
    return bool(codex_cfg and (codex_cfg.enabled or is_codex_model(model)))


def create_provider(config):
    """Create an LLM provider from config using codex-first explicit routing."""
    model = config.agents.defaults.model

    if use_codex_provider(config):
        codex_cfg = config.providers.codex
        # Single source of truth: Codex always uses agent workspace.
        workspace = str(config.workspace_path)
        return CodexSDKProvider(
            default_model=model,
            profile=codex_cfg.profile,
            workspace=workspace,
            timeout_seconds=codex_cfg.timeout_seconds,
            max_internal_native_steps=codex_cfg.max_internal_native_steps,
            strict_auth=codex_cfg.strict_auth,
            skip_git_repo_check=codex_cfg.skip_git_repo_check,
            native_tools=codex_cfg.native_tools,
        )

    p = config.get_provider()
    if not (p and p.api_key) and not model.startswith("bedrock/"):
        raise RuntimeError(
            "No API key configured. Set one in ~/.nanobot/config.json under providers section"
        )

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=config.get_provider_name(),
    )
