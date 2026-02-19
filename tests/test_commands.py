import shutil
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

import pytest
from click.exceptions import Exit

from nanobot.cli.commands import _is_gpt52_planner_mode, _make_codex_worker_provider, _make_provider, app
from nanobot.config.schema import Config

runner = CliRunner()


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.utils.helpers.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_status_shows_oauth_provider_line(monkeypatch, tmp_path):
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5-codex"

    fake_config_path = tmp_path / "config.json"
    fake_config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: fake_config_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Model: openai-codex/gpt-5-codex" in result.stdout
    assert "OpenAI Codex: ✓ (OAuth)" in result.stdout


def test_make_provider_surfaces_provider_creation_error(monkeypatch):
    cfg = Config()
    cfg.agents.defaults.model = "anthropic/claude-3-5-haiku"
    cfg.providers.anthropic.api_key = "test-key"

    monkeypatch.setattr(
        "nanobot.providers.litellm_provider.LiteLLMProvider.__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider init failed")),
    )

    with pytest.raises(RuntimeError, match="provider init failed"):
        _make_provider(cfg)


def test_is_gpt52_planner_mode():
    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5.2"
    assert _is_gpt52_planner_mode(cfg) is True

    cfg.agents.defaults.model = "openai-codex/gpt-5-codex"
    assert _is_gpt52_planner_mode(cfg) is False


def test_make_codex_worker_provider_uses_configured_model(monkeypatch, tmp_path):
    captured = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("nanobot.providers.codex_sdk_provider.CodexSDKProvider", FakeWorker)

    cfg = Config()
    cfg.agents.defaults.model = "openai-codex/gpt-5.2"
    cfg.agents.defaults.workspace = str(tmp_path)
    cfg.agents.codex_worker.model = "openai-codex/gpt-5.3-codex"
    cfg.agents.codex_worker.sandbox_mode = "danger-full-access"
    cfg.agents.codex_worker.approval_policy = "never"
    cfg.agents.codex_worker.network_access_enabled = True
    cfg.agents.codex_worker.web_search_enabled = True

    worker = _make_codex_worker_provider(cfg)

    assert worker is not None
    assert captured["default_model"] == "openai-codex/gpt-5.3-codex"
    assert captured["sandbox_mode"] == "danger-full-access"
    assert captured["approval_policy"] == "never"
