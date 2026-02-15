import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from click.exceptions import Exit
from typer.testing import CliRunner

from nanobot.cli.commands import app, _make_provider
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


def test_status_shows_codex_session_state(monkeypatch, tmp_path):
    cfg = Config()
    cfg.providers.codex.enabled = True
    cfg.providers.codex.profile = "default"
    cfg.agents.defaults.model = "codex/default"

    fake_config_path = tmp_path / "config.json"
    fake_config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("nanobot.config.loader.get_config_path", lambda: fake_config_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr(
        "nanobot.providers.codex_transport.CodexTransport.probe_session",
        lambda profile=None: (False, "No Codex session detected"),
    )

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Codex: enabled" in result.stdout
    assert "No Codex session detected" in result.stdout


def test_make_provider_surfaces_codex_auth_error(monkeypatch):
    cfg = Config()
    cfg.providers.codex.enabled = True

    monkeypatch.setattr(
        "nanobot.providers.factory.create_provider",
        lambda config: (_ for _ in ()).throw(RuntimeError("Codex session is unavailable")),
    )

    with pytest.raises(Exit) as exc:
        _make_provider(cfg)

    assert exc.value.exit_code == 1
