from pathlib import Path

from nanobot.agent.skills import SkillsLoader


def _write_skill(root: Path, name: str, body: str = "# Skill\n\nDemo") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def test_disabled_skills_are_hidden_from_list(tmp_path):
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    (workspace / "skills").mkdir(parents=True, exist_ok=True)
    builtin.mkdir(parents=True, exist_ok=True)

    _write_skill(workspace / "skills", "clawhub")
    _write_skill(workspace / "skills", "weather")
    _write_skill(builtin, "github")
    _write_skill(builtin, "clawhub")

    loader = SkillsLoader(
        workspace=workspace,
        builtin_skills_dir=builtin,
        disabled_skills=["ClAwHuB"],
    )

    names = sorted(s["name"] for s in loader.list_skills(filter_unavailable=False))
    assert names == ["github", "weather"]


def test_disabled_skill_cannot_be_loaded(tmp_path):
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    _write_skill(workspace / "skills", "clawhub", body="# ClawHub\n\nSensitive")
    _write_skill(builtin, "github", body="# GitHub\n\nSafe")

    loader = SkillsLoader(
        workspace=workspace,
        builtin_skills_dir=builtin,
        disabled_skills=["clawhub"],
    )

    assert loader.load_skill("clawhub") is None
    assert loader.load_skill("github")
