from pathlib import Path

from nanobot.agent.tools.filesystem import ListDirTool, ReadFileTool


def _write_skill(root: Path, name: str, body: str = "# Skill\n\nDemo") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(body, encoding="utf-8")
    return skill_file


async def test_read_file_blocks_disabled_skill_paths(tmp_path):
    workspace_skills = tmp_path / "workspace" / "skills"
    builtin_skills = tmp_path / "builtin"
    blocked_file = _write_skill(workspace_skills, "clawhub", body="# ClawHub\n\nSensitive")
    _write_skill(workspace_skills, "weather", body="# Weather\n\nSafe")
    _write_skill(builtin_skills, "clawhub", body="# ClawHub\n\nBuiltin")

    tool = ReadFileTool(
        blocked_skill_names={"clawhub"},
        skill_roots=[workspace_skills, builtin_skills],
    )
    result = await tool.execute(path=str(blocked_file))

    assert "Access denied" in result
    assert "clawhub" in result


async def test_list_dir_blocks_disabled_skill_directory(tmp_path):
    workspace_skills = tmp_path / "workspace" / "skills"
    blocked_dir = workspace_skills / "clawhub"
    blocked_dir.mkdir(parents=True, exist_ok=True)

    tool = ListDirTool(
        blocked_skill_names={"clawhub"},
        skill_roots=[workspace_skills],
    )
    result = await tool.execute(path=str(blocked_dir))

    assert "Access denied" in result
    assert "clawhub" in result

