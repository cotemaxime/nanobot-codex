from nanobot.config.loader import _migrate_config


def test_migrate_moves_disabled_skills_from_agents_defaults():
    data = {
        "agents": {
            "defaults": {
                "workspace": "~/.nanobot/workspace",
                "disabledSkills": ["clawhub"],
            }
        }
    }

    migrated = _migrate_config(data)

    assert migrated["agents"]["disabledSkills"] == ["clawhub"]
    assert "disabledSkills" not in migrated["agents"]["defaults"]


def test_migrate_does_not_override_existing_agents_disabled_skills():
    data = {
        "agents": {
            "disabledSkills": ["github"],
            "defaults": {
                "disabledSkills": ["clawhub"],
            },
        }
    }

    migrated = _migrate_config(data)

    assert migrated["agents"]["disabledSkills"] == ["github"]
    assert "disabledSkills" not in migrated["agents"]["defaults"]

