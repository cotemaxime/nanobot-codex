import os
from pathlib import Path

from nanobot.utils.env import with_nvm_env


def test_with_nvm_env_prefers_existing_nvm_bin(tmp_path: Path):
    nvm_bin = tmp_path / "nvm-bin"
    nvm_bin.mkdir()
    env = with_nvm_env(
        {
            "PATH": "/usr/bin:/bin",
            "NVM_DIR": str(tmp_path),
            "NVM_BIN": str(nvm_bin),
        }
    )
    assert env["NVM_BIN"] == str(nvm_bin)
    assert env["PATH"].split(os.pathsep)[0] == str(nvm_bin)


def test_with_nvm_env_resolves_default_alias(tmp_path: Path):
    nvm_dir = tmp_path
    (nvm_dir / "alias").mkdir(parents=True)
    (nvm_dir / "alias" / "default").write_text("v20.11.1\n", encoding="utf-8")
    target_bin = nvm_dir / "versions" / "node" / "v20.11.1" / "bin"
    target_bin.mkdir(parents=True)

    env = with_nvm_env({"PATH": "/usr/bin", "NVM_DIR": str(nvm_dir)})

    assert env["NVM_BIN"] == str(target_bin)
    assert env["PATH"].split(os.pathsep)[0] == str(target_bin)


def test_with_nvm_env_falls_back_to_latest_installed(tmp_path: Path):
    nvm_dir = tmp_path
    (nvm_dir / "versions" / "node" / "v18.19.1" / "bin").mkdir(parents=True)
    latest_bin = nvm_dir / "versions" / "node" / "v22.3.0" / "bin"
    latest_bin.mkdir(parents=True)

    env = with_nvm_env({"PATH": "/usr/bin", "NVM_DIR": str(nvm_dir)})

    assert env["NVM_BIN"] == str(latest_bin)
    assert env["PATH"].split(os.pathsep)[0] == str(latest_bin)
