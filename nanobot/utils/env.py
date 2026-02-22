"""Environment helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping


def with_nvm_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an env dict with nvm node bin prepended to PATH when available."""
    env = dict(base_env or os.environ)

    nvm_dir_value = env.get("NVM_DIR")
    nvm_dir = Path(nvm_dir_value).expanduser() if nvm_dir_value else (Path.home() / ".nvm")
    if not nvm_dir.exists():
        return env

    nvm_bin = _resolve_nvm_bin(nvm_dir, env)
    if not nvm_bin:
        return env

    env["NVM_DIR"] = str(nvm_dir)
    env["NVM_BIN"] = str(nvm_bin)

    path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    nvm_bin_str = str(nvm_bin)
    if nvm_bin_str not in path_parts:
        env["PATH"] = os.pathsep.join([nvm_bin_str, *path_parts]) if path_parts else nvm_bin_str

    return env


def _resolve_nvm_bin(nvm_dir: Path, env: Mapping[str, str]) -> Path | None:
    """Resolve the most likely nvm bin directory from env, aliases, or installed versions."""
    nvm_bin_env = env.get("NVM_BIN")
    if nvm_bin_env:
        candidate = Path(nvm_bin_env).expanduser()
        if candidate.exists():
            return candidate

    default_alias = nvm_dir / "alias" / "default"
    if default_alias.exists():
        target = default_alias.read_text(encoding="utf-8").strip()
        resolved = _resolve_alias_target(nvm_dir, target, depth=0)
        if resolved:
            return resolved

    return _latest_installed_nvm_bin(nvm_dir)


def _resolve_alias_target(nvm_dir: Path, target: str, depth: int) -> Path | None:
    """Resolve an nvm alias target like v20.11.1, lts/*, or another alias."""
    if not target or depth > 5:
        return None

    # Alias can include comments or metadata in some setups.
    clean = target.split()[0]

    version_bin = nvm_dir / "versions" / "node" / clean / "bin"
    if version_bin.exists():
        return version_bin

    alias_file = nvm_dir / "alias" / clean
    if alias_file.exists():
        nested = alias_file.read_text(encoding="utf-8").strip()
        return _resolve_alias_target(nvm_dir, nested, depth + 1)

    # Bare semver aliases may omit the "v" prefix.
    if re.fullmatch(r"\d+\.\d+\.\d+", clean):
        candidate = nvm_dir / "versions" / "node" / f"v{clean}" / "bin"
        if candidate.exists():
            return candidate

    return None


def _latest_installed_nvm_bin(nvm_dir: Path) -> Path | None:
    """Pick the highest installed nvm node version by semver ordering."""
    versions_dir = nvm_dir / "versions" / "node"
    if not versions_dir.exists():
        return None

    bins: list[tuple[tuple[int, int, int], Path]] = []
    for version_dir in versions_dir.glob("v*"):
        if not version_dir.is_dir():
            continue
        m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version_dir.name)
        if not m:
            continue
        bin_dir = version_dir / "bin"
        if not bin_dir.exists():
            continue
        bins.append(((int(m.group(1)), int(m.group(2)), int(m.group(3))), bin_dir))

    if not bins:
        return None

    bins.sort(key=lambda item: item[0], reverse=True)
    return bins[0][1]
