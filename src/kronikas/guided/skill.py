"""Locating and installing the packaged assistant skill.

The guided workflow ships with a skill — instructions an AI assistant follows
to conduct the whole interview, plus its reference material and templates.
``kronikas skill install`` copies it where the assistant will find it, so a
user who has run ``pip install kronikas`` needs nothing else.
"""

from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from pathlib import Path

SKILL_NAME = "election-forecast"


def default_target() -> Path:
    """Where Claude Code looks for a user's personally installed skills."""
    return Path.home() / ".claude" / "skills"


def copy_to(target_dir: str | Path, *, force: bool = False) -> Path:
    """Copy the packaged skill into *target_dir*.

    Parameters
    ----------
    target_dir:
        Directory that holds skills; the skill lands in a subdirectory named
        after it.  Created if missing.
    force:
        Overwrite an existing copy.  Off by default: a user may have edited
        the installed skill, and silently discarding that would be rude.

    Returns
    -------
    pathlib.Path
        The installed skill directory.

    Raises
    ------
    FileExistsError
        If the destination exists and *force* is false.
    """
    destination = Path(target_dir).expanduser() / SKILL_NAME
    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists. Pass --force to overwrite it, or "
            "choose another directory with --dir."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with as_file(files("kronikas") / "skill") as source:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    return destination


def packaged_path() -> Path:
    """Filesystem path of the skill inside the installed package."""
    with as_file(files("kronikas") / "skill") as source:
        return Path(source)
