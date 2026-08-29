"""The guided workflow: plain-language settings in, a readable report out.

Built for people who have polls and a question but neither statistics nor
Python.  A settings file states the race, the poll file, and what the user
believes about pollster bias; :func:`~kronikas.guided.runner.run` fits the
model and writes a self-contained HTML report.

Driven from the command line by ``kronikas guided``, and by an AI assistant
through the skill in :mod:`kronikas.skill` (``kronikas skill install``).
"""

from .report import build as build_report
from .runner import run
from .settings import (
    Plan,
    PollsterBelief,
    SettingsError,
    build_model_config,
    describe,
    load_plan,
    normalise,
    read_settings_file,
)

__all__ = [
    "Plan",
    "PollsterBelief",
    "SettingsError",
    "build_model_config",
    "build_report",
    "describe",
    "load_plan",
    "normalise",
    "read_settings_file",
    "run",
]
