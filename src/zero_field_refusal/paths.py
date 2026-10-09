"""Repository locations.

Data and run outputs live outside version control. Set ``DATA_DIR`` and ``RUNS_DIR`` to move them
to a larger disk, e.g. on a rented machine.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

# Valid for the editable install that `uv sync` creates.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"

if not (REPO_ROOT / "pyproject.toml").is_file() or not CONFIG_ROOT.is_dir():
    raise RuntimeError(
        f"Expected pyproject.toml and configs/ in {REPO_ROOT}: the package must be installed from "
        "its repository in editable mode (run `uv sync` in the repository)."
    )


def load_env() -> None:
    """Read ``.env`` from the repository root.

    Variables already set in the environment win, and blank values are ignored, so an empty
    ``KEY=`` line does not override a library default.
    """
    for key, value in dotenv_values(REPO_ROOT / ".env").items():
        if value and key not in os.environ:
            os.environ[key] = value


def data_root() -> Path:
    return Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data")


def runs_root() -> Path:
    return Path(os.environ.get("RUNS_DIR") or REPO_ROOT / "runs")
