import os
from pathlib import Path

import pytest

from zero_field_refusal import paths


def test_load_env_skips_blank_values_and_keeps_the_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("BLANK=\nFROM_FILE=file\nFROM_SHELL=file\n")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    env = {"FROM_SHELL": "shell"}
    monkeypatch.setattr(os, "environ", env)
    paths.load_env()
    assert env == {"FROM_FILE": "file", "FROM_SHELL": "shell"}
