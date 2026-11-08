"""Study 5's GPU code, pinned (docs/study5-code-identity.json): HEAD's GPU code identity
(``code_identity.gpu_code_identity``: every GPU path's blob at the commit) is the recorded one, so
that any later change of a GPU path fails here until the file is written again after a new
engineering check (docs/study5-preregistration.md, "Protocol": study 3's F runs on seed 10403,
rerun under the code and compared by ``scripts/study5_engineering_check.py``)."""

from __future__ import annotations

import json
import subprocess

import pytest

from zero_field_refusal import code_identity
from zero_field_refusal.paths import REPO_ROOT

RECORD = REPO_ROOT / "docs" / "study5-code-identity.json"
FIELDS = {"what", "rule", "commit", "files", "gpu_code_sha256", "engineering_check"}


def recorded() -> dict:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert set(record) == FIELDS
    assert record["rule"] == "zero_field_refusal.code_identity.gpu_code_identity"
    assert len(record["commit"]) == 40 and len(record["gpu_code_sha256"]) == 64
    return record


def test_heads_gpu_code_is_the_recorded_one() -> None:
    record = recorded()
    head = code_identity.gpu_code_identity("HEAD")
    assert (head["sha256"], head["files"]) == (record["gpu_code_sha256"], record["files"]), (
        "a GPU path changed since docs/study5-code-identity.json: rerun the engineering check, "
        "then record the new identity"
    )


def test_the_recorded_commit_has_that_code_and_no_gpu_path_changed_since() -> None:
    record = recorded()
    found = subprocess.run(
        ["git", "merge-base", "--is-ancestor", record["commit"], "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if found.returncode != 0:
        pytest.skip(f"the recorded commit {record['commit'][:7]} is not in this history")
    identity = code_identity.gpu_code_identity(record["commit"])
    assert identity["sha256"] == record["gpu_code_sha256"]
    assert code_identity.gpu_paths_changed(record["commit"], "HEAD") == []
