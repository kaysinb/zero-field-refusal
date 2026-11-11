"""Study 5's GPU code, pinned. The registered runs' code (docs/study5-code-identity.json, pods 1
and 2), extension 1's (docs/study5-extension-1-code-identity.json, pod 3) and extension 2's
(docs/study5-extension-2-code-identity.json, pod 4): HEAD's GPU code identity
(``code_identity.gpu_code_identity``: every GPU path's blob at the commit) is the latest record's,
so that any later change of a GPU path fails here until a record is written again after a new
engineering check (docs/study5-preregistration.md, "Protocol", "Extension 1" and "Extension 2").
The earlier records stay: each commit keeps its code, and the GPU paths changed from one record's
commit to the next are exactly those the later record names."""

from __future__ import annotations

import json
import subprocess

import pytest

from zero_field_refusal import code_identity
from zero_field_refusal.paths import REPO_ROOT

RECORD = REPO_ROOT / "docs" / "study5-code-identity.json"
EXTENSION = REPO_ROOT / "docs" / "study5-extension-1-code-identity.json"
EXTENSION_2 = REPO_ROOT / "docs" / "study5-extension-2-code-identity.json"
FIELDS = {"what", "rule", "commit", "files", "gpu_code_sha256", "engineering_check"}


def recorded(path=RECORD, fields=FIELDS) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record) == fields
    assert record["rule"] == "zero_field_refusal.code_identity.gpu_code_identity"
    assert len(record["commit"]) == 40 and len(record["gpu_code_sha256"]) == 64
    return record


def extension() -> dict:
    record = recorded(EXTENSION, FIELDS | {"changed_from"})
    assert record["changed_from"]["record"] == "docs/study5-code-identity.json"
    return record


def extension_2() -> dict:
    record = recorded(EXTENSION_2, FIELDS | {"changed_from"})
    assert record["changed_from"]["record"] == "docs/study5-extension-1-code-identity.json"
    return record


def in_history(commit: str) -> bool:
    found = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return found.returncode == 0


def test_heads_gpu_code_is_the_latest_recorded_one() -> None:
    record = extension_2()
    head = code_identity.gpu_code_identity("HEAD")
    assert (head["sha256"], head["files"]) == (record["gpu_code_sha256"], record["files"]), (
        "a GPU path changed since docs/study5-extension-2-code-identity.json: rerun the "
        "engineering check, then record the new identity"
    )


def test_each_recorded_commit_has_its_code() -> None:
    for record in (recorded(), extension(), extension_2()):
        if not in_history(record["commit"]):
            pytest.skip(f"the recorded commit {record['commit'][:7]} is not in this history")
        identity = code_identity.gpu_code_identity(record["commit"])
        assert identity["sha256"] == record["gpu_code_sha256"]
        assert identity["files"] == record["files"]


def test_only_the_named_paths_changed_from_record_to_record() -> None:
    chain = (recorded(), extension(), extension_2())
    if not all(in_history(r["commit"]) for r in chain):
        pytest.skip("the recorded commits are not in this history")
    for earlier, later in zip(chain, chain[1:], strict=False):
        changed = code_identity.gpu_paths_changed(earlier["commit"], later["commit"])
        assert changed == sorted(later["changed_from"]["paths"])
        assert earlier["gpu_code_sha256"] != later["gpu_code_sha256"]
    assert code_identity.gpu_paths_changed(chain[-1]["commit"], "HEAD") == []
