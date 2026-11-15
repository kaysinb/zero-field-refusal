"""Study 5's GPU code, pinned. The registered runs' code (docs/study5-code-identity.json, pods 1
and 2), extension 1's (docs/study5-extension-1-code-identity.json, pod 3), extension 2's
(docs/study5-extension-2-code-identity.json, pod 4), extension 2's after its fourth and fifth
amendments (docs/study5-extension-2-amendment-5-code-identity.json) and the merge of the E2
top-up's stream-prefix fix into it (docs/study5-extension-2-topup-fix-code-identity.json): HEAD's
GPU code identity (``code_identity.gpu_code_identity``: every GPU path's blob at the commit) is the
latest record's, so that any later change of a GPU path fails here until a record is written again
after a new engineering check (docs/study5-preregistration.md, "Protocol", "Extension 1" and
"Extension 2"). The earlier records stay: each commit keeps its code, and the GPU paths changed
from one record's commit to the next are exactly those the later record names. The fix's own
record (docs/study5-topup-fix-code-identity.json) branches from extension 2's: the merge joins the
two lines, each side changing exactly the other side's paths."""

from __future__ import annotations

import json
import subprocess

import pytest

from zero_field_refusal import code_identity
from zero_field_refusal.paths import REPO_ROOT

RECORD = REPO_ROOT / "docs" / "study5-code-identity.json"
EXTENSION = REPO_ROOT / "docs" / "study5-extension-1-code-identity.json"
EXTENSION_2 = REPO_ROOT / "docs" / "study5-extension-2-code-identity.json"
AMENDMENT_5 = REPO_ROOT / "docs" / "study5-extension-2-amendment-5-code-identity.json"
TOPUP_FIX = REPO_ROOT / "docs" / "study5-topup-fix-code-identity.json"
MERGED = REPO_ROOT / "docs" / "study5-extension-2-topup-fix-code-identity.json"
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


def amendment_5() -> dict:
    record = recorded(AMENDMENT_5, FIELDS | {"changed_from"})
    assert record["changed_from"]["record"] == "docs/study5-extension-2-code-identity.json"
    return record


def topup_fix() -> dict:
    record = recorded(TOPUP_FIX, FIELDS | {"changed_from"})
    assert record["changed_from"]["record"] == "docs/study5-extension-2-code-identity.json"
    return record


def merged() -> dict:
    record = recorded(MERGED, FIELDS | {"changed_from"})
    assert record["changed_from"]["record"] == (
        "docs/study5-extension-2-amendment-5-code-identity.json"
    )
    return record


def in_history(commit: str) -> bool:
    found = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return found.returncode == 0


def test_heads_gpu_code_is_the_latest_recorded_one() -> None:
    record = merged()
    head = code_identity.gpu_code_identity("HEAD")
    assert (head["sha256"], head["files"]) == (record["gpu_code_sha256"], record["files"]), (
        "a GPU path changed since docs/study5-extension-2-topup-fix-code-identity.json: rerun "
        "the engineering check, then record the new identity"
    )


def test_each_recorded_commit_has_its_code() -> None:
    for record in (recorded(), extension(), extension_2(), amendment_5(), topup_fix(), merged()):
        if not in_history(record["commit"]):
            pytest.skip(f"the recorded commit {record['commit'][:7]} is not in this history")
        identity = code_identity.gpu_code_identity(record["commit"])
        assert identity["sha256"] == record["gpu_code_sha256"]
        assert identity["files"] == record["files"]


def test_only_the_named_paths_changed_from_record_to_record() -> None:
    chain = (recorded(), extension(), extension_2(), amendment_5(), merged())
    if not all(in_history(r["commit"]) for r in chain):
        pytest.skip("the recorded commits are not in this history")
    for earlier, later in zip(chain, chain[1:], strict=False):
        changed = code_identity.gpu_paths_changed(earlier["commit"], later["commit"])
        assert changed == sorted(later["changed_from"]["paths"])
        assert earlier["gpu_code_sha256"] != later["gpu_code_sha256"]
    assert code_identity.gpu_paths_changed(chain[-1]["commit"], "HEAD") == []


def test_the_fix_branches_from_extension_2_and_the_merge_joins_both_lines() -> None:
    """The fix's record changes exactly its paths from extension 2's record; from the fix to the
    merge exactly the fifth amendment's paths change, and from the fifth amendment to the merge
    exactly the fix's."""
    base, amendment, fix, joined = extension_2(), amendment_5(), topup_fix(), merged()
    if not all(in_history(r["commit"]) for r in (base, amendment, fix, joined)):
        pytest.skip("the recorded commits are not in this history")
    changed = code_identity.gpu_paths_changed
    assert changed(base["commit"], fix["commit"]) == sorted(fix["changed_from"]["paths"])
    assert changed(fix["commit"], joined["commit"]) == sorted(amendment["changed_from"]["paths"])
    assert sorted(joined["changed_from"]["paths"]) == sorted(fix["changed_from"]["paths"])
    assert len({r["gpu_code_sha256"] for r in (base, amendment, fix, joined)}) == 4
