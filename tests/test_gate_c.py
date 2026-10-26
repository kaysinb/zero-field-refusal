"""Gate C (amendment 4: the analysis, the simulation and the fixation) changes no code a GPU run
executes: no GPU path (``code_identity.is_gpu_path``) and none of the modules the launcher and
the GPU tools load, which the GPU identity leaves out (``analysis/__init__.py`` and its imports,
``analysis/stage_a.py``; the final review of gate B, I5 and M10).

The guarantee is study 2's: it held for every study-2 commit up to its last one, ``STUDY2_LAST``
(study 2's versions record after amendment 5), and the tests compare the gate-B commit with that
one, not with HEAD. Study 3 changes GPU paths by design and carries its own GPU-code identity in
every qualification record.

The gate-B record (docs/study2-short-gate-b.json) names the commit gate B was reviewed at and its
GPU-code sha256; it is updated if gate B changes. These tests read only that record and the two
commits."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from zero_field_refusal import code_identity
from zero_field_refusal.paths import REPO_ROOT

RECORD = REPO_ROOT / "docs" / "study2-short-gate-b.json"
# The entry points of every GPU run and GPU tool (the launcher's, the drift diagnostic's and the
# numerical check's), whose loaded modules make up what a GPU run imports.
GPU_ENTRY_POINTS = ("launch", "train", "schedule", "fa_control_diagnostic", "drift")
# Study 2's last commit (its versions record after amendment 5): the end of the guarantee
STUDY2_LAST = "028b953"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def _gate_b() -> dict:
    """The gate-B record; the test skips, with the reason, only when its commit or study 2's
    last one is not in this clone (a shallow or partial checkout)."""
    record = json.loads(RECORD.read_text())
    for name, commit in (("gate-B", record["commit"]), ("study 2's last", STUDY2_LAST)):
        if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            pytest.skip(f"the {name} commit {commit} is not in this clone")
    return record


def test_gate_c_changes_no_gpu_path() -> None:
    record = _gate_b()
    check = code_identity.reuse_check(record["commit"], STUDY2_LAST)
    assert check["changed"] == [] and check["reusable"], check["changed"]
    assert check["base"]["sha256"] == record["gpu_code_sha256"]
    assert code_identity.gpu_code_identity(STUDY2_LAST)["sha256"] == record["gpu_code_sha256"]


def test_gate_c_leaves_the_launchers_imports_alone() -> None:
    record = _gate_b()
    imports = "; ".join(f"import zero_field_refusal.{name}" for name in GPU_ENTRY_POINTS)
    program = (
        f"import json, sys; {imports}; "
        "print(json.dumps(sorted(m.__file__ for name, m in sys.modules.items() "
        "if name.split('.')[0] == 'zero_field_refusal' and getattr(m, '__file__', None))))"
    )
    loaded = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, capture_output=True,
                            text=True, check=True)  # fmt: skip
    files = [str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
             for path in json.loads(loaded.stdout)]  # fmt: skip
    assert "src/zero_field_refusal/launch.py" in files
    assert "src/zero_field_refusal/analysis/__init__.py" in files
    assert "src/zero_field_refusal/analysis/stage_a.py" in files
    changed = _git("diff", "--name-only", "--no-renames", record["commit"], STUDY2_LAST, "--",
                   *files)  # fmt: skip
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.split() == []
