"""The identity of the code that a GPU run executes (study 2, amendment 4, section 1).

Amendment 4's numerical qualification holds for the code its runs executed, not for a commit:
analysis-only changes (the selection, the looks, the simulation, the docs) move the commit without
changing what a GPU run computes. This module names the files that can change a run, hashes them
at a commit, and lists the ones two commits differ in, so that a qualification made at one commit
is reused at a later one only after a recorded check that no such file changed.

A **GPU path** is every tracked file under ``src/zero_field_refusal/``, ``scripts/``,
``configs/`` and ``docker/``, and ``pyproject.toml``, ``uv.lock`` and the files that build the
run environment's image (``Dockerfile``, its build context's ``.dockerignore`` and
``.python-version``; ``docker/project-run.py`` is the image's launcher), except the analysis
package (its ``stage_a.py`` stays in: the launcher imports it to qualify terminal records), the
scripts that only read records or plan campaigns, and the stand-alone diagnostics under
``scripts/diagnostics/``, which no run executes (docs/study3-preregistration.md, amendment 7). The
default is inclusion, so an omission errs toward a stricter check.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

INCLUDED = ("src/zero_field_refusal/", "scripts/", "configs/", "docker/")
# The dependency manifests and the files that build the image every Pod run executes in
INCLUDED_FILES = ("pyproject.toml", "uv.lock", "Dockerfile", ".dockerignore", ".python-version")
EXCLUDED_PREFIXES = ("src/zero_field_refusal/analysis/",)
KEPT_IN_EXCLUDED = ("src/zero_field_refusal/analysis/stage_a.py",)  # imported by the launcher
DIAGNOSTICS_PREFIX = "scripts/diagnostics/"  # by location: nothing traces what such a file imports
# Scripts that read records, plan campaigns or write reports; none runs on a GPU
ANALYSIS_SCRIPTS = frozenset(
    f"scripts/{name}"
    for name in (
        "archive.py",
        "build_analysis.py",
        "check_docs.py",
        "check_theta0_utility.py",
        "gate1_report.py",
        "index_runs.py",
        "judge_check_qualify.py",
        "pilot_momentum_campaigns.py",
        "pilot_momentum_prediction.py",
        "pilot_momentum_summary.py",
        "simulate_analysis.py",
        "stage1_descriptive.py",
        "stage_a_barrier.py",
        "stage_a_campaigns.py",
        "stage_a_summary.py",
        "study2_short_campaigns.py",
        "study2_short_simulate.py",
        "study2_short_summary.py",
        # Study 3's analysis entry points, listed before they exist: adding a name later would
        # change this file, which is inside the identity (a name matters once its file exists)
        "study3_analysis.py",
        "study3_audit.py",
        "study3_campaigns.py",
        "study3_simulate.py",
        "study3_spread.py",
        "study3_summary.py",
    )
)


def is_gpu_path(path: str) -> bool:
    """Whether a repository-relative path belongs to the code a GPU run executes."""
    if path in KEPT_IN_EXCLUDED:
        return True
    if path in ANALYSIS_SCRIPTS or path.startswith((*EXCLUDED_PREFIXES, DIAGNOSTICS_PREFIX)):
        return False
    if "/__pycache__/" in path or path.endswith(".pyc"):
        return False
    return path in INCLUDED_FILES or path.startswith(INCLUDED)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout  # fmt: skip


def gpu_code_identity(commit: str = "HEAD", repo: Path = REPO_ROOT) -> dict[str, Any]:
    """The GPU paths of ``commit`` as committed (not the working tree): the resolved commit, the
    number of files, and a sha256 over their sorted (path, blob) pairs."""
    resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    entries = []
    for line in _git(repo, "ls-tree", "-r", "--full-tree", resolved).splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind == "blob" and is_gpu_path(path):
            entries.append((path, mode, blob))
    digest = hashlib.sha256()
    for path, mode, blob in sorted(entries):
        digest.update(f"{path}\0{mode}\0{blob}\n".encode())
    return {"commit": resolved, "files": len(entries), "sha256": digest.hexdigest()}


def gpu_paths_changed(base: str, head: str = "HEAD", repo: Path = REPO_ROOT) -> list[str]:
    """The GPU paths that differ between two commits, renames as both paths."""
    changed = _git(repo, "diff", "--name-only", "--no-renames", base, head).splitlines()
    return sorted(p for p in changed if is_gpu_path(p))


def reuse_check(base: str, head: str = "HEAD", repo: Path = REPO_ROOT) -> dict[str, Any]:
    """The recorded check that lets a qualification made at ``base`` hold at ``head``: equal GPU
    identities and no GPU path changed between them."""
    a, b = gpu_code_identity(base, repo), gpu_code_identity(head, repo)
    changed = gpu_paths_changed(a["commit"], b["commit"], repo)
    return {"base": a, "head": b, "changed": changed,
            "reusable": a["sha256"] == b["sha256"] and not changed}  # fmt: skip
