import subprocess
from pathlib import Path

import pytest

from zero_field_refusal.code_identity import (
    gpu_code_identity,
    gpu_paths_changed,
    is_gpu_path,
    reuse_check,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/zero_field_refusal/optim.py", True),
        ("src/zero_field_refusal/schedule.py", True),
        ("src/zero_field_refusal/analysis/stage_a.py", True),
        ("src/zero_field_refusal/analysis/study2_short.py", False),
        ("src/zero_field_refusal/analysis/study2_short_sim.py", False),
        # Study 3's verdict rules are analysis; the harness that writes the records is not
        ("src/zero_field_refusal/analysis/qualification_verdicts.py", False),
        ("src/zero_field_refusal/qualification.py", True),
        ("scripts/launch.py", True),
        ("scripts/numerical_check.py", True),
        ("scripts/study2_short_summary.py", False),
        ("scripts/check_theta0_utility.py", False),
        ("scripts/study2_short_campaigns.py", False),
        ("configs/experiments/study2_short.yaml", True),
        ("pyproject.toml", True),
        ("uv.lock", True),
        # The run environment's image (review M4): its build and its launcher
        ("Dockerfile", True),
        (".dockerignore", True),
        (".python-version", True),
        ("docker/project-run.py", True),
        ("README.md", False),
        (".env.example", False),
        (".gitignore", False),
        ("docs/study2-short-fixation.json", False),
        ("tests/test_optim.py", False),
        ("src/zero_field_refusal/__pycache__/optim.cpython-312.pyc", False),
    ],
)
def test_gpu_paths(path: str, expected: bool) -> None:
    assert is_gpu_path(path) is expected


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


def test_analysis_only_commit_keeps_the_qualification(repo: Path) -> None:
    # Gate B pins the GPU code; Gate C changes analysis and docs only: the qualification is reused
    gate_b = _commit(repo, {"src/zero_field_refusal/optim.py": "a = 1\n",
                            "configs/x.yaml": "k: 1\n",
                            "src/zero_field_refusal/analysis/study2_short.py": "s = 1\n"},
                     "gate B")  # fmt: skip
    gate_c = _commit(repo, {"src/zero_field_refusal/analysis/study2_short.py": "s = 2\n",
                            "docs/study2-short-fixation.json": "{}\n",
                            "scripts/study2_short_summary.py": "x = 1\n"},
                     "gate C")  # fmt: skip
    check = reuse_check(gate_b, gate_c, repo)
    assert check["reusable"] is True
    assert check["changed"] == []
    assert check["base"]["sha256"] == check["head"]["sha256"]
    assert check["base"]["commit"] != check["head"]["commit"]


def test_a_gpu_path_change_voids_the_qualification(repo: Path) -> None:
    gate_b = _commit(repo, {"src/zero_field_refusal/optim.py": "a = 1\n"}, "gate B")
    later = _commit(repo, {"src/zero_field_refusal/optim.py": "a = 2\n"}, "touches the optimizer")
    check = reuse_check(gate_b, later, repo)
    assert check["reusable"] is False
    assert check["changed"] == ["src/zero_field_refusal/optim.py"]
    assert gpu_paths_changed(gate_b, later, repo) == ["src/zero_field_refusal/optim.py"]


def test_identity_reads_the_commit_not_the_working_tree(repo: Path) -> None:
    head = _commit(repo, {"configs/x.yaml": "k: 1\n"}, "one")
    before = gpu_code_identity(head, repo)
    (repo / "configs" / "x.yaml").write_text("k: 2\n")  # uncommitted
    assert gpu_code_identity(head, repo) == before
    assert before["files"] == 1


def test_an_image_change_voids_the_qualification(repo: Path) -> None:
    """Review M4: the files that build the run environment are GPU paths, so a change of the
    image's definition is a GPU-code change."""
    files = {"Dockerfile": "FROM a\n", "docker/project-run.py": "x = 1\n",
             ".python-version": "3.12\n", ".dockerignore": "**\n",
             "src/zero_field_refusal/optim.py": "a = 1\n"}  # fmt: skip
    gate_b = _commit(repo, files, "gate B")
    assert gpu_code_identity(gate_b, repo)["files"] == len(files)
    for name, text in (("Dockerfile", "FROM b\n"), ("docker/project-run.py", "x = 2\n"),
                       (".python-version", "3.13\n"), (".dockerignore", "**\n!x\n")):  # fmt: skip
        later = _commit(repo, {name: text}, f"changes {name}")
        check = reuse_check(gate_b, later, repo)
        assert check["reusable"] is False and check["changed"] == [name], name
        gate_b = later
