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
        ("src/zero_field_refusal/qualification_constants.py", True),
        ("scripts/launch.py", True),
        ("scripts/numerical_check.py", True),
        ("scripts/study2_short_summary.py", False),
        ("scripts/check_theta0_utility.py", False),
        ("scripts/study2_short_campaigns.py", False),
        ("configs/experiments/study2_short.yaml", True),
        ("pyproject.toml", True),
        ("uv.lock", True),
        # The run environment's image: its build and its launcher
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


def test_a_diagnostic_script_is_no_gpu_path() -> None:
    assert not is_gpu_path("scripts/diagnostics/v0_backends.py")
    assert is_gpu_path("scripts/v0_backends.py")


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
    """The files that build the run environment are GPU paths, so a change of the
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


def test_nothing_a_qualification_part_executes_is_defined_under_analysis() -> None:
    """Study 3's harness takes from the analysis package (outside the GPU identity) the four
    verdict functions alone, which it re-exports and never calls; every constant it executes or
    records is ``qualification_constants``', a GPU path."""
    import ast

    from zero_field_refusal import qualification, qualification_constants
    from zero_field_refusal.analysis import qualification_verdicts

    root = Path(qualification.__file__).resolve().parents[2]
    tree = ast.parse(Path(qualification.__file__).read_text())
    from_analysis = {alias.asname or alias.name for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and "analysis" in (node.module or "")
                     for alias in node.names}  # fmt: skip
    verdicts = {"case_verdict", "control_verdict", "rate_verdict", "engineering_summary",
                "rest_from_trajectories"}  # fmt: skip
    assert from_analysis == verdicts
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not used & verdicts, node.name
    # The verdict module defines no constant the harness has: its own are verdict reasons
    module = ast.parse(Path(qualification_verdicts.__file__).read_text())
    defined = {t.id for node in module.body if isinstance(node, ast.Assign)
               for t in node.targets if isinstance(t, ast.Name)}  # fmt: skip
    assert not defined & set(vars(qualification)), defined & set(vars(qualification))
    for name in ("SHIFTS", "REST", "STEPS", "BATCH", "COMPARISONS", "SCIENTIFIC", "TECHNICAL",
                 "PARTS", "record_sha256", "QualificationError"):  # fmt: skip
        assert getattr(qualification, name) is getattr(qualification_constants, name), name
    for path, gpu in ((qualification_constants.__file__, True), (qualification.__file__, True),
                      (qualification_verdicts.__file__, False)):  # fmt: skip
        assert is_gpu_path(str(Path(path).resolve().relative_to(root))) is gpu, path


STUDY3_ANALYSIS = ("study3_analysis.py", "study3_audit.py", "study3_campaigns.py",
                   "study3_simulate.py", "study3_spread.py", "study3_summary.py",
                   "check_theta0_utility.py")  # fmt: skip
GPU_MODULES = ("qualification", "schedule", "train", "launch", "runtime", "fa_control_diagnostic")


# What loads code by its name or its path at run time: a script such as scripts/qualify.py could be
# run so
LOADERS = ("spec_from_file_location", "run_path", "run_module", "SourceFileLoader", "load_source",
           "import_module", "__import__", "exec")  # fmt: skip
# What runs a program: a call of one whose arguments name scripts/qualify.py runs it
RUNNERS = ("run", "Popen", "call", "check_call", "check_output", "system", "execv", "execvp",
           "spawnv")  # fmt: skip


def gpu_uses(path: Path, package: str | None = None) -> set[str]:
    """What a script, or a module of ``package`` (its relative imports resolved), imports that
    runs on a GPU (the GPU modules, torch) or as a script (``qualify``, ``scripts.qualify``:
    scripts/qualify.py), what it calls to load code by its name or path at run time (``LOADERS``)
    or to run scripts/qualify.py (``RUNNERS``, an argument naming ``qualify.py``), and its uses of
    CUDA."""
    import ast

    text = path.read_text()
    imported, loads = set(), set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and package:
                parts = package.split(".")
                module = ".".join([*parts[: len(parts) - node.level + 1], *filter(None, [module])])
            elif node.level:
                module = "." * node.level + module
            imported |= {module, *(f"{module}.{a.name}" for a in node.names)}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in LOADERS:
                loads.add(f"loads code at run time: {name}")
            named = (n.value for n in ast.walk(node) if isinstance(n, ast.Constant))
            if name in RUNNERS and any(isinstance(v, str) and "qualify.py" in v for v in named):
                loads.add(f"runs scripts/qualify.py: {name}")
    found = {
        i
        for i in imported
        for m in GPU_MODULES
        if i == f"zero_field_refusal.{m}" or i.startswith(f"zero_field_refusal.{m}.")
    }
    found |= {i for i in imported if i == "torch" or i.startswith("torch.")}
    found |= {
        i
        for i in imported
        for m in ("qualify", "scripts.qualify")
        if i == m or i.startswith(f"{m}.")
    }
    return found | loads | {use for use in ("torch.cuda", ".cuda(") if use in text}  # fmt: skip


def test_study3_analysis_scripts_are_listed_and_do_no_gpu_work(tmp_path) -> None:
    """Study 3's analysis entry points are outside the identity from the start; each either does
    not exist yet or imports nothing that runs on a GPU (the GPU modules, torch) or as a script
    (scripts/qualify.py), loads no code by its name or path at run time, runs no subprocess whose
    arguments name scripts/qualify.py, and uses no CUDA. The check is shown to catch such a
    script, and such a module of the analysis package (its relative imports resolved), which may
    import the package's own modules."""
    from zero_field_refusal.code_identity import ANALYSIS_SCRIPTS
    from zero_field_refusal.paths import REPO_ROOT

    assert {f"scripts/{n}" for n in STUDY3_ANALYSIS} <= ANALYSIS_SCRIPTS
    assert {p for p in ANALYSIS_SCRIPTS if "study3" in p} == {
        f"scripts/{n}" for n in STUDY3_ANALYSIS if "study3" in n}  # fmt: skip
    for name in STUDY3_ANALYSIS:
        path = REPO_ROOT / "scripts" / name
        assert is_gpu_path(f"scripts/{name}") is False
        if path.exists():
            assert not gpu_uses(path), (name, gpu_uses(path))
    for text, found in (("import torch\n", {"torch"}),
                        ("x = 1\ntorch.cuda.synchronize()\n", {"torch.cuda"}),
                        ("from zero_field_refusal import qualification\n",
                         {"zero_field_refusal.qualification"}),
                        ("from zero_field_refusal.schedule import run_scheduled\n",
                         {"zero_field_refusal.schedule",
                          "zero_field_refusal.schedule.run_scheduled"})):  # fmt: skip
        sample = tmp_path / "sample.py"
        sample.write_text(text)
        assert gpu_uses(sample) >= found, text
    for text, found in (("import qualify\n", {"qualify"}),
                        ("import scripts.qualify\n", {"scripts.qualify"}),
                        ("from scripts import qualify\n", {"scripts.qualify"}),
                        ("import subprocess\n"
                         "subprocess.run(['python', 'scripts/qualify.py', 'verdict'])\n",
                         {"runs scripts/qualify.py: run"}),
                        ("import os\nos.system('python scripts/qualify.py driver')\n",
                         {"runs scripts/qualify.py: system"}),
                        ("import subprocess\nsubprocess.run(['git', 'status'])\n", set()),
                        ("print('python scripts/qualify.py driver --campaign x')\n", set()),
                        ("import importlib.util\n"
                         "importlib.util.spec_from_file_location('q', 'scripts/qualify.py')\n",
                         {"loads code at run time: spec_from_file_location"}),
                        ("import runpy\nrunpy.run_path('scripts/qualify.py')\n",
                         {"loads code at run time: run_path"})):  # fmt: skip
        sample = tmp_path / "sample.py"
        sample.write_text(text)
        assert gpu_uses(sample) == found, text
    analysis = "zero_field_refusal.analysis"
    for text, found in (("from ..qualification import Case\n",
                         {"zero_field_refusal.qualification",
                          "zero_field_refusal.qualification.Case"}),
                        ("from .. import schedule, train\n",
                         {"zero_field_refusal.schedule", "zero_field_refusal.train"}),
                        ("from . import qualification_verdicts\n"
                         "from ..qualification_constants import BATCH\n", set())):  # fmt: skip
        sample = tmp_path / "sample.py"
        sample.write_text(text)
        assert gpu_uses(sample, analysis) == found, text
