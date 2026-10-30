"""Study 3's analysis package as a whole (the modules ``analysis/study3_*.py``,
``analysis/qualification_records.py`` and ``analysis/campaign_prep.py``, and the scripts
``scripts/study3_campaigns.py`` and ``scripts/study3_summary.py``): its boundaries.

- No GPU path has changed since the GPU code study 3 qualifies, as of package 1d
  (``QUALIFIED_GPU_CODE``, ``code_identity.reuse_check``).
- No module of the package or script imports a GPU module directly or loads a script (read as
  code).
- No label is read: the label readers' entry points raise wherever they are bound, before the
  package's modules and scripts are loaded, and none is called.
- The sealing boundary: the five sealed folders (``study3_records.SEALED``): a run's
  ``responses/`` and ``observations/``, its judge outputs (``judge/``) and their superseded copies
  (``judge-superseded/``), and the labeling ledger (``RUNS_DIR/study3/labeling/``). On the campaign
  writer's route world, run once for the module (every campaign kind and every summary command, in
  process, on files, in a Git checkout: the real ``progress`` and ``versions_problems``, the guard
  among them), every open of a path under a sealed folder fails, and none is made; the sentinel is
  in no output, refusal, problem or written file; and runs whose ``config.yaml`` and ``meta.json``
  link into their sealed folders, and a launch log and qualification records that do, are named
  problems. The qualification's reader, whose reading the route gives, reads the same files too,
  and opens no such link.
- A run's record files are opened only outside its sealed folders (``study3_records``, and the
  qualification's reader): a record file, or a run's folder, that resolves through a link under a
  ``responses/`` or ``observations/`` folder is refused unopened, a named problem; so is a launch
  log or a record of the qualification (the base is the runs root); and a journal start's command
  composes only a study-3 config.
- The label readers' fence (``LABEL_READERS``: ``analysis.study3_label_cli``,
  ``analysis.study3_labels`` and ``scripts/study3_spread.py``; "Judge and audit", "Sample size from
  validation"): they import no GPU module and load no script; a sealed path is opened only inside
  a named opener (``SEALED_OPENERS``, ``NamedOpens``); and each reader refuses a point that is
  closed: ``spread`` outside candidate N's choice step (it closes once the n record is written),
  ``final`` until the freeze gate opens it, reading nothing of a run's sealed folders meanwhile.

Every sealed file of these worlds holds the sentinel ``SENTINEL``, and every read of a sealed path
fails under ``SealedGuard``."""

import builtins
import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import test_study3_records
from omegaconf import OmegaConf
from test_code_identity import gpu_uses
from test_study3_campaigns import (
    ARM_OF,
    LABELS,
    OPT,
    START,
    campaign_of,
    run_config,
    script_module,
    target,
)
from test_study3_campaigns import World as CampaignWorld
from test_study3_decisions import World as DecisionWorld
from test_study3_records import World as RunWorld
from test_study3_records import by_id, config_of, header, write_run
from test_study3_versions import (
    AMENDMENTS,
    CONFIRMATION_C1,
    FIRST,
    REGISTRATION,
    SECOND,
    VALIDATION_C1,
    git,
    stand_in,
)
from test_study3_versions import Study as VersionsStudy

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import qualification_records, study3_progress, study3_records
from zero_field_refusal.analysis import study3_versions as sv
from zero_field_refusal.analysis.study3_decisions import decision_path, decisions
from zero_field_refusal.analysis.study3_pins import CONTROLS, spread_pin
from zero_field_refusal.analysis.study3_progress import THETA0_RUN, progress
from zero_field_refusal.analysis.study3_records import (
    SEEDS,
    EndpointReader,
    journal_lines,
    read_endpoints,
    read_holds,
    read_theta0,
)
from zero_field_refusal.analysis.study3_search import adjusted_rate
from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.launch import campaign_header, load_campaign
from zero_field_refusal.paths import CONFIG_ROOT, REPO_ROOT
from zero_field_refusal.tracking import git_state

SENTINEL = "PKG2-SENTINEL-qX7v-Rk3W-9zJd"  # in every responses/ and observations/ file here
SEALED = ("responses", "observations")
JUDGE = "judge"  # a run's judge outputs (<run>/judge/<judge>.jsonl): sealed for the checks
# Sealed below the runs root as the judge outputs are: their superseded copies
# (<run>/judge-superseded/) and the labeling ledger (study3/labeling/)
SUPERSEDED, LEDGER = "judge-superseded", "labeling"
UNDER_RUNS = (JUDGE, SUPERSEDED, LEDGER)
# The sealed folders as a refusal names them (``study3_records.SEALED_WHY``)
FOLDER = "a responses, observations, judge, judge-superseded or labeling folder"
RECORD_FILES = ("config.yaml", "meta.json", "terminal.json", "summary.json", "evaluations.jsonl",
                "metrics.jsonl", "monitor.jsonl")  # fmt: skip
ANALYSIS = REPO_ROOT / "src" / "zero_field_refusal" / "analysis"
PACKAGE = (
    ANALYSIS / "study3_blocks.py",
    ANALYSIS / "study3_decisions.py",
    ANALYSIS / "study3_inference.py",
    ANALYSIS / "study3_pins.py",
    ANALYSIS / "study3_progress.py",
    ANALYSIS / "study3_records.py",
    ANALYSIS / "study3_search.py",
    ANALYSIS / "study3_seed_vectors.py",
    ANALYSIS / "study3_simulation.py",
    ANALYSIS / "study3_versions.py",
    ANALYSIS / "qualification_records.py",
    ANALYSIS / "campaign_prep.py",
)
# study 3's label readers (package 2b), each with its own sealing tests and this file's fence: two
# modules of the analysis package and the spread's decision script
LABEL_READERS: tuple[Path, ...] = (ANALYSIS / "study3_label_cli.py", ANALYSIS / "study3_labels.py",
                                   REPO_ROOT / "scripts" / "study3_spread.py")  # fmt: skip
# The label readers' functions that open a sealed path, by module and name: a run's judge outputs
# (and the responses they are labeled from), their superseded copies or the labeling ledger.
# Every such open is inside one of them (``NamedOpens``)
SEALED_OPENERS = frozenset({
    "study3_labels.read_ledger", "study3_labels.append_event", "study3_labels.ledger_lock",
    "study3_labels.diagnoses", "study3_labels.torn_tail", "study3_labels.resolve_torn",
    "study3_labels._read_run", "study3_labels._receipt_sha256",
    "study3_label_cli._label_pass", "study3_label_cli._rounds_used",
    "study3_label_cli._supersede", "study3_label_cli._complete_moves",
    "study3_label_cli.label_controls", "study3_label_cli._controls",
    "study3_label_cli.write_diagnosis",
})  # fmt: skip
# The labeling ledger's own openers: the labeling command takes the ledger's lock and reads it
# before it checks the point, through these alone
LEDGER_OPENERS = frozenset({"study3_labels.read_ledger", "study3_labels.ledger_lock",
                            "study3_labels.torn_tail"})  # fmt: skip
OTHER = (
    ANALYSIS / "__init__.py",
    ANALYSIS / "bootstrap.py",
    ANALYSIS / "build.py",
    ANALYSIS / "curves.py",
    ANALYSIS / "decisions.py",
    ANALYSIS / "eligibility.py",
    ANALYSIS / "evidence.py",
    ANALYSIS / "gate1.py",
    ANALYSIS / "kstar.py",
    ANALYSIS / "outputs.py",
    ANALYSIS / "pilot_momentum.py",
    ANALYSIS / "primary.py",
    ANALYSIS / "qualification_verdicts.py",
    ANALYSIS / "runs.py",
    ANALYSIS / "simulate.py",
    ANALYSIS / "stage_a.py",
    ANALYSIS / "study2_audit.py",
    ANALYSIS / "study2_audit_cli.py",
    ANALYSIS / "study2_label_cli.py",
    ANALYSIS / "study2_labels.py",
    ANALYSIS / "study2_short.py",
    ANALYSIS / "study2_short_sim.py",
    ANALYSIS / "sweeps.py",
    ANALYSIS / "zero_field_law.py",
)
SCRIPTS = (
    REPO_ROOT / "scripts" / "study3_campaigns.py",
    REPO_ROOT / "scripts" / "study3_summary.py",
)
SUMMARY = SCRIPTS[1]
LABEL_MODULES = ("zero_field_refusal.analysis.study2_labels", "zero_field_refusal.labeling",
                 "zero_field_refusal.run_judging",
                 "zero_field_refusal.analysis.study3_labels",
                 "zero_field_refusal.analysis.study3_label_cli")  # fmt: skip
# The GPU code study 3 qualifies, as of package 1d: its last commit that changed a GPU path, the
# launcher's further recovery rounds, the driver's control-only path, the export of a technical
# case's checkpoints, restart-case's refusals of a control-alone state, of departing numerics and
# of a binding refusal, an established technical case's early export, a scientific case whose
# same-path control failed (it counts as scientific, its evidence kept while the driver goes on),
# control-only's quiet answer, a present checkpoint hashed once, the launch journal read once per
# engineering cycle and the record that names an interrupted attempt
QUALIFIED_GPU_CODE = "14c131f"
REAL_IDENTITY = code_identity.gpu_code_identity
REAL_READING = qualification_records.read_qualification  # the route replaces it
# The route: A's lower rate kept by the search on these offsets in ln K, then repaired; N = 99
# needs a budget record; the confirmation supplement's seeds
OFFSETS = dict(zip(SEEDS["search"][1], (0.0, 0.03, 0.06, 0.3), strict=True))
N = 99
SUPPLEMENT_SEEDS = ("--bootstrap-seed", 20261101, "--audit-seed", 20261102, "--agreement-seed",
                    20261103)  # fmt: skip


class SealedGuard:
    """``builtins.open``, ``io.open``, ``os.open``, ``Path.open``, ``Path.read_text`` and
    ``Path.read_bytes`` wrapped: while armed, opening a path that resolves (links followed) under
    a ``responses/`` or ``observations/`` folder, in any mode, fails, and the path is kept in
    ``attempts`` (a caller that swallows the failure is still caught); every path opened while
    armed, resolved, is kept in ``opened``. With ``runs_root``, so does
    a path under a run's ``judge/`` or ``judge-superseded/`` folder or the labeling ledger's
    ``labeling/`` (such a component below the runs root: the checkout's ``configs/judge/`` stays
    readable). ``off()`` disarms it while a test writes a run as the tracker does."""

    def __init__(self, monkeypatch, runs_root: Path | None = None) -> None:
        self.attempts: list[str] = []
        self.opened: list[str] = []
        self.armed = True
        self.runs_root = None if runs_root is None else Path(os.path.realpath(runs_root))
        real_open, real_os_open = builtins.open, os.open

        def open_(file, *args, **kwargs):
            self.check(file)
            return real_open(file, *args, **kwargs)

        def os_open(path, flags, mode=0o777, *, dir_fd=None):
            self.check(path, dir_fd)
            return real_os_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(builtins, "open", open_)
        monkeypatch.setattr(io, "open", open_)
        monkeypatch.setattr(os, "open", os_open)
        for name in ("open", "read_text", "read_bytes"):
            real = getattr(Path, name)

            def wrapper(self_, *args, _real=real, **kwargs):
                self.check(self_)
                return _real(self_, *args, **kwargs)

            monkeypatch.setattr(Path, name, wrapper)

    def check(self, path, dir_fd=None) -> None:
        if not self.armed or isinstance(path, int):
            return
        text = os.fsdecode(path)
        if dir_fd is not None and not os.path.isabs(text):
            text = os.path.join(f"/proc/self/fd/{dir_fd}", text)
        resolved = Path(os.path.realpath(text))
        self.opened.append(str(resolved))
        judged = (self.runs_root is not None and resolved.is_relative_to(self.runs_root)
                  and any(part in UNDER_RUNS
                          for part in resolved.relative_to(self.runs_root).parts))  # fmt: skip
        if judged or any(part in SEALED for part in resolved.parts):
            self.attempts.append(text)
            raise AssertionError(f"a sealed path was opened: {text}")

    @contextlib.contextmanager
    def off(self):
        self.armed = False
        try:
            yield
        finally:
            self.armed = True


def stack_names() -> set[str]:
    """``<module file's stem>.<function>`` of every frame of this thread's stack and, in another
    thread (a judge client's worker), of the main thread's stack, which waits for it."""
    frames = [sys._getframe(1)]
    if threading.current_thread() is not threading.main_thread():
        frames.append(sys._current_frames().get(threading.main_thread().ident))
    names = set()
    for frame in frames:
        while frame is not None:
            names.add(f"{Path(frame.f_code.co_filename).stem}.{frame.f_code.co_name}")
            frame = frame.f_back
    return names


class NamedOpens(SealedGuard):
    """``SealedGuard`` where a sealed open is allowed only inside a named opener (``openers``,
    ``SEALED_OPENERS`` by default): a frame of its stack, or, in a judge client's worker thread,
    of the main thread's stack that waits for it (``stack_names``). With ``under``, only a path
    under that folder is allowed so. Each allowed open is kept in ``allowed``."""

    def __init__(self, monkeypatch, runs_root: Path | None = None, openers=SEALED_OPENERS,
                 under: Path | None = None) -> None:  # fmt: skip
        super().__init__(monkeypatch, runs_root)
        self.openers = frozenset(openers)
        self.under = None if under is None else Path(os.path.realpath(under))
        self.allowed: list[str] = []

    def check(self, path, dir_fd=None) -> None:
        try:
            super().check(path, dir_fd)
        except AssertionError:
            resolved = Path(self.opened[-1])
            if (self.under is not None and not resolved.is_relative_to(self.under)) or not (
                    stack_names() & self.openers):  # fmt: skip
                raise
            self.attempts.pop()
            self.allowed.append(str(resolved))


# ------------------------------------------------------------------ a run's sealed folders


@pytest.fixture
def runs(tmp_path, monkeypatch) -> RunWorld:
    """The records tests' world, its sealed files holding ``SENTINEL``."""
    monkeypatch.setattr(test_study3_records, "SENTINEL", SENTINEL)
    return RunWorld(tmp_path / "runs")


def link(path: Path, target: Path | str) -> None:
    """``path`` replaced by a symbolic link to ``target``."""
    path.unlink()
    path.symlink_to(target)


def test_every_analysis_module_is_classified_once() -> None:
    """Every analysis/*.py module is in exactly one fence: the label-free study-3 package (its
    label-free checks), study 3's label readers (their own sealing tests) or the others (study 1
    and 2, shared modules, stage_a among them, which stays a GPU path). A new module of any name
    fails here until it is classified."""
    found = set(ANALYSIS.glob("*.py"))
    lists = (set(PACKAGE), {path for path in LABEL_READERS if path.parent == ANALYSIS}, set(OTHER))
    assert set().union(*lists) == found
    assert sum(len(x) for x in lists) == len(found)
    assert ANALYSIS / "stage_a.py" in set(OTHER)


@pytest.mark.parametrize("name", RECORD_FILES)
def test_a_record_file_linked_into_its_runs_responses_is_refused_unopened(runs, monkeypatch,
                                                                         name) -> None:  # fmt: skip
    """A record file of a run that links into the run's ``responses/`` folder: the attempt is
    refused before anything of the run is read, the file named; nothing sealed is opened, and the
    run beside it reads as before."""
    linked = runs.run("A", 1.45e-5, 10300)
    ordinary = runs.run("A", 2.11e-5, 10300)
    run_dir = runs.root / "study3" / linked
    link(run_dir / name, Path("responses") / "harmful_evaluation.jsonl")
    guard = SealedGuard(monkeypatch)
    found, problems = runs.read()
    assert guard.attempts == []
    endpoint = by_id(found)[linked]
    assert (endpoint.state, endpoint.counts) == ("refused", False)
    why = f"its {name} resolves under {FOLDER}: never opened"
    assert why in endpoint.reasons
    assert [p for p in problems if p.startswith(linked)] == [f"{linked}: {why}"]
    assert by_id(found)[ordinary].state == "usable"
    assert SENTINEL not in repr((found, problems))
    # a reading of the same snapshot opens nothing sealed either
    reader = EndpointReader(runs.root, theta0=runs.theta0)
    reader.read(runs.prescribed)
    assert reader.meta_status(run_dir) == (None if name == "meta.json" else "completed")
    assert guard.attempts == []


def test_records_linked_into_observations_or_another_runs_responses_are_refused(runs, monkeypatch):
    """A K-only run's config and meta linked into its ``observations/`` folder, and a regular
    run's terminal record linked into another run's ``responses/`` (an absolute link): each refused,
    every such file named; nothing sealed opened."""
    k_only = runs.run("A", 1.45e-5, 10100, role="search")
    regular = runs.run("F", 1.3e-5, 10300)
    other = runs.run("F", 1.83e-5, 10300)
    folder = runs.root / "study3"
    for name in ("config.yaml", "meta.json"):
        link(folder / k_only / name, Path("observations") / "measurements.jsonl")
    link(folder / regular / "terminal.json",
         (folder / other / "responses" / "harmful_evaluation.jsonl").absolute())  # fmt: skip
    guard = SealedGuard(monkeypatch)
    found, problems = runs.read()
    assert guard.attempts == []
    assert (f"{k_only}: its config.yaml and meta.json resolve under {FOLDER}: never "
            "opened") in problems  # fmt: skip
    assert f"{regular}: its terminal.json resolves under {FOLDER}: never opened" in problems
    assert by_id(found)[other].state == "usable"
    assert SENTINEL not in repr((found, problems))


def test_a_run_folder_linked_under_another_runs_responses_is_refused_unopened(runs, monkeypatch):
    """A run's folder that is itself a link to a folder under another run's ``responses/``: every
    record file of it resolves there, and none is opened."""
    moved = runs.run("A", 1.45e-5, 10300)
    other = runs.run("A", 2.11e-5, 10300)
    folder = runs.root / "study3"
    target = folder / other / "responses" / moved
    shutil.move(folder / moved, target)
    (folder / moved).symlink_to(target)
    guard = SealedGuard(monkeypatch)
    found, problems = runs.read()
    assert guard.attempts == []
    assert (f"{moved}: its run folder resolves under {FOLDER}: nothing of it is "
            "opened") in problems  # fmt: skip
    assert by_id(found)[moved].state == "refused" and by_id(found)[other].state == "usable"
    assert SENTINEL not in repr((found, problems))


def test_a_link_that_stays_outside_the_sealed_folders_reads_as_before(runs, monkeypatch) -> None:
    """The rule is the sealed folders, not links: a record file that links to a copy elsewhere
    is read, and the run is usable."""
    run_id = runs.run("A", 1.45e-5, 10300)
    run_dir = runs.root / "study3" / run_id
    elsewhere = runs.root.parent / "copies" / run_id
    elsewhere.mkdir(parents=True)
    for name in ("meta.json", "config.yaml", "terminal.json"):
        shutil.copy(run_dir / name, elsewhere / name)
        link(run_dir / name, elsewhere / name)
    guard = SealedGuard(monkeypatch)
    found, problems = runs.read()
    assert guard.attempts == [] and problems == []
    assert by_id(found)[run_id].state == "usable"


def test_the_config_a_qualification_binds_is_never_read_through_a_sealed_link(runs, monkeypatch):
    """``qualification_records`` binds a start's case through ``study3_records._config``: a
    ``config.yaml`` linked into the run's ``responses/`` is a refusal, named, and is not opened;
    an ordinary one binds."""
    linked = runs.run("A", 1.45e-5, 10200, role="numerical")
    ordinary = runs.run("A", 2.11e-5, 10200, role="numerical")
    folder = runs.root / "study3"
    (folder / linked / "responses").mkdir()
    (folder / linked / "responses" / "harmful_evaluation.jsonl").write_text(SENTINEL + "\n")
    link(folder / linked / "config.yaml", Path("responses") / "harmful_evaluation.jsonl")
    _, chash, command = config_of("A", 1.45e-5, 10200, "numerical")
    _, other_hash, other_command = config_of("A", 2.11e-5, 10200, "numerical")
    guard = SealedGuard(monkeypatch)
    found, unbound, refusal = study3_records._config(folder / linked,
                                                     {"command": list(command)}, chash)  # fmt: skip
    assert (found, unbound) == (None, None)
    assert refusal == f"its config.yaml resolves under {FOLDER}: never opened"
    found, unbound, refusal = study3_records._config(folder / ordinary,
                                                     {"command": list(other_command)},
                                                     other_hash)  # fmt: skip
    assert config_hash(found) == other_hash and (unbound, refusal) == (None, None)
    assert guard.attempts == []


def test_theta0s_summary_is_never_read_through_a_sealed_link(runs, monkeypatch) -> None:
    """θ₀'s evaluation is a run's too: its ``summary.json`` linked into its ``responses/``, or a
    θ₀ named inside a sealed folder, is refused unopened; the ordinary one is read."""
    assert read_theta0(runs.theta0)["benchmark_accuracy"] == 0.60
    sealed = runs.theta0 / "responses"
    sealed.mkdir()
    (sealed / "harmful_evaluation.jsonl").write_text(SENTINEL + "\n")
    shutil.copy(runs.theta0 / "summary.json", sealed / "summary.json")
    guard = SealedGuard(monkeypatch)
    for named in (sealed, sealed / "summary.json"):
        with pytest.raises(ValueError, match=f"{FOLDER}: never opened"):
            read_theta0(named)
    link(runs.theta0 / "summary.json", Path("responses") / "harmful_evaluation.jsonl")
    with pytest.raises(ValueError, match=f"{FOLDER}: never opened") as found:
        read_theta0(runs.theta0)
    assert SENTINEL not in str(found.value) and guard.attempts == []


# ------------------------------------------------------------------ a start's command


def test_a_start_command_composes_only_a_study3_config(runs, monkeypatch) -> None:
    """A journal start's command is composed again only when its config is one the launcher
    composes study 3's runs from (``experiments/study3``, ``experiments/study3_k_only``), and its
    section overrides name config files by their plain names; any other is refused by name and
    never loaded."""
    loaded = []
    real = study3_records.load_run_config

    def recorded(name, overrides=()):
        loaded.append(name)
        return real(name, overrides)

    monkeypatch.setattr(study3_records, "load_run_config", recorded)
    for role in ("validation", "search"):
        _, chash, command = config_of("A", 1.45e-5, 10300 if role == "validation" else 10100, role)
        assert config_hash(study3_records._command_config(list(command))) == chash
    assert loaded == ["experiments/study3", "experiments/study3_k_only"]
    _, _, command = config_of("A", 1.45e-5, 10300, "validation")
    script, _, *overrides = command
    for name in ("experiments/smoke", "experiments/no_such_config", "experiments/study3.yaml",
                 "experiments/../experiments/study3", "../configs/experiments/study3",
                 "configs/experiments/study3", str(CONFIG_ROOT / "experiments" / "study3"),
                 str(CONFIG_ROOT / "experiments" / "study3.yaml"), "experiments/sub/study3",
                 "/etc/hostname", "", "experiments/study3/"):  # fmt: skip
        with pytest.raises(FileNotFoundError, match="not a study-3 config") as found:
            study3_records._command_config([script, name, *overrides])
        assert repr(name) in str(found.value)
    for bad in ("optimizer=../optimizer/adamw", "optimizer=/etc/hostname", "optimizer=sub/adamw",
                "optimizer=..", "model=../../runs/x"):  # fmt: skip
        with pytest.raises(ValueError, match="names no config file by its plain name") as found:
            study3_records._command_config([script, "experiments/study3", *overrides, bad])
        assert repr(bad) in str(found.value)
    assert loaded == ["experiments/study3", "experiments/study3_k_only"]  # none of them loaded


def test_a_start_whose_command_names_another_config_gives_no_config(runs, monkeypatch) -> None:
    """Without a ``config.yaml``, a start whose command names a config the launcher does not
    compose study 3 from binds nothing: technical, the reason named, nothing loaded; the
    launcher's own command binds as before."""
    ordinary = runs.run("A", 1.45e-5, 10300, end=None, directory=False)
    crafted = runs.run("A", 2.11e-5, 10300, end=None, directory=False)
    journal = runs.root / "launch" / "study3-c1.jsonl"
    test_study3_records.edit_jsonl(journal, lambda rows: [
        {**r, "command": [r["command"][0], "../configs/experiments/study3", *r["command"][2:]]}
        if r.get("run_id") == crafted else r for r in rows])  # fmt: skip
    loaded = []
    real = study3_records.load_run_config
    monkeypatch.setattr(
        study3_records,
        "load_run_config",
        lambda name, overrides=(): loaded.append(name) or real(name, overrides),
    )
    found, problems = runs.read()
    assert problems == []
    assert (by_id(found)[ordinary].state, by_id(found)[ordinary].arm) == ("pending", "A")
    endpoint = by_id(found)[crafted]
    assert (endpoint.state, endpoint.arm) == ("technical", None)
    assert endpoint.reasons == (
        "no config.yaml, and its start's command names the config "
        "'../configs/experiments/study3', not a study-3 config (experiments/study3, "
        "experiments/study3_k_only): not loaded",
    )
    assert loaded == ["experiments/study3"]


# ------------------------------------------------------------------ records under the runs root

SEALED_WHY = f"resolves under {FOLDER}: never opened"


def qualification_reading(root: Path) -> dict:
    """Candidate 1's qualification as the real reader reads ``root``."""
    return REAL_READING(root, 1, expected_code="e" * 64,
                        settings=load_config("experiments/qualification"))  # fmt: skip


def test_logs_and_qualification_records_linked_into_a_runs_responses_are_refused(runs, monkeypatch):
    """A launch log, a case-restart record and a supersession record linked into a run's
    ``responses/`` (the base is the runs root): each is a named problem of every reader of the
    journal and of the qualification's records, and none is opened; the ordinary log beside them
    is read as before."""
    run_id = runs.run("A", 1.45e-5, 10300)
    other = runs.run("A", 2.11e-5, 10300)
    sealed = runs.root / "study3" / run_id / "responses" / "harmful_evaluation.jsonl"
    (runs.root / "launch" / "linked.jsonl").symlink_to(sealed)
    case = runs.root / "qualification" / "c1" / "numerical" / "s10200" / "adamw-1.45e-05"
    case.mkdir(parents=True)
    restart, cycle = (
        case / "restart-1.json",
        runs.root / "qualification" / "c1" / "supersede-1.json",
    )
    restart.symlink_to(sealed)
    cycle.symlink_to(sealed)
    log = f"launch/linked.jsonl: a launch log that {SEALED_WHY}"
    guard = SealedGuard(monkeypatch)
    found, problems = runs.read()
    held = f"qualification/c1/supersede-1.json: a supersession record that {SEALED_WHY}"
    assert {log, f"{restart}: a case-restart record that {SEALED_WHY}", held} <= set(problems)
    assert by_id(found)[other].state == "usable"
    logs, unreadable = journal_lines(runs.root)
    assert [entry.path for entry in logs] == ["launch/study3-c1.jsonl"] and unreadable == [log]
    _, unreadable = read_holds(runs.root)
    assert unreadable == [log, held]  # the same problem, in the same form
    pinned, found_logs = sv.qualification_logs(runs.root, 1)
    assert [p for p in found_logs if p.startswith(log)]
    reading = qualification_reading(runs.root)
    for where, what in (
        (runs.root / "launch" / "linked.jsonl", "launch log"),
        (restart, "case-restart record"),
        (cycle, "supersession record"),
    ):
        assert f"{where}: a {what} that {SEALED_WHY}" in reading["problems"], where
    (listed,) = qualification_records.supersessions(runs.root, 1)
    assert listed["sha256"] is None and listed["diagnosis"] is None
    assert guard.attempts == []
    assert SENTINEL not in repr((found, problems, reading, listed, pinned, found_logs))


def test_the_qualification_reader_reads_no_run_file_through_a_sealed_link(runs, monkeypatch):
    """The qualification reader's run files go through the rule with the run's folder as base: a
    numerical run whose ``config.yaml`` and ``meta.json`` link into its ``observations/`` is
    refused unopened, named, and no launcher run is located there; the ordinary run beside it is
    located and its identity files read. A start whose command names another config is a problem
    that says why."""
    linked = runs.run("A", 1.45e-5, 10200, role="numerical")
    ordinary = runs.run("F", 1.3e-5, 10200, role="numerical")
    crafted = runs.run("P", 1.3e-5, 10200, role="numerical")
    folder = runs.root / "study3"
    for name in ("config.yaml", "meta.json"):
        link(folder / linked / name, Path("observations") / "measurements.jsonl")
    journal = runs.root / "launch" / "study3-c1.jsonl"
    test_study3_records.edit_jsonl(journal, lambda rows: [
        {**r, "command": [r["command"][0], "experiments/smoke", *r["command"][2:]]}
        if r.get("run_id") == crafted and r["event"] == "start" else r for r in rows])  # fmt: skip
    guard = SealedGuard(monkeypatch)
    reading = qualification_reading(runs.root)
    why = f"its config.yaml and meta.json resolve under {FOLDER}"
    assert (f"{folder / linked}: {why}: never opened; no launcher run is located there"
            in reading["problems"])  # fmt: skip
    assert any(p.startswith(f"{linked} (A-1.45e-05/s10200): a start of candidate 1's "
                            "qualification refused: its config.yaml resolves under")
               for p in reading["problems"])  # fmt: skip
    assert any("P-1.3e-05/s10200" in p and "does not compose under this checkout's configs "
               "(FileNotFoundError: names the config 'experiments/smoke', not a study-3 config"
               in p for p in reading["problems"])  # fmt: skip
    assert qualification_records.rate_of(reading, "F", 1.3e-5)["launchers"][10200] is not None
    assert qualification_records.launcher_identity(folder / linked) is None
    assert qualification_records.launcher_identity(folder / ordinary)["run_id"] == ordinary
    assert guard.attempts == [] and SENTINEL not in repr(reading)


# ------------------------------------------------------------------ a run folder is a plain child


def plant(folder: Path) -> list[str]:
    """Files a crafted journal could point a reader at: values that must never show in a reason.
    Their resolved paths."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(
        json.dumps(
            {"config_hash": "PLANTED-hash", "status": "completed", "campaign": {"name": SENTINEL}}
        )
    )
    (folder / "config.yaml").write_text(f"seed: 777\nname: {SENTINEL}\n")
    (folder / "summary.json").write_text(json.dumps({"benchmark_accuracy": 0.123,
                                                     "factual_accuracy": 0.456,
                                                     "text": SENTINEL}))  # fmt: skip
    return [
        os.path.realpath(folder / name) for name in ("meta.json", "config.yaml", "summary.json")
    ]


CRAFTED = ("a folder inside a run's responses", "..", "an absolute path")


def crafted_start(runs: RunWorld, real: str, kind: str, header_: dict,
                  campaign: str = "study3-c1") -> tuple[str, list[str]]:  # fmt: skip
    """A crafted start whose run id is no plain name (``kind``: ``<run>/responses/sub``, ``..`` or
    an absolute path), with files planted where it points; its run id and the planted files."""
    if kind == CRAFTED[0]:
        run_id = f"{real}/responses/sub"
        folder = runs.root / "study3" / run_id
    elif kind == "..":
        run_id, folder = "..", runs.root  # the experiment's folder's parent: the runs root
    else:
        folder = runs.root.parent / "planted" / "run"
        run_id = str(folder)
    planted = plant(folder)
    _, chash, command = config_of("A", 2.11e-5, 10300, "validation")
    runs.log(campaign, {"event": "start", "run": "A-2.11e-05/s10300", "attempt": 1,
                        "run_id": run_id, "experiment": "study3", "config_hash": chash,
                        "command": list(command), "header": header_})  # fmt: skip
    return run_id, planted


PLANTED = ("PLANTED", "777", "0.123", "0.456", SENTINEL)


@pytest.mark.parametrize("kind", CRAFTED)
def test_a_start_whose_run_id_is_no_plain_name_binds_no_run(runs, monkeypatch, kind) -> None:
    """A journal start whose run id is not a plain name (a folder inside a run's ``responses/``,
    ``..`` or an absolute path; files planted where it points) binds no run: a named problem,
    nothing under it opened, and no value of those files in any reason; the ordinary run beside
    it binds as before."""
    real = runs.run("A", 1.45e-5, 10300)
    run_id, planted = crafted_start(runs, real, kind, header(1, "validation", None))
    guard = SealedGuard(monkeypatch)
    found, problems = runs.read()
    assert guard.attempts == [] and not set(planted) & set(guard.opened)
    why = (f"its run id {run_id!r} is not a plain name: it names no run folder, and nothing "
           "under it is opened")  # fmt: skip
    assert f"{run_id}: {why}" in problems
    assert by_id(found)[run_id].state == "refused" and by_id(found)[real].state == "usable"
    shown = repr((found, problems))
    assert not [value for value in PLANTED if value in shown]
    for bad in ("", ".", "..", "a/b", "/abs", "a\\b", None, 3):
        assert study3_records.plain_name(bad) is False, bad
    assert study3_records.plain_name(real) is True


@pytest.mark.parametrize("kind", CRAFTED)
def test_the_versions_check_binds_no_run_whose_id_is_no_plain_name(tmp_path, monkeypatch, kind):
    """The same starts under a validation campaign whose header names the versions record:
    ``versions_problems`` names each, opens nothing under it, and shows none of its values; the
    ordinary runs beside it raise nothing."""
    monkeypatch.setattr(test_study3_records, "SENTINEL", SENTINEL)
    study = VersionsStudy(tmp_path, monkeypatch)
    study.validated()
    real, _ = study.block(10300)
    assert study.problems(step="validation", candidate=1) == []
    run_id, planted = crafted_start(study.runs, real, kind, study.header(),
                                    campaign="study3-c1-validation-x")  # fmt: skip
    guard = SealedGuard(monkeypatch)
    problems = study.problems(step="validation", candidate=1)
    assert guard.attempts == [] and not set(planted) & set(guard.opened)
    assert problems == [f"study3-c1-validation-x: A-2.11e-05/s10300 ({run_id}): its run id "
                        f"{run_id!r} is not a plain name: it names no run folder, and nothing "
                        "under it is opened"]  # fmt: skip
    assert not [value for value in PLANTED if value in repr(problems)]


def test_a_theta0_inside_a_sealed_folder_is_refused_unopened(runs, monkeypatch) -> None:
    """θ₀ named anywhere inside a run's ``responses/`` (two levels down, files planted there) is
    refused unopened, whether its folder or its file is named; the ordinary θ₀ is read."""
    real = runs.run("A", 1.45e-5, 10300)
    inner = runs.root / "study3" / real / "responses" / "inner"
    plant(inner)
    guard = SealedGuard(monkeypatch)
    for named in (inner, inner / "summary.json"):
        with pytest.raises(ValueError, match=f"{FOLDER}: never opened") as found:
            read_theta0(named)
        assert not [value for value in PLANTED if value in str(found.value)]
    assert read_theta0(runs.theta0)["benchmark_accuracy"] == 0.60
    assert guard.attempts == []


# ------------------------------------------------------------------ loops and sealed logs


LOOP_WHY = "cannot be resolved (a link loop): never opened"


def test_a_link_loop_is_a_named_problem_not_a_crash(runs, monkeypatch, tmp_path) -> None:
    """A record file that is a link to itself, a launch log, a decision record, θ₀'s summary and a
    campaign file that are one: each refused unopened (fail-closed) and named as a link loop, not
    as a sealed folder; the readers go on."""
    looped = runs.run("A", 1.45e-5, 10300)
    ordinary = runs.run("A", 2.11e-5, 10300)
    meta = runs.root / "study3" / looped / "meta.json"
    link(meta, meta)
    log = runs.root / "launch" / "loop.jsonl"
    log.symlink_to(log)
    record = decision_path(runs.root, "engineering")
    record.parent.mkdir(parents=True)
    record.symlink_to(record)
    guard = SealedGuard(monkeypatch)
    found, problems = runs.read()
    assert f"{looped}: its meta.json {LOOP_WHY}" in problems
    assert f"launch/loop.jsonl: a launch log that {LOOP_WHY}" in problems
    assert not [p for p in problems if "loop" in p and FOLDER in p]
    assert by_id(found)[ordinary].state == "usable"
    _, unreadable = journal_lines(runs.root)
    assert unreadable == [f"launch/loop.jsonl: a launch log that {LOOP_WHY}"]
    reading = qualification_reading(runs.root)
    assert f"{log}: a launch log that {LOOP_WHY}" in reading["problems"]
    entry = decisions(runs.root, 1, repo=tmp_path)["engineering"]
    assert entry["state"] == "invalid" and f"{record}: {LOOP_WHY}" in entry["problems"]
    summary = runs.theta0 / "summary.json"
    kept = summary.read_bytes()
    link(summary, summary)
    with pytest.raises(ValueError, match=r"cannot be resolved \(a link loop\): never opened"):
        read_theta0(runs.theta0)
    summary.unlink()
    summary.write_bytes(kept)
    campaign = tmp_path / "loop.yaml"
    campaign.symlink_to(campaign)
    script = script_module(SUMMARY, "study3_summary")
    with pytest.raises(SystemExit) as refused:
        script.main(["check", "--runs-root", str(runs.root), "--campaign", str(campaign)])
    assert "cannot be resolved (a link loop)" in str(refused.value.code)
    assert guard.attempts == []


def test_the_recovery_reading_opens_nothing_of_a_start_that_binds_no_run(runs, monkeypatch):
    """A journal start whose run id is ``../elsewhere`` (files planted at ``RUNS_DIR/elsewhere``)
    is refused; which runs the launcher would open a recovery round for
    (``study3_progress._recoverable``) is read without opening anything of it, and it is none."""
    real = runs.run("A", 1.45e-5, 10300, end=None, status="failed", terminal=False)
    planted = plant(runs.root / "elsewhere")
    _, chash, command = config_of("A", 2.11e-5, 10300, "validation")
    runs.log("study3-c1", {"event": "start", "run": "A-2.11e-05/s10300", "attempt": 1,
                           "run_id": "../elsewhere", "experiment": "study3",
                           "config_hash": chash, "command": list(command),
                           "header": header(1, "validation", None)})  # fmt: skip
    guard = SealedGuard(monkeypatch, runs_root=runs.root)
    reader = EndpointReader(runs.root, theta0=runs.theta0)
    endpoints, _ = reader.read(runs.prescribed)
    assert by_id(endpoints)["../elsewhere"].state == "refused"
    found = study3_progress._recoverable(reader, endpoints)
    assert "../elsewhere" not in found and real in found
    assert guard.attempts == [] and not set(planted) & set(guard.opened)


def test_a_sealed_launch_log_is_named_in_the_decision_records(tmp_path, monkeypatch) -> None:
    """``decisions`` read on its own: a launch log that resolves under a sealed folder makes each
    existing record fail closed, the log named by its path under the runs root."""
    world = DecisionWorld(tmp_path)
    world.write("engineering", world.engineering())
    (world.runs / "launch").mkdir()
    (world.runs / "launch" / "linked.jsonl").symlink_to(
        world.run / "responses" / "harmful_evaluation.jsonl")  # fmt: skip
    guard = SealedGuard(monkeypatch)
    entry = decisions(world.runs, 1, repo=world.repo)["engineering"]
    assert entry["state"] == "invalid"
    assert (f"{decision_path(world.runs, 'engineering')}: a launch log resolves under a sealed "
            "folder (launch/linked.jsonl), so whether a campaign has named this record cannot be "
            "shown") in entry["problems"]  # fmt: skip
    assert guard.attempts == []


# ------------------------------------------------------------------ no GPU path


def test_the_package_changes_no_gpu_path(monkeypatch) -> None:
    """No GPU path has changed since the GPU code study 3 qualifies, as of package 1d
    (``QUALIFIED_GPU_CODE``): the qualification made under that GPU code holds at HEAD
    (``code_identity.reuse_check``)."""
    try:
        known = subprocess.run(["git", "merge-base", "--is-ancestor", QUALIFIED_GPU_CODE, "HEAD"],
                               cwd=REPO_ROOT, capture_output=True).returncode == 0  # fmt: skip
    except OSError:
        known = False
    if not known:
        pytest.skip(f"the GPU code study 3 qualifies ({QUALIFIED_GPU_CODE}) is not in this "
                    "repository's history")  # fmt: skip
    monkeypatch.setattr(code_identity, "gpu_code_identity", REAL_IDENTITY)
    found = code_identity.reuse_check(QUALIFIED_GPU_CODE, "HEAD")
    assert found["changed"] == [], found["changed"]
    assert found["reusable"] is True, found


# ------------------------------------------------------------------ imports


def test_the_package_imports_no_gpu_module_and_loads_no_script() -> None:
    """Every module of the package and both scripts, read as code (``test_code_identity.gpu_uses``,
    relative imports resolved, imports inside functions included): none imports directly
    ``zero_field_refusal.qualification``, ``schedule``, ``train`` or any other GPU module, nor
    torch; none uses CUDA; none imports ``scripts/qualify.py`` (``qualify``, ``scripts.qualify``),
    loads code at run time through the import machinery, or runs a subprocess whose arguments name
    ``qualify.py``. The check is on what each file imports directly, not on what those modules
    load in turn, and it is per file. Transitively, the analysis scripts do load torch,
    ``zero_field_refusal.schedule``, ``train`` and ``runtime``, through
    ``analysis.qualification_verdicts`` → ``fa_control_diagnostic``: they call none of their GPU
    paths. Its allowed edges and what they load:
    - ``analysis.qualification_verdicts``, which ``qualification_records`` imports (the other
      modules import ``qualification_records`` only inside the functions that read the
      qualification), imports ``fa_control_diagnostic``, which loads torch and
      ``zero_field_refusal.schedule``, ``train`` and ``runtime``;
    - ``campaign_prep``'s lazy import of ``analysis.study2_short`` (study 2's ``prep`` only) loads
      torch and ``zero_field_refusal.runtime``.
    The scripts run no training: none calls a GPU path."""
    assert len(PACKAGE) == 12 and all(path.exists() for path in (*PACKAGE, *SCRIPTS))
    for path in PACKAGE:
        assert gpu_uses(path, "zero_field_refusal.analysis") == set(), path.name
    for path in SCRIPTS:
        assert gpu_uses(path) == set(), path.name


# ------------------------------------------------------------------ the route on files


@dataclasses.dataclass
class Ran:
    """A command run in process: its script, its arguments, its exit code, its output, and its
    error output with the message it exited with."""

    script: str
    argv: tuple[str, ...]
    code: int
    out: str
    err: str


class LabelReaders:
    """The public entry points of the label readers (the functions ``LABEL_MODULES`` define),
    replaced wherever a loaded module binds them by one that records the call and raises; ``lift``
    gives every binding of one its real function back for a test. ``ACTIVE`` lists the instances
    in force (the route's, for its module)."""

    ACTIVE: list["LabelReaders"] = []

    def __init__(self, monkeypatch) -> None:
        self.calls: list[str] = []
        self.names: list[str] = []
        self.real: dict[int, tuple[Any, Any]] = {}  # id(raiser) -> (raiser, real function)
        found: dict[int, tuple[Any, Any]] = {}
        for name in LABEL_MODULES:
            module = importlib.import_module(name)
            for attr, value in sorted(vars(module).items()):
                if (attr.startswith("_") or not inspect.isfunction(value)
                        or value.__module__ != name):  # fmt: skip
                    continue
                self.names.append(f"{name}.{attr}")
                found[id(value)] = (value, self._raiser(f"{name}.{attr}"))
                self.real[id(found[id(value)][1])] = (found[id(value)][1], value)
        for module in list(sys.modules.values()):
            try:
                items = list(vars(module).items())
            except TypeError:
                continue
            for attr, value in items:
                entry = found.get(id(value))
                if entry is not None and entry[0] is value:
                    monkeypatch.setattr(module, attr, entry[1])

    def _raiser(self, name: str):
        def raiser(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"a label reader was called: {name}")

        return raiser

    def lift(self, monkeypatch) -> None:
        """Every binding of a replaced entry point, in every module loaded now, back to the real
        function, for the test of ``monkeypatch`` (undone with it)."""
        for module in list(sys.modules.values()):
            try:
                items = list(vars(module).items())
            except TypeError:
                continue
            for attr, value in items:
                entry = self.real.get(id(value))
                if entry is not None and entry[0] is value:
                    monkeypatch.setattr(module, attr, entry[1])


def judged(run_dir: Path) -> None:
    """A judge's labels in the run's ``judge/`` folder, as the labeling writes them, and a copy
    moved aside to its ``judge-superseded/`` folder: the sentinel."""
    for folder in (run_dir / JUDGE, run_dir / SUPERSEDED / "b0001" / "g1-unverified"):
        folder.mkdir(parents=True)
        (folder / "wildguard.jsonl").write_text(json.dumps({"label": SENTINEL}) + "\n")


def round0(arm, rate, seed, candidate) -> dict:
    """The search's round 0: A's lower rate kept (its median within 0.05 in ln K) though it
    misses on one seed; F's upper rate 0.2 off its target; every other run at its target."""
    if (arm, rate) == ("A", START["A"]["l"]):
        return {"k": 4.8 * math.exp(OFFSETS[seed])}
    found = target(arm, rate, seed, candidate)
    return {"k": found["k"] * math.exp(0.2)} if (arm, rate) == ("F", START["F"]["u"]) else found


def numerical(arm, rate, seed, candidate) -> dict:
    """The numerical runs: A's lower rate 0.3 off its target, so that it misses over its 4 + 3
    seeds and is repaired."""
    if (arm, rate) == ("A", START["A"]["l"]):
        return {"k": 4.8 * math.exp(0.3)}
    return target(arm, rate, seed, candidate)


class Route(CampaignWorld):
    """The campaign writer's world (``test_study3_campaigns.World``) in a Git checkout, every
    campaign kind and every summary command run in process on its files, from the engineering
    campaign to the confirmation. The state is the real ``progress``, the versions checks are the
    real ``versions_problems``, the GPU code is the checkout's (``gpu_code_identity``); only the
    qualification's reading (``read_qualification``) is given: its rates qualified, its input
    records on disk. Every sealed file holds ``SENTINEL``."""

    def __init__(self, base: Path, monkeypatch) -> None:
        super().__init__(base, monkeypatch, None)
        self.base = base
        git(self.repo, "init", "-q")
        (self.repo / ".gitignore").write_text("/evidence/\n")  # the decision records' evidence
        (self.repo / "docs").mkdir()
        (self.repo / REGISTRATION).write_text("# Study 3 preregistration\n" + AMENDMENTS)
        code = self.repo / "src" / "zero_field_refusal" / "train.py"
        code.parent.mkdir(parents=True)
        code.write_text("x = 1\n")
        stand_in(self.repo)  # the spread script and the control list the fixation pins
        self.commit("the registration and the code")
        monkeypatch.setattr(code_identity, "gpu_code_identity", self.identity)
        head = git(self.repo, "rev-parse", "HEAD")
        self.code = {"commit": head, "gpu_code_sha256": REAL_IDENTITY(head, self.repo)["sha256"]}
        monkeypatch.setattr(qualification_records, "read_qualification", self.qualification)
        monkeypatch.setattr("zero_field_refusal.launch.load_run_config", run_config)
        self.summary = script_module(SUMMARY, "study3_summary")
        monkeypatch.setattr(self.summary, "REPO", self.repo)
        self.inputs = base / "inputs"
        self.inputs.mkdir()
        for theta0 in (self.runs.theta0, self.root / THETA0_RUN):  # θ₀'s run has them too
            for folder, name in (("responses", "harmful_evaluation.jsonl"),
                                 ("observations", "measurements.jsonl")):  # fmt: skip
                (theta0 / folder).mkdir()
                (theta0 / folder / name).write_text(json.dumps({"text": SENTINEL}) + "\n")
            judged(theta0)
        # the labeling ledger, in the experiment's folder: no run, never opened
        ledger = self.root / "study3" / LEDGER / "journal.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"label": SENTINEL}) + "\n")
        self.ran: list[Ran] = []
        self.guards: list[tuple[int, int]] = []  # (seed, exit code) of each guard run
        self.launched: dict[str, list[str]] = {}  # campaign file -> its run ids
        self.readings: list[tuple[int, str]] = []
        self.guard: SealedGuard | None = None
        self.prescribed: dict = {}
        self.blocks: dict[int, Path] = {}

    # -- the checkout and the qualification

    def identity(self, commit: str = "HEAD", repo: Path = REPO_ROOT) -> dict:
        """``gpu_code_identity``, the world's checkout standing for this one (the writer reads
        its expected code from the default checkout, the summary script from its ``REPO``)."""
        if Path(repo).resolve() == REPO_ROOT.resolve():
            repo = self.repo
        return REAL_IDENTITY(commit, repo)

    def commit(self, message: str, *paths: str) -> None:
        git(self.repo, "add", *(paths or ("-A",)))
        git(self.repo, "commit", "-qm", message)

    def qualification(self, runs_root, candidate, *, expected_code, settings) -> dict:
        """The qualification's reading: the rates in ``qualified``, each with its case's input
        record (written by ``qualify``)."""
        self.readings.append((candidate, expected_code))
        out = {"rates": {}, "states": {}, "cases": [], "voided": {}, "supersessions": [],
               "held": [], "problems": [], "inputs": []}  # fmt: skip
        for arm, lr in sorted(self.qualified[candidate]):
            key = f"{OPT[arm]}-{lr:g}"
            case = self.case(candidate, key)
            out["inputs"].append(str(case / "baseline-attempt-1.json"))
            out["rates"][key] = {"key": key, "arm": arm, "optimizer": OPT[arm], "lr": lr,
                                 "state": "qualified", "runs": [], "held": [],
                                 "cases": [{"folder": str(case), "key": key}],
                                 "verdict": {"state": "qualified", "control": None,
                                             "not_counted": []}}  # fmt: skip
            out["states"][key] = "qualified"
        out["inputs"].sort()
        return out

    def case(self, candidate: int, key: str) -> Path:
        seed = SEEDS["numerical"][candidate][0]
        return self.root / "qualification" / f"c{candidate}" / "numerical" / f"s{seed}" / key

    def qualify(self, rates) -> None:
        """Candidate 1's rates qualified, each with its case's input record on disk."""
        for arm, lr in rates:
            self.qualified[1].add((arm, lr))
            folder = self.case(1, f"{OPT[arm]}-{lr:g}")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "baseline-attempt-1.json").write_text(json.dumps({"part": "baseline"}))

    # -- the commands, in process

    def run(self, script: str, *argv) -> Ran:
        module = self.script if script == "campaigns" else self.summary
        argv = tuple(str(a) for a in argv)
        out, err = io.StringIO(), io.StringIO()
        code, message = 0, ""
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                module.main(list(argv))
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
                message = exc.code if isinstance(exc.code, str) else ""
        ran = Ran(script, argv, code, out.getvalue(), err.getvalue() + message)
        self.ran.append(ran)
        return ran

    def ok(self, script: str, *argv) -> Ran:
        ran = self.run(script, *argv)
        assert ran.code == 0, (ran.argv, ran.err)
        return ran

    def refused(self, script: str, *argv) -> Ran:
        ran = self.run(script, *argv)
        assert ran.code not in (0, None), ran.argv
        return ran

    def campaign(self, kind: str, *args) -> Path:
        """The writer's campaign of ``kind`` (θ₀ named, but for the confirmation, which reads it
        where ``progress`` reads it by default): its path, a folder for the confirmation."""
        out = self.out / (
            f"{kind}-{next(self.names)}" + ("" if kind == "confirmation" else ".yaml")
        )
        theta0 = () if kind == "confirmation" else ("--theta0", self.runs.theta0)
        self.ok("campaigns", kind, "--runs-root", self.root, "--candidate", 1, *theta0, "--out",
                out, *args)  # fmt: skip
        return out

    def record(self, kind: str, record: dict) -> None:
        source = self.inputs / f"{kind}.json"
        source.write_text(json.dumps(record))
        self.ok("summary", "record", kind, "--runs-root", self.root, "--from", source)

    def guarded(self, seed: int, path: Path) -> Ran:
        ran = self.run("summary", "guard", "--runs-root", self.root, "--candidate", 1, "--seed",
                       seed, "--campaign", path)  # fmt: skip
        self.guards.append((seed, ran.code))
        return ran

    def state(self) -> dict:
        return progress(self.root, repo=self.repo, expected_code=self.code["gpu_code_sha256"],
                        theta0=self.runs.theta0)  # fmt: skip

    def step(self) -> dict:
        state = self.state()
        assert state["problems"] == [], state["problems"]
        return state["step"]

    # -- the launcher and the tracker

    def launch(self, path: Path, values=target, technical=(), dirty=()) -> list[str]:
        """The campaign's runs started, written and ended as the launcher and the tracker do it
        (``World.launched``), the guard off while they are written; a run of ``technical``'s
        (arm, rate) ends technical, its meta.json failed, no terminal record; one of ``dirty``'s
        ran on a checkout with uncommitted changes (its meta.json's ``git.dirty``)."""
        name, runs = load_campaign(path)
        header = campaign_header(path)
        ids = []
        with self.guard.off():
            for run in runs:
                cfg = run_config(run.config, run.overrides)
                arm, rate = ARM_OF[str(cfg.optimizer.name)], float(cfg.optimizer.lr)
                run_id = (f"20261027T{next(self.run_ids):06d}Z-{run.cell[:28]}-s{run.seed}-"
                          f"{run.config_hash[:8]}-a1")  # fmt: skip
                self.runs.log(name, {"event": "start", "run": run.label, "attempt": 1,
                                     "run_id": run_id, "experiment": run.experiment,
                                     "config_hash": run.config_hash,
                                     "command": run.command()[1:], "header": header,
                                     "max_attempts": 1})  # fmt: skip
                failed = (arm, rate) in technical
                outcome = {"terminal": False, "status": "failed"} if failed else {}
                outcome |= values(arm, rate, run.seed, header["candidate"])
                write_run(self.root / run.experiment / run_id,
                          OmegaConf.to_yaml(cfg, resolve=True), run.config_hash,
                          {"name": name, "label": run.label, "attempt": 1, "header": header},
                          k_only=OmegaConf.select(cfg, "eval.k_only", default=False) is True,
                          git={**git_state(self.repo), "dirty": (arm, rate) in dirty},
                          **outcome)  # fmt: skip
                judged(self.root / run.experiment / run_id)
                self.runs.end(name, run.label, 1, run_id, "technical" if failed else "done")
                ids.append(run_id)
        self.launched[str(path)] = ids
        return ids

    # -- the route

    def build(self) -> None:
        """Candidate 1 from its engineering campaign to its confirmation's fourth block: every
        campaign kind, every summary command; the search's two rounds, a repair, the staged
        fixation and two versions records, N = 99 with its budget record, a P run under another
        commit (a dirty checkout) that P's closure record closes."""
        root, theta0 = self.root, self.runs.theta0
        assert self.step()["record"] == "engineering"
        self.launch(self.campaign("engineering"))
        self.record("engineering", self.records.engineering(code=self.code))
        self.record("simulation", self.records.simulation(code=self.code))
        assert self.step()["campaign"] == "search-round0"
        self.launch(self.campaign("search-round0"), round0)
        assert self.step()["campaign"] == "search-round1"
        self.launch(self.campaign("search-round1"))
        assert self.step()["campaign"] == "numerical"
        self.numerical_runs = self.launch(self.campaign("numerical"), numerical)
        adjusted = adjusted_rate(START["F"]["u"], math.log(7.5) + 0.2, 7.5, 1.30)
        self.qualify({(arm, START[arm][label]) for arm in OPT for label in LABELS
                      if (arm, label) != ("F", "u")} | {("F", adjusted)})  # fmt: skip
        assert self.step()["campaign"] == "repair"
        self.launch(self.campaign("repair"))
        self.qualify({("A", adjusted_rate(START["A"]["l"], math.log(4.8) + 0.3, 4.8, 1.25))})
        assert self.step()["campaign"] == "validation"
        # the validation fixation and the first versions record, each committed alone
        self.ok("summary", "progress", "--runs-root", root, "--theta0", theta0)
        self.ok("summary", "fixation", "validation", "--runs-root", root, "--candidate", 1,
                "--theta0", theta0)  # fmt: skip
        self.commit("the validation fixation", VALIDATION_C1)
        self.ok("summary", "versions", "--out", FIRST, "--reason", "validation_c1", "--fixation",
                VALIDATION_C1, "--runs-root", root)  # fmt: skip
        self.commit("the versions record", FIRST)
        validation = self.campaign("validation")
        self.ok("summary", "check", "--runs-root", root, "--campaign", validation)
        self.validation_runs = self.launch(validation)
        state = self.state()
        assert (state["stage"], state["step"]["record"], state["problems"]) == (
            "choice", "n-c1", [])  # fmt: skip
        fields = state["step"]["fields"]
        self.record("n-c1", self.records.n(1, N, rates=fields["rates"], active=fields["active"],
                                          cohort=fields["cohort"], code=self.code))  # fmt: skip
        self.record("budget-c1", self.records.budget(1, N, code=self.code))
        assert self.step()["campaign"] == "confirmation"
        # the confirmation supplement and the versions record that supersedes the first
        self.ok("summary", "fixation", "confirmation", "--runs-root", root, "--candidate", 1,
                *SUPPLEMENT_SEEDS)  # fmt: skip
        self.commit("the confirmation supplement", CONFIRMATION_C1)
        self.ok("summary", "versions", "--out", SECOND, "--reason", "confirmation_c1",
                "--fixation", CONFIRMATION_C1, "--runs-root", root, "--supersedes", FIRST,
                "--supersedes-reason", "confirmation_c1")  # fmt: skip
        self.commit("the second versions record", SECOND)
        # three blocks, each started only when the real guard lets it
        out = self.campaign("confirmation", "--blocks", 10400, 10402)
        self.blocks = {campaign_of(p)["cells"][0]["seeds"][0]: p for p in out.glob("*.yaml")}
        assert sorted(self.blocks) == [10400, 10401, 10402]
        self.prescribed = self.state()["prescribed"]
        self.ok("summary", "check", "--runs-root", root, "--campaign", self.blocks[10400])
        assert self.guarded(10400, self.blocks[10400]).code == 0
        assert self.guarded(10401, self.blocks[10401]).code != 0  # out of seed order
        self.launch(self.blocks[10400])
        assert self.guarded(10401, self.blocks[10401]).code == 0
        self.launch(self.blocks[10401])
        assert self.guarded(10402, self.blocks[10402]).code == 0
        self.launch(self.blocks[10402], dirty={("P", START["P"]["u"])})
        step = self.state()["step"]
        assert (step["kind"], step.get("technical")) == ("blocked", True), step
        assert self.guarded(10402, self.blocks[10402]).code != 0  # the mark holds it
        self.refused("campaigns", "confirmation", "--runs-root", root, "--candidate", 1, "--out",
                     self.out / "blocked")  # fmt: skip
        self.refused("summary", "closure", "--runs-root", root, "--candidate", 1, "--arm", "A")
        self.ok("summary", "closure", "--runs-root", root, "--candidate", 1, "--arm", "P")
        # P closed: the next block holds A and F
        out = self.campaign("confirmation")
        (block,) = out.glob("*.yaml")
        assert campaign_of(block)["cells"][0]["seeds"] == [10403]
        assert len(campaign_of(block)["cells"]) == 4
        self.blocks[10403] = block
        assert self.guarded(10403, block).code == 0
        self.launch(block)
        assert self.ok("summary", "progress", "--runs-root", root).out
        state = self.state()
        assert state["problems"] == [] and state["confirmation"]["started"] == [10400, 10401, 10402,
                                                                              10403]  # fmt: skip
        self.ok("campaigns", "prep", "--seeds", 10100, 10300, "--commit", self.code["commit"],
                "--out", self.out / "prep.sh")  # fmt: skip


@pytest.fixture(scope="module")
def route(tmp_path_factory):
    """The route built once, with the label readers replaced from the start and every sealed read
    failing while the commands run (``SealedGuard``, kept as ``route.attempts``); the guard is
    lifted once the route is built."""
    # the label readers' test modules, which import this one, loaded before the entry points are
    # replaced: the replacement then covers their bindings, and its undo restores them
    importlib.import_module("test_study3_label_cli")
    importlib.import_module("test_study3_spread")
    patch = pytest.MonkeyPatch()
    sealed = pytest.MonkeyPatch()
    try:
        labels = LabelReaders(patch)
        LabelReaders.ACTIVE.append(labels)
        patch.setattr(test_study3_records, "SENTINEL", SENTINEL)
        world = Route(tmp_path_factory.mktemp("route"), patch)
        world.labels = labels
        world.guard = SealedGuard(sealed, runs_root=world.root)
        world.build()
        # the guard is live: a sealed file, a judge output, a superseded one or the ledger cannot
        # be read under it
        live = len(world.guard.attempts)
        live_files = (next(world.root.glob("study3/*/responses/harmful_evaluation.jsonl")),
                      next(world.root.glob(f"study3/*/{JUDGE}/wildguard.jsonl")),
                      next(world.root.glob(f"study3/*/{SUPERSEDED}/*/*/wildguard.jsonl")),
                      world.root / "study3" / LEDGER / "journal.jsonl")  # fmt: skip
        for sealed_file in live_files:
            with pytest.raises(AssertionError):
                sealed_file.read_text()
        assert len(world.guard.attempts) == live + len(live_files)
        del world.guard.attempts[live:]
        world.attempts = list(world.guard.attempts)
        sealed.undo()
        yield world
    finally:
        sealed.undo()
        patch.undo()
        LabelReaders.ACTIVE.clear()


def sealed_part(path: Path, route) -> bool:
    """Whether a file of the route's world lies in a sealed folder, a run's judge folder or its
    superseded copies, or the labeling ledger."""
    below = path.relative_to(route.base).parts
    judge = path.is_relative_to(route.root) and any(
        part in UNDER_RUNS for part in path.relative_to(route.root).parts)  # fmt: skip
    return judge or any(part in SEALED for part in below)


def written(route) -> dict[Path, bytes]:
    """Every file of the route's world but the sealed folders', the judge outputs and Git's own,
    as bytes (read with no guard)."""
    return {path: path.read_bytes() for path in sorted(route.base.rglob("*"))
            if path.is_file() and not path.is_symlink() and not sealed_part(path, route)
            and ".git" not in path.relative_to(route.base).parts}  # fmt: skip


def test_the_route_runs_every_campaign_kind_and_summary_command_on_files(route) -> None:
    """Every campaign kind of the writer and every command of the summary script ran on the
    route's files, and the real guard let each confirmation block start only in its turn."""
    done = {(r.script, r.argv[0] if r.argv[0] != "record" and r.argv[0] != "fixation"
             else f"{r.argv[0]} {r.argv[1]}") for r in route.ran if r.code == 0}  # fmt: skip
    assert {kind for script, kind in done if script == "campaigns"} == {
        "engineering", "search-round0", "search-round1", "numerical", "repair", "validation",
        "confirmation", "prep"}  # fmt: skip
    assert {kind for script, kind in done if script == "summary"} == {
        "progress", "guard", "check", "versions", "fixation validation",
        "fixation confirmation", "record engineering", "record simulation", "record n-c1",
        "record budget-c1", "closure"}  # fmt: skip
    assert route.guards == [(10400, 0), (10401, 1), (10401, 0), (10402, 0), (10402, 1),
                            (10403, 0)]  # fmt: skip
    # the guard's state and versions check were the real ones, on the checkout's GPU code
    assert {code for _, code in route.readings} == {route.code["gpu_code_sha256"]}
    # the validation fixation the summary command wrote pins the spread and the control list
    fixation = json.loads((route.repo / VALIDATION_C1).read_text())
    assert fixation["spread"] == spread_pin(route.repo, fixation["code"]["commit"])
    assert [entry["path"] for entry in fixation["spread"]] == [
        "scripts/study3_spread.py", "src/zero_field_refusal/train.py"]  # fmt: skip
    controls = hashlib.sha256((route.repo / CONTROLS).read_bytes()).hexdigest()
    assert fixation["judge_controls"] == {"path": CONTROLS, "sha256": controls}
    # the sentinel is in every sealed file and every judge output of the world
    sealed = [path for path in route.base.rglob("*") if path.is_file() and sealed_part(path, route)]
    assert len(sealed) > 400
    assert len([path for path in sealed if path.parent.name == JUDGE]) > 250
    assert all(SENTINEL.encode() in path.read_bytes() for path in sealed)


def test_the_route_reads_no_label(route) -> None:
    """Nothing on the route touched a label reader, every entry point of which raised (they were
    replaced before the scripts were loaded)."""
    assert route.labels.calls == []
    assert {"zero_field_refusal.analysis.study2_labels.read_endpoint_labels",
            "zero_field_refusal.labeling.load_labels",
            "zero_field_refusal.run_judging.label_runs"} <= set(route.labels.names)  # fmt: skip


def test_the_route_opens_nothing_sealed_and_shows_no_sentinel(route) -> None:
    """No command opened a path under a ``responses/`` or ``observations/`` folder or a run's
    judge outputs, and the sentinel is in no output, error, refusal or written file."""
    assert route.attempts == []
    for ran in route.ran:
        assert SENTINEL not in ran.out + ran.err, ran.argv
    files = written(route)
    assert len(files) > 1000
    assert [path for path, data in files.items() if SENTINEL.encode() in data] == []


def test_the_route_opens_nothing_superseded_and_no_ledger(route) -> None:
    """Every run's judge outputs moved aside (``judge-superseded/``) and the labeling ledger
    (``study3/labeling/``) lay in the route's world, each holding the sentinel, while every
    command ran: none opened a path under them (the guard refuses both below the runs root),
    and no walk took the ledger's folder for a run."""
    superseded = list(route.root.glob(f"study3/*/{SUPERSEDED}/*/*/wildguard.jsonl"))
    ledger = route.root / "study3" / LEDGER / "journal.jsonl"
    assert len(superseded) > 250 and SENTINEL.encode() in ledger.read_bytes()
    assert route.attempts == []
    assert route.state()["problems"] == []


@contextlib.contextmanager
def linked(route):
    """A confirmation run's ``config.yaml`` and ``meta.json`` replaced by links into its
    ``responses/``, and a numerical (K-only) run's into its ``observations/``; and a launch log, a
    case-restart record (in a case folder of the qualification) and a supersession record added,
    each a link into the confirmation run's ``responses/``. Yields the two runs and the added
    records (what each is -> its path); everything is restored after."""
    regular = route.launched[str(route.blocks[10400])][0]
    k_only = route.numerical_runs[0]
    kept = {}
    for run_id, sealed in ((regular, Path("responses") / "harmful_evaluation.jsonl"),
                           (k_only, Path("observations") / "measurements.jsonl")):  # fmt: skip
        for name in ("config.yaml", "meta.json"):
            path = route.root / "study3" / run_id / name
            kept[path] = path.read_bytes()
            link(path, sealed)
    sealed_file = route.root / "study3" / regular / "responses" / "harmful_evaluation.jsonl"
    added = {
        "launch log": route.root / "launch" / "linked.jsonl",
        "case-restart record": route.case(1, f"adamw-{START['A']['u']:g}") / "restart-1.json",
        "supersession record": route.root / "qualification" / "c1" / "supersede-1.json",
    }
    for path in added.values():
        path.symlink_to(sealed_file)
    try:
        yield regular, k_only, added
    finally:
        for path in added.values():
            path.unlink()
        for path, data in kept.items():
            path.unlink()
            path.write_bytes(data)


KINDS = ("engineering", "search-round0", "search-round1", "numerical", "repair", "validation",
         "confirmation")  # fmt: skip


def test_runs_whose_config_and_meta_link_into_their_sealed_folders_are_named_problems(
    route, monkeypatch
) -> None:
    """The commands again, with a confirmation run's ``config.yaml`` and ``meta.json`` linked into
    its ``responses/``, a numerical run's into its ``observations/``, and a launch log, a
    case-restart record and a supersession record linked into a run's ``responses/``
    (``linked``): each run and each record is a problem that names it, and the step is blocked;
    every campaign kind, the guard and a new versions record refuse; the other commands refuse
    or write what reads no run (the streams driver); the readers name the same problems. Nothing
    sealed is opened, no label reader is called, and the sentinel is in no output, refusal,
    problem or written file."""
    root, repo = route.root, route.repo
    third = "docs/study3-versions-3.json"
    with linked(route) as (regular, k_only, added):
        guard = SealedGuard(monkeypatch, runs_root=route.root)
        first = len(route.ran)
        why = f"its config.yaml and meta.json resolve under {FOLDER}"
        state = json.loads(route.ok("summary", "progress", "--runs-root", root).out)
        assert state["step"]["kind"] == "blocked"
        for run_id in (regular, k_only):
            assert f"{run_id}: {why}: never opened" in state["problems"]
        assert f"launch/linked.jsonl: a launch log that {SEALED_WHY}" in state["problems"]
        restart = f"{added['case-restart record']}: a case-restart record that {SEALED_WHY}"
        cycle = f"qualification/c1/supersede-1.json: a supersession record that {SEALED_WHY}"
        assert restart in state["problems"] and cycle in state["problems"]
        assert route.guarded(10403, route.blocks[10403]).code != 0
        check = route.refused("summary", "check", "--runs-root", root, "--campaign",
                              route.blocks[10403])  # fmt: skip
        assert f"launch/linked.jsonl: a launch log that {SEALED_WHY}" in check.err
        for kind in KINDS:
            error = route.refused("campaigns", kind, "--runs-root", root, "--candidate", 1,
                                  "--theta0", route.runs.theta0, "--out",
                                  route.out / f"linked-{kind}").err  # fmt: skip
            assert "the step is blocked" in error and regular in error, (kind, error)
        route.ok("campaigns", "prep", "--seeds", 10101, "--commit", route.code["commit"], "--out",
                 route.out / "linked-prep.sh")  # fmt: skip
        for step in ("validation", "confirmation"):
            route.refused("summary", "fixation", step, "--runs-root", root, "--candidate", 1,
                          *(SUPPLEMENT_SEEDS if step == "confirmation" else ()))  # fmt: skip
        versions = route.refused("summary", "versions", "--out", third, "--reason",
                                 "linked_runs", "--fixation", CONFIRMATION_C1, "--runs-root",
                                 root, "--supersedes", SECOND, "--supersedes-reason",
                                 "linked_runs")  # fmt: skip
        assert f"launch/linked.jsonl: a launch log that {SEALED_WHY}" in versions.err
        route.refused("summary", "record", "engineering", "--runs-root", root, "--from",
                      route.inputs / "engineering.json")  # fmt: skip
        route.refused("summary", "closure", "--runs-root", root, "--candidate", 1, "--arm", "F")
        # the readers, called directly
        found = {}
        for name, read in (
            ("read_endpoints", lambda: read_endpoints(root, theta0=route.runs.theta0,
                                                      prescribed=route.prescribed)),
            ("versions_problems", lambda: sv.versions_problems(
                repo / SECOND, runs_root=root, repo=repo, step="confirmation", candidate=1)),
            ("uncounted_runs", lambda: sv.uncounted_runs(read_endpoints(
                root, theta0=route.runs.theta0, prescribed=route.prescribed)[0], repo=repo)),
            ("read_validation_fixation", lambda: sv.read_validation_fixation(
                1, runs_root=root, repo=repo)),
            ("read_confirmation_fixation", lambda: sv.read_confirmation_fixation(
                1, runs_root=root, repo=repo)),
            ("decisions", lambda: decisions(root, 1, repo=repo)),
            ("read_holds", lambda: read_holds(root)),
            ("journal_lines", lambda: journal_lines(root)),
            ("qualification_logs", lambda: sv.qualification_logs(root, 1)),
        ):  # fmt: skip
            try:
                found[name] = repr(read())
            except Exception as exc:  # noqa: BLE001  (its message is checked too)
                found[name] = f"raised {type(exc).__name__}: {exc}"
        _, problems = read_endpoints(root, theta0=route.runs.theta0, prescribed=route.prescribed)
        for run_id in (regular, k_only):
            assert f"{run_id}: {why}: never opened" in problems
        assert guard.attempts == []
        ran = route.ran[first:]
        files = written(route)
    assert not (repo / third).exists()
    assert route.labels.calls == []
    for done in ran:
        assert SENTINEL not in done.out + done.err, done.argv
    for name, text in found.items():
        assert SENTINEL not in text and "AssertionError" not in text, name
    assert [path for path, data in files.items() if SENTINEL.encode() in data] == []


def real_reading(route) -> dict:
    """Candidate 1's qualification as the real ``read_qualification`` reads the route's files
    (they hold no qualification of package 1's: its rates are not qualified, and its problems say
    so); its launcher runs located and their identity files read."""
    return REAL_READING(route.root, 1, expected_code=route.code["gpu_code_sha256"],
                        settings=load_config("experiments/qualification"))  # fmt: skip


def test_the_qualification_reader_opens_nothing_sealed_on_the_routes_files(route, monkeypatch):
    """The route replaces the qualification's reading; the real reader, on the same files, opens
    nothing sealed while it locates the launcher runs and reads their identity files, and calls no
    label reader."""
    guard = SealedGuard(monkeypatch, runs_root=route.root)
    reading = real_reading(route)
    assert guard.attempts == []
    assert any(identity is not None for rate in reading["rates"].values()
               for identity in rate["launchers"].values())  # fmt: skip
    assert SENTINEL not in repr(reading)
    assert route.labels.calls == []


def test_the_qualification_reader_opens_no_record_linked_into_its_sealed_folders(
    route, monkeypatch
) -> None:
    """With the links of ``linked``, the real qualification reader opens none of them: each run
    and each record is a named problem; no launcher run is located through the numerical run
    whose files are linked, and ``launcher_identity`` gives none for it, unopened; the other
    rates' launcher runs are located as before; no label reader is called."""
    with linked(route) as (regular, k_only, added):
        guard = SealedGuard(monkeypatch, runs_root=route.root)
        reading = real_reading(route)
        identity = qualification_records.launcher_identity(route.root / "study3" / k_only)
        found = list(guard.attempts)
        guard.armed = False
    assert found == [] and identity is None
    why = f"its config.yaml and meta.json resolve under {FOLDER}"
    for run_id in (regular, k_only):
        assert (f"{route.root / 'study3' / run_id}: {why}: never opened; no launcher run is "
                "located there") in reading["problems"]  # fmt: skip
    for what, path in added.items():
        assert f"{path}: a {what} that {SEALED_WHY}" in reading["problems"], what
    assert any(p.startswith(f"{k_only} (") and "a start of candidate 1's qualification refused: "
               f"its config.yaml {SEALED_WHY}" in p for p in reading["problems"])  # fmt: skip
    seed = SEEDS["numerical"][1][0]
    lower = qualification_records.rate_of(reading, "A", START["A"]["l"])
    assert lower["launchers"].get(seed) is None  # its seed's run: refused, located nowhere
    assert qualification_records.rate_of(reading, "A", START["A"]["u"])["launchers"][seed]
    assert SENTINEL not in repr(reading)
    assert route.labels.calls == []


# ------------------------------------------------------------------ no label

LOADED = """\
import importlib, importlib.util, inspect, json, sys
from pathlib import Path

labels, modules, scripts = sys.argv[1:4]
labels, modules, scripts = json.loads(labels), json.loads(modules), json.loads(scripts)
calls, readers = [], {}
if "--patched" in sys.argv:  # the label readers' entry points raise from the start
    for name in labels:
        module = importlib.import_module(name)
        for attr, value in list(vars(module).items()):
            if attr.startswith("_") or not inspect.isfunction(value) or value.__module__ != name:
                continue

            def raiser(*args, _name=f"{name}.{attr}", **kwargs):
                calls.append(_name)
                raise AssertionError(_name)

            readers[id(value)], readers[id(raiser)] = (value, attr), (raiser, attr)
            setattr(module, attr, raiser)
loaded = {name: importlib.import_module(name) for name in modules}
for path in scripts:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded[path] = module
bound = sorted(f"{where}.{attr}" for where, module in loaded.items()
               for attr, value in vars(module).items()
               if id(value) in readers and readers[id(value)][0] is value)
print(json.dumps({"calls": calls, "bound": bound, "readers": len(readers) // 2,
                  "loaded": [name for name in labels if name in sys.modules]}))
"""


def test_the_package_loads_and_binds_no_label_reader() -> None:
    """Each module of the package and both scripts, imported in a fresh process: with the label
    readers' entry points raising from the start, the imports call none and bind none; without
    them, the imports load no label reader at all."""
    modules = [f"zero_field_refusal.analysis.{path.stem}" for path in PACKAGE]
    argv = [sys.executable, "-c", LOADED, json.dumps(LABEL_MODULES), json.dumps(modules),
            json.dumps([str(path) for path in SCRIPTS])]  # fmt: skip
    patched = json.loads(subprocess.run([*argv, "--patched"], capture_output=True, text=True,
                                        check=True, cwd=REPO_ROOT).stdout)  # fmt: skip
    assert patched["readers"] > 10 and patched["calls"] == [] and patched["bound"] == []
    plain = json.loads(subprocess.run(argv, capture_output=True, text=True, check=True,
                                      cwd=REPO_ROOT).stdout)  # fmt: skip
    assert plain == {"calls": [], "bound": [], "readers": 0, "loaded": []}


# ------------------------------------------------------------------ the label readers' fence


class Labeled:
    """``test_study3_label_cli.World``'s labeling (its runs, controls, stand-in fixation and fake
    judge server) with a validation cohort of two blocks, ``r1``–``r8`` (A and F at both rates
    per block, each arm's weight 1/2), at candidate 1's choice step; and the spread ready to run
    on it (``test_study3_spread``'s pieces): the stand-in spread script pinned by the fixation,
    the engineering and simulation records, the checkout in Git, the blocks' endpoints given.
    The validation gate's minimum is lowered to two supported blocks in this world alone: its
    tests are of the opens, not of the gate."""

    RUNS = tuple(f"r{i}" for i in range(1, 9))
    SEEDS = (10300, 10301)

    def __init__(self, tmp_path: Path, monkeypatch, capfd) -> None:
        # they import this module: here only
        import test_study3_spread as spread_test
        from test_study3_label_cli import World as LabelWorld
        from test_study3_labels import study3_run

        from zero_field_refusal.analysis import study3_decisions, study3_label_cli, study3_labels

        for readers in LabelReaders.ACTIVE:  # the route's replacement, lifted for this world
            readers.lift(monkeypatch)
        self.labels = LabelWorld(tmp_path, monkeypatch, capfd)
        self.root, self.repo = self.labels.root, self.labels.repo
        for name in self.RUNS[3:]:
            study3_run(self.root, name, harmful=(0,) if int(name[1:]) % 2 else (0, 1, 3),
                       unparseable=())  # fmt: skip
        blocks = [spread_test.block(seed, self.RUNS[4 * i:4 * i + 4])
                  for i, seed in enumerate(self.SEEDS)]  # fmt: skip
        self.endpoints = [e for found in blocks for e in found]
        cohort = {"seeds": list(self.SEEDS), "run_ids": list(self.RUNS)}
        self.state = {"stage": "choice", "candidate": 1, "problems": [], "prescribed": {},
                      "step": {"kind": "decision", "record": "n-c1",
                               "fields": {"candidate": 1, "rates": spread_test.RATES,
                                          "active": ["A-F"], "cohort": cohort}}}  # fmt: skip
        monkeypatch.setattr(study3_labels, "_state", lambda runs_root, repo_root: self.state)
        monkeypatch.setattr(study3_label_cli, "_bound", self.bound)
        monkeypatch.setitem(study3_decisions.SUPPORTED_AT, 28, len(self.SEEDS))
        records = spread_test.Records(self.root, self.repo, monkeypatch)
        records.stand_in()
        records.write("engineering", records.engineering())
        records.write("simulation", records.simulation())
        self.labels.fixation["spread"] = spread_pin(self.repo)
        named = self.repo / sv.fixation_name("validation", 1)
        named.write_text(json.dumps(self.labels.fixation) + "\n")
        git(self.repo, "init", "-q")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "the stand-ins")
        self.script = spread_test.spread_module()
        monkeypatch.setattr(self.script, "REPO", self.repo)
        monkeypatch.setattr(self.script, "read_endpoints",
                            lambda runs_root, *, theta0, prescribed: (list(self.endpoints),
                                                                      []))  # fmt: skip

    def bound(self, runs_root, state, step, candidate, run_ids, theta0=None) -> None:
        from zero_field_refusal.analysis.study3_label_cli import BatchRefused

        if (step, candidate) != ("validation", 1) or set(run_ids) - set(self.RUNS):
            raise BatchRefused(f"{run_ids} are not candidate 1's validation runs")

    def batch(self, run_ids, **judge):
        return self.labels.batch(run_ids, **judge)

    def harm(self, name: str):
        return self.labels.harm(name)

    def spread(self) -> None:
        """The cohort's runs labeled (those not labeled yet, in a batch), then the spread."""
        self.labels.batch(list(self.RUNS))
        self.script.main(["--runs-root", str(self.root), "--candidate", "1"])

    def n_record(self) -> dict:
        from zero_field_refusal.analysis.study3_decisions import read_decision

        record, problems = read_decision(self.root, "n-c1", repo=self.repo)
        assert problems == [] and record is not None, problems
        return record


@pytest.fixture
def labeled_world(tmp_path, monkeypatch, capfd) -> Labeled:
    return Labeled(tmp_path, monkeypatch, capfd)


class Points:
    """``test_study3_label_cli.World``'s runs ``r1``–``r3``, their responses holding the sentinel,
    labeled and covered in a batch at candidate 1's choice step; ``to(state)`` then puts the study
    at another state: ``no_fixation`` (the validation under way, no validation fixation),
    ``validating`` (the same, the fixation written), ``choice`` (the spread point open) or
    ``n_written`` (the confirmation under way). Every refusal and output is kept
    (``everything_shown``)."""

    STATES = {
        "no_fixation": {"stage": "validation", "step": {"kind": "campaign",
                                                        "campaign": "validation"}},
        "validating": {"stage": "validation", "step": {"kind": "campaign",
                                                       "campaign": "validation"}},
        "choice": {"stage": "choice", "step": {"kind": "decision", "record": "n-c1",
                                               "fields": {"cohort": {"run_ids": ["r1", "r2",
                                                                                 "r3"]}}}},
        "n_written": {"stage": "confirmation", "step": {"kind": "campaign",
                                                        "campaign": "confirmation"}},
    }  # fmt: skip

    def __init__(self, tmp_path: Path, monkeypatch, capfd) -> None:
        import test_study3_spread as spread_test  # they import this module: here only
        from test_study3_label_cli import World as LabelWorld

        from zero_field_refusal.analysis import study3_labels

        self.monkeypatch, self.capfd = monkeypatch, capfd
        for readers in LabelReaders.ACTIVE:  # the route's replacement, lifted for this world
            readers.lift(monkeypatch)
        self.labels = LabelWorld(tmp_path, monkeypatch, capfd)
        self.root, self.repo = self.labels.root, self.labels.repo
        for name in ("r1", "r2", "r3"):
            path = self.labels.run(name) / "responses" / "harmful_evaluation.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text("".join(json.dumps({**row, "text": f"{row['text']} {SENTINEL}"})
                                    + "\n" for row in rows))  # fmt: skip
        assert self.labels.batch(["r1", "r2", "r3"]).outcome == "accepted"
        self.cohort_run = self.labels.run("r1")
        self.script = spread_test.spread_module()
        monkeypatch.setattr(self.script, "REPO", self.repo)
        self.shown: list[str] = [self.labels.out]
        self.state = "choice"
        monkeypatch.setattr(study3_labels, "_state", self.state_of)

    def state_of(self, runs_root, repo_root) -> dict:
        return {"candidate": 1, "problems": [], "prescribed": {}, **self.STATES[self.state]}

    def to(self, state: str) -> None:
        self.state = state
        if state == "no_fixation":
            self.monkeypatch.setattr(self.labels, "fixation", None)
            (self.repo / sv.fixation_name("validation", 1)).unlink()

    def _read(self, how: str, run: Path, point: str):
        from zero_field_refusal.analysis import study3_labels

        read = getattr(study3_labels, how)
        try:
            return read(run, point=point, candidate=1, runs_root=self.root, repo_root=self.repo)
        except Exception as exc:
            self.shown.append(str(exc))
            raise

    def harm(self, run: Path, *, point: str):
        return self._read("run_harm", run, point)

    def items(self, run: Path, *, point: str):
        return self._read("run_items", run, point)

    def refused(self, command: str, step: str | None = None) -> bool:
        """Whether the labeling command (a batch of ``r1`` at ``step``) or the spread refuses."""
        from zero_field_refusal.analysis.study3_label_cli import BatchRefused

        try:
            if command == "label":
                self.labels.batch(["r1"], step=step)
            else:
                self.script.main(["--runs-root", str(self.root), "--candidate", "1"])
        except (BatchRefused, SystemExit) as exc:
            self.shown.append(str(exc))
            return True
        finally:
            self.shown.append(self.labels.out + "".join(self.capfd.readouterr()))
        return False

    def everything_shown(self) -> str:
        return "\n".join(self.shown)


@pytest.fixture
def points_world(tmp_path, monkeypatch, capfd) -> Points:
    return Points(tmp_path, monkeypatch, capfd)


def test_the_label_readers_import_no_gpu_module_and_load_no_script():
    """Each label reader, read as code (``test_code_identity.gpu_uses``): the two modules (their
    relative imports resolved) and the spread's script import no GPU module and no torch
    directly, use no CUDA, and load no code by its name or path at run time."""
    for path in LABEL_READERS:
        package = None if path.parent.name == "scripts" else "zero_field_refusal.analysis"
        assert gpu_uses(path, package) == set(), path.name


def test_the_named_sealed_openers_are_functions_of_the_label_readers():
    """Every name of ``SEALED_OPENERS`` is a function (or a method of the batch) of its module: no
    name stands for nothing."""
    from zero_field_refusal.analysis import study3_label_cli, study3_labels

    modules = {"study3_labels": study3_labels, "study3_label_cli": study3_label_cli}
    for name in SEALED_OPENERS | LEDGER_OPENERS:
        stem, function = name.split(".")
        module = modules[stem]
        found = getattr(module, function, None) or getattr(study3_label_cli._Batch, function, None)
        assert callable(found), name


def test_sealed_paths_open_only_through_named_functions(labeled_world, monkeypatch):
    guard = NamedOpens(monkeypatch, runs_root=labeled_world.root)
    labeled_world.batch(["r1"])
    labeled_world.harm("r1")
    labeled_world.spread()
    assert guard.attempts == [] and any("/judge/" in p for p in guard.opened)
    assert any("/labeling/" in p for p in guard.allowed)
    assert any("/responses/" in p for p in guard.allowed)
    assert labeled_world.n_record()["N"] >= 60


def test_a_relabel_and_its_completed_move_open_sealed_paths_only_through_named_functions(
        labeled_world, monkeypatch):  # fmt: skip
    """A control change, a crash between its ``superseded`` line and the move, and the resume
    that completes the move and relabels: every sealed open inside a named opener."""
    guard = NamedOpens(monkeypatch, runs_root=labeled_world.root)
    labeled_world.labels.crash_before_rename()
    with pytest.raises(SystemExit):
        labeled_world.batch(["r1"], flips={(1, "control-3")})
    assert labeled_world.batch(None).outcome == "accepted"
    assert labeled_world.harm("r1").receipt_sha256
    assert guard.attempts == [] and any("/judge-superseded/" in p for p in guard.opened)


@pytest.mark.parametrize("state", ["no_fixation", "validating", "choice", "n_written"])
def test_each_reader_refuses_before_its_point_opens(points_world, state, monkeypatch):
    """At each state of the study, every label reader refuses a closed point: ``run_harm`` and
    ``run_items`` at ``final`` (closed until the freeze gate is wired) and, outside candidate 1's
    choice step, at ``spread``; the labeling command for the confirmation always, and for the
    validation and the spread outside the choice step. While the spread point is closed, nothing
    of a run's sealed folders is opened (its responses, judge outputs or superseded copies) and
    the labeling ledger is opened only by its own openers (``LEDGER_OPENERS``: the labeling command
    reads its ledger under its lock before it checks the point); no sentinel is shown."""
    from zero_field_refusal.analysis.study3_labels import POINTS, SealedPoint, ledger_dir

    points_world.to(state)
    spread_open = state == "choice"
    if spread_open:  # the spread's reads open the runs' labels, through the named openers alone
        guard = NamedOpens(monkeypatch, runs_root=points_world.root)
    else:
        guard = NamedOpens(monkeypatch, runs_root=points_world.root, openers=LEDGER_OPENERS,
                           under=ledger_dir(points_world.root))  # fmt: skip
    reads = (points_world.harm, points_world.items)  # run_harm, run_items
    for read, point in itertools.product(reads, POINTS):
        if point == "final" or not spread_open:  # final: closed until the freeze gate
            with pytest.raises(SealedPoint):
                read(points_world.cohort_run, point=point)
        else:
            assert read(points_world.cohort_run, point=point)
    assert points_world.refused("label", "confirmation")
    if not spread_open:
        assert points_world.refused("label", "validation") and points_world.refused("spread")
        assert guard.attempts == [] and SENTINEL not in points_world.everything_shown()
        under_runs = [p for p in guard.opened
                      if Path(p).is_relative_to(Path(os.path.realpath(points_world.root))
                                                / "study3" / "r1")]  # fmt: skip
        assert under_runs == []
    assert guard.attempts == []


def test_the_route_ends_with_every_point_closed(route, monkeypatch):
    """On the route's files and its real ``progress`` (the n record written, the confirmation
    under way, no freeze), every point is closed: ``run_harm`` raises ``SealedPoint`` at both and
    opens nothing sealed. Its entry points are the real ones for this test (the route replaced
    them with raisers)."""
    from zero_field_refusal.analysis import study3_labels

    route.labels.lift(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=route.root)  # real progress: n written, no freeze
    run = route.root / "study3" / route.validation_runs[0]
    for point in study3_labels.POINTS:
        with pytest.raises(study3_labels.SealedPoint):
            study3_labels.run_harm(run, point=point, candidate=1, runs_root=route.root,
                                   repo_root=route.repo)  # fmt: skip
    assert guard.attempts == []
