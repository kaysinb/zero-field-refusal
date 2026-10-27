"""Study 3's analysis package as a whole (the modules ``analysis/study3_*.py``,
``analysis/qualification_records.py`` and ``analysis/campaign_prep.py``, and the scripts
``scripts/study3_campaigns.py`` and ``scripts/study3_summary.py``): its boundaries.

- A run's record files are opened only outside its sealed folders (``study3_records``): a record
  file, or a run's folder, that resolves through a link under a ``responses/`` or
  ``observations/`` folder is refused unopened, a named problem; and a journal start's command
  composes only a study-3 config.

Every sealed file of these worlds holds the sentinel ``SENTINEL``, and every read of a sealed path
fails under ``SealedGuard``."""

import builtins
import io
import os
import shutil
from pathlib import Path

import pytest
import test_study3_records
from test_study3_records import World as RunWorld
from test_study3_records import by_id, config_of

from zero_field_refusal.analysis import study3_records
from zero_field_refusal.analysis.study3_records import EndpointReader, read_theta0
from zero_field_refusal.config import config_hash
from zero_field_refusal.paths import CONFIG_ROOT

SENTINEL = "PKG2-SENTINEL-qX7v-Rk3W-9zJd"  # in every responses/ and observations/ file here
SEALED = ("responses", "observations")
RECORD_FILES = ("config.yaml", "meta.json", "terminal.json", "summary.json", "evaluations.jsonl",
                "metrics.jsonl", "monitor.jsonl")  # fmt: skip


class SealedGuard:
    """``builtins.open``, ``io.open``, ``os.open``, ``Path.open``, ``Path.read_text`` and
    ``Path.read_bytes`` wrapped: while armed, opening a path that resolves (links followed) under
    a ``responses/`` or ``observations/`` folder, in any mode, fails, and the path is kept in
    ``attempts`` (a caller that swallows the failure is still caught). ``off()`` disarms it while a
    test writes a run as the tracker does."""

    def __init__(self, monkeypatch) -> None:
        self.attempts: list[str] = []
        self.armed = True
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
        if any(part in SEALED for part in Path(os.path.realpath(text)).parts):
            self.attempts.append(text)
            raise AssertionError(f"a sealed path was opened: {text}")

    class _Off:
        def __init__(self, guard) -> None:
            self.guard = guard

        def __enter__(self) -> None:
            self.guard.armed = False

        def __exit__(self, *exc) -> None:
            self.guard.armed = True

    def off(self) -> "SealedGuard._Off":
        return self._Off(self)


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
    why = f"its {name} resolves under a responses or observations folder: never opened"
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
    assert (f"{k_only}: its config.yaml and meta.json resolve under a responses or observations "
            "folder: never opened") in problems  # fmt: skip
    assert (f"{regular}: its terminal.json resolves under a responses or observations folder: "
            "never opened") in problems  # fmt: skip
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
    assert (f"{moved}: its run folder resolves under a responses or observations folder: nothing "
            "of it is opened") in problems  # fmt: skip
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
    assert (
        refusal == "its config.yaml resolves under a responses or observations folder: never opened"
    )
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
        with pytest.raises(ValueError, match="responses or observations folder: never opened"):
            read_theta0(named)
    link(runs.theta0 / "summary.json", Path("responses") / "harmful_evaluation.jsonl")
    with pytest.raises(ValueError, match="responses or observations folder: never opened") as found:
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
