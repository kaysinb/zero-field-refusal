"""Study 3's campaign writer (scripts/study3_campaigns.py) and the prep driver it shares with
study 2 (analysis/campaign_prep.py).

The writer runs on synthetic RUNS_DIR trees built with the reader tests' helpers: the decision
records with their evidence (in a checkout, the writer's), the launch journal and the run
directories, each regular run holding a harmful-response file and an observations file with a
sentinel text (never opened). The state is ``study3_progress.progress``'s over those files, the
qualification's reading given; or, where a test names a route the files would take long to build,
``advance``'s over the progress tests' records. Every campaign is read back with the launcher's
``load_campaign``, and its runs, started, written and ended as the launcher and the tracker do,
are bound by ``study3_records`` without a problem."""

import builtins
import copy
import functools
import hashlib
import importlib.util
import io
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from test_study3_decisions import CODE as DECISION_CODE
from test_study3_decisions import World as DecisionWorld
from test_study3_decisions import held as decision_hold
from test_study3_progress import Study
from test_study3_records import SEALED, SENTINEL, write_run
from test_study3_records import World as RunWorld

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import campaign_prep, qualification_records
from zero_field_refusal.analysis.study3_decisions import decisions
from zero_field_refusal.analysis.study3_progress import progress
from zero_field_refusal.analysis.study3_records import SEEDS, read_endpoints
from zero_field_refusal.analysis.study3_search import CANDIDATES, adjusted_rate
from zero_field_refusal.config import config_hash
from zero_field_refusal.launch import campaign_header, load_campaign
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import check_run_config, load_run_config

COMMIT = "c" * 40
STUDY2 = REPO_ROOT / "scripts" / "study2_short_campaigns.py"


def script_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joined(text: str) -> list[str]:
    """A driver's commands, continued lines joined and indentation dropped."""
    return [line.strip() for line in re.sub(r" *\\\n *", " ", text).splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_reads(monkeypatch) -> list[str]:
    """From now on a read of any file under a ``responses/`` or ``observations/`` folder fails
    (``open``, ``io.open`` and ``Path.open`` for reading, ``Path.read_text`` and ``read_bytes``),
    while the tests still write them as the tracker does (``test_study3_records.seal``, for
    reads); the list holds every such attempt."""
    attempts: list[str] = []

    def guard(path, mode="r") -> None:
        if isinstance(path, int) or any(flag in mode for flag in "wax"):
            return
        if any(part in SEALED for part in Path(os.fsdecode(path)).parts):
            attempts.append(str(path))
            raise AssertionError(f"a sealed file was opened: {path}")

    real_open = builtins.open

    def open_(file, mode="r", *args, **kwargs):
        guard(file, mode)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_)
    monkeypatch.setattr(io, "open", open_)
    real_path_open = Path.open

    def path_open(self, mode="r", *args, **kwargs):
        guard(self, mode)
        return real_path_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    for name in ("read_text", "read_bytes"):
        real = getattr(Path, name)

        def reader(self, *args, _real=real, **kwargs):
            guard(self)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(Path, name, reader)
    return attempts


# ------------------------------------------------------------------ the shared prep


def test_study2s_prep_is_the_shared_one() -> None:
    """Study 2's script runs the moved prep: its driver, its arguments and the names its tests
    and its other commands read."""
    module = script_module(STUDY2, "study2_campaigns")
    assert module.prep_driver is campaign_prep.prep_driver
    assert module.prep is campaign_prep.prep
    assert module.comment is campaign_prep.comment and module.revision is campaign_prep.revision
    for name in ("INVENTORIES", "V3_RECORD", "CURVATURE_PANEL", "CLEAN", "EXISTS", "STAGE_A",
                 "TEACHER", "PASSED", "STAGE0_CONFIG", "STAGE0_OVERRIDES"):  # fmt: skip
        assert getattr(module, name) == getattr(campaign_prep, name), name


def test_the_streams_driver_generates_each_seeds_streams_under_its_own_config(tmp_path) -> None:
    configs = {10100: "experiments/study3_k_only study3.role=search",
               10300: "experiments/study3 study3.role=validation"}  # fmt: skip
    text = campaign_prep.streams_driver([10100, 10300], COMMIT, configs=configs, batch=20,
                                        steps=25, writer="scripts/study3_campaigns.py prep",
                                        title="Study 3", written_at=COMMIT)  # fmt: skip
    out = tmp_path / "prep.sh"
    out.write_text(text)
    subprocess.run(["bash", "-n", str(out)], check=True)
    assert all(len(line) <= 100 for line in text.splitlines() if line.startswith("#"))
    header = " ".join(line[2:] for line in text.splitlines() if line.startswith("# "))
    assert f"Written by scripts/study3_campaigns.py prep at {COMMIT}; do not edit." in header
    assert "Study 3, from commit" in header and "seeds 10100 10300 at batch 20" in header
    commands = joined(text)
    run = 'project-run "$SRC" python scripts/'
    assert f'test "$(git rev-parse HEAD)" = {COMMIT}' in commands
    assert f'project-run "$SRC" python -c "{campaign_prep.CLEAN}"' in commands
    for seed, config in configs.items():
        assert (f"step {run}generate_teacher.py {config} --stream training --seeds {seed} "
                "--responses 500") in commands  # fmt: skip
        assert (f"kept {run}generate_teacher.py {config} optimizer=adam_frozen train.batch_size=20 "
                f"--stream preconditioner --seeds {seed}") in commands  # fmt: skip
    assert "stage0" not in text and "check-streams" not in text and "curvature" not in text
    # the commands compose: the configs as the generator loads them (its overrides after it)
    for config in configs.values():
        name, *overrides = config.split()
        assert load_run_config(name, overrides).train.batch_size == 20
        pre = load_run_config(name, [*overrides, "optimizer=adam_frozen", "train.batch_size=20"])
        assert pre.optimizer.preconditioner.responses == 1000


# ------------------------------------------------------------------ the writer's world


SCRIPT = REPO_ROOT / "scripts" / "study3_campaigns.py"
EXPECTED = DECISION_CODE["gpu_code_sha256"]  # the GPU code every decision record here carries
OPT = {"A": "adamw", "F": "adam_frozen", "P": "adamw_prewarmed"}
ARM_OF = {name: arm for arm, name in OPT.items()}
START = CANDIDATES[1]["start"]
LABELS = ("l", "u")
VERSIONS = "docs/study3-versions.json"


@functools.lru_cache
def _cell_config(config: str, overrides: tuple[str, ...]):
    return load_run_config(config, overrides)


def run_config(config: str, overrides=()):
    """A campaign run's config as the launcher loads it (``load_run_config``), composed once per
    cell and its seed set: the same config and hash (``test_one_composition_per_cell_is_the_
    launchers``), so that the 168 runs of a validation load in seconds."""
    seeds = [o for o in overrides if o.startswith("seed=")]
    rest = tuple(o for o in overrides if not o.startswith("seed="))
    cfg = copy.deepcopy(_cell_config(config, rest))
    if seeds:
        cfg.seed = int(seeds[-1].split("=", 1)[1])
    check_run_config(cfg)
    return cfg


def target(arm: str, rate: float, seed: int, candidate: int) -> dict:
    """K at the target of the rate's label (the nearer starting rate's)."""
    start = CANDIDATES[candidate]["start"][arm]
    label = min(LABELS, key=lambda name: abs(math.log(rate / start[name])))
    return {"k": CANDIDATES[candidate]["targets"][label]}


class World:
    """A RUNS_DIR with its decision records and their evidence (in a checkout, the writer's), the
    runs as the launcher and the tracker write them, and the writer loaded as a module. The
    qualification's reading qualifies the rates in ``qualified``; every other reader is real."""

    def __init__(self, tmp_path: Path, monkeypatch, capsys) -> None:
        self.records = DecisionWorld(tmp_path)
        shutil.rmtree(self.records.run)  # the records tests' own run: none here without a start
        self.root, self.repo = self.records.runs, self.records.repo
        self.runs = RunWorld(self.root)
        (self.root / "launch").mkdir()
        (self.root / "launch" / "setup.jsonl").write_text("")  # a journal (decision records)
        self.capsys, self.monkeypatch = capsys, monkeypatch
        self.script = script_module(SCRIPT, "study3_campaigns")
        monkeypatch.setattr(self.script, "REPO", self.repo)
        stamps = itertools.count(1)
        monkeypatch.setattr(self.script, "stamp", lambda: f"20261027T{next(stamps):06d}Z")
        monkeypatch.setattr(
            code_identity,
            "gpu_code_identity",
            lambda *a, **k: {"commit": "a" * 40, "files": 1, "sha256": EXPECTED},
        )
        self.qualified: dict[int, set[tuple[str, float]]] = {1: set(), 2: set()}
        monkeypatch.setattr(qualification_records, "read_qualification", self.reading)
        self.out = tmp_path / "campaigns"
        self.out.mkdir()
        self.names = itertools.count()
        self.run_ids = itertools.count()
        self.checked: list[tuple] = []  # the versions checks the writer made

    # -- the writer

    def argv(self, kind, *args, candidate=1, out=None) -> tuple[list[str], Path]:
        out = out or self.out / (
            f"{kind}-{next(self.names)}" + ("" if kind == "confirmation" else ".yaml")
        )
        return [
            kind,
            "--runs-root",
            str(self.root),
            "--candidate",
            str(candidate),
            "--theta0",
            str(self.runs.theta0),
            "--out",
            str(out),
            *args,
        ], out

    def write(self, kind, *args, **kw) -> Path:
        """The campaign of ``kind`` written; its path (a folder for the confirmation)."""
        argv, out = self.argv(kind, *args, **kw)
        self.script.main(argv)
        return out

    def refused(self, kind, *args, **kw) -> str:
        """The writer refuses: non-zero, nothing written; its message."""
        argv, out = self.argv(kind, *args, **kw)
        self.capsys.readouterr()
        with pytest.raises(SystemExit) as found:
            self.script.main(argv)
        assert found.value.code not in (0, None)
        assert not out.exists()
        error = self.capsys.readouterr().err
        return error + (found.value.code if isinstance(found.value.code, str) else "")

    def versions(self, problems=()) -> None:
        """A versions record in the checkout; its check replaced, giving ``problems``."""
        path = self.repo / VERSIONS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"what": "versions"}\n')

        def check(record, **kw):
            self.checked.append((Path(record), kw))
            return list(problems)

        self.monkeypatch.setattr(self.script, "versions_problems", check)

    def core(self, study: Study) -> None:
        """The state is ``advance``'s over the progress tests' records."""
        self.monkeypatch.setattr(self.script, "progress", lambda *a, **k: study.advance())

    # -- the launcher, the tracker and the readers

    def launched(self, path: Path, values=target, end="done") -> list[str]:
        """The campaign's runs started, written and ended as the launcher and the tracker do it
        (``launch.load_campaign``, ``campaign_header``), each with ``values(arm, rate, seed,
        candidate)``, ``write_run``'s keywords (K at its target by default): their run ids."""
        name, runs = load_campaign(path)
        header = campaign_header(path)
        ids = []
        for run in runs:
            cfg = run_config(run.config, run.overrides)
            arm, rate = ARM_OF[str(cfg.optimizer.name)], float(cfg.optimizer.lr)
            run_id = (f"20261027T{next(self.run_ids):06d}Z-{run.cell[:28]}-s{run.seed}-"
                      f"{run.config_hash[:8]}-a1")  # fmt: skip
            self.runs.log(name, {"event": "start", "run": run.label, "attempt": 1,
                                 "run_id": run_id, "experiment": run.experiment,
                                 "config_hash": run.config_hash, "command": run.command()[1:],
                                 "header": header, "max_attempts": 1})  # fmt: skip
            write_run(self.root / run.experiment / run_id, OmegaConf.to_yaml(cfg, resolve=True),
                      run.config_hash, {"name": name, "label": run.label, "attempt": 1,
                                        "header": header},
                      k_only=OmegaConf.select(cfg, "eval.k_only", default=False) is True,
                      **values(arm, rate, run.seed, header["candidate"]))  # fmt: skip
            self.runs.end(name, run.label, 1, run_id, end)
            ids.append(run_id)
        return ids

    def reading(self, root, candidate, *, expected_code, settings) -> dict:
        out = {"rates": {}, "states": {}, "cases": [], "voided": {}, "supersessions": [],
               "held": [], "problems": []}  # fmt: skip
        for arm, lr in self.qualified[candidate]:
            key = f"{OPT[arm]}-{lr:g}"
            out["rates"][key] = {"key": key, "arm": arm, "optimizer": OPT[arm], "lr": lr,
                                 "state": "qualified", "runs": [], "held": [],
                                 "verdict": {"state": "qualified", "control": None,
                                             "not_counted": []}}  # fmt: skip
            out["states"][key] = "qualified"
        return out

    def state(self) -> dict:
        return progress(self.root, repo=self.repo, expected_code=EXPECTED,
                        theta0=self.runs.theta0, settings={})  # fmt: skip

    def bound(self, state: dict) -> list:
        """Every run bound by ``study3_records`` under the cells the state prescribes, none
        refused: the endpoints."""
        endpoints, problems = read_endpoints(self.root, theta0=self.runs.theta0,
                                             prescribed=state["prescribed"])  # fmt: skip
        assert problems == []
        assert all(e.state != "refused" for e in endpoints)
        return endpoints

    def ready(self) -> None:
        """The engineering and simulation records passed."""
        self.records.write("engineering", self.records.engineering())
        self.records.write("simulation", self.records.simulation())


@pytest.fixture
def world(tmp_path, monkeypatch, capsys) -> World:
    return World(tmp_path, monkeypatch, capsys)


def campaign_of(path: Path) -> dict:
    """The campaign file as YAML, and its comment lines."""
    text = path.read_text()
    data = yaml.safe_load(text)
    data["comments"] = " ".join(line[2:] for line in text.splitlines() if line.startswith("# "))
    return data


def cells_of(runs) -> list[tuple[str, float, int]]:
    """The (arm, rate, seed) of a campaign's runs, in the launcher's order."""
    out = []
    for run in runs:
        cfg = run_config(run.config, run.overrides)
        out.append((ARM_OF[str(cfg.optimizer.name)], float(cfg.optimizer.lr), run.seed))
    return out


def overrides_of(runs) -> set[tuple[str, ...]]:
    return {tuple(o for o in run.overrides if not o.startswith("seed=")) for run in runs}


def printed(world) -> str:
    return world.capsys.readouterr().out


def qualify_cases(path: Path, engineering: bool, candidate: int = 1) -> list[str]:
    """``scripts/qualify.py``'s reading of a qualification campaign: why it is not one (none)."""
    from test_qualify import SCOPE
    from test_qualify import load_script as load_qualify

    return load_qualify().campaign_cases(path, SCOPE, candidate, engineering)[2]


def test_one_composition_per_cell_is_the_launchers() -> None:
    for config, overrides in (
        ("experiments/study3", ("optimizer=adam_frozen", "train.batch_size=20",
                                "optimizer.lr=1.83e-05", "study3.role=validation")),
        ("experiments/study3_k_only", ("optimizer=adamw_prewarmed", "train.batch_size=20",
                                       "optimizer.lr=1.3e-05", "study3.role=numerical",
                                       "tracking.endpoint_checkpoint=/root/ckpt")),
    ):  # fmt: skip
        run_config(config, (*overrides, "seed=10300"))  # the cell composed with one seed
        fast = run_config(config, (*overrides, "seed=10401"))
        slow = load_run_config(config, (*overrides, "seed=10401"))
        assert config_hash(fast) == config_hash(slow)
        assert OmegaConf.to_yaml(fast) == OmegaConf.to_yaml(slow)


# ------------------------------------------------------------------ each kind, through the route


def test_the_route_from_engineering_to_confirmation_through_the_writer(world, monkeypatch):
    """Every kind but the repair, written from the state the records give, read back by the
    launcher, its runs written as the launcher and the tracker write them and bound by
    ``study3_records`` without a problem, each moving the state to its next step; and nothing
    under ``responses/`` or ``observations/`` is opened on the way, the sentinel nowhere in what
    the writer printed or wrote."""
    sealed = sealed_reads(monkeypatch)
    shown = []

    # engineering: candidate 1's starting rates on 10000-10003, the bridge's checkpoint saved
    assert world.state()["step"]["record"] == "engineering"
    out = world.write("engineering")
    shown.append(printed(world))
    campaign = campaign_of(out)
    assert campaign["name"].startswith("study3-c1-engineering-")
    assert campaign["header"] == {"study": 3, "candidate": 1, "step": "engineering",
                                  "round": None}  # fmt: skip
    name, runs = load_campaign(out)
    assert campaign_header(out) == campaign["header"] and name == campaign["name"]
    assert sorted(cells_of(runs)) == sorted(
        (arm, START[arm][label], seed) for arm in OPT for label in LABELS
        for seed in SEEDS["engineering"][1])  # fmt: skip
    assert {run.config for run in runs} == {"experiments/study3_k_only"}
    for overrides in overrides_of(runs):
        assert {"study3.role=engineering", "train.batch_size=20",
                "tracking.endpoint_checkpoint=/root/ckpt"} <= set(overrides)  # fmt: skip
    assert (f"python scripts/qualify.py driver --campaign {out} --candidate 1 --engineering"
            in shown[-1])  # fmt: skip
    world.launched(out)
    state = world.state()
    assert (state["stage"], state["step"]["record"], state["problems"]) == (
        "engineering", "engineering", [])  # fmt: skip
    found = world.bound(state)
    assert {e.cell.key for e in found} == {"c1/engineering"} and len(found) == 24

    # the search, round 0 then round 1: the header names the engineering record by its sha256
    world.ready()
    out = world.write("search-round0")
    shown.append(printed(world))
    campaign = campaign_of(out)
    named = {"engineering_record": "study3/decisions/engineering.json",
             "engineering_record_sha256": world.records.digest("engineering")}  # fmt: skip
    assert campaign["header"] == {"study": 3, "candidate": 1, "step": "search", "round": 0,
                                  **named}  # fmt: skip
    _, runs = load_campaign(out)
    assert sorted(cells_of(runs)) == sorted(
        (arm, START[arm][label], seed) for arm in OPT for label in LABELS
        for seed in SEEDS["search"][1])  # fmt: skip
    assert all("tracking.endpoint_checkpoint" not in " ".join(o) for o in overrides_of(runs))
    assert f"python scripts/launch.py {out} --max-attempts 1" in shown[-1]

    def off_target(arm, rate, seed, candidate):  # F's upper rate misses by 0.2 in ln K
        values = target(arm, rate, seed, candidate)
        return (
            {"k": values["k"] * math.exp(0.2)} if (arm, rate) == ("F", START["F"]["u"]) else values
        )

    world.launched(out, off_target)
    state = world.state()
    assert (state["step"]["campaign"], state["problems"]) == ("search-round1", [])
    out = world.write("search-round1")
    shown.append(printed(world))
    campaign = campaign_of(out)
    assert campaign["header"] == {"study": 3, "candidate": 1, "step": "search", "round": 1,
                                  **named}  # fmt: skip
    _, runs = load_campaign(out)
    adjusted = adjusted_rate(START["F"]["u"], math.log(7.5) + 0.2, 7.5, 1.30)
    assert cells_of(runs) == [("F", adjusted, seed) for seed in SEEDS["search"][1]]
    world.launched(out)

    # the numerical qualification: its launcher runs save the step-25 checkpoint
    state = world.state()
    assert (state["step"]["campaign"], state["problems"]) == ("numerical", [])
    out = world.write("numerical")
    shown.append(printed(world))
    campaign = campaign_of(out)
    assert campaign["header"] == {"study": 3, "candidate": 1, "step": "numerical", "round": None}
    _, runs = load_campaign(out)
    selected = {**{(arm, label): START[arm][label] for arm in OPT for label in LABELS},
                ("F", "u"): adjusted}  # fmt: skip
    assert sorted(cells_of(runs)) == sorted((arm, rate, seed) for (arm, _), rate in selected.items()
                                            for seed in SEEDS["numerical"][1])  # fmt: skip
    for overrides in overrides_of(runs):
        assert {"study3.role=numerical", "tracking.endpoint_checkpoint=/root/ckpt"} <= set(
            overrides
        )
    assert f"python scripts/qualify.py driver --campaign {out} --candidate 1 --commit" in shown[-1]
    world.launched(out)
    world.qualified[1] = {(arm, rate) for (arm, _), rate in selected.items()}

    # validation: 28 whole blocks, under the versions record that stands and the simulation
    state = world.state()
    assert (state["step"]["campaign"], state["problems"]) == ("validation", [])
    world.versions()
    monkeypatch.setattr("zero_field_refusal.launch.load_run_config", run_config)
    out = world.write("validation")
    shown.append(printed(world))
    ((record, kw),) = world.checked
    assert record == world.repo / VERSIONS
    assert kw == {"runs_root": world.root, "repo": world.repo, "step": "validation",
                  "candidate": 1}  # fmt: skip
    campaign = campaign_of(out)
    versioned = {"versions_record": VERSIONS,
                 "versions_record_sha256": sha256(world.repo / VERSIONS)}  # fmt: skip
    assert campaign["header"] == {
        "study": 3, "candidate": 1, "step": "validation", "round": None, **versioned,
        "simulation_record": "study3/decisions/simulation.json",
        "simulation_record_sha256": world.records.digest("simulation")}  # fmt: skip
    _, runs = load_campaign(out)
    blocks = defaultdict(list)
    for arm, rate, seed in cells_of(runs):
        blocks[seed].append((arm, rate))
    assert sorted(blocks) == list(SEEDS["validation"][1][:28])
    whole = sorted((arm, rate) for (arm, _), rate in selected.items())
    assert all(sorted(cells) == whole and len(cells) == 6 for cells in blocks.values())
    assert {run.config for run in runs} == {"experiments/study3"}
    assert all("tracking.endpoint_checkpoint" not in " ".join(o) for o in overrides_of(runs))
    world.launched(out)
    state = world.state()
    assert (state["stage"], state["step"]["record"], state["problems"]) == ("choice", "n-c1", [])
    assert len([e for e in world.bound(state) if e.cell.key == "c1/validation"]) == 168

    # the confirmation: one block per file, in seed order, and the guarded driver
    fields = state["step"]["fields"]
    world.records.write("n-c1", world.records.n(1, 60, rates=fields["rates"],
                                                active=fields["active"], cohort=fields["cohort"],
                                                cap=70))  # fmt: skip
    state = world.state()
    assert (state["step"]["campaign"], state["problems"]) == ("confirmation", [])
    world.checked.clear()
    out = world.write("confirmation", "--blocks", "10400", "10402")
    shown.append(printed(world))
    assert [kw["step"] for _, kw in world.checked] == ["confirmation"]
    files = sorted(out.glob("*.yaml"))
    assert [campaign_of(f)["name"].split("-s")[-1].split("-")[0] for f in files] == [
        "10400", "10401", "10402"]  # fmt: skip
    header = {"study": 3, "candidate": 1, "step": "confirmation", "round": None, **versioned,
              "n_record": "study3/decisions/n-c1.json",
              "n_record_sha256": world.records.digest("n-c1")}  # fmt: skip
    for seed, path in zip((10400, 10401, 10402), files, strict=True):
        assert campaign_of(path)["header"] == header
        _, runs = load_campaign(path)
        assert sorted(cells_of(runs)) == sorted(
            (arm, rate, seed) for (arm, _), rate in selected.items()
        )
    assert f"bash {out / 'run.sh'}" in shown[-1]
    for path in sorted(files, key=lambda f: campaign_of(f)["name"]):
        world.launched(path)
    state = world.state()
    assert state["problems"] == [] and state["confirmation"]["started"] == [10400, 10401, 10402]
    assert (state["step"]["campaign"], state["step"]["cells"][0][2]) == ("confirmation", 10403)
    world.bound(state)

    assert sealed == []
    written = [p.read_text() for p in world.out.rglob("*") if p.is_file()]
    assert all(SENTINEL not in text for text in [*shown, *written])
    with pytest.raises(AssertionError):  # the guard is live
        next((world.root / "study3").glob("*/responses/*.jsonl")).read_text()


def test_the_repair_campaign_qualifies_a_repaired_rate(world, monkeypatch) -> None:
    """A's lower rate kept by the search (median within 0.05) and missing over its 4 + 3 seeds:
    its repair round, a numerical qualification of the repaired rate, header step repair with no
    round, its checkpoint where qualify.py expects it."""
    sealed = sealed_reads(monkeypatch)
    world.ready()
    offsets = dict(zip(SEEDS["search"][1], (0.0, 0.03, 0.06, 0.3), strict=True))

    def search(arm, rate, seed, candidate):
        values = target(arm, rate, seed, candidate)
        return ({"k": 4.8 * math.exp(offsets[seed])} if (arm, rate) == ("A", START["A"]["l"])
                else values)  # fmt: skip

    world.launched(world.write("search-round0"), search)

    def numerical(arm, rate, seed, candidate):
        values = target(arm, rate, seed, candidate)
        return {"k": 4.8 * math.exp(0.3)} if (arm, rate) == ("A", START["A"]["l"]) else values

    world.launched(world.write("numerical"), numerical)
    world.qualified[1] = {(arm, START[arm][label]) for arm in OPT for label in LABELS}
    state = world.state()
    assert (state["step"]["campaign"], state["problems"]) == ("repair", [])
    out = world.write("repair")
    shown = printed(world)
    campaign = campaign_of(out)
    assert campaign["header"] == {"study": 3, "candidate": 1, "step": "repair", "round": None}
    _, runs = load_campaign(out)
    repaired = adjusted_rate(START["A"]["l"], math.log(4.8) + 0.3, 4.8, 1.25)
    assert cells_of(runs) == [("A", repaired, seed) for seed in SEEDS["numerical"][1]]
    (overrides,) = overrides_of(runs)
    assert {"study3.role=numerical", "tracking.endpoint_checkpoint=/root/ckpt"} <= set(overrides)
    assert qualify_cases(out, engineering=False) == []
    assert f"python scripts/qualify.py driver --campaign {out} --candidate 1 --commit" in shown
    world.launched(out)
    world.qualified[1].add(("A", repaired))
    state = world.state()
    assert (state["step"]["campaign"], state["problems"]) == ("validation", [])
    assert state["rates"]["A"]["l"] == repaired
    assert sealed == [] and SENTINEL not in shown + out.read_text()


def test_qualify_reads_the_engineering_and_numerical_campaigns_and_their_checkpoint(world):
    """``scripts/qualify.py``'s reading of the campaigns: K-only runs of the stage's role at B 20
    and 25 steps, its seeds, every endpoint checkpoint in one absolute directory (``--checkpoints``
    sets it), each FA rate's case on the first numerical seed."""
    out = world.write("engineering")
    assert qualify_cases(out, engineering=True) == []
    world.ready()
    world.launched(world.write("search-round0"))
    elsewhere = world.write("numerical", "--checkpoints", "/mnt/ckpt")
    _, runs = load_campaign(elsewhere)
    assert all("tracking.endpoint_checkpoint=/mnt/ckpt" in o for o in overrides_of(runs))
    assert qualify_cases(elsewhere, engineering=False) == []
    first = world.write("numerical", "--seeds", "10200")
    _, runs = load_campaign(first)
    assert {run.seed for run in runs} == {10200} and len(runs) == 6
    assert qualify_cases(first, engineering=False) == []
    assert "absolute" in world.refused("numerical", "--checkpoints", "ckpt")
    assert "--checkpoints" in world.refused("search-round0", "--checkpoints", "/mnt/ckpt")


# ------------------------------------------------------------------ what the step allows


def test_a_kind_or_a_candidate_the_step_does_not_name_is_refused(world) -> None:
    study = Study()
    study.ready()
    world.records.write("engineering", world.records.engineering())
    world.core(study)
    for kind in ("engineering", "search-round1", "numerical", "repair", "validation",
                 "confirmation"):  # fmt: skip
        error = world.refused(kind)
        assert f"the step is not {kind}" in error and '"campaign": "search-round0"' in error
    error = world.refused("search-round0", candidate=2)
    assert "candidate 1" in error and '"campaign": "search-round0"' in error
    world.write("search-round0")  # the step's own
    # the engineering campaign only while the step is the engineering record
    simulation = Study()
    simulation.record("engineering")
    world.core(simulation)
    error = world.refused("engineering")
    assert "the step is not engineering" in error and '"record": "simulation"' in error
    world.core(Study())
    world.write("engineering")


def test_a_blocked_step_and_a_terminal_stage_are_refused(world) -> None:
    study = Study()
    study.ready()
    study.problems.append("20261026T000099Z-x-s10100-abcd1234-a1: a run with no start")
    world.core(study)
    for kind in ("search-round0", "numerical"):
        error = world.refused(kind)
        assert "the step is blocked" in error and '"kind": "blocked"' in error
        assert "a run with no start" in error
    terminal = Study()
    terminal.record("engineering", "failed")
    world.core(terminal)
    for kind in ("engineering", "search-round0"):
        error = world.refused(kind)
        assert "technical_feasibility_report" in error and "terminal" in error


def test_a_step_with_no_cells_is_refused(world) -> None:
    """A campaign step that names no cell (the readers never give one) writes no empty file."""
    study = Study()
    study.ready()
    world.records.write("engineering", world.records.engineering())

    def empty(*args, **kwargs):
        state = study.advance()
        state["step"]["cells"] = []
        return state

    world.monkeypatch.setattr(world.script, "progress", empty)
    assert "names no cells" in world.refused("search-round0")


def test_validation_without_its_versions_record_is_refused(world) -> None:
    """The record that stands is missing: the real check says so, and nothing is written; any
    problem of the check refuses, printed."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    world.ready()
    world.core(study)
    error = world.refused("validation")
    assert f"{world.repo / VERSIONS} is missing" in error
    world.versions(["docs/study3-versions.json: HEAD is not the recorded commit"])
    error = world.refused("validation")
    assert "HEAD is not the recorded commit" in error
    world.versions()
    world.write("validation", "--seeds", "10300")


def test_the_versions_record_that_stands_is_the_latest(world) -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    world.ready()
    world.core(study)
    world.versions()
    later = world.repo / "docs" / "study3-versions-2.json"
    later.write_text('{"what": "versions, the second"}\n')
    out = world.write("validation", "--seeds", "10300")
    assert world.checked[-1][0] == later
    assert campaign_of(out)["header"]["versions_record"] == "docs/study3-versions-2.json"
    assert campaign_of(out)["header"]["versions_record_sha256"] == sha256(later)


def test_validation_issues_whole_blocks_of_six_cells_or_four_once_p_is_closed(world) -> None:
    study = Study()
    study.ready()
    study.search(special={("P", "l", 10100): {"state": "scientific"}})  # P fails: P–F closes
    study.numerical(arms=("A", "F"))
    world.ready()
    world.versions()
    world.core(study)
    out = world.write("validation", "--seeds", "10300", "10301")
    _, runs = load_campaign(out)
    blocks = defaultdict(list)
    for arm, rate, seed in cells_of(runs):
        blocks[seed].append((arm, rate))
    assert sorted(blocks) == [10300, 10301]
    assert all(
        sorted(cells) == sorted((arm, START[arm][label]) for arm in ("A", "F") for label in LABELS)
        for cells in blocks.values()
    )
    full = Study()
    full.ready()
    full.search()
    full.numerical()
    world.core(full)
    _, runs = load_campaign(world.write("validation", "--seeds", "10300"))
    assert len(runs) == 6 and {arm for arm, _, _ in cells_of(runs)} == {"A", "F", "P"}


def test_seeds_select_the_steps_first_blocks_and_never_add_skip_or_reorder(world) -> None:
    study = Study()
    study.ready()
    world.records.write("engineering", world.records.engineering())
    world.core(study)
    _, runs = load_campaign(world.write("search-round0", "--seeds", "10100"))
    assert {run.seed for run in runs} == {10100} and len(runs) == 6
    _, runs = load_campaign(world.write("search-round0", "--seeds", "10100", "10101"))
    assert [run.seed for run in runs][:2] == [10100, 10101] and len(runs) == 12
    assert "skips" in world.refused("search-round0", "--seeds", "10101")
    assert "skips" in world.refused("search-round0", "--seeds", "10100", "10102")
    assert "never reorders" in world.refused("search-round0", "--seeds", "10101", "10100")
    assert "never adds" in world.refused("search-round0", "--seeds", "10100", "10200")
    assert "twice" in world.refused("search-round0", "--seeds", "10100", "10100")
    assert "--blocks" in world.refused("search-round0", "--blocks", "10100", "10101")


def confirmation_study(world, N=75, budget=False) -> Study:
    """Candidate 1 through validation with its n record (and budget record): the confirmation's
    first block is next. The records the headers name are written (valid)."""
    world.records.chain(1, N, through="budget" if budget else "n")
    study = Study()
    study.through_validation()
    study.n(N=N)
    if budget:
        study.budget()
    world.versions()
    world.core(study)
    return study


def test_the_confirmation_writes_one_file_per_block_and_names_its_records(world) -> None:
    confirmation_study(world, N=75)
    out = world.write("confirmation")
    (only,) = out.glob("*.yaml")
    campaign = campaign_of(only)
    assert campaign["name"].startswith("study3-c1-confirmation-s10400-")
    assert only.name == f"{campaign['name']}.yaml"
    assert campaign["header"] == {
        "study": 3, "candidate": 1, "step": "confirmation", "round": None,
        "versions_record": VERSIONS, "versions_record_sha256": sha256(world.repo / VERSIONS),
        "n_record": "study3/decisions/n-c1.json",
        "n_record_sha256": world.records.digest("n-c1")}  # fmt: skip
    _, runs = load_campaign(only)
    assert sorted(cells_of(runs)) == sorted((arm, START[arm][label], 10400) for arm in OPT
                                            for label in LABELS)  # fmt: skip
    assert {run.config for run in runs} == {"experiments/study3"}
    assert (out / "run.sh").is_file()
    out = world.write("confirmation", "--seeds", "10400", "10401")
    assert len(list(out.glob("*.yaml"))) == 2
    # the cap N + max(10, ⌈0.12·N⌉) = 85 blocks: 10400-10484
    out = world.write("confirmation", "--blocks", "10400", "10484")
    assert len(list(out.glob("*.yaml"))) == 85
    assert "never adds" in world.refused("confirmation", "--blocks", "10400", "10485")
    assert "skips" in world.refused("confirmation", "--blocks", "10401", "10402")
    assert "skips" in world.refused("confirmation", "--seeds", "10401")
    assert "never reorders" in world.refused("confirmation", "--seeds", "10401", "10400")
    assert "not both" in world.refused("confirmation", "--seeds", "10400", "--blocks", "10400",
                                       "10400")  # fmt: skip


def test_a_next_block_that_is_not_whole_is_refused(world) -> None:
    """The confirmation's next block names every active arm at its two rates; a step that names
    fewer (the readers never give one) is refused rather than repeated to the cap."""
    study = confirmation_study(world)

    def short(*args, **kwargs):
        state = study.advance()
        state["step"]["cells"] = state["step"]["cells"][1:]
        return state

    world.monkeypatch.setattr(world.script, "progress", short)
    assert "is not whole" in world.refused("confirmation", "--blocks", "10400", "10401")


def test_the_headers_name_their_records_as_immutable_reads_them(world) -> None:
    """Once a campaign that names a decision record has started, the record is immutable
    (``study3_decisions.decisions``, through ``immutable``): the confirmation's n record, the
    validation's simulation record, the search's engineering record, each changed after its
    campaign's runs started, is invalid, the campaign named."""
    confirmation_study(world)
    (block,) = world.write("confirmation").glob("*.yaml")
    world.launched(block)
    validation = Study()
    validation.ready()
    validation.search()
    validation.numerical()
    world.core(validation)
    world.launched(validation_file := world.write("validation", "--seeds", "10300"))
    search = Study()
    search.ready()
    world.core(search)
    world.launched(search_file := world.write("search-round0", "--seeds", "10100"))
    assert all(decisions(world.root, 1, repo=world.repo)[kind]["state"] == "passed"
               for kind in ("engineering", "simulation", "n-c1"))  # fmt: skip
    for kind, path in (("n-c1", block), ("simulation", validation_file),
                       ("engineering", search_file)):  # fmt: skip
        world.records.change(kind, written_at="2026-10-26T13:00:00Z")  # a valid record still
        entry = decisions(world.root, 1, repo=world.repo)[kind]
        assert entry["state"] == "invalid"
        assert any(campaign_of(path)["name"] in p and "sha256" in p for p in entry["problems"])


def test_the_confirmation_names_its_budget_record_when_n_needs_one(world) -> None:
    confirmation_study(world, N=99, budget=True)
    out = world.write("confirmation")
    (only,) = out.glob("*.yaml")
    header = campaign_of(only)["header"]
    assert (header["budget_record"], header["budget_record_sha256"]) == (
        "study3/decisions/budget-c1.json", world.records.digest("budget-c1"))  # fmt: skip
    assert header["n_record_sha256"] == world.records.digest("n-c1")


def test_a_started_blocks_missing_cells_are_issued_alone(world) -> None:
    """A started block with a never-issued cell is completed first: the step's cells, nothing
    else, and no later block with them."""
    study = confirmation_study(world)
    study.confirmation(count=1)
    study.drop(next(e.run_id for e in study.endpoints if e.cell.step == "confirmation"
                    and e.arm == "P" and e.lr == START["P"]["u"]))  # fmt: skip
    out = world.write("confirmation")
    (only,) = out.glob("*.yaml")
    _, runs = load_campaign(only)
    assert cells_of(runs) == [("P", START["P"]["u"], 10400)]
    error = world.refused("confirmation", "--blocks", "10400", "10401")
    assert "completed first" in error


def test_the_driver_runs_the_guard_before_each_block_and_stops_at_its_refusal(world, tmp_path):
    """run.sh with stand-ins for the guard and the launcher on PATH: the guard of each block
    before its launch with --max-attempts 1, one block at a time in seed order; the first guard
    that refuses (or a launch that fails) stops it."""
    confirmation_study(world)
    out = world.write("confirmation", "--blocks", "10400", "10402")
    driver = out / "run.sh"
    text = driver.read_text()
    subprocess.run(["bash", "-n", str(driver)], check=True)
    assert "&" not in re.sub(r"&&|>&2|2>&1", "", text)  # nothing runs in the background
    files = {campaign_of(f)["name"].split("-s")[-1].split("-")[0]: f for f in out.glob("*.yaml")}
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "calls.txt"
    (bin_ / "python").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {calls}\n'
        'case "$*" in\n'
        '  *"guard"*"--seed $REFUSE"*) exit 3 ;;\n'
        '  *"launch.py"*"$FAIL"*) exit 1 ;;\n'
        "esac\n")  # fmt: skip
    (bin_ / "python").chmod(0o755)

    def run(refuse="none", fail="none"):
        calls.write_text("")
        env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "REFUSE": refuse,
               "FAIL": fail}  # fmt: skip
        done = subprocess.run(["bash", str(driver)], env=env, capture_output=True, text=True)
        return done, calls.read_text().splitlines()

    guard = f"scripts/study3_summary.py guard --runs-root {world.root} --candidate 1 --seed"
    done, lines = run()
    assert done.returncode == 0, done.stderr
    assert lines == [line for seed in ("10400", "10401", "10402") for line in (
        f"{guard} {seed}", f"scripts/launch.py {files[seed]} --max-attempts 1")]  # fmt: skip
    done, lines = run(refuse="10401")
    assert done.returncode == 3 and "the guard refused block 10401" in done.stderr
    assert lines == [f"{guard} 10400", f"scripts/launch.py {files['10400']} --max-attempts 1",
                     f"{guard} 10401"]  # fmt: skip
    done, lines = run(fail=files["10400"].name)
    assert done.returncode == 1
    assert lines == [f"{guard} 10400", f"scripts/launch.py {files['10400']} --max-attempts 1"]
    files["10401"].write_text(files["10401"].read_text() + "# edited\n")
    done, lines = run()
    assert done.returncode != 0 and "is not the file written for block 10401" in done.stderr
    assert f"scripts/launch.py {files['10401']} --max-attempts 1" not in lines


def test_a_held_or_closed_label_is_never_issued_again(world) -> None:
    """A step that names a cell whose label the launch journal holds, or whose run a closure
    record closed, is refused (the readers never issue one; the writer checks again)."""
    study = Study()
    study.ready()
    study.search()
    world.ready()
    world.core(study)
    label = "c1-numerical-adam_frozen-1.83e-05/s10200"
    world.runs.log("held", {"event": "held", "run": label, "run_id": "20261026T000001Z-x-a1",
                            "experiment": "study3", "config_hash": "c" * 12,
                            "reason": "no recorded commit"})  # fmt: skip
    error = world.refused("numerical")
    assert label in error and "held" in error
    (world.root / "launch" / "held.jsonl").unlink()
    closure = decision_hold(world.records, arm="F", rate=1.83e-5, seed=10200)
    world.records.write("closure-c1-F", closure)
    error = world.refused("numerical")
    assert "(F, 1.83e-05, 10200)" in error and "closure" in error


def test_a_campaign_name_whose_log_exists_is_refused(world, monkeypatch) -> None:
    study = Study()
    study.ready()
    world.records.write("engineering", world.records.engineering())
    world.core(study)
    monkeypatch.setattr(world.script, "stamp", lambda: "20261027T000000Z")
    (world.root / "launch" / "study3-c1-search-round0-20261027T000000Z.jsonl").write_text("")
    error = world.refused("search-round0")
    assert "study3-c1-search-round0-20261027T000000Z" in error and "already" in error
    # nor is a file written over
    existing = world.out / "existing.yaml"
    existing.write_text("kept\n")
    argv, _ = world.argv("search-round1", out=existing)
    with pytest.raises(SystemExit):
        world.script.main(argv)
    assert "exists" in world.capsys.readouterr().err and existing.read_text() == "kept\n"


# ------------------------------------------------------------------ prep


def test_prep_writes_the_streams_of_two_seeds_under_their_runs_configs(world, monkeypatch):
    sealed = sealed_reads(monkeypatch)
    out = world.out / "prep.sh"
    world.script.main(["prep", "--seeds", "10100", "10300", "--commit", COMMIT, "--out", str(out)])
    text = out.read_text()
    subprocess.run(["bash", "-n", str(out)], check=True)
    commands = joined(text)
    run = 'project-run "$SRC" python scripts/'
    for seed, config in ((10100, "experiments/study3_k_only study3.role=search"),
                         (10300, "experiments/study3 study3.role=validation")):  # fmt: skip
        assert (f"step {run}generate_teacher.py {config} --stream training --seeds {seed} "
                "--responses 500") in commands  # fmt: skip
        assert (f"kept {run}generate_teacher.py {config} optimizer=adam_frozen train.batch_size=20 "
                f"--stream preconditioner --seeds {seed}") in commands  # fmt: skip
    assert re.search(r"Written by scripts/study3_campaigns\.py prep at ", text)
    assert sealed == []
    for bad, why in ((["--seeds", "2000"], "registered"),
                     (["--seeds", "10100", "10100"], "twice"),
                     (["--seeds", "10100", "--commit", "abc"], "40-character")):  # fmt: skip
        argv = ["prep", "--commit", COMMIT, *bad, "--out", str(world.out / "bad.sh")]
        world.capsys.readouterr()
        with pytest.raises(SystemExit) as found:
            world.script.main(argv)
        assert found.value.code not in (0, None)
        assert why in world.capsys.readouterr().err
    assert not (world.out / "bad.sh").exists()


def test_the_writer_imports_no_gpu_path() -> None:
    """The writer and the shared prep import no GPU path and no study-2 analysis; the state's
    reader loads the qualification's reader (and torch with it) only when it runs."""
    code = ("import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('writer', {str(SCRIPT)!r}); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "print([m for m in ('torch', 'transformers', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.qualification', 'zero_field_refusal.analysis.stage_a', "
            "'zero_field_refusal.analysis.study2_short') if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
