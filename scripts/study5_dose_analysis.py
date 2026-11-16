"""Study 5's dose experiment analysed on a CPU (docs/study5-dose-registration.md): the dose runs'
records and WildGuard labels read, every quantity computed by ``analysis.study5_dose``, written
once as one JSON record.

    python scripts/study5_dose_analysis.py --runs ROOT --seeds docs/study5-dose-seeds.json
        --grid FILE --grid-sha256 HEX --theta0-run DIR --forecasts FILE --forecasts-sha256 HEX
        --out RECORD.json

**Inputs.** ROOT is the runs root: the dose runs are the run directories under ``ROOT/study5/``
whose ``config.yaml`` gives ``study5.role: dose``; no other run is read further. --seeds is the
committed seeds record, its seeds the draw's (``dose.draw_seeds``). --grid is the grid record the
Pods' driver writes (``scripts/study5_dose_campaigns.py grid``: kind ``study5-dose-grid``, both
Pods' pilot tables whole and ``grid``, ``dose.grid_choice``'s return value on them), read only when
the whole file's sha256 is --grid-sha256; its ``grid`` must be the rule's on its tables, and its
``grid.chosen`` pair names two late candidates, each by its name and its rate. --theta0-run is study 5's θ₀
evaluation run (``experiments/study5_theta0``, run type ``theta0``): its accuracies decide gross
collapse and the utility losses. --forecasts holds the forecasts of C and A, each exactly the ten
keys ``study5_dose.FIELDS`` (``study5_dose.check_forecasts``), read only when its sha256 is
--forecasts-sha256.

**The runs.** Each dose run's key (seed, early level, late level) is ``study5_dose.select_runs``'
from its config (``seed``, ``train.rate_schedule.early_lr`` and ``early_steps``,
``optimizer.lr``) and whether its ``terminal.json`` exists. Each keyed run is read as the analysis
CLI reads an F run (``study5_analysis_cli.f_run``: its config's seed, late rate and optimizer; its
stop, steps and step-25 evaluation), with its monitor reading at step 5 (``monitor.jsonl``), its
metrics log's rows of steps 0 … 4 (``metrics.jsonl``, the rows with a ``loss``) and its WildGuard
labels at the step-25 condition as that CLI reads them (``study5_analysis_cli.Labels`` through
``run_labels``: every check of study 2's reader, the receipt's sha256 kept). A record or label
file that does not read is the run's reason, named where it leaves a quantity, never a refusal.

**The record** (--out, never overwritten, written whole by a link): every quantity of
``study5_dose.analysis``; the inputs' sha256 (the seeds, grid and forecasts files, θ₀'s config and
summary and every run file read, by its root and relative path, and each prompt set's); the label
receipts' sha256 by run; the selection (the keyed runs, those not ended, of other seeds, or whose
config does not read); the git state and the UTC time. Strict JSON, sorted keys, a non-finite
number as ``{"missing": "nonfinite"}`` (``study5_analysis.finite_tree``). It holds numbers, ids,
states and reasons, never a text. stdout carries numbers, ids and hashes only. Exits 0 when
written.

**Refusals** exit 1 with ``study5_dose_analysis: <why>`` and write nothing: an existing --out, a
grid or forecasts file whose sha256 is not the one given or that does not read, a seeds record
that is not the draw, a chosen pair that is not two late candidates by name and rate, forecasts
that ``check_forecasts`` refuses, a θ₀ run that is none or does not read, a dose run whose rate is
no level or whose early steps are not 5, and a key with two ended runs (naming them).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from zero_field_refusal import dose
from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_dose as sd
from zero_field_refusal.analysis.study3_records import read_theta0
from zero_field_refusal.analysis.study5_analysis_cli import Refused, refuse
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

PROG = "study5_dose_analysis"
KIND, VERSION = "study5-dose-analysis", 1
EXPERIMENT = "study5"  # the dose runs' folder under the runs root
CONDITION = cli.ENDPOINT  # step-25
THETA0_RUN_TYPE = "theta0"


# ------------------------------------------------------------------ the inputs


def checked(reader: cli.Reader, path: Path, digest: str, what: str) -> bytes:
    """A file's bytes, refused unless its sha256 is ``digest``."""
    try:
        data = reader.bytes(path)
    except OSError as exc:
        refuse(f"{what} does not read ({type(exc).__name__})")
    found = cli.sha256(data)
    if found != digest.strip().lower():
        refuse(f"{what}: its sha256 {found} is not the one given, {digest}")
    return data


def json_of(data: bytes, what: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        refuse(f"{what} is not JSON")


def seeds_of(reader: cli.Reader, path: Path) -> list[int]:
    """The seeds record's seeds, refused unless they are the draw's."""
    try:
        found = json.loads(reader.bytes(path))
    except (OSError, ValueError) as exc:
        refuse(f"the seeds record does not read ({type(exc).__name__})")
    seeds = found.get("seeds") if isinstance(found, dict) else None
    if seeds != sorted(dose.draw_seeds()):
        refuse("the seeds record's seeds are not the draw's (dose.draw_seeds)")
    return list(seeds)


GRID_KIND = "study5-dose-grid"  # the Pods' driver's grid record (scripts/study5_dose_campaigns.py)


def chosen_of(record: Any) -> dict[str, Any]:
    """The grid record's chosen pair (``record["grid"]["chosen"]``): the record the Pods' driver
    writes, its ``grid`` the rule's (``dose.grid_choice``) on the two pilot tables it holds; two
    late candidates, each named with its exact rate, the lower first."""
    if not isinstance(record, dict) or record.get("kind") != GRID_KIND:
        refuse(f"the grid record is not a {GRID_KIND} record")
    tables = record.get("tables")
    if not isinstance(tables, dict) or sorted(tables) != sorted(dose.PODS) or not all(
            isinstance(tables[pod], dict) and isinstance(tables[pod].get("rows"), list)
            for pod in dose.PODS):  # fmt: skip
        refuse("the grid record does not hold both Pods' pilot tables")
    try:
        rule = dose.grid_choice([row for pod in dose.PODS for row in tables[pod]["rows"]],
                                [int(tables[pod]["seed"]) for pod in dose.PODS])  # fmt: skip
    except (ValueError, KeyError, TypeError) as exc:
        refuse(f"the grid record's tables are not the pilot's ({type(exc).__name__}: {exc})")
    grid = record.get("grid")
    if grid != rule:
        refuse("the grid record's choice is not the rule's on its tables")
    chosen = grid.get("chosen")
    rates = dict(dose.LATE_CANDIDATES)
    if not isinstance(chosen, dict) or chosen.get("lo") not in rates or (
            chosen.get("hi") not in rates):  # fmt: skip
        refuse("the grid record's chosen pair does not name two late candidates")
    lo, hi = chosen["lo"], chosen["hi"]
    if chosen.get("L_lo") != rates[lo] or chosen.get("L_hi") != rates[hi] or not (
            rates[lo] < rates[hi]):  # fmt: skip
        refuse(f"the grid record's chosen pair ({lo!r}, {hi!r}) does not give the candidates' "
               "rates, the lower first")  # fmt: skip
    return {"lo": lo, "hi": hi, "L_lo": rates[lo], "L_hi": rates[hi],
            "covered": chosen.get("covered")}  # fmt: skip


def theta0_of(reader: cli.Reader, run_dir: Path) -> dict[str, float]:
    """θ₀'s accuracies (``study3_records.read_theta0``) from its evaluation run, its run type
    ``theta0``; refused otherwise."""
    try:
        cfg = OmegaConf.create(reader.bytes(run_dir / "config.yaml").decode())
        run_type = cfg.get("run_type") if isinstance(cfg, DictConfig) else None
        reader.bytes(run_dir / "summary.json")  # hashed as an input; read_theta0 reads it
        accuracies = read_theta0(run_dir)
    except (OSError, ValueError) as exc:
        refuse(f"θ₀'s evaluation does not read: {reader.scrub(str(exc))}")
    if run_type != THETA0_RUN_TYPE:
        refuse(f"{run_dir.name} is not a θ₀ evaluation run (run type {run_type!r})")
    return accuracies


# ------------------------------------------------------------------ the runs


def dose_dirs(reader: cli.Reader, root: Path) -> tuple[list[dict[str, Any]], list[dict]]:
    """Every run directory under ``ROOT/study5/`` whose config gives role ``dose``, as
    ``study5_dose.select_runs`` takes it (with its directory), its config hashed as an input; and
    the directories whose config does not read."""
    folder = root / EXPERIMENT
    found, unreadable = [], []
    if not folder.is_dir():
        return found, unreadable
    for run_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        path = run_dir / "config.yaml"
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            cfg = OmegaConf.create(data.decode())
            role = OmegaConf.select(cfg, "study5.role", default=None)
        except Exception as exc:  # any config that does not parse is listed, not read further
            unreadable.append({"run_id": run_dir.name, "reason": type(exc).__name__})
            continue
        if role != sd.ROLE:
            continue
        reader.hashes[reader.name(path)] = cli.sha256(data)
        found.append({
            "run_id": run_dir.name, "run_dir": run_dir,
            "seed": OmegaConf.select(cfg, "seed", default=None),
            "early_lr": OmegaConf.select(cfg, "train.rate_schedule.early_lr", default=None),
            "early_steps": OmegaConf.select(cfg, "train.rate_schedule.early_steps", default=None),
            "late_lr": OmegaConf.select(cfg, "optimizer.lr", default=None),
            "ended": (run_dir / "terminal.json").is_file(),
        })  # fmt: skip
    return found, unreadable


def monitor_reading(reader: cli.Reader, run_dir: Path) -> dict[str, Any] | str:
    """The run's monitor reading after update 5 (``monitor.jsonl``, its row of step 5), or why
    there is none."""
    path = run_dir / "monitor.jsonl"
    if not path.is_file():
        return "no monitor.jsonl"
    try:
        rows = reader.jsonl(path)
    except (OSError, ValueError) as exc:
        return f"monitor.jsonl does not read ({type(exc).__name__})"
    found = [r for r in rows if isinstance(r, dict) and r.get("step") == sd.MONITOR_STEP]
    if len(found) != 1:
        return f"{len(found)} monitor readings at step {sd.MONITOR_STEP}"
    return {key: found[0].get(key) for key in sd.MONITOR_METRICS}


def first_steps(reader: cli.Reader, run_dir: Path) -> dict[int, Any] | str:
    """The run's metrics log rows of steps 0 … 4 (the rows with a ``loss``: the updates 1 … 5),
    by step, each its ``STEP_METRICS`` or why it is not one row; why the log does not read."""
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return "no metrics.jsonl"
    try:
        rows = [r for r in reader.jsonl(path) if isinstance(r, dict) and "loss" in r]
    except (OSError, ValueError) as exc:
        return f"metrics.jsonl does not read ({type(exc).__name__})"
    out: dict[int, Any] = {}
    for step in sd.FIRST_STEPS:
        found = [r for r in rows if r.get("step") == step]
        out[step] = ({key: found[0].get(key) for key in sd.STEP_METRICS} if len(found) == 1
                     else f"{len(found)} rows with a loss")  # fmt: skip
    return out


def read_run(reader: cli.Reader, labels: cli.Labels, run_dir: Path,
             key: sd.Key) -> dict[str, Any]:  # fmt: skip
    """One keyed run's readings (``study5_dose``'s ``Run``)."""
    seed, early, late = key
    found = cli.f_run(reader, run_dir, seed, dict(dose.LATE_CANDIDATES)[late])
    run: dict[str, Any] = {"run_id": run_dir.name, "seed": seed, "early": early, "late": late,
                           "records": None, "stop": None, "steps": None, "row": None}  # fmt: skip
    if isinstance(found, str):
        run["records"] = found
    else:
        run |= {"stop": found["stop"], "steps": found["steps"], "row": found["row"]}
    run["monitor5"] = monitor_reading(reader, run_dir)
    run["first_steps"] = first_steps(reader, run_dir)
    got = labels.of(run_dir)
    if isinstance(got, Mapping):
        run["labels"] = got["conditions"].get(CONDITION, f"labels: no condition {CONDITION}")
        run["receipt_sha256"] = got["receipt_sha256"]
    else:
        run["labels"], run["receipt_sha256"] = got, None
    return run


# ------------------------------------------------------------------ the record


def record_of(args: argparse.Namespace) -> dict[str, Any]:
    roots = {"runs": args.runs, "theta0": args.theta0_run, "grid": args.grid.parent,
             "forecasts": args.forecasts.parent, "seeds": args.seeds.parent,
             "repo": REPO_ROOT}  # fmt: skip
    reader = cli.Reader(roots)
    grid_data = checked(reader, args.grid, args.grid_sha256, "the grid record")
    forecasts_data = checked(reader, args.forecasts, args.forecasts_sha256, "the forecasts")
    chosen = chosen_of(json_of(grid_data, "the grid record"))
    try:
        forecasts = sd.check_forecasts(json_of(forecasts_data, "the forecasts"))
    except ValueError as exc:
        refuse(str(exc))
    seeds = seeds_of(reader, args.seeds)
    theta0 = theta0_of(reader, args.theta0_run)
    entries, unreadable = dose_dirs(reader, args.runs)
    try:
        selection = sd.select_runs(entries, seeds)
    except ValueError as exc:
        refuse(str(exc))
    dirs = {e["run_id"]: e["run_dir"] for e in entries}
    labels = cli.Labels(reader, cli.prompt_sets(reader))
    runs = {key: read_run(reader, labels, dirs[run_id], key)
            for key, run_id in selection["runs"].items()}  # fmt: skip
    result = sd.analysis(runs, seeds, chosen, theta0, forecasts)
    return sa.finite_tree({
        "kind": KIND, "version": VERSION,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(REPO_ROOT),
        "inputs": {
            "seeds": {"file": reader.name(args.seeds),
                      "sha256": reader.hashes[reader.name(args.seeds)]},
            "grid": {"file": reader.name(args.grid), "sha256": cli.sha256(grid_data),
                     "chosen": chosen},
            "forecasts": {"file": reader.name(args.forecasts),
                          "sha256": cli.sha256(forecasts_data), "forecasts": forecasts},
            "theta0": {"run_id": Path(args.theta0_run).resolve().name, "accuracies": theta0},
            "prompt_sets": dict(sorted(reader.prompt_sets.items())),
            "files": dict(sorted(reader.hashes.items())),
        },
        "label_receipts": {run["run_id"]: run["receipt_sha256"] for run in runs.values()},
        "selection": {
            "experiment": EXPERIMENT,
            "keyed": [{"seed": s, "early": e, "late": late, "run_id": run_id}
                      for (s, e, late), run_id in selection["runs"].items()],
            "not_ended": selection["not_ended"], "other_seeds": selection["other_seeds"],
            "unreadable": unreadable,
        },
        **result,
    })  # fmt: skip


def _n(value: Any) -> str:
    """A number as stdout carries it; ``none`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "none"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def report(record: Mapping[str, Any], digest: str) -> None:
    """Numbers, ids and hashes only."""
    main, kstar = record["main"], record["kstar"]
    not_ended = len(record["selection"]["not_ended"])
    print(f"runs {len(record['runs'])} third_candidate {len(record['third_candidate'])} "
          f"missing {len(record['missing'])} not_ended {not_ended}")  # fmt: skip
    for key in record["missing"]:
        print(f"missing s{key['seed']} early {key['early']} late {key['late']}")
    for name in ("d", "c_L_lo", "c_L_hi"):
        found = main[name]
        print(f"main {name} n {found['n']} mean {_n(found['mean'])} lower {_n(found['lower'])} "
              f"upper {_n(found['upper'])} excluded {len(found['excluded'])}")  # fmt: skip
    signs = main["signs"]
    print(f"signs n {signs['n']} positive {signs['positive']} negative {signs['negative']} "
          f"zero {signs['zero']} p {_n(signs['p_one_sided'])}")  # fmt: skip
    k = kstar["k"]
    print(f"kstar k n {k['n']} mean {_n(k['mean'])} lower {_n(k['lower'])} upper "
          f"{_n(k['upper'])} unsupported {kstar['unsupported']['n']} without_labels "
          f"{kstar['without_labels']['n']}")  # fmt: skip
    for name in (f"{sd.HIGH}/{sd.LOW}", f"{sd.MID}/{sd.LOW}"):
        found = record["k5_ratios"][name]
        print(f"k5 {name} n {found['n']} geometric_mean {_n(found['geometric_mean'])}")
    first = record["first_updates"]
    print(f"first_updates compared {first['compared_cells']} differing "
          f"{first['differing_cells']} max_abs {_n((first['max_abs'] or {}).get('value'))} "
          f"max_rel {_n((first['max_rel'] or {}).get('value'))}")  # fmt: skip
    for name in ("main", "kstar"):
        found = record["decision"][name]
        print(f"decision {name} {found['code']} C {_n(found['C'])} A {_n(found['A'])} lower "
              f"{_n(found['lower'])} upper {_n(found['upper'])} refuted "
              f"{','.join(found['refuted'] or []) or 'none'}")  # fmt: skip
    print(f"record sha256 {digest}")


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--grid-sha256", required=True)
    parser.add_argument("--theta0-run", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--forecasts-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            refuse(f"{args.out.name} exists: never overwritten")
        record = record_of(args)
        data = cli.encoded(record)
        cli.write_once(args.out, data)
    except Refused as exc:
        sys.exit(f"{PROG}: {exc}")
    report(record, cli.sha256(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
