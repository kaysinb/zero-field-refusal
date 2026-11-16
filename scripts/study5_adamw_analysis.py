"""Study 5's order repeat under AdamW analysed on a CPU (docs/study5-adamw-registration.md): the A
runs and E2's F runs of the drawn seeds read with their WildGuard labels, every quantity computed by
``analysis.study5_adamw``, written once as one JSON record.

    python scripts/study5_adamw_analysis.py --runs ROOT [--runs ROOT ...]
        --seeds docs/study5-adamw-seeds.json --theta0-run DIR --out RECORD.json

**Inputs.** Each --runs ROOT is a runs root (A's restored Pod runs, the roots holding E2's runs),
named ``runs1``, ``runs2`` … in the order given; each must hold a ``study5/`` folder. The runs are
the run directories under ``ROOT/study5/`` whose ``config.yaml`` gives ``study5.role``
``order_a`` (arm A) or ``control`` (arm F when it is one of E2's, ``study5_adamw.select_runs``);
no other run is read further. --seeds is the committed seeds record, its seeds the draw's
(``adamw_orders.draw_seeds``). --theta0-run is study 5's θ₀ evaluation run
(``experiments/study5_theta0``, run type ``theta0``): its accuracies decide gross collapse and the
utility losses of both arms.

**The runs.** Each run's key (arm, seed, order, rate) is ``study5_adamw.select_runs``' from its
config (``study5.role``, ``seed``, ``optimizer.name`` and ``optimizer.lr``,
``train.order_seed``) and whether its ``terminal.json`` exists. Each keyed run is read as the
study-5 analysis CLI reads an F run (``study5_analysis_cli.f_run``: its config's seed and rate;
its stop, steps and step-25 evaluation), its optimizer the arm's (``adamw`` or ``adam_frozen``),
and its WildGuard labels at the step-25 condition as the dose analysis reads them
(``study5_analysis_cli.Labels`` through ``run_labels``: every check of study 2's reader, the
receipt's sha256 kept). A record or label file that does not read is the run's reason, named where
it leaves a quantity, never a refusal.

**The record** (--out, never overwritten, written whole by a link): every quantity of
``study5_adamw.analysis``; the inputs' sha256 (the seeds file, θ₀'s config and summary and every
run file read, by its root and relative path, and each prompt set's); the label receipts' sha256
by run; the selection (the keyed runs, those not ended, of other seeds, the control runs that are
not E2's, those whose config does not read); the git state and the UTC time. Strict JSON, sorted
keys, a non-finite number as ``{"missing": "nonfinite"}`` (``study5_analysis.finite_tree``). It
holds numbers, ids, states and reasons, never a text. stdout carries numbers, ids and hashes
only. Exits 0 when written.

**Refusals** exit 1 with ``study5_adamw_analysis: <why>`` and write nothing: an existing --out, a
runs root without a ``study5/`` folder, a seeds record that is not the draw, a θ₀ run that is none
or does not read, an A run of no order, a run whose optimizer is not its arm's, a run whose rate is
none of its arm's, and a key with two ended runs (naming them).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from zero_field_refusal import adamw_orders
from zero_field_refusal.analysis import study5_adamw as sad
from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis.study3_records import read_theta0
from zero_field_refusal.analysis.study5_analysis_cli import Refused, refuse
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

PROG = "study5_adamw_analysis"
KIND, VERSION = "study5-adamw-analysis", 1
EXPERIMENT = "study5"  # the runs' folder under a runs root
CONDITION = cli.ENDPOINT  # step-25
THETA0_RUN_TYPE = "theta0"


# ------------------------------------------------------------------ the inputs


def seeds_of(reader: cli.Reader, path: Path) -> list[int]:
    """The seeds record's seeds, refused unless they are the draw's."""
    try:
        found = json.loads(reader.bytes(path))
    except (OSError, ValueError) as exc:
        refuse(f"the seeds record does not read ({type(exc).__name__})")
    seeds = found.get("seeds") if isinstance(found, dict) else None
    if seeds != sorted(adamw_orders.draw_seeds()):
        refuse("the seeds record's seeds are not the draw's (adamw_orders.draw_seeds)")
    return list(seeds)


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


def order_dirs(reader: cli.Reader,
               roots: Mapping[str, Path]) -> tuple[list[dict[str, Any]], list[dict]]:  # fmt: skip
    """Every run directory under ``ROOT/study5/`` of each runs root whose config gives role
    ``order_a`` or ``control``, as ``study5_adamw.select_runs`` takes it (with its directory), its
    config hashed as an input; and the directories whose config does not read."""
    found, unreadable = [], []
    for name, root in roots.items():
        folder = Path(root) / EXPERIMENT
        for run_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            path = run_dir / "config.yaml"
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
                cfg = OmegaConf.create(data.decode())
                role = OmegaConf.select(cfg, "study5.role", default=None)
            except Exception as exc:  # any config that does not parse is listed, not read further
                unreadable.append({"root": name, "run_id": run_dir.name,
                                   "reason": type(exc).__name__})  # fmt: skip
                continue
            if role not in sad.ARM_OF_ROLE:
                continue
            reader.hashes[reader.name(path)] = cli.sha256(data)
            found.append({
                "run_id": run_dir.name, "root": name, "run_dir": run_dir, "role": role,
                "optimizer": OmegaConf.select(cfg, "optimizer.name", default=None),
                "seed": OmegaConf.select(cfg, "seed", default=None),
                "lr": OmegaConf.select(cfg, "optimizer.lr", default=None),
                "order_seed": OmegaConf.select(cfg, "train.order_seed", default=None),
                "ended": (run_dir / "terminal.json").is_file(),
            })  # fmt: skip
    return found, unreadable


def run_records(reader: cli.Reader, run_dir: Path, seed: int, lr: float,
                optimizer: str) -> dict[str, Any] | str:  # fmt: skip
    """A keyed run's records as ``study5_analysis_cli.f_run`` reads an F run (its stop, steps
    and step-25 evaluation row; its config the key's seed and rate), its optimizer ``optimizer``
    (the arm's); the reason when they do not read."""
    try:
        cfg = OmegaConf.create(reader.bytes(run_dir / "config.yaml").decode())
        if (int(cfg.seed) != seed or not math.isclose(float(cfg.optimizer.lr), lr, rel_tol=1e-9)
                or cfg.optimizer.name != optimizer):  # fmt: skip
            return f"{run_dir.name}: its config is not seed {seed}, rate {lr!r}, {optimizer}"
        terminal = reader.json(run_dir / "terminal.json")
        reader.json(run_dir / "meta.json")  # read, and hashed, as f_run reads it
        rows = (reader.jsonl(run_dir / "evaluations.jsonl")
                if (run_dir / "evaluations.jsonl").is_file() else [])  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return f"{run_dir.name}: its records do not read ({type(exc).__name__})"
    endpoint = [r for r in rows if isinstance(r, dict) and r.get("condition") == CONDITION]
    return {"stop": (terminal.get("stop") or {}).get("reason"), "steps": terminal.get("steps"),
            "row": endpoint[0] if len(endpoint) == 1 else None}  # fmt: skip


def read_run(reader: cli.Reader, labels: cli.Labels, entry: Mapping[str, Any],
             key: sad.Key) -> dict[str, Any]:  # fmt: skip
    """One keyed run's readings (``study5_adamw``'s ``Run``)."""
    arm, seed, order, rate = key
    run_dir = Path(entry["run_dir"])
    run: dict[str, Any] = {"run_id": run_dir.name, "root": entry["root"], "arm": arm,
                           "seed": seed, "order": order, "rate": rate, "records": None,
                           "stop": None, "steps": None, "row": None}  # fmt: skip
    found = run_records(reader, run_dir, seed, dict(sad.RATES[arm])[rate], sad.OPTIMIZERS[arm])
    if isinstance(found, str):
        run["records"] = found
    else:
        run |= found
    got = labels.of(run_dir)
    if isinstance(got, Mapping):
        run["labels"] = got["conditions"].get(CONDITION, f"labels: no condition {CONDITION}")
        run["receipt_sha256"] = got["receipt_sha256"]
    else:
        run["labels"], run["receipt_sha256"] = got, None
    return run


# ------------------------------------------------------------------ the record


def record_of(args: argparse.Namespace) -> dict[str, Any]:
    roots = {f"runs{i}": Path(root) for i, root in enumerate(args.runs, 1)}
    reader = cli.Reader({**roots, "theta0": args.theta0_run, "seeds": args.seeds.parent,
                         "repo": REPO_ROOT})  # fmt: skip
    for name, root in roots.items():
        if not (root / EXPERIMENT).is_dir():
            refuse(f"{name} holds no {EXPERIMENT}/ folder")
    seeds = seeds_of(reader, args.seeds)
    theta0 = theta0_of(reader, args.theta0_run)
    entries, unreadable = order_dirs(reader, roots)
    try:
        selection = sad.select_runs(entries, seeds)
    except ValueError as exc:
        refuse(str(exc))
    labels = cli.Labels(reader, cli.prompt_sets(reader))
    runs = {key: read_run(reader, labels, entry, key)
            for key, entry in selection["runs"].items()}  # fmt: skip
    result = sad.analysis(runs, seeds, theta0)
    return sa.finite_tree({
        "kind": KIND, "version": VERSION,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(REPO_ROOT),
        "inputs": {
            "seeds": {"file": reader.name(args.seeds),
                      "sha256": reader.hashes[reader.name(args.seeds)]},
            "roots": list(roots),
            "theta0": {"run_id": Path(args.theta0_run).resolve().name, "accuracies": theta0},
            "prompt_sets": dict(sorted(reader.prompt_sets.items())),
            "files": dict(sorted(reader.hashes.items())),
        },
        "label_receipts": {run["run_id"]: run["receipt_sha256"] for run in runs.values()},
        "selection": {
            "experiment": EXPERIMENT,
            "keyed": [{"arm": arm, "seed": s, "order": order, "rate": rate,
                       "run_id": entry["run_id"], "root": entry["root"]}
                      for (arm, s, order, rate), entry in selection["runs"].items()],
            "not_ended": selection["not_ended"], "other_seeds": selection["other_seeds"],
            "ignored": selection["ignored"], "unreadable": unreadable,
        },
        **result,
    })  # fmt: skip


def _n(value: Any) -> str:
    """A number as stdout carries it; ``none`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "none"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _code(words: str) -> str:
    return words.replace(" ", "_")


def report(record: Mapping[str, Any], digest: str) -> None:
    """Numbers, ids, reading codes and hashes only."""
    selection = record["selection"]
    print(f"runs {len(record['runs'])} missing {len(record['missing'])} not_ended "
          f"{len(selection['not_ended'])} other_seeds {len(selection['other_seeds'])} ignored "
          f"{len(selection['ignored'])} unreadable {len(selection['unreadable'])}")  # fmt: skip
    for key in record["missing"]:
        print(f"missing {key['arm']} s{key['seed']} {key['order']} {key['rate']}")
    kstar = record["kstar"]
    print(f"kstar unsupported {kstar['unsupported']['n']} without_labels "
          f"{kstar['without_labels']['n']}")  # fmt: skip
    parts = (("harmful", record["primary"]),
             ("harmful_high", record["secondary"]["harmful_high"]))  # fmt: skip
    for measure, part in parts:
        noise = part["label_noise"]
        print(f"{measure} label_noise runs {noise['runs']} h_bar {_n(noise['h_bar'])} value_pp "
              f"{_n(noise['value_pp'])}")  # fmt: skip
        bases = (("kstar", part["kstar"]), *part["fixed_rate"].items())
        for basis, found in bases:
            for name in ("abs_A", "abs_F", "paired"):
                m = found[name]
                print(f"{measure} {basis} {name} n {m['n']} mean {_n(m['mean'])} lower "
                      f"{_n(m['lower'])} upper {_n(m['upper'])} excluded "
                      f"{len(m['excluded'])}")  # fmt: skip
            words = found["reading"]
            print(f"{measure} {basis} reading A {_code(words['order_sensitivity'])} against_F "
                  f"{_code(words['against_F'])}")  # fmt: skip
            d = found["direction"]
            print(f"{measure} {basis} direction n {d['n']} agreements {d['agreements']} "
                  f"opposite {d['opposite']} with_zero {d['with_zero']} p "
                  f"{_n(d['p_one_sided'])}")  # fmt: skip
    found = record["secondary"]["brackets"]
    print(f"brackets pairs {found['pairs']} decided {found['decided']} bracketing "
          f"{found['bracketing']} share {_n(found['share'])}")  # fmt: skip
    print(f"record sha256 {digest}")


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--theta0-run", type=Path, required=True)
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
