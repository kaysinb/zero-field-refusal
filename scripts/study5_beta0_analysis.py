"""Study 5's run without momentum analysed on a CPU (docs/study5-beta0-registration.md): the runs
of role ``beta0`` read with their WildGuard labels, every quantity computed by
``analysis.study5_beta0``, written once as one JSON record.

    python scripts/study5_beta0_analysis.py --runs ROOT [--runs ROOT ...] --rate RATE.json
        --reference docs/study5-beta0-reference.json --out RECORD.json

**Inputs.** Each --runs ROOT is a runs root holding a ``study5/`` folder (the restored Pod runs).
The runs are the run directories under ``ROOT/study5/`` whose ``config.yaml`` gives
``study5.role`` ``beta0`` and no K-only evaluation (the pilot's runs are listed, never keyed).
--rate is the main plan's rate record (``scripts/study5_beta0_campaigns.py rate``): its η₀ is
recomputed from the pilot table it holds (``beta0.rate_choice``) and must agree. --reference is
the committed frozen-Adam reference (``beta0.REFERENCE_FILE``): its F, E, L, D and h̄ are
recomputed from its per-edit values (``beta0.summary_of``) and must agree.

**The runs.** A run's key (seed, order) comes from its config: the seed; the order ``stream``
without an order key, ``reorder`` when ``train.order_seed`` is the seed, and ``early-k`` or
``late-k`` for a shuffle of batches 1–5 or 21–25 with draw k. Its rate must be η₀ and its
``optimizer.beta1`` 0. A run of another seed, order or rate is listed and enters nothing; two
ended runs of one key are refused. Each run is read as the AdamW analysis reads one (its stop,
steps and step-25 evaluation; its labels at the step-25 condition through
``study5_analysis_cli.Labels``, the receipt's sha256 kept). A record or label file that does not
read is the run's reason, never a refusal.

**The record** (--out, never overwritten): every quantity of ``study5_beta0.analysis``; the inputs'
sha256; the label receipts by run; the selection; the git state and the UTC time. It holds
numbers, ids, states and reasons, never a text; stdout carries numbers, ids and hashes only.
Exits 0 when written; a refusal exits 1 with ``study5_beta0_analysis: <why>`` and writes nothing.
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

from omegaconf import OmegaConf

from zero_field_refusal import beta0
from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_beta0 as sb
from zero_field_refusal.analysis.study5_analysis_cli import Refused, refuse
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

PROG = "study5_beta0_analysis"
KIND, VERSION = "study5-beta0-analysis", 1
EXPERIMENT = "study5"
CONDITION = cli.ENDPOINT
ROLE = "beta0"


def order_name(seed: Any, order_seed: Any, edit: Any) -> str | None:
    """The order of a run's config (module docstring, "The runs"); None for any other."""
    if order_seed is None and edit is None:
        return "stream"
    if edit is None:
        return "reorder" if order_seed == seed else None
    if order_seed is not None or not isinstance(edit, Mapping) or edit.get("kind") != "shuffle":
        return None
    for name, first, last in beta0.WINDOWS:
        drawn = edit.get("draw") in beta0.DRAWS
        if edit.get("first") == first and edit.get("last") == last and drawn:
            return f"{name}-{edit['draw']}"
    return None


def rate_of(reader: cli.Reader, path: Path) -> float:
    """η₀ of the rate record, refused unless the rule gives it from the pilot table it holds."""
    try:
        record = json.loads(reader.bytes(path))
        targets = beta0.targets_from(json.loads(reader.bytes(REPO_ROOT / beta0.TARGETS_FILE)))
        choice = beta0.rate_choice(record["table"]["rows"], targets)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        refuse(f"the rate record does not read ({type(exc).__name__})")
    if choice != record.get("choice") or choice["eta0"] is None:
        refuse("the rate record's η₀ is not the rule's on its pilot table")
    return float(choice["eta0"])


def reference_of(reader: cli.Reader, path: Path) -> dict[str, Any]:
    try:
        record = json.loads(reader.bytes(path))
        summary = beta0.summary_of(record["seeds"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        refuse(f"the reference does not read ({type(exc).__name__})")
    if any(not math.isclose(summary[k], record[k], rel_tol=1e-12) for k in summary):
        refuse("the reference's means are not those of its per-edit values")
    return record


def beta0_dirs(reader: cli.Reader,
               roots: Mapping[str, Path]) -> tuple[list[dict], list[dict], list[str]]:  # fmt: skip
    """The runs of role ``beta0`` without a K-only evaluation; the configs that do not read; the
    K-only (pilot) runs, listed and never keyed: a pilot run at a grid rate equal to η₀ is not a
    main run."""
    found, unreadable, pilot = [], [], []
    for name, root in roots.items():
        for run_dir in sorted(p for p in (Path(root) / EXPERIMENT).iterdir() if p.is_dir()):
            path = run_dir / "config.yaml"
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
                cfg = OmegaConf.create(data.decode())
                role = OmegaConf.select(cfg, "study5.role", default=None)
            except Exception as exc:  # a config that does not parse is listed, not read further
                unreadable.append({"root": name, "run_id": run_dir.name,
                                   "reason": type(exc).__name__})  # fmt: skip
                continue
            if role != ROLE:
                continue
            if OmegaConf.select(cfg, "eval.k_only", default=False) is True:
                pilot.append(run_dir.name)
                continue
            reader.hashes[reader.name(path)] = cli.sha256(data)
            edit = OmegaConf.select(cfg, "train.order_edit", default=None)
            edit = OmegaConf.to_container(edit, resolve=True) if edit is not None else None
            seed = OmegaConf.select(cfg, "seed", default=None)
            found.append({
                "run_id": run_dir.name, "root": name, "run_dir": run_dir, "seed": seed,
                "order": order_name(seed, OmegaConf.select(cfg, "train.order_seed", default=None),
                                    edit),
                "lr": OmegaConf.select(cfg, "optimizer.lr", default=None),
                "beta1": OmegaConf.select(cfg, "optimizer.beta1", default=None),
                "ended": (run_dir / "terminal.json").is_file(),
            })  # fmt: skip
    return found, unreadable, pilot


def select(entries: list[dict], eta0: float) -> dict[str, Any]:
    keys = set(sb.design())
    runs: dict[sb.Key, dict] = {}
    other, not_ended = [], []
    for entry in entries:
        key = (entry["seed"], entry["order"])
        lr = entry["lr"]
        if (key not in keys or entry["beta1"] != 0 or not isinstance(lr, int | float)
                or not math.isclose(float(lr), eta0, rel_tol=1e-9)):  # fmt: skip
            other.append({"run_id": entry["run_id"], "seed": entry["seed"],
                          "order": entry["order"], "lr": lr})  # fmt: skip
            continue
        if not entry["ended"]:
            not_ended.append(entry["run_id"])
            continue
        if key in runs:
            refuse(f"two ended runs of s{key[0]}/{key[1]}: {runs[key]['run_id']}, "
                   f"{entry['run_id']}")  # fmt: skip
        runs[key] = entry
    return {"runs": runs, "other": other, "not_ended": not_ended}


def read_run(reader: cli.Reader, labels: cli.Labels, entry: Mapping[str, Any],
             key: sb.Key) -> dict[str, Any]:  # fmt: skip
    seed, order = key
    run_dir = Path(entry["run_dir"])
    run: dict[str, Any] = {"run_id": run_dir.name, "root": entry["root"], "seed": seed,
                           "order": order, "records": None, "stop": None, "steps": None,
                           "row": None}  # fmt: skip
    try:
        terminal = reader.json(run_dir / "terminal.json")
        reader.json(run_dir / "meta.json")
        rows = (reader.jsonl(run_dir / "evaluations.jsonl")
                if (run_dir / "evaluations.jsonl").is_file() else [])  # fmt: skip
        endpoint = [r for r in rows if isinstance(r, dict) and r.get("condition") == CONDITION]
        run |= {"stop": (terminal.get("stop") or {}).get("reason"), "steps": terminal.get("steps"),
                "row": endpoint[0] if len(endpoint) == 1 else None}  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        run["records"] = f"{run_dir.name}: its records do not read ({type(exc).__name__})"
    got = labels.of(run_dir)
    if isinstance(got, Mapping):
        run["labels"] = got["conditions"].get(CONDITION, f"labels: no condition {CONDITION}")
        run["receipt_sha256"] = got["receipt_sha256"]
    else:
        run["labels"], run["receipt_sha256"] = got, None
    return run


def record_of(args: argparse.Namespace) -> dict[str, Any]:
    roots = {f"runs{i}": Path(root) for i, root in enumerate(args.runs, 1)}
    reader = cli.Reader({**roots, "rate": args.rate.parent, "reference": args.reference.parent,
                         "repo": REPO_ROOT})  # fmt: skip
    for name, root in roots.items():
        if not (root / EXPERIMENT).is_dir():
            refuse(f"{name} holds no {EXPERIMENT}/ folder")
    eta0 = rate_of(reader, args.rate)
    reference = reference_of(reader, args.reference)
    entries, unreadable, pilot = beta0_dirs(reader, roots)
    selection = select(entries, eta0)
    labels = cli.Labels(reader, cli.prompt_sets(reader))
    runs = {key: read_run(reader, labels, entry, key) for key, entry in selection["runs"].items()}
    result = sb.analysis(runs, reference, eta0)
    return sa.finite_tree({
        "kind": KIND, "version": VERSION,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(REPO_ROOT),
        "inputs": {"roots": list(roots), "prompt_sets": dict(sorted(reader.prompt_sets.items())),
                   "files": dict(sorted(reader.hashes.items()))},
        "label_receipts": {run["run_id"]: run["receipt_sha256"] for run in runs.values()},
        "selection": {"keyed": {f"s{s}/{o}": e["run_id"]
                                for (s, o), e in selection["runs"].items()},
                      "other": selection["other"], "not_ended": selection["not_ended"],
                      "unreadable": unreadable, "pilot": pilot},
        **result,
    })  # fmt: skip


def _n(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "none"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def report(record: Mapping[str, Any], digest: str) -> None:
    print(
        f"eta0 {_n(record['eta0'])} keyed {len(record['selection']['keyed'])} of "
        f"{len(record['design'])} missing {len(record['missing'])} other "
        f"{len(record['selection']['other'])} not_ended {len(record['selection']['not_ended'])}"
    )
    print(f"h_bar0 {_n(record['h_bar0'])} label_noise {_n(record['label_noise'])}")
    for name in ("F", "E", "L", "D"):
        iv = record["intervals"][name]
        print(f"{name} {_n(record['statistics'][name])} lower {_n(iv['lower'])} upper "
              f"{_n(iv['upper'])} reference {_n(record['reference'][name])}")  # fmt: skip
    r = record["result"]
    print(
        f"outcome {r['outcome']} strength {r.get('strength')} threshold_F "
        f"{_n(r.get('threshold_F'))} two_N {_n(r.get('two_N'))} scaled {r.get('scaled')}"
    )
    seeds = r.get("seeds_per_class") or {}
    print(f"conditional {r.get('conditional')} technical {len(r.get('technical') or [])} stopped "
          f"{len(r.get('stopped') or [])} seeds F {seeds.get('F')} E {seeds.get('E')} "
          f"L {seeds.get('L')} pilot_runs {len(record['selection']['pilot'])}")  # fmt: skip
    print(f"record sha256 {digest}")


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--rate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
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
