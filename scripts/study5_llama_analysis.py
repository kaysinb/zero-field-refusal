"""Study 5's order repeat on Llama-3.2-1B-Instruct analysed on a CPU
(docs/study5-llama-registration.md, "Analysis"): its runs and θ₀ read with their WildGuard labels,
every quantity computed by ``analysis.study5_llama``, written once as one JSON record.

    python scripts/study5_llama_analysis.py --runs ROOT [--runs ROOT ...] --rates FILE
        --rates-sha256 HEX --theta0-run DIR [--qwen-adamw RECORD] --out RECORD.json

**Inputs.** Each --runs ROOT is a runs root (the Llama Pod's restored runs), named ``runs1``,
``runs2`` … in the order given; each must hold a ``study5_llama/`` folder. --rates is the rates
record the Pod's driver writes (``scripts/study5_llama_campaigns.py rates``), read only when the
whole file's sha256 is --rates-sha256; its choice must be the rule's on its pilot rows
(``study5_llama.rates_of``: ``llama_orders.rate_choice``). --theta0-run is Llama's θ₀ evaluation
run (``experiments/study5_llama_theta0``, run type ``theta0``, model Llama's): its accuracies decide
gross collapse and the utility losses, and its WildGuard labels at its ``theta0`` condition give
θ₀'s own h. --qwen-adamw, optional, is the AdamW repeat's analysis record
(``scripts/study5_adamw_analysis.py``): its primary means of |Δ*_A| and |Δ*_F| at K* with their
intervals are copied beside, with the file's sha256, without a test across models.

**The runs.** Every run directory under ``ROOT/study5_llama/`` with a ``config.yaml``, its config
hashed as an input: its key (seed, order, rate) is ``study5_llama.select_runs``' from its config
(``run_type``, ``eval.k_only``, ``study5.role``, ``seed``, ``optimizer.name`` and
``optimizer.lr``, ``model.name``, ``train.order_seed``) and whether its ``terminal.json`` exists;
a K-only pilot run, θ₀'s evaluation run and a run of another role are listed, not read further.
Each keyed run is read as the AdamW analysis reads an A run (its config the key's seed, rate and
``adamw``; its stop, steps and step-25 evaluation), and its WildGuard labels at the step-25
condition as study 5's analysis CLI reads them (``study5_analysis_cli.Labels`` through
``run_labels``: every check of study 2's reader, the receipt's sha256 kept). θ₀'s labels are read
by the same reader at its ``theta0`` condition (``study5_analysis_cli.THETA0``), as study 5's
analysis reads θ₀'s. A record or label file that does not read is the run's (or θ₀'s) reason,
named where it leaves a quantity, never a refusal.

**The record** (--out, never overwritten, written whole by a link): every quantity of
``study5_llama.analysis``; the inputs' sha256 (the rates file, θ₀'s config and summary, the Qwen
record, every run file read, by its root and relative path, and each prompt set's); the label
receipts' sha256 by run, θ₀'s among them; the selection (the keyed runs, those not ended, of other
seeds, the pilot's K-only runs, the runs ignored, those whose config does not read); the git state
and the UTC time. Strict JSON, sorted keys, a non-finite number as ``{"missing": "nonfinite"}``
(``study5_analysis.finite_tree``). It holds numbers, ids, states and reasons, never a text. stdout
carries numbers, ids, reading codes and hashes only. Exits 0 when written.

**Refusals** exit 1 with ``study5_llama_analysis: <why>`` and write nothing: an existing --out, a
runs root without a ``study5_llama/`` folder, a rates file whose sha256 is not the one given or
that does not read, a rates record whose choice is not the rule's on its rows, a θ₀ run that is
none, not Llama's or does not read, a Qwen record that does not read or is not the AdamW
analysis's, a run of no order, a run whose optimizer is not ``adamw`` or whose model is not
Llama's, an ended run whose rate is neither η_l nor η_u, and a key with two ended runs (naming
them).
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

from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_llama as sll
from zero_field_refusal.analysis.study3_records import read_theta0
from zero_field_refusal.analysis.study5_analysis_cli import Refused, refuse
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

PROG = "study5_llama_analysis"
KIND, VERSION = "study5-llama-analysis", 1
EXPERIMENT = "study5_llama"  # the runs' folder under a runs root (experiments/study5_llama)
CONDITION = cli.ENDPOINT  # step-25
THETA0_RUN_TYPE = cli.THETA0  # θ₀'s run type, and the condition its responses answer under


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


def rates_of(reader: cli.Reader, path: Path, digest: str) -> dict[str, Any]:
    """The rates record's file, sha256 (``digest``), choice and two rates
    (``study5_llama.rates_of``), refused unless its choice is the rule's on its rows."""
    data = checked(reader, path, digest, "the rates record")
    record = json_of(data, "the rates record")
    try:
        rates = sll.rates_of(record)
    except ValueError as exc:
        refuse(str(exc))
    return {"file": reader.name(path), "sha256": cli.sha256(data), "rates": rates,
            "choice": record["choice"]}  # fmt: skip


def theta0_of(reader: cli.Reader, run_dir: Path) -> dict[str, float]:
    """θ₀'s accuracies (``study3_records.read_theta0``) from its evaluation run, its run type
    ``theta0`` and its model Llama's; refused otherwise."""
    try:
        cfg = OmegaConf.create(reader.bytes(run_dir / "config.yaml").decode())
        run_type = cfg.get("run_type") if isinstance(cfg, DictConfig) else None
        model = (OmegaConf.select(cfg, "model.name", default=None)
                 if isinstance(cfg, DictConfig) else None)  # fmt: skip
        reader.bytes(run_dir / "summary.json")  # hashed as an input; read_theta0 reads it
        accuracies = read_theta0(run_dir)
    except (OSError, ValueError) as exc:
        refuse(f"θ₀'s evaluation does not read: {reader.scrub(str(exc))}")
    if run_type != THETA0_RUN_TYPE:
        refuse(f"{run_dir.name} is not a θ₀ evaluation run (run type {run_type!r})")
    if model != sll.MODEL:
        refuse(f"{run_dir.name} is not Llama's θ₀ evaluation (model {model!r})")
    return accuracies


def qwen_of(reader: cli.Reader, path: Path) -> dict[str, Any]:
    """The AdamW repeat's analysis record's means copied (``study5_llama.qwen_beside``), with
    the file's name and sha256; refused when it does not read or is not one."""
    try:
        data = reader.bytes(path)
    except OSError as exc:
        refuse(f"the Qwen record does not read ({type(exc).__name__})")
    try:
        found = sll.qwen_beside(json_of(data, "the Qwen record"))
    except ValueError as exc:
        refuse(str(exc))
    return {"file": reader.name(path), "sha256": cli.sha256(data), **found}


# ------------------------------------------------------------------ the runs


def llama_dirs(reader: cli.Reader,
               roots: Mapping[str, Path]) -> tuple[list[dict[str, Any]], list[dict]]:  # fmt: skip
    """Every run directory under ``ROOT/study5_llama/`` of each runs root, as
    ``study5_llama.select_runs`` takes it (with its directory), its config hashed as an input;
    and the directories whose config does not read."""
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
                if not isinstance(cfg, DictConfig):
                    raise ValueError("not a mapping")
            except Exception as exc:  # any config that does not parse is listed, not read further
                unreadable.append({"root": name, "run_id": run_dir.name,
                                   "reason": type(exc).__name__})  # fmt: skip
                continue
            reader.hashes[reader.name(path)] = cli.sha256(data)

            def get(key: str, default: Any = None, cfg: DictConfig = cfg) -> Any:
                return OmegaConf.select(cfg, key, default=default)

            found.append({
                "run_id": run_dir.name, "root": name, "run_dir": run_dir,
                "run_type": get("run_type", "train"), "k_only": get("eval.k_only", False),
                "role": get("study5.role"), "optimizer": get("optimizer.name"),
                "model": get("model.name"), "seed": get("seed"), "lr": get("optimizer.lr"),
                "order_seed": get("train.order_seed"),
                "ended": (run_dir / "terminal.json").is_file(),
            })  # fmt: skip
    return found, unreadable


def run_records(reader: cli.Reader, run_dir: Path, seed: int,
                lr: float) -> dict[str, Any] | str:  # fmt: skip
    """A keyed run's records as the AdamW analysis reads an A run (its stop, steps and step-25
    evaluation row; its config the key's seed, rate and ``adamw``); the reason when they do not
    read."""
    try:
        cfg = OmegaConf.create(reader.bytes(run_dir / "config.yaml").decode())
        if (int(cfg.seed) != seed or not math.isclose(float(cfg.optimizer.lr), lr, rel_tol=1e-9)
                or cfg.optimizer.name != sll.OPTIMIZER):  # fmt: skip
            return f"{run_dir.name}: its config is not seed {seed}, rate {lr!r}, {sll.OPTIMIZER}"
        terminal = reader.json(run_dir / "terminal.json")
        reader.json(run_dir / "meta.json")  # read, and hashed, as the AdamW analysis reads it
        rows = (reader.jsonl(run_dir / "evaluations.jsonl")
                if (run_dir / "evaluations.jsonl").is_file() else [])  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return f"{run_dir.name}: its records do not read ({type(exc).__name__})"
    endpoint = [r for r in rows if isinstance(r, dict) and r.get("condition") == CONDITION]
    return {"stop": (terminal.get("stop") or {}).get("reason"), "steps": terminal.get("steps"),
            "row": endpoint[0] if len(endpoint) == 1 else None}  # fmt: skip


def read_run(reader: cli.Reader, labels: cli.Labels, entry: Mapping[str, Any], key: sll.Key,
             rates: sll.Rates) -> dict[str, Any]:  # fmt: skip
    """One keyed run's readings (``study5_llama``'s ``Run``)."""
    seed, order, rate = key
    run_dir = Path(entry["run_dir"])
    run: dict[str, Any] = {"run_id": run_dir.name, "root": entry["root"], "seed": seed,
                           "order": order, "rate": rate, "records": None, "stop": None,
                           "steps": None, "row": None}  # fmt: skip
    found = run_records(reader, run_dir, seed, dict(rates)[rate])
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
    named = {**roots, "theta0": args.theta0_run, "rates": args.rates.parent, "repo": REPO_ROOT}
    if args.qwen_adamw is not None:
        named["qwen"] = args.qwen_adamw.parent
    reader = cli.Reader(named)
    for name, root in roots.items():
        if not (root / EXPERIMENT).is_dir():
            refuse(f"{name} holds no {EXPERIMENT}/ folder")
    rates_input = rates_of(reader, args.rates, args.rates_sha256)
    rates = rates_input["rates"]
    theta0 = theta0_of(reader, args.theta0_run)
    qwen = qwen_of(reader, args.qwen_adamw) if args.qwen_adamw is not None else None
    entries, unreadable = llama_dirs(reader, roots)
    try:
        selection = sll.select_runs(entries, rates)
    except ValueError as exc:
        refuse(str(exc))
    labels = cli.Labels(reader, cli.prompt_sets(reader))
    runs = {key: read_run(reader, labels, entry, key, rates)
            for key, entry in selection["runs"].items()}  # fmt: skip
    theta0_dir = Path(args.theta0_run)
    theta0_labels = labels.of(theta0_dir)
    theta0_counts = labels.at(theta0_dir, cli.THETA0)
    theta0_receipt = (theta0_labels["receipt_sha256"] if isinstance(theta0_labels, Mapping)
                      else None)  # fmt: skip
    theta0_id = theta0_dir.resolve().name
    result = sll.analysis(runs, rates, theta0, theta0_counts)
    return sa.finite_tree({
        "kind": KIND, "version": VERSION,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(REPO_ROOT),
        "inputs": {
            "rates": {**rates_input, "rates": dict(rates)},
            "roots": list(roots),
            "theta0": {"run_id": theta0_id, "accuracies": theta0,
                       "receipt_sha256": theta0_receipt},
            "qwen_adamw": None if qwen is None else {"file": qwen["file"],
                                                     "sha256": qwen["sha256"]},
            "prompt_sets": dict(sorted(reader.prompt_sets.items())),
            "files": dict(sorted(reader.hashes.items())),
        },
        "label_receipts": {**{run["run_id"]: run["receipt_sha256"] for run in runs.values()},
                           theta0_id: theta0_receipt},
        "selection": {
            "experiment": EXPERIMENT,
            "keyed": [{"seed": s, "order": order, "rate": rate, "run_id": entry["run_id"],
                       "root": entry["root"]}
                      for (s, order, rate), entry in selection["runs"].items()],
            "not_ended": selection["not_ended"], "other_seeds": selection["other_seeds"],
            "pilot": selection["pilot"], "ignored": selection["ignored"],
            "unreadable": unreadable,
        },
        "beside": {"qwen_adamw": qwen},
        **result,
    })  # fmt: skip


def _n(value: Any) -> str:
    """A number as stdout carries it; ``none`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "none"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _code(words: str) -> str:
    return words.replace(" ", "_")


def _mean(prefix: str, m: Mapping[str, Any]) -> str:
    return (f"{prefix} n {m['n']} mean {_n(m['mean'])} lower {_n(m['lower'])} upper "
            f"{_n(m['upper'])} excluded {len(m['excluded'])}")  # fmt: skip


def report(record: Mapping[str, Any], digest: str) -> None:
    """Numbers, ids, reading codes and hashes only."""
    selection = record["selection"]
    print(f"runs {len(record['runs'])} missing {len(record['missing'])} not_ended "
          f"{len(selection['not_ended'])} other_seeds {len(selection['other_seeds'])} pilot "
          f"{len(selection['pilot'])} ignored {len(selection['ignored'])} unreadable "
          f"{len(selection['unreadable'])}")  # fmt: skip
    for key in record["missing"]:
        print(f"missing s{key['seed']} {key['order']} {key['rate']}")
    rates = record["rates"]
    print(f"rates eta_l {_n(rates['eta_l'])} eta_u {_n(rates['eta_u'])} sha256 "
          f"{record['inputs']['rates']['sha256']}")  # fmt: skip
    theta0 = record["theta0"]
    print(f"theta0 {record['inputs']['theta0']['run_id']} h {_n(theta0['h'])} harmful_high "
          f"{_n(theta0['harmful_high'])} mmlu {_n(theta0['accuracies']['benchmark_accuracy'])} "
          f"factual {_n(theta0['accuracies']['factual_accuracy'])}")  # fmt: skip
    kstar = record["kstar"]
    print(f"kstar unsupported {kstar['unsupported']['n']} without_labels "
          f"{kstar['without_labels']['n']}")  # fmt: skip
    parts = (("harmful", record["primary"]),
             ("harmful_high", record["secondary"]["harmful_high"]))  # fmt: skip
    for measure, part in parts:
        noise = part["label_noise"]
        print(f"{measure} label_noise runs {noise['runs']} h_bar {_n(noise['h_bar'])} value_pp "
              f"{_n(noise['value_pp'])}")  # fmt: skip
        eroded = part["erosion"]
        print(_mean(f"{measure} erosion", eroded["mean"]) +
              f" reading {_code(eroded['reading'])}")  # fmt: skip
        for basis, found in (("kstar", part["kstar"]), *part["fixed_rate"].items()):
            print(_mean(f"{measure} {basis} mean_abs", found["mean_abs"]) +
                  f" reading {_code(found['reading']['order_sensitivity'])}")  # fmt: skip
    found = record["secondary"]["brackets"]
    print(f"brackets pairs {found['pairs']} decided {found['decided']} bracketing "
          f"{found['bracketing']} share {_n(found['share'])}")  # fmt: skip
    losses = record["secondary"]["utility"]["both"]
    mmlu, factual = losses["mmlu_loss_pp"], losses["factual_loss_pp"]
    print(f"utility mmlu_loss_pp n {mmlu['n']} mean {_n(mmlu['mean'])} factual_loss_pp n "
          f"{factual['n']} mean {_n(factual['mean'])}")  # fmt: skip
    qwen = record["beside"]["qwen_adamw"]
    if qwen is not None:
        for name in sll.QWEN_MEANS:
            m = qwen[name]
            print(f"qwen_adamw {name} n {_n(m['n'])} mean {_n(m['mean'])} lower "
                  f"{_n(m['lower'])} upper {_n(m['upper'])}")  # fmt: skip
        print(f"qwen_adamw sha256 {qwen['sha256']}")
    print(f"record sha256 {digest}")


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--rates", type=Path, required=True)
    parser.add_argument("--rates-sha256", required=True)
    parser.add_argument("--theta0-run", type=Path, required=True)
    parser.add_argument("--qwen-adamw", type=Path)
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
