"""Print the state of study 2's short-regime stage (amendment 3, design A) and, once its
confirmation blocks are labeled, its looks.

    python scripts/study2_short_summary.py --runs DIR [DIR ...] --theta0 RUN_DIR \
        (--w-max W | --fixation FILE) [--runs-root RUNS_DIR] [--judge NAME] [--out OUT.json]

--runs holds every run so far, Stage A's included (as for scripts/study2_short_campaigns.py),
--theta0 the θ₀ run, the utility baseline, and --runs-root RUNS_DIR, whose launch/ journal tells
which runs the launcher gave up on (technical failures of their blocks; without it such a block
waits). The status is the stage's state machine (analysis.study2_short.progress), on K, numerics
and utility only: the candidates (B, U) with their costs, every attempt and round, the frozen
rates and FA controls, the validation verdict, the support of the confirmation blocks with the
counts of the confirmation loop (the first block without runs, the supported blocks complete in
order, those that may still be supported), the problems of the records, and the next campaign.

--judge NAME (configs/judge/NAME.yaml) adds the looks, on the label counts of the bracket runs
of every complete confirmation block (analysis.study2_labels.read_endpoint_labels; a run without
labels yet keeps its block waiting): the two sequential looks with the finite-cohort bounds, the
same with every missing label counted as harmful, the missing labels per arm, the secondary
contrasts at the stopping look and the utility family over the blocks started. The looks need
the simulation's fixation record (--fixation: w_max, the look levels, the SE floor, the bias
allowance, the interval with its bootstrap resamples and seed, the look sizes and cap, and the
equivalence and direction minima; analysis.study2_short.read_fixation) and are refused without
one; none of its values is ever defaulted. The looks print the interval, resamples and seed they
used, and each bootstrap look its stream's seed and dropped resamples. Only counts, K, rates and
intervals are printed, never a prompt, a response or a judge output. OUT.json, which must not
exist, gets the whole record with the command and the code revision.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.stage_a import load_trajectories, theta0_problems
from zero_field_refusal.analysis.study2_labels import read_endpoint_labels
from zero_field_refusal.analysis.study2_short import (
    ARMS,
    CAP,
    FA_CONTROL_ARM,
    LOOK_SIZES,
    PRIMARY_ARMS,
    confirmation_guard,
    exhausted_runs,
    look_blocks,
    looks,
    progress,
    read_fixation,
    run_facts,
    secondary_contrasts,
    utility_family,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.data import load_prompt_set
from zero_field_refusal.judge import judge_identity
from zero_field_refusal.launch import launch_records
from zero_field_refusal.tracking import git_state

# The fixation record's values that ``looks`` takes (its look sizes and cap equal the module's).
LOOK_SETTINGS = ("interval", "se_floor", "bias_allowance", "equivalence_min_supported",
                 "direction_min_supported", "bootstrap_resamples", "bootstrap_seed")  # fmt: skip


def status(state: dict[str, Any]) -> dict[str, Any]:
    """The state machine's status, briefly."""
    step = state["next"]
    out: dict[str, Any] = {"next": {k: v for k, v in step.items() if k != "runs"}}
    if "runs" in step:
        out["next"]["runs"] = [f"{r['arm']} B={r['batch']} lr={r['lr']:g} seed={r['seed']}"
                               for r in step["runs"]]  # fmt: skip
    out["problems"] = state.get("problems", [])
    chosen = state.get("selection")
    if chosen:
        out["seconds_per_chunk"] = chosen["seconds_per_chunk"]
        out["candidates"] = [{k: c[k] for k in ("batch", "U", "seconds", "nearest", "reasons")}
                             for c in chosen.get("candidates", [])]  # fmt: skip
        out["attempts"] = [_attempt(a) for a in chosen.get("attempts", [])]
        out["rounds"] = {key: {kind: {"new_rates": r["new_rates"], "proposal": _proposal(r)}
                               for kind, r in kinds.items()}
                         for key, kinds in chosen.get("rounds", {}).items()}  # fmt: skip
        if "prewarmed" in chosen:
            pre = chosen["prewarmed"]
            out["prewarmed"] = {
                "initial": pre["initial"],
                "outcome": pre["outcome"],
                "rates": pre["rates"],
                "selections": [_selection(s) for s in pre["selections"]],
            }
    if "frozen" in state:
        frozen = state["frozen"]
        out["frozen"] = {**{k: v for k, v in frozen.items() if k != "fa_controls"},
                         "fa_controls": {lr: {k: v[k] for k in ("run", "contamination", "bound",
                                                                "accepted")}
                                         for lr, v in frozen["fa_controls"].items()}}  # fmt: skip
    if "validation" in state:
        verdict = state["validation"]
        out["validation"] = {
            "passes": verdict["passes"],
            "reasons": verdict["reasons"],
            "jointly_supported": len(verdict.get("jointly_supported", [])),
            "causes": verdict.get("causes"),
            "unstable": verdict.get("unstable"),
            "utility": _utility(verdict.get("utility")),
            "prewarmed_utility": state.get("validation_prewarmed_utility"),
        }
    if "confirmation" in state:
        blocks = state["confirmation"]
        complete = [b for b in blocks if b["complete"]]
        lost = {arm: [b["supports"][arm] for b in complete if arm in b["supports"]
                      and not b["supports"][arm]["supported"]] for arm in ARMS}  # fmt: skip
        causes = {arm: dict(Counter(s["cause"] for s in found)) for arm, found in lost.items()}
        # The confirmation loop's counts (K only): where the next range starts, and how many
        # supported blocks the next look may already have.
        counts = confirmation_guard(blocks, 1, 1)
        out["confirmation"] = {
            "blocks_with_runs": sum(1 for b in blocks if b["runs"]),
            "first_block_without_runs": counts["first_open"],
            "complete": len(complete),
            "complete_in_order": next((b["block"] - 1 for b in blocks if not b["complete"]),
                                      len(blocks)),
            "supported": sum(1 for b in complete if b["supported"]),
            "supported_in_order": counts["supported_in_order"],
            "possibly_supported": counts["possibly_supported"],
            "look_sizes": list(LOOK_SIZES),
            "unsupported_by_cause": causes,
            "given_up_by_the_launcher": state.get("exhausted", {}),
        }  # fmt: skip
    return out


def _selection(s: dict[str, Any]) -> dict[str, Any]:
    best = s.get("best_unconstrained")
    return {"rates": s["rates"], "feasible": s["feasible"], "unmet": s["unmet"],
            "usable": s["usable"], "excluded": s["excluded"],
            "deviations": (s["selected"] or best or {}).get("deviations"),
            "best_unconstrained": best["rates"] if best else None}  # fmt: skip


def _attempt(a: dict[str, Any]) -> dict[str, Any]:
    return {"batch": a["batch"], "U": a["U"], "outcome": a["outcome"], "reasons": a["reasons"],
            "fa_excluded": a["fa_excluded"],
            "arms": {arm: [_selection(s) for s in d["selections"]]
                     for arm, d in a["arms"].items()}}  # fmt: skip


def _proposal(r: dict[str, Any]) -> Any:
    proposal = r["proposal"]
    if r["kind"] == "targeted":
        return proposal["targets"] + [{"skipped": s} for s in proposal["skipped"]]
    return proposal["reasons"]


def _utility(family: dict[str, Any] | None) -> dict[str, Any] | None:
    if family is None:
        return None
    keys = ("arm", "rate", "runs", "mmlu_upper_pp", "factual_upper_pp")
    return {"passes": family["passes"], "alpha_per_bound": family["alpha_per_bound"],
            "family": [{k: e[k] for k in keys} for e in family["family"]]}  # fmt: skip


def read_labels(state, runs, judge: str) -> dict[str, list[dict[str, Any]]]:
    """The label rows of the bracket runs of every arm supported in a complete confirmation
    block, by run id; a run without a label file is not labeled yet and left out."""
    by_id = {run.run_id: run for run in runs}
    identity = judge_identity(load_config(f"judge/{judge}"))
    prompts: dict[str, dict[str, str]] = {}
    labels: dict[str, list[dict[str, Any]]] = {}
    for block in state.get("confirmation", []):
        for support in block["supports"].values():
            if not support["supported"]:
                continue
            for side in ("lo", "hi"):
                run = by_id[support["bracket"][side]["run"]]
                if run.run_id in labels or not label_path(run.run_dir, judge).is_file():
                    continue
                cfg = OmegaConf.load(run.run_dir / "config.yaml")
                name = str(cfg.eval.harmful.prompt_set)
                if name not in prompts:
                    prompts[name] = {item["id"]: item["text"]
                                     for item in load_prompt_set(name, cfg.data)}  # fmt: skip
                labels[run.run_id] = read_endpoint_labels(run.run_dir, judge, identity,
                                                          prompts[name])  # fmt: skip
    return labels


def compute_looks(state, runs, labels, fixation, theta0) -> dict[str, Any]:
    """The looks on the labels of the complete confirmation blocks (see the module docstring)."""
    if state["next"]["campaign"] != "confirmation":
        at = state["next"]["campaign"]
        return {"available": False, "reason": f"the stage is at {at}, not confirmation"}
    kstar, blocks = state["frozen"]["K_star"], state["confirmation"]
    entries = look_blocks(blocks, labels, kstar)
    fixed = {key: fixation[key] for key in LOOK_SETTINGS}
    primary = looks(entries, fixation["levels"], **fixed)
    worst = looks([{**e, "delta": e.get("delta_missing_harmful")} for e in entries],
                  fixation["levels"], **fixed)  # fmt: skip
    started = (primary["looks"][-1] if primary["stopped"] else primary["waiting"])["started"]
    read = entries[:started]
    supports = {b["seed"]: {arm: b["supports"][arm] for arm in PRIMARY_ARMS}
                for b in blocks[:started]}  # fmt: skip
    primary_runs = [r for r in runs if r.arm in PRIMARY_ARMS and r.seed in supports]
    bootstrap = fixation["interval"] == "bootstrap_t"
    return {
        "available": True,
        "fixation": fixation,
        "interval": {
            "method": fixation["interval"],
            "resamples": fixation["bootstrap_resamples"] if bootstrap else None,
            "seed": fixation["bootstrap_seed"] if bootstrap else None,
        },
        "looks": primary,
        "looks_missing_harmful": worst,
        "missing_labels": {
            arm: sum(e.get("missing", {}).get(arm, 0) for e in read) for arm in ARMS
        },
        "waiting_for_labels": [e["block"] for e in entries[:CAP] if e.get("unlabeled")],
        "secondary": secondary_contrasts(entries, started) if primary["stopped"] else None,
        "utility": utility_family(primary_runs, supports, theta0),
        "blocks": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--theta0", type=Path, required=True)
    parser.add_argument("--w-max", type=float, help="the realized-width bound w_max")
    parser.add_argument("--fixation", type=Path, help="the simulation's fixation record (JSON)")
    parser.add_argument("--judge", help="a judge config in configs/judge/: adds the looks")
    parser.add_argument("--runs-root", type=Path, help="RUNS_DIR, whose launch/ journal is read")
    parser.add_argument("--out", type=Path, help="the whole record (JSON); must not exist")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error(f"{args.out} exists")
    if args.judge and args.fixation is None:
        parser.error("--judge needs the fixation record (--fixation): the looks' levels, SE floor "
                     "and bias allowance are never defaulted")  # fmt: skip
    fixation, w_max = None, args.w_max
    if args.fixation is not None:
        try:
            fixation = read_fixation(args.fixation)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if w_max is not None and w_max != fixation["w_max"]:
            parser.error(f"--w-max {w_max} differs from the fixation record's {fixation['w_max']}")
        w_max = fixation["w_max"]
    if w_max is None or not w_max > 0:
        parser.error("w_max is needed: --w-max or --fixation")
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary.get(key) for key in ("benchmark_accuracy", "factual_accuracy")}
    if theta0_problems(theta0):
        parser.error("; ".join(theta0_problems(theta0)))
    runs = load_trajectories(*args.runs)
    facts = {r.run_id: run_facts(r.run_dir) for r in runs if r.arm in (*ARMS, FA_CONTROL_ARM)}
    records = launch_records(args.runs_root) if args.runs_root is not None else []
    state = progress(runs, theta0, w_max, facts=facts, exhausted=exhausted_runs(records))
    shown = status(state)
    shown["launch_journal"] = str(args.runs_root / "launch") if args.runs_root else None
    record: dict[str, Any] = {"command": sys.argv, "code": git_state(),
                              "theta0": {"run": args.theta0.name, **theta0}, "w_max": w_max,
                              "progress": state}  # fmt: skip
    if args.judge:
        labels = read_labels(state, runs, args.judge)
        found = compute_looks(state, runs, labels, fixation, theta0)
        record["looks"] = found
        shown["looks"] = {k: v for k, v in found.items() if k != "blocks"}
    print(json.dumps(shown, indent=1, default=str))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
