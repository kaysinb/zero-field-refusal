"""Print the state of study 2's short-regime stage (amendment 3, design A) and, once its
confirmation blocks are labeled, its looks and, at the confirmation's end, amendment 4's final
analysis.

    python scripts/study2_short_summary.py --runs DIR [DIR ...] --theta0 RUN_DIR \
        (--w-max W | --fixation FILE) [--runs-root RUNS_DIR] [--abandoned FILE] \
        [--judge NAME [--versions FILE]] [--out OUT.json]
    python scripts/study2_short_summary.py versions --judge NAME --fixation FILE \
        [--allow-descendants] [--out FILE]

--runs holds every run so far, Stage A's included (as for scripts/study2_short_campaigns.py),
--theta0 the θ₀ run, the utility baseline, and --runs-root RUNS_DIR, whose numerical/ holds the
numerical-check verdicts (without it the selection stops, incomplete, where it needs one) and
whose launch/ journal tells which runs the launcher gave up on (technical failures of their
blocks; without it such a block waits). A run is given up only once the last attempt of its
recovery round has ended, or that attempt is in the abandonment record --abandoned FILE
(analysis.study2_short.read_abandonment: a JSON object
``{"abandoned": [{"run_id": ATTEMPT, "reason": TEXT}]}``, ATTEMPT the journal's run_id, kept
under $RUNS_DIR outside the checkout's tracked tree and archived with an inventory); an attempt
that may still be running keeps its run pending. The status is the stage's state
machine (analysis.study2_short.progress), on K, numerics and utility only: the candidates (U, B)
in amendment 4's order with their costs (B = 20 only, review fa22), every attempt (its
selections, numerical checks and validation, its failures, its arms blocked by technical marks
only, the kind of every rate excluded up front, and each arm's search allowance, spent there or
carried) and round, amendment 3's FA controls (reported with their verdicts, never used), the
frozen rates, the validation verdict, the support of the confirmation blocks with the counts of
the confirmation loop (the first block without runs, the block the next range starts at, after
the blocks of the ranges the launcher started too, the supported blocks complete in order,
those that may still be supported; with a three-arm selection the jointly supported blocks
complete in order, those that may still be, and the room for new blocks, at most the blocks left
to the cap), the mechanism cohort's
state (``mechanism``, amendment 4, section 3, on K and support only: the blocks started, the
jointly supported ones complete in order, whether the cohort has ended, by 60 jointly supported
blocks or at the cap of 80 started and complete, and its attrition by reason; a two-arm selection
has none), the problems of the records, and the next campaign (a stage ``blocked`` by technical
marks only with its gate, arms, rates and reasons; an ``infeasible`` one with its B = 20
feasibility report). The time
course's runs (amendment 4, section 4) are split from the others where they are loaded
(analysis.study2_short.split_timecourse): no look, label read, utility family, sensitivity or
contrast ever sees them; the status prints their counts with the failed and missing runs per
seed and arm (``timecourse``), and OUT.json's ``progress.timecourse`` reports each (its numerical
consistency with its endpoint run: identical, consistent, excluded or pending, review fa21),
without labels. ``scalars`` reports the update scalars' coverage of every
regular run of every role (analysis.study2_short.scalars_coverage): whether update_scalars.jsonl
exists with one row per update step (1-25 for a run that finished) and, for the pre-warmed arm,
the denominator's departure at steps 5, 10 and 25; a gap is reported and gates nothing.

--judge NAME (configs/judge/NAME.yaml) adds the looks, on the label counts of the bracket runs
of every complete confirmation block (analysis.study2_labels.read_endpoint_labels). A run without
labels yet, or whose labels have no completed judging receipt matching its label file
(``labels_pending``, with the reason), keeps its block waiting: its endpoints are not yet
labeled, never "not harmful". It prints the two sequential looks with the finite-cohort bounds,
the missing-label sensitivities at the primary's looks (``missing_labels``, the worst case for
the contrast, and ``all_missing_harmful``; analysis.study2_short.sensitivity_looks), the missing
labels of the primary arms, the blocks after the stopping look as a descriptive
``continuation`` (amendment 4: the confirmation goes on for the mechanism cohort; nothing there
revises the decision), and the utility family over the blocks started. The looks wait for every
supported arm's labels, the pre-warmed arm's included, but neither the printed status nor OUT.json
carries the pre-warmed arm's interpolated harm or missing labels (the looks' ``blocks`` hold the
primary arms' only), nor the reason a pre-warmed run's labels are pending (``labels_pending``
shows its generic status): the final analysis opens them. Amendment 3's secondary contrasts are not
computed: amendment 4's mechanism cohort replaces them. Then ``final``, amendment 4's final
analysis, computed once the confirmation has ended: with three arms, once the mechanism cohort
has ended, its blocks are labeled and the primary has stopped; with two arms, once the primary
has stopped. It holds the mechanism cohort's analysis (three arms;
analysis.study2_short.mechanism_analysis: C1 = pre-warmed − FA and C2 = AdamW − pre-warmed on
the cohort, their sum against AdamW − FA, the max-t intervals at the fixation record's
``mechanism`` level with their critical value, the per-contrast 97.5 % intervals beside them,
the readings, the attrition, the missing-label sensitivities, and each cohort block's harm and
missing labels per arm) and the time course with its labels (harm per step of each run
numerically consistent with its endpoint run: identical or consistent). Before, ``final`` says
"not yet" with the reason, and no label is read for it.
--judge refuses without the committed versions record (--versions FILE, default
docs/study2-short-versions.json), or when the running commit, the model, the judge's identity
or the fixation record departs from it (analysis.study2_short.versions_problems); OUT.json then
records the versions record's path and sha256. The looks need
the simulation's fixation record (--fixation: w_max, the look levels, the SE floor, the bias
allowance, the interval with its bootstrap resamples and seed, the look sizes and cap, the
equivalence and direction minima, and the mechanism cohort's values;
analysis.study2_short.read_fixation) and are refused without
one; none of its values is ever defaulted. The looks print the interval, resamples and seed they
used, and each bootstrap look its stream's seed and dropped resamples. Beside each look they
print the bias allowance b, the interval widened by b and the pointwise readings about Δ(K*)
under the look's equivalence and direction minima (analysis.study2_short.pointwise, as the
simulation's pointwise_readings); decisions never use them. Only counts, K, rates and intervals
are printed, never a prompt, a response or a judge output. OUT.json, which must not exist, gets
the whole record with the command and the code revision; write it under $RUNS_DIR (ignored by
Git) and archive it with an inventory, never commit it (a later commit would make HEAD a
descendant of the versions record's commit).

``versions`` writes the versions record that amendment 3 requires before the validation campaign
(default docs/study2-short-versions.json, which must not exist; it is then committed): the
commit of this checkout, which must be clean (the commit the validation and confirmation run
from), whether a later commit may run them (``allow_descendants``, false unless
--allow-descendants is given; committing the record itself is always allowed), the model and its
revision of experiments/study2_short, the identity that every label of judge NAME carries
(judge.judge_identity), and the fixation record's path and sha256.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.stage_a import index, load_trajectories, theta0_problems
from zero_field_refusal.analysis.study2_labels import LabelsPending, read_endpoint_labels
from zero_field_refusal.analysis.study2_short import (
    ARMS,
    CAP,
    CONFIRMATION_SEEDS,
    FA_CONTROL_ARM,
    IDENTITY_COUNTED,
    LOOK_SIZES,
    PREWARMED,
    PRIMARY_ARMS,
    SENSITIVITIES,
    VERSIONS_CONFIG,
    VERSIONS_RECORD,
    confirmation_guard,
    continuation,
    exhausted_runs,
    issued_ranges,
    issued_w_max,
    look_blocks,
    looks,
    mechanism_analysis,
    mechanism_blocks,
    progress,
    read_abandonment,
    read_fixation,
    read_numerical,
    run_facts,
    scalars_coverage,
    sensitivity_looks,
    split_timecourse,
    timecourse_summary,
    utility_family,
    versions_problems,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.data import load_prompt_set
from zero_field_refusal.judge import judge_identity
from zero_field_refusal.launch import launch_records
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

# The fixation record's values that ``looks`` takes (its look sizes and cap equal the module's).
LOOK_SETTINGS = ("interval", "se_floor", "bias_allowance", "equivalence_min_supported",
                 "direction_min_supported", "bootstrap_resamples", "bootstrap_seed")  # fmt: skip
# The interval settings a sensitivity takes from the fixation record: the primary's.
SENSITIVITY_SETTINGS = ("interval", "se_floor", "bootstrap_resamples", "bootstrap_seed")


def status(state: dict[str, Any], issued=()) -> dict[str, Any]:
    """The state machine's status, briefly; ``issued``: the confirmation ranges the launcher
    started (``issued_ranges``), whose blocks the guard's counts take as started."""
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
        # Amendment 3's FA controls, reported with their verdicts; amendment 4 does not use them.
        out["amendment3_controls"] = [
            {k: c[k] for k in ("run", "batch", "lr", "bound", "contamination", "accepted", "used")}
            for c in chosen.get("amendment3_controls", [])
        ]
        out["not_reached"] = chosen.get("not_reached", [])
        if "pending" in chosen:  # the selection's step, held while the records are not whole
            pending = chosen["pending"]
            out["pending"] = {k: v for k, v in pending.items() if k != "runs"}
            if "runs" in pending:
                out["pending"]["runs"] = len(pending["runs"])
        if "beyond_stale" in chosen:
            out["beyond_stale"] = chosen["beyond_stale"]
    if "frozen" in state:
        out["frozen"] = state["frozen"]
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
        # The confirmation loop's counts (K and the launcher's journal only): where the next range
        # starts (after the blocks with runs and those of the issued ranges), how many supported
        # blocks the next look may already have and, with three arms, how many jointly supported
        # blocks the mechanism cohort may already have.
        three_arm = state["frozen"]["three_arm"]
        counts = confirmation_guard(blocks, 1, 1, three_arm=three_arm, issued=issued)
        out["confirmation"] = {
            "three_arm": three_arm,
            "blocks_with_runs": sum(1 for b in blocks if b["runs"]),
            "first_block_without_runs": next((b["block"] for b in blocks if not b["runs"]),
                                             len(blocks) + 1),
            "next_range_starts_at": counts["first_open"],
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
        if three_arm:
            out["confirmation"].update(
                jointly_supported_in_order=counts["jointly_supported_in_order"],
                possibly_jointly_supported=counts["possibly_jointly_supported"],
                room=max(counts["room"], 0),
            )
        out["mechanism"] = state["mechanism"]
    if state.get("timecourse") is not None:  # without labels, on every run
        course = state["timecourse"]
        out["timecourse"] = {key: course[key] for key in ("counts", "failures", "missing")}
    return out


def _selection(s: dict[str, Any]) -> dict[str, Any]:
    best = s.get("best_unconstrained")
    return {"rates": s["rates"], "feasible": s["feasible"], "unmet": s["unmet"],
            "usable": s["usable"], "excluded": s["excluded"],
            "deviations": (s["selected"] or best or {}).get("deviations"),
            "best_unconstrained": best["rates"] if best else None}  # fmt: skip


def _attempt(a: dict[str, Any]) -> dict[str, Any]:
    validation = a.get("validation")
    return {"batch": a["batch"], "U": a["U"], "three_arm": a["three_arm"],
            "resolve_carried": a.get("resolve_carried", False),
            "outcome": a["outcome"], "failed_arms": a["failed_arms"],
            "failed_at": a.get("failed_at"), "failures": a.get("failures", {}),
            "blocked_arms": a.get("blocked_arms", []), "blocked_at": a.get("blocked_at"),
            "stopped_technical": a.get("stopped_technical", []),
            "reasons": a["reasons"], "excluded": a["excluded"],
            "excluded_kinds": a.get("excluded_kinds", {}),
            # The search allowance per arm (review fa22), spent here or carried from pass 1
            "allowance": a.get("allowance", {}),
            "allowance_carried": a.get("allowance_carried", []),
            "allowance_owed": a.get("allowance_owed", []),
            "arms": {arm: [_selection(s) for s in d["selections"]]
                     for arm, d in a["arms"].items()},
            # The rates the visit holds once eligible (None before), each check with the rates
            # it was asked for, and the validation's runs given up by the launcher (review fa22,
            # C5: what a reader of the status needs to follow the order)
            "rates": a.get("rates"),
            "numerical": [{k: e.get(k) for k in ("check", "verdict", "request", "state",
                                                 "reasons", "unusable", "unfinished", "history",
                                                 "commits", "given_up")}
                          for e in a["numerical"]],
            "validation": None if validation is None else {
                "passes": validation["passes"], "reasons": validation["reasons"],
                "jointly_supported": len(validation.get("jointly_supported", [])),
                "exhausted": validation.get("exhausted", {})}}  # fmt: skip


def _proposal(r: dict[str, Any]) -> Any:
    proposal = r["proposal"]
    if r["kind"] == "targeted":
        return proposal["targets"] + [{"skipped": s} for s in proposal["skipped"]]
    if r["kind"] == "first":  # the pre-warmed arm's first round
        return {"rates": proposal["rates"], "joint": proposal["joint"]}
    return proposal["reasons"]


def _utility(family: dict[str, Any] | None) -> dict[str, Any] | None:
    if family is None:
        return None
    keys = ("arm", "rate", "runs", "mmlu_upper_pp", "factual_upper_pp")
    return {"passes": family["passes"], "alpha_per_bound": family["alpha_per_bound"],
            "family": [{k: e[k] for k in keys} for e in family["family"]]}  # fmt: skip


def read_labels(state, runs, judge: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """The label rows of the bracket runs of every arm supported in a complete confirmation
    block, by run id, and the runs whose labels are pending, by run id with the reason. A run
    without a label file is not labeled yet and left out; one whose labels have no completed
    judging receipt matching them (``LabelsPending``) is pending and left out too. Any other
    problem of a label file raises."""
    by_id = {run.run_id: run for run in runs}
    identity = judge_identity(load_config(f"judge/{judge}"))
    prompts: dict[str, dict[str, str]] = {}
    labels: dict[str, list[dict[str, Any]]] = {}
    pending: dict[str, str] = {}
    for block in state.get("confirmation", []):
        for support in block["supports"].values():
            if not support["supported"]:
                continue
            for side in ("lo", "hi"):
                run = by_id[support["bracket"][side]["run"]]
                if run.run_id in labels or run.run_id in pending:
                    continue
                if not label_path(run.run_dir, judge).is_file():
                    continue
                cfg = OmegaConf.load(run.run_dir / "config.yaml")
                name = str(cfg.eval.harmful.prompt_set)
                if name not in prompts:
                    prompts[name] = {item["id"]: item["text"]
                                     for item in load_prompt_set(name, cfg.data)}  # fmt: skip
                try:
                    labels[run.run_id] = read_endpoint_labels(run.run_dir, judge, identity,
                                                              prompts[name])  # fmt: skip
                except LabelsPending as exc:
                    pending[run.run_id] = str(exc)
    return labels, pending


def compute_looks(state, runs, labels, fixation, theta0, pending=None) -> dict[str, Any]:
    """The looks on the labels of the complete confirmation blocks (see the module docstring);
    ``pending``: the runs whose labels are pending, with the reason (``read_labels``)."""
    if state["next"]["campaign"] != "confirmation":
        at = state["next"]["campaign"]
        return {"available": False, "reason": f"the stage is at {at}, not confirmation"}
    kstar, blocks = state["frozen"]["K_star"], state["confirmation"]
    entries = look_blocks(blocks, labels, kstar)
    fixed = {key: fixation[key] for key in LOOK_SETTINGS}
    primary = looks(entries, fixation["levels"], **fixed)
    # The sensitivities at the primary's actual looks, with its interval and levels: no stopping
    # path of their own.
    same = {key: fixation[key] for key in SENSITIVITY_SETTINGS}
    sensitivities = {name: sensitivity_looks(primary, entries, name, **same)
                     for name in SENSITIVITIES}  # fmt: skip
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
        "sensitivities": sensitivities,
        # The primary arms' missing labels; the pre-warmed arm's wait for the final analysis.
        "missing_labels": {
            arm: sum(e.get("missing", {}).get(arm, 0) for e in read) for arm in PRIMARY_ARMS
        },
        "waiting_for_labels": [e["block"] for e in entries[:CAP] if e.get("unlabeled")],
        "labels_pending": _sealed_pending(pending or {}, runs),
        # Amendment 4: the blocks after the stopping look, descriptive (the mechanism cohort's
        # analysis replaces amendment 3's secondary contrasts, which are not computed).
        "continuation": continuation(entries, primary),
        "utility": utility_family(primary_runs, supports, theta0),
        "blocks": [_sealed(entry) for entry in entries],
    }


PENDING_SEALED = "labeling pending (sealed until the final analysis)"


def _sealed_pending(pending: dict[str, str], runs) -> dict[str, str]:
    """The runs whose labels are pending, with the reason, as the summary records them: the
    pre-warmed arm's reasons carry its missing-label counts, which only the final analysis opens,
    so its runs show a generic status (the run stays listed; the primary arms' reasons stay)."""
    prewarmed = {run.run_id for run in runs if run.arm == PREWARMED}
    return {run_id: PENDING_SEALED if run_id in prewarmed else reason
            for run_id, reason in pending.items()}  # fmt: skip


def open_pending(found: dict[str, Any], pending: dict[str, str], final: dict[str, Any]) -> None:
    """With the final analysis available, the looks' ``labels_pending`` carries the full reasons
    (``final_analysis`` does not read it: it is computed on the sealed ``found``)."""
    if final["available"] and "labels_pending" in found:
        found["labels_pending"] = dict(pending)


def _sealed(entry: dict[str, Any]) -> dict[str, Any]:
    """A look block as the summary records it: without the pre-warmed arm's h and missing labels,
    which only the final analysis opens (the mechanism cohort's; the coordinator's ruling on
    Task 3's concern 3). ``look_blocks`` itself is untouched: the looks read the same entries."""
    return {key: ({arm: v for arm, v in value.items() if arm != PREWARMED}
                  if key in ("h", "missing") else value)
            for key, value in entry.items()}  # fmt: skip


def read_timecourse_labels(state, courses, judge: str) -> tuple[dict, dict[str, str]]:
    """The label rows of the time course's runs that count (``progress``'s ``timecourse``:
    identical or numerically consistent with their endpoint run, ``IDENTITY_COUNTED``, review
    fa21), by run id, and those whose labels are pending, with the reason, as ``read_labels``
    reads them: only in the final analysis (amendment 4, section 4: sealed until then)."""
    by_id = {run.run_id: run for run in courses}
    identity = judge_identity(load_config(f"judge/{judge}"))
    prompts: dict[str, dict[str, str]] = {}
    labels: dict[str, list[dict[str, Any]]] = {}
    pending: dict[str, str] = {}
    for entry in state["timecourse"]["trajectories"]:
        if entry["state"] not in IDENTITY_COUNTED:
            continue
        run = by_id[entry["run"]]
        if not label_path(run.run_dir, judge).is_file():
            continue
        cfg = OmegaConf.load(run.run_dir / "config.yaml")
        name = str(cfg.eval.harmful.prompt_set)
        if name not in prompts:
            prompts[name] = {item["id"]: item["text"]
                             for item in load_prompt_set(name, cfg.data)}  # fmt: skip
        try:
            labels[run.run_id] = read_endpoint_labels(run.run_dir, judge, identity, prompts[name])
        except LabelsPending as exc:
            pending[run.run_id] = str(exc)
    return labels, pending


def final_analysis(state, runs, courses, labels, fixation, found, judge, exhausted) -> dict:
    """Amendment 4's final analysis, once the confirmation has ended: with three arms, the
    mechanism cohort ended (``progress``'s ``mechanism``, K only), its blocks labeled and the
    primary stopped (``found``: ``compute_looks``' result); with two arms (no mechanism cohort),
    the primary stopped. Then, once: ``mechanism_analysis`` (three arms) on the fixation record's
    ``mechanism`` values, and the time course with its labels (``timecourse_summary``). Before,
    "not yet" with the reason, and no label is read for either."""
    if state["next"]["campaign"] != "confirmation" or not found.get("available"):
        return {"available": False, "reason": "not yet: the stage is not at its confirmation"}
    frozen, primary = state["frozen"], found["looks"]
    mechanism = state["mechanism"]
    if frozen["three_arm"]:
        if not mechanism["ended"]:
            return {"available": False,
                    "reason": f"not yet: the mechanism cohort has not ended "
                              f"({mechanism['jointly_supported_in_order']} of "
                              f"{mechanism['target']} jointly supported blocks complete in "
                              f"order, {mechanism['started']} of {mechanism['cap']} blocks "
                              "started)"}  # fmt: skip
        cohort = set(mechanism["cohort"])
        unlabeled = [e["block"] for e in found["blocks"] if e["block"] in cohort
                     and not e["complete"]]  # fmt: skip
        if unlabeled:
            why = f"not yet: cohort blocks {unlabeled} wait for their labels"
            return {"available": False, "reason": why}
    if not primary["stopped"]:
        return {"available": False, "reason": "not yet: the primary has not stopped (its "
                "blocks wait for their labels)"}  # fmt: skip
    out: dict[str, Any] = {"available": True, "mechanism": None}
    if frozen["three_arm"]:
        last = mechanism["last_block"]
        entries = mechanism_blocks(state["confirmation"][:last], labels, frozen["K_star"])
        out["mechanism"] = mechanism_analysis(entries, fixation["mechanism"])
    else:
        out["mechanism_reason"] = mechanism["reason"]
    course_labels, pending = read_timecourse_labels(state, courses, judge)
    endpoints, _ = index(r for r in runs if r.arm in ARMS and r.seed in CONFIRMATION_SEEDS)
    out["timecourse"] = timecourse_summary(courses, course_labels, endpoints=endpoints,
                                           rates=frozen["rates"], batch=frozen["batch"],
                                           exhausted=exhausted)  # fmt: skip
    out["timecourse_labels_pending"] = pending
    return out


def _final_shown(final: dict[str, Any]) -> dict[str, Any]:
    """The final analysis, briefly: the mechanism cohort's analysis whole, and of the time course
    its counts, its unlabeled runs and each trajectory's rows (step, K_scoring, harm)."""
    if not final["available"]:
        return final
    course = final["timecourse"]
    return {**{k: v for k, v in final.items() if k != "timecourse"},
            "timecourse": {"counts": course["counts"], "unlabeled": course.get("unlabeled"),
                           "trajectories": [{k: t[k] for k in ("seed", "arm", "state", "rows")}
                                            for t in course["trajectories"]]}}  # fmt: skip


def _coverage_shown(coverage: dict[str, Any]) -> dict[str, Any]:
    """The update scalars' coverage, briefly: the counts per role and each run with a gap."""
    keys = ("run", "enabled", "file", "unreadable", "steps_taken", "missing_steps",
            "repeated_steps", "unexpected_steps", "departure_missing")  # fmt: skip
    return {"by_role": coverage["by_role"], "gaps": coverage["gaps"],
            "status": coverage["status"],
            "with_gaps": [{k: e[k] for k in keys if k in e} for e in coverage["runs"]
                          if e["gap"]]}  # fmt: skip


def versions_record(
    judge: str, fixation: Path, *, allow_descendants: bool = False
) -> dict[str, Any]:
    """The versions record (see the module docstring); ``ValueError`` unless the checkout is a
    clean one of a known commit."""
    code = git_state()
    if not code.get("commit") or code.get("dirty") is not False:
        raise ValueError(f"the versions record needs a clean checkout of a known commit, not "
                         f"{code}")  # fmt: skip
    cfg = load_config(*VERSIONS_CONFIG)
    return {
        "what": "The versions of study 2's short-regime validation and confirmation (amendment "
        "3), written before the validation campaign.",
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": {"commit": code["commit"], "allow_descendants": allow_descendants},
        "model": {"name": str(cfg.model.name), "revision": str(cfg.model.revision)},
        "judge": {"config": f"judge/{judge}", **judge_identity(load_config(f"judge/{judge}"))},
        "fixation": {
            "path": str(fixation),
            "sha256": hashlib.sha256(fixation.read_bytes()).hexdigest(),
        },
    }


def versions_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="study2_short_summary.py versions",
                                     description="Write the versions record.")  # fmt: skip
    parser.add_argument("--judge", required=True, help="a judge config in configs/judge/")
    parser.add_argument("--fixation", type=Path, required=True, help="the fixation record")
    parser.add_argument("--allow-descendants", action="store_true",
                        help="a later commit may run the validation and confirmation")  # fmt: skip
    parser.add_argument("--out", type=Path, default=Path(VERSIONS_RECORD),
                        help="must not exist")  # fmt: skip
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists: the versions record is written once")
    try:
        read_fixation(args.fixation)
        record = versions_record(args.judge, args.fixation,
                                 allow_descendants=args.allow_descendants)  # fmt: skip
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(args.out)


def main() -> None:
    if sys.argv[1:2] == ["versions"]:
        versions_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--theta0", type=Path, required=True)
    parser.add_argument("--w-max", type=float, help="the realized-width bound w_max")
    parser.add_argument("--fixation", type=Path, help="the simulation's fixation record (JSON)")
    parser.add_argument("--judge", help="a judge config in configs/judge/: adds the looks")
    parser.add_argument("--runs-root", type=Path, help="RUNS_DIR, whose launch/ journal is read")
    parser.add_argument("--abandoned", type=Path,
                        help="the abandonment record of launcher attempts (JSON)")  # fmt: skip
    parser.add_argument("--versions", type=Path, default=REPO_ROOT / VERSIONS_RECORD,
                        help="with --judge: the versions record")  # fmt: skip
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
    if args.judge:  # the looks read labels: the versions record must hold
        problems = versions_problems(args.versions, fixation=args.fixation, judge=args.judge)
        if problems:
            parser.error("the versions record does not hold: " + "; ".join(problems))
    abandoned = {}
    if args.abandoned is not None:
        try:
            abandoned = read_abandonment(args.abandoned)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary.get(key) for key in ("benchmark_accuracy", "factual_accuracy")}
    if theta0_problems(theta0):
        parser.error("; ".join(theta0_problems(theta0)))
    loaded = load_trajectories(*args.runs)
    facts = {r.run_id: run_facts(r.run_dir) for r in loaded if r.arm in (*ARMS, FA_CONTROL_ARM)}
    # The time course's runs, split once here: the looks, their labels, the utility family, the
    # sensitivities and the contrasts read ``runs`` only (review of Task 5, C1).
    runs, courses = split_timecourse(loaded, facts)
    records = launch_records(args.runs_root) if args.runs_root is not None else []
    try:
        exhausted = exhausted_runs(records, abandoned=abandoned)
    except ValueError as exc:
        parser.error(str(exc))
    numerical = read_numerical(args.runs_root) if args.runs_root is not None else None
    state = progress(runs, theta0, w_max, facts=facts, exhausted=exhausted, numerical=numerical,
                     timecourse=courses, issued_w_max=issued_w_max(records))  # fmt: skip
    shown = status(state, issued_ranges(records))
    shown["launch_journal"] = str(args.runs_root / "launch") if args.runs_root else None
    shown["abandoned"] = abandoned
    record: dict[str, Any] = {"command": sys.argv, "code": git_state(),
                              "theta0": {"run": args.theta0.name, **theta0}, "w_max": w_max,
                              "abandoned": abandoned, "progress": state}  # fmt: skip
    if args.abandoned is not None:
        digest = hashlib.sha256(args.abandoned.read_bytes()).hexdigest()
        record["abandonment_record"] = {"path": str(args.abandoned), "sha256": digest}
    # The update scalars' coverage of every regular run, the time course's included: reported,
    # never gating (review of Task 6, M4).
    coverage = scalars_coverage([*runs, *courses], facts)
    record["scalars"] = coverage
    shown["scalars"] = _coverage_shown(coverage)
    if args.judge:
        digest = hashlib.sha256(args.versions.read_bytes()).hexdigest()
        record["versions_record"] = {"path": str(args.versions), "sha256": digest}
        labels, pending = read_labels(state, runs, args.judge)
        found = compute_looks(state, runs, labels, fixation, theta0, pending)
        final = final_analysis(state, runs, courses, labels, fixation, found, args.judge,
                               exhausted)  # fmt: skip
        open_pending(found, pending, final)
        record["looks"] = found
        shown["looks"] = {k: v for k, v in found.items() if k != "blocks"}
        record["final"] = final
        shown["final"] = _final_shown(final)
    print(json.dumps(shown, indent=1, default=str))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
