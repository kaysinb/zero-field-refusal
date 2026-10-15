"""Study 2's SGD-momentum pilot (docs/study2-pilot-momentum.md) over its own records; never reads
a harm label.

- ``screen``: the 100-step soft-target controls with momentum 0.9 at every rate of the pilot's
  grid at both batches on seed 2000, and each batch's two largest passing rates.
- ``adamw_evidence``: Stage A's AdamW-long, the comparison arm, from Stage A's own records once
  Stage A's whole search has resolved, checked as Stage A checks them.
- ``search``: the pilot's downward search at each batch, U = 7 settled jointly with AdamW-long.
- ``progress``: that search as it stands, with every problem of the records.
- ``outcome``: the pilot's outcome.

The pilot's runs carry the arms ``sgd_momentum_long`` and ``soft_control_momentum``, which Stage
A's conformance refuses, so that none can enter Stage A's selection; they live in their own
namespaces (``study2_pilot_momentum``, ``study2_pilot_momentum_controls``).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import load_config
from .stage_a import (
    BATCHES,
    CONTROL_SEED,
    DISCOVERY_SEEDS,
    LOWER,
    SCREEN_STEPS,
    UPPER,
    Trajectory,
    _long_rates,
    _name,
    _ok,
    _rate_rule,
    cells,
    index,
    level_summary,
    monitor_qualification,
    monitor_spec,
    numerical_acceptance,
    rate,
    schedule_spec,
    screen_candidates,
    search_order,
    search_progress,
    stream_pairing,
    theta0_problems,
)

ARM, CONTROL_ARM = "sgd_momentum_long", "soft_control_momentum"
MOMENTUM = 0.9
GRID = (0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006)
MAIN, CONTROLS = "experiments/study2_pilot_momentum", "experiments/study2_pilot_momentum_controls"
ROLES = ("discovery", "screen", "full_control")


@functools.cache
def prescribed() -> dict[str, Any]:
    """The model, protocol version, schedules, monitor and momentum the pilot's campaigns
    prescribe, read from its configs."""
    long, control = load_config(MAIN), load_config(CONTROLS)
    return {
        "model": f"{long.model.name}@{long.model.revision}",
        "protocol_version": int(long.protocol_version),
        "long": schedule_spec(long),
        "control": schedule_spec(control),
        "monitor": monitor_spec(long),
        "momentum": float(long.optimizer.momentum),
    }


def pilot_conformance(runs: Sequence[Trajectory], role: str) -> list[str]:
    """How the pilot's runs depart from its campaigns: role, arm, momentum, model, protocol
    version, discovery seeds, schedule and monitor (a full control's length and evaluation steps
    are checked against its regular runs by ``numerical_acceptance``)."""
    spec, problems = prescribed(), []
    free = ("train_steps", "evaluate_at")
    arm = ARM if role == "discovery" else CONTROL_ARM
    for run in runs:
        name = f"{role}: {run.run_id}"
        if run.role != role:
            problems.append(f"{name} has the role {run.role!r}")
        if run.arm != arm or not math.isclose(run.momentum, spec["momentum"]):
            problems.append(f"{name} is of the arm {run.arm!r} with momentum {run.momentum}")
        if run.model != spec["model"] or run.protocol_version != spec["protocol_version"]:
            problems.append(f"{name} ran {run.model} under protocol {run.protocol_version}")
        if run.seed not in DISCOVERY_SEEDS:
            problems.append(f"{name} has the seed {run.seed}, not a discovery seed")
        if role == "discovery":
            want, have = spec["long"], run.spec
            if run.monitor_spec != spec["monitor"]:
                problems.append(f"{name} read another monitor: {run.monitor_spec}")
        else:
            want = {k: v for k, v in spec["control"].items() if k not in free}
            have = {k: v for k, v in run.spec.items() if k not in free}
            if role == "screen":
                want |= {"train_steps": SCREEN_STEPS, "evaluate_at": [SCREEN_STEPS]}
                have |= {k: run.spec[k] for k in free}
        differ = sorted(k for k in want if have.get(k) != want[k])
        if differ:
            problems.append(f"{name} departs from the prescribed schedule in {differ}")
    return problems


def screen(runs: Sequence[Trajectory]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """The pilot's screen and each batch's candidates, with the problems of its records."""
    screened, problems = screen_candidates(runs, grid=GRID, arm=CONTROL_ARM)
    return screened, problems + pilot_conformance(runs, "screen")


def adamw_evidence(
    stage_a: Sequence[Trajectory], theta0: Mapping[str, float]
) -> tuple[dict[tuple[int, float], bool], list[Trajectory], list[str]]:
    """Stage A's AdamW-long as the comparison arm: from Stage A's own records, checked and
    searched as Stage A checks and searches them (``search_progress``), and only once Stage A's
    whole search has resolved. Returns, at each batch and U, whether some rate meets
    AdamW-long's rule there; the AdamW-long runs it rests on; and the problems that leave it
    unknown."""
    by_role = {role: [r for r in stage_a if r.role == role] for role in ROLES}
    state, problems = search_progress(by_role["screen"], by_role["discovery"],
                                      by_role["full_control"], theta0)  # fmt: skip
    problems = [f"Stage A: {p}" for p in problems]
    if problems:
        return {}, [], problems
    manifest = state["manifest"]
    if not manifest["complete"]:
        waiting = {k: e["pending"] for k, e in manifest["searches"].items() if e["pending"]}
        return {}, [], [f"Stage A: its search has not resolved: {waiting}"]
    meets, runs = {}, []
    for batch in BATCHES:
        entry = manifest["searches"][f"adamw_long/B={batch}"]
        for upper in UPPER:
            meets[(batch, upper)] = any(t[_name(upper)]["ok"] for t in entry["tested"].values())
        runs += [state["present"][("adamw_long", batch, lr, s)] for lr in entry["tested"]
                 for s in DISCOVERY_SEEDS]  # fmt: skip
    return meets, runs, []


def search(
    present: Mapping[tuple, Trajectory],
    acceptance: Mapping[tuple[int, float], Mapping[str, Any]],
    candidates: Mapping[int, Sequence[float]],
    theta0: Mapping[str, float],
    adamw: Mapping[tuple[int, float], bool],
    seeds: Sequence[int] = DISCOVERY_SEEDS,
) -> dict[str, Any]:
    """The pilot's downward search at each batch, as an auditable manifest.

    A rate resolves once all ``seeds`` have run it and its full control has a verdict. The
    descent at a batch stops at the first rate that meets the pilot's rule at U = 8. Once every
    batch has resolved, the comparison is feasible at U = 8 where such a rate meets AdamW-long
    meeting its rule there (``adamw``). If no batch is, U = 7 is settled jointly: at every batch
    where AdamW-long meets its rule at 7 and no rate run meets the pilot's rule at 7, the descent
    resumes below the rates run. Rates below a search's stopping rate are "not required"."""
    searches: dict[str, dict[str, Any]] = {}

    def descend(batch: int, entry: dict[str, Any], stop_at: float) -> None:
        for lr in entry["order"]:
            if lr in entry["tested"]:
                continue
            missing = [s for s in seeds if (ARM, batch, lr, s) not in present]
            if missing:
                entry["pending"] = {"rate": lr, "seeds": missing}
                return
            if (batch, lr) not in acceptance:
                entry["pending"] = {"rate": lr, "control": CONTROL_SEED}
                return
            cell = [present[(ARM, batch, lr, s)] for s in seeds]
            entry["tested"][lr] = {_name(u): _rate_rule(ARM, batch, lr, cell, u, theta0, seeds,
                                                        acceptance) for u in UPPER}  # fmt: skip
            if entry["tested"][lr][_name(stop_at)]["ok"]:
                return

    for batch in BATCHES:
        entry = {"arm": ARM, "batch": batch, "order": search_order(ARM, batch, candidates),
                 "tested": {}, "pending": None}  # fmt: skip
        descend(batch, entry, 8.0)
        searches[f"{ARM}/B={batch}"] = entry

    def meets(batch: int, upper: float) -> bool:
        tested = searches[f"{ARM}/B={batch}"]["tested"].values()
        return any(t[_name(upper)]["ok"] for t in tested)

    phase = "U=8"
    if all(e["pending"] is None for e in searches.values()) and not any(
        meets(b, 8.0) and adamw[(b, 8.0)] for b in BATCHES
    ):
        phase = "U=7"
        for batch in BATCHES:
            if adamw[(batch, 7.0)] and not meets(batch, 7.0):
                descend(batch, searches[f"{ARM}/B={batch}"], 7.0)
    complete = all(e["pending"] is None for e in searches.values())
    for entry in searches.values():
        entry["not_required"] = (
            [] if entry["pending"] else [lr for lr in entry["order"] if lr not in entry["tested"]]
        )
        entry["not_required_status"] = "final" if complete else "provisional"
    return {"phase": phase, "complete": complete, "searches": searches}


def progress(
    screen_runs: Sequence[Trajectory],
    discovery: Sequence[Trajectory],
    controls: Sequence[Trajectory],
    theta0: Mapping[str, float],
    stage_a: Sequence[Trajectory],
) -> tuple[dict[str, Any], list[str]]:
    """The pilot's search as it stands, with every problem of its records and of the AdamW-long
    evidence it is compared with (``adamw_evidence``)."""
    screened, problems = screen(screen_runs)
    problems += theta0_problems(theta0)
    adamw, adamw_runs, missing = adamw_evidence(stage_a, theta0)
    problems += missing
    if problems:
        return {"screen": screened}, problems
    candidates = {batch: entry["candidates"] for batch, entry in screened.items()}
    present, problems = index(discovery)
    problems += pilot_conformance(discovery, "discovery")
    problems += pilot_conformance(controls, "full_control")
    # One stream per seed, shared with the AdamW-long runs the pilot is compared with.
    problems += stream_pairing([*screen_runs, *discovery, *controls, *adamw_runs])
    resolved = {
        b: [lr for lr in candidates[b] if all((ARM, b, rate(lr), s) in present
                                              for s in DISCOVERY_SEEDS)]
        for b in BATCHES
    }  # fmt: skip
    acceptance, control_problems = numerical_acceptance(
        controls, present, resolved, arm=ARM, control_arm=CONTROL_ARM, every_read=True
    )
    problems += [p for p in control_problems if not p.startswith("controls: missing")]
    manifest = search(present, acceptance, candidates, theta0, adamw)
    searched = {
        (ARM, e["batch"], lr, s)
        for e in manifest["searches"].values()
        for lr in [*e["tested"], *([e["pending"]["rate"]] if e["pending"] else [])]
        for s in DISCOVERY_SEEDS
    }
    problems += [f"discovery: run outside the search {key}"
                 for key in sorted(set(present) - searched)]  # fmt: skip
    state = {"screen": screened, "candidates": candidates, "present": present,
             "acceptance": acceptance, "manifest": manifest, "adamw": adamw}  # fmt: skip
    return state, problems


def outcome(
    screen_runs: Sequence[Trajectory],
    discovery: Sequence[Trajectory],
    controls: Sequence[Trajectory],
    theta0: Mapping[str, float],
    stage_a: Sequence[Trajectory],
) -> dict[str, Any]:
    """The pilot's outcome (docs/study2-pilot-momentum.md, "Outcomes"): ``incomplete``,
    ``numerics_fail``, ``feasible``, ``unmatched``, ``partial`` or ``barrier``. ``stage_a`` holds
    Stage A's records (screen, discovery and full controls); its AdamW-long is the comparison."""
    seeds = DISCOVERY_SEEDS
    state, problems = progress(screen_runs, discovery, controls, theta0, stage_a)
    out: dict[str, Any] = {"seeds": list(seeds), "screen": state["screen"]}
    if problems:
        out.update(outcome="incomplete", problems=list(dict.fromkeys(problems)))
        return out
    if not any(state["candidates"].values()):
        out["outcome"] = "numerics_fail"
        return out
    manifest, acceptance, present = state["manifest"], state["acceptance"], state["present"]
    out["search"] = manifest
    out["acceptance"] = {f"B={b} lr={lr:g}": a for (b, lr), a in acceptance.items()}
    out["adamw_long"] = {f"B={b} U={u:g}": ok for (b, u), ok in state["adamw"].items()}
    if not manifest["complete"]:
        out.update(outcome="incomplete", problems=[
            f"search: {name} needs {entry['pending']}"
            for name, entry in manifest["searches"].items() if entry["pending"]
        ])  # fmt: skip
        return out
    out["monitor_qualification"] = {
        b: monitor_qualification([r for r in present.values() if r.batch_size == b], arms=(ARM,))
        for b in BATCHES
    }
    if not any(a["accepted"] for a in acceptance.values()):
        out["outcome"] = "numerics_fail"
        return out
    tested = [present[(ARM, e["batch"], lr, s)] for e in manifest["searches"].values()
              for lr in e["tested"] for s in seeds]  # fmt: skip
    pilot_cells = cells(tested)
    batches: dict[int, dict[str, Any]] = {}
    for batch in BATCHES:
        entry: dict[str, Any] = {"levels": {}}
        for upper in UPPER:
            k_star = math.sqrt(LOWER * upper)
            pilot = _long_rates(pilot_cells, ARM, batch, upper, k_star, theta0, seeds, acceptance)
            entry["levels"][_name(upper)] = {
                "K_star": k_star,
                ARM: pilot,
                "adamw_long_meets_its_rule": state["adamw"][(batch, upper)],
                "feasible": bool(_ok(pilot)) and state["adamw"][(batch, upper)],
            }
        feasible = [u for u in UPPER if entry["levels"][_name(u)]["feasible"]]
        entry["U"] = feasible[0] if feasible else None
        accepted = [lr for (b, lr), a in acceptance.items()
                    if b == batch and a["accepted"] and (ARM, b, lr) in pilot_cells]  # fmt: skip

        def reaching(level: float, batch: int = batch, accepted: list = accepted) -> list[float]:
            return sorted(lr for lr in accepted if level_summary(
                pilot_cells[(ARM, batch, lr)], level, theta0, seeds)["all_seeds"])  # fmt: skip

        entry["accepted_reaching"] = {_name(level): reaching(level) for level in (LOWER, 7.0)}
        batches[batch] = entry
    out["batches"] = batches
    if any(e["U"] is not None for e in batches.values()):
        out["outcome"] = "feasible"
        out["feasible"] = {b: e["U"] for b, e in batches.items() if e["U"] is not None}
    elif any(e["accepted_reaching"][_name(7.0)] for e in batches.values()):
        out["outcome"] = "unmatched"  # reached U = 7, but the rule or AdamW-long fails there
    elif any(e["accepted_reaching"][_name(LOWER)] for e in batches.values()):
        out["outcome"] = "partial"
    else:
        out["outcome"] = "barrier"
    return out


__all__ = [
    "ARM",
    "CONTROL_ARM",
    "CONTROL_SEED",
    "CONTROLS",
    "GRID",
    "MAIN",
    "MOMENTUM",
    "ROLES",
    "adamw_evidence",
    "outcome",
    "pilot_conformance",
    "prescribed",
    "progress",
    "screen",
    "search",
]
