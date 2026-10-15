"""Study 2's SGD-momentum pilot (docs/study2-pilot-momentum.md) over its own records; never reads
a harm label.

- ``screen``: the 100-step soft-target controls with momentum 0.9 at every rate of the pilot's
  grid at both batches on seed 2000, and each batch's two largest passing rates.
- ``progress``: the downward search of SGD-momentum-long at each batch under monitor version 2,
  as amendment 1's manifest (``analysis.stage_a.search_manifest`` with this arm alone), with
  every problem of the records.
- ``outcome``: the pilot's outcome. A batch is feasible at U when a rate of the pilot meets its
  rule there (every seed reaches U; utility in bounds at 5, K* and U; three separate
  evaluations; its full control accepted) and Stage A's AdamW-long meets its own rule at that
  batch and U.

The pilot's runs carry the arms ``sgd_momentum_long`` and ``soft_control_momentum``, which Stage
A's conformance refuses, so that none can enter Stage A's selection. Its AdamW-long evidence is
Stage A's version-2 discovery runs, checked by Stage A's own conformance.
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
    cells,
    conformance,
    index,
    level_summary,
    monitor_qualification,
    monitor_spec,
    numerical_acceptance,
    rate,
    schedule_spec,
    screen_candidates,
    search_manifest,
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


def progress(
    screen_runs: Sequence[Trajectory],
    discovery: Sequence[Trajectory],
    controls: Sequence[Trajectory],
    theta0: Mapping[str, float] | None,
) -> tuple[dict[str, Any], list[str]]:
    """The pilot's search as it stands, with every problem of its records."""
    screened, problems = screen(screen_runs)
    if problems:
        return {"screen": screened}, problems
    candidates = {batch: entry["candidates"] for batch, entry in screened.items()}
    if discovery and theta0 is None:
        raise ValueError("θ₀'s utility is needed to judge the pilot's long runs")
    if theta0_problems(theta0):
        return {"screen": screened}, theta0_problems(theta0)
    present, problems = index(discovery)
    problems += pilot_conformance(discovery, "discovery")
    problems += pilot_conformance(controls, "full_control")
    problems += stream_pairing([*screen_runs, *discovery, *controls])
    resolved = {
        b: [lr for lr in candidates[b] if all((ARM, b, rate(lr), s) in present
                                              for s in DISCOVERY_SEEDS)]
        for b in BATCHES
    }  # fmt: skip
    acceptance, control_problems = numerical_acceptance(
        controls, present, resolved, arm=ARM, control_arm=CONTROL_ARM
    )
    problems += [p for p in control_problems if not p.startswith("controls: missing")]
    manifest = search_manifest(
        list(present.values()), acceptance, candidates, theta0 or {}, DISCOVERY_SEEDS, arms=(ARM,)
    )
    searched = {
        (e["arm"], e["batch"], lr, s)
        for e in manifest["searches"].values()
        for lr in [*e["tested"], *([e["pending"]["rate"]] if e["pending"] else [])]
        for s in DISCOVERY_SEEDS
    }
    problems += [f"discovery: run outside the search {key}"
                 for key in sorted(set(present) - searched)]  # fmt: skip
    state = {"screen": screened, "candidates": candidates, "present": present,
             "acceptance": acceptance, "manifest": manifest}  # fmt: skip
    return state, problems


def outcome(
    screen_runs: Sequence[Trajectory],
    discovery: Sequence[Trajectory],
    controls: Sequence[Trajectory],
    theta0: Mapping[str, float],
    stage_a: Sequence[Trajectory],
) -> dict[str, Any]:
    """The pilot's outcome (docs/study2-pilot-momentum.md, "Outcomes"): ``incomplete``,
    ``numerics_fail``, ``feasible``, ``partial`` or ``barrier``. ``stage_a`` holds Stage A's
    discovery runs; its version-2 AdamW-long runs are the comparison arm."""
    seeds = DISCOVERY_SEEDS
    state, problems = progress(screen_runs, discovery, controls, theta0)
    out: dict[str, Any] = {"seeds": list(seeds), "screen": state["screen"]}
    adamw = [r for r in stage_a if r.arm == "adamw_long" and r.monitor_version > 1]
    problems += conformance(adamw, "discovery")
    adamw_present, duplicates = index(adamw)
    problems += duplicates
    if problems or "manifest" not in state:
        out.update(outcome="incomplete", problems=list(dict.fromkeys(problems)))
        return out
    if not any(state["candidates"].values()):
        out["outcome"] = "numerics_fail"
        return out
    manifest, acceptance, present = state["manifest"], state["acceptance"], state["present"]
    out["search"] = manifest
    out["acceptance"] = {f"B={b} lr={lr:g}": a for (b, lr), a in acceptance.items()}
    if not manifest["complete"]:
        out.update(outcome="incomplete", problems=[
            f"search: {name} needs {entry['pending']}"
            for name, entry in manifest["searches"].items() if entry["pending"]
        ])  # fmt: skip
        return out
    out["monitor_qualification"] = monitor_qualification(list(present.values()), arms=(ARM,))
    if not any(a["accepted"] for a in acceptance.values()):
        out["outcome"] = "numerics_fail"
        return out
    tested = [present[(ARM, e["batch"], lr, s)] for e in manifest["searches"].values()
              for lr in e["tested"] for s in seeds]  # fmt: skip
    pilot_cells, adamw_cells = cells(tested), cells(adamw_present.values())
    batches: dict[int, dict[str, Any]] = {}
    for batch in BATCHES:
        entry: dict[str, Any] = {"levels": {}}
        for upper in UPPER:
            k_star = math.sqrt(LOWER * upper)
            pilot = _long_rates(pilot_cells, ARM, batch, upper, k_star, theta0, seeds, acceptance)
            adam = _long_rates(adamw_cells, "adamw_long", batch, upper, k_star, theta0, seeds, None)
            entry["levels"][_name(upper)] = {
                "K_star": k_star,
                ARM: pilot,
                "adamw_long": adam,
                "feasible": bool(_ok(pilot) and _ok(adam)),
            }
        feasible = [u for u in UPPER if entry["levels"][_name(u)]["feasible"]]
        entry["U"] = feasible[0] if feasible else None
        accepted = [lr for (b, lr), a in acceptance.items() if b == batch and a["accepted"]]
        entry["reaches_lower_on_all_seeds"] = sorted(
            lr for lr in accepted
            if (ARM, batch, lr) in pilot_cells
            and level_summary(pilot_cells[(ARM, batch, lr)], LOWER, theta0, seeds)["all_seeds"]
        )  # fmt: skip
        batches[batch] = entry
    out["batches"] = batches
    if any(e["U"] is not None for e in batches.values()):
        out["outcome"] = "feasible"
        out["feasible"] = {b: e["U"] for b, e in batches.items() if e["U"] is not None}
    elif any(e["reaches_lower_on_all_seeds"] for e in batches.values()):
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
    "outcome",
    "pilot_conformance",
    "prescribed",
    "progress",
    "screen",
]
