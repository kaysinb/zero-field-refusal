"""The zero-field law of plain SGD-long and its check on other runs; reads K only
(docs/study2-stage-a-analysis.md, docs/study2-pilot-momentum.md).

In the zero field, plain SGD follows K(t) ≈ (η/B)·g(ηt). ``g_table`` estimates g from SGD-long's
version-2 runs: each finite scoring diagnostic gives the point (ηt, K_scoring / (η/B)); the
points fall into logarithmic bins of ηt; a run's value in a bin is the median of its points there,
and g in the bin is the median over the runs, kept where at least ``MIN_RUNS`` runs contribute.
``g_at`` interpolates linearly in log ηt between the bins' geometric centres and is undefined
outside them. ``check`` compares a run with the law at an effective rate (η/(1 − β) for heavy-ball
momentum β): the ratio of each finite scoring diagnostic's K to (η_eff/B)·g(η_eff·t), at every
diagnostic step whose η_eff·t lies within the table.

A non-finite scoring diagnostic can only be a run's terminal one (``load_trajectory`` refuses any
other); it never enters a table or a check, and both report it.

``freeze`` takes the table's source runs from Stage A's resolved search alone (every SGD-long run
it tested, on every discovery seed) and records their identity: each run's ID and the SHA-256 of
the files the table reads, and the table's own hash. ``verify`` recomputes all of it from Stage
A's records and lists every difference from a frozen table.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .stage_a import DISCOVERY_SEEDS, ROLES, Trajectory, search_progress

BIN_FROM = 0.01  # the lower edge of the first bin of ηt
BIN_RATIO = 1.25  # each bin's upper edge over its lower edge
MIN_RUNS = 2  # runs a bin needs for g to be estimated there
MIN_POINTS = 5  # finite eligible ratios a run needs for its check
SOURCE_FILES = ("config.yaml", "diagnostics.jsonl", "terminal.json")  # what the table reads


def _bin(x: float) -> int:
    return math.floor(math.log(x / BIN_FROM) / math.log(BIN_RATIO))


def _points(run: Trajectory, rate: float) -> tuple[list[tuple[int, float, float]], list[int]]:
    """(step, ηt, K_scoring / (η/B)) of every finite scoring diagnostic after step 0, and the
    steps of the non-finite ones (a run's terminal diagnostic at most)."""
    scale = rate / run.batch_size
    points, nonfinite = [], []
    for row in run.diagnostics:
        if row["step"] <= 0:
            continue
        if math.isfinite(row["K_scoring"]):
            points.append((int(row["step"]), rate * row["step"], row["K_scoring"] / scale))
        else:
            nonfinite.append(int(row["step"]))
    return points, nonfinite


def g_table(runs: Sequence[Trajectory]) -> dict[str, Any]:
    """g from plain SGD-long's version-2 runs among ``runs``, with the runs it used and the
    non-finite diagnostics it left out. Raises ``ValueError`` on a table that is empty or not
    finite."""
    source = [r for r in runs if r.arm == "sgd_long" and r.monitor_version > 1 and not r.momentum]
    by_bin: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    excluded = []
    for run in source:
        points, nonfinite = _points(run, run.lr)
        excluded += [{"run_id": run.run_id, "step": step} for step in nonfinite]
        for _, x, y in points:
            if x >= BIN_FROM:
                by_bin[_bin(x)][run.run_id].append(y)
    bins = []
    for b in sorted(by_bin):
        per_run = [statistics.median(v) for v in by_bin[b].values()]
        if len(per_run) >= MIN_RUNS:
            low = BIN_FROM * BIN_RATIO**b
            bins.append({"x": low * math.sqrt(BIN_RATIO), "g": statistics.median(per_run),
                         "runs": len(per_run)})  # fmt: skip
    table = {
        "runs": sorted(run.run_id for run in source),
        "bin_from": BIN_FROM,
        "bin_ratio": BIN_RATIO,
        "min_runs": MIN_RUNS,
        "bins": bins,
        "excluded_nonfinite": excluded,
    }
    table_problems(table, strict=True)
    return table


def table_problems(table: Mapping[str, Any], *, strict: bool = False) -> list[str]:
    """Why a g table cannot be used: no bins, or a bin not finite, positive and in order."""
    bins = table.get("bins") or []
    problems = [] if bins else ["the g table has no bins"]
    for left, right in zip(bins, bins[1:], strict=False):
        if not left["x"] < right["x"]:
            problems.append(f"the g table's bins are out of order at x = {right['x']}")
    for row in bins:
        if not (math.isfinite(row["x"]) and math.isfinite(row["g"]) and row["x"] > 0
                and row["g"] > 0):  # fmt: skip
            problems.append(f"the g table has a bin that is not finite and positive: {row}")
    if strict and problems:
        raise ValueError("; ".join(problems))
    return problems


def g_at(table: Mapping[str, Any], x: float) -> float | None:
    """g at ηt = x, linear in log x between the bins' centres; None outside them."""
    bins = table["bins"]
    if not bins or not math.isfinite(x) or x <= 0 or not bins[0]["x"] <= x <= bins[-1]["x"]:
        return None
    for left, right in zip(bins, bins[1:], strict=False):
        if left["x"] <= x <= right["x"]:
            w = math.log(x / left["x"]) / math.log(right["x"] / left["x"])
            return left["g"] + w * (right["g"] - left["g"])
    return bins[0]["g"]  # a one-bin table at its centre


def check(run: Trajectory, table: Mapping[str, Any]) -> dict[str, Any]:
    """The run against the law at its effective rate η/(1 − momentum): the median of the finite
    ratios of its scoring K to the prediction over its eligible diagnostics (None below
    ``MIN_POINTS``), with the non-finite diagnostics left out."""
    table_problems(table, strict=True)
    rate = run.lr / (1.0 - run.momentum)
    points, nonfinite = _points(run, rate)
    ratios = []
    for step, x, y in points:
        g = g_at(table, x)
        if g is not None and math.isfinite(y / g):
            ratios.append((step, y / g))
    return {
        "run_id": run.run_id,
        "effective_rate": rate,
        "eligible_steps": [s for s, _ in ratios],
        "excluded_nonfinite_steps": nonfinite,
        "median_ratio": statistics.median(r for _, r in ratios)
        if len(ratios) >= MIN_POINTS
        else None,
    }


def cohort(
    stage_a: Sequence[Trajectory], theta0: Mapping[str, float]
) -> tuple[list[Trajectory], list[str]]:
    """The table's source runs: every SGD-long run that Stage A's resolved search tested, on every
    discovery seed, from Stage A's records checked as its selection checks them."""
    by_role = {role: [r for r in stage_a if r.role == role] for role in ROLES}
    state, problems = search_progress(by_role["screen"], by_role["discovery"],
                                      by_role["full_control"], theta0)  # fmt: skip
    if problems:
        return [], [f"Stage A: {p}" for p in problems]
    manifest = state["manifest"]
    if not manifest["complete"]:
        waiting = {k: e["pending"] for k, e in manifest["searches"].items() if e["pending"]}
        return [], [f"Stage A: its search has not resolved: {waiting}"]
    runs = [state["present"][("sgd_long", e["batch"], lr, s)]
            for e in manifest["searches"].values() if e["arm"] == "sgd_long"
            for lr in e["tested"] for s in DISCOVERY_SEEDS]  # fmt: skip
    return sorted(runs, key=lambda r: r.run_id), []


def identity(runs: Sequence[Trajectory]) -> list[dict[str, Any]]:
    """Each run's ID and the SHA-256 of the files the table reads from it."""
    return [{"run_id": run.run_id,
             "files": {name: hashlib.sha256((run.run_dir / name).read_bytes()).hexdigest()
                       for name in SOURCE_FILES}}
            for run in sorted(runs, key=lambda r: r.run_id)]  # fmt: skip


def table_sha256(frozen: Mapping[str, Any]) -> str:
    """The hash of a frozen table's content: its bins, settings, exclusions and source identity."""
    keys = ("bin_from", "bin_ratio", "min_runs", "bins", "excluded_nonfinite", "source")
    content = json.dumps({key: frozen[key] for key in keys}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def freeze(stage_a: Sequence[Trajectory], theta0: Mapping[str, float]) -> dict[str, Any]:
    """The g table from Stage A's resolved search, with its source identity and hash. Raises
    ``ValueError`` when Stage A's records cannot give it."""
    runs, problems = cohort(stage_a, theta0)
    if problems:
        raise ValueError("; ".join(problems))
    table = g_table(runs)
    if table["runs"] != [run.run_id for run in runs]:
        raise ValueError("the table's source runs are not Stage A's SGD-long cohort")
    frozen = {**table, "source": identity(runs)}
    frozen["table_sha256"] = table_sha256(frozen)
    return frozen


def verify(
    frozen: Mapping[str, Any], stage_a: Sequence[Trajectory], theta0: Mapping[str, float]
) -> list[str]:
    """Every difference between a frozen table and what Stage A's records give now."""
    problems = table_problems(frozen)
    if frozen.get("table_sha256") != table_sha256(frozen):
        problems.append("the frozen table's content does not match its recorded hash")
    try:
        now = freeze(stage_a, theta0)
    except ValueError as exc:
        return [*problems, f"the table cannot be recomputed: {exc}"]
    if now["source"] != frozen.get("source"):
        problems.append("Stage A's SGD-long cohort or its files differ from the frozen source")
    if now["table_sha256"] != frozen.get("table_sha256"):
        problems.append("the recomputed table differs from the frozen one")
    return problems
