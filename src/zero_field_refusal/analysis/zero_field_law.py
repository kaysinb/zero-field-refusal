"""The zero-field law of plain SGD-long and its check on other runs; reads K only
(docs/study2-stage-a-analysis.md, docs/study2-pilot-momentum.md).

In the zero field, plain SGD follows K(t) ≈ (η/B)·g(ηt). ``g_table`` estimates g from Stage A's
version-2 SGD-long runs: each scoring diagnostic gives the point (ηt, K_scoring / (η/B)); the
points fall into logarithmic bins of ηt; a run's value in a bin is the median of its points there,
and g in the bin is the median over the runs, kept where at least ``MIN_RUNS`` runs contribute.
``g_at`` interpolates linearly in log ηt between the bins' geometric centres and is undefined
outside them. ``check`` compares a run with the law at an effective rate (η/(1 − β) for heavy-ball
momentum β): the ratio of each scoring diagnostic's K to (η_eff/B)·g(η_eff·t), at every diagnostic
step whose η_eff·t lies within the table.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .stage_a import Trajectory

BIN_FROM = 0.01  # the lower edge of the first bin of ηt
BIN_RATIO = 1.25  # each bin's upper edge over its lower edge
MIN_RUNS = 2  # runs a bin needs for g to be estimated there
MIN_POINTS = 5  # eligible diagnostics a run needs for its check


def _bin(x: float) -> int:
    return math.floor(math.log(x / BIN_FROM) / math.log(BIN_RATIO))


def _points(run: Trajectory, rate: float) -> list[tuple[int, float, float]]:
    """(step, ηt, K_scoring / (η/B)) of every scoring diagnostic after step 0."""
    scale = rate / run.batch_size
    return [(int(r["step"]), rate * r["step"], r["K_scoring"] / scale)
            for r in run.diagnostics if r["step"] > 0]  # fmt: skip


def g_table(runs: Sequence[Trajectory]) -> dict[str, Any]:
    """g from plain SGD-long's version-2 runs, with the runs it used."""
    source = [r for r in runs if r.arm == "sgd_long" and r.monitor_version > 1 and not r.momentum]
    by_bin: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in source:
        for _, x, y in _points(run, run.lr):
            if x >= BIN_FROM:
                by_bin[_bin(x)][run.run_id].append(y)
    bins = []
    for b in sorted(by_bin):
        per_run = [statistics.median(v) for v in by_bin[b].values()]
        if len(per_run) >= MIN_RUNS:
            low = BIN_FROM * BIN_RATIO**b
            bins.append({"x": low * math.sqrt(BIN_RATIO), "g": statistics.median(per_run),
                         "runs": len(per_run)})  # fmt: skip
    return {
        "runs": sorted(run.run_id for run in source),
        "bin_from": BIN_FROM,
        "bin_ratio": BIN_RATIO,
        "min_runs": MIN_RUNS,
        "bins": bins,
    }


def g_at(table: Mapping[str, Any], x: float) -> float | None:
    """g at ηt = x, linear in log x between the bins' centres; None outside them."""
    bins = table["bins"]
    if not bins or x <= 0 or not bins[0]["x"] <= x <= bins[-1]["x"]:
        return None
    for left, right in zip(bins, bins[1:], strict=False):
        if left["x"] <= x <= right["x"]:
            w = math.log(x / left["x"]) / math.log(right["x"] / left["x"])
            return left["g"] + w * (right["g"] - left["g"])
    return bins[0]["g"]  # a one-bin table at its centre


def check(run: Trajectory, table: Mapping[str, Any]) -> dict[str, Any]:
    """The run against the law at its effective rate η/(1 − momentum): the median ratio of its
    scoring K to the prediction over its eligible diagnostics (None below ``MIN_POINTS``)."""
    rate = run.lr / (1.0 - run.momentum)
    ratios = []
    for step, x, y in _points(run, rate):
        g = g_at(table, x)
        if g is not None and g > 0:
            ratios.append((step, y / g))
    return {
        "run_id": run.run_id,
        "effective_rate": rate,
        "eligible_steps": [s for s, _ in ratios],
        "median_ratio": statistics.median(r for _, r in ratios)
        if len(ratios) >= MIN_POINTS
        else None,
    }
