"""Study 5's run without momentum (docs/study5-beta0-registration.md): its seeds, orders, pilot
grid, the rule that fixes the rate η₀ from the pilot, the drift gate and the campaign cells.

Each run is arm F with ``optimizer.beta1`` 0 (``experiments/study5_beta0``, role ``beta0``) on
its seed's own responses and v₀, 25 updates of B = 20. The rates are decimal literals, so that a
config, a run record and this module name the same float.

**Orders** (``ORDERS``): the stream order (no order key); the order of
``experiments/study5_control_order`` (``+train.order_seed`` = the seed); the shuffles of whole
batches 1–5 and 21–25, draws 0 and 1 (``+train.order_edit``, the permutations of extension 2's
M6). A shuffle whose permutation is the identity repeats the stream order and is not run
(``is_identity``; seed 50003's late draw 0).

**The rate rule** (``rate_choice``), fixed before the pilot: the pilot runs ``PILOT_SEEDS`` in their
stream order at the ``GRID`` rates with a K-only evaluation. A run gives a value when its launcher
end is ``done`` (``ACCEPTED_END``: never ``diverged``, ``conflict`` or ``technical``) and it ended
all 25 updates with a finite, positive K_benign at step 25. For each pilot seed, the first pair of
consecutive grid rates (in increasing rate) whose two values bracket the seed's target
(``TARGETS_FILE``, inclusive, the two values different) gives η_s by linear interpolation in
(ln η, ln K_benign). η₀ is the geometric mean of the two η_s, rounded to three significant
digits. A pilot seed without a bracket leaves no η₀: the experiment stops before any main run.

**The drift gate** (``drift_gate``): the six stream-order runs at η₀ pass when all six give a value
(as a pilot run does) and the geometric mean over the seeds of K_benign / target lies in ``GATE``
(inclusive). The campaign checks the verdict again from the runs at every start before any other
order (``scripts/study5_beta0_campaigns.py check-gate``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .seeding import derive_seed

CONFIG = "experiments/study5_beta0"
PILOT_CONFIG = "experiments/study5_beta0_k_only"
SEEDS = (50000, 50001, 50002, 50003, 50004, 50005)  # extension 2's control seeds (M6)
PILOT_SEEDS = (50000, 50001)
# × η_l = 1.3e-05 and the rate; the grid brackets the first-order guess 1.18 × η_l both ways
GRID = (("0.71", 9.2e-06), ("1.0", 1.3e-05), ("1.41", 1.83e-05), ("2.0", 2.6e-05))
STEPS = 25
GATE = (0.8, 1.25)
ACCEPTED_END = "done"  # the launcher end of a run whose K counts
# K_benign at step 25 of each seed's frozen-Adam (β₁ = 0.9) stream-order run at η_l: study 5's
# fresh runs (pod 2), read from their evaluations.jsonl
TARGET_RUNS = {
    50000: "20261109T192912Z-s5-fresh-1.3e-05-s50000-s50000-f3e969f1-a1",
    50001: "20261109T200448Z-s5-fresh-1.3e-05-s50001-s50001-52976e1f-a1",
    50002: "20261109T204049Z-s5-fresh-1.3e-05-s50002-s50002-99efec8b-a1",
    50003: "20261109T211734Z-s5-fresh-1.3e-05-s50003-s50003-21cff731-a1",
    50004: "20261109T215417Z-s5-fresh-1.3e-05-s50004-s50004-c431ca6c-a1",
    50005: "20261109T223154Z-s5-fresh-1.3e-05-s50005-s50005-d5d976a3-a1",
}
WINDOWS = (("early", 1, 5), ("late", 21, 25))
DRAWS = (0, 1)
ORDERS = ("stream", "reorder", "early-0", "early-1", "late-0", "late-1")
TARGETS_FILE = "docs/study5-beta0-targets.json"
REFERENCE_FILE = "docs/study5-beta0-reference.json"


def shuffle_permutation(seed: int, first: int, last: int, draw: int) -> list[int]:
    """The permutation of batches ``first`` … ``last`` that ``study5_options.edited_order`` draws
    for a shuffle (tests/test_study5_beta0.py checks the two agree)."""
    rng = np.random.default_rng(derive_seed(seed, "study5", "shuffle", first, last, draw))
    return [int(i) for i in rng.permutation(last - first + 1)]


def is_identity(seed: int, first: int, last: int, draw: int) -> bool:
    return shuffle_permutation(seed, first, last, draw) == list(range(last - first + 1))


def edit_of(order: str) -> tuple[int, int, int] | None:
    """(first, last, draw) of a shuffle order name, None for the stream order and the reorder."""
    for name, first, last in WINDOWS:
        for draw in DRAWS:
            if order == f"{name}-{draw}":
                return first, last, draw
    if order in ("stream", "reorder"):
        return None
    raise ValueError(f"no order {order!r}")


def orders_of(seed: int) -> list[str]:
    """The seed's orders that are run: ``ORDERS`` without an identity shuffle."""
    out = []
    for order in ORDERS:
        edit = edit_of(order)
        if edit is not None and is_identity(seed, *edit):
            continue
        out.append(order)
    return out


def rate_override(lr: float) -> str:
    return f"optimizer.lr={lr!r}"


def order_overrides(seed: int, order: str) -> list[str]:
    edit = edit_of(order)
    if order == "reorder":
        return [f"+train.order_seed={seed}"]
    if edit is None:
        return []
    first, last, draw = edit
    return [f"+train.order_edit={{kind: shuffle, first: {first}, last: {last}, draw: {draw}}}"]


def pilot_cell(seed: int, level: str) -> dict[str, Any]:
    """One pilot run: ``seed`` in its stream order at the ``GRID`` level ``level``, K only."""
    lr = dict(GRID)[level]
    return {"name": f"s5-b0-pilot-{lr!r}-s{seed}", "config": PILOT_CONFIG,
            "overrides": [rate_override(lr)], "seeds": [seed]}  # fmt: skip


def cell(seed: int, order: str, lr: float) -> dict[str, Any]:
    """One main run: ``seed`` in ``order`` (``ORDERS``) at η₀ = ``lr``."""
    if order not in orders_of(seed):
        raise ValueError(f"seed {seed} runs no order {order!r}")
    return {"name": f"s5-b0-{order}-{lr!r}-s{seed}", "config": CONFIG,
            "overrides": [rate_override(lr), *order_overrides(seed, order)],
            "seeds": [seed]}  # fmt: skip


def targets_from(record: Mapping[str, Any]) -> dict[int, float]:
    """The targets of the committed record, checked against ``TARGET_RUNS``."""
    runs = {int(s): r for s, r in record["runs"].items()}
    if runs != TARGET_RUNS:
        raise ValueError(f"{TARGETS_FILE}: its runs are not TARGET_RUNS")
    return {int(s): float(k) for s, k in record["k_benign"].items()}


def _value(row: Mapping[str, Any]) -> float | None:
    k = row.get("K")
    ok = (row.get("outcome") == ACCEPTED_END and row.get("complete") is True
          and isinstance(k, int | float) and not isinstance(k, bool)
          and math.isfinite(k) and k > 0)  # fmt: skip
    return float(k) if ok else None


def interpolate(eta_a: float, k_a: float, eta_b: float, k_b: float, target: float) -> float:
    """η at which ln K reaches ln target on the line through (ln η_a, ln K_a), (ln η_b, ln K_b)."""
    t = (math.log(target) - math.log(k_a)) / (math.log(k_b) - math.log(k_a))
    return math.exp(math.log(eta_a) + t * (math.log(eta_b) - math.log(eta_a)))


def rate_choice(rows: Iterable[Mapping[str, Any]], targets: Mapping[int, float]) -> dict[str, Any]:
    """η₀ from the pilot (module docstring, "The rate rule"). ``rows``: one per pilot run, with
    ``seed``, ``level`` (a ``GRID`` name), ``K`` and ``complete``; exactly the pilot seeds × the
    grid are expected (``ValueError`` otherwise)."""
    table: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["level"]))
        if key in table:
            raise ValueError(f"two pilot runs of {key}")
        table[key] = row
    expected = {(s, level) for s in PILOT_SEEDS for level, _ in GRID}
    if set(table) != expected:
        raise ValueError(f"the pilot's {len(expected)} runs are expected, got {sorted(table)}")
    grid = sorted(GRID, key=lambda item: item[1])
    per_seed: dict[str, Any] = {}
    for seed in PILOT_SEEDS:
        target = float(targets[seed])
        values = [(level, lr, _value(table[(seed, level)])) for level, lr in grid]
        found = None
        for (la, ea, ka), (lb, eb, kb) in zip(values, values[1:], strict=False):
            if ka is None or kb is None or ka == kb:
                continue
            if min(ka, kb) <= target <= max(ka, kb):
                found = {"levels": [la, lb], "eta_s": interpolate(ea, ka, eb, kb, target)}
                break
        per_seed[str(seed)] = {"target": target, "K": {lv: k for lv, _, k in values},
                               "bracket": found}  # fmt: skip
    etas = [v["bracket"]["eta_s"] for v in per_seed.values() if v["bracket"] is not None]
    if len(etas) != len(PILOT_SEEDS):
        return {"per_seed": per_seed, "eta0": None,
                "stop": "a pilot seed's target is not bracketed by the grid"}  # fmt: skip
    eta0 = float(f"{math.exp(sum(math.log(e) for e in etas) / len(etas)):.3g}")
    return {"per_seed": per_seed, "eta0": eta0, "stop": None}


def drift_gate(rows: Iterable[Mapping[str, Any]], targets: Mapping[int, float]) -> dict[str, Any]:
    """The drift gate (module docstring) on the stream-order runs at η₀: ``rows`` one per seed
    with ``seed``, ``K`` and ``complete``."""
    found = {int(r["seed"]): _value(r) for r in rows}
    if sorted(found) != list(SEEDS):
        raise ValueError(f"the six stream-order runs are expected, got {sorted(found)}")
    missing = [s for s, k in found.items() if k is None]
    if missing:
        return {"passed": False, "ratio": None, "missing": missing,
                "why": "a stream-order run without a value"}  # fmt: skip
    ratio = math.exp(sum(math.log(found[s] / float(targets[s])) for s in SEEDS) / len(SEEDS))
    passed = GATE[0] <= ratio <= GATE[1]
    return {"passed": passed, "ratio": ratio, "missing": [],
            "why": None if passed else f"ratio {ratio:.4f} outside {GATE}"}  # fmt: skip


def reference_record(ext2: Mapping[str, Any], ext2_sha256: str) -> dict[str, Any]:
    """The frozen-Adam reference (``REFERENCE_FILE``) from extension 2's labeled record: per seed
    the stream-order run's h and the Δh of the reorder (``m2.control_pairs``) and of each shuffle
    (``m6.shuffles.rows``, an identity permutation left out by ``is_identity``), in pp; and the
    means F, E, L, D = E − L over the edits and h̄ over the stream-order runs."""
    shuffles = ext2["m6"]["shuffles"]["rows"]
    pairs = ext2["m2"]["control_pairs"]
    seeds: dict[str, Any] = {}
    for seed in SEEDS:
        rows = [r for r in shuffles if int(r["seed"]) == seed]
        stream = {round(float(r["stream"]["harmful"]), 12) for r in rows}
        if len(stream) != 1:
            raise ValueError(f"seed {seed}: its shuffles name {len(stream)} stream-order runs")
        edits: dict[str, Any] = {}
        for r in rows:
            e = r["edit"]
            name = next(n for n, first, _ in WINDOWS if first == e["first"])
            if is_identity(seed, e["first"], e["last"], e["draw"]):
                continue
            d = r["outcome_difference"]
            edits[f"{name}-{e['draw']}"] = {"dh": 100 * d["harmful"],
                                            "dK_harmful": d["K_harmful"],
                                            "dK_benign": d["K_benign"]}  # fmt: skip
        pair = pairs[str(seed)]
        edits["reorder"] = {
            "dh": 100 * pair["harm_difference"],
            "dK_harmful": pair["K_harmful_difference"],
            "dK_benign": None,
        }
        if sorted(edits) != sorted(o for o in orders_of(seed) if o != "stream"):
            raise ValueError(f"seed {seed}: edits {sorted(edits)} are not its orders")
        seeds[str(seed)] = {"h_stream": 100 * stream.pop(), "edits": edits}
    return {"what": "The frozen-Adam (beta1 0.9) reference at eta_l of the run without momentum "
                    "(docs/study5-beta0-registration.md): Δh = h(order) − h(stream order), pp.",
            "source": {"record": "extension 2's labeled analysis record (m2.control_pairs, "
                                 "m6.shuffles.rows)", "sha256": ext2_sha256},
            "seeds": seeds, **summary_of(seeds)}  # fmt: skip


def summary_of(seeds: Mapping[str, Any]) -> dict[str, float]:
    """F, E, L (mean |Δh| over the reorders, early and late shuffles), D = E − L and h̄."""

    def mean_abs(kind: str) -> float:
        xs = [abs(v["dh"]) for s in seeds.values() for o, v in s["edits"].items()
              if (o == "reorder" if kind == "reorder" else o.startswith(kind + "-"))]  # fmt: skip
        return sum(xs) / len(xs)

    f, e, late = mean_abs("reorder"), mean_abs("early"), mean_abs("late")
    h_bar = sum(s["h_stream"] for s in seeds.values()) / len(seeds)
    return {"F": f, "E": e, "L": late, "D": e - late, "h_bar": h_bar}
