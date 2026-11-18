"""Study 5's run without momentum (docs/study5-beta0-registration.md, "Analysis") as pure functions
of the runs' readings and label counts. Nothing here reads a file:
``scripts/study5_beta0_analysis.py`` reads the run directories, the label files, the rate record
and the frozen-Adam reference and passes their numbers in.

**A run's readings** (``Run``) are those of ``analysis.study5_adamw``: ``run_id``, ``seed``,
``order`` (``beta0.ORDERS``), ``records`` (None, or why they do not read), ``stop``, ``steps``,
``row`` (its step-25 evaluation) and ``labels`` (counts at the step-25 condition, or why absent).
h is ``study5_dose.harm``: harmful compliance at step 25 over 521 in pp, for a run that ended all 25
updates and whose labels read.

**Statistics** over the design (``beta0.SEEDS`` × ``beta0.orders_of``): Δh(s, o) = h(s, o) −
h(s, stream order) for each edited order with both h. F₀, E₀ and L₀ are the means of |Δh| over the
reorders, the shuffles of batches 1–5 and of batches 21–25; D₀ = E₀ − L₀. N₀ is the label-noise
value 100·√(2 h̄₀(1 − h̄₀)/521)·√(2/π) at h̄₀, the mean h of the stream-order runs with h (as a
share). Each has a 95 % interval from ``RESAMPLES`` stream-level bootstrap resamples: the seeds
drawn with replacement by ``numpy.random.default_rng(BOOTSTRAP_SEED)`` as one (10 000, 6) array, a
seed's edits entering together; a statistic of a resample without any of its edits is left out of
that statistic's percentiles; the 2.5 and 97.5 percentiles (numpy's linear).

**A run's state** (``run_state``): ``ok`` when it has h; ``stopped`` when the scheduler stopped it
for one of the protocol's instability reasons (``study3_records.INSTABILITY``) after a whole number
of updates from 1 to 25, the 25th included (a scientific outcome, never rerun); ``technical``
otherwise (no run; records or labels that do not read; any other stop reason, an interruption
among them; a missing or invalid number of updates). An edit's state is the worse of its run's
and its stream order's (technical over stopped over ok). Under the addendum of 2026-11-18 an edit
without a run whose stream-order run is ``stopped`` is ``not_run``: it was not attempted because
its baseline stopped (a structural exclusion, never a technical gap, never an invented stopped
run); its seed stays in the bootstrap population.

**The outcome** (``outcome``), on the point estimates, with F the reference's F and h̄ its h̄. T is
F/2, multiplied by h̄₀/h̄ when that ratio lies outside [0.5, 2]. The regions are mutually exclusive
and taken in this order:

- (ii) F₀ ≤ 2 N₀ and E₀ ≤ 2 N₀;
- (i) F₀ > 2 N₀, F₀ ≥ T and D₀ ≥ 2 N₀;
- (iii) F₀ > 2 N₀, F₀ ≥ T and D₀ < 2 N₀;
- ``reduced`` otherwise.

An outcome among (i)–(iii) is ``clear`` when the intervals of its statistics lie in the same region
(for (ii) both upper bounds ≤ 2 N₀; for (i) and (iii) F₀'s lower bound > 2 N₀ and ≥ T, and D₀'s
lower bound ≥ 2 N₀, or upper bound < 2 N₀) and every class those statistics use (F and E for (ii),
F, E and L for (i) and (iii)) has edits with Δh from at least ``MIN_SEEDS_CLEAR`` seeds; it is
``leaning`` otherwise.

- *Technical gaps*: any edit or stream-order run of the design in the state ``technical`` withholds
  the outcome (``withheld``); the statistics are still reported, with the seeds that contribute to
  each class.
- *Scientific stops*: edits in the state ``stopped`` or ``not_run`` enter no statistic; the
  outcome is read on the rest and marked ``conditional``, with the stopped edits, the edits not
  run and the stopped stream-order runs listed.
- Without F₀, E₀ or L₀ (no edit with Δh in a class) the outcome is ``undetermined``.

**The amended drift gate** (``amended_gate``, the addendum of 2026-11-18; ``beta0.drift_gate``
keeps the registered rule, recorded beside it): the six stream-order runs at η₀ pass when at least
``GATE_MIN_VALUES`` of them give a value (as a pilot run does), every one without a value ended in
a scientific stop (``gate_stop``: launcher end ``done`` or ``diverged``, an instability reason of
``INSTABILITY`` and a whole number of updates from 1 to 25), and the geometric mean of K_benign /
target over the runs with a value lies in ``beta0.GATE``. A run without a value and without such a
stop (a technical gap) fails the gate.

**Common survivors** (supplementary, fixed by the addendum before any edit ran): the seeds whose
every early and late edit has Δh; E, L and D over them alone, with the seeds and, per class, the
seeds that contribute and the edits stopped or not run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .. import beta0
from . import study5_dose as sd
from .study3_records import INSTABILITY

Run = Mapping[str, Any]
Key = tuple[int, str]  # (seed, order)
BOOTSTRAP_SEED = 20261118
RESAMPLES = 10_000
PERCENTILES = (2.5, 97.5)
DENOMINATOR = 521
SCALE_BOUNDS = (0.5, 2.0)
CLASSES = {"F": "reorder", "E": "early", "L": "late"}
MIN_SEEDS_CLEAR = 5  # a clear outcome rests on at least five of the six seeds in each class
STATE_RANK = {"ok": 0, "not_run": 1, "stopped": 1, "technical": 2}
GATE_MIN_VALUES = 5  # the addendum of 2026-11-18: at least five of the six stream-order runs
SCIENTIFIC_ENDS = ("done", "diverged")  # launcher ends of a run that stopped as an outcome


def class_of(order: str) -> str | None:
    if order == "reorder":
        return "F"
    for name, tag in (("early", "E"), ("late", "L")):
        if order.startswith(name + "-"):
            return tag
    return None


def design() -> list[Key]:
    return [(s, o) for s in beta0.SEEDS for o in beta0.orders_of(s)]


def run_state(run: Run | None) -> str:
    """``ok``, ``stopped`` or ``technical`` (module docstring, "A run's state")."""
    if _number(sd.harm(run)):
        return "ok"
    if run is None or run.get("records") is not None:
        return "technical"
    steps = run.get("steps")
    valid = isinstance(steps, int) and not isinstance(steps, bool) and 1 <= steps <= beta0.STEPS
    return "stopped" if run.get("stop") in INSTABILITY and valid else "technical"


def not_run(runs: Mapping[Key, Run], seed: int, order: str) -> bool:
    """An edit without a run whose stream-order run stopped (module docstring)."""
    return (order != "stream" and runs.get((seed, order)) is None
            and run_state(runs.get((seed, "stream"))) == "stopped")  # fmt: skip


def deltas(runs: Mapping[Key, Run]) -> dict[str, Any]:
    """Per edited order of the design: its state, Δh, ΔK_harmful and ΔK_benign against the stream
    order, or why absent."""
    out: dict[str, Any] = {}
    for seed, order in design():
        if order == "stream":
            continue
        key = f"s{seed}/{order}"
        run_o, run_s = runs.get((seed, order)), runs.get((seed, "stream"))
        if not_run(runs, seed, order):
            out[key] = {"seed": seed, "order": order, "state": "not_run", "dh": None,
                        "why": "not run: its stream-order run stopped"}  # fmt: skip
            continue
        state = max(run_state(run_o), run_state(run_s), key=STATE_RANK.__getitem__)
        if state != "ok":
            h_o = sd.harm(run_o)
            out[key] = {"seed": seed, "order": order, "state": state, "dh": None,
                        "why": h_o if not _number(h_o) else sd.harm(run_s)}  # fmt: skip
            continue
        h_o, h_s = sd.harm(run_o), sd.harm(run_s)
        row_o = runs[(seed, order)].get("row") or {}
        row_s = runs[(seed, "stream")].get("row") or {}
        out[key] = {"seed": seed, "order": order, "state": "ok", "dh": h_o - h_s, "why": None,
                    "dK_harmful": _diff(row_o.get("K_harmful"), row_s.get("K_harmful")),
                    "dK_benign": _diff(row_o.get("K_benign"), row_s.get("K_benign"))}  # fmt: skip
    return out


def _class_mean(entries: Sequence[Mapping[str, Any]], tag: str, seeds: Sequence[int],
                field: str = "dh") -> float | None:  # fmt: skip
    xs = [abs(e[field]) for s in seeds for e in entries
          if e["seed"] == s and class_of(e["order"]) == tag and _number(e.get(field))]  # fmt: skip
    return sum(xs) / len(xs) if xs else None


def seeds_per_class(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many seeds have an edit with Δh in each class."""
    return {tag: len({e["seed"] for e in entries if class_of(e["order"]) == tag
                      and _number(e.get("dh"))}) for tag in CLASSES}  # fmt: skip


def statistics_of(entries: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    f, e, late = (_class_mean(entries, tag, seeds) for tag in ("F", "E", "L"))
    d = e - late if e is not None and late is not None else None
    return {"F": f, "E": e, "L": late, "D": d}


def common_survivors(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The supplementary comparison on the common survivors (module docstring)."""
    seeds = [s for s in beta0.SEEDS
             if all(_number(e.get("dh")) for e in entries
                    if e["seed"] == s and class_of(e["order"]) in ("E", "L"))]  # fmt: skip
    usable = [e for e in entries if e["state"] == "ok"]
    stats = statistics_of(usable, seeds)
    per_class = {
        tag: {"seeds": len({e["seed"] for e in usable if class_of(e["order"]) == tag}),
              "stopped": sorted(f"s{e['seed']}/{e['order']}" for e in entries
                                if class_of(e["order"]) == tag and e["state"] == "stopped"),
              "not_run": sorted(f"s{e['seed']}/{e['order']}" for e in entries
                                if class_of(e["order"]) == tag and e["state"] == "not_run")}
        for tag in CLASSES
    }  # fmt: skip
    return {"seeds": seeds, "E": stats["E"], "L": stats["L"], "D": stats["D"],
            "per_class": per_class}  # fmt: skip


def gate_stop(row: Mapping[str, Any]) -> bool:
    """A stream-order run at η₀ that ended in a scientific stop (module docstring, "The amended
    drift gate"); ``row`` as the campaign's gate rows: ``outcome``, ``stop``, ``steps``."""
    steps = row.get("steps")
    valid = isinstance(steps, int) and not isinstance(steps, bool) and 1 <= steps <= beta0.STEPS
    return row.get("outcome") in SCIENTIFIC_ENDS and row.get("stop") in INSTABILITY and valid


def amended_gate(rows: Sequence[Mapping[str, Any]], targets: Mapping[int, float]) -> dict[str, Any]:
    """The amended drift gate (module docstring) on the six stream-order runs at η₀."""
    by_seed = {int(r["seed"]): r for r in rows}
    if sorted(by_seed) != list(beta0.SEEDS) or len(rows) != len(beta0.SEEDS):
        raise ValueError(f"the six stream-order runs are expected, got {sorted(by_seed)}")
    found = {s: beta0._value(r) for s, r in by_seed.items()}
    missing = sorted(s for s, k in found.items() if k is None)
    stopped = [s for s in missing if gate_stop(by_seed[s])]
    technical = [s for s in missing if s not in stopped]
    seeds = sorted(s for s, k in found.items() if k is not None)
    base = {"missing": missing, "stopped": stopped, "technical": technical, "seeds": seeds}
    if technical:
        why = "a stream-order run without a value and without a scientific stop"
        return {"passed": False, "ratio": None, **base, "why": why}
    if len(seeds) < GATE_MIN_VALUES:
        return {"passed": False, "ratio": None, **base,
                "why": f"fewer than {GATE_MIN_VALUES} stream-order runs with a value"}  # fmt: skip
    ratio = math.exp(sum(math.log(found[s] / float(targets[s])) for s in seeds) / len(seeds))
    passed = beta0.GATE[0] <= ratio <= beta0.GATE[1]
    return {"passed": passed, "ratio": ratio, **base,
            "why": None if passed else f"ratio {ratio:.4f} outside {beta0.GATE}"}  # fmt: skip


def label_noise(h_bar_pp: float) -> float:
    h = h_bar_pp / 100
    return 100 * math.sqrt(2 * h * (1 - h) / DENOMINATOR) * math.sqrt(2 / math.pi)


def bootstrap(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each statistic's 95 % interval over stream-level resamples (module docstring)."""
    seeds = list(beta0.SEEDS)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(seeds), (RESAMPLES, len(seeds)))
    found: dict[str, list[float]] = {name: [] for name in ("F", "E", "L", "D")}
    for row in draws:
        resample = [seeds[int(i)] for i in row]
        stats = statistics_of(entries, resample)
        for name, value in stats.items():
            if value is not None:
                found[name].append(value)
    out = {}
    for name, values in found.items():
        if values:
            lo, hi = np.percentile(np.array(values), PERCENTILES)
            out[name] = {"lower": float(lo), "upper": float(hi), "resamples": len(values)}
        else:
            out[name] = {"lower": None, "upper": None, "resamples": 0}
    return out


def outcome(stats: Mapping[str, Any], intervals: Mapping[str, Any], noise: float,
            reference: Mapping[str, float], h_bar0: float,
            seeds: Mapping[str, int] | None = None) -> dict[str, Any]:  # fmt: skip
    """The registered outcome and its strength (module docstring), before any technical gap or
    scientific stop is taken into account (``analysis``). ``seeds``: the seeds per class
    (``seeds_per_class``); all six when not given."""
    seeds = dict(seeds) if seeds is not None else dict.fromkeys(CLASSES, len(beta0.SEEDS))
    half = reference["F"] / 2
    ratio = h_bar0 / reference["h_bar"]
    scaled = not (SCALE_BOUNDS[0] <= ratio <= SCALE_BOUNDS[1])
    if scaled:
        half *= ratio
    two_n = 2 * noise
    f, e, d = stats["F"], stats["E"], stats["D"]
    base = {"threshold_F": half, "two_N": two_n, "h_ratio": ratio, "scaled": scaled}
    if f is None or e is None or d is None:
        return {**base, "outcome": "undetermined", "strength": None}

    def bounds(name: str) -> tuple[float, float] | None:
        iv = intervals[name]
        return None if iv["lower"] is None else (iv["lower"], iv["upper"])

    def enough(classes: str) -> bool:
        return all(seeds.get(tag, 0) >= MIN_SEEDS_CLEAR for tag in classes)

    fb, eb, db = bounds("F"), bounds("E"), bounds("D")
    if f <= two_n and e <= two_n:
        name = "(ii)"
        clear = (fb is not None and eb is not None and fb[1] <= two_n and eb[1] <= two_n
                 and enough("FE"))  # fmt: skip
    elif f > two_n and f >= half:
        persists = fb is not None and fb[0] > two_n and fb[0] >= half and enough("FEL")
        if d >= two_n:
            name, clear = "(i)", persists and db is not None and db[0] >= two_n
        else:
            name, clear = "(iii)", persists and db is not None and db[1] < two_n
    else:
        return {**base, "outcome": "reduced", "strength": None}
    return {**base, "outcome": name, "strength": "clear" if clear else "leaning"}


def analysis(runs: Mapping[Key, Run], reference: Mapping[str, Any], eta0: float) -> dict[str, Any]:
    """Every quantity of the record (module docstring)."""
    entries = list(deltas(runs).values())
    usable = [e for e in entries if e["state"] == "ok"]
    stats = statistics_of(usable, beta0.SEEDS)
    intervals = bootstrap(usable)
    seeds = seeds_per_class(usable)
    h_stream = {s: sd.harm(runs.get((s, "stream"))) for s in beta0.SEEDS}
    hs = [h for h in h_stream.values() if _number(h)]
    h_bar0 = sum(hs) / len(hs) if hs else None
    noise = label_noise(h_bar0) if h_bar0 is not None else None
    technical = sorted(
        f"s{s}/{o}"
        for s, o in design()
        if run_state(runs.get((s, o))) == "technical" and not not_run(runs, s, o)
    )
    stopped = sorted(f"s{e['seed']}/{e['order']}" for e in entries if e["state"] == "stopped")
    skipped = sorted(f"s{e['seed']}/{e['order']}" for e in entries if e["state"] == "not_run")
    stopped_stream = sorted(
        f"s{s}/stream" for s in beta0.SEEDS if run_state(runs.get((s, "stream"))) == "stopped"
    )
    if technical:
        result: dict[str, Any] = {"outcome": "withheld", "strength": None,
                                  "why": "technical gaps"}  # fmt: skip
    elif noise is None:
        result = {"outcome": "undetermined", "strength": None}
    else:
        result = outcome(stats, intervals, noise, reference, h_bar0, seeds)
    result |= {"conditional": bool(stopped or skipped or stopped_stream) and not technical,
               "technical": technical, "stopped": stopped, "not_run": skipped,
               "stopped_stream": stopped_stream, "seeds_per_class": seeds}  # fmt: skip
    paired = {}
    for e in usable:
        ref = reference["seeds"][str(e["seed"])]["edits"].get(e["order"])
        frozen = abs(ref["dh"]) if ref else None
        paired[f"s{e['seed']}/{e['order']}"] = {
            "abs_dh_beta0": abs(e["dh"]),
            "abs_dh_frozen": frozen,
        }
    return {
        "eta0": eta0,
        "design": [f"s{s}/{o}" for s, o in design()],
        "missing": [f"s{s}/{o}" for s, o in design() if (s, o) not in runs],
        "h_stream": {str(s): h for s, h in h_stream.items()},
        "h_bar0": h_bar0,
        "label_noise": noise,
        "deltas": {f"s{e['seed']}/{e['order']}": e for e in entries},
        "statistics": stats,
        "intervals": intervals,
        "reference": {k: reference[k] for k in ("F", "E", "L", "D", "h_bar")},
        "result": result,
        "descriptive": {
            "E_over_F": stats["E"] / stats["F"] if stats["E"] is not None and stats["F"] else None,
            "mean_abs_dK_harmful": {
                t: _class_mean(usable, t, beta0.SEEDS, "dK_harmful") for t in CLASSES
            },
            "mean_abs_dK_benign": {
                t: _class_mean(usable, t, beta0.SEEDS, "dK_benign") for t in CLASSES
            },
            "paired": paired,
            "common_survivors": common_survivors(entries),
            "mmlu_stream": {
                str(s): ((runs.get((s, "stream")) or {}).get("row") or {}).get("benchmark_accuracy")
                for s in beta0.SEEDS
            },
        },  # fmt: skip
    }


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _diff(a: Any, b: Any) -> float | None:
    return float(a) - float(b) if _number(a) and _number(b) else None
