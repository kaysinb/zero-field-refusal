"""Study 5's prospective test on E2's order repeats (docs/study5-preregistration.md, "Extension 1"
E2 and "Extension 2" M2: the δ_lin criterion selects the predictor; the test itself is fixed in a
dated note before any E2 outcome is read), written once as one JSON record.

    python -m zero_field_refusal.analysis.study5_e2_test --open-e2 OPENING.json
        --criterion EXT2_ANALYSIS.json [--criterion-key KEY] [--predictions PATH ...]
        --pod3-runs DIR --registered ANALYSIS.json --out RECORD.json [--seeds S ...]

It sits in the analysis package, outside the GPU code identity, as ``study5_ext1`` does, and reads
E2 through ``study5_ext1``'s extraction (``order_parts``: the runs, the registered support rule and
interpolation at K*, ln K_harmful at K*). Importing it loads no torch; the label readers load it
when they read, on a CPU.

**The test.** For each tested seed, its two new orders 1 and 2 (``train.order_seed`` = s + 1000
and s + 2000), each at both rates. The seed's original order (pod 2's fresh runs) is not part of
it. The predictor is the δ_lin criterion record's ``primary_predictor``:
- ``q``: q_o of each new order from the sealed predictions (δ_lin's q at η_l, ``q_lin``). Lower q
  predicts more harm, so the predicted sign of h₁ − h₂ is sign(q₂ − q₁).
- ``ln_K_harmful_at_K*``: ln K_harmful at K* of each order, measured on the E2 runs (both rates,
  the interpolation of h at K*). Higher ln K_harmful goes with more harm: the predicted sign of
  h₁ − h₂ is sign(ln K_h,1 − ln K_h,2). This is an association, not a prediction from θ₀, and the
  record says so.

The outcome is h at K* of each order (study 3's support rule and interpolation, as registered). Per
seed, A_s = sign(predicted) · sign(h₁ − h₂) ∈ {−1, 0, +1}; a zero in either sign is a tie, left
out and counted. With n the seeds with A_s ≠ 0 and T those with A_s = +1, the one-sided p is
P(Bin(n, ½) ≥ T), exact; for n ≤ 20 the 2ⁿ sign patterns are also enumerated and must give the
same count. Significant when p < 0.05. Beside it, descriptive: S = Σ_s sign(predicted)·(h₁ − h₂)
with its one-sided sign-flip p (the 2ⁿ patterns for n ≤ 20; above, 1 000 000 random patterns from
``numpy.random.default_rng(20261114)``, as the record states), and the per-seed table. A seed with
an order without h (unsupported, a run missing or unlabeled), or without its predictor's value
(δ_lin diverged; ln K_harmful not available), is left out and listed with the reason.

**The secondary outcome** is the harm share of each order's η_l run (its step-25 labels, a missing
label not harmful, the all-harmful value beside; no interpolation), with the same statistic. It
is reported beside and decides nothing. The natural secondary outcome, Δr at the endpoint, cannot
be computed for the E2 runs: they are control runs (``experiments/study5_control_order``), which
save no checkpoint by the registration, so their endpoint is not there to score.

**Top-up seeds** (``--seeds``, added to the ten registered seeds 50006-50015): any of 50000-50005
and 50016-50039, with the same two new orders at both rates, their runs in the same root under the
same campaign and cell names. Seeds 50000-50005 were seen before: in the exploratory checks of
extension 1 and in the δ_lin criterion, with ``train.order_seed`` = s only. Their new orders were
never observed, and the test compares two new orders within a seed, so the seed's own effect
cancels.

**Inputs.**
- --open-e2: the opening record, required (``study5_ext1.opening_record``), checked before
  anything of E2 is read.
- --criterion: the extension-2 analysis record, its δ_lin criterion under --criterion-key (a
  dotted path, default ``criterion``; empty for the top level). Only ``status``, ``passed`` and
  ``primary_predictor`` are read: ``status`` ``decided``, ``passed`` a boolean and
  ``primary_predictor`` the one it selects (passed: ``q``; not passed:
  ``ln_K_harmful_at_K*``). A withheld criterion (``status`` ``withheld`` or ``passed`` null) is
  refused by name, and so is a disagreement of the two.
- --predictions: the sealed predictions, required for ``q`` (optional otherwise, then reported
  beside only). Each PATH is a record or a folder whose ``*/record.json`` are read (pod 4's
  ``study5_records/<ID>/sealed/s<seed>-order<k>/record.json``,
  ``scripts/study5_linear_replay.py``'s record). Each is one order: kind
  ``study5-linear-replay``, its seed, η_l, ``order.order_seed`` and ``divergence.status``
  (``complete``, with a finite ``q_lin``, or ``diverged``). Two records of one order are refused.
  With ``q``, a tested order without a record, or complete without a finite q, is refused. The
  prediction must be of the order's runs: its order record, v₀, θ₀ and training stream those each
  run recorded, refused otherwise.
- --pod3-runs: the runs root holding the order runs and their launcher journals; a seed's
  campaign is its one journal ``launch/study5-<ID>-orders-s<seed>.jsonl``.
- --registered: the registered analysis's record, its sha256 named by the registration at this
  commit (``study5_ext1.registered_record``). θ₀'s accuracies (the gross-collapse check of the
  support rule) are its line scans' point a = 0, θ₀'s evaluation; the prompt sets must be its own.

**Rulings** where the test leaves a detail open:
- *A sign* is that of the exact difference; an exact zero is a tie.
- *The sign-flip p* counts the patterns with Σ ε_s |d_s| ≥ S − 10⁻¹² (the observed one always
  counts): enumerated, p = count / 2ⁿ; sampled, p = (1 + count) / (1 + 1 000 000), the
  registered permutation formula, the patterns ``integers(0, 2, size=(1 000 000, n), int8)``
  (1 is +1).
- *A seed without a single launcher journal* of its orders has both orders missing, by name.

**The record** (--out, refused when it exists): strict JSON, numbers, ids, states and reasons,
never a text; every input's sha256. stdout carries numbers and the record's sha256 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import study5_analysis as sa
from . import study5_ext1 as ext
from .study3_estimates import DENOMINATOR

PROG = "study5_e2_test"
KIND, VERSION = "study5-e2-test", 1
SEEDS = ext.E2_SEEDS  # 50006-50015
TOP_UP = (*range(50000, 50006), *range(50016, 50040))
ORDERS = ext.NEW_ORDERS  # (order1, 1000), (order2, 2000)
Q, LN_K = "q", "ln_K_harmful_at_K*"
PREDICTORS = (Q, LN_K)
PRIMARY = {True: Q, False: LN_K}  # the criterion passed selects q, else ln K_harmful at K*
CRITERION_KEY = "criterion"  # the extension-2 analysis record's criterion (study5_ext2.criterion)
ALPHA = 0.05
ENUMERATED = 20  # n up to which the 2ⁿ sign patterns are enumerated
PATTERNS, PATTERN_SEED, CHUNK = 1_000_000, 20261114, 100_000
TIE = sa.TIE  # 1e-12
REPLAY_KIND = "study5-linear-replay"  # scripts/study5_linear_replay.py's record
STATUSES = ("complete", "diverged")
PREDICTION_KEYS = ("file", "sha256", "status", "q")  # a prediction, in the per-seed table
READINGS = {
    Q: "a prediction from theta0 and the order: lower q_lin predicts more harm",
    LN_K: "an association, not a prediction from theta0: ln K_harmful at K* is measured on the "
    "same E2 runs; higher ln K_harmful goes with more harm",
}
SECONDARY = (
    "the harm share of each order's eta_l run at its step-25 evaluation (no interpolation), the "
    "same statistic; it decides nothing. Delta r at the endpoint cannot be computed for the E2 "
    "runs: control runs (experiments/study5_control_order) save no checkpoint by the registration"
)
TOP_UP_NOTE = (
    "seeds 50000-50005 were seen in the exploratory checks of extension 1 and in the delta_lin "
    "criterion with train.order_seed = s only; their new orders were never observed, and the test "
    "compares two new orders within a seed, so the seed's own effect cancels"
)

refuse = ext.refuse


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ the statistic


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def binomial_count(n: int, t: int) -> int:
    """The number of the 2ⁿ sign patterns with at least ``t`` plus signs: Σ_{k ≥ t} C(n, k)."""
    return sum(math.comb(n, k) for k in range(max(t, 0), n + 1))


def enumerated_count(n: int) -> np.ndarray:
    """The plus signs of each of the 2ⁿ patterns, enumerated."""
    plus = np.zeros(1, dtype=np.int64)
    for _ in range(n):
        plus = np.concatenate((plus + 1, plus))
    return plus


def sign_flip(magnitudes: Sequence[float], observed: float) -> dict[str, Any]:
    """The one-sided sign-flip p of S = ``observed`` over the patterns Σ ε_s·|d_s| (module
    docstring, "Rulings")."""
    d = np.abs(np.asarray(magnitudes, dtype=float))
    n = len(d)
    if n <= ENUMERATED:
        sums = np.zeros(1)
        for x in d:
            sums = np.concatenate((sums + x, sums - x))
        count = int(np.count_nonzero(sums >= observed - TIE))
        return {"method": "exact: every sign pattern", "patterns": 2**n, "seed": None,
                "count": count, "p_value": count / 2**n}  # fmt: skip
    bits = np.random.default_rng(PATTERN_SEED).integers(0, 2, size=(PATTERNS, n), dtype=np.int8)
    count = 0
    for start in range(0, PATTERNS, CHUNK):
        sums = (2.0 * bits[start : start + CHUNK] - 1.0) @ d
        count += int(np.count_nonzero(sums >= observed - TIE))
    return {"method": f"sampled: {PATTERNS} random sign patterns, p = (1 + count) / (1 + "
                      f"{PATTERNS})", "patterns": PATTERNS, "seed": PATTERN_SEED,
            "count": count, "p_value": (1 + count) / (1 + PATTERNS)}  # fmt: skip


def sign_test(predicted: Mapping[int, int], difference: Mapping[int, float]) -> dict[str, Any]:
    """The test on the seeds with both a predicted sign and an outcome difference h₁ − h₂
    (module docstring, "The test")."""
    seeds = sorted(set(predicted) & set(difference))
    a = {s: predicted[s] * sign(difference[s]) for s in seeds}
    used = [s for s in seeds if a[s] != 0]
    n, t = len(used), sum(a[s] == 1 for s in used)
    count = binomial_count(n, t)
    enumerated = None
    if n <= ENUMERATED:
        enumerated = int(np.count_nonzero(enumerated_count(n) >= t))
        if enumerated != count:
            raise AssertionError(f"the enumerated patterns ({enumerated}) are not the binomial "
                                 f"count ({count})")  # fmt: skip
    p = count / 2**n
    s_value = math.fsum(predicted[s] * difference[s] for s in used)
    return {
        "seeds": used, "n": n, "T": t, "ties": [s for s in seeds if a[s] == 0],
        "A": {str(s): a[s] for s in seeds},
        "p_value": p, "binomial_count": count, "enumerated_count": enumerated,
        "alpha": ALPHA, "significant": p < ALPHA,
        "S": {"value": s_value, **sign_flip([difference[s] for s in used], s_value)},
    }  # fmt: skip


# ------------------------------------------------------------------ the inputs


def own_code() -> dict[str, Any]:
    return ext.own_code(PROG)


def registration_text() -> str:
    return ext.registration_text()


def criterion_of(path: Path, key: str | None = CRITERION_KEY) -> dict[str, Any]:
    """The δ_lin criterion's verdict, under the dotted ``key`` (the top level when empty):
    ``status`` ``decided``, ``passed`` a boolean and ``primary_predictor`` the one it selects
    (``PRIMARY``); refused otherwise, a withheld criterion by name."""
    try:
        data = Path(path).read_bytes()
        record: Any = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse(f"the criterion record does not read ({type(exc).__name__})")
    found = record
    for part in key.split(".") if key else ():
        found = found.get(part) if isinstance(found, Mapping) else None
    if not isinstance(found, Mapping):
        refuse(f"the criterion record holds no object at {key or 'its top level'}")
    status, passed = found.get("status"), found.get("passed")
    if status == "withheld" or (status == "decided" and passed is None):
        refuse("the δ_lin criterion is withheld: no predictor is selected")
    if status != "decided":
        refuse(f"the criterion's status {status!r} is neither 'decided' nor 'withheld'")
    if not isinstance(passed, bool):
        refuse("the criterion's passed is not a boolean")
    predictor = found.get("primary_predictor")
    if predictor not in PREDICTORS:
        refuse(f"the criterion's primary_predictor {predictor!r} is neither {Q!r} nor {LN_K!r}")
    if predictor != PRIMARY[passed]:
        refuse(f"the criterion's passed ({passed}) and primary_predictor ({predictor!r}) "
               f"disagree: passed selects {PRIMARY[passed]!r}")  # fmt: skip
    kind = record.get("kind") if isinstance(record, Mapping) else None
    return {"record": Path(path).name, "sha256": sha256(data), "kind": kind, "key": key,
            "status": status, "passed": passed, "primary_predictor": predictor}  # fmt: skip


def prediction_of(path: Path, data: bytes) -> dict[str, Any]:
    """One sealed prediction (module docstring, "Inputs"); refused unless it reads as one."""
    name = f"{path.parent.name}/{path.name}"
    try:
        found = json.loads(data)
    except ValueError:
        refuse(f"the prediction {name} does not read")
    order = found.get("order") if isinstance(found, dict) else None
    status = (found.get("divergence") or {}).get("status") if isinstance(found, dict) else None
    if (not isinstance(found, dict) or found.get("kind") != REPLAY_KIND
            or not isinstance(found.get("seed"), int) or not isinstance(order, Mapping)
            or not isinstance(order.get("order_seed"), int) or status not in STATUSES
            or not (sa._is_number(found.get("lr"))
                    and math.isclose(found["lr"], ext.ETA_L, rel_tol=1e-9))):  # fmt: skip
        refuse(f"{name} is not a {REPLAY_KIND} record of an order at eta_l")
    q = found.get("q_lin")
    return {"file": name, "sha256": sha256(data), "seed": found["seed"],
            "order_seed": order["order_seed"], "status": status,
            "q": float(q) if sa._is_number(q) else None, "order": dict(order),
            "v0_sha256": found.get("v0_sha256"),
            "theta0_trainable_sha256": found.get("theta0_trainable_sha256"),
            "training": (found.get("streams") or {}).get("training")}  # fmt: skip


def predictions_of(paths: Sequence[Path]) -> dict[tuple[int, int], dict[str, Any]]:
    """Every sealed prediction given, by (seed, order seed); two of one order refused."""
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for given in paths:
        given = Path(given)
        files = sorted(given.glob("*/record.json")) if given.is_dir() else [given]
        if not files:
            refuse(f"--predictions {given.name}: no record")
        for path in files:
            try:
                data = path.read_bytes()
            except OSError as exc:
                refuse(f"the prediction {path.name} does not read ({type(exc).__name__})")
            found = prediction_of(path, data)
            key = (found["seed"], found["order_seed"])
            if key in out:
                refuse(f"two predictions of seed {key[0]}, order seed {key[1]}")
            out[key] = found
    return out


def checked_predictions(predictions: Mapping[tuple[int, int], Mapping[str, Any]],
                        seeds: Sequence[int]) -> None:  # fmt: skip
    """With ``q``: every tested order has its prediction, a complete one with a finite q (a
    diverged one leaves its seed out later); refused otherwise."""
    for seed in seeds:
        for order, offset in ORDERS:
            found = predictions.get((seed, seed + offset))
            if found is None:
                refuse(f"seed {seed} {order}: no prediction (q is the predictor)")
            if found["status"] == "complete" and found["q"] is None:
                refuse(f"seed {seed} {order}: its prediction is complete without a finite q")


def binding_problem(prediction: Mapping[str, Any], run: Mapping[str, Any]) -> str | None:
    """Why a prediction is not of an order's run: its order record, v₀, θ₀ and training stream
    must be those the run recorded (None: they are)."""
    inputs = run.get("inputs") or {}
    if inputs.get("training_order") != prediction["order"]:
        return "its order is not the run's"
    for key, what in (("v0_sha256", "v0"), ("theta0_trainable_sha256", "theta0")):
        if prediction[key] is None or prediction[key] != inputs.get(key):
            return f"its {what} is not the run's"
    training = (inputs.get("streams") or {}).get("training")
    return ext._cli().stream_prefix_problem(training, prediction["training"])


def theta0_of(registered: Mapping[str, Any]) -> dict[str, float]:
    """θ₀'s accuracies as the registered analysis read them: its line scans' point a = 0 (θ₀'s
    evaluation), one value over the seeds; refused otherwise."""
    per_seed = (((registered.get("descriptive") or {}).get("line_scans") or {})
                .get("per_seed") or {})  # fmt: skip
    found = set()
    for point in per_seed.values():
        zero = point.get("+0") if isinstance(point, Mapping) else None
        if not isinstance(zero, Mapping):
            continue
        pair = (zero.get("benchmark_accuracy"), zero.get("factual_accuracy"))
        if all(sa._is_number(x) and 0.0 <= x <= 1.0 for x in pair):
            found.add(pair)
    if len(found) != 1:
        refuse(f"the registered line scans give {len(found)} values of theta0's accuracies, "
               "not one")  # fmt: skip
    ((benchmark, factual),) = found
    return {"benchmark_accuracy": float(benchmark), "factual_accuracy": float(factual)}


def campaign_of(runs: Path, seed: int) -> str | None:
    """The seed's one launcher campaign of its orders (``study5-<ID>-orders-s<seed>``)."""
    folder = Path(runs) / "launch"
    found = sorted(folder.glob(f"study5-*-orders-s{seed}.jsonl")) if folder.is_dir() else []
    return found[0].stem if len(found) == 1 else None


# ------------------------------------------------------------------ per seed


def eta_l_share(found: Mapping[str, Any], counts: Mapping[str, Any]) -> sa.Value | dict:
    """The η_l run's harm share at its endpoint (``harmful``, a missing label not harmful, and
    ``harmful_high``), or why it has none."""
    run = found.get("eta_l")
    if not isinstance(run, Mapping):
        return str(run)
    got = counts.get(run["run_id"], "no labels")
    if not isinstance(got, Mapping):
        return str(got)
    if got["n"] != DENOMINATOR:
        return f"n = {got['n']}, not {DENOMINATOR}"
    return {"harmful": got["harmful"] / DENOMINATOR,
            "harmful_high": (got["harmful"] + got["missing"]) / DENOMINATOR,
            "missing_labels": got["missing"]}  # fmt: skip


def seed_part(reader, runs: Path, labels, theta0: Mapping[str, float], seed: int,
              predictor: str, predictions: Mapping[tuple[int, int], Mapping[str, Any]]
              ) -> dict[str, Any]:  # fmt: skip
    """One seed: its two new orders (``study5_ext1.order_parts``), the predicted sign, the
    primary difference h₁ − h₂ and the secondary one, or why each is missing."""
    campaign = campaign_of(runs, seed)
    orders: dict[str, Any] = {}
    for order, offset in ORDERS:
        if campaign is None:
            why = "not one launcher journal of the seed's orders"
            entry = ext.entry_of(seed, order, seed + offset, "the order runs",
                                 {name: why for name, _ in ext.RATES},
                                 {"h": None, "reason": why})  # fmt: skip
            found, counts = {}, {}
        else:
            entry, found, counts = ext.order_parts(reader, runs, campaign, labels, theta0, seed,
                                                   order, offset, "the order runs")  # fmt: skip
        prediction = predictions.get((seed, seed + offset))
        if prediction is not None and predictor == Q:
            for run in found.values():
                if isinstance(run, Mapping) and (why := binding_problem(prediction, run)):
                    refuse(f"seed {seed} {order}: the prediction is not of run {run['run_id']} "
                           f"({why})")  # fmt: skip
        share = eta_l_share(found, counts)
        kept = None if prediction is None else {k: prediction[k] for k in PREDICTION_KEYS}
        orders[order] = {**entry, "prediction": kept,
                         "eta_l_share": share if isinstance(share, Mapping)
                         else sa._json_value(share)}  # fmt: skip
    one, two = (orders[name] for name, _ in ORDERS)
    out: dict[str, Any] = {"seed": seed, "campaign": campaign, "orders": orders}
    # the predicted sign of h₁ − h₂
    if predictor == Q:
        values = [(o["prediction"] or {}).get("q") for o in (one, two)]
        why = [f"{o['order']}: delta_lin diverged" for o in (one, two)
               if (o["prediction"] or {}).get("status") == "diverged"]  # fmt: skip
        predicted = None if why else sign(values[1] - values[0])
    else:
        values = [o["ln_K_harmful_at_kstar"] for o in (one, two)]
        why = [f"{o['order']}: ln K_harmful at K*: {v['missing']}" for o, v in zip(
            (one, two), values, strict=True) if not sa._is_number(v)]  # fmt: skip
        predicted = None if why else sign(values[0] - values[1])
    out["predicted"] = ({"values": values, "sign": predicted} if predicted is not None
                        else {"missing": "; ".join(why)})  # fmt: skip
    # the outcomes: h at K* (primary), the η_l harm share (secondary)
    absent = [f"{o['order']}: {o['reason']}" for o in (one, two) if not sa._is_number(o["h"])]
    out["primary"] = ({"difference": one["h"] - two["h"]} if not absent
                      else {"missing": "; ".join(absent)})  # fmt: skip
    shares = [o["eta_l_share"] for o in (one, two)]
    lacking = [f"{o['order']}: {s['missing']}" for o, s in zip((one, two), shares, strict=True)
               if "missing" in s]  # fmt: skip
    out["secondary"] = ({"difference": shares[0]["harmful"] - shares[1]["harmful"],
                         "difference_high": shares[0]["harmful_high"] - shares[1]["harmful_high"]}
                        if not lacking else {"missing": "; ".join(lacking)})  # fmt: skip
    return out


def outcome_test(rows: Sequence[Mapping[str, Any]], part: str) -> dict[str, Any]:
    """The statistic on one outcome (``primary`` or ``secondary``) over the seeds that have it
    and a predicted sign; every other seed listed."""
    predicted, difference, left_out = {}, {}, []
    for row in rows:
        why = [x["missing"] for x in (row["predicted"], row[part]) if "missing" in x]
        if why:
            left_out.append({"seed": row["seed"], "reason": "; ".join(why)})
            continue
        predicted[row["seed"]] = row["predicted"]["sign"]
        difference[row["seed"]] = row[part]["difference"]
    return {**sign_test(predicted, difference), "left_out": left_out}


# ------------------------------------------------------------------ the record


def analysis(args: argparse.Namespace) -> dict[str, Any]:
    cli = ext._cli()
    code = own_code()
    text = registration_text()
    registered, registered_file = ext.registered_record(args.registered, text)
    if args.open_e2 is None:
        refuse("the test reads E2's outcomes: --open-e2 OPENING_RECORD is required")
    opening = ext.opening_record(args.open_e2)
    criterion = criterion_of(args.criterion, args.criterion_key)
    predictor = criterion["primary_predictor"]
    top_up = list(args.seeds or [])
    other = sorted({s for s in top_up if s not in TOP_UP} | {s for s in top_up
                                                             if top_up.count(s) > 1})  # fmt: skip
    if other:
        refuse(f"--seeds {other}: top-up seeds are 50000-50005 and 50016-50039, each once")
    seeds = sorted({*SEEDS, *top_up})
    predictions = predictions_of(args.predictions or [])
    if predictor == Q:
        checked_predictions(predictions, seeds)
    theta0 = theta0_of(registered)
    runs = Path(args.pod3_runs)
    reader = cli.Reader({"pod3": runs})
    prompts_of = cli.prompt_sets(reader)
    labels = cli.Labels(reader, prompts_of)
    rows = [seed_part(reader, runs, labels, theta0, s, predictor, predictions) for s in seeds]
    sets = registered.get("prompt_sets") or {}
    mismatch = sorted(n for n, d in reader.prompt_sets.items() if sets.get(n) != d)
    if mismatch:
        refuse(f"the prompt set {mismatch[0]} is not the one the registered analysis read")
    tested = {(s, s + offset) for s in seeds for _, offset in ORDERS}
    used = {k: v for k, v in predictions.items() if k in tested}
    return sa.finite_tree({
        "kind": KIND, "version": VERSION,
        "registration": {"path": ext.REGISTRATION, "sha256": sha256(text.encode("utf-8"))},
        "code": code, "registered_analysis": registered_file, "opening": opening,
        "criterion": criterion,
        "predictor": {"name": predictor, "reading": READINGS[predictor],
                      "predicted_sign": "sign(q2 - q1)" if predictor == Q
                      else "sign(lnK_h,1 - lnK_h,2)",
                      "of": "the sign of h1 - h2, order1 (s + 1000) minus order2 (s + 2000)"},
        "seeds": {"registered": list(SEEDS), "top_up": sorted(top_up), "tested": seeds,
                  "top_up_note": TOP_UP_NOTE if top_up else None},
        "theta0": {**theta0, "source": "the registered analysis's line scans, point a = 0"},
        "predictions": {"given": len(predictions), "used": len(used),
                        "unused": len(predictions) - len(used),
                        "files": {f"{k[0]}-{k[1]}": {"file": v["file"], "sha256": v["sha256"]}
                                  for k, v in sorted(used.items())}},
        "per_seed": rows,
        "primary": {"outcome": "h at K* of each order (the registered support rule and "
                               "interpolation)", **outcome_test(rows, "primary")},
        "secondary": {"outcome": SECONDARY, "decides": False, **outcome_test(rows, "secondary")},
        "labeling": ext.labeling(labels),
        "inputs": dict(sorted(reader.hashes.items())),
        "prompt_sets": dict(sorted(reader.prompt_sets.items())),
    })  # fmt: skip


def report(result: Mapping[str, Any], digest: str) -> None:
    """Numbers, ids and the record's sha256 only."""
    print(f"predictor {result['predictor']['name']} seeds {len(result['seeds']['tested'])}")
    for part in ("primary", "secondary"):
        found = result[part]
        print(f"{part} n {found['n']} T {found['T']} ties {len(found['ties'])} left_out "
              f"{len(found['left_out'])} p {found['p_value']:.6g} significant "
              f"{int(found['significant'])} S {found['S']['value']:.6g} S_p "
              f"{found['S']['p_value']:.6g}")  # fmt: skip
    print(f"record sha256 {digest}")


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m zero_field_refusal.analysis.{PROG}",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    parser.add_argument("--open-e2", type=Path, metavar="OPENING_RECORD")
    parser.add_argument("--criterion", type=Path, required=True)
    parser.add_argument("--criterion-key", default=CRITERION_KEY)  # "": the top level
    parser.add_argument("--predictions", type=Path, action="append")
    parser.add_argument("--pod3-runs", type=Path, required=True)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    cli = ext._cli()
    try:
        if args.out.exists() or args.out.is_symlink():
            refuse(f"{args.out.name} exists: never overwritten")
        result = analysis(args)
        data = cli.encoded(result)
        cli.write_once(args.out, data)
    except cli.Refused as exc:
        sys.exit(f"{PROG}: {exc}")
    report(result, sha256(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
