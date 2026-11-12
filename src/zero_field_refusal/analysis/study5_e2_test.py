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
``numpy.random.default_rng(20261114)``, as the record states), and the per-seed table.

**Exclusions and gaps.** A seed is *excluded* and listed, with the reason, only for a scientific
cause established by the registered rules:
- an order not supported at K* by the registered support rule (a failed endpoint, gross collapse,
  no bracket, too wide, tied);
- δ_lin declared diverged in its prediction (``q`` only);
- ln K_harmful at K* recorded but not positive and finite (``ln_K_harmful_at_K*`` only).

A zero difference is a tie, counted. Any other missing input leaves the seed *technically
unresolved*: a launcher journal, a run or its end, a label file or its receipt, or an unreadable
record. While any tested seed is unresolved, the verdict is withheld: ``status`` ``withheld``,
p, T and significance null, and each unresolved seed named with its reason. The numbers over the
resolved seeds are shown under ``descriptive_partial`` alone, with no p-value. An excluded cause
in either order outweighs a gap in the other: the seed is excluded whatever the missing input
holds. A missing prediction is refused (below), not withheld.

**The secondary outcome** is the harm share of each order's η_l run (its step-25 labels, a missing
label not harmful, the all-harmful value beside; no interpolation), with the same statistic. It
is reported beside and decides nothing. Its cohort is fixed here, before the opening. It is
paired: the seeds whose two new orders' η_l runs both have an η_l share (the run read and its
step-25 labels) and a predicted sign. The support at K* plays no part. A seed is excluded when its
predictor is (as above) or when an η_l run stopped by an instability (no step-25 responses). An η_l
run or its labels missing is a technical gap and withholds the secondary verdict in the same way.
The natural secondary outcome, Δr at the endpoint, cannot
be computed for the E2 runs: they are control runs (``experiments/study5_control_order``), which
save no checkpoint by the registration, so their endpoint is not there to score.

**Top-up seeds** (``--seeds``, added to the ten registered seeds 50006-50015): any of 50000-50005
and 50016-50039, with the same two new orders at both rates, their runs in the same root under the
same campaign and cell names. Seeds 50000-50005 were seen before: in the exploratory checks of
extension 1 and in the δ_lin criterion, with ``train.order_seed`` = s only. Their new orders were
never observed, and the test compares two new orders within a seed, so the seed's own effect
cancels.

**Inputs.**
- --open-e2: the opening record, required, checked before anything of E2 is read
  (``study5_ext1.read_opening``: the dated note at its commit, an ancestor of this checkout's
  HEAD, with the sha256 the record names, and unchanged at HEAD). It also binds the test's inputs
  (``opening_inputs``), and any difference is refused before any E2 read:
  - ``criterion_sha256``: the sha256 of the file given as --criterion;
  - ``pair_floors``: 10, equal to the criterion's ``part3_pairs.unresolved_floors``;
  - ``seeds``: exactly the tested seeds (the ten, with any top-up);
  - ``predictions``: a manifest {``<seed>-<order seed>`` or the record's path as given
    (``<folder>/record.json``): sha256}. The predictions given must be exactly these files, each
    with its sha256, with none extra and none missing.
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
  With ``q``, a tested order without a record, or complete without a finite q, is refused.
- --pod3-runs: the runs root holding the order runs and their launcher journals; a seed's
  campaign is its one journal ``launch/study5-<ID>-orders-s<seed>.jsonl``.
- --registered: the registered analysis's record, its sha256 named by the registration at this
  commit (``study5_ext1.registered_record``). θ₀'s accuracies (the gross-collapse check of the
  support rule) are its line scans' point a = 0, θ₀'s evaluation; the prompt sets must be its own.

**Preflight.** Before any outcome or label of an order run is read (``preflight``), each run the
launcher journals name is checked by its config.yaml and meta.json alone:
- its run id, seed, rate, frozen Adam, order seed and role ``control``;
- its prompt set: the registered identity and the prompts the run recorded;
- with ``q``, the prediction bound to it: its order record, v₀, θ₀ and training stream are the
  run's.

A difference is refused. A run not found or not readable is left to the outcome reading, where it
is technically unresolved.

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
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from . import study5_analysis as sa
from . import study5_ext1 as ext
from .study3_estimates import DENOMINATOR
from .study3_records import INSTABILITY

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
OPTIMIZER = "adam_frozen"  # arm F, as every order run's config names it
PAIR_FLOORS = 10  # the δ_lin criterion's part-3 multiplier the opening fixes
SHA256 = re.compile(r"[0-9a-f]{64}")
MANIFEST_KEY = re.compile(r"(?P<seed>[0-9]+)-(?P<order_seed>[0-9]+)")
OK, EXCLUDED, UNRESOLVED = "ok", "excluded", "unresolved"
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
SECONDARY_COHORT = (
    "paired, fixed before the opening: the seeds whose two new orders' eta_l runs both have an "
    "eta_l share (the run read and its step-25 labels) and a predicted sign; support at K* plays "
    "no part. Excluded and listed: a seed whose predictor is excluded or whose eta_l run stopped "
    "by an instability. An eta_l run or its labels missing is a technical gap: while any tested "
    "seed has one, the secondary verdict is withheld"
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
    part3 = found.get("part3_pairs")
    floors = part3.get("unresolved_floors") if isinstance(part3, Mapping) else None
    if not sa._is_number(floors):
        refuse("the criterion's part3_pairs.unresolved_floors is not a number")
    kind = record.get("kind") if isinstance(record, Mapping) else None
    return {"record": Path(path).name, "sha256": sha256(data), "kind": kind, "key": key,
            "status": status, "passed": passed, "primary_predictor": predictor,
            "unresolved_floors": floors}  # fmt: skip


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


def binding_problem(prediction: Mapping[str, Any], inputs: Mapping[str, Any]) -> str | None:
    """Why a prediction is not of an order's run (``inputs``: what its meta.json records): its
    order record, v₀, θ₀ and training stream must be the run's (None: they are)."""
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


def cell_of(campaign: str, seed: int, order: str, eta: float) -> tuple[str, str]:
    return campaign, f"s5-{order}-{eta!r}-s{seed}/s{seed}"


# ------------------------------------------------------------------ the bindings


def opening_inputs(found: Mapping[str, Any]) -> dict[str, Any]:
    """The inputs the opening record binds (module docstring, "Inputs"): ``criterion_sha256``,
    the ``predictions`` manifest, the tested ``seeds`` and ``pair_floors`` (10); refused when one
    is absent or of another form."""
    criterion, manifest = found.get("criterion_sha256"), found.get("predictions")
    seeds, floors = found.get("seeds"), found.get("pair_floors")
    if not isinstance(criterion, str) or not SHA256.fullmatch(criterion):
        refuse("--open-e2: the opening record holds no criterion_sha256")
    if not isinstance(manifest, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in manifest.items()):  # fmt: skip
        refuse("--open-e2: the opening record holds no predictions manifest {name: sha256}")
    if (
        not isinstance(seeds, list)
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        refuse("--open-e2: the opening record holds no list of distinct tested seeds")
    if not sa._is_number(floors) or floors != PAIR_FLOORS:
        refuse(f"--open-e2: the opening's pair_floors is {floors!r}, not {PAIR_FLOORS}")
    return {"criterion_sha256": criterion, "predictions": dict(manifest),
            "seeds": sorted(seeds), "pair_floors": floors}  # fmt: skip


def checked_manifest(predictions: Mapping[tuple[int, int], Mapping[str, Any]],
                     manifest: Mapping[str, str]) -> None:  # fmt: skip
    """The predictions given are exactly the opening's manifest, each with its sha256: a key is
    ``<seed>-<order seed>`` or a record's path as given (``<folder>/record.json``); refused
    otherwise."""
    by_file = {found["file"]: key for key, found in predictions.items()}
    named: dict[tuple[int, int], str] = {}
    for name, digest in sorted(manifest.items()):
        match = MANIFEST_KEY.fullmatch(name)
        key = (int(match["seed"]), int(match["order_seed"])) if match else by_file.get(name)
        if key is None or key not in predictions:
            refuse(f"--open-e2: the manifest names the prediction {name}, which is not given")
        if key in named:
            refuse(f"--open-e2: the manifest names seed {key[0]}, order seed {key[1]} twice")
        named[key] = digest
    extra = sorted(set(predictions) - set(named))
    if extra:
        refuse(f"--open-e2: the prediction of seed {extra[0][0]}, order seed {extra[0][1]} is "
               "given but not in the opening's manifest")  # fmt: skip
    for key, digest in sorted(named.items()):
        if predictions[key]["sha256"] != digest:
            refuse(f"--open-e2: the prediction of seed {key[0]}, order seed {key[1]} does not "
                   "have the sha256 the opening's manifest names")  # fmt: skip


def preflight(reader, runs: Path, prompts_of, registered_sets: Mapping[str, str],
              seeds: Sequence[int], predictor: str,
              predictions: Mapping[tuple[int, int], Mapping[str, Any]]) -> int:  # fmt: skip
    """Before any outcome or label of an order run is read: each run the launcher journals name
    (``study5_analysis_cli.completed``) is checked by its config.yaml and meta.json alone (its
    run id, seed, rate, frozen Adam, order seed and role ``control``), its prompt set by the
    registered identity (its sha256 and the prompts the run recorded), and with ``q`` the
    prediction bound to it (``binding_problem``). Each a refusal; a run not found or not
    readable is left to the outcome reading, where it is technically unresolved. The runs
    checked."""
    cli = ext._cli()
    checked = 0
    for seed in seeds:
        campaign = campaign_of(runs, seed)
        if campaign is None:
            continue
        for order, offset in ORDERS:
            for name, eta in ext.RATES:
                run_dir = cli.completed(reader, runs, cell_of(campaign, seed, order, eta), name)
                if isinstance(run_dir, str):
                    continue
                what = f"seed {seed} {order} {name} ({run_dir.name})"
                try:
                    meta_data = reader.bytes(run_dir / "meta.json")
                    meta = json.loads(meta_data)
                    cfg = OmegaConf.create(reader.bytes(run_dir / "config.yaml").decode())
                    found = (meta.get("run_id"), cfg.get("seed"),
                             OmegaConf.select(cfg, "optimizer.lr"),
                             OmegaConf.select(cfg, "optimizer.name"),
                             OmegaConf.select(cfg, "train.order_seed"),
                             OmegaConf.select(cfg, "study5.role"))  # fmt: skip
                except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError,
                        OmegaConfBaseException):  # fmt: skip
                    continue  # the outcome reading names it: technically unresolved
                lr = found[2]
                if (found[0] != run_dir.name or found[1] != seed or found[3] != OPTIMIZER
                        or not (sa._is_number(lr) and math.isclose(lr, eta, rel_tol=1e-9))
                        or found[4] != seed + offset or found[5] != "control"):  # fmt: skip
                    refuse(
                        f"{what}: its config or meta.json is not of the cell (run id, seed, "
                        f"rate, {OPTIMIZER}, order seed {seed + offset}, role control)"
                    )
                prediction = predictions.get((seed, seed + offset))
                if predictor == Q and prediction is not None:
                    why = binding_problem(prediction, meta.get("inputs") or {})
                    if why:
                        refuse(f"{what}: the prediction is not of the run ({why})")
                try:
                    prompts = prompts_of(cfg)
                    set_name = str(cfg.eval.harmful.prompt_set)
                    ids = cli.prompt_ids(cfg, prompts)
                except (OSError, ValueError, KeyError, TypeError, AttributeError,
                        OmegaConfBaseException):  # fmt: skip
                    continue  # the label reading names it: technically unresolved
                if registered_sets.get(set_name) != reader.prompt_sets.get(set_name):
                    refuse(f"{what}: its prompt set {set_name} is not the one the registered "
                           "analysis read")  # fmt: skip
                try:
                    cli.study2_labels._check_recorded_prompts(run_dir, cfg, ids, meta_data)
                except ValueError:
                    refuse(f"{what}: the prompts it recorded are not the registered set")
                checked += 1
    return checked


# ------------------------------------------------------------------ per seed


def outcome_state(entry: Mapping[str, Any]) -> tuple[str, str | None]:
    """An order's h at K*: ``ok``; ``excluded``, established non-support at K* by the registered
    rule (support ``unsupported``); or ``unresolved``, a technical gap (a run, its end or its
    labels missing, a record unreadable: support ``open``, ``unresolved`` or none, or supported
    without labels)."""
    if sa._is_number(entry.get("h")):
        return OK, None
    why = f"{entry['order']}: {entry['reason']}"
    state = (entry.get("support") or {}).get("state")
    return (EXCLUDED, why) if state == "unsupported" else (UNRESOLVED, why)


def ln_k_state(entry: Mapping[str, Any], found: Mapping[str, Any]) -> tuple[str, Any]:
    """An order's ln K_harmful at K*: its value; ``excluded`` when its order is not supported at
    K* or an endpoint's K_harmful is recorded but not positive and finite; ``unresolved`` when
    the support is a technical gap or a K_harmful is not recorded."""
    value = entry["ln_K_harmful_at_kstar"]
    if sa._is_number(value):
        return OK, value
    why = f"{entry['order']}: ln K_harmful at K*: {value['missing']}"
    support = entry.get("support") or {}
    if support.get("state") == "unsupported":
        return EXCLUDED, why
    if support.get("state") != "supported":
        return UNRESOLVED, why
    rows = {run["run_id"]: run["row"] or {} for run in found.values() if isinstance(run, Mapping)}
    ks = [rows.get(support.get(end), {}).get("K_harmful") for end in ("lo", "hi")]
    recorded = all(isinstance(k, int | float) and not isinstance(k, bool) for k in ks)
    return (EXCLUDED, why) if recorded else (UNRESOLVED, why)


def share_state(found: Mapping[str, Any], counts: Mapping[str, Any]) -> tuple[str, Any]:
    """An order's η_l harm share (``harmful``, ``harmful_high``, missing labels): ``excluded``
    when its η_l run stopped by an instability (the registered failed endpoint: no step-25
    responses); ``unresolved`` when the run or its labels are missing or unreadable."""
    run = found.get("eta_l")
    if not isinstance(run, Mapping):
        return UNRESOLVED, str(run) if run else "no eta_l run"
    if run.get("stop") in INSTABILITY:
        return EXCLUDED, f"its eta_l run stopped ({run['stop']})"
    got = counts.get(run["run_id"], "no labels")
    if not isinstance(got, Mapping):
        return UNRESOLVED, str(got)
    if got["n"] != DENOMINATOR:
        return UNRESOLVED, f"n = {got['n']}, not {DENOMINATOR}"
    return OK, {"harmful": got["harmful"] / DENOMINATOR,
                "harmful_high": (got["harmful"] + got["missing"]) / DENOMINATOR,
                "missing_labels": got["missing"]}  # fmt: skip


def joined(states: Sequence[tuple[str, Any]]) -> tuple[str, list[str]]:
    """A seed's state over its parts: ``excluded`` when any part is (established whatever the
    rest), else ``unresolved`` when any part is, else ``ok``; the reasons."""
    for kind in (EXCLUDED, UNRESOLVED):
        why = [str(found) for state, found in states if state == kind]
        if why:
            return kind, why
    return OK, []


def seed_part(reader, runs: Path, labels, theta0: Mapping[str, float], seed: int,
              predictor: str, predictions: Mapping[tuple[int, int], Mapping[str, Any]]
              ) -> dict[str, Any]:  # fmt: skip
    """One seed: its two new orders (``study5_ext1.order_parts``), the predicted sign, the
    primary difference h₁ − h₂ and the secondary one, each ``ok``, ``excluded`` or
    ``unresolved`` with its reasons."""
    campaign = campaign_of(runs, seed)
    orders, states = {}, {}
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
        if predictor == Q:
            guess = ((EXCLUDED, f"{order}: delta_lin diverged")
                     if prediction["status"] == "diverged" else (OK, prediction["q"]))  # fmt: skip
        else:
            guess = ln_k_state(entry, found)
        share = share_state(found, counts)
        if share[0] != OK:
            share = (share[0], f"{order}: {share[1]}")
        states[order] = {"outcome": outcome_state(entry), "predictor": guess, "share": share}
        kept = None if prediction is None else {k: prediction[k] for k in PREDICTION_KEYS}
        orders[order] = {**entry, "prediction": kept,
                         "outcome_state": states[order]["outcome"][0],
                         "eta_l_share": share[1] if share[0] == OK else {"missing": share[1]},
                         "eta_l_share_state": share[0]}  # fmt: skip
    one, two = (states[name] for name, _ in ORDERS)
    out: dict[str, Any] = {"seed": seed, "campaign": campaign, "orders": orders}
    state, why = joined([one["predictor"], two["predictor"]])
    if state == OK:
        values = [one["predictor"][1], two["predictor"][1]]
        predicted = sign(values[1] - values[0]) if predictor == Q else sign(values[0] - values[1])
        out["predicted"] = {"state": OK, "values": values, "sign": predicted}
    else:
        out["predicted"] = {"state": state, "reasons": why}
    state, why = joined([one["outcome"], two["outcome"]])
    h = [orders[name]["h"] for name, _ in ORDERS]
    out["primary"] = ({"state": OK, "difference": h[0] - h[1]} if state == OK
                      else {"state": state, "reasons": why})  # fmt: skip
    state, why = joined([one["share"], two["share"]])
    if state == OK:
        a, b = one["share"][1], two["share"][1]
        out["secondary"] = {"state": OK, "difference": a["harmful"] - b["harmful"],
                            "difference_high": a["harmful_high"] - b["harmful_high"]}  # fmt: skip
    else:
        out["secondary"] = {"state": state, "reasons": why}
    return out


def outcome_test(rows: Sequence[Mapping[str, Any]], part: str) -> dict[str, Any]:
    """The statistic on one outcome (``primary`` or ``secondary``). A seed is excluded and listed
    when its predictor or its outcome is (module docstring, "Exclusions and gaps"); while any
    tested seed is technically unresolved the verdict is withheld: no p, T or significance,
    the reasons named, the numbers over the resolved seeds under ``descriptive_partial``
    alone."""
    predicted, difference, excluded, unresolved = {}, {}, [], []
    for row in rows:
        state, why = joined(
            [
                (row["predicted"]["state"], "; ".join(row["predicted"].get("reasons", []))),
                (row[part]["state"], "; ".join(row[part].get("reasons", []))),
            ]
        )
        if state == EXCLUDED:
            excluded.append({"seed": row["seed"], "reason": "; ".join(why)})
        elif state == UNRESOLVED:
            unresolved.append({"seed": row["seed"], "reason": "; ".join(why)})
        else:
            predicted[row["seed"]] = row["predicted"]["sign"]
            difference[row["seed"]] = row[part]["difference"]
    found = sign_test(predicted, difference)
    if not unresolved:
        return {"status": "tested", **found, "excluded": excluded, "withheld": []}
    partial = {"label": "descriptive only: no p-value and no reading while a tested seed is "
                        "technically unresolved",
               "seeds": found["seeds"], "n": found["n"], "T": found["T"],
               "ties": found["ties"], "S": found["S"]["value"]}  # fmt: skip
    return {"status": "withheld", "withheld": unresolved, "excluded": excluded,
            "seeds": None, "n": None, "T": None, "ties": found["ties"], "A": found["A"],
            "p_value": None, "binomial_count": None, "enumerated_count": None,
            "alpha": ALPHA, "significant": None, "S": None,
            "descriptive_partial": partial}  # fmt: skip


# ------------------------------------------------------------------ the record


def analysis(args: argparse.Namespace) -> dict[str, Any]:
    cli = ext._cli()
    code = own_code()
    text = registration_text()
    registered, registered_file = ext.registered_record(args.registered, text)
    if args.open_e2 is None:
        refuse("the test reads E2's outcomes: --open-e2 OPENING_RECORD is required")
    opening, found = ext.read_opening(args.open_e2)
    bound = opening_inputs(found)
    criterion = criterion_of(args.criterion, args.criterion_key)
    if criterion["sha256"] != bound["criterion_sha256"]:
        refuse("--open-e2: the criterion record given is not the one the opening names "
               "(criterion_sha256)")  # fmt: skip
    if criterion["unresolved_floors"] != bound["pair_floors"]:
        refuse(f"the criterion's part 3 counts {criterion['unresolved_floors']!r} floors, not the "
               f"opening's {bound['pair_floors']}")  # fmt: skip
    predictor = criterion["primary_predictor"]
    top_up = list(args.seeds or [])
    other = sorted({s for s in top_up if s not in TOP_UP} | {s for s in top_up
                                                             if top_up.count(s) > 1})  # fmt: skip
    if other:
        refuse(f"--seeds {other}: top-up seeds are 50000-50005 and 50016-50039, each once")
    seeds = sorted({*SEEDS, *top_up})
    if seeds != bound["seeds"]:
        refuse(f"--open-e2: the tested seeds ({len(seeds)}) are not the opening's "
               f"({len(bound['seeds'])}): {sorted(set(seeds) ^ set(bound['seeds']))}")  # fmt: skip
    predictions = predictions_of(args.predictions or [])
    checked_manifest(predictions, bound["predictions"])
    if predictor == Q:
        checked_predictions(predictions, seeds)
    theta0 = theta0_of(registered)
    runs = Path(args.pod3_runs)
    reader = cli.Reader({"pod3": runs})
    prompts_of = cli.prompt_sets(reader)
    checked = preflight(reader, runs, prompts_of, registered.get("prompt_sets") or {}, seeds,
                        predictor, predictions)  # fmt: skip
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
        "code": code, "registered_analysis": registered_file,
        "opening": {**opening, **{k: v for k, v in bound.items() if k != "predictions"},
                    "predictions": len(bound["predictions"])},
        "criterion": criterion,
        "predictor": {"name": predictor, "reading": READINGS[predictor],
                      "predicted_sign": "sign(q2 - q1)" if predictor == Q
                      else "sign(lnK_h,1 - lnK_h,2)",
                      "of": "the sign of h1 - h2, order1 (s + 1000) minus order2 (s + 2000)"},
        "seeds": {"registered": list(SEEDS), "top_up": sorted(top_up), "tested": seeds,
                  "top_up_note": TOP_UP_NOTE if top_up else None},
        "theta0": {**theta0, "source": "the registered analysis's line scans, point a = 0"},
        "preflight": {"runs_checked": checked,
                      "read": "config.yaml and meta.json of each run, before any outcome"},
        "predictions": {"given": len(predictions), "used": len(used),
                        "unused": len(predictions) - len(used),
                        "files": {f"{k[0]}-{k[1]}": {"file": v["file"], "sha256": v["sha256"]}
                                  for k, v in sorted(used.items())}},
        "per_seed": rows,
        "primary": {"outcome": "h at K* of each order (the registered support rule and "
                               "interpolation)", **outcome_test(rows, "primary")},
        "secondary": {"outcome": SECONDARY, "cohort": SECONDARY_COHORT, "decides": False,
                      **outcome_test(rows, "secondary")},
        "labeling": ext.labeling(labels),
        "inputs": dict(sorted(reader.hashes.items())),
        "prompt_sets": dict(sorted(reader.prompt_sets.items())),
    })  # fmt: skip


def _number(value: Any) -> str:
    return "none" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def report(result: Mapping[str, Any], digest: str) -> None:
    """Numbers, ids and the record's sha256 only."""
    print(f"predictor {result['predictor']['name']} seeds {len(result['seeds']['tested'])}")
    for part in ("primary", "secondary"):
        found = result[part]
        s = found["S"] or {}
        print(f"{part} {found['status']} n {_number(found['n'])} T {_number(found['T'])} ties "
              f"{len(found['ties'])} excluded {len(found['excluded'])} withheld "
              f"{len(found['withheld'])} p {_number(found['p_value'])} significant "
              f"{_number(found['significant'])} S {_number(s.get('value'))} S_p "
              f"{_number(s.get('p_value'))}")  # fmt: skip
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
