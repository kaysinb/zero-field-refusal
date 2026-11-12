"""Study 5's extension 2 analysis (``analysis.study5_ext2`` and its CLI
``analysis.study5_ext2_cli``) on synthetic records only, each built in the schema its writer
writes (``scripts/study5_linear_replay.py``, ``study5_vectors.py``, ``study5_pairs.py``,
``study5_force.py``, ``study5_vector_points.py``, ``study5_tokens.py``,
``study5_engineering_check.py``, ``study5_surrogate_endpoint.py`` and the registered analysis):
the δ_lin criterion's three parts by hand (the permutation p against the exact distribution, the
scale ratio, the pairs' signs, ties and divergence), the floors' flags, M3's halving rule, the E2
guard, the CLI end to end (written once, numbers only, labels through the registered reader,
missing optional inputs as null with a reason) and its refusals."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_ext2 as ext2
from zero_field_refusal.analysis import study5_ext2_cli as ecli
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import _content_hash, judge_identity
from zero_field_refusal.paths import REPO_ROOT

SENTINEL = "TEXT-NEVER-STORED"
IDS = [f"judged:{i}" for i in range(521)]
PROMPTS = {pid: f"Prompt {SENTINEL} {i}" for i, pid in enumerate(IDS)}
SET = "harmful_evaluation"
ARCHIVE = "pod4-20261111"
POD2 = "pod2-20261109"
R0 = 12.44
RATE = "1.3e-05"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


# ------------------------------------------------------------------ the criterion


def exact_p_greater(q, dr) -> float:
    """The exact one-sided p: the share of all permutations of q whose ρ with Δr is at or above
    the observed one."""
    observed = stats.spearmanr(q, dr).statistic
    found = [stats.spearmanr(perm, dr).statistic for perm in itertools.permutations(q)]
    return sum(r >= observed - 1e-12 for r in found) / len(found)


def test_the_permutation_p_is_the_exact_distributions_limit_and_its_formula() -> None:
    q = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3]
    dr = [0.05, 0.30, 0.10, 0.70, 0.20, 0.35]
    found = ext2.permutation_greater(q, dr, permutations=100_000, rng_seed=20261112)
    assert found["rho"] == pytest.approx(stats.spearmanr(q, dr).statistic, abs=1e-15)
    assert found["p_value"] == (1 + found["count"]) / (1 + 100_000)
    assert found["p_value"] == pytest.approx(exact_p_greater(q, dr), abs=0.004)
    assert ext2.permutation_greater(q, dr, permutations=100_000, rng_seed=20261112) == found
    # by hand: q's ranks permuted by the registered generator, the count of ρ_perm ≥ ρ_obs
    rq = stats.rankdata(q) - stats.rankdata(q).mean()
    rd = stats.rankdata(dr) - stats.rankdata(dr).mean()
    den = math.sqrt(float(rq @ rq) * float(rd @ rd))
    rng = np.random.default_rng(20261112)
    count = sum(float(rd @ rng.permutation(rq)) / den >= found["rho"] - 1e-12
                for _ in range(2000))  # fmt: skip
    again = ext2.permutation_greater(q, dr, permutations=2000, rng_seed=20261112)
    assert again["count"] == count
    with pytest.raises(ValueError, match="constant"):
        ext2.permutation_greater([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], permutations=10, rng_seed=1)


def own_rows(n: int = 12, *, slope: float = 0.9, diverged=(), noise: float = 0.01) -> dict:
    rng = np.random.default_rng(3)
    seeds = list(range(50000, 50000 + n))
    rows = {}
    for s in seeds:
        dr = -0.5 - 0.5 * rng.random()
        rows[s] = {"q_lin": None if s in diverged else slope * dr + noise * rng.normal(),
                   "delta_r": dr, "diverged": s in diverged, "missing": None}  # fmt: skip
    return rows


def pair_rows(dq=(-0.01, 0.02, -0.03, 0.01, -0.02, 0.01), dh=(0.1, -0.1, 0.2, -0.1, 0.3, 0.1),
              diverged=()) -> dict:  # fmt: skip
    return {s: {"q_control": 0.5 + a, "q_stream": 0.5, "h_difference": b,
                "diverged": s in diverged, "missing": None}
            for s, a, b in zip(ext2.M_SEEDS, dq, dh, strict=True)}  # fmt: skip


def test_the_criterion_passes_on_all_three_parts_and_names_q() -> None:
    own = own_rows(diverged=(50011,))
    found = ext2.criterion(own, pair_rows(), seeds=list(own), permutations=20_000)
    assert found["status"] == "decided" and found["diverged"] == [50011]
    one, two, three = found["part1_rank"], found["part2_scale"], found["part3_pairs"]
    used = [s for s in own if s != 50011]
    assert one["n"] == 11 and found["seeds"] == used
    assert one["rho"] == pytest.approx(
        sa.spearman([own[s]["q_lin"] for s in used], [own[s]["delta_r"] for s in used])
    )
    assert one["passed"] and one["p_value"] < 0.05 and one["scipy_p_greater"] < 0.05
    mean_q = np.mean([own[s]["q_lin"] for s in used])
    mean_dr = np.mean([own[s]["delta_r"] for s in used])
    assert two["ratio"] == pytest.approx(mean_q / mean_dr) and two["passed"]
    assert three["agreements"] == 5 and three["passed"]  # the last pair: same signs
    assert [p["agrees"] for p in three["pairs"]] == [True] * 5 + [False]
    assert found["passed"] is True and found["primary_predictor"] == "q"


def test_each_part_can_fail_and_then_the_predictor_is_ln_k_harmful() -> None:
    own = own_rows()
    scaled = {s: {**r, "q_lin": 3 * r["q_lin"]} for s, r in own.items()}  # ratio ≈ 2.7
    found = ext2.criterion(scaled, pair_rows(), seeds=list(own), permutations=2000)
    assert found["part1_rank"]["passed"] and not found["part2_scale"]["passed"]
    assert found["passed"] is False and found["primary_predictor"] == "ln_K_harmful_at_K*"
    # a tie and a diverged pair do not agree: 4 of 6
    pairs = pair_rows(dq=(-0.01, 0.0, -0.03, 0.01, -0.02, -0.01), diverged=(50002,))
    found = ext2.criterion(own, pairs, seeds=list(own), permutations=2000)
    reasons = {p["seed"]: p["reason"] for p in found["part3_pairs"]["pairs"]}
    assert reasons[50001] == "a tie" and reasons[50002] == "δ_lin diverged"
    assert found["part3_pairs"]["agreements"] == 4 and not found["part3_pairs"]["passed"]
    # a negative association fails part 1
    flipped = {s: {**r, "q_lin": -r["q_lin"]} for s, r in own.items()}
    found = ext2.criterion(flipped, pair_rows(), seeds=list(own), permutations=2000)
    assert not found["part1_rank"]["passed"] and found["primary_predictor"] == "ln_K_harmful_at_K*"


def test_a_zero_mean_delta_r_fails_the_scale_part() -> None:
    own = {50000 + i: {"q_lin": float(i), "delta_r": d, "diverged": False, "missing": None}
           for i, d in enumerate([-1.0, 1.0, -2.0, 2.0])}  # fmt: skip
    found = ext2.criterion(own, pair_rows(), seeds=list(own), permutations=100)
    assert found["part2_scale"]["ratio"] is None and found["part2_scale"]["passed"] is False


def test_a_missing_input_withholds_the_verdict() -> None:
    own = own_rows()
    own[50003]["missing"] = "no Δr (absent)"
    pairs = pair_rows()
    pairs[50004]["missing"] = "no control linear record"
    found = ext2.criterion(own, pairs, seeds=list(own), permutations=2000)
    assert found["status"] == "withheld" and found["passed"] is None
    assert found["primary_predictor"] is None
    assert found["withheld"] == ["s50003: no Δr (absent)", "pair s50004: no control linear record"]
    assert found["part1_rank"]["passed"] is None and found["part1_rank"]["rho"] is not None
    assert 50003 not in found["seeds"]


# ------------------------------------------------------------------ floors, locality, the guard


def test_floor_flags_take_one_order_of_magnitude() -> None:
    assert ext2.resolution(1e-3, 1e-4) == {"ratio": pytest.approx(10.0), "flag": "resolved"}
    assert ext2.resolution(-9e-4, 1e-4)["flag"] == "numerically_unresolved"
    assert ext2.resolution(1e-3, 0.0) == {"ratio": None, "floor_zero": True, "flag": "resolved"}
    assert ext2.resolution(0.0, 0.0)["flag"] == "numerically_unresolved"
    assert ext2.resolution(None, 1e-4)["flag"] is None
    assert ext2.resolution(1.0, None)["reason"] == "no floor"


def small_record(seed: int, r) -> dict:
    rows = [
        {
            "a": a,
            "condition": f"vector_small_s{seed}_{a:+g}",
            "r": r(a),
            "harmful_R": -1.0 + a,
            "harmful_C": -9.0 - a,
            "harmful_contrast": r(a),
            "benign_R": -2.0,
            "benign_C": -3.0,
            "benign_contrast": 1.0,
            "K_benign": 100 * a * a,
            "K_harmful": 50 * a * a,
            "evaluated": False,
            "reason": "readout_only",
        }
        for a in (0.1, -0.1, 0.05, -0.05, 0.025, -0.025)
    ]
    return {"kind": "study5-vector", "seed": seed, "rows": rows}  # fmt: skip


def test_m3_halving_reads_a_quadratic_as_local_and_a_cubic_odd_part_as_not() -> None:
    quadratic = small_record(50000, lambda a: R0 - 2 * a - 30 * a * a)
    found = ext2.small_amplitudes(quadratic, "tokens: absent", R0)
    assert found["parts"]["0.1"]["odd"] == pytest.approx(-0.2)
    assert found["parts"]["0.1"]["even"] == pytest.approx(-0.3)
    for key in ("0.1/0.05", "0.05/0.025"):
        assert found["halving"][key]["odd"] == {"ratio": pytest.approx(2.0), "local": True}
        assert found["halving"][key]["even"] == {"ratio": pytest.approx(4.0), "local": True}
    assert found["tokens"] == {"available": False, "reason": "tokens: absent"}
    cubic = small_record(50000, lambda a: R0 - 2 * a + 4000 * a**3)
    found = ext2.small_amplitudes(cubic, "x", R0)
    assert found["halving"]["0.1/0.05"]["odd"]["local"] is False  # (−0.2+4)/(−0.1+0.5) ≈ 9.5
    assert found["halving"]["0.1/0.05"]["even"] == {"ratio": None, "local": None}  # 0/0


def test_a_linear_row_reads_k_absence_and_the_bound() -> None:
    """A record written without K (``--no-k``) says so; a δ_lin that crossed its bound by less
    than ten times the repeat's relative difference is numerically ambiguous, one well past it
    is not."""
    record = {"kind": "study5-linear-replay", "seed": 50000, "K": None, "q_hat": -0.1,
              "q_lin": None, "r": {"theta0": R0, "delta_hat": R0 - 0.1, "delta_lin": None},
              "reference": {"bound": 20.0},
              "divergence": {"status": "diverged", "step": 9, "reason": "bound", "norm": 20.0001},
              "per_step": [{"step": 8, "delta_lin_norm": 19.0},
                           {"step": 9, "delta_lin_norm": 20.0001}],
              "checks": [{"step": 5, "half": 1e-3, "double": None}],
              "vectors": {"delta_hat": {"sha256": "a"}}}  # fmt: skip
    row = ext2.linear_row(record, lin_floor=1e-5)
    assert row["delta_hat"]["K_reason"] == "no K readouts (--no-k)"
    assert row["diverged"] and row["delta_lin"]["q"] is None
    assert row["near_bound"]["bound_ratio"] == pytest.approx(1.000005)
    assert row["near_bound"]["numerically_ambiguous"] is True
    assert row["epsilon_checks"] == {"steps": [{"step": 5, "half": 1e-3, "double": None,
                                                "largest": 1e-3}],
                                     "largest": 1e-3, "missing_steps": [15, 25]}  # fmt: skip
    far = {**record, "per_step": [{"step": 9, "delta_lin_norm": 26.0}]}
    assert ext2.linear_row(far, lin_floor=1e-5)["near_bound"]["numerically_ambiguous"] is False
    assert ext2.linear_row(far)["near_bound"]["numerically_ambiguous"] is None  # no floor


def test_the_e2_guard_names_the_sealed_order_repeats() -> None:
    assert ext2.e2_repeat({"seed": 50006, "train": {"order_seed": 51006}}, "r1")
    assert ext2.e2_repeat({"seed": 50015, "train": {"order_seed": 52015}}, "r2")
    assert ext2.e2_repeat(None, "20261111T0Z-s5-order2-1.3e-05-s50009-s50009-ab12-a1")
    assert ext2.e2_repeat({"seed": 50001, "train": {"order_seed": 52010}}, "r3")
    # the registered order controls (order seed = seed), a fresh run and an edited order are not
    assert ext2.e2_repeat({"seed": 50003, "train": {"order_seed": 50003}},
                          "x-s5-order-1.3e-05-s50003-s50003-a1") is None  # fmt: skip
    assert ext2.e2_repeat({"seed": 50007, "train": {}}, "x-s5-fresh-1.3e-05-s50007") is None
    assert ext2.e2_repeat({"seed": 50000, "train": {"order_edit": {"kind": "shuffle"}}},
                          "x-s5-edit-shuffle1-5-0") is None  # fmt: skip


def test_the_modules_import_no_torch() -> None:
    code = ("import sys; import zero_field_refusal.analysis.study5_ext2, "
            "zero_field_refusal.analysis.study5_ext2_cli; "
            "assert 'torch' not in sys.modules, 'torch imported'")  # fmt: skip
    found = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin"},
    )
    assert found.returncode == 0, found.stderr


# ------------------------------------------------------------------ the synthetic pod 4


def gram_numbers(G: np.ndarray, seeds: list[int]) -> dict:
    """study5_vectors.gram_numbers' arithmetic (its module imports torch)."""
    n = len(seeds)
    d = np.diag(G).copy()
    norms, total, rows = np.sqrt(d), float(G.sum()), G.sum(axis=1)
    loo = np.sqrt(total - 2 * rows + d) / (n - 1)
    p = (rows - d) / (n - 1) / loo
    a = np.zeros(n)
    a[0::2] = 1 / len(range(0, n, 2))
    b = np.zeros(n)
    b[1::2] = 1 / len(range(1, n, 2))

    def spread(v):
        m, s = float(v.mean()), float(v.std(ddof=1))
        return {"mean": m, "sd": s, "se": s / math.sqrt(v.size), "cv": s / abs(m)}

    cos = G / np.outer(norms, norms)
    return {
        "norms": norms.tolist(),
        "mean": {
            "norm_sq_noise_free": (total - d.sum()) / (n * (n - 1)),
            "norm_sq": total / n**2,
            "norm": math.sqrt(total) / n,
            "mean_norm_sq": float(d.mean()),
            "excess": total / n**2 - (total - d.sum()) / (n * (n - 1)),
            "mean_pairwise_cosine": float((cos.sum() - np.trace(cos)) / (n * (n - 1))),
        },
        "leave_one_out": {
            "seeds": seeds,
            "p": p.tolist(),
            "cos": (p / norms).tolist(),
            "norm": loo.tolist(),
            "p_spread": spread(p),
            "cos_spread": spread(p / norms),
        },
        "halves": {
            "even": {"seeds": seeds[0::2], "norm": math.sqrt(a @ G @ a)},
            "odd": {"seeds": seeds[1::2], "norm": math.sqrt(b @ G @ b)},
            "cosine": float(a @ G @ b) / math.sqrt((a @ G @ a) * (b @ G @ b)),
        },
    }


TWIN = {"kind": "swap_first", "step": 2}
EDITS = {f"swap{t}": {"kind": "swap_batches", "step": t} for t in (2, 12, 23)} | {
    f"shuffle{a}-{z}-{d}": {"kind": "shuffle", "first": a, "last": z, "draw": d}
    for a, z in ((1, 5), (21, 25)) for d in (0, 1)}  # fmt: skip


class Pod4:
    """A synthetic pod 4: its records folder, its vector and edited runs (some labeled, every
    response holding ``SENTINEL``), pod 2's endpoint records and the registered analysis."""

    def __init__(self, tmp: Path) -> None:
        self.tmp, self.runs, self.pod2 = tmp, tmp / "runs", tmp / "pod2"
        self.records = self.runs / "study5_records" / ARCHIVE
        self.identity = judge_identity(load_config("judge/wildguard"))
        rng = np.random.default_rng(11)
        self.seeds = list(ext2.SEEDS)
        self.replay = {s: f"r-replay-s{s}" for s in self.seeds}
        self.dr = {s: -0.4 - 0.4 * rng.random() for s in self.seeds}
        self.q_lin = {s: 0.85 * self.dr[s] + 0.03 * rng.normal() for s in self.seeds}
        self.q_hat = {s: 0.3 * self.dr[s] + 0.03 * rng.normal() for s in self.seeds}
        self.diverged, self.near = 50039, 50038
        self.dtheta = {s: 2.0 + 0.1 * (s % 7) for s in self.seeds}
        self.harmful: dict[str, int] = {}
        self.dq = dict(zip(ext2.M_SEEDS, (-0.01, 0.02, -0.03, 0.01, -0.02, 0.01), strict=True))
        self.dh = dict(zip(ext2.M_SEEDS, (0.10, -0.10, 0.20, -0.10, 0.30, 0.10), strict=True))
        self.vectors()
        self.linear()
        self.points()
        self.force()
        self.twins()
        self.checks()
        self.edited()
        self.analysis()

    # -------------------------------------------------------------- labeled runs

    def labeled(self, run_dir: Path, conditions, *, harmful: int | None,
                missing: int = 0) -> None:  # fmt: skip
        """Responses of every condition, as the run writes them, and with ``harmful`` (a count)
        WildGuard's labels and receipt, as the labeling writes them."""
        responses, labels = [], []
        for c_index, condition in enumerate(conditions):
            for i, pid in enumerate(IDS):
                text = f"Reply {SENTINEL} {condition} {i}"
                responses.append({"condition": condition, "item_id": pid, "text": text})
                n = (harmful or 0) + c_index
                label = (None if i >= 521 - missing else "harmful_compliance" if i < n
                         else "refusal" if i < n + 300 else "safe_non_refusal")  # fmt: skip
                labels.append({"item_id": f"{condition}/{pid}", **self.identity,
                               "content_sha256": _content_hash(PROMPTS[pid], text),
                               "label": label,
                               "created_at": "2026-11-13T12:00:00+00:00"})  # fmt: skip
        lines(run_dir / "responses" / f"{SET}.jsonl", responses)
        if harmful is None:  # before the judge
            return
        lines(run_dir / "judge" / "wildguard.jsonl", labels)
        expected = [f"{c}/{pid}" for c in conditions for pid in IDS]
        digest = sha((run_dir / "judge" / "wildguard.jsonl").read_bytes())
        dump(run_dir / "judge" / "wildguard.receipt.json", {
            "judge": "wildguard", "identity": self.identity, "expected_items": expected,
            "items": len(expected), "rounds": 3 if missing else 1,
            "missing": missing * len(conditions), "labels_sha256": digest,
            "completed_at": "2026-11-13T13:00:00+00:00"})  # fmt: skip
        self.harmful[run_dir.name] = harmful

    def meta(self, run_dir: Path) -> None:
        dump(
            run_dir / "meta.json",
            {
                "run_id": run_dir.name,
                "inputs": {
                    "harmful_prompts": {
                        "set": SET,
                        "count": 521,
                        "ids_sha256": sha(json.dumps(IDS).encode()),
                    }
                },
            },
        )

    # -------------------------------------------------------------- M1

    def vectors(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(40, 6)) + np.array([1.0, 0.5, 0, 0, 0, 0])
        G = X @ X.T
        self.gram = G
        numbers = gram_numbers(G, self.seeds)
        self.grad = rng.normal(size=40)
        dump(self.records / "vectors" / "m1" / "record.json", {
            "kind": "study5-vectors", "n": 40, "seeds": self.seeds, "lr": 1.3e-05, "steps": 25,
            "runs": [{"seed": s, "run_id": self.replay[s], "displacement_norm_recorded":
                      numbers["norms"][j], "displacement_norm": numbers["norms"][j]}
                     for j, s in enumerate(self.seeds)],
            "skipped": [], "gram": G.tolist(), **numbers,
            "grad_r": {"sha256": "g" * 64, "norm": 3.0, "per_run": self.grad.tolist(),
                       "mean": float(self.grad.mean()), "sd": 1.0, "se": 0.1, "cv": 1.0,
                       "mean_vector": 0.1, "vectors": {}},
            "removals": [{"seed": s, "delta_norm": 2.0} for s in ext2.M_SEEDS],
            "mixtures": [{"k": k, "scale": 1.0, "unscaled_norm": 0.5, "target_norm": 0.5}
                         for k in (1, 2, 3)]})  # fmt: skip

    # -------------------------------------------------------------- M2

    def linear_record(self, s: int, *, q_hat: float, q_lin: float | None, order="stream",
                      k: bool = True, tag: str = "", ratio: float = 0.1) -> dict:  # fmt: skip
        bound = 10 * self.dtheta[s]
        diverged = q_lin is None
        steps = 20 if diverged else 25
        norms = [bound * ratio * t / 25 for t in range(1, steps + 1)]
        if diverged:
            norms[-1] = 1.3 * bound
        return {
            "kind": "study5-linear-replay", "seed": s, "lr": 1.3e-05, "steps": 25,
            "epsilon": 0.01, "check_steps": [5, 15, 25], "order": order,
            "reference": {"norm": self.dtheta[s], "bound": bound, "factor": 10.0},
            "divergence": {"status": "diverged", "step": 20, "reason": "bound",
                           "norm": norms[-1], "previous_norm": norms[-2]} if diverged else
            {"status": "complete", "step": None, "reason": None, "norm": None,
             "previous_norm": None},
            "r": {"theta0": R0, "delta_hat": R0 + q_hat,
                  "delta_lin": None if diverged else R0 + q_lin},
            "q_hat": q_hat, "q_lin": q_lin,
            "K": None if not k else {"delta_hat": {"K_benign": 3.0, "K_harmful": 1.5},
                                     "delta_lin": None if diverged else
                                     {"K_benign": 5.0, "K_harmful": 2.5},
                                     "delta_lin_unavailable": "diverged: no delta_lin"
                                     if diverged else None},
            "norms": {"delta_hat": 0.5, "delta_lin": None if diverged else 1.5,
                      "relative_difference": None if diverged else 2.0,
                      "cosine": None if diverged else 0.7},
            "per_step": [{"step": t, "delta_lin_norm": v, "delta_hat_norm": 0.02 * t}
                         for t, v in enumerate(norms, 1)],
            "checks": [{"step": t, "half": 1e-3 * i, "double": 2e-3 * i}
                       for i, t in enumerate((5, 15, 25), 1) if t <= steps],
            "vectors": {"delta_hat": {"sha256": f"hat-s{s}{tag}"},
                        **({} if diverged else {"delta_lin": {"sha256": f"lin-s{s}{tag}"}})},
        }  # fmt: skip

    def pairs(self, rows) -> dict:
        return {"kind": "study5-pairs", "r": {}, "pairs": [
            {"a": a, "b": b, "dot": 0.1, "norm_a": na, "norm_b": nb, "cosine": cos,
             "distance": dist} for a, b, na, nb, cos, dist in rows]}  # fmt: skip

    def vec(self, sha256: str) -> dict:
        return {"kind": "vector", "path": f"/vec/{sha256}.pt", "sha256": sha256}

    def end(self, s: int) -> dict:
        return {
            "kind": "ckpt",
            "path": f"/ckpt/{self.replay[s]}.pt",
            "sha256": "e" * 64,
            "run": self.replay[s],
            "step": "25",
        }

    def linear(self) -> None:
        R = self.records
        for s in self.seeds:
            q_lin = None if s == self.diverged else self.q_lin[s]
            ratio = 0.999995 if s == self.near else 0.1
            dump(
                R / "linear" / f"s{s}" / "record.json",
                self.linear_record(s, q_hat=self.q_hat[s], q_lin=q_lin, ratio=ratio),
            )
            dump(
                R / "linear" / f"pairs-hat-s{s}.json",
                self.pairs([(self.vec(f"hat-s{s}"), self.end(s), 0.5, self.dtheta[s], 0.3, 2.0)]),
            )
            if q_lin is None:
                dump(
                    R / "linear" / f"pairs-lin-s{s}.json",
                    {"available": False, "missing": "delta_lin.pt", "reason": "x"},
                )
            else:
                dump(
                    R / "linear" / f"pairs-lin-s{s}.json",
                    self.pairs(
                        [(self.vec(f"lin-s{s}"), self.end(s), 1.5, self.dtheta[s], 0.6, 1.0)]
                    ),
                )
            # pod 2's surrogate endpoint record of the seed
            dump(self.pod2 / "study5_records" / POD2 / "surrogate-endpoint" / f"s{s}.json",
                 {"kind": "study5-surrogate-endpoint", "seed": s, "r": R0 + self.dr[s],
                  "run_id": f"pod2-fresh-s{s}"})  # fmt: skip
        s = ext2.FORCE_SEED  # the amendment's repeat
        dump(R / "linear-repeat" / f"s{s}" / "record.json", {
            **self.linear_record(s, q_hat=self.q_hat[s] + 1e-6, q_lin=self.q_lin[s] + 2e-6,
                                 tag="-b"),
            "K": {"delta_hat": {"K_benign": 3.0 + 1e-7, "K_harmful": 1.5},
                  "delta_lin": {"K_benign": 5.0, "K_harmful": 2.5 + 3e-7},
                  "delta_lin_unavailable": None}})  # fmt: skip
        for kind, norm, cos in (("hat", 0.5, 0.3), ("lin", 1.5, 0.6)):
            dump(R / "linear-repeat" / f"pairs-{kind}-s{s}.json", self.pairs([
                (self.vec(f"{kind}-s{s}"), self.vec(f"{kind}-s{s}-b"), norm, norm, 1.0,
                 1e-5 * norm),
                (self.vec(f"{kind}-s{s}-b"), self.end(s), norm, self.dtheta[s], cos + 1e-7,
                 2.0)]))  # fmt: skip
        for s in ext2.M_SEEDS:  # the registered order controls' orders (no K)
            dump(R / "linear-control" / f"s{s}" / "record.json", self.linear_record(
                s, q_hat=self.q_hat[s] + 0.5 * self.dq[s], q_lin=self.q_lin[s] + self.dq[s],
                order={"order_seed": s, "responses": 500, "permutation_sha256": "c" * 64},
                k=False))  # fmt: skip
        for s in ext2.E2_SEEDS:  # the sealed orders' predictions
            for offset in (1000, 2000):
                dump(R / "sealed" / f"s{s}-order{offset // 1000}" / "record.json",
                     self.linear_record(s, q_hat=-0.1, q_lin=-0.3, k=False, order={
                         "order_seed": s + offset, "responses": 500,
                         "permutation_sha256": "d" * 64}))  # fmt: skip
        for s in ext2.M_SEEDS:  # the edited orders' predictions
            tags = [t for t in EDITS if t.startswith("shuffle")]
            if s in ext2.SWAP_SEEDS:
                tags += ["swap2", "swap12", "swap23", "twin"]
            for tag in tags:
                edit = EDITS.get(tag, TWIN)
                shift = 0.05 if tag.startswith("shuffle1-") else 0.005
                dump(R / "edits" / f"s{s}-{tag}" / "record.json", self.linear_record(
                    s, q_hat=self.q_hat[s] + shift, q_lin=self.q_lin[s] + shift, k=False,
                    order={"order_seed": None, "responses": 500, "permutation_sha256": "f" * 64,
                           "order_edit": edit}))  # fmt: skip

    # -------------------------------------------------------------- M3, M4

    def vector_run(self, label: str, seed: int, points, *, full=(), match: str | None = None,
                   labeled: int | None = None, run_id: str | None = None) -> None:  # fmt: skip
        """A vector record and its run directory (labeled when ``labeled`` is a count)."""
        run_id = run_id or f"s5-{ARCHIVE}-{label.replace('_', '-')}"
        rows = []
        for a in points:
            row = {"a": a, "condition": f"vector_{label}_{a:+g}", "r": R0 - 0.1 * a - 0.02 * a * a,
                   "harmful_R": -1.0, "harmful_C": -9.0, "harmful_contrast": R0,
                   "benign_R": -2.0, "benign_C": -3.0, "benign_contrast": 1.0,
                   "K_benign": 6.0 * a * a, "K_harmful": 3.0 * a * a}  # fmt: skip
            if a in full:
                row |= {"evaluated": True, "full": {"K_benign": 6.0 * a * a, "K_harmful": 3.0,
                                                    "heldout_nll_per_token": 1.1,
                                                    "factual_accuracy": 0.5,
                                                    "benchmark_accuracy": 0.6}}  # fmt: skip
            else:
                row |= {"evaluated": False, "reason": "readout_only"}
            rows.append(row)
        matched = None
        if match is not None:
            search = {"status": "bracketed" if match != "not_bracketed" else "not_bracketed",
                      "extensions": 2 if match != "not_bracketed" else 8, "monotone": True,
                      "scales": [{"c": 0.5, "K_benign": 2.0}, {"c": 1.0, "K_benign": 5.0},
                                 {"c": 2.0, "K_benign": 9.0}]}  # fmt: skip
            matched = {"search": search, "status": match}
            if match != "not_bracketed":
                k = 6.1 if match == "matched" else 7.0
                matched |= {"c": 1.1, "refine": {"c1": 1.1, "K1": k, "c2": None, "K2": None,
                                                 "chosen": "c1", "c": 1.1, "K": k,
                                                 "out_of_tolerance": match != "matched",
                                                 "status": match}}  # fmt: skip
                rows.append({"a": 1.1, "condition": f"vector_{label}_kstar", "r": R0 - 0.2,
                             "K_benign": k, "K_harmful": 3.0, "matched_k": True,
                             "evaluated": True, "full": {"heldout_nll_per_token": 1.2,
                                                         "factual_accuracy": 0.49,
                                                         "benchmark_accuracy": 0.59}})  # fmt: skip
        dump(self.records / "points" / f"{label}.json", {
            "kind": "study5-vector", "label": label, "seed": seed, "sign": 1,
            "vector": {"file": "v.pt", "sha256": "v" * 64}, "vector_norm": 0.7, "rows": rows,
            "matched": matched, "experiment": "study5_vector", "run_id": run_id,
            "conditions": [r["condition"] for r in rows],
            "evaluated": [r["condition"] for r in rows if r["evaluated"]]})  # fmt: skip
        run_dir = self.runs / "study5_vector" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(json.dumps(
            {"run_type": "study5_vector", "seed": seed, "eval": {"harmful": {"prompt_set": SET}},
             "train": {}}))  # fmt: skip
        self.meta(run_dir)
        lines(run_dir / "vector_points.jsonl", rows)
        evaluated = [r["condition"] for r in rows if r["evaluated"]]
        dump(run_dir / "summary.json", {"vector": {"conditions": [r["condition"] for r in rows],
                                                   "evaluated": evaluated}})  # fmt: skip
        if evaluated:
            self.labeled(run_dir, evaluated, harmful=labeled)

    def points(self) -> None:
        small = (0.1, -0.1, 0.05, -0.05, 0.025, -0.025)
        for s in ext2.M_SEEDS:
            self.vector_run(f"small_s{s}", s, small)
            dump(
                self.records / "tokens" / f"s{s}.json",
                {
                    "kind": "study5-tokens",
                    "seed": s,
                    "theta0": {
                        "r": R0,
                        "tokens": {"entropy": 1.2, "rare_mass": 0.01, "count": 4000},
                    },
                    "points": [
                        {
                            "a": a,
                            "r": R0 - 0.1 * a,
                            "tokens": {
                                "entropy": 1.2 + 0.1 * abs(a),
                                "rare_mass": 0.01 + 0.01 * abs(a),
                                "delta_logprob": -0.05 * abs(a),
                                "count": 4000,
                            },
                        }
                        for a in (0.1, -0.1, 1.0, -1.0)
                    ],
                    "pairs": [{"a": a, "odd": -0.1 * a, "even": 0.0} for a in (0.1, 1.0)],
                },
            )
            self.vector_run(f"rmean_s{s}", s, (1.0,), full=(1.0,),
                            labeled=30 if s == 50000 else None)  # fmt: skip
            self.vector_run(f"rmixture_s{s}", s, (1.0,), full=(1.0,),
                            labeled=10 if s == 50000 else None)  # fmt: skip
            self.vector_run(f"dlin_s{s}", s, (1.0,), full=(1.0,),
                            labeled=40 if s == 50000 else None)  # fmt: skip
        self.vector_run("mean", 50000, (1.0, -1.0, 2.0, -2.0), full=(1.0, -1.0, 2.0, -2.0),
                        labeled=20)  # fmt: skip
        self.vector_run("mean_k", 50000, (), match="matched", labeled=25)
        self.vector_run("mean_neg_k", 50000, (), match="not_bracketed")
        for k in (1, 2, 3):
            self.vector_run(f"mix{k}", 50000, (1.0,), full=(1.0,),
                            match="out_of_tolerance" if k == 2 else "matched",
                            labeled=5 if k == 1 else None)  # fmt: skip

    # -------------------------------------------------------------- M5

    def force_record(self, panel: int, steps, epsilons, *, scale: float = 1.0,
                     files: bool = False, tag: str = "") -> dict:  # fmt: skip
        def parts(step, eps):
            base = scale * step * (1 + 0.1 * eps)
            return {label: {"linear_norm": 10.0, "force_norm": 2.0 * base,
                            "Dforce_norm": base, "projections": [0.3 * base * f, -0.1 * base * f]}
                    for label, f in (("full", 1.0), ("first_half", 1.1),
                                     ("second_half", 0.9))}  # fmt: skip

        return {"kind": "study5-force", "seed": 50000, "panel": {"seed": panel},
                "projections": [{"name": "mean_minus_50000.pt", "kind": "mean"},
                                {"name": "grad_r.pt", "kind": "grad_r"}],
                "g0_norm": {"full": 1.0},
                "checkpoints": [{"step": t, "delta_norm": 0.1 * t, "total_norm": {"full": 1.0},
                                 "epsilons": [{"epsilon": e, "parts": parts(t, e),
                                               "halves_difference": {"force_norm": 0.2,
                                                                     "Dforce_norm": 0.1 * t},
                                               "rounding": {"plus": 1e-9},
                                               "force_file": {"name": "f.pt",
                                                              "sha256": f"force{tag}",
                                                              "norm": 1.0} if files else None}
                                              for e in epsilons]}
                                for t in steps]}  # fmt: skip

    def force(self) -> None:
        R = self.records
        dump(
            R / "force" / "s50000-panel59990.json",
            self.force_record(59990, (5, 10), (5e-2, 1e-2, 5e-3)),
        )
        dump(
            R / "force" / "s50000-panel59991.json",
            self.force_record(59991, (5, 10), (5e-2, 1e-2, 5e-3), scale=1.2),
        )
        dump(R / "force-repeat" / "a.json", self.force_record(59990, (5,), (1e-2,), files=True,
                                                              tag="a"))  # fmt: skip
        dump(R / "force-repeat" / "b.json", self.force_record(
            59990, (5,), (1e-2,), scale=1.0 + 1e-4, files=True, tag="b"))  # fmt: skip
        dump(
            R / "force-repeat" / "pairs.json",
            self.pairs([(self.vec("forcea"), self.vec("forceb"), 5.0, 5.0, 1.0, 5e-4)]),
        )

    # -------------------------------------------------------------- M6

    def twins_record(self, s: int, other: str, *, gap: float, r_gap: float) -> dict:
        rows, r = [], {"theta0": R0}
        for t in ext2.TWIN_STEPS:
            suffix = "" if t == 25 else f"-step-{t}"
            a = {"kind": "ckpt", "path": f"/ckpt/{self.replay[s]}{suffix}.pt",
                 "sha256": f"{t:064d}", "run": self.replay[s], "step": str(t)}  # fmt: skip
            b = {"kind": "ckpt", "path": f"/ckpt/{other}{suffix}.pt", "sha256": f"{t + 1:064d}",
                 "run": other, "step": str(t)}  # fmt: skip
            rows.append({"a": a, "b": b, "dot": 1.0, "norm_a": 0.1 * t, "norm_b": 0.1 * t,
                         "cosine": 0.99, "distance": gap * t})  # fmt: skip
            r[Path(a["path"]).name] = R0 - 0.02 * t
            r[Path(b["path"]).name] = R0 - 0.02 * t - r_gap * t
        return {"kind": "study5-pairs", "pairs": rows, "r": r}

    def twins(self) -> None:
        for s in ext2.SWAP_SEEDS:
            dump(
                self.records / "twins" / f"s{s}.json",
                self.twins_record(s, f"twin-s{s}", gap=1e-3, r_gap=1e-3),
            )
        dump(
            self.records / "twins" / "floor-s50000.json",
            self.twins_record(50000, "floor-s50000", gap=1e-6, r_gap=1e-7),
        )

    def checks(self) -> None:
        def record(exact: bool) -> dict:
            pair = {"rerun": "x", "config_hash": "h", "exact": exact, "within_tolerance": True,
                    "passed": True, "answers_differing": 0 if exact else 1, "files": [],
                    "endpoint_tensors": {"equal": exact, "distance": None if exact else
                                         {"ratio": 3e-5, "available": True}}}  # fmt: skip
            return {"kind": "study5-engineering-check", "passed": True, "exact": exact,
                    "within_tolerance": True, "answers_differing": pair["answers_differing"],
                    "pairs": [pair], "tolerance": {"relative": 1e-4}}  # fmt: skip

        dump(self.records / "engineering-check-numerical.json", record(True))
        for i in range(0, 40, 4):
            dump(self.records / "replays" / f"check-g{50000 + i}.json", record(i != 8))

    def edited_run(self, s: int, tag: str, *, labeled: int | None, config=None) -> Path:
        block = "swaps" if tag.startswith("swap") else "shuffles"
        campaign = f"study5-{ARCHIVE}-{block}-s{s}"
        if tag == "twin":  # launched with its group's replays
            campaign = f"study5-{ARCHIVE}-replays-g50000"
        label = f"s5-edit-{tag}-{RATE}-s{s}/s{s}"
        run_id = f"20261112T000000Z-s5-edit-{tag}-{RATE}-s{s}-s{s}-ab-a1"
        journal = self.runs / "launch" / f"{campaign}.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a") as f:
            for event in ({"event": "start", "run": label, "run_id": run_id,
                           "experiment": "study5", "time": "2026-11-12T00:00:00+00:00"},
                          {"event": "end", "run": label, "run_id": run_id, "outcome": "done",
                           "time": "2026-11-12T00:30:00+00:00"}):  # fmt: skip
                f.write(json.dumps(event) + "\n")
        run_dir = self.runs / "study5" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = config or {
            "run_type": "train",
            "seed": s,
            "eval": {"harmful": {"prompt_set": SET}},
            "optimizer": {"name": "adam_frozen", "lr": 1.3e-05},
            "train": {"order_edit": EDITS.get(tag, TWIN)},
            "schedule": {"steps": 25},
        }
        (run_dir / "config.yaml").write_text(json.dumps(cfg))
        self.meta(run_dir)
        dump(run_dir / "terminal.json", {"steps": 25, "stop": {"reason": "steps"},
                                         "evaluations": [25]})  # fmt: skip
        lines(run_dir / "evaluations.jsonl", [{
            "step": 25, "condition": "step-25", "K_benign": 6.2, "K_harmful": 3.3,
            "benchmark_accuracy": 0.6, "factual_accuracy": 0.5,
            "heldout_nll_per_token": 1.0}])  # fmt: skip
        self.labeled(run_dir, ["step-25"], harmful=labeled)
        return run_dir

    def edited(self) -> None:
        for s in ext2.M_SEEDS:
            tags = [t for t in EDITS if t.startswith("shuffle")]
            if s in ext2.SWAP_SEEDS:
                tags = ["swap2", "swap12", "swap23", *tags]
            for tag in tags:
                self.edited_run(s, tag, labeled=(60 if tag.startswith("shuffle1-") else 52)
                                if s == 50000 else None)  # fmt: skip
        self.edited_run(50000, "twin", labeled=None)  # the other seeds' twins: no journal entry

    # -------------------------------------------------------------- the registered analysis

    def analysis(self) -> None:
        inputs = {}
        for s in self.seeds:
            rel = f"study5_records/{POD2}/surrogate-endpoint/s{s}.json"
            inputs[f"runs:{rel}"] = sha((self.pod2 / rel).read_bytes())
        self.h = {s: 0.05 + 0.002 * (s % 13) for s in self.seeds}
        outcomes = [
            {
                "seed": s,
                "support": {"state": "unsupported" if s == 50005 else "supported"},
                "h": None if s == 50005 else self.h[s],
                "p": 0.0,
            }
            for s in self.seeds
        ]
        stream = {str(s): {"+1": {"harmful": 0.1, "K_benign": 6.0, "K_harmful": 3.0,
                                  "benchmark_accuracy": 0.6, "factual_accuracy": 0.5,
                                  "heldout_nll_per_token": 1.0, "refusal": 0.5,
                                  "safe_non_refusal": 0.4}} for s in self.seeds}  # fmt: skip
        record = {"kind": "study5-analysis", "outcomes": outcomes, "inputs": inputs,
                  "descriptive": {
                      "controls": {"order": {"per_seed": {
                          "harmful": {str(s): self.dh[s] for s in ext2.M_SEEDS},
                          "K_harmful": {str(s): -self.dh[s] for s in ext2.M_SEEDS}}}},
                      "line_scans": {"per_seed": stream}}}  # fmt: skip
        self.analysis_path = self.pod2 / "study5-final" / "analysis.json"
        self.analysis_sha = sha(dump(self.analysis_path, record))

    def argv(self, out: Path) -> list[str]:
        return [str(a) for a in ("--records", self.records, "--runs-root", self.runs,
                                 "--runs-root", self.pod2, "--analysis", self.analysis_path,
                                 "--out", out)]  # fmt: skip


@pytest.fixture
def pod4(tmp_path, monkeypatch) -> Pod4:
    monkeypatch.setattr(cli, "harmful_prompts", lambda cfg: PROMPTS)
    found = Pod4(tmp_path)
    monkeypatch.setattr(ecli, "REGISTERED_ANALYSIS_SHA256", found.analysis_sha)
    return found


def call(argv) -> tuple[int, str]:
    try:
        return ecli.main(argv), ""
    except SystemExit as exc:
        return 1, str(exc.code)


# ------------------------------------------------------------------ the CLI end to end


def test_the_record_holds_every_section_once_with_numbers_only(pod4, tmp_path, capsys) -> None:
    out = tmp_path / "ext2.json"
    assert call(pod4.argv(out)) == (0, "")
    printed = capsys.readouterr().out
    data = out.read_text()
    record = json.loads(data)
    assert SENTINEL not in data and SENTINEL not in printed
    assert f"record sha256 {sha(out.read_bytes())}" in printed
    assert record["kind"] == ext2.KIND and record["archive_id"] == ARCHIVE
    assert record["registered_analysis"]["sha256"] == pod4.analysis_sha
    # never overwritten
    before = out.read_bytes()
    code, why = call(pod4.argv(out))
    assert code == 1 and "never overwritten" in why and out.read_bytes() == before

    # Δr = r(θ₂₅) of pod 2's endpoint record − the seed's r(θ₀)
    m1 = record["m1"]
    for s in (50000, 50017):
        assert m1["per_seed"][str(s)]["delta_r"] == pytest.approx(pod4.dr[s], abs=1e-12)
    # M1: copied from the vectors record; ρ(p_i, h_i) over the 39 supported seeds
    numbers = gram_numbers(pod4.gram, pod4.seeds)
    assert m1["norm_sq_noise_free"] == pytest.approx(numbers["mean"]["norm_sq_noise_free"])
    assert m1["split_half_cosine"] == pytest.approx(numbers["halves"]["cosine"])
    supported = [s for s in pod4.seeds if s != 50005]
    p = dict(zip(pod4.seeds, numbers["leave_one_out"]["p"], strict=True))
    assert m1["rho_p_h"]["n"] == 39
    assert m1["rho_p_h"]["rho"] == pytest.approx(
        sa.spearman([p[s] for s in supported], [pod4.h[s] for s in supported])
    )
    dot = m1["dot_against_delta_r"]["dot_grad_r"]
    assert dot["mean"] == pytest.approx(pod4.grad.mean())
    assert dot["se"] == pytest.approx(pod4.grad.std(ddof=1) / math.sqrt(40))
    groups = m1["replay_checks"]
    assert groups["g50008"]["exact"] is False and groups["g50008"]["passed"] is True
    assert groups["g50008"]["largest_endpoint_distance_ratio"] == pytest.approx(3e-5)
    assert record["cases"]["checks_not_exact"] == [
        {"check": "g50008", "passed": True, "within_tolerance": True}
    ]

    # the criterion: by hand over the 39 seeds whose δ_lin did not diverge
    c = record["criterion"]
    used = [s for s in pod4.seeds if s != pod4.diverged]
    q = [pod4.q_lin[s] for s in used]
    dr = [pod4.dr[s] for s in used]
    assert c["status"] == "decided" and c["diverged"] == [pod4.diverged] and c["seeds"] == used
    found = ext2.permutation_greater(q, dr, permutations=100_000, rng_seed=20261112)
    assert c["part1_rank"]["p_value"] == found["p_value"] and c["part1_rank"]["rho"] > 0
    assert c["part1_rank"]["scipy_p_greater"] == pytest.approx(
        stats.spearmanr(q, dr, alternative="greater").pvalue
    )
    assert c["part2_scale"]["ratio"] == pytest.approx(np.mean(q) / np.mean(dr))
    assert c["part3_pairs"]["agreements"] == 5
    assert [p["agrees"] for p in c["part3_pairs"]["pairs"]] == [True] * 5 + [False]
    assert c["passed"] is True and c["primary_predictor"] == "q"
    assert c["part3_pairs"]["pairs"][0]["q_difference_floor"]["flag"] == "resolved"

    # M2: δ̂ and δ_lin per seed, their cosines with Δθ from the pairs records
    m2 = record["m2"]
    row = m2["per_seed"]["50001"]
    assert row["delta_hat"]["cosine_with_delta_theta"] == 0.3
    assert row["delta_lin"]["norm_ratio"] == pytest.approx(1.5 / pod4.dtheta[50001])
    assert row["epsilon_checks"]["largest"] == pytest.approx(6e-3)
    assert m2["per_seed"][str(pod4.diverged)]["delta_lin"]["cosine_with_delta_theta"] is None
    assert m2["means"]["delta_lin"]["q"]["n"] == 39
    assert m2["rho_q_delta_r"]["delta_lin"]["n"] == 39
    pair = m2["control_pairs"]["50000"]
    assert pair["q_lin"]["difference"] == pytest.approx(pod4.dq[50000])
    assert pair["signs"]["q_lin"]["q_opposite_harm"] is True
    assert sorted(m2["e2_predictions"]) == sorted(
        f"s{s}-order{k}" for s in ext2.E2_SEEDS for k in (1, 2)
    )
    assert m2["e2_predictions"]["s50006-order1"]["q_lin"] == -0.3
    # the seed near its bound is numerically ambiguous under the repeat's floor (1e-5)
    assert record["cases"]["near_bound"] == [
        {"where": "own", "key": str(pod4.near), "bound_ratio": pytest.approx(0.999995)}
    ]
    assert {"where": "own", "key": str(pod4.diverged), "step": 20,
            "reason": "bound"} in record["cases"]["diverged"]  # fmt: skip

    # the floors: the linear repeat's, beside q and K
    floors = record["floors"]["linear_repeat"]["floors"]
    assert floors["q_hat"] == pytest.approx(1e-6) and floors["q_lin"] == pytest.approx(2e-6)
    assert floors["vector_lin_relative"] == pytest.approx(1e-5)
    assert row["floors"]["delta_lin"]["q"]["flag"] == "resolved"
    assert row["floors"]["lin_against_hat_relative_difference"]["ratio"] == pytest.approx(2e5)
    assert row["floors"]["delta_hat"]["K_harmful"] == {
        "ratio": None,
        "floor_zero": True,
        "flag": "resolved",
    }

    # M3: the even part with the token record's r(θ₀)
    m3 = record["m3"]["per_seed"]["50000"]
    assert m3["r_theta0_source"] == "tokens"
    assert m3["parts"]["0.1"]["odd"] == pytest.approx(-0.01)
    assert m3["parts"]["0.1"]["even"] == pytest.approx(-0.02 * 0.01)
    assert m3["halving"]["0.1/0.05"]["even"]["local"] is True
    assert m3["tokens"]["points"]["+1"]["entropy"] == pytest.approx(1.3)

    # M4: harm by the registered reader, the parity contrast, the searches' states
    m4 = record["m4"]
    plus = m4["parity"]["1x"]["plus"]
    assert plus["harm"]["harmful"] == pytest.approx(20 / 521)
    assert m4["parity"]["1x"]["minus"]["harm"]["harmful"] == pytest.approx(21 / 521)
    assert m4["parity"]["1x"]["contrast_plus_minus"]["harmful"] == pytest.approx(-1 / 521)
    assert m4["matched_k"]["plus"]["point"]["harm"]["harmful"] == pytest.approx(25 / 521)
    assert m4["removals"]["50000"]["contrast_mean_minus_mixture"]["harmful"] == pytest.approx(
        20 / 521
    )
    unlabeled = m4["removals"]["50001"]["mean_removed"]["harm"]
    assert unlabeled["available"] is False and "wildguard" in unlabeled["reason"]
    assert record["cases"]["unbracketed"] == [
        {"point": "mean_neg_k", "status": "not_bracketed", "extensions": 8}
    ]
    assert record["cases"]["out_of_tolerance"] == [{"point": "mix2", "K": 7.0}]
    assert record["m2"]["dlin_points"]["50000"]["points"]["+1"]["harm"]["harmful"] == (
        pytest.approx(40 / 521)
    )

    # M5: the panels' agreement and the ε-convergence, the force repeat's floor
    m5 = record["m5"]
    agree = m5["panels_agreement"]["5"]["0.01"]["mu_hat_minus_i"]
    base = 0.3 * 5 * (1 + 0.1 * 0.01)
    assert agree["difference"] == pytest.approx(base - 1.2 * base)
    assert agree["relative"] == pytest.approx(0.2 * base / (1.1 * base))
    conv = m5["panels"]["59990"]["steps"]["10"]["convergence"]["0.05"]
    assert conv["Dforce_norm"] == pytest.approx(abs((1.005 - 1.001) / 1.001))
    force = record["floors"]["force_repeat"]
    assert force["vector_difference_norm"] == 5e-4 and force["relative"] == pytest.approx(1e-4)
    assert force["projections"]["mu_hat_minus_i"] == pytest.approx(base * 1e-4)

    # M6: an edited run's outcome and harm against the stream order's, its predicted Δq
    swaps = {(r["seed"], r["tag"]): r for r in record["m6"]["swaps"]["rows"]}
    row = swaps[(50000, "swap2")]
    assert row["outcome"]["harm"]["harmful"] == pytest.approx(52 / 521)
    assert row["outcome_difference"]["harmful"] == pytest.approx(52 / 521 - 0.1)
    assert row["outcome_difference"]["K_harmful"] == pytest.approx(0.3)
    assert row["predicted_difference"]["q_lin"]["difference"] == pytest.approx(0.005)
    assert swaps[(50001, "swap2")]["outcome"]["harm"]["available"] is False
    early = record["m6"]["shuffles"]["early_late"]
    assert early["1-5"]["q_lin"]["mean"] == pytest.approx(0.05)
    assert early["21-25"]["q_lin"]["mean"] == pytest.approx(0.005)
    twin = record["m6"]["twins"]["50000"]
    step = twin["steps"]["5"]
    assert step["distance"] == pytest.approx(5e-3) and step["distance_floor"]["ratio"] == (
        pytest.approx(1000.0)
    )
    assert step["r_floor"]["flag"] == "resolved"
    assert twin["endpoint"]["delta_r_twin"] == pytest.approx(-0.02 * 25 - 1e-3 * 25)
    assert twin["outcome"]["K_harmful"] == 3.3 and twin["outcome"]["harm"]["available"] is False
    assert twin["outcome_difference"]["K_harmful"] == pytest.approx(0.3)
    assert record["m6"]["twins"]["50001"]["outcome"]["available"] is False
    assert twin["endpoint"]["excess_twin_lin"] == pytest.approx(
        -0.5 - 0.025 - (pod4.q_lin[50000] + 0.005)
    )

    # r(θ₀) from every source read, and their spread
    assert record["r_theta0"]["spread"] == 0.0 and record["r_theta0"]["n"] >= 40
    # every input by its sha256; no harmful text anywhere
    assert any(k.startswith("records:linear/s50000/") for k in record["inputs"])
    assert any(k.startswith("runs2:study5_records/pod2-") for k in record["inputs"])


def test_missing_optional_inputs_give_null_with_a_reason(pod4, tmp_path) -> None:
    """The shuffles block not admitted (no records, journals or runs), the force records and a
    seed's tokens absent: null with the reason, never a crash; the criterion still decides."""
    import shutil

    for path in pod4.records.glob("edits/s*-shuffle*"):
        shutil.rmtree(path)
    for path in (pod4.runs / "launch").glob("*shuffles*"):
        path.unlink()
    shutil.rmtree(pod4.records / "force")
    (pod4.records / "tokens" / "s50003.json").unlink()
    out = tmp_path / "ext2.json"
    assert call(pod4.argv(out)) == (0, "")
    record = json.loads(out.read_text())
    rows = record["m6"]["shuffles"]["rows"]
    assert rows and all(not r["outcome"]["available"] for r in rows)
    assert all(not r["prediction"]["available"] for r in rows)
    assert "no journal" in rows[0]["outcome"]["reason"]
    assert record["m5"]["panels"]["59990"] == {
        "available": False,
        "reason": "force/s50000-panel59990.json: absent",
    }
    assert record["m5"]["panels_agreement"] is None
    assert record["m3"]["per_seed"]["50003"]["r_theta0_source"] == "linear"
    assert {"what": "tokens/s50003.json",
            "reason": "tokens/s50003.json: absent"} in record["missing"]  # fmt: skip
    assert record["criterion"]["status"] == "decided"


def test_a_seed_without_its_linear_record_withholds_the_criterion(pod4, tmp_path) -> None:
    import shutil

    shutil.rmtree(pod4.records / "linear" / "s50020")
    out = tmp_path / "ext2.json"
    assert call(pod4.argv(out)) == (0, "")
    c = json.loads(out.read_text())["criterion"]
    assert c["status"] == "withheld" and c["passed"] is None and c["primary_predictor"] is None
    assert c["withheld"] == ["s50020: no linear record (linear/s50020/record.json: absent)"]


@pytest.mark.parametrize("remove, why", [
    ("vectors/m1/record.json", "the vectors record is required"),
    ("linear", "the linear block is required"),
])  # fmt: skip
def test_a_missing_required_record_refuses(pod4, tmp_path, remove, why) -> None:
    import shutil

    target = pod4.records / remove
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    out = tmp_path / "ext2.json"
    code, message = call(pod4.argv(out))
    assert code == 1 and why in message and not out.exists()


def test_another_analysis_record_than_the_registered_one_refuses(pod4, tmp_path,
                                                                 monkeypatch) -> None:  # fmt: skip
    monkeypatch.setattr(ecli, "REGISTERED_ANALYSIS_SHA256", "0" * 64)
    out = tmp_path / "ext2.json"
    code, message = call(pod4.argv(out))
    assert code == 1 and "not the registered analysis record" in message and not out.exists()


def no_outcome_read(monkeypatch, sealed: str) -> None:
    """Fail the test if anything but the sealed run's config is read through the registered
    readers."""
    real_f_run, real_at = cli.f_run, cli.Labels.at

    def f_run(reader, run_dir, *a, **kw):
        assert sealed not in str(run_dir), "an E2 run's records were read"
        return real_f_run(reader, run_dir, *a, **kw)

    def at(self, run_dir, condition):
        assert sealed not in str(run_dir), "an E2 run's labels were read"
        return real_at(self, run_dir, condition)

    monkeypatch.setattr(cli, "f_run", f_run)
    monkeypatch.setattr(cli.Labels, "at", at)


def test_an_e2_run_directory_by_its_config_refuses(pod4, tmp_path, monkeypatch) -> None:
    """An edited cell's journal naming a run whose config is an E2 order repeat (seed 50006,
    order seed 51006): the analysis refuses before it reads anything else of that run."""
    import shutil

    cfg = {
        "run_type": "train",
        "seed": 50006,
        "optimizer": {"name": "adam_frozen", "lr": 1.3e-05},
        "train": {"order_seed": 51006},
        "eval": {"harmful": {"prompt_set": SET}},
    }
    for path in (pod4.runs / "study5").glob("*-s5-edit-swap12-*-s50002-*"):
        shutil.rmtree(path)
    (pod4.runs / "launch" / f"study5-{ARCHIVE}-swaps-s50002.jsonl").unlink()
    for tag in ("swap2", "swap23"):
        pod4.edited_run(50002, tag, labeled=None)
    sealed = pod4.edited_run(50002, "swap12", labeled=7, config=cfg)
    no_outcome_read(monkeypatch, sealed.name)
    out = tmp_path / "ext2.json"
    code, message = call(pod4.argv(out))
    assert code == 1 and "E2" in message and "51006" in message and not out.exists()


def test_an_e2_run_directory_by_its_name_refuses(pod4, tmp_path, monkeypatch) -> None:
    """A vector record pointing at a run directory whose id holds an E2 cell refuses."""
    name = "20261111T0Z-s5-order1-1.3e-05-s50006-s50006-ab12-a1"
    pod4.vector_run("mix3", 50000, (1.0,), full=(1.0,), match="matched", labeled=3, run_id=name)
    no_outcome_read(monkeypatch, name)
    out = tmp_path / "ext2.json"
    code, message = call(pod4.argv(out))
    assert code == 1 and "order repeat (E2) by its name" in message and not out.exists()


def test_the_cli_is_analysis_code_outside_the_gpu_identity() -> None:
    from zero_field_refusal import code_identity

    for path in ("src/zero_field_refusal/analysis/study5_ext2.py",
                 "src/zero_field_refusal/analysis/study5_ext2_cli.py",
                 "tests/test_study5_ext2_analysis.py"):  # fmt: skip
        assert not code_identity.is_gpu_path(path)
