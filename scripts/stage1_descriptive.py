"""Descriptive stage-1 analyses for the short report of the corrected pilot (reports/gate-1.md).

    python scripts/stage1_descriptive.py RUNS_ROOT --gate1-15 GATE1_JSON --gate1-05 GATE1_JSON
        --qualification QUALIFICATION_JSON --build BUILD_DIR --out DIR

Reads the Gate 1 reports of both models (scripts/gate1_report.py), the judge-check qualification
(scripts/judge_check_qualify.py), the 1.5B analysis build (scripts/build_analysis.py) and the
run directories under RUNS_ROOT. It writes into DIR, which must not exist:

- ``descriptive.json``: the sections below;
- ``manifest.json``: the sha256 of every file read, the command, the code revision and the
  package versions, and the sha256 of descriptive.json.

Sections:

- the effect cell's seed spread (both models);
- utility per cell;
- the 48-token bridge;
- token-count normalization;
- the line scans (A2: ratios with a bootstrap over the KL prompts, and the per-run refusal
  asymmetry with Newcombe's method 10);
- the soft-target control;
- the components of the 0.5B design;
- SGD's cliff, by K and utility and from the per-step logs;
- the judge's agreement;
- the gradient and displacement projections;
- the 1.5B analysis build with its exclusions, unrounded.

No model, judge or GPU is used. Aggregates only, never texts.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from zero_field_refusal.analysis.eligibility import cell_eligibility
from zero_field_refusal.analysis.evidence import read_runs
from zero_field_refusal.analysis.outputs import file_sha256, write_json, write_new_directory
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import judge_identity
from zero_field_refusal.tracking import git_state

JUDGE = "wildguard"
SCAN_A = (0.25, 0.5, -0.25, -0.5, -1.0)
RESAMPLES = 4000
SEED = 20261012
PACKAGES = ("numpy", "scipy", "pandas", "zero-field-refusal")


class Inputs:
    """Every file read, with its sha256, keyed by its path relative to the runs root."""

    def __init__(self, root: Path) -> None:
        self.root, self.sha256 = Path(root), {}

    def text(self, path: Path) -> str:
        data = Path(path).read_bytes()
        self.sha256[self._key(path)] = hashlib.sha256(data).hexdigest()
        return data.decode()

    def json(self, path: Path):
        return json.loads(self.text(path))

    def jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in self.text(path).splitlines() if line.strip()]

    def note(self, path: Path) -> None:
        self.sha256[self._key(path)] = file_sha256(path)

    def _key(self, path: Path) -> str:
        path = Path(path).resolve()
        try:
            return path.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)


# --- statistics -------------------------------------------------------------------------------


def t_interval(values) -> dict:
    """Mean, SD and 95 % t-interval over the values (seeds)."""
    v = np.asarray(list(values), float)
    n, mean = len(v), float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else math.nan
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else math.nan
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def wilson(x: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = x / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def newcombe_paired(pairs: list[tuple[bool, bool]]) -> dict:
    """Newcombe (1998), method 10: the difference p₁ − p₂ of two paired proportions with a 95 %
    score interval (Wilson limits combined through the phi correlation, no continuity
    correction)."""
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    d = sum(1 for x, y in pairs if not x and not y)
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(denom) if denom > 0 else 0.0
    diff = p1 - p2
    low = diff - math.sqrt(
        max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0)
    )
    high = diff + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2,
                                0))  # fmt: skip
    return {"n": n, "p1": p1, "p2": p2, "difference": diff, "ci_low": low, "ci_high": high,
            "discordant": [b, c]}  # fmt: skip


# --- readers ----------------------------------------------------------------------------------


def labels_of(io: Inputs, run_dir: Path, identity: dict) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for r in io.jsonl(run_dir / "judge" / f"{JUDGE}.jsonl"):
        if any(r.get(k) != identity.get(k) for k in identity):
            raise ValueError(f"{run_dir}: a label record from another judge identity")
        if r["label"] is not None or r["item_id"] not in out:
            out[r["item_id"]] = r["label"]
    return out


def run_dirs(root: Path, experiment: str, tag: str) -> list[Path]:
    return sorted((root / experiment).glob(f"*Z-{tag}-*"))


# --- sections ---------------------------------------------------------------------------------


def seed_spread(g: dict) -> dict:
    eff = g["effect"]
    runs = {r["run_id"]: r for r in g["runs"]}
    rows = [{key: runs[run_id][key] for key in ("seed", "K", "refusal_rate", "harmful_rate",
                                                "refusal_drop", "factual_drop", "benchmark_drop")}
            for run_id in eff["runs"]]  # fmt: skip
    rho = float(stats.spearmanr(np.log([r["K"] for r in rows]),
                                [r["refusal_drop"] for r in rows]).statistic)  # fmt: skip
    sd = eff["refusal_drop"]["sd"]
    return {
        "runs": rows,
        "refusal_drop": eff["refusal_drop"],
        "harmful_rise": eff["harmful_rise"],
        "spearman_logK_drop": rho,
        "half_width_at_observed_sd": float(stats.t.ppf(0.975, 4) * sd / math.sqrt(5)),
        "half_width_at_assumed_sd": float(stats.t.ppf(0.975, 4) * 0.15 / math.sqrt(5)),
    }


def per_cell(g: dict) -> list[dict]:
    """Seed means of each cell of a Gate 1 report's runs table."""
    cells = defaultdict(list)
    for r in g["runs"]:
        cells[(r["experiment"], r["arm"], r["lr"])].append(r)
    out = []
    for (experiment, arm, lr), rs in sorted(cells.items()):
        mean = {q: float(np.mean([r[q] for r in rs])) for q in (
            "refusal_rate", "harmful_rate", "refusal_drop", "harmful_rise", "factual_drop",
            "benchmark_drop")}  # fmt: skip
        out.append({"experiment": experiment, "arm": arm, "lr": lr, "runs": len(rs),
                    "seeds": sorted(r["seed"] for r in rs),
                    "geometric_mean_K": float(np.exp(np.mean(np.log([r["K"] for r in rs])))),
                    **mean})  # fmt: skip
    return out


def bridge(io: Inputs, root: Path, model: str, g: dict, identity: dict) -> dict:
    reading = read_runs(root, f"stage1_bridge48_{model}", JUDGE, identity)
    io.sha256.update(reading.files)
    th = g["theta0"]
    inc = reading.considered[reading.considered["included"]]
    by_seed = {}
    for run_id, seed, k in zip(inc["run_id"], inc["seed"], inc["K"], strict=True):
        counts, summary = reading.label_counts[run_id], reading.summaries[run_id]
        judged = sum(counts.values())
        by_seed[int(seed)] = {
            "K": float(k), "refusal_drop": th["refusal_rate"] - counts.get("refusal", 0) / judged,
            "harmful_rise": counts.get("harmful_compliance", 0) / judged - th["harmful_rate"],
            "factual_drop": th["factual_accuracy"] - summary["factual_accuracy"],
            "benchmark_drop": th["benchmark_accuracy"] - summary["benchmark_accuracy"]}  # fmt: skip
    main = {r["seed"]: r for r in g["runs"] if r["experiment"] == f"stage1_main_{model}"
            and r["arm"] == "adamw" and r["lr"] == 5e-5}  # fmt: skip
    seeds = sorted(set(by_seed) & set(main))
    left_out = reading.considered[~reading.considered["included"]]
    out = {"seeds": seeds,
           "excluded": dict(zip(left_out["run_id"], left_out["reason"], strict=True))}  # fmt: skip
    for q in ("refusal_drop", "harmful_rise", "factual_drop", "benchmark_drop"):
        out[q] = {"bridge48": t_interval(by_seed[s][q] for s in seeds),
                  "main96": t_interval(main[s][q] for s in seeds),
                  "main96_minus_bridge48": t_interval(main[s][q] - by_seed[s][q]
                                                      for s in seeds)}  # fmt: skip
    out["K_geometric_mean"] = {
        "bridge48": float(np.exp(np.mean(np.log([by_seed[s]["K"] for s in seeds])))),
        "main96": float(np.exp(np.mean(np.log([main[s]["K"] for s in seeds]))))}  # fmt: skip
    return out


def token_count(g: dict, model: str) -> dict:
    runs = [r for r in g["runs"] if r["experiment"] == f"stage1_token_count_{model}"]
    main = {r["seed"]: r for r in g["runs"] if r["experiment"] == f"stage1_main_{model}"
            and r["arm"] == "adamw" and r["lr"] == 5e-5}  # fmt: skip
    by_seed = {r["seed"]: r for r in runs}
    seeds = sorted(set(by_seed) & set(main))
    return {"seeds": seeds,
            "token_count": t_interval(by_seed[s]["refusal_drop"] for s in seeds),
            "constant": t_interval(main[s]["refusal_drop"] for s in seeds),
            "constant_minus_token_count": t_interval(main[s]["refusal_drop"]
                                                     - by_seed[s]["refusal_drop"] for s in seeds),
            "effect_disappears": None if g.get("effect_disappears") is None
            else g["effect_disappears"]["disappears"]}  # fmt: skip


def kl_values(io: Inputs, run_dir: Path) -> dict[str, dict[str, list[float]]]:
    vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in io.jsonl(run_dir / "observations" / "measurements.jsonl"):
        if r["measure"] == "kl_harmful":
            vals[r["condition"]][r["item_id"]].append(r["value"])
    return vals


def a2_run(io: Inputs, run_dir: Path, identity: dict, rng: np.random.Generator) -> dict:
    vals = kl_values(io, run_dir)
    prompts = sorted(vals["line_scan_+1"])
    sums = {c: np.array([sum(vals[c][p]) for p in prompts]) for c in vals}
    counts = {c: np.array([len(vals[c][p]) for p in prompts]) for c in vals}

    def k_of(condition: str, idx: np.ndarray) -> np.ndarray:
        return sums[condition][idx].sum(axis=-1) / counts[condition][idx].sum(axis=-1)

    idx = rng.integers(0, len(prompts), size=(RESAMPLES, len(prompts)))
    full = np.arange(len(prompts))[None]
    k1, k1_boot = float(k_of("line_scan_+1", full)[0]), k_of("line_scan_+1", idx)
    rel_se = float(k1_boot.std(ddof=1) / k1)
    ratios = {}
    for a in SCAN_A:
        condition = f"line_scan_{a:+g}"
        boot = k_of(condition, idx) / (a * a * k1_boot)
        low, high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        reading = ("faster" if low > 1.25 else "slower" if high < 0.8
                   else "meets" if low <= 1.25 and high >= 0.8 else "between")  # fmt: skip
        ratios[f"{a:+g}"] = {"R": float(k_of(condition, full)[0]) / (a * a * k1),
                             "ci_low": low, "ci_high": high, "reading": reading}  # fmt: skip
    labels = labels_of(io, run_dir, identity)
    responses = io.jsonl(run_dir / "responses" / "harmful_judged.jsonl")
    judged = sorted({r["item_id"] for r in responses if r["condition"] == "line_scan_+1"})
    pairs = [(labels.get(f"line_scan_+1/{p}") == "refusal",
              labels.get(f"line_scan_-1/{p}") == "refusal") for p in judged]  # fmt: skip
    unparsed = sum(labels.get(f"line_scan_{s}/{p}") is None for s in ("+1", "-1") for p in judged)
    random = {kind: float(np.mean([v for p in prompts for v in vals[kind][p]])) / k1
              for kind in ("random_global", "random_per_tensor")}  # fmt: skip
    return {"K1": k1, "relative_se_K1": rel_se,
            "evaluable": k1 >= 1.0 and rel_se <= 0.10, "ratios": ratios,
            "random_K_over_K1": random,
            "refusal_plus_minus": newcombe_paired(pairs), "unparsed": unparsed}  # fmt: skip


def a2(io: Inputs, root: Path, identity: dict) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for model in ("1.5b", "0.5b"):
        experiment = f"stage1_scans_{model}"
        for run_dir in sorted((root / experiment).glob("*")):
            out[run_dir.name] = {"experiment": experiment, **a2_run(io, run_dir, identity, rng)}
    return out


def soft_targets(io: Inputs, root: Path, g: dict, identity: dict) -> dict:
    soft = next((root / "stage1_soft_targets_1.5b").glob("*"))
    hard = run_dirs(root, "stage1_main_1.5b", "sgd-0.0280092-1.5b-s0")[0]
    responses = io.jsonl(hard / "responses" / "harmful_judged.jsonl")
    prompts = sorted({r["item_id"] for r in responses if r["condition"] == "trained"})
    th, out, labels = g["theta0"], {}, {}
    for name, run_dir in (("hard", hard), ("soft", soft)):
        labels[name] = labels_of(io, run_dir, identity)
        summary = io.json(run_dir / "summary.json")
        said = Counter(labels[name].get(f"trained/{p}") for p in prompts)
        out[name] = {"run_id": run_dir.name, "K": summary["K_harmful"],
                     "refusal_rate": said["refusal"] / len(prompts),
                     "harmful_rate": said["harmful_compliance"] / len(prompts),
                     "unparsed": said[None],
                     "factual_drop": th["factual_accuracy"] - summary["factual_accuracy"],
                     "benchmark_drop": th["benchmark_accuracy"]
                     - summary["benchmark_accuracy"]}  # fmt: skip
    out["refusal_soft_minus_hard"] = newcombe_paired(
        [(labels["soft"].get(f"trained/{p}") == "refusal",
          labels["hard"].get(f"trained/{p}") == "refusal") for p in prompts])  # fmt: skip
    out["theta0_refusal_rate"] = th["refusal_rate"]
    return out


def components_05b(root: Path, g05: dict, cells05: list[dict], bridge05: dict,
                   token05: dict) -> dict:  # fmt: skip
    adamw = {c["lr"]: c for c in cells05
             if c["experiment"] == "stage1_main_0.5b" and c["arm"] == "adamw"}  # fmt: skip
    has_soft = (root / "stage1_soft_targets_0.5b").exists()
    return {
        "effect_5e-5": {"status": g05["effect"]["status"],
                        "refusal_drop": g05["effect"]["refusal_drop"],
                        "harmful_rise": g05["effect"]["harmful_rise"],
                        "factual_drop": g05["effect"]["factual_drop"]},
        "adamw_cells": {f"{lr:g}": {k: c[k] for k in ("geometric_mean_K", "refusal_drop",
                                                     "harmful_rise", "factual_drop",
                                                     "benchmark_drop")}
                        for lr, c in sorted(adamw.items())},
        "token_count": token05,
        "bridge48": {q: bridge05[q]["bridge48"]["mean"] for q in (
            "refusal_drop", "harmful_rise", "factual_drop", "benchmark_drop")},
        "sgd": "no stable rate: every pilot from 1/300 to 9 collapsed, 27 diverged",
        "soft_targets": "run" if has_soft else "not run: SGD has no stable rate on 0.5B",
        "a1": "a diagnostic on 0.5B, as on 1.5B (stage-0 decision): reported, not tested",
    }  # fmt: skip


def sgd_cliff(io: Inputs, root: Path, model: str, g: dict) -> dict:
    th = {"factual": g["theta0"]["factual_accuracy"],
          "benchmark": g["theta0"]["benchmark_accuracy"]}  # fmt: skip
    experiments = ([f"stage1_main_{model}", "stage1_sgd100x5_1.5b"] if model == "1.5b"
                   else [f"stage1_main_{model}"])  # fmt: skip
    cells = []
    for experiment in experiments:
        for run_dir in sorted((root / experiment).glob("*")):
            for name in ("meta.json", "config.yaml", "summary.json"):
                if (run_dir / name).exists():
                    io.note(run_dir / name)
        table = cell_eligibility(root, experiment, th)
        for r in table[table["arm"] == "sgd"].itertuples():
            cells.append({"experiment": experiment, "lr": r.lr, "completed": r.completed,
                          "diverged": bool(r.diverged),
                          "geometric_mean_K": math.exp(r.mean_log_k),
                          "factual_drop": th["factual"] - r.factual,
                          "benchmark_drop": th["benchmark"] - r.benchmark,
                          "collapse": r.collapse, "eligible": bool(r.eligible)})  # fmt: skip
    out = {"cells": cells}
    if model != "1.5b":
        return out

    def steps(experiment: str, tag: str) -> dict:
        run_dir = run_dirs(root, experiment, tag)[0]
        rows = io.jsonl(run_dir / "metrics.jsonl")
        k = io.json(run_dir / "summary.json")["K_harmful"]
        return {"run_id": run_dir.name, "K": k,
                "steps": [{q: r[q] for q in ("step", "loss", "grad_norm", "displacement_norm")}
                          for r in rows]}  # fmt: skip

    out["seed0_pairs"] = {
        "main": {"stable": steps("stage1_main_1.5b", "sgd-0.0280092-1.5b-s0"),
                 "collapsed": steps("stage1_main_1.5b", "sgd-0.03-1.5b-s0")},
        "100x5": {"stable": steps("stage1_sgd100x5_1.5b", "sgd100x5-0.0198701-1.5b-s0"),
                  "collapsed": steps("stage1_sgd100x5_1.5b", "sgd100x5-0.0212824-1.5b-s0")},
    }  # fmt: skip
    amplification = []
    for experiment, prefix in (("stage1_main_1.5b", "sgd"), ("stage1_sgd100x5_1.5b", "sgd100x5")):
        for run_dir in sorted((root / experiment).glob(f"*Z-{prefix}-*-1.5b-s0-*")):
            rows = io.jsonl(run_dir / "metrics.jsonl")
            g = [r["grad_norm"] for r in rows[:9]]
            amplification.append({"experiment": experiment, "run_id": run_dir.name,
                                  "lr": float(run_dir.name.split("Z-")[1].split("-")[1]),
                                  "peak_over_step0": max(g) / g[0],
                                  "peak_step": int(np.argmax(g))})  # fmt: skip
    out["early_amplification_seed0"] = sorted(amplification,
                                              key=lambda a: (a["experiment"], a["lr"]))  # fmt: skip
    return out


def judge(io: Inputs, root: Path, g: dict, qualification: dict, identity: dict) -> dict:
    strata = {}
    for variants in qualification["panels"].values():
        for outcome in variants.values():
            for s, st in outcome["result"]["strata"].items():
                strata[s] = {k: st[k] for k in ("n", "agreed", "agreement", "harmful",
                                                "not_harmful", "se", "fp",
                                                "discrimination_lower")}  # fmt: skip
    unparsed = {}
    for experiment in ("stage1_main_1.5b", "stage1_sgd100x5_1.5b", "stage1_main_0.5b"):
        for run_dir in sorted((root / experiment).glob("*")):
            if not (run_dir / "judge" / f"{JUDGE}.jsonl").exists():
                continue
            n = sum(1 for k, v in labels_of(io, run_dir, identity).items()
                    if k.startswith("trained/") and v is None)  # fmt: skip
            if n:
                unparsed[run_dir.name] = n
    agreement = g["calibration"]["agreement"]
    return {"calibration": {"accuracy": g["calibration"]["accuracy"], "n": g["calibration"]["n"],
                            "kappa": agreement["kappa"], "confusion": agreement["confusion"]},
            "judge_check_strata": strata,
            "not_harmful_tables": qualification["not_harmful_tables"],
            "unparsed_trained_responses": unparsed}  # fmt: skip


def projections(io: Inputs, root: Path) -> list[dict]:
    rows = []
    for experiment in ("stage1_main_1.5b", "stage1_sgd100x5_1.5b", "stage1_main_0.5b"):
        for run_dir in sorted((root / experiment).glob("*")):
            meta = io.json(run_dir / "meta.json")
            arm = run_dir.name.split("Z-", 1)[1].split("-")[0]
            if meta.get("status") != "completed" or arm == "theta0":
                continue
            steps = io.jsonl(run_dir / "metrics.jsonl")
            disp = np.array([s["displacement_projection"] for s in steps])
            grad = np.array([s["grad_projection"] for s in steps])
            increments = np.diff(np.vstack([np.zeros((1, disp.shape[1])), disp]), axis=0)
            drift = disp[-1] ** 2 / np.maximum((increments**2).sum(axis=0), 1e-300)
            z = grad.mean(axis=0) / (grad.std(axis=0, ddof=1) / math.sqrt(len(grad)))
            rows.append({"experiment": experiment, "run_id": run_dir.name, "arm": arm,
                         "steps": len(steps), "grad_norm_step0": steps[0]["grad_norm"],
                         "displacement_norm_final": steps[-1]["displacement_norm"],
                         "drift_ratio": drift.tolist(),
                         "grad_projection_z": z.tolist()})  # fmt: skip
    return rows


def build_summary(io: Inputs, build: Path) -> dict:
    results = io.json(build / "results.json")
    io.note(build / "manifest.json")
    estimates = {f: pd.read_csv(build / f, float_precision="round_trip")
                 for f in ("estimates.csv", "all_cells_estimates.csv")}  # fmt: skip
    for f in (*estimates, "exclusions.csv", "all_cells_arms.csv", "arms.csv"):
        io.note(build / f)

    def at_k_star(table: pd.DataFrame) -> dict:
        rows = table[table["level"] == "k_star"]
        return {f"{q} ({a})": {k: float(r[k]) for k in ("K", "estimate", "ci_low", "ci_high")}
                for q, a, (_, r) in zip(rows["quantity"], rows["arms"], rows.iterrows(),
                                        strict=True)}  # fmt: skip

    exclusions = pd.read_csv(build / "exclusions.csv")
    every = results["all_cells"]
    return {
        "eligible": {"selection": results["analysis"]["selection"],
                     "decision_label": results["analysis"]["decision"],
                     "at_k_star": at_k_star(estimates["estimates.csv"])},
        "all_cells": {"selection": every["k_star"],
                      "decision_label": every["analysis"]["decision"],
                      "at_k_star": at_k_star(estimates["all_cells_estimates.csv"]),
                      "arms": pd.read_csv(build / "all_cells_arms.csv").to_dict("records"),
                      "fit_exclusion_threshold": (every["k_star"]["c_lo"]
                                                  / every["k_star"]["exclusion_factor"])},
        "exclusions": exclusions[["run_id", "reason", "detail"]].to_dict("records"),
        "settings": results["settings"],
    }  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--gate1-15", type=Path, required=True, dest="gate1_15")
    parser.add_argument("--gate1-05", type=Path, required=True, dest="gate1_05")
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: results are never overwritten")
    root, io = args.runs_root, Inputs(args.runs_root)
    identity = judge_identity(load_config(f"judge/{JUDGE}"))
    g15, g05 = io.json(args.gate1_15), io.json(args.gate1_05)
    qualification = io.json(args.qualification)
    cells05 = per_cell(g05)
    bridge05 = bridge(io, root, "0.5b", g05, identity)
    token05 = token_count(g05, "0.5b")
    result = {
        "note": "descriptive; comparisons between optimizers are exploratory and preliminary",
        "seed_spread": {"1.5b": seed_spread(g15), "0.5b": seed_spread(g05)},
        "cells": {"1.5b": per_cell(g15), "0.5b": cells05},
        "bridge48": {"1.5b": bridge(io, root, "1.5b", g15, identity), "0.5b": bridge05},
        "token_count": {"1.5b": token_count(g15, "1.5b"), "0.5b": token05},
        "a2": a2(io, root, identity),
        "soft_targets": soft_targets(io, root, g15, identity),
        "components_0.5b": components_05b(root, g05, cells05, bridge05, token05),
        "sgd_cliff": {"1.5b": sgd_cliff(io, root, "1.5b", g15),
                      "0.5b": sgd_cliff(io, root, "0.5b", g05)},
        "judge": judge(io, root, g15, qualification, identity),
        "projections": projections(io, root),
        "build_1.5b": build_summary(io, args.build),
    }  # fmt: skip

    def write(directory: Path) -> None:
        write_json(directory / "descriptive.json", result)
        write_json(directory / "manifest.json", {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": [Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
            "code": {"git": git_state(), "python": platform.python_version(),
                     "packages": {name: metadata.version(name) for name in PACKAGES}},
            "settings": {"resamples": RESAMPLES, "seed": SEED, "judge": JUDGE},
            "inputs": dict(sorted(io.sha256.items())),
            "outputs": {"descriptive.json": file_sha256(directory / "descriptive.json")},
        })  # fmt: skip

    write_new_directory(args.out, write)
    print(json.dumps({"out": str(args.out), "inputs": len(io.sha256)}))


if __name__ == "__main__":
    main()
