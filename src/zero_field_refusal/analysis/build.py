"""Versioned analysis outputs of one experiment, rebuilt on the CPU from its run directories.

``build_analysis`` reads the runs (``evidence.collect_runs``), fixes K\\* (selected from the
included runs, or read frozen from an earlier build), runs ``analyze`` with the stage-1 interval
settings and writes the outputs into a new directory. An existing directory is never
overwritten: results are versioned. The directory is written under a temporary name and renamed
when complete, so it appears whole or not at all (``outputs.write_new_directory``). Numbers are
written unrounded (shortest exact decimal); non-finite numbers are empty in CSV files and null in
JSON.

- ``runs.csv``: every run directory considered, with ``included``, ``reason`` and ``detail``
  (``evidence.CONSIDERED_COLUMNS``);
- ``exclusions.csv``: the runs left out and why;
- ``cells.csv``: per cell (arm, lr): the included runs and their seeds, the seeds of the analysis
  without a run in the cell (``absent_seeds``), the mean of log K and whether the cell is decisive;
- ``arms.csv``: per arm, the curve logit h = intercept + slope·log K (K in nats), the runs used and
  excluded, and the fits without a unique estimate or without convergence (point, resamples,
  jackknife);
- ``estimates.csv``: every interval, with its level K, estimate, CI, SE and unit: h of each arm and
  Δ = h_AdamW − h_SGD at K\\* and the quarter points, and the H2 differences at K\\*;
- ``k_star.json``: the K\\* selection used, its status, the runs it was computed from (with their
  gates) and, for a frozen K\\*, the file it was read from and that file's sha256;
- ``results.json``: settings, units, the analyzed runs, ``k_star_status``, the K\\* record and the
  whole ``AnalysisResult``;
- ``figure.svg``: harmful-compliance rate against K per arm with the fitted curves, drawn from
  ``runs.csv``, ``arms.csv`` and ``results.json`` alone (``draw_figure``);
- ``manifest.json``: the sha256 of every input file read, the run ids, analysis settings, RNG
  seed, protocol versions of the runs, judge identity, code revision (``tracking.git_state``),
  package versions, the sha256 of every other output and ``created_at``.

A rebuild from the same inputs with the same code and packages writes the same bytes, except
``created_at`` in the manifest.

K\\* comes from one of two sources:
- *provisional* (``K_STAR_STATUS``, no ``k_star`` given): selected from the analyzed runs
  themselves, a sweep-style selection that suits the stage-0 fixture check and the stage-1 sweep.
  It is not the Gate 2 K\\*, which docs/preregistration.md fixes from the sweep's KL values before
  look 1 and never re-estimates, so the build refuses it when any run directory of the experiment
  is a Gate 2 run (``tracking.gate: 2``, failed runs included);
- *frozen* (``K_STAR_FROZEN``): ``k_star`` names the ``k_star.json`` of an earlier build, whose
  selection is used as it is. A selection computed from Gate 2 runs is refused: the Gate 2 K\\*
  comes from the sweep alone. Freezing the stage-1 sweep build's ``k_star.json`` before look 1
  fixes the Gate 2 K\\*; look 2 reads the same file again.

Per-run look membership is not an input yet: a build analyzes every included run, so the look-1
analysis must be built before any look-2 run joins the experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy.special import expit

from ..tracking import git_state
from .evidence import Evidence, collect_runs
from .kstar import LEVEL_NAMES, KStarSelection, cell_means, k_star_levels, select_k_star
from .outputs import file_sha256, read_csv, write_csv, write_json, write_new_directory
from .primary import FROZEN_VS_ADAMW, FROZEN_VS_SGD, PRIMARY_CONTRAST, AnalysisResult, analyze
from .runs import ARMS


@dataclass(frozen=True)
class IntervalSettings:
    ci_method: str
    level: float  # two-sided, per look
    resamples: int


# docs/preregistration.md, "Interval": percentile CI at 99.5 % per look, 4000 resamples (stage 1).
STAGE1_INTERVAL = IntervalSettings(ci_method="percentile", level=0.995, resamples=4000)

# K* is selected from the analyzed runs (select_k_star): not the frozen Gate 2 K*.
K_STAR_STATUS = "provisional_from_analyzed_runs"
# K* read from an earlier build's k_star.json and used unchanged: what a Gate 2 look needs.
K_STAR_FROZEN = "frozen"
K_STAR_FILE = "k_star.json"
GATE_2 = 2
PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "omegaconf", "zero-field-refusal")
UNITS = {
    "K": "nats: the sequence-level KL(θ₀ ‖ θ) per θ₀ response to the KL prompts (K_harmful)",
    "lr": "the optimizer's learning rate as configured",
    "h": "probability that a judged response is labeled harmful_compliance",
    "delta": "difference of two arms' h at the same K (probability difference)",
    "intercept": "logit h at K = 1 nat",
    "slope": "change of logit h per unit of ln(K / 1 nat)",
    "harmful": "count of responses labeled harmful_compliance",
    "judged": "count of judged responses of the trained model",
}
CONTRASTS = {"delta": PRIMARY_CONTRAST, "h2_vs_sgd": FROZEN_VS_SGD, "h2_vs_adamw": FROZEN_VS_ADAMW}


def build_analysis(
    root: Path,
    experiment: str,
    judge: str,
    identity: Mapping[str, Any],
    out: Path,
    *,
    seed: int = 0,
    interval: IntervalSettings = STAGE1_INTERVAL,
    k_star: Path | None = None,
) -> Path:
    """Analyze ``<root>/<experiment>`` with the labels of ``judge`` into the new directory ``out``.

    ``identity`` is the judge's (``judge.judge_identity``); ``seed`` seeds the bootstrap;
    ``k_star`` is the ``k_star.json`` of an earlier build to use as the frozen K\\* (None: select a
    provisional one from the analyzed runs, refused for Gate 2 runs). Put ``out`` outside the
    checkout or under an ignored path: the manifest records whether the checkout is dirty, and an
    untracked ``out`` inside it (or its partial directory) makes it so.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: results are versioned, never overwritten")
    evidence = collect_runs(root, experiment, judge, identity)
    if k_star is None:
        gate_2 = sorted(evidence.considered.loc[_is_gate_2(evidence.considered), "run_id"])
        if gate_2:
            raise ValueError(
                f"{len(gate_2)} run directories are Gate 2 runs (tracking.gate: 2; first: "
                f"{gate_2[0]}): their K* must be the frozen one, fixed from the sweep before look "
                f"1. Pass the {K_STAR_FILE} of the build that fixed it (k_star, --k-star); a K* "
                "selected from these runs is not allowed"
            )
        selection = select_k_star(evidence.runs)
        k_star_record = {"status": K_STAR_STATUS, "computed_from": _computed_from(evidence),
                         "source": None}  # fmt: skip
    else:
        selection, k_star_record = read_k_star(k_star)
    result = analyze(
        evidence.runs,
        selection,
        level=interval.level,
        ci_method=interval.ci_method,
        resamples=interval.resamples,
        seed=seed,
    )
    return write_new_directory(
        out,
        lambda directory: _write_outputs(
            directory, evidence, selection, k_star_record, result, interval, seed
        ),
    )


def read_k_star(path: Path) -> tuple[KStarSelection, dict[str, Any]]:
    """The frozen K\\* in the ``k_star.json`` of an earlier build, and its record for this build.

    Raises ``ValueError`` if the file is not such a record, if its levels do not follow from its
    common range, or if its selection was computed from Gate 2 runs.
    """
    path = Path(path)
    data = path.read_bytes()
    try:
        saved = json.loads(data)
        status, selection, computed_from = (saved["status"], saved["selection"],
                                            saved["computed_from"])  # fmt: skip
        if status not in (K_STAR_STATUS, K_STAR_FROZEN):
            raise ValueError(f"unknown status {status!r}")
        gates = computed_from["gates"]
        if not (isinstance(computed_from["runs"], list) and isinstance(gates, list)):
            raise ValueError("computed_from needs the lists runs and gates")
        frozen = _selection(selection)
    except (ValueError, KeyError, TypeError) as exc:  # JSONDecodeError is a ValueError
        raise ValueError(f"{path} is not the {K_STAR_FILE} of a build: {exc}") from exc
    if GATE_2 in gates:
        raise ValueError(
            f"The K* in {path} was computed from Gate 2 runs; the Gate 2 K* is fixed from the "
            "sweep alone, before look 1"
        )
    record = {"status": K_STAR_FROZEN, "computed_from": computed_from,
              "source": {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                         "status": status}}  # fmt: skip
    return frozen, record


def _selection(data: Mapping[str, Any]) -> KStarSelection:
    """A ``KStarSelection`` from its JSON form (``k_star.json``), checked."""
    numbers = {name: data[name] for name in ("c_lo", "c_hi", *LEVEL_NAMES)}
    for name, value in numbers.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise ValueError(f"{name} must be a positive number, not {value!r}")
    if not numbers["c_lo"] <= numbers["c_hi"]:
        raise ValueError("c_lo exceeds c_hi")
    expected = dict(zip(LEVEL_NAMES, k_star_levels(numbers["c_lo"], numbers["c_hi"]), strict=True))
    if any(not math.isclose(numbers[name], expected[name], rel_tol=1e-12) for name in LEVEL_NAMES):
        raise ValueError(f"the levels {[numbers[n] for n in LEVEL_NAMES]} do not follow from "
                         f"[c_lo, c_hi] = [{numbers['c_lo']}, {numbers['c_hi']}]")  # fmt: skip
    if not isinstance(data["h2_descriptive_only"], bool):
        raise ValueError("h2_descriptive_only must be true or false")
    factor = data["exclusion_factor"]
    factor = math.inf if factor is None else factor  # JSON null: no exclusion
    if isinstance(factor, bool) or not isinstance(factor, int | float) or not factor > 0:
        raise ValueError(f"exclusion_factor must be a positive number or null, not {factor!r}")
    unknown = sorted({*data["arm_ranges"], *data["decisive"]} - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    return KStarSelection(
        arm_ranges={arm: (float(lo), float(hi)) for arm, (lo, hi) in data["arm_ranges"].items()},
        c_lo=float(numbers["c_lo"]),
        c_hi=float(numbers["c_hi"]),
        k_star=float(numbers["k_star"]),
        lower_quarter=float(numbers["lower_quarter"]),
        upper_quarter=float(numbers["upper_quarter"]),
        h2_descriptive_only=data["h2_descriptive_only"],
        decisive={arm: tuple(float(lr) for lr in lrs) for arm, lrs in data["decisive"].items()},
        exclusion_factor=float(factor),
    )


def _is_gate_2(considered: pd.DataFrame) -> pd.Series:
    return considered["gate"].eq(GATE_2).fillna(False).astype(bool)


def _computed_from(evidence: Evidence) -> dict[str, Any]:
    """The runs a provisional K\\* is selected from, with their gates."""
    included = evidence.considered[evidence.considered["included"]]
    return {"experiment": evidence.experiment, "runs": sorted(included["run_id"]),
            "gates": sorted(int(g) for g in included["gate"].dropna().unique())}  # fmt: skip


def _write_outputs(
    directory: Path,
    evidence: Evidence,
    selection: KStarSelection,
    k_star_record: Mapping[str, Any],
    result: AnalysisResult,
    interval: IntervalSettings,
    seed: int,
) -> None:
    considered = evidence.considered
    write_csv(considered, directory / "runs.csv")
    excluded = considered.loc[~considered["included"], ["run_id", "status", "reason", "detail",
                                                         "arm", "lr", "seed"]]  # fmt: skip
    write_csv(excluded, directory / "exclusions.csv")
    write_csv(_cells(evidence.runs, selection), directory / "cells.csv")
    write_csv(_arms(result), directory / "arms.csv")
    write_csv(_estimates(result, selection), directory / "estimates.csv")

    settings = {
        "ci_method": interval.ci_method,
        "level": interval.level,
        "resamples": interval.resamples,
        "seed": seed,
        "margin": result.margin,
        "exclusion_factor": selection.exclusion_factor,
    }
    analyzed = sorted(evidence.runs["run_id"])
    status = k_star_record["status"]
    write_json(directory / K_STAR_FILE, {**k_star_record, "selection": asdict(selection)})
    analysis = asdict(result)
    analysis["selection"]["exclusion_threshold"] = selection.exclusion_threshold
    write_json(directory / "results.json", {
        "experiment": evidence.experiment,
        "judge": {"name": evidence.judge, "identity": evidence.judge_identity},
        "settings": settings,
        "units": UNITS,
        "analyzed_runs": analyzed,
        "k_star_status": status,
        "k_star": k_star_record,
        "analysis": analysis,
    })  # fmt: skip
    draw_figure(directory)

    included = considered[considered["included"]]
    write_json(directory / "manifest.json", {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": evidence.experiment,
        "judge": {"name": evidence.judge, "identity": evidence.judge_identity},
        "runs": {
            "considered": considered["run_id"].tolist(),
            "included": included["run_id"].tolist(),
            "excluded": considered.loc[~considered["included"], "run_id"].tolist(),
        },
        "inputs": evidence.files,  # path relative to the runs root -> sha256
        "settings": settings,
        "k_star_status": status,
        "k_star": k_star_record,
        "rng_seed": seed,
        "protocol_versions": sorted(int(v) for v in included["protocol_version"].dropna().unique()),
        "code": {
            "git": git_state(),
            "python": platform.python_version(),
            "packages": {name: metadata.version(name) for name in PACKAGES},
        },
        "outputs": {path.name: file_sha256(path) for path in sorted(directory.iterdir())},
    })  # fmt: skip


def _cells(runs: pd.DataFrame, selection: KStarSelection) -> pd.DataFrame:
    seeds = sorted(set(runs["seed"]))
    rows = []
    for cell in cell_means(runs).itertuples(index=False):
        present = sorted(runs.loc[(runs["arm"] == cell.arm) & (runs["lr"] == cell.lr), "seed"])
        rows.append({
            "arm": cell.arm,
            "lr": cell.lr,
            "runs": cell.n_runs,
            "seeds": " ".join(str(s) for s in present),
            "absent_seeds": " ".join(str(s) for s in seeds if s not in present),
            "mean_log_K": cell.mean_log_k,
            "geometric_mean_K": math.exp(cell.mean_log_k),
            "decisive": cell.lr in selection.decisive.get(cell.arm, ()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def _arms(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for arm in sorted({*result.arms, *result.unfitted_arms}):
        if arm in result.unfitted_arms:  # all runs excluded: no curve
            rows.append({"arm": arm, "fitted": False, "runs_used": 0,
                         "runs_excluded": result.unfitted_arms[arm]})  # fmt: skip
            continue
        fit = result.arms[arm]
        rows.append({
            "arm": arm, "fitted": True, "runs_used": fit.runs_used,
            "runs_excluded": fit.runs_excluded, "intercept": fit.intercept, "slope": fit.slope,
            **{f"separated_{part}": n for part, n in fit.separated.items()},
            **{f"not_converged_{part}": n for part, n in fit.not_converged.items()},
        })  # fmt: skip
    return pd.DataFrame(rows)


def _estimates(result: AnalysisResult, selection: KStarSelection) -> pd.DataFrame:
    def row(quantity: str, arms: str, level: str, interval: Any, unit: str) -> dict[str, Any]:
        return {"quantity": quantity, "arms": arms, "level": level,
                "K": selection.levels[level], "estimate": interval.estimate,
                "ci_low": interval.ci_low, "ci_high": interval.ci_high, "se": interval.se,
                "dropped_replicates": interval.dropped_replicates,
                "ci_method": result.ci_method, "ci_level": result.level, "unit": unit}  # fmt: skip

    rows = [
        row("h", arm, level, result.arms[arm].h[level], "probability")
        for arm in sorted(result.arms)
        for level in LEVEL_NAMES
    ]
    delta = _contrast_name(CONTRASTS["delta"])
    rows += [row("delta", delta, level, result.primary[level], "probability difference")
             for level in LEVEL_NAMES]  # fmt: skip
    for quantity, interval in (("h2_vs_sgd", result.h2.vs_sgd),
                               ("h2_vs_adamw", result.h2.vs_adamw)):  # fmt: skip
        if interval is not None:  # None: no frozen-Adam curve
            rows.append(row(quantity, _contrast_name(CONTRASTS[quantity]), "k_star", interval,
                            "probability difference"))  # fmt: skip
    return pd.DataFrame(rows)


def _contrast_name(coefficients: Mapping[str, float]) -> str:
    """A contrast of unit coefficients as text: ``{"adamw": 1, "sgd": -1}`` -> ``adamw - sgd``."""
    terms = " ".join(f"{'-' if c < 0 else '+'} {arm}" for arm, c in coefficients.items())
    return terms.removeprefix("+ ")


# --- figure -------------------------------------------------------------------------------------

# Colour follows the arm, never its rank; markers and line styles back it up. Light surface.
ARM_STYLE = {
    "sgd": ("SGD", "#2a78d6", "o", "-"),
    "adamw": ("AdamW", "#eb6834", "s", "-"),
    "adam_frozen": ("Adam, frozen second moment", "#1baf7a", "^", "-"),
    "sgd_momentum": ("SGD with momentum", "#6250d6", "D", "--"),
    "adamw_prewarmed": ("AdamW, pre-warmed second moment", "#e87ba4", "v", "--"),
}
assert set(ARM_STYLE) == set(ARMS)
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Fixed font, SVG ids and metadata: the same tables give the same bytes.
FIGURE_RC = {
    "svg.hashsalt": "zero-field-refusal",
    "svg.fonttype": "path",
    "font.family": "DejaVu Sans",
    "font.size": 9,
}


def draw_figure(directory: Path) -> Path:
    """``figure.svg`` from ``runs.csv``, ``arms.csv`` and ``results.json`` in ``directory``."""
    runs = read_csv(directory / "runs.csv")
    runs = runs[runs["included"]]
    arms = read_csv(directory / "arms.csv").set_index("arm")
    results = json.loads((directory / "results.json").read_text())
    analysis = results["analysis"]
    selection = analysis["selection"]
    path = directory / "figure.svg"
    with matplotlib.rc_context():
        matplotlib.rcdefaults()  # ignore any local matplotlibrc
        matplotlib.rcParams.update(FIGURE_RC)
        fig = Figure(figsize=(6.4, 4.2), facecolor=SURFACE)
        ax = fig.add_subplot(facecolor=SURFACE)
        k = np.geomspace(runs["K"].min() / 1.5, runs["K"].max() * 1.5, 200)
        ax.axvspan(selection["c_lo"], selection["c_hi"], color=GRID, alpha=0.6, lw=0,
                   label="common K range")  # fmt: skip
        provisional = "provisional " if results["k_star_status"] == K_STAR_STATUS else ""
        ax.axvline(selection["k_star"], color=SECONDARY, ls=":", lw=1.2,
                   label=f"{provisional}K* = {selection['k_star']:.3g} nats")  # fmt: skip
        for arm, (name, color, marker, style) in ARM_STYLE.items():
            points = runs[runs["arm"] == arm]
            if points.empty:
                continue
            fitted = arm in arms.index and bool(arms.loc[arm, "fitted"])
            ax.scatter(points["K"], points["harmful"] / points["judged"], s=30, marker=marker,
                       color=color, edgecolors=SURFACE, linewidths=0.8, zorder=3)  # fmt: skip
            if fitted:
                curve = expit(arms.loc[arm, "intercept"] + arms.loc[arm, "slope"] * np.log(k))
                ax.plot(k, curve, color=color, ls=style, lw=2, zorder=2)
            # Legend entry: the arm's marker on its line style.
            ax.plot([], [], color=color, ls=style if fitted else "none", lw=2, marker=marker,
                    markersize=6, markeredgecolor=SURFACE,
                    label=name if fitted else f"{name} (no curve)")  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(k[0], k[-1])
        ax.xaxis.set_major_formatter(FuncFormatter(_log_tick))
        ax.xaxis.set_minor_formatter(FuncFormatter(_log_tick))
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Displacement K (nats, log scale)", color=SECONDARY)
        ax.set_ylabel("Harmful-compliance rate", color=SECONDARY)
        delta = analysis["primary"]["k_star"]
        ax.set_title(
            f"Δ(K*) = h_AdamW − h_SGD = {_number(delta['estimate'])}, "
            f"{_percent(results['settings']['level'])} CI [{_number(delta['ci_low'])}, "
            f"{_number(delta['ci_high'])}]: {analysis['decision']}",
            loc="left", fontsize=9, color=INK,
        )  # fmt: skip
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(path, format="svg", metadata={"Date": None}, facecolor=SURFACE)
    return path


def _log_tick(value: float, _: int) -> str:
    """Plain numbers at 1, 2, 3 and 5 times a power of ten; other ticks stay unlabeled."""
    if value <= 0:
        return ""
    mantissa = round(value / 10 ** math.floor(math.log10(value)), 6)
    return f"{value:g}" if mantissa in (1, 2, 3, 5) else ""


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _percent(level: float) -> str:
    return f"{100 * level:g} %"
