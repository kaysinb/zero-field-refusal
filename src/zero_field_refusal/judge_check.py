"""The blind sample of the judge check across optimizers (preregistration, Gate 1).

Responses of 1.5B runs at comparable displacement, one stratum per optimizer (SGD, frozen Adam,
AdamW), chosen from K alone: no judge label is read. Only cells with all five seeds 0–4 completed
qualify. A stratum's cell is the qualifying cell whose cell-mean log K is nearest to log K\\*, the
provisional K\\* of the main-condition runs (``kstar.select_k_star``, from KL values only); if it
is more than a factor 2 from K\\*, the two qualifying cells bracketing K\\* share the stratum; with
no bracketing pair, the nearest cell stays, flagged with its distance. Without a K\\* (SGD and
AdamW ranges do not overlap), SGD and AdamW come from the pair of qualifying cells nearest to each
other in log K, frozen Adam from its cell nearest to their geometric mean. When frozen Adam's cell
is more than a factor 2 from its target with no bracketing pair, its stratum is drawn but the
criteria apply to SGD against AdamW only (``criteria_apply``); the same when H2 is descriptive
only, and frozen Adam without a qualifying cell has no stratum. Without a qualifying cell for SGD
or AdamW the check cannot be made. Optionally a fourth stratum, SGD under 100 steps × batch 5,
chosen the same way against K\\* recomputed with those SGD cells.

Prompts: one random permutation of the judged prompt ids (fixed seed); round r takes positions
[100(r − 1), 100 r), so the pre-specified addition (round 2) uses 100 further prompts. They form
5 groups of 20; group g takes its responses from the seed-g runs, the same prompts and seeds in
every stratum. With two cells, the first 10 prompts of a group come from the lower cell, the
other 10 from the upper. Items are shuffled and given opaque ids; the key holds the origin.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .analysis.evidence import TRAINED, arm_of
from .analysis.kstar import CommonRangeUndefined, select_k_star

STRATA = ("sgd", "adam_frozen", "adamw")
SEEDS = (0, 1, 2, 3, 4)
PER_STRATUM = 100
GROUP = PER_STRATUM // len(SEEDS)
NEAR = math.log(2.0)  # a cell within a factor 2 of its target is near


@dataclass(frozen=True)
class Stratum:
    name: str
    arm: str
    experiment: str
    target_k: float
    lrs: tuple[float, ...]  # one cell, or the lower and the upper of a bracketing pair
    cell_k: tuple[float, ...]  # their geometric-mean K
    choice: str  # nearest | bracketing | nearest_far | pair_without_k_star
    criteria_apply: bool


def runs_table(root: Path, experiment: str) -> pd.DataFrame:
    """Completed training runs of ``root/experiment`` with a positive, finite K; no labels read."""
    rows = []
    for run_dir in sorted(p for p in (Path(root) / experiment).glob("*") if p.is_dir()):
        try:
            meta = json.loads((run_dir / "meta.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            cfg = OmegaConf.load(run_dir / "config.yaml")
        except (OSError, ValueError):
            continue
        k = summary.get("K_harmful")
        if meta.get("status") != "completed" or cfg.get("run_type", "train") != "train":
            continue
        if (
            isinstance(k, bool)
            or not isinstance(k, int | float)
            or not (math.isfinite(k) and k > 0)
        ):
            continue
        rows.append({"run_id": run_dir.name, "arm": arm_of(cfg), "lr": float(cfg.optimizer.lr),
                     "seed": int(cfg.seed), "K": float(k)})  # fmt: skip
    return pd.DataFrame(rows, columns=["run_id", "arm", "lr", "seed", "K"])


def qualifying_cells(runs: pd.DataFrame) -> pd.DataFrame:
    """Cells (arm, lr) with every seed of ``SEEDS``, and their mean log K over those seeds."""
    runs = runs[runs["seed"].isin(SEEDS)]
    rows = []
    for (arm, lr), group in runs.groupby(["arm", "lr"], sort=True):
        if set(group["seed"]) == set(SEEDS) and len(group) == len(SEEDS):
            rows.append(
                {"arm": arm, "lr": float(lr), "mean_log_k": float(np.log(group["K"]).mean())}
            )
    return pd.DataFrame(rows, columns=["arm", "lr", "mean_log_k"])


def provisional_k_star(runs: pd.DataFrame) -> tuple[float | None, bool]:
    """K\\* from every main-condition run, pilots included (None if SGD and AdamW do not
    overlap), and whether H2 is descriptive only (frozen Adam misses their overlap)."""
    table = runs.assign(harmful=0, judged=1)  # outcomes are not used
    try:
        selection = select_k_star(table)
    except CommonRangeUndefined:
        return None, True
    return selection.k_star, selection.h2_descriptive_only


def select_strata(
    main: pd.DataFrame,
    sgd_100x5: pd.DataFrame | None = None,
    *,
    main_experiment: str = "",
    experiment_100x5: str = "",
) -> list[Stratum]:
    """The strata of the check, from the main-condition runs (and the 100 × 5 SGD runs)."""
    cells = qualifying_cells(main)
    missing = [arm for arm in ("sgd", "adamw") if arm not in set(cells["arm"])]
    if missing:
        raise ValueError(f"No cell with seeds {list(SEEDS)} completed for {missing}: the check "
                         "cannot be made, and the qualification is undetermined")  # fmt: skip
    k_star, descriptive_only = provisional_k_star(main)
    strata = []
    if k_star is None:
        sgd, adamw = _nearest_pair(cells[cells["arm"] == "sgd"], cells[cells["arm"] == "adamw"])
        for arm, cell in (("sgd", sgd), ("adamw", adamw)):
            k = math.exp(cell["mean_log_k"])
            choice = "pair_without_k_star"
            strata.append(Stratum(arm, arm, main_experiment, math.nan, (cell["lr"],), (k,), choice,
                                  True))  # fmt: skip
        target = math.exp((sgd["mean_log_k"] + adamw["mean_log_k"]) / 2)
    else:
        target = k_star
        for arm in ("sgd", "adamw"):
            strata.append(_choose(arm, arm, main_experiment, cells[cells["arm"] == arm], target))
    frozen_cells = cells[cells["arm"] == "adam_frozen"]
    if not frozen_cells.empty:  # without a qualifying cell frozen Adam has no stratum
        frozen = _choose("adam_frozen", "adam_frozen", main_experiment, frozen_cells, target)
        if descriptive_only:  # H2 descriptive only: reported, no criteria
            frozen = dataclasses.replace(frozen, criteria_apply=False)
        strata.insert(1, frozen)
    if sgd_100x5 is not None:
        if k_star is None:
            raise ValueError("A 100 x 5 stratum needs K* of the main condition")
        combined = pd.concat([sgd_100x5, main[main["arm"] != "sgd"]], ignore_index=True)
        k_100x5, _ = provisional_k_star(combined)
        if k_100x5 is None:
            raise ValueError("SGD under 100 x 5 does not overlap AdamW: no K* for its stratum")
        cells_100x5 = qualifying_cells(sgd_100x5)
        strata.append(_choose("sgd_100x5", "sgd", experiment_100x5,
                              cells_100x5[cells_100x5["arm"] == "sgd"], k_100x5))  # fmt: skip
    return strata


def _choose(name: str, arm: str, experiment: str, cells: pd.DataFrame, target: float) -> Stratum:
    if cells.empty:
        raise ValueError(f"No qualifying cell for stratum {name}")
    t = math.log(target)
    ranked = cells.assign(distance=(cells["mean_log_k"] - t).abs().round(12))
    nearest = ranked.sort_values(["distance", "lr"], kind="stable").iloc[0]  # a tie: smaller lr
    if abs(nearest["mean_log_k"] - t) <= NEAR:
        return Stratum(name, arm, experiment, target, (nearest["lr"],),
                       (math.exp(nearest["mean_log_k"]),), "nearest", True)  # fmt: skip
    below, above = cells[cells["mean_log_k"] <= t], cells[cells["mean_log_k"] > t]
    if not below.empty and not above.empty:
        low, high = below.loc[below["mean_log_k"].idxmax()], above.loc[above["mean_log_k"].idxmin()]
        return Stratum(name, arm, experiment, target, (low["lr"], high["lr"]),
                       (math.exp(low["mean_log_k"]), math.exp(high["mean_log_k"])), "bracketing",
                       True)  # fmt: skip
    far = (math.exp(nearest["mean_log_k"]),)
    return Stratum(name, arm, experiment, target, (nearest["lr"],), far, "nearest_far",
                   arm != "adam_frozen")  # fmt: skip


def _nearest_pair(sgd: pd.DataFrame, adamw: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    best = None
    for _, a in sgd.iterrows():
        for _, b in adamw.iterrows():
            gap = abs(a["mean_log_k"] - b["mean_log_k"])
            if best is None or gap < best[0] - 1e-12:
                best = (gap, a, b)
    assert best is not None
    return best[1], best[2]


def round_prompts(prompt_ids: Sequence[str], round_: int, seed: int) -> list[str]:
    """The 100 prompts of a round: positions [100(r − 1), 100 r) of one seeded permutation."""
    if round_ < 1:
        raise ValueError("Rounds start at 1")
    order = sorted(prompt_ids)
    random.Random(seed).shuffle(order)
    chosen = order[PER_STRATUM * (round_ - 1) : PER_STRATUM * round_]
    if len(chosen) < PER_STRATUM:
        raise ValueError(f"Round {round_} needs {PER_STRATUM} prompts beyond the earlier rounds")
    return chosen


def draw_sample(
    strata: Sequence[Stratum],
    runs: Mapping[str, pd.DataFrame],
    responses: Callable[[str, str], Mapping[str, str]],
    prompts: Mapping[str, str],
    *,
    round_: int = 1,
    seed: int = 0,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """Blinded items and their key.

    ``runs`` maps an experiment to its runs table; ``responses(experiment, run_id)`` gives a run's
    trained-model responses by prompt id; ``prompts`` the judged prompt texts by id.
    """
    chosen = round_prompts(list(prompts), round_, seed)
    items = []
    for stratum in strata:
        table = runs[stratum.experiment]
        for g, run_seed in enumerate(SEEDS):
            group = chosen[GROUP * g : GROUP * (g + 1)]
            parts = ([group] if len(stratum.lrs) == 1 else
                     [group[: GROUP // 2], group[GROUP // 2 :]])  # fmt: skip
            for lr, part in zip(stratum.lrs, parts, strict=True):
                run = table[(table["arm"] == stratum.arm) & (table["lr"] == lr)
                            & (table["seed"] == run_seed)]  # fmt: skip
                assert len(run) == 1, (stratum.name, lr, run_seed)
                run_id = run["run_id"].iloc[0]
                texts = responses(stratum.experiment, run_id)
                for prompt_id in part:
                    items.append({"prompt": prompts[prompt_id], "response": texts[prompt_id],
                                  "stratum": stratum.name, "arm": stratum.arm, "lr": lr,
                                  "seed": run_seed, "run_id": run_id, "prompt_id": prompt_id,
                                  "round": round_})  # fmt: skip
    rng = random.Random(f"judge-check-{seed}-{round_}")
    rng.shuffle(items)
    blinded, key = [], {}
    for item in items:
        item_id = f"c{rng.getrandbits(48):012x}"
        while item_id in key:
            item_id = f"c{rng.getrandbits(48):012x}"
        blinded.append({"item_id": item_id, "prompt": item["prompt"], "response": item["response"]})
        key[item_id] = {k: v for k, v in item.items() if k not in ("prompt", "response")}
    return blinded, key


def trained_responses(run_dir: Path) -> dict[str, str]:
    """The trained model's responses of a run's harmful prompt set, by prompt id."""
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    path = Path(run_dir) / "responses" / f"{cfg.eval.harmful.prompt_set}.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record["condition"] == TRAINED:
                out[record["item_id"]] = record["text"]
    return out


def selection_record(strata: Sequence[Stratum], round_: int, seed: int) -> dict[str, Any]:
    """What was drawn and why, for selection.json (an undefined target K is null)."""

    def plain(stratum: Stratum) -> dict[str, Any]:
        data = asdict(stratum)
        data["target_k"] = None if math.isnan(stratum.target_k) else stratum.target_k
        return data

    return {"round": round_, "seed": seed, "per_stratum": PER_STRATUM, "seeds": list(SEEDS),
            "strata": [plain(s) for s in strata]}  # fmt: skip
