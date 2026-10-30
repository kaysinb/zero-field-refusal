"""Study 2's whole seed vectors for study 3's simulation (analysis/study3_seed_vectors.py), on
synthetic study-2 run directories in ``tmp_path``. Every run here also holds responses,
observations and judge outputs with a sentinel text, a step-25 evaluation row with a keyword
refusal rate and K_harmful, and a meta.json and summary.json with the sentinel: none of it may be
opened or reach the record."""

import builtins
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from zero_field_refusal.analysis import study3_seed_vectors as sv
from zero_field_refusal.analysis.study3_search import SLOPES

SENTINEL = "STUDY2-HARM-SENTINEL-v8Qk-3pLw"
THETA0 = {"benchmark_accuracy": 0.60, "factual_accuracy": 0.80}
OPTIMIZERS = {"A": "adamw", "F": "adam_frozen", "P": "adamw_prewarmed"}
# per arm, the rates of a seed and ln K at them (a seed effect added): two rates for F and P,
# three for A
RATES = {"A": (1.45e-5, 1.75e-5, 2.11e-5), "F": (1.30e-5, 1.83e-5), "P": (1.30e-5, 1.85e-5)}


def config(arm: str, lr: float, seed: int, **change) -> str:
    """A resolved study-2 config.yaml (the keys the builder selects, and others)."""
    cfg = {
        "experiment": "study2_short",
        "run_type": "train",
        "control": None,  # the zero-field protocol's null control
        "seed": seed,
        "optimizer": {"name": OPTIMIZERS[arm], "lr": lr, "beta1": 0.9},
        "train": {"steps": 25, "batch_size": 20},
        "schedule": {"policy": "fixed", "evaluate_at": [25]},
        "study2_short": {"role": "discovery"},
    }
    for key, value in change.items():
        if "__" not in key:
            cfg[key] = value
        else:
            section, name = key.split("__")
            cfg[section][name] = value
    return json.dumps(cfg)  # JSON is YAML


def k_of(arm: str, lr: float, seed: int) -> float:
    """A K that grows with the rate, with a seed effect."""
    return math.exp(1.8 + SLOPES[arm] * math.log(lr / 1.6e-5) + 0.02 * (seed - 2100))


def write_run(
    root: Path,
    run_id: str,
    arm: str,
    lr: float,
    seed: int,
    *,
    k: float | None = None,
    stop: str = "steps",
    steps: int = 25,
    row: dict | None = None,
    **change,
) -> Path:
    """One study-2 run directory with its record files and its sealed folders."""
    run = root / run_id
    for folder, name in (
        ("responses", "harmful_evaluation.jsonl"),
        ("observations", "measurements.jsonl"),
        ("judge", "wildguard.jsonl"),
    ):
        (run / folder).mkdir(parents=True)
        (run / folder / name).write_text(json.dumps({"text": SENTINEL}) + "\n")
    (run / "config.yaml").write_text(config(arm, lr, seed, **change))
    (run / "meta.json").write_text(json.dumps({"note": SENTINEL}))
    (run / "summary.json").write_text(json.dumps({"note": SENTINEL}))
    detail = {"gradient_spike": {"grad_norm": 9.0, "reference_median": 0.5}}.get(stop)
    (run / "terminal.json").write_text(
        json.dumps({"stop": {"reason": stop, "detail": detail}, "steps": steps, "seconds": 10.0})
    )
    k = k_of(arm, lr, seed) if k is None else k
    rows = [
        {
            "step": 0,
            "K_benign": 0.0,
            "benchmark_accuracy": 0.60,
            "factual_accuracy": 0.80,
            "keyword_refusal_rate": 0.9,
            "K_harmful": 0.0,
        }
    ]
    if steps == 25 and stop == "steps":
        loss = 0.004 * math.log(k)  # MMLU loss grows with ln K
        rows.append(
            {
                "step": 25,
                "K_benign": k,
                "benchmark_accuracy": 0.60 - loss,
                "factual_accuracy": 0.80 - loss / 2,
                "keyword_refusal_rate": 0.42,
                "K_harmful": 3.3,
                "harm_note": SENTINEL,
                **(row or {}),
            }
        )
    (run / "evaluations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def study2_world(
    tmp_path: Path, *, p_one_rate: int | None = None, seeds=(2100, 2101), extra=None
) -> tuple[list[Path], Path]:
    """Two study-2 roots and θ₀'s evaluation: whole seed vectors for ``seeds``, except that P
    of ``p_one_rate`` has one rate; runs of other kinds beside them (a control, a K-stopped run,
    a B = 5 run, a θ₀ run, a folder with no config). ``extra``: more runs, (root index, run id,
    arm, rate, seed, keywords)."""
    roots = [tmp_path / "runs" / name for name in sv.STUDY2_ROOTS]
    for root in roots:
        root.mkdir(parents=True)
    stamp = iter(range(100, 10_000))
    for seed in seeds:
        for arm, rates in RATES.items():
            for i, lr in enumerate(rates):
                if arm == "P" and seed == p_one_rate and i > 0:
                    continue
                where = roots[i % 2]  # the two roots pool
                write_run(where, f"20261001T{next(stamp):06d}Z-{arm}-{lr:g}-s{seed}", arm, lr, seed)
    a = roots[0]
    write_run(a, "20261001T000001Z-control", "F", 1.3e-5, 2100, control="soft_targets")
    write_run(a, "20261001T000002Z-long", "A", 1.45e-5, 2100, schedule__policy="k_stopped")
    write_run(a, "20261001T000003Z-b5", "A", 1.45e-5, 2100, train__batch_size=5)
    write_run(a, "20261001T000004Z-theta0", "A", 1.45e-5, 2100, run_type="theta0")
    write_run(a, "20261001T000005Z-sgd", "A", 1.45e-5, 2100, optimizer__name="sgd")
    (a / "20261001T000006Z-empty").mkdir()
    for index, run_id, arm, lr, seed, kw in extra or ():
        write_run(roots[index], run_id, arm, lr, seed, **kw)
    theta0 = tmp_path / "runs" / "study2_theta0" / "20261014T192927Z-theta0-s2000-abb703a1-a1"
    theta0.mkdir(parents=True)
    (theta0 / "summary.json").write_text(json.dumps({**THETA0, "K_benign": 0.0}))
    return roots, theta0


def vector(seed: int, **arms) -> sv.Vector:
    """A vector with the given (rate, K) points per arm, utility proportional to ln K."""
    return sv.Vector(
        seed=seed,
        arms={
            arm: tuple(
                sv.Point(
                    rate=r,
                    K=k,
                    mmlu_loss_pp=0.4 * math.log(k),
                    factual_loss_pp=0.2 * math.log(k),
                    run_id=f"r{seed}{arm}{i}",
                )
                for i, (r, k) in enumerate(points)
            )
            for arm, points in arms.items()
        },
    )


def opened_files(monkeypatch) -> list[Path]:
    """Every file opened from now on, through ``Path.open``, ``open`` or ``io.open``, once per
    opening (``Path.open`` itself calls ``io.open``)."""
    opened: list[Path] = []
    real_path_open, real_open = Path.open, builtins.open
    inside = [0]

    def path_open(self, *a, **k):
        opened.append(self)
        inside[0] += 1
        try:
            return real_path_open(self, *a, **k)
        finally:
            inside[0] -= 1

    def open_(file, *a, **k):
        if not inside[0] and not isinstance(file, int):
            opened.append(Path(os.fsdecode(file)))
        return real_open(file, *a, **k)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(builtins, "open", open_)
    monkeypatch.setattr(io, "open", open_)
    return opened


# ------------------------------------------------------------------ what is opened and kept


def test_the_opened_files_helper_sees_every_way_of_opening(tmp_path, monkeypatch) -> None:
    files = [tmp_path / f"f{i}.txt" for i in range(4)]
    for f in files:
        f.write_text("x")
    opened = opened_files(monkeypatch)
    files[0].read_text()
    with open(files[1]) as fh:
        fh.read()
    with io.open(files[2]) as fh:  # noqa: UP020 - io.open is what is checked
        fh.read()
    with files[3].open() as fh:
        fh.read()
    assert opened == files


def test_the_builder_opens_only_the_record_files_and_keeps_k_and_utility(tmp_path, monkeypatch):
    roots, theta0 = study2_world(tmp_path)
    opened = opened_files(monkeypatch)
    points, excluded = sv.study2_points(roots, theta0)
    vectors, skipped = sv.seed_vectors(points)
    text = json.dumps(sv.inputs_record(vectors, {**excluded, **skipped}, theta0={}, sources=["x"]))
    assert SENTINEL not in text and "keyword_refusal" not in text and "K_harmful" not in text
    assert {p.name for p in opened} <= {*sv.RUN_FILES, "summary.json"}
    assert not any({"responses", "observations", "judge"} & set(p.parts) for p in opened)
    # θ₀'s summary.json is the only summary.json opened
    assert [p for p in opened if p.name == "summary.json"] == [theta0 / "summary.json"]
    assert [v.seed for v in vectors] == [2100, 2101]
    assert len(points) == 2 * 7
    # the other kinds of runs are not points, and not counted as left out
    assert excluded == {
        "instability": 0,
        "incomplete": 0,
        "nonpositive_k": 0,
        "duplicates": {"count": 0, "largest_relative_k_difference": None},
    }


def test_a_point_holds_k_and_the_losses_against_theta0(tmp_path) -> None:
    roots, theta0 = study2_world(tmp_path, seeds=(2100,))
    points, _ = sv.study2_points(roots, theta0)
    seed, arm, point = next(p for p in points if p[1] == "F" and p[2].rate == 1.83e-5)
    k = k_of("F", 1.83e-5, 2100)
    assert seed == 2100 and k == point.K
    assert point.mmlu_loss_pp == pytest.approx(100 * 0.004 * math.log(k))
    assert point.factual_loss_pp == pytest.approx(100 * 0.002 * math.log(k))
    assert point.run_id.endswith("-F-1.83e-05-s2100")


@pytest.mark.parametrize("name", sv.RUN_FILES)
@pytest.mark.parametrize(
    "target",
    [
        "responses/harmful_evaluation.jsonl",
        "observations/measurements.jsonl",
        "judge/wildguard.jsonl",
    ],
)
def test_a_file_linked_into_a_sealed_folder_is_refused_unopened(
    tmp_path, monkeypatch, name, target
) -> None:
    roots, theta0 = study2_world(tmp_path, seeds=(2100,))
    run = sorted(roots[0].glob("*-A-1.45e-05-s2100"))[0]
    (run / name).unlink()
    (run / name).symlink_to(Path(target))
    opened = opened_files(monkeypatch)
    with pytest.raises(ValueError, match="never opened") as refusal:
        sv.study2_points(roots, theta0)
    assert run.name in str(refusal.value) and SENTINEL not in str(refusal.value)
    assert not any({"responses", "observations", "judge"} & set(p.parts) for p in opened)
    assert not any(p.parent == run for p in opened)  # nothing of the run is opened


def test_a_run_folder_linked_into_a_sealed_folder_is_refused_unopened(tmp_path, monkeypatch):
    roots, theta0 = study2_world(tmp_path, seeds=(2100,))
    other = sorted(roots[0].glob("*-A-1.45e-05-s2100"))[0]
    (roots[0] / "20261001T999999Z-linked").symlink_to(other / "judge")
    opened = opened_files(monkeypatch)
    with pytest.raises(ValueError, match="never opened"):
        sv.study2_points(roots, theta0)
    assert not any({"responses", "observations", "judge"} & set(p.parts) for p in opened)


# ------------------------------------------------------------------ which points and seeds


def test_a_seed_is_whole_only_with_two_rates_in_every_arm(tmp_path):
    # seed 2100 complete; seed 2101 has P at one rate only
    vectors, skipped = sv.seed_vectors(
        sv.study2_points(*study2_world(tmp_path, p_one_rate=2101))[0]
    )
    assert [v.seed for v in vectors] == [2100] and skipped == {"incomplete_seed": 1}
    assert {arm: [p.rate for p in points] for arm, points in vectors[0].arms.items()} == {
        arm: list(rates) for arm, rates in RATES.items()
    }


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"stop": "gradient_spike", "steps": 7}, "instability"),
        ({"stop": "nonfinite_loss", "steps": 3}, "instability"),
        ({"stop": "steps", "steps": 24}, "incomplete"),
        ({"row": {"K_benign": 0.0}}, "nonpositive_k"),
        ({"row": {"K_benign": -1.0}}, "nonpositive_k"),
        ({"row": {"benchmark_accuracy": None}}, "incomplete"),
        ({"row": {"K_benign": "x"}}, "incomplete"),
    ],
)
def test_a_stopped_run_is_left_out_and_counted(tmp_path, change, reason):
    run = ("20261001T999990Z-A-1.6e-05-s2100", "A", 1.6e-5, 2100)
    roots, theta0 = study2_world(tmp_path, seeds=(2100,), extra=[(0, *run, change)])
    points, excluded = sv.study2_points(roots, theta0)
    assert excluded[reason] == 1
    assert sum(v for k, v in excluded.items() if k != "duplicates") == 1
    assert not any(p.run_id == run[0] for _, _, p in points)


@pytest.mark.parametrize(
    "path,overrides,arm",
    [
        ("experiments/study2_short", ["optimizer=adamw", "optimizer.lr=1.45e-5"], "A"),
        ("experiments/study2_short", ["optimizer=adam_frozen", "optimizer.lr=1.3e-5"], "F"),
        ("experiments/study2_short", ["optimizer=adamw_prewarmed", "optimizer.lr=1.85e-5"], "P"),
        ("experiments/study2_short_k_only", ["optimizer=adamw", "optimizer.lr=1.45e-5"], "A"),
        ("experiments/study2_stage_a", ["schedule=short", "optimizer=adam_frozen",
                                        "optimizer.lr=2e-5"], "F"),
        ("experiments/study2_stage_a", ["optimizer=adamw", "optimizer.lr=2e-5"], None),
        ("experiments/study2_controls", ["schedule=short", "optimizer=adam_frozen",
                                         "optimizer.lr=1.3e-5"], None),
        ("experiments/study2_short_fa_controls", ["optimizer.lr=1.3e-5"], None),
    ],
)  # fmt: skip
def test_a_config_composed_as_study2s_runs_is_selected_by_its_schedule_and_control(
    tmp_path, path, overrides, arm
) -> None:
    """A run's config.yaml as the tracker saves it (resolved), composed from this checkout's
    configs: the fixed 25-update runs at B = 20 of the three arms are points (the zero-field
    protocol's null control included); a K-stopped run and a soft-target control are not."""
    from omegaconf import OmegaConf

    from zero_field_refusal.config import load_config

    roots, theta0 = study2_world(tmp_path, seeds=())
    cfg = load_config(path, [*overrides, "seed=2100"])
    run = write_run(roots[0], "20261001T999999Z-composed", "A", 1.0e-5, 2100)
    (run / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    points, excluded = sv.study2_points(roots, theta0)
    assert [(seed, a) for seed, a, _ in points] == ([(2100, arm)] if arm else [])
    assert sum(v for k, v in excluded.items() if k != "duplicates") == 0


@pytest.mark.parametrize(
    "damage",
    ["no terminal record", "a config that is not a mapping", "a config that does not parse"],
)
def test_a_run_with_a_damaged_record_is_incomplete(tmp_path, damage) -> None:
    roots, theta0 = study2_world(tmp_path, seeds=(2100,))
    run = sorted(roots[0].glob("*-A-1.45e-05-s2100"))[0]
    if damage == "no terminal record":
        (run / "terminal.json").unlink()
    else:
        (run / "config.yaml").write_text("- a list\n" if "mapping" in damage else "a: [b\n")
    points, excluded = sv.study2_points(roots, theta0)
    assert excluded["incomplete"] == 1 and not any(p.run_id == run.name for _, _, p in points)


def test_duplicates_keep_the_earliest_run(tmp_path):
    k = k_of("A", 1.45e-5, 2100)
    extra = [
        (1, "20261001T000099Z-A-dup-s2100", "A", 1.45e-5, 2100, {"k": 1.1 * k}),
        (0, "20261001T999999Z-A-dup-s2100", "A", 1.45e-5, 2100, {"k": 0.95 * k}),
    ]
    roots, theta0 = study2_world(tmp_path, seeds=(2100,), extra=extra)
    points, excluded = sv.study2_points(roots, theta0)
    at = [p for seed, arm, p in points if (seed, arm, p.rate) == (2100, "A", 1.45e-5)]
    assert [p.run_id for p in at] == ["20261001T000099Z-A-dup-s2100"] and 1.1 * k == at[0].K
    assert excluded["duplicates"]["count"] == 2
    # relative to the kept run's K: |K − K_kept| / K_kept, the largest
    assert excluded["duplicates"]["largest_relative_k_difference"] == pytest.approx(
        (1.1 - 0.95) / 1.1
    )


# ------------------------------------------------------------------ the rate and utility maps


def test_ln_k_interpolates_in_ln_rate_and_extends_with_the_registered_slope():
    v = vector(2100, A=[(1.5e-5, 5.0), (2.0e-5, 7.0)])
    mid = math.log(5.0) + (math.log(7.0 / 5.0)) * math.log(1.75 / 1.5) / math.log(2.0 / 1.5)
    assert v.ln_k("A", 1.75e-5) == pytest.approx(mid)
    assert v.ln_k("A", 2.2e-5) == pytest.approx(math.log(7.0) + 1.25 * math.log(2.2 / 2.0))
    assert v.ln_k("A", 1.3e-5) == pytest.approx(math.log(5.0) - 1.25 * math.log(1.5 / 1.3))
    assert v.ln_k("A", 2.0e-5) == pytest.approx(math.log(7.0))


def test_the_losses_are_piecewise_linear_in_ln_k_and_extend_with_the_pooled_slope():
    v = vector(2100, F=[(1.3e-5, 5.0), (1.5e-5, 6.0), (1.8e-5, 8.0)])
    slopes = {"F": (2.0, 1.5)}
    # inside: the points' own line (0.4 ln K, 0.2 ln K)
    assert v.losses("F", math.log(7.0), slopes) == pytest.approx(
        (0.4 * math.log(6.0) + 0.4 * (math.log(7.0) - math.log(6.0)), 0.2 * math.log(7.0))
    )
    # beyond the points: the pooled slopes
    up = math.log(8.0) + 0.3
    assert v.losses("F", up, slopes) == pytest.approx(
        (0.4 * math.log(8.0) + 2.0 * 0.3, 0.2 * math.log(8.0) + 1.5 * 0.3)
    )
    down = math.log(5.0) - 0.2
    assert v.losses("F", down, slopes) == pytest.approx(
        (0.4 * math.log(5.0) - 2.0 * 0.2, 0.2 * math.log(5.0) - 1.5 * 0.2)
    )


def test_the_losses_at_an_array_are_the_losses_at_each_value():
    v = vector(2100, F=[(1.3e-5, 5.0), (1.5e-5, 6.0), (1.8e-5, 8.0)])
    slopes = {"F": (2.0, 1.5)}
    at = np.linspace(math.log(4.0), math.log(9.5), 41)
    mmlu, factual = v.losses_at("F", at, slopes)
    assert [(m, f) for m, f in zip(mmlu, factual, strict=True)] == pytest.approx(
        [v.losses("F", x, slopes) for x in at]
    )
    # where the MMLU loss crosses a level: on a segment and on an extension
    for level in (0.4 * math.log(5.5), 0.4 * math.log(8.0) + 2.0 * 0.5, 0.4 * math.log(5.0) - 0.2):
        crossings = v.mmlu_crossings("F", level, slopes)
        assert len(crossings) == 1
        assert v.losses("F", crossings[0], slopes)[0] == pytest.approx(level)


def test_the_utility_slopes_are_pooled_within_seeds():
    """Least squares with one intercept per seed: a seed's own level does not enter the slope."""
    pts = {
        2100: [(1.3e-5, 5.0, 1.0, 0.5), (1.8e-5, 8.0, 2.0, 1.1)],
        2101: [(1.3e-5, 4.0, 3.0, 2.0), (1.5e-5, 5.0, 3.5, 2.1), (1.8e-5, 6.5, 4.5, 2.6)],
    }

    def make(seed):
        arm = tuple(sv.Point(r, k, m, f, f"r{seed}{i}") for i, (r, k, m, f) in enumerate(pts[seed]))
        return sv.Vector(seed=seed, arms={"A": arm, "F": arm, "P": arm})

    slopes = sv.utility_slopes([make(2100), make(2101)])
    x = {s: [math.log(k) for _, k, _, _ in p] for s, p in pts.items()}
    for column, index in (("mmlu", 2), ("factual", 3)):
        y = {s: [row[index] for row in p] for s, p in pts.items()}
        num = sum(
            (xi - sum(x[s]) / len(x[s])) * (yi - sum(y[s]) / len(y[s]))
            for s in pts
            for xi, yi in zip(x[s], y[s], strict=True)
        )
        den = sum((xi - sum(x[s]) / len(x[s])) ** 2 for s in pts for xi in x[s])
        want = num / den
        got = slopes["A"][0 if column == "mmlu" else 1]
        assert got == pytest.approx(want)
    assert set(slopes) == {"A", "F", "P"}


def test_a_vector_needs_two_increasing_rates_per_arm():
    with pytest.raises(ValueError):
        vector(2100, A=[(1.5e-5, 5.0)])
    with pytest.raises(ValueError):
        vector(2100, A=[(2.0e-5, 7.0), (1.5e-5, 5.0)])
    with pytest.raises(ValueError):
        vector(2100, A=[(1.5e-5, 5.0), (1.5e-5, 5.5)])


# ------------------------------------------------------------------ the record


def test_read_inputs_refuses_an_unknown_field_and_round_trips(tmp_path):
    roots, theta0 = study2_world(tmp_path)
    points, excluded = sv.study2_points(roots, theta0)
    vectors, skipped = sv.seed_vectors(points)
    record = sv.inputs_record(
        vectors,
        {**excluded, **skipped},
        theta0={"path": str(theta0), **THETA0},
        sources=[str(r) for r in roots],
    )
    assert record["kind"] == "study3-simulation-inputs"
    assert record["excluded"] == {
        "instability": 0,
        "incomplete": 0,
        "nonpositive_k": 0,
        "incomplete_seed": 0,
    }
    assert record["duplicates"] == {"count": 0, "largest_relative_k_difference": None}
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(record, allow_nan=False, indent=2, sort_keys=True))
    read, slopes, digest = sv.read_inputs(path)
    assert read == vectors and slopes == sv.utility_slopes(vectors)
    import hashlib

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    for broken in (
        {**record, "harm": 0.1},
        {**record, "kind": "other"},
        {**record, "vectors": [{**record["vectors"][0], "K_harmful": 1.0}]},
        {
            **record,
            "vectors": [
                {
                    "seed": 2100,
                    "arms": {
                        **record["vectors"][0]["arms"],
                        "A": [
                            {**record["vectors"][0]["arms"]["A"][0], "keyword_refusal_rate": 0.4},
                            *record["vectors"][0]["arms"]["A"][1:],
                        ],
                    },
                }
            ],
        },
        {**record, "vectors": [{"seed": 2100, "arms": {"A": record["vectors"][0]["arms"]["A"]}}]},
        {**record, "excluded": {"instability": -1}},
        {**record, "theta0": {"path": "x", "benchmark_accuracy": 1.5, "factual_accuracy": 0.8}},
        {**record, "vectors": []},
        {k: v for k, v in record.items() if k != "duplicates"},
    ):
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            sv.read_inputs(path)
    path.write_text("{not json")
    with pytest.raises(ValueError):
        sv.read_inputs(path)


def test_the_module_imports_no_torch_and_no_gpu_or_study2_reader():
    code = (
        "import sys; import zero_field_refusal.analysis.study3_seed_vectors; "
        "bad = ('torch', 'transformers', 'zero_field_refusal.analysis.stage_a', "
        "'zero_field_refusal.analysis.study2_short', 'zero_field_refusal.analysis.study2_labels'); "
        "sys.exit(any(m == b or m.startswith(b + '.') for m in sys.modules for b in bad))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
