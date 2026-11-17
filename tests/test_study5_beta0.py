"""Study 5's run without momentum (docs/study5-beta0-registration.md): the ``beta0`` role, its
configs against experiments/study5's, and the K-only pilot config against the main one."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from test_golden_single_process import OVERRIDES
from test_schedule import scheduled
from test_study5_options import F_RUN

from zero_field_refusal.config import config_hash
from zero_field_refusal.protocol import ProtocolError, load_run_config

B0_RUN = (*F_RUN, "optimizer.beta1=0.0")  # the golden F run without momentum
COMMON = ["optimizer.lr=1.83e-05", "seed=50000"]


def _resolved(name: str, overrides: list[str]) -> dict:
    return OmegaConf.to_container(load_run_config(name, overrides), resolve=True)


def test_the_config_is_study5_but_for_beta1_and_its_role():
    ours = load_run_config("experiments/study5_beta0", COMMON)
    assert config_hash(ours) and ours.optimizer.beta1 == 0.0
    b0 = _resolved("experiments/study5_beta0", COMMON)
    f = _resolved("experiments/study5", [*COMMON, "study5.role=fresh"])
    assert b0.pop("study5") == {"role": "beta0"} and f.pop("study5") == {"role": "fresh"}
    assert b0["optimizer"].pop("beta1") == 0.0 and f["optimizer"].pop("beta1") == 0.9
    assert b0 == f


@pytest.mark.parametrize("extra", [
    [], ["+train.order_seed=50000"],
    ["+train.order_edit={kind: shuffle, first: 1, last: 5, draw: 0}"],
])  # fmt: skip
def test_the_k_only_pilot_matches_the_main_run_in_training(extra):
    main = _resolved("experiments/study5_beta0", [*COMMON, *extra])
    pilot = _resolved("experiments/study5_beta0_k_only", [*COMMON, *extra])
    for part in ("train", "optimizer", "execution", "data", "seed", "study5"):
        assert pilot[part] == main[part], part
    assert pilot["eval"]["k_only"] is True


@pytest.mark.parametrize("overrides", [
    ("+study5.role=beta0",),
    ("+study5.role=beta0", "+train.order_seed=3"),
    ("+study5.role=beta0", "+train.order_edit={kind: shuffle, first: 1, last: 3, draw: 0}"),
])  # fmt: skip
def test_the_beta0_role_accepts_f_without_momentum(tmp_path, overrides) -> None:
    scheduled(tmp_path, *B0_RUN, *overrides, policy="short")


@pytest.mark.parametrize(("base", "overrides", "message"), [
    (F_RUN, ("+study5.role=beta0",), "beta1"),
    (B0_RUN, ("+study5.role=fresh",), "beta1"),
    (B0_RUN, ("+study5.role=control", "+train.order_seed=3"), "beta1"),
    (B0_RUN, ("+study5.role=beta0", "+train.order_seed=3",
              "+train.order_edit={kind: swap_batches, step: 1}"), "at most one"),
    (B0_RUN, ("+study5.role=beta0", "+optimizer.preconditioner.source_seed=3"), "at most one"),
    (B0_RUN, ("+study5.role=beta0", "+train.rate_schedule={early_steps: 1, early_lr: 0.0005}"),
     "dose role only"),
    (("optimizer=adamw", *OVERRIDES), ("+study5.role=beta0",), "adam_frozen"),
])  # fmt: skip
def test_bad_beta0_runs_are_refused(tmp_path, base, overrides, message) -> None:
    with pytest.raises(ProtocolError, match=message):
        scheduled(tmp_path, *base, *overrides, policy="short")


# ------------------------------------------------------------------ the design and the rules

import json  # noqa: E402
import math  # noqa: E402

from zero_field_refusal import beta0  # noqa: E402
from zero_field_refusal.analysis import study5_beta0 as sb  # noqa: E402
from zero_field_refusal.paths import REPO_ROOT  # noqa: E402
from zero_field_refusal.study5_options import edited_order  # noqa: E402

TARGETS = beta0.targets_from(json.loads((REPO_ROOT / beta0.TARGETS_FILE).read_text()))
REFERENCE = json.loads((REPO_ROOT / beta0.REFERENCE_FILE).read_text())


def test_the_shuffle_permutations_are_study5_options():
    cfg = OmegaConf.create({"seed": 0, "train": {"steps": 25, "batch_size": 20}})
    for seed in beta0.SEEDS:
        cfg.seed = seed
        for _, first, last in beta0.WINDOWS:
            for draw in beta0.DRAWS:
                order = edited_order(cfg, {"kind": "shuffle", "first": first, "last": last,
                                           "draw": draw})  # fmt: skip
                batches = [order[i * 20] // 20 for i in range(first - 1, last)]
                assert [b - (first - 1) for b in batches] == beta0.shuffle_permutation(
                    seed, first, last, draw)  # fmt: skip


def test_the_design_is_m6s_shuffles_without_the_identity():
    assert beta0.is_identity(50003, 21, 25, 0)
    assert sum(len(beta0.orders_of(s)) for s in beta0.SEEDS) == 35
    assert beta0.orders_of(50003) == ["stream", "reorder", "early-0", "early-1", "late-1"]
    with pytest.raises(ValueError, match="no order"):
        beta0.cell(50003, "late-0", 1.5e-05)


@pytest.mark.parametrize("order", ["stream", "reorder", "early-1", "late-0"])
def test_a_cells_config_passes_the_protocol(order):
    cell = beta0.cell(50000, order, 1.55e-05)
    cfg = load_run_config(cell["config"], [*cell["overrides"], "seed=50000"])
    assert config_hash(cfg) and cfg.study5.role == "beta0" and cfg.optimizer.lr == 1.55e-05
    pilot = beta0.pilot_cell(50001, "1.41")
    cfg = load_run_config(pilot["config"], [*pilot["overrides"], "seed=50001"])
    assert config_hash(cfg) and cfg.eval.k_only is True and cfg.optimizer.lr == 1.83e-05


def test_the_targets_and_the_reference_are_the_committed_records():
    assert sorted(TARGETS) == list(beta0.SEEDS)
    assert math.isclose(TARGETS[50000], 4.9139, abs_tol=1e-4)
    summary = beta0.summary_of(REFERENCE["seeds"])
    assert all(math.isclose(summary[k], REFERENCE[k]) for k in summary)
    assert round(REFERENCE["F"], 1) == 12.0 and round(REFERENCE["E"], 1) == 11.4
    assert round(REFERENCE["L"], 1) == 1.2 and round(REFERENCE["h_bar"], 1) == 12.8


def _pilot_rows(k_of):
    return [{"seed": s, "level": level, "outcome": "done", "complete": True, "K": k_of(s, lr)}
            for s in beta0.PILOT_SEEDS for level, lr in beta0.GRID]  # fmt: skip


def test_the_rate_rule_interpolates_in_log_and_takes_the_geometric_mean():
    choice = beta0.rate_choice(_pilot_rows(lambda s, lr: 3.0 * (lr / 1.3e-05) ** 2), TARGETS)
    expected = [1.3e-05 * math.sqrt(TARGETS[s] / 3.0) for s in beta0.PILOT_SEEDS]
    assert choice["stop"] is None
    assert choice["eta0"] == float(f"{math.sqrt(expected[0] * expected[1]):.3g}")
    assert choice["per_seed"]["50000"]["bracket"]["levels"] == ["1.0", "1.41"]


def test_the_rate_rule_stops_without_a_bracket_and_refuses_a_partial_table():
    choice = beta0.rate_choice(_pilot_rows(lambda s, lr: 1.0), TARGETS)
    assert choice["eta0"] is None and "not bracketed" in choice["stop"]
    rows = _pilot_rows(lambda s, lr: 3.0 * (lr / 1.3e-05) ** 2)
    rows[2] |= {"complete": False}  # 50000 at 1.41 x: no value, so 1.0 and 1.41 do not bracket
    choice = beta0.rate_choice(rows, TARGETS)
    assert choice["per_seed"]["50000"]["bracket"] is None and choice["eta0"] is None
    with pytest.raises(ValueError, match="expected"):
        beta0.rate_choice(rows[:-1], TARGETS)


def test_the_drift_gate():
    rows = [
        {"seed": s, "outcome": "done", "complete": True, "K": TARGETS[s] * 1.1} for s in beta0.SEEDS
    ]
    assert beta0.drift_gate(rows, TARGETS)["passed"] and math.isclose(
        beta0.drift_gate(rows, TARGETS)["ratio"], 1.1)  # fmt: skip
    rows = [
        {"seed": s, "outcome": "done", "complete": True, "K": TARGETS[s] * 1.3} for s in beta0.SEEDS
    ]
    assert not beta0.drift_gate(rows, TARGETS)["passed"]
    rows[0] |= {"complete": False}
    assert beta0.drift_gate(rows, TARGETS)["missing"] == [50000]


def _labels(h_pp: float) -> dict:
    harmful = round(h_pp / 100 * 521)
    return {"n": 521, "harmful": harmful, "refusal": 521 - harmful, "safe_non_refusal": 0,
            "missing": 0}  # fmt: skip


def _runs(h_of) -> dict:
    """Synthetic complete runs: h of (seed, order) in pp."""
    out = {}
    for seed, order in sb.design():
        out[(seed, order)] = {"run_id": f"r{seed}{order}", "seed": seed, "order": order,
                              "records": None, "stop": "steps", "steps": 25,
                              "row": {"K_benign": 5.0, "K_harmful": 5.0,
                                      "benchmark_accuracy": 0.55},
                              "labels": _labels(h_of(seed, order))}  # fmt: skip
    return out


def _scenario(f: float, e: float, late: float):
    """Stream h 12 pp; |Δh| alternating in sign, sized by class, varied a little by seed."""

    def h_of(seed, order):
        if order == "stream":
            return 12.0
        size = {"F": f, "E": e, "L": late}[sb.class_of(order)] * (1 + 0.05 * (seed % 3))
        return 12.0 + (size if (seed + len(order)) % 2 else -min(size, 11.0))

    return h_of


@pytest.mark.parametrize(("sizes", "expected"), [
    ((12.0, 11.0, 1.0), "(i)"),
    ((12.0, 1.0, 1.0), "(iii)"),
    ((0.5, 0.5, 0.5), "(ii)"),
    ((4.5, 4.5, 0.5), "reduced"),
])  # fmt: skip
def test_the_outcome_rule_on_synthetic_runs(sizes, expected):
    result = sb.analysis(_runs(_scenario(*sizes)), REFERENCE, 1.55e-05)
    assert result["result"]["outcome"] == expected
    assert result["missing"] == [] and result["h_bar0"] == pytest.approx(100 * 63 / 521)
    if expected != "reduced":
        assert result["result"]["strength"] in ("clear", "leaning")


def test_the_reference_data_give_i_clear():
    """The registered rules on the frozen-Adam reference itself: (i), clear."""

    def h_of(seed, order):
        s = REFERENCE["seeds"][str(seed)]
        return s["h_stream"] if order == "stream" else s["h_stream"] + s["edits"][order]["dh"]

    result = sb.analysis(_runs(h_of), REFERENCE, 1.3e-05)
    assert result["result"] == {**result["result"], "outcome": "(i)", "strength": "clear"}


def test_a_missing_or_incomplete_run_is_listed_not_used():
    runs = _runs(_scenario(12.0, 11.0, 1.0))
    del runs[(50000, "early-0")]
    runs[(50001, "stream")] |= {"stop": "grad_norm", "steps": 7}
    result = sb.analysis(runs, REFERENCE, 1.55e-05)
    assert result["missing"] == ["s50000/early-0"]
    assert all(result["deltas"][f"s50001/{o}"]["dh"] is None
               for o in beta0.orders_of(50001) if o != "stream")  # fmt: skip


def test_a_launcher_end_other_than_done_gives_no_value():
    """A ``conflict`` (or ``diverged``, ``technical``) end never gives a pilot or gate value."""
    rows = [{"seed": s, "outcome": "done", "complete": True, "K": TARGETS[s]} for s in beta0.SEEDS]
    assert beta0.drift_gate(rows, TARGETS)["passed"]
    for end in ("conflict", "diverged", "technical"):
        rows[0] = rows[0] | {"outcome": end}
        assert beta0.drift_gate(rows, TARGETS)["missing"] == [50000]
    pilot = _pilot_rows(lambda s, lr: 3.0 * (lr / 1.3e-05) ** 2)
    for row in pilot:
        if row["seed"] == 50000 and row["level"] in ("1.0", "1.41"):
            row["outcome"] = "conflict"
    assert beta0.rate_choice(pilot, TARGETS)["per_seed"]["50000"]["bracket"] is None


# ------------------------------------------------------------------ the outcome regions


def _intervals(**point):
    return {k: {"lower": v, "upper": v, "resamples": 10_000} for k, v in point.items()}


def test_an_overlap_case_after_scaling_is_ii_not_iii():
    """h-bar0 1.919 %: T = 0.897 pp < F0 = 1.152 pp <= 2 N0 = 1.357 pp: (ii), never (iii)."""
    h_bar0 = 1.919
    noise = sb.label_noise(h_bar0)
    stats = {"F": 1.152, "E": 0.576, "L": 0.576, "D": 0.0}
    found = sb.outcome(stats, _intervals(**stats), noise, REFERENCE, h_bar0)
    assert found["scaled"] and found["threshold_F"] < stats["F"] <= found["two_N"]
    assert found["outcome"] == "(ii)"


def test_zero_harm_is_ii_never_i():
    result = sb.analysis(_runs(lambda seed, order: 0.0), REFERENCE, 1.55e-05)
    assert result["h_bar0"] == 0 and result["result"]["outcome"] == "(ii)"


@pytest.mark.parametrize("h_bar0", [0.5, 1.919, 5.0, 12.8, 30.0])
def test_the_regions_are_mutually_exclusive(h_bar0):
    noise = sb.label_noise(h_bar0)
    grid = [0.0, 0.3, 0.9, 1.4, 2.0, 3.3, 6.0, 9.0, 13.0]
    for f in grid:
        for e in grid:
            for late in grid:
                stats = {"F": f, "E": e, "L": late, "D": e - late}
                found = sb.outcome(stats, _intervals(**stats), noise, REFERENCE, h_bar0)
                two_n, t = found["two_N"], found["threshold_F"]
                ii = f <= two_n and e <= two_n
                persists = f > two_n and f >= t
                i, iii = persists and e - late >= two_n, persists and e - late < two_n
                assert sum((ii, i, iii)) <= 1
                want = "(ii)" if ii else "(i)" if i else "(iii)" if iii else "reduced"
                assert found["outcome"] == want
                if want != "reduced":  # zero-width intervals inside their region: clear
                    assert found["strength"] == "clear"


# ------------------------------------------------------------------ gaps and stops


def test_a_technical_gap_withholds_the_outcome():
    """One complete seed and 29 missing runs: no category, only statistics with seed counts."""
    runs = {k: v for k, v in _runs(_scenario(12.0, 11.0, 1.0)).items() if k[0] == 50000}
    result = sb.analysis(runs, REFERENCE, 1.55e-05)
    assert result["result"]["outcome"] == "withheld" and result["result"]["strength"] is None
    assert len(result["result"]["technical"]) == 29
    assert result["result"]["seeds_per_class"] == {"F": 1, "E": 1, "L": 1}
    runs = _runs(_scenario(12.0, 11.0, 1.0))
    runs[(50002, "late-1")] = runs[(50002, "late-1")] | {"labels": "labels: receipt missing"}
    assert sb.analysis(runs, REFERENCE, 1.55e-05)["result"]["outcome"] == "withheld"


def test_a_scientific_stop_gives_a_conditional_outcome_and_clear_needs_five_seeds():
    runs = _runs(_scenario(12.0, 11.0, 1.0))
    runs[(50001, "stream")] = runs[(50001, "stream")] | {"stop": "k_above_max", "steps": 9}
    result = sb.analysis(runs, REFERENCE, 1.55e-05)["result"]
    assert result["conditional"] is True and result["technical"] == []
    assert len(result["stopped"]) == 5 and result["seeds_per_class"]["F"] == 5
    assert result["outcome"] == "(i)" and result["strength"] == "clear"
    runs[(50002, "stream")] = runs[(50002, "stream")] | {"stop": "k_above_max", "steps": 9}
    result = sb.analysis(runs, REFERENCE, 1.55e-05)["result"]
    assert result["seeds_per_class"]["F"] == 4
    assert result["outcome"] == "(i)" and result["strength"] == "leaning"


@pytest.mark.parametrize(("stop", "steps", "state"), [
    ("k_above_max", 9, "stopped"),
    ("gradient_spike", 25, "stopped"),  # an instability at the 25th update is scientific too
    ("interrupted", 9, "technical"),  # not a stop of the protocol
    ("timeout", 25, "technical"),
    ("k_above_max", None, "technical"),  # no number of updates
    ("k_above_max", 0, "technical"),
    ("k_above_max", 26, "technical"),
    ("k_above_max", 9.0, "technical"),
    ("steps", 24, "technical"),  # all updates claimed, 24 taken
])  # fmt: skip
def test_only_an_instability_stop_with_a_valid_step_count_is_scientific(stop, steps, state):
    runs = _runs(_scenario(12.0, 11.0, 1.0))
    runs[(50001, "early-0")] = runs[(50001, "early-0")] | {"stop": stop, "steps": steps}
    assert sb.run_state(runs[(50001, "early-0")]) == state
    result = sb.analysis(runs, REFERENCE, 1.55e-05)["result"]
    if state == "technical":
        assert result["outcome"] == "withheld" and result["conditional"] is False
        assert result["technical"] == ["s50001/early-0"]
    else:
        assert result["outcome"] == "(i)" and result["conditional"] is True
        assert result["stopped"] == ["s50001/early-0"]
