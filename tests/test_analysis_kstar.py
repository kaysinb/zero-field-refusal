import math

import pandas as pd
import pytest

from zero_field_refusal.analysis.kstar import (
    CommonRangeUndefined,
    cell_means,
    common_range,
    decisive_cells,
    k_star_levels,
    select_k_star,
)


def _runs(cells: dict[tuple[str, float], list[float]]) -> pd.DataFrame:
    """Runs with the given K values per cell (arm, lr); seeds 0, 1, ... within each cell."""
    rows = [
        {"run_id": f"{arm}-{lr}-{seed}", "seed": seed, "arm": arm, "lr": lr, "K": k}
        for (arm, lr), values in cells.items()
        for seed, k in enumerate(values)
    ]
    return pd.DataFrame(rows)


# Geometric cell means: sgd 1, 4, 16; adamw 2, 8, 32; adam_frozen 3, 6.
CELLS = {
    ("sgd", 1.0): [0.5, 2.0],
    ("sgd", 2.0): [4.0],
    ("sgd", 3.0): [8.0, 32.0],
    ("adamw", 1.0): [2.0],
    ("adamw", 2.0): [8.0],
    ("adamw", 3.0): [32.0],
    ("adam_frozen", 1.0): [3.0],
    ("adam_frozen", 2.0): [6.0],
}


def test_cell_mean_is_the_mean_of_log_k() -> None:
    means = cell_means(_runs(CELLS)).set_index(["arm", "lr"])
    assert means.loc[("sgd", 1.0), "mean_log_k"] == pytest.approx(math.log(1.0))
    assert means.loc[("sgd", 3.0), "mean_log_k"] == pytest.approx(math.log(16.0))
    assert means.loc[("sgd", 3.0), "n_runs"] == 2


def test_common_range_is_the_overlap_of_the_three_primary_arms() -> None:
    c_lo, c_hi, descriptive_only = common_range(cell_means(_runs(CELLS)))
    assert (c_lo, c_hi) == (pytest.approx(3.0), pytest.approx(6.0))
    assert not descriptive_only


def test_k_star_is_the_geometric_midpoint_with_quarter_points() -> None:
    lower, k_star, upper = k_star_levels(3.0, 6.0)
    assert k_star == pytest.approx(math.sqrt(18.0))
    assert lower == pytest.approx(3.0 * 2**0.25)
    assert upper == pytest.approx(3.0 * 2**0.75)


def test_frozen_adam_outside_the_overlap_makes_h2_descriptive() -> None:
    cells = {key: ks for key, ks in CELLS.items() if key[0] != "adam_frozen"}
    cells |= {("adam_frozen", 1.0): [20.0], ("adam_frozen", 2.0): [25.0]}
    c_lo, c_hi, descriptive_only = common_range(cell_means(_runs(cells)))
    assert (c_lo, c_hi) == (pytest.approx(2.0), pytest.approx(16.0))
    assert descriptive_only


def test_no_sgd_adamw_overlap_leaves_k_star_undefined() -> None:
    cells = {key: ks for key, ks in CELLS.items() if key[0] != "adamw"}
    cells |= {("adamw", 1.0): [20.0], ("adamw", 2.0): [40.0]}
    with pytest.raises(CommonRangeUndefined):
        select_k_star(_runs(cells))


def test_a_missing_primary_arm_is_an_error() -> None:
    cells = {key: ks for key, ks in CELLS.items() if key[0] != "adam_frozen"}
    with pytest.raises(ValueError, match="adam_frozen"):
        select_k_star(_runs(cells))


def test_decisive_cells_take_the_nearest_cell_on_each_side_or_the_single_closest() -> None:
    cells = CELLS | {
        ("sgd_momentum", 1.0): [10.0],
        ("sgd_momentum", 2.0): [20.0],
        ("adamw_prewarmed", 1.0): [1.0],
        ("adamw_prewarmed", 2.0): [2.0],
    }
    decisive = decisive_cells(cell_means(_runs(cells)), math.sqrt(18.0))
    assert decisive == {
        "adam_frozen": (1.0, 2.0),
        "adamw": (1.0, 2.0),
        "adamw_prewarmed": (2.0,),
        "sgd": (2.0, 3.0),
        "sgd_momentum": (1.0,),
    }


def test_selection_fixes_k_star_from_kl_values_only() -> None:
    selection = select_k_star(_runs(CELLS))
    assert selection.k_star == pytest.approx(math.sqrt(18.0))
    assert selection.levels == {
        "lower_quarter": pytest.approx(3.0 * 2**0.25),
        "k_star": pytest.approx(math.sqrt(18.0)),
        "upper_quarter": pytest.approx(3.0 * 2**0.75),
    }
    assert selection.exclusion_threshold == pytest.approx(0.3)
    assert selection.arm_ranges["sgd"] == (pytest.approx(1.0), pytest.approx(16.0))
    assert selection.decisive["sgd"] == (2.0, 3.0)
