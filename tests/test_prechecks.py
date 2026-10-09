"""The χ² upper bound of the path-field checks: coverage, scaling and edge cases."""

import numpy as np
import pytest

from zero_field_refusal.prechecks import chi_square_bound

# Per-response sums of χ² are nonnegative and can be right-skewed; the bound must keep close to
# its 97.5 % level for moderate skew. (A Student quantile drops to ≈ 0.93 for lognormal σ = 1 at
# n = 160; very heavy tails, lognormal σ ≥ 1.5, undercover with any method; see the docstring.)
CASES = [
    # name, draw, true mean, responses, minimum coverage
    ("normal", lambda rng, n: 1.0 + 0.3 * rng.standard_normal(n), 1.0, 160, 0.96),
    ("exponential", lambda rng, n: rng.exponential(1.0, n), 1.0, 160, 0.955),
    ("lognormal", lambda rng, n: rng.lognormal(0.0, 1.0, n), np.exp(0.5), 160, 0.94),
    ("lognormal", lambda rng, n: rng.lognormal(0.0, 1.0, n), np.exp(0.5), 500, 0.95),
]


@pytest.mark.parametrize(("name", "draw", "mean", "responses", "minimum"), CASES)
def test_the_bound_covers_the_mean_at_close_to_its_level(name, draw, mean, responses, minimum):
    rng = np.random.default_rng(1)
    repetitions, batch, steps = 1000, 20, 25
    covered = sum(
        chi_square_bound(draw(rng, responses), batch_size=batch, steps=steps, resamples=999)[
            "bound_upper"
        ]
        >= steps * batch * mean
        for _ in range(repetitions)
    )
    # Monte Carlo standard error of the coverage ≈ 0.005.
    assert minimum <= covered / repetitions <= 0.99, name


def test_the_bound_scales_with_steps_and_batch_size():
    values = np.random.default_rng(0).exponential(1e-6, 64)
    one = chi_square_bound(values, batch_size=1, steps=1)
    more = chi_square_bound(values, batch_size=20, steps=25)
    assert more["bound_upper"] == pytest.approx(500 * one["bound_upper"])
    assert more["chi_square_per_batch"] == pytest.approx(20 * values.mean())
    assert one["quantile"] > 0 and one["skewness"] > 0 and 0 < one["largest_share"] < 1


def test_identical_paths_give_a_zero_bound_and_too_few_responses_fail():
    zero = chi_square_bound([0.0] * 10, batch_size=20, steps=25)
    assert zero["bound_upper"] == 0.0 and zero["quantile"] == 0.0
    with pytest.raises(ValueError, match="at least 2"):
        chi_square_bound([1e-6], batch_size=20, steps=25)
