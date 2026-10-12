"""The statistics of scripts/stage1_descriptive.py: t-intervals and paired-proportion intervals."""

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage1_descriptive.py"
spec = importlib.util.spec_from_file_location("stage1_descriptive", SCRIPT)
descriptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(descriptive)


def test_the_t_interval_matches_the_hand_computation() -> None:
    out = descriptive.t_interval([0.1, 0.2, 0.3, 0.4, 0.5])
    half = 2.7764451051977934 * math.sqrt(0.025) / math.sqrt(5)
    assert out["mean"] == pytest.approx(0.3)
    assert (out["ci_low"], out["ci_high"]) == pytest.approx((0.3 - half, 0.3 + half))


def test_the_wilson_interval_of_a_known_case() -> None:
    low, high = descriptive.wilson(81, 263)  # Newcombe (1998), single proportion, method 3
    assert (low, high) == pytest.approx((0.2553, 0.3662), abs=5e-4)


def test_newcombe_paired_interval_contains_the_estimate_and_mirrors() -> None:
    pairs = [(True, True)] * 40 + [(True, False)] * 12 + [(False, True)] * 3 + [(False, False)] * 45
    out = descriptive.newcombe_paired(pairs)
    assert out["difference"] == pytest.approx(0.09)
    assert -1 <= out["ci_low"] < out["difference"] < out["ci_high"] <= 1
    mirrored = descriptive.newcombe_paired([(y, x) for x, y in pairs])
    assert mirrored["ci_low"] == pytest.approx(-out["ci_high"])
    assert mirrored["ci_high"] == pytest.approx(-out["ci_low"])
    same = descriptive.newcombe_paired([(True, True)] * 30 + [(False, False)] * 70)
    assert same["difference"] == 0 and same["ci_low"] < 0 < same["ci_high"]
