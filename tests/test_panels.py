import pytest

from zero_field_refusal.panels import monitor_agreement
from zero_field_refusal.protocol import load_run_config


def test_agreement_uses_only_states_from_k_benign_four() -> None:
    scoring = [1.0, 3.9, 4.0, 5.0, 8.0]
    monitor = [3.0, 1.0, 4.1, 5.1, 8.0]  # the two low states would fail; they do not count
    result = monitor_agreement(monitor, scoring)
    assert result["pairs"] == 3
    assert result["median"] == pytest.approx(0.02)
    assert result["passes"] is True


def test_agreement_fails_on_the_median_or_the_95th_percentile() -> None:
    scoring = [5.0] * 20
    assert monitor_agreement([5.3] * 20, scoring)["passes"] is False  # median 6 %
    tail = [5.0] * 18 + [6.0, 6.0]  # median 0, 95th percentile 20 %
    result = monitor_agreement(tail, scoring)
    assert result["median"] == 0.0 and result["p95"] == pytest.approx(0.2)
    assert result["passes"] is False
    assert monitor_agreement([1.0], [2.0]) == {"pairs": 0, "invalid": 0, "median": None,
                                               "p95": None, "passes": None}  # fmt: skip


@pytest.mark.parametrize(
    ("monitor", "scoring"),
    [
        ([5.0, 100.0], [5.0, float("nan")]),  # the reviewer's case: NaN dropped by the K filter
        ([5.0] * 99 + [float("inf")], [5.0] * 100),  # one in 100, beyond the 95th percentile
        ([float("nan"), 5.0], [1.0, 5.0]),  # even below the K at which agreement is read
    ],
)
def test_any_non_finite_reading_fails_the_agreement(monitor, scoring) -> None:
    result = monitor_agreement(monitor, scoring)
    assert result["passes"] is False and result["invalid"] == 1


def test_study2_configs_add_the_monitor_panel_and_the_evaluation_set_to_study1s(tmp_path) -> None:
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("defaults: [experiments/smoke, data/study2, eval/study2]\n")
    plain = load_run_config("experiments/smoke")
    cfg = load_run_config(experiment)
    assert cfg.data.alpaca.monitor_prompts == 25
    assert cfg.eval.monitor.samples_per_prompt == 2 and cfg.eval.monitor.every == 5
    assert cfg.data.harmful.evaluation.jaccard == 0.6
    assert cfg.eval.harmful.prompt_set == "harmful_evaluation"
    cfg.data.alpaca.pop("monitor_prompts")
    cfg.data.harmful.pop("evaluation")
    assert cfg.data == plain.data  # nothing else changes in the data config
    cfg.eval.pop("monitor")
    cfg.eval.harmful.prompt_set = plain.eval.harmful.prompt_set
    assert cfg.eval == plain.eval
