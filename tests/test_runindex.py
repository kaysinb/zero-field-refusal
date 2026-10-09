from pathlib import Path

from omegaconf import OmegaConf

from zero_field_refusal.runindex import index_runs
from zero_field_refusal.tracking import Run


def test_index_flattens_runs_and_keeps_failed_ones(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    cfg = OmegaConf.create({"experiment": "e", "seed": 3, "optimizer": {"name": "sgd", "lr": 0.1},
                            "tracking": {"wandb": {"enabled": False}}})  # fmt: skip
    with Run(cfg, run_id="ok") as run:
        run.summarize({"K_harmful": 1.5, "controls": {"random_global": {"K_harmful": 2.0}}})
        run.record_inputs({"streams": {"training": {"sha256": "abc"}}})
    try:
        with Run(cfg, run_id="broken"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    table = index_runs(tmp_path).set_index("run_dir")
    assert list(table.index) == ["e/broken", "e/ok"]
    ok = table.loc["e/ok"]
    assert ok["config.seed"] == 3 and ok["config.optimizer.lr"] == 0.1
    assert ok["summary.K_harmful"] == 1.5
    assert ok["summary.controls.random_global.K_harmful"] == 2.0
    assert ok["meta.inputs.streams.training.sha256"] == "abc"
    assert table.loc["e/broken", "meta.status"] == "failed"
    assert not table.loc["e/broken", "has_summary"]
