import json
import os
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from zero_field_refusal import tracking
from zero_field_refusal.tracking import Run, git_diff, git_state


@pytest.fixture(autouse=True)
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in ("RUN_ID", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def make_cfg(**tracking_cfg: object) -> DictConfig:
    return OmegaConf.create(
        {"experiment": "unit", "seed": 0, "tracking": {"wandb": {"enabled": False}, **tracking_cfg}}
    )


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_writes_config_meta_metrics_summary(runs_dir: Path) -> None:
    with Run(make_cfg(), run_id="r1") as run:
        run.log({"loss": 1.5}, step=0)
        run.log({"loss": 1.25}, step=1)
        run.summarize({"kl": 0.5})
        out = run.path("outputs", "responses.jsonl")
        out.write_text("{}\n")

    run_dir = runs_dir / "unit" / "r1"
    assert OmegaConf.load(run_dir / "config.yaml").seed == 0
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["config_hash"] == run.config_hash
    assert [line["loss"] for line in read_lines(run_dir / "metrics.jsonl")] == [1.5, 1.25]
    assert json.loads((run_dir / "summary.json").read_text()) == {"kl": 0.5}
    assert out.exists()


def test_meta_records_provenance_without_initializing_cuda(runs_dir: Path) -> None:
    Run(make_cfg(), run_id="p").finish()
    meta = json.loads((runs_dir / "unit" / "p" / "meta.json").read_text())
    assert len(meta["uv_lock_sha256"]) == 64
    assert isinstance(meta["argv"], list)
    assert {"gpus", "driver", "torch_cuda", "cuda_visible_devices", "world_size"} <= set(
        meta["hardware"]
    )
    assert not torch.cuda.is_initialized()


def test_meta_records_the_launched_snapshot_and_image(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_SNAPSHOT_ID", "snap-1")
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    Run(make_cfg(), run_id="s").finish()
    meta = json.loads((runs_dir / "unit" / "s" / "meta.json").read_text())
    assert meta["launch"] == {"source_snapshot_id": "snap-1", "container_image": None}


def test_failure_is_recorded(runs_dir: Path) -> None:
    with pytest.raises(RuntimeError), Run(make_cfg(), run_id="r2"):
        raise RuntimeError("boom")
    meta = json.loads((runs_dir / "unit" / "r2" / "meta.json").read_text())
    assert meta["status"] == "failed"


def test_run_id_is_never_reused(runs_dir: Path) -> None:
    Run(make_cfg(), run_id="r3").finish()
    with pytest.raises(FileExistsError):
        Run(make_cfg(), run_id="r3")


def test_generated_id_is_sanitized_and_unique(runs_dir: Path) -> None:
    cfg = make_cfg()
    cfg.name = "lr=1e-5 / " + "x" * 80
    cfg.experiment = "../outside"
    first, second = Run(cfg), Run(cfg)
    assert first.run_id != second.run_id
    assert len(first.run_id) <= 64
    assert "/" not in first.run_id and "=" not in first.run_id
    assert first.dir.parent == runs_dir / "outside"


def test_invalid_run_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Run id"):
        Run(make_cfg(), run_id="../r")


def test_non_main_process_writes_nothing(runs_dir: Path) -> None:
    run = Run(make_cfg(), run_id="r4", is_main=False)
    run.log({"loss": 1.0}, step=0)
    run.summarize({"kl": 0.1})
    run.finish()
    assert not (runs_dir / "unit").exists()


def test_rank_decides_main(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    Run(make_cfg(), run_id="r5").finish()
    assert not (runs_dir / "unit").exists()


def test_run_id_from_environment(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_ID", "shared-id")
    assert Run(make_cfg()).dir == runs_dir / "unit" / "shared-id"


def test_distributed_run_needs_a_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="process group"):
        Run(make_cfg())


def _rank_worker(rank: int, world_size: int, tmp: Path) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size))
    init = f"file://{tmp / 'process_group'}"
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world_size)
    try:
        run = Run(make_cfg())
        run.log({"rank": rank}, step=0)
        run.finish()
        (tmp / f"id-{rank}").write_text(run.run_id)
    finally:
        dist.destroy_process_group()


def test_ranks_share_the_id_generated_by_rank_0(runs_dir: Path) -> None:
    mp.spawn(_rank_worker, args=(2, runs_dir), nprocs=2)
    run_id = (runs_dir / "id-0").read_text()
    assert (runs_dir / "id-1").read_text() == run_id
    assert [p.name for p in (runs_dir / "unit").iterdir()] == [run_id]
    records = read_lines(runs_dir / "unit" / run_id / "metrics.jsonl")
    assert [record["rank"] for record in records] == [0]


def test_array_values_become_plain_json(runs_dir: Path) -> None:
    with Run(make_cfg(), run_id="v") as run:
        run.log(
            {
                "a": np.float32(1.5),
                "b": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
                "nested": {"kl": torch.tensor(0.5), "pair": (np.int64(1), torch.tensor(2))},
                "grad": torch.tensor(3.0, requires_grad=True),
                "nan": float("nan"),
            },
            step=np.int64(3),
        )
        run.summarize({"final": {"kl": torch.tensor(0.25)}})
    [record] = read_lines(runs_dir / "unit" / "v" / "metrics.jsonl")
    assert record["step"] == 3 and type(record["step"]) is int
    assert record["a"] == 1.5 and record["b"] == [1.0, 2.0] and record["grad"] == 3.0
    assert record["nested"] == {"kl": 0.5, "pair": [1, 2]}
    assert record["nan"] != record["nan"]  # NaN round-trips through Python's json
    summary = json.loads((runs_dir / "unit" / "v" / "summary.json").read_text())
    assert summary == {"final": {"kl": 0.25}}


def test_bad_values_fail_loudly_and_write_nothing(runs_dir: Path) -> None:
    with Run(make_cfg(), run_id="bad") as run:
        for metrics in ({"step": 1}, {"time_s": 1.0}):
            with pytest.raises(ValueError, match="Reserved metric names"):
                run.log(metrics, step=0)
        with pytest.raises(TypeError):
            run.log({"loss": object()}, step=0)
        with pytest.raises(TypeError):
            run.summarize({"kl": object()})
        run.summarize({"kl": 0.5})
    assert not (runs_dir / "unit" / "bad" / "metrics.jsonl").exists()
    assert json.loads((runs_dir / "unit" / "bad" / "summary.json").read_text()) == {"kl": 0.5}


def test_clean_git_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking, "git_state", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(RuntimeError, match="require_clean_git"):
        Run(make_cfg(require_clean_git=True), run_id="r6")
    monkeypatch.setattr(tracking, "git_state", lambda: {"commit": "abc", "dirty": False})
    Run(make_cfg(require_clean_git=True), run_id="r7").finish()


def test_dirty_tree_is_saved_with_the_run(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking, "git_state", lambda: {"commit": "abc", "dirty": True})
    monkeypatch.setattr(tracking, "git_diff", lambda: "the diff\n")
    Run(make_cfg(), run_id="d").finish()
    assert (runs_dir / "unit" / "d" / "git.diff").read_text() == "the diff\n"


def test_git_state_against_temporary_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
        subprocess.run(
            ["git", *identity, "-c", "commit.gpgsign=false", *args], cwd=repo, check=True
        )

    git("init", "-q")
    (repo / "a.txt").write_text("one\n")
    git("add", "a.txt")
    git("commit", "-q", "-m", "init")
    state = git_state(repo)
    assert len(state["commit"]) == 40 and state["dirty"] is False

    (repo / "a.txt").write_text("two\n")
    (repo / "new.txt").write_text("x\n")
    assert git_state(repo)["dirty"] is True
    diff = git_diff(repo)
    assert "+two" in diff and diff.endswith("# Untracked files:\nnew.txt\n")

    (repo / "sub").mkdir()
    assert git_state(repo / "sub") == {"commit": None, "dirty": None}  # not the top level


@pytest.fixture
def offline_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    for name in ("CONFIG", "CACHE", "DATA"):
        monkeypatch.setenv(f"WANDB_{name}_DIR", str(tmp_path / f"wandb-{name.lower()}"))


def wandb_cfg(**options: object) -> DictConfig:
    cfg = make_cfg()
    cfg.tracking.wandb = {"enabled": True, "project": "unit", **options}
    return cfg


@pytest.mark.usefixtures("offline_wandb")
def test_wandb_mirror_records_no_host_or_home() -> None:
    with Run(wandb_cfg(), run_id="w") as run:
        run.log({"loss": 1.0}, step=5)
        run.log({"eval_loss": 2.0}, step=2)
        run.summarize({"kl": 0.5})
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["wandb"] == {"id": "w", "project": "unit"}
    [synced] = (run.dir / "wandb").glob("offline-run-*/run-w.wandb")
    data = synced.read_bytes()
    assert b"eval_loss" in data
    assert socket.gethostname().encode() not in data
    assert str(Path.home()).encode() not in data


@pytest.mark.usefixtures("offline_wandb")
def test_wandb_failure_is_not_fatal(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wandb

    def broken_init(**kwargs: object) -> None:
        raise ConnectionError("no network")

    monkeypatch.setattr(wandb, "init", broken_init)
    with pytest.warns(UserWarning, match="W&B"), Run(wandb_cfg(), run_id="f") as run:
        run.log({"loss": 1.0}, step=0)
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert "no network" in meta["wandb"]["error"]

    with pytest.raises(ConnectionError):
        Run(wandb_cfg(required=True), run_id="g")
    meta = json.loads((runs_dir / "unit" / "g" / "meta.json").read_text())
    assert meta["status"] == "failed" and "no network" in meta["wandb"]["error"]
