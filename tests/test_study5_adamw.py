"""Study 5's order repeat under AdamW (docs/study5-adamw-registration.md): the seeds record, the
``order_a`` role, its config against the F order control's, and a tiny A run on CPU that trains
on exactly the F control's batches of the same order seed."""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf
from test_golden_single_process import OVERRIDES
from test_schedule import scheduled
from test_study5_options import F_RUN, inputs_of, key, own_runs_dir, ready, spy_batches
from test_train import env as env  # noqa: F401  (the fixture)
from test_train import read_jsonl

from zero_field_refusal import adamw_orders
from zero_field_refusal.config import config_hash
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import ProtocolError, load_run_config
from zero_field_refusal.schedule import run_scheduled

A_RUN = ("optimizer=adamw", *OVERRIDES)  # the golden run's overrides under arm A


def test_the_committed_seeds_record_is_the_draw():
    record = json.loads((REPO_ROOT / adamw_orders.SEEDS_FILE).read_text())
    assert record == adamw_orders.seeds_record()
    assert len(record["seeds"]) == 10 and set(record["seeds"]) <= set(adamw_orders.E2_SEEDS)
    assert len(adamw_orders.E2_SEEDS) == 37
    assert set(adamw_orders.EXCLUDED).isdisjoint(adamw_orders.E2_SEEDS)
    assert set(adamw_orders.E2_SEEDS) | set(adamw_orders.EXCLUDED) == set(range(50000, 50040))


def test_the_config_is_the_f_order_controls_but_for_its_optimizer_and_role():
    common = ["optimizer.lr=1.45e-05", "seed=50019", "train.order_seed=51019"]
    ours = load_run_config(adamw_orders.CONFIG, common)
    control = load_run_config("experiments/study5_control_order", common)
    assert config_hash(ours) and ours.optimizer.name == "adamw"
    a, f = (OmegaConf.to_container(c, resolve=True) for c in (ours, control))
    assert a.pop("study5") == {"role": "order_a"} and f.pop("study5") == {"role": "control"}
    assert a.pop("optimizer")["lr"] == 1.45e-05 and f.pop("optimizer")["name"] == "adam_frozen"
    assert a == f
    with pytest.raises((ProtocolError, ValueError), match="order_seed|Missing|order_a"):
        config_hash(load_run_config(adamw_orders.CONFIG, common[:2]))


def test_the_cells_are_e2s_orders_at_as_rates():
    cells = [adamw_orders.cell(50019, order, rate) for order, _ in adamw_orders.ORDER_OFFSETS
             for rate, _ in adamw_orders.RATES]  # fmt: skip
    assert [c["overrides"] for c in cells] == [
        ["optimizer.lr=1.45e-05", "train.order_seed=51019"],
        ["optimizer.lr=2.11e-05", "train.order_seed=51019"],
        ["optimizer.lr=1.45e-05", "train.order_seed=52019"],
        ["optimizer.lr=2.11e-05", "train.order_seed=52019"],
    ]
    assert len({c["name"] for c in cells}) == 4


@pytest.mark.parametrize("overrides", [
    ("+study5.role=order_a", "+train.order_seed=3"),
    ("+study5.role=order_a", "+train.order_seed=3", "tracking.endpoint_checkpoint=/x"),
])  # fmt: skip
def test_the_order_a_role_accepts_arm_a_in_a_new_order(tmp_path, overrides) -> None:
    scheduled(tmp_path, *A_RUN, *overrides, policy="short")


@pytest.mark.parametrize(("base", "overrides", "message"), [
    (A_RUN, ("+study5.role=order_a",), "order_seed"),
    (F_RUN, ("+study5.role=order_a", "+train.order_seed=3"), "adamw"),
    (A_RUN, ("+study5.role=order_a", "+train.order_seed=3",
             "+train.order_edit={kind: swap_batches, step: 1}"), "order_seed and no other"),
    (A_RUN, ("+study5.role=order_a", "+train.order_seed=3",
             "+train.rate_schedule={early_steps: 1, early_lr: 0.0005}"), "dose role only"),
    (A_RUN, ("+study5.role=control", "+train.order_seed=3"), "adam_frozen"),
    (A_RUN, ("+study5.role=fresh",), "adam_frozen"),
])  # fmt: skip
def test_bad_order_a_runs_are_refused(tmp_path, base, overrides, message) -> None:
    with pytest.raises(ProtocolError, match=message):
        scheduled(tmp_path, *base, *overrides, policy="short")


def test_an_a_run_trains_on_the_f_controls_batches_of_its_order_seed(env, tmp_path, monkeypatch) -> None:  # noqa: F811,E501
    """The same order seed gives the A run and the F control the same batches, in the same order;
    the A run estimates no v₀ (adaptive Adam) and its updates differ from F's."""
    torch.set_num_threads(1)
    seen = spy_batches(monkeypatch)
    runs = {}
    for name, extra in (("f", (*F_RUN, "+study5.role=control")),
                        ("a", (*A_RUN, "+study5.role=order_a"))):  # fmt: skip
        cfg = scheduled(tmp_path, *extra, "+train.order_seed=7", policy="short")
        if name == "f":  # the streams both read; A reads no preconditioner stream
            cfg = ready(env, cfg)
        own_runs_dir(monkeypatch, tmp_path, name)
        seen.clear()
        run_dir = run_scheduled(cfg, env, run_id=name)
        runs[name] = run_dir, [[key(s) for s in b] for b in seen], cfg
    (f_dir, f_batches, _), (a_dir, a_batches, a_cfg) = runs["f"], runs["a"]
    assert a_batches == f_batches and len(a_batches) == 3
    a_in, f_in = inputs_of(a_dir), inputs_of(f_dir)
    assert a_in["training_order"] == f_in["training_order"]
    assert "v0_sha256" not in a_in and "v0_sha256" in f_in
    assert a_in["theta0_trainable_sha256"] == f_in["theta0_trainable_sha256"]
    a_steps = [r for r in read_jsonl(a_dir / "metrics.jsonl") if "loss" in r]
    f_steps = [r for r in read_jsonl(f_dir / "metrics.jsonl") if "loss" in r]
    assert a_steps[0]["loss"] == f_steps[0]["loss"]  # the first batch at θ₀
    assert a_steps[-1]["loss"] != f_steps[-1]["loss"]  # the optimizers differ
    assert a_cfg.optimizer.name == "adamw"
