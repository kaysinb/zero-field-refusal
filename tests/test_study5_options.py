"""Study 5's options of the scheduled run on CPU (docs/study5-preregistration.md, "Descriptive
measurements"): the training order of the crossed control (b), the preconditioner source of
control (c), and the study-5 roles of the protocol (the line scan is a run of its own:
tests/test_line_scan.py).
Every run is a tiny fixed-policy F run; with every option absent the run is the golden one
(tests/test_golden_single_process.py)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest
import torch
from test_endpoint_checkpoint import final_models
from test_golden_single_process import OVERRIDES, snapshot
from test_schedule import scheduled
from test_train import env as env  # noqa: F401  (the fixture)
from test_train import prepare, read_jsonl

from zero_field_refusal import linear_response, schedule, study5_options
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import ProtocolError
from zero_field_refusal.runtime import StreamPrefix, extend_stream
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.teacher import StreamError

F_RUN = ("optimizer=adam_frozen", *OVERRIDES, "optimizer.preconditioner.batches=3")  # golden F


def ready(env, cfg, *, seeds=(0,)):
    """The streams a run reads, the training stream included, so that two runs of a test read
    the same published chunks (no run extends it)."""
    prepare(env, cfg, "kl_monitor", "preconditioner")
    for seed in seeds:
        if seed != cfg.seed:
            prepare(env, cfg, "preconditioner", seed=seed)
    responses = cfg.train.steps * cfg.train.batch_size
    if StreamPrefix(cfg, env, cfg.seed).available < responses:
        extend_stream(cfg, env, cfg.seed, responses, batch_size=4)
    return cfg


def own_runs_dir(monkeypatch, tmp_path, name: str) -> None:
    """A runs directory per run, so that two runs share a run id and their checkpoints a name."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / f"runs-{name}"))


def spy_batches(monkeypatch) -> list[list]:
    """The training batches a run trains on, read by a spy that changes nothing (the
    preconditioner's batches go through ``train``'s own reference, not this one)."""
    seen, original = [], schedule.accumulate_gradient

    def spy(model, samples, *args, **kwargs):
        seen.append(list(samples))
        return original(model, samples, *args, **kwargs)

    monkeypatch.setattr(schedule, "accumulate_gradient", spy)
    return seen


def key(sample) -> tuple:
    return sample.prompt_id, sample.response_ids


def inputs_of(run_dir) -> dict:
    return json.loads((run_dir / "meta.json").read_text())["inputs"]


# ---------------------------------------------------------------------------------------- order


def test_order_seed_permutes_only_order(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """The same responses in another fixed order: the multiset of the responses trained on is
    unchanged, the order differs, v₀ and θ₀ are the same, and the predictor's batches are the
    reordered run's. Three chunks of 5, so that a batch draws from several."""
    torch.set_num_threads(1)
    seen = spy_batches(monkeypatch)
    dirs = {}
    for name, extra in (
        ("plain", ("+study5.role=fresh",)),
        ("ordered", ("+study5.role=control", "+train.order_seed=7")),
    ):
        cfg = ready(env, scheduled(tmp_path, *F_RUN, "teacher.chunk_size=5", *extra,
                                   policy="short"))  # fmt: skip
        seen.clear()
        dirs[name] = run_scheduled(cfg, env, run_id=name), [list(b) for b in seen], cfg
    (plain_dir, plain, plain_cfg), (ordered_dir, ordered, cfg) = dirs["plain"], dirs["ordered"]
    assert len(plain) == len(ordered) == 3
    flat = [key(s) for b in plain for s in b]
    assert Counter(flat) == Counter(key(s) for b in ordered for s in b)
    assert [key(s) for b in ordered for s in b] != flat

    order = study5_options.training_order(cfg)
    assert order == study5_options.training_order(cfg) and sorted(order) == list(range(12))
    assert study5_options.training_order(plain_cfg) is None
    stream = [s for b in plain for s in b]  # the plain run reads the stream in order
    assert [key(s) for b in ordered for s in b] == [key(stream[i]) for i in order]

    batches, training = linear_response.feed_batches(cfg, env)
    assert batches == ordered
    a, b = inputs_of(plain_dir), inputs_of(ordered_dir)
    assert training == b["streams"]["training"] == a["streams"]["training"]
    assert b["training_order"] == {
        "order_seed": 7, "responses": 12,
        "permutation_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest()}  # fmt: skip
    assert "training_order" not in a
    for name in ("v0_sha256", "theta0_trainable_sha256"):
        assert a[name] == b[name], name
    assert a["streams"]["preconditioner"] == b["streams"]["preconditioner"]
    checks = read_jsonl(ordered_dir / "chunk_checks.jsonl")
    assert [r["chunk"] for r in checks] == [0, 1, 2]  # every chunk, before the first update


# ---------------------------------------------------------------------------- preconditioner


def test_source_seed_own_equals_default_bitwise(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """``source_seed`` set to the run's own seed is the default: every record file and the
    endpoint state equal bit for bit; meta.json adds only the source seed."""
    torch.set_num_threads(1)
    models = final_models(monkeypatch)
    out = {}
    for name, extra in (
        ("default", ("+study5.role=fresh",)),
        ("own", ("+study5.role=control", "+optimizer.preconditioner.source_seed=0")),
    ):
        ckpt = tmp_path / f"ckpt-{name}"
        cfg = ready(
            env,
            scheduled(
                tmp_path, *F_RUN, *extra, f"tracking.endpoint_checkpoint={ckpt}", policy="short"
            ),
        )
        own_runs_dir(monkeypatch, tmp_path, name)
        run_dir = run_scheduled(cfg, env, run_id="f")
        out[name] = {"records": snapshot(run_dir),
                     "student": {n: p.detach().clone() for n, p in
                                 freeze(models[-1], cfg.train.freeze)},
                     "ckpt": torch.load(ckpt / "f.pt", weights_only=True)}  # fmt: skip
    default, own = out["default"], out["own"]
    assert set(own["records"]) == set(default["records"])
    inputs = json.loads(own["records"]["inputs"][0])
    assert inputs.pop("preconditioner_source_seed") == 0
    assert json.dumps(inputs, sort_keys=True) == default["records"]["inputs"][0]
    for name in default["records"]:
        if name != "inputs":
            assert own["records"][name] == default["records"][name], name
    assert own["ckpt"].keys() == default["ckpt"].keys() == default["student"].keys()
    for n, p in default["student"].items():
        assert torch.equal(p, own["student"][n]), n
        assert torch.equal(own["ckpt"][n], default["ckpt"][n]), n


def test_source_seed_other_changes_v0_only(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """v₀ from seed 1's preconditioner stream: v₀ and the preconditioner stream are seed 1's, the
    training batches, θ₀ and the step-1 gradient (at θ₀) are the run's own; the predictor's v₀
    follows the config. A seed whose stream is not in the cache stops the run before training."""
    torch.set_num_threads(1)
    seen = spy_batches(monkeypatch)
    runs = {}
    for name, extra in (
        ("own", ("+study5.role=fresh",)),
        ("other", ("+study5.role=control", "+optimizer.preconditioner.source_seed=1")),
    ):
        cfg = ready(env, scheduled(tmp_path, *F_RUN, *extra, policy="short"), seeds=(0, 1))
        seen.clear()
        run_dir = run_scheduled(cfg, env, run_id=name)
        runs[name] = inputs_of(run_dir), [list(b) for b in seen], read_jsonl(
            run_dir / "metrics.jsonl"), cfg  # fmt: skip
    (own, own_batches, own_metrics, _), (other, batches, metrics, cfg) = runs["own"], runs["other"]
    assert other["preconditioner_source_seed"] == 1 and "preconditioner_source_seed" not in own
    assert other["v0_sha256"] != own["v0_sha256"]
    assert other["streams"]["preconditioner"] != own["streams"]["preconditioner"]
    assert other["streams"]["preconditioner"]["path"].endswith("seed-1")
    assert other["streams"]["training"] == own["streams"]["training"]
    assert other["theta0_trainable_sha256"] == own["theta0_trainable_sha256"]
    assert batches == own_batches
    steps, own_steps = ([r for r in rows if "loss" in r] for rows in (metrics, own_metrics))
    assert len(steps) == len(own_steps) == 3
    assert steps[0]["grad_norm"] == own_steps[0]["grad_norm"]  # g₁ at θ₀: v₀ not yet used
    assert steps[0]["loss"] == own_steps[0]["loss"]
    assert steps[-1]["loss"] != own_steps[-1]["loss"]  # the updates differ

    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    v0, identity = linear_response.theta0_second_moment(cfg, env, model, trainable)
    assert linear_response.v0_sha256(trainable, v0) == other["v0_sha256"]
    assert identity == other["streams"]["preconditioner"]

    missing = scheduled(tmp_path, *F_RUN, "+study5.role=control",
                        "+optimizer.preconditioner.source_seed=2", policy="short")  # fmt: skip
    seen.clear()
    with pytest.raises(StreamError, match="source_seed"):
        run_scheduled(missing, env, run_id="missing")
    assert seen == []  # nothing trained


# ----------------------------------------------------------------------------------- the roles


def role_config(tmp_path, *overrides: str):
    return scheduled(tmp_path, *F_RUN, *overrides, policy="short")


@pytest.mark.parametrize("overrides", [
    ("+study5.role=fresh", "tracking.endpoint_checkpoint=/x"),
    ("+study5.role=fresh",),
    ("+study5.role=control", "+train.order_seed=3", "tracking.endpoint_checkpoint=/x"),
    ("+study5.role=control", "+optimizer.preconditioner.source_seed=50001"),
])  # fmt: skip
def test_study5_roles_accept_their_keys(tmp_path, overrides) -> None:
    role_config(tmp_path, *overrides)


@pytest.mark.parametrize(("overrides", "message"), [
    (("+study5.role=other",), "study5.role"),
    (("+study5.role=fresh", "+train.order_seed=3"), "train.order_seed"),
    (("+study5.role=fresh", "+optimizer.preconditioner.source_seed=1"), "source_seed"),
    (("+study5.role=control",), "exactly one"),
    (("+study5.role=control", "+train.order_seed=3", "+optimizer.preconditioner.source_seed=1"),
     "exactly one"),
    (("+study5.role=control", "+train.order_seed=true"), "train.order_seed"),
    (("+study5.role=fresh", "optimizer=adamw", "optimizer.lr=1e-3"), "adam_frozen"),
    (("+train.order_seed=3",), "study-5 role"),
    (("+optimizer.preconditioner.source_seed=1",), "study-5 role"),
])  # fmt: skip
def test_study5_keys_outside_their_roles_are_refused(tmp_path, overrides, message) -> None:
    with pytest.raises(ProtocolError, match=message):
        role_config(tmp_path, *overrides)


def test_study3_roles_are_not_loosened() -> None:
    from zero_field_refusal.protocol import load_run_config

    base = ["optimizer=adam_frozen", "optimizer.lr=1e-5", "seed=3000"]
    with pytest.raises(ProtocolError, match="endpoint checkpoint"):
        load_run_config("experiments/study3", ["study3.role=confirmation", *base,
                                               "tracking.endpoint_checkpoint=/x"])  # fmt: skip
    for extra in ("+train.order_seed=3", "+optimizer.preconditioner.source_seed=1",
                  "+study5.role=fresh"):  # fmt: skip
        with pytest.raises(ProtocolError, match="study-5|study3"):
            load_run_config("experiments/study3", ["study3.role=engineering", *base, extra])
    load_run_config("experiments/study3", ["study3.role=engineering", *base,
                                           "tracking.endpoint_checkpoint=/x"])  # fmt: skip
