"""Study 5's options of the scheduled run on CPU (docs/study5-preregistration.md, "Descriptive
measurements"): the line scan after the last evaluation, the training order of the crossed
control (b), the preconditioner source of control (c), and the study-5 roles of the protocol.
Every run is a tiny fixed-policy F run; with every option absent the run is the golden one
(tests/test_golden_single_process.py)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest
import torch
from omegaconf import OmegaConf
from test_endpoint_checkpoint import final_models, rng_state, same_rng
from test_golden_single_process import OVERRIDES, normalize, snapshot
from test_schedule import scheduled
from test_train import env as env  # noqa: F401  (the fixture)
from test_train import prepare, read_jsonl

from zero_field_refusal import linear_response, schedule, study5_options
from zero_field_refusal.evaluation import line_scan_condition
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import ProtocolError
from zero_field_refusal.run_judging import response_conditions
from zero_field_refusal.runtime import StreamPrefix, extend_stream
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.teacher import StreamError
from zero_field_refusal.train import Evaluator

F_RUN = ("optimizer=adam_frozen", *OVERRIDES, "optimizer.preconditioner.batches=3")  # golden F
POINTS = [-1, -0.5, -0.25, 0.25, 0.5]
SCAN = f"+eval.controls.line_scan.points=[{','.join(str(a) for a in POINTS)}]"


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


def f_run(env, tmp_path, role: str, *overrides: str):
    return ready(env, scheduled(tmp_path, *F_RUN, f"+study5.role={role}", *overrides,
                                policy="short"))  # fmt: skip


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


def run_config(run_dir):
    return OmegaConf.load(run_dir / "config.yaml")


def without(rows: list[str], prefix: str) -> list[str]:
    return [row for row in rows if not json.loads(row)["condition"].startswith(prefix)]


# ------------------------------------------------------------------------------------ line scan


def test_line_scan_points_and_restore(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """Each point is θ₀ + a(θ₃ − θ₀) exactly, evaluated in full on the step-3 evaluation's
    prompts; the student holds θ₃ again afterwards, bit for bit, the endpoint checkpoint holds
    θ₃, and every record but the scan's own rows equals a run without the scan."""
    torch.set_num_threads(1)
    models = final_models(monkeypatch)
    at_points = []  # the trainable parameters at each evaluation of a scan point
    original = Evaluator.__call__

    def spy(self, model, condition, **kwargs):
        if condition.startswith("line_scan_"):
            at_points.append((condition, {n: p.detach().clone() for n, p in
                                          freeze(model, self.cfg.train.freeze)}))  # fmt: skip
        return original(self, model, condition, **kwargs)

    monkeypatch.setattr(Evaluator, "__call__", spy)
    out = {}
    for name, extra in (("plain", ()), ("scan", (SCAN,))):
        ckpt = tmp_path / f"ckpt-{name}"
        cfg = f_run(env, tmp_path, "fresh", f"tracking.endpoint_checkpoint={ckpt}", *extra)
        own_runs_dir(monkeypatch, tmp_path, name)
        run_dir = run_scheduled(cfg, env, run_id="f")
        out[name] = {"dir": run_dir, "records": snapshot(run_dir), "rng": rng_state(),
                     "student": {n: p.detach().clone() for n, p in
                                 freeze(models[-1], cfg.train.freeze)},
                     "ckpt": torch.load(ckpt / "f.pt", weights_only=True)}  # fmt: skip
    plain, scan = out["plain"], out["scan"]

    conditions = [line_scan_condition(a) for a in POINTS]
    assert conditions == ["line_scan_-1", "line_scan_-0.5", "line_scan_-0.25", "line_scan_+0.25",
                          "line_scan_+0.5"]  # fmt: skip
    assert [c for c, _ in at_points] == conditions  # in the config's order
    theta0 = dict(freeze(env.new_model(), cfg.train.freeze))
    end = scan["ckpt"]
    for (condition, params), a in zip(at_points, POINTS, strict=True):
        for n, p in params.items():  # θ₀ + aΔθ with Δθ = θ₃ − θ₀ in fp32
            expected = torch.add(theta0[n].detach(), end[n] - theta0[n].detach(), alpha=a)
            assert p.dtype == torch.float32 and torch.equal(p, expected), (condition, n)
        assert not torch.equal(params[n], end[n])

    # restored: the student, the checkpoint, and both equal the run without the scan
    for n, p in scan["student"].items():
        assert torch.equal(p, end[n]) and torch.equal(p, plain["student"][n]), n
        assert torch.equal(plain["ckpt"][n], end[n]), n
    assert same_rng(plain["rng"], scan["rng"])

    rows = read_jsonl(scan["dir"] / "line_scan.jsonl")
    assert [(r["a"], r["condition"], r["step"], r["evaluated"]) for r in rows] == [
        (float(a), c, 3, True) for a, c in zip(POINTS, conditions, strict=True)]  # fmt: skip
    keys = {"K_benign", "K_harmful", "heldout_nll_per_token", "factual_accuracy",
            "benchmark_accuracy"}  # fmt: skip
    assert all(keys <= set(r) for r in rows)
    summary = json.loads((scan["dir"] / "summary.json").read_text())
    assert summary.pop("line_scan") == {
        "points": [float(a) for a in POINTS],
        "conditions": conditions,
        "evaluated": conditions,
    }

    # the same prompts and measures at every point as at step 3
    got, want = scan["records"], plain["records"]
    measurements = [json.loads(r) for r in got["observations/measurements"]]
    per = Counter((r["condition"], r["measure"], r["item_id"]) for r in measurements)
    step = sorted((m, i) for (c, m, i) in per if c == "step-3")
    for condition in conditions:
        assert sorted((m, i) for (c, m, i) in per if c == condition) == step
    responses = [json.loads(r) for r in got["responses/harmful_evaluation"]]
    ids = [r["item_id"] for r in responses if r["condition"] == "step-3"]
    for condition in conditions:
        assert [r["item_id"] for r in responses if r["condition"] == condition] == ids

    # every other record equals the run without the scan
    for name in want:
        if name in ("observations/measurements", "responses/harmful_evaluation"):
            assert without(got[name], "line_scan_") == want[name], name
        elif name == "summary":
            assert json.dumps(normalize(summary), sort_keys=True) == want[name][0]
        else:
            assert got[name] == want[name], name
    assert set(got) == set(want)
    # the judge reads the scan's conditions after the run's evaluations
    assert response_conditions(run_config(scan["dir"]), scan["dir"]) == ["step-3", *conditions]
    assert response_conditions(run_config(plain["dir"]), plain["dir"]) == ["step-3"]


def test_a_non_finite_scan_point_is_recorded_and_the_scan_goes_on(
    env,
    tmp_path,
    monkeypatch,  # noqa: F811
) -> None:
    from zero_field_refusal.tracking import Diverged

    original = Evaluator.__call__

    def broken(self, model, condition, **kwargs):
        if condition == "line_scan_-1":
            raise Diverged("K_benign not finite")
        return original(self, model, condition, **kwargs)

    monkeypatch.setattr(Evaluator, "__call__", broken)
    cfg = f_run(env, tmp_path, "fresh", SCAN)
    run_dir = run_scheduled(cfg, env, run_id="broken")
    rows = read_jsonl(run_dir / "line_scan.jsonl")
    assert [r["evaluated"] for r in rows] == [False, True, True, True, True]
    assert rows[0]["reason"] == "nonfinite_k"
    conditions = response_conditions(run_config(run_dir), run_dir)
    assert conditions == ["step-3", *(line_scan_condition(a) for a in POINTS[1:])]


def test_no_scan_after_an_early_stop(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    norms = iter([1.0, 50.0])
    real = schedule._norm

    def norm(tensors):  # a gradient spike at step 1, the weight check stays real
        first = next(t for t in tensors if t is not None)
        return next(norms) if first.requires_grad is False and first.grad is None else real(tensors)

    monkeypatch.setattr(schedule, "_norm", norm)
    cfg = f_run(env, tmp_path, "fresh", SCAN)
    run_dir = run_scheduled(cfg, env, run_id="spiked")
    assert json.loads((run_dir / "terminal.json").read_text())["stop"]["reason"] == "gradient_spike"
    assert not (run_dir / "line_scan.jsonl").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["line_scan"]["evaluated"] == [] and "skipped" in summary["line_scan"]


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
    ("+study5.role=fresh", SCAN, "tracking.endpoint_checkpoint=/x"),
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
    (("+study5.role=control", "+train.order_seed=3", SCAN), "line_scan.points"),
    (("+study5.role=control", "+train.order_seed=true"), "train.order_seed"),
    (("+study5.role=fresh", "+eval.controls.line_scan.points=[0.5,1]"), "line_scan.points"),
    (("+study5.role=fresh", "+eval.controls.line_scan.points=[0,0.5]"), "line_scan.points"),
    (("+study5.role=fresh", "+eval.controls.line_scan.points=[0.5,0.5]"), "line_scan.points"),
    (("+study5.role=fresh", "+eval.controls.line_scan.points=[]"), "line_scan.points"),
    (("+study5.role=fresh", SCAN, "+eval.k_only=true"), "K-only"),
    (("+study5.role=fresh", "optimizer=adamw", "optimizer.lr=1e-3"), "adam_frozen"),
    (("+train.order_seed=3",), "study-5 role"),
    ((SCAN,), "study-5 role"),
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
    for extra in ("+train.order_seed=3", "+optimizer.preconditioner.source_seed=1", SCAN,
                  "+study5.role=fresh"):  # fmt: skip
        with pytest.raises(ProtocolError, match="study-5|study3"):
            load_run_config("experiments/study3", ["study3.role=engineering", *base, extra])
    load_run_config("experiments/study3", ["study3.role=engineering", *base,
                                           "tracking.endpoint_checkpoint=/x"])  # fmt: skip
