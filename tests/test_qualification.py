"""Study 3's numerical qualification: the harness and bridge at step 25, its failure kinds and
verdicts (docs/study3-preregistration.md, "Numerical qualification", "Failure routes",
"Amendment 2").

Tiny K-only launcher runs with the endpoint checkpoint, on the CPU; the harness's parts run on
them. CPU bitwise agreement is a test here, not an acceptance condition across processes on a GPU.
"""

import copy
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from conftest import tiny_model, tiny_tokenizer
from omegaconf import OmegaConf
from test_fa_control_diagnostic import forbid_outcomes
from test_schedule import ARMS, arm_config, scheduled
from test_schedule import env as env  # noqa: F401  (the fixture)
from test_train import pass_path_check, prepare, prompt_sets

from zero_field_refusal import config, runtime, schedule
from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal import qualification as q
from zero_field_refusal.config import load_config
from zero_field_refusal.models import install_neutral_generation_config
from zero_field_refusal.runtime import Environment
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.tensor_hash import file_sha256

CPU = torch.device("cpu")
REGISTERED = load_config("experiments/qualification")
# The tiny world's seeds: candidate 1's numerical seeds hold the tiny runs' seeds 0 and 1
SETTINGS = OmegaConf.merge(REGISTERED, {"seeds": {"numerical": {1: [0, 1, 2]}}})
SCOPE = q.QualificationScope.of(SETTINGS)
FA = q.QualificationCase(0, "adam_frozen", 1e-3, 1, False, batch=4)
ADAMW = q.QualificationCase(1, "adamw", 1e-3, 1, False, batch=4)
LAUNCHER = ("tracking.provenance_hashes=true", "+eval.k_only=true")
CONFIRMATION = q.confirmation_config  # experiments/study3's, which no tiny run can agree with


CODE_ENVIRONMENT = q.code_environment_problems  # the real check, before the fixture below


@pytest.fixture(scope="module", autouse=True)
def harness_code_environment():
    """The tiny launcher runs are made by this process from this checkout, which may hold
    uncommitted work under test: the code and environment check is tested on its own
    (``test_the_launcher_run_needs_the_harness_code_and_environment``)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q, "code_environment_problems", lambda run_dir: [])
        yield


@pytest.fixture(scope="module", autouse=True)
def tiny_confirmation(tmp_path_factory):
    """The tiny runs' own confirmation config ("Config agreement"): the tiny schedule at the
    case's arm, rate, seed, batch and steps, as ``arm_config`` makes the launcher runs. Module
    scope, so that the module's own fixtures run under it too."""
    base = tmp_path_factory.mktemp("confirmation")

    def tiny(case):
        return scheduled(base, f"train.steps={case.steps}", "schedule.evaluate_at=[25]",
                         *ARMS[case.optimizer], f"optimizer.lr={case.lr!r}", f"seed={case.seed}",
                         f"train.batch_size={case.batch}", policy="short")  # fmt: skip

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q, "confirmation_config", tiny)
        yield


def launcher_run(env, base, arm, run_id, *overrides, seed=0):
    """A K-only launcher run of the bridge: its checkpoint right after update 25."""
    cfg = arm_config(env, base, arm, f"tracking.endpoint_checkpoint={base / 'launcher'}",
                     *LAUNCHER, f"seed={seed}", *overrides)  # fmt: skip
    return run_scheduled(cfg, env, run_id=run_id,
                         numerics_setup=runtime.configure_numerics(cfg, CPU))  # fmt: skip


def spiked(monkeypatch, module=schedule, name="_norm"):
    """The gradient norms scripted to fire the ×10 rule at step 3 (then 1.0); the weight check
    stays real."""
    norms = iter([1.0, 1.0, 1.0, 50.0])
    real = getattr(module, name)

    def norm(tensors):
        first = next(t for t in tensors if t is not None)
        if first.requires_grad is False and first.grad is None:
            return next(norms, 1.0)
        return real(tensors)

    monkeypatch.setattr(module, name, norm)


def build_world(base: Path):
    """The tiny launcher runs: FA on seed 0, AdamW on seed 1 (its seed's F v₀ prepared under the
    F group, as an AdamW case reads it for s1 and s2), and FA stopped by the gradient rule."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RUNS_DIR", str(base / "runs"))
        tokenizer = tiny_tokenizer()
        end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
        sets = prompt_sets()

        def new_model():
            model = tiny_model(tied=False, head_std=0.02)
            install_neutral_generation_config(model, end_ids, tokenizer.pad_token_id)
            return model

        env = Environment(tokenizer=tokenizer, end_ids=end_ids, pad_id=tokenizer.pad_token_id,
                          device=CPU, new_model=new_model, prompt_set=lambda name: sets[name],
                          data_root=base / "data")  # fmt: skip
        pass_path_check(env)
        runs = {"fa": launcher_run(env, base, "adam_frozen", "launcher-fa")}
        adamw = arm_config(env, base, "adamw", *LAUNCHER, "seed=1")
        prepare(env, q.setup_config(adamw, "adam_frozen"), "preconditioner", seed=1)
        runs["adamw"] = launcher_run(env, base, "adamw", "launcher-adamw", seed=1)
        spiked(patch)
        runs["spiked"] = launcher_run(env, base, "adam_frozen", "launcher-spiked")
    return env, runs


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("qualification"))


@pytest.fixture(scope="module")
def fa_parts(world, tmp_path_factory):
    """The FA case's baseline and rest, once."""
    env, runs = world
    base = tmp_path_factory.mktemp("fa-parts")
    out, ckpt = base / "out", base / "checkpoints"
    baseline = q.run_part(SETTINGS, env, FA, runs["fa"], out, part="baseline", attempt=1,
                          checkpoints=ckpt)  # fmt: skip
    rest = q.run_part(SETTINGS, env, FA, runs["fa"], out, part="rest", attempt=1,
                      checkpoints=ckpt)  # fmt: skip
    return out, ckpt, baseline, rest


def part(env, run_dir, out, ckpt, name, attempt=1, case=FA, **kwargs):
    return q.run_part(SETTINGS, env, case, run_dir, out, part=name, attempt=attempt,
                      checkpoints=ckpt, **kwargs)  # fmt: skip


def nothing_runs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a trajectory ran")

    monkeypatch.setattr(fcd, "twin_setup", forbidden)
    monkeypatch.setattr(fcd, "_control_setup", forbidden)


# ------------------------------------------------------------------ the scope and the cases


def test_the_scope_is_the_registered_one() -> None:
    s = q.QualificationScope.of(REGISTERED)
    assert (s.k_star(1), s.k_star(2)) == (6.0, 5.0)
    assert (s.bound(1), s.bound(2)) == pytest.approx((0.06, 0.05))
    assert s.norm == 5e-05 and s.gaussian == (("s3", 3), ("s4", 4))
    assert (s.survival_share, s.rounding_scale) == (0.5, 10.0)
    assert s.warning(0.0, 1) == pytest.approx(6.0 * 0.001)  # the floor
    assert s.warning(0.01, 2) == pytest.approx(5.0 * 0.03)
    assert s.seeds(1) == (10200, 10201, 10202) and s.seeds(2) == (10210, 10211, 10212)
    assert s.engineering_seeds == (10000, 10001, 10002, 10003)
    assert (s.control_arm, s.control_steps) == ("adam_frozen", 25)
    assert s.control_bound(1) == pytest.approx(0.06)
    assert s.spec == "study 3, numerical qualification (docs/study3-preregistration.md)"
    with pytest.raises(q.QualificationError):
        s.k_star(3)
    with pytest.raises(q.QualificationError):
        q.QualificationScope.of(OmegaConf.merge(REGISTERED, {"gaussian": {"p3": 3, "p4": 4}}))


def test_the_thirteen_comparisons() -> None:
    assert q.COMPARISONS == (
        "launcher~baseline", "launcher~repeat", "baseline~repeat", "launcher~reference",
        "baseline~reference", "launcher~s1", "baseline~s1", "launcher~s2", "baseline~s2",
        "launcher~s3", "baseline~s3", "launcher~s4", "baseline~s4",
    )  # fmt: skip


def test_case_names_and_paths() -> None:
    case = q.QualificationCase(10200, "adam_frozen", 1.83e-05, 1, False)
    assert case.name == "s10200/adam_frozen-1.83e-05" and (case.batch, case.steps) == (20, 25)
    assert case.attempt_path("rest", 2) == (
        "c1/numerical/s10200/adam_frozen-1.83e-05/rest-attempt-2.json"
    )
    engineering = q.QualificationCase(10000, "adamw", 2.11e-05, 1, True)
    assert engineering.attempt_path("baseline", 1).startswith("c1/engineering/s10000/")


# --------------------------------------------------------------------- the launcher run


def test_a_conforming_launcher_run(world) -> None:
    _, runs = world
    assert q.launcher_problems(runs["fa"], FA) == (None, [])
    assert q.launcher_problems(runs["adamw"], ADAMW) == (None, [])


def test_a_gradient_spike_is_scientific_and_nothing_runs(world, tmp_path, monkeypatch) -> None:
    env, runs = world
    kind, reasons = q.launcher_problems(runs["spiked"], FA)
    assert kind == "scientific" and len(reasons) == 1
    assert reasons[0].startswith("instability: the launcher run stopped by 'gradient_spike' "
                                 "after 3 steps")  # fmt: skip
    nothing_runs(monkeypatch)
    record = part(env, runs["spiked"], tmp_path / "out", tmp_path / "ckpt", "baseline")
    assert record["status"] == "not_run" and record["kind"] == "scientific"
    assert record["reasons"] == reasons and "trajectories" not in record
    assert json.loads((tmp_path / "out" / FA.attempt_path("baseline", 1)).read_text()) == record
    verdict = q.case_verdict(record, None, SCOPE)
    assert (verdict["state"], verdict["kind"]) == ("failing", "scientific")
    assert verdict["reasons"] == [f"baseline: {reasons[0]}"]  # no rest is needed


def test_identity_comes_before_an_instability_stop(world, tmp_path) -> None:
    """A run stopped by the gradient rule that departs from its case is technical."""
    _, runs = world
    for other in (q.QualificationCase(0, "adam_frozen", 2e-3, 1, False, batch=4),
                  q.QualificationCase(5, "adam_frozen", 1e-3, 1, False, batch=4),
                  q.QualificationCase(0, "adam_frozen", 1e-3, 1, False)):  # fmt: skip
        kind, reasons = q.launcher_problems(runs["spiked"], other)
        assert kind == "technical" and all("does not conform" in r for r in reasons), other
    copied = tmp_path / "spiked"
    shutil.copytree(runs["spiked"], copied)
    meta = json.loads((copied / "meta.json").read_text())
    del meta["inputs"]["numerics_setup"]
    (copied / "meta.json").write_text(json.dumps(meta))
    kind, reasons = q.launcher_problems(copied, FA)
    assert kind == "technical" and "numerics setup" in reasons[0]


def test_a_launcher_run_that_did_not_complete_is_technical(world, tmp_path) -> None:
    _, runs = world
    for name, change in (("terminal.json", None), ("summary.json", {"steps": 25})):
        copied = tmp_path / name
        shutil.copytree(runs["fa"], copied)
        if change is None:
            (copied / name).unlink()
        else:
            summary = json.loads((copied / name).read_text())
            summary["terminal"] = {**summary["terminal"], **change, "evaluations": []}
            (copied / name).write_text(json.dumps(summary))
        kind, reasons = q.launcher_problems(copied, FA)
        assert kind == "technical" and "terminal record" in " ".join(reasons), name
    copied = tmp_path / "unevaluated"
    shutil.copytree(runs["fa"], copied)
    for name in ("terminal.json", "summary.json"):
        record = json.loads((copied / name).read_text())
        terminal = record if name == "terminal.json" else record["terminal"]
        terminal["evaluations"] = []
        (copied / name).write_text(json.dumps(record))
    expected = ("technical", ["the launcher run has no step-25 evaluation"])
    assert q.launcher_problems(copied, FA) == expected


def test_a_corrupted_checkpoint_is_technical_before_any_run(env, tmp_path, monkeypatch) -> None:
    run_dir = launcher_run(env, tmp_path, "adam_frozen", "launcher-corrupted")
    record = json.loads((run_dir / "summary.json").read_text())["endpoint_checkpoint"]
    path = Path(record["path"])
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01  # one byte flipped
    path.write_bytes(bytes(data))
    kind, reasons = q.launcher_problems(run_dir, FA)
    assert kind == "technical" and reasons == [f"the checkpoint {path}'s sha256 is not the "
                                               "recorded one"]  # fmt: skip
    nothing_runs(monkeypatch)
    out = part(env, run_dir, tmp_path / "out", tmp_path / "ckpt", "baseline")
    assert (out["status"], out["kind"], out["reasons"]) == ("not_run", "technical", reasons)
    path.unlink()
    assert q.launcher_problems(run_dir, FA)[1] == [f"the checkpoint {path} is missing"]


def test_numerics_are_set_first(world, tmp_path, monkeypatch) -> None:
    env, runs = world
    calls = []
    configure, check = runtime.configure_numerics, q.prechecks.require_path_check

    def numerics(cfg, device):
        calls.append("numerics")
        return configure(cfg, device)

    def path_check(*args):
        calls.append("path check")
        return check(*args)

    real, given = q.launcher_problems, []

    def problems(*args, **kwargs):
        calls.append("launcher")
        given.append(kwargs["numerics"])
        return real(*args, **kwargs)

    monkeypatch.setattr(q.runtime, "configure_numerics", numerics)
    monkeypatch.setattr(q.prechecks, "require_path_check", path_check)
    monkeypatch.setattr(q, "launcher_problems", problems)
    nothing_runs(monkeypatch)
    record = part(env, runs["fa"], tmp_path / "out", tmp_path / "ckpt", "baseline")
    assert calls == ["numerics", "launcher", "path check"]  # the launcher check reads files only
    setup = configure(fcd.run_config(runs["fa"]), CPU)
    assert record["numerics_setup"] == setup and given == [setup]
    assert record["status"] == "failed" and "a trajectory ran" in record["reason"]
    calls.clear()
    part(env, runs["spiked"], tmp_path / "out", tmp_path / "ckpt", "baseline", attempt=2)
    assert calls == ["numerics", "launcher"]  # nothing runs, so no path check is needed


def edited(run_dir, target, name, change):
    """A copy of a launcher run with one record changed."""
    shutil.copytree(run_dir, target)
    if name == "config.yaml":
        cfg = OmegaConf.load(target / name)
        change(cfg)
        OmegaConf.save(cfg, target / name)
    else:
        record = json.loads((target / name).read_text())
        change(record)
        (target / name).write_text(json.dumps(record))
    return target


def test_the_launcher_run_must_agree_with_the_confirmation_config(world, tmp_path) -> None:
    _, runs = world
    for source in ("fa", "spiked"):  # an instability stop does not hide the departure
        copied = edited(runs[source], tmp_path / source, "config.yaml",
                        lambda cfg: cfg.execution.update(micro_batch_size=2))  # fmt: skip
        kind, reasons = q.launcher_problems(copied, FA)
        assert kind == "technical", source
        assert reasons == ["the launcher run does not conform: execution departs from the "
                           "confirmation config's"]  # fmt: skip


def test_the_registered_confirmation_config_agrees_with_the_k_only_runs(monkeypatch) -> None:
    monkeypatch.setattr(q, "confirmation_config", CONFIRMATION)
    case = q.QualificationCase(10200, "adam_frozen", 1.83e-05, 1, False)
    reference = q.confirmation_config(case)
    assert reference.study3.role == "confirmation" and reference.train.batch_size == 20
    assert (reference.optimizer.name, reference.optimizer.lr, reference.seed) == (
        "adam_frozen", 1.83e-05, 10200)  # fmt: skip
    for arm in ("adamw", "adam_frozen", "adamw_prewarmed"):
        k_only = load_config("experiments/study3_k_only", [
            "study3.role=numerical", f"optimizer={arm}", "optimizer.lr=1.83e-05", "seed=10200",
            "train.batch_size=20", "tracking.endpoint_checkpoint=/x"])  # fmt: skip
        mine = q.QualificationCase(10200, arm, 1.83e-05, 1, False)
        assert q._agreement_problems(k_only, mine) == [], arm
        k_only.execution.micro_batch_size = 2
        assert q._agreement_problems(k_only, mine) == [
            "execution departs from the confirmation config's"]  # fmt: skip


def test_the_launcher_run_needs_the_harness_code_and_environment(world, tmp_path, monkeypatch):
    """A launcher run made at a commit whose GPU code is not this checkout's, from a dirty
    checkout, or in another environment (packages, GPU, Python, image) is technical before
    anything runs or is classified: here even a run its instability rule stopped."""
    _, runs = world
    head = q.code_identity.gpu_code_identity()["commit"]
    monkeypatch.setattr(q, "git_state", lambda *a, **k: {"commit": head, "dirty": False})
    here = fcd.current_environment()
    environment = {"hardware": {"gpus": here["hardware.gpus"], "driver": here["hardware.driver"],
                                "torch_cuda": here["hardware.torch_cuda"]},
                   "python": here["python"], "packages": here["packages"]}  # fmt: skip

    def with_meta(name, **change):
        copied = tmp_path / name
        shutil.copytree(runs["spiked"], copied)
        meta = json.loads((copied / "meta.json").read_text())
        meta.update({"git": {"commit": head, "dirty": False}, **copy.deepcopy(environment)})
        meta.update(change)
        (copied / "meta.json").write_text(json.dumps(meta))
        return copied

    assert CODE_ENVIRONMENT(with_meta("same")) == []
    old_code = with_meta("old-code", git={"commit": "028b953", "dirty": False})
    assert any(p.startswith("code: ") and "GPU code differs" in p
               for p in CODE_ENVIRONMENT(old_code))  # fmt: skip
    dirty = with_meta("dirty", git={"commit": head, "dirty": True})
    assert any("not a clean checkout" in p for p in CODE_ENVIRONMENT(dirty))
    packages = {**environment["packages"], "torch": "0.0.0"}
    other = with_meta("other-packages", packages=packages)
    assert any(p.startswith("environment: packages.torch") for p in CODE_ENVIRONMENT(other))
    image = with_meta("other-image", launch={"container_image": "another:image"})
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    assert any("container_image" in p for p in CODE_ENVIRONMENT(image))
    monkeypatch.setattr(q, "code_environment_problems", CODE_ENVIRONMENT)
    kind, reasons = q.launcher_problems(old_code, FA)  # stopped by its instability rule
    assert kind == "technical" and any("code: " in r for r in reasons), reasons
    assert q.launcher_problems(with_meta("again"), FA)[0] == "scientific"


def test_the_launcher_run_must_have_this_processs_numerics(world, tmp_path) -> None:
    _, runs = world
    setup = runtime.configure_numerics(fcd.run_config(runs["fa"]), CPU)
    assert q.launcher_problems(runs["fa"], FA, numerics=setup) == (None, [])
    for source in ("fa", "spiked"):
        copied = edited(runs[source], tmp_path / source, "meta.json",
                        lambda meta: meta["inputs"]["numerics_setup"].update(
                            matmul_precision="high"))  # fmt: skip
        kind, reasons = q.launcher_problems(copied, FA, numerics=setup)
        assert kind == "technical" and "is not this process's" in reasons[0], source


def test_a_launcher_run_without_the_v0_backends_is_refused_and_not_bridged(world, tmp_path) -> None:
    """Old code's run, whose numerics setup lacks the v₀ policy, against a part with it: the
    launcher check refuses it, and the bridge's own equality compares the key."""
    _, runs = world
    setup = runtime.configure_numerics(fcd.run_config(runs["fa"]), CPU)
    old = edited(runs["fa"], tmp_path / "old", "meta.json",
                 lambda meta: meta["inputs"]["numerics_setup"].pop("v0_sdpa_backends"))  # fmt: skip
    kind, reasons = q.launcher_problems(old, FA, numerics=setup)
    assert kind == "technical" and "is not this process's" in reasons[0]
    rows = [{"step": t, "tokens": 1, "denominator": 1} for t in range(25)]
    facts = {"rows": rows, "training_stream": {"path": "p"}, "v0_sha256": None,
             "theta0_trainable_sha256": "t", "names_sha256": "n", "optimizer": {"o": 1},
             "numerics_setup": setup}  # fmt: skip
    harness = {k: v for k, v in facts.items() if k != "rows"}
    assert q.bridge_checks(facts, rows, harness)["numerics_setup"] is True
    before = {k: v for k, v in setup.items() if k != "v0_sdpa_backends"}
    assert q.bridge_checks({**facts, "numerics_setup": before}, rows, harness)[
        "numerics_setup"] is False  # fmt: skip


def test_the_v0_hash_is_one_across_the_launcher_and_the_harness(world) -> None:
    """The hash ``schedule``'s path recorded in the launcher run against the harness's estimate."""
    env, runs = world
    cfg = fcd.run_config(runs["fa"])
    model, trainable = fcd._model(env, cfg)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    v0, _ = fcd.second_moment(cfg, env, model, params)
    recorded = json.loads((runs["fa"] / "meta.json").read_text())["inputs"]["v0_sha256"]
    assert fcd.tensors_sha256(list(zip(names, v0, strict=True))) == recorded


def test_the_public_names_keep_their_private_aliases() -> None:
    assert config._apply_override is config.apply_override
    assert schedule._publish is schedule.publish


# ------------------------------------------------------------------------- the harness


def test_the_baseline_of_a_conforming_run(world, fa_parts) -> None:
    _, runs = world
    out, ckpt, baseline, _ = fa_parts
    assert baseline["status"] == "complete", baseline.get("reason")
    assert baseline["comparisons"]["launcher~baseline"]["D_twin"] == 0.0  # CPU: the same run
    assert baseline["bridge"] and all(baseline["bridge"].values()), baseline["bridge"]
    run = baseline["trajectories"]["baseline"]
    assert run["steps"] == 25 and run["stop"] is None and run["bitwise_rows"] is True
    meta = json.loads((runs["fa"] / "meta.json").read_text())
    assert baseline["numerics_setup"] == meta["inputs"]["numerics_setup"]
    assert baseline["v0_sha256"] == meta["inputs"]["v0_sha256"] and baseline["v0_unchanged"]
    saved = baseline["checkpoint"]
    assert saved["path"] == str(ckpt / "harness-baseline.pt") and saved["step"] == 25
    assert saved["sha256"] == file_sha256(ckpt / "harness-baseline.pt")
    assert saved["names_sha256"] == meta["inputs"]["endpoint_checkpoint"]["names_sha256"]
    on_disk = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    assert on_disk == baseline
    provenance = baseline["provenance"]
    assert provenance["spec"] == SETTINGS.spec
    assert provenance["settings"] == q.settings_record(SETTINGS)
    assert len(provenance["gpu_code"]["sha256"]) == 64
    assert set(provenance["host_memory"]) >= {"peak_rss_gb", "available_gb"}


def test_the_rest(world, fa_parts) -> None:
    _, runs = world
    out, ckpt, baseline, rest = fa_parts
    assert rest["status"] == "complete", rest.get("reason")
    assert set(rest["trajectories"]) == set(q.REST)
    assert set(rest["comparisons"]) == set(q.COMPARISONS) - {"launcher~baseline"}
    for target in ("launcher", "baseline"):  # the repeat is the run, bit for bit, on the CPU
        assert rest["comparisons"][f"{target}~repeat"]["D_twin"] == 0.0
    assert rest["reference_mode"] == fcd.FLOAT64_FULL and rest["fallback_reason"] is None
    for name in ("s1", "s2", "s3", "s4"):
        shift = rest["trajectories"][name]
        assert shift["rounding"][0]["intended_norm"] == pytest.approx(5e-05, rel=1e-6), name
        assert shift["survived"] is True and shift["initial_difference"]["D_twin"] > 0, name
        assert rest["comparisons"][f"baseline~{name}"]["D_twin"] > 0, name
    meta = json.loads((runs["fa"] / "meta.json").read_text())
    assert rest["v0_sha256"] == meta["inputs"]["v0_sha256"] and rest["v0_unchanged"] is True
    assert all(rest["bridge"].values()), rest["bridge"]
    path = out / FA.attempt_path("baseline", 1)
    assert rest["baseline_record"] == {"path": str(path), "sha256": file_sha256(path)}
    assert rest["baseline_record"]["sha256"] == q.record_sha256(baseline)
    assert rest["baseline_checkpoint_sha256"] == baseline["checkpoint"]["sha256"]
    verdict = q.case_verdict(baseline, rest, SCOPE, e_eng=0.0)
    assert list(verdict["comparisons"]) == list(q.COMPARISONS) and verdict["complete"]
    assert all(fcd._number(c["D"]) for c in verdict["comparisons"].values())
    assert verdict["comparisons"]["baseline~repeat"] == {"D": 0.0, "within": True}
    assert verdict["warning"] is False and verdict["tau"] == pytest.approx(0.006)
    assert (verdict["state"], verdict["kind"], verdict["reasons"]) == ("qualified", None, [])
    assert verdict["fallback"] is False and verdict["launcher"] == baseline["launcher"]["identity"]


def rest_copy(fa_parts, tmp_path):
    """The FA case's records in a folder of their own (its checkpoints read where they are)."""
    out, ckpt, baseline, _ = fa_parts
    shutil.copytree(out, tmp_path / "out")
    return tmp_path / "out", ckpt, baseline


def test_a_shift_beyond_the_bound_does_not_stop_the_rest(world, fa_parts, tmp_path,
                                                         monkeypatch) -> None:  # fmt: skip
    """Amendment 8: the battery goes on past a shift's result, so all four are attempted."""
    env, runs = world
    out, ckpt, _ = rest_copy(fa_parts, tmp_path)
    real, ran = q.run_trajectory, []

    def trajectory(name, *args, **kwargs):
        ran.append(name)
        run, endpoint = real(name, *args, **kwargs)
        if name == "s1":
            endpoint = {label: {**m, "D_twin": 0.5} for label, m in endpoint.items()}
        return run, endpoint

    monkeypatch.setattr(q, "run_trajectory", trajectory)
    record = part(env, runs["fa"], out, ckpt, "rest", attempt=2)
    assert ran == list(q.REST)
    assert (record["status"], record.get("kind")) == ("complete", None)
    assert record["comparisons"]["baseline~s1"]["D_twin"] == 0.5
    assert set(record["trajectories"]) == set(q.REST)


def test_a_non_finite_shift_leaves_the_next_shift_as_it_was(world, fa_parts, tmp_path,
                                                            monkeypatch) -> None:  # fmt: skip
    """A shift that ends non-finite: the next one starts from θ₀ and ends as it would have."""
    env, runs = world
    _, _, _, rest = fa_parts
    out, ckpt, _ = rest_copy(fa_parts, tmp_path)
    real = q.run_trajectory

    def trajectory(name, cfg, twins, *args, **kwargs):
        run, endpoint = real(name, cfg, twins, *args, **kwargs)
        if name == "s1":
            for _, parameter in twins.trainable:
                parameter.data.fill_(math.nan)
            return {**run, "stop": "non-finite loss or gradient at step 3"}, {}
        return run, endpoint

    monkeypatch.setattr(q, "run_trajectory", trajectory)
    record = part(env, runs["fa"], out, ckpt, "rest", attempt=2)
    assert (record["status"], record.get("kind")) == ("complete", None), record.get("reasons")
    for name in ("s2", "s3", "s4"):
        assert (record["trajectories"][name]["endpoint_sha256"]
                == rest["trajectories"][name]["endpoint_sha256"]), name  # fmt: skip


def test_a_scientific_trajectory_result_stops_the_rest_at_once(world, fa_parts, tmp_path,
                                                              monkeypatch) -> None:  # fmt: skip
    """Each trajectory's result is published, bound to its inputs, before the next starts; a
    valid scientific failure (here the baseline's D₂₅ against the reference) publishes the part
    at once, decided, and runs no further trajectory."""
    env, runs = world
    out, ckpt, baseline = rest_copy(fa_parts, tmp_path)
    real, ran = q.run_trajectory, []

    def trajectory(name, *args, **kwargs):
        ran.append(name)
        run, endpoint = real(name, *args, **kwargs)
        if name == "reference":
            endpoint = {**endpoint, "baseline": {**endpoint["baseline"], "D_twin": 0.5}}
        return run, endpoint

    monkeypatch.setattr(q, "run_trajectory", trajectory)
    record = part(env, runs["fa"], out, ckpt, "rest", attempt=2)
    assert ran == ["repeat", "reference"]
    assert (record["status"], record["kind"]) == ("decided", "scientific")
    assert record["reasons"][0].startswith("baseline~reference: D₂₅ 0.5")
    assert list(record["trajectories"]) == ["repeat", "reference"]
    folder = out / FA.attempt_path("rest", 2).rsplit("/", 1)[0]
    files = sorted(f.name for f in folder.glob("rest-attempt-2.trajectory-*.json"))
    assert files == ["rest-attempt-2.trajectory-reference.json",
                     "rest-attempt-2.trajectory-repeat.json"]  # fmt: skip
    result = json.loads((folder / files[0]).read_text())
    baseline_file = out / FA.attempt_path("baseline", 1)
    assert result["baseline_record"]["sha256"] == file_sha256(baseline_file)
    assert result["baseline_checkpoint_sha256"] == baseline["checkpoint"]["sha256"]
    assert result["launcher"]["identity"] == baseline["launcher"]["identity"]
    assert result["launcher_checkpoint_sha256"] == baseline["launcher"]["identity"]["checkpoint"][
        "sha256"]  # fmt: skip
    assert result["provenance"]["settings"] == q.settings_record(SETTINGS)
    assert all(result["bridge"].values()) and result["v0_unchanged"] is True
    v = q.case_verdict(json.loads(baseline_file.read_text()), record, SCOPE)
    assert (v["state"], v["kind"]) == ("failing", "scientific")


def nan_gradient(patch, at: int) -> None:
    """The ``at``-th gradient norm ``run_twin`` reads is NaN (the weight norms stay real): the
    trajectory stops there, non-finite, as a real divergence stops it."""
    real, count = fcd.grad_norm, [0]

    def norm(tensors):
        first = next(t for t in tensors if t is not None)
        if first.requires_grad is False and first.grad is None:  # gradients, not weights
            count[0] += 1
            if count[0] == at:
                return math.nan
        return real(tensors)

    patch.setattr(fcd, "grad_norm", norm)


def test_a_non_finite_baseline_is_scientific(world, tmp_path, monkeypatch) -> None:
    """A NaN at a step of the harness baseline stops it there (``run_twin``): its bridge holds on
    the launcher run's rows of the same length, and the case is a scientific failure."""
    env, runs = world
    nan_gradient(monkeypatch, at=4)
    record = part(env, runs["fa"], tmp_path / "out", tmp_path / "ckpt", "baseline")
    run = record["trajectories"]["baseline"]
    assert run["stop"].startswith("non-finite loss or gradient at step"), run["stop"]
    assert record["status"] == "complete", record.get("reason")
    assert 0 < len(run["rows"]) < 25 and record["checkpoint"] is None
    assert all(record["bridge"].values()), record["bridge"]
    v = q.case_verdict(record, None, SCOPE)
    assert (v["state"], v["kind"]) == ("failing", "scientific"), v["reasons"]
    assert v["reasons"][0].startswith("instability: the harness baseline")


def test_a_non_finite_repeat_is_scientific(world, fa_parts, tmp_path, monkeypatch) -> None:
    """A NaN at a step of the repeat: the part stops there, decided, and the case fails."""
    env, runs = world
    out, ckpt, baseline = rest_copy(fa_parts, tmp_path)
    real = q.run_trajectory

    def trajectory(name, *args, **kwargs):
        if name != "repeat":
            return real(name, *args, **kwargs)
        with pytest.MonkeyPatch.context() as patch:
            nan_gradient(patch, at=6)
            return real(name, *args, **kwargs)

    monkeypatch.setattr(q, "run_trajectory", trajectory)
    record = part(env, runs["fa"], out, ckpt, "rest", attempt=2)
    repeat = record["trajectories"]["repeat"]
    assert repeat["stop"].startswith("non-finite loss or gradient at step")
    assert all(record["bridge"].values()), record["bridge"]
    assert (record["status"], record["kind"]) == ("decided", "scientific")
    assert list(record["trajectories"]) == ["repeat"]
    v = q.case_verdict(json.loads((out / FA.attempt_path("baseline", 1)).read_text()), record,
                       SCOPE)  # fmt: skip
    assert (v["state"], v["kind"]) == ("failing", "scientific"), v["reasons"]
    assert any(r.startswith("instability: the harness repeat") for r in v["reasons"])


def test_a_stopped_trajectory_is_bridged_on_the_launchers_prefix() -> None:
    rows = [{"step": t, "tokens": 10 + t, "denominator": 5} for t in range(25)]
    full = {"path": "p", "count": 100, "chunks": ["a", "b", "c", "d"], "sha256": "s"}
    facts = {"rows": rows, "training_stream": full, "v0_sha256": None,
             "theta0_trainable_sha256": "t", "names_sha256": "n", "optimizer": {"o": 1},
             "numerics_setup": {"x": 1}}  # fmt: skip
    setup = {k: v for k, v in facts.items() if k != "rows"}
    read = {"path": "p", "count": 28, "chunks": ["a", "b"], "sha256": "t"}  # 7 steps of 4
    assert all(q.bridge_checks(facts, rows[:7], {**setup, "training_stream": read},
                               stopped=True).values())  # fmt: skip
    assert not q.bridge_checks(facts, rows[:7], setup)["steps"]  # not stopped: all 25 rows
    other = [*rows[:6], {**rows[6], "tokens": 0}]
    assert not q.bridge_checks(facts, other, setup, stopped=True)["tokens"]
    elsewhere = {**read, "chunks": ["a", "x"]}
    assert not q.bridge_checks(facts, rows[:7], {**setup, "training_stream": elsewhere},
                               stopped=True)["training_stream"]  # fmt: skip


def test_an_interruption_keeps_the_trajectory_results(world, fa_parts, tmp_path,
                                                      monkeypatch) -> None:  # fmt: skip
    """A part killed after its third trajectory leaves no record but the three results, durable
    and bound: as a record (``rest_from_trajectories``) they are read as any rest."""
    env, runs = world
    out, ckpt, _ = rest_copy(fa_parts, tmp_path)
    real = q.publish

    def publish(path, record):
        real(path, record)
        if path.name.endswith("trajectory-s1.json"):
            raise KeyboardInterrupt  # the process stops: no exception handler runs

    monkeypatch.setattr(q, "publish", publish)
    with pytest.raises(KeyboardInterrupt):
        part(env, runs["fa"], out, ckpt, "rest", attempt=2)
    assert not (out / FA.attempt_path("rest", 2)).exists()
    folder = out / FA.attempt_path("rest", 2).rsplit("/", 1)[0]
    results = [json.loads(f.read_text()) for f in folder.glob("rest-attempt-2.trajectory-*")]
    record = q.rest_from_trajectories(results)
    assert record["from_trajectories"] == ["repeat", "reference", "s1"]
    assert record["status"] == "interrupted" and all(record["bridge"].values())
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    v = q.case_verdict(baseline, record, SCOPE)
    assert v["kind"] is None and v["state"] == "incomplete"
    assert v["comparisons"]["baseline~s1"]["D"] is not None
    assert v["comparisons"]["baseline~s2"]["D"] is None
    mixed = [results[0], {**results[1], "baseline_checkpoint_sha256": "x" * 64}]
    assert q.case_verdict(baseline, q.rest_from_trajectories(mixed), SCOPE)["kind"] == "technical"


def test_each_trajectory_starts_afresh(world, fa_parts, monkeypatch) -> None:
    """The repeat after the reference and the shifts, and after a forced out-of-memory fallback
    of the reference, ends where a repeat run first ends, bit for bit on the CPU."""
    env, runs = world
    out, ckpt, _, first = fa_parts
    monkeypatch.setattr(q, "REST", ("reference", "s1", "s2", "s3", "s4", "repeat"))
    later = part(env, runs["fa"], out, ckpt, "rest", attempt=2)
    real = fcd.run_twin

    def oom(cfg, twins, *, precision="fp32", **kwargs):
        if precision == "float64":
            raise torch.OutOfMemoryError("CUDA out of memory (forced)")
        return real(cfg, twins, precision=precision, **kwargs)

    monkeypatch.setattr(fcd, "run_twin", oom)
    fallback = part(env, runs["fa"], out, ckpt, "rest", attempt=3)
    reference = fallback["trajectories"]["reference"]
    assert reference["mode"] == fcd.FLOAT64_ACCUMULATION
    assert reference["failed_attempt"]["error"].startswith("OutOfMemoryError: CUDA out of memory")
    assert fallback["fallback_reason"].startswith("OutOfMemoryError")
    endpoint = first["trajectories"]["repeat"]["endpoint_sha256"]
    for record in (later, fallback):
        assert record["status"] == "complete" and record["v0_unchanged"] is True
        assert record["trajectories"]["repeat"]["monitor"] == first["trajectories"]["repeat"][
            "monitor"]  # fmt: skip
        assert record["trajectories"]["repeat"]["instability"] is None
        assert record["trajectories"]["repeat"]["endpoint_sha256"] == endpoint
        assert record["comparisons"]["baseline~repeat"]["D_twin"] == 0.0
    for name in ("s1", "s2", "s3", "s4"):  # the shifts are the same whatever came before
        assert (later["trajectories"][name]["endpoint_sha256"]
                == first["trajectories"][name]["endpoint_sha256"])  # fmt: skip
    verdict = q.case_verdict(json.loads((out / FA.attempt_path("baseline", 1)).read_text()),
                             fallback, SCOPE)  # fmt: skip
    assert verdict["fallback"] is True and verdict["reference_mode"] == fcd.FLOAT64_ACCUMULATION


def test_the_fallback_is_for_out_of_memory_alone(world, fa_parts, monkeypatch) -> None:
    env, runs = world
    out, ckpt, baseline, _ = fa_parts
    real = fcd.run_twin

    def broken(cfg, twins, *, precision="fp32", **kwargs):
        if precision != "fp32":
            raise RuntimeError("an error that is not out of memory")
        return real(cfg, twins, precision=precision, **kwargs)

    monkeypatch.setattr(fcd, "run_twin", broken)
    record = part(env, runs["fa"], out, ckpt, "rest", attempt=4)
    assert record["status"] == "failed" and record["kind"] == "technical"
    assert "RuntimeError: an error that is not out of memory" in record["reason"]
    assert "reference" not in record["trajectories"]  # no fallback was taken
    assert record["trajectories"]["repeat"]["stop"] is None  # what ran before it is kept
    verdict = q.case_verdict(baseline, record, SCOPE)
    assert (verdict["state"], verdict["kind"]) == ("not_qualified", "technical")
    assert any("RuntimeError" in r for r in verdict["reasons"])


def test_an_adamw_case_reads_the_seeds_f_v0_for_its_shifts(world, tmp_path) -> None:
    env, runs = world
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    baseline = part(env, runs["adamw"], out, ckpt, "baseline", case=ADAMW)
    assert baseline["status"] == "complete", baseline.get("reason")
    assert baseline["comparisons"]["launcher~baseline"]["D_twin"] == 0.0
    assert all(baseline["bridge"].values()) and baseline["setup"]["v0_sha256"] is None
    assert baseline["setup"]["optimizer"]["name"] == "adamw"  # the run's own trajectories
    rest = part(env, runs["adamw"], out, ckpt, "rest", case=ADAMW)
    assert rest["status"] == "complete", rest.get("reason")
    assert rest["directions"]["direction_norms"]["p1"] > 0 and rest["directions"]["epsilon"] == 1e-8
    assert rest["comparisons"]["baseline~repeat"]["D_twin"] == 0.0
    assert q.case_verdict(baseline, rest, SCOPE)["complete"]


def test_diagnostic_steps_are_the_engineering_stages(world, tmp_path) -> None:
    env, runs = world
    with pytest.raises(q.QualificationError):
        part(env, runs["fa"], tmp_path / "out", tmp_path / "ckpt", "baseline",
             diagnostic_steps=[5])  # fmt: skip
    case = q.QualificationCase(0, "adam_frozen", 1e-3, 1, True, batch=4)
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    baseline = part(env, runs["fa"], out, ckpt, "baseline", case=case, diagnostic_steps=[10, 5])
    assert baseline["diagnostic_steps"] == [5, 10]
    assert sorted(p.name for p in (ckpt / "baseline-states").iterdir()) == [
        "A-step-10.pt", "A-step-5.pt"]  # fmt: skip
    rest = part(env, runs["fa"], out, ckpt, "rest", case=case, diagnostic_steps=[5, 10])
    assert rest["status"] == "complete", rest.get("reason")
    for name in q.REST:
        series = rest["trajectories"][name]["D_twin"]
        assert [r["step"] for r in series] == [5, 10], name
    assert rest["trajectories"]["repeat"]["max_D_twin"] == 0.0
    summary = q.engineering_summary([baseline, rest])
    assert summary["e_eng"] == 0.0 and summary["repeats"] == 1
    assert set(summary["path_maxima"]) == {"repeat", "reference", "shift"}
    assert summary["path_maxima"]["repeat"] == 0.0
    assert summary["margins"]["launcher~repeat"] == math.inf
    assert summary["margins"]["baseline~shift"] == pytest.approx(
        0.06 / max(rest["comparisons"][f"baseline~{s}"]["D_twin"] for s in ("s1", "s2", "s3",
                                                                            "s4")))  # fmt: skip
    other = part(env, runs["fa"], out, ckpt, "rest", attempt=2, case=case, diagnostic_steps=[5])
    assert other["status"] == "not_run" and "diagnostic steps" in other["reasons"][0]


def test_the_rest_needs_its_baseline(world, tmp_path) -> None:
    env, runs = world
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    record = part(env, runs["fa"], out, ckpt, "rest")
    assert (record["status"], record["kind"]) == ("not_run", "technical")
    assert record["reasons"] == ["no readable baseline record"]
    baseline = part(env, runs["fa"], out, ckpt, "baseline")
    (ckpt / "harness-baseline.pt").write_bytes(b"overwritten")
    record = part(env, runs["fa"], out, ckpt, "rest", attempt=2)
    assert record["status"] == "not_run" and "sha256" in record["reasons"][0]
    assert q.case_verdict(baseline, record, SCOPE)["kind"] == "technical"


def test_the_same_path_control(world, tmp_path) -> None:
    env, runs = world
    with pytest.raises(q.QualificationError):  # the control is FA's, on the first seed
        part(env, runs["adamw"], tmp_path / "out", tmp_path / "ckpt", "control", case=ADAMW)
    record = part(env, runs["fa"], tmp_path / "out", tmp_path / "ckpt", "control")
    assert record["status"] == "complete", record.get("reason")
    assert all(check["equal"] for check in record["shared_state"].values())
    assert [r["step"] for r in record["control"]["scoring"]] == [25]
    verdict = q.control_verdict(record, SCOPE, 1)
    assert verdict["state"] == "accepted" and verdict["kind"] is None, verdict["reasons"]
    assert verdict["K_scoring"] <= 0.06
    spiked = part(env, runs["spiked"], tmp_path / "out", tmp_path / "ckpt", "control", attempt=2)
    assert (spiked["status"], spiked["kind"]) == ("not_run", "scientific")
    assert q.control_verdict(spiked, SCOPE, 1)["state"] == "failing"


def test_nothing_reads_a_harmful_prompt_set_or_generates(world, tmp_path, monkeypatch) -> None:
    env, runs = world
    guarded, requested = forbid_outcomes(monkeypatch, env)
    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    for name in ("baseline", "rest", "control"):
        record = part(guarded, runs["fa"], out, ckpt, name)
        assert record["status"] == "complete", (name, record.get("reason"))
    assert requested and not [name for name in requested if "harmful" in name]


# ------------------------------------------------------------- case verdicts on synthetic records

SETTINGS_RECORD = q.settings_record(REGISTERED)
CODE = {"commit": "c" * 40, "files": 1, "sha256": "g" * 64}
CASE = q.QualificationCase(10200, "adam_frozen", 1.83e-05, 1, False)
CHECKPOINT = {"path": "/x.pt", "sha256": "c" * 64, "step": 25, "names_sha256": "n"}
IDENTITY = {"run": "study3/r1", "run_id": "r1", "config_hash": "h", "sha256": "s",
            "checkpoint": CHECKPOINT}  # fmt: skip


def synthetic(case=CASE, d=0.001, identity=IDENTITY):
    """A qualified case's records, each D₂₅ ``d``."""
    provenance = {"settings": SETTINGS_RECORD, "gpu_code": CODE}
    launcher = {"kind": None, "reasons": [], "identity": identity}
    common = {"case": case.record(), "launcher": launcher, "provenance": provenance,
              "v0_unchanged": True,
              "bridge": dict.fromkeys(q.BRIDGE_KEYS, True)}  # fmt: skip
    baseline = {"part": "baseline", "status": "complete", **copy.deepcopy(common),
                "checkpoint": {"sha256": "b" * 64}, "trajectories": {"baseline": {"stop": None}},
                "comparisons": {"launcher~baseline": {"D_twin": d}}}  # fmt: skip
    rest = {"part": "rest", "status": "complete", **copy.deepcopy(common),
            "baseline_checkpoint_sha256": "b" * 64,
            "trajectories": {name: {"stop": None, **({"survived": True} if name[0] == "s" else {})}
                             for name in q.REST},
            "comparisons": {name: {"D_twin": d} for name in q.COMPARISONS[1:]}}  # fmt: skip
    return baseline, bind(rest, baseline)


def bind(rest, baseline):
    rest["baseline_record"] = {"path": "b.json", "sha256": q.record_sha256(baseline)}
    return rest


def verdict_of(change=None, *, e_eng=None):
    baseline, rest = synthetic()
    if change is not None:
        change(baseline, rest)
        bind(rest, baseline)
    return q.case_verdict(baseline, rest, SCOPE, e_eng=e_eng)


def test_a_qualified_case() -> None:
    v = verdict_of()
    assert (v["state"], v["kind"], v["reasons"]) == ("qualified", None, [])
    assert list(v["comparisons"]) == list(q.COMPARISONS) and len(v["comparisons"]) == 13
    assert all(c == {"D": 0.001, "within": True} for c in v["comparisons"].values())
    assert v["launcher"] == IDENTITY and v["gpu_code"] == CODE["sha256"]
    assert v["settings"] == SETTINGS_RECORD and v["warning"] is None
    at = verdict_of(lambda b, r: r["comparisons"]["baseline~reference"].update(D_twin=0.06))
    assert at["state"] == "qualified"  # D₂₅ ≤ 0.01·K*


def beyond(name, d=0.07):
    def change(baseline, rest):
        record = baseline if name == "launcher~baseline" else rest
        record["comparisons"][name]["D_twin"] = d

    return change


def stopped(name, stop="non-finite loss or gradient at step 7"):
    def change(baseline, rest):
        (baseline if name == "baseline" else rest)["trajectories"][name]["stop"] = stop

    return change


def launcher_instability(baseline, rest):
    reason = "instability: the launcher run stopped by 'gradient_spike'"
    baseline.update(status="not_run", kind="scientific", reasons=[reason])


def baseline_error(baseline, rest):
    baseline.update(status="failed", kind="technical", reasons=["error: OSError: disk full"])


def both(*changes):
    def change(baseline, rest):
        for c in changes:
            c(baseline, rest)

    return change


SCIENTIFIC_RULES = {
    "the launcher run's instability stop": (launcher_instability, "instability: the launcher"),
    "the baseline's instability stop": (stopped("baseline"), "instability: the harness baseline"),
    "the repeat's instability stop": (stopped("repeat"), "instability: the harness repeat"),
    "a non-finite reference": (stopped("reference"), "a non-finite loss, gradient or model in "
                               "the reference"),
    "baseline~reference beyond": (beyond("baseline~reference"), "baseline~reference: D₂₅ 0.07"),
    "scientific over technical": (both(beyond("baseline~reference"), beyond("launcher~repeat")),
                                  "baseline~reference: D₂₅ 0.07"),
    "an instability stop before an error": (both(stopped("baseline"), baseline_error),
                                            "instability: the harness baseline"),
}  # fmt: skip


@pytest.mark.parametrize("rule", list(SCIENTIFIC_RULES))
def test_each_scientific_rule(rule) -> None:
    change, reason = SCIENTIFIC_RULES[rule]
    v = verdict_of(change)
    assert (v["state"], v["kind"]) == ("failing", "scientific"), v["reasons"]
    assert any(r.startswith(reason) or r.startswith(f"baseline: {reason}") for r in v["reasons"])


def stopped_shift(baseline, rest):
    rest["trajectories"]["s3"]["stop"] = "non-finite loss or gradient at step 7"
    for target in ("launcher", "baseline"):
        del rest["comparisons"][f"{target}~s3"]


@pytest.mark.parametrize("change", [
    beyond("baseline~s2"), beyond("launcher~s3"),
    both(beyond("baseline~s4"), beyond("launcher~s4")), stopped_shift,
], ids=["against the baseline", "against the launcher alone", "both", "a non-finite shift"])  # fmt: skip
def test_a_shift_rejects_nothing_and_stays_shown(change) -> None:
    """Amendment 8: the four shifts are measured and reported; none of their results is a reason."""
    baseline, rest = synthetic()
    change(baseline, rest)
    bind(rest, baseline)
    v = q.case_verdict(baseline, rest, SCOPE)
    assert (v["state"], v["kind"], v["reasons"]) == ("qualified", None, [])
    assert v["complete"] is True
    assert v["comparisons"]["baseline~s2"]["D"] == rest["comparisons"]["baseline~s2"]["D_twin"]


def lost(baseline, rest):
    rest["trajectories"]["s1"]["survived"] = False


def test_a_shift_lost_to_rounding_rejects_nothing() -> None:
    v = verdict_of(lost)
    assert (v["state"], v["kind"], v["reasons"]) == ("qualified", None, [])


def test_a_rest_decided_on_a_shift_under_the_old_rule_keeps_its_verdict() -> None:
    """The engineering stage's four cases: the record's scientific kind stands, and the shifts it
    did not run stay unmeasured."""
    baseline, rest = synthetic()
    reason = "baseline~s2: D₂₅ 0.07 beyond the bound 0.06"
    rest.update(status="decided", kind="scientific", reasons=[reason])
    rest["comparisons"]["baseline~s2"]["D_twin"] = 0.07
    for name in ("s3", "s4"):
        del rest["trajectories"][name]
        for target in ("launcher", "baseline"):
            del rest["comparisons"][f"{target}~{name}"]
    bind(rest, baseline)
    v = q.case_verdict(baseline, rest, SCOPE)
    assert (v["state"], v["kind"]) == ("failing", "scientific")
    assert v["reasons"] == [f"rest: {reason}"] and v["complete"] is False
    assert v["comparisons"]["baseline~s3"] == {"D": None, "within": None}


def test_a_rest_interrupted_before_a_shift_ran_stays_incomplete() -> None:
    """A shift missing without a stop of its own is not measured: execution, not a result."""
    baseline, rest = synthetic()
    rest["status"] = "interrupted"
    del rest["trajectories"]["s4"]
    for target in ("launcher", "baseline"):
        del rest["comparisons"][f"{target}~s4"]
    bind(rest, baseline)
    v = q.case_verdict(baseline, rest, SCOPE)
    assert v["state"] != "qualified" and v["complete"] is False


def failed_bridge(where):
    def change(baseline, rest):
        (baseline if where == "baseline" else rest)["bridge"]["tokens"] = False

    return change


def nonfinite_d(baseline, rest):
    rest["comparisons"]["launcher~s2"]["D_twin"] = math.inf


def raised(baseline, rest):
    rest.update(status="failed", kind="technical", reasons=["error: RuntimeError: boom"])
    del rest["trajectories"]["s4"]


def v0_changed(baseline, rest):
    rest["v0_unchanged"] = False


def other_code(baseline, rest):
    rest["provenance"] = {**rest["provenance"], "gpu_code": {"sha256": "o" * 64}}


TECHNICAL_RULES = {
    "launcher~baseline beyond": (beyond("launcher~baseline"), "launcher~baseline: D₂₅"),
    "launcher~repeat beyond": (beyond("launcher~repeat"), "launcher~repeat: D₂₅"),
    "baseline~repeat beyond": (beyond("baseline~repeat"), "baseline~repeat: D₂₅"),
    "a launcher comparison alone beyond": (beyond("launcher~reference"),
                                           "launcher~reference: D₂₅ 0.07 beyond the bound 0.06, "
                                           "its baseline counterpart not"),
    "a non-finite D between finite models": (nonfinite_d, "launcher~s2: a non-finite D"),
    "a failed bridge check": (failed_bridge("baseline"), "the bridge failed: ['tokens']"),
    "a failed bridge check of the rest": (failed_bridge("rest"), "the bridge failed in the rest"),
    "an exception": (raised, "rest: error: RuntimeError: boom"),
    "v₀ written": (v0_changed, "v₀ changed during the rest part"),
    "parts under other GPU code": (other_code, "the parts were written under other settings"),
}  # fmt: skip


@pytest.mark.parametrize("rule", list(TECHNICAL_RULES))
def test_each_technical_rule(rule) -> None:
    change, reason = TECHNICAL_RULES[rule]
    v = verdict_of(change)
    assert (v["state"], v["kind"]) == ("not_qualified", "technical"), v["reasons"]
    assert any(r.startswith(reason) for r in v["reasons"]), v["reasons"]


def test_a_missing_part_is_technical() -> None:
    baseline, rest = synthetic()
    v = q.case_verdict(baseline, None, SCOPE)
    assert (v["state"], v["kind"], v["reasons"]) == ("not_qualified", "technical",
                                                      ["the rest part has no record"])  # fmt: skip
    assert v["comparisons"]["launcher~baseline"] == {"D": 0.001, "within": True}
    assert v["comparisons"]["baseline~reference"] == {"D": None, "within": None}
    v = q.case_verdict(None, rest, SCOPE)
    assert v["kind"] == "technical" and "the baseline part has no record" in v["reasons"]


def test_a_rest_of_another_baseline_is_refused() -> None:
    baseline, rest = synthetic()
    other, _ = synthetic(d=0.002)  # another baseline record of the same case
    v = q.case_verdict(other, rest, SCOPE)
    assert v["kind"] == "technical"
    assert "the rest part does not build on this baseline record and checkpoint" in v["reasons"]
    rest["baseline_checkpoint_sha256"] = "x" * 64
    assert q.case_verdict(baseline, rest, SCOPE)["kind"] == "technical"
    _, swapped = synthetic(case=q.QualificationCase(10201, "adam_frozen", 1.83e-05, 1, False))
    v = q.case_verdict(baseline, swapped, SCOPE)  # a rest of another case
    assert v["kind"] == "technical" and "the rest record is another case's" in v["reasons"]


def test_no_scientific_verdict_on_a_failed_identity_or_bridge_check() -> None:
    """The identity and bridge checks hold before any measurement is read: a false check (here
    the training stream, or θ₀'s hash) or a second Pod makes the case technical, whatever
    scientific evidence its records show, and no measurement of it is read."""

    def false(where, key):
        def change(baseline, rest):
            (baseline if where == "baseline" else rest)["bridge"][key] = False

        return change

    def spike(baseline, rest):
        baseline["trajectories"]["baseline"]["instability"] = {
            "reason": "gradient_spike", "step": 3, "detail": {"grad_norm": 50.0}}  # fmt: skip

    def pods(baseline, rest):
        baseline["provenance"]["pod_id"], rest["provenance"]["pod_id"] = "p1", "p2"

    def baseline_pods(baseline, rest):  # the baseline on another Pod than its launcher run
        baseline["launcher"]["pod_id"], baseline["provenance"]["pod_id"] = "p0", "p1"

    evidence = {
        "a false training-stream check and an excessive D": both(
            false("rest", "training_stream"), beyond("baseline~reference", 0.5)),
        "a false θ₀ hash and a gradient spike": both(
            false("baseline", "theta0_trainable_sha256"), spike),
        "a second Pod and an excessive D": both(pods, beyond("baseline~s2", 0.5)),
        "the baseline's second Pod and a gradient spike": both(baseline_pods, spike),
    }  # fmt: skip
    for name, change in evidence.items():
        v = verdict_of(change)
        assert (v["state"], v["kind"]) == ("not_qualified", "technical"), (name, v["reasons"])
        assert not any("instability" in r or "D₂₅" in r for r in v["reasons"]), name
        assert all(c["D"] is None for c in v["comparisons"].values()), name
    # Without the false check, the same evidence is scientific
    for change in (beyond("baseline~reference", 0.5), spike):
        assert verdict_of(change)["kind"] == "scientific"


def test_an_unbound_rest_never_hides_a_decided_baseline() -> None:
    """A baseline that is a scientific failure on bound, verified records decides the case: an
    unbound rest (another baseline record, other GPU code, a second Pod, a false check of its
    own) is listed as that record's technical problem, none of its measurements is read, and the
    case and its rate stay failing."""

    def spike(baseline, rest):
        baseline["trajectories"]["baseline"]["instability"] = {
            "reason": "gradient_spike", "step": 3, "detail": {"grad_norm": 50.0}}  # fmt: skip

    def other_baseline(baseline, rest):
        rest["baseline_record"]["sha256"] = "x" * 64

    def other_code(baseline, rest):
        rest["provenance"]["gpu_code"] = {"sha256": "o" * 64}

    def second_pod(baseline, rest):
        rest["provenance"]["pod_id"] = "p2"  # the baseline ran on p1

    def false_check(baseline, rest):
        rest["bridge"]["tokens"] = False

    problems = {other_baseline: "the rest part does not build on this baseline record",
                other_code: "the parts were written under other settings or GPU code",
                second_pod: "the case ran on more than one Pod",
                false_check: "the bridge failed in the rest part"}  # fmt: skip
    for change, problem in problems.items():
        baseline, rest = synthetic()
        spike(baseline, rest)
        baseline["provenance"]["pod_id"] = "p1"
        bind(rest, baseline)  # bound to this baseline record, unless the change unbinds it
        rest["comparisons"]["baseline~reference"]["D_twin"] = 0.5  # would be scientific
        change(baseline, rest)
        v = q.case_verdict(baseline, rest, SCOPE)
        assert (v["state"], v["kind"]) == ("failing", "scientific"), (change.__name__, v)
        assert v["reasons"][0].startswith("instability: the harness baseline")
        assert v["reasons"][1:] == [r for r in v["reasons"][1:] if r.startswith(problem)]
        assert any(r.startswith(problem) for r in v["reasons"]), (change.__name__, v["reasons"])
        assert not any("baseline~reference" in r for r in v["reasons"])  # nothing of it read
        assert all(c["D"] is None for n, c in v["comparisons"].items() if n != "launcher~baseline")
        out = rate([v], optimizer=CASE.optimizer)  # the registered seeds, CASE's rate
        assert out["state"] == "failing" and out["counted"] == [CASE.seed], change.__name__
    # Without the decided baseline, the same unbound rest makes the case technical
    baseline, rest = synthetic()
    rest["comparisons"]["baseline~reference"]["D_twin"] = 0.5
    other_baseline(baseline, rest)
    assert q.case_verdict(baseline, rest, SCOPE)["kind"] == "technical"


def test_the_bridge_checks_are_the_registered_set() -> None:
    """``bridge_checks`` produces exactly ``BRIDGE_KEYS``, the set the verdicts require."""
    rows = [{"step": t, "tokens": 1, "denominator": 1} for t in range(25)]
    facts = {"rows": rows, "training_stream": {"path": "p"}, "v0_sha256": None,
             "theta0_trainable_sha256": "t", "names_sha256": "n", "optimizer": {"o": 1},
             "numerics_setup": {"x": 1}}  # fmt: skip
    setup = {k: v for k, v in facts.items() if k != "rows"}
    checks = q.bridge_checks(facts, rows, setup)
    assert tuple(checks) == q.BRIDGE_KEYS and all(checks.values())
    assert set(q.BRIDGE_KEYS) == {"steps", "tokens", "denominator", "training_stream",
                                  "v0_sha256", "theta0_trainable_sha256", "names_sha256",
                                  "optimizer", "numerics_setup"}  # fmt: skip


@pytest.mark.parametrize("missing", ["all but steps", *q.BRIDGE_KEYS])
def test_an_incomplete_bridge_is_not_established(missing) -> None:
    """A bridge missing a registered check (``{"steps": True}`` alone, or any single key) is
    not established: the case is technical and no measurement of that record is read, even
    evidence that would be scientific."""
    for where in ("baseline", "rest"):
        baseline, rest = synthetic()
        record = baseline if where == "baseline" else rest
        if missing == "all but steps":
            record["bridge"] = {"steps": True}
        else:
            del record["bridge"][missing]
        rest["comparisons"]["baseline~reference"]["D_twin"] = 0.5  # would be scientific
        baseline["trajectories"]["baseline"]["instability"] = (
            {"reason": "gradient_spike", "step": 3, "detail": {}} if where == "baseline" else None
        )
        bind(rest, baseline)
        v = q.case_verdict(baseline, rest, SCOPE)
        assert (v["state"], v["kind"]) == ("not_qualified", "technical"), (missing, where)
        label = "the bridge failed" if where == "baseline" else "the bridge failed in the rest"
        assert any(r.startswith(label) and "(missing)" in r for r in v["reasons"]), v["reasons"]
        assert not any("instability" in r or "D₂₅" in r for r in v["reasons"])
        assert all(c["D"] is None for n, c in v["comparisons"].items()
                   if where == "baseline" or n != "launcher~baseline")  # fmt: skip


def test_a_part_that_failed_before_its_bridge_has_nothing_read() -> None:
    """A baseline whose part failed before its bridge checks were established: its trajectory's
    instability is not read (the case is technical); with the checks recorded, it is."""

    def failed(baseline, rest):
        baseline["trajectories"]["baseline"]["instability"] = {
            "reason": "gradient_spike",
            "step": 3,
            "detail": {},
        }
        baseline.update(status="failed", kind="technical", reasons=["error: OSError: full"])
        del baseline["bridge"]  # fmt: skip

    v = verdict_of(failed)
    assert (v["state"], v["kind"]) == ("not_qualified", "technical")
    assert v["reasons"] == ["baseline: error: OSError: full"]


def test_one_pod_per_case() -> None:
    """The launcher run's and the parts' pod ids, where present, are one; a difference is
    technical."""

    def pods(launcher, baseline_pod, rest_pod):
        def change(baseline, rest):
            for record, pod in ((baseline, baseline_pod), (rest, rest_pod)):
                record["launcher"]["pod_id"] = launcher
                record["provenance"]["pod_id"] = pod

        return change

    assert verdict_of(pods("p1", "p1", "p1"))["state"] == "qualified"
    assert verdict_of(pods(None, None, "p1"))["state"] == "qualified"
    for case in (pods("p1", "p1", "p2"), pods(None, "p1", "p2"), pods("p0", "p1", "p1")):
        v = verdict_of(case)
        assert (v["state"], v["kind"]) == ("not_qualified", "technical")
        assert any(r.startswith("the case ran on more than one Pod") for r in v["reasons"])
    record = control_record()
    record["launcher"]["pod_id"], record["provenance"]["pod_id"] = "p1", "p2"
    v = q.control_verdict(record, SCOPE, 1)
    assert (v["state"], v["kind"]) == ("not_qualified", "technical")
    record["provenance"]["pod_id"] = "p1"
    assert q.control_verdict(record, SCOPE, 1)["state"] == "accepted"


def test_the_warning_is_a_diagnosis_only() -> None:
    v = verdict_of(lambda b, r: r["comparisons"]["baseline~repeat"].update(D_twin=0.01),
                   e_eng=0.0003)  # fmt: skip
    assert v["tau"] == pytest.approx(6.0 * 0.001) and v["warning"] is True
    assert v["state"] == "qualified"  # the pairs are measured and judged as ever
    assert verdict_of(e_eng=0.001)["warning"] is False  # τ = 6·0.003 = 0.018 > 0.001


# ------------------------------------------------------------------------------ rate verdicts

SEEDS = (10200, 10201, 10202)


def case_verdicts(states=("qualified",) * 3, seeds=SEEDS, *, optimizer="adamw"):
    out = []
    for seed, state in zip(seeds, states, strict=True):
        case = q.QualificationCase(seed, optimizer, 1.83e-05, 1, False)
        identity = {**IDENTITY, "run_id": f"r{seed}"}
        baseline, rest = synthetic(case, identity=identity)
        changes = {"failing": beyond("baseline~reference"), "not_qualified": beyond(
            "baseline~repeat"), "incomplete": None, "qualified": None}  # fmt: skip
        if changes[state] is not None:
            changes[state](baseline, rest)
            bind(rest, baseline)
        verdict = q.case_verdict(baseline, rest, SCOPE)
        if state == "incomplete":  # no rule fired, but not every comparison is measured
            verdict = {**verdict, "state": "incomplete", "complete": False}
        assert verdict["state"] == state
        out.append(verdict)
    return out


def rate(cases, *, optimizer="adamw", code=CODE, policy=SETTINGS_RECORD, **kwargs):
    return q.rate_verdict(cases, scope=q.QualificationScope.of(REGISTERED), candidate=1,
                          optimizer=optimizer, lr=1.83e-05, code=code, policy=policy,
                          **kwargs)  # fmt: skip


def test_the_rate_states_in_order() -> None:
    assert rate(case_verdicts())["state"] == "qualified"
    assert rate(case_verdicts(("qualified", "incomplete", "qualified")))["state"] == "incomplete"
    # failing > not_qualified > qualified > incomplete
    for states, expected in (
        (("not_qualified", "failing", "incomplete"), "failing"),
        (("qualified", "failing", "qualified"), "failing"),
        (("incomplete", "not_qualified", "qualified"), "not_qualified"),
        (("qualified", "qualified", "not_qualified"), "not_qualified"),
    ):
        assert rate(case_verdicts(states))["state"] == expected, states


def test_an_empty_or_partial_list_is_incomplete() -> None:
    assert rate([])["state"] == "incomplete"
    partial = rate(case_verdicts()[:2])
    assert partial["state"] == "incomplete"
    assert "no counted case of seed 10202" in partial["reasons"]
    failing = rate(case_verdicts(("failing",), seeds=(10200,)))
    assert failing["state"] == "failing"  # scientific evidence needs no other seed


def test_a_duplicate_seed_is_incomplete() -> None:
    cases = case_verdicts()
    out = rate([*cases, cases[0]])
    assert out["state"] == "incomplete" and "more than one case of seed 10200" in out["reasons"]


def test_a_case_of_another_seed_or_rate_does_not_count() -> None:
    cases = case_verdicts()
    stray = case_verdicts(("qualified",), seeds=(10210,))  # candidate 2's seed
    out = rate([*cases, *stray])  # an extra stray case: never qualified beside it
    assert out["state"] == "not_qualified" and out["counted"] == [10200, 10201, 10202]
    assert out["not_counted"] == [
        {
            "case": stray[0]["case"]["name"],
            "record_sha256": stray[0]["record_sha256"],
            "why": "another candidate, arm, rate or seed",
            "kind": None,
        }
    ]
    assert rate(cases, optimizer="adamw_prewarmed")["state"] == "not_qualified"
    voided = {stray[0]["record_sha256"]: "superseded"}
    assert rate([*cases, *stray], voided=voided)["state"] == "qualified"


def test_swapped_parts_are_technical() -> None:
    cases = case_verdicts()
    baseline, _ = synthetic(q.QualificationCase(10202, "adamw", 1.83e-05, 1, False),
                            identity={**IDENTITY, "run_id": "r10202"})  # fmt: skip
    _, other = synthetic(q.QualificationCase(10201, "adamw", 1.83e-05, 1, False))
    swapped = q.case_verdict(baseline, other, SCOPE)
    assert rate([*cases[:2], swapped])["state"] == "not_qualified"


def test_a_stale_policy_does_not_count() -> None:
    cases = case_verdicts()
    stale = {**SETTINGS_RECORD, "norm": 1e-4}
    out = rate(cases, policy=stale)  # every case shown, with why it does not count
    assert out["state"] == "not_qualified" and out["counted"] == []
    assert [entry["why"] for entry in out["not_counted"]] == ["other settings"] * 3
    assert rate(cases, code={**CODE, "sha256": "o" * 64})["state"] == "not_qualified"
    assert rate(cases, code=None)["state"] == "not_qualified"
    expected = {s: {**IDENTITY, "run_id": f"r{s}"} for s in SEEDS}
    assert rate(cases, launchers=expected)["state"] == "qualified"
    assert rate(cases, launchers={**expected, 10201: IDENTITY})["state"] == "not_qualified"


def test_a_rate_whose_launcher_runs_are_not_all_located_is_at_most_incomplete() -> None:
    cases = case_verdicts()
    expected = {s: {**IDENTITY, "run_id": f"r{s}"} for s in SEEDS}
    out = rate(cases, launchers={**expected, 10201: None})
    assert out["state"] == "incomplete" and out["counted"] == list(SEEDS)
    assert "the launcher run of seed 10201 is not located" in out["reasons"]
    assert rate(cases, launchers={10200: expected[10200]})["state"] == "incomplete"
    failing = case_verdicts(("qualified", "failing", "qualified"))
    assert rate(failing, launchers={**expected, 10201: None})["state"] == "failing"
    technical = case_verdicts(("qualified", "not_qualified", "qualified"))
    assert rate(technical, launchers={**expected, 10201: None})["state"] == "not_qualified"


def stale_case(state, seed=10200):
    """A case of this rate written under other GPU code."""
    (verdict,) = case_verdicts((state,), seeds=(seed,))
    return {**verdict, "gpu_code": "o" * 64}


def test_a_stale_scientific_case_keeps_the_rate_failing_until_voided() -> None:
    fresh = case_verdicts()
    old = stale_case("failing")
    out = rate([old, *fresh])
    assert out["state"] == "failing" and out["counted"] == list(SEEDS)
    assert out["not_counted"][0]["kind"] == "scientific"
    voided = rate([old, *fresh], voided={old["record_sha256"]: "superseded by an engineering "
                                         "cycle"})  # fmt: skip
    assert voided["state"] == "qualified"
    assert voided["voided"] == [{"case": old["case"]["name"], "record_sha256": old["record_sha256"],
                                 "reason": "superseded by an engineering cycle"}]  # fmt: skip
    assert rate([stale_case("qualified"), *fresh])["state"] == "not_qualified"


def control_record(k=1e-12, *, diverged=False, equal=True, case=None, code=CODE,
                   identity=None):  # fmt: skip
    case = case or q.QualificationCase(10200, "adam_frozen", 1.83e-05, 1, False)
    control = {"steps": 25, "stop": None, "diverged": diverged,
               "rows": [{"step": 0, "loss": 1.0, "grad_norm": 1.0}], "monitor": [],
               "scoring": [{"step": 25, "K_scoring": k}]}  # fmt: skip
    identity = identity or {**IDENTITY, "run_id": "r10200"}  # the first seed's case's run
    return {"part": "control", "status": "complete", "case": case.record(), "control": control,
            "shared_state": {"v0_sha256": {"equal": equal}},
            "launcher": {"kind": None, "reasons": [], "identity": identity},
            "provenance": {"settings": SETTINGS_RECORD, "gpu_code": code}}  # fmt: skip


def test_control_verdicts() -> None:
    s = q.QualificationScope.of(REGISTERED)
    assert q.control_verdict(control_record(), s, 1)["state"] == "accepted"
    assert q.control_verdict(control_record(0.06), s, 1)["state"] == "accepted"  # ≤ 0.01·K*
    for record in (None, control_record(0.0601), control_record(math.nan),
                   control_record(diverged=True), control_record(equal=False)):  # fmt: skip
        v = q.control_verdict(record, s, 1)
        assert (v["state"], v["kind"]) == ("not_qualified", "technical"), record
    assert q.control_verdict(control_record(), s, 2)["state"] == "not_qualified"


def test_an_fa_rate_needs_its_accepted_control() -> None:
    s = q.QualificationScope.of(REGISTERED)
    cases = case_verdicts(optimizer="adam_frozen")
    assert rate(cases, optimizer="adam_frozen")["state"] == "not_qualified"  # missing
    accepted = q.control_verdict(control_record(), s, 1)
    assert rate(cases, optimizer="adam_frozen", control=accepted)["state"] == "qualified"
    rejected = q.control_verdict(control_record(0.1), s, 1)
    assert rate(cases, optimizer="adam_frozen", control=rejected)["state"] == "not_qualified"
    other = q.control_verdict(control_record(case=q.QualificationCase(
        10200, "adam_frozen", 1.5e-05, 1, False)), s, 1)  # fmt: skip
    assert rate(cases, optimizer="adam_frozen", control=other)["state"] == "not_qualified"
    stale = q.control_verdict(control_record(code={**CODE, "sha256": "o" * 64}), s, 1)
    assert stale["state"] == "accepted"  # on its own; not under the expected GPU code
    out = rate(cases, optimizer="adam_frozen", control=stale)
    assert out["state"] == "not_qualified"
    assert "control: the control was written under other GPU code" in out["reasons"]
    elsewhere = q.control_verdict(control_record(identity={**IDENTITY, "run_id": "other"}), s, 1)
    out = rate(cases, optimizer="adam_frozen", control=elsewhere)
    assert out["state"] == "not_qualified"
    assert "control: the control read another launcher run than the first seed's case" in (
        out["reasons"])  # fmt: skip
    voided = rate(cases, optimizer="adam_frozen", control=accepted,
                  voided={accepted["record_sha256"]: "superseded"})  # fmt: skip
    assert voided["state"] == "not_qualified"  # a voided control is a missing one
    assert rate(case_verdicts())["control"] is None  # AdamW has none


def scientific_control(**kwargs):
    """A control record whose launcher run an instability rule stopped."""
    return {**control_record(**kwargs), "status": "not_run", "kind": "scientific",
            "reasons": ["instability: the launcher run stopped by 'gradient_spike'"]}  # fmt: skip


def test_another_rates_scientific_control_is_not_this_rates_evidence() -> None:
    s = q.QualificationScope.of(REGISTERED)
    cases = case_verdicts(optimizer="adam_frozen")
    other = q.control_verdict(scientific_control(case=q.QualificationCase(
        10200, "adam_frozen", 1.5e-05, 1, False)), s, 1)  # fmt: skip
    assert other["state"] == "failing"  # on its own, for its own rate
    out = rate(cases, optimizer="adam_frozen", control=other)
    assert (out["state"], out["control"]["kind"]) == ("not_qualified", "technical")
    assert "control: the control is not this rate's" in out["reasons"]
    stale = q.control_verdict(scientific_control(code={**CODE, "sha256": "o" * 64}), s, 1)
    assert rate(cases, optimizer="adam_frozen", control=stale)["state"] == "failing"


def test_a_superseded_control_counts_as_a_superseded_case_does() -> None:
    """A superseded control record that no restart voids keeps the rate from ``qualified``;
    a scientific one makes it ``failing``; another rate's is never this rate's science."""
    s = q.QualificationScope.of(REGISTERED)
    cases = case_verdicts(optimizer="adam_frozen")
    accepted = q.control_verdict(control_record(), s, 1)
    technical = q.control_verdict(control_record(0.1), s, 1)
    out = rate(cases, optimizer="adam_frozen", control=accepted, superseded_controls=[technical])
    assert out["state"] == "not_qualified"
    assert out["not_counted"] == [{"case": "control", "record_sha256": technical["record_sha256"],
                                   "why": "a superseded same-path control",
                                   "kind": "technical"}]  # fmt: skip
    science = q.control_verdict(scientific_control(), s, 1)
    out = rate(cases, optimizer="adam_frozen", control=accepted, superseded_controls=[science])
    assert out["state"] == "failing"
    voided = rate(cases, optimizer="adam_frozen", control=accepted, superseded_controls=[science],
                  voided={science["record_sha256"]: "an engineering cycle"})  # fmt: skip
    assert voided["state"] == "qualified"
    elsewhere = q.control_verdict(scientific_control(case=q.QualificationCase(
        10200, "adam_frozen", 1.5e-05, 1, False)), s, 1)  # fmt: skip
    out = rate(cases, optimizer="adam_frozen", control=accepted, superseded_controls=[elsewhere])
    assert out["state"] == "not_qualified" and out["not_counted"][0]["kind"] is None


def test_a_voided_superseded_control_neither_counts_nor_blocks() -> None:
    s = q.QualificationScope.of(REGISTERED)
    cases = case_verdicts(optimizer="adam_frozen")
    accepted = q.control_verdict(control_record(), s, 1)
    old = q.control_verdict(control_record(0.1), s, 1)  # technical, superseded by a restart
    out = rate(cases, optimizer="adam_frozen", control=accepted, superseded_controls=[old],
               voided={old["record_sha256"]: "the Pod stopped"})  # fmt: skip
    assert out["state"] == "qualified" and out["not_counted"] == []
    assert out["voided"] == [{"case": "control", "record_sha256": old["record_sha256"],
                              "reason": "the Pod stopped"}]  # fmt: skip


def test_the_engineering_summary_on_synthetic_records() -> None:
    baseline, rest = synthetic(d=0.003)
    for record in (baseline, rest):
        record.update(K_star=6.0, bound=0.06)
    rest["comparisons"]["baseline~repeat"]["D_twin"] = 0.012
    rest["comparisons"]["launcher~s3"]["D_twin"] = 0.0
    rest["trajectories"]["s2"]["D_twin"] = [{"step": 5, "D_twin": 0.12},
                                            {"step": 10, "D_twin": 0.6}]  # fmt: skip
    summary = q.engineering_summary([baseline, rest])
    assert summary["e_eng"] == pytest.approx(0.002) and summary["repeats"] == 1
    assert summary["path_maxima"] == {"shift": pytest.approx(0.1)}
    assert summary["margins"]["baseline~repeat"] == pytest.approx(5.0)
    assert summary["margins"]["launcher~shift"] == pytest.approx(20.0)  # 0.06 / 0.003; 0 is ∞
    assert q.engineering_summary([])["e_eng"] is None


def test_a_rest_not_bound_to_its_baseline_is_read_for_nothing() -> None:
    """A rest under other GPU code, of another launcher run or not built on this baseline is
    technical, and none of its measurements counts, a scientific one included."""

    def other_launcher(baseline, rest):
        rest["launcher"] = {**rest["launcher"], "identity": {**IDENTITY, "run_id": "other"}}

    def unbound(baseline, rest):
        baseline["checkpoint"] = {"sha256": "x" * 64}

    for problem in (other_code, other_launcher, unbound):
        baseline, rest = synthetic()
        rest["comparisons"]["baseline~reference"]["D_twin"] = 0.5  # would be scientific
        rest["trajectories"]["s2"]["stop"] = "non-finite loss or gradient at step 3"
        problem(baseline, rest)
        if problem is not unbound:
            bind(rest, baseline)
        v = q.case_verdict(baseline, rest, SCOPE)
        assert (v["state"], v["kind"]) == ("not_qualified", "technical"), problem.__name__
        assert not any("non-finite" in r or "baseline~reference" in r for r in v["reasons"])
        assert all(c["D"] is None for n, c in v["comparisons"].items() if n != "launcher~baseline")


# ------------------------------------------------- the declared instability rules, after the run


def rows(norms, *, loss=1.0):
    return [{"step": i, "loss": loss, "grad_norm": n} for i, n in enumerate(norms)]


@pytest.fixture(scope="module")
def run_cfg(world):
    _, runs = world
    cfg = fcd.run_config(runs["fa"])
    rules = schedule.Schedule.from_config(cfg)
    assert (rules.grad_factor, rules.grad_window, rules.k_max) == (10.0, 20, 16.0)
    return cfg


def test_the_gradient_rule_after_the_run(run_cfg) -> None:
    def fired(norms, **kwargs):
        return q.harness_instability(run_cfg, {"rows": rows(norms, **kwargs), "stop": None})

    assert fired([1.0] * 25) is None
    assert fired([5.0, 50.0]) is None  # exactly 10× the step-0 norm: not above
    assert fired([5.0, 50.1]) == {
        "reason": "gradient_spike",
        "step": 1,
        "detail": {"grad_norm": 50.1, "reference_median": 5.0},
    }
    # The reference is the last 20 norms: 80 < 10 × 9, though above 10 × the median of all 40
    assert fired([1.0] * 20 + [9.0] * 20 + [80.0]) is None
    assert fired([1.0] * 20 + [9.0] * 20 + [91.0])["step"] == 40
    assert fired([1.0, math.nan])["reason"] == "nonfinite_gradient"
    assert fired([1.0, 1.0], loss=math.inf)["reason"] == "nonfinite_loss"


def test_the_monitor_rule_and_the_weights_after_the_run(run_cfg) -> None:
    every = int(run_cfg.eval.monitor.every)

    def fired(k, stop=None):
        run = {"rows": rows([1.0] * 25), "stop": stop,
               "monitor": [{"step": t, "K_monitor": 1.0 if t != 2 * every else k}
                           for t in range(every, 26, every)]}  # fmt: skip
        return q.harness_instability(run_cfg, run)

    assert fired(16.0) is None  # at the bound: not above
    assert fired(16.5) == {
        "reason": "k_above_max",
        "step": 2 * every,
        "detail": {"K_monitor": 16.5, "k_max": 16.0},
    }
    assert fired(math.nan)["reason"] == "nonfinite_k"
    stopped_ = fired(1.0, stop="non-finite weights after step 24")
    assert stopped_["reason"] == "nonfinite_weight" and stopped_["step"] is None


def test_the_rules_hold_for_the_unshifted_trajectories_alone() -> None:
    spike = {"reason": "gradient_spike", "step": 3, "detail": {"grad_norm": 50.0}}
    for name in ("baseline", "repeat"):
        v = verdict_of(lambda b, r, n=name: (b if n == "baseline" else r)["trajectories"][n]
                       .update(instability=spike))  # fmt: skip
        assert (v["state"], v["kind"]) == ("failing", "scientific"), name
        assert v["reasons"][0].startswith(f"instability: the harness {name}: 'gradient_spike' "
                                          "at step 3")  # fmt: skip

    def finite_spike(baseline, rest):  # a ×10 spike and a K above 16 in a shift and the reference
        for name in ("s1", "reference"):
            rest["trajectories"][name].update(
                rows=rows([1.0, 1.0, 1.0, 50.0]), instability=spike,
                monitor=[{"step": 5, "K_monitor": 20.0}])  # fmt: skip

    v = verdict_of(finite_spike)
    assert (v["state"], v["kind"], v["reasons"]) == ("qualified", None, [])


def test_a_real_gradient_spike_of_the_harness_baseline(world, tmp_path, monkeypatch) -> None:
    env, runs = world
    spiked(monkeypatch, fcd, "grad_norm")  # the norms run_twin records, not its training
    record = part(env, runs["fa"], tmp_path / "out", tmp_path / "ckpt", "baseline")
    run = record["trajectories"]["baseline"]
    assert run["stop"] is None and run["steps"] == 25
    fired = run["instability"]  # the setup's residual took the first scripted norm
    assert (fired["reason"], fired["detail"]) == (
        "gradient_spike", {"grad_norm": 50.0, "reference_median": 1.0})  # fmt: skip
    assert run["rows"][fired["step"]]["grad_norm"] == 50.0
    launcher = [r["step"] for r in fcd.read_jsonl(runs["fa"] / "monitor.jsonl")]
    assert [r["step"] for r in run["monitor"]] == launcher  # the launcher's cadence
    v = q.case_verdict(record, None, SCOPE)
    assert (v["state"], v["kind"]) == ("failing", "scientific")
    assert v["reasons"][0].startswith("instability: the harness baseline: 'gradient_spike'")


# ---------------------------------------------- rate verdicts from the records run_part writes


def rate_of(records, case):
    verdict = q.case_verdict(*records, SCOPE)
    return q.rate_verdict([verdict], scope=SCOPE, candidate=1, optimizer=case.optimizer,
                          lr=case.lr, code=q.code_identity.gpu_code_identity(),
                          policy=q.settings_record(SETTINGS), batch=4,
                          control=q.control_verdict(None, SCOPE, 1))  # fmt: skip


def test_a_spiked_launcher_run_makes_its_rate_failing(world, tmp_path) -> None:
    env, runs = world
    record = part(env, runs["spiked"], tmp_path / "out", tmp_path / "ckpt", "baseline")
    assert record["launcher"]["identity"]["run_id"] == runs["spiked"].name
    assert record["launcher"]["identity"]["checkpoint"] is None
    out = rate_of((record, None), FA)
    assert out["state"] == "failing" and out["counted"] == [0], out["reasons"]


def test_a_record_is_published_once_whatever_its_launcher_identity(world, tmp_path, monkeypatch):
    """The infrastructure reason is in the record's one publish; an exception reading the
    launcher run's identity leaves it missing, with the error, and the record is published."""
    env, runs = world
    published, real = [], q.publish

    def spy(path, record):
        published.append((path, copy.deepcopy(dict(record))))
        real(path, record)

    def unreadable(run_dir):
        raise RuntimeError("an unreadable identity")

    monkeypatch.setattr(q, "publish", spy)
    monkeypatch.setattr(q, "launcher_identity", unreadable)
    nothing_runs(monkeypatch)
    reason = {"reason": "the Pod stopped", "previous": [{"attempt": 1, "state": "interrupted"}]}
    record = part(env, runs["fa"], tmp_path / "out", tmp_path / "ckpt", "baseline", attempt=2,
                  infrastructure=reason)  # fmt: skip
    assert [path for path, _ in published] == [tmp_path / "out" / FA.attempt_path("baseline", 2)]
    assert published[0][1]["infrastructure"] == reason == record["infrastructure"]
    launcher = record["launcher"]
    assert launcher["identity"] is None and launcher["pod_id"] is None
    assert launcher["identity_error"] == "RuntimeError: an unreadable identity"
    assert record["status"] == "failed" and record["kind"] == "technical"


@pytest.mark.parametrize("raises", [None, "numerics", "check"])
def test_a_record_classifies_its_launcher_run_once_the_check_returned(
    world, tmp_path, monkeypatch, raises
) -> None:
    """``launcher.classified`` turns true only once ``launcher_problems`` returned, with its
    kind (here the spiked run's declared stop, scientific); an exception in the numerics setup,
    or in the check itself, publishes a technical record that names the run without classifying
    it (``classified`` false, ``kind`` None)."""
    env, runs = world

    def broken(*args, **kwargs):
        raise RuntimeError("it raised")

    nothing_runs(monkeypatch)
    if raises == "numerics":
        monkeypatch.setattr(q.runtime, "configure_numerics", broken)
    elif raises == "check":
        monkeypatch.setattr(q, "launcher_problems", broken)
    record = part(env, runs["spiked"], tmp_path / "out", tmp_path / "ckpt", "baseline")
    launcher = record["launcher"]
    assert launcher["identity"]["run_id"] == runs["spiked"].name
    if raises is None:
        assert (launcher["classified"], launcher["kind"], record["kind"]) == (
            True, "scientific", "scientific")  # fmt: skip
    else:
        assert (launcher["classified"], launcher["kind"], launcher["reasons"]) == (False, None, [])
        assert (record["status"], record["kind"]) == ("failed", "technical")


def test_publish_syncs_the_directory_after_the_rename(tmp_path, monkeypatch) -> None:
    import os
    import stat

    synced, real = [], os.fsync

    def fsync(fd):
        synced.append((stat.S_ISDIR(os.fstat(fd).st_mode), (tmp_path / "r.json").exists()))
        real(fd)

    monkeypatch.setattr(q.os, "fsync", fsync)
    q.publish(tmp_path / "r.json", {"a": 1})
    assert synced[-1] == (True, True)  # the directory, once the record is in place
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1}


def test_a_launcher_run_of_another_case_makes_its_rate_not_qualified(world, tmp_path) -> None:
    env, runs = world
    other = q.QualificationCase(0, "adam_frozen", 2e-3, 1, False, batch=4)
    record = part(env, runs["spiked"], tmp_path / "out", tmp_path / "ckpt", "baseline",
                  case=other)  # fmt: skip
    assert (record["status"], record["kind"]) == ("not_run", "technical")
    assert rate_of((record, None), other)["state"] == "not_qualified"
    empty = tmp_path / "empty-run"
    empty.mkdir()
    failed = part(env, empty, tmp_path / "out", tmp_path / "ckpt", "baseline", attempt=2)
    assert failed["status"] == "failed" and failed["launcher"]["identity"]["run_id"] == "empty-run"
    assert rate_of((failed, None), FA)["state"] == "not_qualified"
