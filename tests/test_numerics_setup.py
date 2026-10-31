"""The one numerics setup of every entry point (docs/study3-preregistration.md, "Numerical
qualification" → "The bridge's conditions") and the stage-0 record's fingerprint of it."""

import json
import runpy
import types
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from test_train import BOUND, config

from zero_field_refusal import checks, prechecks, protocol, runtime, schedule, train
from zero_field_refusal import config as config_module
from zero_field_refusal.prechecks import (
    CheckFailed,
    environment_fingerprint,
    record_path,
    require_path_check,
    write_check_record,
)
from zero_field_refusal.runtime import configure_numerics

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# The fingerprint of study 2's protocol-2 path-field record of Qwen2.5-1.5B on the L40S, as
# written by stage 0 (DATA_DIR/checks/Qwen--Qwen2.5-1.5B-Instruct/path-field-4e1d7d928d6c.json).
STUDY2_RECORD = {
    "attn_implementation": "sdpa", "check_version": 3, "cuda": "13.0", "cudnn": 92400,
    "device": "cuda", "device_name": "NVIDIA L40S", "dtype": "float32",
    "generation_batch_size": 32, "micro_batch_size": 4, "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "protocol_version": 2, "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "sampler_version": 1, "tf32": False, "torch": "2.14.1", "transformers": "5.18.0",
}  # fmt: skip


@pytest.fixture(autouse=True)
def numerics_restored(monkeypatch):
    """Every test starts without a cuBLAS setting, as a fresh process before its CUDA work (other
    tests may have used the GPU), and leaves torch's numerics as it found them."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32,
             torch.are_deterministic_algorithms_enabled())  # fmt: skip
    yield
    torch.set_float32_matmul_precision(saved[0])  # first: it sets the matmul TF32 flag too
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[1:3]
    torch.use_deterministic_algorithms(saved[3])


def test_configure_numerics_returns_what_it_set() -> None:
    out = configure_numerics(config(), "cpu")
    assert out == {"tf32_matmul": False, "tf32_cudnn": False, "matmul_precision": "highest",
                   "deterministic": False, "cublas_workspace_config": None,
                   "memory_limit_gb": None, "memory_fraction": None,
                   "device_index": None, "v0_sdpa_backends": ["MATH"]}  # fmt: skip
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.are_deterministic_algorithms_enabled()


def test_the_numerics_record_carries_the_v0_backends() -> None:
    """As a list: the bridge compares this process's record with the one a launcher run's
    ``meta.json`` reads back."""
    out = configure_numerics(config(), "cpu")
    assert out["v0_sdpa_backends"] == ["MATH"]
    assert json.loads(json.dumps(out)) == out


def test_determinism_sets_the_cublas_workspace_and_is_read_as_given(monkeypatch) -> None:
    out = configure_numerics(config("execution.deterministic=true"), "cpu")
    assert out["deterministic"] is True and torch.are_deterministic_algorithms_enabled()
    assert out["cublas_workspace_config"] == ":4096:8"
    again = configure_numerics(config(), "cpu")  # false turns it off; the setting stays as found
    assert again["deterministic"] is False and not torch.are_deterministic_algorithms_enabled()
    assert again["cublas_workspace_config"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # replaced, so both entry points agree
    replaced = configure_numerics(config("execution.deterministic=true"), "cpu")
    assert replaced["cublas_workspace_config"] == ":4096:8"


def test_tf32_and_the_memory_limit_follow_the_config() -> None:
    overrides = ["model.tf32=true", "execution.memory_limit_gb=46"]  # TF32 is off-protocol
    out = configure_numerics(config_module.load_config("experiments/smoke", overrides), "cpu")
    assert (out["tf32_matmul"], out["tf32_cudnn"], out["matmul_precision"]) == (True, True, "high")
    # On the CPU the limit is recorded but caps nothing.
    assert out["memory_limit_gb"] == 46.0
    assert out["memory_fraction"] is None and out["device_index"] is None


def test_the_record_is_read_back_from_torch(monkeypatch) -> None:
    monkeypatch.setattr(runtime.models, "set_numerics", lambda model_cfg: None)
    torch.set_float32_matmul_precision("medium")
    assert configure_numerics(config(), "cpu")["matmul_precision"] == "medium"


def test_the_cublas_workspace_cannot_be_set_after_cuda_work(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before any CUDA work"):
        configure_numerics(config("execution.deterministic=true"), "cpu")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # already in force: nothing to set
    assert configure_numerics(config("execution.deterministic=true"), "cpu")["deterministic"]


def test_an_unavailable_gpu_is_refused_before_any_setting(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        configure_numerics(config("execution.deterministic=true"), "cuda")
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_the_device_index_is_normalized_and_the_cap_applied_on_a_gpu() -> None:
    out = configure_numerics(config("execution.memory_limit_gb=1"), "cuda")
    assert out["device_index"] == torch.cuda.current_device()
    total = torch.cuda.get_device_properties(out["device_index"]).total_memory
    assert out["memory_fraction"] == pytest.approx(min(1.0, 1e9 / total))
    configure_numerics(config("execution.memory_limit_gb=1e6"), "cuda")  # lift the cap again


# The entry points: each calls the setup once, before environment() builds anything.


def entry_point(monkeypatch, cfg) -> list:
    calls = []
    setup = {"numerics": "as set"}

    def configure(c, device):
        calls.append(("configure_numerics", device))
        return setup

    def environment(c, device):
        calls.append(("environment", device))
        return types.SimpleNamespace(new_model=lambda: "model")

    monkeypatch.setattr(runtime, "configure_numerics", configure)
    monkeypatch.setattr(runtime, "environment", environment)
    monkeypatch.setattr(config_module, "from_cli", lambda: cfg)
    monkeypatch.setattr(protocol, "check_run_config", lambda c: None)
    monkeypatch.setattr(protocol, "load_run_config", lambda path, overrides: cfg)
    return calls


def test_the_launcher_entry_point_calls_the_one_setup_first(monkeypatch) -> None:
    cfg = OmegaConf.create({"schedule": {"steps": 25}, "execution": {"device": "cuda"}})
    calls = entry_point(monkeypatch, cfg)

    def run(c, env, *, run_id=None, numerics_setup=None):
        calls.append(("run_scheduled", numerics_setup))
        return "run-dir"

    monkeypatch.setattr(schedule, "run_scheduled", run)
    runpy.run_path(str(SCRIPTS / "train.py"), run_name="__main__")
    assert calls == [("configure_numerics", "cuda"), ("environment", "cuda"),
                     ("run_scheduled", {"numerics": "as set"})]  # fmt: skip


def test_study1_runs_take_the_same_setup(monkeypatch) -> None:
    calls = entry_point(monkeypatch, OmegaConf.create({"execution": {"device": "cuda"}}))
    monkeypatch.setattr(train, "run_training", lambda c, env: calls.append(("run_training",)))
    runpy.run_path(str(SCRIPTS / "train.py"), run_name="__main__")
    assert calls == [("configure_numerics", "cuda"), ("environment", "cuda"), ("run_training",)]


def test_teacher_generation_calls_the_setup_before_its_cuda_work(monkeypatch) -> None:
    cfg = OmegaConf.create({"execution": {"device": "cuda", "generation_batch_size": 4}})
    calls = entry_point(monkeypatch, cfg)
    monkeypatch.setattr(runtime, "generate_stream", lambda *a, **k: calls.append(("generate",)))
    monkeypatch.setattr("sys.argv", ["generate_teacher.py", "cfg", "--stream", "kl_harmful"])
    runpy.run_path(str(SCRIPTS / "generate_teacher.py"), run_name="__main__")
    assert calls == [("configure_numerics", "cuda"), ("environment", "cuda"), ("generate",)]


def test_the_stage0_checks_call_the_setup_before_their_cuda_work(monkeypatch) -> None:
    calls = entry_point(monkeypatch, OmegaConf.create({"execution": {"device": "cuda"}}))

    def stage0(c, env):
        calls.append(("stage0",))
        return types.SimpleNamespace(dir="run-dir")

    monkeypatch.setattr(checks, "run_stage0_checks", stage0)
    runpy.run_path(str(SCRIPTS / "stage0_checks.py"), run_name="__main__")
    assert calls == [("configure_numerics", "cuda"), ("environment", "cuda"), ("stage0",)]


def test_the_theta0_evaluation_calls_the_setup_before_its_cuda_work(monkeypatch) -> None:
    """θ₀'s utility check of study 3's engineering stage runs through this entry point."""
    cfg = OmegaConf.create({"execution": {"device": "cuda"}})
    calls, given = entry_point(monkeypatch, cfg), []
    monkeypatch.setattr(config_module, "from_cli", lambda args=None: given.append(args) or cfg)
    monkeypatch.setattr(train, "evaluate_theta0", lambda c, env: calls.append(("evaluate",)))
    monkeypatch.setattr("sys.argv", ["evaluate_theta0.py", "cfg"])
    runpy.run_path(str(SCRIPTS / "evaluate_theta0.py"), run_name="__main__")
    assert calls == [("configure_numerics", "cuda"), ("environment", "cuda"), ("evaluate",)]
    assert given == [["cfg", "run_type=theta0"]]


# The stage-0 record's fingerprint: determinism and the cuBLAS setting enter it only when set,
# so study 2's records keep matching study-2 configs and never match a deterministic run.


@pytest.fixture
def l40s(monkeypatch):
    """The study-2 environment of STUDY2_RECORD, as seen from a CPU-only test."""
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA L40S")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 92400)
    monkeypatch.setattr(prechecks, "package_versions",
                        lambda: {"torch": "2.14.1", "transformers": "5.18.0"})  # fmt: skip
    return torch.device("cuda")


def study2(*overrides):
    return protocol.load_run_config(
        "experiments/study2_short", ["study2_short.role=confirmation", *overrides]
    )


def passed_record(root, fingerprint, cfg):
    certified = {"criterion": protocol.PATH_FIELD_BOUND, "steps": cfg.train.steps,
                 "batch_size": cfg.train.batch_size, "cap": cfg.train.max_response_tokens,
                 "chi_square_bound": BOUND}  # fmt: skip
    write_check_record(root, fingerprint, certified, passed=True, run_id="stage0")


def test_study2_records_keep_matching_study2_configs(tmp_path, l40s) -> None:
    assert record_path(tmp_path, STUDY2_RECORD).name == "path-field-4e1d7d928d6c.json"
    cfg = study2()
    assert environment_fingerprint(cfg, l40s) == STUDY2_RECORD
    passed_record(tmp_path, STUDY2_RECORD, cfg)
    assert require_path_check(tmp_path, cfg, l40s)["fingerprint"] == STUDY2_RECORD


def test_a_record_without_determinism_does_not_match_a_deterministic_run(
    tmp_path, l40s, monkeypatch
) -> None:
    passed_record(tmp_path, STUDY2_RECORD, study2())
    deterministic = study2("execution.deterministic=true")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # as configure_numerics sets it
    fingerprint = environment_fingerprint(deterministic, l40s)
    assert fingerprint == STUDY2_RECORD | {"deterministic": True,
                                           "cublas_workspace_config": ":4096:8"}  # fmt: skip
    with pytest.raises(CheckFailed, match="No stage-0 path-field check"):
        require_path_check(tmp_path, deterministic, l40s)
    passed_record(tmp_path, fingerprint, deterministic)  # its own record unlocks it
    assert require_path_check(tmp_path, deterministic, l40s)["fingerprint"] == fingerprint


def test_a_cublas_setting_alone_is_a_different_environment(tmp_path, l40s, monkeypatch) -> None:
    passed_record(tmp_path, STUDY2_RECORD, study2())
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert environment_fingerprint(study2(), l40s) == STUDY2_RECORD | {
        "cublas_workspace_config": ":4096:8"
    }
    with pytest.raises(CheckFailed, match="No stage-0 path-field check"):
        require_path_check(tmp_path, study2(), l40s)
    # cuBLAS takes no part on the CPU.
    assert "cublas_workspace_config" not in environment_fingerprint(study2(), torch.device("cpu"))
