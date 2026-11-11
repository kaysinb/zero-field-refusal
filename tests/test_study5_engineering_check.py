"""scripts/study5_engineering_check.py: a rerun's records against the archived ones, after the
volatile fields are removed (``analysis.record_normalize``, the golden fixture's rule)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch
from test_schedule import scheduled
from test_train import env as env  # noqa: F401  (the fixture)
from test_train import prepare

from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.schedule import run_scheduled


def script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = script_module(REPO_ROOT / "scripts" / "study5_engineering_check.py", "study5_eng_check")

OVERRIDES = ("optimizer=adam_frozen", "train.steps=3", "schedule.evaluate_at=[3]",
             "eval.monitor.every=1", "eval.diagnostics.every=1", "tracking.update_scalars=true",
             "tracking.provenance_hashes=true", "optimizer.lr=1e-3",
             "optimizer.preconditioner.batches=3")  # fmt: skip
SECRET = "RESPONSE-TEXT-NEVER-PRINTED"


def tiny_runs(env, tmp_path: Path) -> tuple[Path, Path]:  # noqa: F811
    """Two tiny scheduled F runs of one config, as the archive and the rerun."""
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES="
    torch.set_num_threads(1)
    cfg = scheduled(tmp_path, *OVERRIDES, policy="short")
    prepare(env, cfg, "kl_monitor", "preconditioner")
    # The first run grows the appendable training stream; the two compared runs find it whole,
    # as study 3's run and its rerun both find their seed's stream generated beforehand.
    run_scheduled(cfg, env, run_id="grows-the-stream-a1")
    first = run_scheduled(cfg, env, run_id="archived-a1")
    second = run_scheduled(cfg, env, run_id="rerun-a1")
    archived = tmp_path / "archived" / first.name
    shutil.copytree(first, archived)
    return archived, second


def flip_digit(path: Path) -> None:
    """Change one byte of the file: its first digit after the first line's first colon."""
    data = bytearray(path.read_bytes())
    start = data.index(b":")
    for i in range(start, len(data)):
        if chr(data[i]).isdigit():
            data[i] = ord("1") if data[i] != ord("1") else ord("2")
            break
    path.write_bytes(bytes(data))


def test_two_runs_of_one_code_are_equal_and_one_byte_differs(env, tmp_path, capsys):  # noqa: F811
    archived, rerun = tiny_runs(env, tmp_path)
    (archived / "judge").mkdir()  # the labeling's outputs, written later: never compared
    (archived / "judge" / "wildguard.jsonl").write_text('{"x": 1}\n')
    out = tmp_path / "equal.json"
    assert check.main(["--archived", str(archived), "--rerun", str(rerun), "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["equal"] is True and len(record["pairs"]) == 1
    pair = record["pairs"][0]
    names = {f["name"] for f in pair["files"]}
    assert {"metrics.jsonl", "summary.json", "terminal.json", "meta.json", "config.yaml",
            "observations/measurements.jsonl"} <= names  # fmt: skip
    assert any(name.startswith("responses/") for name in names)
    assert not any(name.startswith(("judge/", "wandb/")) for name in names)
    assert all(f["equal"] for f in pair["files"])

    for name in ("metrics.jsonl", "summary.json"):
        changed = tmp_path / f"changed-{name}"
        shutil.copytree(rerun, changed)
        flip_digit(changed / name)
        out = tmp_path / f"differ-{name}.json"
        code = check.main(["--archived", str(archived), "--rerun", str(changed), "--out", str(out)])
        assert code == 2, name
        record = json.loads(out.read_text())
        assert record["equal"] is False
        assert [f["name"] for f in record["pairs"][0]["files"] if not f["equal"]] == [name]


def test_a_response_difference_is_named_by_its_file_only(env, tmp_path, capsys):  # noqa: F811
    archived, rerun = tiny_runs(env, tmp_path)
    responses = sorted((rerun / "responses").glob("*.jsonl"))[0]
    rows = [json.loads(line) for line in responses.read_text().splitlines()]
    key = next(k for k, v in rows[0].items() if isinstance(v, str) and k != "id")
    rows[0][key] = SECRET
    responses.write_text("".join(json.dumps(r) + "\n" for r in rows))
    capsys.readouterr()
    out = tmp_path / "differ.json"
    assert check.main(["--archived", str(archived), "--rerun", str(rerun), "--out", str(out)]) == 2
    printed = capsys.readouterr()
    text = out.read_text()
    assert SECRET not in printed.out + printed.err and SECRET not in text
    entry = next(f for f in json.loads(text)["pairs"][0]["files"] if not f["equal"])
    assert entry["name"] == f"responses/{responses.name}"
    assert entry["differing_lines"] == 1 and "keys" not in entry


# ------------------------------------------------------------------ synthetic run directories


def fake_run(root: Path, name: str, config_hash: str = "aaaaaaaaaaaa", **files: str) -> Path:
    run = root / name
    (run / "responses").mkdir(parents=True)
    meta = {"run_id": name, "config_hash": config_hash, "status": "completed",
            "launch": {"pod_id": name}, "campaign": {"name": name}, "git": {"commit": name},
            "inputs": {"v0_sha256": "f" * 64}}  # fmt: skip
    (run / "meta.json").write_text(json.dumps(meta))
    (run / "config.yaml").write_text("seed: 1\nexperiment: study3\n")
    (run / "metrics.jsonl").write_text('{"step": 0, "loss": 1.5, "seconds": 3.0}\n')
    (run / "responses" / "harmful_evaluation.jsonl").write_text('{"id": "h1", "text": "x"}\n')
    for rel, text in files.items():
        (run / rel).write_text(text)
    return run


def test_the_placement_and_the_volatile_fields_are_not_compared(tmp_path):
    a = fake_run(tmp_path / "a", "run-a")
    b = fake_run(tmp_path / "b", "run-b")
    (b / "metrics.jsonl").write_text('{"step": 0, "loss": 1.5, "seconds": 9.0}\n')
    (b / "wandb").mkdir()
    (b / "wandb" / "x.log").write_text("anything")
    out = tmp_path / "r.json"
    assert check.main(["--archived", str(a), "--rerun", str(b), "--out", str(out)]) == 0


def test_a_file_on_one_side_only_differs(tmp_path):
    a = fake_run(tmp_path / "a", "run-a", **{"summary.json": "{}"})
    b = fake_run(tmp_path / "b", "run-b")
    out = tmp_path / "r.json"
    assert check.main(["--archived", str(a), "--rerun", str(b), "--out", str(out)]) == 2
    files = json.loads(out.read_text())["pairs"][0]["files"]
    assert {"name": "summary.json", "equal": False, "only_in": "archived"} in files


def test_runs_pair_by_config_hash_and_an_unpaired_run_differs(tmp_path):
    a1 = fake_run(tmp_path / "a", "a1", "111111111111")
    a2 = fake_run(tmp_path / "a", "a2", "222222222222")
    b2 = fake_run(tmp_path / "b", "b2", "222222222222")
    b1 = fake_run(tmp_path / "b", "b1", "111111111111")
    out = tmp_path / "r.json"
    argv = ["--archived", str(a1), str(a2), "--rerun", str(b2), str(b1), "--out", str(out)]
    assert check.main(argv) == 0
    assert [p["config_hash"] for p in json.loads(out.read_text())["pairs"]] == [
        "111111111111", "222222222222"]  # fmt: skip
    b3 = fake_run(tmp_path / "b", "b3", "333333333333")
    out2 = tmp_path / "r2.json"
    argv = ["--archived", str(a1), str(a2), "--rerun", str(b3), str(b1), "--out", str(out2)]
    assert check.main(argv) == 2
    record = json.loads(out2.read_text())
    assert record["unpaired"] == {"archived": ["222222222222"], "rerun": ["333333333333"]}


@pytest.mark.parametrize("problem", ["no meta", "bad json", "out exists", "no run"])
def test_errors_exit_1(tmp_path, problem):
    a = fake_run(tmp_path / "a", "run-a")
    b = fake_run(tmp_path / "b", "run-b")
    out = tmp_path / "r.json"
    if problem == "no meta":
        (b / "meta.json").unlink()
    elif problem == "bad json":
        (b / "metrics.jsonl").write_text("{not json\n")
    elif problem == "out exists":
        out.write_text("{}")
    else:
        shutil.rmtree(b)
    assert check.main(["--archived", str(a), "--rerun", str(b), "--out", str(out)]) == 1
    if problem != "out exists":
        assert not out.exists()


def test_an_ignored_key_is_dropped_only_when_asked(tmp_path):
    """Extension 2: a rerun that records its step checkpoints beside the training is equal under
    ``--ignore step_checkpoints`` and differs without it; the record names the ignored key."""
    a = fake_run(tmp_path / "a", "run-a")
    b = fake_run(tmp_path / "b", "run-b")
    meta = json.loads((b / "meta.json").read_text())
    meta.setdefault("inputs", {})["step_checkpoints"] = [{"step": 2, "sha256": "f" * 64}]
    (b / "meta.json").write_text(json.dumps(meta))
    out = tmp_path / "r.json"
    assert check.main(["--archived", str(a), "--rerun", str(b), "--out", str(out)]) == 2
    out2 = tmp_path / "r2.json"
    argv = ["--archived", str(a), "--rerun", str(b), "--out", str(out2), "--ignore",
            "step_checkpoints"]  # fmt: skip
    assert check.main(argv) == 0
    assert json.loads(out2.read_text())["ignored_keys"] == ["step_checkpoints"]
    with pytest.raises(SystemExit):
        check.main(["--archived", str(a), "--rerun", str(b), "--out", str(tmp_path / "x.json"),
                    "--ignore", "metrics"])  # fmt: skip
    check.IGNORED = frozenset()


def _checkpoint(path: Path, value: float) -> str:
    torch.save({"w": torch.full((3,), value), "b": torch.zeros(2)}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_endpoint(run: Path, ckpt: Path, sha: str, directory: str, steps=None) -> None:
    meta = json.loads((run / "meta.json").read_text())
    meta["inputs"]["endpoint_checkpoint"] = {"path": str(ckpt), "sha256": sha, "step": 25,
                                             "names_sha256": "n" * 64}  # fmt: skip
    if steps is not None:
        meta["inputs"]["step_checkpoints"] = steps
    (run / "meta.json").write_text(json.dumps(meta))
    lines = ["seed: 1", "experiment: study3", "tracking:", f"  endpoint_checkpoint: {directory}"]
    if steps is None:
        lines.append("  step_checkpoints: null")
    (run / "config.yaml").write_text("\n".join(lines) + "\n")


def test_storage_is_canonicalized_and_the_endpoint_compared_by_its_tensors(tmp_path):
    """Extension 2: a rerun that saves its endpoint in another folder under another name (so its
    file hash differs) and records a null step list is equal when its tensors are the archived
    ones (hashed by scripts/study5_tensor_hash.py); other tensors differ; a missing expected
    value or a missing rerun hash is an error. The check itself loads no checkpoint."""
    hasher = script_module(REPO_ROOT / "scripts" / "study5_tensor_hash.py", "s5_tensor_hash")
    a = fake_run(tmp_path / "a", "run-a")
    b = fake_run(tmp_path / "b", "run-b")
    ckpt_a, ckpt_b = tmp_path / "a.pt", tmp_path / "other-name.pt"
    sha_a = _checkpoint(ckpt_a, 1.0)
    sha_b = _checkpoint(ckpt_b, 1.0)
    assert sha_a != sha_b  # torch.save writes the file's name into its archive
    _with_endpoint(a, ckpt_a, sha_a, "/root/ckpt/study5", steps=None)
    meta_a = json.loads((a / "meta.json").read_text())
    meta_a["inputs"].pop("step_checkpoints", None)
    (a / "meta.json").write_text(json.dumps(meta_a))
    (a / "config.yaml").write_text("seed: 1\nexperiment: study3\ntracking:\n"
                                   "  endpoint_checkpoint: /root/ckpt/study5\n")  # fmt: skip
    _with_endpoint(b, ckpt_b, sha_b, "/workspace/ckpt/pod4", steps=None)
    from zero_field_refusal.tensor_hash import tensors_sha256

    expected = tensors_sha256(list(torch.load(ckpt_a, weights_only=True).items()))
    hashed = tmp_path / "hashed.json"
    assert hasher.main([str(b), "--out", str(hashed)]) == 0
    assert json.loads(hashed.read_text())["run-b"] == {"file_ok": True, "tensors_sha256": expected}
    out = tmp_path / "r.json"
    argv = ["--archived", str(a), "--rerun", str(b), "--out", str(out)]
    good = ["--endpoint-tensors", f"aaaaaaaaaaaa={expected}", "--rerun-endpoint-tensors",
            str(hashed)]  # fmt: skip
    assert check.main([*argv, *good]) == 0
    assert json.loads(out.read_text())["pairs"][0]["endpoint_tensors"]["equal"] is True
    assert check.main([*argv[:-1], str(tmp_path / "r1.json"), "--rerun-endpoint-tensors",
                       str(hashed)]) == 1  # no expected value: an error  # fmt: skip
    assert check.main([*argv[:-1], str(tmp_path / "r2.json"), "--endpoint-tensors",
                       f"aaaaaaaaaaaa={expected}"]) == 1  # no rerun hash: an error  # fmt: skip
    _checkpoint(ckpt_b, 2.0)  # other tensors, same name: the file no longer matches its record
    hashed2 = tmp_path / "hashed2.json"
    assert hasher.main([str(b), "--out", str(hashed2)]) == 0
    assert json.loads(hashed2.read_text())["run-b"]["file_ok"] is False
    sha_b2 = hashlib.sha256(ckpt_b.read_bytes()).hexdigest()
    _with_endpoint(b, ckpt_b, sha_b2, "/workspace/ckpt/pod4", steps=None)
    hashed3 = tmp_path / "hashed3.json"
    assert hasher.main([str(b), "--out", str(hashed3)]) == 0
    out4 = tmp_path / "r4.json"
    assert check.main([*argv[:-1], str(out4), "--endpoint-tensors", f"aaaaaaaaaaaa={expected}",
                       "--rerun-endpoint-tensors", str(hashed3)]) == 2  # fmt: skip
    tensors = json.loads(out4.read_text())["pairs"][0]["endpoint_tensors"]
    assert tensors["equal"] is False and tensors["rerun"] != expected
    assert hasher.main([str(b), "--out", str(hashed3)]) == 1  # never overwritten
