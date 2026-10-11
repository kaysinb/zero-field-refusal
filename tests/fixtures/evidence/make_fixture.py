"""Regenerate the evidence fixture: small synthetic run directories of one experiment.

    uv run python tests/fixtures/evidence/make_fixture.py

Writes ``runs/evidence_fixture/<run id>/`` next to this file, replacing an earlier copy, and
``expected_runs.csv``, the runs table the fixture must give: the values drawn here, not values
read back by the code under test. Texts are benign placeholders; nothing here is a model output.
The files are committed, so tests never depend on running this script; it is deterministic, so a
rerun rewrites the same bytes as long as the configs it composes are unchanged.

Layout: the three primary arms × 2 learning rates × seeds 0-2, every arm on the same seeds (one
teacher stream per seed, shared across arms), with
- a missing run: no directory for adam_frozen at 3e-5, seed 2;
- a failed run: sgd at 0.03, seed 0, with status ``failed`` and no summary or responses;
- incomplete judge labels: adamw at 1e-5, seed 1 (one response unlabeled, one unparseable);
- a label record of another judge: adamw at 3e-5, seed 2;
- a control run outside the primary analysis: AdamW with token-count normalization.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.special import expit

from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.judge import JudgeClient
from zero_field_refusal.protocol import load_run_config

HERE = Path(__file__).resolve().parent
EXPERIMENT = "evidence_fixture"
RESPONSES = 10  # eval.harmful.limit of experiments/smoke
STAMP = "2026-10-01T00:00:00+00:00"

# Cell means of log K (nats) per (arm, learning rate): the arms' ranges overlap, so K* exists.
CELLS = {
    ("sgd", 0.01): -0.9, ("sgd", 0.03): 0.45,
    ("adam_frozen", 1e-5): -0.7, ("adam_frozen", 3e-5): 0.65,
    ("adamw", 1e-5): -0.5, ("adamw", 3e-5): 0.8,
}  # fmt: skip
SEEDS = (0, 1, 2)
# Harmful-compliance curves: logit h = -0.5 + 1.2·log K, shifted per arm.
SHIFT = {"sgd": 0.0, "adam_frozen": 0.4, "adamw": 0.8}

MISSING = ("adam_frozen", 3e-5, 2)
FAILED = ("sgd", 0.03, 0)
INCOMPLETE = ("adamw", 1e-5, 1)
OTHER_JUDGE = ("adamw", 3e-5, 2)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_id(arm: str, lr: float, seed: int) -> str:
    return f"{arm}_lr{lr:g}_s{seed}"


def identity(name: str) -> dict:
    """A judge config's identity, as ``JudgeClient`` writes it into every record."""
    client = JudgeClient.from_config(load_config(f"judge/{name}"), env={}, client=object())
    return {"backend": client.backend, "prompt_version": client.prompt_version,
            "judge_model": client.model, "judge_revision": client.revision,
            "rubric_version": client.rubric_version}  # fmt: skip


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_jsonl(path: Path, records: list[dict], **dumps) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, **dumps) + "\n" for r in records))


def label_record(index: int, harmful: bool, judge: dict) -> dict:
    """A label record of a trained response, in ``JudgeClient``'s format."""
    label = "harmful_compliance" if harmful else "refusal"
    fields = {"harmful_request": True, "response_refusal": not harmful,
              "harmful_response": harmful}  # fmt: skip
    yes = {True: "yes", False: "no"}
    raw = (f"Harmful request: yes\nResponse refusal: {yes[not harmful]}\n"
           f"Harmful response: {yes[harmful]}")  # fmt: skip
    if judge["backend"] == "rubric":
        fields, raw = None, f"LABEL: {label}"
    return {
        "item_id": f"trained/judged:{index}",
        "label": label,
        "fields": fields,
        "raw_output": raw,
        "finish_reason": "stop",
        **judge,
        "created_at": STAMP,
        "content_sha256": sha(
            json.dumps([f"Placeholder prompt {index}.", f"Placeholder response {index}."])
        ),
    }


def write_run(root: Path, name: str, cfg, *, status: str, outcome: tuple | None, judges) -> None:
    """One run directory; ``outcome`` is (K, harmful flags), None for a run that failed."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    streams = {
        "training": {"path": f"teacher/training-seed{cfg.seed}", "count": 40,
                     "sha256": sha(f"training stream of seed {cfg.seed}")},
        "kl_harmful": {"path": "teacher/kl_harmful", "count": 20, "sha256": sha("kl_harmful")},
        "kl_benign": {"path": "teacher/kl_benign", "count": 20, "sha256": sha("kl_benign")},
    }  # fmt: skip
    write_json(run_dir / "meta.json", {
        "run_id": name, "experiment": EXPERIMENT, "name": None, "config_hash": config_hash(cfg),
        "status": status, "started_at": STAMP, "finished_at": STAMP, "argv": [],
        "git": {"commit": None, "dirty": None},
        "launch": {"source_snapshot_id": None, "container_image": None},
        "uv_lock_sha256": None, "python": "3.12", "packages": {}, "hardware": {"gpus": []},
        "wandb": None,
        "inputs": {"path_check": {"run_id": "fixture", "fingerprint": sha("environment")},
                   "prompt_manifest_sha256": sha("prompt manifest"), "streams": streams},
    })  # fmt: skip
    if outcome is None:
        return
    k, harmful = outcome
    write_json(
        run_dir / "summary.json",
        {"K_harmful": k, "K_benign": k / 2, "keyword_refusal_rate": 1 - float(np.mean(harmful))},
    )
    write_jsonl(run_dir / "responses" / "harmful_judged.jsonl", [
        {"condition": "trained", "item_id": f"judged:{i}", "response_ids": [11, 12, 13],
         "text": f"Placeholder response {i}.", "ended": True, "keyword_refusal": not h}
        for i, h in enumerate(harmful)
    ], sort_keys=True)  # fmt: skip
    records = [
        label_record(i, h, judge) for i, (h, judge) in enumerate(zip(harmful, judges, strict=True))
    ]
    write_jsonl(run_dir / "judge" / "wildguard.jsonl", records, ensure_ascii=False)


def main() -> None:
    root = HERE / "runs" / EXPERIMENT
    if root.parent.exists():
        shutil.rmtree(root.parent)
    rng = np.random.default_rng(2026)
    wildguard, rubric = identity("wildguard"), identity("rubric_gemini")
    expected = []
    for (arm, lr), mean_log_k in CELLS.items():
        for seed in SEEDS:
            log_k = mean_log_k + rng.normal(0.0, 0.15)
            k = float(np.exp(log_k))
            harmful = (rng.random(RESPONSES) < expit(-0.5 + 1.2 * log_k + SHIFT[arm])).tolist()
            if (arm, lr, seed) == MISSING:
                continue
            cfg = load_run_config("experiments/smoke", [
                f"experiment={EXPERIMENT}", f"optimizer={arm}", f"optimizer.lr={lr}",
                f"seed={seed}",
            ])  # fmt: skip
            judges = [wildguard] * RESPONSES
            if (arm, lr, seed) == OTHER_JUDGE:
                judges[3] = rubric
            name = run_id(arm, lr, seed)
            failed = (arm, lr, seed) == FAILED
            write_run(root, name, cfg, status="failed" if failed else "completed",
                      outcome=None if failed else (k, harmful), judges=judges)  # fmt: skip
            if (arm, lr, seed) == INCOMPLETE:
                path = root / name / "judge" / "wildguard.jsonl"
                lines = path.read_text().splitlines()[:-1]  # the last response is unlabeled
                record = json.loads(lines[0]) | {
                    "label": None,
                    "fields": None,
                    "raw_output": "Harmful request: yes",
                }
                lines[0] = json.dumps(record, ensure_ascii=False)  # an unparseable answer
                path.write_text("".join(line + "\n" for line in lines))
            if not failed and (arm, lr, seed) not in (INCOMPLETE, OTHER_JUDGE):
                expected.append({"run_id": name, "seed": seed, "arm": arm, "lr": lr, "K": k,
                                 "harmful": int(sum(harmful)), "judged": RESPONSES})  # fmt: skip

    # A control arm in the same experiment: complete, but not an arm of the primary analysis.
    cfg = load_run_config("experiments/smoke", [
        f"experiment={EXPERIMENT}", "optimizer.lr=3e-5", "seed=0",
        "control=token_count_normalization", "train.loss.normalization=token_count",
    ])  # fmt: skip
    harmful = (rng.random(RESPONSES) < 0.5).tolist()
    write_run(root, "adamw-token-count_lr3e-05_s0", cfg, status="completed",
              outcome=(1.5, harmful), judges=[wildguard] * RESPONSES)  # fmt: skip

    table = pd.DataFrame(expected).sort_values("run_id")
    table.to_csv(HERE / "expected_runs.csv", index=False, lineterminator="\n")
    print(f"Wrote {root} and {HERE / 'expected_runs.csv'}")


if __name__ == "__main__":
    main()
