"""Rescore the factual answers stored in run directories with the current matcher; no model runs.

    python scripts/rescore_factual_qa.py RUN_DIR [RUN_DIR ...]

Reads each run's factual-QA observations (the decoded answers) and writes
``observations/factual_qa_rescored.json`` next to them: the accuracy and per-item verdicts under
the current ``data.factual_answer_correct``, the accuracy the run recorded, and the code version.
The run's own files are left unchanged.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from zero_field_refusal.data import factual_answer_correct, load_prompt_set
from zero_field_refusal.tracking import git_state

RESCORED = "factual_qa_rescored.json"


def rescore(run_dir: Path) -> dict:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    answers = {item["id"]: item["answers"] for item in load_prompt_set("factual_qa", cfg.data)}
    lines = (run_dir / "observations" / "measurements.jsonl").read_text().splitlines()
    verdicts: dict[str, dict[str, tuple[bool, bool]]] = {}  # condition -> item -> (recorded, now)
    for row in map(json.loads, lines):
        if row["measure"] == "factual_qa":
            now = factual_answer_correct(row["text"], answers[row["item_id"]])
            verdicts.setdefault(row["condition"], {})[row["item_id"]] = (row["value"] == 1.0, now)
    result = {
        "rescored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": git_state(),
        "conditions": {
            condition: {
                "items": len(items),
                "recorded_accuracy": sum(old for old, _ in items.values()) / len(items),
                "accuracy": sum(now for _, now in items.values()) / len(items),
                "changed": sorted(item for item, (old, now) in items.items() if old != now),
            }
            for condition, items in verdicts.items()
        },
    }
    (run_dir / "observations" / RESCORED).write_text(json.dumps(result, indent=1) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    for run_dir in parser.parse_args().run_dirs:
        result = rescore(run_dir)
        for condition, c in result["conditions"].items():
            print(f"{run_dir.name} {condition}: {c['recorded_accuracy']:.3f} -> {c['accuracy']:.3f}"
                  f" ({len(c['changed'])} items changed)")  # fmt: skip


if __name__ == "__main__":
    main()
