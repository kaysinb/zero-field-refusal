"""The pool of judge-calibration responses, gathered from calibration run directories.

Each item carries the prompt text, the response text and the hidden conditions used for the
stratified blind sample: the model and whether the response comes from θ₀ or a trained model.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

CALIBRATION_SET = "harmful_calibration"


def calibration_pool(
    run_dirs: Iterable[Path], load_prompts: Callable[[DictConfig], list[dict]]
) -> list[dict[str, Any]]:
    """Items ``{item_id, prompt, response, model, state, run_id, prompt_id, ...}``.

    ``load_prompts(cfg)`` returns the calibration prompt set of a run's config.
    """
    pool = []
    for run_dir in run_dirs:
        cfg = OmegaConf.load(run_dir / "config.yaml")
        if cfg.eval.harmful.prompt_set != CALIBRATION_SET:
            raise ValueError(f"{run_dir} did not answer the calibration prompts")
        texts = {item["id"]: item["text"] for item in load_prompts(cfg)}
        run_id = run_dir.name
        for line in (run_dir / "responses" / f"{CALIBRATION_SET}.jsonl").read_text().splitlines():
            record = json.loads(line)
            pool.append(
                {
                    "item_id": f"{run_id}/{record['item_id']}",
                    "prompt": texts[record["item_id"]],
                    "response": record["text"],
                    "model": cfg.model.name,
                    "state": "theta0" if record["condition"] == "theta0" else "trained",
                    "condition": record["condition"],
                    "run_id": run_id,
                    "prompt_id": record["item_id"],
                    "optimizer": cfg.optimizer.name,
                    "lr": cfg.optimizer.lr,
                    "seed": cfg.seed,
                }
            )
    return pool
