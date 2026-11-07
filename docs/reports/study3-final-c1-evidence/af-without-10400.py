"""Descriptive, not registered: A-F's block differences (the analysis's own reading), the count with
d < 0, and A-F's mean and bootstrap-t interval at L without block 10400. Prints numbers only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
spec = importlib.util.spec_from_file_location("study3_analysis", REPO / "scripts/study3_analysis.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
from zero_field_refusal.analysis import study3_estimates as est  # noqa: E402
from zero_field_refusal.analysis.study3_inference import bootstrap_t, stream  # noqa: E402

root, theta0, sealed = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
freeze_, problems = sa.read_freeze(1, runs_root=root, repo=sa.REPO)
assert freeze_ is not None, problems
inputs = sa.in_force(root, 1, freeze_)
state, endpoints = sa.read_state(root, repo=sa.REPO, expected_code=freeze_["code"]["gpu_code_sha256"],
                                 theta0=theta0)
harm, _ = sa.labels(freeze_, endpoints, root, sealed_dir=sealed)
blocks = freeze_["comparisons"]["A-F"]["blocks"]
d = est.block_differences(blocks, harm, ("A", "F"))["primary"]
seeds = [b["seed"] for b in blocks]
level, boot = inputs["level"], inputs["bootstrap"]
full = bootstrap_t(d, level=level, resamples=boot["resamples"], seed=stream(boot["seed"], "analysis", "A-F"))
keep = np.array([s != 10400 for s in seeds])
part = bootstrap_t(d[keep], level=level, resamples=boot["resamples"],
                   seed=stream(boot["seed"], "descriptive-without-10400", "A-F"))
print(json.dumps({
    "n": len(d), "negative": int((d < 0).sum()), "zero": int((d == 0).sum()), "positive": int((d > 0).sum()),
    "full_mean_pp": 100 * float(d.mean()), "full_interval_pp": [100 * full.lower, 100 * full.upper],
    "without_10400": {"n": int(keep.sum()), "mean_pp": 100 * float(d[keep].mean()),
                      "interval_pp": [100 * part.lower, 100 * part.upper], "level": level,
                      "resamples": boot["resamples"],
                      "stream": "stream(seed, 'descriptive-without-10400', 'A-F')"},
}))
