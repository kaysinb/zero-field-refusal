"""Describe SGD-long's barrier in study 2's Stage A from K, curvature and the SGD screen alone.

    python scripts/stage_a_barrier.py --runs DIR [DIR ...] --curvature RUN_DIR --out OUT_DIR

Reads no harm label and no response text: the scoring diagnostics and monitor readings of the
long runs, the stops of the screen's soft-target controls and the Lanczos coefficients of the
curvature run. OUT_DIR, which must not exist, receives barrier.json (unrounded numbers, run IDs,
the command and the code revision) and barrier.png, drawn from that table.

- **Zero-field scaling of SGD-long** (docs/study2-stage-a-analysis.md): K / (η/B) against ηt from
  the finite scoring diagnostics of every version-2 SGD-long run, the table of g over them
  (analysis.zero_field_law.g_table) and each run's median ratio to it (``check``); a terminal
  non-finite diagnostic is left out and listed.
- **Its bound.** The Hutchinson estimate of tr(H) from the first Lanczos coefficient of every
  start (a unit vector with iid normal entries, so E[vᵀHv] = tr(H) / n over the n trainable
  parameters), and g(∞) = tr(H) / 4 with ρ = 1.
- **Stability.** The screen's largest stable and smallest unstable rate at each batch, and the
  curvature run's top eigenvalues.
- **AdamW-long's plateaus.** The median monitor K over the last 100 steps of every run stopped by
  plateau or T_max, by batch and rate, with the monitor version it read.
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from zero_field_refusal.analysis.stage_a import load_trajectories
from zero_field_refusal.analysis.zero_field_law import check, g_table
from zero_field_refusal.tracking import git_state


def trainable_count(model: str) -> int:
    """Trainable parameters of ``name@revision``: all but the (tied) input embeddings."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    name, revision = model.split("@")
    with torch.device("meta"):
        net = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name, revision=revision))
    frozen = {id(net.get_input_embeddings().weight), id(net.get_output_embeddings().weight)}
    return sum(p.numel() for p in net.parameters() if id(p) not in frozen)


def sgd_scaling(runs) -> dict:
    table = g_table(runs)
    out = {"g_table": table, "runs": []}
    for run in runs:
        scale = run.lr / run.batch_size
        rows = [(run.lr * r["step"], r["K_scoring"] / scale) for r in run.diagnostics
                if r["step"] and math.isfinite(r["K_scoring"])]  # fmt: skip
        out["runs"].append({
            "run_id": run.run_id, "seed": run.seed, "batch_size": run.batch_size, "lr": run.lr,
            "stop": run.stop, "steps": run.steps,
            "last_100_monitor_K": statistics.median(k for s, k in run.monitor
                                                    if s > run.steps - 100),
            "median_ratio_to_g": check(run, table)["median_ratio"],
            "excluded_nonfinite_steps": [int(r["step"]) for r in run.diagnostics
                                         if not math.isfinite(r["K_scoring"])],
            "eta_t_and_K_over_eta_by_B": rows,
        })  # fmt: skip
    return out


def curvature_summary(run_dir: Path, n: int) -> dict:
    data = json.loads((run_dir / "curvature.json").read_text())
    measurements = [("pooled", data["pooled"])] + [
        (f"B={m['batch_size']} batch {m['batch']}", m) for m in data["single"]
    ]
    rows = []
    for name, m in measurements:
        for start in m["starts"]:
            rows.append({"measurement": name, "top": start["top"], "bottom": start["bottom"],
                         "vHv": start["alphas"][0], "trace_H": n * start["alphas"][0]})  # fmt: skip
    traces = [r["trace_H"] for r in rows]
    trace = statistics.median(traces)
    return {
        "trainable_parameters": n,
        "starts": rows,
        "trace_H": {"median": trace, "min": min(traces), "max": max(traces)},
        "g_infinity_rho_1": trace / 4,
        "tops": {name: m["top"] for name, m in measurements},
    }


def stability(screen) -> dict:
    out = {}
    for batch in sorted({r.batch_size for r in screen}):
        rows = sorted((r for r in screen if r.batch_size == batch), key=lambda r: r.lr)
        stable = [r.lr for r in rows if r.stable]
        unstable = [r for r in rows if not r.stable]
        first = unstable[0] if unstable else None
        out[str(batch)] = {
            "rates": [{"lr": r.lr, "stop": r.stop, "steps": r.steps} for r in rows],
            "largest_stable": max(stable) if stable else None,
            "smallest_unstable": first.lr if first else None,
            "eta_over_B": [max(stable) / batch if stable else None,
                           first.lr / batch if first else None],
        }  # fmt: skip
    return out


def adamw_plateaus(runs) -> list[dict]:
    out = []
    for run in sorted(runs, key=lambda r: (r.batch_size, r.lr, r.seed, r.monitor_version)):
        if run.stop not in ("plateau", "t_max"):
            continue
        out.append({
            "run_id": run.run_id, "batch_size": run.batch_size, "lr": run.lr, "seed": run.seed,
            "monitor_version": run.monitor_version, "stop": run.stop, "steps": run.steps,
            "last_100_monitor_K": statistics.median(k for s, k in run.monitor
                                                    if s > run.steps - 100),
        })  # fmt: skip
    return out


def figure(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for run in result["sgd_long"]["runs"]:
        xs, ys = zip(*run["eta_t_and_K_over_eta_by_B"], strict=True)
        left.plot(xs, ys, lw=1, label=f"B={run['batch_size']} η={run['lr']:g} s{run['seed']}")
    bound = result["curvature"]["g_infinity_rho_1"]
    left.axhline(bound, color="k", ls="--", lw=1, label="tr(H)/4 (ρ = 1)")
    left.set_ylim(bottom=0)
    left.set(xscale="log", xlabel="η t", ylabel="K_scoring / (η / B)",
             title="SGD-long: zero-field scaling")  # fmt: skip
    left.legend(fontsize=7)
    groups: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in result["adamw_long_plateaus"]:
        groups[row["batch_size"]][row["lr"]].append(row["last_100_monitor_K"])
    for batch, by_lr in sorted(groups.items()):
        lrs = sorted(by_lr)
        right.plot(lrs, [statistics.median(by_lr[lr]) for lr in lrs], "o-", label=f"B={batch}")
        for lr in lrs:
            right.scatter([lr] * len(by_lr[lr]), by_lr[lr], s=8, color="grey")
    right.set(xscale="log", yscale="log", xlabel="AdamW rate", ylabel="monitor K, last 100 steps",
              title="AdamW-long: plateau or T_max")  # fmt: skip
    right.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--curvature", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    runs = load_trajectories(*args.runs)
    sgd = [r for r in runs if r.arm == "sgd_long" and r.role == "discovery"
           and r.monitor_version > 1]  # fmt: skip
    adamw = [r for r in runs if r.arm == "adamw_long" and r.role == "discovery"]
    screen = [r for r in runs if r.role == "screen"]
    model = OmegaConf.load(args.curvature / "config.yaml").model
    n = trainable_count(f"{model.name}@{model.revision}")
    result = {
        "command": sys.argv,
        "code": git_state(),
        "sgd_long": sgd_scaling(sorted(sgd, key=lambda r: (r.batch_size, r.lr, r.seed))),
        "curvature": {"run": args.curvature.name, **curvature_summary(args.curvature, n)},
        "stability": stability(screen),
        "adamw_long_plateaus": adamw_plateaus(adamw),
    }
    args.out.mkdir(parents=True)
    (args.out / "barrier.json").write_text(json.dumps(result, indent=1))
    figure(result, args.out / "barrier.png")
    print(args.out)


if __name__ == "__main__":
    main()
