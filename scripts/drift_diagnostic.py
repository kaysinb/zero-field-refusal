"""Run the first-step drift diagnostic of study 2 (amendment 4, section 4).

    python scripts/drift_diagnostic.py prepare --batch B [--device D] [--dry-run]
    python scripts/drift_diagnostic.py plan --batch B [--runs-root DIR] [--out DIR] [--dry-run]
    python scripts/drift_diagnostic.py run --batch B [--runs-root DIR] [--out DIR] [--device D]
        [--dry-run]
    python scripts/drift_diagnostic.py driver --batch B --commit SHA --out FILE [--dry-run]

B is the selected batch (20 is expected); it is never written in the settings. Descriptive and
blind to outcomes: only norms, dot products, losses and v₀'s hash are computed and written
(src/zero_field_refusal/drift.py); no harmful prompt set, stream or generation, no judge and no
refusal, and every prompt set whose name contains "harmful" raises if any code asks for it
(``guard_prompt_sets``). The settings are the tracked configs/experiments/drift_diagnostic.yaml:
``run`` and ``driver`` refuse a ``--config`` that resolves to any other file unless they only
print (``--dry-run``).

- ``prepare`` generates the streams that are missing, on ``--device``, as
  scripts/generate_teacher.py does (``drift.prepare_streams``): seed 2300's drift stream (its
  appendable training stream, 64·B responses, grown by whole chunks) and its v₀ preconditioner
  stream (50·B responses). Streams that exist are located and kept. ``--dry-run`` only locates
  them.
- ``plan`` prints what ``run`` computes, its cost, where the streams are and the free host memory
  against the requirement (1.5 × 21 GB + 16 GB); it exits 1 when the host memory falls short.
  ``--dry-run`` prints without reading DATA_DIR or the host.
- ``run`` computes the diagnostic (``drift.run_drift``) and writes
  ``RUNS_DIR/diagnostics/drift/b{B}/result.json`` (``--out``): the statistics, the peaks,
  ``numerics``, the provenance with the GPU-code identity, and the stream identities. It runs
  only from a clean checkout (``tracking.git_state``), under the memory rule (``memory_rule_gb``:
  46 GB reserved, 1e9 bytes, a cap on the process's CUDA memory) and after the host-memory check
  (exit 5 when it falls short). A result written from a clean checkout at a commit with this GPU
  code (``code_identity.gpu_code_identity``) is kept: running again skips it. Another is moved
  aside to ``stale/<its commit>-<time>/`` first. A failure is recorded in ``failed.json`` (the
  error, the peaks, the commit) before the non-zero exit (4 when out of memory under the rule); a
  later success renames that record, which is kept. ``--dry-run`` prints what would run without
  loading a model or writing anything.
- ``driver`` writes the bash driver for a Pod, from the source checkout of SHA (``--dry-run``
  prints it).

Cost on one L40S at 1.5B (fp32), from the study-2 timings at B 20 (scratchpad batch-cost
analysis of 2026-10-19: a training step ≈ 1.2 s, ≈ 0.06 s per response; generation ≈ 0.12 s per
training response and ≈ 0.09 s per preconditioner response, process start included); the
per-batch update and float64 reductions (≈ 0.5 s) and the setup are estimates. At B 20:
- ``prepare`` ≈ 5.5 min when both streams are generated (two chunks of 1 000 training responses,
  1 000 preconditioner responses);
- ``run`` ≈ 12 min: v₀'s 50 batch gradients, 4 × 64 batch gradients with their updates, the
  setup and the cross cosines in host memory.
GPU ≈ 22 GB of tensors at 1.5B (the model, a gradient, the two half-sums) and a step's
activations, under the 46 GB cap; host memory ≈ 21 GB for the four sums and 5.2 GB of pinned v₀.
"""

import argparse
import json
import sys
import textwrap
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal import drift
from zero_field_refusal.config import _locate, load_config
from zero_field_refusal.fa_control_diagnostic import (
    apply_memory_rule,
    guarded_prompt_sets,
    peak_memory,
    set_aside,
    write_json,
)
from zero_field_refusal.paths import CONFIG_ROOT
from zero_field_refusal.tracking import CLEAN_CHECKOUT

CONFIG = "experiments/drift_diagnostic"
SCRIPT = 'project-run "$SRC" python scripts/drift_diagnostic.py'
PROBLEM = 1  # the plan: the host memory falls short; the driver: a check failed
OUT_OF_MEMORY = 4  # run: the memory rule's cap was met
HOST_MEMORY = 5  # run: the free host memory falls short of the requirement
GPU_USED_MIB = 1000  # the driver starts only below this much GPU memory in use
DISK_GB = 5  # the free space the streams (≈ 3 000 responses), the result and the log need
FAILED = "failed.json"
# The driver's failure record when a step left none (python -c, through project-run): no $,
# backtick, backslash or double quote inside.
FAILURE_RECORD = (
    "import json, sys; from datetime import UTC, datetime; "
    "from zero_field_refusal.tracking import git_state; "
    "json.dump({'part': 'driver', 'reason': sys.argv[2], 'exit_code': int(sys.argv[3]), "
    "'written_at': datetime.now(UTC).isoformat(timespec='seconds'), "
    "'provenance': {'code': git_state()}}, open(sys.argv[1], 'w'), indent=1)"
)
# Seconds on one L40S at 1.5B (module docstring), per response unless stated
GRADIENT_S = 0.06
UPDATE_S = 0.5  # per batch: the per-parameter updates and float64 reductions (estimate)
GENERATION_S = {"training": 0.12, "preconditioner": 0.09}
SETUP_S = 240  # the model, the streams, v₀'s hash and the cross cosines (estimate)


def comment(text: str) -> list[str]:
    return textwrap.wrap(text, 100, initial_indent="# ", subsequent_indent="# ",
                         break_long_words=False, break_on_hyphens=False)  # fmt: skip


def guard_prompt_sets(env):
    """The environment with every prompt set whose name contains "harmful" refused: the
    diagnostic reads the teacher prompts alone, and this makes any path to a harmful set
    raise."""
    return guarded_prompt_sets(env, "harmful", "The drift diagnostic")


def config_path(config: str) -> Path:
    """The settings file ``--config`` names, found as ``config.load_config`` finds it."""
    return _locate(config, CONFIG_ROOT)


def tracked_config(config: str) -> bool:
    """Whether ``--config`` resolves to the tracked settings, whatever the working directory."""
    return config_path(config) == (CONFIG_ROOT / f"{CONFIG}.yaml").resolve()


def costs(scope: drift.DriftScope, batch: int, chunk_size: int) -> dict[str, float]:
    """GPU minutes at B: ``prepare`` when both streams are generated, and ``run``."""
    chunks = -(-scope.batches * batch // chunk_size)
    prepare = (chunks * chunk_size * GENERATION_S["training"]
               + scope.v0_batches * batch * GENERATION_S["preconditioner"])  # fmt: skip
    gradients = (scope.v0_batches + len(scope.optimizers) * scope.batches) * batch * GRADIENT_S
    updates = len(scope.optimizers) * scope.batches * UPDATE_S
    return {"prepare": prepare / 60, "run": (gradients + updates + SETUP_S) / 60}


def describe(scope: drift.DriftScope, cfg, out: Path) -> list[str]:
    """What ``run`` computes, and its cost."""
    batch, chunk_size = int(cfg.train.batch_size), int(cfg.teacher.chunk_size)
    cost = costs(scope, batch, chunk_size)
    return [
        f"output: {out / drift.RESULT}",
        f"B {batch}, seed {scope.seed}: {scope.batches} batches of the drift stream (responses "
        f"0-{scope.batches * batch - 1}), halves batches 0-{scope.half - 1} and "
        f"{scope.half}-{scope.batches - 1}; v₀ from {scope.v0_batches} batches at B",
        f"first updates {', '.join(scope.optimizers)} (ε {scope.eps:g}, β₂ {scope.beta2:g}), one "
        f"at a time: {len(scope.optimizers)} × {scope.batches} batch gradients",
        f"cost ≈ {cost['run']:.0f} GPU min for run, ≈ {cost['prepare']:.0f} more for prepare if "
        f"both streams are generated; reserved GPU memory capped at {scope.memory_rule_gb:g} GB; "
        f"free host memory required {scope.host_required_gb:g} GB",
    ]  # fmt: skip


def host_summary(check: dict) -> str:
    verdict = "sufficient" if check["sufficient"] else "INSUFFICIENT"
    available = check["available_gb"]
    shown = "unknown" if available is None else f"{available:.1f} GB"
    return (f"host memory: {shown} free (meminfo {check['meminfo_gb']}, cgroup "
            f"{check['cgroup_gb']}), {check['required_gb']:g} GB required: {verdict}")  # fmt: skip


def stream_lines(state: dict) -> list[str]:
    d, p = state["drift"], state["preconditioner"]
    return [f"{'found' if d['ready'] else 'MISSING'}: drift stream {d['path']}: {d['chunks']} "
            f"chunks, {d['needed']} needed ({d['responses']} responses)",
            f"{'found' if p['ready'] else 'MISSING'}: v₀ stream {p['path']} "
            f"({p['responses']} responses)"]  # fmt: skip


def require_clean() -> dict:
    """A run writes only from a clean checkout of a commit."""
    state = drift.git_state()
    if state.get("commit") is None or state.get("dirty") is not False:
        sys.exit("The checkout is not clean (tracking.git_state): a result made from it would "
                 "not name its code; commit first")  # fmt: skip
    return state


def counted(out: Path, gpu_code: dict) -> dict | None:
    """The result already under ``out``, if it counts: written from a clean checkout at a
    commit with this GPU code."""
    path = out / drift.RESULT
    if not path.exists() or "error" in gpu_code:
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    made = record.get("provenance") or {}
    same = (made.get("gpu_code") or {}).get("sha256") == gpu_code.get("sha256")
    return record if same and (made.get("code") or {}).get("dirty") is False else None


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run(settings, scope, cfg, args, out: Path) -> None:
    """The ``run`` part after its checks: the result, or a failure record and a non-zero exit."""
    from zero_field_refusal.runtime import environment

    env = guard_prompt_sets(environment(cfg, args.device))
    rule = apply_memory_rule(env.device, scope.memory_rule_gb)
    print(json.dumps({"memory_rule": rule}))
    failed = out / FAILED
    try:
        result = drift.run_drift(settings, cfg, env)
    except BaseException as exc:  # recorded whatever it is, then the exit
        write_json(failed, {
            "part": "run", "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
            "traceback": traceback.format_exc()[-8000:], "peaks": peak_memory(env.device),
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "provenance": {"spec": drift.SPEC, "code": drift.git_state(),
                           "gpu_code": drift.code_identity()},
        })  # fmt: skip
        if isinstance(exc, torch.OutOfMemoryError):
            first = str(exc).splitlines()[0] if str(exc) else ""
            print(f"run failed: out of memory under the {rule['limit_gb']:g} GB rule: {first}",
                  file=sys.stderr)  # fmt: skip
            sys.exit(OUT_OF_MEMORY)
        raise
    write_json(out / drift.RESULT, result)
    if failed.exists():  # an earlier attempt's failure, kept under another name
        failed.rename(failed.with_name(f"failed-before-{stamp()}.json"))
    print(json.dumps({name: {key: stats[key] for key in ("mean_norm2", "trace_cov", "ratio",
                                                           "half_cosine")}
                      for name, stats in result["optimizers"].items()}))  # fmt: skip
    print(json.dumps({"cross_cosines": result["cross_cosines"],
                      "gradients_identical": result["numerics"]["gradients_identical"],
                      "peak": result["peaks"]["all"]}))  # fmt: skip
    print(out / drift.RESULT)


def driver_script(commit: str, written_at: str, batch: int) -> str:
    """The bash driver of the diagnostic on a Pod (module docstring)."""
    stamp_now = "$(date -u +%FT%TZ)"
    args = f"--batch {batch}"
    lines = [
        "#!/usr/bin/env bash",
        *comment(f"Written by scripts/drift_diagnostic.py driver at {written_at}; do not edit."),
        *comment(f"The first-step drift diagnostic of study 2 (amendment 4, section 4) at B "
                 f"{batch}, from commit {commit}: the plan (the host memory), the streams of seed "
                 "2300 (generated if missing), then the run. Descriptive and blind to outcomes: "
                 "no harmful prompt set, stream or generation, no judge, no refusal."),
        *comment("Before it runs, DATA_DIR is restored with scripts/pod_setup.sh (the prompt "
                 "sets; the drift streams too once they are archived). The GPU must be free (no "
                 f"other compute process), the volume must hold {DISK_GB} GB, and the host "
                 "memory must reach the plan's requirement."),
        *comment(f"Outputs: $OUT (result.json), with this driver's log. It can be run again: a "
                 "result that counts is kept, and streams that exist are kept. Before any "
                 f"non-zero exit a failure is recorded in $OUT/{FAILED} (the run's own record, "
                 "or the driver's when the failing step left none); an earlier failure record is "
                 "kept under another name."),
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        f'OUT="${{RUNS_DIR:?}}/diagnostics/drift/b{batch}"',
        'mkdir -p "$OUT"',
        'exec > >(tee -a "$OUT/driver-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1',
        "# An earlier attempt's failure record is kept under another name",
        f'if [ -f "$OUT/{FAILED}" ]; then',
        f'  mv "$OUT/{FAILED}" "$OUT/failed-before-$(date -u +%Y%m%dT%H%M%SZ).json"',
        "fi",
        "# Every stop records the failure first, unless the failing step recorded its own",
        "stop() {",
        f'  echo "== {stamp_now} $1" >&2',
        f'  if [ ! -f "$OUT/{FAILED}" ]; then',
        f'    project-run "$SRC" python -c "{FAILURE_RECORD}" "$OUT/{FAILED}" "$1" "$2" || true',
        "  fi",
        '  exit "$2"',
        "}",
        "step() {",
        "  local name=$1 rc=0",
        "  shift",
        f'  echo "== {stamp_now} start: $name"',
        '  "$@" || rc=$?',
        '  if [ "$rc" -ne 0 ]; then stop "$name failed (exit $rc)" "$rc"; fi',
        f'  echo "== {stamp_now} done: $name"',
        "}",
        "# Any other failing command stops the driver through stop as well",
        f"trap 'stop \"a driver command failed (line $LINENO)\" {PROBLEM}' ERR",
        f'cd "$SRC" || stop "no source checkout at $SRC" {PROBLEM}',
        f'test "$(git rev-parse HEAD)" = {commit} || stop "the checkout is not at {commit}" '
        f"{PROBLEM}",
        f'project-run "$SRC" python -c "{CLEAN_CHECKOUT}" || stop "the checkout is not clean" '
        f"{PROBLEM}",
        "# No GPU job beside the diagnostic: no compute process, and (as a container may not see",
        "# other processes) no GPU memory in use",
        'if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then',
        f'  stop "a process holds the GPU; the diagnostic does not start" {PROBLEM}',
        "fi",
        "used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)",
        f"if echo \"$used\" | awk '$1 + 0 >= {GPU_USED_MIB} {{busy = 1}} END {{exit !busy}}'; then",
        f'  stop "GPU memory in use ($used MiB); the diagnostic does not start" {PROBLEM}',
        "fi",
        'free_kb=$(df -Pk "$OUT" | awk \'NR == 2 {print $4}\')',
        f'if [ "$free_kb" -lt {DISK_GB * 1_000_000} ]; then',
        f'  stop "{DISK_GB} GB free are needed under $OUT, $free_kb KB are" {PROBLEM}',
        "fi",
        "# The plan checks the free host memory against the requirement before anything starts",
        f"step plan {SCRIPT} plan {args}",
        f"step prepare {SCRIPT} prepare {args}",
        f"step run {SCRIPT} run {args}",
        f'echo "== {stamp_now} the drift diagnostic is complete: $OUT/{drift.RESULT}"',
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(*, untracked_config_for_tests: bool = False) -> None:
    """The command line (module docstring). ``untracked_config_for_tests`` lets the tests run
    ``run`` on settings other than the tracked ones; nothing on the command line does."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["prepare", "plan", "run", "driver"])
    parser.add_argument("--batch", type=int, required=True, help="the selected B")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--runs-root", type=Path, default=None, help="RUNS_DIR")
    parser.add_argument("--out", type=Path, default=None, help="output directory (driver: file)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit")
    args = parser.parse_args()

    settings = load_config(config_path(args.config))  # the file tracked_config judges
    try:
        scope = drift.DriftScope.of(settings)
        cfg = drift.run_config(scope, args.batch)
    except drift.DriftError as exc:
        sys.exit(str(exc))
    writes = args.part in ("run", "driver") and not args.dry_run
    if writes and not (tracked_config(args.config) or untracked_config_for_tests):
        sys.exit(f"{args.part} writes only under the tracked settings ({CONFIG}), not "
                 f"{config_path(args.config)}; --dry-run only prints")  # fmt: skip
    if args.part == "driver":
        script = driver_script(args.commit or "<SHA>", drift.git_state().get("commit")
                               or datetime.now(UTC).isoformat(timespec="seconds"),
                               args.batch)  # fmt: skip
        if args.dry_run:
            print(script, end="")
            return
        if not args.commit or not args.out or args.out.exists():
            sys.exit("driver needs --commit SHA and an --out FILE that does not exist")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(script)
        print(args.out)
        return

    out = args.out or drift.output_root(args.batch, args.runs_root)
    responses = scope.batches * args.batch
    if args.part in ("plan", "run") and args.dry_run:
        print("\n".join(describe(scope, cfg, out)))
        if (out / drift.RESULT).exists():
            print(f"a result exists: {out / drift.RESULT}")
        return

    from zero_field_refusal.runtime import environment

    if args.part == "prepare":
        device = "cpu" if args.dry_run else args.device  # a dry run loads no model
        env = guard_prompt_sets(environment(cfg, device))
        if args.dry_run:
            print("\n".join(stream_lines(drift.stream_state(cfg, env, responses))))
            return
        state = drift.prepare_streams(cfg, env, responses)
        print("\n".join([*stream_lines(state), *(f"generated: {p}" for p in state["generated"])]))
        return
    if args.part == "plan":
        print("\n".join(describe(scope, cfg, out)))
        env = guard_prompt_sets(environment(cfg, "cpu"))
        print("\n".join(stream_lines(drift.stream_state(cfg, env, responses))))
        check = drift.host_memory_check(scope)
        print(host_summary(check))
        if not check["sufficient"]:
            sys.exit(PROBLEM)
        return

    state = require_clean()
    gpu_code = drift.code_identity()
    if counted(out, gpu_code) is not None:
        print(f"kept: {out / drift.RESULT} was written with this GPU code "
              f"({gpu_code.get('sha256')})")  # fmt: skip
        return
    check = drift.host_memory_check(scope)
    print(host_summary(check))
    if not check["sufficient"]:
        write_json(out / FAILED, {
            "part": "run", "error": host_summary(check), "host_memory": check,
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "provenance": {"spec": drift.SPEC, "code": state, "gpu_code": gpu_code},
        })  # fmt: skip
        sys.exit(HOST_MEMORY)
    if (out / drift.RESULT).exists():  # a result that does not count, kept aside
        try:
            record = json.loads((out / drift.RESULT).read_text())
            commit = ((record.get("provenance") or {}).get("code") or {}).get("commit")
        except (OSError, json.JSONDecodeError, AttributeError):
            commit = "unreadable"
        set_aside(out, drift.RESULT, f"{commit}-{stamp()}")
    run(settings, scope, cfg, args, out)


if __name__ == "__main__":
    main()
