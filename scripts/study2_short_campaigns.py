"""Write the campaigns of study 2's short-regime stage (amendment 3, design A) from the run records,
and the shell driver of their prerequisites.

    python scripts/study2_short_campaigns.py targeted --runs DIR... --theta0 RUN \
        (--w-max W | --fixation FILE) [--seeds S ...] [--cells NAME ...] --out FILE
    python scripts/study2_short_campaigns.py KIND --runs DIR... --theta0 RUN --fixation FILE \
        [--runs-root RUNS_DIR] [--seeds S ...] [--cells NAME ...] --out FILE
    python scripts/study2_short_campaigns.py confirmation --blocks FIRST LAST [--after-look K] \
        --runs DIR... --theta0 RUN --fixation FILE --runs-root RUNS_DIR [--cells NAME ...] \
        --out FILE
    python scripts/study2_short_campaigns.py prep --seeds S [S ...] --batch B --commit SHA \
        [--archived] [--once --stage0-run-id ID] --out FILE
    python scripts/study2_short_campaigns.py check-streams --seeds S [S ...] --batch B
    python scripts/study2_short_campaigns.py (soft-targets-aside | soft-targets-restore) \
        --stage0-run-id ID [--aside-root DIR]

KIND is the campaign the stage's state machine names (analysis.study2_short.progress), and each
kind refuses when it names another step, or when the records are not whole: ``targeted`` or
``repair`` (a round of discovery runs at the rates it prescribes, on seeds 2000-2002, of the
primary arms or of the pre-warmed arm), ``controls`` (one FA soft-target control per selected FA
rate on seed 2000, experiments/study2_short_fa_controls with schedule.instability.contamination_k
= 0.01 × K*), ``prewarmed`` (the pre-warmed arm's first discovery runs, at the union of the frozen
AdamW and FA rates), ``validation`` (seeds 2100-2111, every arm at its frozen rates) and
``confirmation`` (the blocks FIRST to LAST, 1-80, in the preassigned seed order 3000-3079, every
arm at its frozen rates, once the validation passed; blocks 1-10 run
experiments/study2_short_rayleigh, which measures the endpoint Rayleigh quotient). A campaign
holds the runs still missing; ``--seeds`` and ``--cells`` only split it over Pods.

- --runs holds every run so far: Stage A's (study2_stage_a, study2_controls), whose 25-step grids
  are the first discovery runs and whose version-1 discovery set fixes the stream time, and this
  stage's (study2_short, study2_short_fa_controls). --theta0 is the θ₀ run (summary.json).
- **w_max** is the simulation's: the fixation record (--fixation, ``read_fixation``) is required
  for every kind but ``targeted``, whose rates do not depend on it; there --w-max may stand in.
  A --w-max given beside a record must equal it. Every campaign's header records w_max, where it
  came from and the code revision. Fix w_max before any repair round: rounds already made under
  another w_max leave the stage incomplete.
- --runs-root is RUNS_DIR, whose launch/ journal (``launch.launch_records``) tells which runs the
  launcher gave up on after their recovery round (``exhausted_runs``: technical failures of
  their blocks) and which confirmation ranges were issued; validation and confirmation need it.
- **The confirmation loop** (``confirmation_guard``; the looks at 36 and 60 supported blocks, cap
  80): (1) write and launch at most the blocks the guard allows: the next range starts at the
  first block without a run and ends before the next look could fall; (2) let them complete;
  (3) read the support counts (scripts/study2_short_summary.py); (4) label the runs; (5) run the
  summary with --judge and --fixation; (6) go on only if the look did not stop, passing
  ``--after-look K`` for look K. Re-issuing exactly a range written before completes it.

Nothing here reads a judge label or K_harmful.

``prep`` writes a bash driver of the prerequisites of the runs of the given seeds at batch B, from
the source checkout of commit SHA on a Pod, laid out as the Stage-A drivers. With ``--once``
(and its ``--stage0-run-id``) it starts with the stage-0 checks under protocol version 3 and the
curvature panel. Then, per seed, either it checks that the seed's streams exist (``--archived``:
seeds whose streams are archived, nothing is generated, a missing stream stops it) or it
generates them: the training stream to 25 × B responses (the stream is appendable; the default
length, study2_stage_a's 4000 steps at its batch 20, is not needed) and v₀'s preconditioner
stream at B. The driver can be run again after a failure. ``check-streams`` is that check, which
the driver runs on the Pod through ``runtime.locate_stream``. FILE must not exist; write
campaigns under an ignored directory such as $RUNS_DIR/campaigns/.

Stage 0 under a new protocol (docs/operations.md) needs the distributions of its path-field
check's soft-target stream, which the retention policy deleted: DATA_DIR holds only the restored
manifest and samples. Around the stage-0 checks the driver runs:

- ``soft-targets-aside`` before them: the restored copy of the stream, located as stage 0 locates
  it (``checks.path_field_config``, ``runtime.locate_stream``, under DATA_DIR), moves to
  ``--aside-root``/soft-targets-aside-YYYYMMDD/ (default /workspace/tmp), never deleted, so that
  stage 0 generates the stream again. A copy already set aside is kept there: nothing moves.
- ``soft-targets-restore`` once the run passed: the sha256 of the new samples and distributions
  against the archived manifest (its ``sha256`` and its distributions' hash), written with the
  regenerated manifest into the stage-0 run's directory (soft_targets_regeneration.json, laid
  out as the record of 2026-10-14); then the regenerated distributions are deleted, the rest of
  the regenerated stream moves into the dated directory (under the run's id), and the archived
  copy goes back, so that DATA_DIR matches its inventories. Each step is skipped once done, so
  the driver can be run again at any point.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import textwrap
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from zero_field_refusal.analysis.stage_a import BATCHES, load_trajectories, rate, theta0_problems
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    ARMS,
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    FA,
    FA_CONTROL_ARM,
    FA_CONTROLS,
    LOOK_SIZES,
    PREWARMED,
    RAYLEIGH,
    REGULAR,
    STEPS,
    VALIDATION_SEEDS,
    confirmation_guard,
    exhausted_runs,
    issued_ranges,
    measures_rayleigh,
    progress,
    read_fixation,
    run_facts,
    seed_role,
)
from zero_field_refusal.launch import launch_records
from zero_field_refusal.tracking import git_state

KINDS = ("targeted", "repair", "controls", "prewarmed", "validation", "confirmation")
JOURNAL_KINDS = ("validation", "confirmation")  # past the selection: the journal is read
OPTIMIZER = {ADAMW: [], FA: ["optimizer=adam_frozen"], PREWARMED: ["optimizer=adamw_prewarmed"]}
STAGE_SEEDS = (*DISCOVERY_SEEDS, *VALIDATION_SEEDS, *CONFIRMATION_SEEDS)
STAGE_A = "experiments/study2_stage_a"  # the streams' config, as Stage A generated them
TEACHER = 'project-run "$SRC" python scripts/generate_teacher.py'
# generate_teacher.py's refusal of a stream that exists (runtime.generate_stream).
EXISTS = "streams are the record and are never regenerated"
# A stage-0 run counts only if it completed, ran under protocol version 3 and its path-field check
# passed: a completed run can still have failed the check (review r34).
PASSED = (
    "import json, sys; from pathlib import Path; from omegaconf import OmegaConf; "
    "d = Path(sys.argv[1]); m = json.loads((d / 'meta.json').read_text()); "
    "s = json.loads((d / 'summary.json').read_text()); "
    "c = OmegaConf.load(d / 'config.yaml'); "
    "sys.exit(not (m.get('status') == 'completed' and c.get('protocol_version') == 3 "
    "and s.get('path_field', {}).get('passed') is True))"
)
# The stage-0 checks the driver runs, and where the restored soft-target copy waits meanwhile.
STAGE0_CONFIG = "experiments/stage0_checks"
STAGE0_OVERRIDES = ("model=qwen2.5-1.5b-instruct", "protocol_version=3")
ASIDE_ROOT = Path("/workspace/tmp")  # on the Pod's volume, as DATA_DIR
ASIDE_PREFIX = "soft-targets-aside-"  # + the UTC date
SOFT_TARGET_KINDS = ("soft-targets-aside", "soft-targets-restore")
REGENERATION_RECORD = "soft_targets_regeneration.json"  # in the stage-0 run's directory
REGENERATED_MANIFEST = "soft_targets_manifest_regenerated.json"


def cell(name: str, config: str, overrides: list[str], seeds: list[int]) -> dict:
    return {"name": name, "config": config, "overrides": overrides, "seeds": seeds}


def regular_cells(runs: list[dict], suffix: str = "") -> list[dict]:
    """One cell per arm, rate and config (the first 10 confirmation blocks measure R), over its
    seeds, with the role of its seeds. The launcher grants a recovery round once per experiment
    and cell name, so a name never recurs in another of the stage's campaigns: ``suffix`` names
    the validation or the confirmation blocks."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in runs:
        seed = r["seed"]
        key = (ARMS.index(r["arm"]), r["batch"], r["lr"], measures_rayleigh(seed), seed_role(seed))
        groups[key].append(seed)
    items = []
    for (arm_index, batch, lr, rayleigh, role), seeds in sorted(groups.items()):
        arm = ARMS[arm_index]
        overrides = [*OPTIMIZER[arm], f"train.batch_size={batch}", f"optimizer.lr={lr:g}",
                     f"study2_short.role={role}"]  # fmt: skip
        name = f"{arm}-b{batch}-{lr:g}{suffix}" + ("-rayleigh" if rayleigh else "")
        items.append(cell(name, RAYLEIGH if rayleigh else REGULAR, overrides, sorted(seeds)))
    return items


def control_cells(runs: list[dict], upper: float) -> list[dict]:
    """One FA soft-target control per rate on seed 2000, its bound 0.01 × K* written in full;
    a rate may have a control under each U, so the cell's name carries U."""
    items = []
    for r in sorted(runs, key=lambda r: r["lr"]):
        overrides = [f"train.batch_size={r['batch']}", f"optimizer.lr={r['lr']:g}",
                     f"schedule.instability.contamination_k={r['contamination_k']!r}"]  # fmt: skip
        items.append(cell(f"fa-control-b{r['batch']}-{r['lr']:g}-u{upper:g}", FA_CONTROLS,
                          overrides, [r["seed"]]))  # fmt: skip
    return items


def revision() -> str:
    """The code revision that writes a file: its commit, and whether the tree was clean."""
    state = git_state()
    commit = state.get("commit") or "an unknown commit"
    return commit + (" with uncommitted changes" if state.get("dirty") else "")


def comment(text: str) -> list[str]:
    """``text`` as comment lines of at most 100 characters, broken at spaces only."""
    return textwrap.wrap(text, 100, initial_indent="# ", subsequent_indent="# ",
                         break_long_words=False, break_on_hyphens=False)  # fmt: skip


def prep_driver(
    seeds: list[int],
    batch: int,
    commit: str,
    *,
    once: bool,
    stage0_run_id: str | None,
    archived: bool,
    written_at: str,
) -> str:
    """The bash driver of the runs' prerequisites (see the module docstring)."""
    stamp = "$(date -u +%FT%TZ)"
    listed = " ".join(map(str, seeds))
    streams = (f"checks that the training and preconditioner streams of seeds {listed} at batch "
               f"{batch} exist, generating nothing" if archived else
               f"generates the training and preconditioner streams of seeds {listed} at batch "
               f"{batch}")  # fmt: skip
    steps = ("runs the stage-0 checks under protocol version 3 (with the restored soft-target "
             "copy set aside, compared and put back) and generates the curvature panel, then "
             f"{streams}") if once else streams  # fmt: skip
    lines = [
        "#!/usr/bin/env bash",
        *comment(f"Written by scripts/study2_short_campaigns.py prep at {written_at}; do not "
                 "edit."),
        *comment(f"Study 2's short-regime stage, from commit {commit}: it {steps}."),
        *comment("Before it runs, DATA_DIR is restored with scripts/pod_setup.sh from the "
                 "inventories of the prompt sets, the teacher streams (the training streams of "
                 "the seeds already run, with their chunks), the monitor-v2 stream, the "
                 "preconditioner streams and the protocol-2 path-field record (the stage-0 "
                 "record), so that the runs read the streams that Stage A's runs read."),
        *comment("It stops at the first failure and can be run again: a completed stage-0 run "
                 "is kept, a soft-target copy set aside or put back stays so, a stream that "
                 "exists is kept, and a training stream only grows to what the runs read."),
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        'test -z "$(git status --porcelain)"',
        f'step() {{ echo "== {stamp} start: $*"; "$@"; echo "== {stamp} done: $*"; }}',
        "kept() {",
        "  local log",
        "  log=$(mktemp)",
        f'  echo "== {stamp} start: $*"',
        f'  if "$@" 2>&1 | tee "$log"; then echo "== {stamp} done: $*"',
        f'  elif grep -q "{EXISTS}" "$log"; then',
        f'    echo "== {stamp} kept, it exists: $*"',
        '  else rm -f "$log"; return 1; fi',
        '  rm -f "$log"',
        "}",
    ]  # fmt: skip
    if once:
        soft_targets = 'project-run "$SRC" python scripts/study2_short_campaigns.py soft-targets'
        lines += [
            f"STAGE0={stage0_run_id}",
            f"PASSED=\"{PASSED}\"",
            "stage0() {",
            '  local dir="${RUNS_DIR:?}/stage0_checks/$STAGE0"',
            '  if [ ! -e "$dir" ]; then',
            "    # The restored soft-target copy lacks its distributions: stage 0 makes them anew.",
            f'    step {soft_targets}-aside --stage0-run-id "$STAGE0"',
            '    step env RUN_ID="$STAGE0" project-run "$SRC" python scripts/stage0_checks.py \\',
            f"      {STAGE0_CONFIG} {' '.join(STAGE0_OVERRIDES)}",
            "  fi",
            '  if project-run "$SRC" python -c "$PASSED" "$dir"; then',
            f'    echo "== {stamp} stage-0 run $STAGE0 passed under protocol version 3"',
            "  else",
            f'    echo "== {stamp} stage-0 run $STAGE0 did not complete, or did not pass under" \\',
            '      "protocol version 3: read it, then run again with a new --stage0-run-id" >&2',
            "    return 1",
            "  fi",
            f'  step {soft_targets}-restore --stage0-run-id "$STAGE0"',
            "}",
        ]  # fmt: skip
    lines.append("# The steps")
    if once:
        lines += ["stage0", f"kept {TEACHER} {RAYLEIGH} \\", "  --stream curvature_panel"]
    if archived:
        lines += [
            'step project-run "$SRC" python scripts/study2_short_campaigns.py check-streams \\',
            f"  --seeds {listed} --batch {batch}",
        ]
    else:
        for seed in seeds:
            lines += [
                f"step {TEACHER} {STAGE_A} --stream training \\",
                f"  --seeds {seed} --responses {STEPS * batch}",
                f"kept {TEACHER} {STAGE_A} \\",
                f"  optimizer=adam_frozen train.batch_size={batch} --stream preconditioner \\",
                f"  --seeds {seed}",
            ]
    lines.append(f'echo "== {stamp} the prerequisites are in place"')
    return "\n".join(lines) + "\n"


def check_streams(seeds: list[int], batch: int) -> tuple[list[str], list[str]]:
    """Where each seed's training stream (enough chunks for its runs) and v₀'s preconditioner
    stream at the batch are, located as every run locates them, and which are missing."""
    from zero_field_refusal.protocol import load_run_config
    from zero_field_refusal.runtime import environment, locate_stream
    from zero_field_refusal.teacher import (
        MANIFEST_FILE,
        chunk_directory,
        load_manifest,
        published_chunks,
    )

    training = load_run_config(STAGE_A)
    preconditioner = load_run_config(STAGE_A, ["optimizer=adam_frozen",
                                               f"train.batch_size={batch}"])  # fmt: skip
    env = environment(training, "cpu")
    need = -(-STEPS * batch // int(training.teacher.chunk_size))
    found, missing = [], []
    for seed in seeds:
        _, path, _ = locate_stream(training, env, "training", seed)
        have = published_chunks(path)
        if have < need:
            missing.append(f"seed {seed}: training stream {path} holds {have} chunks, the runs "
                           f"read {need}")  # fmt: skip
        else:
            digest = load_manifest(chunk_directory(path, 0))["sha256"]
            found.append(f"seed {seed}: training {path}: {have} chunks, chunk 0 sha256 {digest}")
        _, path, _ = locate_stream(preconditioner, env, "preconditioner", seed)
        if not (path / MANIFEST_FILE).is_file():
            missing.append(f"seed {seed}: no preconditioner stream at {path}")
        else:
            found.append(f"seed {seed}: preconditioner {path}: sha256 "
                         f"{load_manifest(path)['sha256']}")  # fmt: skip
    return found, missing


# ------------------------------------------------------------------ stage 0's soft-target stream


def stage0_soft_targets() -> tuple[Path, Path]:
    """The path-field check's soft-target stream (seed 0) of the stage-0 checks the driver runs,
    located as stage 0 locates it, and DATA_DIR."""
    from zero_field_refusal.checks import path_field_config
    from zero_field_refusal.protocol import load_run_config
    from zero_field_refusal.runtime import environment, locate_stream

    cfg = load_run_config(STAGE0_CONFIG, list(STAGE0_OVERRIDES))
    env = environment(cfg, "cpu")
    _, path, _ = locate_stream(path_field_config(cfg), env, "soft_targets", 0)
    return path, env.data_root


def set_aside_copies(path: Path, data_root: Path, aside_root: Path) -> list[Path]:
    """The copies of the stream at ``path`` set aside under ``aside_root``, at most one."""
    relative = path.relative_to(data_root)
    found = sorted(p for p in aside_root.glob(f"{ASIDE_PREFIX}*/{relative}") if p.is_dir())
    if len(found) > 1:
        sys.exit(f"several copies of {relative} are set aside: {[str(p) for p in found]}; keep "
                 "the archived one only")  # fmt: skip
    return found


def set_aside(path: Path, data_root: Path, aside_root: Path, today: str) -> str:
    """Move the restored copy of the stream at ``path`` under ``aside_root`` (dated ``today``),
    unless a copy is already set aside there; then nothing moves."""
    from zero_field_refusal.teacher import MANIFEST_FILE

    found = set_aside_copies(path, data_root, aside_root)
    if found:
        return f"kept: the restored copy is already set aside at {found[0]}"
    if not (path / MANIFEST_FILE).is_file():
        sys.exit(f"no restored soft-target copy at {path}, and none set aside under {aside_root}: "
                 "restore DATA_DIR with scripts/pod_setup.sh first")  # fmt: skip
    target = aside_root / f"{ASIDE_PREFIX}{today}" / path.relative_to(data_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, target)
    return f"set aside: {path} -> {target}"


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    out: dict[str, object] = {}
    for key, item in value.items():
        out.update(_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
    return out


def regeneration_record(
    archived: Path, regenerated: Path, run_dir: Path, *, stream: Path, kept: Path
) -> dict:
    """The comparison of the regenerated stream with the archived copy's manifest, laid out as
    runs/runpod/evidence/soft-targets-regeneration-20261014.json (``archived`` for its ``pilot``:
    the copy set aside, whatever made it)."""
    from zero_field_refusal.teacher import (
        DISTRIBUTIONS_FILE,
        MANIFEST_FILE,
        SAMPLES_FILE,
        file_sha256,
    )

    old = json.loads((archived / MANIFEST_FILE).read_text())
    new = json.loads((regenerated / MANIFEST_FILE).read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    samples = file_sha256(regenerated / SAMPLES_FILE)
    distributions = file_sha256(regenerated / DISTRIBUTIONS_FILE)
    old_leaves, new_leaves = _leaves(old), _leaves(new)
    differing = sorted(key for key in old_leaves.keys() | new_leaves.keys()
                       if old_leaves.get(key) != new_leaves.get(key))  # fmt: skip

    def facts(manifest: dict) -> dict:
        return {"driver": (manifest.get("hardware") or {}).get("driver"),
                "generated": str(manifest.get("started_at") or "")[:10] or None}  # fmt: skip

    commit = (meta.get("git") or {}).get("commit")
    return {
        "what": f"Stage 0 under protocol version 3 (run {run_dir.name}, commit {commit}) "
        f"regenerated the soft-target stream {stream} of its path-field check, whose "
        "distributions the retention policy had deleted. scripts/study2_short_campaigns.py "
        f"(prep --once) set the restored copy (manifest and samples) aside at {archived} "
        "before the run and compared the regenerated stream with it after the run passed.",
        "pod": os.environ.get("RUNPOD_POD_ID"),
        "archived": {
            **facts(old),
            "manifest_file_sha256": file_sha256(archived / MANIFEST_FILE),
            "samples_sha256": old.get("sha256"),
            "distributions_sha256": (old.get("distributions") or {}).get("sha256"),
        },
        "regenerated": {
            **facts(new),
            "manifest_file_sha256": file_sha256(regenerated / MANIFEST_FILE),
            "manifest_file": REGENERATED_MANIFEST,
            "samples_sha256": samples,
            "distributions_sha256": distributions,
            "distributions_bytes": (regenerated / DISTRIBUTIONS_FILE).stat().st_size,
            "distributions_sha256_measured_by": "teacher.file_sha256 of the file on the Pod",
            "distributions_sha256_in_manifest": (new.get("distributions") or {}).get("sha256"),
        },
        "comparison": {
            "samples_equal": samples == old.get("sha256"),
            "distributions_equal": distributions == (old.get("distributions") or {}).get("sha256"),
            "manifest_fields_differing": differing,
        },
        "after": f"{DISTRIBUTIONS_FILE} deleted after the check (retention policy); the archived "
        "copy put back in DATA_DIR, so DATA_DIR matches the existing inventories; the rest of "
        f"the regenerated stream (manifest and samples) kept at {kept}.",
    }


def _write_new(path: Path, text: str) -> None:
    """Write ``path`` whole or not at all."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text)
    partial.replace(path)


def restore(path: Path, data_root: Path, aside_root: Path, run_dir: Path) -> str:
    """After a passed stage-0 run: the comparison into ``run_dir``, then the regenerated
    distributions deleted, the rest of the regenerated stream moved beside the set-aside copy and
    that copy put back at ``path``. Each step done before is skipped."""
    from zero_field_refusal.teacher import DISTRIBUTIONS_FILE, MANIFEST_FILE

    found = set_aside_copies(path, data_root, aside_root)
    record = run_dir / REGENERATION_RECORD
    distributions = path / DISTRIBUTIONS_FILE
    if not found:
        if record.is_file():
            return f"kept: the archived copy is back at {path}; the comparison is {record}"
        if distributions.exists():
            sys.exit(f"{path} holds distributions and no copy is set aside under {aside_root}: "
                     "compare and clean up by hand (docs/operations.md, Stage 0 under a new "
                     "protocol)")  # fmt: skip
        return (f"nothing set aside for run {run_dir.name} and no regenerated distributions at "
                f"{path}: nothing to compare or restore")  # fmt: skip
    archived = found[0]
    relative = path.relative_to(data_root)
    kept = aside_root / archived.relative_to(aside_root).parts[0] / run_dir.name / relative
    done = []
    if not record.is_file():
        if not distributions.is_file():
            sys.exit(f"no regenerated distributions at {path} to compare with {archived}")
        found_record = regeneration_record(archived, path, run_dir, stream=relative, kept=kept)
        _write_new(run_dir / REGENERATED_MANIFEST, (path / MANIFEST_FILE).read_text())
        _write_new(record, json.dumps(found_record, indent=2) + "\n")
        compared = found_record["comparison"]
        done.append(f"compared: samples equal {compared['samples_equal']}, distributions equal "
                    f"{compared['distributions_equal']} ({record})")  # fmt: skip
    if distributions.exists():
        distributions.unlink()
        done.append(f"deleted {distributions}")
    if path.exists():
        if kept.exists():
            sys.exit(f"{kept} exists: the regenerated stream at {path} has nowhere to go")
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, kept)
        done.append(f"kept the rest of the regenerated stream at {kept}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(archived, path)
    done.append(f"put back: {archived} -> {path}")
    return "\n".join(done)


def prep(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    if not args.seeds or args.batch is None or not args.commit:
        parser.error("prep needs --seeds, --batch and --commit")
    if len(args.commit) != 40 or set(args.commit) - set("0123456789abcdef"):
        parser.error("--commit takes the full 40-character commit hash")
    if args.once != bool(args.stage0_run_id):
        parser.error("--once and --stage0-run-id go together: the stage-0 run's id names the GPU "
                     "and the day, and has no default")  # fmt: skip
    return prep_driver(args.seeds, args.batch, args.commit, once=args.once,
                       stage0_run_id=args.stage0_run_id, archived=args.archived,
                       written_at=revision())  # fmt: skip


def w_max_of(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[float, str]:
    """w_max and where it came from: the fixation record, required but for ``targeted``."""
    if args.fixation is None:
        if args.kind != "targeted":
            parser.error(f"{args.kind} needs the fixation record (--fixation): its w_max is the "
                         "simulation's, fixed before any repair round")  # fmt: skip
        if args.w_max is None or not args.w_max > 0:
            parser.error("w_max is needed: --w-max or --fixation")
        return args.w_max, "--w-max"
    try:
        fixed = read_fixation(args.fixation)["w_max"]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.w_max is not None and args.w_max != fixed:
        parser.error(f"--w-max {args.w_max} differs from the fixation record's {fixed}")
    digest = hashlib.sha256(args.fixation.read_bytes()).hexdigest()
    return fixed, f"the fixation record {args.fixation}, sha256 {digest}"


def campaign(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[dict, float, str]:
    if args.theta0 is None or not args.runs:
        parser.error(f"{args.kind} needs --runs and --theta0")
    if (args.kind == "confirmation") != (args.blocks is not None):
        parser.error("--blocks goes with confirmation, and confirmation needs it")
    if args.after_look and args.kind != "confirmation":
        parser.error("--after-look goes with confirmation")
    if args.kind in ("confirmation", "controls") and args.seeds:
        parser.error("--seeds splits discovery and validation campaigns only")
    if args.kind in JOURNAL_KINDS and args.runs_root is None:
        parser.error(f"{args.kind} needs --runs-root, whose launch journal tells which runs the "
                     "launcher gave up on")  # fmt: skip
    w_max, source = w_max_of(parser, args)
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary.get(key) for key in ("benchmark_accuracy", "factual_accuracy")}
    if theta0_problems(theta0):
        parser.error("; ".join(theta0_problems(theta0)))
    runs = load_trajectories(*args.runs)
    facts = {r.run_id: run_facts(r.run_dir) for r in runs if r.arm in (*ARMS, FA_CONTROL_ARM)}
    records = launch_records(args.runs_root) if args.runs_root is not None else []
    exhausted = exhausted_runs(records)
    state = progress(runs, theta0, w_max, facts=facts, exhausted=exhausted)
    step = state["next"]
    if state["problems"]:
        parser.error("the stage's records are not whole: " + "; ".join(state["problems"]))
    if step["campaign"] != args.kind:
        shown = {k: v for k, v in step.items() if k != "runs"}
        parser.error(f"the stage's next step is {step['campaign']}, not {args.kind}: {shown}")
    batch, upper = step["batch"], step["U"]
    if args.kind == "controls":
        written = {"name": f"study2-short-controls-b{batch}-u{upper:g}",
                   "cells": control_cells(step["runs"], upper)}  # fmt: skip
        return written, w_max, source
    if args.kind == "confirmation":
        first, last = args.blocks
        guard = confirmation_guard(state["confirmation"], first, last,
                                   after_look=args.after_look or 0,
                                   issued=issued_ranges(records))  # fmt: skip
        if guard["problems"]:
            parser.error(f"blocks {first}-{last}: " + "; ".join(guard["problems"]))
        seeds = CONFIRMATION_SEEDS[first - 1 : last]
        present = {(r.arm, r.batch_size, rate(r.lr), r.seed) for r in runs
                   if r.arm in ARMS and r.seed in seeds} | set(exhausted)  # fmt: skip
        wanted = [{"arm": arm, "batch": batch, "lr": rate(lr), "seed": seed}
                  for seed in seeds for arm, lrs in step["rates"].items() for lr in lrs
                  if (arm, batch, rate(lr), seed) not in present]  # fmt: skip
        if not wanted:
            parser.error(f"blocks {first}-{last} have every run already")
        written = {"name": f"study2-short-confirmation-{first}-{last}",
                   "cells": regular_cells(wanted, f"-blocks-{first}-{last}")}  # fmt: skip
        return written, w_max, source
    wanted = step["runs"]
    if args.seeds:
        unknown = sorted(set(args.seeds) - {r["seed"] for r in wanted})
        if unknown:
            parser.error(f"no run of seeds {unknown} in this campaign")
        wanted = [r for r in wanted if r["seed"] in args.seeds]
    # The pre-warmed arm's own rounds, apart from the primary arms' of the same batch and U.
    own = args.kind in ("targeted", "repair") and {r["arm"] for r in wanted} == {PREWARMED}
    kind = f"{args.kind}-prewarmed" if own else args.kind
    suffix = "-validation" if args.kind == "validation" else ""
    written = {"name": f"study2-short-{kind}-b{batch}-u{upper:g}",
               "cells": regular_cells(wanted, suffix)}  # fmt: skip
    return written, w_max, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("prep", "check-streams", *SOFT_TARGET_KINDS, *KINDS))
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--theta0", type=Path, help="the θ₀ run directory (summary.json)")
    parser.add_argument("--w-max", type=float, help="targeted: w_max without a fixation record")
    parser.add_argument("--fixation", type=Path, help="the simulation's fixation record (JSON)")
    parser.add_argument("--runs-root", type=Path, help="RUNS_DIR, whose launch/ journal is read")
    parser.add_argument("--seeds", type=int, nargs="+", help="keep only these seeds (one Pod)")
    parser.add_argument("--cells", nargs="+", help="keep only these cells (one Pod each)")
    parser.add_argument("--blocks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="confirmation: the blocks FIRST to LAST, 1-80")  # fmt: skip
    parser.add_argument("--after-look", type=int, choices=range(1, len(LOOK_SIZES) + 1),
                        help="confirmation: look K was read and did not stop")  # fmt: skip
    parser.add_argument("--batch", type=int, choices=BATCHES, help="prep: the batch size")
    parser.add_argument("--commit", help="prep: the commit whose source checkout the Pod runs")
    parser.add_argument(
        "--archived",
        action="store_true",
        help="prep: the seeds' streams are archived: check them, generate nothing",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="prep: also the stage-0 checks and the curvature panel",
    )
    parser.add_argument("--stage0-run-id", help="prep with --once: RUN_ID of the stage-0 checks")
    parser.add_argument("--aside-root", type=Path, default=ASIDE_ROOT,
                        help="soft-targets-*: where the restored copy waits")  # fmt: skip
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = args.seeds or []
    if set(seeds) - set(STAGE_SEEDS) or len(set(seeds)) != len(seeds):
        parser.error("--seeds: distinct seeds of the stage (2000-2002, 2100-2111, 3000-3079)")
    if args.kind in SOFT_TARGET_KINDS:
        if not args.stage0_run_id:
            parser.error(f"{args.kind} needs --stage0-run-id")
        from zero_field_refusal.paths import runs_root

        path, data_root = stage0_soft_targets()
        if args.kind == "soft-targets-aside":
            print(set_aside(path, data_root, args.aside_root,
                            datetime.now(UTC).strftime("%Y%m%d")))  # fmt: skip
        else:
            run_dir = runs_root() / "stage0_checks" / args.stage0_run_id
            print(restore(path, data_root, args.aside_root, run_dir))
        return
    if args.kind == "check-streams":
        if not args.seeds or args.batch is None:
            parser.error("check-streams needs --seeds and --batch")
        found, missing = check_streams(args.seeds, args.batch)
        print("\n".join(found))
        if missing:
            sys.exit("missing streams: restore them from the archive first:\n" + "\n".join(missing))
        return
    if args.out is None:
        parser.error(f"{args.kind} needs --out")
    if args.out.exists():
        parser.error(f"{args.out} exists")
    if args.kind == "prep":
        text = prep(parser, args)
    else:
        written, w_max, source = campaign(parser, args)
        if args.cells:  # a campaign split over Pods, cell by cell, without choosing anything
            unknown = sorted(set(args.cells) - {c["name"] for c in written["cells"]})
            if unknown:
                parser.error(f"no such cells in this campaign: {unknown}")
            written["cells"] = [c for c in written["cells"] if c["name"] in args.cells]
        writer = f"scripts/study2_short_campaigns.py {args.kind} at {revision()}"
        header = comment(f"Written by {writer}, with w_max {w_max!r} from {source}. Do not edit.")
        text = "\n".join(header) + "\n" + yaml.safe_dump(written, sort_keys=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(args.out)


if __name__ == "__main__":
    main()
