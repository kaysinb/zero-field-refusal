"""Write the campaigns of study 2's short-regime stage (amendment 3, design A, its batch and rates
selected under amendment 4) from the run records and the numerical-check verdicts, and the shell
driver of their prerequisites.

    python scripts/study2_short_campaigns.py KIND --runs DIR... --theta0 RUN --fixation FILE \
        --runs-root RUNS_DIR [--abandoned FILE] [--versions FILE] [--seeds S ...] \
        [--cells NAME ...] --out FILE
    python scripts/study2_short_campaigns.py confirmation --blocks FIRST LAST [--after-look K] \
        --runs DIR... --theta0 RUN --fixation FILE --runs-root RUNS_DIR [--abandoned FILE] \
        [--versions FILE] [--cells NAME ...] --out FILE
    python scripts/study2_short_campaigns.py timecourse --runs DIR... --theta0 RUN \
        --fixation FILE --runs-root RUNS_DIR [--abandoned FILE] [--versions FILE] \
        [--seeds S ...] [--cells NAME ...] --out FILE
    python scripts/study2_short_campaigns.py prep --seeds S [S ...] --batch B --commit SHA \
        [--archived] [--once --stage0-run-id ID] --out FILE
    python scripts/study2_short_campaigns.py check-streams --seeds S [S ...] --batch B
    python scripts/study2_short_campaigns.py (soft-targets-aside | soft-targets-restore) \
        --stage0-run-id ID [--aside-root DIR]

KIND is the campaign the stage's state machine names (analysis.study2_short.progress), and each
kind refuses when it names another step, or when the records are not whole. The candidates
(U, B) come in amendment 5's order (8, 20), (7, 20), (6, 20), each for the three arms (B = 20 is
the study's fixed scope: no B = 5 candidate is ever prescribed):
``discovery`` (the primary arms' targeted round on seeds 2000-2002, holding the pre-warmed arm's
first round, at the union of the rates of the primary arms' targeted assignments, when it is
due), ``repair`` (a repair round, of the primary arms or of the pre-warmed arm; or, after a
numerical exclusion left an arm without a feasible triple, its search allowance's one repair
round, named ``...-allowance-<check>`` after the check that spent it, its cells new rates of
their arm and batch), ``prewarmed``
(the pre-warmed arm's first round alone, at the union of the selected AdamW and FA rates),
``targeted`` (the pre-warmed arm's own targeted round), ``numerical`` (the regular runs of the
numerical qualification's cases, role numerical, under experiments/study2_short_k_only: K-only,
no response to a harmful prompt; every selected rate on seed 2200 and each arm's U-rate on 2201
and 2202; the streams of those seeds at B come first, from ``prep``),
``validation`` (seeds 2100-2111, every arm at its selected rates) and ``confirmation`` (the blocks
FIRST to LAST, 1-80, in the preassigned seed order 3000-3079, every arm at its frozen rates, once
a candidate's validation passed; blocks 1-10 run experiments/study2_short_rayleigh, which
measures the endpoint Rayleigh quotient). Between ``numerical`` and ``validation`` the step is
``numerical-check``, no launcher campaign: the check of scripts/numerical_check.py, with the
rates, the cases and the verdict path under $RUNS_DIR that the step names (the summary prints
it); this kind writes its rate input, the JSON selection record ``{"rates": {optimizer:
[r1, r2, r3]}, "batch": B, "U": U}`` that numerical_check.py --selection reads (with
``"given_up": [case, ...]`` when the launcher gave up on a numerical run), and prints the whole
command that writes the check's driver (``--commit`` the numerical runs' commit, ``--out``
beside the record), the request's ``runs_commits`` and w_max, the directory the check writes
into and the selection's verdict there (verdicts/<digest>.json).
It is the one kind written while the records are not whole, and only when every problem is a run
made beyond a check that went stale (the selection's ``beyond_stale``): those runs wait for the
check, which is run again.
When no candidate is eligible with the three arms, (7, 20) and (6, 20) are visited again for the
primary arms alone, the pre-warmed arm dropped (never (8, 20), which the three-arm validation's
records decide: amendment 5). When none is eligible either, the stage ends
``infeasible``: every kind is refused, and the B = 20 feasibility report (JSON: per candidate and
arm, the scientific failures apart from the technical ones) is printed; it is not a
confirmation. A stage stopped by technical marks only (``blocked``: a rate not qualified after its
arm's search allowance was spent) refuses every kind with the rates and their reasons: the
decision is a human one. A campaign holds the runs still missing; ``--seeds`` and ``--cells``
only split it over Pods.

``timecourse`` (amendment 4, section 4, "Harm over steps"; exploratory) is written beside the
confirmation, never instead of it: separate runs of the first 10 confirmation blocks (seeds
3000-3009), each arm of the frozen selection (three, or two when the pre-warmed arm was dropped)
at its central rate, the K* rate, the middle of its three: 30 runs, or 20. Each is its endpoint
run's cell (experiments/study2_short_rayleigh, as blocks 1-10 run it) but for
``study2_short.role=timecourse`` and ``schedule.evaluate_at=[1,5,10,25]``: the regular sealed
evaluation at those steps (K_scoring, utility, and the harmful responses stored for the judge,
never scored). It is refused until the stage's next step is ``confirmation`` (the selection done,
its validation passed) and the launcher's journal shows blocks 1-10 in issued confirmation ranges,
so that their streams exist (``timecourse_guard``); like the confirmation it needs the versions
record. A run whose endpoint run the launcher gave up on is left out, named in the header: its
identity could never be shown. The loaded runs are split here once (``split_timecourse``): a
time-course run never counts as a run of its key in another campaign (a confirmation range
re-issued to complete it offers an endpoint run lost beside its time-course twin), and never
enters a block, a look or the mechanism cohort; the stage's summary reports each against its
endpoint run (``timecourse_identity``: identical bit for bit, or numerically consistent, review
fa21; else excluded), with the failed and missing
ones per seed and arm (``progress``'s ``timecourse``).

- --runs holds every run so far: Stage A's (study2_stage_a, study2_controls), whose 25-step grids
  are the first discovery runs and whose version-1 discovery set fixes the stream time, and this
  stage's (study2_short, and amendment 3's study2_short_fa_controls, reported and never used).
  --theta0 is the θ₀ run (summary.json).
- **w_max** is the simulation's: the fixation record (--fixation, ``read_fixation``) is required
  for every kind. A --w-max given beside it must equal it. Every campaign's header records
  w_max, where it came from and the code revision, and its YAML ``header: {w_max: ...}``, which
  the launcher logs with every start it makes: w_max is frozen since S1, and a replay under
  another value than one an issued campaign recorded is refused (``issued_w_max``; the final
  review of gate B, I4).
- --runs-root is RUNS_DIR, required for every kind: its numerical/ holds the numerical-check
  verdicts (``read_numerical``), and its launch/ journal (``launch.launch_records``) tells which
  runs the launcher gave up on after their recovery round (``exhausted_runs``: technical
  failures of their blocks) and which confirmation ranges were issued.
  A run is given up only once the last attempt of its recovery round ended, or that attempt is
  in the abandonment record --abandoned FILE (``read_abandonment``: a JSON object
  ``{"abandoned": [{"run_id": ATTEMPT, "reason": TEXT}]}``, ATTEMPT the journal's run_id, kept
  under $RUNS_DIR outside the checkout and archived with an inventory); an attempt without either
  may still be running, and its run stays pending. The campaign's header records the abandonment
  record's path and sha256.
- **The versions record** (--versions FILE, default the current record,
  docs/study2-short-versions-2.json; the first, docs/study2-short-versions.json, holds for its
  own commit; written by ``scripts/study2_short_summary.py versions`` and committed):
  ``validation``, ``confirmation`` and ``timecourse`` refuse without it, or when the running
  commit, the model, the judge's identity or the fixation record departs from it
  (``versions_problems``: the record tracked and unchanged at HEAD; HEAD is the recorded commit
  or a child that adds the record alone, or any descendant when the record allows it; no
  uncommitted change or untracked file, so campaign files go under $RUNS_DIR), or when an issued
  confirmation or time-course campaign named another record (``issued_versions``, from the
  launch journal: a confirmation keeps the record it started with). The campaign's YAML header
  names its record (``versions_record``, ``versions_sha256``), which the launcher logs with every
  start. The discovery and numerical kinds (``discovery``, ``repair``,
  ``prewarmed``, ``targeted``, ``numerical``) precede the record and do not read it.
- **The confirmation loop** (``confirmation_guard``; cap 80). A block is started once it has runs
  or belongs to a range the launcher has started (its journal), even before the launcher reaches
  it. Either way the next range starts at the first block not started and touches no started
  block, and re-issuing exactly a range written before completes it.
  - A three-arm selection (amendment 4): the confirmation runs until the mechanism cohort has 60
    jointly supported blocks (all three arms supported) complete in order, or until all 80 blocks
    are started and complete, without pausing at the primary's looks, which are computed
    afterwards from the same first 36 and 60 primary-supported blocks: (1) write and launch at
    most the blocks the guard allows, min(60 − p₃, 80 − the blocks started), p₃ the started
    blocks that may still be jointly supported; (2) let them complete; (3) read the counts
    (scripts/study2_short_summary.py: its ``mechanism``); (4) go on while the cohort has not
    ended. ``--after-look`` is refused, also on a re-issue. The runs are labeled as they
    complete; the summary with --judge and --fixation reads the looks, and the mechanism's final
    analysis once the cohort has ended.
  - A two-arm selection (the pre-warmed arm dropped, or the primary-only fallback; looks at 36
    and 60 supported blocks): (1) write and launch at most the blocks the guard allows: the next
    range ends before the next look could fall; (2) let them complete; (3) read the support
    counts (scripts/study2_short_summary.py); (4) label the runs; (5) run the summary with
    --judge and --fixation; (6) go on only if the look did not stop, passing ``--after-look K``
    for look K.

Nothing here reads a judge label or K_harmful.

``prep`` writes a bash driver of the prerequisites of the runs of the given seeds at batch B, from
the source checkout of commit SHA on a Pod, laid out as the Stage-A drivers. It first checks that
the checkout is at SHA and clean as the runs read it (``tracking.git_state``: no uncommitted
change and no untracked file its .gitignore does not ignore, whatever status.showUntrackedFiles,
.git/info/exclude and core.excludesFile say). With ``--once`` (and its ``--stage0-run-id``) it
starts with the stage-0 checks under
protocol version 3 and the curvature panel. That was done on P0 on 2026-10-16
(docs/operations.md records it) and must not be run again: later drivers leave out ``--once``,
and their Pods restore the inventories of its stage-0 v3 record and of the curvature panel
(``V3_RECORD``, ``CURVATURE_PANEL``; the header of every driver names the inventories it needs).
Then, per seed, either it checks that the seed's
streams exist (``--archived``: seeds whose streams are archived, nothing is generated, a missing
stream stops it) or it
generates them: the training stream to 25 × B responses (the stream is appendable; the default
length, study2_stage_a's 4000 steps at its batch 20, is not needed) and v₀'s preconditioner
stream at B, 50 × B responses (1000 at B = 20; those of seeds 2000-2002 at both batches are
archived). The numerical seeds 2200-2202 are prepared so at the candidate's B before its
numerical campaign. The driver can be run again after a failure. ``check-streams`` is that
check, which the driver runs on the Pod through ``runtime.locate_stream``. FILE must not exist;
write campaigns under an ignored directory such as $RUNS_DIR/campaigns/.

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
  copy goes back, so that DATA_DIR matches its inventories. A regeneration that differs from the
  archive is kept whole in the dated directory, under the run's id with a ``-regenerated``
  suffix, for inspection; the archived copy goes back and the step stops the driver with an
  error, on every later run with that run id too. Each step is skipped once done, so the driver
  can be run again at any point.
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
    ARMS,
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    FA_CONTROL_ARM,
    LOOK_SIZES,
    NUMERICAL_SEEDS,
    PREWARMED,
    RAYLEIGH,
    STEPS,
    TIMECOURSE,
    VALIDATION_SEEDS,
    VERSIONS_RECORD,
    confirmation_guard,
    exhausted_runs,
    issued_ranges,
    issued_versions,
    issued_w_max,
    measures_rayleigh,
    progress,
    read_abandonment,
    read_fixation,
    read_numerical,
    run_facts,
    run_prescription,
    seed_role,
    split_timecourse,
    timecourse_guard,
    versions_problems,
)
from zero_field_refusal.launch import launch_records
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import CLEAN_CHECKOUT, git_state

KINDS = ("discovery", "repair", "prewarmed", "targeted", "numerical", "numerical-check",
         "validation", "confirmation", "timecourse")  # fmt: skip
VERSIONED_KINDS = ("validation", "confirmation", "timecourse")  # they need the versions record
STAGE_SEEDS = (*DISCOVERY_SEEDS, *NUMERICAL_SEEDS, *VALIDATION_SEEDS, *CONFIRMATION_SEEDS)
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
# The source checkout is clean as the runs read it (tracking.git_state: no uncommitted change, no
# untracked file that its .gitignore does not ignore, whatever status.showUntrackedFiles,
# .git/info/exclude and core.excludesFile say); the check is shared with the other Pod drivers.
CLEAN = CLEAN_CHECKOUT
# The stage-0 checks the driver runs, and where the restored soft-target copy waits meanwhile.
STAGE0_CONFIG = "experiments/stage0_checks"
STAGE0_OVERRIDES = ("model=qwen2.5-1.5b-instruct", "protocol_version=3")
ASIDE_ROOT = Path("/workspace/tmp")  # on the Pod's volume, as DATA_DIR
ASIDE_PREFIX = "soft-targets-aside-"  # + the UTC date
SOFT_TARGET_KINDS = ("soft-targets-aside", "soft-targets-restore")
REGENERATION_RECORD = "soft_targets_regeneration.json"  # in the stage-0 run's directory
REGENERATED_MANIFEST = "soft_targets_manifest_regenerated.json"
# The inventories under runs/archive/ from which pod_setup.sh restores DATA_DIR for this stage
# (docs/operations.md, "Prerequisites on a Pod"). Every campaign kind reads the first group: K's
# reference streams (kl_harmful, kl_benign), the prompt sets, the protocol-2 path-field record,
# the monitor-v2 stream, v₀'s preconditioner streams and the latest training stream per seed.
INVENTORIES = (
    "inventory-2026-10-11.tsv",
    "inventory-2026-10-14-study2-prompts.tsv",
    "inventory-2026-10-14-study2-a-phase0.tsv",
    "inventory-2026-10-15-study2-monitor-v2.tsv",
    "inventory-2026-10-14-study2-a-preconditioner-s2000.tsv",
    "inventory-2026-10-14-study2-a-preconditioner-s2001.tsv",
    "inventory-2026-10-14-study2-a-preconditioner-s2002.tsv",
    "inventory-2026-10-16-study2-a-stream-s2000-final.tsv",
    "inventory-2026-10-15-study2-a-stream-s2001-search-s2001-2.tsv",
    "inventory-2026-10-16-study2-a-stream-s2002-final.tsv",
)
# The prep of 2026-10-16 (prep --once on P0): the stage-0 record under protocol version 3 (made
# for amendment 3's FA controls; amendment 4 launches none) and the curvature panel, which
# confirmation blocks 1-10 need. Every later Pod restores both, whatever it runs: pod_setup.sh
# checks every file in DATA_DIR against the inventories, so the v3 record is restored for that
# check (review M9).
V3_RECORD = "inventory-2026-10-16-study2-stage0-v3-record.tsv"
CURVATURE_PANEL = "inventory-2026-10-16-study2-curvature-panel.tsv"


def cell(name: str, config: str, overrides: list[str], seeds: list[int]) -> dict:
    return {"name": name, "config": config, "overrides": overrides, "seeds": seeds}


def regular_cells(runs: list[dict], suffix: str = "", *, timecourse: bool = False) -> list[dict]:
    """One cell per arm, rate and config (the first 10 confirmation blocks measure R), over its
    seeds, with the role of its seeds. The launcher grants a recovery round once per experiment
    and cell name, so a name never recurs in another of the stage's campaigns: ``suffix`` names
    the numerical check (whose seeds of one rate can differ between checks), the validation, the
    confirmation blocks or the time course. With ``timecourse``, the time course's runs: each
    the endpoint run's cell of its seed, arm and rate but for the role timecourse and the
    evaluation steps 1, 5, 10 and 25 (``TIMECOURSE_OVERRIDES``)."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in runs:
        seed = r["seed"]
        key = (ARMS.index(r["arm"]), r["batch"], r["lr"], measures_rayleigh(seed), seed_role(seed))
        groups[key].append(seed)
    items = []
    for (arm_index, batch, lr, rayleigh, role), seeds in sorted(groups.items()):
        arm = ARMS[arm_index]
        # The config and overrides that conformance checks every run's config hash against
        config, overrides = run_prescription(arm, batch, lr, seeds[0],
                                             TIMECOURSE if timecourse else role)  # fmt: skip
        name = f"{arm}-b{batch}-{lr:g}{suffix}" + ("-rayleigh" if rayleigh else "")
        items.append(cell(name, config, overrides, sorted(seeds)))
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
        *comment("Before it runs, DATA_DIR is restored with scripts/pod_setup.sh from these "
                 "inventories under runs/archive/, so that the runs read the streams that Stage "
                 f"A's runs read: for every campaign kind {', '.join(INVENTORIES)} (the prompt "
                 "sets, K's reference streams, the protocol-2 path-field record, the monitor-v2 "
                 "stream, the preconditioner streams and the training streams); the stage-0 v3 "
                 f"record {V3_RECORD} (no run of amendment 4 reads it: it is restored because "
                 "pod_setup.sh checks the whole DATA_DIR against the inventories); for "
                 f"confirmation blocks 1-10 also the curvature panel {CURVATURE_PANEL}. Every Pod "
                 "after the prep of 2026-10-16 restores these two as well, and the inventories of "
                 "the fresh seeds' streams once they are archived."),
        *comment("It stops at the first failure and can be run again: a completed stage-0 run "
                 "is kept, a soft-target copy set aside or put back stays so, a stream that "
                 "exists is kept, and a training stream only grows to what the runs read."),
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        f'project-run "$SRC" python -c "{CLEAN}"',
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


def regenerated_place(dated: Path, stream: Path, run_id: str, *, equal: bool) -> Path:
    """Where the regenerated stream goes in the dated aside directory, under the run's id (two
    runs on one day never meet): its manifest and samples when it equals the archive; otherwise
    the whole stream, distributions included, with a ``-regenerated`` suffix, for inspection."""
    return dated / run_id / (stream if equal else f"{stream}-regenerated")


def regeneration_record(
    archived: Path, regenerated: Path, run_dir: Path, *, stream: Path, dated: Path
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
    comparison = {
        "samples_equal": samples == old.get("sha256"),
        "distributions_equal": distributions == (old.get("distributions") or {}).get("sha256"),
        "manifest_fields_differing": differing,
    }
    equal = comparison["samples_equal"] and comparison["distributions_equal"]
    kept = regenerated_place(dated, stream, run_dir.name, equal=equal)
    if equal:
        after = (f"{DISTRIBUTIONS_FILE} deleted after the check (retention policy); the archived "
                 "copy put back in DATA_DIR, so DATA_DIR matches the existing inventories; the "
                 f"rest of the regenerated stream (manifest and samples) kept at "
                 f"{kept}.")  # fmt: skip
    else:
        after = ("The regeneration differs from the archive: the archived copy put back in "
                 "DATA_DIR, so DATA_DIR matches the existing inventories; the regenerated stream, "
                 f"{DISTRIBUTIONS_FILE} included, kept at {kept} for inspection; the driver "
                 "stopped.")  # fmt: skip

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
        "comparison": comparison,
        "after": after,
    }


def _write_new(path: Path, text: str) -> None:
    """Write ``path`` whole or not at all."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text)
    partial.replace(path)


def _equal(record: Path) -> bool:
    compared = json.loads(record.read_text())["comparison"]
    return bool(compared["samples_equal"] and compared["distributions_equal"])


def _differs(record: Path, done: list[str]) -> None:
    """Stop: the regeneration differs from the archive."""
    sys.exit("\n".join([*done, f"the regeneration differs from the archive ({record}): the "
                        "archived copy is back in DATA_DIR and the regenerated stream is kept "
                        "for inspection; the lead decides on the comparison, and only then may "
                        "the driver run again, with a new --stage0-run-id"]))  # fmt: skip


def restore(path: Path, data_root: Path, aside_root: Path, run_dir: Path) -> str:
    """After a passed stage-0 run: the comparison into ``run_dir``; then, when the regenerated
    samples and distributions equal the archive's, the regenerated distributions deleted and the
    rest of the regenerated stream moved into the dated aside directory, otherwise the whole
    regenerated stream kept there for inspection (``regenerated_place``); then the archived copy
    put back at ``path``. A regeneration that differs stops with an error, also when run again.
    Each step done before is skipped."""
    from zero_field_refusal.teacher import DISTRIBUTIONS_FILE, MANIFEST_FILE

    found = set_aside_copies(path, data_root, aside_root)
    record = run_dir / REGENERATION_RECORD
    distributions = path / DISTRIBUTIONS_FILE
    if not found:
        if record.is_file():
            if not _equal(record):
                _differs(record, [])
            return f"kept: the archived copy is back at {path}; the comparison is {record}"
        if distributions.exists():
            sys.exit(f"{path} holds distributions and no copy is set aside under {aside_root}: "
                     "compare and clean up by hand (docs/operations.md, Stage 0 under a new "
                     "protocol)")  # fmt: skip
        if not (path / MANIFEST_FILE).is_file():
            sys.exit(f"no soft-target stream at {path}, and none set aside under {aside_root}: "
                     "restore DATA_DIR with scripts/pod_setup.sh first")  # fmt: skip
        return (f"nothing set aside for run {run_dir.name} and no regenerated distributions at "
                f"{path}: nothing to compare or restore")  # fmt: skip
    archived = found[0]
    relative = path.relative_to(data_root)
    dated = aside_root / archived.relative_to(aside_root).parts[0]
    done = []
    if not record.is_file():
        if not distributions.is_file():
            sys.exit(f"no regenerated distributions at {path} to compare with {archived}")
        found_record = regeneration_record(archived, path, run_dir, stream=relative, dated=dated)
        _write_new(run_dir / REGENERATED_MANIFEST, (path / MANIFEST_FILE).read_text())
        _write_new(record, json.dumps(found_record, indent=2) + "\n")
        compared = found_record["comparison"]
        done.append(f"compared: samples equal {compared['samples_equal']}, distributions equal "
                    f"{compared['distributions_equal']} ({record})")  # fmt: skip
    equal = _equal(record)
    if equal and distributions.exists():
        distributions.unlink()
        done.append(f"deleted {distributions}")
    if path.exists():
        kept = regenerated_place(dated, relative, run_dir.name, equal=equal)
        if kept.exists():
            sys.exit(f"{kept} exists: the regenerated stream at {path} has nowhere to go")
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, kept)
        done.append(f"kept the {'rest of the ' if equal else ''}regenerated stream at {kept}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(archived, path)
    done.append(f"put back: {archived} -> {path}")
    if not equal:
        _differs(record, done)
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
    """w_max and where it came from: the fixation record, required for every kind."""
    if args.fixation is None:
        parser.error(f"{args.kind} needs the fixation record (--fixation): its w_max is the "
                     "simulation's")  # fmt: skip
    try:
        fixed = read_fixation(args.fixation)["w_max"]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.w_max is not None and args.w_max != fixed:
        parser.error(f"--w-max {args.w_max} differs from the fixation record's {fixed}")
    digest = hashlib.sha256(args.fixation.read_bytes()).hexdigest()
    return fixed, f"the fixation record {args.fixation}, sha256 {digest}"


def campaign(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[dict, float, str, list[str], list[str]]:
    """The campaign of ``args.kind``, w_max with its source, the records it was written under
    (the versions and abandonment records, path and sha256) and its prerequisites, for its
    header."""
    if args.theta0 is None or not args.runs:
        parser.error(f"{args.kind} needs --runs and --theta0")
    if (args.kind == "confirmation") != (args.blocks is not None):
        parser.error("--blocks goes with confirmation, and confirmation needs it")
    if args.after_look and args.kind != "confirmation":
        parser.error("--after-look goes with confirmation")
    if args.kind in ("confirmation", "numerical-check") and args.seeds:
        parser.error("--seeds splits the discovery, numerical, validation and time-course "
                     "campaigns only")  # fmt: skip
    if args.runs_root is None:
        parser.error(f"{args.kind} needs --runs-root: the stage reads the numerical-check "
                     "verdicts and the launch journal there")  # fmt: skip
    w_max, source = w_max_of(parser, args)
    notes, prerequisites = [], []
    records = launch_records(args.runs_root)
    if args.kind in VERSIONED_KINDS:
        # The record this campaign is written under; a confirmation's earlier campaigns named
        # theirs in the journal (amendment 5: two records, each for its own commit).
        problems = versions_problems(args.versions, fixation=args.fixation,
                                     issued=issued_versions(records))  # fmt: skip
        if problems:
            parser.error("the versions record does not hold: " + "; ".join(problems))
        notes.append(f"the versions record {args.versions}, sha256 {_sha256(args.versions)}")
    abandoned = {}
    if args.abandoned is not None:
        try:
            abandoned = read_abandonment(args.abandoned)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        notes.append(f"the abandonment record {args.abandoned}, sha256 "
                     f"{_sha256(args.abandoned)}")  # fmt: skip
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary.get(key) for key in ("benchmark_accuracy", "factual_accuracy")}
    if theta0_problems(theta0):
        parser.error("; ".join(theta0_problems(theta0)))
    loaded = load_trajectories(*args.runs)
    facts = {r.run_id: run_facts(r.run_dir) for r in loaded if r.arm in (*ARMS, FA_CONTROL_ARM)}
    # The time course's runs, split once here: no campaign's bookkeeping ever counts one of them
    # as a run of its key (review of Task 5, I1).
    runs, courses = split_timecourse(loaded, facts)
    try:
        exhausted = exhausted_runs(records, abandoned=abandoned)
    except ValueError as exc:
        parser.error(str(exc))
    state = progress(runs, theta0, w_max, facts=facts, exhausted=exhausted,
                     numerical=read_numerical(args.runs_root), timecourse=courses,
                     issued_w_max=issued_w_max(records))  # fmt: skip
    step = state["next"]
    chosen = state.get("selection") or {}
    waiting = chosen.get("beyond_stale") or []
    if (args.kind == "numerical-check" and state["problems"]
            and set(state["problems"]) <= set(waiting)):  # fmt: skip
        # Runs made beyond a check that went stale wait for it: its rate input may be written.
        step = chosen["pending"]
    elif state["problems"]:
        parser.error("the stage's records are not whole: " + "; ".join(state["problems"]))
    if step.get("blocked"):  # technical marks only (review fa22): no campaign, a human decision
        parser.error(blocked_message(step))
    if step["campaign"] == "infeasible":  # the B = 20 feasibility report, not a confirmation
        print(json.dumps(step["feasibility"], indent=2))
        parser.error(f"the stage ended infeasible: no candidate at B = 20 qualifies (scope "
                     f"{step['scope']}); its feasibility report is printed, and no campaign "
                     "follows")  # fmt: skip
    if args.kind == "timecourse":  # beside the confirmation, once the selection is done
        if step["campaign"] != "confirmation":
            shown = {k: v for k, v in step.items() if k != "runs"}
            parser.error("the time course waits for the selection to be done and its "
                         f"validation passed: the stage's next step is {step['campaign']}: "
                         f"{shown}")  # fmt: skip
        problems = timecourse_guard(issued_ranges(records))
        if problems:
            parser.error("; ".join(problems))
        course = state["timecourse"]
        # A run whose endpoint run was given up could never show its identity: not run, named.
        left_out = [r for r in course["missing"] if "endpoint_given_up" in r]
        wanted = [{"arm": r["arm"], "batch": course["batch"], "lr": r["lr"], "seed": r["seed"]}
                  for r in course["missing"] if "endpoint_given_up" not in r]  # fmt: skip
        if args.seeds:
            unknown = sorted(set(args.seeds) - {r["seed"] for r in wanted})
            if unknown:
                parser.error(f"no run of seeds {unknown} in this campaign")
            wanted = [r for r in wanted if r["seed"] in args.seeds]
            left_out = [r for r in left_out if r["seed"] in args.seeds]
        if not wanted:
            parser.error("the time course has every run already, or given up by the launcher "
                         "with its endpoint run")  # fmt: skip
        prerequisites += [f"Left out: the time course's {r['arm']} run at {r['lr']:g} on seed "
                          f"{r['seed']}, whose endpoint run the launcher gave up on "
                          f"({r['endpoint_given_up']}): its identity could never be shown."
                          for r in left_out]  # fmt: skip
        written = {"name": f"study2-short-timecourse-b{course['batch']}-u{step['U']:g}",
                   "cells": regular_cells(wanted, "-timecourse", timecourse=True)}  # fmt: skip
        return written, w_max, source, notes, prerequisites
    if step["campaign"] != args.kind:
        shown = {k: v for k, v in step.items() if k != "runs"}
        parser.error(f"the stage's next step is {step['campaign']}, not {args.kind}: {shown}")
    batch, upper = step["batch"], step["U"]
    if args.kind == "numerical-check":  # no launcher campaign: the check's rate input
        written = {"check": step["check"], "waiting": waiting if state["problems"] else []}
        return written, w_max, source, notes, prerequisites
    if args.kind == "confirmation":
        first, last = args.blocks
        guard = confirmation_guard(state["confirmation"], first, last,
                                   three_arm=step["three_arm"], after_look=args.after_look or 0,
                                   issued=issued_ranges(records))  # fmt: skip
        if guard["problems"]:
            parser.error(f"blocks {first}-{last}: " + "; ".join(guard["problems"]))
        seeds = CONFIRMATION_SEEDS[first - 1 : last]
        # The endpoint runs there or given up (``runs`` holds no time-course run, and a given-up
        # time-course run keeps its own 5-tuple key).
        present = {(r.arm, r.batch_size, rate(r.lr), r.seed) for r in runs
                   if r.arm in ARMS and r.seed in seeds}  # fmt: skip
        present |= {key for key in exhausted if len(key) == 4}
        wanted = [{"arm": arm, "batch": batch, "lr": rate(lr), "seed": seed}
                  for seed in seeds for arm, lrs in step["rates"].items() for lr in lrs
                  if (arm, batch, rate(lr), seed) not in present]  # fmt: skip
        if not wanted:
            parser.error(f"blocks {first}-{last} have every run already")
        written = {"name": f"study2-short-confirmation-{first}-{last}",
                   "cells": regular_cells(wanted, f"-blocks-{first}-{last}")}  # fmt: skip
        return written, w_max, source, notes, prerequisites
    wanted = step["runs"]
    if args.seeds:
        unknown = sorted(set(args.seeds) - {r["seed"] for r in wanted})
        if unknown:
            parser.error(f"no run of seeds {unknown} in this campaign")
        wanted = [r for r in wanted if r["seed"] in args.seeds]
    if args.kind == "numerical":
        streams = step["streams"]
        seeds = " ".join(map(str, streams["seeds"]))
        prerequisites.append(f"The streams of seeds {seeds} at batch {batch} come first (prep "
                             f"--seeds {seeds} --batch {batch}): the training stream of "
                             f"{streams['training_responses']} responses and v₀'s preconditioner "
                             f"stream of {streams['preconditioner_responses']}. The check "
                             f"{step['check']} follows the runs.")  # fmt: skip
        # The check's selection digest keeps every cell name unique across checks.
        tag = f"u{upper:g}-{step['check']}"
        written = {"name": f"study2-short-numerical-b{batch}-{tag}",
                   "cells": regular_cells(wanted, f"-numerical-{tag}")}  # fmt: skip
        return written, w_max, source, notes, prerequisites
    # The pre-warmed arm's own rounds, apart from the primary arms' of the same batch and U.
    own = args.kind in ("targeted", "repair") and {r["arm"] for r in wanted} == {PREWARMED}
    kind = f"{args.kind}-prewarmed" if own else args.kind
    if not step["three_arm"]:  # the fallback: the primary arms alone
        kind += "-primary"
    suffix = "-validation" if args.kind == "validation" else ""
    name = f"study2-short-{kind}-b{batch}-u{upper:g}"
    if step.get("allowance"):  # a search allowance's round: named after the check that spent it
        name += f"-allowance-{step['allowance']}"
    written = {"name": name, "cells": regular_cells(wanted, suffix)}
    return written, w_max, source, notes, prerequisites


def blocked_message(step: dict) -> str:
    """The refusal of a stage stopped by technical marks only (review fa22): the candidate, the
    gate, the reasons and the rates with theirs, for a human decision."""
    rates = "; ".join(f"{r['arm']} {r['lr']:g}: {r['why']}"
                      + (f" ({'; '.join(r['reasons'])})" if r.get("reasons") else "")
                      for r in step["rates"])  # fmt: skip
    scope = "three arms" if step["three_arm"] else "primary pair"
    return (
        f"the stage is blocked by technical marks only at U={step['U']:g}, B={step['batch']} "
        f"({scope}, {step['at']}): {'; '.join(step['reasons'])}; the rates: {rates}; "
        f"{step['decision']}; no campaign is written"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _in_checkout(path: Path) -> str:
    """A path relative to the checkout when it is in it (a versions record always is)."""
    resolved = path.resolve()
    inside = resolved.is_relative_to(REPO_ROOT.resolve())
    return resolved.relative_to(REPO_ROOT.resolve()).as_posix() if inside else str(path)


def check_command(check: dict, selection: Path, checkout: str | None = None) -> list[str]:
    """The numerical check's whole command and what it needs (the final review of gate B, M1):
    the driver written from the selection record at ``selection`` for the commit whose source
    the Pod runs (the numerical runs' own, ``runs_commits``; any of them when there are several,
    their GPU code being one), with the request's w_max, the cases given up by the launcher and
    where the check writes. When no numerical run is located (every case given up by the
    launcher), ``runs_commits`` is empty and the driver runs from the stage's checkout commit
    (review fa14, concern 4): ``checkout``, the commit of the clean checkout writing the request,
    or COMMIT, said so, when that checkout is not clean."""
    commits = check["runs_commits"]
    if len(commits) == 1:
        commit = commits[0]
    elif not commits and checkout:
        commit = checkout
    else:
        commit = "COMMIT"
    if len(commits) > 1:
        which = "; COMMIT is one of them, their GPU code being one"
    elif not commits:
        unclean = (" (COMMIT: this checkout is not clean; write the request from a clean "
                   "checkout of the stage's commit)")  # fmt: skip
        which = ("; none: no numerical run is located (every case given up by the launcher), so "
                 "the driver runs from the stage's checkout commit, the one writing this request"
                 + ("" if checkout else unclean))  # fmt: skip
    else:
        which = ""
    driver = selection.with_name(f"{selection.stem}-driver.sh")
    lines = [f"python scripts/numerical_check.py driver --selection {selection} --batch "
             f"{check['batch']} --upper {check['U']:g} --commit {commit} --out {driver}",
             f"runs_commits (the numerical runs' commits): {commits}{which}",
             f"w_max {check['w_max']!r}; outputs under $RUNS_DIR/{check['out']}, the selection's "
             f"verdict at $RUNS_DIR/{check['verdict']}"]  # fmt: skip
    if check["given_up"]:
        lines.append(f"given up by the launcher, named in the record: {check['given_up']}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("prep", "check-streams", *SOFT_TARGET_KINDS, *KINDS))
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--theta0", type=Path, help="the θ₀ run directory (summary.json)")
    parser.add_argument("--w-max", type=float, help="w_max, which must equal the fixation record's")
    parser.add_argument("--fixation", type=Path, help="the simulation's fixation record (JSON)")
    parser.add_argument("--runs-root", type=Path,
                        help="RUNS_DIR: its numerical/ verdicts and launch/ journal")  # fmt: skip
    parser.add_argument("--abandoned", type=Path,
                        help="the abandonment record of launcher attempts (JSON)")  # fmt: skip
    parser.add_argument("--versions", type=Path, default=REPO_ROOT / VERSIONS_RECORD,
                        help="validation and confirmation: the versions record")  # fmt: skip
    parser.add_argument("--seeds", type=int, nargs="+", help="keep only these seeds (one Pod)")
    parser.add_argument("--cells", nargs="+", help="keep only these cells (one Pod each)")
    parser.add_argument("--blocks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="confirmation: the blocks FIRST to LAST, 1-80")  # fmt: skip
    parser.add_argument("--after-look", type=int, choices=range(1, len(LOOK_SIZES) + 1),
                        help="confirmation of a two-arm selection: look K was read and did not "
                             "stop (refused for three arms)")  # fmt: skip
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
        help="prep: also the stage-0 checks and the curvature panel (done on 2026-10-16: never "
        "again)",
    )
    parser.add_argument("--stage0-run-id", help="prep with --once: RUN_ID of the stage-0 checks")
    parser.add_argument("--aside-root", type=Path, default=ASIDE_ROOT,
                        help="soft-targets-*: where the restored copy waits")  # fmt: skip
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = args.seeds or []
    if set(seeds) - set(STAGE_SEEDS) or len(set(seeds)) != len(seeds):
        parser.error("--seeds: distinct seeds of the stage (2000-2002, 2200-2202, 2100-2111, "
                     "3000-3079)")  # fmt: skip
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
        written, w_max, source, notes, prerequisites = campaign(parser, args)
        if args.kind == "numerical-check":
            if args.cells:
                parser.error("--cells splits launcher campaigns only")
            check = written["check"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(check["selection"], indent=2) + "\n")
            print(args.out)
            state = git_state()  # the stage's checkout: its commit when clean
            checkout = state.get("commit") if state.get("dirty") is False else None
            print("\n".join(check_command(check, args.out, checkout)))
            if written["waiting"]:
                print("Written while the stage's records wait for this check, which is run again:"
                      "\n" + "\n".join(written["waiting"]))  # fmt: skip
            return
        if args.cells:  # a campaign split over Pods, cell by cell, without choosing anything
            unknown = sorted(set(args.cells) - {c["name"] for c in written["cells"]})
            if unknown:
                parser.error(f"no such cells in this campaign: {unknown}")
            written["cells"] = [c for c in written["cells"] if c["name"] in args.cells]
        writer = f"scripts/study2_short_campaigns.py {args.kind} at {revision()}"
        under = "".join(f", under {note}" for note in notes)
        header = comment(f"Written by {writer}, with w_max {w_max!r} from {source}{under}. "
                         "Do not edit.")  # fmt: skip
        header += [line for text in prerequisites for line in comment(text)]
        # The final review of gate B, I4: the header's w_max, which the launcher logs with every
        # start, so that a replay under another w_max is refused (``issued_w_max``); and the
        # versions record a versioned campaign is written under (``issued_versions``)
        values = {"w_max": w_max}
        if args.kind in VERSIONED_KINDS:
            values.update(versions_record=_in_checkout(args.versions),
                          versions_sha256=_sha256(args.versions))  # fmt: skip
        recorded = {"header": values, **written}
        text = "\n".join(header) + "\n" + yaml.safe_dump(recorded, sort_keys=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(args.out)


if __name__ == "__main__":
    main()
