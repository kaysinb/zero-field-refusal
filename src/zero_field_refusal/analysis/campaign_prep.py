"""The prerequisites' driver of a campaign: the bash script that prepares a Pod for its runs, the
teacher streams of their fresh seeds above all. Shared by study 2's short-regime stage
(``scripts/study2_short_campaigns.py prep``) and study 3 (``scripts/study3_campaigns.py prep``);
it writes text only, and reads no record.

- ``prep_driver`` and ``prep``: study 2's driver and its command's arguments (that script's
  docstring): the stage-0 checks under protocol version 3 once (``--once``), then per seed either
  the check that its archived streams exist (``--archived``) or their generation, under
  ``STAGE_A``, the config Stage A generated them with.
- ``streams_driver``: a study's fresh seeds, each seed's streams generated under the config its
  runs locate them with (study 3: its role's), nothing else.
- ``comment`` and ``revision``: the comment lines of a written file and the revision that wrote it.

**Every driver** runs from the source checkout of one commit on a Pod (``SRC``), first checks that
the checkout is at that commit and clean as the runs read it (``CLEAN``: ``tracking.git_state``, no
uncommitted change and no untracked file its .gitignore does not ignore), and then, per seed, grows
the training stream to the responses its runs read (25 updates × B; the stream is appendable) and
generates v₀'s preconditioner stream at B (50 × B responses). It stops at the first failure and
can be run again: a stream that exists is kept (``kept``: the generator's refusal of an existing
stream, ``EXISTS``, is no failure), and a training stream only grows.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Mapping, Sequence

from ..tracking import CLEAN_CHECKOUT, git_state

STAGE_A = "experiments/study2_stage_a"  # study 2's streams' config, as Stage A generated them
TEACHER = 'project-run "$SRC" python scripts/generate_teacher.py'
# generate_teacher.py's refusal of a stream that exists (runtime.generate_stream).
EXISTS = "streams are the record and are never regenerated"
# A stage-0 run counts only if it completed, ran under protocol version 3 and its path-field check
# passed: a completed run can still have failed the check.
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
# The stage-0 checks study 2's driver runs once.
STAGE0_CONFIG = "experiments/stage0_checks"
STAGE0_OVERRIDES = ("model=qwen2.5-1.5b-instruct", "protocol_version=3")
# The inventories under runs/archive/ from which pod_setup.sh restores DATA_DIR for study 2's
# short-regime stage (docs/operations.md). Every campaign kind reads the first group: K's
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
# Study 2's prep of 2026-10-16 (prep --once on P0): the stage-0 record under protocol version 3
# (made for amendment 3's FA controls; amendment 4 launches none) and the curvature panel, which
# confirmation blocks 1-10 need. Every later Pod restores both, whatever it runs: pod_setup.sh
# checks every file in DATA_DIR against the inventories, so the v3 record is restored for that
# check.
V3_RECORD = "inventory-2026-10-16-study2-stage0-v3-record.tsv"
CURVATURE_PANEL = "inventory-2026-10-16-study2-curvature-panel.tsv"
STAMP = "$(date -u +%FT%TZ)"


def revision() -> str:
    """The code revision that writes a file: its commit, and whether the tree was clean."""
    state = git_state()
    commit = state.get("commit") or "an unknown commit"
    return commit + (" with uncommitted changes" if state.get("dirty") else "")


def comment(text: str) -> list[str]:
    """``text`` as comment lines of at most 100 characters, broken at spaces only."""
    return textwrap.wrap(text, 100, initial_indent="# ", subsequent_indent="# ",
                         break_long_words=False, break_on_hyphens=False)  # fmt: skip


def _start(header: Sequence[str], commit: str) -> list[str]:
    """A driver's first lines: its header, the checkout at the commit and clean, and the step
    functions (``step``: a command, stamped; ``kept``: one that may find its stream there)."""
    return [
        "#!/usr/bin/env bash",
        *header,
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        f'project-run "$SRC" python -c "{CLEAN}"',
        f'step() {{ echo "== {STAMP} start: $*"; "$@"; echo "== {STAMP} done: $*"; }}',
        "kept() {",
        "  local log",
        "  log=$(mktemp)",
        f'  echo "== {STAMP} start: $*"',
        f'  if "$@" 2>&1 | tee "$log"; then echo "== {STAMP} done: $*"',
        f'  elif grep -q "{EXISTS}" "$log"; then',
        f'    echo "== {STAMP} kept, it exists: $*"',
        '  else rm -f "$log"; return 1; fi',
        '  rm -f "$log"',
        "}",
    ]  # fmt: skip


def _streams(seeds: Sequence[int], batch: int, steps: int,
             configs: Mapping[int, str]) -> list[str]:  # fmt: skip
    """Per seed, its training stream grown to ``steps`` × ``batch`` responses and v₀'s
    preconditioner stream at ``batch``, under the seed's config (``configs``: the config's name
    and its overrides, as the generator takes them)."""
    lines = []
    for seed in seeds:
        lines += [
            f"step {TEACHER} {configs[seed]} --stream training \\",
            f"  --seeds {seed} --responses {steps * batch}",
            f"kept {TEACHER} {configs[seed]} \\",
            f"  optimizer=adam_frozen train.batch_size={batch} --stream preconditioner \\",
            f"  --seeds {seed}",
        ]
    return lines


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
    """Study 2's bash driver of the runs' prerequisites (``scripts/study2_short_campaigns.py``'
    docstring)."""
    from .study2_short import RAYLEIGH, STEPS

    listed = " ".join(map(str, seeds))
    streams = (f"checks that the training and preconditioner streams of seeds {listed} at batch "
               f"{batch} exist, generating nothing" if archived else
               f"generates the training and preconditioner streams of seeds {listed} at batch "
               f"{batch}")  # fmt: skip
    steps = ("runs the stage-0 checks under protocol version 3 (with the restored soft-target "
             "copy set aside, compared and put back) and generates the curvature panel, then "
             f"{streams}") if once else streams  # fmt: skip
    header = [
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
    ]  # fmt: skip
    lines = _start(header, commit)
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
            f'    echo "== {STAMP} stage-0 run $STAGE0 passed under protocol version 3"',
            "  else",
            f'    echo "== {STAMP} stage-0 run $STAGE0 did not complete, or did not pass under" \\',
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
        lines += _streams(seeds, batch, STEPS, dict.fromkeys(seeds, STAGE_A))
    lines.append(f'echo "== {STAMP} the prerequisites are in place"')
    return "\n".join(lines) + "\n"


def streams_driver(
    seeds: Sequence[int],
    commit: str,
    *,
    configs: Mapping[int, str],
    batch: int,
    steps: int,
    writer: str,
    title: str,
    written_at: str,
) -> str:
    """The bash driver that generates the streams of fresh ``seeds`` at ``batch`` (module
    docstring): each seed's under ``configs[seed]`` (the config's name and its overrides, those
    its runs locate the streams with), the training stream to ``steps`` × ``batch`` responses.
    ``writer``: the command that writes it; ``title``: what it serves."""
    listed = " ".join(map(str, seeds))
    header = [
        *comment(f"Written by {writer} at {written_at}; do not edit."),
        *comment(f"{title}, from commit {commit}: it generates the training and preconditioner "
                 f"streams of seeds {listed} at batch {batch}, each under its runs' config: "
                 + "; ".join(f"{seed}: {configs[seed]}" for seed in seeds) + "."),
        *comment("Before it runs, DATA_DIR holds what every run reads besides its seed's streams "
                 "(the prompt sets, K's reference streams and the monitor stream), restored with "
                 "scripts/pod_setup.sh."),
        *comment("It stops at the first failure and can be run again: a stream that exists is "
                 "kept, and a training stream only grows to what the runs read."),
    ]  # fmt: skip
    lines = [*_start(header, commit), "# The steps", *_streams(seeds, batch, steps, configs)]
    lines.append(f'echo "== {STAMP} the prerequisites are in place"')
    return "\n".join(lines) + "\n"


def prep(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Study 2's ``prep`` command: its arguments checked, its driver."""
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
