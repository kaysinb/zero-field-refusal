"""Study 5's Pod campaigns, each one detached driver of blocks chained in order
(docs/study5-preregistration.md, "Seeds and order", "The confirmatory test", "Extension 1";
docs/operations.md, "Study 5 campaigns").

    python scripts/study5_campaigns.py pod1 --out DIR --archive-id ID [--surrogate SPEC]
        [--dry-run]
    python scripts/study5_campaigns.py pod2 --out DIR --archive-id ID [--n 40|24 (the committed)]
        --engineering-check RECORD --surrogate-checks RECORD --retrospective RECORD
        [--surrogate SPEC] [--checkpoints DIR] [--dry-run]
    python scripts/study5_campaigns.py pod3 --out DIR --archive-id ID [--dry-run]

On a Pod: ``project-run "$SRC" python scripts/study5_campaigns.py pod1 ...``, detached. The other
commands (``deadline``, ``fetch-streams``, ``fetch-run``, ``with-runs``, ``archive``,
``snapshot``, ``receipt``, ``check-checkpoints``, ``checkpoint``, ``completion``,
``export-checkpoints``, ``predictions``, ``publish``, ``check-published``) are the steps the
drivers run (``restore-runs`` restores a Pod's runs for the analysis, ``restore_runs``); each
prints numbers, ids, paths and hashes only, and exits 0 on success, 1 otherwise.

**The plan.** A campaign is a plan: blocks of steps, each step a command line (its argv) with
what its failure does (``stop`` the campaign, ``record`` it and go on, or ``skip-group``: skip the
rest of its seed's steps), built as data from the arguments, the checkout's commit and study 3's
records alone (``pod1_plan``, ``pod2_plan``); nothing is run to build it. ``--dry-run`` prints it
and writes nothing. Every block begins by setting the Pod's deadline file (``deadline``: now +
2 × the estimated minutes of its steps outside the seed groups + 60, through
``scripts/pod_deadline_watch.py --set``; ``ZFR_DEADLINE_EPOCH`` stays its cap), and so does every
seed group (2 × the group's minutes + 60), so that a dead driver or a hung step leaves the Pod
idle one group's window at most. A deadline written is never above the cap (min(requested,
cap), both printed; ``cmd_deadline``), and on pod 2 a block or seed group is admitted only while
its window and the close's export window (``EXPORT_WINDOW``, 76 minutes) both end before the cap:
otherwise its deadline step fails, a stop, and the close runs inside the cap. The minutes are
study 3's measured unit costs where one exists (``MINUTES``). Every step runs in its own session
with a timeout (3 × its minutes + 20): past it its process group is killed and the step has
failed, under its rule. A block ends by archiving its records to the bucket, verified
(``scripts/archive.py``: every object's size and MD5; each inventory archived too and read back),
a snapshot of the campaign's journal and logs, and the block's receipt. A last block, ``close``,
run at every end, after a stop and on any exception of the driver or SIGTERM (its deadline in a
``finally``): on pod 2 it first sets a window for, and attempts, the export of every recorded
endpoint checkpoint not yet verified in the archive (``export-checkpoints``: uploaded, verified
against the run's recorded sha256; a failure that remains is in its record and printed with the
path, the sha256 and the error, and the close goes on); then it archives the plan, the launcher
campaigns, the receipts and that record and sets the deadline 15 minutes ahead, so that the
watcher stops the Pod and nothing is left running.

**Pod 1** (order steps 1-2 of the registration): ``engineering``: seed 10403's streams fetched
from the bucket; study 3's archived records of its F runs at both rates (the run ids from study
3's analysis freeze, ``docs/study3-analysis-freeze-c1.json``; ``judge/`` and ``wandb/`` are not
fetched); the two runs rerun with this checkout under study 3's own config
(experiments/study3, ``study3.role=confirmation``: the config hashes of the archived runs); and
``scripts/study5_engineering_check.py``, archived against rerun. Any failure here stops the
campaign before anything else runs. ``theta0``: θ₀'s evaluation on the 521 prompts
(experiments/study5_theta0, the launcher's ``theta0`` entry; the analysis needs it, so its failure
stops the campaign). ``surrogate``:
``scripts/study5_surrogate_checks.py``; any non-zero exit stops the campaign (exit 2: the sets are
then changed once, by the researcher, before any extraction). ``retrospective``: the streams of
study 3's 60 A-F confirmation seeds (its freeze's A-F cohort, in its order) fetched into DATA_DIR
from the inventory study 3 archived them with, then per seed ``scripts/study5_predict.py`` at η_l
and ``scripts/study5_stream_features.py`` under that seed's study-3 F config.

**Pod 2** (order steps 4-5), fresh seeds 50000-50039 (``--n 40``) or 50000-50023 (``--n 24``; any
other n is refused): ``streams``: per seed its training and preconditioner streams
(``scripts/generate_teacher.py``, as study 3 generated them: 25 × 20 training responses grown,
v₀'s stream at B = 20, under the η_l run's config), archived and verified before anything reads
them, then ``study5_predict.py`` (η_l) and ``study5_stream_features.py``. ``predictions``: one
record of p per seed, seeds in order, each seed's record by its sha256, with its own sha256 beside
it (``<record>.sha256``); it is refused unless every seed has its prediction. It is copied to the
bucket, read back and compared, and the object's creation time is recorded in a receipt; only
then: ``fresh``, which first checks the receipt against the record (``check-published``), then
per seed: the F run at η_l (experiments/study5, ``study5.role=fresh``: a plain F run with its
endpoint checkpoint); the surrogate r at its endpoint from that checkpoint
(``scripts/study5_surrogate_endpoint.py``); the line scan of its endpoint, a run of its own from
the checkpoint (``scripts/study5_scan.py``, experiment ``study5_scan``; a partial run directory
is moved aside before it runs again); then the checkpoint removed; then the F run at η_u (the same
config), each F run in a launcher campaign of its own; after the last seed, the completion record
(``completion``: every fresh run the launcher journals show as started, with its end and
outcome, written only when no started one lacks an end; kept only while the journals still say
the same), which the analysis requires before any fresh label. The checkpoint is archived and
verified right after the η_l run (``checkpoint --keep``), so that a Pod stop, which wipes the
container disk, loses none: the endpoint's r and the scan fetch it back from that archive, its
sha256 checked, when it is gone (``with-runs --restore-*``). A failure of the endpoint's r or of the
scan is the seed's, recorded; it takes nothing from the η_l run. Each
endpoint checkpoint is archived, verified and then removed from the container disk by
``remove_checkpoint``, which refuses any path but that run's own ``<dir>/<run id>.pt``; a seed's
runs start only while at most one checkpoint is on the disk (``check-checkpoints``), so it holds
at most two. ``controls``: seeds 50000-50005 (those of the campaign's seeds): the two crossed
controls at η_l (experiments/study5_control_order; experiments/study5_control_v0 with v₀ from
seed 50000 + ((s - 50000 + 1) mod 6)), then the ladder's rungs ``gaussian_d``, ``linear_no_d`` and
``linear`` (``scripts/study5_ladder.py``, on the seed's fresh F run at η_l; ``linear_shuffled_d``
runs only on a later decision). ``check-published`` also requires the record to hold exactly the
plan's seeds in order, its n and the plan's GPU code identity, and its sha256 the published
copy's (the receipt's and the sidecar's); its commit is information only. A mismatch stops the
campaign
before any F run. No labeling: labels are opened only after every fresh run has
ended, in a separate step.

**Pod 3** (extension 1, after the registered analysis; docs/study5-preregistration.md,
"Extension 1"), its GPU code the one ``docs/study5-extension-1-code-identity.json`` records
(``EXTENSION_IDENTITY``) and its registration holding the extension (``EXTENSION_HEADING``).
``engineering``: seed 50000's streams fetched from pod 2's archive (``POD2_STREAMS``: the attempt's
inventory of the seed's streams, ``fetch-streams``), pod 2's order control of seed 50000 at η_l
fetched from pod 2's runs (``fetch-run --prefix``, ``POD2_RUNS``), rerun under this checkout with
pod 2's cell (``eng-launch``: the same config hash) and compared by
``scripts/study5_engineering_check.py``; a difference or any failure stops the campaign before
anything else. ``fetch``: the streams of seeds 50001-50015 and the fresh F runs at η_l of seeds
50000-50005 (``POD2_F_RUNS``), the ladder's reference runs. ``ladder``:
per seed 50000-50005 the rungs ``gaussian_d`` and ``linear_no_d`` (``EXTENSION_RUNGS``) by
``scripts/study5_ladder.py --extensions 8`` on the fetched F run (a partial rung run moved aside
first), records under ``RUNS_DIR/study5_records/<ID>/ladder/``. ``orders``: per seed 50006-50015
(``ORDER_SEEDS``) one launcher campaign of four F runs, ``experiments/study5_control_order`` with
``train.order_seed`` = s + 1000 and s + 2000 (``ORDER_OFFSETS``), each at η_l and η_u. No checkpoint
is written, so the close exports none. No labeling. Before anything is written, the preflight
requires an archive id of pod 3's own and a runs root holding nothing of another archive
(``own_root_problems``), and each block and seed group is admitted only when ``ADMIT_FACTOR`` × its
minutes and ``CLOSE_BUDGET`` end before the cap (``deadline --admit``).

**Failures.** Every run goes through ``scripts/launch.py FILE --max-attempts 1`` (one attempt; a
rerun is a recovery round of the run's own, zero_field_refusal.launch, "Recovery rounds"). A step
that fails or exits non-zero is recorded in the journal and the campaign goes on to the next seed
where the step is per seed, except where the plan stops: the engineering check, θ₀'s run, the
surrogate checks, the fetch of study 3's streams, the predictions record and its publication, the
checkpoint count, every archive and every deadline. The checkpoint step refuses a checkpoint
whose run records no sha256 for it (the file stays, and the next seed's count stops the
campaign). A failed step is not run again by a restart; a stop recorded stops again.

**Attempts.** Each execution of the driver is an attempt with its own id (``new_attempt``: the UTC
time and a short random suffix; in the journal's ``driver`` line). Every file that can change
between attempts goes to the bucket under ``study5/<ID>/attempts/<attempt>/``: the launcher
journals and logs, the directories of runs without an end, a scan or ladder run before its
record, the gates' records and the completion record (``runs/``), the snapshots, the plan, the
campaigns and the receipts (``campaign/``), every inventory (``inventories/``). What is written
once keeps its stable place: the records of the scripts, the streams, the directory of a run its
journal shows ended, a scan or ladder run with its record, the checkpoints (and their
inventories, named by the run), the predictions record. So a later attempt never writes an
object an earlier one wrote; one it finds already there is checked identical (``archive.py
--no-clobber``), never replaced. ``restore-runs`` restores the attempts first, in their
chronological order, then each completed stable run directory whole, so that a partial run an
earlier attempt archived never survives beside its completed rerun.

**Resumable.** The driver keeps ``journal.jsonl`` in --out (a start and an end line per step;
each run of a step's output in ``logs/``). Started again with the same arguments from the same
commit (the plan must equal ``plan.json``), it skips a block whose receipt exists (its archive
verified) and, inside a block, every step that ended, every launch whose runs all completed (by the
launcher's journal) and every step whose outputs exist; a step left started without an end (the
Pod stopped) runs again. A gate's record is kept only by its verdict (``gate_problem``): the
engineering check equal for reruns of the plan's GPU code from a clean checkout, the surrogate
checks passed, each written under the plan's GPU code identity from a clean checkout (a commit
is compared by its GPU code identity, never by itself); otherwise it is moved aside beside itself
(``<record>.aside-<UTC>``, never deleted) and the gate runs again. The engineering reruns of
another GPU code are moved aside too (``eng-launch``, to ``RUNS_DIR/aside/``, never deleted), as
the launcher would reuse them by their config hash, and the cells run again in their one recovery
round. A predictions record that does not stand fails its step: a stop. So a stopped campaign goes
on from a new --out with the same --archive-id, from the same commit or a later one of the same
GPU code: records that stand are kept, completed runs are not launched again, identical archives
are no conflict. SIGTERM to the driver kills the running step and runs the close; a killed launch
step's run has used its one attempt and needs a recovery round (docs/operations.md). A ladder
rung's partial run directory is first moved aside (``RUNS_DIR/aside/``), never reused. A launcher
run the Pod stopped is settled by the launcher.

**Where.** --out (the campaign folder; outside the checkout, which must be clean) holds the plan,
the journal, the logs, the launcher campaigns, the inventories, the receipts and study 3's fetched
records. Runs go to RUNS_DIR, the records of the study-5 scripts to
``RUNS_DIR/study5_records/<ID>/``; the endpoint checkpoints to --checkpoints (default
``/root/ckpt/study5``, the container disk). In the bucket (``ARTIFACT_REMOTE``, read from the
environment; the archive key ``ARTIFACT_GCS_KEY`` too; neither is written anywhere):
``study5/<ID>/runs``, ``checkpoints``, ``predictions``, ``inventories`` (the checkpoints') and
``attempts/<attempt>/`` (``runs``, ``campaign``, ``inventories``: "Attempts"), and the fresh
streams under ``data/`` as every stream. Before it starts, the driver checks the checkout
is clean and that its GPU code is the one ``docs/study5-code-identity.json`` records.

**Pod 2's preflight** (every start) also requires, each given by path: the registration at the
checkout's commit ``Status: `registered``` and naming the sha256 of the committed
``docs/study5-fixation.json`` and of the retrospective record (``--retrospective``, of that
fixation record); the n chosen before generation, ``docs/study5-confirmatory-seeds.json`` at that
commit (``{"n", "seeds"}``: 40 seeds 50000-50039 or 24 seeds 50000-50023), its sha256 named in the
registration there and the plan's fresh seeds exactly its seeds (``--n`` defaults to its n;
another is refused); pod 1's engineering-check record (``--engineering-check``) equal for reruns
under the plan's GPU code identity; pod 1's surrogate-checks record (``--surrogate-checks``)
passed, under that identity, of the plan's surrogate (the same sha256). Both plans name one
surrogate spec (``--surrogate``, default ``configs/study5/surrogate-2.yaml``, a path in the
checkout) and its sha256, and pass it to the checks, the predictor and the endpoint's r. A second
set, after failed checks, is a new file there: a GPU path, so a new
``docs/study5-code-identity.json`` and the engineering check again come first.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from zero_field_refusal import code_identity
from zero_field_refusal.analysis.campaign_prep import EXISTS, comment
from zero_field_refusal.analysis.study3_freeze import _stored, freeze_name  # its schema reader
from zero_field_refusal.paths import REPO_ROOT, data_root, runs_root
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT
SELF = "scripts/study5_campaigns.py"
ETA_L, ETA_U = 1.3e-05, 1.83e-05  # study 3's candidate-1 rates of F
BATCH, UPDATES = 20, 25
FRESH_FIRST = 50000
N_CHOICES = (40, 24)
CONTROL_SEEDS = tuple(range(50000, 50006))
RUNGS = ("gaussian_d", "linear_no_d", "linear")  # linear_shuffled_d: only on a later decision
SCAN_EXPERIMENT = "study5_scan"  # line_scan.EXPERIMENT: the scan runs' directory under RUNS_DIR
ENGINEERING_SEED = 10403
STUDY3_CONFIG = "experiments/study3"
STUDY3_F = ("optimizer=adam_frozen", f"train.batch_size={BATCH}")
# Study 3's archive of its confirmation's records and of its streams' inventory, relative to
# ARTIFACT_REMOTE (docs/reports/study3-confirmation-c1.md, "Records")
STUDY3_RUNS_REMOTE = "study3/confirmation/study3-confirmation-c1-20261105/runs-20261105T001016Z"
STUDY3_STREAMS_INVENTORY = "runs/archive/inventory-2026-11-05-study3-confirmation-c1-streams.tsv"
SKIPPED = ("wandb", "judge", "judge-superseded")  # never fetched from study 3's run folders
CHECKPOINTS = Path("/root/ckpt/study5")
IDENTITY_FILE = "docs/study5-code-identity.json"
# Extension 1 (docs/study5-preregistration.md, "Extension 1"): pod 3's GPU code, its runs and inputs
EXTENSION_IDENTITY = "docs/study5-extension-1-code-identity.json"
EXTENSION_HEADING = "### Extension 1 (2026-11-11): the ladder's unbracketed rungs and order repeats"
EXTENSION_RUNGS = ("gaussian_d", "linear_no_d")
LADDER_EXTENSIONS = 8  # the extension's most ×2 (÷2) extensions of a rung's scale grid
ORDER_SEEDS = tuple(range(50006, 50016))  # the ten fresh seeds after the control subset
ORDER_OFFSETS = (1000, 2000)  # train.order_seed = s + offset: two new orders per seed
POD2_ARCHIVE = "study5/pod2-20261109"  # pod 2's archive, relative to ARTIFACT_REMOTE
# The attempt that archived pod 2's streams; a seed's inventory of its two streams there
POD2_STREAMS = (
    f"{POD2_ARCHIVE}/attempts/20261109t135336z-5aaaad/inventories/"
    "streams--s{seed}--archive-streams.tsv"
)
POD2_RUNS = POD2_ARCHIVE + "/runs/study5"  # pod 2's completed run directories
# Pod 2's runs the extension reads, as pod 2's journals record them: the fresh F runs at η_l of the
# ladder's seeds (the rungs' reference runs) and the order control of seed 50000 at η_l (rerun by
# the engineering check)
POD2_F_RUNS = {
    50000: "20261109T192912Z-s5-fresh-1.3e-05-s50000-s50000-f3e969f1-a1",
    50001: "20261109T200448Z-s5-fresh-1.3e-05-s50001-s50001-52976e1f-a1",
    50002: "20261109T204049Z-s5-fresh-1.3e-05-s50002-s50002-99efec8b-a1",
    50003: "20261109T211734Z-s5-fresh-1.3e-05-s50003-s50003-21cff731-a1",
    50004: "20261109T215417Z-s5-fresh-1.3e-05-s50004-s50004-c431ca6c-a1",
    50005: "20261109T223154Z-s5-fresh-1.3e-05-s50005-s50005-d5d976a3-a1",
}
POD2_ORDER_RUN = "20261110T192907Z-s5-order-1.3e-05-s50000-s50000-3173f4e9-a1"
REGISTRATION = "docs/study5-preregistration.md"
FIXATION = "docs/study5-fixation.json"
REGISTERED = re.compile(r"^Status: `registered`", re.MULTILINE)
# The n chosen before generation and its seeds, committed; the registration names its sha256
SEEDS_FILE = "docs/study5-confirmatory-seeds.json"
# The surrogate's spec (scripts/study5_surrogate_checks.py, study5_predict.py and
# study5_surrogate_endpoint.py take it by --surrogate): a second set, if the checks at θ₀ fail,
# goes in a new file of the checkout, named here by --surrogate of both plans
SURROGATE = "configs/study5/surrogate-2.yaml"  # the second set (surrogate.yaml without R4)
RECEIPT_NAME = "predictions-receipt.json"  # beside the predictions record
CLOSE = "close"
# Replaced in a step's argv by the execution attempt's id when the step runs (``execute``)
ATTEMPT = "{attempt}"
END_MINUTES = 15  # the deadline after the close: the watcher stops the Pod
MARGIN = 60
COMPLETED, ERROR, STOPPED = 0, 1, 3
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,23}")
# Minutes per step. Measured on study 3's L40S Pod where a unit cost exists: an F run 6.9; an
# evaluation without training 3.1; a K-only point 1.5; five scan points (five evaluations) 15.5;
# a stream 91.7 s per 1000 responses (1500 per seed: 500 training, 1000 for v₀); the predictor
# 1.7 per seed; a ladder rung three K-only points, two more for the matched-K step (K at c₁ and
# at most one refinement c₂, ladder.refine) and one evaluation, 10.6; an endpoint checkpoint's
# save 0.5. Assumed where none is measured, as named. They set each block's and each
# seed group's deadline (2 × the sum + MARGIN) and each step's timeout (timeout_of).
F_RUN, EVALUATION, K_POINT, SCAN, CKPT_SAVE = 6.9, 3.1, 1.5, 15.5, 0.5
SCAN_LOAD = 2.0  # assumed: two models loaded, a 5.24 GB checkpoint hashed and read, the panels
STREAM_PER_RESPONSE = 91.7 / 60 / 1000
MINUTES = {
    "fetch-seed": 1.0,  # assumed: one seed's two streams, about 10 MB
    "fetch-seeds": 15.0,  # assumed: 60 seeds' streams, about 0.6 GB
    "fetch-run": 1.0,  # assumed: a run directory, under 1 MB
    "engineering-runs": 2 * F_RUN,
    "engineering-check": 1.0,  # assumed: a CPU comparison of two run directories
    "theta0": EVALUATION,
    "surrogate": EVALUATION,  # assumed: forward passes on 200 prompts, as an evaluation
    "predict": 1.7,
    "features": 0.5,  # assumed: no model, 500 responses decoded
    "training-stream": 500 * STREAM_PER_RESPONSE,
    "preconditioner-stream": 1000 * STREAM_PER_RESPONSE,
    "fresh-lower": F_RUN + CKPT_SAVE,
    "scan": SCAN + SCAN_LOAD,
    "fresh-upper": F_RUN,
    # assumed, ≈ 1 min: a 5.24 GB checkpoint hashed and read, the surrogate's 2 × 1200 rows
    "surrogate-endpoint": 1.0,
    "checkpoint": 4.0,  # assumed: a 5.24 GB checkpoint hashed and uploaded at about 25 MB/s
    "checkpoint-remove": 0.5,  # assumed: the checkpoint hashed again, then removed
    "controls": 2 * F_RUN,
    "rung": 5 * K_POINT + EVALUATION,
    # extension 1, at most: the first grid's three K points, eight extensions, the two matched-K
    # points (c₁ and one refinement c₂) and one evaluation
    "rung-extended": (3 + LADDER_EXTENSIONS + 2) * K_POINT + EVALUATION,
    "orders": 4 * F_RUN,
    "archive": 1.0,  # assumed: a seed's run directories and records, a few MB
    "small": 0.5,
}
TIMEOUT_FACTOR, TIMEOUT_SLACK = 3, 20  # a step's timeout: 3 × its minutes + 20, in minutes
# The close's export window (2 × two checkpoints' archive + MARGIN = 76): pod 2 admits a block or
# a seed group only while its window and this one both fit before the cap ZFR_DEADLINE_EPOCH
EXPORT_WINDOW = int(math.ceil(2 * 2 * MINUTES["checkpoint"] + 60))
NOT_ADMITTED = 4  # the deadline step's exit when a window does not fit before the cap
# Pod 3 admits a block or a seed group only when ADMIT_FACTOR × its estimated minutes (its steps
# and its archive) and the close's CLOSE_BUDGET end before the cap ZFR_DEADLINE_EPOCH; its
# watchdog window stays 2 × the minutes + MARGIN, cut at the cap
ADMIT_FACTOR = 1.5
CLOSE_BUDGET = 10  # assumed: the close's snapshot and its archive, in minutes


def timeout_of(minutes: float) -> int:
    """A step's timeout in seconds: past it the step is killed and counts as failed."""
    return int(round(60 * (TIMEOUT_FACTOR * minutes + TIMEOUT_SLACK)))


def deadline_minutes(minutes: float) -> int:
    """A deadline window for work estimated at ``minutes``: 2 × it + MARGIN."""
    return int(math.ceil(2 * minutes + MARGIN))


class StepError(RuntimeError):
    """A step that cannot be done: its reason is printed and the step exits 1."""


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    on_fail: str = "record"  # stop | record | skip-group
    kind: str = "run"  # run | launch | archive | receipt | deadline
    outputs: tuple[str, ...] = ()  # all present: the step is done (kept)
    exists_ok: bool = False  # generate_teacher.py's refusal of an existing stream is no failure
    group: str = ""  # a seed's steps, for skip-group
    always: bool = False  # run at every start (the close)
    minutes: float = 1.0
    timeout: int | None = None  # seconds; past it the step is killed and counts as failed
    gate: str = ""  # engineering | surrogate | predictions | completion: kept only by its verdict
    kept_if_runs: tuple[str, ...] = ()  # "<campaign>\t<label>": done when each run completed


@dataclass(frozen=True)
class Block:
    name: str
    steps: tuple[Step, ...]


@dataclass
class Plan:
    pod: str
    archive_id: str
    out: Path
    runs: Path
    data: Path
    checkpoints: Path
    commit: str
    gpu_code: str  # the GPU code identity at the commit (code_identity.gpu_code_identity)
    n: int | None
    blocks: list[Block]
    campaigns: dict[str, str] = field(default_factory=dict)  # launcher campaign name -> YAML
    surrogate: str = SURROGATE  # the surrogate's spec, relative to the checkout
    surrogate_sha256: str = ""  # its sha256 (surrogate.load_spec)

    def to_json(self) -> dict[str, Any]:
        return {
            "pod": self.pod,
            "archive_id": self.archive_id,
            "out": str(self.out),
            "runs": str(self.runs),
            "data": str(self.data),
            "checkpoints": str(self.checkpoints),
            "commit": self.commit,
            "gpu_code": self.gpu_code,
            "n": self.n,
            "surrogate": self.surrogate,
            "surrogate_sha256": self.surrogate_sha256,
            "blocks": [
                {"name": b.name, "steps": [asdict(s) for s in b.steps]} for b in self.blocks
            ],
            "campaigns": {
                name: hashlib.sha256(text.encode()).hexdigest()
                for name, text in sorted(self.campaigns.items())
            },
        }


def study3_cohort(repo: Path = REPO) -> tuple[list[int], list[str]]:
    """Study 3's A-F confirmation cohort (its analysis freeze, in its order) and the run ids of
    its F runs on seed 10403, as the freeze names them (lo and hi by K)."""
    record, _, problems = _stored(repo, freeze_name(1), 1)
    if record is None:
        raise StepError(f"{freeze_name(1)}: " + "; ".join(problems))
    blocks = record["comparisons"]["A-F"]["blocks"]
    seeds = [int(block["seed"]) for block in blocks]
    engineering = next((b for b in blocks if b["seed"] == ENGINEERING_SEED), None)
    if engineering is None:
        raise StepError(f"seed {ENGINEERING_SEED} is not in study 3's A-F cohort")
    runs = [engineering["arms"]["F"][side] for side in ("lo", "hi")]
    return seeds, runs


def surrogate_spec(path: str, repo: Path | None = None) -> tuple[str, str]:
    """The surrogate's spec as the plan names it: a path relative to the checkout (``SURROGATE``,
    or a second set's new file), and its sha256 (``surrogate.load_spec``)."""
    from zero_field_refusal.surrogate import load_spec

    repo = REPO if repo is None else repo
    rel = PurePosixPath(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"--surrogate {path}: a path inside the checkout, relative to it")
    if not (repo / rel).is_file():
        raise ValueError(f"--surrogate {path}: no such file in the checkout")
    return rel.as_posix(), load_spec(repo / rel).sha256


def fresh_seeds(n: int) -> list[int]:
    if n not in N_CHOICES:
        raise ValueError(
            f"--n {n}: the registration's n is 40, or 24 when decided before generation"
        )
    return list(range(FRESH_FIRST, FRESH_FIRST + n))


def source_seed(seed: int) -> int:
    """The v₀ control's source: 50000 + ((s − 50000 + 1) mod 6)."""
    return FRESH_FIRST + ((seed - FRESH_FIRST + 1) % len(CONTROL_SEEDS))


def rate(value: float) -> str:
    return f"optimizer.lr={value!r}"


def retro_args(seed: int) -> tuple[str, ...]:
    """A study-3 seed's F run at η_l, as study 3 ran it."""
    return (STUDY3_CONFIG, *STUDY3_F, rate(ETA_L), "study3.role=confirmation", f"seed={seed}")


FRESH = ("experiments/study5", "study5.role=fresh")  # a fresh seed's plain F run


def fresh_args(seed: int) -> tuple[str, ...]:
    """A fresh seed's F run at η_l (its checkpoint's directory is not in its config)."""
    return (*FRESH, rate(ETA_L), f"seed={seed}")


def label(cell: dict[str, Any]) -> str:
    return f"{cell['name']}/s{cell['seeds'][0]}"


def campaign_text(name: str, header: dict[str, Any], cells: list[dict[str, Any]]) -> str:
    notes = comment(
        f"Written by {SELF} from the plan of {header['pod']} ({header['archive_id']}); "
        "do not edit. Run by its driver: scripts/launch.py FILE --max-attempts 1."
    )
    data = {"name": name, "header": header, "cells": cells}
    return "\n".join(notes) + "\n" + yaml.safe_dump(data, sort_keys=False)


class Builder:
    """The steps of one plan, with its folders and the bucket's places."""

    def __init__(
        self, pod: str, archive_id: str, out: Path, runs: Path, data: Path, checkpoints: Path
    ) -> None:
        if not _ID.fullmatch(archive_id):
            raise ValueError(
                f"--archive-id {archive_id!r}: lowercase letters, digits and '-', at most 24"
            )
        if not checkpoints.is_absolute():
            raise ValueError(f"--checkpoints {checkpoints}: an absolute directory")
        self.pod, self.id, self.out = pod, archive_id, Path(out)
        self.runs, self.data, self.checkpoints = Path(runs), Path(data), Path(checkpoints)
        self.records = self.runs / "study5_records" / archive_id
        self.base = f"study5/{archive_id}"
        # the execution attempt's own places (``ATTEMPT`` is replaced when a step runs): every
        # mutable file goes there, so that a later attempt never writes an earlier one's object
        self.attempt = f"{self.base}/attempts/{ATTEMPT}"
        self.campaigns: dict[str, str] = {}
        self.blocks: list[Block] = []
        self.inventories: list[str] = []
        self.current: list[Step] = []
        self.reserve = 0  # minutes kept free before the cap for the close's export (pod 2)
        self.admit = False  # pod 3: admitted only when the work and the close fit (``deadline``)

    def own(self, *args: str) -> tuple[str, ...]:
        return (SELF, *map(str, args))

    def step(self, name: str, argv, *, minutes: str | float = "small", **kw) -> Step:
        value = MINUTES[minutes] if isinstance(minutes, str) else float(minutes)
        found = Step(
            name=name, argv=tuple(map(str, argv)), minutes=value, timeout=timeout_of(value), **kw
        )
        self.current.append(found)
        return found

    def launcher(self, name: str, block: str, cells: list[dict[str, Any]]) -> str:
        """A launcher campaign of this plan; its file's path."""
        full = f"study5-{self.id}-{name}"
        header = {"study": 5, "pod": self.pod, "archive_id": self.id, "block": block}
        self.campaigns[full] = campaign_text(full, header, cells)
        return str(self.out / "campaigns" / f"{full}.yaml")

    def launch(self, name: str, block: str, cells, *, minutes: str, **kw) -> tuple[Step, str]:
        path = self.launcher(name, block, cells)
        full = Path(path).stem
        group = kw.get("group", "")
        tag = f"{block}/{group}" if group else block
        found = self.step(
            f"{tag}/launch-{name.removesuffix('-' + group)}" if group else f"{tag}/launch",
            ("scripts/launch.py", path, "--max-attempts", "1"),
            kind="launch",
            minutes=minutes,
            # a run that completed is not launched again: its launcher would refuse a removed
            # checkpoint as a conflict
            kept_if_runs=tuple(f"{full}\t{label(cell)}" for cell in cells),
            **kw,
        )
        return found, full

    def archive(
        self,
        name: str,
        *,
        root: Path,
        remote: str,
        paths=(),
        campaign: str | tuple[str, ...] | None = None,
        streams: int | None = None,
        group: str = "",
        kind: str = "archive",
        mutable=(),
        run_dirs=(),
        attempt_remote: str | None = None,
    ) -> Step:
        """An archive step; ``kind`` ``archive`` runs also after a stop and in a skipped group
        (it archives what exists), ``run`` only when its group goes on (a seed's streams).
        ``paths`` are written once and go to ``remote``; ``mutable`` paths (a gate's record) and
        a ``run_dirs`` entry ``(dir, record)`` whose record does not exist yet (a partial run)
        go to ``attempt_remote``, as do the launcher campaigns' journals and the directories of
        their runs that have no end (``cmd_archive``)."""
        inventory = self.out / "inventories" / f"{name.replace('/', '--')}.tsv"
        self.inventories.append(str(inventory))
        argv = [
            "archive",
            "--inventory",
            inventory,
            "--remote",
            remote,
            "--root",
            root,
            "--inventories-remote",
            f"{self.attempt}/inventories",
        ]
        if attempt_remote is not None:
            argv += ["--attempt-remote", attempt_remote]
        for path in paths:
            argv += ["--path", path]
        for path in mutable:
            argv += ["--mutable-path", path]
        for path, record in run_dirs:
            argv += ["--run-dir", f"{path}={record}"]
        campaigns = (campaign,) if isinstance(campaign, str) else tuple(campaign or ())
        for name_ in campaigns:
            argv += ["--campaign", name_]
        if campaigns:
            argv += ["--runs", self.runs]
        if streams is not None:
            argv += ["--streams", streams]
        return self.step(
            name,
            self.own(*argv),
            kind=kind,
            on_fail="stop",
            outputs=(str(inventory),),
            group=group,
            minutes="archive",
        )

    def runs_archive(
        self,
        name: str,
        *,
        paths=(),
        campaign: str | tuple[str, ...] | None = None,
        group: str = "",
        mutable=(),
        run_dirs=(),
    ) -> Step:
        return self.archive(
            name,
            root=self.runs,
            remote=f"{self.base}/runs",
            paths=paths,
            campaign=campaign,
            group=group,
            mutable=mutable,
            run_dirs=run_dirs,
            attempt_remote=f"{self.attempt}/runs",
        )

    def record(self, *parts: str) -> Path:
        return self.records.joinpath(*parts)

    def rel(self, path: Path) -> str:
        return path.relative_to(self.runs).as_posix()

    def begin(self) -> None:
        self.current, self.inventories = [], []

    def end(self, block: str) -> None:
        """The block's deadline first, its snapshot and receipt last."""
        inventory = self.out / "inventories" / f"snapshot-{block}.tsv"
        self.step(
            f"{block}/snapshot",
            self.own(
                "snapshot",
                "--out",
                self.out,
                "--label",
                block,
                "--remote",
                f"{self.attempt}/campaign",
                "--inventories-remote",
                f"{self.attempt}/inventories",
            ),
            kind="archive",
            on_fail="stop",
            outputs=(str(inventory),),
            minutes="archive",
        )
        self.inventories.append(str(inventory))
        self.step(
            f"{block}/receipt",
            self.own("receipt", "--out", self.out, "--block", block, *self.inventories),
            kind="receipt",
            on_fail="stop",
            outputs=(str(self.out / "receipts" / f"{block}.json"),),
        )
        # The block's window covers its steps outside the seed groups; each group sets its own
        # first, so a dead driver or a hung step leaves the Pod idle at most one group's window.
        loose = sum(s.minutes for s in self.current if not s.group)
        steps = [self.deadline(f"{block}/deadline", loose)]
        for found in self.current:
            if found.group and (steps[-1].group != found.group):
                minutes = sum(s.minutes for s in self.current if s.group == found.group)
                steps.append(
                    self.deadline(f"{block}/{found.group}/deadline", minutes, group=found.group)
                )
            steps.append(found)
        self.blocks.append(Block(block, tuple(steps)))
        self.current = []

    def deadline(self, name: str, minutes: float, group: str = "") -> Step:
        """The deadline file set to now + 2 × ``minutes`` + MARGIN (the cap stays), again at every
        start that reaches it: a restart rewrites the window. With ``reserve`` (pod 2), the
        block or group is admitted only when its window and the close's export window both fit
        before the cap (``cmd_deadline``); otherwise the step fails, a stop: no further group,
        the close follows."""
        argv = ["deadline", "--minutes", deadline_minutes(minutes)]
        if self.reserve:
            argv += ["--reserve", self.reserve]
        if self.admit:
            argv += ["--admit", int(math.ceil(ADMIT_FACTOR * minutes + CLOSE_BUDGET))]
        return Step(
            name=name,
            argv=self.own(*argv),
            on_fail="stop",
            kind="deadline",
            always=True,
            group=group,
            timeout=timeout_of(MINUTES["small"]),
        )

    def close(
        self,
        checkpoints: tuple[tuple[str, str], ...] = (),
        records: tuple[str, ...] = (),
        json_from: Path | None = None,
    ) -> None:
        """The close: with ``checkpoints`` (each fresh η_l run's launcher campaign and label),
        first a deadline window for their export, then the export of every recorded endpoint
        checkpoint not yet verified in the archive (``cmd_export_checkpoints``; its record in
        ``close/``); with ``records`` (folders relative to the runs root, pod 4), those folders
        archived to the attempt's own place, so that a block that never reached its archive keeps
        its records; with ``json_from`` (pod 4's vector folders), first a copy of every JSON
        record there, its relative path kept, into the records folder (``json-snapshot``: a record
        written before its settle; never a tensor); then the plan, the campaigns, the receipts and
        that record to the bucket, and the deadline 15 minutes ahead."""
        self.current = []
        include = ["plan.json", "campaigns", "receipts"]
        if json_from is not None:
            self.step(f"{CLOSE}/json-snapshot", self.own(
                "json-snapshot", "--src", json_from, "--dest",
                self.records / "vec-json" / ATTEMPT), on_fail="record", always=True)  # fmt: skip
        if records:
            include.append(CLOSE)
            inventory = self.out / CLOSE / f"records-{ATTEMPT}.tsv"
            argv = ["archive", "--inventory", inventory, "--remote", f"{self.attempt}/runs",
                    "--root", self.runs, "--inventories-remote", f"{self.attempt}/inventories",
                    "--attempt-remote", f"{self.attempt}/runs"]  # fmt: skip
            for path in records:
                argv += ["--mutable-path", path]
            self.step(f"{CLOSE}/records", self.own(*argv), on_fail="record", always=True,
                      kind="archive", minutes="archive")  # fmt: skip
        if checkpoints:
            # set in order, before the export: the close's own deadline comes last
            self.step(f"{CLOSE}/export-deadline",
                      self.own("deadline", "--minutes", EXPORT_WINDOW, "--needs",
                               2 * MINUTES["checkpoint"], "--record",
                               self.out / CLOSE / f"export-deadline-{ATTEMPT}.json"),
                      on_fail="record", always=True)  # fmt: skip
            argv = ["export-checkpoints", "--runs", self.runs, "--dir", self.checkpoints,
                    "--remote", f"{self.base}/checkpoints", "--inventories-remote",
                    f"{self.base}/inventories", "--inventories", self.out / "inventories",
                    "--record", self.out / CLOSE / f"checkpoints-{ATTEMPT}.json"]  # fmt: skip
            for campaign, run_label in checkpoints:
                argv += ["--run", f"{campaign}={run_label}"]
            self.step(f"{CLOSE}/checkpoints", self.own(*argv), on_fail="record", always=True,
                      minutes=2 * MINUTES["checkpoint"])  # fmt: skip
            if CLOSE not in include:
                include.append(CLOSE)
        self.step(
            f"{CLOSE}/archive",
            self.own(
                "snapshot",
                "--out",
                self.out,
                "--label",
                CLOSE,
                "--unique",
                *(x for name in include for x in ("--include", name)),
                "--remote",
                f"{self.attempt}/campaign",
                "--inventories-remote",
                f"{self.attempt}/inventories",
            ),
            kind="archive",
            on_fail="record",
            always=True,
        )
        self.step(
            f"{CLOSE}/deadline",
            self.own(
                "deadline",
                "--minutes",
                END_MINUTES,
                "--record",
                self.out / CLOSE / f"deadline-{ATTEMPT}.json",
            ),
            kind="deadline",
            on_fail="record",
            always=True,
        )
        self.blocks.append(Block(CLOSE, tuple(self.current)))

    def plan(self, commit: str, gpu_code: str, n: int | None,
             surrogate: tuple[str, str]) -> Plan:  # fmt: skip
        return Plan(
            pod=self.pod,
            archive_id=self.id,
            out=self.out,
            runs=self.runs,
            data=self.data,
            checkpoints=self.checkpoints,
            commit=commit,
            gpu_code=gpu_code,
            n=n,
            blocks=self.blocks,
            campaigns=self.campaigns,
            surrogate=surrogate[0],
            surrogate_sha256=surrogate[1],
        )

    def fetch_streams(self, name: str, seeds: list[int], minutes: str) -> Step:
        return self.step(
            name,
            self.own(
                "fetch-streams",
                "--inventory",
                STUDY3_STREAMS_INVENTORY,
                "--data",
                self.data,
                "--work",
                self.out,
                "--seeds",
                *seeds,
            ),
            on_fail="stop",
            minutes=minutes,
        )


def pod1_plan(
    *,
    out: Path,
    runs: Path,
    data: Path,
    checkpoints: Path = CHECKPOINTS,
    archive_id: str,
    commit: str,
    gpu_code: str,
    repo: Path = REPO,
    surrogate: str = SURROGATE,
) -> Plan:
    """Pod 1's plan (module docstring, "Pod 1")."""
    spec = surrogate_spec(surrogate)
    b = Builder("pod1", archive_id, out, runs, data, checkpoints)
    seeds, f_runs = study3_cohort(repo)

    b.begin()  # the engineering check: any failure stops the campaign
    b.fetch_streams("engineering/fetch-streams", [ENGINEERING_SEED], "fetch-seed")
    fetched = b.out / "fetched" / "study3"
    for run_id in f_runs:
        b.step(
            f"engineering/fetch-run-{run_id[-11:]}",
            b.own("fetch-run", "--run-id", run_id, "--dest", fetched),
            on_fail="stop",
            outputs=(str(fetched / run_id / "meta.json"),),
            minutes="fetch-run",
        )
    cells = [
        {
            "name": f"s5-eng-adam_frozen-{eta!r}-s{ENGINEERING_SEED}",
            "config": STUDY3_CONFIG,
            "overrides": [*STUDY3_F, rate(eta), "study3.role=confirmation"],
            "seeds": [ENGINEERING_SEED],
        }
        for eta in (ETA_L, ETA_U)
    ]
    # The reruns are reused only when they ran under the plan's GPU code: the launcher reuses a
    # completed run of the same config hash from any campaign (eng-launch)
    eng_file = b.launcher("engineering", "engineering", cells)
    eng = Path(eng_file).stem
    b.step(
        "engineering/launch",
        b.own("eng-launch", "--runs", b.runs, "--campaign-file", eng_file, "--gpu-code", gpu_code),
        kind="launch",
        on_fail="stop",
        minutes="engineering-runs",
    )
    check = b.record("engineering-check.json")
    labels = [label(cell) for cell in cells]
    b.step(
        "engineering/check",
        b.own(
            "with-runs",
            "--runs",
            b.runs,
            "--campaign",
            eng,
            "--",
            "scripts/study5_engineering_check.py",
            "--archived",
            *(fetched / r for r in f_runs),
            "--rerun",
            *(f"{{run:{x}}}" for x in labels),
            "--out",
            check,
        ),
        on_fail="stop",
        outputs=(str(check),),
        minutes="engineering-check",
        gate="engineering",
    )
    b.runs_archive("engineering/archive", mutable=(b.rel(check),), campaign=eng)
    b.end("engineering")

    b.begin()  # θ₀'s evaluation on the 521 prompts: the analysis needs it, so a failure stops
    cell = {
        "name": f"s5-theta0-s{FRESH_FIRST}",
        "config": "experiments/study5_theta0",
        "entry": "theta0",
        "overrides": [],
        "seeds": [FRESH_FIRST],
    }
    _, theta0 = b.launch("theta0", "theta0", [cell], minutes="theta0", on_fail="stop")
    b.runs_archive("theta0/archive", campaign=theta0)
    b.end("theta0")

    b.begin()  # the surrogate checks: any non-zero exit stops the campaign
    checks = b.record("surrogate-checks.json")
    b.step(
        "surrogate/checks",
        (
            "scripts/study5_surrogate_checks.py",
            "experiments/study5_theta0",
            "--out",
            checks,
            "--surrogate",
            spec[0],
        ),  # fmt: skip
        on_fail="stop",
        outputs=(str(checks),),
        minutes="surrogate",
        gate="surrogate",
    )
    b.runs_archive("surrogate/archive", mutable=(b.rel(checks),))
    b.end("surrogate")

    b.begin()  # the retrospective predictor on study 3's 60 A-F seeds
    b.fetch_streams("retrospective/fetch-streams", seeds, "fetch-seeds")
    for seed in seeds:
        group = f"s{seed}"
        predict = b.record("predictions", "retro", f"s{seed}.json")
        features = b.record("features", "retro", f"s{seed}.json")
        b.step(
            f"retrospective/{group}/predict",
            (
                "scripts/study5_predict.py",
                *retro_args(seed),
                "--out",
                predict,
                "--surrogate",
                spec[0],
            ),  # fmt: skip
            outputs=(str(predict),),
            group=group,
            minutes="predict",
        )
        b.step(
            f"retrospective/{group}/features",
            ("scripts/study5_stream_features.py", *retro_args(seed), "--out", features),
            outputs=(str(features),),
            group=group,
            minutes="features",
        )
        b.runs_archive(
            f"retrospective/{group}/archive", paths=(b.rel(predict), b.rel(features)), group=group
        )
    b.end("retrospective")
    b.close()
    return b.plan(commit, gpu_code, None, spec)


def pod2_plan(
    *,
    out: Path,
    runs: Path,
    data: Path,
    checkpoints: Path = CHECKPOINTS,
    archive_id: str,
    commit: str,
    gpu_code: str,
    n: int = 40,
    surrogate: str = SURROGATE,
) -> Plan:
    """Pod 2's plan (module docstring, "Pod 2")."""
    seeds = fresh_seeds(n)
    spec = surrogate_spec(surrogate)
    b = Builder("pod2", archive_id, out, runs, data, checkpoints)
    b.reserve = EXPORT_WINDOW  # every block and group keeps the close's export inside the cap
    teacher = ("scripts/generate_teacher.py", *FRESH, rate(ETA_L))

    b.begin()  # streams, then the predictor and the features of every fresh seed
    for seed in seeds:
        group = f"s{seed}"
        b.step(
            f"streams/{group}/training",
            (*teacher, "--stream", "training", "--seeds", seed, "--responses", UPDATES * BATCH),
            on_fail="skip-group",
            group=group,
            minutes="training-stream",
        )
        b.step(
            f"streams/{group}/preconditioner",
            (*teacher, "--stream", "preconditioner", "--seeds", seed),
            on_fail="skip-group",
            exists_ok=True,
            group=group,
            minutes="preconditioner-stream",
        )
        b.archive(
            f"streams/{group}/archive-streams",
            root=b.data,
            remote="data",
            streams=seed,
            group=group,
            kind="run",  # a seed whose generation failed has no stream to archive
        )
        predict = b.record("predictions", "fresh", f"s{seed}.json")
        features = b.record("features", "fresh", f"s{seed}.json")
        b.step(
            f"streams/{group}/predict",
            (
                "scripts/study5_predict.py",
                *fresh_args(seed),
                "--out",
                predict,
                "--surrogate",
                spec[0],
            ),  # fmt: skip
            outputs=(str(predict),),
            group=group,
            minutes="predict",
        )
        b.step(
            f"streams/{group}/features",
            ("scripts/study5_stream_features.py", *fresh_args(seed), "--out", features),
            outputs=(str(features),),
            group=group,
            minutes="features",
        )
        b.runs_archive(
            f"streams/{group}/archive", paths=(b.rel(predict), b.rel(features)), group=group
        )
    b.end("streams")

    b.begin()  # the predictions record, in the bucket and verified before any fresh F run
    record = b.record("predictions-record.json")
    receipt = b.record(RECEIPT_NAME)
    b.step(
        "predictions/record",
        b.own(
            "predictions",
            "--records",
            b.record("predictions", "fresh"),
            "--out",
            record,
            "--seeds",
            *seeds,
        ),
        on_fail="stop",
        outputs=(str(record),),
        gate="predictions",
    )
    b.step(
        "predictions/publish",
        b.own(
            "publish",
            "--record",
            record,
            "--receipt",
            receipt,
            "--remote",
            f"{b.base}/predictions",
            "--inventory",
            b.out / "inventories" / "predictions.tsv",
        ),
        on_fail="stop",
        outputs=(str(receipt),),
        minutes="archive",
    )
    b.inventories.append(str(b.out / "inventories" / "predictions.tsv"))
    b.end("predictions")

    b.begin()  # the fresh F runs: η_l with its checkpoint, its endpoint's r and scan, then η_u
    b.step(
        "fresh/check-published",
        b.own(
            "check-published",
            "--record",
            record,
            "--receipt",
            receipt,
            "--gpu-code",
            gpu_code,
            "--seeds",
            *seeds,
        ),
        on_fail="stop",
    )
    fresh: dict[int, tuple[str, str]] = {}
    uppers: list[str] = []
    for seed in seeds:
        group = f"s{seed}"
        lower = {
            "name": f"s5-fresh-{ETA_L!r}-s{seed}",
            "config": FRESH[0],
            "overrides": [*FRESH[1:], rate(ETA_L), f"tracking.endpoint_checkpoint={b.checkpoints}"],
            "seeds": [seed],
        }
        upper = {
            "name": f"s5-fresh-{ETA_U!r}-s{seed}",
            "config": FRESH[0],
            "overrides": [*FRESH[1:], rate(ETA_U)],
            "seeds": [seed],
        }
        b.step(
            f"fresh/{group}/check-checkpoints",
            b.own("check-checkpoints", "--dir", b.checkpoints, "--max", 1),
            on_fail="stop",
            group=group,
        )
        # η_l alone first; its checkpoint archived and verified at once (a Pod stop wipes the
        # container disk), then r at its endpoint and its scan, each restoring the checkpoint
        # from that archive when it is gone, then the local file removed, before η_u starts
        _, name = b.launch(
            f"fresh-lower-{group}", "fresh", [lower], minutes="fresh-lower", group=group
        )
        fresh[seed] = (name, label(lower))
        inventory = b.out / "inventories" / f"checkpoint-{group}.tsv"
        where = ("--runs", b.runs, "--campaign", name)
        store = ("--remote", f"{b.base}/checkpoints", "--inventories-remote",
                 f"{b.base}/inventories")  # fmt: skip
        checkpoint = ("checkpoint", *where, "--label", label(lower), "--dir", b.checkpoints,
                      "--inventory", inventory, *store)  # fmt: skip
        b.step(f"fresh/{group}/checkpoint-archive", b.own(*checkpoint, "--keep"), group=group,
               minutes="checkpoint")  # fmt: skip
        restore = ("--restore-label", label(lower), "--restore-dir", b.checkpoints,
                   "--restore-remote", f"{b.base}/checkpoints", "--restore-inventory", inventory,
                   "--restore-inventories-remote", f"{b.base}/inventories")  # fmt: skip
        # r at the η_l run's endpoint, from its checkpoint; a failure is the seed's, recorded
        endpoint = b.record("surrogate-endpoint", f"s{seed}.json")
        b.step(
            f"fresh/{group}/surrogate-endpoint",
            b.own(
                "with-runs",
                *where,
                *restore,
                "--",
                "scripts/study5_surrogate_endpoint.py",
                *fresh_args(seed),
                "--f-run",
                f"{{run:{label(lower)}}}",
                "--checkpoints",
                b.checkpoints,
                "--out",
                endpoint,
                "--surrogate",
                spec[0],
            ),
            outputs=(str(endpoint),),
            group=group,
            minutes="surrogate-endpoint",
        )
        # the line scan of the η_l run's endpoint, a run of its own from the checkpoint; a
        # partial run directory (the Pod cut it off) is moved aside before it runs again
        scan = b.record("scan", f"s{seed}.json")
        scan_id = f"s5-{b.id}-scan-s{seed}"
        b.step(
            f"fresh/{group}/scan",
            b.own(
                "with-runs",
                *where,
                "--aside",
                f"{SCAN_EXPERIMENT}/{scan_id}",
                "--unless",
                scan,
                *restore,
                "--",
                "scripts/study5_scan.py",
                *fresh_args(seed),
                "--f-run",
                f"{{run:{label(lower)}}}",
                "--checkpoints",
                b.checkpoints,
                "--out",
                scan,
                "--run-id",
                scan_id,
            ),
            outputs=(str(scan),),
            group=group,
            minutes="scan",
        )
        # the local file removed (guarded: archived and verified, its recorded sha256)
        b.step(f"fresh/{group}/checkpoint", b.own(*checkpoint), group=group,
               minutes="checkpoint-remove")  # fmt: skip
        b.inventories.append(str(b.out / "inventories" / f"checkpoint-{group}.tsv"))  # if any
        _, upper_name = b.launch(
            f"fresh-upper-{group}", "fresh", [upper], minutes="fresh-upper", group=group
        )
        uppers.append(upper_name)
        b.runs_archive(
            f"fresh/{group}/archive",
            paths=(b.rel(endpoint), b.rel(scan)),
            run_dirs=((f"{SCAN_EXPERIMENT}/{scan_id}", b.rel(scan)),),
            campaign=(name, upper_name),
            group=group,
        )
    # every fresh run that started has ended: the record of their ends, before any fresh label
    completion = b.record("fresh-completion.json")
    lowers = [fresh[s][0] for s in seeds]
    b.step(
        "fresh/completion",
        b.own("completion", "--runs", b.runs, "--out", completion,
              *(x for c in (*lowers, *uppers) for x in ("--campaign", c))),
        outputs=(str(completion),),
        gate="completion",
    )  # fmt: skip
    b.runs_archive("fresh/completion-archive", mutable=(b.rel(completion),))
    b.end("fresh")

    b.begin()  # the crossed controls and the ladder, seeds 50000-50005
    for seed in [s for s in seeds if s in CONTROL_SEEDS]:
        group = f"s{seed}"
        order = {
            "name": f"s5-order-{ETA_L!r}-s{seed}",
            "config": "experiments/study5_control_order",
            "overrides": [rate(ETA_L)],
            "seeds": [seed],
        }
        v0 = {
            "name": f"s5-v0-{ETA_L!r}-s{seed}",
            "config": "experiments/study5_control_v0",
            "overrides": [rate(ETA_L), f"optimizer.preconditioner.source_seed={source_seed(seed)}"],
            "seeds": [seed],
        }
        _, controls = b.launch(
            f"controls-{group}", "controls", [order, v0], minutes="controls", group=group
        )
        name, lower = fresh[seed]
        ladder_paths, ladder_runs = [], []
        for rung in RUNGS:
            out_record = b.record("ladder", f"s{seed}-{rung}.json")
            run_id = f"s5-{b.id}-ladder-{rung}-s{seed}"
            b.step(
                f"controls/{group}/ladder-{rung}",
                b.own(
                    "with-runs",
                    "--runs",
                    b.runs,
                    "--campaign",
                    name,
                    "--aside",
                    f"study5_ladder/{run_id}",
                    "--unless",
                    out_record,
                    "--",
                    "scripts/study5_ladder.py",
                    *fresh_args(seed),
                    "--rung",
                    rung,
                    "--f-run",
                    f"{{run:{lower}}}",
                    "--out",
                    out_record,
                    "--run-id",
                    run_id,
                ),
                outputs=(str(out_record),),
                group=group,
                minutes="rung",
            )
            ladder_paths.append(b.rel(out_record))
            ladder_runs.append((f"study5_ladder/{run_id}", b.rel(out_record)))
        b.runs_archive(f"controls/{group}/archive", paths=ladder_paths, run_dirs=ladder_runs,
                       campaign=controls, group=group)  # fmt: skip
    b.end("controls")
    b.close(checkpoints=tuple(fresh[s] for s in seeds))
    return b.plan(commit, gpu_code, n, spec)


def order_cell(seed: int, offset: int, eta: float) -> dict[str, Any]:
    """Extension 1's F run of ``seed`` in the order of ``train.order_seed`` = seed + offset."""
    return {
        "name": f"s5-order{offset // 1000}-{eta!r}-s{seed}",
        "config": "experiments/study5_control_order",
        "overrides": [rate(eta), f"train.order_seed={seed + offset}"],
        "seeds": [seed],
    }


def pod3_plan(
    *,
    out: Path,
    runs: Path,
    data: Path,
    checkpoints: Path = CHECKPOINTS,
    archive_id: str,
    commit: str,
    gpu_code: str,
    surrogate: str = SURROGATE,
) -> Plan:
    """Pod 3's plan, extension 1 (module docstring, "Pod 3")."""
    spec = surrogate_spec(surrogate)
    b = Builder("pod3", archive_id, out, runs, data, checkpoints)
    b.admit = True  # each block and seed group admitted only when it and the close fit the cap
    fetched = b.out / "fetched" / "study5"

    def fetch_streams(name: str, seed: int) -> None:
        b.step(name, b.own("fetch-streams", "--inventory", POD2_STREAMS.format(seed=seed),
                           "--data", b.data, "--work", b.out, "--seeds", seed),
               on_fail="stop", minutes="fetch-seed")  # fmt: skip

    def fetch_run(name: str, run_id: str) -> None:
        b.step(name, b.own("fetch-run", "--run-id", run_id, "--dest", fetched,
                           "--prefix", POD2_RUNS),
               on_fail="stop", outputs=(str(fetched / run_id / "meta.json"),),
               minutes="fetch-run")  # fmt: skip

    b.begin()  # the engineering check: pod 2's order control of 50000 rerun; any failure stops
    first = CONTROL_SEEDS[0]
    fetch_streams("engineering/fetch-streams", first)
    fetch_run("engineering/fetch-run", POD2_ORDER_RUN)
    cell = {  # pod 2's cell of that run: the same config hash
        "name": f"s5-order-{ETA_L!r}-s{first}",
        "config": "experiments/study5_control_order",
        "overrides": [rate(ETA_L)],
        "seeds": [first],
    }
    eng_file = b.launcher("engineering", "engineering", [cell])
    eng = Path(eng_file).stem
    b.step("engineering/launch",
           b.own("eng-launch", "--runs", b.runs, "--campaign-file", eng_file, "--gpu-code",
                 gpu_code),
           kind="launch", on_fail="stop", minutes="fresh-upper")  # fmt: skip
    check = b.record("engineering-check.json")
    b.step("engineering/check",
           b.own("with-runs", "--runs", b.runs, "--campaign", eng, "--",
                 "scripts/study5_engineering_check.py", "--archived", fetched / POD2_ORDER_RUN,
                 "--rerun", f"{{run:{label(cell)}}}", "--out", check),
           on_fail="stop", outputs=(str(check),), minutes="engineering-check",
           gate="engineering")  # fmt: skip
    b.runs_archive("engineering/archive", mutable=(b.rel(check),), campaign=eng)
    b.end("engineering")

    b.begin()  # the other seeds' streams and the ladder's reference F runs
    for seed in (*CONTROL_SEEDS[1:], *ORDER_SEEDS):
        fetch_streams(f"fetch/s{seed}/streams", seed)
    for seed in CONTROL_SEEDS:
        fetch_run(f"fetch/s{seed}/f-run", POD2_F_RUNS[seed])
    b.end("fetch")

    b.begin()  # the ladder's two unbracketed rungs, at most LADDER_EXTENSIONS extensions
    for seed in CONTROL_SEEDS:
        group = f"s{seed}"
        paths, run_dirs = [], []
        for rung in EXTENSION_RUNGS:
            record = b.record("ladder", f"s{seed}-{rung}.json")
            run_id = f"s5-{b.id}-ladder-{rung}-s{seed}"
            # with-runs only for --aside/--unless: the command names no {run:} of a campaign
            b.step(f"ladder/{group}/{rung}",
                   b.own("with-runs", "--runs", b.runs, "--campaign", eng, "--aside",
                         f"study5_ladder/{run_id}", "--unless", record, "--",
                         "scripts/study5_ladder.py", *fresh_args(seed), "--rung", rung,
                         "--f-run", fetched / POD2_F_RUNS[seed], "--out", record,
                         "--run-id", run_id, "--extensions", LADDER_EXTENSIONS),
                   outputs=(str(record),), group=group, minutes="rung-extended")  # fmt: skip
            paths.append(b.rel(record))
            run_dirs.append((f"study5_ladder/{run_id}", b.rel(record)))
        b.runs_archive(f"ladder/{group}/archive", paths=paths, run_dirs=run_dirs, group=group)
    b.end("ladder")

    b.begin()  # two new orders of each seed's own responses, both rates
    for seed in ORDER_SEEDS:
        group = f"s{seed}"
        cells = [order_cell(seed, offset, eta) for offset in ORDER_OFFSETS
                 for eta in (ETA_L, ETA_U)]  # fmt: skip
        _, name = b.launch(f"orders-{group}", "orders", cells, minutes="orders", group=group)
        b.runs_archive(f"orders/{group}/archive", campaign=name, group=group)
    b.end("orders")
    b.close()
    return b.plan(commit, gpu_code, None, spec)


# ------------------------------------------------------------------ pod 4: extension 2

EXT2_IDENTITY = "docs/study5-extension-2-code-identity.json"
EXT2_HEADING = "### Extension 2 (2026-11-11): diagnostics of the erosion's mechanism"
POD4_PREFIX = "pod4-"  # pod 4's archive namespace
POD2_FRESH_L = {  # pod 2's fresh η_l runs (the replays' archived twins), by seed
    50000: "20261109T192912Z-s5-fresh-1.3e-05-s50000-s50000-f3e969f1-a1",
    50001: "20261109T200448Z-s5-fresh-1.3e-05-s50001-s50001-52976e1f-a1",
    50002: "20261109T204049Z-s5-fresh-1.3e-05-s50002-s50002-99efec8b-a1",
    50003: "20261109T211734Z-s5-fresh-1.3e-05-s50003-s50003-21cff731-a1",
    50004: "20261109T215417Z-s5-fresh-1.3e-05-s50004-s50004-c431ca6c-a1",
    50005: "20261109T223154Z-s5-fresh-1.3e-05-s50005-s50005-d5d976a3-a1",
    50006: "20261109T230923Z-s5-fresh-1.3e-05-s50006-s50006-cbb2a234-a1",
    50007: "20261109T234657Z-s5-fresh-1.3e-05-s50007-s50007-bdb09490-a1",
    50008: "20261110T002145Z-s5-fresh-1.3e-05-s50008-s50008-5ceafe6b-a1",
    50009: "20261110T005738Z-s5-fresh-1.3e-05-s50009-s50009-0c547c54-a1",
    50010: "20261110T013333Z-s5-fresh-1.3e-05-s50010-s50010-7191026a-a1",
    50011: "20261110T020709Z-s5-fresh-1.3e-05-s50011-s50011-1e205b37-a1",
    50012: "20261110T024200Z-s5-fresh-1.3e-05-s50012-s50012-1c4109be-a1",
    50013: "20261110T031731Z-s5-fresh-1.3e-05-s50013-s50013-96652855-a1",
    50014: "20261110T035237Z-s5-fresh-1.3e-05-s50014-s50014-6fb06d91-a1",
    50015: "20261110T042846Z-s5-fresh-1.3e-05-s50015-s50015-5ff1ca1b-a1",
    50016: "20261110T050600Z-s5-fresh-1.3e-05-s50016-s50016-f76d4a71-a1",
    50017: "20261110T054243Z-s5-fresh-1.3e-05-s50017-s50017-d466168f-a1",
    50018: "20261110T061650Z-s5-fresh-1.3e-05-s50018-s50018-ff1d7425-a1",
    50019: "20261110T065316Z-s5-fresh-1.3e-05-s50019-s50019-b5fb1a8b-a1",
    50020: "20261110T073048Z-s5-fresh-1.3e-05-s50020-s50020-e68806db-a1",
    50021: "20261110T080723Z-s5-fresh-1.3e-05-s50021-s50021-880d44ea-a1",
    50022: "20261110T084313Z-s5-fresh-1.3e-05-s50022-s50022-0b14001f-a1",
    50023: "20261110T092021Z-s5-fresh-1.3e-05-s50023-s50023-7ea65215-a1",
    50024: "20261110T095648Z-s5-fresh-1.3e-05-s50024-s50024-8c5627af-a1",
    50025: "20261110T103427Z-s5-fresh-1.3e-05-s50025-s50025-42bf277b-a1",
    50026: "20261110T111019Z-s5-fresh-1.3e-05-s50026-s50026-5d6e8213-a1",
    50027: "20261110T114649Z-s5-fresh-1.3e-05-s50027-s50027-68d7fd80-a1",
    50028: "20261110T122124Z-s5-fresh-1.3e-05-s50028-s50028-24cd96c3-a1",
    50029: "20261110T125804Z-s5-fresh-1.3e-05-s50029-s50029-2b1b7423-a1",
    50030: "20261110T133402Z-s5-fresh-1.3e-05-s50030-s50030-51436cb3-a1",
    50031: "20261110T141200Z-s5-fresh-1.3e-05-s50031-s50031-23f99580-a1",
    50032: "20261110T144641Z-s5-fresh-1.3e-05-s50032-s50032-bfc26888-a1",
    50033: "20261110T152311Z-s5-fresh-1.3e-05-s50033-s50033-65961fb9-a1",
    50034: "20261110T155803Z-s5-fresh-1.3e-05-s50034-s50034-b8b71433-a1",
    50035: "20261110T163412Z-s5-fresh-1.3e-05-s50035-s50035-4b54149c-a1",
    50036: "20261110T171107Z-s5-fresh-1.3e-05-s50036-s50036-4a2e4971-a1",
    50037: "20261110T174522Z-s5-fresh-1.3e-05-s50037-s50037-55510982-a1",
    50038: "20261110T181849Z-s5-fresh-1.3e-05-s50038-s50038-83886ff4-a1",
    50039: "20261110T185234Z-s5-fresh-1.3e-05-s50039-s50039-74838283-a1",
}
# pod 2's fresh η_l runs: (config hash, the endpoint's trainable tensors hash as its surrogate
# endpoint record holds it), the replays' expected endpoints (study5_engineering_check.py
# --endpoint-tensors)
POD2_ENDPOINTS = {
    50000: ("f3e969f1c2f0", "48ebf1e849ece2acaf1f9fc8474898c71b36691771b99e9421ddd896158b4924"),
    50001: ("52976e1f6c44", "60c4a7683726e23ccd21579294e835e53d84fb15d7beec43777095c0c4043057"),
    50002: ("99efec8b692b", "b14dd160e9d23ddef92738fc17698be2e5c568c58c0b668ff8d67615561167d1"),
    50003: ("21cff731cd08", "ea97c0d628a5581c62556c118b97fd26994b0486cfa7c9b4520027aaf3232675"),
    50004: ("c431ca6cd040", "f1bf7e3a3683b21d3c558994f6e266482b24f1019fbc53a99ca475619e532ff3"),
    50005: ("d5d976a37510", "cf4fd5b134577f52f5b9acd94f5684965ae85cab624b5f1c4341d179b2488fc9"),
    50006: ("cbb2a2346600", "df6e412d1fa296378ac5feba4270ad265a6df959b7d3ae80acf8c4e03aeb5864"),
    50007: ("bdb0949001cd", "601018809750fbb528e5a1190a19df4c8734bc52d21460fec084620c7984b982"),
    50008: ("5ceafe6ba7c3", "9627d1cf0bcbee0c43b32ec96865744042727c82f02312c908d8d8ff62cc73fd"),
    50009: ("0c547c549141", "6ed04c1d5b7759a08bb64025d82ced358b920570efa95640819949821c5ada88"),
    50010: ("7191026ac2c4", "7f8fc93ed30d75965375892e47f472d728b1899f75ad02ebc39eada633e3b026"),
    50011: ("1e205b37d419", "7ac680c5e4dddb15b17c59a792eaf421277d8a5d343d163d1c1aa87aab731009"),
    50012: ("1c4109be778b", "68e3e178c6d783b66de65128e23196a3c6e1e03da94edf6dc7bb8ec2f021731b"),
    50013: ("966528554bd1", "785d5b8b80509fcdca9deb9a54a65cb2c2a8ce854a3f93564d0775aab36e34aa"),
    50014: ("6fb06d91c01e", "85ee698c17dc328669993d6ac8b36852d3630921765f6f7a138510f3eb22345d"),
    50015: ("5ff1ca1bc35a", "8078318814f0e49c526201c64bcb0e28f9b6901041b85f1d8dc12831ccbc9371"),
    50016: ("f76d4a7191eb", "a2fcdd930f4f5ef55e45cc68de936186915f5e94a85cdff0efa07dd439e56d81"),
    50017: ("d466168f8a6f", "c91edd6eb036eaeb21752dae94da20c35133892ea89bb301fb764c7cc84d8a74"),
    50018: ("ff1d7425e19a", "4954396239d1a63f4753dd8d04a2bff6952d7f6a0d4f2944a31cfc24806e7ea1"),
    50019: ("b5fb1a8b6d92", "a8ddad3273940a11811076a665b8b75b569b36c5e5a570eacd09900cd5d5ed9b"),
    50020: ("e68806dbc0a2", "021ed875667af6ea3ed74a4f8c6090e9ecc0fa001f718c7c9df0d777354c6f9b"),
    50021: ("880d44ea24ee", "59025e205de2a99dae61a1ff6696b6511f5e7d5cc58b9932f934e2e8f1aa415c"),
    50022: ("0b14001f75ac", "425cfb5b8db681152a15c14f27a3427c220c0ffd53f34b83e92c084ccc1c86e1"),
    50023: ("7ea652155308", "d20d9b54faa38ce6e9230cc48a7db62107e72dd0ce14665ea3eb6f1cd10fc5c8"),
    50024: ("8c5627af3237", "f28f1bcbd403f85bbdd8c684109dd2c4758dab3dde73b4cf276bd3d7a956651c"),
    50025: ("42bf277ba718", "868e5a0360e8e9058816d9ebe11809a0402f9ae8aad5136eb88a1630ba750e16"),
    50026: ("5d6e821339ae", "557a2720f8f0f7f0945fbd35f6c5fab3d436b04b0ba4cf12cdefc76b31e5bcad"),
    50027: ("68d7fd80adda", "e3b2bdb2e41680caa061884fadc6978c01eaaf1f5632bd481be8d164dc71e332"),
    50028: ("24cd96c31794", "deda0ab4af93c2f4606ab701777c9d5e7d95a0b2dc606e30043043ff226ebcd6"),
    50029: ("2b1b74235620", "2ac97b100e5cf196631eae79b09007bab0c8cef3cded305f81a383daab24eb56"),
    50030: ("51436cb3133e", "5b9fb08c3105d31e94c07d60d7256080cecdeca71818f17fe37b7765d2dacef3"),
    50031: ("23f99580fded", "9b898b783d6f5947ab07cff10abfff2ce75358e136d6217c711ebe4ac34933f3"),
    50032: ("bfc26888eb43", "2c180349027e639b1290570b07b833b189118c874eae65f8018fb938c3cb86e1"),
    50033: ("65961fb9d610", "5442736344ae9c216ea70d0ffe86537b01602092cfe0710fcb179ad84cf1f746"),
    50034: ("b8b714337449", "c38fcdda54c742596f1087c5f55a9247d61bcb61fdadf007ac5916393c022dba"),
    50035: ("4b54149c07c3", "7f6dcab0f0b26f7329ab798412d54124689ed7aaa2d31532c370f4a124dddaea"),
    50036: ("4a2e4971fde0", "726c7682c282a64338f2e642c50365081a024b0fe4281d4ee0974e8134a1f83f"),
    50037: ("555109822a26", "710f311d9e608e6e40786a29643aba3190f7ea3ff4bfbe3b538dff093f3f34b9"),
    50038: ("83886ff40487", "dff35a346c041c27c390e75479cbcf4c286206ae1d42178051bc296ab7ac5c39"),
    50039: ("748382835e47", "08fec5139f298dd59672a6424dbed71d7b912771a82792c1881e5896a426e9fd"),
}
# pod 2's fresh η_l runs: the endpoint checkpoint file's sha256 as its archive's inventory
# shows it verified (study5/pod2-20261109/inventories/checkpoint-<run id>.tsv), by seed
POD2_CHECKPOINT_SHA256 = {
    50000: "cc908d6c7e6da8e0abb7230bea211d3772d3ff4ee9545e9964c36af3ab0aa465",
    50001: "bea679963e2594bd55a4ae842dcf6114592464be9f49039195298bc8a546f976",
    50002: "3e9e131a83d75a17ace3a1f7321708ef5d90bfa430756a8926ba3aa83b68cf7b",
    50003: "0fc612f954f7352b5a1c6dfdc033b1be86709fa09155f62d9edf86e86d545336",
    50004: "eec2d659ddd5266b7e44e172b50f212262f68a0ded8352e1d53b169fd647b41d",
    50005: "6bbd3b876b97acff718194eb2f4645cd7f1458e393e4fa7e52c5129e22ed7e48",
    50006: "3b89a4cf434301e61ba33c9f274375bb1b30217bc737a5a3d59ef6ff86cc0074",
    50007: "0e25a210a19d79679b36ff509019713fae9e2712d041773293df7d98f9947c5f",
    50008: "80bcd21e3780ce4fe264efbff89feb30067ddf1308274cf0a956f5ca36847cc4",
    50009: "04218bb742f422c16cda5418519e2104017bfd5c5070c67c67926bfb2fe6d889",
    50010: "85b565bef6b60fae3550f15b61a12d39da3e5fdfa1083bc3297f833c1355c424",
    50011: "6306775f12e57192420d374c81e2dea5a4aa15f14fc19c7f97064287e976cac8",
    50012: "b9042a38e2a848202c520a9f3467640b83ef33e9b786c1d7b82bbd0afa38f8de",
    50013: "f03cedf366a7ba2cb96d7d741aa476abf39deee088e01224651e817a73541933",
    50014: "03d0711335244a90ef6e081956fc17a0f49ba02b6479f1c12325217138bab5b1",
    50015: "f72f0785ab9a76bd65960d0e3937754dde11439ef70346972ade86bd5e04c5d1",
    50016: "55e1202dd77dee21e1912206056714a046107dab772a6766864eefd44d53584d",
    50017: "938b8141cae6ef9a715d9182a3f5676355ae6e7c1fc7c5dbda3187d38485186d",
    50018: "39572f93596309099baefe8c310802d36b546efbdb59b94cee28ed487f5a1ccd",
    50019: "cc6bc6bffa6caa8b49d5f84652b174b8cd0775f1e3f686fb19e9ffda9b87388c",
    50020: "90a57af6eb013e1589a145ae1b21a504319477cc2c0bbe42457f263b8013d933",
    50021: "796e06308a2baffbbc7fc33f6956fd6165ce8de675193c28dad88ff2ec7f1b49",
    50022: "d84b42096763dd36dae935706d730612f532ad105bd5069b3783cb58c923928a",
    50023: "a5804b72a22b8817b58b1423b5b871f028fc75349ae2d31d791fdc19d3c682ab",
    50024: "2b29b5a5185e00370918a359cf047685d706fc8df9bd41dba21f5a9589104d45",
    50025: "720e2e7b238167c16c635052ffbf78dd4aa6db01fcdea7f78a1f93384fc2cdb5",
    50026: "e6133c3ea19c8f4a420c3f01471ec77b6cc444b5785766f20daff378fff69806",
    50027: "a8b3b8118c5bd5dc531b8cc8dab276d9ff4af9cc1160be49babf327960b0abb2",
    50028: "da9b2da8a0b551419b0d7a31b4d8d34f7fb0b139d88760b23a44814dc99bb3d3",
    50029: "8c30cf92c7d127efcc8ead9a56fee3ad33f6cb8717da6ef071f0e4a5d86152a4",
    50030: "4d0da3038770474a8409d11eec88a505c76a3357c0bdb17a739cd600271b44c3",
    50031: "f64c3dc90a88ee5d4fc6aa298ecd92b5735467246846b266e39600cba625f203",
    50032: "c210339619989144951325f18acb3567cea1b1e820387fd0858dea0dffe80c4f",
    50033: "bfeb29ab244caa418afa3e70136e3a35c04df172cb8934f7aa170556a6fcd668",
    50034: "31154e93e2f4ebd6cf1827db16fe655db3082ca343ef10af038b11aea6de784e",
    50035: "414daa4e2401c5213b768d3e326f8bde9717f3cbfb0e70c01844518a18666344",
    50036: "2db209d629f8667c87cdf526935cec12941cb9db3ff40d824bed8bb689ac061e",
    50037: "db3ecbec0633d0da6591c0442e96c91a587fab620f9f147ba394397e93b3c9a7",
    50038: "8dae551e130deb4112c99d37a4435d1298380fc55db1199df0d1b2d6c3d64042",
    50039: "40643cef96e6be0666550d118190d0237ab112e0e478663d883f409ee1575090",
}
# the diagnostic rerun of pod 4's first attempt (the order control's second rerun), copied
# into the runs root, and its run directory there
POD4_DIAG = "study5_diag"
POD4_DIAG_RERUN = "study5/20261111T164620Z-s5-order-1.3e-05-s50000-s50000-3173f4e9-a1"
# extension 2's numerical equivalence (docs/study5-preregistration.md, "Extension 2
# amendment"), as study5_engineering_check.py --tolerance and --endpoint-tolerance record it;
# the endpoint bound 1e-3 by "Extension 2 amendment 2" (it was 1e-4). A check that passed under
# a stricter endpoint bound stands (``gate_problem``)
EXT2_TOLERANCE = {"relative": 1e-4, "absolute": 1e-9, "answers_allowed": 2,
                  "endpoint_distance": 1e-3}  # fmt: skip
# a replay group's check, accepted for endpoint analysis ("Extension 2 amendment 3":
# study5_engineering_check.py --endpoint-analysis, the gate ``replay``) under the endpoint bound;
# a record of its own: the earlier records (check-{group}.json, check-{group}-endpoint-0.001.json)
# stay as they were written
EXT2_REPLAY_CHECK = "check-{group}-endpoint-analysis-{bound:g}.json"
EXT2_SEEDS = tuple(range(FRESH_FIRST, FRESH_FIRST + 40))  # the 40 fresh seeds
EXT2_GROUP = 4  # seeds per replay or linear group: each group admitted on its own
M_SEEDS = CONTROL_SEEDS  # M3, M4's removals, the δ_lin points evaluated in full, the shuffles
SWAP_SEEDS = CONTROL_SEEDS[:4]  # M6's swaps and twins
SWAP_STEPS = (2, 12, 23)
TWIN_EDIT = "{kind: swap_first, step: 2}"
TWIN_STEPS = (2, 3, 5, 8, 12, 18)
FORCE_SEED, FORCE_STEPS = 50000, (5, 10)
FORCE_REPEAT_EPSILON = 1e-2  # the amendment's repeated force measurement
FLOOR_EXPERIMENT = "study5_floor"  # the twins' floor run: the replay again, apart from it
FORCE_SEED_STEPS = (1, 2, 3, 5, 8, 10, 12, 18)  # the force pilot's steps and the twins'
PANEL_SEEDS, PANEL_RESPONSES = (59990, 59991), 1000  # streams no studied run trains on
SMALL_POINTS = (0.1, -0.1, 0.05, -0.05, 0.025, -0.025)
TOKEN_SCALES = (0.1, -0.1, 1.0, -1.0)
FORCE_EPSILONS = (5e-2, 1e-2, 5e-3)
SHUFFLE_RANGES, SHUFFLE_DRAWS = ((1, 5), (21, 25)), (0, 1)
MIXTURES = 3
EDIT_CONFIG = "experiments/study5_control_edit"
CONTROL_ORDER = "experiments/study5_control_order"
VECTOR_EXPERIMENT = "study5_vector"  # vector_points.EXPERIMENT
POD4_MINUTES = {
    "replay": F_RUN + CKPT_SAVE,
    "replay-steps": F_RUN + 9 * CKPT_SAVE,  # the endpoint and eight step checkpoints
    "grad-r": 3.0,  # assumed: θ₀, the panel's 1100 rows forward and backward
    # assumed from a 52M-parameter run scaled up: 40 checkpoints memory-mapped, two passes, 22
    # files of 5.24 GB written
    "vectors": 90.0,
    "linear": 6.5,  # assumed: v₀, 85 batch gradients, three r readouts, K on both panels twice
    "linear-q": 4.5,  # the same without the K readouts (--no-k): predictions of q only
    "pairs": 3.0,  # assumed: checkpoints read and hashed, r at each
    "tensor-hash": 3.0,  # assumed: four 5.24 GB endpoints hashed as files and as tensors
    "force-repeat": 12.0,  # assumed: one step and one ε of the force run, its vector saved
    "fetch-endpoint": 4.0,  # assumed: a 5.24 GB checkpoint fetched and hashed at about 25 MB/s
    "tokens": 6.0,
    "force": 45.0,  # assumed: two steps × three ε × four panel gradients of 25 batches, halves
    "panels": 2 * PANEL_RESPONSES * STREAM_PER_RESPONSE,
    "settle": 0.5,
    "edit-run": F_RUN,
}


def vector_minutes(points: int, full: int = 0, match: bool = False) -> float:
    """A vector run: the models and panels loaded, a readout (K both panels, r) per point, a full
    evaluation per full point, and the matched-K search (the extension's 8 extensions)."""
    minutes = 1.0 + 2.0 * points + EVALUATION * full
    if match:
        minutes += (3 + LADDER_EXTENSIONS + 2) * K_POINT + 2.0 + EVALUATION
    return minutes


def replay_cell(b: Builder, seed: int, steps: tuple[int, ...] = ()) -> dict[str, Any]:
    """The seed's fresh F run at η_l as pod 2 ran it (the same config hash), with its endpoint
    checkpoint and, for the twins' and the force pilot's seeds, step checkpoints."""
    overrides = [*FRESH[1:], rate(ETA_L), f"tracking.endpoint_checkpoint={b.checkpoints}"]
    if steps:
        overrides.append(f"tracking.step_checkpoints=[{','.join(map(str, steps))}]")
    return {"name": f"s5-fresh-{ETA_L!r}-s{seed}", "config": FRESH[0], "overrides": overrides,
            "seeds": [seed]}  # fmt: skip


def floor_cell(b: Builder, seed: int) -> dict[str, Any]:
    """The twins' numerical floor (the amendment): the seed's replay again, its config and so its
    config hash unchanged, with the twins' step checkpoints, in an experiment folder of its own
    (``FLOOR_EXPERIMENT``; ``experiment`` is outside the config hash), so that the launcher, which
    reuses a completed run of the same experiment and config hash, runs it anew."""
    cell = replay_cell(b, seed, TWIN_STEPS)
    return {**cell, "name": f"s5-floor-{ETA_L!r}-s{seed}",
            "overrides": [*cell["overrides"], f"experiment={FLOOR_EXPERIMENT}"]}  # fmt: skip


def edit_cell(seed: int, tag: str, edit: str, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """An edited order of the seed's own responses at η_l (``EDIT_CONFIG``)."""
    return {"name": f"s5-edit-{tag}-{ETA_L!r}-s{seed}", "config": EDIT_CONFIG,
            "overrides": [rate(ETA_L), f"+train.order_edit={edit}", *extra],
            "seeds": [seed]}  # fmt: skip


def edit_args(seed: int, edit: str) -> tuple[str, ...]:
    return (EDIT_CONFIG, rate(ETA_L), f"seed={seed}", f"+train.order_edit={edit}")


def shuffle_edit(first: int, last: int, draw: int) -> str:
    return f"{{kind: shuffle, first: {first}, last: {last}, draw: {draw}}}"


def pod4_plan(
    *,
    out: Path,
    runs: Path,
    data: Path,
    checkpoints: Path,
    archive_id: str,
    commit: str,
    gpu_code: str,
    surrogate: str = SURROGATE,
) -> Plan:
    """Pod 4's plan, extension 2 (docs/operations.md, "Study 5, extension 2: pod 4").

    Evidence and disposable files: the run directories and every record (``*.json`` under
    ``study5_records/<id>/``) are evidence, archived as their group ends; the checkpoints and the
    vector files (``*.pt`` under ``--checkpoints``) are disposable, reproducible from the replays,
    and removed only by ``purge`` and ``drop-checkpoints`` once the records that read them are in
    a verified archive (``archived_problems``). The replays' endpoints are never removed here.
    A writer of a vector folder moves an earlier incomplete folder aside first (``--aside-dir``);
    a step whose input a diverged δ_lin did not write records that instead (``--only-if``)."""
    spec = surrogate_spec(surrogate)
    b = Builder("pod4", archive_id, out, runs, data, checkpoints)
    b.admit = True  # each block and group admitted only when it and the close fit the cap
    fetched = b.out / "fetched" / "study5"
    vec = b.checkpoints / "vec"
    teacher = ("scripts/generate_teacher.py", *FRESH, rate(ETA_L))
    minutes = {**MINUTES, **POD4_MINUTES}

    def step(name: str, argv, *, minutes_key: str | float = "small", **kw) -> Step:
        value = minutes[minutes_key] if isinstance(minutes_key, str) else float(minutes_key)
        return b.step(name, argv, minutes=value, **kw)

    def pod2_run(seed: int) -> Path:
        return fetched / POD2_FRESH_L[seed]

    def with_runs(*argv, campaign: str | None = None) -> tuple[str, ...]:
        return b.own("with-runs", "--runs", b.runs, "--campaign", campaign or eng, *argv)

    def settle(name: str, folder: Path, records: Path, group: str = "") -> None:
        step(name, b.own("settle", "--dir", folder, "--records", records), group=group,
             minutes_key="settle")  # fmt: skip

    def purge(name: str, folder: Path, inventories, consumers, group: str = "") -> None:
        argv = ["purge", "--dir", folder, "--runs", b.runs]
        argv += [x for inv in inventories for x in ("--inventory", inv)]
        argv += [x for c in consumers for x in ("--consumer", c)]
        step(name, b.own(*argv), group=group, minutes_key="settle")

    def drop(name: str, campaign: str, labels, inventories, consumers, *, endpoint: bool,
             group: str = "") -> None:  # fmt: skip
        argv = ["drop-checkpoints", "--runs", b.runs, "--campaign", campaign]
        argv += [x for lab in labels for x in ("--run", lab)]
        argv += [x for inv in inventories for x in ("--inventory", inv)]
        argv += [x for c in consumers for x in ("--consumer", c)]
        step(name, b.own(*argv, *(["--endpoint"] if endpoint else [])), group=group,
             minutes_key="settle")  # fmt: skip

    def inventory(archive_step: Step) -> str:
        return archive_step.outputs[0]

    def vector_run(name: str, label_: str, seed: int, source: tuple[str, ...], *,
                   points: tuple[float, ...] = (1.0,), full: tuple[float, ...] = (),
                   match: bool = False, sign: int = 1, campaign: str | None = None,
                   only_if: Path | None = None, group: str = "") -> tuple[str, str]:  # fmt: skip
        """A vector run (scripts/study5_vector_points.py): its record and run directory,
        relative to the runs root. With ``only_if``, an absent input is recorded instead."""
        record = b.record("points", f"{label_}.json")
        run_id = f"s5-{b.id}-{label_.replace('_', '-')}"
        guard = ("--only-if", only_if, "--else-record", record) if only_if is not None else ()
        argv = [*guard, "--aside", f"{VECTOR_EXPERIMENT}/{run_id}", "--unless", record, "--",
                "scripts/study5_vector_points.py", *fresh_args(seed), *source, "--label", label_,
                "--points", *points, "--out", record, "--run-id", run_id, "--surrogate",
                spec[0]]  # fmt: skip
        if full:
            argv += ["--full", *full]
        if match:
            argv += ["--match"]
        if sign != 1:
            argv += ["--sign", sign]
        step(name, with_runs(*argv, campaign=campaign), outputs=(str(record),), group=group,
             minutes_key=vector_minutes(len(points), len(full), match))  # fmt: skip
        return b.rel(record), f"{VECTOR_EXPERIMENT}/{run_id}"

    def linear(name: str, config: tuple[str, ...], folder: Path, reference: tuple[str, ...],
               group: str = "", k: bool = True) -> str:  # fmt: skip
        """δ̂, δ_lin, q and (with ``k``) K of one order into ``folder`` (an incomplete one moved
        aside); without ``k`` the predictions of q only (``--no-k``)."""
        record = folder / "record.json"
        step(name, with_runs("--unless", record, "--aside-dir", folder, "--",
                             "scripts/study5_linear_replay.py", *config, "--out-dir", folder,
                             *reference, "--surrogate", spec[0], *(() if k else ("--no-k",))),
             outputs=(str(record),), group=group,
             minutes_key="linear" if k else "linear-q")  # fmt: skip
        return str(record)

    def pairs(name: str, seed: int, items: list[tuple[str, str]], record: Path, *,
              r: bool = False, only_if: Path | None = None, campaign: str | None = None,
              group: str = "") -> str:  # fmt: skip
        guard = ("--only-if", only_if, "--else-record", record) if only_if is not None else ()
        argv = [*guard, "--unless", record, "--", "scripts/study5_pairs.py", *fresh_args(seed)]
        for a, z in items:
            argv += ["--pair", a, z]
        argv += [*(["--r"] if r else []), "--out", record, "--surrogate", spec[0]]
        step(name, with_runs(*argv, campaign=campaign), outputs=(str(record),), group=group,
             minutes_key="pairs")  # fmt: skip
        return b.rel(record)

    # ---- engineering: pod 2's order control of 50000 rerun; any failure stops (as pod 3)
    b.begin()
    first = CONTROL_SEEDS[0]
    step("engineering/fetch-streams", b.own("fetch-streams", "--inventory",
         POD2_STREAMS.format(seed=first), "--data", b.data, "--work", b.out, "--seeds", first),
         on_fail="stop", minutes_key="fetch-seed")  # fmt: skip
    step("engineering/fetch-run", b.own("fetch-run", "--run-id", POD2_ORDER_RUN, "--dest", fetched,
         "--prefix", POD2_RUNS), on_fail="stop", minutes_key="fetch-run",
         outputs=(str(fetched / POD2_ORDER_RUN / "meta.json"),))  # fmt: skip
    cell = {"name": f"s5-order-{ETA_L!r}-s{first}", "config": CONTROL_ORDER,
            "overrides": [rate(ETA_L)], "seeds": [first]}  # fmt: skip
    eng_file = b.launcher("engineering", "engineering", [cell])
    eng = Path(eng_file).stem
    step("engineering/launch", b.own("eng-launch", "--runs", b.runs, "--campaign-file", eng_file,
         "--gpu-code", gpu_code), kind="launch", on_fail="stop",
         minutes_key="fresh-upper")  # fmt: skip
    # under the amendment, every rerun of the order control on this volume in one numerical
    # check: the campaign's own and the diagnostic rerun beside it (POD4_DIAG); the first
    # attempt's bitwise record stays as written, archived with them
    diag = b.runs / POD4_DIAG
    reruns = [f"{{run:{label(cell)}}}", *([diag / POD4_DIAG_RERUN] if diag.is_dir() else [])]
    earlier = [b.rel(p) for p in (b.record("engineering-check.json"), diag) if p.exists()]
    check = b.record("engineering-check-numerical.json")
    step("engineering/check", with_runs("--", "scripts/study5_engineering_check.py", "--archived",
         fetched / POD2_ORDER_RUN, "--rerun", *reruns, "--out", check, "--tolerance",
         EXT2_TOLERANCE["relative"], "--endpoint-tolerance", EXT2_TOLERANCE["endpoint_distance"]),
         on_fail="stop", outputs=(str(check),), minutes_key="engineering-check",
         gate="engineering")  # fmt: skip
    b.runs_archive("engineering/archive", paths=earlier, mutable=(b.rel(check),), campaign=eng)
    b.end("engineering")

    # ---- the other seeds' streams, pod 2's fresh η_l runs, two fresh panels
    b.begin()
    for seed in EXT2_SEEDS[1:]:
        step(f"fetch/s{seed}/streams", b.own("fetch-streams", "--inventory",
             POD2_STREAMS.format(seed=seed), "--data", b.data, "--work", b.out, "--seeds", seed),
             on_fail="stop", minutes_key="fetch-seed")  # fmt: skip
    for seed in EXT2_SEEDS:
        step(f"fetch/s{seed}/f-run", b.own("fetch-run", "--run-id", POD2_FRESH_L[seed], "--dest",
             fetched, "--prefix", POD2_RUNS), on_fail="stop",
             outputs=(str(pod2_run(seed) / "meta.json"),), minutes_key="fetch-run")  # fmt: skip
    step("fetch/panels", (*teacher, "--stream", "training", "--seeds", *PANEL_SEEDS,
         "--responses", PANEL_RESPONSES), on_fail="stop", exists_ok=True,
         minutes_key="panels")  # fmt: skip
    b.end("fetch")

    # ---- M1's replays, each group checked against pod 2's runs by numerical equivalence (the
    # amendment): tensors of the endpoints, and where they differ the distance to pod 2's
    # checkpoint fetched back from its archive (removed once read); M6's twins on the first four
    # seeds, their pairs, and in FORCE_SEED's group an unchanged-order repeat of its replay with
    # the twins' checkpoints (the twins' numerical floor); then the twins' and the repeat's
    # checkpoints and the originals' step checkpoints (not their endpoints) dropped once the
    # pairs are archived
    replays: dict[int, tuple[str, str]] = {}  # seed -> (campaign, label)
    floor_read: tuple[list[str], list[str]] = ([], [])  # the floor's inventory and record
    pod2_checkpoints = b.checkpoints / "pod2"
    b.begin()
    for i in range(0, len(EXT2_SEEDS), EXT2_GROUP):
        seeds = EXT2_SEEDS[i : i + EXT2_GROUP]
        group = f"g{seeds[0]}"
        cells = [replay_cell(b, s, FORCE_SEED_STEPS if s == FORCE_SEED else
                             TWIN_STEPS if s in SWAP_SEEDS else ()) for s in seeds]  # fmt: skip
        twins = {s: edit_cell(s, "twin", TWIN_EDIT, (
            f"tracking.endpoint_checkpoint={b.checkpoints}",
            f"tracking.step_checkpoints=[{','.join(map(str, TWIN_STEPS))}]")) for s in seeds
            if s in SWAP_SEEDS}  # fmt: skip
        minute = sum(minutes["replay-steps" if s in SWAP_SEEDS else "replay"] for s in seeds)
        minute += len(twins) * minutes["replay-steps"]
        _, name = b.launch(f"replays-{group}", "replays", [*cells, *twins.values()],
                           minutes=minute, group=group)  # fmt: skip
        for s, c in zip(seeds, cells, strict=True):
            replays[s] = (name, label(c))
        floor = None  # (campaign, label) of the unchanged-order repeat
        if FORCE_SEED in seeds:
            repeat = floor_cell(b, FORCE_SEED)
            _, again = b.launch(f"repeat-{group}", "replays", [repeat],
                                minutes=minutes["replay-steps"], group=group)  # fmt: skip
            floor = (again, label(repeat))
        hashed = b.record("replays", f"tensors-{group}.json")
        step(f"replays/{group}/tensors", with_runs("--unless", hashed, "--",
             "scripts/study5_tensor_hash.py", *[f"{{run:{label(c)}}}" for c in cells], "--out",
             hashed, campaign=name), on_fail="stop", outputs=(str(hashed),), group=group,
             minutes_key="tensor-hash")  # fmt: skip
        distances = {}
        for s, c in zip(seeds, cells, strict=True):
            sha = POD2_CHECKPOINT_SHA256[s]
            fetched_ckpt = pod2_checkpoints / f"{POD2_FRESH_L[s]}.pt"
            step(f"replays/{group}/fetch-s{s}", b.own("fetch-endpoint", "--runs", b.runs,
                 "--campaign", name, "--label", label(c), "--hashed", hashed, "--expected",
                 POD2_ENDPOINTS[s][1], "--run-id", POD2_FRESH_L[s], "--sha256", sha, "--remote",
                 f"{POD2_ARCHIVE}/checkpoints", "--inventories-remote",
                 f"{POD2_ARCHIVE}/inventories", "--dest", pod2_checkpoints), group=group,
                 minutes_key="fetch-endpoint")  # fmt: skip
            distances[s] = b.record("replays", f"distance-s{s}.json")
            pairs(f"replays/{group}/distance-s{s}", s,
                  [(f"ckpt:{fetched_ckpt}:{sha}", f"run:{{run:{label(c)}}}:{UPDATES}")],
                  distances[s], only_if=fetched_ckpt, campaign=name, group=group)  # fmt: skip
            step(f"replays/{group}/remove-s{s}", b.own("remove-fetched", "--path", fetched_ckpt,
                 "--sha256", sha, "--record", distances[s]), group=group,
                 minutes_key="settle")  # fmt: skip
        check = b.record("replays", EXT2_REPLAY_CHECK.format(
            group=group, bound=EXT2_TOLERANCE["endpoint_distance"]))  # fmt: skip
        argv = ["--", "scripts/study5_engineering_check.py", "--archived",
                *[pod2_run(s) for s in seeds], "--rerun", *[f"{{run:{label(c)}}}" for c in cells],
                "--out", check, "--rerun-endpoint-tensors", hashed, "--tolerance",
                EXT2_TOLERANCE["relative"], "--endpoint-tolerance",
                EXT2_TOLERANCE["endpoint_distance"], "--endpoint-analysis"]  # fmt: skip
        argv += [x for s in seeds for x in ("--endpoint-tensors",
                                            "=".join(POD2_ENDPOINTS[s]))]  # fmt: skip
        argv += [x for s in seeds for x in ("--endpoint-distance",
                                            f"{POD2_ENDPOINTS[s][0]}={distances[s]}")]  # fmt: skip
        if twins:
            argv += ["--ignore", "step_checkpoints"]
        step(f"replays/{group}/check", with_runs(*argv, campaign=name), on_fail="stop",
             outputs=(str(check),), group=group, minutes_key="engineering-check",
             gate="replay")  # fmt: skip
        paths = [b.rel(hashed), b.rel(check), *[b.rel(d) for d in distances.values()]]
        for s, twin in twins.items():
            items = [(f"run:{{run:{replays[s][1]}}}:{t}", f"run:{{run:{label(twin)}}}:{t}")
                     for t in (*TWIN_STEPS, UPDATES)]  # fmt: skip
            paths.append(pairs(f"replays/{group}/twins-s{s}", s, items,
                               b.record("twins", f"s{s}.json"), r=True, campaign=name,
                               group=group))  # fmt: skip
        if floor is not None:
            items = [(f"run:{{run:{replays[FORCE_SEED][1]}@{name}}}:{t}",
                      f"run:{{run:{floor[1]}@{floor[0]}}}:{t}")
                     for t in (*TWIN_STEPS, UPDATES)]  # fmt: skip
            paths.append(pairs(f"replays/{group}/floor-s{FORCE_SEED}", FORCE_SEED, items,
                               b.record("twins", f"floor-s{FORCE_SEED}.json"), r=True,
                               campaign=name, group=group))  # fmt: skip
        archived = b.runs_archive(f"replays/{group}/archive", paths=paths,
                                  campaign=(name, floor[0]) if floor else name,
                                  group=group)  # fmt: skip
        for s, twin in twins.items():
            consumers = [b.rel(b.record("twins", f"s{s}.json"))]
            if s == FORCE_SEED and floor is not None:
                consumers.append(b.rel(b.record("twins", f"floor-s{FORCE_SEED}.json")))
            drop(f"replays/{group}/drop-twin-s{s}", name, [label(twin)], [inventory(archived)],
                 consumers, endpoint=True, group=group)  # fmt: skip
            if s != FORCE_SEED:  # the force pilot reads 50000's step checkpoints later
                drop(f"replays/{group}/drop-steps-s{s}", name, [replays[s][1]],
                     [inventory(archived)], consumers, endpoint=False, group=group)  # fmt: skip
        if floor is not None:
            floor_record = b.rel(b.record("twins", f"floor-s{FORCE_SEED}.json"))
            floor_read = ([inventory(archived)], [floor_record])
            drop(f"replays/{group}/drop-repeat", floor[0], [floor[1]], [inventory(archived)],
                 [floor_record], endpoint=True, group=group)  # fmt: skip
    b.end("replays")

    def replay_dir(seed: int) -> str:
        campaign, run_label = replays[seed]
        return f"{{run:{run_label}@{campaign}}}"

    # ---- M1's vectors: ∇r(θ₀), then the Gram, μ̂, the leave-one-out means, mixtures, removals
    b.begin()
    grad = vec / "grad_r"
    step("vectors/grad-r", with_runs("--unless", grad / "record.json", "--aside-dir", grad, "--",
         "scripts/study5_linear_replay.py", *fresh_args(first), "--grad-r-only", "--out-dir",
         grad, "--surrogate", spec[0]), on_fail="stop", outputs=(str(grad / "record.json"),),
         minutes_key="grad-r")  # fmt: skip
    m1 = vec / "m1"
    step("vectors/vectors", with_runs("--unless", m1 / "record.json", "--aside-dir", m1, "--",
         "scripts/study5_vectors.py", *fresh_args(first), "--runs",
         *[replay_dir(s) for s in EXT2_SEEDS], "--checkpoints", b.checkpoints, "--out-dir", m1,
         "--grad-r", grad / "grad_r.pt", "--loo-seeds", *M_SEEDS, "--removal-seeds", *M_SEEDS,
         "--mixtures", MIXTURES), on_fail="stop", outputs=(str(m1 / "record.json"),),
         minutes_key="vectors")  # fmt: skip
    settle("vectors/settle-grad-r", grad, b.record("vectors", "grad_r"))
    settle("vectors/settle", m1, b.record("vectors", "m1"))
    varchived = b.runs_archive("vectors/archive", paths=(b.rel(b.record("vectors")),))
    b.end("vectors")

    # ---- M5: the force pilot on seed 50000 at steps 5 and 10, two panels
    b.begin()
    projections = (m1 / f"mean_minus_{FORCE_SEED}.pt", grad / "grad_r.pt")
    force_paths = []
    for panel in PANEL_SEEDS:
        record = b.record("force", f"s{FORCE_SEED}-panel{panel}.json")
        steps_ = [x for t in FORCE_STEPS for x in ("--step", t)]
        step(f"force/panel{panel}", with_runs("--unless", record, "--",
             "scripts/study5_force.py", *fresh_args(FORCE_SEED), "--panel-config", f"seed={panel}",
             "train.steps=50", "schedule.evaluate_at=[50]", "--run", replay_dir(FORCE_SEED),
             "--checkpoints", b.checkpoints, *steps_, "--out", record, "--epsilon",
             *FORCE_EPSILONS, "--project", *projections, "--split-halves"),
             outputs=(str(record),), minutes_key="force")  # fmt: skip
        force_paths.append(b.rel(record))
    # the amendment's noise check: one force measurement made twice on the same inputs (the
    # first panel, the first step, ε = FORCE_REPEAT_EPSILON), its vector saved each time; the
    # two vectors compared by their difference, the records by their projections
    saved = {}
    for k in ("a", "b"):
        record = b.record("force-repeat", f"{k}.json")
        saved[k] = vec / "force-repeat" / k
        step(f"force/repeat-{k}", with_runs("--unless", record, "--aside-dir", saved[k], "--",
             "scripts/study5_force.py", *fresh_args(FORCE_SEED), "--panel-config",
             f"seed={PANEL_SEEDS[0]}", "train.steps=50", "schedule.evaluate_at=[50]", "--run",
             replay_dir(FORCE_SEED), "--checkpoints", b.checkpoints, "--step", FORCE_STEPS[0],
             "--out", record, "--epsilon", FORCE_REPEAT_EPSILON, "--project", *projections,
             "--split-halves", "--save-force", saved[k]), outputs=(str(record),),
             minutes_key="force-repeat")  # fmt: skip
        force_paths.append(b.rel(record))
    name_ = (
        f"force-step-{FORCE_STEPS[0]}-eps-{FORCE_REPEAT_EPSILON:g}.pt"  # study5_force.force_path
    )
    force_paths.append(pairs("force/repeat-pairs", FORCE_SEED,
                             [(f"vector:{saved['a'] / name_}", f"vector:{saved['b'] / name_}")],
                             b.record("force-repeat", "pairs.json")))  # fmt: skip
    farchived = b.runs_archive("force/archive", paths=force_paths)
    repeat_records = [p for p in force_paths if "/force-repeat/" in f"/{p}"]
    for k, folder_ in saved.items():
        purge(f"force/purge-repeat-{k}", folder_, [inventory(farchived)], repeat_records)
    # FORCE_SEED's step checkpoints go only once the force records and the twins' floor (which
    # read them too) are in verified archives
    drop("force/drop-steps", replays[FORCE_SEED][0], [replays[FORCE_SEED][1]],
         [inventory(farchived), *floor_read[0]], [*force_paths, *floor_read[1]],
         endpoint=False)  # fmt: skip
    vectors_record = f"{b.rel(b.record('vectors', 'm1'))}/record.json"
    purge("force/purge-grad-r", grad, [inventory(varchived), inventory(farchived)],
          [vectors_record, *force_paths])  # fmt: skip
    b.end("force")

    # ---- M2: δ̂ and δ_lin (q, K) of every seed's own order; θ₀ + δ_lin in full on 50000-50005;
    # the cosines with Δθ; then the vector files purged once their consumers are archived
    b.begin()
    for i in range(0, len(EXT2_SEEDS), EXT2_GROUP):
        seeds = EXT2_SEEDS[i : i + EXT2_GROUP]
        group = f"g{seeds[0]}"
        paths, run_dirs, consumers = [], [], {}
        for s in seeds:
            folder = vec / "linear" / f"s{s}"
            linear(f"linear/{group}/s{s}", fresh_args(s), folder, ("--reference-run",
                   pod2_run(s)), group=group)  # fmt: skip
            target = b.record("linear", f"s{s}")
            settle(f"linear/{group}/s{s}-settle", folder, target, group)
            mine = [b.rel(target / "record.json")]
            hat = pairs(f"linear/{group}/s{s}-pairs-hat", s,
                        [(f"vector:{folder / 'delta_hat.pt'}", f"run:{replay_dir(s)}:{UPDATES}")],
                        b.record("linear", f"pairs-hat-s{s}.json"), group=group)  # fmt: skip
            lin = pairs(f"linear/{group}/s{s}-pairs-lin", s,
                        [(f"vector:{folder / 'delta_lin.pt'}", f"run:{replay_dir(s)}:{UPDATES}")],
                        b.record("linear", f"pairs-lin-s{s}.json"),
                        only_if=folder / "delta_lin.pt", group=group)  # fmt: skip
            mine += [hat, lin]
            if s == first:  # the amendment's repeat: the same computation on the same inputs
                again = vec / "linear-repeat" / f"s{s}"
                linear(f"linear/{group}/s{s}-repeat", fresh_args(s), again, ("--reference-run",
                       pod2_run(s)), group=group)  # fmt: skip
                repeat_target = b.record("linear-repeat", f"s{s}")
                settle(f"linear/{group}/s{s}-repeat-settle", again, repeat_target, group)
                hat_items = [
                    (f"vector:{folder / 'delta_hat.pt'}", f"vector:{again / 'delta_hat.pt'}"),
                    (f"vector:{again / 'delta_hat.pt'}", f"run:{replay_dir(s)}:{UPDATES}"),
                ]
                lin_items = [
                    (f"vector:{folder / 'delta_lin.pt'}", f"vector:{again / 'delta_lin.pt'}"),
                    (f"vector:{again / 'delta_lin.pt'}", f"run:{replay_dir(s)}:{UPDATES}"),
                ]
                repeat_pairs = [
                    pairs(f"linear/{group}/s{s}-repeat-pairs-hat", s, hat_items,
                          b.record("linear-repeat", f"pairs-hat-s{s}.json"), group=group),
                    pairs(f"linear/{group}/s{s}-repeat-pairs-lin", s, lin_items,
                          b.record("linear-repeat", f"pairs-lin-s{s}.json"),
                          only_if=again / "delta_lin.pt", group=group),
                ]  # fmt: skip
                mine += repeat_pairs  # they read this seed's own vectors too
                paths.append(b.rel(repeat_target))
                repeated = [b.rel(repeat_target / "record.json"), *repeat_pairs]
            if s in M_SEEDS:  # θ₀ + δ_lin evaluated in full, for the labels
                rec, run_dir = vector_run(f"linear/{group}/s{s}-points", f"dlin_s{s}", s,
                                          ("--vector", folder / "delta_lin.pt"), full=(1.0,),
                                          only_if=folder / "delta_lin.pt", group=group)  # fmt: skip
                mine.append(rec)
                run_dirs.append((run_dir, rec))
            paths += [b.rel(target), *mine[1:]]
            consumers[s] = mine
        archived = b.runs_archive(f"linear/{group}/archive", paths=sorted(set(paths)),
                                  run_dirs=run_dirs, group=group)  # fmt: skip
        for s in seeds:
            purge(f"linear/{group}/s{s}-purge", vec / "linear" / f"s{s}", [inventory(archived)],
                  consumers[s], group)  # fmt: skip
            if s == first:
                purge(f"linear/{group}/s{s}-repeat-purge", vec / "linear-repeat" / f"s{s}",
                      [inventory(archived)], repeated, group)  # fmt: skip
    group = "controls"  # the six registered order controls' orders: the criterion's pairs
    paths = []
    for s in M_SEEDS:
        folder = vec / "linear-control" / f"s{s}"
        linear(f"linear/{group}/s{s}", (CONTROL_ORDER, rate(ETA_L), f"seed={s}"), folder,
               ("--reference-fresh-run", pod2_run(s)), group=group, k=False)  # fmt: skip
        settle(f"linear/{group}/s{s}-settle", folder, b.record("linear-control", f"s{s}"), group)
        paths.append(b.rel(b.record("linear-control", f"s{s}")))
    archived = b.runs_archive(f"linear/{group}/archive", paths=paths, group=group)
    for s in M_SEEDS:
        purge(f"linear/{group}/s{s}-purge", vec / "linear-control" / f"s{s}",
              [inventory(archived)], [f"{b.rel(b.record('linear-control', f's{s}'))}/record.json"],
              group)  # fmt: skip
    b.end("linear")

    # ---- predictions only, for extension 1's twenty new orders (no E2 run is read)
    b.begin()
    for s in ORDER_SEEDS:
        group = f"s{s}"
        paths, made = [], []
        for offset in ORDER_OFFSETS:
            tag = f"s{s}-order{offset // 1000}"
            folder = vec / "sealed" / tag
            linear(f"sealed/{group}/order{offset // 1000}", (CONTROL_ORDER, rate(ETA_L),
                   f"seed={s}", f"train.order_seed={s + offset}"), folder,
                   ("--reference-fresh-run", pod2_run(s)), group=group, k=False)  # fmt: skip
            target = b.record("sealed", tag)
            settle(f"sealed/{group}/order{offset // 1000}-settle", folder, target, group)
            paths.append(b.rel(target))
            made.append((folder, f"{b.rel(target)}/record.json"))
        archived = b.runs_archive(f"sealed/{group}/archive", paths=paths, group=group)
        for folder, record in made:
            purge(f"sealed/{group}/{folder.name}-purge", folder, [inventory(archived)], [record],
                  group)  # fmt: skip
    b.end("sealed")

    # ---- M3 (small amplitudes, tokens) and M4's removals per seed; then, in a group of their
    # own, M4's parity, matched K and mixtures; the vector files purged after both are archived
    b.begin()
    inventories, consumers = [], []
    for s in M_SEEDS:
        group = f"s{s}"
        paths, run_dirs = [], []
        source = ("--f-run", replay_dir(s), "--checkpoints", b.checkpoints)
        rec, run_dir = vector_run(f"points/{group}/small", f"small_s{s}", s, source,
                                  points=SMALL_POINTS, campaign=replays[s][0],
                                  group=group)  # fmt: skip
        paths.append(rec)
        run_dirs.append((run_dir, rec))
        tokens = b.record("tokens", f"s{s}.json")
        step(f"points/{group}/tokens", with_runs("--unless", tokens, "--",
             "scripts/study5_tokens.py", *fresh_args(s), *source, "--scales", *TOKEN_SCALES,
             "--out", tokens, "--surrogate", spec[0], campaign=replays[s][0]),
             outputs=(str(tokens),), group=group, minutes_key="tokens")  # fmt: skip
        paths.append(b.rel(tokens))
        for kind in ("mean", "mixture"):
            rec, run_dir = vector_run(f"points/{group}/removal-{kind}", f"r{kind}_s{s}", s,
                                      ("--vector", m1 / f"removal_{kind}_{s}.pt"), full=(1.0,),
                                      group=group)  # fmt: skip
            paths.append(rec)
            run_dirs.append((run_dir, rec))
            consumers.append(rec)
        archived = b.runs_archive(f"points/{group}/archive", paths=paths, run_dirs=run_dirs,
                                  group=group)  # fmt: skip
        inventories.append(inventory(archived))
    group = "global"
    paths, run_dirs = [], []
    mean = ("--vector", m1 / "mean.pt")
    for name, label_, kw in (
        ("points/global/mean", "mean",
         {"points": (1.0, -1.0, 2.0, -2.0), "full": (1.0, -1.0, 2.0, -2.0)}),
        ("points/global/mean-k", "mean_k", {"match": True}),
        ("points/global/mean-neg-k", "mean_neg_k", {"match": True, "sign": -1}),
    ):  # fmt: skip
        rec, run_dir = vector_run(name, label_, first, mean, group=group, **kw)
        paths.append(rec)
        run_dirs.append((run_dir, rec))
    for k in range(1, MIXTURES + 1):
        rec, run_dir = vector_run(f"points/global/mix{k}", f"mix{k}", first,
                                  ("--vector", m1 / f"mixture_{k}.pt"), full=(1.0,), match=True,
                                  group=group)  # fmt: skip
        paths.append(rec)
        run_dirs.append((run_dir, rec))
    archived = b.runs_archive("points/global/archive", paths=paths, run_dirs=run_dirs,
                              group=group)  # fmt: skip
    purge("points/global/purge-m1", m1, [*inventories, inventory(archived), inventory(farchived)],
          [*consumers, *paths, *force_paths], group)  # fmt: skip
    b.end("points")

    # ---- M6: single adjacent swaps on four seeds; every edited order's predictions
    def edited_block(block: str, seeds, edits_of) -> None:
        b.begin()
        for s in seeds:
            group = f"s{s}"
            edits = edits_of(s)
            cells = [edit_cell(s, tag, edit) for tag, edit in edits if tag != "twin"]
            _, name = b.launch(f"{block}-{group}", block, cells, minutes=len(cells) * F_RUN,
                               group=group)  # fmt: skip
            paths, made = [], []
            for tag, edit in edits:
                folder = vec / "edits" / f"s{s}-{tag}"
                linear(f"{block}/{group}/{tag}-predict", edit_args(s, edit), folder,
                       ("--reference-fresh-run", pod2_run(s)), group=group, k=False)  # fmt: skip
                target = b.record("edits", f"s{s}-{tag}")
                settle(f"{block}/{group}/{tag}-settle", folder, target, group)
                paths.append(b.rel(target))
                made.append((folder, f"{b.rel(target)}/record.json"))
            archived = b.runs_archive(f"{block}/{group}/archive", paths=paths, campaign=name,
                                      group=group)  # fmt: skip
            for folder, record in made:
                purge(f"{block}/{group}/{folder.name}-purge", folder, [inventory(archived)],
                      [record], group)  # fmt: skip
        b.end(block)

    edited_block("swaps", SWAP_SEEDS, lambda s: [
        *[(f"swap{t}", f"{{kind: swap_batches, step: {t}}}") for t in SWAP_STEPS],
        ("twin", TWIN_EDIT)])  # fmt: skip
    # ---- step 2: early and late shuffles of whole batches, with their predictions
    edited_block("shuffles", M_SEEDS, lambda s: [
        (f"shuffle{a}-{z}-{d}", shuffle_edit(a, z, d)) for a, z in SHUFFLE_RANGES
        for d in SHUFFLE_DRAWS])  # fmt: skip
    b.close(records=(b.rel(b.records), VECTOR_EXPERIMENT), json_from=vec)
    return b.plan(commit, gpu_code, None, spec)


# ------------------------------------------------------------------ the executor


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def new_attempt() -> str:
    """An execution attempt's id: the UTC time and a short random suffix."""
    return f"{_stamp().lower()}-{secrets.token_hex(3)}"


def attempt_argv(argv: tuple[str, ...], attempt: str) -> list[str]:
    """A step's argv with ``ATTEMPT`` replaced by the attempt's id."""
    return [arg.replace(ATTEMPT, attempt) for arg in argv]


def append(journal: Path, entry: dict[str, Any]) -> None:
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"time": _now(), **entry}) + "\n")
        out.flush()
        os.fsync(out.fileno())


def read_journal(journal: Path) -> dict[str, str]:
    """Each step's last state: done, kept, failed, skipped, or interrupted (a start with no end).
    A line that is not whole (a crash in the middle of an append) is ignored."""
    status: dict[str, str] = {}
    if not journal.exists():
        return status
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        step = entry.get("step") if isinstance(entry, dict) else None
        if not isinstance(step, str):
            continue
        if entry.get("event") == "start":
            status[step] = "interrupted"
        elif entry.get("event") in ("end", "skip"):
            status[step] = entry.get("status", "failed")
    return status


def _attempt_log(logs: Path, step: Step) -> Path:
    stem = step.name.replace("/", "__")
    logs.mkdir(parents=True, exist_ok=True)
    k = 1
    while (logs / f"{stem}.{k}.log").exists():
        k += 1
    return logs / f"{stem}.{k}.log"


def _kept_by_runs(step: Step, runs: Path) -> bool:
    """Every run the launch step names completed, by its launcher journal."""
    if not step.kept_if_runs:
        return False
    for entry in step.kept_if_runs:
        campaign, run_label = entry.split("\t", 1)
        try:
            completed_run(runs, campaign, run_label)
        except StepError:
            return False
    return True


def _set_aside(path: Path) -> Path:
    """Move a record aside beside itself (never deleted); its new path."""
    target = path.with_name(f"{path.name}.aside-{_stamp()}")
    path.rename(target)
    return target


def _run(
    argv: list[str], *, cwd: Path, env: dict[str, str], log: Path, timeout: int | None
) -> int | str:
    """The step's process, in its own session; past ``timeout`` seconds its whole process
    group is killed and the outcome is ``"timeout"``."""
    with log.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return "timeout"
        except BaseException:  # the driver itself is stopping: its step goes with it
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def run_step(
    step: Step,
    status: dict[str, str],
    *,
    plan: Plan,
    journal: Path,
    logs: Path,
    python: list[str],
    cwd: Path,
    env: dict[str, str],
    attempt: str = "",
) -> str:
    """Run ``step`` unless its journal or its outputs say it is done; its outcome (its argv's
    ``ATTEMPT`` the execution attempt's id). A gate's
    record is kept only by its verdict, for the plan's commit and GPU code (``gate_problem``):
    otherwise an engineering or surrogate record is moved aside and the step runs again, and a
    predictions record fails the step."""
    before = status.get(step.name)
    if not step.always and before in ("done", "kept", "failed"):
        return before
    if not step.always and step.outputs and all(Path(p).exists() for p in step.outputs):
        problem = gate_problem(step.gate, Path(step.outputs[0]), plan) if step.gate else None
        if problem is None:
            append(journal, {"event": "end", "step": step.name, "status": "kept", "exit": None})
            status[step.name] = "kept"
            return "kept"
        if step.gate == "predictions":
            append(
                journal,
                {
                    "event": "end",
                    "step": step.name,
                    "status": "failed",
                    "exit": None,
                    "problem": problem,
                },
            )
            print(f"== {_now()} failed {step.name}: {problem}", flush=True)
            status[step.name] = "failed"
            return "failed"
        moved = _set_aside(Path(step.outputs[0]))
        append(
            journal, {"event": "aside", "step": step.name, "record": moved.name, "problem": problem}
        )
        print(f"== {_now()} {step.name}: its record set aside ({problem})", flush=True)
    elif not step.always and _kept_by_runs(step, plan.runs):
        append(journal, {"event": "end", "step": step.name, "status": "kept", "exit": None})
        status[step.name] = "kept"
        return "kept"
    log = _attempt_log(logs, step)
    append(journal, {"event": "start", "step": step.name, "log": log.name})
    print(f"== {_now()} start {step.name}", flush=True)
    argv = attempt_argv(step.argv, attempt)
    code = _run([*python, *argv], cwd=cwd, env=env, log=log, timeout=step.timeout)
    outcome = "done" if code == 0 else "failed"
    if code != 0 and step.exists_ok and EXISTS in log.read_text(encoding="utf-8", errors="replace"):
        outcome = "kept"
    problem = None
    if outcome == "done" and step.gate:  # the record just written must stand too
        problem = gate_problem(step.gate, Path(step.outputs[0]), plan)
        if problem is not None:
            outcome = "failed"
    end = {"event": "end", "step": step.name, "status": outcome, "exit": code}
    append(journal, end if problem is None else {**end, "problem": problem})
    print(f"== {_now()} {outcome} {step.name} (exit {code})", flush=True)
    status[step.name] = outcome
    return outcome


def _final(status: dict[str, str]) -> bool:
    return status in ("done", "kept", "failed")


def execute(plan: Plan, *, python: list[str], cwd: Path, env: dict[str, str]) -> int:
    """Run the plan's blocks in order (module docstring, "Failures", "Resumable"); then the close,
    whatever ends the blocks (an exception too): its deadline last, in a ``finally``. Each
    execution is an attempt of its own (``new_attempt``, in the journal's ``driver`` line): its
    mutable files go to the bucket under its id. COMPLETED, or STOPPED when a stop rule fired."""
    journal, logs = plan.out / "journal.jsonl", plan.out / "logs"
    attempt = new_attempt()
    run = {"plan": plan, "journal": journal, "logs": logs, "python": python, "cwd": cwd, "env": env,
           "attempt": attempt}  # fmt: skip
    close = next((b for b in plan.blocks if b.name == CLOSE), Block(CLOSE, ()))
    status: dict[str, str] = {}
    stopped = False
    try:
        status = read_journal(journal)
        append(journal, {"event": "driver", "pod": plan.pod, "commit": plan.commit,
                         "attempt": attempt})  # fmt: skip
        stopped = _blocks(plan, status, run)
    finally:
        deadlines = [s for s in close.steps if s.kind == "deadline"]
        try:
            for step in close.steps:
                if step.kind != "deadline":
                    run_step(step, status, **run)
        finally:
            for step in deadlines:
                try:
                    run_step(step, status, **run)
                except Exception as exc:  # the journal itself may be unwritable
                    print(f"== close deadline through the journal failed ({exc}); set directly")
                    _run(
                        [*python, *attempt_argv(step.argv, attempt)],
                        cwd=cwd,
                        env=env,
                        log=Path(tempfile.mkstemp(suffix=".log")[1]),
                        timeout=step.timeout,
                    )
    return STOPPED if stopped else COMPLETED


def _blocks(plan: Plan, status: dict[str, str], run: dict[str, Any]) -> bool:
    """The blocks but the close; whether a stop rule fired."""
    journal = run["journal"]
    stopped = False
    for block in plan.blocks:
        if block.name == CLOSE:
            continue
        if (plan.out / "receipts" / f"{block.name}.json").exists():
            append(journal, {"event": "block", "block": block.name, "status": "archived"})
            continue
        skipped_groups: set[str] = set()
        touched: set[str] = set()  # at a stop: the groups that had begun
        for step in block.steps:
            blocked = stopped or (step.group and step.group in skipped_groups)
            # after a stop, the block's archives of what exists: its own and its begun groups'
            archives = step.kind == "archive" and (not stopped or step.group in touched)
            if blocked and not archives:
                if not _final(status.get(step.name)):
                    append(journal, {"event": "skip", "step": step.name, "status": "skipped"})
                    status[step.name] = "skipped"
                continue
            if (
                step.kind == "deadline"
                and step.group
                and all(
                    _final(status.get(s.name))
                    for s in block.steps
                    if s.group == step.group and s is not step
                )
            ):
                continue  # a group already ended sets no window
            outcome = run_step(step, status, **run)
            if outcome == "failed":
                if step.on_fail == "stop":
                    stopped = True
                    touched = {
                        "",
                        *(
                            s.group
                            for s in block.steps
                            if status.get(s.name) not in (None, "skipped")
                        ),
                    }
                    print(f"== {_now()} stop: {step.name} failed", flush=True)
                elif step.on_fail == "skip-group" and step.group:
                    skipped_groups.add(step.group)
        if stopped:
            break
    return stopped


def tolerance_stands(found: Any) -> bool:
    """Whether a check's recorded tolerance stands for the plan's (``EXT2_TOLERANCE``): the same
    measurement rule (relative and absolute parts, answers allowed) and an endpoint bound no
    looser than the plan's. A check that passed under a stricter endpoint bound (extension 2's
    amendment, before its second) passes under the plan's too."""
    if not isinstance(found, dict) or set(found) != set(EXT2_TOLERANCE):
        return False
    if any(found[k] != EXT2_TOLERANCE[k] for k in EXT2_TOLERANCE if k != "endpoint_distance"):
        return False
    bound = found["endpoint_distance"]
    return isinstance(bound, float) and 0 < bound <= EXT2_TOLERANCE["endpoint_distance"]


def gate_problem(kind: str, path: Path, plan: Plan) -> str | None:
    """Why a gate's existing record does not stand for the plan, or None. Its verdict (the
    engineering check equal, the surrogate checks passed; the predictions record of exactly the
    plan's seeds in order and as published) and its code: the plan's GPU code identity, from a
    clean checkout. A commit is compared by its GPU code identity, never by itself: a later
    commit that changes no GPU path keeps every gate."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"unreadable ({type(exc).__name__})"
    if not isinstance(record, dict):
        return "not a record"
    if kind == "predictions":
        return predictions_problem(record, fresh_seeds(plan.n), plan.gpu_code) or (
            published_problem(path, path.parent / RECEIPT_NAME, required=False)
        )
    if kind in ("engineering", "replay"):
        # the verdict: ``passed`` (exact, or within extension 2's numerical equivalence when the
        # record holds a tolerance; ``exact`` and ``within_tolerance`` apart), or an earlier
        # record's ``equal``; a replay group's: ``accepted_for_endpoint_analysis`` (extension 2's
        # third amendment)
        verdict = record.get("passed") if "passed" in record else record.get("equal")
        if kind == "replay":
            verdict = record.get("accepted_for_endpoint_analysis")
        if verdict is not True:
            if kind == "replay":
                return "the replay check did not accept the replays for endpoint analysis"
            return "the engineering check found a difference"
        found = record.get("tolerance")
        if found is not None and not tolerance_stands(found):
            return "the engineering check passed under another tolerance"
        if not record.get("pairs"):
            return "the engineering check holds no pair"
        for pair in record["pairs"]:
            if pair.get("rerun_dirty") is not False:
                return f"rerun {pair.get('rerun')} was not run from a clean checkout"
            if identity_of(pair.get("rerun_commit")) != plan.gpu_code:
                return f"rerun {pair.get('rerun')} ran under another GPU code"
    elif kind == "surrogate":
        if record.get("passes") is not True:
            return "the surrogate checks did not pass"
    elif kind == "completion":  # it stands while the launcher journals still say the same
        try:
            runs = completion_runs(plan.runs, list(record.get("campaigns") or []))
        except StepError as exc:
            return str(exc)
        return None if runs == record.get("runs") else "the fresh runs' journals changed since"
    else:
        raise ValueError(f"unknown gate {kind!r}")
    return code_problem(record, plan.gpu_code)


_IDENTITIES: dict[str, str | None] = {}


def identity_of(commit: Any) -> str | None:
    """The GPU code identity of ``commit`` in this checkout's history; None when it cannot be
    told (no commit, or one this checkout cannot place)."""
    if not isinstance(commit, str) or not commit:
        return None
    if commit not in _IDENTITIES:
        try:
            _IDENTITIES[commit] = code_identity.gpu_code_identity(commit)["sha256"]
        except Exception:
            _IDENTITIES[commit] = None
    return _IDENTITIES[commit]


def code_problem(record: dict[str, Any], gpu_code: str) -> str | None:
    """The record's code against the plan's: a clean checkout (``git.dirty`` false where
    recorded) and the GPU code identity of its commit (``code.commit``, else ``git.commit``;
    one this checkout can place), with which a recorded ``code.gpu_code_sha256`` agrees."""
    code = record.get("code") if isinstance(record.get("code"), dict) else {}
    git = record.get("git") if isinstance(record.get("git"), dict) else {}
    if "dirty" in git and git["dirty"] is not False:
        return "it was written from a checkout that was not clean"
    commit = code.get("commit") or git.get("commit")
    found = identity_of(commit)
    if found is None:
        return f"the GPU code of its commit {str(commit)[:12]} cannot be told"
    recorded = code.get("gpu_code_sha256")
    if recorded is not None and recorded != found:
        return "its recorded GPU code is not its commit's"
    if found != gpu_code:
        return "its GPU code identity is not the plan's"
    return None


def predictions_problem(record: dict[str, Any], seeds: list[int], gpu_code: str) -> str | None:
    """The predictions record against the plan: its kind, exactly the plan's seeds in order, its
    n and its GPU code identity (``code.gpu_code_sha256``) the plan's. Its commit
    (``code.commit``) is information: a later commit of the same GPU code keeps it."""
    if record.get("kind") != "study5-predictions":
        return "not a predictions record"
    if record.get("seeds") != list(seeds) or record.get("n") != len(seeds):
        return (
            f"its seeds ({record.get('n')}) are not the plan's {len(seeds)} seeds "
            f"{seeds[0]}-{seeds[-1]}, in order"
        )
    if [e.get("seed") for e in record.get("predictions") or []] != list(seeds):
        return "its predictions are not the plan's seeds, in order"
    if (record.get("code") or {}).get("gpu_code_sha256") != gpu_code:
        return "its GPU code identity is not the plan's"
    return None


def published_problem(record: Path, receipt: Path, *, required: bool = True) -> str | None:
    """The record against its publication: its sha256 that of the bucket copy (the receipt's,
    written after the copy was read back) and its sidecar's. With ``required`` false, a record
    not yet published has no problem here."""
    if not receipt.is_file():
        return "it is not published (no receipt)" if required else None
    try:
        published = json.loads(receipt.read_text(encoding="utf-8")).get("sha256")
        sidecar = record.with_name(record.name + ".sha256").read_text(encoding="utf-8")
    except (OSError, ValueError, AttributeError) as exc:
        return f"its receipt or sidecar is unreadable ({type(exc).__name__})"
    digest = sha256_of(record)
    if published != digest or sidecar != f"{digest}  {record.name}\n":
        return "its sha256 is not the published copy's"
    return None


# ------------------------------------------------------------------ the steps' helpers


def artifact_remote() -> str:
    remote = os.environ.get("ARTIFACT_REMOTE", "")
    if not remote.startswith("gs://"):
        raise StepError("ARTIFACT_REMOTE is not a gs:// path")
    return remote.rstrip("/")


def gcloud(args: list[str]) -> str:
    key = os.environ.get("ARTIFACT_GCS_KEY")
    if not key or not Path(key).is_file():
        raise StepError("ARTIFACT_GCS_KEY is not a readable key file")
    env = {
        **os.environ,
        "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": key,
        "CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED": "False",
    }
    done = subprocess.run(["gcloud", "storage", *args], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise StepError(f"gcloud storage {args[0]} failed (exit {done.returncode})")
    return done.stdout


def archive(inventory: Path, prefix: str, root: Path, paths: list[str]) -> dict[str, int]:
    """``scripts/archive.py INVENTORY PREFIX ROOT PATH...`` as its own process (every object's
    size and MD5 checked, the inventory appended); the files and their verified count."""
    done = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "archive.py"),
            str(inventory),
            prefix,
            str(root),
            *paths,
        ],
        capture_output=True,
        text=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise StepError(f"scripts/archive.py failed (exit {done.returncode})")
    result = json.loads(lines[-1])
    if done.returncode != 0 and not result.get("mismatched"):
        raise StepError(f"scripts/archive.py failed (exit {done.returncode})")
    return result


def md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def safe_rel(rel: str) -> str:
    path = PurePosixPath(rel)
    if path.is_absolute() or not path.parts or any(p in ("..", ".", "") for p in path.parts):
        raise StepError(f"{rel!r}: not a relative path inside its root")
    return path.as_posix()


def journal_events(runs: Path, campaign: str) -> list[dict[str, Any]]:
    path = runs / "launch" / f"{campaign}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def campaign_paths(runs: Path, campaign: str) -> list[str]:
    """The run directories a launcher campaign started and its logs, those that exist, relative
    to RUNS_DIR."""
    return sorted(set().union(*campaign_parts(runs, campaign)))


def campaign_parts(runs: Path, campaign: str) -> tuple[list[str], list[str]]:
    """A launcher campaign's paths that exist, relative to RUNS_DIR, as (written once, mutable):
    the directory of a run its journal shows ended is written once; its journal, its logs and the
    directory of a run without an end (cut off, settled later) can still change."""
    events = journal_events(runs, campaign)
    ended = {e.get("run_id") for e in events if e.get("event") == "end"}
    once, mutable = set(), {f"launch/{campaign}.jsonl", f"launch/{campaign}"}
    for entry in events:
        if entry.get("event") == "start" and isinstance(entry.get("run_id"), str):
            experiment = entry.get("experiment")
            if isinstance(experiment, str):
                rel = f"{safe_rel(experiment)}/{safe_rel(entry['run_id'])}"
                (once if entry["run_id"] in ended else mutable).add(rel)
    mutable -= once
    return (sorted(p for p in once if (runs / p).exists()),
            sorted(p for p in mutable if (runs / p).exists()))  # fmt: skip


def _start_of(runs: Path, campaign: str, run_id: str) -> dict[str, Any]:
    for entry in journal_events(runs, campaign):
        if entry.get("event") == "start" and entry.get("run_id") == run_id:
            return entry
    raise StepError(f"{campaign}: no start of run {run_id}")


def completed_run(runs: Path, campaign: str, run_label: str) -> Path:
    """The run directory of ``run_label``'s completed run in the launcher campaign's journal."""
    ends = [
        e
        for e in journal_events(runs, campaign)
        if e.get("event") == "end" and e.get("run") == run_label
    ]
    if not ends or ends[-1].get("outcome") != "done":
        found = ends[-1].get("outcome") if ends else "no end"
        raise StepError(f"{campaign}: {run_label} has no completed run ({found})")
    run_id = ends[-1]["run_id"]
    start = _start_of(runs, campaign, run_id)
    path = runs / safe_rel(start["experiment"]) / safe_rel(run_id)
    if not path.is_dir():
        raise StepError(f"{campaign}: the run directory of {run_label} is missing")
    return path


def latest_run_id(runs: Path, campaign: str, run_label: str) -> str | None:
    starts = [
        e
        for e in journal_events(runs, campaign)
        if e.get("event") == "start" and e.get("run") == run_label
    ]
    return starts[-1]["run_id"] if starts else None


RUN_REF = re.compile(r"\{run:([^{}@]+)(?:@([^{}@]+))?\}")


def substituted(argv: list[str], runs: Path, campaign: str) -> list[str]:
    """Every ``{run:LABEL}`` replaced by that label's completed run directory in ``campaign``,
    and every ``{run:LABEL@CAMPAIGN}`` by that label's in CAMPAIGN (extension 2), also inside a
    longer argument (``run:{run:LABEL}:5``)."""

    def run_dir(found: re.Match) -> str:
        return str(completed_run(runs, found.group(2) or campaign, found.group(1)))

    return [RUN_REF.sub(run_dir, arg) for arg in argv]


def move_aside(runs: Path, rel: str) -> Path | None:
    """Move a partial run directory to ``RUNS_DIR/aside/``, never reused; None when absent."""
    path = runs / safe_rel(rel)
    if not path.exists():
        return None
    target = runs / "aside" / f"{rel.replace('/', '--')}-{_stamp()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    return target


STREAM = re.compile(r"teacher/[^/]+/(training|preconditioner)/[^/]+/seed-(\d+)/.+")


def stream_rows(rows: list[dict[str, str]], seeds: list[int]) -> list[dict[str, str]]:
    """The inventory's rows of the seeds' training and preconditioner streams; each seed needs
    both, each row its object at ``ARTIFACT_REMOTE/data/<path>`` and the status ``ok``."""
    remote = artifact_remote()
    chosen, kinds = [], {seed: set() for seed in seeds}
    for row in rows:
        path = safe_rel(row.get("path", ""))
        found = STREAM.fullmatch(path)
        if found is None or int(found.group(2)) not in kinds:
            continue
        if row.get("url") != f"{remote}/data/{path}":
            raise StepError(f"{path}: its url is not under ARTIFACT_REMOTE/data")
        if row.get("status") != "ok":
            raise StepError(f"{path}: its inventory status is not ok")
        kinds[int(found.group(2))].add(found.group(1))
        chosen.append(row)
    for seed, found in kinds.items():
        if found != {"training", "preconditioner"}:
            raise StepError(
                f"seed {seed}: the inventory lacks its streams "
                f"{sorted({'training', 'preconditioner'} - found)}"
            )
    return chosen


def stream_dirs(data: Path, seed: int) -> list[str]:
    """A fresh seed's training and preconditioner stream directories, relative to DATA_DIR."""
    out = []
    for kind in ("training", "preconditioner"):
        found = sorted(data.glob(f"teacher/*/{kind}/*/seed-{seed}"))
        if len(found) != 1:
            raise StepError(f"seed {seed}: {len(found)} {kind} streams in DATA_DIR, not 1")
        out.append(found[0].relative_to(data).as_posix())
    return out


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def readback(url: str, expected: str) -> None:
    """Download ``url`` again and compare its sha256."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "readback"
        gcloud(["cp", "--quiet", url, str(local)])
        if sha256_of(local) != expected:
            raise StepError(f"{url.rsplit('/', 1)[-1]}: the copy read back differs")


def archive_paths(
    root: Path,
    remote: str,
    paths: list[str],
    inventory: Path,
    inventories_remote: str,
    *,
    mutable: list[str] = (),
    attempt_remote: str | None = None,
    remote_name: str | None = None,
) -> int:
    """Archive ``paths`` under ``root`` to ``ARTIFACT_REMOTE/<remote>`` and ``mutable`` to
    ``<attempt_remote>`` (the attempt's own place), every object verified (``scripts/archive.py
    --no-clobber``), then the inventory itself to ``<inventories_remote>/<remote_name>`` (its own
    name by default), read back; the inventory is in place only once both hold. The number of
    files archived."""
    base = artifact_remote()
    if mutable and attempt_remote is None:
        raise StepError("mutable paths need the attempt's place (--attempt-remote)")
    groups = []
    for prefix, rels in ((remote, paths), (attempt_remote, mutable)):
        present = [p for p in dict.fromkeys(safe_rel(p) for p in rels) if (root / p).exists()]
        if present:
            groups.append((prefix, present))
    name = remote_name or inventory.name
    staging = inventory.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / name
    staged.unlink(missing_ok=True)
    files = 0
    for prefix, present in groups:
        result = archive(staged, f"{base}/{prefix}", root, present)
        if result["mismatched"]:
            raise StepError(f"{result['mismatched']} of {result['files']} objects do not match")
        files += result["files"]
    if not groups:
        staged.write_text("path\tsize\tsha256\tmd5_b64\turl\tstatus\n", encoding="utf-8")
    self_inventory = staging / f"{Path(name).stem}.self.tsv"
    self_inventory.unlink(missing_ok=True)
    copy = archive(self_inventory, f"{base}/{inventories_remote}", staging, [name])
    if copy["mismatched"]:
        raise StepError(f"the inventory {name} does not match its copy")
    readback(f"{base}/{inventories_remote}/{name}", sha256_of(staged))
    os.replace(staged, inventory)
    self_inventory.unlink(missing_ok=True)
    return files


def remove_checkpoint(path: Path, root: Path, sha256: str, inventory: list[dict[str, str]]) -> None:
    """Remove an endpoint checkpoint from the container disk: only a regular file directly in
    ``root`` (no link), named ``<run id>.pt``, whose sha256 is ``sha256`` and which the archive's
    inventory shows verified with that sha256."""
    if path.is_symlink() or not path.is_file():
        raise StepError(f"{path}: not a regular checkpoint file")
    if path.parent.resolve() != root.resolve() or path.suffix != ".pt":
        raise StepError(f"{path}: outside the checkpoint directory {root}")
    rows = [r for r in inventory if r.get("path") == path.name]
    if not rows or any(r.get("status") != "ok" or r.get("sha256") != sha256 for r in rows):
        raise StepError(f"{path.name}: the archive does not show it verified")
    if sha256_of(path) != sha256:
        raise StepError(f"{path.name}: its sha256 is not the recorded one")
    path.unlink()


def predictions_record(records: Path, seeds: list[int], identity: dict[str, Any]) -> dict[str, Any]:
    """The predictions record (module docstring, "Pod 2"): every seed's p at η_l, in order."""
    entries = []
    for seed in seeds:
        path = records / f"s{seed}.json"
        if not path.is_file():
            raise StepError(f"seed {seed}: no prediction record")
        data = path.read_bytes()
        found = json.loads(data)
        if found.get("kind") != "study5-prediction" or found.get("seed") != seed:
            raise StepError(f"seed {seed}: {path.name} is not its prediction record")
        if found.get("lr") != ETA_L:
            raise StepError(f"seed {seed}: its prediction's lr {found.get('lr')!r} is not η_l")
        p = found.get("p")
        if not isinstance(p, float) or not math.isfinite(p):
            raise StepError(f"seed {seed}: its p is not a finite number")
        entries.append(
            {"seed": seed, "p": p, "record": path.name, "sha256": hashlib.sha256(data).hexdigest()}
        )
    return {
        "kind": "study5-predictions",
        "n": len(seeds),
        "seeds": list(seeds),
        "lr": ETA_L,
        "predictions": entries,
        "code": {"commit": identity.get("commit"), "gpu_code_sha256": identity.get("sha256")},
        "written_utc": _now(),
    }


def write_record(path: Path, record: dict[str, Any]) -> str:
    """Write the record and ``<path>.sha256`` beside it, each whole; its sha256."""
    if path.exists():
        raise StepError(f"{path} exists; a record is never overwritten")
    data = (json.dumps(record, indent=1, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, content in (
        (path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode()),
        (path, data),
    ):
        partial = target.with_name(f".{target.name}.partial")
        partial.write_bytes(content)
        os.replace(partial, target)
    return digest


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


# ------------------------------------------------------------------ the steps


def deadline_cap() -> int | None:
    """The hard cap ``ZFR_DEADLINE_EPOCH`` (Unix seconds) of the Pod's environment; the watcher
    stops the Pod at it whatever the file says. None when it is not set or does not read."""
    try:
        return int(os.environ["ZFR_DEADLINE_EPOCH"])
    except (KeyError, ValueError):
        return None


def _iso(epoch: int | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, UTC).isoformat()


def cmd_deadline(args) -> int:
    """The deadline file set to now + ``--minutes``, never above the cap: the effective deadline
    is min(requested, cap), printed and (``--record``) recorded with both. With ``--reserve``,
    the window is admitted only when it and the reserved minutes (the close's export) both end
    before the cap; otherwise nothing is written and the step exits ``NOT_ADMITTED``. With
    ``--needs``, a window shorter than that is said so explicitly. With ``--admit`` (pod 3), the
    work is admitted only when that many minutes from now end before the cap (the work, its
    archive and the close); otherwise nothing is written and the step exits ``NOT_ADMITTED``,
    and the window itself stays the requested one, cut at the cap."""
    now = int(time.time())
    requested, cap = now + 60 * args.minutes, deadline_cap()
    found: dict[str, Any] = {"kind": "study5-deadline", "requested": _iso(requested),
                             "cap": _iso(cap), "minutes": args.minutes,
                             "reserve_minutes": args.reserve, "admit_minutes": args.admit,
                             "written_utc": _iso(now)}  # fmt: skip
    if args.admit and cap is not None and now + 60 * args.admit > cap:
        found |= {"admitted": False, "effective": None}
        print(
            f"not admitted: the work and the close need {args.admit} min, past the cap "
            f"{_iso(cap)}: no further group, the close follows"
        )
        if args.record:
            write_json(args.record, found)
        return NOT_ADMITTED
    if args.reserve and cap is not None and requested + 60 * args.reserve > cap:
        found |= {"admitted": False, "effective": None}
        print(
            f"not admitted: the window of {args.minutes} min and the close's {args.reserve} "
            f"min end after the cap {_iso(cap)}: no further group, the close follows"
        )
        if args.record:
            write_json(args.record, found)
        return NOT_ADMITTED
    effective = requested if cap is None else min(requested, cap)
    window = (effective - now) / 60
    found |= {"admitted": True, "effective": _iso(effective), "effective_minutes": window}
    if args.needs is not None:
        found |= {"needs_minutes": args.needs, "short": window < args.needs}
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "pod_deadline_watch.py"), "--set", str(effective)]
    )
    found["exit"] = done.returncode
    print(f"deadline requested {_iso(requested)} cap {_iso(cap)} effective {_iso(effective)} "
          f"exit {done.returncode}")  # fmt: skip
    if found.get("short"):
        print(f"WARNING: the effective window, {window:.1f} min before the cap, is shorter than "
              f"the {args.needs} min the export needs")  # fmt: skip
    if args.record:
        write_json(args.record, found)
    return done.returncode


def cmd_fetch_streams(args) -> int:
    base = artifact_remote()
    local = args.work / "fetched" / Path(args.inventory).name
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(f".{local.name}.partial")
        gcloud(["cp", "--quiet", f"{base}/{safe_rel(args.inventory)}", str(partial)])
        os.replace(partial, local)
    rows = stream_rows(read_inventory(local), args.seeds)
    kept = fetched = 0
    for row in rows:
        target = args.data / row["path"]
        if target.exists():
            if sha256_of(target) != row["sha256"]:
                raise StepError(f"{row['path']}: in DATA_DIR with another sha256; never replaced")
            kept += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial-fetch")
        gcloud(["cp", "--quiet", row["url"], str(partial)])
        if partial.stat().st_size != int(row["size"]) or sha256_of(partial) != row["sha256"]:
            partial.unlink()
            raise StepError(f"{row['path']}: the copy does not match the inventory")
        os.replace(partial, target)
        fetched += 1
    print(f"seeds {len(args.seeds)} files {len(rows)} fetched {fetched} kept {kept}")
    return 0


def cmd_fetch_run(args) -> int:
    base = artifact_remote()
    run_id = safe_rel(args.run_id)
    target = args.dest / run_id
    if target.exists():
        print(f"kept {run_id}")
        return 0
    prefix = f"{base}/{safe_rel(args.prefix)}/{run_id}"  # study 3's runs unless --prefix
    partial = args.dest / f".{run_id}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    count = 0
    for meta in json.loads(gcloud(["objects", "list", f"{prefix}/**", "--format=json"])):
        url = f"gs://{meta['bucket']}/{meta['name']}"
        rel = safe_rel(url[len(prefix) + 1 :])
        if rel.split("/", 1)[0] in SKIPPED:
            continue
        local = partial / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        gcloud(["cp", "--quiet", url, str(local)])
        md5 = md5_b64(local)
        if md5 != meta.get("md5_hash") or local.stat().st_size != int(meta["size"]):
            raise StepError(f"{run_id}/{rel}: the copy does not match its object")
        count += 1
    if not (partial / "meta.json").is_file():
        raise StepError(f"{run_id}: no meta.json in the archive")
    os.replace(partial, target)
    print(f"fetched {run_id} files {count}")
    return 0


def stale_reruns(runs: Path, experiment: str, config_hash: str, gpu_code: str) -> list[Path]:
    """The run directories of one config that did not run under ``gpu_code`` from a clean
    checkout (``meta.json``'s git state and its commit's GPU code identity)."""
    out = []
    for run_dir in sorted(p for p in (runs / experiment).glob("*") if p.is_dir()):
        try:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # the launcher settles a run without a readable meta.json
        if meta.get("config_hash") != config_hash:
            continue
        git = meta.get("git") or {}
        if git.get("dirty") is not False or identity_of(git.get("commit")) != gpu_code:
            out.append(run_dir)
    return out


def campaign_hashes(campaign_file: Path) -> list[tuple[str, str]]:
    """Each run's experiment and config hash in a launcher campaign, as the launcher computes
    them (its cell's config with its overrides, then ``seed=N``)."""
    from zero_field_refusal.config import config_hash
    from zero_field_refusal.protocol import load_run_config

    data = yaml.safe_load(campaign_file.read_text(encoding="utf-8"))
    out = []
    for cell in data["cells"]:
        for seed in cell["seeds"]:
            cfg = load_run_config(cell["config"], [*cell.get("overrides", []), f"seed={seed}"])
            out.append((str(cfg.experiment), config_hash(cfg)))
    return out


def engineering_launch(campaign_file: Path, moved: bool, gpu_code: str) -> list[str]:
    """The launcher's command for the engineering reruns: plain, or, once a stale rerun was
    moved aside (its start is then a used attempt), the cells' one recovery round."""
    argv = ["scripts/launch.py", str(campaign_file), "--max-attempts", "1"]
    if moved:
        argv += ["--recovery", f"reruns under GPU code {gpu_code[:12]}"]
    return argv


def cmd_eng_launch(args) -> int:
    """The engineering reruns: a rerun of another GPU code (or of an unclean checkout) is moved
    aside to ``RUNS_DIR/aside/`` (never deleted, never reused), then the launcher runs."""
    moved = False
    for experiment, found in campaign_hashes(args.campaign_file):
        for run_dir in stale_reruns(args.runs, experiment, found, args.gpu_code):
            target = move_aside(args.runs, run_dir.relative_to(args.runs).as_posix())
            print(f"moved aside {run_dir.name}: not of GPU code {args.gpu_code[:12]} ({target})")
            moved = True
    argv = engineering_launch(args.campaign_file, moved, args.gpu_code)
    return subprocess.run([sys.executable, str(REPO / argv[0]), *argv[1:]], cwd=REPO).returncode


def cmd_with_runs(args) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise StepError("with-runs needs a script after --")
    done = bool(args.unless and Path(args.unless).exists())
    if args.aside and not done:
        moved = move_aside(args.runs, args.aside)
        if moved is not None:
            print(f"moved aside {args.aside} to {moved.relative_to(args.runs)}")
    if args.aside_dir and not done:  # extension 2: an incomplete output folder, kept aside
        folder = Path(args.aside_dir)
        if folder.is_dir() and any(folder.iterdir()):
            target = folder.with_name(f"{folder.name}.aside-{_stamp()}")
            folder.rename(target)
            print(f"moved aside {folder.name} to {target.name}")
    if args.only_if and not Path(args.only_if).exists():  # extension 2: the input was not made
        out = Path(args.else_record)
        if not out.exists():
            reason = "its producer did not write it (its record says why)"
            write_json(out, {"available": False, "missing": Path(args.only_if).name,
                             "reason": reason})  # fmt: skip
        print(f"unavailable: {Path(args.only_if).name}")
        return 0
    if args.restore_label and not done:  # after a Pod stop: the checkpoint from its archive
        restore_checkpoint(args)
    argv = substituted(command, args.runs, args.campaign)
    return subprocess.run([sys.executable, str(REPO / argv[0]), *argv[1:]], cwd=REPO).returncode


def cmd_settle(args) -> int:
    """Extension 2: the JSON records of ``--dir`` (a vector folder on the volume) copied to
    ``--records`` (in the runs root, archived with its block); an identical copy already there is
    kept, another refused. Nothing is removed here (``purge``)."""
    folder, records = args.dir, args.records
    if not folder.is_dir():
        raise StepError(f"{folder}: no such folder")
    records.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(folder.glob("*.json")):
        target = records / path.name
        data = path.read_bytes()
        if target.exists():
            if target.read_bytes() != data:
                raise StepError(f"{target} exists with other content")
            continue
        partial = target.with_name(f".{target.name}.partial")
        partial.write_bytes(data)
        os.replace(partial, target)
        copied += 1
    print(f"settle {folder.name}: copied {copied}")
    return 0


def archived_problems(runs: Path, inventories: list[Path], consumers: list[str]) -> list[str]:
    """Why the ``consumers`` (records relative to the runs root) are not yet safely kept: each
    must exist and appear in one of the archive ``inventories`` (written only after the archive
    verified it) with status ok and the local file's sha256."""
    rows: dict[str, dict[str, str]] = {}
    problems = []
    for inventory in inventories:
        if not Path(inventory).is_file():
            problems.append(f"{Path(inventory).name}: not archived yet")
            continue
        rows.update({row["path"]: row for row in read_inventory(Path(inventory))})
    for rel in consumers:
        local = runs / safe_rel(rel)
        row = rows.get(rel)
        if not local.is_file():
            problems.append(f"{rel}: not written")
        elif row is None or row.get("status") != "ok" or row.get("sha256") != sha256_of(local):
            problems.append(f"{rel}: not in a verified archive")
    return problems


def cmd_json_snapshot(args) -> int:
    """Pod 4's close: every ``*.json`` below ``--src`` copied to ``--dest`` at the same relative
    path (an existing copy replaced); no other file. The close then archives it."""
    if not args.src.is_dir():
        print(f"json-snapshot: {args.src} absent, nothing copied")
        return 0
    copied = 0
    for path in sorted(args.src.rglob("*.json")):
        target = args.dest / path.relative_to(args.src)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial")
        partial.write_bytes(path.read_bytes())
        os.replace(partial, target)
        copied += 1
    print(f"json-snapshot: copied {copied}")
    return 0


def cmd_purge(args) -> int:
    """Extension 2: the disposable vector files (``*.pt``) of ``--dir`` removed once every
    ``--consumer`` record is written and archived (``archived_problems``); otherwise nothing is
    removed and the step fails, so that a failed consumer can be rerun on the same inputs. Records
    (``*.json``) are evidence and stay."""
    problems = archived_problems(args.runs, args.inventory, args.consumer)
    if problems:
        raise StepError(f"kept: {problems[0]}")
    removed = 0
    for path in sorted(args.dir.glob("*.pt")):
        path.unlink()
        removed += 1
    print(f"purge {args.dir.name}: removed {removed}")
    return 0


def cmd_drop_checkpoints(args) -> int:
    """Extension 2: the step checkpoints (and, with ``--endpoint``, the endpoint checkpoint) that
    each ``--run`` label's completed run in ``--campaign`` recorded, removed once every
    ``--consumer`` record is written and archived (``archived_problems``); a file is removed only
    when its bytes hash to the recorded sha256 (otherwise it is kept and the step fails).
    Checkpoints are disposable: the records that read them are the evidence."""
    problems = archived_problems(args.runs, args.inventory, args.consumer)
    if problems:
        raise StepError(f"kept: {problems[0]}")
    removed, kept = 0, []
    for run_label in args.run:
        run_dir = completed_run(args.runs, args.campaign, run_label)
        inputs = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["inputs"]
        found = list(inputs.get("step_checkpoints") or [])
        if args.endpoint and isinstance(inputs.get("endpoint_checkpoint"), dict):
            found.append(inputs["endpoint_checkpoint"])
        for record in found:
            path = Path(record["path"])
            if not path.is_file():
                continue
            if sha256_of(path) != record["sha256"]:
                kept.append(path.name)
                continue
            path.unlink()
            removed += 1
    print(f"dropped {removed} checkpoints" + (f", kept {len(kept)} that differ" if kept else ""))
    if kept:
        raise StepError(f"{kept[0]} does not hash to its record")
    return 0


def cmd_archive(args) -> int:
    """Written-once files to ``--remote`` (records, streams, the directories of ended runs and of
    scan or ladder runs whose record exists); mutable ones to ``--attempt-remote`` (``--mutable-
    path``, a ``--run-dir`` whose record does not exist yet, the launcher journals and logs, the
    directories of runs without an end), so that a later attempt never writes an earlier one's
    object (``archive_paths``)."""
    paths, mutable = list(args.path or []), list(args.mutable_path or [])
    for given in args.run_dir or []:
        rel, _, record = given.partition("=")
        if not record:
            raise StepError(f"--run-dir {given}: DIR=RECORD, relative to the root")
        (paths if (args.root / safe_rel(record)).is_file() else mutable).append(rel)
    for campaign in args.campaign or []:
        once, changing = campaign_parts(args.runs, campaign)
        paths += once
        mutable += changing
    if args.streams is not None:
        paths += stream_dirs(args.root, args.streams)
    if args.inventory.exists():
        print(f"kept {args.inventory.name}")
        return 0
    files = archive_paths(args.root, args.remote, paths, args.inventory, args.inventories_remote,
                          mutable=mutable, attempt_remote=args.attempt_remote)  # fmt: skip
    print(f"archived {files} files under {args.remote} ({args.inventory.name})")
    return 0


def cmd_snapshot(args) -> int:
    label = f"{args.label}-{_stamp()}" if args.unique else args.label
    inventory = args.out / "inventories" / f"snapshot-{label}.tsv"
    if inventory.exists():
        print(f"kept {inventory.name}")
        return 0
    target = args.out / "snapshots" / label
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ("journal.jsonl", "logs", "receipts"):
        source = args.out / name
        if source.is_dir():
            shutil.copytree(source, target / name)
        elif source.is_file():
            shutil.copy2(source, target / name)
    paths = [f"snapshots/{label}", *(args.include or [])]
    files = archive_paths(args.out, args.remote, paths, inventory, args.inventories_remote)
    print(f"snapshot {label}: {files} files")
    return 0


def cmd_receipt(args) -> int:
    """The block's receipt: every inventory of its archives verified. The receipt runs only once
    every archive step of the block has ended (each stops the campaign on a failure), so an
    absent inventory is an archive with nothing to archive: a seed's streams, skipped with its
    seed when their generation failed, or a run's checkpoint when it saved none. It is listed as
    absent."""
    entries, absent = [], []
    for path in map(Path, args.inventories):
        if not path.is_file():
            absent.append(path.name)
            continue
        rows = read_inventory(path)
        bad = [r for r in rows if r.get("status") != "ok"]
        if bad:
            raise StepError(f"{path.name}: {len(bad)} rows not verified")
        entries.append({"inventory": path.name, "rows": len(rows), "sha256": sha256_of(path)})
    write_json(
        args.out / "receipts" / f"{args.block}.json",
        {"block": args.block, "inventories": entries, "absent": absent, "written_utc": _now()},
    )
    print(
        f"receipt {args.block}: {len(entries)} inventories, {sum(e['rows'] for e in entries)} files"
    )
    return 0


def cmd_check_checkpoints(args) -> int:
    found = sorted(args.dir.glob("*.pt")) if args.dir.is_dir() else []
    print(f"checkpoints on the disk {len(found)} (at most {args.max} before a run)")
    return 0 if len(found) <= args.max else 1


def run_checkpoint(runs: Path, campaign: str, run_label: str,
                   directory: Path) -> tuple[str | None, Path | None, str | None]:  # fmt: skip
    """The label's latest run id, its checkpoint's path on the disk and the sha256 its run's
    summary records for it (None where there is none)."""
    run_id = latest_run_id(runs, campaign, run_label)
    if run_id is None:
        return None, None, None
    path = Path(directory) / f"{safe_rel(run_id)}.pt"
    summary = runs / safe_rel(_start_of(runs, campaign, run_id)["experiment"])
    summary = summary / run_id / "summary.json"
    record = None
    if summary.is_file():
        record = json.loads(summary.read_text(encoding="utf-8")).get("endpoint_checkpoint")
    return run_id, path, record.get("sha256") if isinstance(record, dict) else None


def cmd_checkpoint(args) -> int:
    """The η_l run's endpoint checkpoint archived and verified, then (without ``--keep``)
    removed from the disk. It is refused while the run's summary records no sha256 for it (the
    file stays). A checkpoint no longer on the disk must be in the bucket when the run recorded
    one. The group archives it right after its run (``--keep``) and removes it after the scan,
    so that a Pod stop in between loses no checkpoint (``restore_checkpoint``)."""
    run_id, path, recorded = run_checkpoint(args.runs, args.campaign, args.label, args.dir)
    if run_id is None:
        print(f"no run of {args.label}: no checkpoint")
        return 0
    if not path.exists():
        if recorded is None:
            print(f"no checkpoint of {run_id}")
            return 0
        if archived_earlier(args, path.name, recorded):
            print(f"checkpoint {run_id} archived earlier, sha256 {recorded}, removed from the disk")
            return 0
        raise StepError(f"{path.name}: recorded by its run but neither on the disk nor archived")
    if not isinstance(recorded, str):
        raise StepError(f"{path.name}: its run's summary records no sha256 for it; it stays")
    digest = sha256_of(path)
    if recorded != digest:
        raise StepError(f"{path.name}: its sha256 is not the one its run recorded")
    if not args.inventory.exists():
        archive_paths(args.dir, args.remote, [path.name], args.inventory, args.inventories_remote,
                      remote_name=checkpoint_inventory(path.name))  # fmt: skip
    rows = [r for r in read_inventory(args.inventory) if r.get("path") == path.name]
    if not rows or any(r.get("status") != "ok" or r.get("sha256") != digest for r in rows):
        raise StepError(f"{path.name}: the archive does not show it verified")
    if args.keep:
        print(f"checkpoint {run_id} sha256 {digest} archived and verified; kept on the disk")
        return 0
    remove_checkpoint(path, args.dir, digest, read_inventory(args.inventory))
    print(f"checkpoint {run_id} sha256 {digest} archived, verified and removed")
    return 0


def restore_checkpoint(args) -> None:
    """Before the endpoint's r or the scan: the η_l run's checkpoint on the disk, or, after a Pod
    stop wiped the container disk, fetched back from its verified archive (the object and its
    archive's inventory showing the run's recorded sha256, ``archived_earlier``), its sha256
    checked before it is put in place. Nothing to do without a run or a recorded sha256 (the
    script then refuses)."""
    run_id, path, recorded = run_checkpoint(args.runs, args.campaign, args.restore_label,
                                            args.restore_dir)  # fmt: skip
    if run_id is None or recorded is None or path.exists():
        return
    found = argparse.Namespace(remote=args.restore_remote, inventory=args.restore_inventory,
                               inventories_remote=args.restore_inventories_remote)  # fmt: skip
    if not archived_earlier(found, path.name, recorded):
        raise StepError(f"{path.name}: not on the disk and not in the bucket")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.restore")
    partial.unlink(missing_ok=True)
    gcloud(
        ["cp", "--quiet", f"{artifact_remote()}/{args.restore_remote}/{path.name}", str(partial)]
    )
    if sha256_of(partial) != recorded:
        partial.unlink()
        raise StepError(f"{path.name}: the copy fetched back is not of its run's sha256")
    os.replace(partial, path)
    print(f"checkpoint {run_id} restored from its archive, sha256 {recorded}")


def cmd_fetch_endpoint(args) -> int:
    """Extension 2's endpoint distance (docs/study5-preregistration.md, "Extension 2
    amendment"): when a replay's endpoint tensors (``scripts/study5_tensor_hash.py``'s record,
    ``--hashed``) are not the archived run's (``--expected``), the archived run's endpoint
    checkpoint is fetched back into ``--dest`` from its verified archive (``archived_earlier``:
    the object, and its archive's inventory showing ``--sha256``), and its sha256 checked before
    it is put in place. Equal
    tensors, or a replay file that does not match its own record (the check fails it), fetch
    nothing; a file already there with that sha256 is kept."""
    run = completed_run(args.runs, args.campaign, args.label)
    found = json.loads(args.hashed.read_text(encoding="utf-8")).get(run.name) or {}
    if found.get("file_ok") is not True:
        print(f"endpoint of {run.name} does not match its record: nothing fetched")
        return 0
    if found.get("tensors_sha256") == args.expected:
        print(f"endpoint tensors of {run.name} equal the archived run's: nothing fetched")
        return 0
    name = f"{safe_rel(args.run_id)}.pt"
    path = args.dest / name
    if path.exists() and sha256_of(path) == args.sha256:
        print(f"checkpoint {args.run_id} already fetched, sha256 {args.sha256}")
        return 0
    where = argparse.Namespace(remote=args.remote, inventory=args.dest / ".none.tsv",
                               inventories_remote=args.inventories_remote)  # fmt: skip
    if not archived_earlier(where, name, args.sha256):
        raise StepError(f"{name}: not in the bucket")
    args.dest.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{name}.fetch")
    partial.unlink(missing_ok=True)
    gcloud(["cp", "--quiet", f"{artifact_remote()}/{args.remote}/{name}", str(partial)])
    if sha256_of(partial) != args.sha256:
        partial.unlink()
        raise StepError(f"{name}: the copy fetched back is not of sha256 {args.sha256}")
    os.replace(partial, path)
    print(f"checkpoint {args.run_id} fetched from its archive, sha256 {args.sha256}")
    return 0


def cmd_remove_fetched(args) -> int:
    """A checkpoint ``fetch-endpoint`` fetched, removed once read: only when ``--record``, the
    distance record that reads it, is written (``scripts/study5_pairs.py``'s, its first item this
    file by ``--sha256``), and only a file of that sha256 (a copy of an archived object); an absent
    one is nothing to do. Otherwise it stays, for the record to be computed again."""
    if not args.path.exists():
        print(f"{args.path.name}: not on the disk")
        return 0
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise StepError(f"{args.path.name}: its distance record is not written; it stays") from None
    rows = record.get("pairs") if isinstance(record, dict) else None
    first = (rows[0].get("a") or {}) if isinstance(rows, list) and len(rows) == 1 else {}
    if record.get("kind") != "study5-pairs" or first.get("sha256") != args.sha256:
        raise StepError(f"{args.path.name}: its distance record does not read it; it stays")
    if sha256_of(args.path) != args.sha256:
        raise StepError(f"{args.path.name}: not of sha256 {args.sha256}; it stays")
    args.path.unlink()
    print(f"{args.path.name}: removed (a copy of its archive)")
    return 0


def checkpoint_inventory(name: str) -> str:
    """The bucket name of a checkpoint archive's inventory, by the checkpoint (``<run id>.pt``):
    one run's checkpoint, written once, so the inventory never changes either."""
    return f"checkpoint-{Path(name).stem}.tsv"


def archived_earlier(args, name: str, recorded: str) -> bool:
    """Whether the checkpoint ``name`` is in the bucket with the sha256 its run recorded: its
    object exists and the inventory of its archive (this --out's when it lists the checkpoint,
    else the copy archived in the bucket, ``checkpoint_inventory``) shows it verified with that
    sha256."""
    base = artifact_remote()
    if not json.loads(gcloud(["objects", "list", f"{base}/{args.remote}/{name}", "--format=json"])):
        return False
    inventory = args.inventory
    if not inventory.is_file() or not any(r.get("path") == name for r in read_inventory(inventory)):
        inventory = args.inventory.parent / ".staging" / f"remote-{checkpoint_inventory(name)}"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        try:
            gcloud(
                [
                    "cp",
                    "--quiet",
                    f"{base}/{args.inventories_remote}/{checkpoint_inventory(name)}",
                    str(inventory),
                ]
            )
        except StepError:
            raise StepError(f"{name}: archived, but no inventory shows its sha256") from None
    rows = [r for r in read_inventory(inventory) if r.get("path") == name]
    if not rows or any(r.get("status") != "ok" or r.get("sha256") != recorded for r in rows):
        raise StepError(f"{name}: the archived object is not shown verified with its run's sha256")
    return True


def cmd_predictions(args) -> int:
    identity = code_identity.gpu_code_identity()
    record = predictions_record(args.records, args.seeds, identity)
    digest = write_record(args.out, record)
    print(f"predictions {len(args.seeds)} seeds sha256 {digest}")
    return 0


def _created(meta: dict[str, Any]) -> str:
    for key in ("creation_time", "timeCreated", "time_created", "creationTime"):
        if isinstance(meta.get(key), str):
            return meta[key]
    raise StepError("the object's description holds no creation time")


def cmd_publish(args) -> int:
    base = artifact_remote()
    record, sidecar = args.record, args.record.with_name(args.record.name + ".sha256")
    digest = sha256_of(record)
    if sidecar.read_text(encoding="utf-8") != f"{digest}  {record.name}\n":
        raise StepError(f"{sidecar.name} does not hold the record's sha256")
    staged = args.inventory.with_name(f".{args.inventory.name}.partial")
    staged.unlink(missing_ok=True)
    result = archive(staged, f"{base}/{args.remote}", record.parent, [record.name, sidecar.name])
    if result["mismatched"]:
        raise StepError("the predictions record's copy does not match")
    url = f"{base}/{args.remote}/{record.name}"
    readback(url, digest)
    meta = json.loads(gcloud(["objects", "describe", url, "--format=json"]))
    # The receipt is archived from a staging folder and put in place last; a receipt staged by
    # an interrupted attempt is the one archived (its object exists), so it is kept.
    staging = args.receipt.parent / ".staging"
    pending = staging / args.receipt.name
    if not pending.is_file():
        write_json(
            pending,
            {
                "kind": "study5-predictions-receipt",
                "record": record.name,
                "sha256": digest,
                "object": f"$ARTIFACT_REMOTE/{args.remote}/{record.name}",
                "created": _created(meta),
                "verified_utc": _now(),
            },
        )
    receipt = json.loads(pending.read_text(encoding="utf-8"))
    if receipt.get("sha256") != digest:
        raise StepError(f"{pending} is the receipt of another record")
    copy = archive(staged, f"{base}/{args.remote}", staging, [pending.name])
    if copy["mismatched"]:
        raise StepError("the receipt's copy does not match")
    os.replace(staged, args.inventory)
    os.replace(pending, args.receipt)
    print(f"published {record.name} sha256 {digest} created {receipt['created']}")
    return 0


def cmd_check_published(args) -> int:
    """Before any fresh F run: the record is the one published (its sha256 that of the bucket
    copy, in the receipt and its sidecar), of exactly the plan's seeds in order and n, under the
    plan's GPU code identity."""
    if not args.record.is_file():
        raise StepError("the predictions record is missing: no fresh run starts")
    problem = published_problem(args.record, args.receipt)
    if problem is None:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        problem = predictions_problem(record, args.seeds, args.gpu_code)
    if problem is not None:
        raise StepError(f"the predictions record does not stand for this plan: {problem}")
    created = json.loads(args.receipt.read_text(encoding="utf-8")).get("created")
    print(f"published {args.record.name} sha256 {sha256_of(args.record)} created {created}")
    return 0


def completion_runs(runs: Path, campaigns: list[str]) -> list[dict[str, Any]]:
    """Every run the launcher campaigns' journals show as started, each with its recorded end
    and outcome, in journal order; ``StepError`` when a started run has no end."""
    out, open_ = [], []
    for campaign in campaigns:
        events = journal_events(runs, campaign)
        ends = {e.get("run_id"): e for e in events if e.get("event") == "end"}
        for start in (e for e in events if e.get("event") == "start"):
            end = ends.get(start.get("run_id"))
            if end is None:
                open_.append(f"{campaign}: {start.get('run_id')}")
                continue
            out.append({"campaign": campaign, "run": start.get("run"),
                        "run_id": start.get("run_id"), "experiment": start.get("experiment"),
                        "started": start.get("time"), "ended": end.get("time"),
                        "outcome": end.get("outcome")})  # fmt: skip
    if open_:
        raise StepError(f"{len(open_)} started fresh runs have no end (first: {open_[0]}): no "
                        "completion record")  # fmt: skip
    return out


def fetch_object(meta: dict[str, Any], target: Path) -> None:
    """One bucket object to ``target``, its size and MD5 checked against the listing."""
    url = f"gs://{meta['bucket']}/{meta['name']}"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial-restore")
    gcloud(["cp", "--quiet", url, str(partial)])
    if partial.stat().st_size != int(meta["size"]) or md5_b64(partial) != meta.get("md5_hash"):
        partial.unlink()
        raise StepError(f"{meta['name']}: the copy does not match its object")
    os.replace(partial, target)


def restore_runs(archive_id: str, dest: Path) -> dict[str, int]:
    """A Pod's runs restored from the bucket into ``dest`` (the analysis's runs root), in the
    order that keeps every completed record: first each attempt's ``attempts/<attempt>/runs/``
    in the attempts' chronological order (the id begins with its UTC time; a later one replaces
    an earlier one's file), then every stable run directory under ``runs/`` as a whole (the
    directory an attempt restored at its path removed first, so that no file of a partial run
    survives beside its completed rerun), then the stable records. Each object's size and MD5
    checked. The counts restored."""
    if not _ID.fullmatch(archive_id):
        raise StepError(f"--archive-id {archive_id!r}: not an archive id")
    prefix = f"{artifact_remote()}/study5/{archive_id}"
    listing = json.loads(gcloud(["objects", "list", f"{prefix}/**", "--format=json"]))
    objects = {f"gs://{m['bucket']}/{m['name']}"[len(prefix) + 1 :]: m for m in listing}
    attempts: dict[str, list[tuple[str, dict]]] = {}
    stable_dirs: dict[str, list[tuple[str, dict]]] = {}
    stable_files: list[tuple[str, dict]] = []
    for rel, meta in objects.items():
        parts = PurePosixPath(rel).parts
        if len(parts) > 3 and parts[0] == "attempts" and parts[2] == "runs":
            attempts.setdefault(parts[1], []).append((safe_rel("/".join(parts[3:])), meta))
        elif len(parts) > 1 and parts[0] == "runs":
            inner = safe_rel("/".join(parts[1:]))
            inner_parts = PurePosixPath(inner).parts
            if len(inner_parts) >= 3 and inner_parts[0] != "study5_records":
                stable_dirs.setdefault("/".join(inner_parts[:2]), []).append((inner, meta))
            else:
                stable_files.append((inner, meta))
    counts = {"attempts": len(attempts), "attempt_files": 0, "run_directories": 0, "files": 0}
    for attempt in sorted(attempts):  # chronological: the id starts with its UTC time
        for rel, meta in sorted(attempts[attempt], key=lambda x: x[0]):
            fetch_object(meta, dest / rel)
            counts["attempt_files"] += 1
    for run_dir, found in sorted(stable_dirs.items()):  # completed: whole and authoritative
        if (dest / run_dir).exists():
            shutil.rmtree(dest / run_dir)
        for rel, meta in sorted(found, key=lambda x: x[0]):
            fetch_object(meta, dest / rel)
        counts["run_directories"] += 1
    for rel, meta in sorted(stable_files, key=lambda x: x[0]):
        fetch_object(meta, dest / rel)
        counts["files"] += 1
    return counts


def cmd_restore_runs(args) -> int:
    counts = restore_runs(args.archive_id, args.dest)
    print(f"restored {args.archive_id}: {counts['attempts']} attempts "
          f"({counts['attempt_files']} files), {counts['run_directories']} completed run "
          f"directories, {counts['files']} records")  # fmt: skip
    return 0


def cmd_completion(args) -> int:
    """Pod 2's completion record, once its last fresh group has ended: every fresh run the
    launcher journals show as started, with its end and outcome; written only when no started
    fresh run lacks an end. The analysis opens no fresh label written before its time."""
    runs = completion_runs(args.runs, list(args.campaign))
    record = {"kind": "study5-fresh-completion", "campaigns": list(args.campaign),
              "runs": runs, "started": len(runs), "written_utc": _now()}  # fmt: skip
    digest = write_record(args.out, record)
    print(f"completion {len(runs)} fresh runs started and ended sha256 {digest}")
    return 0


def cmd_export_checkpoints(args) -> int:
    """The close's export (a planned shutdown: the end, a stop, SIGTERM): every recorded endpoint
    checkpoint of the given η_l runs (``--run CAMPAIGN=LABEL``) not yet verified in the archive
    is uploaded and verified, its sha256 the one its run recorded. Each result goes to the record
    (``--record``); a failure that remains is printed with the path, the recorded sha256 and the
    error, and the close goes on (exit 1)."""
    results = []
    for given in args.run:
        campaign, _, run_label = given.partition("=")
        entry: dict[str, Any] = {"campaign": campaign, "run": run_label}
        try:
            run_id, path, recorded = run_checkpoint(args.runs, campaign, run_label, args.dir)
        except (StepError, OSError, ValueError) as exc:
            results.append({**entry, "status": "failed", "error": str(exc)})
            continue
        if run_id is None or recorded is None:
            continue  # no run, or no checkpoint recorded: nothing to export
        entry |= {"run_id": run_id, "path": str(path), "sha256": recorded}
        local = [i for i in sorted(args.inventories.glob("checkpoint-*.tsv"))
                 if any(r.get("path") == path.name and r.get("status") == "ok"
                        and r.get("sha256") == recorded for r in read_inventory(i))]  # fmt: skip
        try:
            if local:
                status = "verified"  # this --out's archive shows it verified
            elif path.exists():
                if sha256_of(path) != recorded:
                    raise StepError("its sha256 on the disk is not the one its run recorded")
                inventory = args.inventories / f"close-{checkpoint_inventory(path.name)}"
                inventory.unlink(missing_ok=True)
                archive_paths(args.dir, args.remote, [path.name], inventory,
                              args.inventories_remote,
                              remote_name=checkpoint_inventory(path.name))  # fmt: skip
                rows = [r for r in read_inventory(inventory) if r.get("path") == path.name]
                if not rows or any(r.get("status") != "ok" or r.get("sha256") != recorded
                                   for r in rows):  # fmt: skip
                    raise StepError("the archive does not show it verified")
                status = "exported"
            else:
                found = argparse.Namespace(remote=args.remote,
                                           inventory=args.inventories / "none.tsv",
                                           inventories_remote=args.inventories_remote)  # fmt: skip
                if not archived_earlier(found, path.name, recorded):
                    raise StepError("neither on the disk nor in the archive")
                status = "verified"
            results.append({**entry, "status": status})
        except (StepError, OSError, ValueError, subprocess.SubprocessError) as exc:
            results.append({**entry, "status": "failed", "error": str(exc)})
    failed = [r for r in results if r["status"] == "failed"]
    for r in failed:
        print(f"CHECKPOINT NOT EXPORTED: {r.get('path', r['run'])} sha256 {r.get('sha256')}: "
              f"{r['error']}")  # fmt: skip
    write_json(args.record, {"kind": "study5-close-checkpoints", "checkpoints": results,
                             "failed": len(failed), "written_utc": _now()})  # fmt: skip
    print(f"checkpoints {len(results)} exported or verified {len(results) - len(failed)} "
          f"failed {len(failed)}")  # fmt: skip
    return 1 if failed else 0


# ------------------------------------------------------------------ the drivers


def seeds_problems(plan: Plan, repo: Path | None = None) -> list[str]:
    """Pod 2's seeds are the n chosen before generation: ``SEEDS_FILE`` at the plan's commit (the
    clean checkout's) holds n and its seeds, one of the registration's two choices, its sha256
    named in the registration there, and the plan's fresh seeds are exactly its seeds."""
    repo = REPO if repo is None else repo
    try:
        data = (repo / SEEDS_FILE).read_bytes()
        text = (repo / REGISTRATION).read_text(encoding="utf-8")
        chosen = json.loads(data)
    except (OSError, ValueError) as exc:
        return [f"{SEEDS_FILE} or the registration does not read ({type(exc).__name__})"]
    problems = []
    n, listed = (chosen.get("n"), chosen.get("seeds")) if isinstance(chosen, dict) else (None, None)
    if n not in N_CHOICES or listed != fresh_seeds(n):
        problems.append(f"{SEEDS_FILE} is not n = 40 (50000-50039) or n = 24 (50000-50023)")
    if hashlib.sha256(data).hexdigest() not in text:
        problems.append(f"{REGISTRATION} does not name the sha256 of {SEEDS_FILE}")
    if plan.n != n or listed != fresh_seeds(plan.n):
        problems.append(f"the plan's n {plan.n} is not the committed choice in {SEEDS_FILE}")
    return problems


def committed_n(repo: Path | None = None) -> int | None:
    """The n of ``SEEDS_FILE`` (None when it does not read: the preflight then refuses)."""
    try:
        found = json.loads(((REPO if repo is None else repo) / SEEDS_FILE).read_text())
    except (OSError, ValueError):
        return None
    return found.get("n") if isinstance(found, dict) and found.get("n") in N_CHOICES else None


def registration_problems(retrospective: Path | None, repo: Path | None = None) -> list[str]:
    """Pod 2 starts only after the registration (its order, steps 3-4): the registration at the
    checkout's commit says ``Status: `registered``` and names the sha256 of the committed fixation
    record and of the retrospective record (``--retrospective``, of that fixation record)."""
    repo, problems = REPO if repo is None else repo, []
    try:
        text = (repo / REGISTRATION).read_text(encoding="utf-8")
        fixation = sha256_of(repo / FIXATION)
    except OSError as exc:
        return [f"the registration or the fixation record does not read ({type(exc).__name__})"]
    if not REGISTERED.search(text):
        problems.append(f"{REGISTRATION} is not registered (Status: `registered`)")
    if fixation not in text:
        problems.append(f"{REGISTRATION} does not name the sha256 of {FIXATION}")
    if retrospective is None:
        return [*problems, "pod2 needs --retrospective, the record the registration names"]
    try:
        record = json.loads(Path(retrospective).read_text(encoding="utf-8"))
        digest = sha256_of(Path(retrospective))
    except (OSError, ValueError) as exc:
        return [*problems, f"the retrospective record does not read ({type(exc).__name__})"]
    if not isinstance(record, dict) or record.get("kind") != "study5-retrospective":
        problems.append("--retrospective is not study 5's retrospective record")
    elif (record.get("fixation") or {}).get("sha256") != fixation:
        problems.append("the retrospective record is not of the committed fixation record")
    if digest not in text:
        problems.append(f"{REGISTRATION} does not name the retrospective record's sha256")
    return problems


def gate_problems(plan: Plan, engineering: Path | None, checks: Path | None) -> list[str]:
    """Pod 2 starts only on pod 1's gates (its records, copied to the Pod, by path): an equal
    engineering check of reruns under the plan's GPU code identity, and passing surrogate checks
    of the plan's surrogate (``gate_problem``; the same surrogate sha256)."""
    problems = []
    for kind, path, option in (("engineering", engineering, "--engineering-check"),
                               ("surrogate", checks, "--surrogate-checks")):  # fmt: skip
        if path is None:
            problems.append(f"pod2 needs {option}, pod 1's {kind} record")
            continue
        found = gate_problem(kind, Path(path), plan)
        if found is not None:
            problems.append(f"pod 1's {kind} record does not stand: {found}")
        elif kind == "surrogate":
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            if record.get("surrogate_sha256") != plan.surrogate_sha256:
                problems.append(f"pod 1's surrogate checks are not of {plan.surrogate}")
    return problems


POD3_PREFIX = "pod3-"  # pod 3's archive namespace: never pod 1's or pod 2's


def _earlier_attempt(folder: Path, plan: Plan) -> bool:
    """A campaign folder an earlier attempt of the same pod and archive id wrote (its
    ``plan.json``): a stopped campaign goes on from a new --out (module docstring, "Resumable")."""
    try:
        stored = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(stored, dict) and stored.get("pod") == plan.pod
            and stored.get("archive_id") == plan.archive_id)  # fmt: skip


def own_root_problems(plan: Plan) -> list[str]:
    """Pod 3 writes only into a runs root and an archive namespace of its own, so that it can
    neither move nor reuse a registered run or record: its archive id starts with ``pod3-``, and
    the runs root holds nothing of another archive (each a refusal before anything is written):
    no records folder of another id under ``study5_records/``, no launcher journal of another
    campaign under ``launch/``, no run under ``study5/``, ``study3/`` or ``study5_floor/`` whose
    ``meta.json`` names another archive (or that has none that reads), no scan or ladder run of
    another id, no
    campaign folder under ``campaigns/`` but the plan's own --out and those of earlier attempts
    of the same pod and archive id (``_earlier_attempt``)."""
    problems, own = [], f"study5-{plan.archive_id}-"
    prefix = POD4_PREFIX if plan.pod == "pod4" else POD3_PREFIX
    if not plan.archive_id.startswith(prefix):
        problems.append(f"--archive-id {plan.archive_id}: {plan.pod}'s archive id starts with "
                        f"{prefix!r}, never another pod's")  # fmt: skip
    runs = plan.runs

    def children(name: str) -> list[Path]:
        folder = runs / name
        return sorted(folder.iterdir()) if folder.is_dir() else []

    other = [p.name for p in children("study5_records") if p.name != plan.archive_id]
    other += [p.name for p in children("launch") if not p.name.startswith(own)]
    for experiment in ("study5", "study3", FLOOR_EXPERIMENT):
        for run_dir in children(experiment):
            try:
                meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
                archive_id = meta["campaign"]["header"]["archive_id"]
            except (OSError, ValueError, KeyError, TypeError):
                archive_id = None
            if archive_id != plan.archive_id:
                other.append(f"{experiment}/{run_dir.name}")
    for experiment in (SCAN_EXPERIMENT, "study5_ladder", VECTOR_EXPERIMENT):
        other += [f"{experiment}/{p.name}" for p in children(experiment)
                  if not p.name.startswith(f"s5-{plan.archive_id}-")]  # fmt: skip
    out = plan.out.resolve()
    other += [f"campaigns/{p.name}" for p in children("campaigns")
              if p.resolve() != out and not _earlier_attempt(p, plan)]  # fmt: skip
    if other:
        problems.append(f"RUNS_DIR {runs} is not {plan.pod}'s own: it holds {len(other)} entries "
                        f"of another archive or campaign (first: {other[0]})")  # fmt: skip
    return problems


def preflight(plan: Plan, args: argparse.Namespace | None = None) -> list[str]:
    """What forbids the start: the checkout, its GPU code (pod 3's: extension 1's record), the
    bucket, the key, the folder; for pod 2 also the registration, the committed seeds and pod 1's
    gates (``registration_problems``, ``seeds_problems``, ``gate_problems``); for pod 3 the
    registration holding extension 1 and a runs root and archive id of its own
    (``own_root_problems``)."""
    problems = []
    state = git_state(REPO)
    if state.get("dirty") is not False or state.get("commit") != plan.commit:
        problems.append("the checkout is not clean at the plan's commit")
    identity = {"pod3": EXTENSION_IDENTITY, "pod4": EXT2_IDENTITY}.get(plan.pod, IDENTITY_FILE)
    recorded = json.loads((REPO / identity).read_text(encoding="utf-8"))
    if plan.gpu_code != recorded.get("gpu_code_sha256"):
        problems.append(f"HEAD's GPU code is not the one {identity} records")
    heading = {"pod3": (EXTENSION_HEADING, "extension 1"), "pod4": (EXT2_HEADING, "extension 2")}
    if plan.pod in heading and heading[plan.pod][0] not in (REPO / REGISTRATION).read_text(
        encoding="utf-8"
    ):
        problems.append(f"{REGISTRATION} does not hold {heading[plan.pod][1]}")
    if plan.pod in ("pod3", "pod4"):
        problems += own_root_problems(plan)
    if not os.environ.get("ARTIFACT_REMOTE", "").startswith("gs://"):
        problems.append("ARTIFACT_REMOTE is not a gs:// path")
    if not Path(os.environ.get("ARTIFACT_GCS_KEY", "/nonexistent")).is_file():
        problems.append("ARTIFACT_GCS_KEY is not a readable key file")
    if plan.out.resolve().is_relative_to(REPO.resolve()):
        problems.append("--out lies inside the checkout")
    if plan.pod == "pod2":
        given = vars(args) if args is not None else {}
        problems += registration_problems(given.get("retrospective"))
        problems += seeds_problems(plan)
        problems += gate_problems(plan, given.get("engineering_check"),
                                  given.get("surrogate_checks"))  # fmt: skip
    return problems


def prepare(plan: Plan) -> None:
    """Write the plan and its launcher campaigns into --out, or check those already there."""
    stored = plan.out / "plan.json"
    if stored.exists():
        if json.loads(stored.read_text()) != plan.to_json():
            raise StepError(
                f"{stored} is another plan: a restart takes the same arguments and commit"
            )
    elif plan.out.exists() and any(plan.out.iterdir()):
        raise StepError(f"{plan.out} holds files but no plan")
    for name, text in plan.campaigns.items():
        path = plan.out / "campaigns" / f"{name}.yaml"
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise StepError(f"{path.name} is not the plan's campaign")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if not stored.exists():
        write_json(stored, plan.to_json())


def show(plan: Plan) -> None:
    for block in plan.blocks:
        print(f"block {block.name}: {len(block.steps)} steps")
        for step in block.steps:
            print(f"  [{step.on_fail}] {step.name}: {' '.join(step.argv)}")


def install_sigterm() -> None:
    """SIGTERM to the driver raises SystemExit: the running step's process group is killed (its
    start stays without an end: a restart runs it again), and the close, its deadline last, runs
    in ``execute``'s ``finally``."""

    def stop(signum, frame):
        raise SystemExit(STOPPED)

    signal.signal(signal.SIGTERM, stop)


def cmd_pod(args) -> int:
    commit = git_state(REPO).get("commit") or "unknown"
    try:
        gpu_code = code_identity.gpu_code_identity(commit)["sha256"]
    except Exception:  # no commit: the preflight refuses
        gpu_code = "unknown"
    where = {
        "out": args.out.absolute(),
        "runs": runs_root(),
        "data": data_root(),
        "archive_id": args.archive_id,
        "commit": commit,
        "gpu_code": gpu_code,
    }
    if args.kind == "pod1":
        plan = pod1_plan(**where, surrogate=args.surrogate)
    elif args.kind == "pod3":
        plan = pod3_plan(**where, surrogate=args.surrogate)
    elif args.kind == "pod4":
        plan = pod4_plan(**where, checkpoints=args.checkpoints, surrogate=args.surrogate)
    else:
        n = args.n if args.n is not None else committed_n() or N_CHOICES[0]
        plan = pod2_plan(**where, checkpoints=args.checkpoints, n=n, surrogate=args.surrogate)
    if args.dry_run:
        show(plan)
        return COMPLETED
    problems = preflight(plan, args)
    if problems:
        for problem in problems:
            print(f"refused: {problem}")
        return ERROR
    try:
        prepare(plan)
    except Exception:
        # nothing runs: the watcher stops the Pod after the close's window
        subprocess.run(
            [sys.executable, str(REPO / SELF), "deadline", "--minutes", str(END_MINUTES)],
            cwd=REPO,
            timeout=timeout_of(MINUTES["small"]),
        )
        raise
    env = {
        **os.environ,
        "RUNS_DIR": str(plan.runs),
        "DATA_DIR": str(plan.data),
        "PYTHONUNBUFFERED": "1",
    }
    install_sigterm()
    code = execute(plan, python=[sys.executable], cwd=REPO, env=env)
    print(f"== {_now()} {plan.pod} {'completed' if code == COMPLETED else 'stopped'}")
    return code


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("pod1", "pod2", "pod3", "pod4"):
        p = sub.add_parser(kind)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--archive-id", required=True)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--surrogate", default=SURROGATE)
        if kind == "pod4":  # the volume's folder for checkpoints and vector files (absolute)
            p.add_argument("--checkpoints", type=Path, required=True)
        if kind == "pod2":
            p.add_argument("--n", type=int, default=None, choices=N_CHOICES)  # SEEDS_FILE's
            p.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
            p.add_argument("--engineering-check", type=Path)
            p.add_argument("--surrogate-checks", type=Path)
            p.add_argument("--retrospective", type=Path)
    p = sub.add_parser("deadline")
    p.add_argument("--minutes", type=int, required=True)
    p.add_argument("--reserve", type=int, default=0)  # minutes kept free before the cap
    p.add_argument("--needs", type=float)  # minutes the window must hold (the close's export)
    p.add_argument("--admit", type=int, default=0)  # pod 3: minutes the work and close need
    p.add_argument("--record", type=Path)
    p = sub.add_parser("fetch-streams")
    p.add_argument("--inventory", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p = sub.add_parser("fetch-run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dest", type=Path, required=True)
    # the run folders' place relative to ARTIFACT_REMOTE: study 3's, or pod 2's (pod 3)
    p.add_argument("--prefix", default=f"{STUDY3_RUNS_REMOTE}/study3")
    p = sub.add_parser("eng-launch")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign-file", type=Path, required=True)
    p.add_argument("--gpu-code", required=True)
    p = sub.add_parser("settle")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p = sub.add_parser("json-snapshot")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("purge")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--inventory", type=Path, action="append", required=True)
    p.add_argument("--consumer", action="append", required=True)
    p = sub.add_parser("drop-checkpoints")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--inventory", type=Path, action="append", required=True)
    p.add_argument("--consumer", action="append", required=True)
    p.add_argument("--endpoint", action="store_true")
    p = sub.add_parser("with-runs")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--aside")
    p.add_argument("--unless")
    p.add_argument("--restore-label")  # the η_l run whose checkpoint must be on the disk
    p.add_argument("--restore-dir", type=Path)
    p.add_argument("--restore-remote")
    p.add_argument("--restore-inventory", type=Path)
    p.add_argument("--restore-inventories-remote")
    p.add_argument("--aside-dir")  # extension 2: an incomplete output folder moved aside first
    p.add_argument("--only-if")  # extension 2: run only when this input exists
    p.add_argument("--else-record")  # ... else write this record of its absence
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("archive")
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inventories-remote", required=True)
    p.add_argument("--path", action="append")
    p.add_argument("--mutable-path", action="append")
    p.add_argument("--run-dir", action="append")  # DIR=RECORD: written once when RECORD exists
    p.add_argument("--attempt-remote")
    p.add_argument("--campaign", action="append")
    p.add_argument("--runs", type=Path)
    p.add_argument("--streams", type=int)
    p = sub.add_parser("snapshot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--unique", action="store_true")
    p.add_argument("--include", action="append")
    p.add_argument("--remote", required=True)
    p.add_argument("--inventories-remote", required=True)
    p = sub.add_parser("receipt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block", required=True)
    p.add_argument("inventories", nargs="*")
    p = sub.add_parser("check-checkpoints")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--max", type=int, required=True)
    p = sub.add_parser("checkpoint")
    for name in ("--runs", "--dir", "--inventory"):
        p.add_argument(name, type=Path, required=True)
    for name in ("--campaign", "--label", "--remote", "--inventories-remote"):
        p.add_argument(name, required=True)
    p.add_argument("--keep", action="store_true")  # archived and verified, not removed
    p = sub.add_parser("fetch-endpoint")  # extension 2: an archived endpoint, for its distance
    for name in ("--runs", "--hashed", "--dest"):
        p.add_argument(name, type=Path, required=True)
    for name in ("--campaign", "--label", "--expected", "--run-id", "--sha256", "--remote",
                 "--inventories-remote"):  # fmt: skip
        p.add_argument(name, required=True)
    p = sub.add_parser("remove-fetched")
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--record", type=Path, required=True)  # the distance record that read it
    p = sub.add_parser("restore-runs")  # a Pod's runs from the bucket, for the analysis
    p.add_argument("--archive-id", required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("completion")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--campaign", action="append", required=True)
    p = sub.add_parser("export-checkpoints")
    for name in ("--runs", "--dir", "--inventories", "--record"):
        p.add_argument(name, type=Path, required=True)
    for name in ("--remote", "--inventories-remote"):
        p.add_argument(name, required=True)
    p.add_argument("--run", action="append", default=[])
    p = sub.add_parser("predictions")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p = sub.add_parser("publish")
    for name in ("--record", "--receipt", "--inventory"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--remote", required=True)
    p = sub.add_parser("check-published")
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--gpu-code", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser


COMMANDS = {
    "pod1": cmd_pod,
    "pod2": cmd_pod,
    "pod3": cmd_pod,
    "pod4": cmd_pod,
    "deadline": cmd_deadline,
    "settle": cmd_settle,
    "purge": cmd_purge,
    "json-snapshot": cmd_json_snapshot,
    "drop-checkpoints": cmd_drop_checkpoints,
    "fetch-streams": cmd_fetch_streams,
    "fetch-run": cmd_fetch_run,
    "with-runs": cmd_with_runs,
    "eng-launch": cmd_eng_launch,
    "archive": cmd_archive,
    "snapshot": cmd_snapshot,
    "receipt": cmd_receipt,
    "check-checkpoints": cmd_check_checkpoints,
    "checkpoint": cmd_checkpoint,
    "fetch-endpoint": cmd_fetch_endpoint,
    "remove-fetched": cmd_remove_fetched,
    "completion": cmd_completion,
    "restore-runs": cmd_restore_runs,
    "export-checkpoints": cmd_export_checkpoints,
    "predictions": cmd_predictions,
    "publish": cmd_publish,
    "check-published": cmd_check_published,
}


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        return COMMANDS[args.kind](args)
    except (StepError, ValueError, OSError) as exc:
        print(f"study5_campaigns.py {args.kind}: {exc}")
        return ERROR


if __name__ == "__main__":
    sys.exit(main())
