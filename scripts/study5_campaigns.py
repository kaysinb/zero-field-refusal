"""Study 5's two Pod campaigns, each one detached driver of blocks chained in order
(docs/study5-preregistration.md, "Seeds and order", "The confirmatory test"; docs/operations.md,
"Study 5 campaigns").

    python scripts/study5_campaigns.py pod1 --out DIR --archive-id ID [--dry-run]
    python scripts/study5_campaigns.py pod2 --out DIR --archive-id ID [--n 40|24]
        [--checkpoints DIR] [--dry-run]

On a Pod: ``project-run "$SRC" python scripts/study5_campaigns.py pod1 ...``, detached. The other
commands (``deadline``, ``fetch-streams``, ``fetch-run``, ``with-runs``, ``archive``,
``snapshot``, ``receipt``, ``check-checkpoints``, ``checkpoint``, ``predictions``, ``publish``,
``check-published``) are the steps the drivers run; each prints numbers, ids, paths and hashes
only, and exits 0 on success, 1 otherwise.

**The plan.** A campaign is a plan: blocks of steps, each step a command line (its argv) with
what its failure does (``stop`` the campaign, ``record`` it and go on, or ``skip-group``: skip the
rest of its seed's steps), built as data from the arguments, the checkout's commit and study 3's
records alone (``pod1_plan``, ``pod2_plan``); nothing is run to build it. ``--dry-run`` prints it
and writes nothing. Every block begins by setting the Pod's deadline file (``deadline``: now +
2 × the estimated minutes of its steps outside the seed groups + 60, through
``scripts/pod_deadline_watch.py --set``; ``ZFR_DEADLINE_EPOCH`` stays its cap), and so does every
seed group (2 × the group's minutes + 60), so that a dead driver or a hung step leaves the Pod
idle one group's window at most. The minutes are study 3's measured unit costs where one exists
(``MINUTES``). Every step runs in its own session with a timeout (3 × its minutes + 20): past it
its process group is killed and the step has failed, under its rule. A block ends by archiving
its records to the bucket, verified
(``scripts/archive.py``: every object's size and MD5; each inventory archived too and read back),
a snapshot of the campaign's journal and logs, and the block's receipt. A last block, ``close``,
run at every end, after a stop and on any exception of the driver (its deadline in a
``finally``), archives the plan, the launcher campaigns and the receipts and sets the deadline 15
minutes ahead, so that the watcher stops the Pod and nothing is left running.

**Pod 1** (order steps 1-2 of the registration): ``engineering``: seed 10403's streams fetched
from the bucket; study 3's archived records of its F runs at both rates (the run ids from study
3's analysis freeze, ``docs/study3-analysis-freeze-c1.json``; ``judge/`` and ``wandb/`` are not
fetched); the two runs rerun with this checkout under study 3's own config
(experiments/study3, ``study3.role=confirmation``: the config hashes of the archived runs); and
``scripts/study5_engineering_check.py``, archived against rerun. Any failure here stops the
campaign before anything else runs. ``theta0``: θ₀'s evaluation on the 521 prompts
(experiments/study5_theta0, the launcher's ``theta0`` entry). ``surrogate``:
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
per seed the F run at η_l (experiments/study5_scan: the line scan and the endpoint checkpoint),
its checkpoint archived, then the F run at η_u (experiments/study5, ``study5.role=fresh``), each
in a launcher campaign of its own. Each endpoint checkpoint is archived, verified and then removed
from the container disk by
``remove_checkpoint``, which refuses any path but that run's own ``<dir>/<run id>.pt``; a seed's
runs start only while at most one checkpoint is on the disk (``check-checkpoints``), so it holds
at most two. ``controls``: seeds 50000-50005 (those of the campaign's seeds): the two crossed
controls at η_l (experiments/study5_control_order; experiments/study5_control_v0 with v₀ from
seed 50000 + ((s - 50000 + 1) mod 6)), then the ladder's rungs ``isotropic``, ``linear_no_d`` and
``linear`` (``scripts/study5_ladder.py``, on the seed's fresh F run at η_l; ``linear_shuffled_d``
runs only on a later decision). ``check-published`` also requires the record to hold exactly the
plan's seeds in order, its n and the plan's GPU code identity, and its sha256 the published
copy's (the receipt's and the sidecar's); its commit is information only. A mismatch stops the
campaign
before any F run. No labeling: labels are opened only after every fresh run has
ended, in a separate step.

**Failures.** Every run goes through ``scripts/launch.py FILE --max-attempts 1`` (one attempt; a
rerun is a recovery round of the run's own, zero_field_refusal.launch, "Recovery rounds"). A step
that fails or exits non-zero is recorded in the journal and the campaign goes on to the next seed
where the step is per seed, except where the plan stops: the engineering check, the surrogate
checks, the fetch of study 3's streams, the predictions record and its publication, the checkpoint
count, every archive and every deadline. The checkpoint step refuses a checkpoint whose run
records no sha256 for it (the file stays, and the next seed's count stops the campaign). A failed
step is not run again by a restart; a stop
recorded stops again.

**Resumable.** The driver keeps ``journal.jsonl`` in --out (a start and an end line per step;
each attempt's output in ``logs/``). Started again with the same arguments from the same commit
(the plan must equal ``plan.json``), it skips a block whose receipt exists (its archive verified)
and, inside a block, every step that ended, every launch whose runs all completed (by the
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
``study5/<ID>/runs``, ``checkpoints``, ``predictions``, ``campaign`` and ``inventories``, and the
fresh streams under ``data/`` as every stream. Before it starts, the driver checks the checkout
is clean and that its GPU code is the one ``docs/study5-code-identity.json`` records.
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
RUNGS = ("isotropic", "linear_no_d", "linear")  # linear_shuffled_d: only on a later decision
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
RECEIPT_NAME = "predictions-receipt.json"  # beside the predictions record
CLOSE = "close"
END_MINUTES = 15  # the deadline after the close: the watcher stops the Pod
MARGIN = 60
COMPLETED, ERROR, STOPPED = 0, 1, 3
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,23}")
# Minutes per step. Measured on study 3's L40S Pod where a unit cost exists: an F run 6.9; an
# evaluation without training 3.1; a K-only point 1.5; five scan points 15.5; a stream 91.7 s per
# 1000 responses (1500 per seed: 500 training, 1000 for v₀); the predictor 1.7 per seed; a
# ladder rung three K-only points and one evaluation, 7.6; an endpoint checkpoint's save 0.5.
# Assumed where none is measured, as named. They set each block's and each seed group's deadline
# (2 × the sum + MARGIN) and each step's timeout (timeout_of).
F_RUN, EVALUATION, K_POINT, SCAN, CKPT_SAVE = 6.9, 3.1, 1.5, 15.5, 0.5
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
    "fresh-lower": F_RUN + SCAN + CKPT_SAVE,
    "fresh-upper": F_RUN,
    "checkpoint": 4.0,  # assumed: a 5.24 GB checkpoint hashed and uploaded at about 25 MB/s
    "controls": 2 * F_RUN,
    "rung": 3 * K_POINT + EVALUATION,
    "archive": 1.0,  # assumed: a seed's run directories and records, a few MB
    "small": 0.5,
}
TIMEOUT_FACTOR, TIMEOUT_SLACK = 3, 20  # a step's timeout: 3 × its minutes + 20, in minutes


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
    gate: str = ""  # engineering | surrogate | predictions: its record kept only by its verdict
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


def fresh_args(seed: int) -> tuple[str, ...]:
    """A fresh seed's F run at η_l (its scan; its checkpoint's directory is not in its config)."""
    return ("experiments/study5_scan", rate(ETA_L), f"seed={seed}")


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
        self.campaigns: dict[str, str] = {}
        self.blocks: list[Block] = []
        self.inventories: list[str] = []
        self.current: list[Step] = []

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
    ) -> Step:
        """An archive step; ``kind`` ``archive`` runs also after a stop and in a skipped group
        (it archives what exists), ``run`` only when its group goes on (a seed's streams)."""
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
            f"{self.base}/inventories",
        ]
        for path in paths:
            argv += ["--path", path]
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
    ) -> Step:
        return self.archive(
            name,
            root=self.runs,
            remote=f"{self.base}/runs",
            paths=paths,
            campaign=campaign,
            group=group,
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
                f"{self.base}/campaign",
                "--inventories-remote",
                f"{self.base}/inventories",
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
        start that reaches it: a restart rewrites the window."""
        return Step(
            name=name,
            argv=self.own("deadline", "--minutes", deadline_minutes(minutes)),
            on_fail="stop",
            kind="deadline",
            always=True,
            group=group,
            timeout=timeout_of(MINUTES["small"]),
        )

    def close(self) -> None:
        self.current = []
        self.step(
            f"{CLOSE}/archive",
            self.own(
                "snapshot",
                "--out",
                self.out,
                "--label",
                CLOSE,
                "--unique",
                "--include",
                "plan.json",
                "--include",
                "campaigns",
                "--include",
                "receipts",
                "--remote",
                f"{self.base}/campaign",
                "--inventories-remote",
                f"{self.base}/inventories",
            ),
            kind="archive",
            on_fail="record",
            always=True,
        )
        self.step(
            f"{CLOSE}/deadline",
            self.own("deadline", "--minutes", END_MINUTES),
            kind="deadline",
            on_fail="record",
            always=True,
        )
        self.blocks.append(Block(CLOSE, tuple(self.current)))

    def plan(self, commit: str, gpu_code: str, n: int | None) -> Plan:
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
) -> Plan:
    """Pod 1's plan (module docstring, "Pod 1")."""
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
    b.runs_archive("engineering/archive", paths=(b.rel(check),), campaign=eng)
    b.end("engineering")

    b.begin()  # θ₀'s evaluation on the 521 prompts
    cell = {
        "name": f"s5-theta0-s{FRESH_FIRST}",
        "config": "experiments/study5_theta0",
        "entry": "theta0",
        "overrides": [],
        "seeds": [FRESH_FIRST],
    }
    _, theta0 = b.launch("theta0", "theta0", [cell], minutes="theta0")
    b.runs_archive("theta0/archive", campaign=theta0)
    b.end("theta0")

    b.begin()  # the surrogate checks: any non-zero exit stops the campaign
    checks = b.record("surrogate-checks.json")
    b.step(
        "surrogate/checks",
        ("scripts/study5_surrogate_checks.py", "experiments/study5_theta0", "--out", checks),
        on_fail="stop",
        outputs=(str(checks),),
        minutes="surrogate",
        gate="surrogate",
    )
    b.runs_archive("surrogate/archive", paths=(b.rel(checks),))
    b.end("surrogate")

    b.begin()  # the retrospective predictor on study 3's 60 A-F seeds
    b.fetch_streams("retrospective/fetch-streams", seeds, "fetch-seeds")
    for seed in seeds:
        group = f"s{seed}"
        predict = b.record("predictions", "retro", f"s{seed}.json")
        features = b.record("features", "retro", f"s{seed}.json")
        b.step(
            f"retrospective/{group}/predict",
            ("scripts/study5_predict.py", *retro_args(seed), "--out", predict),
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
    return b.plan(commit, gpu_code, None)


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
) -> Plan:
    """Pod 2's plan (module docstring, "Pod 2")."""
    seeds = fresh_seeds(n)
    b = Builder("pod2", archive_id, out, runs, data, checkpoints)
    teacher = ("scripts/generate_teacher.py", "experiments/study5_scan", rate(ETA_L))

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
            ("scripts/study5_predict.py", *fresh_args(seed), "--out", predict),
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

    b.begin()  # the fresh F runs: η_l with its scan and checkpoint, then η_u
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
    for seed in seeds:
        group = f"s{seed}"
        lower = {
            "name": f"s5-fresh-scan-{ETA_L!r}-s{seed}",
            "config": "experiments/study5_scan",
            "overrides": [rate(ETA_L), f"tracking.endpoint_checkpoint={b.checkpoints}"],
            "seeds": [seed],
        }
        upper = {
            "name": f"s5-fresh-{ETA_U!r}-s{seed}",
            "config": "experiments/study5",
            "overrides": ["study5.role=fresh", rate(ETA_U)],
            "seeds": [seed],
        }
        b.step(
            f"fresh/{group}/check-checkpoints",
            b.own("check-checkpoints", "--dir", b.checkpoints, "--max", 1),
            on_fail="stop",
            group=group,
        )
        # η_l alone first: its checkpoint is archived and removed before η_u starts
        _, name = b.launch(
            f"fresh-scan-{group}", "fresh", [lower], minutes="fresh-lower", group=group
        )
        fresh[seed] = (name, label(lower))
        b.step(
            f"fresh/{group}/checkpoint",
            b.own(
                "checkpoint",
                "--runs",
                b.runs,
                "--campaign",
                name,
                "--label",
                label(lower),
                "--dir",
                b.checkpoints,
                "--inventory",
                b.out / "inventories" / f"checkpoint-{group}.tsv",
                "--remote",
                f"{b.base}/checkpoints",
                "--inventories-remote",
                f"{b.base}/inventories",
            ),
            group=group,
            minutes="checkpoint",
        )
        b.inventories.append(str(b.out / "inventories" / f"checkpoint-{group}.tsv"))  # if any
        _, upper_name = b.launch(
            f"fresh-upper-{group}", "fresh", [upper], minutes="fresh-upper", group=group
        )
        b.runs_archive(f"fresh/{group}/archive", campaign=(name, upper_name), group=group)
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
        ladder_paths = []
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
            ladder_paths += [b.rel(out_record), f"study5_ladder/{run_id}"]
        b.runs_archive(
            f"controls/{group}/archive", paths=ladder_paths, campaign=controls, group=group
        )
    b.end("controls")
    b.close()
    return b.plan(commit, gpu_code, n)


# ------------------------------------------------------------------ the executor


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
) -> str:
    """Run ``step`` unless its journal or its outputs say it is done; its outcome. A gate's
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
    code = _run([*python, *step.argv], cwd=cwd, env=env, log=log, timeout=step.timeout)
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
    whatever ends the blocks (an exception too): its deadline last, in a ``finally``. COMPLETED,
    or STOPPED when a stop rule fired."""
    journal, logs = plan.out / "journal.jsonl", plan.out / "logs"
    run = {"plan": plan, "journal": journal, "logs": logs, "python": python, "cwd": cwd, "env": env}
    close = next((b for b in plan.blocks if b.name == CLOSE), Block(CLOSE, ()))
    status: dict[str, str] = {}
    stopped = False
    try:
        status = read_journal(journal)
        append(journal, {"event": "driver", "pod": plan.pod, "commit": plan.commit})
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
                        [*python, *step.argv],
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
    if kind == "engineering":
        if record.get("equal") is not True:
            return "the engineering check found a difference"
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
    found = {f"launch/{campaign}.jsonl", f"launch/{campaign}"}
    for entry in journal_events(runs, campaign):
        if entry.get("event") == "start" and isinstance(entry.get("run_id"), str):
            experiment = entry.get("experiment")
            if isinstance(experiment, str):
                found.add(f"{safe_rel(experiment)}/{safe_rel(entry['run_id'])}")
    return sorted(p for p in found if (runs / p).exists())


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


def substituted(argv: list[str], runs: Path, campaign: str) -> list[str]:
    """``{run:LABEL}`` replaced by that label's completed run directory."""
    out = []
    for arg in argv:
        found = re.fullmatch(r"\{run:(.+)\}", arg)
        out.append(str(completed_run(runs, campaign, found.group(1))) if found else arg)
    return out


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
    root: Path, remote: str, paths: list[str], inventory: Path, inventories_remote: str
) -> int:
    """Archive ``paths`` under ``root`` to ``ARTIFACT_REMOTE/<remote>``, verified, then the
    inventory itself to ``<inventories_remote>``, read back; the inventory is in place only once
    both hold. The number of files archived."""
    base = artifact_remote()
    present = [p for p in dict.fromkeys(safe_rel(p) for p in paths) if (root / p).exists()]
    staging = inventory.parent / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / inventory.name
    staged.unlink(missing_ok=True)
    files = 0
    if present:
        result = archive(staged, f"{base}/{remote}", root, present)
        if result["mismatched"]:
            raise StepError(f"{result['mismatched']} of {result['files']} objects do not match")
        files = result["files"]
    else:
        staged.write_text("path\tsize\tsha256\tmd5_b64\turl\tstatus\n", encoding="utf-8")
    copy = archive(
        staging / f"{inventory.stem}.self.tsv",
        f"{base}/{inventories_remote}",
        staging,
        [inventory.name],
    )
    if copy["mismatched"]:
        raise StepError(f"the inventory {inventory.name} does not match its copy")
    readback(f"{base}/{inventories_remote}/{inventory.name}", sha256_of(staged))
    os.replace(staged, inventory)
    (staging / f"{inventory.stem}.self.tsv").unlink(missing_ok=True)
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


def cmd_deadline(args) -> int:
    epoch = int(time.time()) + 60 * args.minutes
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "pod_deadline_watch.py"), "--set", str(epoch)]
    )
    print(f"deadline {datetime.fromtimestamp(epoch, UTC).isoformat()} exit {done.returncode}")
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
    prefix = f"{base}/{STUDY3_RUNS_REMOTE}/study3/{run_id}"
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
    if args.aside and not (args.unless and Path(args.unless).exists()):
        moved = move_aside(args.runs, args.aside)
        if moved is not None:
            print(f"moved aside {args.aside} to {moved.relative_to(args.runs)}")
    argv = substituted(command, args.runs, args.campaign)
    return subprocess.run([sys.executable, str(REPO / argv[0]), *argv[1:]], cwd=REPO).returncode


def cmd_archive(args) -> int:
    paths = list(args.path or [])
    for campaign in args.campaign or []:
        paths += campaign_paths(args.runs, campaign)
    if args.streams is not None:
        paths += stream_dirs(args.root, args.streams)
    if args.inventory.exists():
        print(f"kept {args.inventory.name}")
        return 0
    files = archive_paths(args.root, args.remote, paths, args.inventory, args.inventories_remote)
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


def cmd_checkpoint(args) -> int:
    """The η_l run's endpoint checkpoint archived, verified, then removed from the disk. It is
    refused while the run's summary records no sha256 for it (the file stays). A checkpoint no
    longer on the disk must be in the bucket when the run recorded one."""
    run_id = latest_run_id(args.runs, args.campaign, args.label)
    if run_id is None:
        print(f"no run of {args.label}: no checkpoint")
        return 0
    path = args.dir / f"{safe_rel(run_id)}.pt"
    summary = args.runs / safe_rel(_start_of(args.runs, args.campaign, run_id)["experiment"])
    summary = summary / run_id / "summary.json"
    record = None
    if summary.is_file():
        record = json.loads(summary.read_text(encoding="utf-8")).get("endpoint_checkpoint")
    recorded = record.get("sha256") if isinstance(record, dict) else None
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
        archive_paths(args.dir, args.remote, [path.name], args.inventory, args.inventories_remote)
    remove_checkpoint(path, args.dir, digest, read_inventory(args.inventory))
    print(f"checkpoint {run_id} sha256 {digest} archived, verified and removed")
    return 0


def archived_earlier(args, name: str, recorded: str) -> bool:
    """Whether the checkpoint ``name`` is in the bucket with the sha256 its run recorded: its
    object exists and the inventory of its archive (this --out's, else the copy archived
    beside the others) shows it verified with that sha256."""
    base = artifact_remote()
    if not json.loads(gcloud(["objects", "list", f"{base}/{args.remote}/{name}", "--format=json"])):
        return False
    inventory = args.inventory
    if not inventory.is_file():
        inventory = args.inventory.parent / ".staging" / f"remote-{args.inventory.name}"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        try:
            gcloud(
                [
                    "cp",
                    "--quiet",
                    f"{base}/{args.inventories_remote}/{args.inventory.name}",
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


# ------------------------------------------------------------------ the drivers


def preflight(plan: Plan) -> list[str]:
    """What forbids the start: the checkout, its GPU code, the bucket, the key, the folder."""
    problems = []
    state = git_state(REPO)
    if state.get("dirty") is not False or state.get("commit") != plan.commit:
        problems.append("the checkout is not clean at the plan's commit")
    recorded = json.loads((REPO / IDENTITY_FILE).read_text(encoding="utf-8"))
    if plan.gpu_code != recorded.get("gpu_code_sha256"):
        problems.append(f"HEAD's GPU code is not the one {IDENTITY_FILE} records")
    if not os.environ.get("ARTIFACT_REMOTE", "").startswith("gs://"):
        problems.append("ARTIFACT_REMOTE is not a gs:// path")
    if not Path(os.environ.get("ARTIFACT_GCS_KEY", "/nonexistent")).is_file():
        problems.append("ARTIFACT_GCS_KEY is not a readable key file")
    if plan.out.resolve().is_relative_to(REPO.resolve()):
        problems.append("--out lies inside the checkout")
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
        plan = pod1_plan(**where)
    else:
        plan = pod2_plan(**where, checkpoints=args.checkpoints, n=args.n)
    if args.dry_run:
        show(plan)
        return COMPLETED
    problems = preflight(plan)
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
    for kind in ("pod1", "pod2"):
        p = sub.add_parser(kind)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--archive-id", required=True)
        p.add_argument("--dry-run", action="store_true")
        if kind == "pod2":
            p.add_argument("--n", type=int, default=40, choices=N_CHOICES)
            p.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    p = sub.add_parser("deadline")
    p.add_argument("--minutes", type=int, required=True)
    p = sub.add_parser("fetch-streams")
    p.add_argument("--inventory", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p = sub.add_parser("fetch-run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("eng-launch")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign-file", type=Path, required=True)
    p.add_argument("--gpu-code", required=True)
    p = sub.add_parser("with-runs")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--aside")
    p.add_argument("--unless")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("archive")
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inventories-remote", required=True)
    p.add_argument("--path", action="append")
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
    "deadline": cmd_deadline,
    "fetch-streams": cmd_fetch_streams,
    "fetch-run": cmd_fetch_run,
    "with-runs": cmd_with_runs,
    "eng-launch": cmd_eng_launch,
    "archive": cmd_archive,
    "snapshot": cmd_snapshot,
    "receipt": cmd_receipt,
    "check-checkpoints": cmd_check_checkpoints,
    "checkpoint": cmd_checkpoint,
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
