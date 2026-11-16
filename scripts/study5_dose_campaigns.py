"""Study 5's dose experiment on two Pods (docs/study5-dose-registration.md; docs/operations.md,
"Study 5, the dose experiment"): on each Pod one detached driver of two plans in order, built and
run by ``scripts/study5_campaigns.py``'s machinery (its ``Builder``, ``execute``, deadlines,
archives, receipts, restarts and close; that module's docstring).

    python scripts/study5_dose_campaigns.py run --pod a|b --out DIR --archive-id dose-a-...
        --other-archive-id dose-b-... --checkpoints DIR [--dry-run [--grid FILE]]
    python scripts/study5_dose_campaigns.py adamw --out DIR --archive-id adamw-... \
        --checkpoints DIR [--dry-run]

``adamw`` runs the order repeat under AdamW (docs/study5-adamw-registration.md) on a Pod of its
own, one plan (``adamw_plan``) with the dose Pods' engineering check first.

The two plans' campaign folders are DIR-pilot and DIR-main, siblings (``RUNS_DIR/campaigns/
study5-dose-a-pilot``: a runs root's campaign folders are each a plan's own or an earlier
attempt's of the same Pod and archive id).

On a Pod: ``project-run "$SRC" python scripts/study5_dose_campaigns.py run ...``, detached. The
other commands (``table``, ``publish``, ``grid``, ``check-grid``) are steps the plans run; each
prints numbers, ids, paths and hashes only and exits 0 on success, 1 otherwise.

**The seeds** are those of ``docs/study5-dose-seeds.json`` (``zero_field_refusal.dose``): Pod A
takes its eight with the pilot seed of the smallest K_benign at η_l, Pod B its eight with the
largest. ``--archive-id`` starts with ``dose-a-`` or ``dose-b-`` by the Pod; ``--other-archive-id``
is the other Pod's.

**The pilot plan** (``pilot_plan``, campaign folder ``DIR-pilot``):

- ``engineering``: pod 2's order control of 50000 at η_l rerun under this checkout and compared by
  ``scripts/study5_engineering_check.py`` under extension 2's numerical equivalence (pods 3 and 4
  and the top-up did the same); any failure stops before anything else.
- ``fetch``: the streams of the Pod's eight seeds from pod 2's archive (``fetch-streams``).
- ``pilot``: the pilot seed's nine runs, three early levels × three late candidates
  (``experiments/study5_dose``), each saving its endpoint checkpoint; then ``table``: per run its
  end, its 25 updates and its K_benign at step 25 (no label, no harmful number), written as the
  Pod's pilot table and ``publish``-ed to the bucket under ``study5/<ID>/pilot/`` (read back).
- ``grid``: ``grid`` waits at most ``WAIT`` minutes for the other Pod's published table, checks
  both (their Pods, pilot seeds, nine runs each and one GPU code), and applies the registered rule
  (``dose.grid_choice``) to the 18 runs. The record holds both tables whole and the choice, so
  either Pod's record is the other's choice from the same inputs. A missing table stops the plan.

**The main plan** (``main_plan``, campaign folder ``DIR-main``), built from that grid record
(``check-grid`` first: the record's sha256 is the one the plan was built from, and its choice is
the rule's on its tables):

- ``pilot-scans``: the scan at a = −0.5 of the pilot seed's three runs at L_lo (``scripts/
  study5_scan.py --points -0.5``); then every pilot checkpoint removed, each once the records that
  read it (the scans; for the runs no scan reads, the grid record) are in a verified archive
  (``drop-checkpoints``).
- ``main``: per other seed, one launcher campaign of six runs (three early levels × L_lo, L_hi;
  the L_lo runs save their endpoint checkpoint), the scan at a = −0.5 of its three L_lo runs, the
  seed's archive, then its checkpoints removed as above. A seed's runs start only while at most
  ``LEFT_MAX`` checkpoints that a failed scan or drop kept are on the disk
  (``check-checkpoints``).

No labeling here: the judge runs elsewhere, after the forecasts are hashed. A failed run or scan
is recorded and the plan goes on; every archive, the engineering check, the tables and the grid
stop it. Each block and seed group is admitted only when ``ADMIT_FACTOR`` × its minutes and the
close fit before the Pod's cap ``ZFR_DEADLINE_EPOCH`` (``deadline --admit``). ``run`` executes the
pilot plan and, only when it completed, the main plan; started again with the same arguments it
skips what is done in both (``scripts/study5_campaigns.py``, "Resumable").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study5_campaigns as c  # noqa: E402  (the driver's machinery)

from zero_field_refusal import adamw_orders, code_identity, dose  # noqa: E402
from zero_field_refusal.paths import data_root, runs_root  # noqa: E402
from zero_field_refusal.tracking import git_state  # noqa: E402

SELF = "scripts/study5_dose_campaigns.py"
IDENTITY = "docs/study5-dose-code-identity.json"
REGISTRATION = "docs/study5-dose-registration.md"
# The order repeat under AdamW (``adamw_plan``): its Pod, GPU code record and registration
ADAMW_POD = "adamw"
ADAMW_IDENTITY = "docs/study5-adamw-code-identity.json"
ADAMW_REGISTRATION = "docs/study5-adamw-registration.md"
CONFIG = "experiments/study5_dose"
WAIT = 45  # minutes a Pod waits for the other Pod's pilot table
POLL = 60  # seconds between two looks for it
POINT = -0.5  # the scan's one point
RUN, RUN_CKPT = c.F_RUN, c.F_RUN + c.CKPT_SAVE  # minutes of a dose run, with its checkpoint
SCAN = c.EVALUATION + c.SCAN_LOAD  # minutes of a one-point scan
TABLE, GRID = "pilot-table.json", "grid.json"
# checkpoints a seed's runs may find on the disk: those a failed scan or drop left (each kept,
# recorded); with a seed's three, at most nine, about 47 GB
LEFT_MAX = 6
TABLE_KIND, GRID_KIND = "study5-dose-pilot-table", "study5-dose-grid"
c.PREFIXES.update({f"dose-{pod}": f"dose-{pod}-" for pod in dose.PODS})  # own_root_problems
c.PREFIXES[ADAMW_POD] = "adamw-"


def other(pod: str) -> str:
    return {"a": "b", "b": "a"}[pod]


def seeds_record(repo: Path = c.REPO) -> dict[str, Any]:
    """``docs/study5-dose-seeds.json``, checked against the draw and the shares' rule."""
    record = json.loads((repo / dose.SEEDS_FILE).read_text(encoding="utf-8"))
    k = {int(s): v for s, v in record["k_benign_eta_l"].items()}
    if record["drawn_in_order"] != dose.draw_seeds() or record["pods"] != dose.assign(
        record["drawn_in_order"], k
    ):
        raise c.StepError(f"{dose.SEEDS_FILE} is not the draw and its shares")
    return record


def cell(seed: int, early: str, late: str, checkpoints: Path | None = None) -> dict[str, Any]:
    """One dose run: ``seed`` at the early level ``early`` and the late level ``late`` (names of
    ``dose.EARLY`` and ``dose.LATE_CANDIDATES``), with its endpoint checkpoint when given."""
    overrides = list(run_overrides(early, late))
    if checkpoints is not None:
        overrides.append(f"tracking.endpoint_checkpoint={checkpoints}")
    return {"name": f"s5-dose-e{early}-l{late}-s{seed}", "config": CONFIG,
            "overrides": overrides, "seeds": [seed]}  # fmt: skip


def run_overrides(early: str, late: str) -> tuple[str, ...]:
    return (c.rate(dose.late_rate(late)),
            f"train.rate_schedule.early_lr={dose.early_rate(early)!r}")  # fmt: skip


def run_args(seed: int, early: str, late: str) -> tuple[str, ...]:
    """The run's config and overrides, as the scan reads its config hash."""
    return (CONFIG, *run_overrides(early, late), f"seed={seed}")


def builder(pod: str, archive_id: str, out: Path, runs: Path, data: Path,
            checkpoints: Path) -> c.Builder:  # fmt: skip
    b = c.Builder(f"dose-{pod}", archive_id, out, runs, data, checkpoints)
    b.admit = True  # each block and seed group admitted only when it and the close fit the cap
    return b


def pilot_campaign(archive_id: str, seed: int) -> str:
    return f"study5-{archive_id}-pilot-s{seed}"


def engineering(b: c.Builder, gpu_code: str):
    """The ``engineering`` block, as pods 3 and 4 and the top-up ran it: pod 2's order control of
    50000 at η_l rerun and compared under extension 2's numerical equivalence; any failure stops.
    Returns the plan's stream fetcher (``fetch_streams(name, seed)``, a stop on failure)."""
    fetched = b.out / "fetched" / "study5"

    def fetch_streams(name: str, seed: int) -> None:
        b.step(name, b.own("fetch-streams", "--inventory", c.POD2_STREAMS.format(seed=seed),
                           "--data", b.data, "--work", b.out, "--seeds", seed),
               on_fail="stop", minutes="fetch-seed")  # fmt: skip

    b.begin()  # pod 2's order control of 50000 rerun; any failure stops
    first = c.CONTROL_SEEDS[0]
    fetch_streams("engineering/fetch-streams", first)
    b.step("engineering/fetch-run", b.own("fetch-run", "--run-id", c.POD2_ORDER_RUN, "--dest",
           fetched, "--prefix", c.POD2_RUNS), on_fail="stop",
           outputs=(str(fetched / c.POD2_ORDER_RUN / "meta.json"),), minutes="fetch-run")  # fmt: skip
    eng_cell = {"name": f"s5-order-{c.ETA_L!r}-s{first}", "config": c.CONTROL_ORDER,
                "overrides": [c.rate(c.ETA_L)], "seeds": [first]}  # fmt: skip
    eng_file = b.launcher("engineering", "engineering", [eng_cell])
    eng = Path(eng_file).stem
    b.step("engineering/launch", b.own("eng-launch", "--runs", b.runs, "--campaign-file",
           eng_file, "--gpu-code", gpu_code), kind="launch", on_fail="stop",
           minutes="fresh-upper")  # fmt: skip
    check = b.record("engineering-check-numerical.json")
    b.step("engineering/check", b.own("with-runs", "--runs", b.runs, "--campaign", eng, "--",
           "scripts/study5_engineering_check.py", "--archived", fetched / c.POD2_ORDER_RUN,
           "--rerun", f"{{run:{c.label(eng_cell)}}}", "--out", check, "--tolerance",
           c.EXT2_TOLERANCE["relative"]), on_fail="stop", outputs=(str(check),),
           minutes="engineering-check", gate="engineering")  # fmt: skip
    b.runs_archive("engineering/archive", mutable=(b.rel(check),), campaign=eng)
    b.end("engineering")
    return fetch_streams


def pilot_plan(*, pod: str, out: Path, runs: Path, data: Path, checkpoints: Path,
               archive_id: str, other_archive_id: str, commit: str, gpu_code: str,
               repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The pilot plan (module docstring)."""
    record = seeds_record(repo)
    share, rival = record["pods"][pod], record["pods"][other(pod)]
    pilot = share["pilot"]
    b = builder(pod, archive_id, out, runs, data, checkpoints)
    fetch_streams = engineering(b, gpu_code)

    b.begin()  # the Pod's seeds' streams, each fetched only if missing
    for seed in share["seeds"]:
        fetch_streams(f"fetch/s{seed}/streams", seed)
    b.end("fetch")

    b.begin()  # the pilot seed's nine runs and the Pod's pilot table, published
    cells = [cell(pilot, e, late, b.checkpoints) for e, _ in dose.EARLY
             for late, _ in dose.LATE_CANDIDATES]  # fmt: skip
    _, campaign = b.launch(f"pilot-s{pilot}", "pilot", cells, minutes=len(cells) * RUN_CKPT)
    assert campaign == pilot_campaign(archive_id, pilot)
    table = b.record("pilot", TABLE)
    argv = [SELF, "table", "--runs", b.runs, "--campaign", campaign, "--pod", pod, "--seed",
            pilot, "--gpu-code", gpu_code, "--out", table]  # fmt: skip
    for e, _ in dose.EARLY:
        for late, _ in dose.LATE_CANDIDATES:
            argv += ["--run", e, late, c.label(cell(pilot, e, late))]
    b.step("pilot/table", argv, on_fail="stop", outputs=(str(table),))
    inventory = b.out / "inventories" / "pilot-publish.tsv"
    b.step("pilot/publish", (SELF, "publish", "--file", table, "--remote",
           f"{b.base}/pilot", "--inventory", inventory), on_fail="stop",
           outputs=(str(inventory),), minutes="archive")  # fmt: skip
    b.inventories.append(str(inventory))
    b.runs_archive("pilot/archive", paths=(b.rel(table),), campaign=campaign)
    b.end("pilot")

    b.begin()  # the other Pod's table, then the registered rule on both
    grid = b.record(GRID)
    theirs = b.record("pilot", f"pilot-table-{other(pod)}.json")
    b.step("grid/choice", (SELF, "grid", "--own", table, "--other-remote",
           f"study5/{other_archive_id}/pilot/{TABLE}", "--fetched", theirs, "--wait", WAIT,
           "--pilot", f"{pod}={pilot}", "--pilot", f"{other(pod)}={rival['pilot']}",
           "--gpu-code", gpu_code, "--out", grid), on_fail="stop", outputs=(str(grid),),
           minutes=WAIT + 1)  # fmt: skip
    b.runs_archive("grid/archive", paths=(b.rel(theirs), b.rel(grid)))
    b.end("grid")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


def read_grid(path: Path) -> dict[str, Any]:
    """A grid record whose choice is the registered rule's on the tables it holds."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("kind") != GRID_KIND:
        raise c.StepError(f"{Path(path).name}: not a dose grid record")
    tables = record.get("tables") or {}
    if sorted(tables) != sorted(dose.PODS):
        raise c.StepError(f"{Path(path).name}: not both Pods' tables")
    rows = [row for pod in dose.PODS for row in tables[pod]["rows"]]
    seeds = [int(tables[pod]["seed"]) for pod in dose.PODS]
    if dose.grid_choice(rows, seeds) != record.get("grid"):
        raise c.StepError(f"{Path(path).name}: its choice is not the rule's on its tables")
    return record


def main_plan(*, pod: str, out: Path, runs: Path, data: Path, checkpoints: Path,
              archive_id: str, grid_path: Path, pilot_out: Path, commit: str, gpu_code: str,
              repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The main plan (module docstring), from the grid record at ``grid_path``; ``pilot_out``
    is the pilot plan's campaign folder (its grid archive's inventory)."""
    record = seeds_record(repo)
    share = record["pods"][pod]
    pilot = share["pilot"]
    grid = read_grid(grid_path)
    lo, hi = grid["grid"]["chosen"]["lo"], grid["grid"]["chosen"]["hi"]
    b = builder(pod, archive_id, out, runs, data, checkpoints)
    grid_rel = b.rel(Path(grid_path))
    grid_inventory = pilot_out / "inventories" / "grid--archive.tsv"
    early = [e for e, _ in dose.EARLY]

    def scan(block: str, group: str, campaign: str, seed: int, e: str) -> tuple[str, str]:
        record_path = b.record("scan", f"s{seed}-e{e}.json")
        run_id = f"s5-{b.id}-scan-e{e}-s{seed}"
        b.step(f"{block}/{group}/scan-e{e}", b.own(
            "with-runs", "--runs", b.runs, "--campaign", campaign, "--aside",
            f"{c.SCAN_EXPERIMENT}/{run_id}", "--unless", record_path, "--",
            "scripts/study5_scan.py", *run_args(seed, e, lo), "--f-run",
            f"{{run:{c.label(cell(seed, e, lo))}}}", "--checkpoints", b.checkpoints, "--out",
            record_path, "--run-id", run_id, "--points", POINT),
            outputs=(str(record_path),), group=group, minutes=SCAN)  # fmt: skip
        return b.rel(record_path), f"{c.SCAN_EXPERIMENT}/{run_id}"

    def drop(name: str, campaign: str, run_label: str, inventory: str, consumer: str,
             group: str) -> None:  # fmt: skip
        b.step(name, b.own("drop-checkpoints", "--runs", b.runs, "--campaign", campaign,
                           "--run", run_label, "--inventory", inventory, "--consumer", consumer,
                           "--endpoint"), group=group)  # fmt: skip

    b.begin()  # the grid this plan was built from; the pilot's scans, then its checkpoints
    b.step("pilot-scans/check-grid", (SELF, "check-grid", "--grid", grid_path, "--sha256",
           c.sha256_of(Path(grid_path))), on_fail="stop")  # fmt: skip
    group = f"s{pilot}"
    campaign = pilot_campaign(archive_id, pilot)
    scans = {e: scan("pilot-scans", group, campaign, pilot, e) for e in early}
    archived = b.runs_archive(f"pilot-scans/{group}/archive",
                              paths=tuple(rec for rec, _ in scans.values()),
                              run_dirs=tuple((run, rec) for rec, run in scans.values()),
                              group=group)  # fmt: skip
    for e in early:
        for late, _ in dose.LATE_CANDIDATES:
            consumer, inventory = ((scans[e][0], archived.outputs[0]) if late == lo
                                   else (grid_rel, str(grid_inventory)))  # fmt: skip
            drop(f"pilot-scans/{group}/drop-e{e}-l{late}", campaign,
                 c.label(cell(pilot, e, late)), inventory, consumer, group)  # fmt: skip
    b.end("pilot-scans")

    b.begin()  # per other seed: six runs, the scans of its three L_lo runs, then its checkpoints
    for seed in share["seeds"]:
        if seed == pilot:
            continue
        group = f"s{seed}"
        b.step(f"main/{group}/check-checkpoints",
               b.own("check-checkpoints", "--dir", b.checkpoints, "--max", LEFT_MAX),
               on_fail="stop", group=group)  # fmt: skip
        cells = [cell(seed, e, lo, b.checkpoints) for e in early]
        cells += [cell(seed, e, hi) for e in early]
        _, campaign = b.launch(f"main-{group}", "main", cells, group=group,
                               minutes=len(early) * (RUN_CKPT + RUN))  # fmt: skip
        scans = {e: scan("main", group, campaign, seed, e) for e in early}
        archived = b.runs_archive(f"main/{group}/archive",
                                  paths=tuple(rec for rec, _ in scans.values()),
                                  run_dirs=tuple((run, rec) for rec, run in scans.values()),
                                  campaign=campaign, group=group)  # fmt: skip
        for e in early:
            drop(f"main/{group}/drop-e{e}", campaign, c.label(cell(seed, e, lo)),
                 archived.outputs[0], scans[e][0], group)  # fmt: skip
    b.end("main")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


def adamw_plan(*, out: Path, runs: Path, data: Path, checkpoints: Path, archive_id: str,
               commit: str, gpu_code: str, repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The order repeat under AdamW (docs/study5-adamw-registration.md) on a Pod of its own:
    ``engineering`` as on the dose Pods; ``fetch``, the ten seeds' streams
    (``docs/study5-adamw-seeds.json``, checked against the draw); ``orders``, per seed one
    launcher campaign of four A runs (``adamw_orders.cell``: E2's two orders × A's two rates),
    archived as its seed group ends; ``close``. No checkpoint, no labeling."""
    record = json.loads((repo / adamw_orders.SEEDS_FILE).read_text(encoding="utf-8"))
    if record != adamw_orders.seeds_record():
        raise c.StepError(f"{adamw_orders.SEEDS_FILE} is not the draw")
    b = c.Builder(ADAMW_POD, archive_id, out, runs, data, checkpoints)
    b.admit = True  # each block and seed group admitted only when it and the close fit the cap
    fetch_streams = engineering(b, gpu_code)

    b.begin()  # the ten seeds' streams, each fetched only if missing
    for seed in record["seeds"]:
        fetch_streams(f"fetch/s{seed}/streams", seed)
    b.end("fetch")

    b.begin()  # per seed: E2's two orders at A's two rates
    for seed in record["seeds"]:
        group = f"s{seed}"
        cells = [adamw_orders.cell(seed, order, rate) for order, _ in adamw_orders.ORDER_OFFSETS
                 for rate, _ in adamw_orders.RATES]  # fmt: skip
        _, campaign = b.launch(f"orders-{group}", "orders", cells, group=group,
                               minutes=len(cells) * RUN)  # fmt: skip
        b.runs_archive(f"orders/{group}/archive", campaign=campaign, group=group)
    b.end("orders")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


# ------------------------------------------------------------------ the steps


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]  # fmt: skip


def table_row(runs: Path, campaign: str, early: str, late: str, run_label: str) -> dict[str, Any]:
    """A pilot run's row: its launcher end, whether it took all 25 updates (``complete``) and its
    K_benign at step 25 (None otherwise). A run the journal shows without an end fails."""
    events = c.journal_events(runs, campaign)
    ends = [e for e in events if e.get("event") == "end" and e.get("run") == run_label]
    if not ends:
        raise c.StepError(f"{campaign}: {run_label} has no end")
    end = ends[-1]
    row: dict[str, Any] = {"early": early, "late": late, "label": run_label,
                           "run_id": end.get("run_id"), "outcome": end.get("outcome"),
                           "steps": None, "stop": None, "complete": False, "K": None}  # fmt: skip
    try:
        start = c._start_of(runs, campaign, end["run_id"])
        run_dir = runs / c.safe_rel(start["experiment"]) / c.safe_rel(end["run_id"])
        terminal = json.loads((run_dir / "terminal.json").read_text(encoding="utf-8"))
    except (c.StepError, OSError, ValueError, KeyError, TypeError):
        return row
    row["steps"] = terminal.get("steps")
    row["stop"] = (terminal.get("stop") or {}).get("reason")
    row["complete"] = row["steps"] == dose.STEPS and row["stop"] == "steps"
    found = [r for r in _rows(run_dir / "evaluations.jsonl") if r.get("step") == dose.STEPS]
    if row["complete"] and len(found) == 1:
        k = found[0].get("K_benign")
        row["K"] = float(k) if isinstance(k, int | float) and not isinstance(k, bool) else None
    return row


def cmd_table(args) -> int:
    rows = []
    for early, late, run_label in args.run:
        found = table_row(args.runs, args.campaign, early, late, run_label)
        rows.append({"seed": args.seed, **found})
    record = {"kind": TABLE_KIND, "pod": args.pod, "seed": args.seed,
              "gpu_code": args.gpu_code, "campaign": args.campaign, "rows": rows,
              "created_utc": c._now()}  # fmt: skip
    c.write_json(args.out, record)
    for row in rows:
        print(f"pilot s{args.seed} e{row['early']} l{row['late']} {row['run_id']} "
              f"{row['outcome']} complete {row['complete']} K {row['K']}")  # fmt: skip
    print(f"table {args.out.name} sha256 {c.sha256_of(args.out)}")
    return 0


def cmd_publish(args) -> int:
    """The file to ``$ARTIFACT_REMOTE/<remote>/``, verified (size and MD5), then read back."""
    base = c.artifact_remote()
    staged = args.inventory.with_name(f".{args.inventory.name}.partial")
    staged.unlink(missing_ok=True)
    result = c.archive(staged, f"{base}/{args.remote}", args.file.parent, [args.file.name])
    if result["mismatched"]:
        raise c.StepError(f"{args.file.name}: the copy does not match")
    digest = c.sha256_of(args.file)
    c.readback(f"{base}/{args.remote}/{args.file.name}", digest)
    os.replace(staged, args.inventory)
    print(f"published {args.file.name} sha256 {digest}")
    return 0


def table_problem(table: Any, pod: str, seed: int, gpu_code: str) -> str | None:
    """Why ``table`` is not Pod ``pod``'s pilot table of ``seed`` under ``gpu_code``."""
    if not isinstance(table, dict) or table.get("kind") != TABLE_KIND:
        return "not a pilot table"
    if table.get("pod") != pod or table.get("seed") != seed:
        return f"Pod {table.get('pod')!r} seed {table.get('seed')!r}, not {pod} {seed}"
    if table.get("gpu_code") != gpu_code:
        return "written under another GPU code"
    rows = table.get("rows")
    expected = {(e, late) for e, _ in dose.EARLY for late, _ in dose.LATE_CANDIDATES}
    if not isinstance(rows, list) or {(r.get("early"), r.get("late")) for r in rows} != expected \
            or len(rows) != len(expected) or any(r.get("seed") != seed for r in rows):  # fmt: skip
        return "not the nine pilot runs of its seed"
    return None


def fetch_table(url: str, target: Path, wait_minutes: float, poll: float = POLL) -> None:
    """The other Pod's table from ``url`` into ``target`` (kept when already there), waiting at
    most ``wait_minutes``; ``StepError`` when it never appears."""
    if target.is_file():
        return
    give_up = time.monotonic() + 60 * wait_minutes
    while True:
        partial = target.with_name(f".{target.name}.partial")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            c.gcloud(["cp", "--quiet", url, str(partial)])
            os.replace(partial, target)
            return
        except c.StepError:
            partial.unlink(missing_ok=True)
        if time.monotonic() + poll > give_up:
            raise c.StepError(f"the other Pod's pilot table did not appear in {wait_minutes} min")
        print(f"waiting for {url.rsplit('/', 2)[-2]}'s table")
        time.sleep(poll)


def cmd_grid(args) -> int:
    pilots = dict(item.split("=", 1) for item in args.pilot)
    pilots = {pod: int(seed) for pod, seed in pilots.items()}
    if sorted(pilots) != sorted(dose.PODS):
        raise c.StepError("--pilot a=SEED and --pilot b=SEED are expected")
    own = json.loads(args.own.read_text(encoding="utf-8"))
    pod = own.get("pod")
    if pod not in dose.PODS:
        raise c.StepError("the own table names no Pod")
    fetch_table(f"{c.artifact_remote()}/{c.safe_rel(args.other_remote)}", args.fetched,
                args.wait)  # fmt: skip
    theirs = json.loads(args.fetched.read_text(encoding="utf-8"))
    tables = {pod: own, other(pod): theirs}
    for name, table in tables.items():
        problem = table_problem(table, name, pilots[name], args.gpu_code)
        if problem is not None:
            raise c.StepError(f"Pod {name}'s pilot table: {problem}")
    rows = [row for name in dose.PODS for row in tables[name]["rows"]]
    grid = dose.grid_choice(rows, [pilots[name] for name in dose.PODS])
    record = {
        "kind": GRID_KIND,
        "gpu_code": args.gpu_code,
        "tables": {name: tables[name] for name in dose.PODS},
        "tables_sha256": {pod: c.sha256_of(args.own), other(pod): c.sha256_of(args.fetched)},
        "grid": grid,
        "created_utc": c._now(),
    }
    c.write_json(args.out, record)
    for pair in grid["pairs"]:
        print(f"pair {pair['lo']} {pair['hi']} covered {pair['covered']} of {grid['cells']}")
    chosen = grid["chosen"]
    print(f"chosen L_lo {chosen['lo']} ({chosen['L_lo']!r}) L_hi {chosen['hi']} "
          f"({chosen['L_hi']!r}); grid {args.out.name} sha256 {c.sha256_of(args.out)}")  # fmt: skip
    return 0


def cmd_check_grid(args) -> int:
    digest = c.sha256_of(args.grid)
    if digest != args.sha256:
        raise c.StepError(f"{args.grid.name}: sha256 {digest}, not the plan's {args.sha256}")
    read_grid(args.grid)
    print(f"grid {args.grid.name} sha256 {digest}")
    return 0


# ------------------------------------------------------------------ the driver


def preflight(plan: c.Plan, repo: Path = c.REPO) -> list[str]:
    """What forbids the start: the checkout clean at the plan's commit, its GPU code the one
    ``IDENTITY`` records (``ADAMW_IDENTITY`` for the AdamW Pod), the registration (the AdamW
    Pod's: ``ADAMW_REGISTRATION``) ``registered``, a runs root and archive id of the
    Pod's own (``own_root_problems``), the bucket, the key, the folder."""
    problems = []
    state = git_state(repo)
    if state.get("dirty") is not False or state.get("commit") != plan.commit:
        problems.append("the checkout is not clean at the plan's commit")
    identity, registration = ((ADAMW_IDENTITY, ADAMW_REGISTRATION) if plan.pod == ADAMW_POD
                              else (IDENTITY, REGISTRATION))  # fmt: skip
    recorded = json.loads((repo / identity).read_text(encoding="utf-8"))
    if plan.gpu_code != recorded.get("gpu_code_sha256"):
        problems.append(f"HEAD's GPU code is not the one {identity} records")
    if not c.REGISTERED.search((repo / registration).read_text(encoding="utf-8")):
        problems.append(f"{registration} is not registered")
    problems += c.own_root_problems(plan)
    if not os.environ.get("ARTIFACT_REMOTE", "").startswith("gs://"):
        problems.append("ARTIFACT_REMOTE is not a gs:// path")
    if not Path(os.environ.get("ARTIFACT_GCS_KEY", "/nonexistent")).is_file():
        problems.append("ARTIFACT_GCS_KEY is not a readable key file")
    if plan.out.resolve().is_relative_to(repo.resolve()):
        problems.append("--out lies inside the checkout")
    return problems


def start(plan: c.Plan) -> int:
    """``scripts/study5_campaigns.py``'s start of a plan: preflight, prepare, execute."""
    problems = preflight(plan)
    if problems:
        for problem in problems:
            print(f"refused: {problem}")
        return c.ERROR
    c.prepare(plan)
    env = {**os.environ, "RUNS_DIR": str(plan.runs), "DATA_DIR": str(plan.data),
           "PYTHONUNBUFFERED": "1"}  # fmt: skip
    code = c.execute(plan, python=[sys.executable], cwd=c.REPO, env=env)
    print(f"== {c._now()} {plan.pod} {Path(plan.out).name} "
          f"{'completed' if code == c.COMPLETED else 'stopped'}")  # fmt: skip
    return code


def cmd_run(args) -> int:
    if not args.archive_id.startswith(f"dose-{args.pod}-"):
        raise c.StepError(f"--archive-id {args.archive_id}: Pod {args.pod}'s starts with "
                          f"dose-{args.pod}-")  # fmt: skip
    if not args.other_archive_id.startswith(f"dose-{other(args.pod)}-"):
        raise c.StepError(f"--other-archive-id {args.other_archive_id}: not Pod "
                          f"{other(args.pod)}'s")  # fmt: skip
    commit = git_state(c.REPO).get("commit") or "unknown"
    try:
        gpu_code = code_identity.gpu_code_identity(commit)["sha256"]
    except Exception:  # no commit: the preflight refuses
        gpu_code = "unknown"
    out = args.out.absolute()
    pilot_out, main_out = (out.with_name(f"{out.name}-{part}") for part in ("pilot", "main"))
    where = {"pod": args.pod, "runs": runs_root(), "data": data_root(),
             "checkpoints": args.checkpoints, "archive_id": args.archive_id, "commit": commit,
             "gpu_code": gpu_code}  # fmt: skip
    pilot = pilot_plan(**where, out=pilot_out, other_archive_id=args.other_archive_id)
    grid = pilot.runs / "study5_records" / args.archive_id / GRID
    if args.dry_run:
        c.show(pilot)
        if args.grid is not None:
            c.show(main_plan(**where, out=main_out, grid_path=args.grid, pilot_out=pilot_out))
        return c.COMPLETED
    c.install_sigterm()
    code = start(pilot)
    if code != c.COMPLETED:
        return code
    return start(main_plan(**where, out=main_out, grid_path=grid, pilot_out=pilot_out))


def cmd_adamw(args) -> int:
    if not args.archive_id.startswith("adamw-"):
        raise c.StepError(f"--archive-id {args.archive_id}: the AdamW Pod's starts with adamw-")
    commit = git_state(c.REPO).get("commit") or "unknown"
    try:
        gpu_code = code_identity.gpu_code_identity(commit)["sha256"]
    except Exception:  # no commit: the preflight refuses
        gpu_code = "unknown"
    plan = adamw_plan(out=args.out.absolute(), runs=runs_root(), data=data_root(),
                      checkpoints=args.checkpoints, archive_id=args.archive_id, commit=commit,
                      gpu_code=gpu_code)  # fmt: skip
    if args.dry_run:
        c.show(plan)
        return c.COMPLETED
    c.install_sigterm()
    return start(plan)


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("run")
    p.add_argument("--pod", choices=dose.PODS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive-id", required=True)
    p.add_argument("--other-archive-id", required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--grid", type=Path)  # the dry run's main plan, from a grid record
    p = sub.add_parser("adamw")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive-id", required=True)
    p.add_argument("--checkpoints", type=Path, required=True)  # unused: no run saves one
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("table")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--pod", choices=dose.PODS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu-code", required=True)
    p.add_argument("--run", nargs=3, action="append", required=True)  # EARLY LATE LABEL
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("publish")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p = sub.add_parser("grid")
    p.add_argument("--own", type=Path, required=True)
    p.add_argument("--other-remote", required=True)
    p.add_argument("--fetched", type=Path, required=True)
    p.add_argument("--wait", type=float, required=True)
    p.add_argument("--pilot", action="append", required=True)  # POD=SEED
    p.add_argument("--gpu-code", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("check-grid")
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    return parser


COMMANDS = {"run": cmd_run, "adamw": cmd_adamw, "table": cmd_table, "publish": cmd_publish, "grid": cmd_grid,
            "check-grid": cmd_check_grid}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        return COMMANDS[args.kind](args)
    except (c.StepError, ValueError, OSError) as exc:
        print(f"study5_dose_campaigns.py {args.kind}: {exc}")
        return c.ERROR


if __name__ == "__main__":
    sys.exit(main())
