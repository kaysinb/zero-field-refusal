"""Study 5's run without momentum on one Pod (docs/study5-beta0-registration.md): one detached
driver of three plans in order, built and run by ``scripts/study5_campaigns.py``'s machinery (its
``Builder``, ``execute``, deadlines, archives, receipts, restarts and close), with the dose Pods'
engineering block (``scripts/study5_dose_campaigns.py``).

    python scripts/study5_beta0_campaigns.py run --out DIR --archive-id beta0-...
        --checkpoints DIR [--dry-run [--rate FILE]]

    python scripts/study5_beta0_campaigns.py run ... --dry-run --rate FILE --gate FILE

The three plans' campaign folders are DIR-pilot, DIR-main and DIR-orders, siblings.

**The pilot plan** (``pilot_plan``):

- ``engineering``: pod 2's order control of 50000 at η_l rerun under this checkout and compared
  under extension 2's numerical equivalence; any failure stops before anything else.
- ``fetch``: the six seeds' streams from pod 2's archive.
- ``pilot``: the pilot seeds' runs at the ``beta0.GRID`` rates, K only
  (``experiments/study5_beta0_k_only``), then ``table`` (per run its end, whether it took all 25
  updates and its K_benign at step 25; no label, no harm).
- ``rate``: ``beta0.rate_choice`` on the table and the committed targets; the record holds the
  table whole and the choice. No η₀ (a target not bracketed) stops the plan before any main run.

**The main plan** (``main_plan``), built from that rate record (``check-rate`` first: its sha256 is
the one the plan was built from, and its choice is the rule's on its table):

- ``stream``: the six stream-order runs at η₀; ``gate`` writes ``gate.json``, the registered
  rule's verdict (``beta0.drift_gate``), kept as written; ``gate-amended`` writes
  ``gate-amended.json``, the verdict of the addendum of 2026-11-18
  (``analysis.study5_beta0.amended_gate``: at least five runs with a value, every other one a
  scientific stop, the geometric mean in ``beta0.GATE``) with the registered verdict recomputed
  beside it; a failed amended gate stops the plan before any other order.

**The orders plan** (``orders_plan``), built from ``gate-amended.json`` (the seeds with a value):

- ``check-gate`` first, run at every start (``always``) and stopping on failure: both verdicts
  recomputed from the stream-order runs' launcher journal and run directories, never taken from the
  records, which must agree with them (``gate.json`` with the registered one, ``gate-amended.json``
  with the amended one); the amended gate passed; the plan's seeds its seeds with a value.
- per seed with a value, one launcher campaign of its other orders at η₀ (``beta0.orders_of``: the
  reorder and the shuffles, an identity permutation not run), archived as its seed group ends. A
  seed whose stream-order run stopped runs no other order (its edits are ``not_run`` in the
  analysis).

No checkpoint is saved and no run is labeled here: the judge runs on its own Pod after the runs are
restored, as for the dose experiment. A failed run is recorded and the plan goes on; every archive,
the engineering check, the table, the rate and the amended gate stop it. ``run`` executes the pilot
plan, only when it completed the main plan, and only when that completed the orders plan; started
again with the same arguments it skips what is done (``scripts/study5_campaigns.py``,
"Resumable").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study5_campaigns as c  # noqa: E402  (the driver's machinery)
import study5_dose_campaigns as d  # noqa: E402  (the engineering block)

from zero_field_refusal import beta0, code_identity  # noqa: E402
from zero_field_refusal.analysis import study5_beta0 as sb  # noqa: E402
from zero_field_refusal.paths import data_root, runs_root  # noqa: E402
from zero_field_refusal.tracking import git_state  # noqa: E402

SELF = "scripts/study5_beta0_campaigns.py"
POD = "beta0"
IDENTITY = "docs/study5-beta0-code-identity.json"
REGISTRATION = "docs/study5-beta0-registration.md"
RUN = c.F_RUN  # minutes of a main run; a K-only pilot run takes less
TABLE, RATE, GATE, AMENDED = "pilot-table.json", "rate.json", "gate.json", "gate-amended.json"
TABLE_KIND, RATE_KIND, GATE_KIND = ("study5-beta0-pilot-table", "study5-beta0-rate",
                                    "study5-beta0-gate")  # fmt: skip
AMENDED_KIND = "study5-beta0-gate-amended"
c.PREFIXES[POD] = "beta0-"


def builder(archive_id: str, out: Path, runs: Path, data: Path, checkpoints: Path) -> c.Builder:
    b = c.Builder(POD, archive_id, out, runs, data, checkpoints)
    b.admit = True  # each block and seed group admitted only when it and the close fit the cap
    return b


def targets(repo: Path = c.REPO) -> dict[int, float]:
    return beta0.targets_from(json.loads((repo / beta0.TARGETS_FILE).read_text(encoding="utf-8")))


def pilot_plan(*, out: Path, runs: Path, data: Path, checkpoints: Path, archive_id: str,
               commit: str, gpu_code: str, repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The pilot plan (module docstring)."""
    targets(repo)  # the committed targets read and match beta0.TARGET_RUNS
    b = builder(archive_id, out, runs, data, checkpoints)
    fetch_streams = d.engineering(b, gpu_code)

    b.begin()  # the six seeds' streams, each fetched only if missing
    for seed in beta0.SEEDS:
        fetch_streams(f"fetch/s{seed}/streams", seed)
    b.end("fetch")

    b.begin()  # the pilot runs, K only, and the table
    cells = [beta0.pilot_cell(s, level) for s in beta0.PILOT_SEEDS for level, _ in beta0.GRID]
    _, campaign = b.launch("pilot", "pilot", cells, minutes=len(cells) * RUN)
    table = b.record("pilot", TABLE)
    argv = [SELF, "table", "--runs", b.runs, "--campaign", campaign, "--gpu-code", gpu_code,
            "--out", table]  # fmt: skip
    for seed in beta0.PILOT_SEEDS:
        for level, _ in beta0.GRID:
            argv += ["--run", seed, level, c.label(beta0.pilot_cell(seed, level))]
    b.step("pilot/table", argv, on_fail="stop", outputs=(str(table),))
    rate = b.record(RATE)
    b.step("pilot/rate", (SELF, "rate", "--table", table, "--out", rate), on_fail="stop",
           outputs=(str(rate),))  # fmt: skip
    b.runs_archive("pilot/archive", paths=(b.rel(table), b.rel(rate)), campaign=campaign)
    b.end("pilot")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


def read_rate(path: Path, repo: Path = c.REPO) -> dict[str, Any]:
    """A rate record whose choice is the rule's on the table it holds, with an η₀."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("kind") != RATE_KIND:
        raise c.StepError(f"{Path(path).name}: not a rate record")
    choice = beta0.rate_choice(record["table"]["rows"], targets(repo))
    if choice != record.get("choice"):
        raise c.StepError(f"{Path(path).name}: its choice is not the rule's on its table")
    if choice["eta0"] is None:
        raise c.StepError(f"{Path(path).name}: no η₀ ({choice['stop']})")
    return record


def main_plan(*, out: Path, runs: Path, data: Path, checkpoints: Path, archive_id: str,
              rate_path: Path, commit: str, gpu_code: str,
              repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The main plan (module docstring), from the rate record at ``rate_path``."""
    eta0 = float(read_rate(rate_path, repo)["choice"]["eta0"])
    b = builder(archive_id, out, runs, data, checkpoints)

    b.begin()  # the rate this plan was built from; the six stream-order runs; the drift gates
    b.step("stream/check-rate", (SELF, "check-rate", "--rate", rate_path, "--sha256",
           c.sha256_of(Path(rate_path))), on_fail="stop")  # fmt: skip
    cells = [beta0.cell(s, "stream", eta0) for s in beta0.SEEDS]
    _, campaign = b.launch("stream", "stream", cells, minutes=len(cells) * RUN)
    gate, amended = b.record(GATE), b.record(AMENDED)
    runs_argv = []
    for seed in beta0.SEEDS:
        runs_argv += ["--run", seed, c.label(beta0.cell(seed, "stream", eta0))]
    b.step("stream/gate", [SELF, "gate", "--runs", b.runs, "--campaign", campaign, "--out", gate,
                           *runs_argv], on_fail="stop", outputs=(str(gate),))  # fmt: skip
    b.step("stream/gate-amended", [SELF, "gate-amended", "--runs", b.runs, "--campaign", campaign,
                                   "--out", amended, *runs_argv],
           on_fail="stop", outputs=(str(amended),))  # fmt: skip
    b.runs_archive("stream/archive", paths=(b.rel(gate), b.rel(amended)), campaign=campaign)
    b.end("stream")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


def stream_campaign(archive_id: str) -> str:
    """The launcher campaign of the main plan's stream-order runs (``Builder.launcher``)."""
    return f"study5-{archive_id}-stream"


def read_amended(path: Path) -> dict[str, Any]:
    """An amended gate record that passed."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("kind") != AMENDED_KIND:
        raise c.StepError(f"{Path(path).name}: not an amended gate record")
    if (record.get("gate") or {}).get("passed") is not True:
        raise c.StepError(f"{Path(path).name}: the amended gate did not pass")
    return record


def orders_plan(*, out: Path, runs: Path, data: Path, checkpoints: Path, archive_id: str,
                rate_path: Path, gate_path: Path, commit: str, gpu_code: str,
                repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The orders plan (module docstring), from the rate record and the amended gate record."""
    eta0 = float(read_rate(rate_path, repo)["choice"]["eta0"])
    seeds = [int(s) for s in read_amended(gate_path)["gate"]["seeds"]]
    b = builder(archive_id, out, runs, data, checkpoints)
    campaign = stream_campaign(archive_id)
    gate, amended = b.record(GATE), b.record(AMENDED)
    if Path(amended) != Path(gate_path):
        raise c.StepError(f"{Path(gate_path).name}: not this archive's amended gate record")

    b.begin()  # both verdicts again, from the runs, at every start; then per seed its orders
    argv = [SELF, "check-gate", "--runs", b.runs, "--campaign", campaign, "--gate", gate,
            "--amended", amended]  # fmt: skip
    for seed in beta0.SEEDS:
        argv += ["--run", seed, c.label(beta0.cell(seed, "stream", eta0))]
    for seed in seeds:
        argv += ["--seed", seed]
    b.step("orders/check-gate", argv, on_fail="stop", always=True)
    for seed in seeds:
        group = f"s{seed}"
        orders = [o for o in beta0.orders_of(seed) if o != "stream"]
        cells = [beta0.cell(seed, o, eta0) for o in orders]
        _, campaign = b.launch(f"orders-{group}", "orders", cells, group=group,
                               minutes=len(cells) * RUN)  # fmt: skip
        b.runs_archive(f"orders/{group}/archive", campaign=campaign, group=group)
    b.end("orders")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


# ------------------------------------------------------------------ the steps


def k_row(runs: Path, campaign: str, run_label: str) -> dict[str, Any]:
    """A run's end, whether it took all 25 updates (``complete``) and its K_benign at step 25
    (None otherwise). A run the journal shows without an end fails."""
    events = c.journal_events(runs, campaign)
    ends = [e for e in events if e.get("event") == "end" and e.get("run") == run_label]
    if not ends:
        raise c.StepError(f"{campaign}: {run_label} has no end")
    end = ends[-1]
    row: dict[str, Any] = {"label": run_label, "run_id": end.get("run_id"),
                           "outcome": end.get("outcome"), "steps": None, "stop": None,
                           "complete": False, "K": None}  # fmt: skip
    try:
        start = c._start_of(runs, campaign, end["run_id"])
        run_dir = runs / c.safe_rel(start["experiment"]) / c.safe_rel(end["run_id"])
        terminal = json.loads((run_dir / "terminal.json").read_text(encoding="utf-8"))
    except (c.StepError, OSError, ValueError, KeyError, TypeError):
        return row
    row["steps"] = terminal.get("steps")
    row["stop"] = (terminal.get("stop") or {}).get("reason")
    row["complete"] = (row["outcome"] == beta0.ACCEPTED_END and row["steps"] == beta0.STEPS
                       and row["stop"] == "steps")  # fmt: skip
    found = [r for r in d._rows(run_dir / "evaluations.jsonl") if r.get("step") == beta0.STEPS]
    if row["complete"] and len(found) == 1:
        k = found[0].get("K_benign")
        row["K"] = float(k) if isinstance(k, int | float) and not isinstance(k, bool) else None
    return row


def cmd_table(args) -> int:
    rows = [{"seed": int(seed), "level": level, **k_row(args.runs, args.campaign, label)}
            for seed, level, label in args.run]  # fmt: skip
    record = {"kind": TABLE_KIND, "gpu_code": args.gpu_code, "campaign": args.campaign,
              "rows": rows, "created_utc": c._now()}  # fmt: skip
    c.write_json(args.out, record)
    for row in rows:
        print(f"pilot s{row['seed']} {row['level']} {row['run_id']} {row['outcome']} "
              f"complete {row['complete']} K {row['K']}")  # fmt: skip
    print(f"table {args.out.name} sha256 {c.sha256_of(args.out)}")
    return 0


def cmd_rate(args) -> int:
    table = json.loads(args.table.read_text(encoding="utf-8"))
    if table.get("kind") != TABLE_KIND:
        raise c.StepError(f"{args.table.name}: not a pilot table")
    choice = beta0.rate_choice(table["rows"], targets())
    record = {"kind": RATE_KIND, "table": table, "table_sha256": c.sha256_of(args.table),
              "choice": choice, "created_utc": c._now()}  # fmt: skip
    c.write_json(args.out, record)
    for seed, found in choice["per_seed"].items():
        bracket = found["bracket"]
        print(f"s{seed} target {found['target']:.4f} bracket "
              f"{bracket['levels'] if bracket else None} eta_s "
              f"{bracket['eta_s'] if bracket else None}")  # fmt: skip
    print(f"eta0 {choice['eta0']} stop {choice['stop']} rate {args.out.name} sha256 "
          f"{c.sha256_of(args.out)}")  # fmt: skip
    return 0 if choice["eta0"] is not None else c.ERROR


def cmd_check_rate(args) -> int:
    digest = c.sha256_of(args.rate)
    if digest != args.sha256:
        raise c.StepError(f"{args.rate.name}: sha256 {digest}, not the plan's {args.sha256}")
    read_rate(args.rate)
    print(f"rate {args.rate.name} sha256 {digest}")
    return 0


def gate_rows(args) -> list[dict[str, Any]]:
    return [{"seed": int(seed), **k_row(args.runs, args.campaign, label)}
            for seed, label in args.run]  # fmt: skip


def cmd_check_gate(args) -> int:
    """Both verdicts recomputed from the runs; each record must agree with its verdict; the
    amended gate passed; the plan's seeds are its seeds with a value."""
    rows = gate_rows(args)
    registered, amended = beta0.drift_gate(rows, targets()), sb.amended_gate(rows, targets())
    for path, verdict, kind in ((args.gate, registered, GATE_KIND),
                                (args.amended, amended, AMENDED_KIND)):  # fmt: skip
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("kind") != kind or stored.get("gate") != verdict:
            raise c.StepError(f"{path.name}: its verdict is not the one the runs give")
    if json.loads(args.amended.read_text(encoding="utf-8")).get("registered") != registered:
        raise c.StepError(f"{args.amended.name}: its registered verdict is not the runs'")
    seeds = sorted(int(s) for s in args.seed)
    if seeds != amended["seeds"]:
        raise c.StepError(f"the plan's seeds {seeds} are not the amended gate's {amended['seeds']}")
    print(f"check-gate registered passed {registered['passed']} amended passed "
          f"{amended['passed']} ratio {amended['ratio']} seeds {amended['seeds']} stopped "
          f"{amended['stopped']}")  # fmt: skip
    return 0 if amended["passed"] else c.ERROR


def cmd_gate(args) -> int:
    """The registered rule's verdict, written as it was; under the addendum of 2026-11-18 the
    amended gate (``gate-amended``) decides whether the plan goes on."""
    rows = gate_rows(args)
    gate = beta0.drift_gate(rows, targets())
    record = {"kind": GATE_KIND, "campaign": args.campaign, "rows": rows, "gate": gate,
              "created_utc": c._now()}  # fmt: skip
    c.write_json(args.out, record)
    print(f"gate passed {gate['passed']} ratio {gate['ratio']} missing {gate['missing']} "
          f"gate {args.out.name} sha256 {c.sha256_of(args.out)}")  # fmt: skip
    return 0


def cmd_gate_amended(args) -> int:
    rows = gate_rows(args)
    gate = sb.amended_gate(rows, targets())
    record = {"kind": AMENDED_KIND, "campaign": args.campaign, "rows": rows, "gate": gate,
              "registered": beta0.drift_gate(rows, targets()),
              "rule": "docs/study5-beta0-registration.md, addendum of 2026-11-18",
              "created_utc": c._now()}  # fmt: skip
    c.write_json(args.out, record)
    print(f"gate-amended passed {gate['passed']} ratio {gate['ratio']} seeds {gate['seeds']} "
          f"stopped {gate['stopped']} technical {gate['technical']} gate {args.out.name} "
          f"sha256 {c.sha256_of(args.out)}")  # fmt: skip
    return 0 if gate["passed"] else c.ERROR


# ------------------------------------------------------------------ the driver


def preflight(plan: c.Plan, repo: Path = c.REPO) -> list[str]:
    """What forbids the start, as for the dose Pods: the checkout clean at the plan's commit, its
    GPU code the one ``IDENTITY`` records, ``REGISTRATION`` registered, a runs root and archive id
    of the Pod's own, the bucket, the key, the folder."""
    problems = []
    state = git_state(repo)
    if state.get("dirty") is not False or state.get("commit") != plan.commit:
        problems.append("the checkout is not clean at the plan's commit")
    path = repo / IDENTITY
    recorded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if plan.gpu_code != recorded.get("gpu_code_sha256"):
        problems.append(f"HEAD's GPU code is not the one {IDENTITY} records")
    if not c.REGISTERED.search((repo / REGISTRATION).read_text(encoding="utf-8")):
        problems.append(f"{REGISTRATION} is not registered")
    problems += c.own_root_problems(plan)
    if not os.environ.get("ARTIFACT_REMOTE", "").startswith("gs://"):
        problems.append("ARTIFACT_REMOTE is not a gs:// path")
    if not Path(os.environ.get("ARTIFACT_GCS_KEY", "/nonexistent")).is_file():
        problems.append("ARTIFACT_GCS_KEY is not a readable key file")
    if plan.out.resolve().is_relative_to(repo.resolve()):
        problems.append("--out lies inside the checkout")
    return problems


def start(plan: c.Plan) -> int:
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
    if not args.archive_id.startswith("beta0-"):
        raise c.StepError(f"--archive-id {args.archive_id}: this Pod's starts with beta0-")
    commit = git_state(c.REPO).get("commit") or "unknown"
    try:
        gpu_code = code_identity.gpu_code_identity(commit)["sha256"]
    except Exception:  # no commit: the preflight refuses
        gpu_code = "unknown"
    out = args.out.absolute()
    pilot_out, main_out, orders_out = (out.with_name(f"{out.name}-{part}")
                                       for part in ("pilot", "main", "orders"))  # fmt: skip
    where = {"runs": runs_root(), "data": data_root(), "checkpoints": args.checkpoints,
             "archive_id": args.archive_id, "commit": commit, "gpu_code": gpu_code}  # fmt: skip
    pilot = pilot_plan(**where, out=pilot_out)
    records = pilot.runs / "study5_records" / args.archive_id
    rate, amended = records / RATE, records / AMENDED
    if args.dry_run:
        c.show(pilot)
        if args.rate is not None:
            c.show(main_plan(**where, out=main_out, rate_path=args.rate))
            if args.gate is not None:
                c.show(orders_plan(**where, out=orders_out, rate_path=args.rate,
                                   gate_path=args.gate))  # fmt: skip
        return c.COMPLETED
    c.install_sigterm()
    code = start(pilot)
    if code != c.COMPLETED:
        return code
    code = start(main_plan(**where, out=main_out, rate_path=rate))
    if code != c.COMPLETED:
        return code
    return start(orders_plan(**where, out=orders_out, rate_path=rate, gate_path=amended))


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive-id", required=True)
    p.add_argument("--checkpoints", type=Path, required=True)  # unused: no run saves one
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rate", type=Path)  # the dry run's main plan, from a rate record
    p.add_argument("--gate", type=Path)  # the dry run's orders plan, from an amended gate record
    p = sub.add_parser("table")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--gpu-code", required=True)
    p.add_argument("--run", nargs=3, action="append", required=True)  # SEED LEVEL LABEL
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("rate")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("check-rate")
    p.add_argument("--rate", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p = sub.add_parser("check-gate")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--amended", type=Path, required=True)
    p.add_argument("--run", nargs=2, action="append", required=True)  # SEED LABEL
    p.add_argument("--seed", action="append", required=True)  # the plan's seeds
    for name in ("gate", "gate-amended"):
        p = sub.add_parser(name)
        p.add_argument("--runs", type=Path, required=True)
        p.add_argument("--campaign", required=True)
        p.add_argument("--run", nargs=2, action="append", required=True)  # SEED LABEL
        p.add_argument("--out", type=Path, required=True)
    return parser


COMMANDS = {"run": cmd_run, "table": cmd_table, "rate": cmd_rate, "check-rate": cmd_check_rate,
            "gate": cmd_gate, "gate-amended": cmd_gate_amended,
            "check-gate": cmd_check_gate}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        return COMMANDS[args.kind](args)
    except (c.StepError, ValueError, OSError) as exc:
        print(f"study5_beta0_campaigns.py {args.kind}: {exc}")
        return c.ERROR


if __name__ == "__main__":
    sys.exit(main())
