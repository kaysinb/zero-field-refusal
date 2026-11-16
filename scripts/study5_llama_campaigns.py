"""Study 5's order repeat on Llama-3.2-1B-Instruct (docs/study5-llama-registration.md) on one
L40S Pod: one detached driver of three plans in order, built and run by
``scripts/study5_campaigns.py``'s machinery (its ``Builder``, ``execute``, deadlines, archives,
receipts, restarts and close), each later plan built from the record the one before wrote.

    python scripts/study5_llama_campaigns.py run --out DIR --archive-id llama-...
        --checkpoints DIR [--dry-run [--checks FILE [--rates FILE]]]

The plans' campaign folders are DIR-checks, DIR-pilot and DIR-main (siblings). The other
commands (``checks-record``, ``table``, ``rates``) are steps the plans run; each prints numbers,
ids, paths and hashes only and exits 0 on success, 1 otherwise.

**The checks plan** (``checks_plan``):

- ``engineering``: the dose Pods' check (pod 2's order control of Qwen's seed 50000 at η_l rerun
  and compared under extension 2's numerical equivalence): the machine and the code reproduce the
  study's runs; any failure stops.
- ``instruments``: Llama's own K panels (``kl_harmful``, ``kl_benign``, ``kl_monitor``), sampled
  from its θ₀ by ``scripts/generate_teacher.py`` under the runs' config; any failure stops.
- ``stage0``: ``scripts/stage0_checks.py experiments/study5_llama_stage0`` (the zero-field checks:
  the cap, the teacher gap, the mean-gradient scaling, the path field, the optimizer, memory), then
  ``checks-record``: its path-field verdict and the teacher-gap thresholds it proposes
  (``llama_orders.checks_record``); a failed check or a missing threshold stops.
- ``theta0``: θ₀'s evaluation on the 521 prompts (``experiments/study5_llama_theta0``, the
  launcher's ``theta0`` entry); a failure stops.
- ``streams``: per seed of ``llama_orders.SEEDS`` its training stream of 500 responses, archived
  with the K panels under ``data/`` in the bucket.

**The pilot plan** (``pilot_plan``, from the checks record): the K-only pilot, ``PILOT_SEEDS`` in
order 1 at each of ``PILOT_RATES`` with the record's thresholds; ``table``: each run's end and
K_benign at step 25; ``rates``: the registered rule (``llama_orders.rate_choice``: the rates of
K = 4.84 and 7.44, then the support of K\\* at the rounded rates) into the rates record; no
bracket or a failed support check stops.

**The main plan** (``main_plan``, from both records): per seed one launcher campaign of six runs
(three orders × the two rates), archived as the seed ends.

No checkpoint, no labeling. Each block and seed group is admitted only when ``ADMIT_FACTOR`` × its
minutes and the close fit before the Pod's cap; ``run`` starts a plan only when the one before
completed, and started again with the same arguments it skips what is done in each.
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
import study5_dose_campaigns as d  # noqa: E402  (its engineering block and pilot table rows)

from zero_field_refusal import code_identity  # noqa: E402
from zero_field_refusal import llama_orders as lo  # noqa: E402
from zero_field_refusal.paths import data_root, runs_root  # noqa: E402
from zero_field_refusal.tracking import git_state  # noqa: E402

SELF = "scripts/study5_llama_campaigns.py"
POD = "llama"
IDENTITY = "docs/study5-llama-code-identity.json"
REGISTRATION = "docs/study5-llama-registration.md"
MODEL_DIR = "teacher/meta-llama--Llama-3.2-1B-Instruct"  # Llama's streams under DATA_DIR
CHECKS_DIR = "checks/meta-llama--Llama-3.2-1B-Instruct"  # its path-field certificate there
INSTRUMENTS = ("kl_harmful", "kl_benign", "kl_monitor")
STAGE0_EXPERIMENT = "study5_llama_stage0"
# the pilot table of the amended grid (llama_orders.PILOT_RATES): pilot-table.json, the first
# grid's table, is archived where an object is written once
CHECKS, TABLE, RATES = "llama-checks.json", "pilot-table-amended.json", "rates.json"
# Minutes, assumed for the 1B model where no measurement exists (the 1.5B model's as an upper bound)
RUN, PILOT_RUN, STAGE0, INSTRUMENT, STREAM = c.F_RUN, 3.0, 60.0, 10.0, 3.0
c.PREFIXES[POD] = "llama-"  # own_root_problems


def builder(archive_id: str, out: Path, runs: Path, data: Path, checkpoints: Path) -> c.Builder:
    b = c.Builder(POD, archive_id, out, runs, data, checkpoints)
    b.admit = True  # each block and seed group admitted only when it and the close fit the cap
    return b


def theta0_args(seed: int) -> tuple[str, ...]:
    """The runs' config for a seed without training (the K panels and the training streams), as
    a θ₀ evaluation: ``run_type=theta0`` needs no teacher-gap threshold, which stage 0 sets only
    later; no key that locates a stream differs from a run's (tests/test_llama_campaigns.py)."""
    return (lo.THETA0_CONFIG, f"seed={seed}", "run_type=theta0")


def checks_plan(*, out: Path, runs: Path, data: Path, checkpoints: Path, archive_id: str,
                commit: str, gpu_code: str, repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The checks plan (module docstring)."""
    b = builder(archive_id, out, runs, data, checkpoints)
    d.engineering(b, gpu_code)

    b.begin()  # Llama's own K panels, sampled from its θ₀
    for kind in INSTRUMENTS:
        b.step(f"instruments/{kind}", ("scripts/generate_teacher.py", *theta0_args(lo.SEEDS[0]),
               "--stream", kind), on_fail="stop", exists_ok=True, minutes=INSTRUMENT)  # fmt: skip
    b.end("instruments")

    b.begin()  # the zero-field checks on Llama, then their record: a failed check stops
    record = b.record(CHECKS)
    # done once its record exists: a later --out (a new commit) never makes a second stage-0 run,
    # which the record step would refuse
    b.step("stage0/checks", ("scripts/stage0_checks.py", lo.STAGE0_CONFIG), on_fail="stop",
           outputs=(str(record),), minutes=STAGE0)  # fmt: skip
    b.step("stage0/record", (SELF, "checks-record", "--runs", b.runs, "--out", record),
           on_fail="stop", outputs=(str(record),))  # fmt: skip
    b.runs_archive("stage0/archive", paths=(b.rel(record), STAGE0_EXPERIMENT))
    # the path-field certificate every Llama run requires (prechecks.require_path_check), under
    # DATA_DIR/checks/: archived with the data, so that a restored Pod's runs find it
    b.archive("stage0/archive-certificate", root=b.data, remote="data", paths=(CHECKS_DIR,))
    b.end("stage0")

    b.begin()  # θ₀'s evaluation on the 521 prompts: the analysis needs it, so a failure stops
    cell = {"name": "s5-llama-theta0", "config": lo.THETA0_CONFIG, "entry": "theta0",
            "overrides": [], "seeds": [lo.SEEDS[0]]}  # fmt: skip
    _, theta0 = b.launch("theta0", "theta0", [cell], minutes=c.EVALUATION, on_fail="stop")
    b.runs_archive("theta0/archive", campaign=theta0)
    b.end("theta0")

    b.begin()  # the ten seeds' training streams, then every Llama stream to the bucket
    for seed in lo.SEEDS:
        b.step(f"streams/s{seed}/training", ("scripts/generate_teacher.py", *theta0_args(seed),
               "--stream", "training", "--seeds", seed, "--responses", c.UPDATES * c.BATCH),
               on_fail="stop", exists_ok=True, minutes=STREAM)  # fmt: skip
    paths = [f"{MODEL_DIR}/{kind}" for kind in INSTRUMENTS] + [f"{MODEL_DIR}/training"]
    b.archive("streams/archive", root=b.data, remote="data", paths=paths)
    b.end("streams")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


def read_checks(path: Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("kind") != "study5-llama-checks" or record.get("path_field_passed") is not True:
        raise c.StepError(f"{Path(path).name}: not a passed Llama checks record")
    lo.threshold_overrides(record)  # both thresholds present
    return record


def pilot_campaign(archive_id: str) -> str:
    return f"study5-{archive_id}-pilot"


def pilot_plan(*, out: Path, runs: Path, data: Path, checkpoints: Path, archive_id: str,
               checks: Path, commit: str, gpu_code: str, repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The pilot plan (module docstring), from the checks record at ``checks``."""
    thresholds = lo.threshold_overrides(read_checks(checks))
    b = builder(archive_id, out, runs, data, checkpoints)
    b.begin()  # the K-only pilot, its table and the rates it fixes; no bracket stops
    cells = [lo.cell(seed, lo.PILOT_ORDER, rate, thresholds, pilot=True)
             for seed in lo.PILOT_SEEDS for rate in lo.PILOT_RATES]  # fmt: skip
    _, campaign = b.launch("pilot", "pilot", cells, minutes=len(cells) * PILOT_RUN)
    assert campaign == pilot_campaign(archive_id)
    table, rates = b.record(TABLE), b.record(RATES)
    argv = [SELF, "table", "--runs", b.runs, "--campaign", campaign, "--out", table]
    for cell in cells:
        argv += ["--run", cell["seeds"][0], cell["overrides"][0].split("=", 1)[1], c.label(cell)]
    b.step("pilot/table", argv, on_fail="stop", outputs=(str(table),))
    b.step("pilot/rates", (SELF, "rates", "--table", table, "--out", rates), on_fail="stop",
           outputs=(str(rates),))  # fmt: skip
    b.runs_archive("pilot/archive", paths=(b.rel(table), b.rel(rates)), campaign=campaign)
    b.end("pilot")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


def read_rates(path: Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = record.get("rows") if isinstance(record, dict) else None
    if not isinstance(rows, list) or lo.rate_choice(rows) != record.get("choice"):
        raise c.StepError(f"{Path(path).name}: its rates are not the rule's on its rows")
    return record


def main_plan(*, out: Path, runs: Path, data: Path, checkpoints: Path, archive_id: str,
              checks: Path, rates: Path, commit: str, gpu_code: str,
              repo: Path = c.REPO) -> c.Plan:  # fmt: skip
    """The main plan (module docstring), from the checks and the rates records."""
    thresholds = lo.threshold_overrides(read_checks(checks))
    choice = read_rates(rates)["choice"]
    b = builder(archive_id, out, runs, data, checkpoints)
    b.begin()  # per seed: three orders at the two rates
    for seed in lo.SEEDS:
        group = f"s{seed}"
        cells = [lo.cell(seed, order, choice[eta], thresholds) for order, _ in lo.ORDER_OFFSETS
                 for eta in ("eta_l", "eta_u")]  # fmt: skip
        _, campaign = b.launch(f"main-{group}", "main", cells, group=group,
                               minutes=len(cells) * RUN)  # fmt: skip
        b.runs_archive(f"main/{group}/archive", campaign=campaign, group=group)
    b.end("main")
    b.close(records=(b.rel(b.records),))
    return b.plan(commit, gpu_code, None, c.surrogate_spec(c.SURROGATE, repo))


# ------------------------------------------------------------------ the steps


def stage0_summary(runs: Path) -> tuple[str, dict[str, Any]]:
    """The one stage-0 run of Llama under the runs root that wrote its path-field verdict."""
    folder = runs / STAGE0_EXPERIMENT
    found = []
    for run_dir in sorted(folder.iterdir()) if folder.is_dir() else []:
        try:
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(summary, dict) and "path_field" in summary:
            found.append((run_dir.name, summary))
    if len(found) != 1:
        raise c.StepError(f"{len(found)} finished stage-0 runs of Llama, not one")
    return found[0]


def cmd_checks_record(args) -> int:
    run_id, summary = stage0_summary(args.runs)
    try:
        record = lo.checks_record(summary, run_id)
    except ValueError as exc:
        raise c.StepError(str(exc)) from exc
    c.write_json(args.out, record | {"created_utc": c._now()})
    t = record["thresholds"]
    print(f"stage0 {run_id} path field passed; thresholds token {t['token_max_abs']!r} "
          f"response {t['response_mean_signed_abs']!r}; eos at cap {record['eos_fraction_at_cap']}")  # fmt: skip
    return 0


def cmd_table(args) -> int:
    rows = []
    for seed, rate, run_label in args.run:
        found = d.table_row(args.runs, args.campaign, "", "", run_label)
        rows.append({"seed": int(seed), "rate": float(rate), "run_id": found["run_id"],
                     "outcome": found["outcome"], "steps": found["steps"],
                     "stop": found["stop"], "complete": found["complete"], "K": found["K"]})  # fmt: skip
    c.write_json(args.out, {"kind": "study5-llama-pilot", "rows": rows, "created_utc": c._now()})
    for row in rows:
        print(f"pilot s{row['seed']} lr {row['rate']!r} {row['outcome']} "
              f"complete {row['complete']} K {row['K']}")  # fmt: skip
    return 0


def cmd_rates(args) -> int:
    rows = json.loads(args.table.read_text(encoding="utf-8"))["rows"]
    try:
        choice = lo.rate_choice(rows)
    except ValueError as exc:
        raise c.StepError(str(exc)) from exc
    c.write_json(args.out, {"kind": "study5-llama-rates", "rows": rows, "choice": choice,
                            "created_utc": c._now()})  # fmt: skip
    print(f"eta_l {choice['eta_l']!r} eta_u {choice['eta_u']!r} per seed {choice['per_seed']}")
    return 0


# ------------------------------------------------------------------ the driver


def preflight(plan: c.Plan, repo: Path = c.REPO) -> list[str]:
    """The checkout clean at the plan's commit, its GPU code ``IDENTITY``'s, the registration
    ``registered``, a runs root and archive id of the Pod's own, the bucket, the key, the
    folder."""
    problems = []
    state = git_state(repo)
    if state.get("dirty") is not False or state.get("commit") != plan.commit:
        problems.append("the checkout is not clean at the plan's commit")
    recorded = json.loads((repo / IDENTITY).read_text(encoding="utf-8"))
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
    if not args.archive_id.startswith("llama-"):
        raise c.StepError(f"--archive-id {args.archive_id}: the Llama Pod's starts with llama-")
    commit = git_state(c.REPO).get("commit") or "unknown"
    try:
        gpu_code = code_identity.gpu_code_identity(commit)["sha256"]
    except Exception:  # no commit: the preflight refuses
        gpu_code = "unknown"
    out = args.out.absolute()
    outs = {part: out.with_name(f"{out.name}-{part}") for part in ("checks", "pilot", "main")}
    where = {"runs": runs_root(), "data": data_root(), "checkpoints": args.checkpoints,
             "archive_id": args.archive_id, "commit": commit, "gpu_code": gpu_code}  # fmt: skip
    records = where["runs"] / "study5_records" / args.archive_id
    checks = args.checks or records / CHECKS
    rates = args.rates or records / RATES
    plans = [lambda: checks_plan(**where, out=outs["checks"]),
             lambda: pilot_plan(**where, out=outs["pilot"], checks=checks),
             lambda: main_plan(**where, out=outs["main"], checks=checks, rates=rates)]
    if args.dry_run:
        c.show(plans[0]())
        if args.checks is not None:
            c.show(plans[1]())
            if args.rates is not None:
                c.show(plans[2]())
        return c.COMPLETED
    c.install_sigterm()
    for build in plans:
        code = start(build())
        if code != c.COMPLETED:
            return code
    return c.COMPLETED


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive-id", required=True)
    p.add_argument("--checkpoints", type=Path, required=True)  # unused: no run saves one
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--checks", type=Path)  # the dry run's pilot plan, from a checks record
    p.add_argument("--rates", type=Path)  # and its main plan, from a rates record
    p = sub.add_parser("checks-record")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("table")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--run", nargs=3, action="append", required=True)  # SEED RATE LABEL
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("rates")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return parser


COMMANDS = {"run": cmd_run, "checks-record": cmd_checks_record, "table": cmd_table,
            "rates": cmd_rates}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        return COMMANDS[args.kind](args)
    except (c.StepError, ValueError, OSError) as exc:
        print(f"study5_llama_campaigns.py {args.kind}: {exc}")
        return c.ERROR


if __name__ == "__main__":
    sys.exit(main())
