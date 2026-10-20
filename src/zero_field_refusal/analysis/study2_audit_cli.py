"""Draw study 2's terminal audit of the judge, and estimate the judge's bias once its reference
labels are resolved (amendment 4; the audit design v2.1; analysis.study2_audit).

    python -m zero_field_refusal.analysis.study2_audit_cli draw --runs DIR [DIR ...] \
        --theta0 RUN_DIR --fixation FILE --judge NAME [--runs-root RUNS_DIR] \
        [--abandoned FILE] [--versions FILE] [--out DIR]
    python -m zero_field_refusal.analysis.study2_audit_cli agreements --sample DIR \
        --fixation FILE --first FILE --second FILE [--out FILE]
    python -m zero_field_refusal.analysis.study2_audit_cli estimate --sample DIR \
        --fixation FILE --reference FILE [--out FILE]

The command line lives in the analysis package, not in scripts/: every file under scripts/ but
``code_identity.ANALYSIS_SCRIPTS`` is a GPU path, and gate C changes none (tests/test_gate_c.py).

``draw`` takes the arguments of scripts/study2_short_summary.py with --judge, and computes the
summary's final analysis as the summary does (its functions, loaded from the script). It refuses
(exit 2) unless that final analysis is available: the confirmation has ended, the cohort blocks are
labeled and the primary has stopped; before, it reads no item's judge label. Then it reads the
judge's label of every response of each bracket run of the primary's stopping-look cohort and of
the mechanism cohort (each checked as analysis.study2_labels.read_endpoint_labels checks it, its
counts against that reader's), builds the frame (analysis.study2_audit.audit_frame), checks that
the frame's judge-rated Δ, C1 and C2 are the final analysis's to 1e-12, and draws the fixation
record's ``audit`` (sizes 400, 400 and 200; SRSWOR per stratum of arm × side × judge label;
``draw_audit``). It writes into --out, which must not exist (default
$RUNS_DIR/audit/study2-short-<UTC time>), the files of study 1's D7 check:

- ``sample.jsonl``: the sampled items in a shuffled order, ``{item_id, prompt, response}``;
- ``unique.jsonl``: each distinct (prompt, response) once, ``{item_id, prompt, response}``, the
  only file the reference labellers see;
- ``key.json``: per sampled item its origin (arm, block, seed, run, prompt id, bracket side,
  weight, prompts, the judge's label), stratum, N_h, n_h, π and unique id: hidden until the
  reference labels are resolved;
- ``frame.jsonl``: every frame item, the same fields and its cohorts (the estimates' coefficients
  and the worst-case placement read it): hidden like key.json;
- ``selection.json``: the audit's settings and seed, the fixation record's sha256, the cohorts,
  per arm and stratum N_h, n_h, π and the stream seed, the estimands with their cohorts' blocks,
  the strata that count and the judge-rated estimates, the dilution (sampled items in the
  stopping-look cohort per arm), and the items and sha256 of every other file.

It prints only counts. The files hold model responses to harmful prompts: keep them under
$RUNS_DIR (ignored by Git), archive them with an inventory, and hand the labellers unique.jsonl
alone.

``agreements`` reads the two labellers' labels of unique.jsonl's items (JSON objects
``{"item_id": label | null}``, GPT's as --first and Grok's as --second; null: a failed output)
and prints the counts, the disagreements and failed outputs to resolve, and the agreements the
researcher checks: ⌈0.1 × agreements⌉ drawn with the stream derive_seed(seed, "agreement_check")
of the fixation record's audit seed (analysis.study2_audit.agreement_check); --out writes the same
record (it must not exist). Item ids only, never a text.

``estimate`` reads the resolved reference labels (--reference: a JSON object
``{"item_id": label | null}`` over every item id of unique.jsonl, a label one of refusal,
safe_non_refusal or harmful_compliance, null a missing reference: failed or unresolved), checks
the files against selection.json's sha256 and the draw against the fixation record (the draw is
made again from frame.jsonl and must give key.json's items), and prints per estimand (Δ on the
stopping-look cohort, the formal reading; C1, C2 and each arm's harm, descriptive) the HT estimate
with its SE and Wald interval under both completions of the missing references, the
boundary-safe interval [L_B, U_B] at the record's bias_level, and for Δ, C1 and C2 the reference
harm of the realized cohort [judge point − U_B, judge point − L_B] with its readings; then per
stratum N_h, n_h, m_h and the missing references. --out writes the whole record (it must not
exist). Numbers only, never a text.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from zero_field_refusal.analysis.evidence import HARMFUL, label_item_id, label_path
from zero_field_refusal.analysis.study2_audit import (
    COHORTS,
    ESTIMANDS,
    agreement_check,
    audit_cohorts,
    audit_estimates,
    audit_frame,
    bracket_runs,
    coefficients,
    draw_audit,
    estimands,
    judge_point,
    reference_value,
    stratum_counts,
)
from zero_field_refusal.analysis.study2_labels import _labels, _responses, read_endpoint_labels
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    FA,
    STEPS,
    VERSIONS_RECORD,
    read_fixation,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.data import load_prompt_set
from zero_field_refusal.judge import judge_identity
from zero_field_refusal.judge_check import unique_texts
from zero_field_refusal.labeling import write_calibration_files
from zero_field_refusal.paths import REPO_ROOT, runs_root
from zero_field_refusal.run_judging import scheduled_conditions
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.tracking import git_state

SUMMARY = REPO_ROOT / "scripts" / "study2_short_summary.py"
ENDPOINT = f"step-{STEPS}"
FILES = ("sample.jsonl", "unique.jsonl", "key.json", "frame.jsonl")
TOLERANCE = 1e-12  # the frame's judge-rated estimates against the final analysis's
KEY_FIELDS = ("arm", "block", "seed", "run", "prompt", "side", "w", "prompts", "judge",
              "y_judge", "judge_label", "stratum")  # fmt: skip


class Refused(Exception):
    """The audit refuses to act (exit 2), with the reason."""


def summary_module():
    """scripts/study2_short_summary.py as a module: the audit computes its final analysis by its
    functions, never by a copy of them."""
    spec = importlib.util.spec_from_file_location("study2_short_summary", SUMMARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ draw


def final_state(args: argparse.Namespace, summary) -> tuple:
    """The fixation record, the stage's state, the runs, the looks and the final analysis, as
    scripts/study2_short_summary.py computes them with --judge (its ``main``, step for step, on
    its functions)."""
    try:
        fixation = summary.read_fixation(args.fixation)
    except (OSError, ValueError) as exc:
        raise Refused(str(exc)) from exc
    problems = summary.versions_problems(args.versions, fixation=args.fixation, judge=args.judge)
    if problems:
        raise Refused("the versions record does not hold: " + "; ".join(problems))
    abandoned = {}
    if args.abandoned is not None:
        try:
            abandoned = summary.read_abandonment(args.abandoned)
        except (OSError, ValueError) as exc:
            raise Refused(str(exc)) from exc
    record = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: record.get(key) for key in ("benchmark_accuracy", "factual_accuracy")}
    if summary.theta0_problems(theta0):
        raise Refused("; ".join(summary.theta0_problems(theta0)))
    loaded = summary.load_trajectories(*args.runs)
    facts = {r.run_id: summary.run_facts(r.run_dir) for r in loaded
             if r.arm in (*summary.ARMS, summary.FA_CONTROL_ARM)}  # fmt: skip
    runs, courses = summary.split_timecourse(loaded, facts)
    records = summary.launch_records(args.runs_root) if args.runs_root is not None else []
    try:
        exhausted = summary.exhausted_runs(records, abandoned=abandoned)
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    numerical = summary.read_numerical(args.runs_root) if args.runs_root is not None else None
    state = summary.progress(runs, theta0, fixation["w_max"], facts=facts, exhausted=exhausted,
                             numerical=numerical, timecourse=courses,
                             issued_w_max=summary.issued_w_max(records))  # fmt: skip
    labels, pending = summary.read_labels(state, runs, args.judge)
    found = summary.compute_looks(state, runs, labels, fixation, theta0, pending)
    final = summary.final_analysis(state, runs, courses, labels, fixation, found, args.judge,
                                   exhausted)  # fmt: skip
    return fixation, state, runs, found, final


def item_labels(run_dir: Path, judge: str, identity, prompts) -> tuple[dict, dict]:
    """The judge's label of each harmful prompt at a run's endpoint, by prompt id in the set's
    order (None: no parsed label), and the sha256 of the files read. The run is checked first as
    ``read_endpoint_labels`` checks it (raising ``LabelsPending`` or ``ValueError``); every file
    read here must be the one it read, and the labels must give its counts."""
    rows = read_endpoint_labels(run_dir, judge, identity, prompts)
    row = next((r for r in rows if r["condition"] == ENDPOINT), None)
    if row is None:
        raise ValueError(f"{run_dir}: no label row of {ENDPOINT}")
    files = row["files"]

    def read(path: Path) -> bytes:
        data = path.read_bytes()
        name = path.relative_to(run_dir).as_posix()
        if files.get(name) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"{path} is not the file the label reader checked")
        return data

    cfg = OmegaConf.create(read(run_dir / "config.yaml").decode())
    conditions = scheduled_conditions(run_dir, read)
    limit = cfg.eval.harmful.get("limit")
    ids = list(prompts)[:limit] if limit else list(prompts)
    texts = _responses(run_dir, cfg, conditions, ids, read)
    items = {label_item_id(c, pid): (prompts[pid], text) for (c, pid), text in texts.items()}
    path = label_path(run_dir, judge)
    labels, _ = _labels(path, read(path), identity, items)
    found = {pid: labels.get(label_item_id(ENDPOINT, pid)) for pid in ids}
    counted = Counter(found.values())
    if (len(found), counted[HARMFUL], counted[None]) != (row["prompts"], row["harmful"],
                                                         row["missing"]):  # fmt: skip
        raise ValueError(f"{run_dir}: the item labels do not give the reader's counts")
    return found, dict(files)


def endpoint_texts(run_dir: Path, files: dict[str, str]) -> dict[str, str]:
    """The responses at a run's endpoint by prompt id, from the responses file the label reader
    checked (its sha256 in ``files``)."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    path = run_dir / "responses" / f"{cfg.eval.harmful.prompt_set}.jsonl"
    data = path.read_bytes()
    if files.get(path.relative_to(run_dir).as_posix()) != hashlib.sha256(data).hexdigest():
        raise ValueError(f"{path} is not the file the label reader checked")
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    return {r["item_id"]: r["text"] for r in rows if r["condition"] == ENDPOINT}


def prompt_texts(run_dir: Path, cache: dict[str, dict[str, str]]) -> dict[str, str]:
    """The harmful prompt set of a run by prompt id, in the set's order, read once per set."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    name = str(cfg.eval.harmful.prompt_set)
    if name not in cache:
        cache[name] = {item["id"]: item["text"] for item in load_prompt_set(name, cfg.data)}
    return cache[name]


def _order(items: list[dict[str, Any]], seed: int) -> list[tuple[str, dict[str, Any]]]:
    """The sampled items shuffled across strata, each with an opaque id (as D7's)."""
    rng = random.Random(derive_seed(seed, "order"))
    shuffled = list(items)
    rng.shuffle(shuffled)
    named, used = [], set()
    for item in shuffled:
        item_id = f"c{rng.getrandbits(48):012x}"
        while item_id in used:
            item_id = f"c{rng.getrandbits(48):012x}"
        used.add(item_id)
        named.append((item_id, item))
    return named


def _plain(stratum: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stratum.items() if k not in ("members", "sampled")}


def draw(args: argparse.Namespace) -> dict[str, Any]:
    summary = summary_module()
    fixation, state, runs, found, final = final_state(args, summary)
    try:  # the gate: the summary's final analysis; no item label is read before it
        cohorts = audit_cohorts(state, found, final)
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    audit = fixation["audit"]
    by_id = {run.run_id: run for run in runs}
    identity = judge_identity(load_config(f"judge/{args.judge}"))
    cache: dict[str, dict[str, str]] = {}
    labels, files = {}, {}
    for run_id in bracket_runs(cohorts):
        run_dir = by_id[run_id].run_dir
        prompts = prompt_texts(run_dir, cache)
        labels[run_id], files[run_id] = item_labels(run_dir, args.judge, identity, prompts)
    frame = audit_frame(cohorts, labels)
    points = {}
    for name, expected in cohorts["judge"].items():
        points[name] = judge_point(frame, coefficients(frame, name))
        if expected is None or abs(points[name] - expected) > TOLERANCE:
            raise ValueError(f"the frame's judge-rated {name} {points[name]!r} is not the final "
                             f"analysis's {expected!r}")  # fmt: skip
    drawn = draw_audit(frame, audit["sizes"], audit["seed"])
    texts: dict[str, dict[str, str]] = {}
    blinded, key = [], {}
    for item_id, entry in _order(drawn["items"], audit["seed"]):
        item = frame[entry["index"]]
        run_dir = by_id[item["run"]].run_dir
        if item["run"] not in texts:
            texts[item["run"]] = endpoint_texts(run_dir, files[item["run"]])
        prompts = prompt_texts(run_dir, cache)
        blinded.append({"item_id": item_id, "prompt": prompts[item["prompt"]],
                        "response": texts[item["run"]][item["prompt"]]})  # fmt: skip
        stratum = next(s for s in drawn["strata"] if s["stratum"] == item["stratum"])
        key[item_id] = {"frame_index": item["index"], **{k: item[k] for k in KEY_FIELDS},
                        "cohorts": list(item["cohorts"]), "N": stratum["N"], "n": stratum["n"],
                        "pi": entry["pi"]}  # fmt: skip
    unique, unique_of = unique_texts(blinded, 1)
    for item_id, unique_id in unique_of.items():
        key[item_id]["unique_id"] = unique_id
    sampled_as = {entry["frame_index"]: item_id for item_id, entry in key.items()}
    primary = {b["block"] for b in cohorts["primary"]}
    out = args.out
    out.mkdir(parents=True)
    write_calibration_files(blinded, key, out / "sample.jsonl", out / "key.json")
    (out / "unique.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False) + "\n"
                                              for u in unique))  # fmt: skip
    (out / "frame.jsonl").write_text(
        "".join(
            json.dumps({**item, "sampled": sampled_as.get(item["index"])}) + "\n" for item in frame
        )
    )
    record = {
        "what": "Study 2's terminal audit of the judge (amendment 4; the audit design v2.1).",
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": sys.argv,
        "code": git_state(),
        "audit": audit,
        "fixation": {"path": str(args.fixation), "sha256": _sha256(args.fixation)},
        "versions_record": {"path": str(args.versions), "sha256": _sha256(args.versions)},
        "K_star": cohorts["K_star"],
        "three_arm": cohorts["three_arm"],
        "stopping_look": cohorts["stopping_look"],
        "cohorts": {cohort: None if cohorts[cohort] is None else [b["block"] for b in
                                                                   cohorts[cohort]]
                    for cohort in COHORTS},
        "arms": drawn["arms"],
        "strata": [_plain(s) for s in drawn["strata"]],
        "estimands": {name: {"cohort": ESTIMANDS[name]["cohort"],
                             "blocks": len({i["block"] for i in frame
                                            if ESTIMANDS[name]["cohort"] in i["cohorts"]}),
                             "judge_point": judge_point(frame, coefficients(frame, name))}
                      for name in estimands(frame)},
        "dilution": {arm: sum(1 for e in drawn["items"] if frame[e["index"]]["arm"] == arm
                              and frame[e["index"]]["block"] in primary)
                     for arm in (ADAMW, FA)},
        "files": {name: {"items": len((out / name).read_text().splitlines()) if name.endswith(
            ".jsonl") else len(key), "sha256": _sha256(out / name)} for name in FILES},
    }  # fmt: skip
    (out / "selection.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def draw_counts(record: dict[str, Any], out: Path) -> dict[str, Any]:
    """What ``draw`` prints: counts only."""
    return {"out": str(out), "arms": record["arms"],
            "strata": [{k: s[k] for k in ("stratum", "N", "n")} for s in record["strata"]],
            "cohorts": {k: None if v is None else len(v) for k, v in record["cohorts"].items()},
            "stopping_look": record["stopping_look"],
            "sampled_in_the_stopping_look_cohort": record["dilution"],
            "files": {k: v["items"] for k, v in record["files"].items()}}  # fmt: skip


# ------------------------------------------------------------------ the sample's files


def read_sample(directory: Path, fixation: dict[str, Any]) -> dict[str, Any]:
    """The draw's files, checked: their sha256 against selection.json's, its audit settings
    against the fixation record's, and the draw made again from frame.jsonl against key.json."""
    selection = json.loads((directory / "selection.json").read_text())
    for name in FILES:
        if _sha256(directory / name) != selection["files"][name]["sha256"]:
            raise Refused(f"{directory / name} is not the file the draw wrote")
    if selection["audit"] != fixation["audit"]:
        raise Refused("the fixation record's audit is not the one the draw used")
    frame = [json.loads(line) for line in (directory / "frame.jsonl").read_text().splitlines()]
    for item in frame:
        item.pop("sampled")
    key = json.loads((directory / "key.json").read_text())
    drawn = draw_audit(frame, fixation["audit"]["sizes"], fixation["audit"]["seed"])
    if sorted(e["index"] for e in drawn["items"]) != sorted(e["frame_index"]
                                                            for e in key.values()):  # fmt: skip
        raise Refused("the draw made again from frame.jsonl is not key.json's")
    return {"selection": selection, "frame": frame, "key": key, "drawn": drawn}


def _labels_file(path: Path) -> dict[str, Any]:
    labels = json.loads(path.read_text())
    if not isinstance(labels, dict):
        raise Refused(f"{path}: a JSON object {{item_id: label | null}}")
    return labels


def _unique_ids(key: dict[str, Any]) -> set[str]:
    return {entry["unique_id"] for entry in key.values()}


def agreements(args: argparse.Namespace) -> dict[str, Any]:
    fixation = read_fixation(args.fixation)
    sample = read_sample(args.sample, fixation)
    first, second = _labels_file(args.first), _labels_file(args.second)
    wanted = _unique_ids(sample["key"])
    for name, labels in (("--first", first), ("--second", second)):
        if set(labels) != wanted:
            raise Refused(f"{name} does not label exactly unique.jsonl's items")
    try:
        found = agreement_check(first, second, fixation["audit"]["agreement_check_share"],
                                fixation["audit"]["seed"])  # fmt: skip
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    return {"sample": str(args.sample), "first": {"path": str(args.first),
                                                  "sha256": _sha256(args.first)},
            "second": {"path": str(args.second), "sha256": _sha256(args.second)},
            "counts": {k: len(found[k]) for k in ("agreements", "disagreements", "failed",
                                                  "check")},
            **found}  # fmt: skip


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    fixation = read_fixation(args.fixation)
    sample = read_sample(args.sample, fixation)
    resolved = _labels_file(args.reference)
    wanted = _unique_ids(sample["key"])
    if set(resolved) != wanted:
        missing, extra = sorted(wanted - set(resolved)), sorted(set(resolved) - wanted)
        raise Refused(f"--reference must give every item id of unique.jsonl (null when missing): "
                      f"{len(missing)} absent, {len(extra)} unknown")  # fmt: skip
    try:
        values = {uid: reference_value(label) for uid, label in resolved.items()}
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    reference = {entry["frame_index"]: values[entry["unique_id"]]
                 for entry in sample["key"].values()}  # fmt: skip
    level = fixation["audit"]["bias_level"]
    found = audit_estimates(sample["frame"], sample["drawn"], reference, level)
    selection = sample["selection"]
    return {"sample": str(args.sample),
            "reference": {"path": str(args.reference), "sha256": _sha256(args.reference)},
            "audit": fixation["audit"], "stopping_look": selection["stopping_look"],
            "cohorts": {k: None if v is None else len(v)
                        for k, v in selection["cohorts"].items()},
            "estimands": found, "strata": stratum_counts(sample["drawn"], reference),
            "missing_references": sum(1 for v in reference.values() if v is None)}  # fmt: skip


def estimate_shown(record: dict[str, Any]) -> dict[str, Any]:
    """What ``estimate`` prints: per estimand the HT estimates, the bounds and the reference
    harm; per stratum its counts."""
    shown = {}
    for name, entry in record["estimands"].items():
        shown[name] = {
            "status": entry["status"],
            "cohort": entry["cohort"],
            "blocks": entry["blocks"],
            "strata": entry["bounds"]["K"],
            "ht": {
                c: {k: entry["ht"][c][k] for k in ("estimate", "se", "wald")}
                for c in ("lower", "upper")
            },
            "bounds": entry["bounds"]["interval"],
            "level": entry["bounds"]["level"],
        }
        if "reference_harm" in entry:
            harm = entry["reference_harm"]
            shown[name]["reference_harm"] = {
                k: harm[k] for k in ("judge_point", "interval", "readings", "conditional")
            }
    return {"cohorts": record["cohorts"], "stopping_look": record["stopping_look"],
            "estimands": shown, "strata": record["strata"],
            "missing_references": record["missing_references"]}  # fmt: skip


# ------------------------------------------------------------------ main


def _tracked_tree(path: Path) -> bool:
    """Whether ``path`` lies in the checkout outside its ignored runs/ directory."""
    found, root = path.resolve(), REPO_ROOT.resolve()
    return found.is_relative_to(root) and not found.is_relative_to(root / "runs")


def _write(path: Path | None, record: dict[str, Any]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m zero_field_refusal.analysis.study2_audit_cli",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("draw", help="draw the audit once the final analysis has run")
    one.add_argument("--runs", type=Path, nargs="+", required=True)
    one.add_argument("--theta0", type=Path, required=True)
    one.add_argument("--fixation", type=Path, required=True)
    one.add_argument("--judge", required=True, help="a judge config in configs/judge/")
    one.add_argument("--runs-root", type=Path, help="RUNS_DIR, whose launch/ journal is read")
    one.add_argument("--abandoned", type=Path, help="the abandonment record (JSON)")
    one.add_argument("--versions", type=Path, default=REPO_ROOT / VERSIONS_RECORD)
    one.add_argument("--out", type=Path, help="a new directory under $RUNS_DIR")
    two = sub.add_parser("agreements", help="the agreements the researcher checks")
    three = sub.add_parser("estimate", help="the bias and the reference harm")
    for later in (two, three):
        later.add_argument("--sample", type=Path, required=True, help="the draw's directory")
        later.add_argument("--fixation", type=Path, required=True)
        later.add_argument("--out", type=Path, help="the whole record (JSON); must not exist")
    two.add_argument("--first", type=Path, required=True, help="GPT's labels (JSON)")
    two.add_argument("--second", type=Path, required=True, help="Grok's labels (JSON)")
    three.add_argument("--reference", type=Path, required=True,
                       help="the resolved reference labels (JSON)")  # fmt: skip
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "draw" and args.out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.out = runs_root() / "audit" / f"study2-short-{stamp}"
    if args.out is not None and args.out.exists():
        parser.error(f"{args.out} exists: the audit's files are written once")
    if args.command == "draw" and _tracked_tree(args.out):
        parser.error(f"{args.out} lies in the checkout outside runs/: the audit's files hold "
                     "model responses and never enter Git; write them under $RUNS_DIR")  # fmt: skip
    try:
        if args.command == "draw":
            record = draw(args)
            print(json.dumps(draw_counts(record, args.out), indent=1))
        elif args.command == "agreements":
            record = agreements(args)
            print(json.dumps({k: v for k, v in record.items() if k != "agreements"}, indent=1))
            _write(args.out, record)
        else:
            record = estimate(args)
            print(json.dumps(estimate_shown(record), indent=1))
            _write(args.out, record)
    except Refused as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
