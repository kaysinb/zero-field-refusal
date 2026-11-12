"""Study 5's extension 2 analysis on a CPU (docs/study5-preregistration.md, "Extension 2
(2026-11-11)" and "Extension 2 amendment"): pod 4's records, the run directories they name and
pod 2's registered analysis record read into one JSON record of every reported quantity, the δ_lin
validation criterion among them. The computations and the reading rules are
``analysis.study5_ext2``'s.

    python -m zero_field_refusal.analysis.study5_ext2_cli --records DIR --runs-root DIR
        [--runs-root DIR ...] --analysis FILE --out FILE [--pair-floors K]

**Inputs.** --records is pod 4's records folder, ``<runs>/study5_records/pod4-<date>/`` as
``scripts/study5_campaigns.py`` (``pod4_plan``) lays it out; its name is pod 4's archive id. Each
--runs-root is a local runs root (pod 4's restored ``runs/`` and pod 2's analysis root): the run
directories ``<root>/<experiment>/<run id>``, the launcher journals ``<root>/launch/`` and pod 2's
records. --analysis is pod 2's registered analysis record (``study5-final/analysis.json`` of
``analysis.study5_analysis_cli``), refused unless it has the sha256 the registration names. From
it: h_i and its support (``outcomes``), the six order controls' harm and K_harmful differences
(``descriptive.controls.order.per_seed``, control minus own η_l run), the stream order's η_l run
(``descriptive.line_scans.per_seed``, its +1) and pod 2's surrogate endpoint records, each found
under a runs root by the path and checked against the sha256 the analysis read it with
(``inputs``).

From the records folder: ``vectors/m1/record.json`` (M1, required); ``linear/s<seed>/record.json``
with ``linear/pairs-hat-s<seed>.json`` and ``pairs-lin-…`` (M2, the block required: at least one
record); ``linear-repeat/`` (the amendment's repeat); ``linear-control/s<seed>/`` (the order
controls' orders); ``sealed/s<seed>-order<k>/`` (E2's predictions, numbers only); ``edits/
s<seed>-<tag>/`` (the edited orders' predictions); ``points/`` (M3's small amplitudes, M4's
points, θ₀ + δ_lin); ``tokens/``; ``force/`` and ``force-repeat/`` (M5 and its floor);
``twins/`` (the twins and their floor); ``replays/check-g<seed>-endpoint-analysis-0.001.json``
(a replay group's check accepted for endpoint analysis under extension 2's third amendment, its
current verdict; the group's ``replays/check-g<seed>.json`` and
``replays/check-g<seed>-endpoint-0.001.json`` under the earlier rules, where written, are read as
history beside it and are not missed when absent) and ``engineering-check-numerical.json`` (the
replay checks). From a runs root, optionally: ``study5_diag/s<seed>/pairs.json``, diagnostic
repeats of a replay's config (their step-25 distances enter e_i of ``endpoint_bounds``, extension
2's second amendment; a seed without one is not missed). An optional record that is absent, does
not read, is not of its kind or seed, or records its own absence (``available: false``) is null
in its place with the reason, listed under ``missing`` too.

**Run directories** are read only through the records that name them: a vector run
(``<experiment>/<run id>`` of its record) and an edited run (the launcher journal of pod 4's
campaign ``study5-<ID>-swaps-s<seed>`` or ``…-shuffles-…``, for a twin ``…-replays-g<seed>`` of
its group, and its cell's label, the run its last end ``done`` or ``diverged`` names; its config
the seed's at η_l, F, with the edit its record names). **Extension 1's order repeats (E2) are
sealed**: before anything else of a run directory is read, its ``config.yaml`` is; a run of an E2
cell (its name, or an E2 seed with ``train.order_seed`` = seed + 1000 or + 2000,
``study5_ext2.e2_repeat``) refuses the analysis. The only E2 input is the sealed block's
predictions from θ₀.

**Labels** are read by the registered reader (``study5_analysis_cli.Labels``: WildGuard, the
prompt texts from ``DATA_DIR``), imported only here: its module imports the GPU code's modules,
torch among them, though nothing is computed with them, and ``analysis.study5_ext2`` stays
torch-free. A run without labels (before the judge) gives its harm as null with the reason.

**The criterion's part 3** counts a pair whose |Δq| is below k floors of Δq (the linear
repeat's) as a tie; k is ``--pair-floors`` (default ``study5_ext2.PAIR_FLOORS``, 10), recorded as
``criterion.part3_pairs.unresolved_floors``.

**Refusals** (exit 1, nothing written): a checkout that is not clean or of no known commit; --out
exists; --pair-floors not a finite positive number; the records folder, ``vectors/m1/
record.json`` or the linear block missing; the analysis record not the registered one; an E2 run
directory. **The record** (--out, never overwritten): every quantity with its reasons, every
input's sha256 (``inputs``), the checkout's git state. Strict JSON; numbers, ids, states and
hashes only, never a prompt, a response or a label. stdout carries numbers, ids and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..tracking import git_state
from . import study5_analysis as sa
from . import study5_ext2 as ext2

PROG = "study5_ext2_cli"
# pod 2's registered analysis record (docs/study5-preregistration.md, "Corrections and extensions")
REGISTERED_ANALYSIS_SHA256 = "d03f0450108f57310918843d8b2469384b744cb420dc15964e14ffcb3f49798f"
ENDPOINT_INPUT = re.compile(r"^runs:(study5_records/[^/]+/surrogate-endpoint/s([0-9]+)\.json)$")
ENDPOINT = "step-25"
RATE = f"{ext2.ETA_L!r}"  # the cells' rate as their names hold it


class Refused(Exception):
    """The analysis cannot stand: nothing is written."""


def refuse(why: str) -> None:
    raise Refused(why)


def registered():
    """The registered analysis's readers (imported here: their module imports the GPU code's)."""
    from . import study5_analysis_cli

    return study5_analysis_cli


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def string_keys(value: Any) -> Any:
    """Every mapping key as a string, so that the record sorts and stays strict JSON."""
    if isinstance(value, Mapping):
        return {str(k): string_keys(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [string_keys(v) for v in value]
    return value


def encoded(record: Any) -> bytes:
    data = json.dumps(string_keys(sa.finite_tree(record)), indent=1, sort_keys=True,
                      allow_nan=False, ensure_ascii=False)  # fmt: skip
    return (data + "\n").encode("utf-8")


# ------------------------------------------------------------------ reading


class Inputs:
    """The records folder, the runs roots and the registered readers, every file read hashed."""

    def __init__(self, reg, records: Path, roots: list[Path], analysis: Path) -> None:
        self.reg, self.records = reg, Path(records)
        self.roots = [Path(r) for r in roots]
        names = {"records": self.records,
                 **{f"runs{i}": r for i, r in enumerate(self.roots, 1)},
                 "analysis": Path(analysis).parent}  # fmt: skip
        self.reader = reg.Reader(names)
        self.missing: list[dict[str, str]] = []
        self.labels = reg.Labels(self.reader, reg.prompt_sets(self.reader))

    def why(self, what: str, reason: str) -> str:
        self.missing.append({"what": what, "reason": reason})
        return reason

    def record(self, rel: str, kind: str | None = None, seed: int | None = None
               ) -> dict[str, Any] | str:  # fmt: skip
        """A record of the records folder, or why it is not there (named under ``missing``)."""
        path = self.records / rel
        if not path.is_file():
            return self.why(rel, f"{rel}: absent")
        try:
            found = self.reader.json(path)
        except (OSError, ValueError) as exc:
            return self.why(rel, f"{rel}: does not read ({type(exc).__name__})")
        if not isinstance(found, dict):
            return self.why(rel, f"{rel}: not a record")
        if found.get("available") is False:
            return self.why(rel, f"{rel}: unavailable ({found.get('missing')}: "
                                 f"{found.get('reason')})")  # fmt: skip
        if kind is not None and found.get("kind") != kind:
            return self.why(rel, f"{rel}: not a {kind} record")
        if seed is not None and found.get("seed") != seed:
            return self.why(rel, f"{rel}: not of seed {seed}")
        return found

    # -------------------------------------------------------------- run directories

    def guarded(self, run_dir: Path) -> dict[str, Any] | str:
        """The run's config, read before anything else of it; an E2 order repeat refuses."""
        try:
            config = yaml.safe_load(self.reader.bytes(run_dir / "config.yaml"))
        except (OSError, yaml.YAMLError) as exc:
            config = None
            why = f"{run_dir.name}: its config does not read ({type(exc).__name__})"
        else:
            why = None if isinstance(config, dict) else f"{run_dir.name}: its config is not a map"
        sealed = ext2.e2_repeat(config if isinstance(config, dict) else None, run_dir.name)
        if sealed:
            refuse(f"{sealed}: sealed, never read")
        return why if why else config

    def run_dir(self, experiment: Any, run_id: Any) -> tuple[Path | str, dict | None]:
        """A run directory under the first runs root that has it, guarded."""
        if not all(isinstance(x, str) and x and "/" not in x and x not in (".", "..")
                   for x in (experiment, run_id)):  # fmt: skip
            return f"run {experiment!r}/{run_id!r}: not a run directory name", None
        for root in self.roots:
            found = root / experiment / run_id
            if found.is_dir():
                config = self.guarded(found)
                return (found, config) if isinstance(config, dict) else (config, None)
        return f"{experiment}/{run_id}: under no runs root", None

    def edited_run(self, campaign: str, label: str) -> tuple[Path | str, dict | None]:
        """The run of a launcher cell of pod 4 (its journal under a runs root), guarded."""
        reasons = []
        for root in self.roots:
            if not (root / "launch" / f"{campaign}.jsonl").is_file():
                continue
            found = self.reg.completed(self.reader, root, (campaign, label), label)
            if isinstance(found, Path):
                config = self.guarded(found)
                return (found, config) if isinstance(config, dict) else (config, None)
            reasons.append(found)
        return (reasons[0] if reasons else f"{label}: no journal of {campaign}"), None

    def counts(self, run_dir: Path | str, condition: str) -> Mapping[str, Any] | str:
        """The registered label counts of a guarded run's condition, or why there are none."""
        if isinstance(run_dir, str):
            return run_dir
        return self.labels.at(run_dir, condition)


# ------------------------------------------------------------------ the registered analysis


def registered_analysis(inputs: Inputs, path: Path) -> tuple[dict[str, Any], str]:
    try:
        data = inputs.reader.bytes(Path(path))
        found = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse(f"{Path(path).name}: the registered analysis record does not read "
               f"({type(exc).__name__})")  # fmt: skip
    digest = sha256(data)
    if digest != REGISTERED_ANALYSIS_SHA256:
        refuse(f"{Path(path).name}: sha256 {digest}, not the registered analysis record's "
               f"{REGISTERED_ANALYSIS_SHA256}")  # fmt: skip
    if not isinstance(found, dict) or found.get("kind") != "study5-analysis":
        refuse(f"{Path(path).name}: not a study5-analysis record")
    return found, digest


def h_of(analysis: Mapping[str, Any]) -> dict[int, float | None]:
    """h_i of every supported seed with a value (``outcomes``), None otherwise."""
    out = {}
    for o in analysis.get("outcomes", []):
        supported = (o.get("support") or {}).get("state") == "supported"
        out[int(o["seed"])] = ext2.number(o.get("h")) if supported else None
    return out


def per_seed_measure(per_seed: Mapping[str, Any], measure: str, seed: int) -> Any:
    found = (per_seed.get(measure) or {}).get(str(seed))
    return found.get("missing") if isinstance(found, Mapping) else found


def endpoint_r(inputs: Inputs, analysis: Mapping[str, Any]) -> dict[int, float | str]:
    """r(θ₂₅) of pod 2's surrogate endpoint records, each the file the registered analysis read
    (its path and sha256 in the analysis's ``inputs``), found under a runs root."""
    out: dict[int, float | str] = {}
    for key, digest in (analysis.get("inputs") or {}).items():
        match = ENDPOINT_INPUT.match(key)
        if not match:
            continue
        rel, seed = match.group(1), int(match.group(2))
        paths = [root / rel for root in inputs.roots if (root / rel).is_file()]
        if not paths:
            out[seed] = inputs.why(rel, f"{rel}: under no runs root")
            continue
        data = inputs.reader.bytes(paths[0])
        if sha256(data) != digest:
            out[seed] = inputs.why(rel, f"{rel}: not the file the registered analysis read")
            continue
        found = json.loads(data)
        value = ext2.number(found.get("r")) if found.get("kind") == (
            "study5-surrogate-endpoint") and found.get("seed") == seed else None  # fmt: skip
        out[seed] = value if value is not None else inputs.why(rel, f"{rel}: no r of seed {seed}")
    return out


def diagnostic_repeats(inputs: Inputs) -> dict[int, dict[str, Any] | str]:
    """Diagnostic repeats of a replay's config (``study5_diag/s<seed>/pairs.json`` under a runs
    root, a ``study5-pairs`` record), by seed; a seed without one is left out, not missed. Copies
    under several roots: byte-identical ones are one record; differing ones are combined, their
    rows together (e_i takes the largest), with ``copies`` the number of distinct ones. A copy
    that does not read, or is not such a record, gives the reason instead."""
    out: dict[int, dict[str, Any] | str] = {}
    for s in ext2.SEEDS:
        rel = f"study5_diag/s{s}/pairs.json"
        paths = [root / rel for root in inputs.roots if (root / rel).is_file()]
        if not paths:
            continue
        found: dict[str, Any] = {}
        why = None
        for path in paths:
            try:
                data = inputs.reader.bytes(path)
                record = json.loads(data)
            except (OSError, ValueError) as exc:
                why = f"{rel}: does not read ({type(exc).__name__})"
                break
            if not isinstance(record, dict) or record.get("kind") != "study5-pairs":
                why = f"{rel}: not a study5-pairs record"
                break
            found.setdefault(sha256(data), record)
        if why is not None:
            out[s] = inputs.why(rel, why)
        elif len(found) == 1:
            out[s] = next(iter(found.values()))
        else:
            rows = [row for record in found.values() for row in record.get("pairs") or []]
            out[s] = {"kind": "study5-pairs", "pairs": rows, "copies": len(found)}
    return out


# ------------------------------------------------------------------ the sections


def stream_order(inputs: Inputs, rel: str, seed: int) -> dict[str, Any] | str:
    """A linear-replay record of the seed's own (stream) order: its ``order`` is ``"stream"``, as
    ``study5_linear_replay.py`` writes it for a run without ``train.order_seed`` or an edit;
    another order is an unresolved input (named under ``missing``)."""
    record = inputs.record(rel, "study5-linear-replay", seed)
    if isinstance(record, dict) and record.get("order") != "stream":
        return inputs.why(rel, f"{rel}: not of the stream order")
    return record


def control_order(inputs: Inputs, rel: str, seed: int) -> dict[str, Any] | str:
    """A linear-replay record of the registered order control's order: ``order.order_seed`` the
    seed itself (``experiments/study5_control_order``) and no edit; otherwise unresolved."""
    record = inputs.record(rel, "study5-linear-replay", seed)
    if isinstance(record, dict):
        order = record.get("order")
        if not (isinstance(order, dict) and order.get("order_seed") == seed
                and order.get("order_edit") is None):  # fmt: skip
            return inputs.why(rel, f"{rel}: not of the registered control's order (seed {seed})")
    return record


def linear_rows(inputs: Inputs, replays: Mapping[int, str]) -> tuple[dict, dict]:
    """M2's own orders with the linear repeat's floor: (rows by seed, the floor)."""

    def row(folder: str, s: int, pairs_folder: str, lin_floor=None) -> dict[str, Any]:
        record = stream_order(inputs, f"{folder}/s{s}/record.json", s)
        return ext2.linear_row(
            record, pairs_hat=inputs.record(f"{pairs_folder}/pairs-hat-s{s}.json", "study5-pairs"),
            pairs_lin=inputs.record(f"{pairs_folder}/pairs-lin-s{s}.json", "study5-pairs"),
            replay_run=replays.get(s), lin_floor=lin_floor)  # fmt: skip

    s0 = ext2.FORCE_SEED
    first = row("linear", s0, "linear")
    again = stream_order(inputs, f"linear-repeat/s{s0}/record.json", s0)
    hat = inputs.record(f"linear-repeat/pairs-hat-s{s0}.json", "study5-pairs")
    lin = inputs.record(f"linear-repeat/pairs-lin-s{s0}.json", "study5-pairs")
    # the repeat's cosine with Δθ is its pairs records' second row (the repeat's vector, the
    # replay's endpoint)
    repeat = ext2.linear_row(again)
    if repeat.get("available"):
        for kind, pairs in (("delta_hat", hat), ("delta_lin", lin)):
            found = ext2.pair_row(pairs, 1, a_sha256=repeat["vector_sha256"].get(kind) or "",
                                  b_run=replays.get(s0))  # fmt: skip
            repeat[kind] |= {"cosine_with_delta_theta": found.get("cosine"),
                             "norm_ratio": found.get("norm_ratio"),
                             "pairs_reason": found.get("reason")}  # fmt: skip
    floor = ext2.linear_floor(first, repeat, hat, lin)
    floors = floor.get("floors", {}) if floor.get("available") else {}
    rows = {s: row("linear", s, "linear", floors.get("vector_lin_relative")) for s in ext2.SEEDS}
    return rows, floor


def m2_section(
    inputs: Inputs,
    rows: Mapping[int, dict],
    floors: Mapping[str, Any],
    delta_r: Mapping[int, Any],
    analysis: Mapping[str, Any],
    edited: Mapping[tuple[int, str], dict],
) -> tuple[dict[str, Any], dict[int, dict]]:
    """M2: the own orders, their means and ρ(q, Δr), the order-control pairs, θ₀ + δ_lin in full,
    and the sealed and edited orders' predictions copied through; and the controls' rows."""
    per_seed = {}
    for s, row in rows.items():
        if row.get("available"):
            for kind in ("delta_hat", "delta_lin"):
                row[kind]["delta_r_minus_q"] = ext2.difference(delta_r.get(s), row[kind]["q"])
            row["delta_r"] = ext2.number(delta_r.get(s))
            row["floors"] = ext2.m2_floors(row, floors)
        per_seed[s] = row
    keys = ("q", "K_benign", "K_harmful", "cosine_with_delta_theta", "norm_ratio", "norm",
            "delta_r_minus_q")  # fmt: skip
    means = {kind: ext2.spreads({s: (r.get(kind) if r.get("available") else None)
                                 for s, r in per_seed.items()}, keys)
             for kind in ("delta_hat", "delta_lin")}  # fmt: skip
    means["epsilon_checks_largest"] = ext2.spread(
        {s: (r.get("epsilon_checks") or {}).get("largest") for s, r in per_seed.items()}
    )
    rho = {kind: ext2.rank_rho({s: (r.get(kind) or {}).get("q") for s, r in per_seed.items()},
                               {s: delta_r.get(s) for s in per_seed})
           for kind in ("delta_hat", "delta_lin")}  # fmt: skip

    controls_per_seed = analysis["descriptive"]["controls"]["order"]["per_seed"]
    controls, pairs = {}, {}
    for s in ext2.M_SEEDS:
        record = control_order(inputs, f"linear-control/s{s}/record.json", s)
        control = ext2.linear_row(record, lin_floor=floors.get("vector_lin_relative"))
        controls[s] = control
        registered_values = {m: per_seed_measure(controls_per_seed, m, s)
                             for m in ("harmful", "K_harmful")}  # fmt: skip
        stream = per_seed.get(s) or {}
        if control.get("available") and stream.get("available"):
            pairs[s] = ext2.control_pair(s, control, stream, registered_values, floors)
        else:
            pairs[s] = {"seed": s, "available": False,
                        "reason": "; ".join(str(x.get("reason")) for x in (control, stream)
                                            if not x.get("available"))}  # fmt: skip

    dlin = {}
    for s in ext2.M_SEEDS:
        dlin[s] = vector_section(inputs, f"points/dlin_s{s}.json", s)

    e2 = {}
    for s in ext2.E2_SEEDS:
        for offset in ext2.E2_OFFSETS:
            tag = f"s{s}-order{offset // 1000}"
            record = inputs.record(f"sealed/{tag}/record.json", "study5-linear-replay", s)
            if isinstance(record, dict) and (record.get("order") or {}).get("order_seed") != (
                    s + offset):  # fmt: skip
                record = inputs.why(tag, f"sealed/{tag}: not of order seed {s + offset}")
            e2[tag] = ext2.prediction(
                ext2.linear_row(record, lin_floor=floors.get("vector_lin_relative"))
            )
    return {"per_seed": per_seed, "means": means, "rho_q_delta_r": rho,
            "controls": {s: ext2.prediction(c) | {"divergence": c.get("divergence")}
                         for s, c in controls.items()},
            "control_pairs": pairs, "dlin_points": dlin, "e2_predictions": e2,
            "edited_predictions": {f"s{s}-{tag}": ext2.prediction(row)
                                   for (s, tag), row in edited.items()}}, controls  # fmt: skip


EDITS: dict[str, dict[str, Any]] = {
    **{f"swap{t}": {"kind": "swap_batches", "step": t} for t in ext2.SWAP_STEPS},
    "twin": {"kind": "swap_first", "step": 2},
    **{f"shuffle{a}-{z}-{d}": {"kind": "shuffle", "first": a, "last": z, "draw": d}
       for a, z in ext2.SHUFFLES for d in ext2.DRAWS},
}  # fmt: skip


def edit_tags(seed: int) -> list[str]:
    tags = [t for t in EDITS if t.startswith("swap") or t == "twin"]
    tags = tags if seed in ext2.SWAP_SEEDS else []
    return tags + [t for t in EDITS if t.startswith("shuffle")]


def edited_rows(inputs: Inputs, lin_floor: float | None) -> dict[tuple[int, str], dict[str, Any]]:
    """Every edited order's prediction record (``edits/s<seed>-<tag>``), its edit checked."""
    out = {}
    for s in ext2.M_SEEDS:
        for tag in edit_tags(s):
            rel = f"edits/s{s}-{tag}/record.json"
            record = inputs.record(rel, "study5-linear-replay", s)
            if isinstance(record, dict) and (record.get("order") or {}).get(
                    "order_edit") != EDITS[tag]:  # fmt: skip
                record = inputs.why(rel, f"{rel}: not of the edit {tag}")
            out[(s, tag)] = ext2.linear_row(record, lin_floor=lin_floor)
    return out


def vector_section(inputs: Inputs, rel: str, seed: int) -> dict[str, Any]:
    """A vector record with the registered label counts of its evaluated points."""
    record = inputs.record(rel, "study5-vector", seed)
    counts: dict[str, Any] = {}
    if isinstance(record, dict):
        run_dir, _ = inputs.run_dir(record.get("experiment"), record.get("run_id"))
        for row in record.get("rows", []):
            if row.get("evaluated"):
                counts[row.get("condition")] = inputs.counts(run_dir, row.get("condition"))
    return ext2.vector_record(record, counts)


def m3_section(inputs: Inputs, linear: Mapping[int, dict]) -> dict[str, Any]:
    per_seed = {}
    for s in ext2.M_SEEDS:
        small = inputs.record(f"points/small_s{s}.json", "study5-vector", s)
        tokens = inputs.record(f"tokens/s{s}.json", "study5-tokens", s)
        r0, source = None, None
        if isinstance(tokens, dict):
            r0, source = ext2.number((tokens.get("theta0") or {}).get("r")), "tokens"
        if r0 is None and (linear.get(s) or {}).get("available"):
            r0, source = linear[s]["r_theta0"], "linear"
        per_seed[s] = ext2.small_amplitudes(small, tokens, r0) | {"r_theta0_source": source}
    means: dict[str, Any] = {"parts": {}, "halving": {}, "points": {}, "tokens": {}}
    for a in ext2.SMALL_AMPLITUDES:
        for part in ("odd", "even"):
            means["parts"][f"{part} {a:g}"] = ext2.spread(
                {s: (r["parts"].get(f"{a:g}") or {}).get(part) for s, r in per_seed.items()}
            )
    for key in ("0.1/0.05", "0.05/0.025"):
        for part in ("odd", "even"):
            means["halving"][f"{part} {key}"] = ext2.spread(
                {s: ((r["halving"].get(key) or {}).get(part) or {}).get("ratio")
                 for s, r in per_seed.items()})  # fmt: skip
    for a in ext2.SMALL_AMPLITUDES:
        for x in (a, -a):
            means["points"][f"{x:+g}"] = ext2.spreads(
                {s: r["points"].get(f"{x:+g}") if r["points"].get("available", True) else None
                 for s, r in per_seed.items()},
                ("r", "harmful_R", "harmful_C", "benign_R", "benign_C", "K_benign",
                 "K_harmful"))  # fmt: skip
    for a in ext2.TOKEN_AMPLITUDES:
        for x in (a, -a):
            means["tokens"][f"{x:+g}"] = ext2.spreads(
                {s: (r["tokens"].get("points") or {}).get(f"{x:+g}") if r["tokens"].get(
                    "available") else None for s, r in per_seed.items()},
                ("r", "entropy", "rare_mass", "delta_logprob"))  # fmt: skip
    return {"reading_rule": {"odd_local": list(ext2.ODD_LOCAL),
                             "even_local": list(ext2.EVEN_LOCAL)},
            "per_seed": per_seed, "means": means}  # fmt: skip


def m4_section(inputs: Inputs) -> dict[str, Any]:
    first = ext2.FORCE_SEED
    names = {
        "mean": first,
        "mean_k": first,
        "mean_neg_k": first,
        **{f"mix{k}": first for k in ext2.MIXTURES},
        **{f"r{kind}_s{s}": s for s in ext2.M_SEEDS for kind in ("mean", "mixture")},
    }
    records = {name: vector_section(inputs, f"points/{name}.json", seed)
               for name, seed in names.items()}  # fmt: skip

    def point(name: str, key: str) -> dict[str, Any] | None:
        found = records.get(name) or {}
        return (found.get("points") or {}).get(key) if found.get("available") else None

    parity = {}
    for scale in ("1", "2"):
        plus, minus = point("mean", f"+{scale}"), point("mean", f"-{scale}")
        parity[f"{scale}x"] = {
            "plus": plus,
            "minus": minus,
            "contrast_plus_minus": ext2.point_contrast(plus, minus),
        }
    matched = {side: {"matched": (records[name].get("matched") if records[name].get("available")
                                  else None), "point": point(name, "kstar"),
                      "reason": records[name].get("reason")}
               for side, name in (("plus", "mean_k"), ("minus", "mean_neg_k"))}  # fmt: skip
    mixtures = {f"{k}": {"natural": point(f"mix{k}", "+1"), "matched_k": point(f"mix{k}", "kstar"),
                         "matched": (records[f"mix{k}"].get("matched")
                                     if records[f"mix{k}"].get("available") else None)}
                for k in ext2.MIXTURES}  # fmt: skip
    removals = {}
    for s in ext2.M_SEEDS:
        mean, mixture = point(f"rmean_s{s}", "+1"), point(f"rmixture_s{s}", "+1")
        removals[s] = {
            "mean_removed": mean,
            "mixture_removed": mixture,
            "contrast_mean_minus_mixture": ext2.point_contrast(mean, mixture),
        }
    keys = ("r", "K_benign", "K_harmful", "harmful", "harmful_high")
    return {
        "records": records, "parity": parity, "matched_k": matched, "mixtures": mixtures,
        "mixtures_spread": {at: ext2.spreads({k: flat(m[at]) for k, m in mixtures.items()}, keys)
                            for at in ("natural", "matched_k")},
        "removals": removals,
        "removals_spread": {
            "mean_removed": ext2.spreads({s: flat(r["mean_removed"])
                                          for s, r in removals.items()}, keys),
            "mixture_removed": ext2.spreads({s: flat(r["mixture_removed"])
                                             for s, r in removals.items()}, keys),
            "contrast_mean_minus_mixture": ext2.spreads(
                {s: r["contrast_mean_minus_mixture"] for s, r in removals.items()}, keys)},
    }  # fmt: skip


def flat(point: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A point's r, K's and harm in one map (for the spreads)."""
    if not point:
        return None
    found = point.get("harm") or {}
    return {k: point.get(k) for k in ("r", "K_benign", "K_harmful")} | {
        "harmful": found.get("harmful"),
        "harmful_high": found.get("harmful_high"),
    }


def m5_section(inputs: Inputs) -> tuple[dict[str, Any], dict[str, Any]]:
    s = ext2.FORCE_SEED
    a = inputs.record("force-repeat/a.json", "study5-force", s)
    b = inputs.record("force-repeat/b.json", "study5-force", s)
    pairs = inputs.record("force-repeat/pairs.json", "study5-pairs")
    floor = ext2.force_floor(a, b, pairs)
    floors = floor if floor.get("available") else {}
    panels, values = {}, {}
    for panel in (59990, 59991):
        record = inputs.record(f"force/s{s}-panel{panel}.json", "study5-force", s)
        if isinstance(record, str):
            panels[panel] = {"available": False, "reason": record}
            continue
        values[panel] = ext2.force_values(record)
        panels[panel] = {"available": True, "g0_norm": record.get("g0_norm"),
                         "projections": record.get("projections"),
                         "steps": ext2.force_panel(values[panel], floors)}  # fmt: skip
    agreement = (ext2.panels_agreement(values[59990], values[59991], floors)
                 if len(values) == 2 else None)  # fmt: skip
    return {"seed": s, "panels": panels, "panels_agreement": agreement}, floor


def m6_section(inputs: Inputs, archive_id: str, own: Mapping[int, dict],
               analysis: Mapping[str, Any], floors: Mapping[str, Any],
               predicted: Mapping[tuple[int, str], dict],
               repeated: Mapping[int, Any]) -> tuple[dict, dict]:  # fmt: skip
    """M6: the swaps and shuffles (prediction and outcome beside the stream order's, which is pod
    2's archived η_l run, its replay's k/521 bound beside) and the twins with their floor; and
    the twins' floor."""
    scans = analysis["descriptive"]["line_scans"]["per_seed"]
    blocks: dict[str, Any] = {"swaps": {"rows": []}, "shuffles": {"rows": []}}
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for s in ext2.M_SEEDS:
        stream = {"source": "archived", "run": "pod 2's η_l run (the registered analysis)",
                  **ext2.stream_values((scans.get(str(s)) or {}).get("+1")),
                  "replay": repeated.get(s)}  # fmt: skip
        for tag in edit_tags(s):
            # the twin is launched with the replays of its seed's group, the others by block
            block = "twins" if tag == "twin" else "swaps" if tag.startswith("swap") else "shuffles"
            group = ext2.SEEDS[(ext2.SEEDS.index(s) // ext2.GROUP) * ext2.GROUP]
            name = f"replays-g{group}" if tag == "twin" else f"{block}-s{s}"
            run_dir, config = inputs.edited_run(f"study5-{archive_id}-{name}",
                                                f"s5-edit-{tag}-{RATE}-s{s}/s{s}")  # fmt: skip
            if isinstance(run_dir, Path):
                edit = ((config or {}).get("train") or {}).get("order_edit")
                if edit != EDITS[tag]:
                    run = f"{run_dir.name}: its config is not the edit {tag}"
                else:
                    run = inputs.reg.f_run(inputs.reader, run_dir, s, ext2.ETA_L)
            else:
                run = inputs.why(f"{block} s{s} {tag}", run_dir)
            counts = inputs.counts(run_dir, ENDPOINT) if not isinstance(run, str) else run
            outcome = ext2.outcome_of(run, counts)
            rows[(s, tag)] = ext2.edited_row(
                s, tag, EDITS[tag], predicted.get((s, tag), {"available": False}), outcome,
                stream, own.get(s) or {}, floors)  # fmt: skip
            if block != "twins":
                blocks[block]["rows"].append(rows[(s, tag)])
    # over seeds: each seed's runs of a kind (its three swaps, a range's two draws) averaged
    # first; per tag one run per seed
    keys = ("harmful", "K_harmful", "K_benign", *ext2.UTILITY)
    for block, found in blocks.items():
        kinds = {"swaps": ("swap",), "shuffles": ("shuffle1-", "shuffle21-")}[block]
        tags = sorted({r["tag"] for r in found["rows"]})
        found["spread"] = {
            "by_kind": {kind: ext2.edited_spreads(
                [r for r in found["rows"] if r["tag"].startswith(kind)], keys) for kind in kinds},
            "by_tag": {tag: ext2.edited_spreads(
                [r for r in found["rows"] if r["tag"] == tag], keys) for tag in tags},
        }  # fmt: skip
    blocks["shuffles"]["early_late"] = ext2.early_late(blocks["shuffles"]["rows"])
    floor = ext2.twins_steps(inputs.record(f"twins/floor-s{ext2.FORCE_SEED}.json", "study5-pairs"))
    twins = {}
    for s in ext2.SWAP_SEEDS:
        found = ext2.twins_steps(inputs.record(f"twins/s{s}.json", "study5-pairs"))
        twins[s] = ext2.twins_row(found, floor, predicted.get((s, "twin"), {}), own.get(s) or {})
        edited = rows.get((s, "twin")) or {}
        twins[s] |= {k: edited.get(k) for k in ("outcome", "outcome_difference",
                                                "predicted_difference", "stream")}  # fmt: skip
        twins[s]["r_stream_source"] = "the seed's replay (pod 4)"
    blocks["twins"] = twins
    return blocks, floor


def checks_section(inputs: Inputs) -> dict[str, Any]:
    out = {"engineering": ext2.check_summary(inputs.record(
        "engineering-check-numerical.json", "study5-engineering-check"))}  # fmt: skip
    for i in range(0, len(ext2.SEEDS), ext2.GROUP):
        group = f"g{ext2.SEEDS[i]}"
        out[group] = ext2.check_summary(inputs.record(ext2.REPLAY_CHECK.format(group=group),
                                                      "study5-engineering-check"))  # fmt: skip
        out[group]["earlier"] = []
        for earlier in (name.format(group=group) for name in ext2.REPLAY_CHECKS_EARLIER):
            out[group]["earlier"].append({"record": earlier} | (
                ext2.check_summary(inputs.record(earlier, "study5-engineering-check"))
                if (inputs.records / earlier).is_file()
                else {"available": False, "reason": f"{earlier}: none written"}
            ))  # fmt: skip
    return out


def r_theta0_section(
    linear: Mapping[int, dict],
    m3: Mapping[str, Any],
    m6: Mapping[str, Any],
    twins_floor: Mapping[str, Any],
) -> dict[str, Any]:
    """Every r(θ₀) read (θ₀ is common to the seeds) and their spread."""
    values: dict[str, float] = {}
    for s, row in linear.items():
        if row.get("available") and row.get("r_theta0") is not None:
            values[f"linear s{s}"] = row["r_theta0"]
    for s, row in m3["per_seed"].items():
        found = ((row.get("tokens") or {}).get("theta0") or {}).get("r")
        if found is not None:
            values[f"tokens s{s}"] = found
    for s, row in m6["twins"].items():
        if row.get("available") and row.get("r_theta0") is not None:
            values[f"twins s{s}"] = row["r_theta0"]
    if twins_floor.get("available") and twins_floor.get("r_theta0") is not None:
        values["twins floor"] = twins_floor["r_theta0"]
    numbers = list(values.values())
    return {"values": values, "n": len(numbers),
            "spread": (max(numbers) - min(numbers)) if numbers else None,
            "rule": "Δr: the seed's linear record; M3: the seed's token record (else its linear "
                    "record); the twins: their pairs record's"}  # fmt: skip


# ------------------------------------------------------------------ the analysis


def own_code() -> dict[str, Any]:
    """This checkout's git state: a clean checkout of a known commit, as the registered
    analysis's."""
    state = git_state()
    if not state.get("commit") or state.get("dirty") is not False:
        refuse("the analysis runs from a clean checkout of a known commit")
    return {
        "commit": state["commit"],
        "dirty": False,
        "module": "zero_field_refusal.analysis.study5_ext2",
    }


def analysis(args: argparse.Namespace) -> dict[str, Any]:
    code = own_code()
    number = ext2.number(args.pair_floors)
    if number is None or number <= 0:
        refuse(f"--pair-floors {args.pair_floors!r}: a finite positive number is needed")
    records = Path(args.records)
    if not records.is_dir():
        refuse(f"{records}: no records folder")
    inputs = Inputs(registered(), records, list(args.runs_root), Path(args.analysis))
    analysis_record, digest = registered_analysis(inputs, Path(args.analysis))
    vectors = inputs.record("vectors/m1/record.json", "study5-vectors")
    if isinstance(vectors, str):
        refuse(f"the vectors record is required: {vectors}")
    if not any((records / "linear" / f"s{s}" / "record.json").is_file() for s in ext2.SEEDS):
        refuse("the linear block is required: no linear/s<seed>/record.json")
    replays = {int(r["seed"]): r.get("run_id") for r in vectors.get("runs", [])}
    h = h_of(analysis_record)
    endpoints = endpoint_r(inputs, analysis_record)

    rows, linear_floor = linear_rows(inputs, replays)
    floors = linear_floor.get("floors", {}) if linear_floor.get("available") else {}
    delta_r: dict[int, float | None] = {}
    delta_r_reason: dict[int, str] = {}
    for s in ext2.SEEDS:
        r_end, row = endpoints.get(s, "no endpoint record"), rows.get(s) or {}
        if isinstance(r_end, str) or not row.get("available") or row.get("r_theta0") is None:
            delta_r[s] = None
            delta_r_reason[s] = r_end if isinstance(r_end, str) else str(
                row.get("reason") or "no r(θ₀)")  # fmt: skip
        else:
            delta_r[s] = r_end - row["r_theta0"]

    checks = checks_section(inputs)
    repeated = ext2.repeats(checks, replays)
    m1 = ext2.m1(vectors, h, delta_r, checks, repeated)
    m1["delta_r_missing"] = delta_r_reason
    edited = edited_rows(inputs, floors.get("vector_lin_relative"))
    m2, controls = m2_section(inputs, rows, floors, delta_r, analysis_record, edited)

    own = {}
    for s in ext2.SEEDS:
        row = rows.get(s) or {}
        missing = None
        if not row.get("available"):
            missing = f"no linear record ({row.get('reason')})"
        elif row["diverged"]:  # outside the criterion's seeds: its Δr is not needed
            missing = None
        elif delta_r[s] is None:
            missing = f"no Δr ({delta_r_reason.get(s)})"
        elif row["delta_lin"]["q"] is None:
            missing = "δ_lin complete without a finite q"
        own[s] = {"q_lin": (row.get("delta_lin") or {}).get("q"), "delta_r": delta_r[s],
                  "diverged": bool(row.get("diverged")), "missing": missing}  # fmt: skip
    pairs = {}
    controls_per_seed = analysis_record["descriptive"]["controls"]["order"]["per_seed"]
    floor_reason = None
    if floors.get("q_lin") is None:  # the linear repeat's floor of q: part 3 needs it
        floor_reason = f"the linear repeat: {linear_floor.get('reason') or 'no q_lin floor'}"
    for s in ext2.M_SEEDS:
        control, stream = controls.get(s) or {}, rows.get(s) or {}
        dh = per_seed_measure(controls_per_seed, "harmful", s)
        missing = None
        if not control.get("available"):
            missing = f"no control linear record ({control.get('reason')})"
        elif not stream.get("available"):
            missing = f"no linear record ({stream.get('reason')})"
        elif ext2.number(dh) is None:
            missing = f"no registered harm difference ({dh})"
        pairs[s] = {"q_control": (control.get("delta_lin") or {}).get("q"),
                    "q_stream": (stream.get("delta_lin") or {}).get("q"),
                    "diverged": bool(control.get("diverged") or stream.get("diverged")),
                    "h_difference": dh, "missing": missing, "q_floor": floors.get("q_lin"),
                    "q_floor_reason": floor_reason}  # fmt: skip
    criterion = ext2.criterion(own, pairs, pair_floors=args.pair_floors)
    for row in criterion["part3_pairs"]["pairs"]:  # the flag beside the k-floor rule
        row["q_difference_floor"] = ext2.resolution(row.get("q_difference"), floors.get("q_lin"))

    m3 = m3_section(inputs, rows)
    m4 = m4_section(inputs)
    m5, force_floor = m5_section(inputs)
    m6, twins_floor = m6_section(inputs, records.name, rows, analysis_record, floors, edited,
                                 repeated)  # fmt: skip
    floor_evidence: dict[int, Any] = {}  # the floor repeat, where its record was written
    if (records / "twins" / f"floor-s{ext2.FORCE_SEED}.json").is_file():
        floor_evidence[ext2.FORCE_SEED] = (twins_floor if twins_floor.get("available")
                                           else str(twins_floor.get("reason")))  # fmt: skip
    envelope = ext2.endpoint_envelope(checks, replays, floor_evidence,
                                      diagnostic_repeats(inputs))  # fmt: skip
    bounds = {"rule": ext2.ENDPOINT_RULE, "per_seed": ext2.seed_bounds(vectors, envelope),
              "m1": ext2.m1_bounds(vectors, envelope),
              "m3": ext2.m3_bounds(m3["per_seed"], vectors, envelope),
              "m4": ext2.m4_bounds(m4, vectors, envelope)}  # fmt: skip
    cases = ext2.cases(m2, m4, m6, checks)
    cases["endpoint_unresolved"] = ext2.bound_cases(bounds)
    return {
        "kind": ext2.KIND,
        "version": ext2.VERSION,
        "registration": {"document": "docs/study5-preregistration.md",
                         "sections": ["Extension 2 (2026-11-11)",
                                      "Extension 2 amendment (2026-11-11)",
                                      "Extension 2 amendment 2 (2026-11-12)",
                                      "Extension 2 amendment 3 (2026-11-12)"]},
        "code": code,
        "archive_id": records.name,
        "registered_analysis": {"sha256": digest, "registered_sha256": REGISTERED_ANALYSIS_SHA256},
        "r_theta0": r_theta0_section(rows, m3, m6, twins_floor),
        "repeats": {
            "rule": "the stream order's outcomes (h, K, harm, utility) are pod 2's archived runs'; "
                    "the replays give Δθ, the twins' r and nothing labeled; a replay whose greedy "
                    "answers differ on k of 521 prompts carries the bound k/521 on its harm; a "
                    "replay is used by its acceptance for endpoint analysis (extension 2's third "
                    "amendment), its changed factual answers and benchmark letters (k of 222 and "
                    "500, flips, accuracy difference) beside; k/521 bounds only the changed "
                    "labels' contribution at the endpoint, not r, K or an intervention's effect",
            "per_seed": repeated,
            "engineering": list((checks.get("engineering") or {}).get("pairs") or []),
        },
        "m1": m1,
        "m2": m2,
        "criterion": criterion,
        "m3": m3,
        "m4": m4,
        "m5": m5,
        "m6": m6,
        "floors": {"linear_repeat": linear_floor, "force_repeat": force_floor,
                   "twins": twins_floor},
        "endpoint_bounds": bounds,
        "cases": cases,
        "missing": inputs.missing,
        "superseded_labels": sorted(inputs.labels.superseded),
        "inputs": dict(sorted(inputs.reader.hashes.items())),
        "prompt_sets": dict(sorted(inputs.reader.prompt_sets.items())),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip


def write_once(path: Path, data: bytes) -> None:
    """``data`` at ``path`` whole, never over an existing file (a hard link)."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        refuse(f"{path.name} exists: never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(data)
    try:
        os.link(partial, path)
    except FileExistsError:
        refuse(f"{path.name} exists: never overwritten")
    finally:
        partial.unlink(missing_ok=True)


def _fmt(value: Any) -> str:
    return "none" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def report(record: Mapping[str, Any], digest: str) -> None:
    """Numbers, ids and hashes only."""
    c = record["criterion"]
    one, two, three = c["part1_rank"], c["part2_scale"], c["part3_pairs"]
    print(f"criterion status {c['status']} passed {_fmt(c['passed'])} primary "
          f"{_fmt(c['primary_predictor'])} withheld {len(c['withheld'])}")  # fmt: skip
    print(f"part1 n {one['n']} rho {_fmt(one['rho'])} p {_fmt(one['p_value'])} scipy_p "
          f"{_fmt(one['scipy_p_greater'])} diverged {len(c['diverged'])}")  # fmt: skip
    print(f"part2 mean_q {_fmt(two['mean_q'])} mean_delta_r {_fmt(two['mean_delta_r'])} ratio "
          f"{_fmt(two['ratio'])}")  # fmt: skip
    print(f"part3 agreements {three['agreements']} of {three['of']}")
    m1 = record["m1"]
    print(f"m1 n {m1['n']} norm_sq_noise_free {_fmt(m1['norm_sq_noise_free'])} mu_hat_norm "
          f"{_fmt(m1['mu_hat_norm'])} split_half_cosine {_fmt(m1['split_half_cosine'])} "
          f"rho_p_h {_fmt(m1['rho_p_h']['rho'])} n {m1['rho_p_h']['n']}")  # fmt: skip
    print(" ".join(f"{k} {len(v)}" for k, v in record["cases"].items()))
    print(f"missing {len(record['missing'])}")
    print(f"record sha256 {digest}")


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m zero_field_refusal.analysis.{PROG}",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    # criterion part 3: a pair's |Δq| below this many floors of Δq counts as a tie
    parser.add_argument("--pair-floors", type=float, default=ext2.PAIR_FLOORS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            refuse(f"{args.out.name} exists: never overwritten")
        record = analysis(args)
        data = encoded(record)
        write_once(args.out, data)
    except Refused as exc:
        sys.exit(f"{PROG}: {exc}")
    report(record, sha256(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
