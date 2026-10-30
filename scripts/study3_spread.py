"""Validation's A–F spread and N from the sealed labels, written as study 3's n record.

The decision script of the sample size:

    python scripts/study3_spread.py --runs-root DIR --candidate N

docs/study3-preregistration.md: "Sample size from validation" (only validation's harm labels are
opened, and only for the spread; the decision script is hashed before labeling; it outputs only
the rule's outcome, N and the A–F spread rounded to 0.5 pp, and no means, signs, per-seed values or
detailed logs), "Judge and audit" (the denominator 521, a missing or unparsable label counted as
not harmful; harm leaves sealed storage at validation's spread step), "Validation" (the blocks),
"Seeds and order" (order step 7: validation's spread and N) and "Voiding and supersession".

**The rule (A–F only).** Per supported validation block of the cohort, d_AF = h_A − h_F, each
arm's h interpolated at K* with its block's weight (``study3_blocks.interpolated``) from h =
harmful/521 of its two endpoints (a missing label counts as not harmful). SD = 100·sd(d_AF) in pp,
the sample SD over the m blocks at full precision. N is ``study3_decisions.sample_size(SD, m, L)``,
the single implementation of the registered rule (SD_up = SD·√((m − 1)/χ²₀.₂₀,ₘ₋₁), power(N) =
Φ(2·√N/SD_up − z_L), N the smallest N ≥ 60 with power(N) ≥ 0.85), L the simulation record's
level; when that record's A–F power is short (``power_af.power_short``), N is at least its
``n_min`` (``n_for``). The cap is N + max(10, ⌈0.12·N⌉) (``cap_of``); the outcome is ``passed``
when the cap fits the 200 reserved confirmation seeds and ``failed`` otherwise: N is never cut.
The registration's examples: at L = 0.975, m = 28 and SD 3.5 pp give N = 60, SD 5.0 pp N = 88, and
m = 22 with SD 5.0 pp N = 92; at L = 0.98, m = 28 and SD 5.0 pp give N = 93.

**What is printed**: one line, ``{"outcome", "N", "spread_pp"}``, the rule's outcome, N and the
spread rounded half up to 0.5 pp (``rounded``). A refusal exits with ``study3_spread.py: <why>``,
naming ids, counts, paths and hashes, never a label, a text, a mean or a per-block value.

**What is written.**
- *The output*, ``RUNS_DIR/study3/decisions/spread-output-c<N>.json`` (``OUTPUT``): ``kind``
  (``spread-output``), ``candidate``, ``level`` (L), ``blocks`` (m), ``outcome``, ``N``, ``cap``,
  ``spread_pp``, ``fixation`` (the validation fixation's path in the checkout, its sha256 and
  ``root: "repo"``), ``labels`` (each cohort run, in the order of the ids, with the sha256 of the
  judging receipt its labels were read under) and ``code`` (the pin the spread ran with). It is
  deterministic and holds no time; it holds no SD, SD_up, power, mean, sign or per-block value. An
  identical existing output is reused; a different one refuses: it is written once.
- *The n record* ``n-c<N>``, through ``study3_decisions.write_decision``: the choice step's rates,
  active comparisons and cohort, L, ``spread`` (``script``, the script by its pinned sha256;
  ``output``, the output's evidence; ``code``, the pin as checked before any label),
  ``spread_pp``, the simulation record's evidence, N, the cap, the state (the outcome),
  ``written_at`` and ``code`` (HEAD's commit and its GPU code, from a clean checkout).

**The order of refusals**, each before anything after it (steps 1–4 read no label):
1. *the spread point*, ``study3_labels``' one rule: the state (``progress`` under this checkout's
   GPU code, θ₀ at its default place) is at candidate N's choice step, its record ``n-c<N>``, with
   no problem;
2. *the code*: candidate N's in-force validation fixation
   (``study3_versions.read_validation_fixation``) without a problem, and the current files equal
   to its ``spread`` pin (``study3_pins.pin_problems``: "the spread's code differs from the
   validation fixation's pin: <paths>"); then a clean checkout of a known commit, the n record's
   ``code``;
3. *the simulation record*, valid and passed: L and the A–F power;
4. *the blocks*: ``study3_records.read_endpoints`` under the state's prescribed cells, and
   ``study3_blocks.block_state`` of A and F per cohort seed: each block's A–F supported, and the A
   and F runs at both endpoints exactly the step's cohort run ids (refused with the counts);
5. *the labels*, ``study3_labels.harm_many`` at the point ``spread``: pending labels or a labeling
   problem refuse, naming the run; nothing is written;
6. *the code again*, just before writing: a pinned file changed during the run refuses, and nothing
   is written;
7. *the output*, then *the n record*: ``write_decision`` checks the pin against the current files
   once more, and a refusal names its problems (never the spread).

**Rulings** where the registration leaves a detail open:
- *The pin.* The spread runs only while its import closure equals the validation fixation's pin,
  checked before any label and again at write time, and records that code in the n record
  (``spread.code``). No reader hashes the current files afterwards: an unrelated later change of a
  pinned module voids nothing written, and a correction that changes N goes through the registered
  supersession ("Voiding and supersession"), a successor of the n record.
- *Half up.* The spread is rounded to 0.5 pp as floor(x/0.5 + 0.5)·0.5.
- *The fixation's file* is the one the in-force reader reads,
  ``study3_versions.fixation_name("validation", N)`` in the checkout.
"""

import argparse
import contextlib
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_labels, study3_versions
from zero_field_refusal.analysis.study3_blocks import block_state, interpolated
from zero_field_refusal.analysis.study3_decisions import (  # its write-once publication
    SIMULATION_RECORD,
    SPREAD_STEP_PP,
    DecisionRefused,
    _publish,
    _under_sealed,
    cap_of,
    decision_path,
    read_decision,
    sample_size,
    write_decision,
)
from zero_field_refusal.analysis.study3_labels import (
    LabelingProblem,
    LabelsPending,
    LedgerError,
    RunHarm,
    SealedPoint,
    harm_many,
)
from zero_field_refusal.analysis.study3_pins import SPREAD_SCRIPT, pin_problems
from zero_field_refusal.analysis.study3_progress import THETA0_RUN
from zero_field_refusal.analysis.study3_records import EXPERIMENT, SEEDS, read_endpoints
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # the checkout: the pinned files, the fixation, the n record's repo evidence
OUTPUT = "study3/decisions/spread-output-c{candidate}.json"
KIND = "spread-output"
ARMS = ("A", "F")
Block = Mapping[str, tuple[str, str, float]]  # arm -> (lower run id, upper run id, weight)


def refuse(why: str) -> NoReturn:
    sys.exit(f"study3_spread.py: {why}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ the rule


def n_for(sd_pp: float, m: int, level: float, n_min: int | None = None) -> int:
    """N of the registered rule (``study3_decisions.sample_size``), raised to ``n_min`` when the
    simulation record's A–F power is short (its ``n_min`` given; None otherwise)."""
    n = sample_size(sd_pp, m, level)
    return n if n_min is None else max(n, n_min)


def rounded(sd_pp: float) -> float:
    """The spread rounded half up to 0.5 pp."""
    return math.floor(sd_pp / SPREAD_STEP_PP + 0.5) * SPREAD_STEP_PP


def block_differences(blocks: Sequence[Block], harm: Mapping[str, RunHarm]) -> list[float]:
    """d = h_A − h_F per block, each arm's h interpolated with its block's weight between its two
    endpoints, h = harmful/n (a missing label counts as not harmful)."""

    def h(run: str) -> float:
        found = harm[run]
        return found.harmful / found.n

    out = []
    for block in blocks:
        (a_lo, a_hi, a_w), (f_lo, f_hi, f_w) = block["A"], block["F"]
        out.append(interpolated(h(a_lo), h(a_hi), a_w) - interpolated(h(f_lo), h(f_hi), f_w))
    return out


# ------------------------------------------------------------------ the steps


def pinned_code(candidate: int, runs_root: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Step 2's fixation: its ``spread`` pin, equal to the current files, and its reference."""
    fixation, problems = study3_versions.read_validation_fixation(candidate, runs_root=runs_root,
                                                                  repo=REPO)  # fmt: skip
    if fixation is None:
        refuse(f"candidate {candidate}'s validation fixation is "
               + (f"not valid ({problems[0]})" if problems else "missing")
               + ": it pins the spread's code")  # fmt: skip
    pin = fixation.get("spread")
    changed(pin)
    named = study3_versions.fixation_name("validation", candidate)
    path = REPO / named
    if _under_sealed(path, REPO):
        refuse(f"{named} resolves under a sealed folder: never opened")
    try:
        digest = sha256(path.read_bytes())
    except OSError as exc:
        refuse(f"{named} cannot be read ({type(exc).__name__})")
    return pin, {"path": named, "sha256": digest, "root": "repo"}


def changed(pin: Any) -> None:
    """Refused unless the current files equal the pin (``study3_pins.pin_problems``)."""
    found = pin_problems(pin, REPO)
    if found:
        refuse(f"the spread's code differs from the validation fixation's pin: {'; '.join(found)}")


def head_code() -> dict[str, str]:
    """HEAD's commit and its GPU code, as the n record's ``code`` names them, from a clean
    checkout."""
    state = git_state(REPO)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse(f"the n record's code needs a clean checkout of a known commit, not {state}")
    identity = code_identity.gpu_code_identity(state["commit"], REPO)
    return {"commit": state["commit"], "gpu_code_sha256": identity["sha256"]}


def simulation_of(runs_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Step 3: the simulation record, valid and passed, and its evidence."""
    record, problems = read_decision(runs_root, "simulation", repo=REPO)
    if record is None or record["state"] != "passed":
        why = (f"not valid ({problems[0]})" if problems else "missing" if record is None
               else "failed")  # fmt: skip
        refuse(f"the simulation record is {why}: it gives L and the A–F power")
    try:
        digest = sha256(decision_path(runs_root, "simulation").read_bytes())
    except OSError as exc:
        refuse(f"the simulation record cannot be read ({type(exc).__name__})")
    return record, {"path": SIMULATION_RECORD, "sha256": digest, "root": "runs"}


def cohort_blocks(runs_root: Path, state: Mapping[str, Any], candidate: int,
                  cohort: Mapping[str, Any]) -> list[Block]:  # fmt: skip
    """Step 4: each cohort block's A and F endpoints and weights, the A–F of every block
    supported and their runs exactly the cohort's run ids."""
    try:
        endpoints, _ = read_endpoints(runs_root, theta0=Path(runs_root) / THETA0_RUN,
                                      prescribed=state.get("prescribed") or {})  # fmt: skip
    except (OSError, ValueError) as exc:
        refuse(f"the run records cannot be read ({type(exc).__name__})")
    seeds = list(cohort["seeds"])
    blocks, named, unsupported = [], [], []
    for seed in seeds:
        try:
            found = block_state(endpoints, seed, candidate, ARMS)
        except ValueError as exc:
            refuse(f"block {seed}: {exc}")
        if not found["AF"]["supported"]:
            unsupported.append(seed)
            continue
        entry = {}
        for arm in ARMS:
            arm_state = found["arms"][arm]
            entry[arm] = (arm_state["lo"].run_id, arm_state["hi"].run_id, arm_state["weight"])
            named += entry[arm][:2]
        blocks.append(entry)
    if unsupported:
        refuse(f"{len(unsupported)} of the cohort's {len(seeds)} blocks are not supported for A "
               f"and F now (seeds {unsupported[:3]}): the cohort is the step's")  # fmt: skip
    given = list(cohort["run_ids"])
    if sorted(given) != sorted(named):
        refuse(f"the cohort's {len(given)} run ids are not the {len(named)} A and F endpoints of "
               f"its {len(blocks)} blocks: {len(set(given) - set(named))} are not among them, "
               f"{len(set(named) - set(given))} of them are not in the cohort")  # fmt: skip
    if len(blocks) < 2:
        refuse(f"the cohort has {len(blocks)} block(s): the spread needs at least two")
    return blocks


def labels(runs_root: Path, candidate: int, run_ids: Sequence[str]) -> dict[str, RunHarm]:
    """Step 5: each cohort run's harm, read at the spread point (``harm_many``)."""
    dirs = [Path(runs_root) / EXPERIMENT / run for run in run_ids]
    try:
        found = harm_many(dirs, point="spread", candidate=candidate, runs_root=Path(runs_root),
                          repo_root=REPO)  # fmt: skip
    except (LabelsPending, LabelingProblem, SealedPoint, LedgerError) as exc:
        refuse(f"the labels are not read for the spread ({type(exc).__name__}): {exc}")
    except (OSError, ValueError) as exc:
        refuse(f"the labels cannot be read ({type(exc).__name__}): {exc}")
    return dict(zip(run_ids, found, strict=True))


def published(runs_root: Path, candidate: int, data: bytes) -> dict[str, str]:
    """The output at ``OUTPUT``, written once: an identical one is reused, a different one
    refuses. Its evidence."""
    named = OUTPUT.format(candidate=candidate)
    path = Path(runs_root) / named
    if _under_sealed(path, Path(runs_root)):
        refuse(f"{named} resolves under a sealed folder: never opened")
    if not (path.exists() or path.is_symlink()):
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(DecisionRefused):  # written meanwhile: compared below
            _publish(path, data)
    try:
        same = path.read_bytes() == data
    except OSError:
        same = False
    if not same:
        refuse(f"{named} exists with other content: the spread's output is written once")
    return {"path": named, "sha256": sha256(data)}


# ------------------------------------------------------------------ the command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="RUNS_DIR: the run records, the labels and the decision records",
    )
    parser.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    args = parser.parse_args(argv)
    root, candidate = args.runs_root, args.candidate
    # 1. the spread point
    try:
        state, _ = study3_labels._spread(candidate, root, REPO)
    except SealedPoint as exc:
        refuse(str(exc))
    kind, fields = state["step"]["record"], state["step"]["fields"]
    # 2. the code, before any label
    pin, fixation = pinned_code(candidate, root)
    code = head_code()
    # 3. the simulation record
    simulation, simulation_ref = simulation_of(root)
    level, power = simulation["level"], simulation["gates"]["power_af"]
    # 4. the blocks
    blocks = cohort_blocks(root, state, candidate, fields["cohort"])
    # 5. the labels
    run_ids = list(fields["cohort"]["run_ids"])
    harm = labels(root, candidate, run_ids)
    # the rule
    d = block_differences(blocks, harm)
    sd_pp = 100 * statistics.stdev(d)
    n = n_for(sd_pp, len(d), level, power["n_min"] if power["power_short"] else None)
    cap = cap_of(n)
    outcome = "passed" if cap <= len(SEEDS["confirmation"][candidate]) else "failed"
    spread_pp = rounded(sd_pp)
    output = {"kind": KIND, "candidate": candidate, "level": level, "blocks": len(blocks),
              "outcome": outcome, "N": n, "cap": cap, "spread_pp": spread_pp,
              "fixation": fixation,
              "labels": [{"run": run, "receipt_sha256": harm[run].receipt_sha256}
                         for run in sorted(run_ids)],
              "code": pin}  # fmt: skip
    data = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # 6. the code again, then 7. the output and the n record
    changed(pin)
    evidence = published(root, candidate, data)
    script = next(entry["sha256"] for entry in pin if entry["path"] == SPREAD_SCRIPT)
    record = {"kind": kind, "state": outcome,
              "written_at": datetime.now(UTC).isoformat(timespec="seconds"), "code": code,
              "candidate": candidate, "rates": fields["rates"], "active": fields["active"],
              "level": level, "cohort": fields["cohort"],
              "spread": {"script": {"path": SPREAD_SCRIPT, "sha256": script, "root": "repo"},
                         "output": evidence, "code": pin},
              "spread_pp": spread_pp, "simulation": simulation_ref, "N": n,
              "cap": cap}  # fmt: skip
    try:
        write_decision(root, kind, record, repo=REPO)
    except DecisionRefused as exc:
        refuse(str(exc))
    print(json.dumps({"outcome": outcome, "N": n, "spread_pp": spread_pp}))


if __name__ == "__main__":
    main()
