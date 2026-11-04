"""Study 3's precommitted spread script: validation's A–F spread, N and the cap, from the sealed
labels of the cohort the route fixes (docs/study3-preregistration.md, "Sample size from
validation").

    python -m zero_field_refusal.analysis.study3_spread --runs-root RUNS_DIR --theta0 THETA0 \
        --candidate 1 --out OUT.json

Only validation's harm labels are opened, and only for the spread. The script is hashed before
labeling (its own sha256 is in its output, ``script_sha256``, and the ``n-c<N>`` record names it
as ``spread.script``).

**The state.** ``study3_progress.read_state`` under the checkout's GPU code
(``code_identity.gpu_code_identity``), θ₀'s evaluation at ``--theta0``: the route must be at
``choice`` for the candidate, its step the ``n-c<N>`` record to write; its fields give the cohort
(the validation blocks supported for A and F, and the A and F runs at both rates). Anything else
refuses. The blocks are rebuilt as the route builds them (``study3_blocks.block_state`` over the
endpoints of each validation seed, the arms still active), and the A–F blocks must be exactly the
route's cohort, seeds and run ids. Each supported arm of a block gives its two endpoints ordered
by K and its weight w at K*.

**The harm** of each cohort run (``harm_counts``): its WildGuard labels as
``study2_labels.read_endpoint_labels`` reads them, strictly (its label file, its completed
judging receipt for that file, every record checked against the response stored now), at its one
evaluation, ``step-25``, over the 521 harmful prompts: n = 521, the harmful labels, and the
missing ones (a missing or unparsed label is missing). A run without a matching completed receipt
is not labeled: the script refuses, naming the run id only (no reader's message, which may hold a
count).

**The rule** (``spread_result``): d_AF per block, primary (``study3_estimates.block_differences``:
A_low − F_low, a missing label not harmful); SD = 100·sd(d_AF), the sample SD (ddof 1) over the
m supported A–F blocks at full precision (``statistics.stdev``); N =
``study3_decisions.sample_size(SD, m, L)``, the single implementation of the rule, L the
simulation record's level, raised to the record's ``n_min`` when its power is short (amendment 7:
N and the cap are raised for the simulated power); cap = ``cap_of(N)``; the rule's outcome
``passed`` when the cap fits the reserved confirmation seeds (``study3_records.SEEDS``, 200),
else ``failed`` (N is never cut). P–F's power is reported, not targeted: over the P–F supported
validation blocks, SD_PF from d_PF primary likewise and
``planning_power(N, upper_sd(SD_PF, m_PF), L)``, rounded to 0.01; null when P–F is closed or has
fewer than two blocks.

**The output** (``OUTPUT_FIELDS``, strict JSON, written once to OUT): ``state`` (the rule's
outcome), ``N``, ``cap``, ``spread_pp`` (SD rounded to 0.5 pp, half up: ``spread_pp``), ``m``,
``power_pf``, ``level`` (L), ``cohort`` (``seeds`` and ``run_ids``, the route's) and
``script_sha256``. No mean, sign, per-block or per-seed value, unrounded spread, P–F spread or
log; its ``state``, ``N``, ``cap``, ``spread_pp``, ``level`` and ``cohort`` are the ``n-c<N>``
record's fields of the same names. Stdout: ``written OUT`` and the output's sha256, nothing
else. A refusal or a failure after a label was read names no value (only the exception's type).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..config import load_config
from ..judge import judge_identity
from ..paths import REPO_ROOT
from .study2_label_cli import harmful_prompts
from .study3_blocks import SUPPORTED, block_state
from .study3_decisions import (
    SPREAD_STEP_PP,
    cap_of,
    planning_power,
    read_decision,
    sample_size,
    upper_sd,
)
from .study3_estimates import DENOMINATOR, block_differences
from .study3_progress import read_state
from .study3_records import SEEDS, Endpoint

SCRIPT = Path(__file__)
JUDGE = "wildguard"
CONDITION = "step-25"
OUTPUT_FIELDS = ("state", "N", "cap", "spread_pp", "m", "power_pf", "level", "cohort",
                 "script_sha256")  # fmt: skip
PAIRS = {"AF": ("A", "F"), "PF": ("P", "F")}


class SpreadRefused(ValueError):
    """The spread cannot be computed: why, naming run ids at most, never a value."""


@dataclass(frozen=True)
class Harm:
    """A run's label counts: ``n`` (521), ``harmful`` and ``missing``."""

    n: int
    harmful: int
    missing: int


def script_sha256(path: Path = SCRIPT) -> str:
    """The sha256 of this script's file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spread_pp(sd_pp: float) -> float:
    """The spread rounded to 0.5 pp, half up."""
    return math.floor(sd_pp / SPREAD_STEP_PP + 0.5) * SPREAD_STEP_PP


def harm_counts(run_dir: Path, identity: Mapping[str, Any], prompts: Mapping[str, str],
                judge: str = JUDGE) -> Harm:  # fmt: skip
    """A run's ``Harm`` from its labels (module docstring, "The harm"); ``SpreadRefused`` naming
    the run id alone when it is not labeled or its labels do not read."""
    from .study2_labels import read_endpoint_labels  # it imports run_judging

    run_id = Path(run_dir).name
    try:
        rows = read_endpoint_labels(Path(run_dir), judge, identity, prompts)
    except Exception:  # its message may hold a count: not shown
        raise SpreadRefused(f"{run_id}: not labeled (no completed judging receipt matches its "
                            "label file) or its labels do not read") from None  # fmt: skip
    if [row["condition"] for row in rows] != [CONDITION] or rows[0]["prompts"] != DENOMINATOR:
        raise SpreadRefused(f"{run_id}: its labels are not of one {CONDITION} evaluation of the "
                            f"{DENOMINATOR} harmful prompts")  # fmt: skip
    return Harm(DENOMINATOR, rows[0]["harmful"], rows[0]["missing"])


def frozen_blocks(states: Sequence[Mapping[str, Any]], pair: str) -> list[dict[str, Any]]:
    """The blocks supported for a comparison (``PAIRS``), in seed order, as
    ``study3_estimates.block_differences`` reads them: ``{"seed", "arms": {arm: {"lo", "hi",
    "w"}}}``, the run ids of the arm's endpoints ordered by K and the weight at K*."""
    out = []
    for state in states:
        if (state.get(pair) or {}).get("state") != SUPPORTED:
            continue
        arms = {}
        for arm in PAIRS[pair]:
            support = state["arms"][arm]
            arms[arm] = {"lo": support["lo"].run_id, "hi": support["hi"].run_id,
                         "w": support["weight"]}  # fmt: skip
        out.append({"seed": state["seed"], "arms": arms})
    return out


def _sd_pp(blocks: Sequence[Mapping[str, Any]], harm: Mapping[str, Harm],
           pair: Sequence[str]) -> float:  # fmt: skip
    """100·sd of the primary block differences (sample SD, ddof 1), in pp."""
    d = block_differences(blocks, harm, pair)["primary"]
    return 100.0 * statistics.stdev(float(x) for x in d)


def spread_result(
    af_blocks: Sequence[Mapping[str, Any]],
    pf_blocks: Sequence[Mapping[str, Any]] | None,
    harm: Mapping[str, Harm],
    *,
    level: float,
    candidate: int,
    cohort: Mapping[str, Sequence],
    script: str,
    n_min: int | None = None,
) -> dict[str, Any]:
    """The output (module docstring, "The rule" and "The output"), pure. ``pf_blocks``: the P–F
    supported blocks, None when P–F is closed; ``n_min``: the simulation record's when its power
    is short; ``script``: the script's sha256."""
    m = len(af_blocks)
    if m < 2:
        raise SpreadRefused(f"{m} supported A–F blocks: the spread needs at least two")
    sd = _sd_pp(af_blocks, harm, ("A", "F"))
    n = sample_size(sd, m, level)
    if n_min is not None:
        n = max(n, n_min)
    cap = cap_of(n)
    power = None
    if pf_blocks is not None and len(pf_blocks) >= 2:
        up = upper_sd(_sd_pp(pf_blocks, harm, ("P", "F")), len(pf_blocks))
        power = round(planning_power(n, up, level), 2)
    return {
        "state": "passed" if cap <= len(SEEDS["confirmation"][candidate]) else "failed",
        "N": n, "cap": cap,
        "spread_pp": spread_pp(sd),
        "m": m, "power_pf": power, "level": level,
        "cohort": {"seeds": list(cohort["seeds"]), "run_ids": list(cohort["run_ids"])},
        "script_sha256": script,
    }  # fmt: skip


def _reading(runs_root: Path, repo: Path, candidate: int, theta0: Path | None,
             expected_code: str) -> tuple[dict[str, Any], list[Endpoint]]:  # fmt: skip
    """The state and its endpoints, at the choice of ``candidate`` with its n record to write."""
    state, endpoints = read_state(runs_root, repo=repo, expected_code=expected_code,
                                  theta0=theta0)  # fmt: skip
    step, kind = state.get("step") or {}, f"n-c{candidate}"
    if (state.get("stage") != "choice" or state.get("candidate") != candidate
            or step.get("kind") != "decision" or step.get("record") != kind):  # fmt: skip
        raise SpreadRefused(f"the route is at {state.get('stage')!r} (candidate "
                            f"{state.get('candidate')}), not at the choice of candidate "
                            f"{candidate} with its {kind} record to write")  # fmt: skip
    return state, endpoints


def spread(
    runs_root: str | Path,
    *,
    candidate: int,
    theta0: str | Path | None,
    expected_code: str,
    repo: str | Path = REPO_ROOT,
    harm_of: Callable[[Endpoint], Harm] | None = None,
    script: str | None = None,
) -> dict[str, Any]:
    """The output from the records under ``runs_root`` (module docstring). ``harm_of``: a run's
    ``Harm`` (by default its WildGuard labels, ``harm_counts``); ``script``: the script's sha256
    (by default this file's)."""
    root, repo = Path(runs_root), Path(repo)
    state, endpoints = _reading(root, repo, candidate, None if theta0 is None else Path(theta0),
                                expected_code)  # fmt: skip
    fields = state["step"]["fields"]
    cohort = fields["cohort"]
    simulation, problems = read_decision(root, "simulation", repo=repo)
    if simulation is None or simulation["state"] != "passed":
        raise SpreadRefused(f"the simulation record is not a valid passed record {problems[:1]}")
    gate = simulation["gates"]["power_af"]
    n_min = gate["n_min"] if gate.get("power_short") else None
    count = state["validation"]["blocks"]
    arms = list(state["active"]["arms"])
    states = [block_state([e for e in endpoints if e.seed in (seed, None)], seed, candidate, arms)
              for seed in SEEDS["validation"][candidate][:count]]  # fmt: skip
    af = frozen_blocks(states, "AF")
    run_ids = sorted(b["arms"][arm][end] for b in af for arm in "AF" for end in ("lo", "hi"))
    if [b["seed"] for b in af] != list(cohort["seeds"]) or run_ids != sorted(cohort["run_ids"]):
        raise SpreadRefused("the A–F blocks rebuilt here are not the cohort the route fixes")
    pf = frozen_blocks(states, "PF") if "P-F" in fields["active"] and "P" in arms else None
    by_id = {e.run_id: e for e in endpoints}
    needed = sorted({b["arms"][arm][end] for blocks, pair in ((af, "AF"), (pf or [], "PF"))
                     for b in blocks for arm in PAIRS[pair] for end in ("lo", "hi")})  # fmt: skip
    harm_of = harm_of or _labels_reader()
    harm = {run_id: harm_of(by_id[run_id]) for run_id in needed}
    try:  # labels are read: a failure names no value
        return spread_result(af, pf, harm, level=simulation["level"], candidate=candidate,
                             cohort=cohort, script=script or script_sha256(),
                             n_min=n_min)  # fmt: skip
    except SpreadRefused:
        raise
    except Exception as exc:
        raise SpreadRefused(f"the computation failed ({type(exc).__name__})") from None


def _labels_reader() -> Callable[[Endpoint], Harm]:
    """``harm_counts`` of an endpoint's run, under WildGuard's identity and the run's prompt set
    (read once per set)."""
    identity = judge_identity(load_config(f"judge/{JUDGE}"))
    sets: dict[str, Mapping[str, str]] = {}

    def read(endpoint: Endpoint) -> Harm:
        cfg = OmegaConf.load(Path(endpoint.run_dir) / "config.yaml")
        assert isinstance(cfg, DictConfig)
        key = OmegaConf.to_yaml(cfg.data) + str(cfg.eval.harmful.prompt_set)
        if key not in sets:
            sets[key] = harmful_prompts(cfg)
        return harm_counts(endpoint.run_dir, identity, sets[key])

    return read


def write_output(path: Path, result: Mapping[str, Any]) -> str:
    """Write the output once (strict JSON) and return its sha256; ``FileExistsError`` when the
    file is there."""
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(handle, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> None:
    from .. import code_identity

    parser = argparse.ArgumentParser(prog="python -m zero_field_refusal.analysis.study3_spread",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--theta0", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=int, choices=(1, 2))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists: the output is written once")
    code = code_identity.gpu_code_identity(repo=REPO_ROOT)["sha256"]
    try:
        result = spread(
            args.runs_root, candidate=args.candidate, theta0=args.theta0, expected_code=code
        )
    except SpreadRefused as exc:
        sys.exit(f"refused: {exc}")
    digest = write_output(args.out, result)
    print(f"written {args.out}")
    print(digest)


if __name__ == "__main__":
    main()
