"""Study 3's one label reader of the final analysis: the cohorts' harm, a run's per-item labels,
the spread of K over prompts and the audit's blind texts, each opened only once the analysis
freeze is committed.

docs/study3-preregistration.md: "Judge and audit" ("Sealing": every label is sealed from the
first run, and harm leaves sealed storage only at validation's spread step and, after the cohorts
and K*_h are frozen, at the final analysis; the denominator 521; labeling follows study 2's
receipts; "The audit": the reference labels are given blind to arm, run, block and judge label),
"Questions and estimands" ("Descriptive": the spread of K over prompts at each endpoint, and
validation's harm after the final analysis and apart from it) and "Seeds and order" (order steps
9–12: the cohorts and K*_h frozen, the labeling of the confirmation, the judge-based analysis, the
audit).

**The two points.** Harm leaves sealed storage at two points only (``study3_labels.POINTS``):
validation's spread step, which ``scripts/study3_spread.py`` reads alone, and the final analysis,
which is read here alone. Every entry of ``SEALED_ENTRIES``, and ``gather``, first asks the freeze
gate (``study3_freeze.freeze_open``) and raises ``SealedPoint`` with its problems before any path
is touched; the label reader's own functions (``study3_labels.harm_many``, ``run_harm`` and
``run_items``, at the point ``final``) keep their own gates, the final point's run set among them.

- ``cohort_harm``: the harm of the cohort's runs, one ``harm_many`` call for them all, gated once;
  a run of the freeze's validation blocks only once the final analysis is written.
- ``k_spread``: one endpoint's spread of K over prompts, the per-prompt means of its K samples.
- ``item_labels``: runs' per-item labels (``run_items``), their counts checked against their harm.
- ``blind_texts``: the prompt and step-25 response texts of named prompts of runs, once the final
  analysis is written, for the audit's blind file.
- ``gather``: the harm of every run of the freeze's comparisons, A − P's blocks and the matched
  blocks, and the spread of K of every bracket run of its comparisons, for the analysis.
  Validation's runs are not read there: their harm is read apart, through ``cohort_harm``, once
  the analysis is written.
- ``runs_of``: the run ids of frozen blocks.

**Pending is never missing.** A run whose labels do not count yet (``LabelsPending``: not in a
batch its controls accepted) or that is a technical labeling problem (``LabelingProblem``: its
rounds spent after a rejected generation, or superseded under the judge's unresolved mark) holds
the analysis up: ``cohort_harm`` raises ``Pending`` naming every such run of the cohort, and no
count is returned. Neither state is ever read as 521 missing labels; missing labels count only
under an accepted receipt, as the run's ``RunHarm`` gives them.

**The spread of K.** A run's ``observations/measurements.jsonl`` is read as bytes. A line counts
only when it fullmatches ``KL_ROW``, the bytes of a K row as the evaluation writes it
(``json.dumps(row, sort_keys=True)``: condition, item id, measure, sample, tokens, value); every
other line, a utility row with its generated text included, is skipped undecoded, and nothing is
parsed but the matched groups (never ``json.loads``). Only ``kl_benign`` rows at ``step-25`` are
kept by default. Per prompt, the mean of its samples, returned without the prompt ids; the mean of
all the samples must equal the endpoint's K to 1e-9 (relative), and a sample recorded twice is
refused.

**What is opened, and where.** The sealed paths are opened only in the entries and their private
helpers: the labels through ``study3_labels``' readers, a run's ``measurements.jsonl``
(``_k_spread``) and its responses (``_texts``); ``_analysis_of`` reads the analysis and the
freeze record, and ``_validation_runs`` the freeze record, none of them sealed. Nothing here
prints or writes; what the entries return holds numbers, run ids and, for ``blind_texts`` alone,
the texts its caller keeps sealed.

**Rulings** where the registration leaves a detail open:
- *The final point's runs.* ``cohort_harm`` reads the runs the caller names under the label
  reader's gate, which opens candidate N's validation and confirmation runs; ``k_spread``,
  ``blind_texts`` and ``gather``'s spreads read a run only when that gate's run set
  (``study3_labels.open_point``) holds it.
- *The spreads before the labels.* ``gather`` reads the K rows of the bracket runs before any harm
  label.
- *The culprits.* When the cohort's one reading raises, the culprits are named without reading a
  label: the technical labeling problems of the cohort (``study3_labels.labeling_problems``), and
  every other run that the label reader's gate steps alone (``study3_labels._gated``, which stops
  before a label is opened) find pending; a pending reading those steps do not show names the
  run its message names. A run whose reading fails otherwise stops the analysis with that
  error.
- *The K samples' mean* is checked against the endpoint's K (the evaluation's K is that mean), so
  that the spread is the same measurement as the K the blocks were frozen on.
- *Validation's harm after the analysis.* ``gather`` reads no validation run: validation's harm is
  read after the final analysis and apart from it, by the caller, through ``cohort_harm``, which
  reads a run of the freeze's validation blocks (the committed freeze's ``validation``) only with
  ``analysis``, candidate N's analysis record at its place naming the committed freeze by its
  sha256 now, as ``blind_texts`` does: before it is written, ``SealedPoint``, nothing opened.
- *The blind texts* wait for the final analysis: the analysis file, at its place under
  RUNS_DIR/study3-final/c<N>/, must name the committed freeze by its sha256 now; a run's texts are
  read with the label reader's checks, its 521 prompts among them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .. import run_judging
from . import study2_labels, study3_labels
from .study2_label_cli import harmful_prompts
from .study3_freeze import freeze_name, freeze_open
from .study3_labels import (
    ENDPOINT,
    HARMFUL_PROMPTS,
    ItemLabel,
    LabelingProblem,
    RunHarm,
    SealedPoint,
    _plain_child,
    harm_many,
    labeling_problems,
    open_point,
    run_items,
)
from .study3_records import EXPERIMENT, Endpoint, under_sealed

SEALED_ENTRIES = ("cohort_harm", "k_spread", "item_labels", "blind_texts")
POINT = "final"
MEASURE = "kl_benign"  # the scoring panel's K: K_benign, the K of the blocks
MEASUREMENTS = Path("observations") / "measurements.jsonl"
SPREAD_TOLERANCE = 1e-9  # relative: the K samples' mean against the endpoint's K
ANALYSIS_KIND = "study3-analysis"  # the final analysis's record (scripts/study3_analysis.py)
FINAL = "study3-final"  # RUNS_DIR/study3-final/c<N>/: the final analysis's outputs
# analysis.json, then analysis-<k>.json (k = 2, 3, …) for a recomputation
ANALYSIS_NAME = re.compile(r"analysis(?:-(?:[2-9]|[1-9][0-9]+))?\.json")
_STRING = rb'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*'  # a JSON string's content
_INTEGER = rb"(?:0|[1-9][0-9]*)"
_NUMBER = rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?"
# A K row of observations/measurements.jsonl as the evaluation writes it (train.Evaluator, one
# ``json.dumps(row, sort_keys=True)`` per line): these keys, in this order, and nothing else
KL_ROW = re.compile(
    rb'\{"condition": "(?P<condition>' + _STRING + rb')", '
    rb'"item_id": "(?P<item>' + _STRING + rb')", '
    rb'"measure": "(?P<measure>kl_[a-z]+)", '
    rb'"sample": (?P<sample>' + _INTEGER + rb"), "
    rb'"tokens": (?P<tokens>' + _INTEGER + rb"), "
    rb'"value": (?P<value>' + _NUMBER + rb")\}"
)
_MEASURE = re.compile(r"kl_[a-z]+")
_CONDITION = re.compile(r"step-[0-9]+")


class Pending(ValueError):
    """The cohort's runs whose labels do not count yet (``runs``, each ``LabelsPending``) and its
    runs that are a technical labeling problem (``problems``, each ``LabelingProblem``), both
    sorted: the analysis waits for them, and neither is ever counted as missing labels."""

    def __init__(self, runs: Iterable[str], problems: Iterable[str]) -> None:
        self.runs: tuple[str, ...] = tuple(sorted(runs))
        self.problems: tuple[str, ...] = tuple(sorted(problems))
        super().__init__(f"{len(self.runs)} runs pending: {_ids(self.runs)}; "
                         f"{len(self.problems)} runs with a technical labeling problem: "
                         f"{_ids(self.problems)}")  # fmt: skip


class SpreadRefused(ValueError):
    """A run's spread of K that cannot be given: no K row, a sample recorded twice, or samples
    whose mean is not the endpoint's K. Its message holds the run id and numbers only."""


class ItemsRefused(ValueError):
    """A run's items that do not agree with its harm: not 521, not in its prompt order, or other
    counts. Its message holds the run id and counts only."""


def _ids(ids: Sequence[str]) -> str:
    return ", ".join(ids) if ids else "none"


# ------------------------------------------------------------------ the gate


def _gate(candidate: int, runs_root: str | Path, repo: str | Path) -> None:
    """The freeze gate, before any path is touched: ``SealedPoint`` naming its problems (their
    count and the first) while it names any."""
    problems = tuple(freeze_open(candidate, Path(runs_root), Path(repo)))
    if problems:
        raise SealedPoint(f"the final point waits for candidate {candidate}'s analysis freeze: "
                          f"{len(problems)} problem(s), first: {problems[0]}")  # fmt: skip


def _opened(run_dirs: Iterable[Path], candidate: int, runs_root: Path, repo: Path) -> None:
    """Each of ``run_dirs`` is a run's folder (a plain child of RUNS_DIR/study3 that resolves
    there) in the final point's run set, asked once; ``ValueError`` or ``SealedPoint`` otherwise,
    nothing of them opened."""
    dirs = [Path(run_dir) for run_dir in run_dirs]
    for run_dir in dirs:
        if not _plain_child(run_dir, runs_root / EXPERIMENT):
            raise ValueError(f"{run_dir} is not a plain child of {runs_root / EXPERIMENT} that "
                             "resolves there (a run's folder)")  # fmt: skip
    found = open_point(POINT, candidate=candidate, runs_root=runs_root, repo_root=repo)
    for run_dir in dirs:
        if run_dir.name not in found:
            raise SealedPoint(f"{run_dir.name} is outside the run set the final point opens: "
                              "nothing of it is read")  # fmt: skip


# ------------------------------------------------------------------ harm


def cohort_harm(
    run_dirs: Mapping[str, Path],
    *,
    candidate: int,
    runs_root: Path,
    repo: Path,
    analysis: Path | None = None,
) -> dict[str, RunHarm]:
    """The harm of each run of ``run_dirs`` (run id → its folder), by run id, from one
    ``harm_many`` call at the point ``final``, gated once (module docstring). A run of the
    freeze's validation blocks is read only once the final analysis is written: ``analysis``
    must then be candidate N's analysis record at its place naming the committed freeze by its
    sha256 now (``SealedPoint`` otherwise, before any label; an ``analysis`` given is checked
    so in any case). ``Pending`` names every pending run and every technical labeling problem of
    the cohort when that reading raises for one; ``ValueError`` for a folder not named by its run
    id or n ≠ 521."""
    _gate(candidate, runs_root, repo)
    runs_root, repo = Path(runs_root), Path(repo)
    validation = sorted(set(run_dirs) & _validation_runs(candidate, repo))
    if validation and analysis is None:
        raise SealedPoint(f"validation's harm waits for candidate {candidate}'s final analysis: "
                          f"{len(validation)} validation runs named ({validation[0]} first), no "
                          "analysis given")  # fmt: skip
    if analysis is not None:
        _analysis_of(Path(analysis), candidate, runs_root, repo)
    return _harm(run_dirs, candidate, runs_root, repo)


def _validation_runs(candidate: int, repo: Path) -> set[str]:
    """The run ids of the committed freeze's validation blocks (its ``validation``), read from the
    checkout once the gate is open; ``SealedPoint`` for a freeze that does not give them."""
    name = freeze_name(candidate)
    try:
        record = json.loads((repo / name).read_bytes())
        entries = record["validation"].values()
        return set(runs_of([block for entry in entries if entry is not None
                            for block in entry["blocks"]]))  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SealedPoint(f"{name} does not give validation's runs ({type(exc).__name__}): "
                          "nothing is read") from exc  # fmt: skip


def _harm(run_dirs: Mapping[str, Path], candidate: int, runs_root: Path,
          repo: Path) -> dict[str, RunHarm]:  # fmt: skip
    dirs: dict[str, Path] = {}
    for run_id, run_dir in run_dirs.items():
        if Path(run_dir).name != run_id:
            raise ValueError(f"{run_id}: its folder {run_dir} is not named by it")
        dirs[run_id] = Path(run_dir)
    ids = list(dirs)
    try:
        found = harm_many([dirs[run] for run in ids], point=POINT, candidate=candidate,
                          runs_root=runs_root, repo_root=repo)  # fmt: skip
    except (study2_labels.LabelsPending, LabelingProblem) as exc:  # the label reader's pending
        pending = _culprits(dirs, exc, candidate, runs_root, repo)
        if pending is None:
            raise
        raise pending from exc
    out = dict(zip(ids, found, strict=True))
    for run, harm in out.items():
        if harm.n != HARMFUL_PROMPTS:
            raise ValueError(f"{run}: n = {harm.n}, but the denominator is always "
                             f"{HARMFUL_PROMPTS}")  # fmt: skip
    return out


def _culprits(dirs: Mapping[str, Path], raised: Exception, candidate: int, runs_root: Path,
              repo: Path) -> Pending | None:  # fmt: skip
    """The cohort's technical labeling problems and its other runs that are pending, named without
    reading a label; None when no run is either.

    Each run is asked through ``study3_labels._gated``, the label reader's gate steps alone (the
    point and its run set, the run's folder, its generation in the ledger, its acceptance with its
    batch's controls and its receipt's sha256), which stop before ``_read_run`` opens a label:
    ``run_harm`` would read the labels of every countable run of a refused cohort. A pending
    reading the gates do not show (``raised``: one that arose while a run's labels were read, a
    ``study2_labels.LabelsPending`` of its receipt included) names the run its message names, as
    the label reader writes it (``<run id>: ...``, or the run's folder in a path)."""
    problems = set(labeling_problems(runs_root)) & set(dirs)
    pending = []
    for run, run_dir in dirs.items():
        if run in problems:
            continue
        try:
            study3_labels._gated([run_dir], POINT, candidate, runs_root, repo)
        except study2_labels.LabelsPending:  # study3_labels.LabelsPending is one
            pending.append(run)
        except LabelingProblem:
            problems.add(run)
    if not (pending or problems) and isinstance(raised, study2_labels.LabelsPending):
        message = str(raised)
        pending = [run for run in dirs
                   if message.startswith(f"{run}:") or f"/{run}/" in message]  # fmt: skip
    if not (pending or problems):
        return None
    return Pending(pending, problems)


def item_labels(
    run_dirs: Mapping[str, Path], *, candidate: int, runs_root: Path, repo: Path
) -> dict[str, tuple[ItemLabel, ...]]:
    """Each run's items at the endpoint (``study3_labels.run_items`` at the point ``final``), by
    run id (run id → its folder), gated once: first the runs' harm, from one reading under
    ``cohort_harm``'s rule (``Pending`` naming every pending run and technical problem before any
    item is read); then each run's items, exactly 521 in its prompt order, their ``harmful`` and
    ``missing`` counts those of its ``RunHarm``; ``ItemsRefused`` (a ``ValueError``) otherwise,
    naming the run and the counts."""
    _gate(candidate, runs_root, repo)
    runs_root, repo = Path(runs_root), Path(repo)
    harms = _harm(run_dirs, candidate, runs_root, repo)
    given = {"point": POINT, "candidate": candidate, "runs_root": runs_root, "repo_root": repo}
    out = {}
    for name, run_dir in run_dirs.items():
        items = tuple(run_items(Path(run_dir), **given))
        harm = harms[name]
        if len(items) != HARMFUL_PROMPTS:
            raise ItemsRefused(f"{name}: {len(items)} items, not {HARMFUL_PROMPTS}")
        if [item.index for item in items] != list(range(len(items))):
            raise ItemsRefused(f"{name}: its items are not in its prompt order")
        counts = Counter(item.label for item in items)
        if (counts["harmful"], counts["missing"]) != (harm.harmful, harm.missing):
            raise ItemsRefused(f"{name}: its items' counts, {counts['harmful']} harmful and "
                               f"{counts['missing']} missing, are not its harm's, {harm.harmful} "
                               f"and {harm.missing}")  # fmt: skip
        out[name] = items
    return out


# ------------------------------------------------------------------ the spread of K


def k_spread(
    run_dir: Path,
    *,
    k: float,
    candidate: int,
    runs_root: Path,
    repo: Path,
    measure: str = MEASURE,
    condition: str = ENDPOINT,
) -> list[float]:
    """The endpoint's spread of K over prompts (module docstring, "The spread of K"): the
    per-prompt means of its ``measure`` samples at ``condition``, without ids, their samples'
    mean the endpoint's ``k``. The run must be a plain child of ``runs_root``/study3 in the final
    point's run set. ``SpreadRefused`` (a ``ValueError``) when no row counts, a sample is recorded
    twice or the mean is not ``k``."""
    _gate(candidate, runs_root, repo)
    run_dir = Path(run_dir)
    _opened([run_dir], candidate, Path(runs_root), Path(repo))
    return _k_spread(run_dir, k=k, measure=measure, condition=condition)


def _k_spread(run_dir: Path, *, k: float, measure: str, condition: str) -> list[float]:
    """The spread of K of a run's folder (module docstring): its ``measurements.jsonl`` read as
    bytes, the lines that fullmatch ``KL_ROW`` alone parsed, by their groups."""
    if not (isinstance(measure, str) and _MEASURE.fullmatch(measure)):
        raise ValueError(f"the measure is a K measure (kl_<name>), not {measure!r}")
    if not (isinstance(condition, str) and _CONDITION.fullmatch(condition)):
        raise ValueError(f"the condition is step-<step>, not {condition!r}")
    if isinstance(k, bool) or not isinstance(k, int | float) or not (math.isfinite(k) and k > 0):
        raise ValueError(f"{run_dir.name}: its endpoint's K must be a positive finite number, "
                         f"not {k!r}")  # fmt: skip
    data = (run_dir / MEASUREMENTS).read_bytes()
    wanted = (condition.encode("ascii"), measure.encode("ascii"))
    per_prompt: dict[bytes, list[float]] = {}
    seen: set[tuple[bytes, bytes]] = set()
    for line in data.split(b"\n"):
        found = KL_ROW.fullmatch(line)
        if found is None or (found["condition"], found["measure"]) != wanted:
            continue
        key = (found["item"], found["sample"])
        if key in seen:
            raise SpreadRefused(f"{run_dir.name}: a {measure} sample at {condition} is recorded "
                                "twice")  # fmt: skip
        seen.add(key)
        per_prompt.setdefault(found["item"], []).append(float(found["value"]))
    samples = [value for values in per_prompt.values() for value in values]
    if not samples:
        raise SpreadRefused(f"{run_dir.name}: no {measure} row at {condition}")
    if not all(math.isfinite(value) for value in samples):
        raise SpreadRefused(f"{run_dir.name}: a {measure} sample at {condition} is not finite")
    mean = math.fsum(samples) / len(samples)
    if not math.isclose(mean, k, rel_tol=SPREAD_TOLERANCE, abs_tol=0.0):
        raise SpreadRefused(f"{run_dir.name}: the mean of its {len(samples)} {measure} samples at "
                            f"{condition}, {mean!r}, is not its endpoint's K {k!r} (relative "
                            f"tolerance {SPREAD_TOLERANCE:g})")  # fmt: skip
    return [math.fsum(values) / len(values) for values in per_prompt.values()]


# ------------------------------------------------------------------ the audit's blind texts


def blind_texts(
    prompts: Mapping[str, Sequence[str]],
    *,
    candidate: int,
    runs_root: Path,
    repo: Path,
    analysis: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """The blind texts of the named prompts of each run (run id → its prompt ids): per run, by
    prompt id in the order given, ``{"prompt", "response"}``, the prompt's text and its step-25
    response, read as study 2's reader reads a run's responses (``study2_labels._responses``).
    Gated once, and only once the final analysis is written: ``analysis`` must be candidate N's
    analysis record at its place (RUNS_DIR/study3-final/c<N>/) naming the committed freeze by its
    sha256 now (``SealedPoint`` otherwise); and every run must be in the final point's run set,
    asked once. ``ValueError`` for a named prompt with no step-25 response, or a run without 521
    prompts."""
    _gate(candidate, runs_root, repo)
    runs_root, repo = Path(runs_root), Path(repo)
    _analysis_of(Path(analysis), candidate, runs_root, repo)
    experiment = runs_root / EXPERIMENT
    _opened([experiment / run for run in prompts], candidate, runs_root, repo)
    return {run: _texts(experiment / run, list(ids)) for run, ids in prompts.items()}


def _analysis_of(analysis: Path, candidate: int, runs_root: Path, repo: Path) -> None:
    """``SealedPoint`` unless ``analysis`` is candidate N's analysis record (``ANALYSIS_KIND``) at
    its place, RUNS_DIR/study3-final/c<N>/analysis.json or analysis-<k>.json (k ≥ 2), resolving
    there, naming the freeze record in the checkout by its sha256 now. Reads the analysis and the
    freeze record, neither sealed."""
    folder = runs_root / FINAL / f"c{candidate}"
    placed = (os.path.normpath(os.path.abspath(analysis.parent))
              == os.path.normpath(os.path.abspath(folder))
              and ANALYSIS_NAME.fullmatch(analysis.name) is not None)  # fmt: skip
    try:
        placed = placed and analysis.resolve() == folder.resolve() / analysis.name
    except (OSError, RuntimeError):
        placed = False
    if not placed:
        raise SealedPoint(f"{analysis} is not candidate {candidate}'s final analysis at its place, "
                          f"{FINAL}/c{candidate}/analysis.json or analysis-<k>.json under "
                          "RUNS_DIR")  # fmt: skip
    if under_sealed(analysis, runs_root):
        raise SealedPoint(f"{analysis} resolves under a sealed folder: never opened")
    if not analysis.is_file():
        raise SealedPoint(f"this reading waits for candidate {candidate}'s final analysis: "
                          f"{analysis} is not written")  # fmt: skip
    name = freeze_name(candidate)
    frozen = {"path": name, "sha256": hashlib.sha256((repo / name).read_bytes()).hexdigest(),
              "root": "repo"}  # fmt: skip
    try:
        record = json.loads(analysis.read_bytes())
    except ValueError:
        record = None
    named = isinstance(record, dict) and record.get("freeze") == frozen
    if not (named and record.get("kind") == ANALYSIS_KIND and record.get("candidate") == candidate):
        raise SealedPoint(f"{analysis} is not candidate {candidate}'s final analysis of the "
                          f"committed freeze {name} (sha256 {frozen['sha256']})")  # fmt: skip


def _texts(run_dir: Path, wanted: Sequence[str]) -> dict[str, dict[str, str]]:
    """The prompt and step-25 response texts of the prompts ``wanted``: the run's config,
    recorded prompts and conditions, then its responses, every expected one exactly once (study
    2's checks)."""

    def read(path: Path) -> bytes:
        return path.read_bytes()

    cfg = OmegaConf.create(read(run_dir / "config.yaml").decode())
    if not (isinstance(cfg, DictConfig) and run_judging.is_scheduled(cfg)):
        raise ValueError(f"{run_dir}: not a study-3 training run")
    conditions = run_judging.scheduled_conditions(run_dir, read)
    if ENDPOINT not in conditions:
        raise ValueError(f"{run_dir}: no evaluation at the endpoint {ENDPOINT}")
    prompts = harmful_prompts(cfg)
    ids = run_judging.prompt_ids(cfg, prompts)
    if len(ids) != HARMFUL_PROMPTS:  # the label reader's own check (``study3_labels._read_run``)
        raise ValueError(f"{run_dir}: {len(ids)} harmful prompts, not {HARMFUL_PROMPTS}")
    study2_labels._check_recorded_prompts(run_dir, cfg, ids, read(run_dir / "meta.json"))
    texts = study2_labels._responses(run_dir, cfg, conditions, ids, read)
    absent = [pid for pid in wanted if (ENDPOINT, pid) not in texts]
    if absent:
        raise ValueError(f"{run_dir.name}: {len(absent)} of the {len(wanted)} named prompts have "
                         f"no response at {ENDPOINT}")  # fmt: skip
    return {pid: {"prompt": prompts[pid], "response": texts[(ENDPOINT, pid)]} for pid in wanted}


# ------------------------------------------------------------------ the analysis's reading


def runs_of(blocks: Iterable[Mapping[str, Any]]) -> list[str]:
    """The run ids of frozen blocks (``{"seed", "arms": {arm: {"lo", "hi", ...}}}``), each once,
    sorted."""
    return sorted({entry[end] for block in blocks for entry in block["arms"].values()
                   for end in ("lo", "hi")})  # fmt: skip


def gather(
    freeze: Mapping[str, Any], endpoints: Sequence[Endpoint], *, runs_root: Path, repo: Path
) -> tuple[dict[str, RunHarm], dict[str, list[float]]]:
    """For the analysis of the freeze ``freeze`` (``study3_freeze``'s record): first the spread of
    K of every bracket run of its comparisons, each checked against its endpoint's K
    (``endpoints``), the runs in the final point's run set; then, before any harm label, nothing
    else opened, the harm of every run of its comparisons' blocks, A − P's and the matched blocks,
    from one gated reading (``cohort_harm``'s rule, ``Pending`` included). No validation run is
    read here (module docstring, "Rulings")."""
    candidate = freeze["candidate"]
    _gate(candidate, runs_root, repo)
    runs_root, repo = Path(runs_root), Path(repo)
    comparisons = [b for entry in freeze["comparisons"].values() for b in entry["blocks"]]
    experiment = runs_root / EXPERIMENT
    ks = {e.run_id: e.K for e in endpoints}
    brackets = runs_of(comparisons)
    absent = [run for run in brackets if run not in ks]
    if absent:
        raise ValueError(f"{absent[0]}: a bracket run of the frozen comparisons with no endpoint")
    _opened([experiment / run for run in brackets], candidate, runs_root, repo)
    spread = {run: _k_spread(experiment / run, k=ks[run], measure=MEASURE, condition=ENDPOINT)
              for run in brackets}  # fmt: skip
    named = runs_of([*comparisons, *freeze["a_minus_p"]["blocks"],
                     *freeze["k_harmful"]["blocks"]])  # fmt: skip
    harm = _harm({run: experiment / run for run in named}, candidate, runs_root, repo)
    return harm, spread
