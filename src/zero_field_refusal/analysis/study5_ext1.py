"""Study 5's extension 1 on a CPU (docs/study5-preregistration.md, "Extension 1 (2026-11-11)",
"What is reported"): E1, the ladder's rungs ``gaussian_d`` and ``linear_no_d`` repeated with at
most 8 extensions on seeds 50000-50005, and E2, the order repeats of seeds 50006-50015, written
once as one JSON record. Descriptive only; pod 2's numbers are the registered analysis's.

    python -m zero_field_refusal.analysis.study5_ext1 --pod3-runs DIR --pod2-runs DIR
        --registered ANALYSIS.json --out RECORD.json [--open-e2 OPENING.json]

It sits in the analysis package, outside the GPU code identity (``code_identity``), as
``study5_analysis_cli`` does, and reuses that module's readers and ``study5_analysis``'s
functions. Importing it loads no torch; the label readers (``run_judging``) load it when they
read, on a CPU.

**Inputs.** --pod3-runs is pod 3's runs root as ``restore-runs`` restored it: the rung records
under ``study5_records/<pod3-…>/ladder/`` (one archive id ``pod3-…`` there), the rungs' runs, the
order runs and the launcher journals (``launch/``), with the label files in the run directories.
--pod2-runs is the runs root the registered analysis read and --registered that analysis's record,
its sha256 named by the registration at this commit. Every file read under --pod2-runs is located
through the record's ``inputs`` and must have the sha256 it lists there; the prompt sets must be
the record's. A refusal otherwise. The checkout must be clean.

**Labels** are read as the registered analysis reads them (``study5_analysis_cli.Labels``: study 2's
checks over ``run_judging.response_conditions``); pod 2's under the registered record's latest
ready-to-label time, so that superseded labels stay unused as there. A missing label is not
harmful in every share, with the all-harmful value beside (``harmful_high``). A run without
labels gives no harm, with the reason.

**E1.** Per seed and rung, pod 3's record (kind ``study5-ladder``, its rung and seed, its F run
the one pod 2's rungs of the seed name, ``extensions_allowed`` 8; a reason otherwise) through the
registered ladder table (``study5_analysis.ladder_table``): status, extensions used, c₀, c, the
matched-K step and the tolerance flag, K_benign, K_harmful and the utilities at the evaluated
point, the three classes; beside each row c/c₀, ``harmful_high`` and the missing labels. A rung's
run is read only after its config (``rung_run``): an E2 order repeat (an E2 cell in its name, an E2
seed with ``train.order_seed`` = s + 1000 or s + 2000, or one of those 20 order seeds, as the
extension-2 analysis refuses them) refuses the analysis, the name checked before the config is
read; a run that is not the seed's rung of its η_l run is a reason. Beside
them (``beside``): the seed's ``linear`` rung from pod 2 (its record and labels, each row checked
against the registered record's row) and its trained endpoint (h_s and the classes at K* as the
registered record holds them). ``means``: per rung (and the endpoint) and measure, the t interval
over the seeds where the rung is evaluated, each other seed listed. ``shuffled_d_trigger`` is the
registered table's: the mean over 50000-50005 of ``linear`` (pod 2) minus ``linear_no_d`` (pod 3)
harm, decided only when all six differences exist. ``reproducibility``: each grid point that both
pods measured, K beside pod 2's, and the largest relative difference per rung.

**E2** is sealed until a separate opening decision. Without --open-e2 nothing of an order run and
none of pod 3's launcher journals is read, and the record holds ``"e2": {"sealed": true}``. With
it, the opening record is checked before anything of E2 is read (``opening_record``): a JSON
object of kind ``study5-e2-opening`` naming the dated note that fixes the test (``note``, its path
in the repository), the commit that holds it (``commit``, a full sha, an ancestor of this
checkout's HEAD or HEAD itself) and the note's sha256 (``note_sha256``, that of its text at the
commit, which HEAD holds unchanged). The record names it by sha256, with the commit, the note and
its sha256. Per seed and order: the
original order is the registered analysis's outcome of the seed's two fresh F runs (its K values
read from those runs); each new order is the launcher's last ``done`` or ``diverged`` end of its
cell (campaign ``study5-<ID>-orders-s<seed>``, cell ``s5-order<1|2>-<η>-s<seed>``), its config
the seed, the rate, frozen Adam, ``train.order_seed`` = s + 1000 or s + 2000 and role ``control``,
through the registered outcome (``study5_analysis.seed_outcome``: study 3's support rule against
θ₀'s accuracies from the registered analysis's θ₀ run, its interpolation in ln K). Reported:
K_benign and K_harmful at both rates, h at K* (``harmful_high`` beside), and K_harmful at K*;
every order without h listed with its reason. Over the seeds with h in all three orders: σ²_w,
σ²_b, the order share with its seed-level bootstrap interval, and the within-seed Spearman ρ of
the deviations; the registered order controls' differences at η_l beside, not pooled.

**Rulings** where the registration leaves a detail open:
- *K_harmful at K\\** is study 3's interpolation in ln K_harmful (its matched-K_harmful
  analysis): ln K_harmful(K*) = (1 − w)·ln K_harmful_lo + w·ln K_harmful_hi with the block's
  endpoints and weight (``study3_blocks.interpolated``); none unless both are positive and finite.
- *The decomposition* takes each seed's three orders as one group: σ²_w the mean over seeds of
  their sample variances (ddof 1), σ²_b = Var(seed means) − σ²_w/3 (ddof 1, kept when negative),
  the share σ²_w/(σ²_w + max(σ²_b, 0)), undefined when that is 0/0. The interval draws n seed
  indices with replacement per resample (``numpy.random.default_rng(20261111).integers``, one call
  per resample, as the registered block bootstrap does), 10 000 resamples, numpy's linear
  percentiles at 2.5 and 97.5; a resample with an undefined share is left out and counted.
- *The Spearman ρ* pools the 3·n pairs of deviations (each value minus its seed's mean of the
  three orders) over the seeds with h and K_harmful at K* in all three orders, by average ranks
  (``study5_analysis.spearman``).
- *A grid point* is a scale of the search, c = 2^k·c₀; the two pods' points are matched by k,
  each with its own c₀. The relative difference is |K₃ − K₂|/K₂; the refinement points c₁ and c₂
  are not grid points.
- *Means* over seeds are t intervals at the fixation record's level (0.95), as the registered
  ladder's.

**The record** (--out, refused when it exists, written whole by a link): strict JSON, sorted keys,
numbers, ids, states and reasons, never a text; every input's sha256 (``inputs``: ``pod3:`` and
``pod2:`` relative paths). stdout carries numbers, ids and the record's sha256 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from ..paths import REPO_ROOT
from ..tracking import git_state
from . import study5_analysis as sa
from .study3_blocks import interpolated
from .study3_estimates import DENOMINATOR
from .study3_records import read_theta0

if TYPE_CHECKING:
    from . import study5_analysis_cli as cli

PROG = "study5_ext1"
KIND, VERSION = "study5-extension-1-analysis", 1
REGISTRATION = "docs/study5-preregistration.md"
HEADING = "### Extension 1 (2026-11-11): the ladder's unbracketed rungs and order repeats"
POD3_PREFIX = "pod3-"  # pod 3's archive namespace (scripts/study5_campaigns.py)
E1_SEEDS = sa.LADDER_SEEDS  # 50000-50005
E1_RUNGS = ("gaussian_d", "linear_no_d")  # the rungs pod 3 repeated
POD2_RUNG = "linear"  # δ̂_s, from pod 2
EXTENSIONS = 8  # the extension's most ×2 (÷2) extensions of a rung's grid
E2_SEEDS = tuple(range(50006, 50016))
ETA_L, ETA_U = 1.3e-05, 1.83e-05  # study 3's candidate-1 rates of F (study5_analysis_cli's)
ENDPOINT = "step-25"  # a run's step-25 evaluation, its outcome's condition
REGISTERED_KIND = "study5-analysis"  # the registered analysis's record
RATES = (("eta_l", ETA_L), ("eta_u", ETA_U))
NEW_ORDERS = (("order1", 1000), ("order2", 2000))  # train.order_seed = s + offset
ORDERS = ("original", *(name for name, _ in NEW_ORDERS))
OPENING_KIND = "study5-e2-opening"
CHECKOUT = REPO_ROOT  # the checkout whose history must hold the dated note the opening names
FULL_SHA = re.compile(r"[0-9a-f]{40}")
# Extension 1's order repeats (E2), refused wherever a run is read that is not one of them, as the
# extension-2 analysis refuses them (study5_ext2.e2_repeat): an E2 cell in the run's name, an E2
# seed with train.order_seed = s + 1000 or s + 2000, or any of these 20 order seeds
E2_NAME = re.compile(r"(?:^|-)s5-order[12]-")
E2_ORDER_SEEDS = frozenset(s + offset for s in E2_SEEDS for _, offset in NEW_ORDERS)
LADDER_RUN_TYPE = "study5_ladder"  # evaluation.LADDER_RUN_TYPE: a rung's evaluation run
LEVEL = 0.95  # the fixation record's scan_intervals level, the registered ladder's
RESAMPLES, BOOTSTRAP_SEED = 10_000, 20261111
GRID_TOLERANCE = 1e-9  # |log2(c/c₀) − k| of a grid point
CHECKED = ("status", "c", "K_benign", "K_harmful", "classes", "h_s")  # pod 2's rows, re-read
MEASURES = ("harmful", "harmful_high", "refusal", "safe_non_refusal", *sa.METRICS, "c_over_c0")
ENDPOINT_MEASURES = ("harmful", "harmful_high", "refusal", "safe_non_refusal")
LADDER_RECORD = re.compile(r"study5_records/[^/]+/ladder/s(?P<seed>[0-9]+)-(?P<rung>[a-z_]+)\.json")


def _cli():
    """The registered analysis's readers (``study5_analysis_cli``), imported when reading: its
    label readers import torch (through ``run_judging``), which importing this module does not."""
    from . import study5_analysis_cli

    return study5_analysis_cli


def refuse(why: str) -> NoReturn:
    """Nothing is written (``study5_analysis_cli.Refused``)."""
    _cli().refuse(why)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ pure computations


def k_harmful_at_kstar(block: Mapping[str, Any] | None, k_harmful: Mapping[str, Any]) -> sa.Value:
    """ln K_harmful at K* (module docstring, "Rulings"): ``k_harmful`` run id → K_harmful of the
    block's two endpoints; the reason when there is none."""
    if block is None:
        return "no supported block"
    entry = block["arms"][sa.ARM]
    ends = [k_harmful.get(entry[end]) for end in ("lo", "hi")]
    if not all(sa._is_number(k) and k > 0 for k in ends):
        return "K_harmful not positive and finite at both endpoints"
    return interpolated(math.log(ends[0]), math.log(ends[1]), entry["w"])


def components(values: np.ndarray) -> dict[str, Any]:
    """The one-way decomposition of ``values`` (seeds × orders): σ²_w, Var(seed means), σ²_b and
    the order share (module docstring, "Rulings"); None for each under two seeds."""
    n, orders = values.shape
    out: dict[str, Any] = dict.fromkeys(("sigma2_w", "var_seed_means", "sigma2_b",
                                         "order_share"))  # fmt: skip
    if n < 2:
        return out
    sigma2_w = float(values.var(axis=1, ddof=1).mean())
    var_means = float(values.mean(axis=1).var(ddof=1))
    sigma2_b = var_means - sigma2_w / orders
    denominator = sigma2_w + max(sigma2_b, 0.0)
    share = sigma2_w / denominator if denominator > 0 else None
    return {"sigma2_w": sigma2_w, "var_seed_means": var_means, "sigma2_b": sigma2_b,
            "order_share": share}  # fmt: skip


def share_interval(values: np.ndarray, *, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED,
                   level: float = LEVEL) -> dict[str, Any]:  # fmt: skip
    """The order share's percentile interval from seed-level bootstrap resamples (module
    docstring, "Rulings")."""
    n = len(values)
    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(resamples):
        share = components(values[rng.integers(0, n, size=n)])["order_share"]
        if share is not None and math.isfinite(share):
            shares.append(share)
    lower, upper = (np.percentile(shares, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
                    if shares else (None, None))  # fmt: skip
    return {"level": level, "resamples": resamples, "seed": seed,
            "method": "seeds resampled with replacement; numpy percentile, linear",
            "undefined": resamples - len(shares),
            "lower": None if lower is None else float(lower),
            "upper": None if upper is None else float(upper)}  # fmt: skip


def deviation_spearman(h: Mapping[int, Sequence[float]],
                       ln_k: Mapping[int, Sequence[float]]) -> dict[str, Any]:  # fmt: skip
    """Spearman ρ between ln K_harmful at K* and h of the deviations from each seed's mean,
    pooled over the seeds (both keys) and their orders."""
    seeds = sorted(set(h) & set(ln_k))
    dh, dk = [], []
    for s in seeds:
        mh, mk = sum(h[s]) / len(h[s]), sum(ln_k[s]) / len(ln_k[s])
        dh += [v - mh for v in h[s]]
        dk += [v - mk for v in ln_k[s]]
    rho = sa.spearman(dk, dh) if len(dh) >= 2 else math.nan
    return {"of": ["ln K_harmful at K*", "h"], "seeds": seeds, "pairs": len(dh),
            "rho": sa._finite(rho)}  # fmt: skip


def grid(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """A rung record's grid points by k, c = 2^k·c₀ (module docstring, "Rulings")."""
    c0 = record.get("c0")
    out: dict[int, dict[str, Any]] = {}
    if not (sa._is_number(c0) and c0 > 0):
        return out
    for scale in (record.get("search") or {}).get("scales") or []:
        c = scale.get("c")
        if not (sa._is_number(c) and c > 0):
            continue
        x = math.log2(c / c0)
        if abs(x - round(x)) <= GRID_TOLERANCE:
            out[round(x)] = {"c": c, "K_benign": scale.get("K_benign")}
    return out


def compare_grids(pod2: Mapping[str, Any] | str, pod3: Mapping[str, Any] | str) -> dict[str, Any]:
    """One seed's rung on both pods: every grid point both measured (k, c and K on each, the
    relative difference), the points pod 2 measured that pod 3 did not, or why there are none."""
    if not isinstance(pod2, Mapping) or not isinstance(pod3, Mapping):
        why = "; ".join(str(r) for r in (pod2, pod3) if not isinstance(r, Mapping))
        return {"c0_pod2": None, "c0_pod3": None, "points": [], "pod2_only": [], "reason": why}
    a, b = grid(pod2), grid(pod3)
    points = []
    for k in sorted(set(a) & set(b)):
        k2, k3 = a[k]["K_benign"], b[k]["K_benign"]
        rel = (abs(k3 - k2) / abs(k2) if sa._is_number(k2) and sa._is_number(k3) and k2 != 0
               else None)  # fmt: skip
        points.append({"k": k, "c_pod2": a[k]["c"], "c_pod3": b[k]["c"], "K_pod2": k2,
                       "K_pod3": k3, "relative_difference": rel})  # fmt: skip
    return {"c0_pod2": pod2.get("c0"), "c0_pod3": pod3.get("c0"), "points": points,
            "pod2_only": sorted(set(a) - set(b)), "reason": None}  # fmt: skip


def reproducibility(pod2: Mapping[int, Mapping[str, Any]],
                    pod3: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """Per E1 rung, pod 3's K at the grid points pod 2 also measured, beside pod 2's, and the
    largest relative difference (a check of reproducibility, not a gate)."""
    out: dict[str, Any] = {"rule": "per grid point c = 2^k c0 measured by both pods (matched by "
                                   "k, each pod's own c0): |K_pod3 - K_pod2| / K_pod2; the "
                                   "largest per rung; not a gate"}  # fmt: skip
    for rung in E1_RUNGS:
        per_seed, largest, count = [], None, 0
        for seed in E1_SEEDS:
            found = compare_grids(pod2[seed][rung], pod3[seed][rung])
            per_seed.append({"seed": seed, **found})
            for point in found["points"]:
                count += 1
                rel = point["relative_difference"]
                if rel is not None and (largest is None or rel > largest["relative_difference"]):
                    largest = {"seed": seed, "k": point["k"], "relative_difference": rel}
        out[rung] = {"per_seed": per_seed, "points": count, "largest": largest}
    return out


# ------------------------------------------------------------------ the inputs


def own_code(prog: str = PROG) -> dict[str, Any]:
    """This checkout's code (``prog``: the analysis module): a clean checkout of a known
    commit."""
    state = git_state(REPO_ROOT)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse("the analysis runs from a clean checkout of a known commit")
    return {
        "commit": state["commit"],
        "dirty": False,
        "module": f"zero_field_refusal.analysis.{prog}",
    }


def registration_text() -> str:
    """The registration at this checkout's commit (``own_code``: a clean checkout)."""
    return (REPO_ROOT / REGISTRATION).read_text(encoding="utf-8")


def registered_record(path: Path, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The registered analysis's record: its sha256 one the registration names, its kind the
    registered analysis's."""
    try:
        data = Path(path).read_bytes()
        found = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse(f"the registered analysis record does not read ({type(exc).__name__})")
    digest = sha256(data)
    if digest not in text:
        refuse(f"the analysis record read ({digest[:12]}) is not one the registration names")
    if not isinstance(found, dict) or found.get("kind") != REGISTERED_KIND:
        refuse(f"{Path(path).name} is not a {REGISTERED_KIND} record")
    return found, {"record": Path(path).name, "sha256": digest}


def opening_record(path: Path) -> dict[str, Any]:
    """The decision that opens E2: a JSON object of kind ``OPENING_KIND`` naming the dated note
    (``note``, its path in the repository), the commit that holds it (``commit``, a full sha) and
    its sha256 (``note_sha256``), the note checked in this checkout's history (``note_problem``);
    refused otherwise. What the record keeps of it."""
    return read_opening(path)[0]


def read_opening(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``opening_record`` and the opening record itself, for the fields a test of E2 binds."""
    path = Path(path)
    if not path.is_file():
        refuse("--open-e2: the opening record does not exist")
    try:
        data = path.read_bytes()
        found = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse(f"--open-e2: the opening record does not read ({type(exc).__name__})")
    if not isinstance(found, dict) or found.get("kind") != OPENING_KIND:
        refuse(f"--open-e2: the opening record is not of kind {OPENING_KIND}")
    if not isinstance(found.get("note_sha256"), str):
        refuse("--open-e2: the opening record holds no note_sha256 string")
    why = note_problem(found.get("commit"), found.get("note"), found["note_sha256"])
    if why is not None:
        refuse(f"--open-e2: {why}")
    return {"record": path.name, "sha256": sha256(data), "commit": found["commit"],
            "note": found["note"], "note_sha256": found["note_sha256"]}, found  # fmt: skip


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=CHECKOUT, capture_output=True, check=False)


def note_problem(commit: Any, note: Any, digest: str) -> str | None:
    """Why the dated note does not stand (None: it does): ``commit`` a full sha in this
    checkout's history, an ancestor of HEAD or HEAD itself; ``note`` a path in the repository,
    whose content at that commit has the sha256 ``digest``, and the same content at HEAD."""
    if not isinstance(commit, str) or not FULL_SHA.fullmatch(commit):
        return "its commit is not a full git sha"
    parts = PurePosixPath(note).parts if isinstance(note, str) else ()
    if not parts or note.startswith("/") or ".." in parts or "\\" in note:
        return "its note is not a path in the repository"
    if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        return f"its commit {commit[:12]} is not in this checkout's history"
    if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return f"its commit {commit[:12]} is not an ancestor of this checkout's HEAD"
    then = _git("show", f"{commit}:{note}")
    if then.returncode != 0:
        return f"its note {note} is not in its commit {commit[:12]}"
    if sha256(then.stdout) != digest:
        return f"its note {note} at {commit[:12]} does not have the sha256 note_sha256 names"
    now = _git("show", f"HEAD:{note}")
    if now.returncode != 0 or now.stdout != then.stdout:
        return f"its note {note} at HEAD is not the note at {commit[:12]}"
    return None


def pod3_archive(runs: Path) -> str:
    """Pod 3's archive id: the one folder ``pod3-…`` under the root's ``study5_records/``."""
    folder = Path(runs) / "study5_records"
    entries = sorted(folder.iterdir()) if folder.is_dir() else []
    found = [p.name for p in entries if p.is_dir() and p.name.startswith(POD3_PREFIX)]
    if len(found) != 1:
        refuse(f"--pod3-runs holds {len(found)} archives {POD3_PREFIX}… under study5_records/, "
               "not one")  # fmt: skip
    return found[0]


class Pod2:
    """The runs root the registered analysis read: each file located through the record's
    ``inputs`` (``runs:<relative path>``) and read by a reader of its own, so that every file
    read is checked against them (``check``)."""

    def __init__(self, root: Path, registered: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.reader = _cli().Reader({"pod2": self.root})
        listed = registered.get("inputs") or {}
        self.inputs = {name.removeprefix("runs:"): digest for name, digest in listed.items()
                       if name.startswith("runs:")}  # fmt: skip

    def run_dir(self, run_id: Any, file: str) -> Path | None:
        """The run directory of ``run_id`` whose ``file`` the registered analysis read."""
        if not isinstance(run_id, str):
            return None
        suffix = f"/{run_id}/{file}"
        found = [rel[: -len(file) - 1] for rel in self.inputs if rel.endswith(suffix)]
        return self.root / found[0] if len(found) == 1 else None

    def ladder(self, seed: int, rung: str) -> dict[str, Any] | str:
        """Pod 2's record of a seed's rung, or why there is none."""
        rels = [rel for rel in self.inputs
                if (m := LADDER_RECORD.fullmatch(rel)) and int(m["seed"]) == seed
                and m["rung"] == rung]  # fmt: skip
        what = f"pod 2 ladder {rung}"
        if len(rels) != 1:
            return f"{what}: {len(rels)} records among the registered analysis's inputs"
        found = _cli().record(self.reader, self.root / rels[0], "study5-ladder", seed, what)
        if isinstance(found, Mapping) and found.get("rung") != rung:
            return f"{what}: the record is of rung {found.get('rung')!r}"
        return found

    def f_run(self, run_id: str, seed: int) -> dict[str, Any] | str:
        """A fresh F run's records (``study5_analysis_cli.f_run``), its rate from its config."""
        run_dir = self.run_dir(run_id, "config.yaml")
        if run_dir is None:
            return f"{run_id}: not among the registered analysis's inputs"
        try:
            cfg = OmegaConf.create(self.reader.bytes(run_dir / "config.yaml").decode())
            lr = float(cfg.optimizer.lr)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return f"{run_id}: its config does not read ({type(exc).__name__})"
        return _cli().f_run(self.reader, run_dir, seed, lr)

    def theta0(self, registered: Mapping[str, Any]) -> dict[str, float]:
        """θ₀'s accuracies (``study3_records.read_theta0``) from the registered analysis's θ₀
        run; refused when it does not read."""
        run_id = ((registered.get("descriptive") or {}).get("theta0") or {}).get("run_id")
        run_dir = self.run_dir(run_id, "summary.json")
        if run_dir is None:
            refuse("θ₀'s evaluation is not among the registered analysis's inputs")
        try:
            self.reader.bytes(run_dir / "summary.json")  # hashed and checked as an input
            return read_theta0(run_dir)
        except (OSError, ValueError) as exc:
            refuse(f"θ₀'s evaluation does not read: {self.reader.scrub(str(exc))}")

    def check(self) -> None:
        """Every file read here is a registered input with the same sha256; refused otherwise."""
        for name, digest in sorted(self.reader.hashes.items()):
            root, _, rel = name.partition(":")
            if root != "pod2" or self.inputs.get(rel) != digest:
                refuse(f"pod 2's {rel or name} is not the file the registered analysis read")


def outcomes_of(registered: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(o["seed"]): o for o in registered.get("outcomes") or [] if isinstance(o, Mapping)}


def h_value(outcome: Mapping[str, Any] | None) -> sa.Value:
    """A fresh seed's registered h_s, or the registered reason it has none."""
    if outcome is None:
        return "not among the registered analysis's outcomes"
    h = outcome.get("h")
    return float(h) if sa._is_number(h) else str(outcome.get("reason"))


# ------------------------------------------------------------------ E1


def rung_problem(found: Mapping[str, Any], rung: str, f_run: str | None) -> str | None:
    """Why pod 3's rung record is not the extension's for the seed (None: it is)."""
    what = f"ladder {rung}"
    if found.get("rung") != rung:
        return f"{what}: the record is of rung {found.get('rung')!r}"
    if f_run is None:
        return f"{what}: pod 2's rungs of the seed name no single F run"
    if (found.get("f_run") or {}).get("run_id") != f_run:
        return f"{what}: the record is not of the seed's η_l run"
    if found.get("extensions_allowed") != EXTENSIONS:
        return (f"{what}: the record allows {found.get('extensions_allowed')!r} extensions, not "
                f"{EXTENSIONS}")  # fmt: skip
    return None


def e2_repeat(config: Mapping[str, Any] | None, name: str) -> str | None:
    """Why a run directory is one of extension 1's order repeats (E2), or None (as
    ``study5_ext2.e2_repeat``): an E2 cell in its name, or in its config an E2 seed with
    ``train.order_seed`` = seed + 1000 or + 2000, or one of those order seeds."""
    if E2_NAME.search(name):
        return f"{name}: an extension-1 order repeat (E2) by its name"
    if not isinstance(config, Mapping):
        return None
    seed = config.get("seed")
    train = config.get("train") if isinstance(config.get("train"), Mapping) else {}
    order = train.get("order_seed")
    offsets = tuple(offset for _, offset in NEW_ORDERS)
    if isinstance(order, int) and not isinstance(order, bool) and (
            order in E2_ORDER_SEEDS or (seed in E2_SEEDS and order - seed in offsets)):  # fmt: skip
        return f"{name}: an extension-1 order repeat (E2), order seed {order}"
    return None


def rung_run(reader: cli.Reader, root: Path, record: Mapping[str, Any], seed: int, rung: str,
             f_run: str | None) -> Path | str:  # fmt: skip
    """The evaluation run a rung record names, guarded before any of its files but its config is
    read: an E2 order repeat (``e2_repeat``: by the name, then by the config) refuses the
    analysis; a run that is not the seed's ``rung`` rung of its η_l run ``f_run`` (run type,
    seed, rung, F run) is a reason."""
    what = f"ladder {rung}"
    experiment, run_id = record.get("experiment"), record.get("run_id")
    if not all(isinstance(x, str) and x and "/" not in x and x not in (".", "..")
               for x in (experiment, run_id)):  # fmt: skip
        return f"{what}: the record names no run directory"
    sealed = e2_repeat(None, run_id)
    if sealed:
        refuse(f"{sealed}: sealed, never read")
    run_dir = Path(root) / experiment / run_id
    if not run_dir.is_dir():
        return f"{what}: its run directory is missing"
    try:
        config = yaml.safe_load(reader.bytes(run_dir / "config.yaml"))
    except (OSError, yaml.YAMLError) as exc:
        return f"{what}: its config does not read ({type(exc).__name__})"
    sealed = e2_repeat(config if isinstance(config, Mapping) else None, run_id)
    if sealed:
        refuse(f"{sealed}: sealed, never read")
    found = None
    if isinstance(config, Mapping):
        ladder = _at(config, "study5", "ladder")
        found = (config.get("run_type"), config.get("seed"), _at(ladder, "rung"),
                 _at(ladder, "f_run"))  # fmt: skip
    if f_run is None or found != (LADDER_RUN_TYPE, seed, rung, f_run):
        return f"{what}: its run is not the seed's {rung} rung of its eta_l run"
    return run_dir


def _at(value: Any, *keys: str) -> Any:
    """``value[k1][k2]…``, None where a level is not a mapping or lacks the key."""
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def rung_labels(labels: cli.Labels, root: Path, record: Mapping[str, Any], seed: int, rung: str,
                f_run: str | None) -> dict[str, int] | str:  # fmt: skip
    from ..evaluation import ladder_condition  # torch, as the label readers

    return labels.at(rung_run(labels.reader, root, record, seed, rung, f_run),
                     ladder_condition(rung))  # fmt: skip


def row_value(row: Mapping[str, Any], measure: str) -> sa.Value:
    """A ladder row's measure, or why it has none."""
    if row["status"] != "evaluated":
        return f"{row['rung']}: {row['status']}"
    if measure in (*sa.CLASSES, "harmful_high") and row.get("classes") is None:
        return f"{row['rung']}: {row.get('labels') or 'no labels'}"
    found = row["classes"][measure] if measure in sa.CLASSES else row.get(measure)
    return found if sa._is_number(found) else f"{row['rung']}: {measure} {found}"


def means_of(values: Mapping[str, Mapping[int, sa.Value]],
             seeds: Sequence[int]) -> dict[str, Any]:  # fmt: skip
    return {m: sa.summary(v, seeds, LEVEL) for m, v in values.items()}


def e1_part(reader: cli.Reader, runs: Path, archive: str, pod2: Pod2, labels3: cli.Labels,
            labels2: cli.Labels, registered: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """E1 (module docstring)."""
    cli = _cli()
    outcomes = outcomes_of(registered)
    h = {s: h_value(outcomes.get(s)) for s in E1_SEEDS}
    records: dict[int, dict[str, Any]] = {}
    theirs: dict[int, dict[str, Any]] = {}
    counts: dict[tuple[int, str], Any] = {}
    for seed in E1_SEEDS:
        theirs[seed] = {rung: pod2.ladder(seed, rung) for rung in (*E1_RUNGS, POD2_RUNG)}
        named = {(r.get("f_run") or {}).get("run_id") for r in theirs[seed].values()
                 if isinstance(r, Mapping)}  # fmt: skip
        f_run = next(iter(named)) if len(named) == 1 else None
        records[seed] = {}
        for rung in E1_RUNGS:
            path = runs / "study5_records" / archive / "ladder" / f"s{seed}-{rung}.json"
            found = cli.record(reader, path, "study5-ladder", seed, f"ladder {rung}")
            if isinstance(found, Mapping):
                found = rung_problem(found, rung, f_run) or found
            records[seed][rung] = found
        records[seed][POD2_RUNG] = theirs[seed][POD2_RUNG]
        for rung, found in records[seed].items():
            if isinstance(found, Mapping) and found.get("status") == "evaluated":
                labels, root = (labels2, pod2.root) if rung == POD2_RUNG else (labels3, runs)
                counts[(seed, rung)] = rung_labels(labels, root, found, seed, rung, f_run)
    table = sa.ladder_table(records, counts, h, E1_SEEDS, (*E1_RUNGS, POD2_RUNG), LEVEL)
    for row in table["rows"]:
        record = records[row["seed"]][row["rung"]]
        found = counts.get((row["seed"], row["rung"]))
        c, c0 = row.get("c"), row.get("c0")
        row |= {
            "pod": 2 if row["rung"] == POD2_RUNG else 3,
            "extensions_allowed": record.get("extensions_allowed"),
            "c_over_c0": c / c0 if sa._is_number(c) and sa._is_number(c0) and c0 > 0 else None,
            "missing_labels": found["missing"] if isinstance(found, Mapping) else None,
            "harmful_high": ((found["harmful"] + found["missing"]) / DENOMINATOR
                             if isinstance(found, Mapping) else None),
            "experiment": record.get("experiment"),
            "run_id": record.get("run_id"),
        }  # fmt: skip
    check_linear_rows(table, registered)
    rows = [r for r in table["rows"] if r["rung"] in E1_RUNGS]
    linear = [r for r in table["rows"] if r["rung"] == POD2_RUNG]
    endpoint = {}
    for seed in E1_SEEDS:
        outcome = outcomes.get(seed) or {}
        classes = outcome.get("classes") or {}
        endpoint[seed] = {m: classes[m] if sa._is_number(classes.get(m)) else h[seed]
                          for m in ENDPOINT_MEASURES}  # fmt: skip
    means: dict[str, Any] = {}
    for rung in (*E1_RUNGS, POD2_RUNG):
        values: dict[str, dict[int, sa.Value]] = {m: {} for m in MEASURES}
        found = {r["seed"]: r for r in table["rows"] if r["rung"] == rung}
        absent = {e["seed"]: e["reason"] for e in table["missing"] if e["rung"] == rung}
        for seed in E1_SEEDS:
            for m in MEASURES:
                values[m][seed] = (row_value(found[seed], m) if seed in found
                                   else absent.get(seed, "no record"))  # fmt: skip
        means[rung] = means_of(values, E1_SEEDS)
    means["endpoint"] = means_of({m: {s: endpoint[s][m] for s in E1_SEEDS}
                                  for m in ENDPOINT_MEASURES}, E1_SEEDS)  # fmt: skip
    return {
        "seeds": list(E1_SEEDS), "rungs": list(E1_RUNGS), "extensions_allowed": EXTENSIONS,
        "rows": rows,
        "not_evaluated": [e for e in table["not_evaluated"] if e["rung"] in E1_RUNGS],
        "missing": [e for e in table["missing"] if e["rung"] in E1_RUNGS],
        "beside": {
            "linear": {"rows": linear,
                       "not_evaluated": [e for e in table["not_evaluated"]
                                         if e["rung"] == POD2_RUNG],
                       "missing": [e for e in table["missing"] if e["rung"] == POD2_RUNG]},
            "endpoint": {str(s): {"h_s": sa._json_value(h[s]),
                                  **{m: sa._json_value(endpoint[s][m]) for m in ENDPOINT_MEASURES}}
                         for s in E1_SEEDS},
        },
        "means": means,
        "shuffled_d_trigger": table["shuffled_d_trigger"],
        "shuffled_d_reading": "a decision of runs makes linear_shuffled_d a separate later "
                              "decision (registration, Extension 1); it is not run here",
        "reproducibility": reproducibility(theirs, records),
    }  # fmt: skip


def check_linear_rows(table: Mapping[str, Any], registered: Mapping[str, Any]) -> None:
    """Each registered row of pod 2's ``linear`` rung on the E1 seeds is read again here, equal in
    its status, scale, K, classes and h_s; refused otherwise."""
    cli = _cli()
    stored = {(int(r["seed"]), r["rung"]): r
              for r in ((registered.get("descriptive") or {}).get("ladder") or {}).get("rows") or []
              if r.get("rung") == POD2_RUNG and int(r["seed"]) in E1_SEEDS}  # fmt: skip
    mine = {(r["seed"], r["rung"]): r for r in table["rows"] if r["rung"] == POD2_RUNG}
    for key in sorted(set(stored) | set(mine)):
        if key not in stored or key not in mine:
            why = next((e["reason"] for e in table["missing"] if (e["seed"], e["rung"]) == key),
                       "no registered row")  # fmt: skip
            refuse(f"pod 2's linear rung of seed {key[0]} is not read as the registered analysis "
                   f"read it ({why})")  # fmt: skip
        try:
            cli.same_tree({k: stored[key].get(k) for k in CHECKED},
                          sa.finite_tree({k: mine[key].get(k) for k in CHECKED}),
                          f"seed {key[0]} linear")  # fmt: skip
        except cli.Mismatch as exc:
            refuse(f"pod 2's linear rung does not give its registered row (at {exc})")


# ------------------------------------------------------------------ E2


def order_run(reader: cli.Reader, run_dir: Path | str, seed: int, lr: float,
              order_seed: int) -> dict[str, Any] | str:  # fmt: skip
    """An order run's records (``study5_analysis_cli.f_run``), its config also of
    ``train.order_seed`` and role ``control``; the reason otherwise."""
    cli = _cli()
    if isinstance(run_dir, str):
        return run_dir
    try:
        cfg = OmegaConf.create(reader.bytes(Path(run_dir) / "config.yaml").decode())
        if not isinstance(cfg, DictConfig):
            raise ValueError("not a mapping")
        found = OmegaConf.select(cfg, "train.order_seed")
        role = OmegaConf.select(cfg, "study5.role")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{Path(run_dir).name}: its config does not read ({type(exc).__name__})"
    if found != order_seed or role != "control":
        return f"{Path(run_dir).name}: its config is not order seed {order_seed}, role control"
    return cli.f_run(reader, run_dir, seed, lr)


def rate_of(lr: Any) -> str | None:
    for name, eta in RATES:
        if sa._is_number(lr) and math.isclose(lr, eta, rel_tol=1e-9):
            return name
    return None


def entry_of(seed: int, order: str, order_seed: int, source: str, runs: Mapping[str, Any],
             outcome: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """One seed and order (``runs``: rate → its ``f_run`` or why it is missing; ``outcome``:
    ``study5_analysis.seed_outcome``'s fields)."""
    k = {key: {} for key in ("K_benign", "K_harmful")}
    by_id: dict[str, Any] = {}
    for name, _ in RATES:
        run = runs.get(name, "no run")
        for key in k:
            if isinstance(run, str):
                k[key][name] = {"missing": run}
            else:
                value = (run["row"] or {}).get(key)
                k[key][name] = (
                    value
                    if sa._is_number(value)
                    else {"missing": f"{key} {value!r}" if run["row"] else "no step-25 evaluation"}
                )
        if isinstance(run, Mapping):
            by_id[run["run_id"]] = (run["row"] or {}).get("K_harmful")
    classes = outcome.get("classes")
    ln_k = k_harmful_at_kstar(outcome.get("block"), by_id)
    return {"seed": seed, "order": order, "order_seed": order_seed, "source": source,
            "runs": {name: run["run_id"] if isinstance(run, Mapping) else None
                     for name, run in runs.items()},
            **k, "support": outcome.get("support"), "h": outcome.get("h"),
            "h_high": (classes or {}).get("harmful_high"), "classes": classes,
            "ln_K_harmful_at_kstar": sa._json_value(ln_k),
            "K_harmful_at_kstar": math.exp(ln_k) if sa._is_number(ln_k) else None,
            "reason": outcome.get("reason")}  # fmt: skip


def original_order(pod2: Pod2, seed: int, outcome: Mapping[str, Any] | None) -> dict[str, Any]:
    """The seed's original order: its registered outcome, the K values from the two runs its
    support names."""
    if outcome is None:
        outcome = {"h": None, "reason": "not among the registered analysis's outcomes"}
    runs: dict[str, Any] = {}
    problems = []
    for end in ("lo", "hi"):
        run_id = (outcome.get("support") or {}).get(end)
        if not isinstance(run_id, str):
            continue
        run = pod2.f_run(run_id, seed)
        name = rate_of(run["lr"]) if isinstance(run, Mapping) else None
        if name is None or name in runs:
            problems.append(run if isinstance(run, str) else f"{run_id}: not at η_l or η_u once")
        else:
            runs[name] = run
    why = "; ".join(problems) or "the registered support names no run at this rate"
    runs = {name: runs.get(name, why) for name, _ in RATES}
    return entry_of(seed, "original", seed, "the registered analysis", runs, outcome)


def new_order(reader: cli.Reader, runs: Path, archive: str, labels: cli.Labels,
              theta0: Mapping[str, float], seed: int, order: str,
              offset: int) -> dict[str, Any]:  # fmt: skip
    """A new order of the seed: its two runs (the launcher's journal), their outcome at K*."""
    campaign = f"study5-{archive}-orders-s{seed}"
    return order_parts(reader, runs, campaign, labels, theta0, seed, order, offset)[0]


Parts = tuple[dict[str, Any], dict[str, Any], dict[str, Any]]


def order_parts(reader: cli.Reader, runs: Path, campaign: str, labels: cli.Labels,
                theta0: Mapping[str, float], seed: int, order: str, offset: int,
                source: str = "pod 3") -> Parts:  # fmt: skip
    """A new order of the seed in the launcher campaign ``campaign``: its entry (``entry_of``:
    the registered outcome at K* of its two runs), the runs (rate → ``order_run``'s records or
    why they are missing) and their label counts at the endpoint (run id → counts or why)."""
    cli = _cli()
    found: dict[str, Any] = {}
    for name, eta in RATES:
        cell = (campaign, f"s5-{order}-{eta!r}-s{seed}/s{seed}")
        run_dir = cli.completed(reader, runs, cell, f"{order} {name}")
        found[name] = order_run(reader, run_dir, seed, eta, seed + offset)
    endpoints, missing, counts = [], [], {}
    for run in found.values():
        if isinstance(run, str):
            missing.append(run)
            continue
        endpoints.append(sa.endpoint_of(run["run_id"], seed=seed, lr=run["lr"], stop=run["stop"],
                                        row=run["row"], theta0=theta0,
                                        steps=run["steps"]))  # fmt: skip
        counts[run["run_id"]] = labels.at(run["run_dir"], ENDPOINT)
    outcome = sa.seed_outcome(seed, endpoints, missing, counts)
    return entry_of(seed, order, seed + offset, source, found, outcome), found, counts


def e2_part(reader: cli.Reader, runs: Path, archive: str, pod2: Pod2, labels: cli.Labels,
            registered: Mapping[str, Any],
            opening: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """E2 (module docstring): read only once the opening record has been checked."""
    theta0 = pod2.theta0(registered)
    outcomes = outcomes_of(registered)
    entries = []
    for seed in E2_SEEDS:
        entries.append(original_order(pod2, seed, outcomes.get(seed)))
        for order, offset in NEW_ORDERS:
            entries.append(new_order(reader, runs, archive, labels, theta0, seed, order, offset))
    by = {(e["seed"], e["order"]): e for e in entries}
    h, ln_k, excluded, excluded_rho = {}, {}, [], []
    for seed in E2_SEEDS:
        found = [by[(seed, o)] for o in ORDERS]
        absent = [f"{e['order']}: {e['reason']}" for e in found if not sa._is_number(e["h"])]
        if absent:
            excluded.append({"seed": seed, "reason": "; ".join(absent)})
            continue
        h[seed] = [float(e["h"]) for e in found]
        lacking = [f"{e['order']}: {e['ln_K_harmful_at_kstar']['missing']}" for e in found
                   if not sa._is_number(e["ln_K_harmful_at_kstar"])]  # fmt: skip
        if lacking:
            excluded_rho.append({"seed": seed, "reason": "; ".join(lacking)})
        else:
            ln_k[seed] = [float(e["ln_K_harmful_at_kstar"]) for e in found]
    seeds = sorted(h)
    values = np.asarray([h[s] for s in seeds], dtype=float).reshape(len(seeds), len(ORDERS))
    with_missing = sum(1 for s in seeds for o in ORDERS
                       if sa._is_number(by[(s, o)]["h_high"])
                       and by[(s, o)]["h_high"] != by[(s, o)]["h"])  # fmt: skip
    decomposition = {
        "rule": "seeds as groups, their three orders (original, order1, order2) the members: "
                "sigma2_w the mean of the seeds' sample variances (ddof 1); sigma2_b = Var(seed "
                "means) (ddof 1) - sigma2_w/3, kept when negative; order share sigma2_w / "
                "(sigma2_w + max(sigma2_b, 0)); descriptive, no test",
        "orders": list(ORDERS), "seeds": seeds, "n": len(seeds), "excluded": excluded,
        **components(values),
        "interval": share_interval(values) if len(seeds) >= 2 else None,
        "entries_with_missing_labels": with_missing,
        "per_seed": {str(s): {"mean": float(np.mean(h[s])), "variance": float(np.var(h[s], ddof=1))}
                     for s in seeds},
    }  # fmt: skip
    spearman = {**deviation_spearman(h, ln_k), "excluded": excluded_rho,
                "rule": "deviations from each seed's mean of its three orders, pooled over seeds "
                        "and orders; Spearman by average ranks"}  # fmt: skip
    order = ((registered.get("descriptive") or {}).get("controls") or {}).get("order")
    controls: dict[str, Any] = {"missing": "no registered order controls"}
    if isinstance(order, Mapping):
        controls = {"source": "the registered analysis's descriptive.controls.order: each measure "
                              "of the order control minus the seed's own fresh run, at eta_l; "
                              "not pooled",
                    "harmful": order.get("harmful"), "per_seed": order.get("per_seed")}  # fmt: skip
    return {
        "sealed": False, "opening": dict(opening), "seeds": list(E2_SEEDS),
        "orders": {"original": "s", **{name: f"s + {offset}" for name, offset in NEW_ORDERS}},
        "theta0": theta0, "entries": entries,
        "unsupported": [{"seed": e["seed"], "order": e["order"], "reason": e["reason"]}
                        for e in entries if e["h"] is None],
        "decomposition": decomposition, "spearman": spearman, "registered_controls": controls,
    }  # fmt: skip


# ------------------------------------------------------------------ the record


def labeling(labels: cli.Labels) -> dict[str, Any]:
    """When each of pod 3's runs read here was labeled (its label rows' earliest ``created_at``
    and its receipt's ``completed_at``)."""
    return {Path(d).name: {"started_at": f.get("started_at"), "completed_at": f.get("completed_at")}
            for d, f in sorted(labels.found.items()) if isinstance(f, Mapping)}  # fmt: skip


def analysis(args: argparse.Namespace) -> dict[str, Any]:
    cli = _cli()
    code = own_code()
    text = registration_text()
    if HEADING not in text:
        refuse("the registration at this commit holds no extension 1")
    registered, registered_file = registered_record(args.registered, text)
    opening = opening_record(args.open_e2) if args.open_e2 is not None else None
    runs = Path(args.pod3_runs)
    archive = pod3_archive(runs)
    reader = cli.Reader({"pod3": runs})
    pod2 = Pod2(Path(args.pod2_runs), registered)
    prompts_of = cli.prompt_sets(reader)
    labels3 = cli.Labels(reader, prompts_of)
    ready = (registered.get("ready_to_label") or {}).get("written_utc")
    labels2 = cli.Labels(pod2.reader, prompts_of, ready=ready)
    e1 = e1_part(reader, runs, archive, pod2, labels3, labels2, registered)
    e2 = ({"sealed": True} if opening is None
          else e2_part(reader, runs, archive, pod2, labels3, registered, opening))  # fmt: skip
    pod2.check()
    sets = registered.get("prompt_sets") or {}
    other = sorted(n for n, d in reader.prompt_sets.items() if sets.get(n) != d)
    if other:
        refuse(f"the prompt set {other[0]} is not the one the registered analysis read")
    inputs = {**reader.hashes, **pod2.reader.hashes}  # pod3:<path> and pod2:<path>
    return sa.finite_tree({
        "kind": KIND, "version": VERSION,
        "registration": {"path": REGISTRATION, "sha256": sha256(text.encode("utf-8")),
                         "section": HEADING.lstrip("# ")},
        "code": code, "registered_analysis": registered_file, "k_star": sa.K_STAR,
        "level": LEVEL, "pod3": {"archive_id": archive},
        "e1": e1, "e2": e2, "labeling": labeling(labels3),
        "inputs": dict(sorted(inputs.items())),
        "prompt_sets": dict(sorted(reader.prompt_sets.items())),
    })  # fmt: skip


def _number(value: Any) -> str:
    return _cli()._number(value) if not isinstance(value, Mapping) else "none"


def report(result: Mapping[str, Any], digest: str) -> None:
    """Numbers, ids and the record's sha256 only."""
    e1 = result["e1"]
    evaluated = sum(r["status"] == "evaluated" for r in e1["rows"])
    print(f"e1 rows {len(e1['rows'])} evaluated {evaluated} not_evaluated "
          f"{len(e1['not_evaluated'])} missing {len(e1['missing'])}")  # fmt: skip
    trigger = e1["shuffled_d_trigger"]
    print(f"e1 shuffled_d_trigger {trigger['decision']} mean {_number(trigger['mean']['mean'])} "
          f"seeds {trigger['seeds_in_mean']}")  # fmt: skip
    for rung in E1_RUNGS:
        found = e1["reproducibility"][rung]
        largest = (found["largest"] or {}).get("relative_difference")
        print(f"e1 reproducibility {rung} points {found['points']} largest {_number(largest)}")
    e2 = result["e2"]
    if e2["sealed"]:
        print("e2 sealed")
    else:
        d, interval = e2["decomposition"], e2["decomposition"]["interval"] or {}
        print(f"e2 entries {len(e2['entries'])} unsupported {len(e2['unsupported'])} seeds "
              f"{d['n']} sigma2_w {_number(d['sigma2_w'])} sigma2_b {_number(d['sigma2_b'])} "
              f"share {_number(d['order_share'])} lower {_number(interval.get('lower'))} upper "
              f"{_number(interval.get('upper'))} rho {_number(e2['spearman']['rho'])}")  # fmt: skip
    print(f"record sha256 {digest}")


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m zero_field_refusal.analysis.{PROG}",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    parser.add_argument("--pod3-runs", type=Path, required=True)
    parser.add_argument("--pod2-runs", type=Path, required=True)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--open-e2", type=Path, metavar="OPENING_RECORD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    cli = _cli()
    try:
        if args.out.exists() or args.out.is_symlink():
            refuse(f"{args.out.name} exists: never overwritten")
        result = analysis(args)
        data = cli.encoded(result)
        cli.write_once(args.out, data)
    except cli.Refused as exc:
        sys.exit(f"{PROG}: {exc}")
    report(result, sha256(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
