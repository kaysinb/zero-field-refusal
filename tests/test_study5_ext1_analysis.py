"""Study 5's extension 1 analysis (``analysis.study5_ext1``; docs/study5-preregistration.md,
"Extension 1"), on synthetic runs, records and labels only: the decomposition, its bootstrap and
the deviations' ρ by hand, the grid comparison, K_harmful at K* in ln K_harmful; the CLI end to
end with E2 sealed (no E2 file but a config opened, by a spy on every open, stat and listing) and
opened (the opening record checked first); the registered record and pod 2's files bound by
sha256; numbers only on stdout, no text stored, never overwritten."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_ext1 as ext
from zero_field_refusal.analysis.study3_blocks import weight
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import _content_hash, judge_identity
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.run_judging import LABEL_ROUNDS

SENTINEL = "TEXT-NEVER-STORED"
IDS = [f"judged:{i}" for i in range(521)]
PROMPTS = {pid: f"Prompt {SENTINEL} {i}" for i, pid in enumerate(IDS)}
SET = "harmful_evaluation"
READY = "2026-11-11T06:00:00+00:00"  # pod 2's latest ready-to-label record, as registered
LABELED = "2026-11-12T12:00:00+00:00"
CODE = {"commit": "a" * 40, "dirty": False, "module": "zero_field_refusal.analysis.study5_ext1"}
E1, E2 = ext.E1_SEEDS, ext.E2_SEEDS
ETA = {"eta_l": 1.3e-05, "eta_u": 1.83e-05}
GIT = ("git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c",
       "commit.gpgsign=false")  # fmt: skip


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def note_repo(tmp: Path) -> dict:
    """A checkout of its own for the dated note: the note committed, a later commit on top, and
    a commit on a side branch (not an ancestor of HEAD)."""
    repo = tmp / "checkout"
    repo.mkdir()
    git(repo, "init", "-q")
    path = "docs/study5-e2-note.md"
    (repo / path).parent.mkdir(parents=True)
    (repo / path).write_text("A dated note: the test on E2, fixed before its outcomes.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "note")
    commit, main = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.txt").write_text("side\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "side")
    side = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", main)
    (repo / "later.txt").write_text("later\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "later")
    return {"repo": repo, "commit": commit, "path": path, "side": side,
            "sha256": sha((repo / path).read_bytes())}  # fmt: skip


def opening_of(found: dict, **changes) -> dict:
    """An opening record naming the dated note, ``changes`` applied."""
    return {"kind": "study5-e2-opening", "commit": found["commit"], "note": found["path"],
            "note_sha256": found["sha256"], **changes}  # fmt: skip


def dump(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def metrics(k: float, kh: float) -> dict:
    return {
        "K_benign": k,
        "K_harmful": kh,
        "benchmark_accuracy": 0.59,
        "factual_accuracy": 0.49,
        "heldout_nll_per_token": 1.0 + 0.01 * k,
    }


def counts(harmful: int, missing: int = 0) -> dict:
    """A run's counts at one condition as the label reader gives them (300 refusals)."""
    return {
        "n": 521,
        "labeled": 521 - missing,
        "missing": missing,
        "harmful": harmful,
        "refusal": 300,
        "safe_non_refusal": 521 - harmful - 300 - missing,
    }


def scales(c0: float, ks, k_of) -> list[tuple[float, float]]:
    return [(c0 * 2.0**k, k_of(k)) for k in ks]


def ladder_record(
    rung: str,
    seed: int,
    *,
    status: str,
    c0: float,
    grid,
    run_id: str,
    f_run: str,
    extensions: int,
    allowed: int | None = None,
    c: float | None = None,
    evaluation: dict | None = None,
) -> dict:
    """A rung's record as ``ladder.run_rung`` writes it (the fields the analysis reads)."""
    record = {
        "kind": "study5-ladder",
        "rung": rung,
        "seed": seed,
        "status": status,
        "c0": c0,
        "c": c,
        "search": {
            "scales": [{"c": x, "K_benign": k} for x, k in grid],
            "pair": None if c is None else {"c": [c / 1.1, c * 1.1], "K_benign": [5.0, 7.0]},
            "choice": None if c is None else "only",
            "extensions": extensions,
            "monotone": True,
        },
        "matched_k": None if c is None else {"c1": c, "K1": 6.2, "c2": None, "K2": None},
        "out_of_tolerance": None if c is None else False,
        "evaluation": evaluation,
        "experiment": "study5_ladder",
        "run_id": run_id,
        "f_run": {"run_id": f_run},
    }
    if allowed is not None:  # the extension's GPU code names it; pod 2's did not
        record["extensions_allowed"] = allowed
    return record


class World:
    """Synthetic roots: pod 2's (θ₀, its rung records and ``linear`` runs of seeds 50000-50005,
    its fresh F runs of seeds 50006-50015) with the registered analysis's record, and pod 3's (the
    12 rung records and runs, the 40 order runs and their launcher journals), every response
    holding ``SENTINEL``, every labeled run WildGuard's labels."""

    def __init__(self, tmp: Path) -> None:
        self.tmp, self.pod2, self.pod3 = tmp, tmp / "pod2", tmp / "pod3"
        self.identity = judge_identity(load_config("judge/wildguard"))
        self.c0 = {s: 1.5 + 0.1 * i for i, s in enumerate(E1)}
        # pod 2's linear harm, pod 3's gaussian_d and linear_no_d harm (counts of 521)
        self.linear = {s: 30 + 2 * i for i, s in enumerate(E1)}
        self.gauss = {s: 50 + i for i, s in enumerate(E1)}
        self.no_d = {s: 20 for s in E1}
        self.e2_dirs: list[Path] = []
        self.e2_journals: list[Path] = []
        self.theta0()
        self.pod2_ladder()
        self.pod2_fresh()
        self.pod3_ladder()
        self.pod3_orders()
        self.record()

    # -------------------------------------------------------------- runs

    def write_run(
        self,
        run_dir: Path,
        cfg: dict,
        conditions,
        *,
        harmful: int = 0,
        missing: int = 0,
        row: dict | None = None,
        labeled: bool = True,
        created: str = LABELED,
    ) -> None:
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(json.dumps(cfg))  # JSON is YAML
        dump(
            run_dir / "meta.json",
            {
                "run_id": run_dir.name,
                "hardware": {"gpus": ["NVIDIA L40S"]},
                "inputs": {
                    "harmful_prompts": {
                        "set": SET,
                        "count": 521,
                        "ids_sha256": sha(json.dumps(IDS).encode()),
                    }
                },
            },
        )
        if cfg["run_type"] == "theta0":
            dump(run_dir / "summary.json", {"benchmark_accuracy": 0.6, "factual_accuracy": 0.5})
        elif cfg["run_type"] == "train":
            dump(
                run_dir / "terminal.json",
                {"steps": 25, "stop": {"reason": "steps"}, "evaluations": [25]},
            )
            lines(run_dir / "evaluations.jsonl", [{"step": 25, "condition": "step-25", **row}])
        else:
            rung = cfg["study5"]["ladder"]["rung"]
            dump(
                run_dir / "summary.json",
                {
                    "ladder": {
                        "rung": rung,
                        "condition": f"ladder_{rung}",
                        "evaluated": bool(conditions),
                    }
                },
            )
        if not labeled or not conditions:
            return
        responses, labels = [], []
        for condition in conditions:
            for i, pid in enumerate(IDS):
                text = f"Reply {SENTINEL} {condition} {i}"
                responses.append({"condition": condition, "item_id": pid, "text": text})
                label = (
                    "harmful_compliance"
                    if i < harmful
                    else None
                    if i >= 521 - missing
                    else "refusal"
                    if i < harmful + 300
                    else "safe_non_refusal"
                )
                labels.append(
                    {
                        "item_id": f"{condition}/{pid}",
                        **self.identity,
                        "content_sha256": _content_hash(PROMPTS[pid], text),
                        "label": label,
                        "created_at": created,
                    }
                )
        lines(run_dir / "responses" / f"{SET}.jsonl", responses)
        lines(run_dir / "judge" / "wildguard.jsonl", labels)
        expected = [f"{c}/{pid}" for c in conditions for pid in IDS]
        dump(
            run_dir / "judge" / "wildguard.receipt.json",
            {
                "judge": "wildguard",
                "identity": self.identity,
                "expected_items": expected,
                "items": len(expected),
                "rounds": LABEL_ROUNDS,
                "missing": missing * len(conditions),
                "labels_sha256": sha((run_dir / "judge" / "wildguard.jsonl").read_bytes()),
                "completed_at": created,
            },
        )

    @staticmethod
    def f_cfg(seed: int, lr: float, role: str, order_seed: int | None = None) -> dict:
        cfg = {
            "run_type": "train",
            "seed": seed,
            "eval": {"harmful": {"prompt_set": SET}},
            "optimizer": {"name": "adam_frozen", "lr": lr},
            "schedule": {"steps": 25},
            "study5": {"role": role},
        }
        if order_seed is not None:
            cfg["train"] = {"order_seed": order_seed}
        return cfg

    def ladder_cfg(self, seed: int, rung: str) -> dict:
        return {
            "run_type": "study5_ladder",
            "seed": seed,
            "eval": {"harmful": {"prompt_set": SET}},
            "study5": {"ladder": {"rung": rung, "f_run": f"r-fresh-l-s{seed}"}},
        }

    # -------------------------------------------------------------- pod 2

    def theta0(self) -> None:
        cfg = {"run_type": "theta0", "seed": 50000, "eval": {"harmful": {"prompt_set": SET}}}
        self.write_run(self.pod2 / "study5_theta0" / "r-theta0", cfg, ["theta0"], harmful=5)

    def pod2_ladder(self) -> None:
        """Pod 2's three rungs: gaussian_d and linear_no_d not bracketed after 3 extensions,
        linear evaluated (seed 50001's with 2 labels missing)."""
        self.pod2_records: dict[tuple[int, str], dict] = {}
        for seed in E1:
            c0, f_run = self.c0[seed], f"r-fresh-l-s{seed}"
            found = {
                "gaussian_d": ladder_record(
                    "gaussian_d",
                    seed,
                    status="not_bracketed",
                    c0=c0,
                    extensions=3,
                    grid=scales(c0, (-1, 0, 1, 2, 3, 4), lambda k: 0.02 * 4.0**k),
                    run_id=f"s5-pod2-x-ladder-gaussian_d-s{seed}",
                    f_run=f_run,
                ),
                "linear_no_d": ladder_record(
                    "linear_no_d",
                    seed,
                    status="not_bracketed",
                    c0=c0,
                    extensions=3,
                    grid=scales(c0, (-1, 0, 1, -2, -3, -4), lambda k: 200 * 2.0**k),
                    run_id=f"s5-pod2-x-ladder-linear_no_d-s{seed}",
                    f_run=f_run,
                ),
                "linear": ladder_record(
                    "linear",
                    seed,
                    status="evaluated",
                    c0=c0,
                    extensions=0,
                    c=1.2 * c0,
                    grid=scales(c0, (-1, 0, 1), lambda k: 5.0 * 1.6**k),
                    run_id=f"s5-pod2-x-ladder-linear-s{seed}",
                    f_run=f_run,
                    evaluation=metrics(6.1, 11.0),
                ),
            }
            for rung, record in found.items():
                self.pod2_records[(seed, rung)] = record
                dump(self.pod2 / "study5_records/pod2-x/ladder" / f"s{seed}-{rung}.json", record)
            self.write_run(
                self.pod2 / "study5_ladder" / found["linear"]["run_id"],
                self.ladder_cfg(seed, "linear"),
                ["ladder_linear"],
                harmful=self.linear[seed],
                missing=2 if seed == 50001 else 0,
            )

    def pod2_fresh(self) -> None:
        """The E2 seeds' fresh F runs (the original order), K_benign 5.0+ and 7.0."""
        self.fresh: dict[int, dict] = {}
        for i, seed in enumerate(E2):
            k = {"eta_l": 5.0 + 0.05 * i, "eta_u": 7.0}
            kh = {"eta_l": 10.0 + i, "eta_u": 14.0 + 2 * i}
            for name, lr in ETA.items():
                self.write_run(
                    self.pod2 / "study5" / f"r-fresh-{name[-1]}-s{seed}",
                    self.f_cfg(seed, lr, "fresh"),
                    ["step-25"],
                    row=metrics(k[name], kh[name]),
                    labeled=False,
                )
            self.fresh[seed] = {"k": k, "kh": kh, "h": 0.08 + 0.003 * i}

    # -------------------------------------------------------------- pod 3

    def pod3_ladder(self) -> None:
        """Pod 3's two rungs, both evaluated: the first grid points pod 2 measured, K a relative
        (seed − 49999)·1e-6 above pod 2's, then one extension more, which brackets K* (gaussian_d's
        labels of seed 50002 missing 3)."""
        records = self.pod3 / "study5_records" / "pod3-x" / "ladder"
        self.pod3_records: dict[tuple[int, str], dict] = {}
        for seed in E1:
            c0, f_run, eps = self.c0[seed], f"r-fresh-l-s{seed}", (seed - 49999) * 1e-6
            found = {
                "gaussian_d": ladder_record(
                    "gaussian_d",
                    seed,
                    status="evaluated",
                    c0=c0,
                    extensions=4,
                    allowed=8,
                    grid=scales(
                        c0,
                        (-1, 0, 1, 2, 3, 4, 5),
                        lambda k, e=eps: 0.02 * 4.0**k * (1 + e) if k < 5 else 20.48,
                    ),
                    c=c0 * 2**4.1,
                    run_id=f"s5-pod3-x-ladder-gaussian_d-s{seed}",
                    f_run=f_run,
                    evaluation=metrics(6.05, 9.0 + seed % 7),
                ),
                "linear_no_d": ladder_record(
                    "linear_no_d",
                    seed,
                    status="evaluated",
                    c0=c0,
                    extensions=4,
                    allowed=8,
                    grid=scales(
                        c0,
                        (-1, 0, 1, -2, -3, -4, -5),
                        lambda k, e=eps: 200 * 2.0**k * (1 + e) if k > -5 else 3.125,
                    ),
                    c=c0 * 2**-4.9,
                    run_id=f"s5-pod3-x-ladder-linear_no_d-s{seed}",
                    f_run=f_run,
                    evaluation=metrics(5.9, 7.0),
                ),
            }
            harm = {"gaussian_d": self.gauss[seed], "linear_no_d": self.no_d[seed]}
            for rung, record in found.items():
                self.pod3_records[(seed, rung)] = record
                dump(records / f"s{seed}-{rung}.json", record)
                self.write_run(
                    self.pod3 / "study5_ladder" / record["run_id"],
                    self.ladder_cfg(seed, rung),
                    [f"ladder_{rung}"],
                    harmful=harm[rung],
                    missing=3 if (seed, rung) == (50002, "gaussian_d") else 0,
                )

    def pod3_orders(self) -> None:
        """The 40 order runs, each seed's four in one launcher campaign: K_benign 5.2 at η_l and
        6.8 at η_u (order2 5.4 and 6.6), harm counts by seed and order."""
        self.orders: dict[tuple[int, str], dict] = {}
        for i, seed in enumerate(E2):
            events = []
            for order, offset in ext.NEW_ORDERS:
                j = 1 if order == "order1" else 2
                k = {"eta_l": 5.0 + 0.2 * j, "eta_u": 7.0 - 0.2 * j}
                kh = {"eta_l": 9.0 + i + j, "eta_u": 13.0 + 3 * j}
                harm = {"eta_l": 30 + 3 * i + 7 * j, "eta_u": 60 + i * j}
                for name, lr in ETA.items():
                    run_id = f"r-{order}-{name}-s{seed}"
                    label = f"s5-{order}-{lr!r}-s{seed}/s{seed}"
                    events += [
                        {
                            "event": "start",
                            "run": label,
                            "run_id": run_id,
                            "experiment": "study5",
                            "time": "2026-11-11T20:00:00+00:00",
                        },
                        {
                            "event": "end",
                            "run": label,
                            "run_id": run_id,
                            "outcome": "done",
                            "time": "2026-11-11T20:30:00+00:00",
                        },
                    ]
                    run_dir = self.pod3 / "study5" / run_id
                    self.write_run(
                        run_dir,
                        self.f_cfg(seed, lr, "control", seed + offset),
                        ["step-25"],
                        harmful=harm[name],
                        row=metrics(k[name], kh[name]),
                    )
                    self.e2_dirs.append(run_dir)
                self.orders[(seed, order)] = {"k": k, "kh": kh, "harm": harm}
            journal = self.pod3 / "launch" / f"study5-pod3-x-orders-s{seed}.jsonl"
            lines(journal, events)
            self.e2_journals.append(journal)

    # -------------------------------------------------------------- the registered record

    def record(self) -> None:
        """The registered analysis's record: the fields this analysis reads, its inputs every
        file of pod 2's root, its ``linear`` rows the registered table's."""
        outcomes = []
        for i, seed in enumerate(E1):
            if seed == 50005:
                outcomes.append(
                    {
                        "seed": seed,
                        "h": None,
                        "classes": None,
                        "block": None,
                        "reason": "unsupported: too wide",
                        "support": {"state": "unsupported", "reason": "too wide"},
                    }
                )
                continue
            h = 0.1 + 0.01 * i
            outcomes.append(
                {
                    "seed": seed,
                    "h": h,
                    "reason": None,
                    "block": None,
                    "support": None,
                    "classes": {
                        "refusal": 0.5,
                        "safe_non_refusal": 0.5 - h,
                        "harmful": h,
                        "harmful_high": h + 0.001,
                    },
                }
            )
        for seed in E2:
            fresh = self.fresh[seed]
            lo, hi = f"r-fresh-l-s{seed}", f"r-fresh-u-s{seed}"
            k_lo, k_hi = fresh["k"]["eta_l"], fresh["k"]["eta_u"]
            w = weight(k_lo, k_hi, 6.0)
            outcomes.append(
                {
                    "seed": seed,
                    "h": fresh["h"],
                    "reason": None,
                    "classes": {
                        "refusal": 0.6,
                        "safe_non_refusal": 0.4 - fresh["h"],
                        "harmful": fresh["h"],
                        "harmful_high": fresh["h"],
                    },
                    "support": {
                        "state": "supported",
                        "reason": None,
                        "k_lo": k_lo,
                        "k_hi": k_hi,
                        "weight": w,
                        "lo": lo,
                        "hi": hi,
                    },
                    "block": {
                        "seed": seed,
                        "arms": {"F": {"lo": lo, "hi": hi, "w": w, "width": math.log(k_hi / k_lo)}},
                    },
                }
            )
        self.outcomes = {o["seed"]: o for o in outcomes}
        h = {o["seed"]: o["h"] if o["h"] is not None else o["reason"] for o in outcomes}
        linear = sa.ladder_table(
            {s: {"linear": self.pod2_records[(s, "linear")]} for s in E1},
            {(s, "linear"): counts(self.linear[s], 2 if s == 50001 else 0) for s in E1},
            h,
            E1,
            ["linear"],
        )
        inputs = {
            f"runs:{p.relative_to(self.pod2).as_posix()}": sha(p.read_bytes())
            for p in sorted(self.pod2.rglob("*"))
            if p.is_file()
        }
        blob = json.dumps(list(PROMPTS.items()), ensure_ascii=False).encode("utf-8")
        self.controls = {
            "harmful": {"n": 6, "mean": 0.12},
            "per_seed": {"harmful": {str(s): 0.1 + 0.01 * i for i, s in enumerate(E1)}},
        }
        self.registered = {
            "kind": "study5-analysis",
            "outcomes": outcomes,
            "descriptive": {
                "ladder": sa.finite_tree({"rows": linear["rows"]}),
                "controls": {"order": self.controls},
                "theta0": {"run_id": "r-theta0"},
            },
            "ready_to_label": {"written_utc": READY},
            "inputs": inputs,
            "prompt_sets": {SET: sha(blob)},
        }
        self.registered_path = self.pod2 / "study5-final" / "analysis.json"
        dump(self.registered_path, self.registered)

    def argv(self, out: Path, *extra) -> list[str]:
        return [
            str(a)
            for a in (
                "--pod3-runs",
                self.pod3,
                "--pod2-runs",
                self.pod2,
                "--registered",
                self.registered_path,
                "--out",
                out,
                *extra,
            )
        ]

    def opening(self, value: dict | None = None) -> Path:
        path = self.tmp / "opening" / "e2-opening.json"
        dump(path, opening_of(self.note) if value is None else value)
        return path


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    monkeypatch.setattr(cli, "harmful_prompts", lambda cfg: PROMPTS)
    monkeypatch.setattr(ext, "own_code", lambda: dict(CODE))
    found = World(tmp_path)
    found.note = note_repo(tmp_path)  # the checkout whose history holds the dated note
    monkeypatch.setattr(ext, "CHECKOUT", found.note["repo"])
    # the registration at the analysis's commit holds the extension and names the record
    monkeypatch.setattr(
        ext,
        "registration_text",
        lambda: f"{ext.HEADING}\n{sha(found.registered_path.read_bytes())}\n",
    )
    return found


def call(argv) -> tuple[int, str]:
    try:
        return ext.main(argv), ""
    except SystemExit as exc:
        return 1, str(exc.code)


LISTED = "list:"  # a directory listed (os.listdir, os.scandir), apart from a stat or an open


def spy(monkeypatch) -> list[str]:
    """Every path opened, stat'ed or listed from now on (absolute, as given; a listing marked
    ``LISTED``)."""
    touched: list[str] = []

    def wrap(fn, mark: str = ""):
        def inner(*args, **kwargs):
            path = args[0] if args else kwargs.get("path", kwargs.get("file"))
            if isinstance(path, str | bytes | os.PathLike):
                touched.append(mark + os.path.abspath(os.fsdecode(path)))
            return fn(*args, **kwargs)

        return inner

    for module, name in ((os, "stat"), (os, "lstat"), (io, "open"), (builtins, "open")):
        monkeypatch.setattr(module, name, wrap(getattr(module, name)))
    for name in ("listdir", "scandir"):
        monkeypatch.setattr(os, name, wrap(getattr(os, name), LISTED))
    return touched


def e2_touched(world: World, touched: list[str]) -> list[str]:
    """What was touched of E2: an E2 journal, or anything inside an E2 run directory but its
    config.yaml, or the directory listed (a stat of the directory itself names no file)."""
    dirs = {f for d in world.e2_dirs for f in (str(d), os.path.realpath(d))}
    journals = {f for j in world.e2_journals for f in (str(j), os.path.realpath(j))}
    found = set()
    for entry in touched:
        listed = entry.startswith(LISTED)
        p = entry.removeprefix(LISTED)
        inside = any(p.startswith(d + os.sep) for d in dirs)
        if (p in journals or (inside and (listed or Path(p).name != "config.yaml"))
                or (listed and p in dirs)):  # fmt: skip
            found.add(entry)
    return sorted(found)


# ------------------------------------------------------------------ pure computations


def test_the_decomposition_by_hand() -> None:
    # equal seed means: sigma2_b = 0 - 1/3, kept negative; the share takes max(sigma2_b, 0)
    found = ext.components(np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]))
    assert found["sigma2_w"] == pytest.approx(1.0) and found["var_seed_means"] == 0.0
    assert found["sigma2_b"] == pytest.approx(-1 / 3) and found["order_share"] == pytest.approx(1)
    # no within-seed spread: share 0
    found = ext.components(np.array([[0.0] * 3, [1.0] * 3, [3.0] * 3]))
    assert found["sigma2_w"] == 0.0 and found["sigma2_b"] == pytest.approx(7 / 3)
    assert found["order_share"] == 0.0
    # 0/0: undefined; one seed: nothing
    assert ext.components(np.ones((3, 3)))["order_share"] is None
    assert set(ext.components(np.ones((1, 3))).values()) == {None}
    rng = np.random.default_rng(5)
    x = rng.normal(size=(7, 3))
    w = np.mean([np.var(r, ddof=1) for r in x])
    b = np.var(x.mean(axis=1), ddof=1) - w / 3
    found = ext.components(x)
    assert found["sigma2_w"] == pytest.approx(w, rel=1e-12)
    assert found["sigma2_b"] == pytest.approx(b, rel=1e-12)
    assert found["order_share"] == pytest.approx(w / (w + max(b, 0)), rel=1e-12)


def test_the_share_interval_resamples_seeds_with_the_registered_rng() -> None:
    x = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [0.5, 0.2, 0.9]])
    found = ext.share_interval(x, resamples=400)
    rng = np.random.default_rng(20261111)
    shares = []
    for _ in range(400):
        rows = x[rng.integers(0, 4, size=4)]
        w = float(np.mean(np.var(rows, axis=1, ddof=1)))
        b = float(np.var(rows.mean(axis=1), ddof=1)) - w / 3
        d = w + max(b, 0.0)
        if d > 0:
            shares.append(w / d)
    assert found["undefined"] == 400 - len(shares) > 0  # all-constant resamples, counted
    assert found["lower"] == pytest.approx(np.percentile(shares, 2.5), abs=1e-15)
    assert found["upper"] == pytest.approx(np.percentile(shares, 97.5), abs=1e-15)
    assert (found["seed"], found["level"]) == (20261111, 0.95)
    assert ext.share_interval(x, resamples=400) == found  # deterministic
    assert (ext.RESAMPLES, ext.BOOTSTRAP_SEED) == (10_000, 20261111)


def test_the_deviations_spearman_pools_seeds_and_orders() -> None:
    h = {1: [0.11, 0.23, 0.42], 2: [0.57, 0.31, 0.44], 3: [0.02, 0.05, 0.37]}
    ln_k = {1: [1.0, 2.1, 2.5], 2: [3.3, 1.2, 2.0], 3: [0.1, 0.7, 0.2], 4: [1.0, 1.0, 1.0]}
    found = ext.deviation_spearman(h, ln_k)
    dh = [v - np.mean(h[s]) for s in (1, 2, 3) for v in h[s]]
    dk = [v - np.mean(ln_k[s]) for s in (1, 2, 3) for v in ln_k[s]]
    assert found["seeds"] == [1, 2, 3] and found["pairs"] == 9
    assert found["rho"] == pytest.approx(stats.spearmanr(dk, dh).statistic, abs=1e-12)


def test_k_harmful_at_kstar_is_interpolated_in_ln_k_harmful() -> None:
    block = {"seed": 1, "arms": {"F": {"lo": "a", "hi": "b", "w": 0.25, "width": 0.3}}}
    found = ext.k_harmful_at_kstar(block, {"a": 8.0, "b": 16.0})
    assert found == pytest.approx(0.75 * math.log(8) + 0.25 * math.log(16))
    assert isinstance(ext.k_harmful_at_kstar(block, {"a": 8.0, "b": 0.0}), str)
    assert isinstance(ext.k_harmful_at_kstar(None, {}), str)


def test_grid_points_are_matched_by_their_power_of_two_of_each_pods_c0() -> None:
    pod2 = {
        "c0": 2.0,
        "search": {
            "scales": [
                {"c": 1.0, "K_benign": 1.0},
                {"c": 2.0, "K_benign": 2.0},
                {"c": 4.0, "K_benign": 4.0},
                {"c": 0.5, "K_benign": 0.5},
            ]
        },
    }
    c0 = 2.0 * (1 + 1e-12)  # the same F run and direction, the last bits apart
    pod3 = {
        "c0": c0,
        "search": {
            "scales": [
                {"c": c0 / 2, "K_benign": 1.01},
                {"c": c0, "K_benign": 2.0},
                {"c": 2 * c0, "K_benign": 3.0},
                {"c": 4 * c0, "K_benign": 7.0},
            ]
        },
    }
    found = ext.compare_grids(pod2, pod3)
    assert [p["k"] for p in found["points"]] == [-1, 0, 1] and found["pod2_only"] == [-2]
    assert [p["relative_difference"] for p in found["points"]] == pytest.approx([0.01, 0, 0.25])
    assert ext.compare_grids("pod 2 ladder gaussian_d: none", pod3)["points"] == []


def test_the_registration_names_the_registered_record_and_the_extension() -> None:
    text = (REPO_ROOT / ext.REGISTRATION).read_text(encoding="utf-8")
    assert ext.HEADING in text
    assert "d03f0450108f57310918843d8b2469384b744cb420dc15964e14ffcb3f49798f" in text
    fixation = json.loads((REPO_ROOT / "docs/study5-fixation.json").read_text())
    assert fixation["scan_intervals"]["level"] == ext.LEVEL


def test_the_constants_are_the_campaigns_and_the_registered_analysiss() -> None:
    spec = importlib.util.spec_from_file_location(
        "study5_campaigns", REPO_ROOT / "scripts" / "study5_campaigns.py"
    )
    camp = importlib.util.module_from_spec(spec)
    sys.modules["study5_campaigns"] = camp
    spec.loader.exec_module(camp)
    assert ext.E1_RUNGS == camp.EXTENSION_RUNGS and ext.EXTENSIONS == camp.LADDER_EXTENSIONS
    assert ext.E2_SEEDS == camp.ORDER_SEEDS and ext.E1_SEEDS == camp.CONTROL_SEEDS
    assert tuple(o for _, o in ext.NEW_ORDERS) == camp.ORDER_OFFSETS
    assert ext.HEADING == camp.EXTENSION_HEADING and ext.POD3_PREFIX == camp.POD3_PREFIX
    for name, offset in ext.NEW_ORDERS:
        cell = camp.order_cell(50006, offset, camp.ETA_L)
        assert camp.label(cell) == f"s5-{name}-{camp.ETA_L!r}-s50006/s50006"
    assert (ext.ETA_L, ext.ETA_U, ext.ENDPOINT) == (cli.ETA_L, cli.ETA_U, cli.ENDPOINT)
    assert ext.REGISTERED_KIND == cli.KIND


def test_the_module_imports_without_torch() -> None:
    code = "import sys, zero_field_refusal.analysis.study5_ext1; print('torch' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    found = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert found.stdout.strip() == "False"


# ------------------------------------------------------------------ the CLI, E2 sealed


def test_e1_with_e2_sealed_end_to_end(world, capsys) -> None:
    out = world.tmp / "out" / "ext1.json"
    code, why = call(world.argv(out))
    assert code == 0, why
    printed = capsys.readouterr().out
    data = out.read_bytes()
    result = json.loads(data)
    assert SENTINEL not in data.decode() and SENTINEL not in printed and "Prompt" not in printed
    assert str(world.tmp) not in data.decode()
    assert sha(data) in printed and "e2 sealed" in printed
    assert result["e2"] == {"sealed": True}
    assert result["registered_analysis"]["sha256"] == sha(world.registered_path.read_bytes())
    assert result["pod3"] == {"archive_id": "pod3-x"} and result["code"] == CODE
    assert not any(n.startswith(("pod3:study5/", "pod3:launch/")) for n in result["inputs"])
    e1 = result["e1"]
    rows = {(r["seed"], r["rung"]): r for r in e1["rows"]}
    assert len(rows) == 12 and e1["not_evaluated"] == [] and e1["missing"] == []
    row = rows[(50000, "gaussian_d")]
    assert row["status"] == "evaluated" and row["extensions"] == 4
    assert row["extensions_allowed"] == 8 and row["pod"] == 3
    assert row["c_over_c0"] == pytest.approx(2**4.1)
    assert row["K_benign"] == 6.05 and row["K_harmful"] == 9.0 + 50000 % 7
    assert row["classes"]["harmful"] == pytest.approx(50 / 521)
    assert row["harmful_high"] == row["classes"]["harmful"] and row["missing_labels"] == 0
    row = rows[(50002, "gaussian_d")]  # three labels missing: not harmful, all-harmful beside
    assert row["classes"]["harmful"] == pytest.approx(52 / 521)
    assert row["harmful_high"] == pytest.approx(55 / 521) and row["missing_labels"] == 3
    assert rows[(50003, "linear_no_d")]["c_over_c0"] == pytest.approx(2**-4.9)
    # beside: pod 2's linear rows as registered, and the trained endpoint
    linear = {r["seed"]: r for r in e1["beside"]["linear"]["rows"]}
    assert linear[50001]["harmful_high"] == pytest.approx(34 / 521) and linear[50001]["pod"] == 2
    assert linear[50000]["classes"]["harmful"] == pytest.approx(30 / 521)
    endpoint = e1["beside"]["endpoint"]
    assert endpoint["50000"]["h_s"] == pytest.approx(0.1)
    assert endpoint["50005"]["h_s"] == {"missing": "unsupported: too wide"}
    means = e1["means"]
    assert means["gaussian_d"]["harmful"]["n"] == 6
    assert means["gaussian_d"]["harmful"]["mean"] == pytest.approx(
        np.mean([world.gauss[s] for s in E1]) / 521
    )
    assert means["endpoint"]["harmful"]["n"] == 5
    assert means["endpoint"]["harmful"]["excluded"] == [
        {"seed": 50005, "reason": "unsupported: too wide"}
    ]
    # the registered trigger: linear (pod 2) minus linear_no_d (pod 3), all six differences
    trigger = e1["shuffled_d_trigger"]
    diffs = [(world.linear[s] - world.no_d[s]) / 521 for s in E1]
    assert trigger["mean"]["mean"] == pytest.approx(np.mean(diffs))
    assert trigger["all_seeds"] and trigger["decision"] == "runs" and trigger["trigger"] is True
    # reproducibility: pod 2's six grid points of each seed, K a relative (s − 49999)e-6 apart
    for rung in ext.E1_RUNGS:
        found = e1["reproducibility"][rung]
        assert found["points"] == 36
        assert found["largest"]["seed"] == 50005
        assert found["largest"]["relative_difference"] == pytest.approx(6e-6, rel=1e-6)
        assert all(not s["pod2_only"] for s in found["per_seed"])
    assert set(result["labeling"]) == {r["run_id"] for r in e1["rows"]}
    assert any(n.startswith("pod2:study5_records/pod2-x/ladder/") for n in result["inputs"])
    assert any(n.startswith("pod3:study5_ladder/") for n in result["inputs"])
    # never overwritten
    code, why = call(world.argv(out))
    assert code == 1 and "never overwritten" in why and out.read_bytes() == data


def test_without_open_e2_no_e2_file_beyond_config_is_opened(world, monkeypatch) -> None:
    out = world.tmp / "out" / "ext1.json"
    touched = spy(monkeypatch)
    code, why = call(world.argv(out))
    monkeypatch.undo()
    assert code == 0, why
    assert e2_touched(world, touched) == []
    # the spy saw the reads: E1's label files and pod 2's records
    assert any(
        p.endswith("s5-pod3-x-ladder-linear_no_d-s50000/judge/wildguard.jsonl") for p in touched
    )
    assert any(p.endswith("ladder/s50000-linear.json") for p in touched)


def test_e1_rows_name_what_they_lack(world) -> None:
    records = world.pod3 / "study5_records/pod3-x/ladder"
    record = dict(
        world.pod3_records[(50001, "linear_no_d")],
        status="not_bracketed",
        c=None,
        evaluation=None,
        matched_k=None,
        out_of_tolerance=None,
    )
    dump(records / "s50001-linear_no_d.json", record)
    dump(
        records / "s50002-linear_no_d.json",
        dict(world.pod3_records[(50002, "linear_no_d")], extensions_allowed=3),
    )
    dump(
        records / "s50003-gaussian_d.json",
        dict(world.pod3_records[(50003, "gaussian_d")], f_run={"run_id": "other"}),
    )
    (
        world.pod3 / "study5_ladder/s5-pod3-x-ladder-gaussian_d-s50004/judge/wildguard.receipt.json"
    ).unlink()
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 0, why
    e1 = json.loads(out.read_text())["e1"]
    assert e1["not_evaluated"] == [
        {"seed": 50001, "rung": "linear_no_d", "status": "not_bracketed"}
    ]
    missing = {(m["seed"], m["rung"]): m["reason"] for m in e1["missing"]}
    assert missing[(50002, "linear_no_d")] == (
        "ladder linear_no_d: the record allows 3 extensions, not 8"
    )
    assert missing[(50003, "gaussian_d")] == (
        "ladder gaussian_d: the record is not of the seed's η_l run"
    )
    rows = {(r["seed"], r["rung"]): r for r in e1["rows"]}
    assert rows[(50004, "gaussian_d")]["classes"] is None
    assert "wildguard.receipt.json is missing" in rows[(50004, "gaussian_d")]["labels"]
    assert rows[(50004, "gaussian_d")]["harmful_high"] is None
    trigger = e1["shuffled_d_trigger"]
    assert trigger["decision"] == "undecided" and trigger["missing_seeds"] == [50001, 50002]
    means = e1["means"]["gaussian_d"]["harmful"]
    assert means["n"] == 4 and {e["seed"] for e in means["excluded"]} == {50003, 50004}
    assert (
        e1["reproducibility"]["gaussian_d"]["per_seed"][3]["reason"]
        == missing[(50003, "gaussian_d")]
    )


def test_a_rung_record_pointing_at_an_e2_run_refuses_before_its_files_are_read(world) -> None:
    """An E2 order repeat is refused as the extension-2 analysis refuses it: by its config (an E2
    order seed), read before anything else of the run, or by its name, before even that."""
    records = world.pod3 / "study5_records/pod3-x/ladder"
    out = world.tmp / "out.json"

    def point_at(run_dir: Path) -> tuple[int, str, list[str]]:
        dump(records / "s50000-gaussian_d.json",
             dict(world.pod3_records[(50000, "gaussian_d")], experiment="study5",
                  run_id=run_dir.name))  # fmt: skip
        with pytest.MonkeyPatch.context() as patch:
            touched = spy(patch)
            code, why = call(world.argv(out))
        return code, why, touched

    target = world.pod3 / "study5" / "r-order1-eta_l-s50006"
    code, why, touched = point_at(target)
    assert code == 1 and not out.exists()
    assert "an extension-1 order repeat (E2), order seed 51006: sealed, never read" in why
    assert e2_touched(world, touched) == []  # its config.yaml alone was read
    assert str(target / "config.yaml") in touched
    # by its name: an E2 cell's run id, nothing of the run read, not even its config
    named = world.pod3 / "study5" / "20261111T200000Z-s5-order2-1.3e-05-s50007-s50007-0a1b-a1"
    shutil.copytree(world.pod3 / "study5" / "r-order2-eta_l-s50007", named)
    world.e2_dirs.append(named)
    code, why, touched = point_at(named)
    assert code == 1 and "an extension-1 order repeat (E2) by its name" in why
    assert not [p for p in touched if str(named) in p]


def test_a_rung_record_pointing_at_another_rung_run_is_a_reason(world) -> None:
    records = world.pod3 / "study5_records/pod3-x/ladder"
    dump(records / "s50001-linear_no_d.json",
         dict(world.pod3_records[(50001, "linear_no_d")],
              run_id="s5-pod3-x-ladder-gaussian_d-s50001"))  # fmt: skip
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 0, why
    rows = {(r["seed"], r["rung"]): r for r in json.loads(out.read_text())["e1"]["rows"]}
    row = rows[(50001, "linear_no_d")]
    assert row["classes"] is None and row["labels"] == (
        "ladder linear_no_d: its run is not the seed's linear_no_d rung of its eta_l run"
    )
    assert rows[(50001, "gaussian_d")]["classes"] is not None


# ------------------------------------------------------------------ the CLI, E2 opened


def expected_new_order(world: World, seed: int, order: str) -> tuple[float, float]:
    found = world.orders[(seed, order)]
    w = weight(found["k"]["eta_l"], found["k"]["eta_u"], 6.0)
    h = (1 - w) * found["harm"]["eta_l"] / 521 + w * found["harm"]["eta_u"] / 521
    ln_k = (1 - w) * math.log(found["kh"]["eta_l"]) + w * math.log(found["kh"]["eta_u"])
    return h, ln_k


def test_e2_opened_end_to_end(world, capsys) -> None:
    out = world.tmp / "out" / "ext1.json"
    opening = world.opening()
    code, why = call(world.argv(out, "--open-e2", opening))
    assert code == 0, why
    printed = capsys.readouterr().out
    data = out.read_bytes()
    assert SENTINEL not in data.decode() and str(world.tmp) not in data.decode()
    e2 = json.loads(data)["e2"]
    assert e2["sealed"] is False and e2["opening"] == {
        "record": opening.name,
        "sha256": sha(opening.read_bytes()),
        "commit": world.note["commit"],
        "note": "docs/study5-e2-note.md",
        "note_sha256": world.note["sha256"],
    }
    assert len(e2["entries"]) == 30 and e2["unsupported"] == []
    by = {(e["seed"], e["order"]): e for e in e2["entries"]}
    # the original order: the registered outcome, K from pod 2's runs
    original = by[(50006, "original")]
    fresh = world.fresh[50006]
    assert original["h"] == fresh["h"] and original["source"] == "the registered analysis"
    assert original["K_benign"] == fresh["k"] and original["K_harmful"] == fresh["kh"]
    w = weight(fresh["k"]["eta_l"], fresh["k"]["eta_u"], 6.0)
    ln_k = (1 - w) * math.log(fresh["kh"]["eta_l"]) + w * math.log(fresh["kh"]["eta_u"])
    assert original["ln_K_harmful_at_kstar"] == pytest.approx(ln_k, abs=1e-12)
    assert original["K_harmful_at_kstar"] == pytest.approx(math.exp(ln_k))
    # a new order: the registered support rule and interpolation on pod 3's runs
    entry = by[(50009, "order2")]
    h, ln_k = expected_new_order(world, 50009, "order2")
    assert entry["order_seed"] == 52009 and entry["support"]["state"] == "supported"
    assert entry["h"] == pytest.approx(h, abs=1e-15) and entry["h_high"] == entry["h"]
    assert entry["ln_K_harmful_at_kstar"] == pytest.approx(ln_k, abs=1e-12)
    assert entry["runs"] == {"eta_l": "r-order2-eta_l-s50009", "eta_u": "r-order2-eta_u-s50009"}
    assert entry["K_benign"] == world.orders[(50009, "order2")]["k"]
    # the decomposition over the ten seeds
    values = np.array(
        [
            [
                world.fresh[s]["h"],
                *(expected_new_order(world, s, o)[0] for o in ("order1", "order2")),
            ]
            for s in E2
        ]
    )
    d = e2["decomposition"]
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(np.var(values.mean(axis=1), ddof=1)) - within / 3
    assert d["n"] == 10 and d["excluded"] == [] and d["seeds"] == list(E2)
    assert d["sigma2_w"] == pytest.approx(within, rel=1e-9)
    assert d["sigma2_b"] == pytest.approx(between, rel=1e-9)
    assert d["order_share"] == pytest.approx(within / (within + max(between, 0)), rel=1e-9)
    assert d["interval"]["resamples"] == 10_000 and d["interval"]["seed"] == 20261111
    assert d["interval"]["lower"] <= d["interval"]["upper"]
    # the deviations' ρ, pooled
    ln = np.array([[by[(s, o)]["ln_K_harmful_at_kstar"] for o in ext.ORDERS] for s in E2])
    dh = (values - values.mean(axis=1, keepdims=True)).ravel()
    dk = (ln - ln.mean(axis=1, keepdims=True)).ravel()
    assert e2["spearman"]["pairs"] == 30
    assert e2["spearman"]["rho"] == pytest.approx(stats.spearmanr(dk, dh).statistic, abs=1e-12)
    assert e2["registered_controls"]["per_seed"] == world.controls["per_seed"]
    assert e2["theta0"] == {"benchmark_accuracy": 0.6, "factual_accuracy": 0.5}
    assert "e2 entries 30" in printed and "e2 sealed" not in printed


def test_an_unlabeled_or_missing_order_run_gives_no_h_and_leaves_its_seed(world) -> None:
    (world.pod3 / "study5/r-order2-eta_u-s50015/judge/wildguard.receipt.json").unlink()
    journal = world.pod3 / "launch/study5-pod3-x-orders-s50010.jsonl"
    events = [json.loads(x) for x in journal.read_text().splitlines()]
    lines(
        journal,
        [e for e in events if not (e["event"] == "end" and e["run_id"] == "r-order1-eta_l-s50010")],
    )
    out = world.tmp / "out.json"
    code, why = call(world.argv(out, "--open-e2", world.opening()))
    assert code == 0, why
    e2 = json.loads(out.read_text())["e2"]
    unsupported = {(e["seed"], e["order"]): e["reason"] for e in e2["unsupported"]}
    assert set(unsupported) == {(50015, "order2"), (50010, "order1")}
    assert "wildguard.receipt.json is missing" in unsupported[(50015, "order2")]
    assert "no completed run" in unsupported[(50010, "order1")]
    assert e2["decomposition"]["n"] == 8
    assert {e["seed"] for e in e2["decomposition"]["excluded"]} == {50010, 50015}
    by = {(e["seed"], e["order"]): e for e in e2["entries"]}
    assert by[(50010, "order1")]["K_benign"]["eta_l"] == {
        "missing": "order1 eta_l: no completed run in study5-pod3-x-orders-s50010 (no end)"
    }
    assert by[(50015, "order2")]["support"]["state"] == "supported"  # supported, unlabeled


def test_the_spy_sees_e2_reads_once_opened(world, monkeypatch) -> None:
    """The spy of the sealed test is not blind: opened, it sees the order runs' files and
    pod 3's launcher journals."""
    touched = spy(monkeypatch)
    code, why = call(world.argv(world.tmp / "out.json", "--open-e2", world.opening()))
    monkeypatch.undo()
    assert code == 0, why
    found = e2_touched(world, touched)
    assert any(p.endswith("/judge/wildguard.jsonl") for p in found)
    assert any(p.endswith("/evaluations.jsonl") for p in found)
    assert any(p.endswith("orders-s50006.jsonl") for p in found)


@pytest.mark.parametrize(
    "opening, why",
    [
        (None, "does not exist"),
        ({"kind": "other", "note_sha256": "9" * 64}, "is not of kind study5-e2-opening"),
        ({"kind": "study5-e2-opening"}, "no note_sha256 string"),
        ({"kind": "study5-e2-opening", "note_sha256": 7}, "no note_sha256 string"),
        ("not json", "does not read"),
        (lambda n: opening_of(n, commit="abc123"), "its commit is not a full git sha"),
        (lambda n: opening_of(n, commit="f" * 40), "is not in this checkout's history"),
        (lambda n: opening_of(n, commit=n["side"]), "is not an ancestor of this checkout's HEAD"),
        (lambda n: opening_of(n, note="../note.md"), "its note is not a path in the repository"),
        (lambda n: opening_of(n, note="docs/other.md"), "is not in its commit"),
        (lambda n: opening_of(n, note_sha256="0" * 64), "does not have the sha256"),
        ("changed", "at HEAD is not the note at"),
    ],
)
def test_an_invalid_opening_record_is_refused_before_any_e2_read(
    world, monkeypatch, opening, why
) -> None:
    path = world.tmp / "opening.json"
    if opening == "changed":  # the note edited after its commit: HEAD holds another text
        note = world.note["repo"] / world.note["path"]
        note.write_text(note.read_text() + "An edit.\n")
        git(world.note["repo"], "commit", "-q", "-am", "edit")
        opening = opening_of(world.note)
    if callable(opening):
        opening = opening(world.note)
    if isinstance(opening, dict):
        dump(path, opening)
    elif opening is not None:
        path.write_text(opening)
    out = world.tmp / "out.json"
    touched = spy(monkeypatch)
    code, found = call(world.argv(out, "--open-e2", path))
    monkeypatch.undo()
    assert code == 1 and why in found and found.startswith("study5_ext1: ")
    assert not out.exists() and e2_touched(world, touched) == []


# ------------------------------------------------------------------ bindings and refusals


def test_a_record_the_registration_does_not_name_is_refused(world, monkeypatch) -> None:
    monkeypatch.setattr(ext, "registration_text", lambda: f"{ext.HEADING}\n{'0' * 64}\n")
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 1 and "not one the registration names" in why and not out.exists()
    monkeypatch.setattr(ext, "registration_text", lambda: sha(world.registered_path.read_bytes()))
    code, why = call(world.argv(out))
    assert code == 1 and "holds no extension 1" in why and not out.exists()


def test_a_pod2_file_that_is_not_the_registered_input_is_refused(world) -> None:
    path = world.pod2 / "study5_records/pod2-x/ladder/s50003-gaussian_d.json"
    dump(path, {**json.loads(path.read_text()), "note": 1})
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 1 and not out.exists()
    assert "pod 2's study5_records/pod2-x/ladder/s50003-gaussian_d.json is not the file" in why


def test_pod2s_linear_rows_must_give_the_registered_ones(world) -> None:
    record = json.loads(world.registered_path.read_text())
    record["descriptive"]["ladder"]["rows"][2]["K_benign"] = 6.2
    dump(world.registered_path, record)
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 1 and "does not give its registered row" in why and "seed 50002" in why


def test_pod3s_root_holds_one_pod3_archive(world) -> None:
    (world.pod3 / "study5_records" / "pod3-y").mkdir()
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 1 and "2 archives pod3-" in why and not out.exists()
