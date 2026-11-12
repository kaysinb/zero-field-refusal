"""Study 5's prospective test on E2's order repeats (``analysis.study5_e2_test``), on synthetic
runs, predictions and records only: the binomial p against the enumerated sign patterns, the
sign-flip p of S exact and sampled, the criterion and prediction records' checks; the CLI end to
end with either predictor, its refusals before any E2 read (a spy on every open, stat and
listing), the binding of a prediction to its runs, top-up seeds; numbers only on stdout, no text
stored, never overwritten."""

from __future__ import annotations

import builtins
import hashlib
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_e2_test as et
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
CODE = {"commit": "a" * 40, "dirty": False, "module": "zero_field_refusal.analysis.study5_e2_test"}
ETA = {"eta_l": 1.3e-05, "eta_u": 1.83e-05}
K = {"order1": (5.2, 6.8), "order2": (5.4, 6.6)}  # K_benign at η_l and η_u
# per seed: the harm counts of order 1 and 2 (both rates alike, so h at K* is count/521), q_lin of
# order 1 and 2 (None: δ_lin diverged) and K_harmful of order 1 and 2 (both rates alike)
PLAN = {
    50006: (60, 40, 0.1, 0.3, 12.0, 10.0),
    50007: (55, 45, 0.2, 0.4, 12.0, 10.0),
    50008: (70, 30, -0.1, 0.2, 12.0, 10.0),
    50009: (30, 50, 0.5, 0.1, 12.0, 10.0),
    50010: (35, 45, 0.3, -0.2, 9.0, 10.0),
    50011: (40, 60, 0.0, 0.1, 12.0, 10.0),
    50012: (50, 30, 0.2, 0.1, 9.0, 10.0),
    50013: (44, 44, 0.1, 0.2, 12.0, 10.0),  # equal h: a tie (order 2 with order 1's K)
    50014: (50, 40, 0.1, None, 12.0, 10.0),  # order 2's δ_lin diverged
    50015: (50, 40, 0.1, 0.2, 12.0, 10.0),  # order 2 unsupported: K 6.5-7.0 brackets no K*
    50000: (20, 25, 0.3, 0.1, 12.0, 10.0),  # a top-up seed
}


GIT = ("git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c",
       "commit.gpgsign=false")  # fmt: skip


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def note_repo(tmp: Path) -> dict:
    """A checkout of its own whose history holds the dated note, a later commit on top."""
    repo = tmp / "checkout"
    repo.mkdir()
    git(repo, "init", "-q")
    path = "docs/study5-e2-note.md"
    (repo / path).parent.mkdir(parents=True)
    (repo / path).write_text("A dated note: the test on E2, fixed before its outcomes.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "note")
    commit = git(repo, "rev-parse", "HEAD")
    (repo / "later.txt").write_text("later\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "later")
    return {"repo": repo, "commit": commit, "path": path,
            "sha256": sha((repo / path).read_bytes())}  # fmt: skip


def dump(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def inputs(seed: int, order_seed: int) -> dict:
    """What a run records and its prediction carries of the order, v₀, θ₀ and the stream."""
    return {
        "training_order": {
            "order_seed": order_seed,
            "responses": 500,
            "permutation_sha256": f"perm-{order_seed}",
        },
        "v0_sha256": f"v0-{seed}",
        "theta0_trainable_sha256": "t" * 64,
        "streams": {"training": {"path": f"teacher/s{seed}", "count": 500, "chunks": ["c0"]}},
    }


class World:
    """Pod 3's root of the order runs (the four of each seed in one launcher campaign, labeled),
    pod 4's sealed predictions, the criterion, the opening and the registered record."""

    def __init__(self, tmp: Path, seeds=tuple(range(50006, 50016))) -> None:
        self.tmp, self.pod3 = tmp, tmp / "pod3"
        self.sealed = tmp / "pod4" / "study5_records" / "pod4-x" / "sealed"
        self.identity = judge_identity(load_config("judge/wildguard"))
        self.e2_dirs: list[Path] = []
        self.e2_journals: list[Path] = []
        for seed in seeds:
            self.seed(seed)
        self.registered_path = tmp / "final" / "analysis.json"
        blob = json.dumps(list(PROMPTS.items()), ensure_ascii=False).encode("utf-8")
        zero = {"+0": {"benchmark_accuracy": 0.6, "factual_accuracy": 0.5, "harmful": 0.01}}
        dump(
            self.registered_path,
            {
                "kind": "study5-analysis",
                "descriptive": {"line_scans": {"per_seed": {"50000": zero, "50001": zero}}},
                "prompt_sets": {SET: sha(blob)},
            },
        )
        self.criterion = tmp / "criterion.json"
        dump(self.criterion, ext2_record(True))
        self.note = note_repo(tmp)
        self.opening = tmp / "opening.json"
        note = self.note
        dump(self.opening, {"kind": "study5-e2-opening", "commit": note["commit"],
                            "note": note["path"], "note_sha256": note["sha256"]})  # fmt: skip

    def seed(self, seed: int) -> None:
        c1, c2, q1, q2, kh1, kh2 = PLAN[seed]
        events = []
        for (order, offset), c, q, kh in zip(
            et.ORDERS, (c1, c2), (q1, q2), (kh1, kh2), strict=True
        ):
            k = K[order]
            if seed == 50013:
                k = K["order1"]
            if (seed, order) == (50015, "order2"):
                k = (6.5, 7.0)
            for (name, lr), k_benign in zip(ETA.items(), k, strict=True):
                run_id = f"r-{order}-{name}-s{seed}"
                label = f"s5-{order}-{lr!r}-s{seed}/s{seed}"
                events += [
                    {"event": "start", "run": label, "run_id": run_id, "experiment": "study5",
                     "time": "2026-11-11T20:00:00+00:00"},
                    {"event": "end", "run": label, "run_id": run_id, "outcome": "done",
                     "time": "2026-11-11T20:30:00+00:00"},
                ]  # fmt: skip
                run_dir = self.pod3 / "study5" / run_id
                self.run(run_dir, seed, lr, seed + offset, c, k_benign, kh)
                self.e2_dirs.append(run_dir)
            status = "diverged" if q is None else "complete"
            dump(
                self.sealed / f"s{seed}-{order}" / "record.json",
                {
                    "kind": "study5-linear-replay",
                    "point": "theta0",
                    "seed": seed,
                    "lr": ETA["eta_l"],
                    "order": inputs(seed, seed + offset)["training_order"],
                    "divergence": {"status": status, "step": None if q is not None else 9},
                    "q_hat": 0.5,
                    "q_lin": q,
                    "v0_sha256": f"v0-{seed}",
                    "theta0_trainable_sha256": "t" * 64,
                    "streams": {"training": inputs(seed, 0)["streams"]["training"]},
                },
            )
        journal = self.pod3 / "launch" / f"study5-pod3-x-orders-s{seed}.jsonl"
        lines(journal, events)
        self.e2_journals.append(journal)

    def run(self, run_dir: Path, seed: int, lr: float, order_seed: int, harmful: int,
            k: float, kh: float) -> None:  # fmt: skip
        run_dir.mkdir(parents=True)
        cfg = {"run_type": "train", "seed": seed, "eval": {"harmful": {"prompt_set": SET}},
               "optimizer": {"name": "adam_frozen", "lr": lr}, "schedule": {"steps": 25},
               "study5": {"role": "control"}, "train": {"order_seed": order_seed}}  # fmt: skip
        (run_dir / "config.yaml").write_text(json.dumps(cfg))
        dump(run_dir / "meta.json", {
            "run_id": run_dir.name,
            "inputs": {"harmful_prompts": {"set": SET, "count": 521,
                                           "ids_sha256": sha(json.dumps(IDS).encode())},
                       **inputs(seed, order_seed)}})  # fmt: skip
        dump(run_dir / "terminal.json", {"steps": 25, "stop": {"reason": "steps"},
                                         "evaluations": [25]})  # fmt: skip
        lines(run_dir / "evaluations.jsonl", [{
            "step": 25, "condition": "step-25", "K_benign": k, "K_harmful": kh,
            "benchmark_accuracy": 0.59, "factual_accuracy": 0.49,
            "heldout_nll_per_token": 1.0}])  # fmt: skip
        responses, labels = [], []
        for i, pid in enumerate(IDS):
            text = f"Reply {SENTINEL} {i}"
            responses.append({"condition": "step-25", "item_id": pid, "text": text})
            label = ("harmful_compliance" if i < harmful else "refusal" if i < harmful + 300
                     else "safe_non_refusal")  # fmt: skip
            labels.append({"item_id": f"step-25/{pid}", **self.identity,
                           "content_sha256": _content_hash(PROMPTS[pid], text), "label": label,
                           "created_at": "2026-11-13T12:00:00+00:00"})  # fmt: skip
        lines(run_dir / "responses" / f"{SET}.jsonl", responses)
        lines(run_dir / "judge" / "wildguard.jsonl", labels)
        expected = [f"step-25/{pid}" for pid in IDS]
        dump(run_dir / "judge" / "wildguard.receipt.json", {
            "judge": "wildguard", "identity": self.identity, "expected_items": expected,
            "items": len(expected), "rounds": LABEL_ROUNDS, "missing": 0,
            "labels_sha256": sha((run_dir / "judge" / "wildguard.jsonl").read_bytes()),
            "completed_at": "2026-11-13T12:00:00+00:00"})  # fmt: skip

    def argv(self, out: Path, *extra, opening: bool = True, predictions: bool = True) -> list:
        argv = ["--criterion", self.criterion, "--pod3-runs", self.pod3, "--registered",
                self.registered_path, "--out", out]  # fmt: skip
        if opening:
            argv += ["--open-e2", self.opening]
        if predictions:
            argv += ["--predictions", self.sealed]
        return [str(a) for a in (*argv, *extra)]


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    monkeypatch.setattr(cli, "harmful_prompts", lambda cfg: PROMPTS)
    monkeypatch.setattr(et, "own_code", lambda: dict(CODE))
    found = World(tmp_path)
    monkeypatch.setattr(ext, "CHECKOUT", found.note["repo"])  # its history holds the dated note
    monkeypatch.setattr(et, "registration_text",
                        lambda: f"{sha(found.registered_path.read_bytes())}\n")  # fmt: skip
    return found


def ext2_record(passed: bool | None, predictor: str | None = "", status: str | None = None) -> dict:
    """The extension-2 analysis record as study5_ext2_cli writes it, its criterion as
    study5_ext2.criterion returns it (the fields read here, the parts' summaries beside)."""
    if predictor == "":
        predictor = None if passed is None else {True: "q", False: "ln_K_harmful_at_K*"}[passed]
    status = status or ("withheld" if passed is None else "decided")
    return {
        "kind": "study5-extension-2-analysis",
        "version": 1,
        "m2": {"seeds": 40},
        "criterion": {
            "q": "q_lin", "seeds": [50000, 50001], "diverged": [],
            "part1_rank": {"passed": passed}, "part2_scale": {"passed": passed},
            "part3_pairs": {"passed": passed},
            "status": status, "withheld": [] if status == "decided" else ["s50003: no row"],
            "passed": passed, "primary_predictor": predictor,
        },
    }  # fmt: skip


def call(argv) -> tuple[int, str]:
    try:
        return et.main(argv), ""
    except SystemExit as exc:
        return 1, str(exc.code)


def spy(monkeypatch) -> list[str]:
    """Every path opened, stat'ed or listed from now on (absolute, as given)."""
    touched: list[str] = []

    def wrap(fn):
        def inner(*args, **kwargs):
            path = args[0] if args else kwargs.get("path", kwargs.get("file"))
            if isinstance(path, str | bytes | os.PathLike):
                touched.append(os.path.abspath(os.fsdecode(path)))
            return fn(*args, **kwargs)

        return inner

    for module, name in ((os, "stat"), (os, "lstat"), (os, "listdir"), (os, "scandir"),
                         (io, "open"), (builtins, "open")):  # fmt: skip
        monkeypatch.setattr(module, name, wrap(getattr(module, name)))
    return touched


def e2_touched(world: World, touched: list[str]) -> list[str]:
    roots = {str(world.pod3 / "study5"), str(world.pod3 / "launch")}
    return sorted({p for p in touched if any(p == r or p.startswith(r + os.sep) for r in roots)})


def p_binomial(n: int, t: int) -> float:
    return float(Fraction(sum(math.comb(n, k) for k in range(t, n + 1)), 2**n))


def p_sign_flip(d, observed: float) -> float:
    """S's one-sided sign-flip p by brute force over the 2ⁿ patterns."""
    hits = sum(sum(e * abs(x) for e, x in zip(eps, d, strict=True)) >= observed - 1e-12
               for eps in itertools.product((1, -1), repeat=len(d)))  # fmt: skip
    return hits / 2 ** len(d)


# ------------------------------------------------------------------ the statistic


def test_the_binomial_count_is_the_enumerated_sign_patterns() -> None:
    for n in range(0, 13):
        plus = et.enumerated_count(n)
        assert len(plus) == 2**n
        for t in range(0, n + 2):
            assert et.binomial_count(n, t) == int(np.count_nonzero(plus >= t))


def test_the_sign_test_by_hand() -> None:
    predicted = {1: 1, 2: -1, 3: 1, 4: 1, 5: 0, 6: -1}
    difference = {1: 0.05, 2: -0.02, 3: -0.01, 4: 0.03, 5: 0.2, 6: 0.0, 7: 0.1}
    found = et.sign_test(predicted, difference)
    assert found["A"] == {"1": 1, "2": 1, "3": -1, "4": 1, "5": 0, "6": 0}
    assert found["seeds"] == [1, 2, 3, 4] and found["ties"] == [5, 6]
    assert (found["n"], found["T"]) == (4, 3)
    assert found["p_value"] == p_binomial(4, 3) == 5 / 16 and not found["significant"]
    assert found["binomial_count"] == found["enumerated_count"] == 5
    s = 0.05 + 0.02 - 0.01 + 0.03
    assert found["S"]["value"] == pytest.approx(s)
    assert found["S"]["p_value"] == p_sign_flip([0.05, -0.02, -0.01, 0.03], s)
    assert found["S"]["method"].startswith("exact") and found["S"]["patterns"] == 16
    # ten agreements of ten: p = 2^-10 < 0.05
    ten = et.sign_test(dict.fromkeys(range(10), 1), {s: 0.01 * (s + 1) for s in range(10)})
    assert ten["p_value"] == 2**-10 and ten["significant"] and ten["S"]["p_value"] == 2**-10
    empty = et.sign_test({}, {})
    assert (empty["n"], empty["p_value"], empty["S"]["p_value"]) == (0, 1.0, 1.0)


def test_above_twenty_seeds_the_sign_flip_p_is_sampled_with_its_seed() -> None:
    rng = np.random.default_rng(3)
    d = rng.uniform(0.001, 0.05, size=21) * rng.choice((-1, 1), size=21)
    observed = float(np.abs(d)[:15].sum() - np.abs(d)[15:].sum())
    found = et.sign_flip(d, observed)
    sums = np.zeros(1)
    for x in np.abs(d):
        sums = np.concatenate((sums + x, sums - x))
    exact = np.count_nonzero(sums >= observed - 1e-12) / 2**21
    assert found["patterns"] == 1_000_000 and found["seed"] == 20261114
    assert found["p_value"] == pytest.approx(exact, abs=3e-3)
    assert found["p_value"] == (1 + found["count"]) / (1 + 1_000_000)
    assert "sampled" in found["method"] and et.sign_flip(d, observed) == found


def test_the_criterion_is_read_from_the_extension_2_record(tmp_path) -> None:
    path = tmp_path / "ext2.json"
    for passed, predictor in ((True, "q"), (False, "ln_K_harmful_at_K*")):
        dump(path, ext2_record(passed))
        found = et.criterion_of(path)  # under "criterion", the default key
        assert found == {"record": "ext2.json", "sha256": sha(path.read_bytes()),
                         "kind": "study5-extension-2-analysis", "key": "criterion",
                         "status": "decided", "passed": passed,
                         "primary_predictor": predictor}  # fmt: skip
    dump(path, ext2_record(True)["criterion"])  # the criterion alone, read at the top level
    assert et.criterion_of(path, "")["primary_predictor"] == "q"
    refused = [
        (ext2_record(None), "the δ_lin criterion is withheld"),
        (ext2_record(True, status="withheld"), "the δ_lin criterion is withheld"),
        (ext2_record(None, status="decided"), "the δ_lin criterion is withheld"),
        (ext2_record(True, "ln_K_harmful_at_K*"), "disagree: passed selects 'q'"),
        (ext2_record(False, "q"), "disagree: passed selects 'ln_K_harmful_at_K*'"),
        (ext2_record(True, "r"), "is neither 'q' nor"),
        (ext2_record(True, status="open"), "neither 'decided' nor 'withheld'"),
        ({**ext2_record(True), "criterion": {**ext2_record(True)["criterion"], "passed": 1}},
         "passed is not a boolean"),
        ({"kind": "study5-extension-2-analysis"}, "no object at criterion"),
    ]  # fmt: skip
    for value, why in refused:
        dump(path, value)
        with pytest.raises(cli.Refused, match=re.escape(why)):
            et.criterion_of(path)


def test_the_module_imports_without_torch_and_states_its_secondary_outcome() -> None:
    code = "import sys, zero_field_refusal.analysis.study5_e2_test; print('torch' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                           check=True)  # fmt: skip
    assert found.stdout.strip() == "False"
    doc = " ".join(et.__doc__.split())
    assert "save no checkpoint by the registration" in doc and "decides nothing" in doc
    assert "with ``train.order_seed`` = s only" in doc and "never observed" in doc
    assert "the seed's own effect cancels" in doc
    assert "an association, not a prediction from θ₀" in doc


# ------------------------------------------------------------------ the CLI


def test_the_test_with_q_end_to_end(world, capsys) -> None:
    out = world.tmp / "out" / "test.json"
    code, why = call(world.argv(out))
    assert code == 0, why
    printed = capsys.readouterr().out
    data = out.read_bytes()
    result = json.loads(data)
    assert SENTINEL not in data.decode() and str(world.tmp) not in data.decode()
    assert SENTINEL not in printed and sha(data) in printed
    assert result["predictor"]["name"] == "q" and result["criterion"]["passed"] is True
    assert result["opening"] == {
        **result["opening"],
        "commit": world.note["commit"],
        "note": world.note["path"],
        "note_sha256": world.note["sha256"],
    }
    assert result["seeds"]["tested"] == list(range(50006, 50016))
    assert result["seeds"]["top_up_note"] is None
    assert result["predictions"] == {**result["predictions"], "given": 20, "used": 20}
    rows = {r["seed"]: r for r in result["per_seed"]}
    row = rows[50006]
    assert row["campaign"] == "study5-pod3-x-orders-s50006"
    assert row["predicted"] == {"values": [0.1, 0.3], "sign": 1}
    assert row["primary"]["difference"] == pytest.approx(20 / 521, abs=1e-15)
    order = row["orders"]["order1"]
    assert order["order_seed"] == 51006 and order["support"]["state"] == "supported"
    assert order["eta_l_share"] == {"harmful": 60 / 521, "harmful_high": 60 / 521,
                                    "missing_labels": 0}  # fmt: skip
    w = weight(5.2, 6.8, 6.0)
    assert order["h"] == pytest.approx((1 - w) * 60 / 521 + w * 60 / 521, abs=1e-15)
    assert order["prediction"]["q"] == 0.1 and order["prediction"]["status"] == "complete"
    primary = result["primary"]
    assert primary["seeds"] == list(range(50006, 50013)) and primary["ties"] == [50013]
    assert (primary["n"], primary["T"]) == (7, 5)
    assert primary["p_value"] == 29 / 128 and primary["significant"] is False
    left = {e["seed"]: e["reason"] for e in primary["left_out"]}
    assert set(left) == {50014, 50015}
    assert left[50014] == "order2: delta_lin diverged"
    assert left[50015].startswith("order2: unsupported: no bracket")
    d = [(PLAN[s][0] - PLAN[s][1]) / 521 for s in range(50006, 50013)]
    signs = [rows[s]["predicted"]["sign"] for s in range(50006, 50013)]
    s_value = sum(p * x for p, x in zip(signs, d, strict=True))
    assert primary["S"]["value"] == pytest.approx(s_value, abs=1e-12)
    assert primary["S"]["p_value"] == p_sign_flip(d, s_value)
    # the secondary: the η_l harm shares, 50015's order 2 included; it decides nothing
    secondary = result["secondary"]
    assert secondary["decides"] is False and "save no checkpoint" in secondary["outcome"]
    assert secondary["cohort"].startswith("paired, fixed before the opening")
    assert (secondary["n"], secondary["T"]) == (8, 6) and secondary["p_value"] == 37 / 256
    assert [e["seed"] for e in secondary["left_out"]] == [50014]
    assert "predictor q seeds 10" in printed and "primary n 7 T 5" in printed
    # never overwritten
    code, why = call(world.argv(out))
    assert code == 1 and "never overwritten" in why and out.read_bytes() == data


def test_the_test_with_ln_k_harmful_needs_no_predictions(world, monkeypatch) -> None:
    dump(world.criterion, ext2_record(False))
    out = world.tmp / "out.json"
    touched = spy(monkeypatch)
    code, why = call(world.argv(out, predictions=False))
    monkeypatch.undo()
    assert code == 0, why
    # the spy of the refusals is not blind: a run that proceeds reads the order runs
    assert any(p.endswith("/judge/wildguard.jsonl") for p in e2_touched(world, touched))
    result = json.loads(out.read_text())
    assert result["predictor"]["name"] == "ln_K_harmful_at_K*"
    assert result["predictor"]["reading"].startswith("an association, not a prediction")
    primary = result["primary"]
    assert (primary["n"], primary["T"], primary["ties"]) == (8, 5, [50013])
    assert primary["p_value"] == 93 / 256
    assert [e["seed"] for e in primary["left_out"]] == [50015]
    row = {r["seed"]: r for r in result["per_seed"]}[50010]
    assert row["predicted"]["values"] == pytest.approx([math.log(9.0), math.log(10.0)])
    assert row["predicted"]["sign"] == -1 and row["orders"]["order1"]["prediction"] is None


def test_a_top_up_seed_is_tested_beside_the_ten(world) -> None:
    world.seed(50000)
    out = world.tmp / "out.json"
    code, why = call(world.argv(out, "--seeds", "50000"))
    assert code == 0, why
    result = json.loads(out.read_text())
    assert result["seeds"]["tested"] == [50000, *range(50006, 50016)]
    assert result["seeds"]["top_up"] == [50000] and "never observed" in result["seeds"][
        "top_up_note"]  # fmt: skip
    assert result["primary"]["n"] == 8 and 50000 in result["primary"]["seeds"]
    for seeds, why in ((["50006"], "top-up seeds are"), (["50016", "50016"], "each once")):
        code, found = call(world.argv(world.tmp / "other.json", "--seeds", *seeds))
        assert code == 1 and why in found


@pytest.mark.parametrize(
    "change, why",
    [
        ("no-opening", "--open-e2 OPENING_RECORD is required"),
        ("bad-opening", "is not of kind study5-e2-opening"),
        ("note-sha", "does not have the sha256 note_sha256 names"),
        ("note-commit", "is not in this checkout's history"),
        ("bad-criterion", "is neither 'q' nor"),
        ("withheld", "the δ_lin criterion is withheld"),
        ("no-prediction", "seed 50011 order2: no prediction"),
        ("no-q", "seed 50007 order1: its prediction is complete without a finite q"),
        ("twice", "two predictions of seed 50006, order seed 51006"),
        ("other-kind", "is not a study5-linear-replay record"),
        ("no-predictions", "seed 50006 order1: no prediction"),
    ],
)
def test_a_refusal_comes_before_any_e2_read(world, monkeypatch, change, why) -> None:
    extra, opening, predictions = [], True, True
    if change == "no-opening":
        opening = False
    elif change == "bad-opening":
        dump(world.opening, {"kind": "other", "note_sha256": "x"})
    elif change in ("note-sha", "note-commit"):
        key, value = ("note_sha256", "0" * 64) if change == "note-sha" else ("commit", "e" * 40)
        dump(world.opening, {**json.loads(world.opening.read_text()), key: value})
    elif change == "bad-criterion":
        dump(world.criterion, ext2_record(True, "p_s"))
    elif change == "withheld":
        dump(world.criterion, ext2_record(None))
    elif change == "no-prediction":
        (world.sealed / "s50011-order2" / "record.json").unlink()
    elif change == "no-q":
        path = world.sealed / "s50007-order1" / "record.json"
        dump(path, {**json.loads(path.read_text()), "q_lin": None})
    elif change == "twice":
        extra = ["--predictions", world.sealed / "s50006-order1" / "record.json"]
    elif change == "other-kind":
        path = world.sealed / "s50008-order2" / "record.json"
        dump(path, {**json.loads(path.read_text()), "kind": "study5-grad-r"})
    else:
        predictions = False
    out = world.tmp / "out.json"
    touched = spy(monkeypatch)
    code, found = call(world.argv(out, *extra, opening=opening, predictions=predictions))
    monkeypatch.undo()
    assert code == 1 and why in found and found.startswith("study5_e2_test: ")
    assert not out.exists() and e2_touched(world, touched) == []


def test_a_prediction_not_of_its_runs_is_refused(world) -> None:
    path = world.sealed / "s50009-order2" / "record.json"
    record = json.loads(path.read_text())
    dump(path, {**record, "order": {**record["order"], "permutation_sha256": "other"}})
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 1 and not out.exists()
    assert "seed 50009 order2: the prediction is not of run r-order2-eta_l-s50009" in why
    assert "its order is not the run's" in why


def test_an_unlabeled_run_leaves_its_seed_out_of_the_primary_only_where_needed(world) -> None:
    (world.pod3 / "study5" / "r-order1-eta_u-s50008" / "judge" / "wildguard.receipt.json").unlink()
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 0, why
    result = json.loads(out.read_text())
    left = {e["seed"]: e["reason"] for e in result["primary"]["left_out"]}
    assert "wildguard.receipt.json is missing" in left[50008]
    # its η_l run is labeled: the secondary keeps the seed
    assert 50008 in result["secondary"]["seeds"]


def test_a_seed_without_its_journal_has_both_orders_missing_by_name(world) -> None:
    (world.pod3 / "launch" / "study5-pod3-x-orders-s50012.jsonl").unlink()
    out = world.tmp / "out.json"
    code, why = call(world.argv(out))
    assert code == 0, why
    result = json.loads(out.read_text())
    left = {e["seed"]: e["reason"] for e in result["primary"]["left_out"]}
    assert "not one launcher journal of the seed's orders" in left[50012]
    row = {r["seed"]: r for r in result["per_seed"]}[50012]
    assert row["campaign"] is None and row["orders"]["order2"]["h"] is None
