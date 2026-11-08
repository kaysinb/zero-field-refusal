"""Study 5's fixation record (docs/study5-preregistration.md; ``analysis.study5_fixation`` and
``scripts/study5_fixation.py``): the extreme blocks on a synthetic freeze with known means (top and
bottom 5, the tie-break), the reference sample's reproducibility and cell counts, the record's
determinism, and the script end to end on study 3's synthetic world (its freeze committed, its
analysis written): the record byte-identical on a rerun and never over a different file; the
reference annotators' input in study 3's audit form, its texts only in unique.jsonl, never
printed, never where Git could track it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_study3_audit_script import Audit
from test_study3_campaigns import git, script_module
from test_study3_package2 import SENTINEL

from zero_field_refusal.analysis import study3_audit, study3_final
from zero_field_refusal.analysis import study5_fixation as fx
from zero_field_refusal.analysis.study3_freeze import freeze_name
from zero_field_refusal.analysis.study3_labels import RunHarm
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.seeding import derive_seed

SCRIPT = REPO_ROOT / "scripts" / "study5_fixation.py"
IDS = [f"judged:{i}" for i in range(521)]


def synthetic(harmful: dict[int, tuple[int, int, int, int]], w: float = 0.25):
    """Frozen A–F blocks and the harm of their runs: per seed the harmful counts of A lo, A hi,
    F lo and F hi; the block's mean at K* is known in closed form."""
    blocks, harm, rates = [], {}, {}
    for seed, counts in harmful.items():
        arms = {}
        for k, arm in enumerate(("A", "F")):
            entry = {"w": w, "width": 0.4}
            for j, side in enumerate(("lo", "hi")):
                run = f"r-{arm}-{side}-s{seed}"
                entry[side] = run
                harm[run] = RunHarm(harmful=counts[2 * k + j], missing=1, receipt_sha256="f" * 64)
                rates[run] = {"A": (1.45e-5, 2.11e-5), "F": (1.3e-5, 1.83e-5)}[arm][j]
            arms[arm] = entry
        blocks.append({"seed": seed, "arms": arms})
    return blocks, harm, rates


def expected_mean(counts, w=0.25) -> float:
    a = (1 - w) * counts[0] / 521 + w * counts[1] / 521
    f = (1 - w) * counts[2] / 521 + w * counts[3] / 521
    return (a + f) / 2


def test_block_means_are_study3s_interpolation_at_kstar() -> None:
    blocks, harm, _ = synthetic({10400: (40, 80, 20, 60), 10401: (0, 4, 8, 12)})
    means = fx.block_means(blocks, harm)
    assert [m["seed"] for m in means] == [10400, 10401]
    for m, counts in zip(means, ((40, 80, 20, 60), (0, 4, 8, 12)), strict=True):
        assert m["mean"] == pytest.approx(expected_mean(counts), abs=1e-15)
        assert m["A"] == pytest.approx(0.75 * counts[0] / 521 + 0.25 * counts[1] / 521)
    # a block that misses an arm is refused, never guessed
    del blocks[0]["arms"]["F"]
    with pytest.raises(ValueError, match="carries no arm F"):
        fx.block_means(blocks, harm)


def test_extreme_blocks_top_and_bottom_five_ties_by_seed() -> None:
    counts = {10400 + i: (i, i, i, i) for i in range(20)}
    counts[10450] = (19, 19, 19, 19)  # ties the highest block 10419
    counts[10399] = (0, 0, 0, 0)  # ties the lowest block 10400
    counts[10451] = (3, 3, 3, 3)  # ties 10403 at the edge of the low group
    blocks, harm, _ = synthetic(counts)
    means = fx.block_means(blocks, harm)
    found = fx.extreme_blocks(means)
    assert [m["seed"] for m in found["high"]] == [10419, 10450, 10418, 10417, 10416]
    assert [m["seed"] for m in found["low"]] == [10399, 10400, 10401, 10402, 10403]
    assert found["high"][0]["mean"] >= found["high"][-1]["mean"]
    assert not {m["seed"] for m in found["high"]} & {m["seed"] for m in found["low"]}
    # the order of the input does not matter
    assert fx.extreme_blocks(list(reversed(means))) == found
    with pytest.raises(ValueError, match="need 10"):
        fx.extreme_blocks(means[:9])


def test_reference_sample_is_reproducible_with_its_cell_counts() -> None:
    blocks, harm, rates = synthetic({10400 + i: (i, 2 * i, i, 3 * i) for i in range(12)})
    found = fx.extreme_blocks(fx.block_means(blocks, harm))
    chosen = fx.chosen_blocks(found, blocks)
    cells = fx.reference_cells(chosen, rates, IDS)
    assert len(cells) == 40 and sum(len(c["prompt_ids"]) for c in cells) == 200
    assert [(c["group"], c["rank"]) for c in cells[::4]] == [
        *(("high", r) for r in range(1, 6)), *(("low", r) for r in range(1, 6))]  # fmt: skip
    assert [(c["arm"], c["side"]) for c in cells[:4]] == [("A", "lo"), ("A", "hi"),
                                                           ("F", "lo"), ("F", "hi")]  # fmt: skip
    for cell in cells:
        assert len(set(cell["prompt_ids"])) == 5 and set(cell["prompt_ids"]) <= set(IDS)
        assert cell["weight"] == 521 / 5
        assert cell["lr"] == rates[cell["run_id"]]
        seed = derive_seed(cell["seed"], "study5", "reference", cell["arm"], cell["lr"])
        assert cell["rng_seed"] == seed
        picked = np.random.default_rng(seed).choice(521, size=5, replace=False)
        assert cell["prompt_ids"] == [IDS[i] for i in picked]
    assert fx.reference_cells(chosen, rates, IDS) == cells  # reproducible
    assert len({tuple(c["prompt_ids"]) for c in cells}) == 40  # each cell its own stream
    assert fx.relabel_runs(chosen) == [c["run_id"] for c in cells]
    with pytest.raises(ValueError, match="521"):
        fx.reference_cells(chosen, rates, IDS[:520])
    same = {run: 1e-5 for run in rates}
    with pytest.raises(ValueError, match="share a rate"):
        fx.reference_cells(chosen, same, IDS)


def test_the_record_is_deterministic_with_the_fixed_settings() -> None:
    blocks, harm, rates = synthetic({10400 + i: (i, 2 * i, i, 3 * i) for i in range(12)})
    freeze = {"candidate": 1, "comparisons": {"A-F": {"blocks": blocks}}}
    kw = {"freeze": freeze, "harm": harm, "rates": rates, "prompt_ids": IDS,
          "inputs": {"x": 1}, "code": {"commit": "c" * 40}}  # fmt: skip
    first, second = fx.fixation_record(**kw), fx.fixation_record(**kw)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    seeds = first["seeds"]
    assert seeds["fresh"]["seeds"] == list(range(50000, 50040))
    assert seeds["alternative"]["seeds"] == list(range(50000, 50024))
    assert seeds["controls"] == seeds["ladder"] == list(range(50000, 50006))
    from test_study5_campaigns import camp

    assert seeds["v0_source"] == {str(s): camp.source_seed(s) for s in range(50000, 50006)}
    test = first["permutation_test"]
    assert (test["permutations"], test["alpha"], test["min_supported_seeds"]) == (100_000, 0.05, 20)
    assert test["seed"] == derive_seed(50000, "study5", "permutation")
    assert test["p_value"] == "(1 + #{rho_perm <= rho_obs}) / (1 + 100000)"
    boot = first["retrospective_interval"]
    assert (boot["resamples"], boot["level"], boot["interval"]) == (10_000, 0.95, "percentile")
    assert boot["seed"] == derive_seed(50000, "study5", "bootstrap")
    assert first["scan_intervals"]["level"] == 0.95
    extreme = first["extreme_blocks"]
    assert len(extreme["high"]) == len(extreme["low"]) == 5
    assert len(extreme["relabel"]["runs"]) == 40 == len(set(extreme["relabel"]["runs"]))
    sample = first["reference_sample"]
    assert (sample["cells"], sample["items"], len(sample["cells_drawn"])) == (40, 200, 40)
    assert "mean over its cells" in sample["estimator"]
    assert sample["order"]["seed"] == derive_seed(50000, "study5", "reference", "order")
    json.dumps(first, allow_nan=False)


# ------------------------------------------------------------------ the script, end to end


class Fixation(Audit):
    """Study 3's synthetic world with its analysis written (``Audit``), the fixation script's
    readers pointed at it: its state's endpoints (their rates), its harm, the runs' label hashes
    and each relabeled run's 521 prompt ids."""

    def __init__(self, tmp_path, monkeypatch, capsys) -> None:
        super().__init__(tmp_path, monkeypatch, capsys)
        self.fix = script_module(SCRIPT, "study5_fixation")
        self.here = tmp_path / "here"  # this checkout, its code committed
        for rel in self.fix.CODE_FILES:
            (self.here / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.here / rel).write_bytes((REPO_ROOT / rel).read_bytes())
        git(self.here, "init", "-q")
        git(self.here, "add", "-A")
        git(self.here, "commit", "-qm", "the code")
        world = self.world
        mp = monkeypatch
        mp.setattr(self.fix, "REPO", self.here)
        mp.setattr(self.fix, "read_state", lambda *a, **kw: (world.state, list(world.endpoints)))
        mp.setattr(self.fix, "label_hashes",
                   lambda run_dir, judge: {"labels_sha256": "e" * 64,
                                           "receipt_sha256": "f" * 64})  # fmt: skip
        mp.setattr(self.fix, "run_prompt_ids", lambda run_dir: (list(IDS), "d" * 64))
        ledger = self.root / "sealed" / "study3-ledger"
        ledger.mkdir(parents=True)
        (ledger / "batch-b1.json").write_text("{}\n")
        (ledger / "acceptance-b1.json").write_text("{}\n")
        self.out = self.here / "docs" / "study5-fixation.json"
        self.theta0 = self.root / "study2_stage_a" / "theta0-run"

    def common(self) -> list:
        return ["--runs-root", self.root, "--study3-repo", self.world.repo]

    def call_fix(self, *argv) -> tuple[int, str, str]:
        self.capsys.readouterr()
        code, message = 0, ""
        try:
            code = self.fix.main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            message = exc.code if isinstance(exc.code, str) else ""
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err + message

    def fixed(self) -> str:
        code, out, error = self.call_fix("fix", *self.common(), "--theta0", self.theta0,
                                         "--sealed-dir", self.root / "sealed",
                                         "--out", self.out)  # fmt: skip
        assert code == 0, error
        return out


@pytest.fixture
def fixation(tmp_path, monkeypatch, capsys) -> Fixation:
    return Fixation(tmp_path, monkeypatch, capsys)


def test_fix_writes_the_record_once_and_a_rerun_is_byte_identical(fixation) -> None:
    out = fixation.fixed()
    data = fixation.out.read_bytes()
    record = json.loads(data)
    assert record["kind"] == "study5-fixation"
    freeze = json.loads((fixation.world.repo / freeze_name(1)).read_bytes())
    assert record["inputs"]["study3_freeze"]["sha256"] == hashlib.sha256(
        (fixation.world.repo / freeze_name(1)).read_bytes()).hexdigest()  # fmt: skip
    assert record["inputs"]["labels"]["count"] == 240
    assert record["inputs"]["prompt_ids"]["count"] == 521
    code = git(fixation.here, "rev-parse", "HEAD")
    assert record["code"]["commit"] == code
    assert set(record["code"]["files"]) == set(fixation.fix.CODE_FILES)
    seeds = [b["seed"] for b in freeze["comparisons"]["A-F"]["blocks"]]
    chosen = [m["seed"] for g in ("high", "low") for m in record["extreme_blocks"][g]]
    assert set(chosen) <= set(seeds) and len(set(chosen)) == 10
    assert hashlib.sha256(data).hexdigest() in out
    for b in record["extreme_blocks"]["high"]:
        assert str(b["seed"]) in out
    # a rerun: the same bytes, nothing rewritten; also once the record is committed
    mtime = fixation.out.stat().st_mtime_ns
    assert "unchanged" in fixation.fixed()
    assert fixation.out.read_bytes() == data and fixation.out.stat().st_mtime_ns == mtime
    git(fixation.here, "add", "-A")
    git(fixation.here, "commit", "-qm", "the record")
    assert "unchanged" in fixation.fixed()
    assert fixation.out.read_bytes() == data
    # a different file at the place is never overwritten
    fixation.out.write_text("{}\n")
    code, _, why = fixation.call_fix("fix", *fixation.common(), "--theta0", fixation.theta0,
                                     "--sealed-dir", fixation.root / "sealed",
                                     "--out", fixation.out)  # fmt: skip
    assert code not in (0, None) and "different" in why
    assert fixation.out.read_text() == "{}\n"


def test_fix_needs_a_clean_checkout(fixation) -> None:
    (fixation.here / "scripts" / "untracked.py").write_text("x")
    code, _, why = fixation.call_fix("fix", *fixation.common(), "--theta0", fixation.theta0,
                                     "--out", fixation.out)  # fmt: skip
    assert code not in (0, None) and "clean" in why
    assert not fixation.out.exists()


def reference(fixation: Fixation, folder: Path) -> tuple[int, str, str]:
    return fixation.call_fix("reference-input", *fixation.common(), "--fixation", fixation.out,
                             "--out", folder)  # fmt: skip


def test_reference_input_is_study3s_blind_form_texts_never_shown(fixation) -> None:
    fixation.fixed()
    record = json.loads(fixation.out.read_bytes())
    folder = fixation.root / "study5-reference-input"
    code, out, error = reference(fixation, folder)
    assert code == 0, error
    assert {p.name for p in folder.iterdir()} == {"unique.jsonl", "key.json", "manifest.json"}
    assert oct(folder.stat().st_mode & 0o777) == "0o700"
    assert all(oct(p.stat().st_mode & 0o777) == "0o600" for p in folder.iterdir())
    lines = (folder / "unique.jsonl").read_text().splitlines()
    unique = [json.loads(line) for line in lines]
    assert len(unique) == 200
    assert all(set(u) == set(study3_audit.BLIND) for u in unique)
    for line, u in zip(lines, unique, strict=True):  # study 3's encoding of a row
        assert line == json.dumps(u, ensure_ascii=False, sort_keys=True)
    assert all(SENTINEL in u["response"] for u in unique)
    key = json.loads((folder / "key.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    assert sorted(key) == sorted(u["item_id"] for u in unique)
    cells = {(c["run_id"], pid) for c in record["reference_sample"]["cells_drawn"]
             for pid in c["prompt_ids"]}  # fmt: skip
    assert {(k["run_id"], k["prompt"]) for k in key.values()} == cells
    # the blinding: study 3's order and opaque ids, under the fixed order seed
    items = [{"cell": i, "prompt": pid}
             for i, c in enumerate(record["reference_sample"]["cells_drawn"])
             for pid in c["prompt_ids"]]  # fmt: skip
    audit_script = script_module(REPO_ROOT / "scripts" / "study3_audit.py", "s3_audit_fx")
    named = audit_script.ordered(items, record["reference_sample"]["order"]["seed"])
    assert [item_id for item_id, _ in named] == [u["item_id"] for u in unique]
    assert [(key[i]["cell"], key[i]["prompt"]) for i, _ in named] == [
        (e["cell"], e["prompt"]) for _, e in named]  # fmt: skip
    assert manifest["fixation"]["sha256"] == hashlib.sha256(fixation.out.read_bytes()).hexdigest()
    digest = hashlib.sha256((folder / "unique.jsonl").read_bytes()).hexdigest()
    assert manifest["files"]["unique.jsonl"] == {"items": 200, "sha256": digest}
    assert len(manifest["cells"]) == 40
    assert sum(len(c["item_ids"]) for c in manifest["cells"]) == 200
    # the texts are in unique.jsonl alone, and never shown
    for name in ("key.json", "manifest.json"):
        assert SENTINEL not in (folder / name).read_text() and "Placeholder" not in (
            folder / name).read_text()  # fmt: skip
    assert SENTINEL not in out + error and "Placeholder" not in out + error
    manifest_sha = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
    assert manifest_sha in out
    # written once
    code, _, why = reference(fixation, folder)
    assert code not in (0, None) and "exists" in why


def test_reference_input_is_refused_where_git_could_track_it(fixation) -> None:
    fixation.fixed()
    folder = REPO_ROOT / "study5-reference-check"
    code, _, why = reference(fixation, folder)
    assert code not in (0, None) and "Git" in why
    assert not folder.exists()


def test_reference_input_waits_for_study3s_analysis(fixation) -> None:
    fixation.fixed()
    (fixation.root / "study3-final" / "c1" / "analysis.json").unlink()
    folder = fixation.root / "study5-reference-input"
    code, _, why = reference(fixation, folder)
    assert code not in (0, None) and "analysis" in why
    assert not folder.exists()
    assert not list(fixation.root.glob(".study5-reference-input-*"))


def test_the_record_names_study3s_runs_and_the_relabel_set(fixation) -> None:
    fixation.fixed()
    record = json.loads(fixation.out.read_bytes())
    freeze = json.loads((fixation.world.repo / freeze_name(1)).read_bytes())
    runs = set(study3_final.runs_of(freeze["comparisons"]["A-F"]["blocks"]))
    assert set(record["inputs"]["labels"]["runs"]) == runs
    relabel = record["extreme_blocks"]["relabel"]["runs"]
    assert set(relabel) <= runs and len(relabel) == 40
    assert set(record["inputs"]["run_configs"]) == set(relabel)


def test_the_script_is_analysis_code_outside_the_gpu_identity() -> None:
    """It reads records and writes the record and the input: outside the GPU code identity, it
    imports nothing that runs on a GPU, and the one file it loads by path is study 3's audit
    script (an analysis script itself, for its blind order and its write-once folder)."""
    from test_code_identity import gpu_uses

    from zero_field_refusal.code_identity import ANALYSIS_SCRIPTS, is_gpu_path

    assert "scripts/study5_fixation.py" in ANALYSIS_SCRIPTS
    assert is_gpu_path("scripts/study5_fixation.py") is False
    assert is_gpu_path("src/zero_field_refusal/analysis/study5_fixation.py") is False
    assert gpu_uses(SCRIPT) == {"loads code at run time: spec_from_file_location"}
    assert '"scripts" / "study3_audit.py"' in SCRIPT.read_text()
    assert "scripts/study3_audit.py" in ANALYSIS_SCRIPTS
    assert gpu_uses(REPO_ROOT / "scripts" / "study3_audit.py") == set()
    assert not gpu_uses(
        REPO_ROOT / "src/zero_field_refusal/analysis/study5_fixation.py",
        "zero_field_refusal.analysis.study5_fixation",
    )


def test_the_record_fixes_the_judges_contrast_and_binds_its_inputs(fixation) -> None:
    """The judges' high-low contrast is the reference's estimator, per cell over all 521
    responses; the block mean at K* chose the blocks only. The study 3 checkout is named as what
    it supplies, the sealed ledger by its files' sha256, and the state's records and θ₀'s run by
    what binds them."""
    fixation.fixed()
    record = json.loads(fixation.out.read_bytes())
    contrast = record["extreme_blocks"]["contrast"]
    assert contrast["judges"] == ["WildGuard", "Llama-Guard-3-8B"]
    assert "all 521" in contrast["cell"] and "own label files" in contrast["cell"]
    assert "unweighted mean over the group's 20 cells" in contrast["group"]
    assert contrast["contrast"] == "high minus low"
    assert "the reference sample's estimator" in contrast["same_as"]
    assert "selection measure only" in contrast["selection"]
    assert record["extreme_blocks"]["count"] * 4 == 20
    inputs = record["inputs"]
    assert "study3_readers" not in inputs
    checkout = inputs["study3_checkout"]
    assert set(checkout) == {"commit", "gpu_code_sha256", "supplies"}
    assert "GPU-code identity" in checkout["supplies"] and str(record["code"]["commit"]) in (
        checkout["supplies"])  # fmt: skip
    ledger = inputs["ledger"]
    folder = fixation.root / "sealed" / "study3-ledger"
    assert ledger["files"] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(folder.iterdir())}  # fmt: skip
    assert ledger["count"] == 2
    assert "freeze" in inputs["state"]["bound_by"] and "lr" in inputs["state"]["bound_by"]
    assert "run id" in inputs["theta0"]["bound_by"]


def test_fix_names_data_dir_when_the_prompt_set_is_missing(fixation, monkeypatch) -> None:
    def missing(run_dir):
        raise FileNotFoundError("no prompt set")

    monkeypatch.setattr(fixation.fix, "run_prompt_ids", missing)
    code, _, why = fixation.call_fix("fix", *fixation.common(), "--theta0", fixation.theta0,
                                     "--sealed-dir", fixation.root / "sealed",
                                     "--out", fixation.out)  # fmt: skip
    assert code not in (0, None) and "DATA_DIR" in why
    assert not fixation.out.exists()
