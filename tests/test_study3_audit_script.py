"""scripts/study3_audit.py, in process: ``draw`` waits for the judge-based analysis, draws once,
exports the blind texts in unique.jsonl alone and never shows a text; ``agreements`` gives the
agreements the researcher checks; ``estimate`` needs every item's reference, counts the missing
ones, and leaves an arm's error rates undetermined below 20 reference-harmful items.

The world is the analysis script's (``test_study3_analysis_script.World``): candidate 1's done
study, its freeze committed, its analysis written. The label reader's per-item labels are stubbed
from each run's counts (its first ``harmful`` prompts harmful, the next ``missing`` missing); each
bracket run's ``responses/`` holds a placeholder reply with the sentinel, read through the final
reader's own entry (its run reading, ``_texts``, stubbed). The freeze gate is the real one."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from test_code_identity import gpu_uses
from test_study3_analysis_script import BOOTSTRAP, World, raisers
from test_study3_campaigns import script_module
from test_study3_freeze import SUPPLEMENT
from test_study3_package2 import SENTINEL, SealedGuard

from zero_field_refusal.analysis import study3_audit as au
from zero_field_refusal.analysis import study3_final
from zero_field_refusal.analysis.study3_freeze import freeze_name
from zero_field_refusal.analysis.study3_labels import ItemLabel
from zero_field_refusal.analysis.study3_versions import AUDIT_PROCEDURE
from zero_field_refusal.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "study3_audit.py"
AUDIT = {"seed": 20261102, "agreement_seed": 20261103, "procedure": AUDIT_PROCEDURE}
FILES = {"unique.jsonl", "key.json", "frame.jsonl", "selection.json"}
REFUSAL = "refusal"
HARMFUL = "harmful_compliance"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    """The analysis world with its freeze committed (and, with ``analyzed``, its analysis
    written), the audit script's in-force supplement given, the per-item labels stubbed from the
    runs' counts (``skew``: a run whose items count one harmful more) and each bracket run's
    placeholder reply in its ``responses/``."""

    def __init__(self, tmp_path, monkeypatch, capsys, analyzed: bool = True) -> None:
        self.capsys = capsys
        self.world = world = World(tmp_path, monkeypatch, capsys)
        world.frozen()
        if analyzed:
            world.run("analyze", "--runs-root", world.root, "--candidate", "1")
        self.root = world.root
        self.folder = world.root / "study3-final" / "c1" / "audit"
        self.script = script_module(SCRIPT, "study3_audit")
        supplement = {"bootstrap": dict(BOOTSTRAP), "audit": dict(AUDIT)}
        mp = monkeypatch
        mp.setattr(self.script, "REPO", world.repo)
        mp.setattr(self.script, "in_force_supplement",
                   lambda c, **kw: (supplement, dict(SUPPLEMENT), []))  # fmt: skip
        mp.setattr(study3_final, "run_items", self.items)
        mp.setattr(study3_final, "_texts", self.texts)
        self.skew: str | None = None
        freeze = json.loads((world.repo / freeze_name(1)).read_text())
        blocks = [b for entry in freeze["comparisons"].values() for b in entry["blocks"]]
        self.brackets = study3_final.runs_of(blocks)
        for run in self.brackets:
            folder = world.root / "study3" / run / "responses"
            folder.mkdir(parents=True)
            (folder / "harmful_evaluation.jsonl").write_text(
                json.dumps({"text": f"Placeholder reply {SENTINEL}"}) + "\n")  # fmt: skip

    def items(self, run_dir, **kw) -> tuple[ItemLabel, ...]:
        harm = self.world.harm(run_dir)
        h = harm.harmful + (Path(run_dir).name == self.skew)
        return tuple(ItemLabel(i, f"p{i}", "harmful" if i < h else
                               "missing" if i < h + harm.missing else "not_harmful")
                     for i in range(521))  # fmt: skip

    def texts(self, run_dir: Path, wanted) -> dict:
        """The run's step-25 texts, its placeholder reply read from its responses/."""
        reply = json.loads((run_dir / "responses" / "harmful_evaluation.jsonl").read_text())
        return {pid: {"prompt": f"Placeholder prompt {pid}.",
                      "response": f"{reply['text']} ({pid})"} for pid in wanted}  # fmt: skip

    def call(self, *argv) -> tuple[int, str, str]:
        self.capsys.readouterr()
        code, message = 0, ""
        try:
            self.script.main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
            message = exc.code if isinstance(exc.code, str) else ""
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err + message

    def run(self, command: str, *argv) -> str:
        code, out, error = self.call(command, "--runs-root", self.root, "--candidate", "1", *argv)
        assert code == 0, error
        return out

    def refused(self, command: str, *argv) -> str:
        code, out, error = self.call(command, "--runs-root", self.root, "--candidate", "1", *argv)
        assert code not in (0, None), out
        return error

    def key(self) -> dict:
        return json.loads((self.folder / "key.json").read_text())

    def labels(self, name: str, labels: dict) -> Path:
        path = self.root.parent / f"{name}.json"
        path.write_text(json.dumps(labels))
        return path


@pytest.fixture
def audit(tmp_path, monkeypatch, capsys) -> Audit:
    return Audit(tmp_path, monkeypatch, capsys)


def test_the_script_does_no_gpu_work() -> None:
    assert gpu_uses(SCRIPT) == set()


# ------------------------------------------------------------------ draw


def test_draw_waits_for_the_analysis_writing_and_opening_nothing(tmp_path, monkeypatch, capsys):
    audit = Audit(tmp_path, monkeypatch, capsys, analyzed=False)
    calls = raisers(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=audit.root)
    assert "after the judge-based analysis" in audit.refused("draw")
    assert not audit.folder.parent.exists()
    assert guard.attempts == [] and calls == []


def test_draw_writes_its_files_once_and_shows_counts_only(audit) -> None:
    out = audit.run("draw")
    assert {p.name for p in audit.folder.iterdir()} == FILES
    selection = json.loads((audit.folder / "selection.json").read_text())
    for name in FILES - {"selection.json"}:
        assert selection["files"][name]["sha256"] == sha256(audit.folder / name)
    assert {arm: entry["draws"] for arm, entry in selection["arms"].items()} == au.SIZES
    assert selection["audit"] == AUDIT and selection["sizes"] == au.SIZES
    analysis = json.loads((audit.folder.parent / "analysis.json").read_text())
    for e, key in au.FROZEN.items():
        mean = analysis["comparisons"][key]["primary"]["mean"]
        assert selection["judge_points"][e] == pytest.approx(mean, abs=1e-12)
    assert selection["analysis"] == {
        "path": "study3-final/c1/analysis.json",
        "root": "runs",
        "sha256": sha256(audit.folder.parent / "analysis.json"),
    }
    key = audit.key()
    unique = [json.loads(line) for line in (audit.folder / "unique.jsonl").read_text().splitlines()]
    assert [u["item_id"] for u in unique] != sorted(key)  # the order stream, not the ids'
    assert sorted(u["item_id"] for u in unique) == sorted(key)
    assert all(set(u) == {"item_id", "prompt", "response"} for u in unique)
    assert sum(entry["multiplicity"] for entry in key.values()) == 1000
    assert "400" in out and str(len(key)) in out
    assert "one terminal draw" in audit.refused("draw")
    assert {p.name for p in audit.folder.parent.iterdir()} == {
        "analysis.json", "validation-harm.json", "audit"}  # fmt: skip


def test_the_texts_are_only_in_unique_jsonl_never_shown(audit) -> None:
    code, out, error = audit.call("draw", "--runs-root", audit.root, "--candidate", "1")
    assert code == 0, error
    assert SENTINEL in (audit.folder / "unique.jsonl").read_text()
    for name in FILES - {"unique.jsonl"}:
        assert SENTINEL not in (audit.folder / name).read_text(), name
    assert SENTINEL not in out + error and "Placeholder" not in out + error


def test_item_labels_that_disagree_with_the_runs_harm_are_refused(audit) -> None:
    audit.skew = audit.brackets[7]
    why = audit.refused("draw")
    assert "counts" in why and audit.brackets[7] in why
    assert not audit.folder.exists() and SENTINEL not in why


def test_draw_is_refused_where_git_could_track_it(audit) -> None:
    code, _, why = audit.call("draw", "--runs-root", REPO_ROOT / "audit-check", "--candidate", "1")
    assert code not in (0, None) and "Git" in why
    assert not (REPO_ROOT / "audit-check").exists()


# ------------------------------------------------------------------ agreements


def labellers(key: dict) -> tuple[dict, dict]:
    """GPT's and Grok's labels: both refusal but on every fifth item, where Grok says harmful
    compliance; GPT's output failed on the first item."""
    ids = sorted(key)
    gpt = {i: REFUSAL for i in ids}
    grok = {i: HARMFUL if n % 5 == 0 else REFUSAL for n, i in enumerate(ids)}
    gpt[ids[1]] = None
    return gpt, grok


def test_agreements_check_a_tenth_of_them_by_the_seed(audit) -> None:
    audit.run("draw")
    gpt, grok = labellers(audit.key())
    short = dict(list(gpt.items())[1:])
    assert "unique.jsonl" in audit.refused("agreements", "--gpt", audit.labels("short", short),
                                           "--grok", audit.labels("grok", grok))  # fmt: skip
    out = audit.run("agreements", "--gpt", audit.labels("gpt", gpt),
                    "--grok", audit.labels("grok", grok))  # fmt: skip
    record = json.loads((audit.folder / "agreements.json").read_text())
    assert len(record["check"]) == math.ceil(0.1 * len(record["agreements"]))
    assert record["check"] == au.agreements(gpt, grok, seed=AUDIT["agreement_seed"])["check"]
    assert record["failed"] == [sorted(gpt)[1]] and record["counts"]["agreements"] > 0
    assert all(item in out for item in record["check"] + record["disagreements"])
    assert "written once" in audit.refused("agreements", "--gpt", audit.labels("gpt", gpt),
                                           "--grok", audit.labels("grok", grok))  # fmt: skip


# ------------------------------------------------------------------ estimate


def resolved(key: dict, harmful: dict[str, int]) -> dict:
    """The resolved references: harmful compliance for the first ``harmful[arm]`` items of each
    arm (in id order), refusal elsewhere."""
    out, seen = {}, dict.fromkeys(harmful, 0)
    for item in sorted(key):
        arm = key[item]["arm"]
        out[item] = HARMFUL if seen[arm] < harmful[arm] else REFUSAL
        seen[arm] += 1
    return out


def drawn_and_agreed(audit: Audit) -> dict:
    audit.run("draw")
    key = audit.key()
    gpt, grok = labellers(key)
    audit.run("agreements", "--gpt", audit.labels("gpt", gpt), "--grok", audit.labels("grok", grok))
    return key


def test_estimate_needs_every_item(audit) -> None:
    key = drawn_and_agreed(audit)
    reference = resolved(key, {"A": 30, "F": 30, "P": 30})
    del reference[sorted(reference)[3]]
    assert "every item" in audit.refused("estimate", "--reference",
                                         audit.labels("reference", reference))  # fmt: skip
    assert not (audit.folder / "estimate.json").exists()


def test_estimate_counts_a_missing_reference_and_leaves_few_harmful_undetermined(audit) -> None:
    key = drawn_and_agreed(audit)
    reference = resolved(key, {"A": 30, "F": 30, "P": 19})
    gone = sorted(reference)[2]
    reference[gone] = None
    out = audit.run("estimate", "--reference", audit.labels("reference", reference))
    record = json.loads((audit.folder / "estimate.json").read_text())
    assert record["missing"] == {"items": 1, "draws": key[gone]["multiplicity"]}
    rates = record["error_rates"]
    assert rates["P"]["undetermined"] is True and rates["P"]["sensitivity"]["mar"] is None
    assert rates["A"]["undetermined"] is False and rates["A"]["harmful_by_reference"] >= 20
    assert set(record["bias"]) == {"AF", "PF"}
    assert record["selection"]["sha256"] == sha256(audit.folder / "selection.json")
    assert record["agreements"]["sha256"] == sha256(audit.folder / "agreements.json")
    text = (audit.folder / "estimate.json").read_text().lower()
    assert "clopper" not in text and "reference_harm" not in text
    assert SENTINEL not in out and "undetermined" in out
    assert "written once" in audit.refused("estimate", "--reference",
                                           audit.labels("reference", reference))  # fmt: skip


def test_estimate_refuses_files_the_draw_did_not_write(audit) -> None:
    key = drawn_and_agreed(audit)
    reference = resolved(key, {"A": 30, "F": 30, "P": 30})
    frame = audit.folder / "frame.jsonl"
    frame.write_text(frame.read_text().replace('"harmful"', '"not_harmful"', 1))
    assert "frame.jsonl" in audit.refused("estimate", "--reference",
                                          audit.labels("reference", reference))  # fmt: skip


def test_estimate_waits_for_the_agreements(audit) -> None:
    audit.run("draw")
    reference = resolved(audit.key(), {"A": 30, "F": 30, "P": 30})
    assert "agreements" in audit.refused("estimate", "--reference",
                                         audit.labels("reference", reference))  # fmt: skip
