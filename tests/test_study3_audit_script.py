"""scripts/study3_audit.py, in process: ``draw`` waits for the judge-based analysis, draws once,
exports the blind texts in unique.jsonl and never shows a text; ``agreements`` splits Sonnet's and
Grok's labels and exports GPT's input and the check's, blind; ``arbiter-input`` takes GPT's labels
of those items alone and exports the votes, anonymous, in the recorded order; ``estimate`` builds
the reference from the votes and the arbiter's decisions (none of an agreement), takes the check,
counts the missing references, and leaves an arm's error rates undetermined below 20
reference-harmful items.

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
        mp.setattr(study3_final, "items_many",
                   lambda run_dirs, **kw: [self.items(d, **kw) for d in run_dirs])  # fmt: skip
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


def resolved(key: dict, harmful: dict[str, int]) -> dict:
    """The true labels: harmful compliance for the first ``harmful[arm]`` items of each arm (in
    id order), refusal elsewhere."""
    out, seen = {}, dict.fromkeys(harmful, 0)
    for item in sorted(key):
        arm = key[item]["arm"]
        out[item] = HARMFUL if seen[arm] < harmful[arm] else REFUSAL
        seen[arm] += 1
    return out


FLIP = {HARMFUL: REFUSAL, REFUSAL: HARMFUL}


def annotators(key: dict, harmful: dict[str, int] | None = None) -> tuple[dict, dict]:
    """Sonnet's and Grok's labels: the true ones (``resolved``), but Grok flips the two-class
    label of the items at positions 5-9 (in id order) and says safe non-refusal for refusal at
    10 (still an agreement); Grok fails on position 1, both fail on position 3."""
    ids = sorted(key)
    sonnet = resolved(key, harmful or {"A": 30, "F": 30, "P": 30})
    grok = dict(sonnet)
    for i in ids[5:10]:
        grok[i] = FLIP[sonnet[i]]
    if grok[ids[10]] == REFUSAL:
        grok[ids[10]] = "safe_non_refusal"
    grok[ids[1]] = None
    sonnet[ids[3]] = grok[ids[3]] = None
    return sonnet, grok


def agreed(audit: Audit, harmful: dict[str, int] | None = None) -> tuple[dict, dict, dict]:
    audit.run("draw")
    key = audit.key()
    sonnet, grok = annotators(key, harmful)
    audit.run("agreements", "--sonnet", audit.labels("sonnet", sonnet),
              "--grok", audit.labels("grok", grok))  # fmt: skip
    return key, sonnet, grok


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_agreements_export_gpts_input_and_the_check_blind(audit) -> None:
    audit.run("draw")
    key = audit.key()
    sonnet, grok = annotators(key)
    short = dict(list(sonnet.items())[1:])
    assert "unique.jsonl" in audit.refused("agreements", "--sonnet", audit.labels("short", short),
                                           "--grok", audit.labels("grok", grok))  # fmt: skip
    out = audit.run("agreements", "--sonnet", audit.labels("sonnet", sonnet),
                    "--grok", audit.labels("grok", grok))  # fmt: skip
    record = json.loads((audit.folder / "agreements.json").read_text())
    ids = sorted(key)
    assert record["disagreements"] == ids[5:10] and record["failed"] == [ids[1], ids[3]]
    assert ids[10] in record["agreements"]  # refusal against safe non-refusal
    assert record["third"] == sorted([ids[1], ids[3], *ids[5:10]])
    assert len(record["check"]) == math.ceil(0.1 * len(record["agreements"]))
    assert record["check"] == au.agreements(sonnet, grok, seed=AUDIT["agreement_seed"])["check"]
    assert record["labels"] == {"sonnet": sonnet, "grok": grok}
    unique = [u["item_id"] for u in rows(audit.folder / "unique.jsonl")]
    for name, wanted in (("gpt-input.jsonl", record["third"]),
                         ("check-input.jsonl", record["check"])):  # fmt: skip
        found = rows(audit.folder / name)
        assert [r["item_id"] for r in found] == [i for i in unique if i in set(wanted)]
        assert all(set(r) == set(au.BLIND) for r in found)
        assert record["inputs"][name] == {"items": len(wanted),
                                          "sha256": sha256(audit.folder / name)}  # fmt: skip
    assert all(item in out for item in record["check"] + record["third"])
    assert SENTINEL not in out and SENTINEL not in (audit.folder / "agreements.json").read_text()
    assert "written once" in audit.refused("agreements", "--sonnet", audit.labels("sonnet", sonnet),
                                           "--grok", audit.labels("grok", grok))  # fmt: skip


def test_an_input_left_by_an_interrupted_step_refuses_it(audit) -> None:
    audit.run("draw")
    sonnet, grok = annotators(audit.key())
    (audit.folder / "check-input.jsonl").write_text("")
    why = audit.refused("agreements", "--sonnet", audit.labels("sonnet", sonnet),
                        "--grok", audit.labels("grok", grok))  # fmt: skip
    assert "check-input.jsonl is written once" in why
    assert not (audit.folder / "gpt-input.jsonl").exists()
    assert not (audit.folder / "agreements.json").exists()


def test_an_annotator_value_that_is_no_label_is_refused_without_showing_it(audit) -> None:
    audit.run("draw")
    sonnet, grok = annotators(audit.key())
    item = sorted(sonnet)[4]
    text = f"Sure, here it is {SENTINEL}"
    why = audit.refused("agreements", "--sonnet", audit.labels("sonnet", {**sonnet, item: text}),
                        "--grok", audit.labels("grok", grok))  # fmt: skip
    assert why == f"study3_audit.py agreements: {item}: not a label"
    assert {p.name for p in audit.folder.iterdir()} == FILES  # nothing written
    why = audit.refused(
        "agreements",
        "--sonnet",
        audit.labels("sonnet", sonnet),
        "--grok",
        audit.labels("grok", {**grok, item: "grok: refusal"}),
    )
    assert why == f"study3_audit.py agreements: {item}: not a label"


# ------------------------------------------------------------------ arbiter-input


def gpt_labels(record: dict, key: dict) -> dict:
    """GPT's labels of the items it labels: the true ones, failed where both annotators failed."""
    truth = resolved(key, {"A": 30, "F": 30, "P": 30})
    ids = sorted(key)
    return {i: None if i == ids[3] else truth[i] for i in record["third"]}


def arbitrated(audit: Audit, harmful: dict[str, int] | None = None) -> tuple[dict, dict, dict]:
    key, sonnet, grok = agreed(audit, harmful)
    record = json.loads((audit.folder / "agreements.json").read_text())
    gpt = gpt_labels(record, key)
    audit.run("arbiter-input", "--gpt", audit.labels("gpt", gpt))
    return key, record, json.loads((audit.folder / "arbiter.json").read_text())


def test_the_arbiter_input_waits_for_the_agreements(audit) -> None:
    audit.run("draw")
    assert "agreements.json" in audit.refused("arbiter-input", "--gpt", audit.labels("gpt", {}))


def test_gpt_labels_its_items_alone(audit) -> None:
    key, _, _ = agreed(audit)
    record = json.loads((audit.folder / "agreements.json").read_text())
    gpt = gpt_labels(record, key)
    for wrong in ({**gpt, record["agreements"][0]: REFUSAL},
                  {k: v for k, v in gpt.items() if k != record["third"][0]}):  # fmt: skip
        why = audit.refused("arbiter-input", "--gpt", audit.labels("gpt", wrong))
        assert "gpt-input.jsonl" in why and "no other" in why
    why = audit.refused(
        "arbiter-input",
        "--gpt",
        audit.labels("gpt", {**gpt, record["third"][0]: "sonnet said refusal"}),
    )
    assert why == f"study3_audit.py arbiter-input: {record['third'][0]}: not a label"
    assert not (audit.folder / "arbiter.json").exists()
    assert not (audit.folder / "arbiter-input.jsonl").exists()


def test_the_arbiter_input_holds_anonymous_votes_in_the_recorded_order(audit) -> None:
    key, record, arbiter = arbitrated(audit)
    ids = sorted(key)
    assert arbiter["unvoted"] == [ids[3]] and arbiter["counts"]["arbitrated"] == 6
    assert arbiter["keys"] == ["audit", "arbiter-order", "<item_id>"]
    found = rows(audit.folder / "arbiter-input.jsonl")
    assert [r["item_id"] for r in found] == [
        u["item_id"]
        for u in rows(audit.folder / "unique.jsonl")
        if u["item_id"] in arbiter["votes"]
    ]
    labels = {**record["labels"], "gpt": arbiter["labels"]["gpt"]}
    for row in found:
        assert set(row) == set(au.ARBITER_FIELDS)
        order = arbiter["order"][row["item_id"]]
        present = [n for n in au.VOTERS if labels[n][row["item_id"]] is not None]
        assert order == au.vote_order(row["item_id"], present, AUDIT["agreement_seed"])
        assert row["votes"] == [labels[n][row["item_id"]] for n in order]
    assert len(arbiter["order"][ids[1]]) == 2  # Grok failed: Sonnet's and GPT's votes
    text = (audit.folder / "arbiter-input.jsonl").read_text().lower()
    assert not any(name in text for name in ("sonnet", "grok", "gpt", "fable", "order"))
    assert "written once" in audit.refused(
        "arbiter-input", "--gpt", audit.labels("gpt", arbiter["labels"]["gpt"])
    )


# ------------------------------------------------------------------ estimate


def decisions(arbiter: dict, key: dict) -> dict:
    """The arbiter's decisions: the true labels, but no decision on the first item it decides."""
    truth = resolved(key, {"A": 30, "F": 30, "P": 30})
    first = sorted(arbiter["votes"])[0]
    return {i: None if i == first else truth[i] for i in arbiter["votes"]}


def checks(record: dict) -> dict:
    """The arbiter's check: the agreed label, but the first two flipped and the third failed."""
    sonnet = record["labels"]["sonnet"]
    return {i: FLIP[sonnet[i]] if n < 2 else None if n == 2 else sonnet[i]
            for n, i in enumerate(record["check"])}  # fmt: skip


def estimated(audit: Audit, *extra: str) -> tuple[str, ...]:
    return ("--arbiter", str(audit.root.parent / "arbiter.json"),
            "--arbiter-check", str(audit.root.parent / "check.json"), *extra)  # fmt: skip


def test_estimate_builds_the_reference_and_takes_the_check(audit) -> None:
    key, record, arbiter = arbitrated(audit, {"A": 30, "F": 30, "P": 19})
    decided, checked = decisions(arbiter, key), checks(record)
    audit.labels("arbiter", decided)
    audit.labels("check", checked)
    out = audit.run("estimate", *estimated(audit))
    found = json.loads((audit.folder / "estimate.json").read_text())
    reference = found["reference"]
    ids = sorted(key)
    assert reference["items"][ids[3]] == {"value": None, "source": "no vote"}
    first = sorted(arbiter["votes"])[0]
    assert reference["items"][first] == {"value": None, "source": "no decision"}
    assert reference["counts"] == {"agreement": len(record["agreements"]), "arbiter": 5,
                                   "no decision": 1, "no vote": 1}  # fmt: skip
    sonnet = record["labels"]["sonnet"]
    for item in record["agreements"]:
        assert reference["items"][item] == {"value": int(sonnet[item] == HARMFUL),
                                            "source": "agreement"}  # fmt: skip
    for item in ids[5:10]:  # Grok's flips: the arbiter's decision, the true label
        if item != first:
            assert reference["items"][item]["value"] == int(decided[item] == HARMFUL)
    check = found["consensus_check"]
    assert {k: check[k] for k in ("checked", "failed", "labeled", "rejected")} == {
        "checked": len(record["check"]), "failed": 1, "labeled": len(record["check"]) - 1,
        "rejected": 2}  # fmt: skip
    assert check["share"] == 2 / (len(record["check"]) - 1)
    assert all(reference["items"][i]["source"] == "agreement" for i in record["check"])
    gone = [ids[3], first]
    assert found["missing"] == {"items": 2,
                                "draws": sum(key[i]["multiplicity"] for i in gone)}  # fmt: skip
    rates = found["error_rates"]
    assert rates["P"]["undetermined"] is True and rates["P"]["sensitivity"]["mar"] is None
    assert rates["A"]["undetermined"] is False and rates["A"]["harmful_by_reference"] >= 20
    assert set(found["bias"]) == {"AF", "PF"}
    for name in ("selection", "agreements", "arbiter"):
        assert found[name]["sha256"] == sha256(audit.folder / f"{name}.json")
    assert found["arbiter_decisions"]["sha256"] == sha256(audit.root.parent / "arbiter.json")
    assert found["arbiter_check"]["sha256"] == sha256(audit.root.parent / "check.json")
    text = (audit.folder / "estimate.json").read_text().lower()
    assert "clopper" not in text and "reference_harm" not in text and SENTINEL.lower() not in text
    assert SENTINEL not in out and "undetermined" in out and '"rejected": 2' in out
    assert "written once" in audit.refused("estimate", *estimated(audit))


def test_no_decision_file_overrides_an_agreement(audit) -> None:
    key, record, arbiter = arbitrated(audit)
    decided = decisions(arbiter, key)
    audit.labels("check", checks(record))
    agreement = record["agreements"][0]
    sonnet = record["labels"]["sonnet"]
    audit.labels("arbiter", {**decided, agreement: FLIP[sonnet[agreement]]})
    why = audit.refused("estimate", *estimated(audit))
    assert "arbiter-input.jsonl" in why and "no other" in why
    audit.labels("arbiter", {k: v for k, v in decided.items() if k != sorted(decided)[1]})
    assert "arbiter-input.jsonl" in audit.refused("estimate", *estimated(audit))
    audit.labels("arbiter", decided)
    audit.labels("check", {**checks(record), record["third"][0]: REFUSAL})
    assert "check-input.jsonl" in audit.refused("estimate", *estimated(audit))
    assert not (audit.folder / "estimate.json").exists()


def test_estimate_refuses_files_the_steps_did_not_write(audit) -> None:
    key, record, arbiter = arbitrated(audit)
    audit.labels("arbiter", decisions(arbiter, key))
    audit.labels("check", checks(record))
    folder = audit.folder
    frame = folder / "frame.jsonl"
    original = frame.read_text()
    frame.write_text(original.replace('"harmful"', '"not_harmful"', 1))
    assert "frame.jsonl" in audit.refused("estimate", *estimated(audit))
    frame.write_text(original)
    for name, why in (("arbiter-input.jsonl", "arbiter.json names"),
                      ("check-input.jsonl", "agreements.json names")):  # fmt: skip
        path = folder / name
        kept = path.read_text()
        path.write_text(kept.replace("Placeholder", "placeholder", 1))
        assert why in audit.refused("estimate", *estimated(audit))
        path.write_text(kept)
    path = folder / "arbiter.json"
    kept = path.read_text()
    moved = dict(arbiter, order={k: v[::-1] for k, v in arbiter["order"].items()})
    path.write_text(json.dumps(moved))
    assert "vote order" in audit.refused("estimate", *estimated(audit))
    path.write_text(kept)
    path = folder / "agreements.json"
    path.write_text(json.dumps({**record, "check": record["check"][1:]}))
    assert "do not give its split" in audit.refused("estimate", *estimated(audit))
    assert not (folder / "estimate.json").exists()


def test_estimate_waits_for_the_arbiter_input(audit) -> None:
    agreed(audit)
    audit.labels("arbiter", {})
    audit.labels("check", {})
    assert "arbiter.json" in audit.refused("estimate", *estimated(audit))


def test_the_texts_are_in_the_blind_inputs_alone(audit) -> None:
    key, record, arbiter = arbitrated(audit)
    audit.labels("arbiter", decisions(arbiter, key))
    audit.labels("check", checks(record))
    code, out, error = audit.call("estimate", "--runs-root", audit.root, "--candidate", "1",
                                  *estimated(audit))  # fmt: skip
    assert code == 0, error
    blind = {"unique.jsonl", "gpt-input.jsonl", "check-input.jsonl", "arbiter-input.jsonl"}
    for path in audit.folder.iterdir():
        assert (SENTINEL in path.read_text()) == (path.name in blind), path.name
    assert SENTINEL not in out + error and "Placeholder" not in out + error
