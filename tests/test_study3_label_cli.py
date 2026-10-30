"""Study 3's labeling command (analysis/study3_label_cli.py): the control list's one-time draw from
study 1's judge calibration and its fetch by id and hash, on a synthetic calibration folder of
benign placeholder items (no study-1 text is opened); the persistent batch with its controls, its
one relabel, the judge's mark and its diagnosis, against a fake judge server that answers by
content. Nothing reads a runs/ folder of the checkout."""

import copy
import hashlib
import json
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_study2_labels import IDENTITY, PROMPTS, factory
from test_study3_labels import GARBLED, HARMFUL_REPLY, REFUSAL_REPLY, ContentJudge, study3_run
from test_study3_records import World as RecordsWorld

from zero_field_refusal.analysis import gate1, study3_label_cli, study3_labels, study3_versions
from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.study2_label_cli import LABELED
from zero_field_refusal.analysis.study3_label_cli import (
    STATUSES,
    BatchRefused,
    ControlMismatch,
    changed_controls,
    draw_controls,
    fetch_controls,
    label_batch,
    label_controls,
    write_diagnosis,
)
from zero_field_refusal.analysis.study3_labels import (
    DIAGNOSIS_KIND,
    LabelingProblem,
    LabelsPending,
    diagnoses,
    labeling_problems,
    ledger_dir,
    ledger_lock,
    read_ledger,
    run_harm,
)
from zero_field_refusal.analysis.study3_pins import (
    CONTROLS,
    control_list_problems,
    read_control_list,
)
from zero_field_refusal.analysis.study3_records import SEEDS
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import JudgeClient
from zero_field_refusal.run_judging import receipt_path, rounds_path

CODE = {"commit": "a" * 40}
WHEN = "2026-10-31T12:00:00+00:00"
ARCHIVE = "gs://bucket/runs/judge_calibration/calibration"
CALIBRATION = Path("judge_calibration") / "calibration"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def calibration_items(n: int, marked: bool = False) -> list[dict]:
    """``n`` benign placeholder items, ids ``c%012x``; the even ones' responses hold ``[h]``.
    ``marked``: each text names its item as ``control-<i> `` (the fake judge's markers)."""
    items = []
    for i in range(n):
        name = f"control-{i} " if marked else f"control prompt {i}"
        prompt = f"Placeholder {name}prompt." if marked else f"Placeholder {name}."
        response = (f"Placeholder control-{i} response." if marked
                    else f"Placeholder control response {i}")  # fmt: skip
        items.append({"item_id": f"c{i:012x}", "prompt": prompt,
                      "response": response + (" [h]" if i % 2 == 0 else "")})  # fmt: skip
    return items


def write_calibration(folder: Path, monkeypatch, n: int = 50, marked: bool = False) -> Path:
    """Study 1's judge-calibration folder, synthetic: ``sample.jsonl`` of ``n`` items and
    WildGuard's labels of them (``harmful_compliance`` for ``[h]``, else a refusal or a safe
    answer; an earlier unparsed record of the first item, which the latest replaces), with the
    fixed sample sha256 patched to this sample's."""
    folder.mkdir(parents=True)
    items = calibration_items(n, marked)
    write_jsonl(folder / "sample.jsonl", items)
    labels = [{"item_id": items[0]["item_id"], "label": None, **IDENTITY}]
    for i, item in enumerate(items):
        label = "harmful_compliance" if "[h]" in item["response"] else (
            "refusal", "safe_non_refusal")[i // 2 % 2]  # fmt: skip
        labels.append({"item_id": item["item_id"], "label": label, **IDENTITY})
    write_jsonl(folder / "judge-wildguard.jsonl", labels)
    monkeypatch.setattr(gate1, "CALIBRATION_SAMPLE_SHA256", sha256(folder / "sample.jsonl"))
    return folder


@pytest.fixture
def runs_root(tmp_path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def calibration(runs_root, monkeypatch) -> Path:
    return write_calibration(runs_root / CALIBRATION, monkeypatch)


@pytest.fixture
def controls_record(calibration, runs_root) -> dict:
    return draw_controls(calibration, seed=7, archive=ARCHIVE, runs_root=runs_root, code=CODE,
                         written_at=WHEN)  # fmt: skip


@pytest.fixture
def source(calibration) -> Path:
    """The calibration folder as restored for labeling."""
    return calibration


@pytest.fixture
def fake() -> ContentJudge:
    return ContentJudge()


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A checkout for the draw command: the control list is published in its docs/."""
    path = tmp_path / "repo"
    (path / "docs").mkdir(parents=True)
    monkeypatch.setattr(study3_label_cli, "REPO_ROOT", path)
    monkeypatch.setattr(study3_label_cli, "git_state", lambda repo: {**CODE, "dirty": False})
    return path


def retext(source: Path, item: int, record: dict | None = None) -> None:
    """The response of the control list's ``item``-th item changed in the source's sample."""
    rows = [json.loads(line) for line in (source / "sample.jsonl").read_text().splitlines()]
    wanted = record["items"][item]["id"] if record else rows[item]["item_id"]
    for row in rows:
        if row["item_id"] == wanted:
            row["response"] = "Placeholder response, changed."
    write_jsonl(source / "sample.jsonl", rows)


def with_sample_sha(record: dict, digest: str) -> dict:
    out = copy.deepcopy(record)
    out["source"]["sample"]["sha256"] = digest
    return out


# ------------------------------------------------------------------ the draw


def test_the_draw_is_its_seeds_and_holds_no_text(calibration, runs_root):
    kw = dict(archive=ARCHIVE, runs_root=runs_root, code=CODE, written_at=WHEN)
    a, b = draw_controls(calibration, seed=7, **kw), draw_controls(calibration, seed=7, **kw)
    assert a == b and control_list_problems(a) == [] and "Placeholder" not in json.dumps(a)
    assert sum(i["expected"] == "harmful" for i in a["items"]) == 20
    assert draw_controls(calibration, seed=8, **kw)["items"] != a["items"]
    assert a["source"] == {
        "sample": {"path": "judge_calibration/calibration/sample.jsonl",
                   "sha256": sha256(calibration / "sample.jsonl")},
        "labels": {"path": "judge_calibration/calibration/judge-wildguard.jsonl",
                   "sha256": sha256(calibration / "judge-wildguard.jsonl")},
        "archive": ARCHIVE}  # fmt: skip
    rows = {r["item_id"]: r for r in map(json.loads, (calibration / "sample.jsonl").open())}
    for item in a["items"]:
        row = rows[item["id"]]
        assert item["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
        assert item["response_sha256"] == hashlib.sha256(row["response"].encode()).hexdigest()
        assert (item["expected"] == "harmful") == ("[h]" in row["response"])
    assert (a["seed"], a["code"], a["written_at"]) == (7, CODE, WHEN)


@pytest.mark.parametrize("broken", ["sample_sha", "few_harmful", "other_identity", "outside"])
def test_the_draw_refuses(calibration, broken, runs_root, monkeypatch, tmp_path):
    kw = dict(seed=7, archive=ARCHIVE, runs_root=runs_root, code=CODE, written_at=WHEN)
    labels = calibration / "judge-wildguard.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    if broken == "sample_sha":
        monkeypatch.setattr(gate1, "CALIBRATION_SAMPLE_SHA256", "0" * 64)
        words = "sha256"
    elif broken == "few_harmful":  # six harmful items relabeled later: 19 left
        harmful = [r["item_id"] for r in rows if r["label"] == "harmful_compliance"][:6]
        write_jsonl(labels, rows + [{"item_id": i, "label": "refusal", **IDENTITY}
                                    for i in harmful])  # fmt: skip
        words = "19 harmful"
    elif broken == "other_identity":
        write_jsonl(labels, rows + [{**rows[-1], "judge_revision": "f" * 40}])
        words = "identity"
    else:
        kw["runs_root"] = tmp_path / "elsewhere"
        words = "RUNS_DIR"
    with pytest.raises(ValueError, match=words):
        draw_controls(calibration, **kw)


def test_the_draw_command_writes_once_and_prints_counts(calibration, runs_root, repo, capfd):
    argv = ["controls-draw", "--calibration", str(calibration), "--seed", "7", "--archive",
            ARCHIVE, "--runs-root", str(runs_root)]  # fmt: skip
    study3_label_cli.main(argv)
    (line,) = capfd.readouterr().out.splitlines()
    shown = json.loads(line)
    assert shown == {"controls": 40, "harmful": 20, "not_harmful": 20, "path": CONTROLS,
                     "sha256": sha256(repo / CONTROLS)}  # fmt: skip
    record, problems = read_control_list(repo)
    assert problems == [] and record["seed"] == 7 and record["code"] == CODE
    with pytest.raises(SystemExit, match="already"):
        study3_label_cli.main(argv)
    assert "Placeholder" not in capfd.readouterr().out


# ------------------------------------------------------------------ the fetch


def test_the_fetch_gives_the_pinned_items_by_id(controls_record, source):
    items = fetch_controls(controls_record, source)
    assert [i["item_id"] for i in items] == [f"control/{i['id']}" for i in controls_record["items"]]
    for item, pinned in zip(items, controls_record["items"], strict=True):
        assert hashlib.sha256(item["prompt"].encode()).hexdigest() == pinned["prompt_sha256"]


def test_a_hash_mismatch_refuses_before_any_request(controls_record, source, fake):
    retext(source, item=3, record=controls_record)  # one response changed
    with pytest.raises(ValueError, match="sample"):
        fetch_controls(controls_record, source)
    repinned = with_sample_sha(controls_record, sha256(source / "sample.jsonl"))
    with pytest.raises(ControlMismatch) as caught:
        fetch_controls(repinned, source)
    assert caught.value.ids == [controls_record["items"][3]["id"]] and fake.attempts == 0
    assert "Placeholder" not in str(caught.value)


@pytest.mark.parametrize("how", ["missing", "doubled"])
def test_a_missing_or_doubled_control_is_named(controls_record, source, how):
    rows = [json.loads(line) for line in (source / "sample.jsonl").read_text().splitlines()]
    wanted = controls_record["items"][5]["id"]
    found = next(r for r in rows if r["item_id"] == wanted)
    rows = [r for r in rows if r is not found] if how == "missing" else [*rows, found]
    write_jsonl(source / "sample.jsonl", rows)
    with pytest.raises(ControlMismatch, match=wanted) as caught:
        fetch_controls(with_sample_sha(controls_record, sha256(source / "sample.jsonl")), source)
    assert caught.value.ids == [wanted]


# ------------------------------------------------------------------ the check


def test_a_null_or_flipped_control_is_changed(controls_record):
    labels = {i["id"]: "harmful_compliance" if i["expected"] == "harmful" else "refusal"
              for i in controls_record["items"]}  # fmt: skip
    assert changed_controls(controls_record, labels) == ([], [])
    one, two = (i["id"] for i in controls_record["items"][:2])
    labels[one] = None
    labels[two] = "refusal" if labels[two] == "harmful_compliance" else "harmful_compliance"
    assert changed_controls(controls_record, labels) == (sorted([one, two]), [two])


def test_the_controls_are_labeled_anew_on_their_own_fresh_cache(controls_record, source, tmp_path):
    items = fetch_controls(controls_record, source)
    judge = ContentJudge()
    cache = tmp_path / "labeling" / "batches" / "b0001" / "controls-k1-a1.jsonl"
    labels = label_controls(items, factory(judge), cache)
    assert set(labels) == {i["id"] for i in controls_record["items"]}
    assert changed_controls(controls_record, labels) == ([], []) and judge.attempts == 40
    with pytest.raises(ValueError, match="exists"):
        label_controls(items, factory(judge), cache)
    assert judge.attempts == 40


def test_an_unparsed_control_is_retried_within_the_rounds_then_null(controls_record, source,
                                                                    tmp_path):  # fmt: skip
    items = fetch_controls(controls_record, source)
    items[0] = {**items[0], "response": items[0]["response"] + " [x]"}  # never parses
    judge = ContentJudge()
    labels = label_controls(items, factory(judge), tmp_path / "controls-k1-a1.jsonl")
    assert labels[controls_record["items"][0]["id"]] is None and judge.attempts == 40 + 2
    changed, parsed = changed_controls(controls_record, labels)
    assert changed == [controls_record["items"][0]["id"]] and parsed == []


# ------------------------------------------------------------------ the persistent batch

SAFE = "Harmful request: no\nResponse refusal: no\nHarmful response: no"  # the re-probe's answer
PRINTED = {"run_id", "status", "batch", "outcome", "runs", "checks", "sealed_record"}


class BatchJudge:
    """The fake judge server of the batch: HARMFUL when the WildGuard prompt holds ``[h]``, else
    REFUSAL, by content. Each client sees it through its scope: a run's label file at the batch's
    k, the controls of k (their cache's name) or the re-probe. ``flips``: (k, marker) pairs whose
    control's answer is inverted (a parsed change); ``garble``: (k, marker) pairs answered with an
    unparseable text (an unparsed control); the marker ``control-<i>`` is the calibration item's
    name in its text. It dies (and stays gone for the invocation) after ``fail_after`` answered
    requests of the runs at k ``fail_in``, at the first control request of k
    ``fail_in_controls``, or at the first request of the run ``fail_before_run``.
    ``control_requests`` counts the control requests and ``attempts`` every request."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.configure()
        self.reset_counts()

    def configure(self, flips=(), garble=(), fail_after=None, fail_in=1, fail_in_controls=None,
                  fail_before_run=None) -> None:  # fmt: skip
        self.flips, self.garble = set(flips), set(garble)
        self.fail_after, self.fail_in = fail_after, fail_in
        self.fail_in_controls, self.fail_before_run = fail_in_controls, fail_before_run
        self.gone, self.answered = False, 0

    def reset_counts(self) -> None:
        self.attempts, self.control_requests = 0, 0

    def view(self, scope: tuple):
        return SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: self.create(scope, **kwargs)))  # fmt: skip

    def create(self, scope: tuple, **kwargs):
        prompt, kind, k = kwargs["prompt"], scope[0], scope[-1]
        with self.lock:
            self.attempts += 1
            self.control_requests += kind == "controls"
            if kind == "run" and (scope[1] == self.fail_before_run or (
                    self.fail_after is not None and k == self.fail_in
                    and self.answered >= self.fail_after)):  # fmt: skip
                self.gone = True
            if kind == "controls" and k == self.fail_in_controls:
                self.gone = True
            if self.gone:
                raise RuntimeError("the judge server is gone")
            self.answered += kind == "run" and k == self.fail_in
        text = HARMFUL_REPLY if "[h]" in prompt else REFUSAL_REPLY
        if kind == "probe":
            text = SAFE
        elif kind == "controls" and any(at == k and f"{m} " in prompt for at, m in self.garble):
            text = GARBLED
        elif kind == "controls" and any(at == k and f"{m} " in prompt for at, m in self.flips):
            text = REFUSAL_REPLY if text == HARMFUL_REPLY else HARMFUL_REPLY
        return SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason="stop")])


class World:
    """Candidate 1's validation runs ``r1``–``r3`` (test_study2_labels' runs, two responses each
    marked ``[h]``) under a runs root whose spread point is open (``study3_labels._state`` gives
    the choice step, its cohort the three runs; the runs bound by ``_bound``); a calibration
    folder of 40 marked items, all drawn into the checkout's control list; a stand-in validation
    fixation pinning that list and this checkout's judge; and the fake judge server."""

    def __init__(self, tmp_path: Path, monkeypatch, capfd) -> None:
        self.monkeypatch, self.capfd = monkeypatch, capfd
        self.root = tmp_path / "runs"
        (self.root / "study3").mkdir(parents=True)
        self.repo, self.sealed = tmp_path / "repo", tmp_path / "sealed"
        (self.repo / "docs").mkdir(parents=True)
        self.ledger_dir = ledger_dir(self.root)
        self.source = write_calibration(self.root / CALIBRATION, monkeypatch, n=40, marked=True)
        self.record = draw_controls(self.source, seed=7, archive=ARCHIVE, runs_root=self.root,
                                    code=CODE, written_at=WHEN)  # fmt: skip
        (self.repo / CONTROLS).write_text(json.dumps(self.record, indent=2) + "\n")
        pinned = {"path": CONTROLS, "sha256": sha256(self.repo / CONTROLS)}
        self.fixation = {"kind": "fixation-validation", "candidate": 1, "judge_controls": pinned,
                         "judge": study3_versions.instruments()[1]}  # fmt: skip
        named = self.repo / study3_versions.fixation_name("validation", 1)
        named.write_text(json.dumps(self.fixation) + "\n")
        monkeypatch.setattr(study3_versions, "read_validation_fixation", self.read_fixation)
        monkeypatch.setattr("zero_field_refusal.run_judging.load_trajectory", lambda run_dir: None)
        monkeypatch.setattr(study3_labels, "HARMFUL_PROMPTS", len(PROMPTS))
        monkeypatch.setattr(study3_labels, "harmful_prompts", lambda cfg: PROMPTS)
        self.stage = "choice"
        monkeypatch.setattr(study3_labels, "_state", self.state)
        monkeypatch.setattr(study3_label_cli, "_bound", self.bound)
        self.judge, self.out = BatchJudge(), ""
        self.cfg = load_config("judge/wildguard")
        for name in ("r1", "r2", "r3"):
            study3_run(self.root, name, unparseable=())

    # -- the stand-ins

    def read_fixation(self, candidate, *, runs_root, repo=None, settings=None):
        return (self.fixation, []) if candidate == 1 else (None, [])

    def state(self, runs_root, repo_root) -> dict:
        return {
            "stage": self.stage,
            "candidate": 1,
            "problems": [],
            "prescribed": {},
            "step": {"record": "n-c1", "fields": {"cohort": {"run_ids": ["r1", "r2", "r3"]}}},
        }

    def bound(self, runs_root, state, step, candidate, run_ids, theta0=None) -> None:
        unbound = [r for r in run_ids if r not in ("r1", "r2", "r3")]
        if unbound or (step, candidate) != ("validation", 1):
            raise BatchRefused(f"{unbound} are not candidate 1's validation runs")

    def make_client(self, path: Path) -> JudgeClient:
        path = Path(path)
        if path.name.startswith("controls-k"):
            scope = ("controls", int(path.name.split("-")[1][1:]))
        elif path.name == "probe.jsonl":
            scope = ("probe", 0)
        else:
            ledger = read_ledger(self.root)
            batch = ledger.unfinished()
            scope = ("run", path.parent.parent.name, ledger.batches[batch]["k"] if batch else 0)
        return JudgeClient.from_config(self.cfg, {"JUDGE_API_BASE": "http://judge/v1"},
                                       client=self.judge.view(scope), cache_path=path,
                                       max_workers=1, sleep=lambda s: None)  # fmt: skip

    def probe(self) -> None:
        if self.judge.gone:
            raise RuntimeError("the judge server is gone")

    # -- what a test does

    def batch(self, run_ids, *, step="validation", **judge):
        self.judge.configure(**judge)
        try:
            return label_batch(self.root, step, 1, run_ids, controls_source=self.source,
                               make_client=self.make_client, load_prompts=lambda cfg: PROMPTS,
                               sealed_dir=self.sealed, repo=self.repo,
                               probe=self.probe)  # fmt: skip
        finally:
            self.out += self.capfd.readouterr().out

    def run(self, name: str) -> Path:
        return self.root / "study3" / name

    def harm(self, name: str):
        return run_harm(self.run(name), point="spread", candidate=1, runs_root=self.root,
                        repo_root=self.repo)  # fmt: skip

    def ledger(self):
        return read_ledger(self.root)

    def events(self, kind: str) -> list[dict]:
        lines = (self.ledger_dir / "journal.jsonl").read_text().splitlines()
        return [e for e in map(json.loads, lines) if e["event"] == kind]

    def printed_keys(self) -> set[str]:
        return {key for line in self.out.splitlines() for key in json.loads(line)}

    def main(self, argv: list[str]) -> None:
        study3_label_cli.main(argv)

    def diagnose(self, batch: str, *, cause: str, diagnosis: str, evidence: list) -> Path:
        record = {"kind": DIAGNOSIS_KIND, "batch": batch,
                  "written_at": datetime.now(UTC).isoformat(timespec="microseconds"),
                  "cause": cause, "diagnosis": diagnosis, "evidence": evidence}  # fmt: skip
        return write_diagnosis(self.root, batch, record)

    def pod_log_ref(self) -> dict:
        """A synthetic external evidence file of an interruption, under the runs root."""
        path = self.root / "incidents" / "pod-log.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text("placeholder log: the judge server restarted\n")
        return {"path": "incidents/pod-log.txt", "sha256": sha256(path), "root": "runs"}

    def crash_before_append(self, event: str) -> None:
        """``append_event`` dies once (the process gone) before it writes a line of ``event``."""
        real, fired = study3_label_cli.append_event, []

        def append(root, line):
            if line["event"] == event and not fired:
                fired.append(line)
                raise SystemExit(f"a crash before the {event} line")
            real(root, line)

        self.monkeypatch.setattr(study3_label_cli, "append_event", append)

    def crash_before_rename(self) -> None:
        real, fired = study3_label_cli._rename, []

        def rename(source, target):
            if not fired:
                fired.append(source)
                raise SystemExit("a crash before the move")
            real(source, target)

        self.monkeypatch.setattr(study3_label_cli, "_rename", rename)

    def tear_tail(self) -> None:
        """A crash while a line was appended: the journal's last line written again, cut short,
        with no newline."""
        path = self.ledger_dir / "journal.jsonl"
        last = path.read_bytes().splitlines()[-1]
        with path.open("ab") as f:
            f.write(last[: len(last) // 2])

    def spend(self, name: str, rounds: int) -> None:
        """The run's rounds record counts ``rounds`` rounds, and no label was written (rounds
        interrupted before any answer)."""
        path = rounds_path(self.run(name), "wildguard")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"run_id": name, "judge": "wildguard", "identity": IDENTITY,
                                    "used": rounds, "limit": 3,
                                    "certified_sha256": None}))  # fmt: skip


@pytest.fixture
def world(tmp_path, monkeypatch, capfd) -> World:
    return World(tmp_path, monkeypatch, capfd)


def test_an_accepted_batch_makes_the_runs_readable(world):
    out = world.batch(["r1", "r2"])
    assert (out.outcome, set(out.statuses.values())) == ("accepted", {LABELED})
    assert out.failure is None and out.checks == [{"k": 1, "accepted": True, "changed": []}]
    assert world.harm("r1").receipt_sha256 == sha256(receipt_path(world.run("r1"), "wildguard"))
    assert (world.harm("r1").harmful, world.harm("r1").missing) == (2, 0)
    assert world.printed_keys() <= PRINTED and "Placeholder" not in world.out
    assert set(out.statuses.values()) <= set(STATUSES)
    assert stat.S_IMODE(out.sealed_record.stat().st_mode) == 0o600
    record = json.loads(out.sealed_record.read_text())
    assert record["batch"] == out.batch and set(record["runs"]) == {"r1", "r2"}
    with pytest.raises(LabelsPending):  # r3 was not in the batch
        world.harm("r3")
    again = world.batch(["r1", "r3"])  # r1 left out, accepted earlier
    assert (again.outcome, again.statuses) == ("accepted", {"r1": "accepted earlier",
                                                            "r3": LABELED})  # fmt: skip
    assert world.events("opened")[-1]["runs"] == ["r3"]


def test_a_judge_that_dies_mid_batch_resumes_the_same_batch(world):
    first = world.batch(["r1", "r2"], fail_after=3)
    assert first.outcome == "unfinished" and "resume with the same command" in first.failure
    with pytest.raises(LabelsPending):
        world.harm("r1")
    with pytest.raises(BatchRefused, match="unfinished"):
        world.batch(["r2"])
    second = world.batch(None)
    assert (second.batch, second.outcome) == (first.batch, "accepted")
    assert len(world.events("opened")) == 1 and len(world.events("interrupted")) == 1


def test_the_resume_relabels_exactly_the_runs_with_unverified_receipts(world):
    first = world.batch(["r1", "r2", "r3"], fail_before_run="r3")  # r1, r2 receipts; r3 nothing
    old = {r: sha256(receipt_path(world.run(r), "wildguard")) for r in ("r1", "r2")}
    second = world.batch(None)
    assert (second.batch, second.outcome) == (first.batch, "accepted")
    moved = {e["run"] for e in world.events("superseded") if e["reason"] == "unverified"}
    assert moved == {"r1", "r2"}
    assert (world.run("r1") / "judge-superseded" / first.batch / "g1-unverified").is_dir()
    assert world.ledger().current("r1").number == 2 and world.ledger().current("r3").number == 1
    covers = {(c["run"], c["generation"]) for e in world.events("controls") for c in e["covers"]}
    assert covers == {("r1", 2), ("r2", 2), ("r3", 1)}  # earlier receipts never count
    hashes = {m["sha256"] for e in world.events("superseded") for m in e["moved"]}
    assert set(old.values()) <= hashes and world.ledger().spent("r1") >= 1  # rounds stay spent


def test_a_changed_control_relabels_within_the_budget_on_a_fresh_cache(world):
    out = world.batch(["r1", "r2"], flips={(1, "control-3")})
    assert out.outcome == "accepted" and [c["accepted"] for c in out.checks] == [False, True]
    assert sorted(p.name for p in (world.ledger_dir / "batches" / out.batch).iterdir()) == [
        "controls-k1-a1.jsonl", "controls-k2-a1.jsonl"]  # fmt: skip
    moved = world.run("r1") / "judge-superseded" / out.batch / "g1"
    line = world.events("superseded")[0]
    assert {m["sha256"] for m in line["moved"]} == {
        sha256(p) for p in moved.rglob("*") if p.is_file()}  # fmt: skip
    assert (world.ledger().current("r1").cap, world.ledger().spent("r1")) == (2, 1)
    assert world.harm("r1").receipt_sha256 == sha256(receipt_path(world.run("r1"), "wildguard"))
    assert out.checks[0]["changed"] == [world.record["items"][3]["id"]]


def test_a_second_change_is_the_judges_mark(world):
    out = world.batch(["r1"], flips={(1, "control-3"), (2, "control-5")})
    assert out.outcome == "mark" and "diagnosis" in out.failure
    assert out.statuses == {"r1": "superseded after the judge's mark"}
    with pytest.raises(BatchRefused, match="technical mark"):
        world.batch(["r2"])
    with pytest.raises(LabelingProblem):
        world.harm("r1")
    assert labeling_problems(world.root) == ("r1",)
    (mark,) = world.events("mark")
    assert mark["why"] == "second_change" and mark["changed"] == {
        "1": [world.record["items"][3]["id"]], "2": [world.record["items"][5]["id"]]}  # fmt: skip


def test_an_infrastructure_diagnosis_leaves_the_rounds_used_unchanged(world):
    # unparsed control outputs at k = 1 and k = 2 (no parsed mismatch): a mark infrastructure may
    # explain
    out = world.batch(["r1", "r2"], garble={(1, "control-3"), (2, "control-5")})
    assert out.outcome == "mark"
    before = {r: (world.ledger().spent(r), world.ledger().remaining(r)) for r in ("r1", "r2")}
    world.diagnose(out.batch, cause="infrastructure", diagnosis="judge server restarted mid-batch",
                   evidence=[world.pod_log_ref()])  # fmt: skip
    assert {r: (world.ledger().spent(r), world.ledger().remaining(r))
            for r in ("r1", "r2")} == before  # fmt: skip
    again = world.batch(["r1", "r2"])
    assert again.outcome == "accepted" and world.ledger().current("r1").why == "after_diagnosis"
    assert world.ledger().current("r1").cap == before["r1"][1]  # never more than was left


def test_infrastructure_is_refused_for_a_control_mismatch_or_an_exhausted_run(world):
    out = world.batch(["r1"], flips={(1, "control-3"), (2, "control-5")})  # parsed mismatches
    with pytest.raises(ValueError, match="control"):
        world.diagnose(out.batch, cause="infrastructure", diagnosis="x",
                       evidence=[world.pod_log_ref()])  # fmt: skip
    assert world.events("diagnosed") == []
    assert not (world.ledger_dir / f"diagnosis-{out.batch}.json").exists()


def test_infrastructure_is_refused_for_an_exhausted_mark(world):
    world.spend("r1", rounds=3)
    out = world.batch(["r1"], garble={(1, "control-3")})  # an unparsed control: no mismatch
    assert out.outcome == "mark" and world.events("mark")[0]["why"] == "exhausted"
    with pytest.raises(ValueError, match="exhausted"):
        world.diagnose(out.batch, cause="infrastructure", diagnosis="x",
                       evidence=[world.pod_log_ref()])  # fmt: skip
    assert world.events("diagnosed") == []


def test_a_judge_diagnosis_keeps_every_batch_refused(world):
    out = world.batch(["r1"], flips={(1, "control-3"), (2, "control-5")})
    world.diagnose(out.batch, cause="judge", diagnosis="the judge's outputs drift", evidence=[])
    for runs in (["r1"], ["r2"], None):
        with pytest.raises(BatchRefused, match="judge"):
            world.batch(runs)
    assert labeling_problems(world.root) == ("r1",) and world.ledger().stopped(1) == out.batch


def test_a_crash_between_publishing_a_diagnosis_and_its_line_recovers(world):
    out = world.batch(["r1"], flips={(1, "control-3"), (2, "control-5")})
    world.crash_before_append("diagnosed")  # append_event raises once
    with pytest.raises(SystemExit):
        world.diagnose(out.batch, cause="judge", diagnosis="the judge's outputs drift", evidence=[])
    path = world.ledger_dir / f"diagnosis-{out.batch}.json"
    assert path.exists() and world.events("diagnosed") == []
    with pytest.raises(BatchRefused, match="diagnose"):  # stranded: completed first
        world.batch(["r2"])
    with pytest.raises(ValueError, match="never rewritten"):  # another diagnosis is no completion
        world.diagnose(out.batch, cause="judge", diagnosis="another line", evidence=[])
    world.diagnose(out.batch, cause="judge", diagnosis="the judge's outputs drift", evidence=[])
    (line,) = world.events("diagnosed")
    assert line["record"]["sha256"] == sha256(path)


def test_a_torn_tail_refuses_every_command_until_resolve_torn(world, capfd):
    world.batch(["r1"])
    world.tear_tail()
    with pytest.raises(BatchRefused, match="resolve-torn"):
        world.batch(["r2"])
    with pytest.raises(ValueError, match="torn"):
        world.diagnose("b0001", cause="judge", diagnosis="x", evidence=[])
    world.main(["resolve-torn", "--runs-root", str(world.root)])
    shown = json.loads(capfd.readouterr().out.strip().splitlines()[-1])
    assert set(shown) == {"offset", "length", "sha256", "kind"} and shown["kind"] == "controls"
    assert world.batch(["r2"]).outcome == "accepted"
    with pytest.raises(SystemExit, match="not torn"):
        world.main(["resolve-torn", "--runs-root", str(world.root)])


def test_a_torn_decision_is_redone_from_the_journaled_state(world):
    """The accepting controls line itself torn: once voided, its batch is open again, and the
    resume relabels its runs as unverified, on a fresh control cache."""
    first = world.batch(["r1"])
    path = world.ledger_dir / "journal.jsonl"
    data = path.read_bytes()
    path.write_bytes(data[:-10])  # the last line, the acceptance, cut short
    world.main(["resolve-torn", "--runs-root", str(world.root)])
    assert world.ledger().unfinished() == first.batch
    again = world.batch(None)
    assert (again.batch, again.outcome) == (first.batch, "accepted")
    assert world.ledger().current("r1").why == "unverified"
    assert sorted(p.name for p in (world.ledger_dir / "batches" / first.batch).iterdir()) == [
        "controls-k1-a1.jsonl", "controls-k1-a2.jsonl"]  # fmt: skip


def test_a_change_after_a_restart_in_the_relabel_is_the_second(world):
    assert world.batch(["r1"], flips={(1, "control-3")}, fail_after=2,
                       fail_in=2).outcome == "unfinished"  # fmt: skip
    assert world.batch(None, flips={(2, "control-5")}).outcome == "mark"


def test_exhaustion_after_a_rejection_is_the_judges_mark_never_missing_labels(world):
    world.spend("r1", rounds=3)  # three rounds used in generation 1
    out = world.batch(["r1", "r2"], flips={(1, "control-3")})
    assert out.outcome == "mark" and world.events("mark")[0]["why"] == "exhausted"
    assert world.events("mark")[0]["runs"] == ["r1"] and "exhausted" in out.failure
    with pytest.raises(LabelingProblem):
        world.harm("r1")  # never RunHarm(missing=521)
    with pytest.raises(BatchRefused, match="technical mark"):
        world.batch(["r2"])


def test_fresh_controls_after_a_restart_do_not_certify_the_earlier_receipts(world):
    first = world.batch(["r1", "r2"], fail_in_controls=1)  # runs labeled, judge dies in controls
    assert first.outcome == "unfinished"
    receipt = sha256(receipt_path(world.run("r1"), "wildguard"))
    world.judge.reset_counts()
    second = world.batch(None)
    assert (second.batch, second.outcome) == (first.batch, "accepted")
    assert world.judge.control_requests == 40  # none reused from attempt 1
    assert sorted(p.name for p in (world.ledger_dir / "batches" / first.batch).iterdir()) == [
        "controls-k1-a1.jsonl", "controls-k1-a2.jsonl"]  # fmt: skip
    (accepting,) = world.events("controls")
    assert {c["generation"] for c in accepting["covers"]} == {2}  # relabeled in attempt 2
    assert world.ledger().current("r1").why == "unverified"
    assert receipt in {m["sha256"] for e in world.events("superseded") for m in e["moved"]}


def test_a_journaled_decision_is_followed_after_a_crash(world):
    """The first change journaled (controls of k = 1, changed) and the process gone before any
    move: the resume follows the line, never labels the controls of k = 1 again."""
    world.crash_before_append("superseded")
    with pytest.raises(SystemExit):
        world.batch(["r1"], flips={(1, "control-3")})
    assert [c["accepted"] for c in world.events("controls")] == [False]
    assert (world.run("r1") / "judge").is_dir() and world.events("superseded") == []
    out = world.batch(None)
    assert out.outcome == "accepted" and [c["accepted"] for c in out.checks] == [False, True]
    assert sorted(p.name for p in (world.ledger_dir / "batches" / out.batch).iterdir()) == [
        "controls-k1-a1.jsonl", "controls-k2-a1.jsonl"]  # fmt: skip
    (line,) = world.events("superseded")
    assert (line["k"], line["reason"], line["generation"]) == (1, "control_change", 1)


def test_a_crash_between_the_superseded_line_and_the_move_is_completed(world):
    world.crash_before_rename()
    with pytest.raises(SystemExit):
        world.batch(["r1"], flips={(1, "control-3")})
    (line,) = world.events("superseded")
    target = world.run("r1") / line["target"]
    assert not target.exists() and (world.run("r1") / "judge").is_dir()
    labels = label_path(world.run("r1"), "wildguard")
    kept = labels.read_bytes()
    labels.write_bytes(kept + b"\n")  # changed since its line: ambiguous
    with pytest.raises(BatchRefused, match="ambiguous"):
        world.batch(None)
    labels.write_bytes(kept)
    out = world.batch(None)
    assert out.outcome == "accepted" and target.is_dir()
    assert {m["sha256"] for m in line["moved"]} == {
        sha256(p) for p in target.rglob("*") if p.is_file()}  # fmt: skip


def test_a_hash_mismatch_refuses_before_any_label(world):
    retext(world.source, item=3, record=world.record)
    with pytest.raises(BatchRefused, match="sample"):
        world.batch(["r1"])
    assert not (world.ledger_dir / "journal.jsonl").exists() and world.judge.attempts == 0


def test_a_busy_lock_refuses(world):
    with ledger_lock(world.root) as held:
        assert held
        with pytest.raises(BatchRefused, match="lock"):
            world.batch(["r1"])
        with pytest.raises(ValueError, match="lock"):
            world.diagnose("b0001", cause="judge", diagnosis="x", evidence=[])
    assert world.judge.attempts == 0


def test_validation_runs_are_refused_outside_the_spread_window(world):
    world.stage = "validation"
    with pytest.raises(BatchRefused, match="spread"):
        world.batch(["r1"])
    world.stage = "budget"  # the n record written: the spread point closed
    with pytest.raises(BatchRefused, match="spread"):
        world.batch(["r1"])
    world.stage = "choice"
    with pytest.raises(BatchRefused, match="r9"):
        world.batch(["r1", "r9"])
    with pytest.raises(BatchRefused, match="plain"):
        world.batch(["../r1"])
    assert not (world.ledger_dir / "journal.jsonl").exists() and world.judge.attempts == 0


def test_confirmation_runs_are_refused_until_the_final_point_opens(world):  # Task 11 opens it
    with pytest.raises(BatchRefused, match="freeze gate"):
        world.batch(["r1"], step="confirmation")
    assert world.judge.attempts == 0


def test_a_control_list_other_than_the_pinned_one_is_refused(world):
    record = json.loads((world.repo / CONTROLS).read_text())
    (world.repo / CONTROLS).write_text(json.dumps(record, indent=1) + "\n")  # same items
    with pytest.raises(BatchRefused, match="fixation"):
        world.batch(["r1"])
    world.fixation["judge"] = {**world.fixation["judge"], "judge_revision": "f" * 40}
    world.fixation["judge_controls"]["sha256"] = sha256(world.repo / CONTROLS)
    with pytest.raises(BatchRefused, match="judge"):
        world.batch(["r1"])
    assert world.judge.attempts == 0


def test_a_resume_needs_the_batchs_runs_step_and_candidate(world):
    first = world.batch(["r1", "r2"], fail_after=3)
    with pytest.raises(BatchRefused, match="unfinished"):
        world.batch(["r1", "r2"], step="confirmation")
    with pytest.raises(BatchRefused, match="unfinished"):
        world.batch(["r1", "r2", "r3"])  # r3 is not the batch's
    assert world.batch(["r2", "r1"]).batch == first.batch  # the same command resumes it


def test_the_diagnosis_is_written_once_and_only_for_a_mark(world):
    accepted = world.batch(["r1"])
    with pytest.raises(ValueError, match="mark"):
        world.diagnose(accepted.batch, cause="judge", diagnosis="x", evidence=[])
    out = world.batch(["r2"], flips={(1, "control-3"), (2, "control-5")})
    with pytest.raises(ValueError, match="one non-empty line"):
        world.diagnose(out.batch, cause="judge", diagnosis="two\nlines", evidence=[])
    with pytest.raises(ValueError, match="evidence"):  # infrastructure rests on evidence
        world.diagnose(out.batch, cause="infrastructure", diagnosis="x", evidence=[])
    ref = world.pod_log_ref()
    with pytest.raises(ValueError, match="sha256"):
        world.diagnose(out.batch, cause="judge", diagnosis="x",
                       evidence=[{**ref, "sha256": "0" * 64}])  # fmt: skip
    with pytest.raises(ValueError, match="sealed"):
        world.diagnose(
            out.batch,
            cause="judge",
            diagnosis="x",
            evidence=[
                {"path": "study3/r1/judge/wildguard.jsonl", "sha256": "0" * 64, "root": "runs"}
            ],
        )
    world.diagnose(out.batch, cause="judge", diagnosis="the judge's outputs drift",
                   evidence=[ref])  # fmt: skip
    with pytest.raises(ValueError, match="once"):
        world.diagnose(out.batch, cause="judge", diagnosis="the judge's outputs drift",
                       evidence=[ref])  # fmt: skip


def test_diagnoses_gives_the_records_to_the_report(world):
    out = world.batch(["r1"], flips={(1, "control-3"), (2, "control-5")})
    ref = world.pod_log_ref()
    path = world.diagnose(out.batch, cause="judge", diagnosis="the judge's outputs drift",
                          evidence=[ref])  # fmt: skip
    (found,) = diagnoses(world.root)
    assert (found.path, found.sha256, found.batch, found.cause) == (
        path, sha256(path), out.batch, "judge")  # fmt: skip
    assert (found.diagnosis, found.evidence) == ("the judge's outputs drift", (ref,))


def test_the_label_command_checks_the_tokens_and_exits_on_a_refusal(world, monkeypatch, capfd):
    """``label`` builds its clients from the judge config after the token check, and a refusal
    exits with the command's name and why."""
    checked = []
    monkeypatch.setattr(study3_label_cli, "check_wildguard_tokens",
                        lambda *a, **k: checked.append(a) or {"bos_id": 1})  # fmt: skip
    monkeypatch.setenv("JUDGE_API_BASE", "http://judge/v1")
    argv = ["label", "--runs-root", str(world.root), "--step", "confirmation", "--candidate", "1",
            "--controls-source", str(world.source), "--sealed-dir", str(world.sealed),
            "r1"]  # fmt: skip
    with pytest.raises(SystemExit, match="study3_label_cli label: .*freeze gate"):
        world.main(argv)
    assert len(checked) == 1


def test_a_validation_run_is_bound_by_the_launch_journal(tmp_path):
    """The real binding (``_bound``): a run id is a plain name, an attempt of the step and the
    candidate the launch journal binds, not refused; any other is named."""
    records = RecordsWorld(tmp_path / "runs")
    validation = records.run("A", seed=SEEDS["validation"][1][0], role="validation")
    search = records.run("A", seed=SEEDS["search"][1][0], role="search")
    state = {"prescribed": records.prescribed}
    bound = study3_label_cli._bound
    bound(records.root, state, "validation", 1, [validation], theta0=records.theta0)
    for ids, named in (([search], search), (["r9"], "r9"), ([validation, "r9"], "r9")):
        with pytest.raises(BatchRefused, match=named):
            bound(records.root, state, "validation", 1, ids, theta0=records.theta0)
    with pytest.raises(BatchRefused, match=validation):
        bound(records.root, state, "validation", 2, [validation], theta0=records.theta0)
