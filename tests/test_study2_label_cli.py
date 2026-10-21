"""Study 2's sealed labeling command (analysis.study2_label_cli; review fa27): which runs are
sealed, what is printed of them on a real labeling state (tests/test_study2_labels.py's runs,
labeled through ``run_judging.label_runs`` against a fake judge server), the sealed record and the
exit codes. Synthetic text only; output is captured at the file-descriptor level (capfd)."""

import fcntl
import json
import logging
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from test_study2_labels import HARMFUL, JUDGE, PROMPTS, REFUSAL, FakeWildGuard, factory, make_run

from zero_field_refusal.analysis import study2_label_cli
from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.stage_a import TrajectoryError
from zero_field_refusal.analysis.study2_label_cli import (
    BUSY,
    ERROR,
    INCOMPLETE,
    LABELED,
    NOT_ATTEMPTED_STATUS,
    PROBE_ITEM,
    STATUSES,
    judge_answers,
    sealed_directory,
    sealed_reason,
)
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    FA,
    OPTIMIZERS,
    PREWARMED,
    TIMECOURSE,
    TIMECOURSE_SEEDS,
)
from zero_field_refusal.judge import JudgeClient
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.run_judging import NOT_ATTEMPTED, label_runs, lock_path, receipt_path

TOKENS = {"bos_id": 1, "probe_tokens": 9}  # the token check's answer, as the server gives it
SAFE = "Harmful request: no\nResponse refusal: no\nHarmful response: no"  # the re-probe's answer
SEALED_KEYS = {"run_id", "sealed", "status"}
# What label_runs.py prints of a run and the CLI's own lines may name; nothing else of a sealed run
COUNT_KEYS = ("unparseable", "missing", "rounds", "items", "kept", "restored", "error", "skipped")


@pytest.fixture(autouse=True)
def published(monkeypatch):
    """The runs here stand for runs whose terminal record was published with whole records
    (``stage_a.load_trajectory``), but those named ``unpublished*``."""

    def load(run_dir):
        if Path(run_dir).name.startswith("unpublished"):
            raise TrajectoryError(f"{run_dir}: unreadable records")

    monkeypatch.setattr("zero_field_refusal.run_judging.load_trajectory", load)


def study2_run(root: Path, name: str, arm: str, *, seed: int = 3020, role="confirmation",
               steps=(25,)) -> Path:  # fmt: skip
    """test_study2_labels' run with what the classification reads: the arm's optimizer under the
    fixed schedule, the seed and ``study2_short.role`` (None: none)."""
    run = make_run(root, steps, name)
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.schedule.policy = "fixed"
    cfg.optimizer = {"name": OPTIMIZERS[arm]}
    cfg.seed = seed
    if role is not None:
        cfg.study2_short = {"role": role}
    OmegaConf.save(cfg, run / "config.yaml")
    return run


def gone(fake) -> bool:
    """Whether the fake server is gone (``fail_after`` requests answered)."""
    return fake.fail_after is not None and fake.calls >= fake.fail_after


def server(fake):
    """The fake server as the command's clients see it: it answers the synthetic re-probe
    (``PROBE_ITEM``) as a judge would, parseably, without counting it in ``calls``, and fails it
    once it is gone; every other request is the fake's."""

    def create(**kwargs):
        if study2_label_cli.PROBE_ITEM["response"] not in kwargs["prompt"]:
            return fake.completions.create(**kwargs)
        if gone(fake):
            raise RuntimeError("the judge server is gone")
        return SimpleNamespace(choices=[SimpleNamespace(text=SAFE, finish_reason="stop")])

    return SimpleNamespace(completions=SimpleNamespace(create=create))


def cli(monkeypatch, tmp_path: Path, fake, *runs: Path, sealed: Path | None = None,
        tokenizer_up: bool = False) -> str | int | None:  # fmt: skip
    """``main`` on ``runs`` against the fake judge server (``server``), the sealed records under
    ``sealed`` (default ``tmp_path/sealed``); returns its exit message (None: exit 0). The token
    check, also part of the re-probe after the sealed call, fails once the fake server is gone,
    unless ``tokenizer_up`` (its /tokenize answers while generation is gone)."""

    class Client:  # JudgeClient.from_config with the fake server behind it
        @staticmethod
        def from_config(cfg, env=None, **kwargs):
            return JudgeClient.from_config(cfg, {"JUDGE_API_BASE": "http://judge/v1"},
                                           client=server(fake), sleep=lambda s: None,
                                           **kwargs)  # fmt: skip

    monkeypatch.setattr(study2_label_cli, "JudgeClient", Client)

    def tokens(*args, **kwargs) -> dict:
        if gone(fake) and not tokenizer_up:
            raise RuntimeError("the judge server is gone")
        return TOKENS

    monkeypatch.setattr(study2_label_cli, "check_wildguard_tokens", tokens)
    monkeypatch.setattr(study2_label_cli, "harmful_prompts", lambda cfg: PROMPTS)
    monkeypatch.setenv("JUDGE_API_BASE", "http://judge/v1")
    try:
        study2_label_cli.main(["--judge", JUDGE, "--sealed-dir", str(sealed or tmp_path / "sealed"),
                               *map(str, runs)])  # fmt: skip
    except SystemExit as stopped:
        return stopped.code
    return None


def records(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "sealed").glob("labeling-*-wildguard.json"))


def printed(capfd) -> tuple[list[dict], str]:
    """The printed lines, as JSON, and stderr (both at the file-descriptor level)."""
    out, err = capfd.readouterr()
    return [json.loads(line) for line in out.splitlines()], err


def assert_allowlisted(lines: list[dict], err: str, sealed: dict[str, str], record: Path | None,
                       opened: tuple[dict, ...] = ()) -> None:  # fmt: skip
    """Exactly: the open runs' summaries (``opened``, in order with the sealed ones' lines), each
    sealed run's allowlisted line with its status, the invocation's line and the record's path;
    nothing on stderr."""
    shown = [line for line in lines if line.get("sealed")]
    assert all(set(line) == SEALED_KEYS and line["status"] in STATUSES for line in shown)
    assert {line["run_id"]: line["status"] for line in shown} == sealed
    rest = [line for line in lines if not line.get("sealed")]
    tail = [{"runs": len(sealed) + len(opened), "token_check": TOKENS}]
    tail += [] if record is None else [{"sealed_record": str(record)}]
    assert rest == [*opened, *tail]
    for line in [*shown, *tail]:
        assert not set(line) & set(COUNT_KEYS), line
    assert err == ""


def test_a_prewarmed_run_with_unparseable_items_shows_its_status_alone(
    monkeypatch, tmp_path, capfd
) -> None:
    """Review fa27, 1: a pre-warmed run whose six items never parse, on its first labeling and on
    the completed receipt's reuse (``kept``): only its allowlisted status is printed and the exit
    is 0 (scripts/label_runs.py would print ``"unparseable": 6`` and exit non-zero); the sealed
    record holds the summary, 0600 in a 0700 directory."""
    run = study2_run(tmp_path / "runs", "prewarmed", PREWARMED)
    fake = FakeWildGuard(["cannot classify"])
    assert cli(monkeypatch, tmp_path, fake, run) is None
    (first,) = records(tmp_path)
    assert_allowlisted(*printed(capfd), {"prewarmed": LABELED}, first)
    assert fake.calls == 18  # three rounds of six items
    whole = json.loads(first.read_text())
    assert whole["runs"] == [
        {
            "run_dir": str(run),
            "run_id": "prewarmed",
            "sealed": "prewarmed",
            "status": LABELED,
            "summary": {"run_id": "prewarmed", "items": 6, "unparseable": 6, "rounds": 3},
        }
    ]
    assert whole["attempted"] and whole["traceback"] is None and whole["judge"] == JUDGE
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert stat.S_IMODE(first.parent.stat().st_mode) == 0o700
    # The completed receipt's reuse path: nothing sent, the same allowlisted line.
    assert cli(monkeypatch, tmp_path, fake, run) is None
    second = next(path for path in records(tmp_path) if path != first)
    assert_allowlisted(*printed(capfd), {"prewarmed": LABELED}, second)
    assert fake.calls == 18
    (entry,) = json.loads(second.read_text())["runs"]
    assert entry["summary"]["kept"] and entry["summary"]["unparseable"] == 6
    assert stat.S_IMODE(second.stat().st_mode) == 0o600


def test_an_open_run_is_printed_in_full(monkeypatch, tmp_path, capfd) -> None:
    """An AdamW confirmation run is printed and exits exactly as scripts/label_runs.py does, and
    no sealed record is written for it."""
    run = study2_run(tmp_path / "runs", "adamw", ADAMW)
    message = cli(monkeypatch, tmp_path, FakeWildGuard(["cannot classify"]), run)
    assert message.startswith("Some outputs did not parse.")
    lines, err = printed(capfd)
    assert lines == [{"run_id": "adamw", "items": 6, "unparseable": 6, "rounds": 3},
                     {"runs": 1, "token_check": TOKENS}]  # fmt: skip
    assert err == "" and not (tmp_path / "sealed").exists()


def test_open_and_sealed_runs_in_one_invocation(monkeypatch, tmp_path, capfd) -> None:
    """The lines keep the runs' order; the open run's counts are printed, the sealed runs' are
    not, and the exit message names the open run's unparseable outputs only."""
    root = tmp_path / "runs"
    runs = [
        study2_run(root, "fa", FA),
        study2_run(root, "prewarmed", PREWARMED),
        study2_run(root, "course", ADAMW, seed=TIMECOURSE_SEEDS[0], role=TIMECOURSE),
    ]
    replies = ["cannot classify"] * 18 + [REFUSAL]  # the open run's three rounds, then labels
    message = cli(monkeypatch, tmp_path, FakeWildGuard(replies), *runs)
    assert message.startswith("Some outputs did not parse.") and "Sealed" not in message
    lines, err = printed(capfd)
    (record,) = records(tmp_path)
    assert lines[0] == {"run_id": "fa", "items": 6, "unparseable": 6, "rounds": 3}
    assert [line["run_id"] for line in lines[:3]] == ["fa", "prewarmed", "course"]
    assert_allowlisted(lines, err, {"prewarmed": LABELED, "course": LABELED}, record,
                       opened=(lines[0],))  # fmt: skip


def test_which_runs_are_sealed(tmp_path) -> None:
    """The pre-warmed arm whatever its role, the time course of every arm (on its seeds), and any
    run that cannot be classified; a primary arm's endpoint run, on a time-course seed too, and a
    run that is not a study-2 run are open."""
    root = tmp_path / "runs"
    first = TIMECOURSE_SEEDS[0]
    assert sealed_reason(study2_run(root, "p-confirmation", PREWARMED)) == "prewarmed"
    assert sealed_reason(study2_run(root, "p-discovery", PREWARMED, seed=2000,
                                    role="discovery")) == "prewarmed"  # fmt: skip
    assert sealed_reason(study2_run(root, "p-none", PREWARMED, role=None)) == "prewarmed"
    for arm in (ADAMW, FA, PREWARMED):
        course = study2_run(root, f"{arm}-course", arm, seed=first, role=TIMECOURSE)
        assert sealed_reason(course) == ("prewarmed" if arm == PREWARMED else TIMECOURSE)
    assert sealed_reason(study2_run(root, "a-endpoint", ADAMW, seed=first)) is None
    assert sealed_reason(study2_run(root, "f-endpoint", FA)) is None
    assert sealed_reason(study2_run(root, "f-discovery", FA, seed=2000, role="discovery")) is None
    # Fail closed: a time-course run off its seeds, a run of a time-course seed without the role
    # of an endpoint run, an unknown arm, a config that does not read or lacks the seed.
    unclassifiable = [study2_run(root, "a-course-elsewhere", ADAMW, role=TIMECOURSE),
                      study2_run(root, "a-roleless", ADAMW, seed=first, role=None)]  # fmt: skip
    odd = study2_run(root, "odd-arm", ADAMW)
    cfg = OmegaConf.load(odd / "config.yaml")
    cfg.optimizer.name = "lion"
    OmegaConf.save(cfg, odd / "config.yaml")
    garbled = study2_run(root, "garbled", ADAMW)
    (garbled / "config.yaml").write_text("schedule: [unclosed\n")
    seedless = make_run(root, (25,), "seedless")  # no optimizer, no seed
    listed = study2_run(root, "listed", ADAMW)
    (listed / "config.yaml").write_text("- a list\n")
    for run in (*unclassifiable, odd, garbled, seedless, listed, root / "missing"):
        assert sealed_reason(run) == "unclassifiable", run
    theta0 = study2_run(root, "theta0", PREWARMED)  # a θ₀ run keeps its schedule section
    cfg = OmegaConf.load(theta0 / "config.yaml")
    cfg.run_type = "theta0"
    OmegaConf.save(cfg, theta0 / "config.yaml")
    assert sealed_reason(theta0) is None


def test_a_timecourse_run_of_adamw_is_sealed(monkeypatch, tmp_path, capfd) -> None:
    run = study2_run(tmp_path / "runs", "course", ADAMW, seed=TIMECOURSE_SEEDS[3],
                     role=TIMECOURSE)  # fmt: skip
    assert cli(monkeypatch, tmp_path, FakeWildGuard(["cannot classify", HARMFUL]), run) is None
    (record,) = records(tmp_path)
    assert_allowlisted(*printed(capfd), {"course": LABELED}, record)
    (entry,) = json.loads(record.read_text())["runs"]
    assert entry["sealed"] == TIMECOURSE and entry["summary"]["rounds"] == 2


def test_an_unclassifiable_run_is_sealed(monkeypatch, tmp_path, capfd) -> None:
    """A study-2 run whose arm the analysis does not know: labeled, its output sealed."""
    run = study2_run(tmp_path / "runs", "unknown", ADAMW)
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.optimizer.name = "lion"
    OmegaConf.save(cfg, run / "config.yaml")
    assert cli(monkeypatch, tmp_path, FakeWildGuard(["cannot classify"]), run) is None
    (record,) = records(tmp_path)
    assert_allowlisted(*printed(capfd), {"unknown": LABELED}, record)
    assert json.loads(record.read_text())["runs"][0]["sealed"] == "unclassifiable"


def test_an_exception_inside_the_sealed_call_is_sealed(monkeypatch, tmp_path, capfd) -> None:
    """Whatever the sealed call prints (to stdout or stderr, where warnings go), logs (to a
    handler holding the original stderr too) or raises goes to the sealed record; the run shows
    "error", the exit message names its status and the record, never a count."""
    run = study2_run(tmp_path / "runs", "prewarmed", PREWARMED)
    stray = logging.getLogger("study2-label-cli-test")
    held = logging.StreamHandler(sys.stderr)  # bound to the stream as it is now, before the call
    stray.addHandler(held)
    stray.propagate = False

    def failing(run_dirs, *args, **kwargs):
        print("unparseable 6 of 6")
        print("a label 'harmful_compliance' on stderr", file=sys.stderr)
        stray.warning("missing 6 in a log line")
        raise ValueError("its receipt records 6 items without a label, the file 5")

    monkeypatch.setattr(study2_label_cli, "label_runs",
                        lambda dirs, *a, **k: failing(dirs) if dirs else [])  # fmt: skip
    try:
        message = cli(monkeypatch, tmp_path, FakeWildGuard([REFUSAL]), run)
    finally:
        stray.removeHandler(held)
        stray.propagate = True
    (record,) = records(tmp_path)
    assert_allowlisted(*printed(capfd), {"prewarmed": ERROR}, record)
    assert message == (f"Sealed runs not labeled yet ({ERROR}): rerun the labeling; their details "
                       f"are sealed in {record}")  # fmt: skip
    whole = json.loads(record.read_text())
    assert "ValueError: its receipt records 6 items without a label" in whole["traceback"]
    for text in ("unparseable 6 of 6", "'harmful_compliance' on stderr", "missing 6 in a log line"):
        assert text in whole["output"]
    assert whole["runs"][0]["summary"] is None


@pytest.mark.parametrize("reply", [REFUSAL, "cannot classify"])
def test_the_statuses_and_exit_codes_of_sealed_runs(monkeypatch, tmp_path, capfd, reply) -> None:
    """Busy and not complete (no published record; a receipt written again, whatever its rounds)
    exit non-zero; labeled (whatever its missing labels) and nothing to label exit 0. Each status
    is the same whether every label parsed or none did."""
    root = tmp_path / "runs"
    busy = study2_run(root, "busy", PREWARMED)
    unpublished = study2_run(root, "unpublished", PREWARMED)
    early = study2_run(root, "early", PREWARMED, steps=())  # stopped before its first evaluation
    restored = study2_run(root, "restored", PREWARMED)
    rewritten = study2_run(root, "rewritten", PREWARMED)
    # Receipts deleted after a completed labeling, and written again for the unchanged label
    # files: one labeled with one round (as scripts/label_runs.py --rounds 1 would), one with up
    # to three (one round when every label parsed, three when none did).
    fake = FakeWildGuard([reply])
    label_runs([restored], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=1)
    label_runs([rewritten], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    for run in (restored, rewritten):
        receipt_path(run, JUDGE).unlink()
    lock = lock_path(busy, JUDGE)
    lock.parent.mkdir()
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        message = cli(monkeypatch, tmp_path, fake, busy, unpublished, early, restored, rewritten)
        fcntl.flock(handle, fcntl.LOCK_UN)
    (record,) = records(tmp_path)
    expected = {"busy": BUSY, "unpublished": INCOMPLETE, "early": LABELED, "restored": INCOMPLETE,
                "rewritten": INCOMPLETE}  # fmt: skip
    assert_allowlisted(*printed(capfd), expected, record)
    assert message == (f"Sealed runs not labeled yet ({INCOMPLETE}; {BUSY}): rerun the "
                       f"labeling; their details are sealed in {record}")  # fmt: skip
    # Rerun: a receipt written again is kept (every label there, or three rounds used) or resumed
    # (none, one round used): labeled either way; the busy run is labeled now.
    assert cli(monkeypatch, tmp_path, fake, busy, early, restored, rewritten) is None
    second = next(path for path in records(tmp_path) if path != record)
    assert_allowlisted(*printed(capfd), {"busy": LABELED, "early": LABELED, "restored": LABELED,
                                         "rewritten": LABELED}, second)  # fmt: skip


def test_a_lost_receipt_shows_the_same_status_whatever_the_rounds(
    monkeypatch, tmp_path, capfd
) -> None:
    """Review fa27, fix round 1 (the reviewer's probe): a run whose outputs all parsed in round 1
    and one that used three rounds, labeled through the command, lose their receipts: both show
    "not complete" (three rounds would tell that outputs stayed unparseable after the second)."""
    root = tmp_path / "runs"
    clean, noisy = study2_run(root, "clean", PREWARMED), study2_run(root, "noisy", PREWARMED)
    assert cli(monkeypatch, tmp_path, FakeWildGuard([REFUSAL]), clean) is None
    assert cli(monkeypatch, tmp_path, FakeWildGuard(["cannot classify"]), noisy) is None
    for run in (clean, noisy):
        receipt_path(run, JUDGE).unlink()
    capfd.readouterr()
    before = set(records(tmp_path))
    message = cli(monkeypatch, tmp_path, FakeWildGuard([REFUSAL]), clean, noisy)
    (record,) = set(records(tmp_path)) - before
    assert_allowlisted(*printed(capfd), {"clean": INCOMPLETE, "noisy": INCOMPLETE}, record)
    assert message == (f"Sealed runs not labeled yet ({INCOMPLETE}): rerun the labeling; their "
                       f"details are sealed in {record}")  # fmt: skip


@pytest.mark.parametrize("tokenizer_up", [False, True])
def test_a_judge_gone_in_the_sealed_call_shows_one_status(
    monkeypatch, tmp_path, capfd, tokenizer_up
) -> None:
    """Review fa27, fix rounds 1 and 2 (the reviewers' probes): the judge answers six requests,
    then its generation is gone, its tokenizer too or (``tokenizer_up``) still answering. A run
    whose outputs parse is labeled in round 1, before it goes, and the re-probe's synthetic
    judgment finds it gone; one whose outputs do not parse needs round 2, which fails. Either way
    every sealed run of the call is "not complete", with the same lines and exit message; a rerun
    with the judge back labels it."""
    shown = []
    for name, reply in (("parsed", REFUSAL), ("unparsed", "cannot classify")):
        run = study2_run(tmp_path / name / "runs", "prewarmed", PREWARMED)
        message = cli(monkeypatch, tmp_path / name, FakeWildGuard([reply], fail_after=6), run,
                      tokenizer_up=tokenizer_up)  # fmt: skip
        (record,) = records(tmp_path / name)
        lines, err = printed(capfd)
        assert_allowlisted(lines, err, {"prewarmed": INCOMPLETE}, record)
        whole = json.loads(record.read_text())
        assert whole["judge_failed"]
        if reply == REFUSAL:  # the re-probe failed; its traceback is in the record only
            assert "the judge server is gone" in whole["probe_traceback"]
        shown.append((lines[:-1], message.replace(str(record), "RECORD")))
        assert cli(monkeypatch, tmp_path / name, FakeWildGuard([reply]), run) is None
        second = next(path for path in records(tmp_path / name) if path != record)
        assert_allowlisted(*printed(capfd), {"prewarmed": LABELED}, second)
    assert shown[0] == shown[1]
    assert shown[0][1] == (f"Sealed runs not labeled yet ({INCOMPLETE}): rerun the labeling; "
                           "their details are sealed in RECORD")  # fmt: skip


def test_the_synthetic_re_probe(tmp_path) -> None:
    """``judge_answers``: one request for ``PROBE_ITEM`` through a client writing to a temporary
    label file, removed after; an answer that does not parse, or a failed request, raises."""
    paths = []

    def make_client(reply: str | None):
        fake = FakeWildGuard([reply] if reply is not None else [SAFE], fail_after=None
                             if reply is not None else 0)  # fmt: skip

        def client(path: Path) -> JudgeClient:
            paths.append(path)
            return factory(fake)(path)

        return client, fake

    client, fake = make_client(SAFE)
    judge_answers(client)
    assert fake.calls == 1 and not paths[-1].exists() and not paths[-1].parent.exists()
    for reply in ("cannot classify", None):  # an unparseable answer; the server gone
        client, _ = make_client(reply)
        with pytest.raises(RuntimeError):
            judge_answers(client)
        assert not paths[-1].parent.exists()
    assert PROBE_ITEM["prompt"] not in PROMPTS.values()


def test_a_failed_judge_in_the_open_or_the_sealed_call(monkeypatch, tmp_path, capfd) -> None:
    """The judge gone in the open call: the sealed call is not made (its runs "not attempted",
    nothing sent). Gone in the sealed call: every run of the call "not complete"; the exit
    messages name statuses only."""
    root = tmp_path / "runs"
    adamw, first, second = (study2_run(root, "adamw", ADAMW),
                            study2_run(root, "first", PREWARMED),
                            study2_run(root, "second", PREWARMED))  # fmt: skip
    message = cli(monkeypatch, tmp_path, FakeWildGuard([REFUSAL], fail_after=3), first, adamw,
                  second)  # fmt: skip
    lines, err = printed(capfd)
    (record,) = records(tmp_path)
    assert "RoundFailed" in lines[1]["error"]
    assert_allowlisted(lines, err, {"first": NOT_ATTEMPTED_STATUS, "second": NOT_ATTEMPTED_STATUS},
                       record, opened=(lines[1],))  # fmt: skip
    assert message == ("Some runs could not be labeled; see their errors above\nSealed runs not "
                       f"labeled yet ({NOT_ATTEMPTED_STATUS}): rerun the labeling; their details "
                       f"are sealed in {record}")  # fmt: skip
    whole = json.loads(record.read_text())
    assert not whole["attempted"] and [r["summary"] for r in whole["runs"]] == [None, None]
    assert not (first / "judge").exists()  # nothing sent, no round counted
    message = cli(monkeypatch, tmp_path, FakeWildGuard([REFUSAL], fail_after=3), first, second)
    second_record = next(path for path in records(tmp_path) if path != record)
    assert_allowlisted(*printed(capfd), {"first": INCOMPLETE, "second": INCOMPLETE},
                       second_record)  # fmt: skip
    assert message == (f"Sealed runs not labeled yet ({INCOMPLETE}): rerun the labeling; their "
                       f"details are sealed in {second_record}")  # fmt: skip
    entries = json.loads(second_record.read_text())["runs"]
    assert "the judge server is gone" in entries[0]["summary"]["error"]
    assert entries[1]["summary"]["skipped"] == NOT_ATTEMPTED


def test_an_error_that_is_no_judge_failure_stays_the_runs_own(monkeypatch, tmp_path, capfd) -> None:
    """With the judge up, a run refused (its label file changed after its receipt) shows "error";
    the other sealed run of the call is labeled."""
    root = tmp_path / "runs"
    good, changed = study2_run(root, "good", PREWARMED), study2_run(root, "changed", PREWARMED)
    fake = FakeWildGuard([REFUSAL])
    label_runs([changed], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    with label_path(changed, JUDGE).open("a") as handle:
        handle.write("\n")
    message = cli(monkeypatch, tmp_path, fake, good, changed)
    (record,) = records(tmp_path)
    assert_allowlisted(*printed(capfd), {"good": LABELED, "changed": ERROR}, record)
    assert message == (f"Sealed runs not labeled yet ({ERROR}): rerun the labeling; their details "
                       f"are sealed in {record}")  # fmt: skip
    whole = json.loads(record.read_text())
    assert not whole["judge_failed"] and "never delete the receipt" in whole["runs"][1][
        "summary"]["error"]  # fmt: skip


def test_a_sealed_directory_git_could_track_is_refused(monkeypatch, tmp_path, capfd) -> None:
    """Review fa27, fix round 1: ``--sealed-dir`` in the checkout outside runs/ (which Git
    ignores) is refused before anything is sent; the default is ``$RUNS_DIR/sealed``."""
    run = study2_run(tmp_path / "runs", "prewarmed", PREWARMED)
    fake = FakeWildGuard([REFUSAL])
    tracked = REPO_ROOT / "docs" / "sealed-test"
    assert cli(monkeypatch, tmp_path, fake, run, sealed=tracked) == 2
    assert "lies in the checkout outside runs/" in capfd.readouterr().err
    assert fake.attempts == 0 and not tracked.exists() and not (run / "judge").exists()
    ignored = REPO_ROOT / "runs" / "sealed-test"
    assert sealed_directory(ignored) == ignored and not ignored.exists()
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "store"))
    assert sealed_directory(None) == tmp_path / "store" / "sealed"
    monkeypatch.setenv("RUNS_DIR", str(REPO_ROOT / "docs"))
    with pytest.raises(ValueError, match="outside runs/"):
        sealed_directory(None)


def test_the_command_line_is_no_gpu_path_and_outside_the_launchers_imports() -> None:
    """Not a GPU path, and loaded by neither the analysis package nor the GPU entry points (the
    modules tests/test_gate_c.py names), in a fresh interpreter."""
    from test_gate_c import GPU_ENTRY_POINTS

    from zero_field_refusal import code_identity

    assert not code_identity.is_gpu_path("src/zero_field_refusal/analysis/study2_label_cli.py")
    imports = "; ".join(f"import zero_field_refusal.{name}"
                        for name in ("analysis", *GPU_ENTRY_POINTS))  # fmt: skip
    program = (f"import sys; {imports}; "
               "print('zero_field_refusal.analysis.study2_label_cli' in sys.modules, "
               "'zero_field_refusal.analysis' in sys.modules)")  # fmt: skip
    loaded = subprocess.run([sys.executable, "-c", program], cwd=REPO_ROOT, capture_output=True,
                            text=True, check=True)  # fmt: skip
    assert loaded.stdout.split() == ["False", "True"]
