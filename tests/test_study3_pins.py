"""Study 3's pins (analysis/study3_pins.py): the spread's static import closure and its pin, on
small checkouts written for each test, and the control list's closed schema, on a synthetic list
of 40 benign placeholder items (``control_list``, which the fixation's and the labeling's tests
reuse with the breakers beside it)."""

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from zero_field_refusal.analysis.study3_pins import (
    CONTROL_ITEMS,
    CONTROLS,
    CONTROLS_KIND,
    CONTROLS_WHAT,
    PER_CLASS,
    SPREAD_SCRIPT,
    control_list_problems,
    import_closure,
    pin_problems,
    read_control_list,
    spread_pin,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import judge_identity
from zero_field_refusal.paths import REPO_ROOT


def repo_with(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


def reader(repo):
    return lambda p: (repo / p).read_bytes() if (repo / p).is_file() else None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------ the control list


def control_list() -> dict:
    """A control list as the draw writes it, of 40 synthetic items: ids ``c%012x``, the hashes of
    the benign placeholder ``f"placeholder {i}"``, the even items harmful (WildGuard's
    ``harmful_compliance``) and the odd ones not (a refusal or a safe answer)."""
    items = []
    for i in range(CONTROL_ITEMS):
        harmful = i % 2 == 0
        label = "harmful_compliance" if harmful else ("refusal", "safe_non_refusal")[i // 2 % 2]
        items.append({"id": f"c{i:012x}", "prompt_sha256": digest(f"placeholder {i}"),
                      "response_sha256": digest(f"placeholder {i}"),
                      "expected": "harmful" if harmful else "not_harmful",
                      "label": label})  # fmt: skip
    return {
        "kind": CONTROLS_KIND, "what": CONTROLS_WHAT, "written_at": "2026-10-27T12:00:00+00:00",
        "code": {"commit": "a" * 40}, "seed": 20261027,
        "source": {"sample": {"path": "judge_calibration/calibration/sample.jsonl",
                              "sha256": digest("sample")},
                   "labels": {"path": "judge_calibration/calibration/judge-wildguard.jsonl",
                              "sha256": digest("labels")},
                   "archive": "gs://bucket/judge_calibration/calibration"},
        "judge": judge_identity(load_config("judge/wildguard")),
        "items": items,
    }  # fmt: skip


@pytest.fixture
def controls() -> dict:
    return control_list()


def drop_item(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["items"].pop()
    return out


def flip_expected(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["items"][0]["expected"] = "not_harmful"  # its label stays harmful_compliance
    return out


def extra_field(record: dict) -> dict:
    return {**copy.deepcopy(record), "note": "drawn again"}


def unsorted(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["items"][0], out["items"][1] = out["items"][1], out["items"][0]
    return out


def twenty_one_harmful(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["items"][1] |= {"expected": "harmful", "label": "harmful_compliance"}
    return out


def text_field(record: dict) -> dict:
    out = copy.deepcopy(record)
    out["items"][0]["prompt"] = "placeholder 0"
    return out


BREAKERS = (drop_item, flip_expected, extra_field, unsorted, twenty_one_harmful, text_field)


# ------------------------------------------------------------------ the import closure


def test_the_closure_follows_every_import_form(tmp_path) -> None:
    repo = repo_with(tmp_path, {
        "scripts/study3_spread.py": "import json\nfrom zero_field_refusal.analysis import a\n",
        "src/zero_field_refusal/__init__.py": "", "src/zero_field_refusal/analysis/__init__.py": "",
        "src/zero_field_refusal/analysis/a.py": "from . import b\nfrom ..c import X\n",
        "src/zero_field_refusal/analysis/b.py": "def f():\n    from .d import g\n",
        "src/zero_field_refusal/analysis/d.py": "import zero_field_refusal.e\n",
        "src/zero_field_refusal/c.py": "X = 1\n", "src/zero_field_refusal/e.py": "",
        "src/zero_field_refusal/unused.py": ""})  # fmt: skip
    assert import_closure(SPREAD_SCRIPT, reader(repo)) == sorted([
        "scripts/study3_spread.py", "src/zero_field_refusal/__init__.py",
        "src/zero_field_refusal/analysis/__init__.py", "src/zero_field_refusal/analysis/a.py",
        "src/zero_field_refusal/analysis/b.py", "src/zero_field_refusal/analysis/d.py",
        "src/zero_field_refusal/c.py", "src/zero_field_refusal/e.py"])  # fmt: skip


def test_imports_under_try_and_type_checking_and_package_imports_are_followed(tmp_path) -> None:
    """An import in a ``try`` or under ``if TYPE_CHECKING`` is followed as any other; a package's
    ``__init__.py`` is parsed and its imports followed; a name a package binds is no module."""
    repo = repo_with(tmp_path, {
        "scripts/study3_spread.py": ("try:\n    from zero_field_refusal import p\n"
                                     "except ImportError:\n    pass\n"
                                     "from typing import TYPE_CHECKING\n"
                                     "if TYPE_CHECKING:\n    import zero_field_refusal.q\n"),
        "src/zero_field_refusal/__init__.py": "from .r import load\nVALUE = 1\n",
        "src/zero_field_refusal/p/__init__.py": "from .. import VALUE, load\n",
        "src/zero_field_refusal/q.py": "",
        "src/zero_field_refusal/r.py": "def load(): ...\n"})  # fmt: skip
    assert import_closure(SPREAD_SCRIPT, reader(repo)) == [
        "scripts/study3_spread.py", "src/zero_field_refusal/__init__.py",
        "src/zero_field_refusal/p/__init__.py", "src/zero_field_refusal/q.py",
        "src/zero_field_refusal/r.py"]  # fmt: skip


def test_a_project_import_with_no_file_fails_closed(tmp_path) -> None:
    repo = repo_with(tmp_path, {"scripts/study3_spread.py": "from zero_field_refusal import gone\n",
                                "src/zero_field_refusal/__init__.py": ""})  # fmt: skip
    with pytest.raises(ValueError, match="gone"):
        import_closure(SPREAD_SCRIPT, reader(repo))


@pytest.mark.parametrize("files, words", [
    ({"scripts/study3_spread.py": "import zero_field_refusal.missing\n"}, "missing"),
    ({"scripts/study3_spread.py": "from . import sibling\n"}, "relative"),
    ({"scripts/study3_spread.py": "import zero_field_refusal.x\n",
      "src/zero_field_refusal/x.py": "from ... import y\n"}, "beyond"),
    ({"scripts/study3_spread.py": "def broken(:\n"}, "parse"),
    ({}, "scripts/study3_spread.py"),
])  # fmt: skip
def test_what_the_closure_cannot_follow_is_refused(tmp_path, files, words) -> None:
    """A project module with no file, a relative import outside the package or beyond its top, a
    file that does not parse and a missing entry are refused by name: the closure fails
    closed."""
    repo = repo_with(tmp_path, files)
    with pytest.raises(ValueError, match=words):
        import_closure(SPREAD_SCRIPT, reader(repo))


def test_a_star_import_from_a_project_package_fails_closed(tmp_path) -> None:
    """A star import from a project package loads the submodules its ``__all__`` names, which
    the closure cannot show: refused. One from a project module (no package) is followed."""
    repo = repo_with(tmp_path, {
        "scripts/study3_spread.py": "from zero_field_refusal.analysis import *\n",
        "src/zero_field_refusal/__init__.py": "",
        "src/zero_field_refusal/analysis/__init__.py": "__all__ = ['hidden']\n",
        "src/zero_field_refusal/analysis/hidden.py": "x = 1\n"})  # fmt: skip
    with pytest.raises(ValueError, match="star import from the project package"):
        import_closure(SPREAD_SCRIPT, reader(repo))
    (repo / SPREAD_SCRIPT).write_text("from zero_field_refusal.analysis.hidden import *\n")
    assert import_closure(SPREAD_SCRIPT, reader(repo)) == [
        "scripts/study3_spread.py", "src/zero_field_refusal/__init__.py",
        "src/zero_field_refusal/analysis/__init__.py",
        "src/zero_field_refusal/analysis/hidden.py"]  # fmt: skip


@pytest.mark.parametrize("files", [
    {"scripts/study3_spread.py": "import helper\n", "scripts/helper.py": "x = 1\n"},
    {"scripts/study3_spread.py": "from helper import x\n", "scripts/helper/__init__.py": ""},
    {"scripts/study3_spread.py": "import zero_field_refusal.m\n",
     "src/zero_field_refusal/m.py": "import helper.part\n", "scripts/helper.py": ""},
])  # fmt: skip
def test_an_import_that_resolves_beside_the_script_fails_closed(tmp_path, files) -> None:
    """A script's own folder is first on its import path: an import, in any file of the closure,
    whose top-level name is a module in ``scripts/`` loads that file, which the closure does not
    follow. Refused, by name."""
    repo = repo_with(tmp_path, files)
    with pytest.raises(ValueError, match="beside the entry"):
        import_closure(SPREAD_SCRIPT, reader(repo))


def test_modules_named_as_sealed_folders_are_read_and_a_link_into_one_is_not(tmp_path) -> None:
    """The project's own ``judge.py`` and ``labeling.py`` are modules, not sealed folders: the
    closure looks for ``judge/__init__.py`` first, which is not there, and reads ``judge.py``. A
    pinned file that links into a run's judge outputs is refused, never opened."""
    repo = repo_with(tmp_path / "repo", {
        "scripts/study3_spread.py": "from zero_field_refusal.judge import LABELS\n",
        "src/zero_field_refusal/judge.py": "from .labeling import x\nLABELS = ()\n",
        "src/zero_field_refusal/labeling.py": "x = 1\n"})  # fmt: skip
    assert [e["path"] for e in spread_pin(repo)] == [
        "scripts/study3_spread.py", "src/zero_field_refusal/judge.py",
        "src/zero_field_refusal/labeling.py"]  # fmt: skip
    outputs = tmp_path / "runs" / "study3" / "r1" / "judge"
    outputs.mkdir(parents=True)
    (outputs / "wildguard.jsonl").write_text('{"label": "x"}\n')
    (repo / "src/zero_field_refusal/labeling.py").unlink()
    (repo / "src/zero_field_refusal/labeling.py").symlink_to(outputs / "wildguard.jsonl")
    with pytest.raises(ValueError, match="never opened"):
        spread_pin(repo)


def test_the_closure_of_a_script_of_this_checkout_can_be_shown() -> None:
    """Every import form of this checkout's analysis code is one the closure follows: the summary
    script's closure (the versions and decisions modules, the judge's constants, the packages'
    ``__init__.py``) is shown, and each file of it is there."""
    found = import_closure("scripts/study3_summary.py", reader(REPO_ROOT))
    analysis = "src/zero_field_refusal/analysis"
    assert {f"{analysis}/study3_versions.py", f"{analysis}/study3_decisions.py",
            f"{analysis}/__init__.py", "src/zero_field_refusal/__init__.py",
            "src/zero_field_refusal/judge.py"} <= set(found)  # fmt: skip
    assert all((REPO_ROOT / path).is_file() for path in found)


def test_the_pin_hashes_each_file_and_names_a_change(tmp_path) -> None:
    repo = repo_with(tmp_path, {"scripts/study3_spread.py": "from zero_field_refusal import c\n",
                                "src/zero_field_refusal/__init__.py": "",
                                "src/zero_field_refusal/c.py": "X = 1\n"})  # fmt: skip
    pin = spread_pin(repo)
    assert [e["path"] for e in pin] == ["scripts/study3_spread.py",
        "src/zero_field_refusal/__init__.py", "src/zero_field_refusal/c.py"]  # fmt: skip
    assert pin_problems(pin, repo) == []
    (repo / "src/zero_field_refusal/c.py").write_text("X = 2\n")
    (problem,) = pin_problems(pin, repo)
    assert "src/zero_field_refusal/c.py" in problem and pin_problems(pin, repo, now=False) == []


def test_the_pin_at_a_commit_reads_the_commit(tmp_path) -> None:
    """With a commit, the closure and the hashes are the commit's, whatever the files are now."""
    repo = repo_with(tmp_path, {"scripts/study3_spread.py": "from zero_field_refusal import c\n",
                                "src/zero_field_refusal/c.py": "X = 1\n"})  # fmt: skip

    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=t@example.org",
                               "-c", "commit.gpgsign=false", *args], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip

    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "the spread")
    commit = git("rev-parse", "HEAD")
    pin = spread_pin(repo, commit)
    assert pin == spread_pin(repo)
    (repo / "scripts/study3_spread.py").write_text("x = 2\n")
    assert spread_pin(repo, commit) == pin != spread_pin(repo)
    assert [e["path"] for e in spread_pin(repo)] == ["scripts/study3_spread.py"]
    with pytest.raises(ValueError, match="scripts/study3_spread.py"):
        spread_pin(repo, "0" * 40)


def test_a_new_import_now_is_named_beside_the_changed_file(tmp_path) -> None:
    """The pin checked now is the files and their closure: a pinned file that imports one more
    module names both the file and the module the closure gained."""
    repo = repo_with(tmp_path, {"scripts/study3_spread.py": "from zero_field_refusal import c\n",
                                "src/zero_field_refusal/c.py": "X = 1\n",
                                "src/zero_field_refusal/d.py": ""})  # fmt: skip
    pin = spread_pin(repo)
    (repo / "src/zero_field_refusal/c.py").write_text("from . import d\n")
    problems = pin_problems(pin, repo)
    assert any("src/zero_field_refusal/c.py" in p and "sha256" in p for p in problems), problems
    assert any("src/zero_field_refusal/d.py" in p and "closure" in p for p in problems), problems


@pytest.mark.parametrize("pin, words", [
    ([], "non-empty list"),
    ("scripts/study3_spread.py", "non-empty list"),
    ([{"path": "src/zero_field_refusal/c.py", "sha256": "0" * 64}], SPREAD_SCRIPT),
    ([{"path": SPREAD_SCRIPT, "sha256": "x"}], "sha256"),
    ([{"path": SPREAD_SCRIPT, "sha256": "0" * 64, "text": "x = 1"}], "text"),
    ([{"path": SPREAD_SCRIPT, "sha256": "0" * 64},
      {"path": "src/zero_field_refusal/../../etc/c.py", "sha256": "0" * 64}], "[1].path"),
    ([{"path": SPREAD_SCRIPT, "sha256": "0" * 64},
      {"path": "configs/judge/wildguard.yaml", "sha256": "0" * 64}], "[1].path"),
    ([{"path": SPREAD_SCRIPT, "sha256": "0" * 64}, {"path": SPREAD_SCRIPT, "sha256": "0" * 64}],
     "order"),
    ([{"path": "src/zero_field_refusal/a.py", "sha256": "0" * 64},
      {"path": SPREAD_SCRIPT, "sha256": "0" * 64}], "order"),
])  # fmt: skip
def test_the_pins_form(tmp_path, pin, words) -> None:
    """A pin is a non-empty list of ``{path, sha256}`` in the order of its paths, each once, that
    names the spread script; every other path a module under src/zero_field_refusal/."""
    found = pin_problems(pin, tmp_path, now=False)
    assert any(words in p for p in found), found


# ------------------------------------------------------------------ the control list's schema


def test_the_control_list_schema(controls) -> None:
    assert control_list_problems(controls) == []
    for broken in (drop_item, flip_expected, extra_field, unsorted, twenty_one_harmful, text_field):
        assert control_list_problems(broken(controls)) != []


@pytest.mark.parametrize("change, words", [
    ({"kind": "controls"}, "kind"),
    ({"what": "other"}, "what"),
    ({"written_at": "2026-10-27T12:00:00+02:00"}, "written_at"),
    ({"code": {"commit": "a" * 40, "gpu_code_sha256": "b" * 64}}, "gpu_code_sha256"),
    ({"seed": 2**32}, "seed"),
    ({"seed": True}, "seed"),
    ({"source": {"sample": {"path": "/abs/sample.jsonl", "sha256": "0" * 64},
                 "labels": {"path": "c/judge-wildguard.jsonl", "sha256": "0" * 64},
                 "archive": "gs://b/c"}}, "source.sample.path"),
    ({"source": {"sample": {"path": "c/sample.jsonl", "sha256": "0" * 64},
                 "labels": {"path": "c/labels.jsonl", "sha256": "0" * 64},
                 "archive": "gs://b/c"}}, "source.labels.path"),
    ({"source": {"sample": {"path": "c/sample.jsonl", "sha256": "0" * 64},
                 "labels": {"path": "c/judge-wildguard.jsonl", "sha256": "0" * 64},
                 "archive": "gs://b/a calibration"}}, "source.archive"),
    ({"judge": {"backend": "wildguard"}}, "judge"),
    ({"items": "none"}, "items"),
])  # fmt: skip
def test_each_field_of_the_control_list_is_checked(controls, change, words) -> None:
    found = control_list_problems({**controls, **change})
    assert any(words in p for p in found), found


@pytest.mark.parametrize("change, words", [
    ({"id": "c00000000000z"}, "id"),
    ({"prompt_sha256": "0" * 63}, "prompt_sha256"),
    ({"label": "unsure"}, "label"),
    ({"expected": "maybe"}, "expected"),
])  # fmt: skip
def test_each_field_of_an_item_is_checked(controls, change, words) -> None:
    controls["items"][2] |= change
    found = control_list_problems(controls)
    assert any(words in p for p in found), found


def test_the_breakers_name_what_they_break(controls) -> None:
    named = {drop_item: "40", flip_expected: "expected", extra_field: "note",
             unsorted: "order", twenty_one_harmful: "20", text_field: "prompt"}  # fmt: skip
    for broken, word in named.items():
        found = control_list_problems(broken(controls))
        assert any(word in p for p in found), (broken.__name__, found)
    assert len(BREAKERS) == len(named)


def test_the_control_list_is_read_from_the_checkout_or_given_bytes(tmp_path, controls) -> None:
    assert read_control_list(tmp_path) == (None, [f"{CONTROLS} is missing"])
    path = tmp_path / CONTROLS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(controls, indent=2) + "\n")
    assert read_control_list(tmp_path) == (controls, [])
    assert read_control_list(tmp_path, data=b"{")[0] is None
    record, problems = read_control_list(tmp_path, data=json.dumps(drop_item(controls)).encode())
    assert record is None and problems
    nan = json.dumps(controls).replace('"seed": 20261027', '"seed": NaN').encode()
    assert read_control_list(tmp_path, data=nan)[0] is None
    twice = json.dumps(controls)[:-1] + ', "seed": 1}'
    assert read_control_list(tmp_path, data=twice.encode())[0] is None
    assert controls["judge"]["backend"] == "wildguard" and PER_CLASS == 20


def test_the_pins_import_no_gpu_module_and_no_label_reader() -> None:
    code = ("import sys; import zero_field_refusal.analysis.study3_pins; "
            "print([m for m in ('torch', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.labeling', 'zero_field_refusal.run_judging', "
            "'zero_field_refusal.analysis.study2_labels') "
            "if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
