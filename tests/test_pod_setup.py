"""The inventory checks of scripts/pod_setup.sh, run as the script runs them (its Python heredocs):
paths that leave DATA_DIR and symbolic links are refused before anything is copied, and the
final verification counts only regular files."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_setup.sh"
REMOTE = "gs://bucket/project"
HEADER = "path\tsize\tsha256\tmd5_b64\turl\tstatus\n"


def heredocs() -> list[str]:
    blocks, current = [], None
    for line in SCRIPT.read_text().splitlines():
        if current is None and line.rstrip().endswith("<<'PY'"):
            current = []
        elif current is not None and line == "PY":
            blocks.append("\n".join(current) + "\n")
            current = None
        elif current is not None:
            current.append(line)
    return blocks


CHECK, VERIFY = heredocs()


def inventory(path: Path, entries: list[tuple[str, str]]) -> Path:
    lines = [f"{rel}\t1\t{sha}\tmd5\t{REMOTE}/{rel}\tok\n" for rel, sha in entries]
    path.write_text(HEADER + "".join(lines))
    return path


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def check(data_dir: Path, tmp_path: Path, *inventories: Path):
    expected = tmp_path / "expected.json"
    result = subprocess.run(
        [sys.executable, "-", REMOTE, str(data_dir), str(expected), *map(str, inventories)],
        input=CHECK, capture_output=True, text=True,
    )  # fmt: skip
    return result, expected


def verify(data_dir: Path, expected: Path):
    return subprocess.run(
        [sys.executable, "-", str(data_dir), str(expected)],
        input=VERIFY, capture_output=True, text=True,
    )  # fmt: skip


def test_the_script_has_both_checks():
    assert len(heredocs()) == 2


def test_listed_data_files_are_restored_and_others_ignored(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    inv = inventory(tmp_path / "1.tsv", [("data/a/x.json", sha("x")), ("data/b.jsonl", sha("b")),
                                         ("runs/r/meta.json", sha("r"))])  # fmt: skip
    result, expected = check(data, tmp_path, inv)
    assert result.returncode == 0, result.stderr
    assert sorted(result.stdout.splitlines()) == [
        f"{REMOTE}/data/a/x.json\ta/x.json",
        f"{REMOTE}/data/b.jsonl\tb.jsonl",
    ]
    assert json.loads(expected.read_text()) == {"a/x.json": sha("x"), "b.jsonl": sha("b")}


@pytest.mark.parametrize(
    "rel", ["data/../escape.json", "data/a/../../escape.json", "data//etc/passwd",
            "data/a//b.json", "data/./a.json", "data/a/"],
)  # fmt: skip
def test_a_path_that_leaves_data_dir_is_refused(tmp_path, rel):
    data = tmp_path / "data"
    data.mkdir()
    result, expected = check(data, tmp_path, inventory(tmp_path / "1.tsv", [(rel, sha("x"))]))
    assert result.returncode != 0
    assert "Unsafe inventory path" in result.stderr
    assert result.stdout == "" and not expected.exists()


def test_a_symbolic_link_in_data_dir_is_refused_before_copying(tmp_path):
    data, outside = tmp_path / "data", tmp_path / "outside"
    data.mkdir()
    outside.mkdir()
    os.symlink(outside, data / "teacher")
    inv = inventory(tmp_path / "1.tsv", [("data/teacher/x.json", sha("x"))])
    result, _ = check(data, tmp_path, inv)
    assert result.returncode != 0
    assert "symbolic links" in result.stderr
    assert result.stdout == ""


def test_inventories_that_disagree_are_refused(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    first = inventory(tmp_path / "1.tsv", [("data/x.json", sha("x"))])
    second = inventory(tmp_path / "2.tsv", [("data/x.json", sha("y"))])
    result, _ = check(data, tmp_path, first, second)
    assert result.returncode != 0
    assert "disagree" in result.stderr


def restored(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    (data / "a").mkdir(parents=True)
    (data / "a" / "x.json").write_text("x")
    inv = inventory(tmp_path / "1.tsv", [("data/a/x.json", sha("x"))])
    result, expected = check(data, tmp_path, inv)
    assert result.returncode == 0, result.stderr
    return data, expected


def test_verification_passes_on_exactly_the_listed_files(tmp_path):
    data, expected = restored(tmp_path)
    result = verify(data, expected)
    assert result.returncode == 0, result.stdout
    assert "1 of 1 files in DATA_DIR verified against 1 inventory entries" in result.stdout


def test_verification_names_unlisted_mismatched_and_linked_files(tmp_path):
    data, expected = restored(tmp_path)
    (data / "extra.json").write_text("e")
    result = verify(data, expected)
    assert result.returncode == 1 and "NOT IN ANY INVENTORY extra.json" in result.stdout
    (data / "extra.json").unlink()
    (data / "a" / "x.json").write_text("changed")
    result = verify(data, expected)
    assert result.returncode == 1 and "SHA256 MISMATCH a/x.json" in result.stdout
    (data / "a" / "x.json").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "y.json").write_text("y")
    os.symlink(outside, data / "linked")
    result = verify(data, expected)
    assert result.returncode == 1 and "SYMBOLIC LINK linked" in result.stdout
    assert "linked/y.json" not in result.stdout  # a linked directory is never walked
