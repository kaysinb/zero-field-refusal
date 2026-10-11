"""scripts/archive.py against a fake gcloud that keeps a bucket in a directory."""

import base64
import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "archive.py"

# `gcloud storage cp [--no-clobber] [--quiet] [--recursive] SRC DST` and
# `gcloud storage objects list URL --format=json`, on FAKE_BUCKETS/<bucket>/<name>.
FAKE_GCLOUD = r"""#!PYTHON
import base64, hashlib, json, os, shutil, sys
from pathlib import Path
assert os.environ["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"]
assert os.environ["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"] == "False"
store = Path(os.environ["FAKE_BUCKETS"])
args = sys.argv[1:]
assert args[0] == "storage"
def local(url):
    return store / url.removeprefix("gs://")
if args[1] == "cp":
    flags = {a for a in args if a.startswith("--")}
    src, dst = [a for a in args[2:] if not a.startswith("--")]
    def put(source, target):
        if target.exists() and "--no-clobber" in flags:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    src = Path(src)
    if src.is_dir():
        for f in src.rglob("*"):
            if f.is_file():
                put(f, local(dst) / src.name / f.relative_to(src))
    else:
        put(src, local(dst))
elif args[1:3] == ["objects", "list"]:
    url = args[3]
    base = local(url.removesuffix("/**"))
    found = [p for p in base.rglob("*") if p.is_file()] if url.endswith("/**") else (
        [base] if base.is_file() else [])
    out = []
    for p in found:
        rel = p.relative_to(store).as_posix()
        bucket, name = rel.split("/", 1)
        data = p.read_bytes()
        out.append({"bucket": bucket, "name": name, "size": str(len(data)),
                    "md5_hash": base64.b64encode(hashlib.md5(data).digest()).decode()})
    print(json.dumps(out))
else:
    sys.exit(f"unsupported: {args}")
"""


def setup(tmp_path: Path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(FAKE_GCLOUD.replace("PYTHON", sys.executable, 1))
    gcloud.chmod(gcloud.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "buckets").mkdir()
    key = tmp_path / "key.json"
    key.write_text("{}")
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_BUCKETS": str(tmp_path / "buckets"), "ARTIFACT_GCS_KEY": str(key)}  # fmt: skip


def run(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], env=env, capture_output=True,
                          text=True)  # fmt: skip


def md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def test_files_and_directories_are_archived_verified_and_inventoried(tmp_path) -> None:
    env = setup(tmp_path)
    root = tmp_path / "local"
    (root / "runs" / "exp" / "r1").mkdir(parents=True)
    (root / "runs" / "exp" / "r1" / "meta.json").write_bytes(b'{"status": "completed"}')
    (root / "runs" / "exp" / "r1" / "summary.json").write_bytes(b"{}")
    (root / "notes.md").write_bytes(b"notes")
    inventory = tmp_path / "inventory.tsv"
    done = run(env, str(inventory), "gs://bucket/project", str(root), "runs/exp", "notes.md")
    assert done.returncode == 0, done.stderr
    assert '"verified": 3' in done.stdout
    stored = tmp_path / "buckets" / "bucket" / "project"
    assert (stored / "runs" / "exp" / "r1" / "meta.json").read_bytes() == b'{"status": "completed"}'
    lines = inventory.read_text().splitlines()
    assert lines[0].split("\t") == ["path", "size", "sha256", "md5_b64", "url", "status"]
    rows = {line.split("\t")[0]: line.split("\t") for line in lines[1:]}
    assert set(rows) == {"runs/exp/r1/meta.json", "runs/exp/r1/summary.json", "notes.md"}
    notes = rows["notes.md"]
    assert notes[1:] == ["5", hashlib.sha256(b"notes").hexdigest(), md5(b"notes"),
                         "gs://bucket/project/notes.md", "ok"]  # fmt: skip


def test_an_existing_object_is_never_overwritten_and_shows_as_a_mismatch(tmp_path) -> None:
    env = setup(tmp_path)
    root = tmp_path / "local"
    root.mkdir()
    (root / "a.txt").write_bytes(b"first")
    inventory = tmp_path / "inventory.tsv"
    assert run(env, str(inventory), "gs://bucket/p", str(root), "a.txt").returncode == 0
    (root / "a.txt").write_bytes(b"changed")
    done = run(env, str(inventory), "gs://bucket/p", str(root), "a.txt")
    assert done.returncode == 1
    assert (tmp_path / "buckets" / "bucket" / "p" / "a.txt").read_bytes() == b"first"
    last = inventory.read_text().splitlines()[-1].split("\t")
    assert last[-1] == f"MISMATCH remote=5,{md5(b'first')}"
    assert inventory.read_text().count("path\tsize") == 1  # one header


def test_a_prefix_that_is_not_a_bucket_path_is_refused(tmp_path) -> None:
    env = setup(tmp_path)
    done = run(env, str(tmp_path / "inv.tsv"), "/tmp/elsewhere", str(tmp_path), "x")
    assert done.returncode != 0 and "gs://" in done.stderr


def test_paths_are_normalized_and_must_stay_inside_the_root(tmp_path) -> None:
    env = setup(tmp_path)
    root = tmp_path / "local"
    (root / "data" / "teacher").mkdir(parents=True)
    (root / "data" / "teacher" / "s.jsonl").write_bytes(b"x")
    inventory = tmp_path / "inventory.tsv"
    done = run(env, str(inventory), "gs://bucket/p/", str(root), "./data/")
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "buckets" / "bucket" / "p" / "data" / "teacher" / "s.jsonl").is_file()
    assert not (tmp_path / "buckets" / "bucket" / "p" / "data" / "data").exists()
    assert inventory.read_text().splitlines()[1].startswith("data/teacher/s.jsonl\t")
    for bad in ("../outside", "/abs/path", "."):
        done = run(env, str(inventory), "gs://bucket/p", str(root), bad)
        assert done.returncode != 0 and "relative path inside ROOT" in done.stderr
