"""Archive local files to ARTIFACT_REMOTE without overwriting, and verify every object.

    python scripts/archive.py INVENTORY.tsv REMOTE_PREFIX LOCAL_ROOT PATH [PATH ...]

Each PATH (a file or a directory, relative to LOCAL_ROOT) lands at REMOTE_PREFIX/PATH, copied with
`gcloud storage cp --no-clobber` under the key ARTIFACT_GCS_KEY as a per-command credential (the
active gcloud account is left alone). Then every object's size and MD5 are compared with the local
file, and one line per file is appended to INVENTORY.tsv: path, size, sha256, md5 (base64),
archive URL, status. An object that already existed with other content shows as a mismatch, never
as overwritten. Exits non-zero on any mismatch. The procedure: docs/operations.md, "Rented
machines".
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HEADER = "path\tsize\tsha256\tmd5_b64\turl\tstatus\n"


def gcloud(args: list[str]) -> str:
    env = {**os.environ, "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": os.environ["ARTIFACT_GCS_KEY"]}
    done = subprocess.run(["gcloud", "storage", *args], env=env, check=True, capture_output=True,
                          text=True)  # fmt: skip
    return done.stdout


def files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(f for f in path.rglob("*") if f.is_file())


def digests(path: Path) -> tuple[int, str, str]:
    data = path.read_bytes()
    md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
    return len(data), hashlib.sha256(data).hexdigest(), md5


def archive(inventory: Path, prefix: str, root: Path, paths: list[str]) -> dict[str, int]:
    prefix = prefix.rstrip("/")
    local = {f.relative_to(root).as_posix(): digests(f) for rel in paths for f in files(root / rel)}
    if not local:
        raise SystemExit("Nothing to archive")
    for rel in paths:
        source, target = root / rel, f"{prefix}/{rel}"
        if source.is_dir():  # the directory itself lands at target
            gcloud(["cp", "--no-clobber", "--quiet", "--recursive", str(source),
                    target.rsplit("/", 1)[0] + "/"])  # fmt: skip
        else:
            gcloud(["cp", "--no-clobber", "--quiet", str(source), target])
    remote = {}
    for rel in paths:
        url = f"{prefix}/{rel}" + ("/**" if (root / rel).is_dir() else "")
        for meta in json.loads(gcloud(["objects", "list", url, "--format=json"])):
            remote[f"gs://{meta['bucket']}/{meta['name']}"] = (int(meta["size"]),
                                                               meta.get("md5_hash"))  # fmt: skip
    bad = 0
    new = not inventory.exists()
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("a") as out:
        if new:
            out.write(HEADER)
        for rel, (size, sha, md5) in sorted(local.items()):
            url = f"{prefix}/{rel}"
            rsize, rmd5 = remote.get(url, (None, None))
            ok = (rsize, rmd5) == (size, md5)
            bad += not ok
            status = "ok" if ok else f"MISMATCH remote={rsize},{rmd5}"
            out.write(f"{rel}\t{size}\t{sha}\t{md5}\t{url}\t{status}\n")
    return {"files": len(local), "verified": len(local) - bad, "mismatched": bad}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inventory", type=Path)
    parser.add_argument("prefix", help="e.g. $ARTIFACT_REMOTE (a gs:// path)")
    parser.add_argument("root", type=Path)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    if not args.prefix.startswith("gs://"):
        raise SystemExit("REMOTE_PREFIX must be a gs:// path")
    result = archive(args.inventory, args.prefix, args.root, args.paths)
    print(json.dumps(result))
    if result["mismatched"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
