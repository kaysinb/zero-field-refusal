#!/usr/bin/python3.12
"""Run a source checkout with the dependencies baked into this container image."""

import hashlib
import os
import sys
import tomllib
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: project-run REPOSITORY COMMAND [ARG ...]")
    repo = Path(sys.argv[1]).resolve()
    baked = Path("/opt/project-deps")
    for filename in ("pyproject.toml", "uv.lock"):
        source = repo / filename
        if not source.is_file():
            raise SystemExit(f"Missing {source}")
        if (
            hashlib.sha256(source.read_bytes()).digest()
            != hashlib.sha256((baked / filename).read_bytes()).digest()
        ):
            raise SystemExit(f"{filename} differs from the image; rebuild the dependency image")
    if not (repo / "src").is_dir() or not (repo / "configs").is_dir():
        raise SystemExit("Expected a source checkout containing src/ and configs/")
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["name"]
    env = os.environ.copy()
    env["PATH"] = "/opt/venv/bin:" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["UV_PROJECT_ENVIRONMENT"] = "/opt/venv"
    env["PYTHONPATH"] = str(repo / "src")
    # Respect explicitly supplied storage locations, including project-specific mounts.
    env.setdefault("DATA_DIR", f"/workspace/data/{project}")
    env.setdefault("RUNS_DIR", f"/workspace/runs/{project}")
    env.setdefault("HF_HOME", "/workspace/cache/huggingface")
    env.setdefault("XDG_CACHE_HOME", "/workspace/cache")
    # Trust only the explicitly selected mounted checkout for Git provenance reads.
    count = int(env.get("GIT_CONFIG_COUNT", "0"))
    env[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{count}"] = str(repo)
    env["GIT_CONFIG_COUNT"] = str(count + 1)
    os.chdir(repo)
    os.execvpe(sys.argv[2], sys.argv[2:], env)


if __name__ == "__main__":
    main()
