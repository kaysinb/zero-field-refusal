"""Build the pinned prompt sets of a config into DATA_DIR/prompts/<config hash>/.

Usage: python scripts/prepare_data.py CONFIG [key=value ...] [--force]
"""

import argparse
import hashlib
import json
import sys

from zero_field_refusal.config import load_config
from zero_field_refusal.data import MANIFEST, build_prompt_sets, prompt_sets_dir
from zero_field_refusal.paths import load_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("config", help="config name under configs/ or a path to a YAML file")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    parser.add_argument("--force", action="store_true", help="overwrite an existing build")
    args = parser.parse_args()

    load_env()  # DATA_DIR and HF_HOME from .env
    cfg = load_config(args.config, args.overrides)
    if "data" not in cfg:
        sys.exit(f"{args.config} has no 'data' section")
    out_dir = prompt_sets_dir(cfg.data)
    if (out_dir / MANIFEST).exists() and not args.force:
        sys.exit(f"{out_dir / MANIFEST} already exists; pass --force to rebuild")

    manifest_path = build_prompt_sets(cfg.data, out_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"manifest: {manifest_path}")
    print(f"manifest sha256: {hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")
    print(f"config hash: {manifest['config_hash']}")
    for name, entry in manifest["sets"].items():
        print(f"{name:22s} {entry['count']:6d}  {entry['sha256'][:16]}")
    dedup = manifest["sets"]["harmful_kl"]["dedup"]
    print(
        f"harmful dedup: {dedup['raw']} raw, {dedup['dropped_exact']} exact, "
        f"{dedup['dropped_near_duplicate']} near-duplicate (Jaccard >= "
        f"{dedup['jaccard_threshold']}), {dedup['dropped_empty']} empty, {dedup['kept']} kept; "
        f"by source {dedup['by_source']}"
    )


if __name__ == "__main__":
    main()
