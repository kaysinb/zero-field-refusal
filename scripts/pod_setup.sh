#!/usr/bin/env bash
# Prepare a Pod for project runs: report GPU, RAM and disk; install the pinned Google Cloud CLI on
# the container disk; decode the archive key; restore into DATA_DIR the files that the given
# archive inventories list under data/, and verify them. Run on the Pod, from the checkout, after
# its deadline is set:
#
#   bash scripts/pod_setup.sh INVENTORY [INVENTORY ...]   # bucket paths of inventories
#   source /root/zfr-env.sh                               # PATH and ARTIFACT_GCS_KEY
#
# Only the listed files are restored, so the bucket may hold others (a superseded build) that
# the Pod never sees. Before anything is copied, every listed path must stay inside DATA_DIR (no
# absolute path, no "..", "." or empty component) and DATA_DIR must hold no symbolic link. Every
# file in DATA_DIR must then be listed, with its sha256, by one of the inventories, and every
# listed file must be present; on a restarted Pod pass the inventories of what it already holds
# too. Run it alone: nothing else may write to DATA_DIR while it checks and restores (the checks
# do not guard against a concurrent writer). The inventories for each stage are named in
# docs/operations.md. Needs ARTIFACT_GCS_KEY_B64 and ARTIFACT_REMOTE in the Pod environment
# (docs/operations.md, "Storage"); prints neither. Existing files in DATA_DIR are never
# overwritten, only checked.
set -euo pipefail

GCLOUD_VERSION=584.0.0
GCLOUD_SHA256=02f0a54a1c5f9e582c568cd8da8258e22275b8ab033532fa03fd46bd9f7a3398
GCLOUD_URL=https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-x86_64.tar.gz
GCLOUD=/root/google-cloud-sdk/bin/gcloud
DATA_DIR=${DATA_DIR:-/workspace/data/zero-field-refusal}
KEY=/root/.artifact-gcs-key.json
INVENTORIES=("$@")
[ ${#INVENTORIES[@]} -gt 0 ] || { echo "usage: pod_setup.sh INVENTORY [INVENTORY ...]" >&2; exit 2; }
: "${ARTIFACT_REMOTE:?set ARTIFACT_REMOTE}"
: "${ARTIFACT_GCS_KEY_B64:?set ARTIFACT_GCS_KEY_B64 from the Runpod secret gcs_archive_key_b64}"

echo "== machine"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free -g | awk '/^Mem:/ {print "host RAM (GB):", $2}'
df -h /workspace | tail -1

echo "== gcloud ${GCLOUD_VERSION}"
installed=$("$GCLOUD" version --format='value("Google Cloud SDK")' 2>/dev/null || true)
if [ "$installed" != "$GCLOUD_VERSION" ]; then
  rm -rf /root/google-cloud-sdk
  tmp=$(mktemp -d)
  curl -sSfL -o "$tmp/gcloud.tar.gz" "$GCLOUD_URL"
  echo "${GCLOUD_SHA256}  $tmp/gcloud.tar.gz" | sha256sum -c --quiet -
  tar -xzf "$tmp/gcloud.tar.gz" -C /root
  rm -rf "$tmp"
fi
installed=$("$GCLOUD" version --format='value("Google Cloud SDK")')
[ "$installed" = "$GCLOUD_VERSION" ] || { echo "gcloud is $installed, not $GCLOUD_VERSION" >&2; exit 1; }
echo "gcloud $installed"

echo "== archive key"
( umask 077; printenv ARTIFACT_GCS_KEY_B64 | base64 -d > "$KEY" )
chmod 600 "$KEY"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d.get('type') == 'service_account'" "$KEY"
cat > /root/zfr-env.sh <<EOF
export PATH=/root/google-cloud-sdk/bin:\$PATH
export ARTIFACT_GCS_KEY=$KEY
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=False
EOF
# shellcheck disable=SC1091
source /root/zfr-env.sh

echo "== restore the inventories' data/ files into DATA_DIR"
mkdir -p "$DATA_DIR"
copies=$(mktemp -d)
i=0
for inventory in "${INVENTORIES[@]}"; do  # numbered copies: distinct inventories never collide
  i=$((i + 1))
  CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$KEY" gcloud storage cp --quiet \
    "$ARTIFACT_REMOTE/$inventory" "$copies/$i.tsv"
done
# The data/ entries of the inventories, checked before anything is copied: every path must stay
# inside DATA_DIR (no absolute path, no empty, "." or ".." component), DATA_DIR holds no symbolic
# link, and the inventories agree on each file's sha256. Writes url<TAB>path lines to restore.
python3 - "$ARTIFACT_REMOTE" "$DATA_DIR" "$copies/expected.json" "$copies"/*.tsv \
  > "$copies/restore.tsv" <<'PY'
import json, os, sys
from pathlib import Path

prefix = sys.argv[1].rstrip("/") + "/data/"
data_dir, out, inventories = Path(sys.argv[2]).resolve(), Path(sys.argv[3]), sys.argv[4:]


def unreadable(error):  # a directory the walk cannot read is never skipped silently
    sys.exit(f"Cannot read {error.filename} in DATA_DIR: {error.strerror}")


links = [os.path.join(top, name) for top, dirs, files in os.walk(data_dir, onerror=unreadable)
         for name in dirs + files if os.path.islink(os.path.join(top, name))]
if links:
    sys.exit(f"DATA_DIR holds symbolic links: {links[:5]}")
expected = {}
for inventory in inventories:
    for line in Path(inventory).read_text().splitlines()[1:]:
        cols = line.split("\t")
        sha, url = cols[2], cols[4]
        if not url.startswith(prefix):
            continue
        rel = url[len(prefix):]
        parts = rel.split("/")
        if rel.startswith("/") or any(part in ("", ".", "..") for part in parts) \
                or not (data_dir / rel).resolve().is_relative_to(data_dir):
            sys.exit(f"Unsafe inventory path {rel!r}: it must stay inside DATA_DIR")
        if expected.setdefault(rel, sha) != sha:
            sys.exit(f"The inventories disagree on {rel}")
        print(f"{url}\t{rel}")
out.write_text(json.dumps(expected))
PY
sort -u "$copies/restore.tsv" | while IFS=$'\t' read -r url rel; do
  [ -e "$DATA_DIR/$rel" ] && continue  # never overwritten; checked below
  mkdir -p "$(dirname "$DATA_DIR/$rel")"
  CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$KEY" gcloud storage cp --no-clobber --quiet "$url" \
    "$DATA_DIR/$rel"
done
python3 - "$DATA_DIR" "$copies/expected.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

data_dir, expected = Path(sys.argv[1]).resolve(), json.loads(Path(sys.argv[2]).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def unreadable(error):  # a directory the walk cannot read is never skipped silently
    sys.exit(f"Cannot read {error.filename} in DATA_DIR: {error.strerror}")


present, links = set(), []
for top, dirs, files in os.walk(data_dir, onerror=unreadable):  # never follows a link
    for name in dirs + files:
        path = Path(top, name)
        if path.is_symlink():
            links.append(path.relative_to(data_dir).as_posix())
        elif name in files:
            present.add(path.relative_to(data_dir).as_posix())
uncovered = sorted(present - set(expected))
missing = sorted(set(expected) - present)
bad = sorted(rel for rel in present & set(expected) if sha256(data_dir / rel) != expected[rel])
print(f"{len(present & set(expected)) - len(bad)} of {len(present)} files in DATA_DIR verified "
      f"against {len(expected)} inventory entries")
for label, items in (("NOT IN ANY INVENTORY", uncovered), ("MISSING", missing),
                     ("SHA256 MISMATCH", bad), ("SYMBOLIC LINK", links)):
    for rel in items:
        print(f"{label} {rel}")
sys.exit(1 if not expected or uncovered or missing or bad or links else 0)
PY
rm -rf "$copies"
echo "== done; next: source /root/zfr-env.sh"
