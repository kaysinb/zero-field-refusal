#!/usr/bin/env bash
# Prepare a Pod for project runs: report GPU, RAM and disk; install the pinned Google Cloud CLI on
# the container disk; decode the archive key; restore DATA_DIR from the bucket and verify every
# file against the archive inventory. Run on the Pod, from the checkout, after its deadline is set:
#
#   bash scripts/pod_setup.sh [INVENTORY_PATH]     # default runs/archive/inventory-2026-10-11.tsv
#   source /root/zfr-env.sh                         # PATH and ARTIFACT_GCS_KEY for later commands
#
# Needs ARTIFACT_GCS_KEY_B64 and ARTIFACT_REMOTE in the Pod environment (docs/operations.md,
# "Storage"); prints neither. Existing files in DATA_DIR are never overwritten.
set -euo pipefail

GCLOUD_VERSION=584.0.0
GCLOUD_SHA256=02f0a54a1c5f9e582c568cd8da8258e22275b8ab033532fa03fd46bd9f7a3398
GCLOUD_URL=https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${GCLOUD_VERSION}-linux-x86_64.tar.gz
INVENTORY=${1:-runs/archive/inventory-2026-10-11.tsv}
DATA_DIR=${DATA_DIR:-/workspace/data/zero-field-refusal}
KEY=/root/.artifact-gcs-key.json
: "${ARTIFACT_REMOTE:?set ARTIFACT_REMOTE}"
: "${ARTIFACT_GCS_KEY_B64:?set ARTIFACT_GCS_KEY_B64 from the Runpod secret gcs_archive_key_b64}"

echo "== machine"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free -g | awk '/^Mem:/ {print "host RAM (GB):", $2}'
df -h /workspace | tail -1

echo "== gcloud ${GCLOUD_VERSION}"
if [ ! -x /root/google-cloud-sdk/bin/gcloud ]; then
  tmp=$(mktemp -d)
  curl -sSfL -o "$tmp/gcloud.tar.gz" "$GCLOUD_URL"
  echo "${GCLOUD_SHA256}  $tmp/gcloud.tar.gz" | sha256sum -c --quiet -
  tar -xzf "$tmp/gcloud.tar.gz" -C /root
  rm -rf "$tmp"
fi
/root/google-cloud-sdk/bin/gcloud version --format='value("Google Cloud SDK")'

echo "== archive key"
( umask 077; printenv ARTIFACT_GCS_KEY_B64 | base64 -d > "$KEY" )
chmod 600 "$KEY"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d.get('type') == 'service_account'" "$KEY"
cat > /root/zfr-env.sh <<EOF
export PATH=/root/google-cloud-sdk/bin:\$PATH
export ARTIFACT_GCS_KEY=$KEY
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
EOF
# shellcheck disable=SC1091
source /root/zfr-env.sh

echo "== restore DATA_DIR from the bucket"
mkdir -p "$DATA_DIR"
CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$KEY" gcloud storage cp --recursive --no-clobber --quiet \
  "$ARTIFACT_REMOTE/data/*" "$DATA_DIR/"
inventory_copy=$(mktemp)
CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$KEY" gcloud storage cp --quiet \
  "$ARTIFACT_REMOTE/$INVENTORY" "$inventory_copy"
python3 - "$inventory_copy" "$DATA_DIR" <<'PY'
import hashlib, sys
from pathlib import Path
inventory, data_dir = sys.argv[1], Path(sys.argv[2])
checked = bad = 0
for line in Path(inventory).read_text().splitlines()[1:]:
    path, size, sha = line.split("\t")[:3]
    if not path.startswith("data/"):
        continue
    local = data_dir / path.removeprefix("data/")
    ok = local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == sha
    checked += 1
    bad += not ok
    if not ok:
        print(f"MISMATCH {path}")
print(f"verified {checked - bad} of {checked} data files against the inventory")
sys.exit(1 if bad else 0)
PY
rm -f "$inventory_copy"
echo "== done; next: source /root/zfr-env.sh"
