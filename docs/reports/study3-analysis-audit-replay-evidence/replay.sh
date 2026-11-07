#!/usr/bin/env bash
# Study 3: the analysis and the audit replayed on synthetic records in the Pod image, on CPU,
# from an uninstalled checkout through project-run, one process per test file.
set -u
L=$(cd "$(dirname "$0")" && pwd)
IMAGE=ghcr.io/kaysinb/zero-field-refusal@sha256:9554679e749e405ad4d85e5a3c3416e687f8c3e7d443d289a18f794704810da5
FILES="test_study3_freeze test_study3_labels test_study3_final test_study3_estimates test_study3_analysis_script
test_study3_audit test_study3_audit_script test_study3_report test_study3_package2 test_study3_label_cli test_study3_spread"
for f in $FILES; do
  start=$(date +%s.%N)
  docker run --rm --network none --entrypoint project-run \
    -e CUDA_VISIBLE_DEVICES= -e CONTAINER_IMAGE="$IMAGE" \
    --mount type=bind,src="$L/source",dst=/workspace/code/zero-field-refusal,readonly \
    --mount type=bind,src="$L/out",dst=/out \
    "$IMAGE" /workspace/code/zero-field-refusal \
    python -m pytest -q -p no:cacheprovider --junitxml=/out/$f.xml tests/$f.py > "$L/out/$f.log" 2>&1
  code=$?
  end=$(date +%s.%N)
  printf '%s\t%s\t%s\n' "$f" "$code" "$(echo "$end - $start" | bc)" >> "$L/out/summary.tsv"
  tail -1 "$L/out/$f.log"
done
echo REPLAY_END
