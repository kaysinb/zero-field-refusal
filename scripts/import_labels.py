"""Import reference labels made outside the project (e.g. by a model ensemble) for a blind sample.

    python scripts/import_labels.py LABELS.jsonl PROVENANCE_DIR SAMPLE.jsonl [--labeler ensemble]

LABELS.jsonl holds ``{item_id, label, labeled_at}`` per sample item, the format of label.py.
Writes reference_labels.jsonl, a copy of PROVENANCE_DIR as reference_provenance/ and
reference_import.json (hashes of every input) next to SAMPLE.jsonl. These files contain responses
to harmful prompts or labels of them: keep them out of git.
"""

import argparse
import json

from zero_field_refusal.labeling import import_reference_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels")
    parser.add_argument("provenance")
    parser.add_argument("sample")
    parser.add_argument("--labeler", default="ensemble")
    args = parser.parse_args()
    manifest = import_reference_labels(args.labels, args.provenance, args.sample,
                                       labeler=args.labeler)  # fmt: skip
    print(json.dumps({k: manifest[k] for k in ("labeler", "label_counts")}
                     | {"provenance_files": len(manifest["provenance"])}))  # fmt: skip


if __name__ == "__main__":
    main()
