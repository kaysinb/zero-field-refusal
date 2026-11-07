# Study 3: the analysis and the audit replayed before the confirmation's labels

[Amendment 7](../study3-preregistration.md#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run)
orders the dry run's pieces: each runs before its first consequential use, on synthetic records, in the Pod image,
from an uninstalled checkout, through `project-run`. The analysis and the audit run before the confirmation's labels.
This is the record of that replay. No confirmation label existed when it ran, and none was opened.

- **Commit:** `fe06c592f066f19488ca274a4a39ed3f9c63e303` (stage0), with the GPU code pinned at `fd54945`. The commit holds the analysis freeze writer,
  the label readers over the labeling command's ledger, the judge-based analysis with validation's harm, the audit
  under [amendment 11](../study3-preregistration.md#amendment-11-the-audits-reference-annotators) and the report.
- **Image:** `ghcr.io/kaysinb/zero-field-refusal@sha256:9554679e749e405ad4d85e5a3c3416e687f8c3e7d443d289a18f794704810da5`,
  the confirmation's.
- **Where:** on CPU, on 2026-11-07, `linux/amd64`, with no network and `CUDA_VISIBLE_DEVICES` empty. The checkout was a
  clean clone at the commit, mounted read-only. Each test file ran in its own process, through `project-run`
  ([`replay.sh`](study3-analysis-audit-replay-evidence/replay.sh)).

## The results

| File | Tests | Failed | Errors | Skipped | Seconds | JUnit sha256 |
|---|---:|---:|---:|---:|---:|---|
| `test_study3_freeze` | 21 | 0 | 0 | 0 | 10.5 | `f35d5dfb8f1e1b22e503f42fdb98d59d8027f185a77da696ca6ceafa7eb43af6` |
| `test_study3_labels` | 31 | 0 | 0 | 0 | 15.9 | `f0c5b0b20283bd067f3dae381ec328b4d674c173ec0c45f625ab7dfef4a612b0` |
| `test_study3_final` | 33 | 0 | 0 | 0 | 5.8 | `08e6a9f29c070f8c7a654e60a79b2f499513fe112ed734691d2b989eb85e38f8` |
| `test_study3_estimates` | 48 | 0 | 0 | 0 | 1.4 | `0b4af6745a5518d70e65bd6558e884be9881ce150d6ee6069837372d0b9f783e` |
| `test_study3_analysis_script` | 12 | 0 | 0 | 0 | 9.6 | `ddb9e690d0264f2c94d329c84519982f5a304d76591c2d2342a812d9ba8d5b0a` |
| `test_study3_audit` | 29 | 0 | 0 | 0 | 9.3 | `8fdf4939667c78be061a6a06d910404e0789d7edc284a9461e5ff97b929b67f8` |
| `test_study3_audit_script` | 17 | 0 | 0 | 0 | 53.4 | `1c6bf8e59c4bda16aaa4242d9dfe539f11da6e4bd52aa205f7d9d09daf2b9b3b` |
| `test_study3_report` | 33 | 0 | 0 | 0 | 11.1 | `d63eadf1474a172120324863928f78520a1b51cf485c79f9bcdad9c90105e26c` |
| `test_study3_package2` | 36 | 0 | 0 | 0 | 242.6 | `fef7ed403744e904bc4834e48c1869ad1cc0cdba3ed7b2d9d5259d22761c9cbe` |
| `test_study3_label_cli` | 13 | 0 | 0 | 0 | 6.8 | `7f5ddcf117c5b276c8c37786c4ad389e2a90006b0b722c7c082728bd2bceeeea` |
| `test_study3_spread` | 14 | 0 | 0 | 0 | 29.8 | `c48030dc0a999036207bdd443fd511a078aa9478172b1a6d4d72d02ab2fdd9fb` |

287 tests, none failed or skipped. The JUnit files are in
[`study3-analysis-audit-replay-evidence/`](study3-analysis-audit-replay-evidence/).

## What this covers

- **The freeze and its gate:** the cohorts and K\*_h written once from label-free records, refused on any trace of a
  confirmation label, and the gate (`freeze_open`) that the labeling command and the final readers pass. The labeling
  command's and the spread's tests are in the set because the gate and the estimates they call changed.
- **The label readers:** a run's labels read only through an acceptance of the labeling command's ledger, both files
  checked against it; pending runs and technical labeling problems named, never counted as missing.
- **The analysis:** the three computations, the missing-labels envelope and validation's harm, through the analysis
  script's `freeze`, `analyze` and `report` on a synthetic study.
- **The audit:** the draw, the blind inputs and the reference under amendment 11 (two blind annotators, the third on
  their disagreements and failures, the arbiter on anonymous votes in a recorded order, the 10 % check of the
  agreements), the Hansen–Hurwitz estimates with their variance from every draw of an arm, through the audit script's
  `draw`, `agreements`, `arbiter-input` and `estimate`.
- **The report:** its facts and tables at each outcome, with a missing audit or amount pending.

The records are synthetic: this shows the code paths run as tested in the Pod image, not any property of the
confirmation's labels.
