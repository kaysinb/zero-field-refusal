# Study 3: the CPU replay at the final pin

[Amendment 6](../study3-preregistration.md#amendment-6-replaying-orchestration-changes-on-cpu) asks for the changed
branches to be replayed on CPU in the Pod image and for the tests, the commit and the results to be recorded.
[Amendment 8](../study3-preregistration.md#amendment-8-the-four-imposed-shifts-become-descriptive) changed the stopping
and reading paths. This is the record of that replay, ahead of the search of candidate 1. It stands beside the
[engineering report](study3-engineering.md), which is not changed.

- **Commit:** `3ce22b6ec2b550ea9aed15bfc9a4630e52a70400`, with the GPU code pinned at `fd54945`, identity
  `717cd02317526179a821eb8eb57e85cefba9e49083108922013b2b353c45ebbc`.
- **Image:** `ghcr.io/kaysinb/zero-field-refusal@sha256:9554679e749e405ad4d85e5a3c3416e687f8c3e7d443d289a18f794704810da5`.
- **Where:** Pod `piixrsw0w2dz5n`, Runpod OC-AU-1, on CPU, on 2026-11-01, one process per file.

## The results

| File | Tests | Failed | Errors | Skipped | Seconds | JUnit sha256 |
|---|---:|---:|---:|---:|---:|---|
| `test_qualification` (first attempt) | 101 | 1 | 0 | 0 | 210.1 | `0f76543507b5d0841d06dc5106ea44d224d0eeb40a9a3ecabf28a59e3c351efb` |
| `test_qualification` (second attempt) | 101 | 0 | 0 | 0 | 185.9 | `7bc786860208ecf10f20597d55afe19ce785c29380653d0906615cee1a9bf5fd` |
| `test_qualification_records` | 76 | 0 | 0 | 0 | 494.2 | `c5c93df31102d7b94b5d30488d6a020359df5940b8a4cba41df071338cd47155` |
| `test_qualify` | 227 | 0 | 0 | 0 | 808.5 | `d805a264071176b38a8642854d0f1215d38409cd70205ffb7e2c4ee255cc8cba` |
| `test_study3_progress` | 222 | 0 | 0 | 1 | 229.0 | `e0290d7c7681311a1d924305e24df430513e3a02c1b7f2b0830b2ef742fae161` |
| `test_study3_replay` | 10 | 0 | 0 | 0 | 1997.9 | `e61e63c83649ebd2b27f2be869e3ffef5b8bcda0172ca7196f83aa1969d27a68` |

- **The failure in the first attempt.** `test_the_launcher_run_needs_the_harness_code_and_environment` unset
  `CONTAINER_IMAGE` for its remaining assertions, and its fixtures had recorded the image the Pod sets, so a run recorded
  under that image read as another image's. It lies in the test, not in the code it checks: the failure reproduces on
  another machine with the variable set, and the test passes there without it. The second attempt ran the file with the
  variable unset. The test now unsets it for its own assertion alone (`b728ff5`), which leaves the GPU-code identity as
  it was.
- **The skip.** `test_a_run_directory_the_reader_may_not_search_blocks`: file permissions do not bind a superuser, and
  the Pod runs as root.

## What this covers

The tests run the battery on a tiny CPU model and read synthetic records. They are not a GPU demonstration: the rest
battery past an exceedance first runs on a GPU in the numerical qualification.
