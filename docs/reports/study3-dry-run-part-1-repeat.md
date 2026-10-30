# Study 3: the Pod dry run, Part I repeated for package 1c

Part I of study 3's dry run was repeated on 2026-10-29 for the GPU code of package 1c, on rented
Runpod Pods with one NVIDIA L40S each. Amendment 4 of the [registration](../study3-preregistration.md)
requires it: GPU code changed after Part I, so Part I's GPU steps are repeated for that code. The
repeat covers the affected paths only, as items R1–R8. [Part I's report](study3-dry-run-part-1.md)
stands for the rest.

The first Pod was unfit and was stopped during step 10, with R2 interrupted (incident 1, below). The whole repeat ran on
the second Pod.

Nothing of the repeat counts for study 3:

- every real-model stage seed was the dry seed 90200 (numerical candidate 1's first dry seed),
  outside the registered 10000–10599 (the test layer's fixture seeds and the instruments' fixed
  seeds are not stage seeds);
- every record lies under the dry folder `/workspace/dryrun/study3-pod-repeat-20261029/` of the Pod
  volume and under the bucket's `dryrun/study3-pod-repeat-20261029/` prefix, never under `runs/` or
  `data/`.

No K or D value was read. The checks read exits, states, kinds, file presence, ids, hashes and
paths.

**Verdict.** R1–R8 hold on the Pod, or on the test layer where this report says so. The four
halts were ruled faults of a check or of the procedure, and watch 4 was judged after the fact
(both below). With this repeat, the researcher declares Part I passed for the engineering stage.
The engineering record's `checks.dry_run` is not marked passed: that needs Part II too.

## Summary

- **Checks.** 346 check rows ran under 341 check ids (five ids ran twice): 342 passed and 4 failed.
  - The frozen evidence snapshot holds the first 323 rows (320 passed, 3 failed). The 23 rows from
    `Z-freeze` on were written after the freeze (22 passed, 1 failed); they and the final
    `checks.tsv` are in addendum 1.
  - The four failed rows are `X7-log` (step 34b), `D3-pre-records` (step 44), `R3-watches` (step 51)
    and `Z-upload-a1` (step 54a). Three are halts that the researcher ruled faults of a check or of
    the procedure; each passed when its step ran again. `R3-watches` fails on watch 4's tail and
    stays a finding of the close.
- **Halts.** Four, in `receipts/halts.tsv`. None came from the code under test:
  1. step 34b: an expectation of the procedure was wrong; the code is right;
  2. step 37a: watch 4 ended `failed 1` after a manual read during the live watch;
  3. step 44: a check's file glob also matched the rest's trajectory files;
  4. step 54a: the upload's path list held an empty top-level folder.
- **What the repeat shows on the Pod.**
  - **R2.** The gate selection T′ passed 320 of 320. The dry-seed selection DSRC_T″ failed in exactly
    the 164 expected cases, each in its class. The seed commit's proof holds, on the Pod's own gate
    junit. Part I's `T-selection-T` failure does not recur.
  - **R4.** One case, X, lost its checkpoints four times. It went through three case restarts, four
    numbered recovery rounds (one attempt each) and five launcher attempts, with three wrong rounds
    refused. Every read of X ends with no problem from the rounds. Step 27's child now fails after its
    run directory exists, and its own `failed` status settles the attempt technical.
  - **R5.** Three technical cases exported their evidence before any launcher call:
    - T1 exported its launcher checkpoint, verified, then released it;
    - T2's checkpoint was lost first, and its manifest says so;
    - T3's corrupt bytes were exported as they are, with `matches_record` false.
  - **R6.** A real FA case, D, lost its checkpoints after its rest. The driver then ran its same-path
    control alone, with no launcher call. Attempt 2 of the control ran by hand, and the case
    finished. D also gave R4's check 5.
  - **R7.** The launcher from another commit skipped T1's completed run, and the journal stayed
    unchanged.
  - **R3.** Seven watches covered the empty `responses/` folders beside the GPU jobs. Watch 4 is
    judged on the window [21:36:51.391183Z, 21:53:41.000000Z), which ends at the manual
    measurement's independently recorded time, by a ruling made after the fact.
- **Test layer only.** P1 (`restart-case` never over a binding refusal), the decided state (a case
  decided by its baseline, its control due) and C2 (the driver's unset bucket variables).
- **Incident 1.** The first Pod, `2xyqxu4mgc0yos` (US-MO-1), had its volume on MooseFS, on a slow
  host. Its 1.016 Pod-hours are the incident. Startup checks 3b and 6b were added after it. Pods that
  hold records need a local volume.
- **Time.**
  - GPU jobs took 88.6 min (1.48 h) on the second Pod. The incident ledger records 0.5 GPU-min more
    for the first.
  - The Pods ran 1.016 and 4.247 Pod-hours: 5.263 in all.
- **Spend.** $5.95 billed: $1.14 for the first Pod and $4.81 for the second, GPU and disk, the
  second Pod's stopped volume included up to its deletion.
- **Closing state.** No Pod of the repeat is left.
  - The second Pod, `jfvugtct592agr`, was stopped at 23:34:42Z (EXITED). It was deleted at
    2026-10-30T00:42Z, after addendum 1's 347 objects were checked in the bucket.
  - The first Pod was deleted at 19:23Z, after its 79 files were verified in the bucket.
- **Archive.** The snapshot `evidence-a1/` (1791 objects) is archived under
  `$ARTIFACT_REMOTE/dryrun/study3-pod-repeat-20261029/`. All 1791 objects verify, and the read-back
  matches.
  - Inventory sha256: `b35f6cd7d324c7158056b343cef1ff72a5bf81e01512a3f2a28fd696d5da136a`.
  - `SHA256SUMS` sha256: `bc35e7f22bf21381dee924c70d173753efe4a9c4edda474ae4ab32f92630fa48`.
  - Addendum 1, under `addendum-1/`, adds what was recorded after the freeze and leaves the snapshot
    unchanged: 347 objects, all verified. Its `SHA256SUMS` lists 345 files and has sha256
    `dde4ee129115cc3eb9c382cd3b0cdc5a6a151c6039111a29aea9a8f25035b1b2`.

## Setup

| Item | Value |
|---|---|
| Pod | `jfvugtct592agr`, US-TX-4, Secure; NVIDIA L40S (46068 MiB), driver 595.91.07, CUDA 13.2; 16 vCPU; container disk 80 GB (overlay); volume 150 GB (XFS, `/dev/md127`); `MemAvailable` 1391414064 kB, cgroup limit 187999997952 bytes; created 19:19:53Z, container first seen 19:25Z, stopped 23:34:42Z, deleted 2026-10-30T00:42Z (`receipts/startup.txt`; the Runpod API's answer at creation) |
| Startup fitness (steps 3b, 6b) | `/workspace` XFS; CPU loop 2.37 s (limit 3.0 s); 200 files with fsync on `/workspace` 45 ms (limit 2000 ms); `tests/test_launch_journal.py` 63 passed in 46.45 s of pytest time (limit 60 s) (`receipts/startup-3b.txt`, `logs/speed-6b.out`) |
| Earlier Pod | `2xyqxu4mgc0yos`, US-MO-1, created 18:16:52Z: `/workspace` on MooseFS (`mfs#us-mo-1.runpod.net:9421`, fuse). Stopped at 19:17:49Z during step 10, with R2 interrupted; deleted at 19:23Z (incident 1) |
| Image | the project image, digest `sha256:9554679e749e405ad4d85e5a3c3416e687f8c3e7d443d289a18f794704810da5`, as Part I's; `/opt/project-deps/pyproject.toml` `039db0b8…`, `uv.lock` `1e0ba860…` (`receipts/image.txt`, `receipts/image-manifests.txt`) |
| Deadline | cap 2026-10-30T02:18:21Z; windows by the file watcher: W0 21:55:47Z, then 21:53:06Z (step 12), 23:29:50Z (step 24), 23:44:27Z (step 41), 00:24:20Z (renewed before step 44 ran again), 00:12:44Z (the close), 00:24:42Z (renewed before step 54a ran again) (`receipts/deadlines.txt`) |
| W&B | `WANDB_MODE=offline`, no W&B key on the Pod |
| Data | `DATA_DIR` on the volume: the four needed items from four of the record store's inventories, 0 uncovered, 17 needed files with their recorded sha256 (`receipts/data-needs.json`, `logs/S-restore.out`); θ₀; Part I's 12 dry stream files from `dryrun/study3-pod-dryrun-20261028/data/`, restored and verified (`receipts/streams.sha256`) |
| Checkpoints | one folder per record group on the container disk, `/root/ckpt-dryrun/<group>` (X, T1, T2, T3, D), so that a `finish`, a release or a wipe in one group never reaches another's files |
| Kept checkpoints | `$DRY/kept-checkpoints/<group>` on the volume (the volume's device, not the container disk's: `receipts/dry-on-volume.txt`); in the bucket under `dryrun/study3-pod-repeat-20261029/kept-checkpoints/<group>/` |
| Groups | T1, T2, T3 and D: one FA case each (`adam_frozen` at 1.3e-05, seed 90200), holding its rate's same-path control. X: the AdamW case (1.3e-05, seed 90200) of an AdamW and FA campaign; every driver run of R4 stopped at X |
| Execution order | R1 and R2, then R5 with R7 (steps 12–23), R4 with step 27 (24–40), R6 with R4's check 5 (41–48), the close (49–56). R5 ran first so that T3's kept, corrupt checkpoint was on the container disk through R4 and R6, where the "no file of another group touched" checks could see it |

### The commits actually run

| Role | Commit |
|---|---|
| GATE′: environment and provenance checks, the gate selection T′ (`stage0`, history 436) | `3f1eedf329310a38b111199806ac13ed47da678d` |
| DRY_SHA″: the seed commit; every GPU job and every reader of dry records (history 437, parent GATE′) | `239edf02293125a6b58ba1612743d4a557322467` |
| FIX″: DRY_SHA″ plus one comment line in `scripts/launch.py`; R7's launcher under another commit (history 438) | `f81981f4134653615c2a81aabc6f91525b92eb89` |

- **GATE′.** Package 1c is merged into `stage0` at `a304832`. `3f1eedf` adds one tests-only commit:
  the stand-in for `project-run` in `tests/test_qualify.py` exports `PYTHONPATH="$src/src"`, as the
  real `docker/project-run.py` does.
  - The defect was found by a local pre-check with a clean venv, and fixed before the Pod. The
    repository's `.venv` carries a `.pth` that puts the main checkout's `src` on its interpreter's
    path, so from a clone the old stand-in ran the main checkout's package.
  - In a clean venv with no installed project, as on the Pod (`/opt/venv`), four early-export
    end-to-end tests failed at `a304832` with `ModuleNotFoundError`: through the venv's `.pth` they
    had imported the main checkout's package, and on the Pod they would have failed. After the fix
    all four passed.
  - Two full suite runs, locally, before the Pod:
    - at `a304832`, with the repository's venv: 3475 passed, 0 failed;
    - at `3f1eedf`, in a clone with a venv that has no project installed, as on the Pod: 3447
      passed, 28 skipped, 0 failed. The skipped tests need a CUDA GPU or archived records:
      `CUDA_VISIBLE_DEVICES` was empty and `RUNS_DIR` private.
  - `code_identity.reuse_check("8418510", "3f1eedf")` finds no GPU path changed. `8418510` is the
    GPU code study 3 qualifies (`QUALIFIED_GPU_CODE`).
- **DRY_SHA″.** It is GATE′ plus the unchanged `dry-seeds.patch`, applied with Part I's committer and
  date.
  - The hunk bodies are byte-identical to the patch. The `SEEDS` block of `study3_records.py` applies
    26 lines lower than in Part I, because code above it grew. The researcher ruled that "the seed
    lines must not have moved" refers to their content, which is unchanged.
  - `seed_commit_check.py` and `seed_proof.py` show that only seed lines change (R2).
  - DRY_SHA″ carries a double prime because an earlier derivation on `a304832` gave a DRY_SHA′
    (`442bb540…`), superseded when GATE′ moved to `3f1eedf`.
- **FIX″.** It was derived with Part I's exact commands. The same commands on Part I's dry-seeds
  bundle give Part I's `7c1b619c…`.
- DRY_SHA″, FIX″ and the throwaway clones are local: never pushed or merged.

### Bundles, lists and tools (sha256)

| File | sha256 |
|---|---|
| `zfr-gate.bundle` (head `refs/heads/stage0` = GATE′) | `5b023776fcd3dc719080badf19d49f2e834b049766c4a1b34119065575f6af9f` |
| `zfr-dry-seeds.bundle` (head `refs/heads/dry-seeds` = DRY_SHA″) | `3604579089aee3b496ad072b92f7921b27db0020807c05c12f6eaefc45cf335c` |
| `dry-seeds.patch` (Part I's, unchanged) | `b4bc493db6b4839d0edb02844f57d36e3aed0d4ecfca95ce09ca0a9c7f2f73d2` |
| `bundles.sha256` (the three above) | `4796b0f7b633831f4f3105b71e860fe929ca70f1bdcb52973f804126075a2301` |
| `tools.sha256` (22 tools) | `ae93658998ee7b774e948097d5d392c9f5d3f09994d0e0fe62b619361542f64e` |
| `tools/pytest-nodes.txt` (155 entries, 320 cases) | `ac85418a4fdb01edfd312b1149439a7c36c753592f9fcff11358437b2be42590` |
| `tools/seed-clone-expected-failures.tsv` (164 rows) | `2dc62e6e935432e48ab08bac391b8b6eae565c1c7892a9c9ef8708dca8f91f12` |
| `upload/secret_scan.py` (step 53's scanner) | `0ed5bd243e6bf011f0446bfb7242c7b74654b7f1c2757d4ac61df65ab91e82c6` |
| `scripts/pod_deadline_watch.py` at GATE′ (unchanged since `07b7568`) | `bc82896f9214106bbec2ce385687afc24243a71282d5005d4e57db62dd2209eb` |
| `inventory-streams.tsv` (Part I's dry streams) | `d016ed1f3fe700f447517d8590d7e108f1162de26e38e3b3180c809278010d63` |

The runner scripts shipped beside the tools are not listed in `tools.sha256`. Their sha256 are
Part I's: `run_detached.py` `a1ffeff5…`, `run_registry.py` `8bff7b6d…`, `night_watch.py` `98bbd2d9…`
(`receipts/runner-scripts.sha256`).

## The checklist

Results are against items R1–R8 of the repeat's scope.

- **Columns.**
  - *Pod*: checks on the dry records this Pod made with the real model.
  - *Test layer*: the gate selection T′ in the image, synthetic worlds driven by the real scripts,
    run on the Pod (step 10).
- **Evidence paths.**
  - Paths are relative to the snapshot (`…/evidence-a1/`).
  - Each check id is a row of `logs/checks.tsv`, with its output in `logs/<id>.out`. A check run
    twice keeps its first output as `logs/<id>.1.out`.
  - Runner outcomes are in `attempts/<name>/status.json`.
  - The rows from `Z-freeze` on were written after the freeze. They and the final `checks.tsv` are
    in addendum 1 (`addendum-1/logs/`).
  - Receipts written after the freeze are in addendum 1 (`addendum-1/receipts-all/`).

### R1. Environment and provenance at GATE′

| Line | Result | Steps | Check ids | Layer | Evidence |
|---|---|---|---|---|---|
| The image's `project-run` environment, both checkouts | pass | 6, 9 | `E1-launcher`, `E1-venv-SRC`, `E1-venv-DSRC`, `E1-manifests-SRC`, `E1-manifests-DSRC`, `E1-clean-SRC`, `E1-clean-DSRC`, `E1-cpu-runtime`; runner `gpu-runtime` completed 0 | Pod | `logs/E1-*.out`, `receipts/image*.txt`, `attempts/gpu-runtime/` |
| The full Git history; the base commits present; FIX″ re-derived | pass | 5 | `E2-head`, `E2-not-shallow`, `E2-history` (436), `E2-base-commits` (`07b7568`, `ad1eb26`, `8418510`), `E2-dsrc-head`, `E2-dsrc-history` (437), `S-fix-line`, `S-fix-commit` | Pod | `logs/E2-*.out`, `logs/S-fix-*.out` |
| `progress` on an empty root, without `unknown_commit` | pass | 8 | `E2-progress-empty` (0), `E2-no-unknown-empty` (grep 1) | Pod | `logs/E2-progress-empty.out` |
| `git.commit` = DRY_SHA″ and clean in every new `meta.json` | pass: 9 run directories, 9 expected | 51 | `E4-meta` | Pod | `logs/E4-meta.out`, `receipts/close-runs.txt` |
| Data, θ₀ and the dry streams | pass | 7 | `S-setup`, six `S-list-inventory-*`, `S-needs`, `S-restore`, `S-theta0`, `S-streams-inventory`, `S-streams` | Pod | `receipts/data-needs.json`, `receipts/streams.sha256` |

### R2. The two test selections under one harness

Both selections ran on the Pod with the same pytest command, node list and environment
(`PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=`, `WANDB_MODE=offline`, and `RUNS_DIR`,
`DATA_DIR` and `TMPDIR` in a private folder). `/etc/rp_environment` and `/root/zfr-env.sh` were
present, as the Pod has them (`R2-pod-env-files`).

| Line | Result | Check ids | Evidence |
|---|---|---|---|
| T′ (GATE′) passes whole | pass: 320 passed in 1177.47 s (locally 626 s) | `T-selection-T` (0), `R2-junit-counts` | `logs/pytest-T.xml`, `logs/T-selection-T.out` |
| DSRC_T″ (DRY_SHA″) fails exactly in its expected set | pass: 156 passed, 162 failed, 2 errors in 468.60 s: the 164 rows of the expected set | `T-selection-DSRC_T` (1), `R2-junit-counts` | `logs/pytest-DSRC_T.xml`, `logs/T-selection-DSRC_T.out` |
| The seed commit's proof | pass: re-derivation `239edf02…`; patch check exit 0; at the gate 320 of 320; in the seed clone 164 failed, by class seed-refusal 80, frozen 54, seed-range 25, seed-literal 4, seed-file 1; "the seed commit's proof holds". It read the Pod's own gate junit, with no local substitute | `S-seed-proof` | `logs/S-seed-proof.out`, `seed-proof/` |
| Every run.sh test in T′ shows that its stand-in ran | pass: 14 `SELF` records and 7 `ran` records, one interpreter (`/opt/venv/bin/python`), no decoy file | `R2-standin-T` | `logs/R2-standin-T.out` |
| No Pod-environment signature in either selection | pass: no "can't open file" and no `ModuleNotFoundError` in either output | `R2-no-pod-signature` (grep 1) | — |

- **The run.sh tests.** All eight passed in T′:
  - the four suite tests that run a written run.sh by bash;
  - the self-test `test_a_stand_in_python_survives_a_pod_environment_that_resets_path`;
  - the dry tool's three.

  They include the five that failed on Part I's Pod. Part I's gap for run.sh's guard refusals on
  the Pod is closed on the test layer, under the hermetic harness.
- **In DSRC_T″** the eight run.sh tests fail by `seed-range` before run.sh is written, so no stand-in
  record exists there (C5, below).
- **The `frozen` class** holds 54 of the 164 seed-clone failures (Part I: 4 of 40).
  - Many of their first lines are generic, for example "assert 3 == 0" 17 times. The set holds only
    together with the case ids and the set equality.
  - Before the Pod, each of the 50 new frozen cases was explained locally by a differential restore
    of the named seed range: in a clone of DRY_SHA″ with only that range restored, the case passed.
    The 50 split as follows:
    - 48 by the numerical candidates;
    - 1 by numerical candidate 1;
    - 1 by validation candidate 1.
  - No case is unclassified.
  - A restore shows that the named range suffices, not that no other dry range contributes. The 48
    pass with every non-numerical range still dry.
- **R2 is the condition** Part I's report set before Part I is declared passed for the engineering
  stage. It holds.

### R3. Empty `responses/` folders, sentinels and watches (J7 (d))

| Line | Result | Steps | Check ids | Evidence |
|---|---|---|---|---|
| An empty `responses/` folder changes no reader's answer | pass in all 8 checks: on a copy with and without the folder, the launcher's dry run (`{"skipped": [<label>]}`), `complete` (3) and `progress` (0) gave the same result | 15, 19, 21, 26, 29, 33, 39, 42 | `R3-<g>-same` and its six parts, for T1, T2, T3, Xa1, Xa2, Xa3, Xa5, D | `logs/R3-*.out` |
| Each completed run gets an empty `responses/` folder and two sentinels | pass: 8 runs, 16 sentinels (`observations/` and `judge/`) | same | `R3-resp-<g>`, `R3-resp-<g>-copy`, `J7d-plant-<g>` | `receipts/gpu-sentinels.tsv` |
| Watches beside the GPU jobs: `WATCH-READY` and `WATCH-END`, no `SENTINEL`, `JUDGE`, `OTHER` or `OVERFLOW` | pass for watches 1–3 and 5–7. Watch 4: 38 failing lines within 30 ms at 21:53:41Z, after a manual read; judged on its window before them, by a ruling made after the fact (below) | 15–47 | `R3-watches` FAIL (finding); runner `sentinel-watch-4` failed 1 | `logs/sentinel-watch-{1..7}.log`; the watch-4 receipt (below) |
| `responses/` is touched only by its registered listing | pass: 46 listings of `responses/` (`REGISTERED`), 0 other events there. 19 of the 46 fall in watch 4's tail and came from the manual read | 51 | `R3-responses` | `logs/R3-responses.out` |
| The content scan: no marker outside the sealed folders, every sentinel unchanged | pass: 1649 files, 36696671617 bytes, 0 problems | 51 | `R3-scan` | `logs/R3-scan.out` |

The watches, from `WATCH-END`'s counts:

| Watch | Window (UTC) | Folders | `REGISTERED` | Failing lines | GPU jobs beside it |
|---|---|---:|---:|---:|---|
| 1 | 20:25:15–21:02:20 | 3 → 9 | 4 | 0 | T2's and T3's launcher runs and baselines to the kill |
| 2 | 21:14:58–21:23:11 | 12 | 5 | 0 | X's a2 |
| 3 | 21:24:14–21:35:46 | 15 | 12 | 0 | X's baseline 1 to the kill; X's a3 |
| 4 | 21:36:51–21:53:58 | 18 | 35 | 38 (18 `JUDGE`, 20 `OTHER`) | X's baseline 2; X's a4 to the kill |
| 5 | 21:57:00–22:04:34 | 18 | 0 | 0 | step 27's child (on a copy); X's a5 |
| 6 | 22:05:38–22:10:17 | 21 | 5 | 0 | X's baseline 3 |
| 7 | 22:23:44–23:05:39 | 24 | 9 | 0 | D's baseline, rest, control 1 to the kill, control 2 |

- **Inotify was heard in every watch.** Each watch logged `WATCH-READY` and `WATCH-END`, and none
  logged `INOTIFY-DEAF`, `OVERFLOW` or `WATCH-LOST`.
- **The `REGISTERED` events** are of `observations/measurements.jsonl` (the launcher's registered
  read, the positive control) and of the `responses/` folders.
- **The launcher's child** ran beside a live watch with planted runs six times: T2's and T3's launcher
  runs, and X's a2, a3, a4 and a5. In Part I it ran so once.
- **J7's audits** (a)–(c), by the audit hook, were not run again. Part I's result stands for them.

### R4. Repeated losses of one case, X (further recovery rounds, the lost-checkpoint note), with step 27

A *wipe* is the simulated loss of the container disk: the group's checkpoint folder removed, after a
receipt of what it held (path, size, sha256). There was no real Pod stop and start, by the
researcher's decision.

| Loss | State at the loss | Wipe removed | The driver, or the launcher, after it | Restart | Round | Next attempt |
|---|---|---|---|---|---|---|
| x1, 21:15:29Z | a1 done, no part. SIGTERM to the driver's bash alone during a1: a1 ended `done`, its run `completed`, no baseline started (`x-drv-1` failed 143) | a1's `.pt` (5241495596 bytes) | the driver: exit 1. Its log shows the conflict ("the endpoint checkpoint … is missing") and the note. `complete` 3; no `checkpoints-removed` | `restart-1` (R1): `"attempts": []`, `checkpoint_missing` true | round 1, `after` [a1], `granted` 1 | a2 done |
| x2, 21:27:34Z | baseline 1 started and killed at about 60 s (21:26:43Z), not recorded | a2's `.pt` | the driver: exit 1, the conflict on a2 | `restart-2` (R2): `[["baseline", 1]]` | three wrong rounds refused, the journal unchanged; then round 2, `after` [a1, a2] | a3 done, its run id ending `-a3` |
| x3, 21:47:41Z | the driver exits 8; baseline 2 run by hand (`--attempt 2 --infrastructure` R2) on a3, recorded | a3's `.pt` and the harness baseline (5241474716 bytes) | the driver: exit 1, the conflict on a3 | `restart-3` (R3): `[["baseline", 1], ["baseline", 2]]` | round 3, `after` [a1–a3] | a4 started, killed in flight at 21:51:48Z |
| x4, 21:52:29Z | a4 started, no end line, no checkpoint | nothing (a4 wrote none) | the launcher without a round: a4 settled `technical` (`settled` true, `returncode` null), `{"exhausted": [<label>]}`, exit 1 | refused (R4): "no attempt to restart …, and the launcher run's checkpoint is not lost" | round 4, `after` [a1–a4] | a5 done; baseline 3 by hand (`--attempt 3 --infrastructure` R3), recorded |

| Line | Result | Steps | Check ids | Evidence |
|---|---|---|---|---|
| The driver's exit 1 after a lost checkpoint shows the conflict and the lost-checkpoint note | pass, three times (a1, a2, a3); the launcher called directly on D's label gives the same (R4's check 5). The note: "A missing endpoint checkpoint is a possible loss requiring diagnosis before anything is moved, deleted or rerun under study 3's qualification (docs/operations.md, "A lost checkpoint"), unless finish removed it." | 27, 30, 34, 48 | `X2-log`, `X4-log`, `X8-log`, `R45-launcher-why` | `runs-X/qualification/c1/driver-numerical-*.log` |
| A case restart per loss, its evidence the lost checkpoint and the interrupted or recorded attempts | pass | 27, 30, 34 | `X2-restart-R1`, `X2-restart-R1-record`, `X4-restart-R2`, `X4-restart-R2-attempts`, `X8-restart-R3`, `X8-restart-R3-attempts` | `runs-X/…/restart-{1,2,3}.json` |
| Before the third wipe, X on its route: a3's `.pt` present with its recorded sha256; baseline 2 recorded on a3; `restart-case` refusing while the checkpoint is on disk; nothing written | pass ("no restart: no evidence of an interruption (no interrupted attempt, and the launcher run's checkpoint is on disk)") | 34 | `X8-pre-a3-on-disk`, `X8-pre-record`, `X8-pre-route` (1), `X8-pre-route-open`, `X8-pre-nothing-written` | `logs/X8-pre-*.out` |
| Wrong rounds refused, nothing written | pass, each exit 1: R1 again ("round 1 … gave this reason already"); a cell round (C1); `--max-attempts 3` ("a further recovery round (round 2; the label had 1) allows one attempt") | 31 | `X5-round-R1-again`, `X5-why-1`, `X5-cell-round`, `X5-why-2`, `X5-round-max3`, `X5-why-3`, `X5-journal-unchanged` | `logs/X5-*.out` |
| A launcher run lost in flight needs no restart, only a round | pass | 35, 36, 38 | `X9-a4-open`, `X10-settle` (1), `X10-why`, `X10-settled`, `X11-restart-R4` (1), `X11-why` | `logs/X9-*.out`–`logs/X11-*.out` |
| The journal: attempts 1–5; rounds 1–4 with their `after` lists, `granted` 1 each, reasons R1–R4, each with its incident time, cause and evidence; a1–a3 superseded with R1–R3 | pass | 40 | `X-journal` | `logs/X-journal.out` |
| The readers | pass: `verdict` exit 0; `read_qualification` with no problem, a5 located, a1–a3 superseded, each with its reason; `progress` exit 0, no `unknown_commit`, 5 problems (one per run directory: "(A, 1.3e-05, 90200) is not prescribed for c1/numerical", a dry root's layout), none naming a restart, a superseded line or a round | 40 | `X-verdict`, `X-qualification`, `X-progress-end`, `X-progress-no-unknown`, `X-progress-routes` | `logs/X-*.out` |
| No checkpoint of another group touched | pass at every wipe and at the end: the one file outside X's folder, T3's corrupt `.pt`, unchanged | 27–40 | `WIPE-x1-others` … `WIPE-x4-others`, `X-others` | `receipts/ckpt-inventory-x*.tsv`, `receipts/wipe-x*.tsv` |

**Step 27: a fault after initialization** (Part I's "not demonstrated 2").
- **The fault.** It ran on a copy of X's records (`copies/S27`), between x4 and round 4. A copy of
  `DATA_DIR` without its path-field record (`checks/`) made the round's child fail inside the
  tracker's run.
- **The child.** It made its run directory (`…-a5`), whose `meta.json` reads `failed`.
- **The launcher.** `journal_crash.py` killed it after 0 of 320 bytes of the end line
  (`s27-crash` failed 137).
- **The journal** stayed whole: no torn or missing line.
- **The settlement.** The next launcher settled the attempt `technical`, with `problem`
  "status 'failed'", `settled` true and exit 1 ("1 runs used up their attempts").
- **The outcome.** The technical outcome is the child's own published status, not a missing run
  directory. Check ids: `CP-S27`, `S27-data`, `S27-no-pathfield`, `S27-published`, `S27-whole`,
  `S27-settled` (1), `S27-settled-why`, `S27-settled-technical`.

### R5. A technical case's export: T1, T2 and T3

Each case: its launcher run done, its baseline killed at about 60 s, then the technical mark by the
registered command, `interrupted --part baseline --attempt 1 --technical "dry run: a technical record
made on purpose"`. Then `complete` exits 3 and `control-only` exits 9: the case is an established
technical case.

| Line | Result | Steps | Check ids | Evidence |
|---|---|---|---|---|
| T1: the driver exports before any launcher call | pass. `t1-drv-2` failed 9. Its log, in order: "a technical case by its recorded parts: its evidence is exported before any launcher call; neither the launcher nor a part is called"; `finish` exit 9; `archive.py` exit 0 (3 files, 3 verified); `kept-check` exit 0; "its checkpoints are kept, on the volume and in the bucket, verified". No `scripts/launch.py` line and no `--part` line. One start of T1's label; `complete` 3 and `control-only` 9 again (incomplete, technical) | 13–16 | `T1-group-empty`, `T1-launcher-done`, `T1-baseline-interrupted`, `T1-interrupted`, `T1-complete-3` (3), `T1-control-only-9` (9), `T1-log`, `T1-starts-1`, `T1-complete-3b`, `T1-control-only-9b`, `T1-manifest`, `T1-inventory`, `T1x-others` | `runs-T1/qualification/c1/driver-numerical-20261029T202608Z.log`, `runs-T1/…/kept-checkpoints-1.json` |
| T1: the manifest | pass: verified; the launcher's `.pt` (5241495596 bytes, `c999e158…`) exported with `matches_record` true; `lost` empty; 3 bucket objects, all `ok`, under `kept-checkpoints/T1/` | 16 | `T1-manifest`, `T1-inventory` | `runs-T1/archive/inventory-kept-20261029T202646Z.tsv` |
| T1: the release through the verified export, after R7 | pass: `finish --release` exit 0, removed T1's `.pt`; `checkpoints-removed-1.json` names `kept-checkpoints-1.json` and `kept-check-1.json`; T1's files gone, no other file changed | 18 | `T1-release`, `T1-release-record`, `T1r-others`, `T1r-gone` | `receipts/ckpt-inventory-t1r-*.tsv` |
| T2: lost before the export | pass. A wipe after the mark (20:43:50Z, T2's `.pt`, `7dbad5b3…`). `t2-drv-2` failed 9: "lost before the export: the technical mark stands; the diagnosis goes on from the records; nothing reruns". The manifest: no file, `lost` naming the launcher's `.pt`. The bucket: `MANIFEST.json` and `SHA256SUMS` alone, both `ok` (C6). `kept-check` 0. One start | 19, 20 | `T2-launcher-done`, `T2-interrupted`, `WIPE-t2`, `WIPE-t2-gone`, `WIPE-t2-others`, `T2-log`, `T2-manifest`, `T2-bucket`, `T2-starts-1` | `receipts/wipe-t2.tsv`, `runs-T2/…/kept-checkpoints-1.json` |
| T2: `restart-case` refused | pass, on a copy: exit 1, "no restart: a completed failure keeps counting: [['baseline attempt 1', 'technical']]" | 23 | `R5-T2-restart-refused` (1), `R5-T2-restart-why` | `logs/R5-T2-restart-refused.out` |
| T3: corrupt bytes on a real record, exported as they are | pass. The first MiB of T3's `.pt` zeroed: size 5241495596 unchanged, sha256 `eea06f88…` (its recorded value) → `539a6e4a…`. `t3-drv-2` failed 9, no launcher line. In the export: `matches_record` false, `verified` true, the copy's sha256 the corrupt file's. `control-only` 9 (technical), `complete` 3, one start | 21, 22 | `T3-launcher-done`, `T3-interrupted`, `T3-corrupt-before`, `T3-corrupt`, `T3-corrupt-after`, `T3-corrupt-differs`, `T3-log`, `T3-export-corrupt`, `T3-still-technical` (9), `T3-complete-3` (3), `T3-starts-1` | `receipts/corrupt-t3-{before,after}.tsv`, `runs-T3/…/kept-checkpoints-1.json` |
| `--kept` on the checkpoint folder's filesystem refused | pass, on a copy: exit 13, "is on the filesystem of the checkpoint directory /root/ckpt-dryrun/T1 (the container disk, lost with it)"; nothing copied | 23 | `R5-kept-13` (13), `R5-kept-13-why` | `logs/R5-kept-13.out` |
| The bucket variables unset at the driver's start: exit 1 before any case | test layer only (C2): node 80 of `tools/pytest-nodes.txt`, `test_the_driver_checks_the_bucket_settings_before_the_first_case`, 5 cases passed | 10, 23 | — | `receipts/coverage.txt` |

- **The early export on the test layer.** The early export is also covered in T′:
  - with a launcher stand-in that refuses (`test_the_driver_exports_a_technical_case_before_a_launcher_that_would_refuse`,
    three cases, the AdamW block's among them);
  - with exit 13 when the bucket copy is not verified (`test_the_driver_stops_with_13_when_the_bucket_copy_is_not_verified`);
  - with the pending-upload receipt (`test_the_driver_leaves_a_pending_upload_receipt_for_an_early_export`).
- **On the Pod**, every export verified.
- **`kept-check`** passed three times per case: in the driver, at step 51 and afresh at step 54b. In
  each case the newest receipt is current for the newest manifest (`Z-final-kept-current-T1`…`T3`).

### R6. The control alone after a lost disk, D, with R4's check 5

D's launcher run, baseline and rest ran by hand, with the driver's exact commands and in the
driver's order (the researcher's decision). The driver itself ran D only after the wipe. The path
under test is the control alone after a lost disk. D's baseline did not decide the case: `decided`
exited 3, its rest due. `control-only` therefore took the branch of a recorded rest.

| Line | Result | Steps | Check ids | Evidence |
|---|---|---|---|---|
| D's launcher run, baseline and rest | pass: a1 done; the baseline and the rest each exit 0 | 42, 43 | `D1-a1-done`, `D2-located`, `D2-decided` (3); runners `d-launcher`, `d-baseline`, `d-rest` completed 0 | `attempts/d-*/` |
| Before the wipe, D bound to its launcher run | pass on the second run of step 44: `control-only` exit 0, bound to `…596d44c3-a1`; the baseline and rest records attempt 1 each, naming that run; one start of D's label; no control attempt | 44 | `D3-pre-records`, `D3-pre-starts`, `D3-pre-no-control` (1) | `logs/D3-pre-*.out` |
| The wipe | 22:56:03Z: D's `.pt` (`08382c3e…`) and its harness baseline (`7fb0806d…`); the 3 files outside D's folder unchanged | 44 | `WIPE-d`, `WIPE-d-gone`, `WIPE-d-others` | `receipts/wipe-d.tsv` |
| The driver runs the control alone | pass. `d-drv-1`: `complete` 3, `control-only` 0, the driver's line "only its same-path control is due (its baseline decided it, or its rest is recorded): the launcher, the baseline and the rest are not called", then `--part control --resume`. SIGKILL to its group at 22:58:38Z, about 60 s after the control's `.started` | 45 | `D4-log` | `attempts/d-kill-1/stdout.log` |
| The next driver run exits 8 | pass: `d-drv-2` failed 8, "control: attempt 1 was started and left no record: its next attempt is run by hand, --attempt 2 --infrastructure REASON" (C3) | 45 | `D5-log` | — |
| Attempt 2 by hand, with no checkpoint on disk | pass: `control-only` 0 names D's launcher run; its `.pt` absent; `part --part control --attempt 2 --infrastructure` R6 completed 0 (3.0 GPU-min) | 46 | `D6-bound`, `D6-no-checkpoint`; runner `d-control-2` | `attempts/d-control-2/` |
| The driver finishes the case | pass: `d-drv-3` completed 0: "complete: s90200/adam_frozen-1.3e-05 is skipped", `finish` exit 0 with `"removed": []` (the wipe had removed D's checkpoints), "every case is complete"; no file outside D's folder changed | 46 | `D7-log`, `D7-others` | `receipts/ckpt-inventory-d3-*.tsv` |
| R6's pass | pass: one start of D's label; control attempt 1 started on `…596d44c3-a1` and attempt 2's record names it; no `scripts/launch.py`, `--part baseline` or `--part rest` line in any D driver log | 47 | `R6-no-new-start`, `R6-bound-run`, `R6-logs-after-wipe` | `runs-D/qualification/c1/driver-numerical-*.log` |
| R4's check 5: a finished case | pass. The driver skips D (`complete` 0), `finish` prints `"removed": []`, no new release record. On a copy, the launcher called directly on D's label: exit 1, the conflict and the note. `restart-case`: exit 1, "the case is complete; a restart voids only an interrupted case" | 48 | `D8-log`, `D8-no-new-release`, `CP-D-ref`, `R45-launcher` (1), `R45-launcher-why`, `R45-restart` (1), `R45-restart-why` | `logs/R45-*.out` |

The decided state, a case decided by its baseline with its control due, is on the test layer only:
`test_a_decided_case_with_its_control_due_runs_the_control_alone`,
`test_control_only_binds_the_decided_baseline_to_its_launcher_run_without_the_checkpoint`,
`test_control_only_refuses_a_decided_case_it_cannot_bind`,
`test_an_interrupted_control_of_a_decided_case_is_rerun_by_hand_after_a_lost_disk` and
`test_a_failed_control_of_a_decided_case_is_kept` passed in T′. No hand-written record was put into
dry records. R6 is a real control on a natural state.

### R7. The launcher's skip branch under another commit

| Line | Result | Steps | Check ids | Evidence |
|---|---|---|---|---|
| A pre-check: T1's `.pt` present with its recorded sha256, in T1's own checkpoint folder | pass | 17 | `R7-precheck` | — |
| The launcher from FIX″, `--only` T1's label, `--max-attempts 1` | pass: `{"skipped": ["c1-numerical-adam_frozen-1.3e-05-s90200/s90200"]}`, exit 0 | 17 | `R7-skip` | `logs/R7-skip.out` |
| The journal unchanged | pass: byte for byte | 17 | `R7-journal-unchanged` | — |

This closes Part I's "not demonstrated 1" and its failed `G2b-launcher-other`. T1's checkpoint stays
reachable because each group has its own checkpoint folder, and R7 ran before T1's release.

### R8. The close

| Line | Result | Steps | Check ids |
|---|---|---|---|
| Separation held to the end | pass: the real record paths absent; both checkouts clean at their commits; the GPU idle; no process; the five run roots whole (no torn or missing line) | 51 | `Z-real-paths-absent`, `Z-checkouts`, `Z-gpu-idle`, `Z-no-process` (1), `Z-runs-X-whole`, `Z-runs-T1-whole`, `Z-runs-T2-whole`, `Z-runs-T3-whole`, `Z-runs-D-whole` |
| Every technical case's export verified before the stop | pass: `control-only` 9 and a current verified export for T1, T2 and T3; X and D `control-only` 3. A fresh `kept-check` 0 for each at step 54b | 50, 51, 54b | `Z-kept-check-T1`…`T3`, `Z-final-exports-try1`, `Z-final-kept-check-T1`…`T3`, `Z-final-kept-current-T1`…`T3`, `Z-final-records-try1`, `Z-final-records-check-try1` |
| Freeze, scan, upload, read-back | pass after step 54a's ruling (below) | 52–54a | `Z-receipts`, `Z-freeze`, `Z-snap-before`, `Z-secret-scan-a1`, `Z-upload-a1` (attempt 1 FAIL, attempt 2 PASS), `Z-readback-get-a1`, `Z-readback-check-a1`, `Z-snap-after-a1`, `Z-inventory-up-a1` |
| The container disk at the close | T3's corrupt `.pt` (`539a6e4a…`), X's a5 `.pt` (`c76facec…`) and X's harness baseline (`ecc437cd…`); nothing of T1, T2 or D | 51, after 54b | `Z-close-inventory-partial`, `Z-post-x-a5`, `Z-post-x-harness`, `Z-post-t3-corrupt` |
| The stop | the Runpod API's `pod-action stop` at 23:34:42Z; `desiredStatus` EXITED. The Pod was deleted at 2026-10-30T00:42Z, after addendum 1 was checked in the bucket | 56 | `addendum-1/runpod/stop-pod2.txt`, `addendum-1/stop-and-cost.md` |

### What each layer covers

| Item | Pod (dry records, real model) | Test layer only (T′ on the Pod) | Not covered |
|---|---|---|---|
| R1 | E1, E2, E4, the GPU runtime | — | — |
| R2 | both selections, the seed proof, the stand-in records | — | — |
| R3 | 7 watches over empty `responses/` folders; 8 equivalence checks; the content scan | — | watch 4 from 21:53:41.000000Z on (the window before it judged by a ruling); J7 (a)–(c) not rerun |
| R4 | X: 4 losses, 3 restarts, a refused restart, 4 rounds, 3 refused rounds, the in-flight settlement; step 27's fault after initialization | — | a real container restart; X's rest |
| R4's check 5 | D, a finished case | — | — |
| R5 | T1, T2, T3; `--kept` on the container disk refused | C2's unset bucket variables (node 80); exit 13 on an unverified bucket copy; the pending-upload receipt; an export holding harness files | an export of an engineering case's diagnostic checkpoints |
| R6 | D: the control alone after a lost disk, its rest recorded; attempt 2 by hand; the finish | the decided state | the decided state on dry records |
| R7 | T1 skipped under FIX″ | — | — |
| P1 | — | `restart-case` never restarts over a binding refusal (`test_restart_case_never_restarts_over_a_binding_refusal`, 4 cases; `test_restart_case_refuses_the_state_control_only_runs_alone`) | on dry records: a "bound elsewhere" state needs records written by hand |
| R8 | the close | — | — |

### Part I's gaps

| Part I's item | What the repeat shows |
|---|---|
| Not demonstrated 1, and the failed `G2b-launcher-other` (step 25's skip branch) | R7: skipped, exit 0, the journal unchanged |
| Not demonstrated 2 (step 27's child-published outcome) | Step 27: the child's run directory `failed`, settled `technical` from it |
| Not demonstrated 3 (J7 (d) for `responses/`) | R3: 7 watches over empty `responses/` folders; watch 4 judged on a truncated window by a ruling made after the fact |
| Failed `T-selection-T` (run.sh's guard on the Pod) | R2: all 8 run.sh tests pass on the Pod under the hermetic harness; both selections under one harness |
| Not rehearsed: a container restart that wipes the container disk | Simulated by six wipes (x1–x4, T2, D). No real restart was produced |
| Open item 1: a second loss in one case | R4: four losses of X, rounds 1–4 |
| Open item 2: the sign of a lost disk | R4: the launcher's lost-checkpoint note, in every conflict on a lost checkpoint |
| Open item 3: a technical case's kept checkpoints | R5: exported and verified before any launcher call and before the stop |
| Open item 4: a decided case whose control is due | Test layer only. On the Pod, the control alone after a lost disk with the rest recorded (R6) |

## What the repeat does not cover

1. **The decided state on dry records.** Test layer only (R6, above). D's baseline did not decide its
   case, and no decided record was written by hand into dry records.
2. **P1 on dry records.** Test layer only. A "bound elsewhere" state needs records written by hand.
3. **C2, the driver's check of unset bucket variables.** Test layer only (node 80). On the Pod the
   generated driver re-sources `/etc/rp_environment` and `/root/zfr-env.sh`, which set them.
4. **An export that fails.** Test layer only: the driver's exit 13 on an unverified bucket copy, and
   the pending-upload receipt. Every export on the Pod verified.
5. **An export holding harness or diagnostic checkpoints.** Test layer only
   (`test_finish_exports_a_technical_cases_checkpoints_and_exits_9`). Every export on the Pod held
   the launcher's endpoint `.pt` alone (T1, T3) or nothing (T2). The size of an engineering case's
   export is not exercised.
6. **A real container restart or Pod stop that wipes the container disk.** Not produced. Each loss
   was a simulated wipe with its receipt.
7. **X's rest to completion.** Not in the repeat's scope.
8. **The driver's own launcher, baseline and rest for D.** They ran by hand with the driver's
   commands (R6). The driver ran D only after the wipe.
9. **J7 (a)–(c).** The audit hook was not run again; Part I's result stands. A `REGISTERED` event is
   still not tied to a process: watch 4's tail holds 19 `REGISTERED` events that the manual read
   caused.
10. **Two GPU jobs with planted runs and no live watch.** X's a1 (`x-drv-1`, 21:07–21:12Z) and D's
    launcher run (`d-launcher`, 22:16–22:22Z) ran while sentinels were planted in earlier groups. The
    procedure starts a group's watch after that group's first launcher run, and no watch was live at
    those two times.
11. **D2's documented negative** (SIGTERM to the launcher alone). Not rehearsed, as in Part I.
12. **Everything outside the affected paths.** Part I's report stands, Part II's items included.

## Watch 4, judged after the fact

Watch 4 failed. It is counted as passed for R3 by a ruling made after the fact, on a truncated
window. The full record follows.

**The watch.**

- It ran from `WATCH-READY` at 21:36:51.391Z to `WATCH-END` at 21:53:58.150Z, over 18 folders.
- Beside it ran the driver to its exit 8 (`x-drv-5`), X's baseline 2 (21:42:25–21:46:42Z), the driver
  to the conflict on a3 (`x-drv-6`), and X's a4 to its kill at 21:51:48Z. Step 36's launcher without
  a round followed (21:52:54Z, no GPU).
- Every one of these jobs had ended before 21:53:41Z.

**The counts, from `logs/sentinel-watch-4.log`.**

| Part | Window (UTC) | Events |
|---|---|---|
| Before the tail | 21:36:51.391183–21:53:41.093712 (none from 21:53:41.000000 on) | 16 `REGISTERED` (8 of `observations/measurements.jsonl`, 8 of `responses/` folders), and nothing else |
| The tail | 21:53:41.093712–21:53:41.123891 (30 ms) | 38 failing lines: 18 `JUDGE`, 20 `OTHER`, 0 `SENTINEL`, 0 `OVERFLOW`. Besides them, 19 `REGISTERED`, all of `responses/` folders |
| After the tail | to `WATCH-END` at 21:53:58.150 | no event |
| Whole watch (`WATCH-END`) | — | `REGISTERED` 35, `JUDGE` 18, `OTHER` 20, `SENTINEL` 0, `OVERFLOW` 0, `WATCH-LOST` 0 |

The 38 failing lines by root:

| Root | `JUDGE` | `OTHER` | Total |
|---|---:|---:|---:|
| `runs-T1` | 2 | 2 | 4 |
| `runs-T2` | 2 | 2 | 4 |
| `runs-T3` | 2 | 3 | 5 |
| `runs-X` | 12 | 13 | 25 |
| **Total** | **18** | **20** | **38** |

- **Folders only.** Every one of the 38 is an event on a sealed folder itself (`judge/` or
  `observations/`), an open or a read with inotify's folder flag (masks `0x40000020` and
  `0x40000001`). None is on a file. No sentinel was opened or read.
- **No other command.** Between step 36's block (21:52:54Z) and step 37a's block (21:53:57Z) nothing
  else ran but the measurement below, and a read of `logs/X10-settle.out` with `nvidia-smi` and
  `open_runners`. That read touched nothing under a runs root or `DATA_DIR`.

**The cause.**

- **The measurement.** A manual quota measurement outside the procedure's blocks ran
  `du -sb` of `/workspace`, of the dry data folder and of `runs-X`. Its receipt line is stamped
  21:53:41Z (`receipts/deviation-workspace-quota.txt`): 19984970174 bytes used of the 150 GB quota.
- **The timing.** It ran before step 37a's `watch_off 4`, while watch 4 was live. The traversal opened
  and read the planted folders, and the watch ended `failed 1` (halt 2, 21:53:59Z).
- **Why it existed.** The measurement existed only because of the first Pod's MooseFS volume. There
  `df` shows the shared filesystem and not the volume's quota, so space decisions were moved to
  `du -sb` against the quota (step 37's space check among them). That rule was still in force on the
  second Pod, whose `/workspace` is local XFS, where `df` alone would have served.

**The ruling.**

- Watch 4 is judged on the window [21:36:51.391183Z, 21:53:41.000000Z): from `WATCH-READY` to the
  manual measurement's receipt time, 21:53:41Z, which was recorded independently of the watch. The
  window ends there, not at the first failing line. It holds 16 `REGISTERED` lines and no
  `SENTINEL`, `JUDGE`, `OTHER` or `OVERFLOW` line. On it, the launcher and the driver pass R3 for
  watch 4.
- Watch 4 as a whole failed, and the ruling was made after the fact. That the tail comes from the
  measurement is an inference consistent with the evidence, not a proof. All 38 failing lines
  fall within 30 ms of the second the measurement ended, under the roots it read, and no other
  command ran then. inotify records no process.
- Step 37a ran again with `job_is sentinel-watch-4 failed 1` in place of `watch_off 4`. `halts.tsv`
  keeps its row.

**The tail stays in the evidence.**

- `logs/sentinel-watch-4.log` is frozen unchanged.
- The receipt of the ruling gives the counts by class and root. Its file name in the evidence is
  `receipts/deviation-watch4-operator-du.txt`.
- `R3-watches` fails on it at the close and stays in `receipts/close-findings.tsv`.
- `R3-responses`' 46 listings of `responses/` include the 19 of the tail.

**Afterwards.** The procedure was narrowed in two ways:

- the `du` measurement against the quota applies only on a MooseFS `/workspace`;
- no command outside the procedure's blocks may read under any runs root, or under the dry data
  folder, while a watch is live.

## Where the code was right and the expectation was not

In each item below, an expectation written into the repeat's scope or its procedure differed from
what the code at GATE′ gives. The researcher ruled the code's version the pass criterion in each.

- **Step 34b, `X7-log` (halt 1).**
  - The procedure expected the driver's exit-8 reason for X's interrupted baseline attempt 1 to read
    "attempt 1 is named by the case's restart or supersession record".
  - The code gives "attempt 1 was started and left no record" for any interrupted latest attempt
    (`scripts/qualify.py:716–724`, `:1645–1650`). `voided` applies only to recorded attempts that a
    restart or supersession record names by their file's sha256. An interrupted attempt has no
    record file.
  - The route is the same: exit 8, then `--attempt 2 --infrastructure REASON`.
- **C1 (step 31).** A round without `--only` was expected to be "no round, exhausted". The code
  refuses it before assessing any run: exit 1, "A study-3 campaign takes no cell recovery round". No
  round is written either way.
- **C2 (step 23).** The bucket variables unset at the driver's start were expected to stop it with
  exit 1. On the Pod the generated driver re-sources the environment files, which set them, so the
  check is reached on the test layer only (node 80).
- **C3 (step 45).** The killed driver was expected to exit 8. A driver killed with its process group
  records no exit: its runner's supervisor is in the group, and `status.json` stays `running`. Exit 8
  comes from the next driver run (`d-drv-2`).
- **C4 (step 51).** The listing of `responses/` was expected as `LIST-ALLOWED`. The watch logs it as
  `REGISTERED <run>/responses/`; `LIST-ALLOWED` is the audit hook's term, and the audit hook was not
  run.
- **C5 (step 11).** Stand-in records were expected for every run.sh test in either selection. In
  DSRC_T″ the eight run.sh tests fail by `seed-range` before run.sh is written, so no record exists.
  There the criterion is no record and no Pod-environment signature.
- **C6 (step 20).** The bucket was expected to get "the manifest alone" for T2. The export holds
  `MANIFEST.json` and `SHA256SUMS`: two objects, no checkpoint.

## Incident 1: the first Pod was unfit

The first Pod, `2xyqxu4mgc0yos` (US-MO-1, one L40S, driver 595.91.07), ran from its creation at
18:16:52Z to its stop at 19:17:49Z: 1.016 Pod-hours. That is the incident. The incident ledger
records 0.5 GPU-min and 61 Pod-min for it (`incidents.tsv`). Paths in this section are relative to
the first Pod's evidence, `pod1-2xyqxu4mgc0yos/`.

- **The volume.** `/workspace` was MooseFS (`mfs#us-mo-1.runpod.net:9421`, fuse), not a local disk. A
  probe at 18:28:17Z (`receipts/deviation-moosefs-probe.txt`) found:
  - umask ignored: a file made under `umask 077` came out mode 666;
  - `cp -a` kept owners and times, but files came out mode 666;
  - root could not `chown` to another owner (`receipts/deviation-rsync-owner.txt`).
- **Slow metadata.** 200 small files written and removed took 996 ms on `/workspace` and 42 ms on
  the container disk.
- **A slow host.**
  - `tests/test_launch_journal.py` took 122.46 s of pytest time on the container disk, against 17.7 s
    locally.
  - T′ had run 112 of its 320 cases in about 22 min, against 70 s locally for those 112. At that pace
    R2 could not end within W0.
- **The journal mode test.** `test_the_append_is_one_write_of_the_whole_line_then_an_fsync` failed on
  MooseFS ("assert 438 == (420 & ~18)": mode 0o666 where 0o644 is expected). It passed on the
  container disk.
  - Run A, on the container disk: 63 of 63 passed.
  - Run B, on MooseFS: 62 passed, 1 failed (`receipts/deviation-r2-moosefs.txt`).
- **The stop.** R2 was stopped at 19:08Z by the researcher's ruling (`receipts/halts.tsv` of the
  first Pod). The Pod was stopped at 19:17:49Z.
- **Its evidence.** Its 79 files went to `dryrun/study3-pod-repeat-20261029/pod1-2xyqxu4mgc0yos/` at
  19:22Z:
  - `archive.py` verified 79, mismatched 0;
  - the read-back matched 78 of 78 by `SHA256SUMS`;
  - inventory `inventory-20261029-pod1-2xyqxu4mgc0yos.tsv`, sha256
    `c7505c8e14505279950698b17c4e224dfe7ac788c85bdb2bb14e65fd0522183b`.

  `known_hosts` and `ssh.env` stayed local as connection files. A scan of the Pod's files found one
  hard match there, the host's public key. The Pod was deleted at 19:23Z.
- **Added after it.** Two startup checks were added for every later Pod of the repeat. A failure stops
  the Pod, records an incident, and another Pod is requested.
  - **Step 3b**, after step 3:
    - `findmnt -no FSTYPE /workspace` must be `xfs` or `ext4`;
    - a CPU loop of 3·10⁷ additions must take at most 3.0 s (local 1.0 s);
    - 200 files written with fsync on `/workspace` must take at most 2000 ms (local 231 ms).
  - **Step 6b**, after step 6: `tests/test_launch_journal.py` whole, under R2's environment, must pass
    all its cases (63 at GATE′) in at most 60 s of pytest time.
- **The second Pod passed both.**
  - Step 3b: XFS, 2.37 s, 45 ms.
  - Step 6b: 63 passed in 46.45 s.
- **The requirement.** Pods that hold records need a local volume (XFS or ext4). Before the
  engineering stage, `docs/operations.md`'s Pod startup procedure gets the `findmnt` check: a Pod
  whose `/workspace` is anything else is stopped and replaced.

## Deviations and rulings

| Pod, step | What departed from the procedure as written | What followed: the observation, a decision or a correction | Evidence |
|---|---|---|---|
| First Pod, 3 (18:21:59Z) | The disk check printed `df -B1G`. GNU df then gives bare numbers, and the check's pattern expects a `G` suffix, so "the disks" failed on any machine | Ruling: `df -BG`, thresholds unchanged. A fault of the check, not an infrastructure incident. Run again at 18:24:26Z: all checks hold | `pod1-2xyqxu4mgc0yos/receipts/deviation-step3-df.txt` |
| First Pod, 3 | `/workspace` on MooseFS: `df` shows the shared filesystem (108701 G free), not the volume's 150 GB quota | Ruling: on MooseFS, space is measured with `du -sb` against the quota. 24095 bytes used; the startup threshold held | `pod1-…/receipts/deviation-workspace-quota.txt` |
| First Pod, 4 (18:25:50Z) | `rsync -a` to the Pod exited 23: root cannot `chown` on MooseFS. The content arrived; the ledger's sha256 was equal | Ruling: the four `rsync` lines to the Pod take `--no-owner --no-group`, since owners carry no meaning in the dry folder. The ledger copy was accepted by its sha256. Not an incident. A probe of the mount followed before step 5 | `pod1-…/receipts/deviation-rsync-owner.txt`, `deviation-moosefs-probe.txt` |
| First Pod, 10 (18:44:18Z–19:08Z) | R2 too slow on MooseFS, and one T′ case failing there only | R2 stopped by the researcher's ruling; the Pod unfit (incident 1, above) | `pod1-…/receipts/deviation-r2-moosefs.txt` |
| Second Pod, 34b (21:38:43Z) | `X7-log` FAIL: the procedure's marker for the exit-8 reason was wrong (halt 1) | Ruling: the code is right (above); the marker corrected. 34b ran again whole and passed; its first output stays as `logs/X7-log.1.out` | `receipts/deviation-step34b-marker.txt` |
| Second Pod, 37a (21:53:59Z) | Watch 4 ended `failed 1` after a manual measurement outside the procedure's blocks read the planted folders (halt 2) | Ruling: judged on the window [21:36:51.391183Z, 21:53:41.000000Z), after the fact (above). 37a ran again with `job_is sentinel-watch-4 failed 1`. The procedure narrowed | the watch-4 receipt (above), `receipts/deviation-workspace-quota.txt` |
| Second Pod, 44 (22:50:59Z) | `D3-pre-records` FAIL (halt 3, D off its route by the check). Its glob `rest-attempt-*.json` also matched the rest's six trajectory files, so it counted 7 "attempts" of 1 and never read the record. The other pre-wipe checks passed | Before any wipe, the halt's keep copied D's folder (its `.pt` and `harness/`) to the volume and uploaded it as `kept-checkpoints/D-unrouted-20261029T224727Z`: 2 files, 2 verified. D was not wiped. Ruling: the check takes only exact `<part>-attempt-<N>.json` names, as the code's own parser does (`qualify.py:2515`); tested on synthetic files. After a deadline renewal, step 44 ran again whole and passed. The keep copy stays as evidence | `receipts/deviation-step44-records-glob.txt`, `logs/KEEP-D-unrouted-*.out` |
| Second Pod, 46–48 | After watch 7 ended (23:05:39Z), a manual `grep` outside the procedure's blocks printed D's case verdict lines to a console | Recorded. The output is in no record | — |
| Second Pod, the close (23:13:40Z) | The scripts and consoles of steps 3–11 were on the container disk | Copied to `$DRY/ops-phase1/` before the freeze (30 files, sha256 equal) | `receipts/deviation-ops-phase1.txt` |
| Second Pod, 51 | `halts.tsv` holds ruled rows, so step 51 took its partial branch, and `R3-watches` failed on watch 4's tail | `R3-watches` is a finding; the close went on. The named container-disk checks (X's a5 `.pt` by sha256, the harness baseline, T3's corrupt sha256) ran after step 54b, as `Z-post-x-a5`, `Z-post-x-harness`, `Z-post-t3-corrupt`: all passed | `receipts/close-findings.tsv` |
| Second Pod, 54a (23:20:48Z) | `Z-upload-a1` exit 1 (halt 4). The path list (`ls -A`) held the snapshot's empty `runs/`, and `gcloud storage cp --recursive` of an empty folder exits 1 ("matched no objects or files"); reproduced locally. 1526 objects of the ten earlier paths were uploaded; 265 files were not, and no inventory was written | Ruling: the path list leaves out empty top-level folders; an empty folder holds no file of `SHA256SUMS`. After a deadline renewal, 54a ran again to the same prefix; its inventory is `…-a1-try1`, since the first attempt wrote none. `--no-clobber` kept the 1526 objects, and the MD5 check verified them with the rest: 1791 of 1791 | `addendum-1/receipts-all/deviation-step54a-empty-runs.txt` (written after the freeze) |
| Second Pod, 52–55 | `ops/` and `ops-phase1/` were not in the freeze list | Copied locally to `final-ops/` (196 files, verified by sha256), and from there into addendum 1 (`addendum-1/ops/`) | `addendum-1/ops/final-ops.sha256` |
| Second Pod, 56 | The planned close after step 55 | The Pod was stopped through the API at 23:34:42Z (EXITED), and its Pod-hours were appended to the ledger. Addendum 1 was built after the billing of 2026-10-29 had posted, then archived and verified. The Pod was deleted at 2026-10-30T00:42Z | `addendum-1/runpod/stop-pod2.txt`, `addendum-1/stop-and-cost.md` |

## The secret scan

- **The scan.** Step 53's scanner (`upload/secret_scan.py`, sha256 `0ed5bd24…`) read every file of
  the frozen snapshot `evidence-a1/`, `external/` included, line by line, before any upload. It never
  printed a line or a value.
  - Hard stops: a `BEGIN … PRIVATE` line, the service-account key field's name, a JSON key body, or a
    run of 200 or more base64 characters.
  - A key-like name (one holding `GCS_KEY` or `ARTIFACT_GCS`) is classified with its whole read or
    write parsed. A write passes only when its whole right side is another variable or the key
    file's path. A default passes only when it is absent, empty, `None`, another variable or the key
    file's path.
- **The result.** 30 files with a match; 0 rejected and 0 hard stops; exit 0
  (`upload/secret-scan-matches-a1.tsv`, sha256 `58a9e584…`).
  - 113 reads:
    - 4 in each of the five generated drivers and their 20 copies, each recognized by its sha256;
    - 1 each in `env.sh`, `receipts/startup.txt` and `tools/restore_needed.py`;
    - 5 each in the two DSRC_T″ outputs.
  - 24 writes accepted by the one exception: `write-other-allowed-line`, 12 matches on 7 lines in
    each of the two files below.
- **The exception, exactly.** It is in the scanner's command, decided before the Pod.
  - **Two files only:** `logs/pytest-DSRC_T.xml` and `logs/T-selection-DSRC_T.out`.
  - **Three lines only,** each by the sha256 of the whole stripped line:
    - `876ea826dd82d6fed022a62a62cb3202d38dc76d8a7125a749ead0b849b06cd3`: `tests/test_qualify.py:484`
      as the seed clone's tracebacks quote it;
    - `d108ff982cbc3a5edfc06bb8fef2e49a5d5d6a0c0cf267f31501a670f4fad738` and
      `143ca834c7031871a4553db49bc462a686e60bc5754b66b6382f305101dbe1dc`: pytest's argument lines for
      the parametrized `None` and `"/no/key"`.
  - **What they are:** test-fixture values printed by expected seed-clone failures.
  - **Still rejected:** the same line in any other file, any other write in those two files, and any
    hard stop anywhere. T′'s junit has no such line, because those tests pass at the gate.
- **The limits.** A passing scan does not prove that any key in the snapshot would have been found.
  The scan looks for four hard-stop patterns and for key-like names. It does not catch:
  - **a ternary fallback.** `os.environ.get(NAME) if c else "<literal>"` is classed a read: only an
    `or D` after a read is parsed as a fallback;
  - **another alias of the environment.** A write through `env = os.environ; env[NAME] = …`, or
    through `environ[NAME] = …` after `from os import environ`, is classed a read: only
    `os.environ[…] =` is parsed as an environment write;
  - **a literal default on a name that is not key-like.** For example
    `KEY_FILE = os.getenv("OTHER", "<literal>")` is never matched.

  A literal there is caught only if it is itself a hard-stop pattern.
- **The receipts** (the scanner, its sha256, the matches) are in addendum 1 (`addendum-1/upload/`).
- **The addendum's own scan,** by the same scanner with no exception, found 8 files with a match,
  all reads: 0 rejected and 0 hard stops (`addendum-1-scan-try1.tsv`).

## GPU and Pod time

**GPU minutes.** They are counted per runner as the close writes them (`receipts/gpu-minutes.tsv`):

- an ended runner from its start to its `finished_at`;
- a runner killed with its process group to the kill time in its `kill_at.sh` receipt;
- `x-drv-1` to the time its group emptied in `term_driver.sh`'s receipt.

| GPU job (runner) | Minutes |
|---|---:|
| Runtime check (`gpu-runtime`) | 0.3 |
| T1: launcher run and baseline to the kill (`t1-drv-1`) | 8.1 |
| T2: the same (`t2-drv-1`) | 7.8 |
| T3: the same (`t3-drv-1`) | 7.9 |
| X: a1 to the group's emptying (`x-drv-1`) | 5.4 |
| X: round 1, a2 (`x-round-1`) | 4.9 |
| X: baseline 1 to the kill (`x-drv-3`) | 2.2 |
| X: round 2, a3 (`x-round-2`) | 4.9 |
| X: baseline 2 (`x-baseline-2`) | 4.3 |
| X: round 3, a4 to the kill (`x-round-3`) | 2.3 |
| Step 27's child, to its crash (`s27-crash`) | 0.3 |
| X: round 4, a5 (`x-round-4`) | 4.9 |
| X: baseline 3 (`x-baseline-3`) | 4.1 |
| D: launcher run (`d-launcher`) | 5.9 |
| D: baseline (`d-baseline`) | 4.2 |
| D: rest (`d-rest`) | 16.5 |
| D: control attempt 1 to the kill (`d-drv-1`) | 1.6 |
| D: control attempt 2 (`d-control-2`) | 3.0 |
| **Total, second Pod** | **88.6 (1.48 h)** |

- **Not counted.** Ten driver runs did no GPU work and took 14.7 min in all: `t1-drv-2`, `t2-drv-2`
  and `t3-drv-2` (the exports), `x-drv-2`, `x-drv-4` and `x-drv-6` (the conflicts), `x-drv-5` and
  `d-drv-2` (exit 8), `d-drv-3` and `d-drv-4` (the finish).
- **The first Pod.** The incident ledger records 0.5 GPU-min for it. Its runtime check ran there.

**Pod time.**

| Pod | Created | Stopped | Pod-hours (`pods.tsv`) |
|---|---|---|---:|
| `2xyqxu4mgc0yos`, US-MO-1: unfit (incident 1) | 18:16:52Z | 19:17:49Z | 1.016 |
| `jfvugtct592agr`, US-TX-4: the repeat | 19:19:53Z | 23:34:42Z | 4.247 |
| **Total** | | | **5.263** |

**Spend.** Runpod's pod billing by day, read after the buckets of 2026-10-30 had posted
(`billing-final-20261030.json`, archived beside addendum 1; it supersedes the earlier read in
`addendum-1/runpod/billing-20261029-day.json`, taken before the last hour of 2026-10-29 was final):

| Pod | Billed: GPU + disk = total |
|---|---|
| `2xyqxu4mgc0yos`, US-MO-1: unfit (incident 1), 2026-10-29 | $1.10 + $0.03 = $1.14 |
| `jfvugtct592agr`, US-TX-4: the repeat, 2026-10-29 (its stopped volume included to midnight) | $4.63 + $0.15 = $4.78 |
| `jfvugtct592agr`'s stopped volume, 2026-10-30, to its deletion at 00:42Z | $0 + $0.03 = $0.03 |
| **Total, 5.263 Pod-hours** | **$5.95** |

- The GPU amounts match the Pod-hours at $1.09/h: 1.013 h and 4.245 h.
- Each total is rounded from the billed amounts, not summed from the rounded parts.

## Archive and provenance

- **Where it is.** `$ARTIFACT_REMOTE/dryrun/study3-pod-repeat-20261029/` only; never the bucket's
  `runs/` or `data/`, and no `runs/archive/` inventory.
  - **`evidence-a1/`: the immutable snapshot.** 1790 files are listed in `SHA256SUMS`, which is
    itself the 1791st object. It holds:
    - `DRY-RUN-NOT-A-RECORD.txt` and `env.sh`;
    - the five run roots whole: `runs-X` 104 files, `runs-T1` 32, `runs-T2` 31, `runs-T3` 31,
      `runs-D` 41;
    - `copies/` (20 copies, 832 files), `logs/` (344), `attempts/` (42 runners, 252 files),
      `receipts/` (76, with the two bundles, the seed patch and `bundles.sha256`), `inputs/` (8)
      and `tools/` (26);
    - `external/` (10 W&B core logs) and `LINKS`, which records 188 links: 141 inside, 47
      `external`.

    The snapshot's empty `runs/` folder has no object.
  - **`kept-checkpoints/`: the exports.**
    - `T1/…/kept-1/`: 3 objects;
    - `T2/…/kept-1/`: 2 objects, no checkpoint;
    - `T3/…/kept-1/`: 3 objects, the corrupt `.pt` among them;
    - `D-unrouted-20261029T224727Z/`: 2 files, step 44's keep.
  - **`pod1-2xyqxu4mgc0yos/`**: the first Pod's 79 files, with its inventory beside them.
  - **The upload inventory** `inventory-study3-pod-repeat-20261029-a1-try1.tsv`, as its own object.
  - **`addendum-1/`: what was recorded after the freeze**, in a separate folder, 347 files. The
    snapshot is left unchanged. It holds:
    - `logs/` (24 files): the final `checks.tsv` and every check output from the first freeze attempt
      on, earlier attempts' `<id>.<n>.out` included, with the post-freeze checks `Z-post-x-a5`,
      `Z-post-x-harness` and `Z-post-t3-corrupt`;
    - `upload/` (24 files): the upload inventory and receipt of every try, the secret scan's matches,
      the scanner's source and its sha256, `checks-final.tsv`, the first Pod's inventory, and step
      54b's `final-records-try1/` (3 manifests, 9 `kept-check` receipts, 3 inventories, with their
      own `SHA256SUMS`);
    - `receipts/` (15 files): the live receipts as step 55 copied them;
    - `receipts-all/` (78 files): every receipt on the Pod's volume, copied and verified just before
      the stop, `deviation-step54a-empty-runs.txt` among them;
    - `ops/` (199 files): the step scripts and their console logs as run, with their sha256 list:
      the phase-1 scripts of steps 3–11 (`ops-phase1/`) and the later phases' (`ops/`), which were
      not in the freeze list;
    - `runpod/`: the stop answer (`get-pod.json`, `stop-pod2.txt`) and the billing for 2026-10-29
      (`billing-20261029-day.json`);
    - `stop-and-cost.md` (the stops, the Pod-hours and the billed amounts), `README.md`, and
      `SHA256SUMS` over 345 files.
  - **`addendum-1-inventory-try1.tsv`**: the addendum's inventory (path, size, sha256), as its own
    object.
  - **`billing-final-20261030.json`** (sha256 `a20d5416ca7378eb9d33512768a6a3c180bfb59c14be2e09ebfb8289ea7a815a`):
    the final billing read, written after addendum 1 and archived beside it, read back equal.
- **The checks in each.**
  - The snapshot's `logs/checks.tsv` holds 323 rows: 320 PASS, 3 FAIL.
  - The final log holds 346: 342 PASS, 4 FAIL.
  - The snapshot's log is a byte-for-byte prefix of the final one.
- **Verification.**
  - `archive.py`, second attempt: 1791 files, 1791 verified (size and MD5), 0 mismatched.
  - The read-back of every object passed `sha256sum -c`.
  - The snapshot was unchanged after the upload, and the inventory was uploaded without overwriting.
  - The local copy of the snapshot passed `sha256sum -c`.
  - Addendum 1: `archive.py` reported 347 files, 347 verified, 0 mismatched. Its read-back matched
    all 345 files of its `SHA256SUMS` by sha256. The nested `upload/final-records-try1/SHA256SUMS` is
    not in that list; its own check and `archive.py`'s MD5 cover it.
  - `final-records-try1/` verified on the Pod and again locally.

| For the aggregate report | sha256 |
|---|---|
| The upload inventory (`inventory-study3-pod-repeat-20261029-a1-try1.tsv`) | `b35f6cd7d324c7158056b343cef1ff72a5bf81e01512a3f2a28fd696d5da136a` |
| The snapshot's `SHA256SUMS` | `bc35e7f22bf21381dee924c70d173753efe4a9c4edda474ae4ab32f92630fa48` |
| The first Pod's inventory (`inventory-20261029-pod1-2xyqxu4mgc0yos.tsv`) | `c7505c8e14505279950698b17c4e224dfe7ac788c85bdb2bb14e65fd0522183b` |
| `final-records-try1/SHA256SUMS` | `b038262a6c34f3f6b848730bd977e1a32d6748f694c0ae3bc4c77ff2e11dafa0` |
| The final `checks.tsv` (346 rows) | `bf39899007dd73be375c62ce34155608cfea486682d40a2e4130fc9105ed2da6` |
| The scan's matches (`secret-scan-matches-a1.tsv`) | `58a9e584eed638c341b3167852d0e3dabcae1620becc9baf31571bae2649cef9` |
| Addendum 1's `SHA256SUMS` | `dde4ee129115cc3eb9c382cd3b0cdc5a6a151c6039111a29aea9a8f25035b1b2` |
| Addendum 1's inventory (`addendum-1-inventory-try1.tsv`) | `8e750c354faa88d47a6ac1c3b4a9c7ff020a9aa119494e975738713df1ad6eea` |
| `tools.sha256` | `ae93658998ee7b774e948097d5d392c9f5d3f09994d0e0fe62b619361542f64e` |
| `zfr-gate.bundle` | `5b023776fcd3dc719080badf19d49f2e834b049766c4a1b34119065575f6af9f` |
| `zfr-dry-seeds.bundle` | `3604579089aee3b496ad072b92f7921b27db0020807c05c12f6eaefc45cf335c` |
| `dry-seeds.patch` | `b4bc493db6b4839d0edb02844f57d36e3aed0d4ecfca95ce09ca0a9c7f2f73d2` |

**What is left where.**

- **The second Pod.** Stopped at 23:34:42Z; `desiredStatus` EXITED, confirmed through the API. It was
  deleted at 2026-10-30T00:42Z, after addendum 1's 347 objects were checked in the bucket. Its volume
  went with it: the run roots, the copies, the data, the volume copies of the exports, the snapshot
  and the read-back. No Pod of the repeat is left.
- **The container disk** held, at the close:
  - T3's corrupt `.pt`, whose export is verified on the volume and in the bucket;
  - X's a5 `.pt` and harness baseline. X is not technical, so by the export rule they had no export.

  They went with the container disk at the stop.
- **The first Pod** was deleted at 19:23Z, after its 79 objects were verified in the bucket.
- **Deleted on purpose on the Pod.**
  - The six wipes, each after its receipt. Five of them removed files, seven in all; x4 found none.
  - T1's `.pt`, by `finish --release` after its verified export.
  - `checks/` in step 27's copy of `DATA_DIR`, the fixture of its fault.
- **What remains of the repeat** is the bucket's dry-run prefix and the local copy (`evidence-a1/`,
  `final-logs/`, `final-receipts-all/`, `upload/`, `final-ops/`, `pod1-2xyqxu4mgc0yos/`,
  `addendum-1/`).
