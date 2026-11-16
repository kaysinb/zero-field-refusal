<!-- Draft of docs/reports/study3-dry-run-part-1.md. Links are written relative to docs/reports/. -->

# Study 3: the Pod dry run, Part I

Part I of study 3's dry run ran on 2026-10-28, on one rented Runpod Pod (an NVIDIA L40S). It
covers the GPU-side drivers in the project image with the real model. Part II comes later and
covers the simulation and the analysis and report drivers on synthetic records (amendment 4 of the
[registration](../study3-preregistration.md)).

Nothing of Part I counts for study 3:

- every real-model stage seed was a dry seed, 90000–90599, outside the registered 10000–10599
  (the test layer's fixture seeds and the instruments' fixed seeds are not stage seeds);
- every record lies under the dry-run folder of the Pod volume and under the bucket's
  `dryrun/study3-pod-dryrun-20261028/` prefix, never under `runs/` or `data/`.

No K, D or verdict value was read; only exit codes, states, kinds, file presence and GPU minutes
were. The engineering record's `checks.dry_run` is not marked passed: that needs Part II too.

## Summary

- **Checks.** 179 checks ran: 177 passed and 2 failed.
  - The frozen evidence snapshot holds the first 172 (170 passed, 2 failed). The last seven, step
    34's from `Z-freeze` on, were recorded after the freeze; they and the final `checks.tsv` are
    in addendum 1 (below).
  - The two failures:
    - `T-selection-T`, step 11: five tests that run run.sh by bash failed on the Pod. There
      `/etc/rp_environment` resets PATH, so the tests' stand-in python is bypassed. The same five
      also fail in the dry-seed checkout's selection, so the seed proof's local baseline (a local
      gate junit in place of the Pod's) does not establish their cause in the Pod comparison.
      run.sh's guard refusals on the Pod are left uncovered.
    - `G2b-launcher-other`, step 25: the launcher refused F's completed run (exit 1, expected 0),
      because `finish` had deleted its endpoint checkpoint from the checkpoint folder that every
      copy shares. No child started; the launcher's skip branch is left uncovered.
  - Three criteria are not demonstrated (step 25's skip branch, step 27's child-published
    outcome, J7 (d) for `responses/`).
- **Separation.** Section 1 of the procedure held. The real record paths were absent throughout,
  both checkouts stayed clean at their commits, and G1 and G2 ended whole (step 33).
- **D1.** One engineering case measured 37.0 GPU-min. Option B is decided before any engineering
  seed.
- **D2.** SIGTERM to the driver's bash alone, during case 2's launcher attempt, behaved as the
  local experiment predicted, with one difference in timing. The driver's runner recorded exit
  143 at 17:06:32Z, 22 s after the SIGTERM and after the EXIT handling, not at once. D2 ended on
  the three required observations: the group empty, the end line recorded, the GPU idle.
- **Time.**
  - GPU jobs took 91.1 min (1.52 h), against an estimate of 114–230 min.
  - The Pod ran from its creation at 15:17:53Z to its stop at about 20:05Z, about 4.8 h, against
    an estimate of 5.8–11.3 h.
- **Cost.**
  - Part I's Pod `5qywtsjpchgr6z` (US-TX-4): $5.37 billed ($5.21 GPU, $0.16 disk), ≈ $5.22 as GPU
    hours × $1.09 by uptime.
  - An earlier Pod, `k89018y06rfzq1` (EU-NL-1), never started its container (the image pull
    failed): $1.03 billed, ≈ $1.02 by uptime.
  - Total: $6.40 billed, ≈ $6.24 by uptime (GPU; the disk is billed besides, about
    $0.03/h per Pod).
- **Closing state.** The Pod was stopped at about 20:05Z (EXITED). It was deleted later, after the
  archive verified (1166 of 1166) and the local copy of the snapshot was checked.
- **Archive.** The immutable evidence snapshot (1166 files) is archived under
  `$ARTIFACT_REMOTE/dryrun/study3-pod-dryrun-20261028/evidence/`. All 1166 objects verify, and the
  read-back matches.
  - Inventory sha256: `e11d11b4f84a37a2091fa61c672a6dfa80894c37a45dcd7b1102fc8632495a11`.
  - `SHA256SUMS` sha256: `9c90fcfd2682b220bd24b1a680ac15195f472b2bf83a9557ed69b723b2805bff`.
  - Addendum 1, under `$ARTIFACT_REMOTE/dryrun/study3-pod-dryrun-20261028/addendum-1/`, adds
    what was recorded after the freeze and leaves the snapshot unchanged. It holds 20 files, and
    its read-back matches.
    - `SHA256SUMS` sha256: `d8fbecbf900dbcc7b0710997ed6d5239955b34ba1a8f5d01998a5db6127ddd3c`.
    - Inventory (`addendum-1-inventory.tsv`) sha256:
      `6fe6b156a400750c61d18ca919f9875d763c8545fa421f55ecc5b4072471abb9`.

## Setup

| Item | Value |
|---|---|
| Pod | `5qywtsjpchgr6z`, US-TX-4; NVIDIA L40S (46068 MiB), driver 595.91.07, CUDA 13.2; container disk 80 GB (overlay); volume 150 GB (XFS); created 15:17:53Z, container up 15:22:21Z, stopped about 20:05Z, deleted after the archive was verified |
| Earlier Pod | `k89018y06rfzq1`, EU-NL-1, created 14:21:47Z: its container never started, because the image pull failed ("unexpected EOF"). Stopped about 15:18Z and deleted; nothing of Part I ran on it |
| Image | the project image, digest `sha256:9554679e749e405ad4d85e5a3c3416e687f8c3e7d443d289a18f794704810da5`; `/opt/project-deps/pyproject.toml` `039db0b8…`, `uv.lock` `1e0ba860…` (`receipts/image.txt`, `receipts/image-manifests.txt`) |
| Deadline | cap 2026-10-29T04:17:17Z; windows W1–W5 set by the file watcher, the last W5 = 2026-10-28T22:55:16Z (`receipts/deadline-watcher-lines.txt`) |
| W&B | `WANDB_MODE=offline`, no W&B key on the Pod |
| Data | `DATA_DIR` on the volume, restored from four of the record store's inventories (`receipts/data-needs.json`); dry streams of 90000 and 90200 archived under the dry prefix's `data/` (12 objects, inventory sha256 `d016ed1f3fe700f447517d8590d7e108f1162de26e38e3b3180c809278010d63`) |
| Checkpoints | `/root/ckpt-dryrun` on the container disk |

### The commits actually run

| Role | Commit |
|---|---|
| Gate: environment and provenance checks, test selection | `07b756883dc7f13381b67eda5c5a069edd86ef31` |
| `DRY_SHA`: the seed commit; every GPU job and every reader of dry records | `1968e468bec42d54de4a671e8a22cc651e09278d` |
| `FIX_SHA`: the throwaway fix of the engineering cycle (steps 25, 29–31) | `7c1b619c0cd87eb6ffdae44c81bd4371d9bf6e13` |
| Versions clone, commit 1: the versions record alone (step 24) | `30f7421a2d0e28ba7a5f66ffbe47155d23038d14` |
| Versions clone, commit 2: another commit (step 24) | `8765b0b98cea95cd6f9d5d70e5b0c6d6b5e7db84` |

`DRY_SHA` sits on top of the gate and changes only seed lines. Step 2 proved this locally, and
steps 6 and 11 again on the Pod (`seed_commit_check.py`, and the patch re-derived to the same SHA). The gate
bundle's history reaches `stage0` at `d6eaaf210516d5cd904c4925d4b5add77efa5cee`. `DRY_SHA`
and the throwaway clones' commits are local: never pushed or merged.

### Bundles, patch and tools (sha256)

| File | sha256 |
|---|---|
| `zfr-gate.bundle` | `626231f92849e757c0a50778983e9b460c875808ddbf122a7d7598ba50a2b46b` |
| `zfr-dry-seeds.bundle` | `ecd3fccbeebef4918939b5ca49b7f3cf6470ae05ecd66bbaccaaa39f9ce4f25f` |
| `dry-seeds.patch` | `b4bc493db6b4839d0edb02844f57d36e3aed0d4ecfca95ce09ca0a9c7f2f73d2` |
| `tools.sha256` as run in steps 4–33 | `7de705b033369193b0bd6db47bbca996742d548c5332b90f385059d2162037fa` |
| `tools.sha256` as run in step 34 (only `freeze_evidence.py` changed; see the deviations) | `ea2665b5240090133f2f5ecc3917f55ce43eaa4f731c0d87a9e765961d2f009a` |

The runner scripts shipped beside the tools are not listed in `tools.sha256`. Their sha256 are:
`run_detached.py` `a1ffeff5…`, `run_registry.py` `8bff7b6d…`, `night_watch.py` `98bbd2d9…`
(`receipts/deviation-step6-bundle-verify.txt`).

## The checklist

Results are against the binding checklist; the map is section 4.8 of the procedure.

- **Columns.**
  - *Real records*: checks on the dry records this Pod made with the real model.
  - *Test layer*: the test selection in the image, synthetic worlds driven by the real scripts,
    run on the Pod (step 11).
- **Evidence paths.**
  - Paths are relative to the snapshot (`…/evidence/`).
  - Each check id is a row of `logs/checks.tsv`, with its output in `logs/<id>.out`.
  - Runner outcomes are in `attempts/<name>/status.json`.
  - The rows of step 34 from `Z-freeze` on were written after the freeze. By design they are
    outside the snapshot. The final `logs/checks.tsv` and these seven checks' `.out` files are in
    addendum 1 (`addendum-1/logs/`).

### Environment and provenance

| Checklist line | Result | Steps | Check ids | Layer | Evidence |
|---|---|---|---|---|---|
| The real `project-run` environment from the image | pass | 4, 7, 12 | `E1-launcher`, `E1-venv-SRC`, `E1-venv-DSRC`, `E1-manifests-SRC`, `E1-manifests-DSRC`, `E1-clean-SRC`, `E1-clean-DSRC`, `E1-cpu-runtime`; runner `gpu-runtime` completed 0 | real | `receipts/startup.txt`, `receipts/image*.txt`, `logs/E1-*.out`, `attempts/gpu-runtime/` |
| A gate checkout with the full Git history; `progress` without `unknown_commit` | pass | 2, 6, 10, 23; 11 | `E2-head`, `E2-not-shallow`, `E2-history`, `E2-base-commits`, `E2-dsrc-head`, `E2-dsrc-history`, `S-fix-line`, `S-fix-commit`, `E2-progress-empty`, `E2-no-unknown-empty`, `E2-progress`, `E2-no-unknown`; `test_the_package_changes_no_gpu_path` passed | real; test layer | `logs/E2-*.out`, `logs/pytest-T.xml` |
| The versions checks: `versions`, and `check` from the same `$SRC` | pass, as refusals only | 24; 11 | `E3-versions`, `E3-check-record` (1), `E3-check-dsrc` (1), `V-other-line`, `G2b-check-other` (1); `test_check_holds_before_a_validation_launch_and_refuses_otherwise` passed | real (negative); test layer | `logs/E3-*.out`, `logs/G2b-check-other.out`, `receipts/versions-clone-commits/` |
| meta.json: the campaign header; `git.commit` and `git.dirty`; the numerics setup | pass (4 run directories, `git.commit` = `DRY_SHA`, clean) | 23 | `E4-meta` | real | `logs/E4-meta.out` |

### Writer, qualification plan and driver

| Checklist line | Result | Steps | Check ids | Layer | Evidence |
|---|---|---|---|---|---|
| The writer feeds `plan` and the driver on a writer-made campaign, one cell per run | pass. Case 1 ran whole: launcher run, baseline with diagnostic steps, `decided` exit 3, the rest, `finish` | 14, 15 | `W1-writer`, `W1-one-cell-per-run`, `W1-plan`, `W1-driver`, `J4-G1-before-launch`; runner `eng-driver-1` | real | `logs/W1-*.out`; `runs/qualification/c1/driver-engineering-20261028T162524Z.log` |
| An interruption and a resumption of the driver, the first seed's same-path FA control kept | pass | 17–22 | `F-writer`, `F-plan`, `F-driver`, `J4-G2-before-launch`; runners `kill-k2` 0, `eng-driver-3` 8, `kill-fa` 0, `fa-driver-2` 8, `fa-control-2` 0 (attempt 2, `--infrastructure`), `fa-driver-3` 1 (F complete and finished; P exhausted) | real | `attempts/`, `runs-numerical/qualification/c1/driver-numerical-*.log` |
| `qualify.py plan` on a partial campaign after a stopped driver | pass | 17, 21 | `W3-plan-eng`, `W3-complete-1` (0), `W3-complete-2` (3), `W3-plan-fa` | real | `logs/W3-*.out` |

### Confirmation guards

| Checklist line | Result | Steps | Check ids | Layer | Evidence |
|---|---|---|---|---|---|
| `run.sh`, run by bash, with its guard: it refuses a new block while a started block is incomplete, once N is reached, once the cap is reached | **not demonstrated on the Pod** (coverage gap below) | 11 | `T-selection-T` FAIL (exit 1, expected 0): the five run.sh-by-bash tests failed, because `/etc/rp_environment` resets PATH and their stand-in python is bypassed; the same five also fail in the dry-seed checkout's selection. The guard's own refusal tests passed on the Pod: `test_the_guard_refuses_while_a_started_block_holds_a_run_not_ended`, `…_once_every_counter_has_reached_n`, `…_once_the_cap_has_started`. The five run.sh-by-bash tests failed. | test layer | `receipts/deviation-step11-pod-env-tests.txt`, `logs/pytest-T.xml`, `logs/T-selection-T.out` |
| No unauthorized child: a held run | pass | 26, 29; 11 | `G2a-hold`, `G2a-launcher` (1), `G2a-recovery` (1), `G2a-complete` (11), `G2a-journal-unchanged`; runner `B-driver` 11; `Y-held-launcher`, `Y-held-recovery`, `Y-held-complete`, `Y-held-never-issued` | real; test layer | `logs/G2a-*.out`, `logs/Y-held-*.out`, `receipts/refusal-reasons.tsv` (each refusal names "held") |
| No unauthorized child: a run under another commit | pass at the qualification level: no child started. `G2b-launcher-other` FAIL (exit 1, expected 0): the launcher refused F's completed run, whose endpoint checkpoint `finish` had deleted from the shared checkpoint folder; the skip branch is left uncovered | 24, 25, 29; 11 | `G2b-stale` (10), `G2b-launcher-other` (1, expected 0), `G2b-journal-unchanged`, `G2b-check-other` (1), `Y-other-commit` (1); confirmation level: `test_a_run_under_a_commit_other_than_the_pinned_one_does_not_count[confirmation]` passed | real; test layer | `receipts/deviation-step25-launcher-other.txt`, `receipts/step25-stop.txt`, `logs/G2b-*.out` |
| No unauthorized child: a recovery round after `check` | pass for a qualification round (no `check` precedes one): one round, then a second refused. The validation round after `check` is on the test layer only | 27; 11 | `J1-crash-end` (137), `J1-settled` (1), `G2c-second-round` (1); `test_a_recovery_round_of_one_block_gives_a_later_block_no_second_attempt` passed | real; test layer | `logs/J1-*.out`, `logs/C1-new-lines.txt` (3 new lines, each naming P's label) |

### Failures near publication and the journal end

| Checklist line | Result | Steps | Check ids | Layer | Evidence |
|---|---|---|---|---|---|
| A crash near the publication of the outcome, and near the journal's end line | pass. The child-published technical outcome (step 27) is not demonstrated | 19, 20, 27; 11 | `F-no-byte`, `F-start-without-end`, `F-checkpoint-kept` (runner `fa-launcher` 137, killed after 0 of 314 bytes of the end line); `J1-crash-end` (137, 0 of 336 bytes), `J2-torn-end` (137, 120 of 336 bytes); the settled end in step 20 | real; test layer | `receipts/deviation-step27-child-outcome.txt`, `receipts/step27-stop.txt` |
| Torn tail: the launcher's repair (a newline, then the `torn` record) | pass | 27, 28, 31 | `J2-repaired`, `J2s-repaired`, `Y3-repaired` | real | `logs/J2-repaired.out` etc. |
| Torn tail: exit 12 from the launcher, `qualify.py` and the driver | pass, with the reason checked in each message | 27, 28, 30, 31 | `J2-launcher-12`, `J2-qualify-12`, `J2s-launcher-12`, `J2s-complete-12`, `J3-launcher-12`, `J3-complete-12`, `J3-verdict-12`, `Y2-launcher-12`, `Y2-complete-12`, `Y3-launcher-12`, `Y3-complete-12`; runners `C2-driver`, `A1-driver`, `A2-driver`, `Q2-driver`, `Q3-driver` 12 | real | `receipts/refusal-reasons.tsv` |
| Torn tail: `--resolve-torn` | pass | 27, 28, 31 | `J2-resolve`, `J2s-resolve`, `Y3-resolve`; `J2-whole`, `J2s-whole`, `Y3-whole` | real | `logs/*-resolve.out`, `logs/*-whole.out` |
| Torn tail: a lost `held`/`superseded` line restored by the two repair calls before the resolution | pass: `superseded` by `restart-case` (A2), `held` by `supersede` (Q3) | 28, 31 | `J2s-crash` (137), `J2s-torn`, `J2s-repair`, `Y3-crash` (137), `Y3-torn`, `Y3-repair` | real | `logs/J2s-*.out`, `logs/Y3-*.out` |
| Torn tail: study 3's reading after the resolution; the fixation refuses while unresolved | pass for `progress`, before and after. The fixation's refusal is on the test layer only | 28; 11 | `J2s-progress-before`, `J2s-progress-after`; `test_the_fixation_pins_no_log_with_an_unresolved_torn_line` passed | real; test layer | `logs/J2s-progress-*.out` |
| Torn tail: two writers on one log, both under the lock | pass on the test layer, with real processes on the Pod volume | 11 | `test_two_writers_take_turns_on_one_log` passed | test layer | `logs/pytest-T.xml` |
| A published record without its journal line: crash, exit 12, repair, the held label still refused | pass | 28, 30 | `J3-crash`, `J3-missing`, `J3-repair`, `J3-whole`, `J3-after`; `Y2-crash`, `Y2-missing`, `Y2-repair`, `Y2-whole`, `Y2-held-still` (1), `Y2-held-complete` (11) | real | `logs/J3-*.out`, `logs/Y2-*.out` |
| `missing_lines` and `torn_lines` empty before the first launch | pass | 9, 14, 18 | `J4-G1-empty`, `J4-G2-empty`, `J4-G1-before-launch`, `J4-G2-before-launch` | real | `logs/J4-*.out` |
| An append fsyncs the log, `launch/` and the runs root; a directory fsync works on the volume (XFS) | pass | 9, 15; 11 | `J5-append`, `J5-dir-fsync`; `test_every_append_fsyncs_the_log_its_folder_and_the_runs_root[*]` passed | real; test layer | `logs/J5-*.out`, `receipts/volume-mount.txt` |
| A closure record, end to end | pass on the test layer only | 11 | `test_closure_writes_the_record_a_held_runs_step_names`, `test_closure_writes_the_closure_of_a_run_under_another_commit` passed | test layer | `logs/pytest-T.xml` |
| Sentinel files in the sealed folders: nothing reads them (J7, in its decided scope) | pass in its scope. (d) for `responses/` is not demonstrated | 17, 20–22, 32; 11 | `J7-plant`, `J7-progress`, `J7-writer`, `J7-verdict-eng`, `J7-complete-eng`, `J7-check`, `J7-fixation`, `J7-verdict-num`, `J7-complete-num`, `J7-launcher-eng`, `J7-launcher-num`, `J7a-readers`, `J7b-launcher`, `J7b-named`, `J7c-no-judge`, `J7-inotify-on`, `J7d-plant-1`…`4`, `J7d-watches`, `J7d-heard`, `J7d-scan` | real; test layer | `logs/sealed-*.log`, `logs/sentinel-watch-{1,2,3,4}.log`, `receipts/gpu-sentinels.tsv`, `receipts/copy-sentinels.tsv` |

### Extra lines

| Line | Result | Steps | Evidence |
|---|---|---|---|
| D1: the engineering stage's GPU time from a measured dry case | measured; option B decided first | 16 | `D1-price`; `receipts/d1.json` |
| D2: SIGTERM to the driver's bash while a launcher runs; the attempt recorded; a rerun resumes at the next baseline | pass | 15, 17 | `D2-recorded`, `D2-no-baseline` (1), `D2-group-empty`, `D2-gpu-idle`; runners `term-l2` 0, `eng-driver-1` 143, `eng-driver-2` (resumed at case 2's baseline) |
| D2's documented negative: SIGTERM to the launcher alone does not stop the driver | not rehearsed (code and operations only) | — | — |

### The procedure's own checks

| Line | Result | Steps | Check ids |
|---|---|---|---|
| Section 1's separation held to the end | pass | 33 | `Z-real-paths-absent`, `Z-checkouts`, `Z-gpu-idle`, `Z-no-process` (1), `Z-G1-whole`, `Z-G2-whole` |
| The seed commit's proof | pass, with a fresh local gate junit (119 of 119) in place of the Pod's (step 11). The five run.sh-by-bash failures also occur in the dry-seed selection, so this substitution does not establish their cause in the Pod comparison | 11 | `T-selection-DSRC_T` (1: 119 cases, 79 passed, 38 failed, 2 errors), `S-seed-proof` |
| Archive and verification before the stop | pass | 34 | `Z-receipts`, `Z-freeze`, `Z-snap-before`, `Z-upload`, `Z-readback-get`, `Z-readback-check`, `Z-snap-after`, `Z-inventory-up` (the seven from `Z-freeze` on are in addendum 1) |

## What Part I does not cover

Every path below is uncovered by Part I. Each is listed with what stands in for it, if anything.

1. **The versions check's passed route on real records.** No dry state holds a committed
   validation fixation. Step 24 shows refusals only. Part II.
2. **run.sh's guard refusals on the Pod.** Not demonstrated: they are checked locally only, because
   `/etc/rp_environment` resets PATH and the tests' stand-in python is bypassed. The same five
   tests also fail in the dry-seed checkout's selection, so the local baseline substitution (a
   local gate junit in place of the Pod's, step 11) does not establish their cause in the Pod
   comparison. The tests are to be made hermetic, and both
   checkout selections rerun on the Pod under that one corrected harness, before Part I is
   declared passed for the engineering stage. The guard refusals on real records belong to Part
   II, since no dry state reaches confirmation.
3. **A run under another commit: the confirmation level on real records.** Test layer only.
4. **A validation recovery round after `check` on real records.** Test layer only.
5. **The fixation's refusal over an unresolved torn line on real records.** Test layer only. A dry
   state refuses first for missing records.
6. **A closure record on real records.** No dry state names a closure. Test layer only; Part II.
7. **The launcher's skip branch for a completed run under another commit (step 25).** Not
   demonstrated: F's endpoint checkpoint had been deleted by `finish` in the shared checkpoint
   folder (the section on the shared checkpoint folder below).
8. **A child-published technical outcome at the end-line crash (step 27).** Not demonstrated.
   - The no-GPU round's child raises in `runtime.configure_numerics` (`runtime.py:142`, called
     from `scripts/train.py:21`) before the tracker makes a run directory.
   - The technical outcome comes from the launcher's settlement ("no run directory"), not from
     the child.
9. **J7 (d) for `responses/` during the GPU part.** Not demonstrated.
   - No planted run had a `responses/` folder, because K-only runs keep none.
   - The watch adds existing folders only, so watches 1–4 watched `observations/` and `judge/`
     alone (4, 6, 6 and 6 folders).
   - The launcher's child before any run had completed (steps 15, 19) had nothing planted beside
     it.
   - The repeat creates empty `responses/` folders in the completed dry runs and watches them. It
     puts no response file in them, since one would change the K-only completeness check.
10. **D2's documented negative** (SIGTERM to the launcher alone). Not rehearsed.
11. **A container restart that wipes the container disk.** Not produced. Every interruption in
    Part I (steps 15, 17, 20, 21) kept the container disk. The section below states how the
    procedure handles it. Its cases, the four open items included, are added to the affected-path
    repeat.

Two conditional gaps did not arise:

- case 1 and F each ran their rest (`decided` exit 3), so the diagnostic rest path and F's rest
  are covered;
- inotify worked: no `INOTIFY-UNAVAILABLE` or `INOTIFY-DEAF` line, and each watch ended with
  `WATCH-END`.

## Coverage gaps and how each is to be demonstrated

Each gap below is repeated on the Pod before the engineering stage, together with the Part I paths
that the fixes for these gaps and for the open items affect (the affected-path repeat).

| Item | Checklist line | What was shown instead | Plan to demonstrate it | Prerequisite |
|---|---|---|---|---|
| Not demonstrated 1: step 25's skip branch | No unauthorized child: a run under another commit | The launcher from `FIX_SHA` refused F's completed run as one that cannot stand (its endpoint checkpoint missing) and started no child; exit 1, journal unchanged. Also `complete` stale (10), `check` from another commit refused, and P's run under the fix commit refused (used up its attempts) | a repeat on the Pod before the engineering stage | Run the skip check before any finish/release that removes its checkpoint, using the original records; alternatively, create the campaign with an independent checkpoint directory from the outset. |
| Not demonstrated 2: step 27's child-published outcome | A crash near the publication of the outcome (the technical form) | the end-line crash (exit 137 after 0 and after 120 of 336 bytes) and the launcher's settled technical end; the `done` form on real GPU records (step 19: killed after 0 of 314 bytes, run directory completed); the test layer | a repeat on the Pod before the engineering stage | a round whose child fails technically after the tracker has made its run directory (the no-GPU child fails before it) |
| Not demonstrated 3: J7 (d) for `responses/` | Sentinel files in the sealed folders: nothing reads them | watches 1–4 on `observations/` and `judge/` of every completed run, beside every later GPU job: no `SENTINEL`, `JUDGE`, `OTHER` or `OVERFLOW` line; the content scan clean; `responses/` sentinels in copies S1 and S2 for the readers and the launcher (step 32 (a)–(c)) | a repeat on the Pod (the GPU part) before the engineering stage | an empty `responses/` folder created in each completed K-only dry run and watched beside the GPU jobs. No response file is put in it, since one would change the K-only completeness check |
| Failed: `T-selection-T` (step 11) | run.sh, run by bash, with its guard refusing a new block | 114 of 119 passed on the Pod, the guard's own refusal tests included; on the Pod run.sh ran the real guard and stopped at its refusal. The five run.sh-by-bash tests pass locally (119 of 119, `logs/pytest-T-local.xml`), but the same five also fail in the dry-seed checkout's selection on the Pod, so the local baseline substitution does not establish their cause in the Pod comparison | a repeat on the Pod before Part I is declared passed for the engineering stage: both checkout selections, the gate's and the dry seed's, rerun under one corrected, hermetic harness | the five tests made hermetic: the executable they run selected explicitly, so that `/etc/rp_environment`'s PATH cannot bypass it, the real guard assertions kept, and the executable that ran checked |
| Failed: `G2b-launcher-other` (step 25) | No unauthorized child: a run under another commit | as in the first row: refusal for a missing checkpoint, no child, exit 1 | a repeat on the Pod before the engineering stage | as in the first row: Run the skip check before any finish/release that removes its checkpoint, using the original records; alternatively, create the campaign with an independent checkpoint directory from the outset. |
| Not rehearsed: a container restart that wipes the container disk | none (the engineering stage's infrastructure route) | the code and the operating text read at `07b7568` (the section below); every interruption of Part I kept the container disk | The affected-path repeat rehearses disk loss, recovery and checkpoint export on the Pod. Decided-case control-only routing is exercised through synthetic driver tests in the Pod image; a real same-path FA control also runs under the repeat commit. Coverage is reported separately for each layer. | the fixes that the four open items call for |

## The shared checkpoint folder

**What step 25 found.** The checkpoint folder `/root/ckpt-dryrun` is on the container disk, and
every copy of the records shares it.

- Step 22's `finish` deleted F's endpoint `.pt`, as designed: F was complete and not technical.
- At step 25 the launcher from the other commit, run on copy B2 with `--only` F's label, assessed
  F's completed run. Its recorded endpoint checkpoint was gone from the shared folder.
- The launcher refused the run as one that cannot stand and exited 1, before any attempt and
  without a child.
- The procedure expected the skip branch (exit 0). That branch is reachable only while the run's
  checkpoint exists.

**What it shows.** The observed failure is exit 1 where 0 was expected, so `G2b-launcher-other`
failed. Its cause is the shared checkpoint folder: step 22's `finish` had deleted the checkpoint
the skip branch needs. The line's substance, that no unauthorized child starts, is shown; the
skip branch is left uncovered. From then on every expected refusal had its reason checked in the
launcher's message, not only its exit code (`receipts/refusal-reasons.tsv`):

- all of them named the expected reason;
- six exit-12 refusals over a torn line also named a missing line beside it.

Run-scoped expectations on copies therefore depend on what earlier steps deleted from the shared
folder.

### A container restart that wipes the container disk, in the engineering stage

This is read from the code and the operating text at `07b7568`. The three files are identical at
`d6eaaf2`. It was not rehearsed in Part I.

**Background.**

- The checkpoint directory (`tracking.endpoint_checkpoint`) is one directory on the container
  disk (`docs/operations.md:1462-1463`). A case needs its launcher run's endpoint and the harness's
  (`docs/operations.md:1494-1498`).
- A Pod stop or container restart loses them (`docs/operations.md:1526-1530`;
  `scripts/qualify.py:104-106`).
- The driver recreates the directory (`mkdir -p "$CKPT"`, `scripts/qualify.py:1965`), so its disk
  check still passes.
- For a case that is not complete, the driver calls the launcher before any part
  (`scripts/qualify.py:52-64`).
- The launcher treats a completed run whose recorded endpoint `.pt` is missing as a conflict ("the
  endpoint checkpoint … is missing", `src/zero_field_refusal/launch.py:788-812`, line 811). It
  stops with `CampaignError` before any attempt (`launch.py:929-934`), exit 1. A `--recovery`
  reason does not get past this, because the conflict is raised before a recovery round is
  considered (`launch.py:929-947`).
- A run that the journal names superseded is a used, technical attempt, not a conflict
  (`launch.py:702-706`).

| State when the disk is lost | What the driver does on its next run | What the procedure says happens | Registered route |
|---|---|---|---|
| A part started but not recorded (`.started`, no record) | `complete` 3, then the launcher refuses the run whose `.pt` is gone (exit 1). It stops there, before `part --resume` could exit 8: the exit-8 rerun (`qualify.py:86-88`; `operations.md:1499-1502`) applies only "while the Pod and its container disk are the ones it started on" (`operations.md:1503-1505`) | A case restart. `restart-case --infrastructure REASON` (`operations.md:1529-1548`; `qualify.py:103-127`, `1058-1121`) has its evidence in the interrupted attempt and the lost checkpoint (`qualify.py:1089-1090`, `1106-1108`). It voids the named attempts, keeping them (`qualify.py:123-127`), and logs the launcher run superseded (`qualify.py:1117-1118`). The launcher's recovery round with the same reason starts a new run (`operations.md:1546-1548`, `1552-1554`; `launch.py:942-947`, `968-970`). Each part then runs as its next attempt with `--infrastructure`, the baseline first, then the driver | Infrastructure: rerun under a new attempt id, not a mark (`docs/study3-preregistration.md:466`, `677-678`). One exception: an interrupted rest whose published trajectory results hold a failure is not restarted, and the failure keeps counting (`qualify.py:96-98`, `119-122`, `1096-1100`): scientific (no reruns to pass) or technical (the engineering cycle) |
| A part recorded, its case not finished | the same stop at the launcher (exit 1) | A case restart. The lost checkpoint is evidence enough, with no interrupted attempt needed (`qualify.py:1070-1072`, `1089-1090`). Recorded attempts that pass so far depend on the lost checkpoints, so they are named and voided with the others (`qualify.py:122-124`). A recorded completed failure refuses the restart and keeps counting (`qualify.py:119-121`, `1096-1100`) | Infrastructure for the voided attempts. A recorded scientific failure: no rerun. A recorded technical failure: the engineering cycle (`study3-preregistration.md:466-468`) |
| A launcher run completed, its `.pt` gone (no part started) | the same stop at the launcher (exit 1) | A case restart even before any attempt: "A launcher run whose checkpoint is lost is evidence enough" (`qualify.py:1070-1072`; `operations.md:1540-1541`). Then the recovery round and the parts. A launcher run still in flight when the disk was lost is settled technical by the next launcher (`launch.py:928`, `995-1004`). The driver then stops at "used up their attempts" (exit 1, `scripts/launch.py:116`), and the run is rerun through its recovery round with the reason (`operations.md:1506-1507`; `qualify.py:60-62`); Part I saw this for P (step 22) | Infrastructure |
| A case already finished | `complete` 0: the case is skipped; `finish` runs again and removes nothing (`qualify.py:41-42`, `843-864`). The launcher is not called for it, and under `--only` no other label is assessed (`launch.py:48-51`) | Nothing reruns. `restart-case` refuses a complete case (`qualify.py:1094-1095`) and does not count a checkpoint `finish` removed (`qualify.py:115-116`, `1042-1050`). A technical case is different: `finish` exits 9 and keeps its checkpoints for the diagnosis (`qualify.py:72-75`; `operations.md:1486-1489`), on the container disk, so a wipe loses them before the diagnosis | None needed for a passed or scientific case. A technical case keeps its technical route (the engineering cycle) |

**Open items:**

1. **A second loss in the same case.**
   - The registered infrastructure route reruns an interrupted attempt with no limit
     (`study3-preregistration.md:466`, `677-678`, `700-701`).
   - The launcher opens one recovery round per label (`launch.py:48-55`, `942-947`). A round
     allows one more attempt (`launch.py:968-970`).
   - So a second container-disk loss within one case leaves the label exhausted, with no further
     recovery round. The same holds for a first loss after the label's recovery round has already
     been used, for example after a technical launcher failure. The procedure names no route for
     that state.
2. **The sign of a lost disk.**
   - After a wipe, the driver's first stop is the launcher's refusal: exit 1, "Completed runs that
     cannot stand … the endpoint checkpoint … is missing".
   - It is not exit 8. `operations.md` describes the case restart but does not name this message
     as the sign that calls for it.
   - The same message also arises when the launcher is called directly on a finished case's label
     (step 25).
3. **A technical case's kept checkpoints.** They live only on the container disk until the
   diagnosis. The operating text names no copy to the volume for the engineering stage; the
   dry-run procedure's export rule (step 33) is its own.
4. **A decided case whose same-path control is still due** (the numerical stage only, FA on the
   first numerical seed; the engineering stage has no control).
   - The restart is refused, because the scientific failure keeps counting
     (`qualify.py:1096-1100`).
   - The driver stops at the launcher before the control (`qualify.py:52-69`), although the control
     reads no checkpoint (`qualify.py:68-69`).
   - The procedure does not say how the due control is then run.

The affected-path repeat rehearses disk loss, recovery and checkpoint export on the Pod. Decided-case control-only routing is exercised through synthetic driver tests in the Pod image; a real same-path FA control also runs under the repeat commit. Coverage is reported separately for each layer.

## J7: scope and limits

**Scope.** J7's scope is the one decided for Part I (step 32), stated as a scope decision, not as a
relaxation, and open to confirmation.

- It holds for Part I's K-only runs only.
- For regular validation and confirmation runs, `_response_problems` also opens
  `responses/<set>.jsonl`. That is outside the launcher's two allowlisted reads, so J7 (b) would
  fail on it. Their allowlist is a separate check, to be made later.

**What was shown.**

- **(a)** Study 3's readers and operating commands opened and listed nothing in a sealed folder:
  `J7a-readers` found no forbidden line.
- **(b)** The launcher made only its two registered reads, by call site: `OPEN-ALLOWED` of
  `observations/measurements.jsonl` and `LIST-ALLOWED` of `responses/`.
  - Beside them, `INOTIFY-ALLOWED` events of `observations/measurements.jsonl` (the positive
    control).
  - The launcher's output named `zz-sentinel-copy.jsonl` as each K-only run's stored response.
- **(c)** No line named `judge/`.
- **(d)** Watches 1–4, beside the GPU jobs of steps 17 and 20–22, logged 0 `SENTINEL`, `JUDGE`,
  `OTHER` or `OVERFLOW` lines and ended with `WATCH-END`.
  - They logged 8, 4, 0 and 6 `REGISTERED` events.
  - The content scan found no marker outside the sealed folders, and every sentinel unchanged
    (`J7d-scan`).
- **inotify** was available and heard its canary in every audit.

**Limits** (section 8 of the procedure):

- **`responses/`.** No sentinel can be in `responses/` during the GPU part. There the harness
  parts and the launcher's child are left to inotify alone. In Part I no `responses/` folder
  existed to watch at all (not demonstrated, above). The repeat watches empty `responses/`
  folders, with no response file put in them.
- **The launcher's child** is covered once, in step 21 (P's launcher run, watch 3: no event of any
  kind). Before step 17 no run had completed, so nothing could be planted.
- **The scan** checks the sentinels' content, not the dry measurement rows that the launcher's
  registered reads open. Comparing those would need the rows to be opened.
- **A clean scan** proves neither that nothing was read nor that no compressed leak occurred.
- **`REGISTERED` events** are not tied to a process.
- **The GPU part's sentinels** stay in G1 and G2 and are part of the evidence snapshot: a
  `judge/` folder and a foreign file in `observations/` of each completed dry run.
- **"GPU busy"** stands as defined: compute processes listed, or memory.used ≥ 1000 MiB. Missing
  or unreadable telemetry never counts as idle. In this container, compute processes are listed
  while GPU memory is in use (`receipts/deviation-step13-visibility-addendum.txt`).

## D1

**The measured case.** From `receipts/d1.json`, engineering case 1 took 37.0 GPU-min:

- launcher run 4.8 min;
- baseline with the diagnostic steps 4.7 min;
- rest 27.5 min.

The rest ran, so the case is fully measured (`flags` is empty).

| Quantity | Value |
|---|---|
| Engineering, central | 24 × 37.0 min + 1.8 h (the known case) + 0.6 h (θ₀, stage 0, streams) + 2.03 h (the dry run's own GPU time: measured, plus 77 central minutes left) = 19.23 GPU-h |
| Engineering, high | × 1.7 = 32.69 GPU-h |
| Decision | option B decided before any engineering seed |

D1 gates the engineering seeds only, and Part I ran to its end. The researcher set two rules
before D1:

- no engineering seed starts until option B is decided;
- the engineering seeds start only on a separate decision after D1's summary.

## D2

**What the local experiment predicted** (procedure, section 4.1):

- bash dies at once, and its EXIT trap runs;
- the runner records the worker's exit at once (`failed`, 143) and kills nothing;
- the launcher finishes its attempt and records its end;
- no later step of the driver starts.

**What the Pod showed.**

| Time (UTC) | Event |
|---|---|
| 17:06:10 | `term_driver.sh` sends SIGTERM to the driver's bash alone. Case 2's launcher (run id `20261028T170551Z-…-834e5711-a1`) runs on |
| 17:06:32 | The runner `eng-driver-1` records `failed 143`, 22 s after the SIGTERM, after the EXIT handling |
| 17:10:56 | The group is empty, the attempt's end line is in the journal, and the GPU is idle. `term-l2` ends `completed 0` |

D2 was not instantaneous: the runner recorded the exit 22 s after the SIGTERM, not at once as
predicted. The checks confirm the rest of the prediction:

- `D2-recorded`: the attempt has its start and an `end` with outcome `done`, and its run
  directory is `completed`;
- `D2-no-baseline`: no baseline was started;
- `D2-group-empty` and `D2-gpu-idle` passed.

D2 thus ended on all three required observations: the group empty, the end line, the GPU idle.

**After the stop.**

- The driver's rerun (`eng-driver-2`) skipped case 1 and ran its `finish`. Its launcher skipped
  case 2's done run, and it resumed case 2 at its baseline.
- `kill-k2` sent SIGKILL to the whole group about 60 s into that baseline.
- The next rerun (`eng-driver-3`) stopped at the interrupted baseline with exit 8.

## Deviations and rulings

| Step | What departed from the procedure as written | What followed: the observation, a decision or a correction | Evidence |
|---|---|---|---|
| 6 | `git bundle verify` refused outside a repository (git 2.43.0), exit 1, before any clone or check | The command was corrected in the procedure to the same check inside an empty repository. Rerun: both bundles verified, complete history | `receipts/deviation-step6-bundle-verify.txt`, `receipts/step6-stop.txt` |
| 11 | `T-selection-T`: pytest exit 1, expected 0. 114 of 119 passed; five tests that run run.sh by bash failed, because `/etc/rp_environment` resets PATH and the stand-in python is bypassed | The observed failure stays: exactly these five named tests. The same five also fail in the dry-seed checkout's selection, so the cause is not established by the Pod comparison. The researcher ruled that `S-seed-proof` use a fresh local gate junit (119 of 119) in place of the Pod's. Coverage gap: the specific refusals of run.sh's guard on the Pod are not demonstrated (checked locally only). The tests are to be made hermetic and both checkout selections rerun on the Pod under that one corrected harness, before Part I is declared passed for the engineering stage | `receipts/deviation-step11-pod-env-tests.txt`, `receipts/step11-stop.txt` |
| 11 | Once, the local checkout's `runs/` was listed, its output discarded; no file contents were opened, and nothing was printed. The procedure keeps the local `runs/` out of the dry run | Recorded | — |
| 12–13 | Step 12's runtime check ran 15 s, too short to show whether compute processes are listed in this container | The researcher ruled three read-only samples during step 13's prep. One compute process was listed at 12959 and 13213 MiB | `receipts/deviation-step13-visibility-addendum.txt`, `receipts/gpu-visibility.txt` |
| 13 | The pass criterion said 500 training responses per seed; the streams hold one chunk of 1000 (`chunk_size: 1000`, `runtime._extend_locked`) | The researcher ruled the "500" a wording error of the procedure, corrected to "at least 500 training responses per seed (one chunk of 1 000)". Nothing was rerun or changed. Step 13 passes | `receipts/deviation-step13-chunk.txt`, `receipts/step13-stop.txt` |
| 14, 16 | The scripts run carried one extra `echo` each beside the procedure's lines | Recorded | — |
| 25 | `G2b-launcher-other`: exit 1, expected 0 (the shared checkpoint folder, above) | The observed failure stays: the launcher refused F's completed run, whose endpoint checkpoint step 22's `finish` had deleted, before any attempt and without a child. The skip branch is left uncovered. Every later expected refusal had its reason checked. Step 25's text now states both branches | `receipts/deviation-step25-launcher-other.txt`, `receipts/step25-stop.txt`, `receipts/refusal-reasons.tsv` |
| 27 | The no-GPU round's child publishes no outcome: it fails before a run directory exists | The child-published technical outcome is left uncovered (not demonstrated), and Part I went on. For steps 27b–32 the researcher set a rule. A coverage-only gap, where every check passed, is recorded as not demonstrated and does not stop the run. A failed check, an unexpected exit, a safety issue, an exit 0 where a refusal is expected, or a guard refusal stops it | `receipts/deviation-step27-child-outcome.txt`, `receipts/step27-stop.txt`, `receipts/not-demonstrated.tsv` |
| 34 | Before step 34's first command, 30 of the 120 links under the paths to freeze were found to resolve outside them. These are W&B core's `<run>/wandb/offline-run-*/logs/debug-core.log`, linking to `/workspace/cache/wandb/logs/core-debug-<stamp>.log`: 4 targets, one link per offline run (G1 2, G2 2, copies 26). `freeze_evidence.py` refused any such link (96). Step 34 stopped; nothing was frozen or uploaded | The researcher ruled a change of `freeze_evidence.py` that allows exactly that kind of link. It is recorded in `LINKS` as `external` with its absolute target, and the target is copied once to `external/<its absolute path>` (in `SHA256SUMS`). Any other outside or dangling link still refuses (96). `freeze_evidence.py` changed from `83806a41…` to `b73e5301…`, `tools.sha256` from `7de705b0…` to `ea2665b5…`. On the Pod, `sha256sum -c tools.sha256` gave 22 of 22 OK. The procedure's step 34 and Appendix A state the change | `receipts/step34-stop.txt`, `receipts/deviation-step34-freeze-external.txt` |
| 34 | A read-only secret scan of the frozen snapshot was added between the freeze and the upload | By the researcher's ruling (the next section) | addendum 1: `secret-scan/secret-scan-paths.txt`, `secret-scan-classes.tsv`, `secret-scan-ruling.txt` (outside the snapshot) |
| 36 | The procedure keeps the Pod stopped, not deleted, until a later decision | The Pod was stopped at about 20:05Z (EXITED). The researcher then decided its deletion, once the archive (1166 of 1166) and the local copy of the snapshot were verified; the engineering stage takes a fresh Pod with fresh checks | addendum 1: `stop-and-cost.md`, `runpod/` |

## The secret scan

- **The scan.** `grep -rlE 'private_key|BEGIN (RSA |EC )?PRIVATE|GCS_KEY|ARTIFACT_GCS|client_email'`
  ran over the frozen snapshot, `external/` included, before any upload. It matched 3 files:

  | File | sha256 |
  |---|---|
  | `env.sh` | `9fa9b69317de2ed06ece3ac8ae21911e13ef3b570b6ba261c2df8caab171ce96` |
  | `receipts/startup.txt` | `53967fb053f551d1020e1ea697c39f0138b597f1ad9d6d6776cbdafd8df94d09` |
  | `tools/restore_needed.py` | `45d1d01c5209d86d7a1b7793456b4125dd8409a70e2800690a6853d985cbea6e` |

- **The matches, line counts per file.** `GCS_KEY` 1 and `ARTIFACT_GCS` 1 in each; `private_key`,
  `BEGIN … PRIVATE` and `client_email` 0.
- **Strong signs.** None of the four in any file: a `BEGIN … PRIVATE` line, `"private_key": "`,
  `"client_email":`, or a run of 200 or more base64 characters.
- **The ruling.** The researcher ruled the three matches variable names, not values:
  - in `env.sh`, a comment;
  - in `tools/restore_needed.py`, the key file's path read from the environment;
  - in `receipts/startup.txt`, the name of a set variable.

  The upload then went ahead from the snapshot as frozen.
- **The receipts** (paths, classes, ruling) are in addendum 1, `secret-scan/`.

## GPU and Pod time

| GPU job (runner) | Minutes |
|---|---:|
| Runtime check (`gpu-runtime`) | 0.3 |
| Prep: streams of 90000 and 90200 (`prep`) | 7.4 |
| Engineering case 1 whole, then case 2's launcher attempt to the end of D2's stop (`eng-driver-1` to `term-l2`'s end) | 45.5 |
| Case 2's resumed baseline to the SIGKILL (`eng-driver-2` to `kill-k2`'s end) | 4.0 |
| F's launcher run, killed after its outcome (`fa-launcher`) | 6.3 |
| F's baseline, rest and control to the SIGKILL (`fa-driver-1` to `kill-fa`'s end) | 23.0 |
| P's launcher run to the SIGKILL (`p-launcher` to `kill-p`'s end) | 1.6 |
| F's control, attempt 2 (`fa-control-2`) | 3.0 |
| **Total** | **91.1 (1.52 h)** |

**Estimates and the Pod.**

- The procedure's estimate was 114–230 GPU-min. The three driver reruns that ran on CPU only
  (`eng-driver-3`, `fa-driver-2`, `fa-driver-3`) took 7.7 min more.
- The Pod ran from its creation at 15:17:53Z (container start 15:22:21Z) to its stop at about
  20:05Z: about 4.8 h. Step 34 ended at 19:50Z. The estimate was 5.8–11.3 h.
- No hours limit of section 6 was approached.

**Cost.** Both Pods were one NVIDIA L40S (Secure) at $1.09/h for the GPU. The billed amounts are
Runpod's pod billing for 2026-10-28 (no record on 2026-10-29), queried at 21:12Z and stable from
21:05Z (it posts about an hour behind).

| Pod | Billed: GPU + disk = total | GPU hours × $1.09 by uptime |
|---|---|---|
| `5qywtsjpchgr6z`, US-TX-4: Part I | $5.21 + $0.16 = $5.37 | 15:17:53Z to about 20:05Z, ≈ 4.79 h: ≈ $5.22 |
| `k89018y06rfzq1`, EU-NL-1: the container never started (the image pull failed) | $1.00 + $0.03 = $1.03 | 14:21:47Z to about 15:18Z, ≈ 0.94 h: ≈ $1.02 |
| **Total** | **$6.21 + $0.19 = $6.40** | **≈ $6.24** |

Part I's Pod is billed for the GPU from about 15:19Z to about 20:06Z, which agrees with its stop.
The disk is billed besides the GPU, at about $0.03/h per Pod. The API's answers are in addendum 1
(`stop-and-cost.md`, `runpod/`).

## Archive and provenance

- **Where it is.** `$ARTIFACT_REMOTE/dryrun/study3-pod-dryrun-20261028/` only; never the bucket's
  `runs/` or `data/`, and no `runs/archive/` inventory.
  - `evidence/`: the immutable snapshot. 1165 files are listed in `SHA256SUMS`, which is itself
    the 1166th object. It holds `DRY-RUN-NOT-A-RECORD.txt`, `env.sh`, G1 and G2 whole, the
    copies, `probe/`, `logs/`, `attempts/`, `receipts/` (with the two bundles, the seed patch and
    `bundles.sha256`), `inputs/`, `tools/`, `upload/inventory-streams.tsv`, `external/` (W&B
    core's four debug logs) and `LINKS`, which records 120 links: 90 inside, 30 `external`.
  - `data/`: the two seeds' dry streams (12 objects).
  - `inventory-20261028-study3-pod-dryrun-20261028.tsv`: the upload inventory, as its own object.
  - `addendum-1/`: what was recorded after the freeze, in a separate immutable folder, 20
    files. The snapshot is left unchanged.
    - `logs/`: the final `checks.tsv` and the `.out` files of the seven checks recorded after the
      freeze (`Z-freeze`, `Z-snap-before`, `Z-upload`, `Z-readback-get`, `Z-readback-check`,
      `Z-snap-after`, `Z-inventory-up`).
    - `upload/`: the upload inventory, the streams' inventory and `receipt.sha256`.
    - `secret-scan/`: the scan's paths, classes and ruling.
    - `stop-and-cost.md` and `runpod/`: the Pods' stop and cost, with the Runpod API's answers.
    - `README.md`, and `SHA256SUMS` over every other file.
  - `addendum-1-inventory.tsv`: the addendum's inventory (path, size, sha256), as its own object.
- **The checks in each.** The snapshot's `logs/checks.tsv` holds 172 checks: 170 PASS, 2 FAIL.
  The final log in addendum 1 holds 179: 177 PASS, 2 FAIL. The snapshot's log is a byte-for-byte
  prefix of the final one; the seven later checks all passed.
- **Verification.**
  - `archive.py`: 1166 files, 1166 verified (size and MD5), 0 mismatched.
  - The read-back of every object passed `sha256sum -c`. Its `SHA256SUMS` and its file list are
    identical to the snapshot's.
  - The snapshot was unchanged after the upload.
  - The inventory object's size and MD5 match the local file.
  - The local copy of the snapshot passed `sha256sum -c`.
  - Addendum 1 was uploaded without overwriting (`--no-clobber`), read back into a fresh folder,
    and every file matched its `SHA256SUMS` (20 of 20). Nothing else under the prefix
    changed.

| For the aggregate report | sha256 |
|---|---|
| The upload inventory (`inventory-20261028-study3-pod-dryrun-20261028.tsv`) | `e11d11b4f84a37a2091fa61c672a6dfa80894c37a45dcd7b1102fc8632495a11` |
| The snapshot's `SHA256SUMS` | `9c90fcfd2682b220bd24b1a680ac15195f472b2bf83a9557ed69b723b2805bff` |
| Addendum 1's `SHA256SUMS` | `d8fbecbf900dbcc7b0710997ed6d5239955b34ba1a8f5d01998a5db6127ddd3c` |
| Addendum 1's inventory (`addendum-1-inventory.tsv`) | `6fe6b156a400750c61d18ca919f9875d763c8545fa421f55ecc5b4072471abb9` |
| `tools.sha256` (as frozen, step 34) | `ea2665b5240090133f2f5ecc3917f55ce43eaa4f731c0d87a9e765961d2f009a` |
| `tools.sha256` (as run in steps 4–33) | `7de705b033369193b0bd6db47bbca996742d548c5332b90f385059d2162037fa` |
| `zfr-gate.bundle` | `626231f92849e757c0a50778983e9b460c875808ddbf122a7d7598ba50a2b46b` |
| `zfr-dry-seeds.bundle` | `ecd3fccbeebef4918939b5ca49b7f3cf6470ae05ecd66bbaccaaa39f9ce4f25f` |
| `dry-seeds.patch` | `b4bc493db6b4839d0edb02844f57d36e3aed0d4ecfca95ce09ca0a9c7f2f73d2` |

**What is left where.**

- **The Pod.** Stopped at about 20:05Z; its `desiredStatus` EXITED was confirmed through the API.
  It was deleted later, after the archive verified (1166 of 1166) and the local copy of the
  snapshot passed `sha256sum -c`. The Runpod API now answers "pod not found" for it (addendum 1).
  Its volume went with it:
  - the dry-run folder whole: G1, G2, the copies, the throwaway clones, the snapshot and the
    read-back;
  - both checkouts and the model cache.
- **The container disk** held case 2's launcher checkpoint (5 241 495 596 bytes) and an empty
  `harness/` folder (`receipts/container-disk-checkpoints.txt`). No dry case ended technical, so
  by the export rule nothing was exported, and they went with the container disk.
- **The earlier Pod** in EU-NL-1, whose container never started, was stopped and deleted. Nothing
  of Part I was on it.
- **Deleted.** Nothing was deleted by hand on the Pod. What remains of Part I is the bucket's
  dry-run prefix and the local copy (`evidence/`, `upload/`, `addendum-1/`).
