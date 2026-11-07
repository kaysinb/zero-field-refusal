# Study 3: the confirmation of candidate 1

The confirmation ran from 2026-11-05 to 2026-11-07 on Pod `f4w7ih75d5yxem`, one NVIDIA L40S, at commit
`c15d18c60bd70f4c09fa14995c1da0b4337de86e` (the second versions record, `docs/study3-versions-2.json`), with the GPU
code pinned at `fd54945` (identity `717cd02317526179a821eb8eb57e85cefba9e49083108922013b2b353c45ebbc`). Its fixation
is `docs/study3-fixation-confirmation-c1.json`: N = 60, cap 74. Each block had six runs on one seed: A at 1.45 and
2.11·10⁻⁵, F at 1.3 and 1.83·10⁻⁵, P at 1.3 and 1.85·10⁻⁵.

**The confirmation stopped at N.** 64 blocks started, on seeds 10400–10463. 61 support A–F and 60 support P–F, so
every active counter reached N = 60 before the cap. Both comparisons are read as confirmatory. No harm label was
opened.

## The counters

| Comparison | Blocks supported | Unsupported | N | Stop rule |
|---|---:|---:|---:|---|
| A–F | 61 of 64 | 3 | 60 | counter reached N |
| P–F | 60 of 64 | 4 | 60 | counter reached N |

- **The analysis cohorts** are each comparison's first 60 supported blocks by seed order:
  - A–F: seeds 10400–10462 without 10428, 10447 and 10460. Seed 10463 supports A–F but lies outside its cohort.
  - P–F: all 60 supported blocks.
- **The cohorts and K\*_h** are fixed by the analysis freeze record before any confirmation label is opened.

## Unsupported blocks

| Seed | A–F | P–F | Reason |
|---|---|---|---|
| 10414 | supported | unsupported | P at 1.85·10⁻⁵ stopped under `k_above_max` (monitored K 17.0 above 16) after 5 steps |
| 10428 | unsupported | unsupported | F's bracket too wide: K 5.40 to 10.14, ln width 0.63 above 0.60 |
| 10447 | unsupported | unsupported | F at 1.83 and P at 1.85·10⁻⁵ stopped under `gradient_spike` at step 3 (74.6 and 45.7, reference median 4.44) |
| 10460 | unsupported | unsupported | F at 1.83·10⁻⁵ stopped under `k_above_max` (monitored K 92.9 above 16) after 5 steps |

Each of these is a failed endpoint or a width failure under the registered rules. Nothing was rerun.

## Per rate

The tables in `study3-confirmation-c1-evidence/` come from the run records alone:

- `runs.tsv`, per arm, rate and block: each run's largest gradient norm and its step, its stop or completion, K at
  the end and its arm's support in the block;
- `blocks.tsv`, per block: A–F and P–F support, with the reason where support fails.

| Rate | Completed | K at step 25: median (range) | Largest gradient norm: median (largest) |
|---|---:|---|---|
| A 1.45·10⁻⁵ | 64 | 4.84 (4.32–5.57) | 5.7 (10.0) |
| A 2.11·10⁻⁵ | 64 | 7.75 (6.81–8.93) | 6.1 (16.0) |
| F 1.3·10⁻⁵ | 64 | 4.95 (4.25–5.93) | 5.9 (15.9) |
| F 1.83·10⁻⁵ | 62 | 7.49 (6.54–10.14) | 6.4 (74.6, a stopped run) |
| P 1.3·10⁻⁵ | 64 | 4.94 (4.24–5.70) | 5.9 (10.2) |
| P 1.85·10⁻⁵ | 62 | 7.57 (6.60–8.80) | 6.5 (45.7, a stopped run) |

- **Four stops,** all listed above. The stopped runs have no K at step 25.
- **Spikes in completed runs.** 19 completed runs exceed a gradient norm of 10. The largest is 25.8, for P at
  1.85·10⁻⁵ on seed 10420, at step 7. A spike creates no exclusion beyond the registered rules.

## Execution and spend

- **Runs.** 384 of 384 ran, one attempt each. The launcher ended every run, with no technical mark and no rerun.
  - A run took 6.7 minutes on average: A 6.0, F 6.9, P 7.1.
  - The training and preconditioner streams for 74 seeds, 10400–10473, were generated on the Pod before the runs
    and archived. Seeds 10464–10473 were not used.
- **Checks before the runs.**
  - The changed readers' tests passed in the Pod image: 808 tests in eight files.
  - `progress` named this confirmation with block 10400's six cells.
  - Before each block, the guard checked the block file against its sha256 and the stop rule.
  - Once both counters reached N, the guard refused block 10464.
- **Threads.** CPU threads were set to the Pod's CPU quota, 13. No trajectory, K, utility or response is computed by
  a CPU reduction that the thread count orders.
- **Records.**
  - Every ended run was archived during the stage, every 3 hours.
  - At the end, the records were verified on the laptop against the Pod's hashes: 10911 files.
  - They are archived under `$ARTIFACT_REMOTE/study3/confirmation/study3-confirmation-c1-20261105/`.
  - Earlier records were unchanged.
- **Spend.**
  - The Pod ran 50.1 hours, from its creation at 23:04Z on 11-04 to its stop at 01:13Z on 11-07.
  - Billed: $56.25 (GPU $54.65, disk $1.60).
  - The Pod and its volume were deleted after the copy was verified.
  - An earlier Pod, `6s0ljn2e0maqm0`, had a network file system as its volume. It was stopped before setup after
    about 10 minutes, for $0.17.
