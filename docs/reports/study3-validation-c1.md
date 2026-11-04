# Study 3: the validation of candidate 1

The validation ran on 2026-11-03 and 2026-11-04 on Pod `3a0r90tn3p6n5p`, one NVIDIA L40S, at commit
`79ce2833773668748df3b213de0f71e7cfe9d505` (the first versions record, `docs/study3-versions.json`), with the GPU code
pinned at `fd54945` (identity `717cd02317526179a821eb8eb57e85cefba9e49083108922013b2b353c45ebbc`). It covered 28 fresh
blocks on seeds 10300–10327, six runs each: A at 1.45 and 2.11·10⁻⁵, F at 1.3 and 1.83·10⁻⁵, P at 1.3 and 1.85·10⁻⁵.

**Both gates pass.** No harm label was opened.

## The gates

| Comparison | Blocks supported | Needed | Utility bounds | Gate |
|---|---:|---:|---|---|
| A–F | 27 of 28 | 22 | pass | **passes** |
| P–F | 26 of 28 | 22 | pass | **passes** |

- **Seed 10305.** F at 1.83·10⁻⁵ stopped under `gradient_spike` at step 18: a failed endpoint, so F is unsupported in
  this block, for both comparisons.
- **Seed 10317.** P at 1.3·10⁻⁵ ended at K 7.38, above K\* = 6.00, so P has no bracket in this block, and P–F is
  unsupported there.
- **The route.** Candidate 1 is chosen with A–F and P–F active. The next step is the `n-c1` record.

## Per rate

From the run records alone, per arm, rate and block: `study3-validation-c1-evidence/runs.tsv` (each run's largest
gradient norm and its step, its stop or completion, K at the end and its arm's support in the block) and
`blocks.tsv` (each block's A–F and P–F support, with the reason where it fails).

| Rate | Completed | K at step 25: median (range) | Largest gradient norm: median (largest) |
|---|---:|---|---|
| A 1.45·10⁻⁵ | 28 | 4.98 (4.23–5.78) | 5.6 (9.2) |
| A 2.11·10⁻⁵ | 28 | 7.78 (6.85–8.61) | 6.0 (8.6) |
| F 1.3·10⁻⁵ | 28 | 4.78 (4.18–5.69) | 5.6 (8.5) |
| F 1.83·10⁻⁵ | 27 | 7.49 (6.48–8.72) | 6.3 (50.0, the stopped run) |
| P 1.3·10⁻⁵ | 28 | 4.81 (4.16–7.38) | 5.6 (24.1) |
| P 1.85·10⁻⁵ | 28 | 7.56 (6.53–8.59) | 6.2 (9.0) |

- **One stop.** The stopped run, F 1.83·10⁻⁵ on seed 10305, has no K at step 25.
- **Spikes in completed runs.** Three exceed a gradient norm of 10:
  - P 1.3·10⁻⁵, seed 10317: 24.1 at step 21;
  - F 1.83·10⁻⁵, seed 10322: 18.5 at step 11;
  - F 1.83·10⁻⁵, seed 10323: 10.8 at step 19.
- **No exclusion from a spike.** None of these runs was stopped, and a spike creates no exclusion beyond the registered
  rules. The P run on seed 10317 is unsupported by its K, as above.
- **The simulation's reading is conditional.** It assumed the numerical qualification passed
  ([amendment 10](../study3-preregistration.md#amendment-10-numerical-eligibility-after-candidate-1s-qualification)).

## Execution and spend

- **Runs.** 168 of 168, one attempt each. The launcher ended every run, with no technical mark and no rerun.
  - A run took 6.2 minutes on average: A 5.5, F 6.4, P 6.6.
  - The 28 seeds' training and preconditioner streams were generated on the Pod before the runs and archived.
- **Checks before the runs.**
  - The changed readers' tests passed in the Pod image: 808 tests in eight files.
  - `progress` named this validation with its 168 cells, and `check` passed the campaign against the versions record.
- **Threads.** CPU threads were set to the Pod's CPU quota, 13. No trajectory, K, utility or response is computed by a
  CPU reduction that the thread count orders.
- **Records.** Verified on the laptop against the Pod's hashes and archived under
  `$ARTIFACT_REMOTE/study3/validation/study3-validation-c1-20261103/`. Earlier records were unchanged.
- **Spend.**
  - The Pod ran 19.5 hours, from its creation at 17:17Z to its stop at 12:49Z.
  - Billed: $21.53 (GPU $20.91, disk $0.61).
  - The Pod and its volume were deleted after the copy was verified.

## Appendix

Numerical eligibility followed [amendment 10](../study3-preregistration.md#amendment-10-numerical-eligibility-after-candidate-1s-qualification).
