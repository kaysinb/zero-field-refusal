# Study 3: the numerical qualification of candidate 1

The qualification ran on 2026-11-02 on Pod `ex376cen7cc76v`, one NVIDIA L40S, Runpod US-TX-4, at commit
`0df225648030e2d10affc826f5f86b3c4c94c6be`, with the GPU code pinned at `fd54945`. It covered both selected rates of A, F and P on
seeds 10200–10202, 18 cases, with F's two same-path controls on seed 10200.

**The measurements.** All 18 cases ran, with no technical marks and no reruns. Fourteen hold every binding bound. Four do not: two by a stop of the launcher
run under the instability rules, two by D₂₅ of the baseline against the reference, below. Their cause is being examined, and this report draws no
conclusion about it.

## The measurements

| Rate | Case | Measurement | Limit |
|---|---|---|---:|
| F 1.3·10⁻⁵ | s10200 | D₂₅ of the baseline against the reference: 0.744 | 0.06 |
| F 1.83·10⁻⁵ | s10201 | the launcher run stopped by `k_above_max` after 5 updates: K 837.96 | 16 |
| P 1.85·10⁻⁵ | s10200 | D₂₅ of the baseline against the reference: 1.945 | 0.06 |
| P 1.85·10⁻⁵ | s10201 | the launcher run stopped by `k_above_max` after 5 updates: K 27.42 | 16 |

- **The other 14 cases,** all of A's included, hold every binding bound.
- **The same-path controls** of F hold at both rates.
- **The repeat** stayed within its bound in every case that ran it: the largest D₂₅ is 1.52·10⁻⁹. Its rows equal the baseline's exactly in 15 of the 16 cases.
- **The two stopped runs** have no battery: their baseline part did not run.
- **The two reference measurements** above ended their battery there, so those cases have no shift measurements.
- **The shifts,** descriptive under [amendment 8](../study3-preregistration.md#amendment-8-the-four-imposed-shifts-become-descriptive), were measured in the other 14 cases: 8 of the 56 comparisons against the baseline lie beyond 0.01·K\*, in 5 cases.

The kept `verdict.json` has sha256 `e6bf315e7b38c6b6d9073716e38ef957396703244642f13a7a4d5df42ec31b42`.

## K at step 25

The search's values, reported here [as it said](study3-search-c1.md), and the qualification's, against the targets 4.80 (lower) and 7.50 (upper).

| Rate | Search: s10100–s10103 | Median | Qualification: s10200–s10202 |
|---|---|---:|---|
| A 1.45·10⁻⁵ | 4.83, 4.91, 4.81, 5.91 | 4.87 | 4.60, 4.97, 4.80 |
| A 2.11·10⁻⁵ | 8.15, 7.55, 7.64, 7.45 | 7.60 | 7.42, 7.58, 7.63 |
| F 1.3·10⁻⁵ | 4.83, 4.84, 4.73, 5.06 | 4.84 | 4.67, 5.06, 4.78 |
| F 1.83·10⁻⁵ | 7.44, 7.44, 7.24, 7.37 | 7.41 | 7.51, stopped, 7.76 |
| P 1.3·10⁻⁵ | 4.82, 4.86, 4.71, 5.06 | 4.84 | 4.62, 5.06, 4.76 |
| P 1.85·10⁻⁵ | 7.52, 7.64, 7.31, 7.44 | 7.48 | 9.98, stopped, 7.57 |

## Execution and spend

- **Runs.** 18 of 18 cases, one attempt each; the driver ended with exit 0. A full case took 26–31 minutes and a case with a stopped launcher run 5.5.
- **Records.** Verified on the laptop against the Pod's hashes and archived under `$ARTIFACT_REMOTE/study3/numerical/study3-numerical-c1-20261102/`. Earlier records unchanged: 1961 after, 482 new, none gone or changed.
- **Incidents.** None.
- **Spend.** 8.59 Pod-hours, $9.65: GPU $9.36, disk $0.28, billed to 10:00Z. The stopped volume is kept.

## The earlier volume

The stage moved to this Pod from `piixrsw0w2dz5n`. That Pod and its volume were deleted on 2026-11-02, after the search's report, with the search's
records verified in the bucket. Its idle disk from its stop to its deletion was billed $0.07.
