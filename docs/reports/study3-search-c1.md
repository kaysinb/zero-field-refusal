# Study 3: the search of candidate 1

The search ran on 2026-11-01 on Pod `piixrsw0w2dz5n`, one NVIDIA L40S, Runpod OC-AU-1, at commit
`3ce22b6ec2b550ea9aed15bfc9a4630e52a70400`, with the GPU code pinned at `fd54945`.

- **Runs.** 24 K-only runs of round 0, both rates of A, F and P on seeds 10100–10103. All 24 finished.
- **Route.** Every median lay within its tolerance, so round 1 did not run and the selected rates are round 0's:
  A 1.45·10⁻⁵ and 2.11·10⁻⁵, F 1.3·10⁻⁵ and 1.83·10⁻⁵, P 1.3·10⁻⁵ and 1.85·10⁻⁵. The next step on the records is the
  numerical qualification.
- **The values** of K are reported with the numerical qualification.
- **The replay** before the runs is recorded in [its own report](study3-engineering-replay-final-pin.md).
- **Incidents.** None.
- **Spend.** 3.70 Pod-hours, $4.15: GPU $4.03, disk $0.12.

## The earlier volume

The stage moved to this Pod from `iu69haoozd6ued`, whose host had no GPU. That Pod and its volume, which held the second
copy of case 17's kept checkpoints, were deleted on 2026-11-02 under
[amendment 9](../study3-preregistration.md#amendment-9-where-a-technical-cases-checkpoints-are-kept), after each of those
files was checked in the bucket against its inventory. The volume's idle disk from its stop to its deletion was billed
$0.37.
