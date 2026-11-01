# Study 3: the engineering stage

Candidate 1's engineering stage ran its 24 cases on seeds 10000–10003, on Runpod Pods with one NVIDIA L40S each. It
passes as a stage: no technical case stands, and the checks below hold. Four cases exceeded the shift bound in force when
they ran, and that informed [amendment 8](../study3-preregistration.md#amendment-8-the-four-imposed-shifts-become-descriptive).

**Four of 24 engineering cases exceeded the former shift bound; this informed the disclosed pre-search amendment, and all
measured shift results and unmeasured comparisons are reported.**

## What ran

- **The first pass,** 2026-10-30, 17 of the 24 cases: 15 qualified, 1 scientific and 1 technical, case 17. An earlier
  tally of "16 passed" was wrong, and the registration records its correction.
- **Case 17** (seed 10002, P at 1.3·10⁻⁵) failed the bridge on v₀'s hash alone. v₀'s estimate under the production
  attention backend does not repeat on the GPU; under the math backend it does: 0 of 110 recomputations differed, against
  9 of 110. The fix runs the whole estimate under the math backend. Its diagnosis is in the
  [registration](../study3-preregistration.md#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).
- **The re-run under the fix,** 2026-10-31 to 11-01: 16 replacements and 7 first executions, 23 cases. Seed 10000's A at
  2.11·10⁻⁵ keeps its first-pass records, a scientific failure preserved as old-policy evidence. So the summary below pools
  records of two GPU codes, and it is not "24 cases under the fix". A's s1 and s2 directions also use v₀, so that case is
  not unaffected by the defect.
- **Case 17 under the fix** passed: v₀ `70157c260d07` at the launcher, the baseline and the rest.

## The results

States: 20 qualified, 4 failing on a shift, none technical. Every value is D₂₅ of the harness baseline against the shift,
in KL; the launcher-checkpoint comparisons agree to four decimals. "Not run": under the rule then in force the part stopped
at the shift that exceeded. These comparisons stay unmeasured.

| Case | s1 | s2 | s3 | s4 | reference | Largest | Bound ÷ largest |
|---|---:|---:|---:|---:|---:|---:|---:|
| s10000/adam_frozen-1.3e-05 | 0.0083 | 0.0379 | 0.0000 | 0.0000 | 0.0000 | 0.0379 | 1.6 |
| s10000/adam_frozen-1.83e-05 | 0.0220 | 0.0307 | 0.0000 | 0.0000 | 0.0000 | 0.0307 | 2.0 |
| s10000/adamw-1.45e-05 | 0.0289 | 0.0264 | 0.0000 | 0.0000 | 0.0000 | 0.0289 | 2.1 |
| **s10000/adamw-2.11e-05** | 0.0400 | 0.0769 | not run | not run | 0.0000 | **0.0769** | 0.8 |
| s10000/adamw_prewarmed-1.3e-05 | 0.0082 | 0.0377 | 0.0000 | 0.0000 | 0.0000 | 0.0377 | 1.6 |
| s10000/adamw_prewarmed-1.85e-05 | 0.0213 | 0.0346 | 0.0000 | 0.0000 | 0.0000 | 0.0346 | 1.7 |
| s10001/adam_frozen-1.3e-05 | 0.0012 | 0.0012 | 0.0000 | 0.0000 | 0.0000 | 0.0012 | 51.6 |
| s10001/adam_frozen-1.83e-05 | 0.0025 | 0.0033 | 0.0000 | 0.0000 | 0.0000 | 0.0033 | 18.0 |
| s10001/adamw-1.45e-05 | 0.0009 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0010 | 58.3 |
| s10001/adamw-2.11e-05 | 0.0008 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0010 | 63.0 |
| s10001/adamw_prewarmed-1.3e-05 | 0.0011 | 0.0011 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 53.6 |
| s10001/adamw_prewarmed-1.85e-05 | 0.0022 | 0.0030 | 0.0000 | 0.0000 | 0.0000 | 0.0030 | 19.9 |
| s10002/adam_frozen-1.3e-05 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | over 500 |
| s10002/adam_frozen-1.83e-05 | 0.0046 | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0046 | 13.0 |
| s10002/adamw-1.45e-05 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | over 500 |
| s10002/adamw-2.11e-05 | 0.0011 | 0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 52.5 |
| s10002/adamw_prewarmed-1.3e-05 | 0.0001 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | over 500 |
| s10002/adamw_prewarmed-1.85e-05 | 0.0012 | 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0012 | 51.8 |
| **s10003/adam_frozen-1.3e-05** | 0.0549 | 0.0851 | not run | not run | 0.0000 | **0.0851** | 0.7 |
| s10003/adam_frozen-1.83e-05 | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 289.1 |
| **s10003/adamw-1.45e-05** | 0.0375 | 0.0339 | 0.0280 | 0.0997 | 0.0297 | **0.0997** | 0.6 |
| s10003/adamw-2.11e-05 | 0.0004 | 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 129.1 |
| **s10003/adamw_prewarmed-1.3e-05** | 0.0021 | 0.0703 | not run | not run | 0.0000 | **0.0703** | 0.9 |
| s10003/adamw_prewarmed-1.85e-05 | 0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 326.3 |

| Comparison family | Smallest bound ÷ D₂₅ |
|---|---:|
| launcher~baseline, launcher~repeat, baseline~repeat | above 10⁴ |
| launcher~reference, baseline~reference | 2.02 |
| launcher~shift, baseline~shift | 0.60 |

- e_eng, the largest D₂₅/K\* over the 24 unshifted repeats, is 2.1·10⁻¹⁰.
- Path maxima of D_t/K\*: repeat 2.0·10⁻¹⁰, reference 0.0040, shifts 0.056.
- 15 cases have a largest shift below 0.005, and 9 at or above 0.028. D does not follow K₂₅ (correlation of the logs −0.09).
- **The clustering.** Three of the four are the lower rate of each arm on seed 10003, whose higher rates read at most 0.0005;
  three failures in one of 8 seed-and-rate groups has P = 0.016 if cases were independent, and they are not: a seed's arms
  share its training stream. D arrives as a jump, for F and P at 1.3·10⁻⁵ between steps 5 and 10, as in study 2's P at
  1.83·10⁻⁵. One case differs in kind, seed 10003's A at 1.45·10⁻⁵, where the Gaussian shifts and the float64 reference move
  too (reference 0.0297). No mechanism is established.
- **The fix did not move D:** the 16 replaced cases keep their per-case largest shift to four decimals.

## The checks

| Check | Result | Evidence |
|---|---|---|
| **The dry run** | Part I and its repeat stand; the replays below passed | this report; [Part I](study3-dry-run-part-1.md) (sha256 `44e85992…`), [its repeat](study3-dry-run-part-1-repeat.md) (`bf4d8c3e…`) |
| **The bridge** | no case failed it among the current records; v₀ repeated across launcher, baseline and rest | `study3-engineering-evidence/verdict.json`; the first pass in `verdict-first-pass.json` |
| **θ₀'s utility** | equal to the reference: benchmark 0.568, factual 0.9955. Run in the first pass; `evaluate_theta0` estimates no v₀ and has no diff | `study3-engineering-evidence/theta0-utility.out` |
| **Stage 0** | attempt 2 passed in the first pass, protocol 2, after a documented step the first attempt missed. Its record's name and folders are unchanged through the re-run's inventories | `study3-engineering-evidence/stage0-passed.out` |

### The replays (amendment 6)

All on CPU, on synthetic records or the tiny model, unless a line says otherwise; none is a GPU demonstration.

- **At 8dcfe20, under the fix, in the Pod image:** through `project-run` from a clone at the commit, CUDA hidden, 188 listed
  entries (sha256 `b7b48aeb…`), the integration replay `tests/test_study3_replay.py` among them. 382 tests, 382 passed, none
  failed, errored or skipped, in 3906.7 s; junit sha256 `e96eb6cf…`.
- **On a GPU at 8dcfe20, on dry seeds:** Part I's affected steps G1, G2 and G3, in 117 minutes, with no finding. P on s90000
  holds one v₀ `757540e3464e` across launcher, baseline and rest; A's baseline and rest the same; F on s90200 `635f4adfce01`,
  its control included.
- **At the final pin, before this record:** the full suite in the main checkout at 2a2e7cc, 4097 passed, none failed or skipped, 50 minutes. With it,
  on a copy of the real records: `qualify.py verdict --engineering` returned the same state, kind, D and summary for all 56
  records, with one reason instead of two on the 4 scientific cases; `progress` named the engineering record as the step,
  with no problem; the chain of engineering cycles against the final pin showed no break.
- **Before the first search:** the five files that hold the changed paths, in the Pod image, from a clone at the commit.
  This report does not hold that result; it is written beside it.

## The GPU code, and why these records are reused

| Code | Commit | GPU identity | What ran under it |
|---|---|---|---|
| Old | first pass | `5c58451a647e…` | the first pass; the kept case |
| The fix | ced3117, at 8dcfe20 | `1421934cf582…` | the 23 cases of the re-run |
| The final pin | fd54945 | `717cd0231752…` | the search and every later run |

**The difference from the fix to the final pin** is one function in one GPU path, `qualification.py`, plus docstrings:
`scientific_result` returns None for a shift:

```diff
-    if run.get("not_run"):
+    if run.get("not_run") or name in SHIFTS:
```

Nothing else changes between ced3117 and fd54945 in a GPU path.

**Why the engineering records are reused under it.** The line decides only whether the rest part goes on after a shift's
result. It runs after each trajectory is done, so no measured trajectory, model, optimizer or data path changes. The code
of the 23 cases and of the final pin computes the same values for what was measured; only s3 and s4 of 3 cases stayed
unmeasured, and they stay so. Each trajectory starts from θ₀ with a fresh optimizer, tested with a non-finite shift. So
this record stands under the final pin without a re-run, and it starts the chain of engineering cycles there. The GPU
continuation past a shift's exceedance first runs in the numerical qualification.

## What this record does not say

- Passing here asserts no robustness to the imposed shifts.
- No study-3 search, qualification, validation or confirmation run exists, and no label of study 3 exists.
- The numerical gates were never part of the simulation's pass probability, and four engineering seeds cannot estimate them.

## Stops and spend

Incidents: one in the first pass, an unfit Pod at 0.105 Pod-hours; none in the re-run.

| Part | Pod-hours | Billed |
|---|---:|---:|
| The first pass, with the unfit Pod | 14.144 | $15.87 |
| The diagnosis of case 17 and its exports | 5.85 | $6.01 |
| The re-run | 19.791 | $22.23 |
| **Total** | **39.79** | **$44.12** |

The diagnosis row is the Pod's billed total less the other two. Each Pod's stopped volume accrues disk time until it is deleted; that is reported with the deletion. Archives:
`$ARTIFACT_REMOTE/study3/engineering/`: `study3-engineering-20261030` and
`study3-engineering-rerun-20261031`, with their inventories.
