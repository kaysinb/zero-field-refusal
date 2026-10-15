# Study 2 pilot: SGD with heavy-ball momentum

Status: `proposed` (2026-10-15). Written during Stage A's downward search of study 2, before any
run of this pilot, with every harm label of study 2 closed. It runs only on the user's decision,
after the coordinator's and the second reviewer's review. It does not change study 2's
[preregistration](study2-preregistration.md) and does not rescue H1.

## Why this pilot

- **The barrier.** In Stage A, plain SGD-long settles far below the comparison zone: K ≈ 0.25 at
  0.002 (B = 20) and at 0.0005 (B = 5), against a lower level of 5
  ([K-only analyses](study2-stage-a-analysis.md)).
- **The law behind it.** In the zero field, plain SGD follows K(t) ≈ (η/B)·g(ηt), and g is set by
  the curvature spectrum. Stability caps η/B at about 1–1.5·10⁻⁴ at both batches. Even full
  equilibrium, g ≤ tr(H)/4 ≈ 6200, would then give at most K ≈ 0.9.
- **What momentum changes.** The gradient noise has the shape of the Fisher information, for every
  optimizer. Heavy-ball momentum β applies no preconditioner to it:
  - For small steps (ηλ ≪ 1), its stationary parameter covariance is approximately isotropic, as
    plain SGD's is. At a finite step a curvature-dependent factor 1/(1 − ηλ/(2(1 + β))) remains,
    that is 1/(1 − ηλ/3.8) at β = 0.9.
  - It raises the effective rate to η/(1 − β).
  - Its linear stability edge for a fixed Hessian rises from 2/λ to 2(1 + β)/λ.
  - At β = 0.9 the equilibrium K that this edge allows is therefore about (1 + β)/(1 − β) = 19
    times higher.
- **What the comparison narrows.** AdamW-long uses β₁ = 0.9 as well. Against it, SGD with momentum
  0.9 is a momentum-matched comparison: it removes momentum as an explanation of a difference and
  narrows the mechanism question toward Adam's per-coordinate adaptive preconditioning. It does
  not isolate that preconditioning exactly. Adam also bias-corrects its first moment, which this
  SGD arm does not, and its second moment changes the steps' scale as well as their shape.

## Question

Does SGD with heavy-ball momentum 0.9 reach the comparison zone under study 2's protocol? Here
that means K_benign 5 up to U ∈ {7, 8}, on all three discovery seeds, within T_max = 4000 steps,
numerically clean and with utility in bounds, at B = 5 or B = 20.

## Arm and protocol

- **SGD-momentum-long:** `torch.optim.SGD` with momentum 0.9, dampening 0, no Nesterov step and
  no weight decay. It runs at one constant rate and is stopped by K
  (`configs/optimizer/sgd_momentum.yaml`).
- **Everything else is Stage A's discovery:**
  - the protocol, model, frozen modules, loss and numerics;
  - the appendable training streams of seeds 2000–2002, shared with Stage A's runs;
  - the discovery schedule: levels 5, √35, √40, 7 and 8, target 8, plateau, instability with
    gradient factor 10, and T_max 4000;
  - monitor version 2 with scoring diagnostics and cached θ₀ logits
    ([amendment 1](study2-preregistration.md#amendment-1-the-monitor-panel)).
- **Namespaces:** experiment `study2_pilot_momentum` for the long runs and
  `study2_pilot_momentum_controls` for the soft-target controls.

## Numerical acceptance

- **Screen.**
  - A 100-step soft-target control on seed 2000 with the same momentum, at both batches, for each
    rate of {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6}·10⁻³.
  - A rate passes when no stopping condition fires, it reaches its final evaluation and the
    contamination stays at most 0.01 × √35 (the gradient rule only logs; below).
  - The two largest passing rates of each batch are its candidates.
- **Why this grid.** For a fixed Hessian, heavy-ball momentum is linearly stable for
  ηλ < 2(1 + β) = 3.8. The plain SGD screen placed the limit between 0.0005 and 0.001 at B = 5,
  and between 0.002 and 0.003 at B = 20. The edges are therefore expected about 1.9 times higher:
  0.00095–0.0019 at B = 5 and 0.0038–0.0057 at B = 20. The grid brackets both. The factor 1.9 is
  a fixed-Hessian heuristic, not a guarantee of stochastic stability; the screen and the full
  controls decide.
- **Momentum in the control.** Momentum is linear in the gradient. Unlike Adam, it does not turn
  the control's tiny residual gradients into full-size steps.
- **The gradient rule in a control only logs** (`grad_rule: log`). A soft-target control's
  gradient is zero at θ₀ up to rounding. Stage A's full controls showed that the ×10 rule, built
  for real gradients, fires on single batches on that noise floor without any growth: both
  stopped on stream responses 11480–11484 of seed 2000, with their scoring K at
  1.3–2.2·10⁻⁹. The pilot fixes this before any of its runs, as follows:
  - the rule's firings are recorded in the terminal record, and the update is taken;
  - a control is rejected when its scoring panel's K_benign exceeds 0.01 × √35 at any prescribed
    evaluation or at the final one; the values are finite and uncached, or bit-equal to the
    uncached reading by that evaluation's cache check;
  - finiteness of loss, gradient, weights and K, the soft-target replay checks, the cache checks
    and the path-field bound remain;
  - every prescribed evaluation and the final one are required. A control that stops before its
    planned last step for any reason is not accepted; there is no early success.

  This holds for the pilot's screen and full controls alike; its regular runs keep the ×10 rule.
- **Full controls.** As in Stage A:
  - one per candidate rate that runs on the three seeds, on seed 2000;
  - as long as the longest regular run of that rate and batch;
  - evaluated at seed 2000's evaluation steps and at its last step;
  - accepted when stable and with max c(t) ≤ 0.01 × √35.

## Search and rule

- **The comparison arm.** Stage A's AdamW-long: its version-2 discovery runs, read from Stage A's
  own records and checked and searched as Stage A's selection checks and searches them, once
  Stage A's whole search has resolved. At each batch and U it either has a rate meeting its own
  rule or not. The pilot's runs and these AdamW-long runs must read each seed's one stream.
  Missing, unresolved or inconsistent comparison evidence leaves the pilot without a conclusion,
  never with a scientific outcome.
- **Search.** At each batch the candidates run in descending order on all three seeds, each with
  its full control, as in
  [amendment 1](study2-preregistration.md#amendment-1-the-monitor-panel)'s descent:
  - The descent at a batch stops at the first rate that meets the pilot's rule at U = 8.
  - Once both batches have resolved, the comparison is feasible at U = 8 at a batch where such a
    rate meets AdamW-long meeting its rule at 8.
  - If no batch is, U = 7 is settled jointly. At every batch where AdamW-long meets its rule at 7
    and no rate run meets the pilot's rule at 7, the descent resumes below the rates run.
  - A rate resolves once its three seeds and its control are there; missing or technical failures
    never count as a rejection.
- **The rule.** A rate meets it at U when:
  - every seed reaches U;
  - utility is in bounds at 5, K\* = √(5U) and U: over the three seeds, the mean MMLU loss is at
    most 5 pp and the mean factual loss at most 10 pp;
  - every seed evaluates 5, K\* and U at three separate evaluations, as AdamW-long must, because a
    later comparison needs a point at each level;
  - its full control is accepted.
- **Monitor.** Version 2's agreement is computed on the pilot's scoring diagnostics with
  K_scoring ≥ 4, at each batch separately, by amendment 1's thresholds: median ≤ 5 %, 95th
  percentile ≤ 15 %, at least 10 states. It is reported. A failure must be resolved before any
  confirmatory use at that batch.

## Outcomes

| Pilot result | Action |
|---|---|
| The records are incomplete: a screen rate, regular run or candidate control missing, duplicated, outside the search or of another provenance; or Stage A's AdamW-long evidence missing, unresolved or on other streams | No conclusion; complete the search |
| No screen rate passes, or no candidate's control is accepted, at either batch | Numerics fail: momentum 0.9 has no clean rate; report it; the block-scaled SGD pilot of the [barrier branch](study2-preregistration.md#outcomes-of-stage-a) may be considered |
| At some batch a rate meets the rule at U ∈ {8, 7}, and Stage A's AdamW-long meets its own rule there at the same U | Feasible: a confirmatory comparison of SGD-momentum-long against AdamW-long at that batch and U becomes possible. It needs its own preregistration and the user's decision |
| No batch is feasible, but an accepted rate reaches 7 on all seeds at some batch: there its rule fails (utility or separate evaluations), or AdamW-long's does | Unmatched: report what fails where; a comparison would need a separate decision |
| An accepted rate reaches 5 on all seeds, but none reaches 7 on all seeds at either batch | Partial: report it; a lower zone would need a separate decision |
| At every accepted rate some seed stays below 5 | The barrier extends to heavy-ball momentum under this horizon; report it; the block-scaled SGD pilot may be considered |

## Harm

- The pilot reads no harm.
- Its runs' harm labels stay closed until a confirmatory design built on the pilot is sealed, or
  until the user decides against one.
- Stage A's discovery labels are opened, if the user so decides, only after this document is
  committed.

## Recorded prediction

Recorded with this document, before any run of the pilot.

- **The model.** For directions with ηλ ≪ 1, heavy-ball SGD with δ ← δ − η·m and
  m ← β·m + g reaches plain SGD's stationary covariance at the effective rate η_eff = η/(1 − β).
  So K(t) ≈ (η_eff/B)·g(η_eff·t), with plain SGD's g.
- **The frozen g.** `analysis.zero_field_law.g_table` estimates g from the SGD-long runs of Stage
  A's resolved search: every rate it tested, on every discovery seed, taken from Stage A's records
  as its selection checks and searches them:
  - each finite scoring diagnostic after step 0 gives the point (ηt, K_scoring / (η/B)); a
    non-finite one, which can only be a run's terminal diagnostic, is left out and listed;
  - the points fall into logarithmic bins of ηt, from 0.01 upward with ratio 1.25;
  - a run's value in a bin is the median of its points there, and g in the bin is the median over
    the runs, kept where at least two runs contribute;
  - between the bins' geometric centres g is interpolated linearly in log ηt, and it is undefined
    outside them.

  `scripts/pilot_momentum_prediction.py` writes the table to `docs/study2-pilot-momentum-g.json`
  once Stage A's search has resolved. With it go the source runs' IDs, the SHA-256 of the files
  the table reads from each (`config.yaml`, `diagnostics.jsonl`, `terminal.json`), and the hash
  of the table's content. The file is committed, and its table hash written here, before the
  pilot's first run; neither changes afterwards. A table that is empty or not finite is refused.
- **The test.**
  - `scripts/pilot_momentum_summary.py` first recomputes the cohort, its files' hashes and the
    table from Stage A's records and compares them, and the table hash, with the frozen ones. It
    refuses to summarize on any difference.
  - For each regular run of the pilot, `analysis.zero_field_law.check` takes the ratio of
    K_scoring to (η_eff/B)·g(η_eff·t) at every finite scoring diagnostic after step 0, up to the
    run's stop, whose η_eff·t lies within the table. A terminal non-finite diagnostic is left out
    and reported.
  - The run's value is the median of these finite ratios, given at least five of them; otherwise
    it is undetermined.
  - The model is contradicted when the median over the pilot's regular runs that have a value lies
    outside 0.7–1.4.
  - `scripts/pilot_momentum_summary.py` reports every run's value and that median.
- **What g is known to be now.** From Stage A's plain SGD-long runs so far, g on the scoring panel
  rises from ≈ 1200–1300 at ηt = 0.05 to ≈ 2300–2500 at ηt ≈ 2–2.8. Beyond that range it lies
  between ≈ 2400 and the bound tr(H)/4 ≈ 6200 (Hutchinson estimate from the curvature run, ρ = 1).
- **Indicative K at T_max = 4000.** These ranges illustrate the prediction; they are not the test.
  A run stopped earlier by its target or a plateau is censored for them, and its stop is reported.
  For every rate below, η_eff·4000 ≥ 20:

| Batch | Rate | η_eff / B | Indicative K at 4000 steps |
|---|---|---|---|
| 5 | 0.0005 | 1.0·10⁻³ | 2.4–6.2 |
| 5 | 0.001 | 2.0·10⁻³ | 4.8–12 |
| 5 | 0.0015 | 3.0·10⁻³ | 7.2–19 |
| 20 | 0.002 | 1.0·10⁻³ | 2.4–6.2 |
| 20 | 0.004 | 2.0·10⁻³ | 4.8–12 |
| 20 | 0.005 | 2.5·10⁻³ | 6.0–16 |

- **What follows from it.** If the screen's candidates reach the predicted edges (at least 0.001
  at B = 5 and 0.004 at B = 20), the pilot is expected to reach 5. Whether it reaches 7 or 8
  depends on g beyond the measured range. Near the stability edge, stiff directions raise K above
  this approximation by the factor above.

## Budget

- **Screen:** ≈ $2–3.
- **Long runs and full controls:** ≈ $8–35, depending on the candidates and how far the search
  descends.
- **Cap:** $40, including Pod setup, on three L40S Pods (one seed per Pod, as in Stage A).

## Implementation

- **Configs:**
  - `configs/experiments/study2_pilot_momentum.yaml` (Stage A's discovery config with the
    momentum optimizer);
  - `configs/experiments/study2_pilot_momentum_controls.yaml` (Stage A's controls with the
    momentum optimizer).
- **Analysis:** `analysis/pilot_momentum.py` applies the screen, the comparison evidence, the
  search, the rule, the acceptance and the outcomes above to the pilot's own records, with Stage
  A's loader and checks. The run's arm is `sgd_momentum_long`, and its controls' arm is
  `soft_control_momentum`, so that no pilot run can enter Stage A's selection.
  `analysis/zero_field_law.py` holds the law's g table and its check.
- **Scripts:**
  - `scripts/pilot_momentum_campaigns.py`, with the kinds screen, search and controls;
  - `scripts/pilot_momentum_prediction.py`, which freezes the g table;
  - `scripts/pilot_momentum_summary.py`, which writes the outcome and the prediction's check.
- **Namespaces:** the pilot's runs never share a directory with Stage A's
  ([operations](operations.md#study-2-stage-a)).
