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
- **What momentum changes.** Heavy-ball momentum β keeps SGD's noise isotropic. Its stationary
  covariance, like plain SGD's, is proportional to the identity, not to a preconditioner.
  - It raises the effective rate to η/(1 − β).
  - Its linear stability edge rises from 2/λ to 2(1 + β)/λ.
  - At β = 0.9 the equilibrium K that stability allows is (1 + β)/(1 − β) = 19 times higher.
- **The question it isolates.** AdamW-long uses β₁ = 0.9 as well. SGD with momentum 0.9 against
  AdamW-long therefore differs in Adam's per-coordinate adaptive preconditioner, not in momentum.

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
  - A rate passes as in Stage A: no instability fires, and the contamination stays at most
    0.01 × √35.
  - The two largest passing rates of each batch are its candidates.
- **Why this grid.** Heavy-ball momentum is linearly stable for ηλ < 2(1 + β) = 3.8. The plain
  SGD screen placed the limit between 0.0005 and 0.001 at B = 5, and between 0.002 and 0.003 at
  B = 20. The edges are therefore expected about 1.9 times higher: 0.00095–0.0019 at B = 5 and
  0.0038–0.0057 at B = 20. The grid brackets both.
- **Momentum in the control.** Momentum is linear in the gradient. Unlike Adam, it does not turn
  the control's tiny residual gradients into full-size steps.
- **Full controls.** As in Stage A:
  - one per candidate rate that runs on the three seeds, on seed 2000;
  - as long as the longest regular run of that rate and batch;
  - evaluated at seed 2000's evaluation steps and at its last step;
  - accepted when stable and with max c(t) ≤ 0.01 × √35.

## Search and rule

- **Search.** At each batch the candidates run in descending order on all three seeds, each with
  its full control. The search follows
  [amendment 1](study2-preregistration.md#amendment-1-the-monitor-panel)'s descent for SGD-long:
  - The descent stops at the first rate that meets the rule at U = 8.
  - If no batch has such a rate, U = 7 is settled from the rates run. Where no rate meets U = 7,
    the descent resumes below them.
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
  K_scoring ≥ 4, by amendment 1's thresholds: median ≤ 5 %, 95th percentile ≤ 15 %, at least 10
  states. It is reported. A failure must be resolved before any confirmatory use.

## Outcomes

| Pilot result | Action |
|---|---|
| The records are incomplete: a screen rate, regular run or candidate control missing, duplicated, outside the search or of another provenance | No conclusion; complete the search |
| No screen rate passes, or no candidate's control is accepted, at either batch | Numerics fail: momentum 0.9 has no clean rate; report it; the block-scaled SGD pilot of the [barrier branch](study2-preregistration.md#outcomes-of-stage-a) may be considered |
| At some batch a rate meets the rule at U ∈ {8, 7}, and Stage A's AdamW-long meets its own rule there at the same U | Feasible: a confirmatory comparison of SGD-momentum-long against AdamW-long at that batch and U becomes possible. It needs its own preregistration and the user's decision |
| An accepted rate reaches 5 on all seeds, but no batch reaches U = 7 | Partial: report it; a lower zone would need a separate decision |
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
- **What g is known to be.** From Stage A's plain SGD-long runs, g on the scoring panel rises from
  ≈ 1300 at ηt = 0.05 to ≈ 2300–2500 at ηt ≈ 2–2.8. Beyond that range it lies between ≈ 2400 and
  the bound tr(H)/4 ≈ 6200 (Hutchinson estimate from the curvature run, ρ = 1).
- **Predicted K at T_max = 4000.** For every rate below, η_eff·4000 ≥ 20:

| Batch | Rate | η_eff / B | Predicted K at 4000 steps |
|---|---|---|---|
| 5 | 0.0005 | 1.0·10⁻³ | 2.4–6.2 |
| 5 | 0.001 | 2.0·10⁻³ | 4.8–12 |
| 5 | 0.0015 | 3.0·10⁻³ | 7.2–19 |
| 20 | 0.002 | 1.0·10⁻³ | 2.4–6.2 |
| 20 | 0.004 | 2.0·10⁻³ | 4.8–12 |
| 20 | 0.005 | 2.5·10⁻³ | 6.0–16 |

- **What follows from it.** If the screen's candidates reach the predicted edges (at least 0.001
  at B = 5 and 0.004 at B = 20), the pilot is predicted to reach 5. Whether it reaches 7 or 8
  depends on g beyond the measured range. Near the stability edge, stiff directions raise K above
  this approximation.
- **What would contradict the model.** On any pilot run, a median ratio of its scoring-diagnostic
  K to (η_eff/B)·g(η_eff·t) outside 0.7–1.4, where η_eff·t lies within the measured range of g.

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
- **Analysis:** `analysis/pilot_momentum.py` applies the screen, the search, the rule, the
  acceptance and the outcomes above to the pilot's own records, with Stage A's loader and
  checks. The run's arm is `sgd_momentum_long`, and its controls' arm is `soft_control_momentum`,
  so that no pilot run can enter Stage A's selection.
- **Campaigns:** `scripts/pilot_momentum_campaigns.py`, with the kinds screen, search and controls.
