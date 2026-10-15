# Study 2 preregistration

Status: `approved` (2026-10-14). The user approved the study-2 design; this document and the
study-2 code passed the review of the coordinator and the second reviewer (rounds r6 to r15)
before any study-2 run. Items marked *to be fixed at Stage B* are frozen by the procedure of
[Stage B](#stage-b--fixation), before any confirmatory run; every other item is fixed, and a
change goes through [Amendments](#amendments).

Study 2 follows the stopped pilot (study 1: [preregistration](preregistration.md),
[decision](decisions.md#study-decisions), [short report](reports/gate-1.md)). It is a new
preregistered study on fresh seeds, with its own run, config and stream names in the namespace
`study2_*`. The pilot's data serve only to plan it. The aim is a decided answer: a difference in
either direction, or equivalence. Confirming H1 is not the aim.

## What the pilot taught

The pilot did not yield a decisive, interpretable optimizer comparison. Study 2 answers its
problems:

| Pilot problem | Study-2 answer |
|---|---|
| One shared budget of 25 steps × batch 20 left SGD a single lever, its learning rate, which hit a stability cliff at K_harmful 2.41, below AdamW's effect zone | Each long arm has its own constant rate and its own number of steps, stopped by K; the arms share the batch, the stream prefix and T_max |
| Gate 1 sat in a collapsed cell with 5 seeds. Under stated distributional assumptions, a simulation gives a real effect of the observed size a pass probability of ≈ 0.54 at the observed seed spread | No harm-based gate; feasibility gates use numerics, reachability and utility; the comparison zone is fixed from pilot data |
| The soft-target control at 0.028, the largest rate called stable, grew ≈ 9–14× per step: that rate was linearly unstable. The threshold was not measured, so whether every nominally stable rate exceeded it is unknown | Curvature measured directly; SGD rates accepted only if a soft-target control shows negligible functional contamination at full length |
| K\* = 2.26 put the comparison where observed harm was low in every arm and discrimination weak | The zone is fixed now in K_benign units, where the pilot's Adam arms showed rises of 16–21 pp with MMLU losses under 5 pp |
| Matching on K_harmful is outcome-adjacent, since θ₀'s responses there are refusals; that it attenuated the contrast is a hypothesis | K_benign is the matching variable; K_harmful analyses are secondary |
| The judge check sampled uniformly at low harmful prevalence | The audit is stratified by the judge's own label, with inclusion probabilities |
| The paired bootstrap was specified, but the number of seeds and the gate were planned with unpaired variance, although the arms share streams | Paired planning and a paired seed-block analysis |

## Relation to the approved design

This document implements the study-2 design approved by the user on 2026-10-14. Two rules
differ from it, both settled before any study-2 data:

- **U also needs reachability.** The design chose U "by utility only", meaning "not by harm".
  U here needs both utility and reachability on all discovery seeds of both long arms. Neither
  condition looks at the outcome. This is the coordinator's clarification within the approved
  design.
- **Batch priority** (a prospective change before any data). Among eligible batches, the larger U
  is preferred first and cost second, where the design ranked by cost alone. At a larger U the
  comparison lies where the effect is larger.

## Question and estimand

- **H1 (study 2).** Under a common batch size, a common appendable teacher-stream prefix and a
  common step limit, at equal functional displacement on held-out benign instructions, does
  long AdamW yield more judge-rated harmful compliance than long SGD?
- **Estimand.** Δ(K\*) = h_AdamW-long(K\*) − h_SGD-long(K\*): the difference of judge-rated
  harmful-compliance rates at the matching level K\*. It is conditional on the training and
  stopping policies below. Each long arm runs at its own constant rate for its own number of
  steps. Reachability and non-arrival are reported with it.
- **Primary outcome.** WildGuard's harmful-compliance label (judge-rated harm). Harm according to
  the reference labels is a separate preregistered analysis ([audit](#judge-and-audit)).
- **Matching variable.** K_benign: the sequence-averaged KL(p_θ₀ ‖ p_θ) on 100 held-out benign
  prompts × 4 θ₀ responses (the scoring panel), defined as in the pilot. It measures functional
  change on held-out benign instructions, not all general capability.
- **Secondary contrasts:**
  - AdamW-25 against each long arm;
  - AdamW-25 against AdamW-long, which asks whether Adam's curve depends on its regime;
  - frozen Adam-25 against AdamW-25, a mechanism diagnostic ([below](#frozen-adam-25));
  - every analysis repeated with K_harmful as the matching variable.
- **Scope of inference.** Uncertainty is over training seeds, conditional on the fixed prompt
  sets. Generalization beyond them needs other prompts or a replication.

## Protocol

- **Model:** Qwen2.5-1.5B-Instruct at the revision pinned in `configs/model/`.
- **Numerics:** fp32 weights, compute and gradients; TF32 off.
- **Frozen parameters:** input embeddings and output head.
- **Update rules:** no weight decay and no gradient clipping. The loss is the summed token NLL
  divided by the constant B × 96.
- **Zero field.** Every training response is a fresh sample from θ₀, as in the pilot's
  [fixed protocol](preregistration.md#fixed-protocol). The step-0 teacher-gap and path-field
  checks are unchanged.
- **Streams.** One teacher stream per seed, shared by every arm of that seed. The stream is
  appendable and prefix-stable: lengthening it never changes the batches already drawn.
- **Protocol version.** `protocol_version` becomes 2 when the study-2 code lands
  ([protocol versions](decisions.md#protocol-versions)).

## Arms

| Arm | Optimizer and regime | Role |
|---|---|---|
| SGD-long | SGD without momentum, one constant rate, stopped by K | primary |
| AdamW-long | AdamW (β 0.9 / 0.999, ε 10⁻⁸) at one small constant rate, stopped by K | primary |
| AdamW-25 | AdamW at three rates, exactly 25 steps each | secondary |
| Frozen Adam-25 | Adam with second moment frozen at θ₀ (as in the pilot), up to three rates, 25 steps each | mechanism diagnostic; [rules below](#frozen-adam-25) |

- **"Long" means stopped by K.** It names the arms stopped by K at a small rate. It does not
  claim that AdamW reaches a settled long-run regime. Each run records its number of steps to
  each level.
- **Batch.** Every arm uses one common batch size B ∈ {5, 20}, chosen in Stage A by the
  [selection rule](#selection-rule). If B = 5 is chosen, AdamW-25 is recalibrated at B = 5, and
  the pilot's AdamW at B = 20 remains historical evidence only.
- **Optional check.** An AdamW-25 check at B = 20 on the same seeds may be added, labeled as a
  different-batch comparison.
- **Why batch 5 is admissible.** The batch is a protocol condition common to all arms. In the
  zero field, SGD's displacement grows roughly as steps × rate² / batch. So only at a small
  batch can SGD reach the zone at a rate below its stability threshold; at B = 20 it either
  oscillates or needs thousands of steps.
- **Limit of that choice.** The batch also changes Adam's dynamics (the bias of its kicks, its
  stability threshold), so conclusions hold for the chosen batch.
- **Fixed before outcomes.** The choice follows the rule below, before any harm is seen.

## Zone and levels

- **Fixed now from pilot data.** The comparison zone is fixed in K_benign units. A level that an
  arm cannot reach stays where it is; non-arrival is a result.
- **Candidate levels:** 5, √35 ≈ 5.92, √40 ≈ 6.32, 7 and 8. Discovery evaluates all of them, so
  that U is chosen from points already recorded.
- **Levels after selection:** 5, K\* = √(5U) and U.
- **The upper level U** is the larger of {7, 8} for which, at the selected batch, both conditions
  hold for both long arms on the discovery seeds:
  - **utility:** the mean MMLU loss at the U evaluations is at most 5 pp;
  - **reachability:** every discovery seed reaches U.

  Neither condition looks at the outcome. If U is 7 because SGD or AdamW does not reach 8, that is
  a reported limitation of feasibility.
- **The reason for this rule.** In the pilot, no run lies between K_benign 9 and 19, so where
  MMLU begins to collapse in these units is unknown. At K_benign ≈ 8, frozen Adam already lost
  4.9 pp on average.
- **What the pilot shows near these levels:**
  - AdamW 2·10⁻⁵ at K_benign ≈ 7.2 raised harmful compliance by 15.9 pp, with an MMLU loss of
    3.6 pp;
  - frozen Adam 2·10⁻⁵ at K_benign ≈ 8.2 raised it by 20.8 pp, with 4.9 pp.

## Panels, monitoring and stopping

### Panels

| Panel | Contents | Use |
|---|---|---|
| Scoring | K_benign on 100 benign prompts × 4 θ₀ responses; K_harmful on 100 harmful prompts × 4; MMLU (500); factual questions (222) | every full evaluation |
| Harmful evaluation | 521 prompts: the deduplicated HarmBench and AdvBench pool outside the 100 K_harmful prompts, cleaned of near-paraphrases (below); greedy, ≤ 256 tokens | harmful compliance at each full evaluation; fixed before Stage A |
| Monitor, version 2 ([amendment 1](#amendment-1-the-monitor-panel)) | the scoring panel's 100 benign prompts × 2 fresh θ₀ responses with seeds of their own; the first response of each prompt (m = 1) is the operational panel, the second only completes the shadow panel m = 2 | m = 1: K every 5 steps; m = 2: at diagnostic states and full evaluations |
| Monitor, version 1 (the discovery runs before amendment 1) | 25 further benign prompts × 2 θ₀ responses, disjoint from the scoring panel, with their own data and RNG | K every 5 steps |
| Scoring diagnostics (monitor version 2) | the scoring panel's K_benign alone: no harm, no utility | every 25 steps |

**The harmful evaluation set** (`harmful_evaluation` of `data/study2`, sha256
`a2365f072eeab68de338babcc8e5c52c159f440efc566352dde6c46395c159ab`, built 2026-10-14):

- **Pool:** the 596 deduplicated prompts outside the 100 K_harmful prompts, study 1's judged
  (300) and judge-calibration (296) prompts. The judge is not tuned in study 2, and the audit
  labels study 2's own responses, so the judge's error is measured on the set that measures harm.
- **Cleaning,** deterministic, in two passes over the pool in source order (HarmBench, then
  AdvBench, by row). First, a prompt is dropped if its word-set Jaccard with a prompt kept before
  it is at least 0.6 (59 near-paraphrases). Then, of the prompts kept, those whose Jaccard with any
  K_harmful prompt is at least 0.6 are dropped (16). A prompt dropped in the second pass still
  removed its later paraphrases in the first; dropping prompts near K first would keep 522.
- **Why 0.6.** In a blind labeling of prompt pairs by Jaccard band (2026-10-14,
  `scripts/harmful_pool_cleaning.py`), the share of pairs asking for the same thing was 48 %,
  58 %, 75 % and 81 % within the pool and 38 %, 55 %, 86 % and 100 % against the K prompts in the
  bands 0.4–0.5, 0.5–0.6, 0.6–0.7 and 0.7–0.8. From 0.6 on it is at least 75 %; below, a lower
  threshold would drop nearly as many different requests as repeats.
- **Composition:** HarmBench 157, AdvBench 364; 260 from study 1's judged and 261 from its
  calibration prompts, marked in each item (`study1_set`) for a descriptive check of the judge's
  error by origin. No prompt is shared with the K_harmful prompts or with the benign held-out,
  scoring or monitor prompts.

### Outcome embargo

- **During training,** only K and utility leave a run.
- **Discovery:** harm labels stay closed until the Stage-A selection is sealed.
- **Validation:** harm labels are opened only to plan variance, after the selection is sealed.
- **Confirmation:** outcomes stay closed until their scheduled look.

### Level crossings

- **A crossing triggers a full evaluation.** When two consecutive monitor readings lie at or
  above a level, the run is fully evaluated in place. The curve uses the scoring panel's measured
  K, never the nominal level. One evaluation that crosses several levels counts as a single
  observation.
- **Monitor agreement.** In Stage A, the monitor must agree with the scoring panel for
  K_benign ≥ 4: median relative discrepancy ≤ 5 %, 95th percentile ≤ 15 %. Otherwise the
  monitor is enlarged before fixation. Version 1 failed this check, and
  [amendment 1](#amendment-1-the-monitor-panel) replaced it; version 2 qualifies by the rule
  stated there, on the scoring diagnostics of the discovery runs that use it.

### Stops

A long arm stops on K and stability only:

1. **Target:** after the evaluation at the top level. That is U in validation and confirmation,
   and 8 in discovery.
2. **Plateau:**
   - never before step 1000;
   - taken over six consecutive windows of 100 steps;
   - stop when the window medians of the monitor K span at most a factor 1.10, the median of the
     last three is at most 1.02 times that of the first three, and all six lie below 90 % of the
     next level.
3. **Instability**, checked before every update from the first gradient:
   - a non-finite loss, gradient, weight or K;
   - or a monitor K above 2U (16 in discovery);
   - or a gradient norm above 10 times the median of the preceding (up to 20) step norms, the
     reference starting at the step-0 norm; one step suffices.

   On the pilot's logs this fires at SGD 0.03 at step 6 (≈ 11×), while the stable runs at 0.028
   peak at ≈ 2×. Discovery reports the rule's margin on stable trajectories and its possible
   false alarms. The factor 10 may be revised only from these, by the rule under
   [Details fixed with the code](#selection-rule), never from harm, and it is frozen before
   validation; a later change would require validation to be repeated.
4. **Limit T_max:**
   - common to both long arms;
   - T_max = min(4000, 1.5 × the larger of the two arms' median step counts to U on the discovery
     seeds);
   - discovery runs use 4000.

- **Earlier observations survive.** A later stop never removes an earlier evaluation. Every run
  keeps a terminal record of its levels reached, levels missed and stop reason.
- **AdamW-25 and frozen Adam-25** run exactly 25 steps and stop early only for instability.

## Numerical acceptance

- **Curvature.** The top curvature of the correctly normalized loss at θ₀ is measured by
  Hessian–vector products, with several starting vectors and batches. It guides the SGD rate
  screen; it does not certify stability. The pilot's soft-target growth suggested λ_max ≈ 350–520,
  i.e. 2/λ_max ≈ 0.004–0.006.
- **SGD rate screen.** The grid is {0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.01, 0.015,
  0.02, 0.028}, at both batches. Every rate runs a 100-step soft-target control on seed 2000.
  A rate passes the screen when no instability fires and the contamination below stays within
  its threshold.
- **Candidate rates.** At each batch, the two largest rates that pass the screen are the
  candidates. Each gets a full-length soft-target control on seed 2000 and runs in discovery.
- **Contamination.**
  - **Definition:** c(t) = K_benign(θ₀ ‖ θ_soft(t)), the scoring panel's K_benign of the
    soft-target control after t steps, evaluated at the regular run's evaluation times.
  - **Length:** the control runs as long as the longest regular discovery run of that rate and
    batch, on the same seed.
  - **Acceptance:** max over t of c(t) ≤ 0.01 × K\*. Until U is chosen, K\* is that of the lowest
    candidate zone, √35 ≈ 5.92, so the threshold is ≈ 0.059 nat.
- **Path bound.** The path drift/noise bound must also hold for the maximum planned exposure.
  The KL threshold does not replace it.
- **A failed rate cannot become primary,** however it is labeled.
- **Adam.** A soft-target control is not meaningful for Adam, which normalizes the residual
  gradient. Adam rates need the path drift/noise bound for their maximum exposure.
- **Descriptive comparison.** At the candidate SGD rates, Stage A compares the gradient growth of
  the soft-target control and of the regular run. This explains why the pilot's regular run at
  0.028 grew only 1.2–1.4× per step.

## Stage A — feasibility

Stage A uses fresh seeds that never enter the confirmation.

### Discovery

Three seeds, run at both batches in parallel. Every run is evaluated at all
[candidate levels](#zone-and-levels).

- **SGD-long:** the two candidate rates of each batch.
- **AdamW-long:** at {1, 3, 5, 10}·10⁻⁶.
- **AdamW-25:**
  - grid {0.5, 1, 1.5, 2, 3, 4}·10⁻⁵;
  - if at B = 5 the smallest rate's median K_benign already exceeds 5, the grid extends downward
    by halving, at most twice.
- **Frozen Adam-25:** the same grid on the same seeds.
- **Monitor version 2** ([amendment 1](#amendment-1-the-monitor-panel)): the long arms are
  searched again under it, rate by rate downward on all three seeds, SGD-long for the first
  time. The selection uses the version-2 records of the long arms; the version-1 long runs stay
  observations with their measured K. The 25-step grids and the screen are unaffected.

### Selection rule

The rule uses numerics, K, reachability, utility and measured cost. It never uses harm.

**Utility in bounds** means: over the three discovery seeds, the mean MMLU loss is at most 5 pp
and the mean factual loss at most 10 pp, at the evaluations used.

1. **Batch.**
   - A batch is eligible when an SGD-long candidate rate passes the numerical acceptance and some
     U ∈ {7, 8} qualifies for both long arms.
   - Among eligible batches, the larger U wins. On a tie, the lower measured GPU time per seed
     block wins.
2. **SGD-long rate:** the larger accepted candidate rate that reaches U on all discovery seeds
   within 4000 steps, with utility in bounds.
3. **AdamW-long rate:** the largest of {1, 3, 5, 10}·10⁻⁶ that meets five conditions:
   - it reaches U on all discovery seeds;
   - it passes the path bound;
   - it keeps utility in bounds;
   - it evaluates 5, K\* and U at three separate evaluations — a coverage condition, not
     evidence of a long-run regime;
   - its step count to U is recorded.
4. **AdamW-25 rates:**
   - three distinct rates whose median K_benign across the discovery seeds lies nearest to 5, K\*
     and U, with utility in bounds;
   - a tie goes to the smaller rate;
   - if fewer than three distinct rates qualify, the qualifying ones are taken and the shortfall
     is flagged.
5. **Frozen Adam-25 rates:**
   - chosen by the same rule on its own grid;
   - if they coincide with AdamW-25's, the mechanism contrast has equal rate, steps and stream;
   - otherwise they keep their own rates: curves are compared at equal K, and covering the levels
     matters more than equal rates;
   - a failure of frozen Adam's coverage never blocks H1.
6. **T_max:** by the rule [above](#stops).

**Details fixed with the code** (`analysis/stage_a.py`, `scripts/stage_a_summary.py`), before any
data:

- **Utility loss** is θ₀'s accuracy minus the run's at an evaluation, in percentage points. The
  evaluations used are those at 5, K\* and U for a long arm, and the one at step 25 for a 25-step
  arm.
- **U qualifies at a batch** when each long arm has a rate that meets its own rule (2 or 3) at
  that U, reachability and utility included.
- **Time per seed block** is the median over seeds of the summed run times of the selected
  SGD-long, AdamW-long and AdamW-25 runs, without stream extension, plus generating the block's
  stream at the measured time per chunk. Frozen Adam-25 is left out, since it runs on 20 blocks
  only.
- **AdamW-25 rates** are the assignment of distinct qualifying rates to the targets 5, K\* and U
  with the least total distance |median K_benign − target|; ties go to the smaller rates.
- **The search must be whole.** The rule runs only on the prescribed search: one 100-step
  screen control per SGD rate and batch on seed 2000; discovery runs of exactly the two screen
  candidates per batch, the AdamW-long grid and the 25-step grids with the extensions they call
  for, one run per arm, batch, rate and seed 2000–2002; one full-length control per candidate
  on seed 2000, of the same model and stream as the regular run of that seed. A run counts on
  whole records agreeing with its published terminal record, whatever its status; every
  evaluation must have answered every harmful evaluation prompt once.
- **The gradient rule's factor** is judged mainly by its margin: the largest ratio of a
  gradient norm to its reference on runs that ended stably. A gradient-rule stop in an arm,
  batch and rate where another seed ran stably is a *possible* false alarm: breakdowns near the
  edge vary by seed, and a stopped run cannot show what it would have done. The factor is raised
  only if the margin exceeds 7 or a possible false alarm occurred, to max(10, 1.5 × the margin),
  and frozen before validation. It is never lowered.

### Validation

Twelve further fresh seeds run at the selected batch and rates, with the instability factor
frozen. Validation passes when:

- at least 10 seed blocks support both long arms around K\*;
- the numerical and monitor checks hold (the selected monitor's agreement is checked again on
  the validation runs' scoring diagnostics, by the thresholds of amendment 1);
- simultaneous one-sided 95 % upper bounds on the mean utility loss, at the three levels and for
  both long arms, stay at most 5 pp on MMLU and at most 10 pp on the factual questions.

Harm does not enter the gate.

### Outcomes of Stage A

| Stage-A result | Action |
|---|---|
| The search or its records are incomplete: a screen rate, discovery run (arm, batch, rate, seed) or candidate control missing, duplicated, outside the prescribed grids, or of another provenance | No conclusion; complete the search |
| Numerical acceptance fails at every SGD rate of both batches | Stop the behavioral comparison; report the numerics |
| Numerically clean SGD reaches U = 7 at neither batch | The *barrier* branch: the primary result is the characterized barrier (K(T) curves, curvature, plateaus) |
| No batch is eligible for another reason (AdamW-long does not reach U, or utility fails) | Stop this confirmatory design; report the feasibility result |
| A batch is eligible but validation fails reachability or utility | Stop this confirmatory design; report the feasibility result |
| The selected batch is 20 and the projected study cost exceeds $320 | A separate decision of the user before Stage B |
| Validation passes | Proceed to Stage B |

**The barrier branch.**

- It may add a separately preregistered pilot of SGD with fixed per-block scales estimated at θ₀
  on a separate stream, with no momentum and no adaptive updates (≈ $10–20).
- It answers a different question and does not rescue H1.
- A failure of plain SGD establishes a barrier under the tested policies and horizon, not
  impossibility for SGD in general.

## Stage B — fixation

Stage B freezes, in a committed record:

- the batch, rates, T_max and U;
- the panels and levels;
- the stopping thresholds;
- the maximum number of seed blocks;
- the analysis choices listed [below](#to-be-fixed-at-stage-b);
- the audit plan;
- the versions of code, model and judge;
- the namespaces.

**The simulation.** It runs the whole procedure. Its inputs:

- a stream effect shared by the arms;
- paired SDs of 3, 5 (central), 10 and 15 pp;
- unsupported blocks, plateaus, instability and judge error;
- the uncertainty of the validation estimate of the SD.

Its requirements:

- power ≥ 0.8 to conclude equivalence at a true Δ = 0, and to conclude the direction at
  |Δ| = 5 pp;
- a false directional result ≤ 2.5 % in each direction, ≤ 5 % in either;
- false equivalence ≤ 5 % at |Δ| = 5 pp.

The maximum, a cap of ≈ 80, counts started blocks. If no maximum within the cap reaches 0.8,
the study runs as an explicitly underpowered, capped study. What is then reported under
uncertainty (the interval and its ±8 reading) is fixed in the Stage-B record.

**Model choice.** The curve model and its band are chosen by a criterion written now: among the
candidates declared in the Stage-B record, the one with the highest simulated power that meets
the error requirements. They are never chosen by how the validation data look.

**Blind analysis.** Arm labels stay hidden until the frozen pipeline has run.

## Stage C — confirmation

### Looks

Two looks, with O'Brien–Fleming-type error spending. The exact numbers are *to be fixed at
Stage B* from the frozen maximum:

| Look | Blocks started (supported) | Nominal two-sided level |
|---|---|---|
| 1 | ≈ 30 (≥ 24) | ≈ 99.53 % |
| 2 | ≈ 63 (≥ 50) | ≈ 95.19 % |

- There is no third look.
- Information is counted in supported blocks, not started ones.
- **Missing minimum.** If a look lacks its minimum of supported blocks, it is postponed until the
  minimum is reached, within the cap on started blocks.
- After look 1 the pods stop until its CPU analysis is done.
- **Utility at both looks.** The utility bounds cover both looks.

### Analysis population

**Pairing.** The primary contrast uses the blocks supported in both long arms: the intersection,
which keeps the pairing. Blocks supported in one arm only are reported.

**Nothing is dropped silently:**

- **Failed fit.** A failed curve fit is not dropped. The Stage-B record fixes three things: the
  regularization of the fit, how failed bootstrap replicates count, and the fallback estimator. The
  candidate fallback is the mean of within-seed interpolated differences. The number of failures
  is reported.
- **Missing utility at a level.** A block that has no utility evaluation at a level enters that
  level's utility bound with the largest loss observed in its arm at that level. The bound is
  also reported over the blocks actually evaluated, with the number missing. A utility claim
  needs the first bound. The block stays in the harm analysis wherever it has K support.
- **Unresolved judge labels.** A WildGuard output that does not parse is rerun until it parses,
  as in the pilot's [judge rule](operations.md#judge-server). A label still missing after the
  reruns counts as not harmful in the primary rate. Their number is reported per arm, with the
  worst-case interval that counts them as harmful.

### Primary analysis

**Curves.** Per arm, harmful compliance against x = ln(K_benign/K\*). The planning model is a
binomial logistic curve with a kink at K\*:
logit h = α + βx + γ·max(x, 0).

- The fit uses the scoring observations in the band [0.8·5, 1.1U].
- Each seed–arm carries equal total weight.
- The model and band are *to be fixed at Stage B* by the criterion [above](#stage-b--fixation).

**Interval.** A paired seed-block bootstrap-t: a resample keeps every arm, rate and evaluation of
a drawn seed. It uses at least 100 000 resamples, or an approximation validated by simulation,
*to be fixed at Stage B*. The sequential levels come from the spending rule and are validated by
simulating the whole analysis.

**Decision.** One sequentially valid interval for Δ(K\*) gives every reading:

- **Equivalence:** the interval lies inside ±5 pp. This is the primary margin.
- **Direction:** the interval excludes 0 (a positive or opposite-sign difference). Direction and
  equivalence can hold together for a small nonzero difference.
- **±8 pp reading:** whether the interval lies inside ±8 pp, read from the same interval. It is
  nested and secondary.
- **Inconclusive:** neither equivalence at ±5 nor direction.

**Stopping and the ±8 reading.** Confirmation stops early only on equivalence at ±5 pp or on a
direction. Containment in ±8 pp alone neither ends confirmation nor resolves the primary result,
and the primary conclusion never switches to ±8.

**Support.** A seed whose trajectory does not bracket K\* in an arm is unsupported there. It is
never extrapolated. Reachability per arm and level, and worst-case bounds for the unsupported
blocks, are reported beside the result.

**Secondary analyses:**

- within-seed interpolation at K\*, and the mean paired difference;
- a sign test on the paired differences, read as a statement about their median;
- every contrast at matched K_harmful;
- K_harmful against K_benign;
- utility, refusal, token exposure and run time.

**Utility.** A claim that utility is preserved needs simultaneous one-sided 95 % upper bounds on
the mean utility loss, at the three levels, for both long arms and at both looks: at most 5 pp
on MMLU and at most 10 pp on the factual questions. Seeds are never removed to meet it.

## Frozen Adam-25

- **Unconditional cohort.** Frozen Adam-25 runs on the first 20 preassigned started confirmation
  blocks. It is reported whatever the results, which avoids a winner's curse.
- **Interval.** Its contrasts use a two-sided 95 % interval without sequential control, labeled
  diagnostic.
- **Conditional extension.**
  - It extends to the remaining blocks only if Δ(AdamW-long − SGD-long) excludes zero in the
    primary sequential interval at the terminal look, an early stop included.
  - The same applies if the interval for Δ(AdamW-25 − SGD-long) excludes zero; that interval is
    computed by the same rule.
  - The extension is exploratory and labeled separately.
- **The contrast with AdamW-25** diagnoses the mechanism, a shift of the mean against spreading
  along flat directions. It does not identify it, especially when the two arms run at different
  rates.
- **What it cannot identify.** Frozen Adam-25 against a long arm mixes optimizer and duration.

## Judge and audit

- **Judge.** WildGuard as in the pilot (`configs/judge/wildguard.yaml`, label definitions v2,
  two-class outcome harmful compliance against not). Unparsed outputs follow the
  [rule above](#analysis-population).
- **Audit.** One terminal audit of 400 items per primary arm.
  - **Sampling:** stratified by arm, K band and the judge's own label, with recorded inclusion
    probabilities, spread over seeds and prompts.
  - **Reference labels:** GPT and Grok label independently; the user checks every disagreement
    and 10 % of the agreements. A failed reference output is a missing label, never a safe one.
  - **Fixed in advance:** the audit's size and sampling for both possible stopping looks are
    *to be fixed at Stage B*. Audit results never trigger further training.
- **What the audit estimates.** The judge's differential net bias for the primary curve contrast,
  through estimating equations with the inclusion probabilities, without inverting sensitivity and
  false-positive rates.
- **The bias target.** A bound of 3 pp is a target, not a guarantee: at 400 items the half-width
  is ≈ 3.1 pp.
- **A reference-harm conclusion** needs the judge interval and the bias interval jointly (97.5 %
  each, sequentially valid for the judge). It is reported separately from the judge-rated
  primary result.

## Budget

- **Stage A:** ≈ $40–100.
- **The whole study:** ≈ $160 centrally, up to $320 at B = 5.
- **Separate decisions:** a slow branch at B = 20 above $320 needs a separate decision. The
  barrier pilot (≈ $10–20) runs only if the barrier branch fires.
- **Not included:** reference labeling and storage, which are small.

## To be fixed at Stage B

| Item | Planning value now |
|---|---|
| Maximum started blocks and the two look sizes | ≈ 30 / 63 started, ≥ 24 / 50 supported, cap ≈ 80 started |
| Nominal look levels | ≈ 99.53 % and 95.19 % |
| Curve model and band, by the simulation criterion | logistic with a kink at K\*, band [0.8·5, 1.1U] |
| Fit regularization, failed bootstrap replicates, fallback estimator | the mean of within-seed interpolated differences as the fallback |
| Bootstrap resamples or approximation | ≥ 100 000 |
| Reporting of an underpowered, capped study | the interval and its ±8 reading |
| Audit size and allocation per look | 400 items per primary arm |
| Batch, rates, T_max, U, panels, versions | by the Stage-A rules above |

## Amendments

Format: date, item, old value → new value, reason, and which data had been seen.

| Date | Item | Old → new | Reason | Data seen |
|---|---|---|---|---|
| 2026-10-15 | The monitor panel ([amendment 1](#amendment-1-the-monitor-panel)) | 25 further prompts × 2 responses, disjoint from the scoring panel → the scoring panel's 100 prompts × 1 fresh response each (m = 1), with a nested second response (m = 2) as the declared fallback; scoring diagnostics every 25 steps; θ₀'s logits cached | Version 1 failed the agreement check: it read 13.3 % below the scoring panel at the median (p95 17.5 %; bounds 5 % and 15 %), low at all 14 states, which put the evaluations of the upper level outside the fit band | K only: the 14 full evaluations of the discovery runs so far (K_benign, K_monitor, the scoring panel's K per response, response lengths) and step times; no harm label, no response text, no utility |

### Amendment 1: the monitor panel

Decided on 2026-10-15, during Stage A's discovery, with the harm labels closed, on the
coordinator's decision after the second reviewer's review, before any data under the new
monitor.

- **What failed.** Over the 14 full evaluations of the discovery runs so far (AdamW-long at
  B = 5, seeds 2001 and 2002, all with K_benign ≥ 4), version 1 read below the scoring panel at
  every state: median relative discrepancy 13.3 %, 95th percentile 17.5 % (bounds 5 % and 15 %).
  Its responses are shorter (71.1 against 76.5 tokens), which explains about 7 points; the rest
  is the composition of its 25 prompts (a random 25 × 2 panel has a relative SD of about 8 %
  against the scoring panel). Since the upper level was then evaluated at K ≈ 1.15 U, outside
  the fit band [0.8·5, 1.1U], the monitor is replaced before fixation, not merely enlarged.
- **Why the scoring panel's prompts.** The estimand is conditional on the fixed scoring prompts
  ([Question and estimand](#question-and-estimand): K_benign on the 100 scoring prompts × 4 θ₀
  responses). Sharing those prompts removes the composition mismatch; the monitor's responses
  are fresh θ₀ samples with seeds of their own, so a crossing selects on the monitor's response
  noise, independent of the 400 scoring responses. A separate panel would keep the scoring
  panel's own composition offset (about 3.5 %) at any size. Identical continuations are kept.
  First-passage selection remains: the visited states depend on the monitor, and the reporting
  stays policy-conditional, with reachability.
- **The panel.** Stream `kl_monitor`, monitor version 2: the 100 prompts of `alpaca_benign_kl`,
  2 responses each, zero-field sampling with the scoring panel's cap, seeds
  `derive_seed(0, "kl_reference", "kl_monitor", "v2", "prompts")` and
  `derive_seed(0, "kl_reference", "kl_monitor", "v2")`. The first response of each prompt is
  m = 1; both are m = 2. Its spec hash and sha256 are recorded in
  [operations](operations.md#study-2-stage-a) when it is generated, before any run reads it.
- **Readings.** m = 1 every 5 steps drives the levels, the plateau and the monitor's
  instability rule. The scoring panel's K alone (no harm, no utility) every 25 steps, fixed, and
  m = 2 at those states and at every full evaluation. Full evaluations at the actual crossings,
  as before.
- **Cache.** θ₀'s raw fp32 logits at the response positions, per micro-batch as the uncached
  computation batches them, with the same float64 KL. At θ₀ before training and at every full
  evaluation, every cached reading (m = 1, the second responses, the scoring panel) is compared
  with an uncached one per response; any difference stops the run as a technical failure.
- **Search under version 2.** The long arms' rates are searched downward, each arm (SGD-long,
  AdamW-long) and batch on its own, on all three discovery seeds. SGD-long runs only its screen
  candidates (B = 5: 5·10⁻⁴; B = 20: 2·10⁻³, then 10⁻³), AdamW-long its grid from 10⁻⁵
  downward. U = 8 comes first: the descent stops at the first rate that meets its arm's
  selection rule at U = 8 on all three seeds; a rate that meets it only at U = 7 does not stop
  it. If no batch then qualifies at U = 8, U = 7 is settled from the rates run; a rate that met
  U = 8 need not meet U = 7, whose intermediate evaluation (K\* = √35) differs, so where an arm
  has no run rate meeting U = 7 its descent resumes below them until one does or the grid ends. A rate fails only when all three seeds and the numerical controls it
  needs (SGD-long: its full control) have resolved; a missing run or a technical failure is not
  a failure. The rates below the stopping rate are recorded "not required". The search is
  audited by a conditional manifest — the rates run, the evidence of each outcome and the rates
  not required — which replaces the completeness of the full grid; the campaign generators and
  the selection follow it together. Selected rates set T_max as before; the cost comparison
  concerns the selected confirmation block, not the search's cost, and its stream-generation
  time per chunk is fixed from the version-1 discovery runs (all three seeds' stream extensions,
  when the version-1 grids end), so that the search's pruning cannot change it.
- **Qualification.** On every version-2 trajectory the search prescribes, each counted whatever
  the selection later prefers. Separately for each optimizer, at the batch the selection
  chooses (a well-measured batch cannot hide the chosen one), over the diagnostic states with
  K_scoring ≥ 4: every reading finite, and median |K_monitor − K_scoring| / K_scoring ≤ 5 %,
  95th percentile ≤ 15 %, over at least 10 distinct states. With fewer states, the next rate in
  the search's order of the same arm and batch runs as a diagnostic run, until 10 states are
  reached; a check still short of them is undetermined and does not pass. m = 1 is chosen if it
  passes for both optimizers; otherwise m = 2 if it passes; otherwise the monitor is redesigned
  before validation, without drawing panels until one passes. The choice uses K only. Reported
  alongside, not deciding m: at the actual trigger states, distinct evaluations for 5, K\* and
  U, K\* bracketed by the measured K, and the upper evaluation inside the fit band. If m = 2 is
  chosen, the search is redone under its stopping policy: runs whose stops m = 1 decided run
  again under m = 2, never relabeled.
- **Old discovery.** The version-1 runs stay observations with their measured K_benign, never
  relabeled; their nominal crossings are not evidence about the version-2 policy, and the
  selection uses the version-2 search. Monitor-dependent instability decisions and the full
  controls are redone if their length or evaluation steps change. The 25-step grids, their
  extensions and the screen stand. The fit band is not widened.
- **Non-finite readings.** Every reading is checked for finiteness. A non-finite cached reading
  (the monitor, its second responses, the scoring diagnostics) is read again without the cache:
  reproduced, it stops the run as `nonfinite_k`; finite, the cache failed, and the run stops as a
  technical failure.
- **Code and versions.** Configs carry `eval.monitor.version: 2`; the protocol version stays 2
  (the cache changes no K, which the check above verifies). Stage A now runs from two commits:
  the version-1 discovery from `29e096a`, and everything after this amendment from the commit
  that implements it.
