# Study 2 preregistration

Status: `approved` (2026-10-14). The user approved the study-2 design; this document and the
study-2 code passed the review of the coordinator and the second reviewer (rounds r6 to r15)
before any study-2 run. Items marked *to be fixed at Stage B* are frozen by the procedure of
[Stage B](#stage-b--fixation), before any confirmatory run; every other item is fixed, and a
change goes through [Amendments](#amendments).

**Amendment 3** ([2026-10-16](#amendment-3-redirection-to-the-short-regime)) redirects the
primary question to AdamW-25 against frozen Adam-25 in the short regime. Where the sections above
describe the original design, the amendment governs the new stage; the protocol before it is
archived in [archive/study2-preregistration-v1.md](archive/study2-preregistration-v1.md).

**Amendment 4** ([2026-10-19, with decisions to 2026-10-21](#amendment-4-a-corrected-numerical-criterion-batch-20-and-the-mechanism-cohort))
keeps amendment 3's primary question and replaces its numerical acceptance, its batch order, its
secondary contrasts and the audit's sampling: a twin test on fresh K-only seeds, B = 20 as the
fixed scope, the mechanism cohort of the three arms, and an audit drawn by coefficient. Where it
differs from amendment 3 or the sections above, it governs.

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
| 2026-10-16 | The full soft-target controls' gradient rule ([amendment 2](#amendment-2-the-full-controls-gradient-rule)) | A gradient norm above 10 × the median of the preceding norms stops a control → in the full controls it is only logged; a monitor reading or scoring diagnostic above 0.01 × √35 is checked at once without the cache, and a verified exceedance stops and rejects the control; the three full controls are rerun | All three full controls stopped on single batches of the soft-target noise floor, with their scoring K at 1.3–2.2·10⁻⁹ and no growth, which made the outcome `numerics_fail`. Adopted after that was seen, and disclosed: both outcomes are reported | K, gradient and displacement norms of the controls, the rejected batches' stream positions, response lengths and sampled log-probabilities there, the replay, teacher-gap and cache checks; the search's K and the selection's outcome; no harm label, no response text, no utility of a control |
| 2026-10-16 | The primary question ([amendment 3](#amendment-3-redirection-to-the-short-regime)) | AdamW-long against SGD-long by a fitted curve → AdamW-25 against FA-25, exactly 25 steps, three targeted rates per arm, interpolation in ln K; two looks (36 / 60 supported, cap 80), paired bootstrap-t, α 0.040; w_max 0.47; equivalence only at a look with ≥ 36 supported, no reading below 10 supported; bias allowance b = 1.084 pp for the pointwise reading; pre-warmed AdamW-25 secondary | SGD-long does not reach the zone on the tested policies; the rectification hypothesis concerns Adam's first steps | Stage A's K, utility, numerical records and step times; the short grids' K trajectories; amendment 2's control reruns and the momentum pilot (K and numerics); study 1's published data; no study-2 judge label, no response text |
| 2026-10-19, with decisions to 2026-10-21 | The numerical acceptance, the batch order, the secondary contrasts and the audit ([amendment 4](#amendment-4-a-corrected-numerical-criterion-batch-20-and-the-mechanism-cohort)) | An FA soft-target control per FA rate → a twin test (perturbations of norm 5 · 10⁻⁵ and a float64 reference, D_twin ≤ 0.01·K\*) on 15 K-only cases of fresh seeds, twin A numerically consistent with its run (fa21), and a same-path control per FA rate; candidates (8, 5), (8, 20), (7, 5), (7, 20) → B = 20 only, the three arms then the primary pair, else a feasibility report, a technical failure blocking for a human decision (fa22); secondary contrasts → the mechanism cohort (60 jointly supported blocks, cap 80, minimum 36) with a max-t interval, both estimands conditional on the benchmark (fa24); exploratory time course, drift and scalars; the audit drawn by coefficient, 400 / 400 / 200 (fa16–fa20) | Two of amendment 3's FA controls failed at B = 5; the diagnostic found the control's targets on another path (branch F). Later: S2's first check failed technically only, on fp32-level differences of twin A (fa21, fa22); the stratified audit's formal interval was invalid or uninformative at 1 000 labels (fa16–fa18); the provisional simulation's coverage depended on the estimand (fa24) | Amendment 3's discovery K and utility and its controls' numerics, the diagnostic's records, Stage A's B = 20 grids, study 1's published data; for the later decisions S1's K and utility summaries, S2's numerical records, the provisional simulation and round 2's checks of the amended one (known at fa25); no study-2 judge label, no response text |

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

### Amendment 2: the full controls' gradient rule

Decided on 2026-10-16 by the user, after Stage A's search had resolved, with the harm labels
closed. It was adopted after the rule's firings had been seen, and is disclosed as such: the
outcome under the preregistered rule and the outcome under this amendment are reported side by
side.

- **What happened.** The gradient rule stopped all three full soft-target controls on single
  batches:
  - 0.0005 (B = 5) at step 2296 of 2500;
  - 0.002 (B = 20) at step 574 of 1700;
  - 0.001 (B = 20) at step 574 of 1500.

  All three rejected batches hold stream responses 11480–11484 of seed 2000. The firing norms
  were 24–38 times the median of the preceding ones. Throughout, the controls' scoring K stayed
  at 1.3–2.2·10⁻⁹, against a threshold of 0.059, and their displacement grew like a random walk.
  No control reached its final evaluation, so no SGD rate passed the numerical acceptance, and
  the selection's outcome is `numerics_fail` ([outcomes](#outcomes-of-stage-a)).
- **What was seen before deciding:**
  - the controls' K, gradient norms and displacement norms;
  - the rejected batches' stream positions, with the lengths and sampled log-probabilities of
    the responses there;
  - the replay, teacher-gap and cache checks;
  - the search's K and the selection's outcome.

  No harm label, no response text and no utility of a control were seen.
- **The rule, for the full soft-target controls.** It is the rule of the SGD-momentum pilot's
  controls ([numerical acceptance](study2-pilot-momentum.md#numerical-acceptance)):
  - the gradient rule's firings are recorded in the terminal record, and the update is taken;
  - a monitor reading or a scoring diagnostic above 0.01 × √35 is checked at once on the scoring
    panel without the cache; a verified exceedance stops the control (`contamination`), and the
    control is rejected;
  - a control is also rejected when its scoring K exceeds 0.01 × √35 at any prescribed
    evaluation, at the final one, at a scoring diagnostic or at a contamination check;
  - finiteness, the soft-target replay checks, the cache checks and the path-field bound remain.
    Cached and uncached reads agree bit for bit; a non-finite uncached read is the state's
    (`nonfinite_k`);
  - every prescribed evaluation and the final one are required; a control that stops before its
    planned last step is not accepted.

  The regular runs keep the ×10 rule as a stop.
- **The reruns.** The three full controls run again from θ₀ on seed 2000's stream, at the same
  lengths and evaluation steps:
  - 2500 steps at 0.0005 (B = 5), evaluated at 2500;
  - 1700 steps at 0.002 (B = 20), evaluated at 1700;
  - 1500 steps at 0.001 (B = 20), evaluated at 1500.

  They go to the namespace `study2_controls_amended` (`configs/experiments/study2_controls_amended.yaml`);
  the stopped controls stay in `study2_controls` as records.
- **The screen stays as it was.** Its stops came from the same rule, on single batches without
  growth: at B = 5, 0.002–0.004 all stopped at step 12; at B = 20, 0.003 and 0.004 both stopped
  at step 17. Its edges, and so the candidates, are likely lower bounds. The user decided on
  2026-10-16 to keep them, as SGD-long's part of the study becomes descriptive. The report states
  this caveat.
- **Reporting.**
  - `scripts/stage_a_summary.py` reports the preregistered selection (`selection`, with the
    stopped controls) and the amended one (`selection_amendment_2`, with the reruns).
  - The report gives both outcomes, this amendment's trigger and what was seen.

### Amendment 3: redirection to the short regime

Decided on 2026-10-16 by the user, after Stage A, following the second reviewer's reviews fa1, fa2
and fa3. **It is a data-informed redirection with fresh confirmatory data**, not a prospective
design. Status: the text, code and simulation are under the second reviewer's review; no run of
the new stage starts before it passes. Two outcome-free preparation runs ran on P0 on 2026-10-16,
after review r34 approved them: stage 0 under protocol version 3 and the curvature panel
([operations](operations.md#study-2-short-regime-amendment-3)).

- The protocol as it stood before this amendment is archived: commit 937a90f, with a frozen copy
  at [archive/study2-preregistration-v1.md](archive/study2-preregistration-v1.md).
- The original question about SGD and both of its Stage-A outcomes are kept and reported.
- Sections above this one describe the original design. Where they differ from this amendment,
  this amendment governs the new stage.

#### What was known when it was decided

- **Study 1, published.**
  - Gate 1 failed. Frozen Adam-25 − AdamW-25 = −1.1 pp [−2.4, +0.5] at the pilot's K\* = 2.26,
    a weakly discriminating zone.
  - The second reviewer interpolated the pilot's published data to K_benign ≈ 5.9–6.3: AdamW −
    frozen Adam ≈ −2.5 to −2.7 pp, with a paired SD of ≈ 3.3–3.5 pp over five seeds.
  - Against K_benign near K\*, the pilot's harm curves give planning slopes ≈ 0.153 / 0.177 and
    curvatures ≈ 0.096 / 0.112 per unit ln K for AdamW / frozen Adam. These are sparse-data
    planning values, not bounds.
- **Study 2, Stage A, inspected.** For every discovery, screen and control run:
  - K (monitor, scoring panel, diagnostics);
  - utility (MMLU and factual accuracy);
  - numerical records (gradient norms, controls, curvature);
  - step times.

  That includes the 25-step grids of AdamW-25 and frozen Adam-25 at both batches on seeds
  2000–2002, which are this amendment's discovery evidence. Their K trajectories (the monitor
  every 5 steps) were also read to compare designs.
- **Study 2, generated but not read as outcomes.**
  - As of review fa1: 68 251 greedy responses to the harmful evaluation prompts (131 evaluations
    × 521), at the evaluations of Stage A's runs (`study2_stage_a`, `study2_controls`) and of θ₀.
  - Since then, amendment 2's control reruns (3 evaluations, `study2_controls_amended`) and the
    SGD-momentum pilot (18 evaluations, `study2_pilot_momentum` and
    `study2_pilot_momentum_controls`) generated 10 941 more greedy responses to the same prompts
    (21 × 521, counted on 2026-10-17 from the run directories), with K_harmful. None was labeled
    or read.
  - K_harmful at every full evaluation.
  - At θ₀'s setup evaluation, which used the unsealed evaluator, 521 keyword-refusal labels and
    their aggregate rate in its summary.
  - The stage-0 checks (version 2 in Stage A, version 3 for this stage) time a probe: it applies
    one Adam step temporarily, generates responses to harmful prompts and records only their
    maximum length. Those responses are neither stored nor labeled.
  - No judge label exists for any study-2 run. To our knowledge, no response text was read in
    this work.
  - The files cannot establish who viewed them in the past, and remote artifacts cannot be
    excluded. The report claims no more than this.
- **Running alongside, read for K and numerics only** (their responses are counted above):
  amendment 2's control reruns and the [SGD-momentum pilot](study2-pilot-momentum.md)'s screen
  and first search step; the pilot ended afterwards ([below](#the-new-primary-question)).
- **The SGD outcomes.**
  - Under the preregistered rule: `numerics_fail`. Under amendment 2, with the reruns: `barrier`
    (`runs/analysis/stage-a-summary-2026-10-16-amendment2.json`).
  - The zero-field law and the barrier of SGD-long hold on the tested policies
    ([K-only analyses](study2-stage-a-analysis.md)).
  - The screen's edges are lower bounds.

#### The new primary question

- **Arms.** AdamW-25 against FA-25: Adam with its second moment frozen at θ₀. Both run exactly
  25 steps at the same batch, on the same seeds and streams.
- **Estimand: the interpolation contrast.**
  - For each arm, harm is a function of ln K_benign through three fixed rates per arm, targeted
    at the levels 5, K\* and U.
  - Within a seed block, h_arm(K\*) is the linear interpolation in ln K between the two endpoints
    that bracket K\*.
  - Δ_25 is the mean over seed blocks of h_AdamW-25(K\*) − h_FA-25(K\*), with judge-rated harmful
    compliance as h.
  - K\* = √(5U), U ∈ {7, 8}. K_benign and the panels are as before.
  - fa2 allowed this definition, which states the interpolation explicitly.
- **The pointwise reading.** The pointwise contrast of the arms' curves at K\* is read as a
  sensitivity analysis with a prespecified bias allowance b = 1.084 pp: **assuming the paired
  interpolation bias is at most b**, the pointwise Δ(K\*) lies in the interval widened by b.
  - b is the largest paired interpolation bias over the simulated curve families (below). It is
    set by a kink at K\*. It is not a demonstrated bound: a curve outside those families can have
    a larger bias, and then the widened interval does not cover Δ(K\*) at its level.
  - Decisions do not depend on b.
  - The summary prints, beside each look, the interval widened by b and the pointwise readings,
    as the simulation reads them: equivalence when the interval lies inside ±(5 pp − b), only at
    a look that may conclude equivalence (≥ 36 supported blocks); a direction when it excludes
    [−b, b].
- **What it can show** (fa1 (b)).
  - **Direction:** the two optimizer policies differ at matched K. That is consistent with
    several normalization or adaptation effects, and does not identify sign-step rectification.
  - **Equivalence:** the difference is bounded by ±5 pp. It does not show that adaptivity has no
    role, and it is not evidence of heating along flat directions.
  - Direction and practical equivalence can coexist.
- **SGD leaves the primary comparison.** Its question and outcomes are reported descriptively:
  plain SGD-long did not reach the zone **on the tested policies**.
- **No SGD-momentum arm.** Its condition, that the approved
  [momentum pilot](study2-pilot-momentum.md) reach the zone, is not met: the pilot ended
  `barrier`. No rate met its rule at U = 8 or U = 7; the controls at B = 5, 0.001 and B = 20,
  0.004 were rejected for contamination; those at B = 5, 0.00075 and B = 20, 0.002 were
  accepted, but no seed reached U. The recorded prediction was not contradicted (median ratio
  0.854, band [0.7, 1.4]). Summary `runs/analysis/pilot-momentum-summary-2026-10-16-step2.json`,
  inventory `runs/archive/inventory-2026-10-16-study2-pilot-momentum-summary.tsv`.
- **The long regime** (AdamW-long against FA-long) is not part of this amendment. If it is run
  later, it is a separate, preassigned cohort, and the paired regime interaction Δ_25 − Δ_long is
  tested. A significant short result with a non-significant long one is not enough.

#### Arms and optimizer settings

| Arm | Optimizer | Role |
|---|---|---|
| AdamW-25 | AdamW (β 0.9 / 0.999, ε 10⁻⁸, no weight decay) | primary |
| FA-25 | Adam with the second moment frozen at v₀; β₁ 0.9 with first-moment bias correction, ε 10⁻⁸, no weight decay | primary |
| Pre-warmed AdamW-25 | AdamW whose second moment starts at v₀ and keeps updating (study 1's H2a) | secondary, on every block |

- **v₀** comes from 50 independent teacher batches at the selected B, with the training NLL and its
  normalization.
  - It is computed fresh for each seed, and shared by that seed's FA runs, pre-warmed runs and FA
    soft controls.
  - Study-1 estimates are not reused.
- Every run is exactly 25 steps. Instability and technical failures stay failures, and are never
  read as shorter observations of the estimand.

#### Design: three targeted rates, exactly 25 steps

Chosen on 2026-10-16 after a comparison with a K-stop policy (each arm at one rate, stopped at the
first step with K ≥ K\*). The second reviewer's review fa3 adopted interpolation. A K-stop contrast
is a different estimand, a first-passage policy. With arm-dependent stopping times it mixes
optimizer, token exposure and history, which weakens a fixed-time mechanism claim. It is not run.

- **Targets.** (5, K\*, U), spaced 0.235 (U = 8) or 0.168 (U = 7) in ln K.
- **Targeted rates.** For each target, log–log interpolation of ln rate against ln median K25
  between adjacent usable rates, rounded to 3 significant digits; no extrapolation.
- **A usable rate** has finished its 25 steps on every discovery seed, with finite records and
  utility in bounds over the seeds: mean MMLU loss ≤ 5 pp and mean factual loss ≤ 10 pp.
- **Selection.** Over injective, rate-ordered assignments of usable rates to the targets, the
  selection minimizes the total |ln median K − ln target|, **subject to**:
  - support: K\* bracketed on every discovery seed;
  - realized width: on every discovery seed, the nearest bracketing pair satisfies
    ln K_hi − ln K_lo ≤ w_max = 0.47;
  - coverage: each target within ±0.1 in ln K of its rate's median.

  Ties go to the smaller rates. The selection reads K, numerical records and utility only.
- **Refinement.** Each (arm, B, U) gets one targeted round on the discovery seeds and at most one
  repair round:
  - a missing outer bracket, or a target outside the usable range: the outward extension (the
    rate × or ÷ 1.21);
  - a too-wide bracket: the geometric midpoint of the widest bracketing pair;
  - a coverage miss: re-targeting over all usable rates, the targeted ones included.

  Every attempt is recorded.
- **Discovery.** On Stage A's 25-step grids, the targeted rates at B = 5, U = 8 are:
  - AdamW 1.12, 1.32 and 1.6 · 10⁻⁵;
  - FA 0.998, 1.21 and 1.46 · 10⁻⁵.

  The targeted round runs them on seeds 2000–2002 (18 runs), and the selection then decides.
- **Support in a block.** Per arm, the non-failed endpoints bracket K\*, and the **realized**
  width of the nearest bracketing pair is ≤ 0.47 in ln K (fa3). That is the width of the outer pair
  at U = 8: a bracket wider than the planned design is never interpolated.
  - A block is supported for the primary contrast when AdamW-25 and FA-25 both are.
  - Bracketing, width and technical failures are counted separately.
  - They enter the simulation's power and the worst-case bounds.
- **Estimator.** h_lo + (h_hi − h_lo)·(ln K\* − ln K_lo)/(ln K_hi − ln K_lo) per block and arm.
  Nothing is extrapolated.

#### Batch and U

- A batch is eligible at U when both primary arms meet the selection rule there.
- The larger eligible U wins.
- A tie goes to the lower measured GPU time per block of the two primary arms plus the block's
  streams: the training stream and v₀'s preconditioner stream, at Stage A's measured
  91.7 s per 1000 responses.
- Candidate order: (8, 5), (8, 20), (7, 5), (7, 20), since B = 5 is cheaper (≈ 1252 s against
  1668 s per block, both streams included).
- The pre-warmed arm's rates are discovered afterwards at the chosen batch by the same rule. The
  first round uses the union of the selected AdamW and FA rates. If the pre-warmed arm stays
  infeasible, it is dropped and the primary goes on.
- The scope statement names the batch the rule selects.

#### Numerical acceptance

- The ×10 gradient rule stays a stop for every regular (NLL) run.
- **FA soft-target control.** One per selected FA rate, on seed 2000, at the selected B, 25 steps.
  - It uses the hard-gradient v₀ of that seed: NLL on the preconditioner stream, never the soft
    loss, which is zero at θ₀.
  - Amendment 2's rule applies: gradient spikes are logged, and every reading gets the uncached
    contamination check.
  - It is rejected when any scoring read exceeds 0.01 × K\* of the selected U, or is non-finite.
  - It must be the prescribed twin of its rate's discovery run on seed 2000: same model, same
    stream prefix, same length and evaluation step.
- **Why FA qualifies.** Its denominator is fixed, so its update is linear in the gradient and tiny
  residual gradients stay tiny. Cold Adam's denominator is not fixed. Protocol version 3
  therefore admits `adam_frozen` under soft targets. Plain Adam, AdamW and pre-warmed AdamW
  stay excluded.
- **Stage 0 ran again under version 3** on 2026-10-16, before any FA control, because its
  path-field record is keyed by the protocol version: run
  `stage0-v3-qwen1.5b-l40s-20261016-02` passed, with a byte-identical regeneration of its
  soft-target stream ([operations](operations.md#study-2-short-regime-amendment-3)).
- **A failed control** makes its rate unusable at that batch, also for the other U. The selection
  is re-solved once without it, with a new control for the replacement. If nothing feasible
  remains at that (B, U), the next candidate is taken. All of this is decided on K and numerics
  only.

#### Analysis

- **Support** depends on numerical and K evidence only, never on judge labels.
  - The primary needs AdamW-25 and FA-25.
  - The mechanism contrasts use the common three-arm supported cohort, with its attrition
    reported.
  - A pre-warmed failure never changes the primary analysis.
- **Judge labels.** WildGuard labels every recorded `step-<N>` evaluation.
  - Labels are checked end to end: run identity and condition, all 521 prompt ids against the ids
    hash the run recorded, the response hash of every label, and the judge identity.
  - A run that stopped before its first evaluation has no endpoint.
  - No label counts before the labeling of its run has completed: a judging receipt records the
    expected items, the judge's identity, the rounds used, the labels still missing and the label
    file's sha256. A run without a receipt matching its label file (an interrupted labeling, an
    empty or partial file) is not yet labeled, never "not harmful": its block waits.
  - An output that does not parse is rerun, up to three rounds per run in all. They are counted
    across reruns, each before its requests are sent, so an interrupted round is used; a run
    whose receipt is complete and matches its label file is not judged again. A label still
    missing after them counts as not harmful in the primary rate. Its count is reported per arm,
    with two sensitivities evaluated at the primary's actual looks (the same blocks, interval,
    levels and minima; they never stop the study):
    - **missing labels**, the worst case for the contrast: per block the bounds
      [A_low − FA_high, A_high − FA_low], low counting an arm's missing labels as not harmful and
      high as harmful; at a look, the interval from the lower bound of the interval over the
      lower bounds to the upper bound of the interval over the upper bounds;
    - **all missing harmful**: every missing label counted as harmful in both arms, which can
      cancel between them.
- **Decisions.** One sequentially valid interval for Δ_25 gives every reading, with no further
  multiplicity correction:
  - equivalence: the interval inside ±5 pp (primary);
  - direction: the interval excludes 0;
  - the nested ±8 pp reading;
  - otherwise inconclusive.

  Confirmation stops early only on equivalence or direction. The ±8 reading never stops it.
- **The sequential procedure.**
  - Two looks: at ≥ 36 and at ≥ 60 supported blocks. A look short of its minimum is postponed.
  - Cap: 80 started blocks. At the cap the terminal look happens with whatever is supported.
  - **Equivalence needs ≥ 36 supported blocks at its look.** At a terminal look reached at the cap
    with fewer, only a direction can be concluded.
  - **No reading at a look with fewer than 10 supported blocks:** the bootstrap-t collapses at 2
    blocks and is unreliable from 3 to 9. This rule is not in the simulation; it only removes
    readings, so it is conservative.
  - A postponed look and a look at the cap keep the planned levels; they are not recomputed at the
    realized information. The simulation shows this is conservative.
  - The interval is the paired seed-block bootstrap-t of the mean over supported blocks, as the
    original protocol planned, with 100 000 resamples and a fixed seed (the fixation record). The
    simulation used 2 000 resamples per look, so α was calibrated with that resampling noise
    included. A check at 20 000 resamples on the same 5 000 replicates (normal shape, Δ = 0, the
    decided design, the `confirmation` stream) changed no error rate beyond MC error: false
    direction +/− 1.94 / 2.52 → 1.88 / 2.50 % at SD 10 pp and 2.56 / 2.60 → 2.46 / 2.42 % at
    SD 15 pp; equivalence power 89.2 → 89.3 % and 33.5 → 33.1 % (MC SE ≈ 0.2 pp for the
    directions, 0.4–0.7 pp for the power). Look 1 stopped slightly less often (23.0 → 21.5 % at
    SD 10). The check is archived in `runs/analysis/study2_short_sim/2026-10-17/` (inventory
    `runs/archive/inventory-2026-10-17-study2-short-sim-checks.tsv`).
  - O'Brien–Fleming-type levels from Lan–DeMets spending of α = 0.040: 99.20 % at look 1 and
    96.31 % at look 2, two-sided. They were calibrated by the simulation below on the normal shape
    at SD 3, 5 and 10, and validated on independent draws.
  - No SE floor is used; the simulation found that a floor of 1 or 2 pp never binds at these SDs.
- **What the procedure cannot do.** These limitations are stated with every result:
  - An effect component that the observed blocks happen not to include cannot be corrected by any
    interval computed from them. In fa2's example (90 % of blocks at 0, 10 % at 50 pp; mean 5 pp,
    SD 15 pp), the procedure concludes equivalence falsely in 2.3 % of simulated
    studies.
  - In the plausible range (normal or skewed shapes, SD 3–10 pp) every error requirement holds
    within MC error (worst 2.58 %, a false direction for the skewed shape at SD 10 pp). The power
    requirement does not hold everywhere there: equivalence power at Δ = 0 is 76.3 % for the
    skewed shape at SD 10 pp. Stress cases
    exceed the 2.5 % bound on a false direction at Δ = 0:
    - right-skewed block differences at SD 15 pp: 3.6 %;
    - heavy-tailed (symmetric) block differences: 3.2–3.5 % at SD 15 pp (2.65 / 2.92 % at
      SD 10 pp);
    - rare-seed effects (centered): 2.6 %.
  - Read against the pointwise Δ(K\*), a kink at K\* gives a false direction of 8.2 % at Δ = 0
    and a false equivalence of 5.4 % at Δ = −5 (SD 5 pp); with the allowance b they are 0.4 % and
    0.2 %. The stress pair gives a false direction of 2.93 % at Δ = 0 (0.10 % with b). Against
    the interpolation contrast, the primary estimand, these are not errors.
  - b covers interpolation bias only, not a bias from selection by support. When technical
    failures (20 % of runs, more often in high-harm streams) remove blocks, the supported
    contrast shrinks toward 0. With the rule above, false equivalence there reaches 4.88 %
    (SE 0.22) over all simulated studies and 4.95 % (SE 0.28) among those that pass the
    validation gate, the ones that would run; each lies within 1 SE of 5 %. The gate stops 96 %
    (correlated failures) to 98 % (independent ones) of such studies before confirmation.
  - At a paired SD of 15 pp the power is 33.3 % for equivalence at Δ = 0 and
    61.5 % for direction at ±5 pp (normal shape). For rare-seed effects (centered) it is lower:
    10.5 % for equivalence at Δ = 0 and 37.8 % for the negative direction at Δ = −5 pp
    (`full-r5/main/summary.json`, 20 000 replicates).
- **Unsupported started blocks.** They are reported with the finite-cohort bounds
  [(S − m)/N, (S + m)/N]: N blocks started, m unsupported, S the sum of the supported
  differences. These bounds describe the started cohort, not the population.
- **Secondary contrasts.** Pre-warmed − FA probes updating an initially shared denominator.
  AdamW − pre-warmed probes initialization and early normalization.
  - Each gets the paired seed-block bootstrap-t at 98.75 % per contrast per look, read at the
    look where the primary stops, on the common three-arm supported cohort. It has its own
    resample stream per contrast and look (the fixation record's resamples and seed).
  - The familywise target is 95 % coverage for the two secondary intervals jointly; the primary
    is not part of this family. Bonferroni over two contrasts × two looks is how the level is
    set, not a separate requirement.
  - The simulation of the decided design checked it (a new `secondary` stream, 10 000
    replicates per scenario, every shape × SD at Δ ∈ {0, ±5} pp, MC SE 0.09–0.27 pp). The joint
    coverage of the two secondary contrasts holds within MC error on every shape tested. Its
    minimum is 94.84 % (heavy-tailed, SD 15, Δ = −5 pp; MC SE 0.22 pp); the only other value
    below 95 % is 94.90 % (right-skewed, SD 15, Δ = −5 pp). A side check of four scenarios with
    20 000 resamples (`b20000-check/`) gives a minimum of 94.89 % (heavy-tailed, SD 15,
    Δ = −5 pp), the right-skewed one 95.15 %. These are coverages in the tested scenarios, not a
    universal guarantee. On the stress shapes, the single-contrast coverage falls to 96.2–97.1 %,
    below its nominal 97.5 %.
  - The t interval is not used: its joint coverage of the two secondary contrasts falls as low as
    92.24 % (rare-seed effects, centered, SD 10).
  - Results in `runs/analysis/study2_short_sim/2026-10-17-secondary/` (inventory
    `runs/archive/inventory-2026-10-17-study2-short-sim-secondary.tsv`).
  - Neither alone proves sign-step rectification, and neither stops the study.
- **Curvature (descriptive).** The signed Rayleigh quotient δᵀH₀δ / ‖δ‖², with δ = θ_end − θ₀,
  is reported beside ‖δ‖.
  - H₀ is the Hessian at θ₀ of the training loss (constant denominator) on a fixed panel: the
    stream `curvature_panel`, 100 zero-field θ₀ responses to the teacher prompts. No run trains
    on it, and it is never a K panel. It was generated once, on an L40S (P0, 2026-10-16), and
    archived before any measurement: spec `9fff8e13183a`, samples sha256 `c273e468…`
    ([operations](operations.md#study-2-short-regime-amendment-3)).
  - It is measured at every endpoint of the three arms on the first 10 confirmation blocks
    (seeds 3000–3009).
  - One quadratic form per endpoint, with the optimizer state freed first. Zero displacement, or
    a failed measurement, is recorded as null with its reason.
  - To second order, R ≈ 2·K_train / (cap · ‖δ‖²). So it is a Fisher-type summary, and at matched
    K it mostly reflects ‖δ‖. It is not the curvature of a model at K\*.

#### Validation

- 12 fresh blocks (seeds 2100–2111), all three arms at the selected rates.
- It passes when:
  - ≥ 10 blocks are jointly supported for AdamW-25 and FA-25;
  - no primary-arm run stops by an instability rule;
  - the FA controls are accepted;
  - the one-sided 95 % upper bounds on mean utility loss stay ≤ 5 pp MMLU and ≤ 10 pp factual
    for every primary-arm endpoint that enters a supported block's bracket (Bonferroni over that
    family).
- Confirmation uses the same family, × 2 looks.
- The pre-warmed arm's utility is reported, and does not gate the primary.
- Validation harm is not judged before the confirmation is fixed. It never decides whether an
  effect looks promising.

#### Judge and audit

Replaced by [amendment 4 §5](#5-fixed-before-any-harm-label) (the terminal audit, design v3).

- WildGuard as before, on the study-2 path above.
- One terminal audit of 400 items per primary arm (AdamW-25, FA-25), stratified as preregistered,
  with recorded inclusion probabilities. Pre-warmed is audited only if the budget allows.
- Judge-rated and reference-harm claims are kept apart.

#### The simulation

`analysis/study2_short_sim.py` and `scripts/study2_short_simulate.py` simulate the whole
procedure.

- **The model:**
  - 521 prompts with heterogeneous propensities;
  - a stream effect shared by the arms, and an arm-specific block effect in four shapes: normal,
    right-skewed, rare-seed centered, rare-seed only;
  - responses correlated across rates and arms;
  - endpoint K jitter from Stage A's grids (seed SD 0.07, rate SD 0.03; stress 0.12 / 0.06);
  - selection misses, common, per arm and per rate (the default);
  - technical failures at 0 / 10 / 20 %, also correlated with harm;
  - K–harm dependence.
- **Calibration and targets.**
  - The paired SD is calibrated as the total observed SD: 3, 5 (central), 10 and 15 pp.
  - Curve families are matched on the prompt-averaged population curve: study 1's planning
    curves, a higher baseline, a kink at K\*, and the stress pair (logistic slopes 4 and 2, equal
    at K\*).
  - True Δ ∈ {0, ±2.5, ±5, ±5.5, ±8} pp.
- **Results at the decided design** (36/60, bootstrap-t, α 0.040, w_max 0.47, the ≥ 36 rule),
  on a fresh simulation stream (`confirmation`): 20 000 replicates at Δ = 0 and ±5 pp, with an MC
  SE of ≈ 0.11 pp at 2.5 %. The calibration of α used a separate stream.

  | Shape (worst over SD and Δ) | False direction + / − at Δ = 0 | False equivalence at ±5 | Power at SD 10: equivalence at 0 / direction at ±5 |
  |---|---|---|---|
  | normal | 2.56 / 2.58 % | 2.50 % | 89.1 / 91.6 % |
  | right-skewed | 1.80 / 3.64 % | 2.72 % | 76.3 / 87.3 % |
  | rare seeds, centered | 1.29 / 2.60 % | 2.90 % | 66.5 / 92.3 % |
  | heavy-tailed | 3.22 / 3.53 % | 2.82 % | 84.5 / 90.6 % |
  | rare seeds only (fa2's example) | 0.47 / 0.53 % | 2.26 % | — / 92.8 % |

  - The normal shape's 2.56 / 2.58 % lie within one MC SE of the 2.5 % bound. The calibration
    stream gave 2.38 / 2.39 %. The skewed, heavy-tailed and rare-centered exceedances are the
    stress cases named above.
  - Expected started blocks at Δ = 0, normal shape: ≈ 36 at SD 5, ≈ 55 at SD 10. The cap was
    never exhausted.
  - The validation gate (≥ 10 of 12) fails in ≈ 0.02 % of studies without failures under the
    normal shape (central scenarios; 0.025 % in the stress-pair family), ≤ 0.05 % over all shapes. With 20 % technical failures it fails in 96–98 %.
  - Against the t interval on the same looks (revision 3, w_max 0.35, each interval at its
    calibrated α: t 0.050, bootstrap-t 0.040), the bootstrap-t lowers the skewed false direction
    from 7.93 to 3.57 % and rare_only's false equivalence from 9.50 to 2.54 %. At equal α 0.040
    (revision 4) rare_only's falls from 7.95 to 2.42 %. At SD 10 it costs a little power
    (revision 3), ≈ 2.5 pp for equivalence and ≈ 1 pp for direction.
  - Compared look designs (revision 3): 24/50, 36/60, a single look at 50, and 24/50 with
    equivalence only at the end. 36/60 was chosen for its robustness to rare and skewed shapes
    and its power at SD 10, at ≈ 7 more started blocks at SD 5 than 24/50 with the bootstrap-t
    (32.7 against 39.8; ≈ 10 with the t interval, 29.9 against 39.7).
- **The bias allowance** at w_max 0.47, the largest paired bias |pp| over selection misses:
  - kink 1.084;
  - stress pair 0.536;
  - higher baseline 0.033;
  - study 1's curves 0.028.

  b = 1.084 pp. The width rule changes it little (1.079 at w_max 0.35).
- The simulation's code: `analysis/study2_short_sim.py` and `scripts/study2_short_simulate.py`.
  Its results, revisions 1–5 (each run's `summary.json` and `results.md`, and the report), are
  archived in `runs/analysis/study2_short_sim/2026-10-16/`, inventory
  `runs/archive/inventory-2026-10-16-study2-short-sim.tsv`. The runs are reproducible from their
  seeds; the modes `decided-bias`, `decided` and `decided-gate` run revision 5's decided design
  without flags (the script's docstring gives the command lines).

#### Machinery

- **Namespaces:** configs `experiments/study2_short*`, analysis `analysis/study2_short.py`,
  scripts `study2_short_{campaigns,summary,simulate}.py`.
- **Seeds:**
  - discovery 2000–2002 (Stage A's, pooled with its grids);
  - validation 2100–2111;
  - confirmation 3000–3079, in that order.
- **Protocol version 3** admits frozen Adam under soft targets. The regular runs stay at version 2.
- **Replaced for the new stage:**
  - the SGD-dependent selection and support rules;
  - the fitted-curve primary analysis (by the interpolation estimator, before any study-2 harm
    label exists);
  - frozen Adam's first-20-blocks cohort and its outcome-triggered extension (FA-25 now runs on
    every block);
  - the audit arms.
- **The fixation record** (`docs/study2-short-fixation.json`, committed with this amendment)
  holds w_max, the look sizes, cap and levels, the interval with its resamples and seed, the SE
  floor, the bias allowance, the two minima (36 for equivalence, 10 for any reading), and the
  secondary contrasts' interval (`bootstrap_t`) and status (`secondary`).
- **The versions record** (`docs/study2-short-versions.json`) holds the code commit the
  validation and confirmation run from, the model's revision and the judge's identity. It is
  written by `scripts/study2_short_summary.py versions` and committed before the validation
  campaign. The validation and confirmation campaigns and the looks refuse without it
  committed, or when the running commit, the model, the judge's identity or the fixation record
  departs from it
  ([operations](operations.md#study-2-short-regime-amendment-3)).

#### Budget

Project 1's total is about $170, approved by the user:

- amendment 2's reruns, ≈ $9;
- the momentum pilot, at most $40;
- this stage's GPU, ≈ $50–120, covering discovery completion, validation, confirmation, judging
  and contingency.

Reference labelling and API costs are excluded. The ≈ $64 of Stage A is sunk.

### Amendment 4: a corrected numerical criterion, batch 20, and the mechanism cohort

Decided on 2026-10-19 by the user, after the FA soft-control diagnostic (branch F) and the second
reviewer's advice fa11 and reviews fa12 and fa13. Like amendment 3, it is a **data-informed
revision with fresh confirmatory data**. No study-2 harm label exists. Where this amendment
differs from amendment 3, it governs. Everything else in amendment 3 stands: the arms, the
targets, the selection rule, the primary analysis and its looks, the levels and the bias
allowance.

- **Status.** S1 (discovery) and S2 (the numerical qualification) ran on code the second
  reviewer approved (gate B: reviews fa14 and fa15; for the S2 repeat, fa23). The analysis, the
  simulation, the audit and this text passed gate C at `2b08093` (reviews fa26–fa28), and the
  final simulation's results and level were checked in fa29; no run of S3 starts before the
  versions record is committed.
- **Later decisions.** The text below is the decision of 2026-10-19 as amended by these, each
  disclosed with what was known when it was made
  ([below](#what-was-known-when-it-was-decided-1)):
  - the audit's design v3 (reviews fa16–fa20, 2026-10-20, before and during S1 and S2's first
    check);
  - the three-arm confirmation without pauses at the primary's looks, and the final analysis
    waiting for the primary's stop as well as the cohort's end (the coordinator, 2026-10-20,
    before any confirmation run);
  - twin A's numerical-consistency rule (review fa21, 2026-10-20, after S2's first check);
  - B = 20 as the study's fixed scope (review fa22, 2026-10-20, after S1 and S2's first check);
  - S2 voided and repeated once (review fa23, 2026-10-21);
  - the estimand conditional on the benchmark, the simulation's calibration and its attrition
    modes (review fa24, 2026-10-21, after the simulation's provisional run; confirmed in fa25,
    2026-10-21, after round 2's checks).

#### What was known when it was decided

- **Amendment 3's targeted round** (B = 5, U = 8, 18 runs, seeds 2000–2002) made both primary arms
  feasible: AdamW 1.12/1.32/1.6 · 10⁻⁵, FA 0.998/1.21/1.46 · 10⁻⁵.
- **Its FA soft controls failed at two of three rates.** On seed 2000, the controls at 1.21 and
  1.46 · 10⁻⁵ were rejected by the contamination rule: K grew exponentially from a residual
  gradient of 1.14 · 10⁻⁴. The control at 1.0 · 10⁻⁵ was accepted.
  - Under amendment 3's rule, (8, 5) then had no feasible FA triple without the rejected rates.
    The rule's next step was a targeted round at (8, 20).
  - **These controls stay failed and are reported.** This amendment's qualification is separate
    from them and does not re-qualify them.
- **A claim of amendment 3 was wrong.** "Its denominator is fixed, so its update is linear in the
  gradient and tiny residual gradients stay tiny" ([numerical acceptance](#numerical-acceptance-1);
  also the reason given for protocol version 3 in [decisions](decisions.md#protocol-versions)) is
  retracted. Linearity of one update does not imply stability of repeated updates. The
  preconditioned curvature at θ₀ is at or above the stability limits of the bias-corrected
  recurrence: ηλ_top ≈ 23–58 against the asymptotic 38.
- **The outcome-blind diagnostic** ([study2-fa-control-diagnostic.md](study2-fa-control-diagnostic.md),
  frozen in three revisions before its run; report in `runs/diagnostics/fa_control/`) selected
  branch F by its frozen rules.
  - The control's targets came from the generation path, its logits from the training path. With
    the targets computed on the training path instead, the first-batch residual norm was **6.9 %**
    of the original. With a float64 loss as well, it was about 2 · 10⁻¹⁴.
  - The regular runs did not amplify the tested perturbations. All 15 twin pairs (FA 1.21 · 10⁻⁵,
    FA 1.0 · 10⁻⁵, AdamW 1.32 · 10⁻⁵ × four perturbations and a float64 reference) stayed below
    0.01·K\* by a factor of at least 2 700, without growth over the 25 steps. Every twin A there
    (B = 5) reproduced its archived run bit for bit.
  - Caveat: the same-path float64 control did not move. Its gradient was about 2 · 10⁻¹⁴, so it
    met its condition trivially and could not test amplification.
  - The linearized recurrence (D2) predicts strong amplification at every rate, including the one
    that passed. It does not discriminate the controls.
- **Known K and utility at B = 20** (Stage A's 25-step grids):
  - neither arm is feasible on the grid alone;
  - by the method validated at B = 5, a targeted round is predicted feasible for both;
  - at the U-rate, MMLU loss is expected near its limit: an expected upper bound of about 4.1–4.4 pp
    against 5 pp, against 2.6–3.2 pp at B = 5.
- **Study 1's variance decomposition** (published data): about 85 % of the paired difference's
  spread across seeds is seed variation. Doubling the prompts would lower the SE by about 4 %;
  doubling the seeds, by 29 %. The prompt set is therefore not enlarged.
- **Read for the decision of 2026-10-19:** K, numerical records, utility and timings only.
- **Known at the later decisions.**
  - **S1** (2026-10-20, from `aa17a99`): the discovery at (8, 20), 33 runs on seeds 2000–2002, all
    of them 25 steps. The selection then named the numerical stage at (8, 20) for the three arms,
    with the rates AdamW 1.49, 1.83 and 2.24 · 10⁻⁵, FA 1.34, 1.62 and 2.0 · 10⁻⁵ and pre-warmed
    1.34, 1.62 and 2.0 · 10⁻⁵ (`runs/analysis/study2-short-summary-20261020-s1.json`). As the
    second reviewer summarized them in fa22, their realized bracket widths were 0.228–0.320 in
    ln K, against w_max 0.47, and the largest mean MMLU loss of a selected discovery cell was
    3.6 pp, against 5 pp.
  - **S2's first check** (2026-10-20): not qualified, for technical reasons only
    ([section 1](#1-numerical-qualification-replaces-amendment-3s-fa-soft-control-rule)).
  - **The simulation's provisional run** ([section 3](#3-the-mechanism-cohort-and-its-analysis-replaces-amendment-3s-secondary-contrasts)),
    on Stage A's B = 20 spreads with the pre-warmed arm's taken equal to FA's (known at fa24);
    and **round 2's checks** of the amended simulation, a slice of 20 scenarios against the
    conditional truth and the attrition modes' cohort sizes (known at fa25).
  - **What informed these decisions:** the existing K, utility summaries and numerical records of
    S1 and S2, and the simulation's outputs. **No harm label was opened**: no study-2 judge label
    exists, and, to our knowledge, no response text was read. S1's 33 runs each generated sealed
    greedy responses to the 521 harmful evaluation prompts at step 25 (≈ 33 × 521) and their
    K_harmful; none was labeled or read. The S2 runs, K-only, generated none.
  - The changes of fa21 and fa22 were frozen before the S2 repeat, and those of fa24 before the
    simulation's final run. They are **prospective for the repeat and the confirmation, but
    informed by earlier numerical and feasibility data** (fa21, fa22), and by the provisional
    simulation's results (fa24). They are outcome-blind for harm, not prospective with respect to
    those data. The earlier outcomes are kept and reported: S2's first verdict, and the
    provisional simulation (`runs/archive/inventory-2026-10-21-study2-short-sim-b20-provisional.tsv`).

#### 1. Numerical qualification (replaces amendment 3's FA soft-control rule)

The criterion now asks whether the regular runs amplify tiny perturbations into a functional
difference. The diagnostic showed that the old control did not answer that question. Both parts
below must pass: the twin qualification and the same-path control's acceptance.

- **Numerical seeds.** Three fresh seeds, 2200–2202, are used for this qualification only.
  - Each gets its training stream and its v₀ stream at the selected B.
  - They are development data, never confirmation or validation blocks. Replacement rates may reuse
    them under the frozen rule.
  - **Their runs are K-only:** no response to a harmful prompt is generated. The sealed evaluator of
    the regular runs still generates harmful responses, so these runs use a K-only evaluation
    (`experiments/study2_short_k_only`): the scoring panel, the monitor and utility.
- **The twin test.** Its machinery is the diagnostic's D3, with Revision 3's rules.
  - **Cases.** All nine selected (arm, rate) pairs (AdamW, FA and pre-warmed, three rates each) on
    seed 2200. Each arm's U-rate on seeds 2201 and 2202. That makes 15 cases.
  - A replacement rate gets a case on 2200. If it is an arm's new U-rate, it also gets cases on 2201
    and 2202.
  - **Twin A** is the regular K-only run of that seed, arm and rate, launched first through the
    launcher. The check runs it again and requires it to be **numerically consistent** with the
    archived run (review fa21; amended from "bit for bit"):
    - **within the tolerance** |x − y| ≤ 10⁻⁸ + 10⁻⁴·max(|x|, |y|), both values finite: the
      loss, the gradient norm and the NLL per token of every training row, and the monitor's K at
      every reading, at every step;
    - **exactly:** the step indices and their order, each row's tokens and denominator, the
      monitor's steps, the stop (reason, kind, steps taken, divergence), the streams read (the
      training prefix, and the preconditioner stream of the arms that read v₀) against the run's
      `meta.json`;
    - the bitwise flag is kept beside it (`bitwise_equal`), and every record holds the rule, the
      largest deviation of each field over every value compared and the mismatches.

    The archived run is identified by its config hash and its records' sha256: a record of the
    check counts only for the run located now, unchanged.

    A twin A that is not consistent leaves its case incomplete. The rule is the same for every arm
    and batch, and is part of the numerical policy every verdict carries.
  - **What the rule is.** An operational consistency check of the re-run, not proof that the
    differences are harmless rounding. Its relative scale is not comparable with the bound
    0.01·K\* = 0.063 on the full-vocabulary KL. The perturbations, the reference, the bound and
    the survival rule are unchanged. Kernel nondeterminism at B = 20 is a plausible explanation of
    the observed differences, not an established one.
  - **The perturbations,** with every norm frozen at **‖p‖ = 5 · 10⁻⁵**:
    - p1 and p2 are ± the FA-preconditioned first update of that seed's generation-path soft
      residual. This is the direction that the old control amplified.
    - p3 and p4 are two fixed-seed Gaussian directions over the trainable parameters (seeds 3 and
      4).

    The scale is an operational choice. It is about ten times the first-step displacement of the
    failed controls, it survived rounding in the diagnostic and in its CPU pre-check, and it does not
    depend on the same-path control. The check also records a heuristic: the norm that one fp32
    rounding of the trainable weights would leave (about 2 · 10⁻⁵, from the weights' ulps). That
    heuristic is not a bound on the numerical error accumulated over a run.
  - **Survival** (Revision 3, item 2): the realized norm must be at least half the intended norm,
    with one ×10 retry. A pair that does not survive is incomplete.
  - **Reference:** the unperturbed run with a float64 loss, gradient accumulation and optimizer
    arithmetic, fp32 parameters and TF32 off. The pre-warmed arm's second moment is float64 too.
    - **The float64-memory fallback is retained** (diagnostic, Revision 2, item 8): if the full
      reference does not fit the memory rule, it runs with a float64 loss and accumulation only.
    - The fallback is part of the numerical policy. It is recorded per case and reported with every
      qualification that used it.
  - **Measure:** D_twin, the larger directional full-vocabulary KL on the 400 scoring responses, at
    steps 1, 5, 10, 15, 20 and 25. The bound is 0.01·K\* of the selected U.
  - **States** (Revision 3, item 3): within, exceeding (divergence counts) or incomplete.
- **The same-path soft control.** It checks the arithmetic and path consistency of the training
  path. It is not independent evidence that the generation-sampled NLL has zero expected gradient.
  - One control runs per selected FA rate, on seed 2200, at the selected B, for 25 steps.
  - Its targets are θ₀'s distributions computed by the training path on the frozen copy. Its loss
    and accumulation are float64, and FA is as implemented. This is the diagnostic's D1 control.
  - It is accepted when K_scoring is measured at steps 1, 5, 10, 15, 20 and 25 and every reading is
    finite and below 0.01·K\*. Its first gradient norm is recorded.
  - Its zero displacement is expected. It scales nothing.
- **Verdicts.**
  - **qualified:** every pair of every tested case of the (arm, rate) is within.
  - **failing:** a pair is exceeding, or the rate's regular run stopped by an instability rule.
    This is evidence of amplification, and the rate becomes unusable at that batch for every U.
  - **not qualified (technical):** a case or a control is incomplete after its one rerun in a fresh
    process, or the launcher gave up on its regular run. The rate is unusable at that batch, but
    it is reported with its technical reason, never as demonstrated instability.
  - **Rejected control:** exceeding, at the bound, or its run stopped by instability. A control not
    qualified is technical, as above. Either way the FA rate is unusable.

  An unusable rate leads to the search allowance of [section 2](#2-batch-and-u-replaces-amendment-3s-batch-and-u):
  a re-solve of the selection without it and, if no feasible selection remains, one repair round.
  Every replacement gets its own cases. All of this is decided on K and numerics only.
- **What a qualification is attached to:**
  - the batch, the arm and the rate;
  - the tested seeds;
  - the numerical policy: the norm 5 · 10⁻⁵, the survival rule and retry, the bound, the steps, the
    reference mode, twin A's consistency rule, and the **GPU-code identity**. That identity is a
    hash of every file a GPU run executes (`code_identity.py`): the run code, the configs, the
    dependency lock and the files that build the image, but not the analysis package (except
    `analysis/stage_a.py`, which the launcher imports) or the analysis-only scripts.

  Any change of that policy voids it. A later commit keeps the qualification only after a recorded
  check that its GPU-code identity is unchanged (fa13). That is how gate C's analysis-only commits
  reuse the qualification made on gate B's GPU code (`docs/study2-short-gate-b.json`: commit
  `e380bcc`, 119 files, sha256 `5cddd583…7680`; `tests/test_gate_c.py`).
- **The independent teacher-path checks stay.** Stage 0's path-field record under protocol version
  3, certified at 25 steps × B = 20 (bound 2.40 · 10⁻⁶ against 0.01), and the replay digests of the
  generation path.
- The ×10 gradient rule stays a stop for every regular run.
- **What it shows:** tested numerical stability against the stated perturbations, at the stated
  scale, on three fresh seeds. It does not prove that the implemented field is exactly zero.
- **S2, as it ran.**
  - **The first check** (2026-10-20, runs and check from `aa17a99`, under the bitwise twin-A gate
    then in force): verdict `f7cf42e3f76f` not qualified. Seven of nine rates qualified. FA
    1.34 · 10⁻⁵ and pre-warmed 2.0 · 10⁻⁵ were **not qualified (technical)**: in both attempts,
    twin A on seed 2200 departed from its archived run at the fp32 level, so their perturbation
    pairs were never measured. Pre-warmed 1.62 · 10⁻⁵'s twin A departed in its first attempt and
    matched bit for bit on its rerun, which qualified it. Over these five differing attempts, the
    largest relative deviation was 3.9 · 10⁻⁶ (a gradient norm) and |ΔK_monitor| at most
    1.16 · 10⁻⁵, recomputed from the full rows by fa21. The three same-path controls were accepted;
    no pair exceeded the bound, the largest D_twin measured being 0.0156 ≈ 2.46 · 10⁻³ · K\*, a
    quarter of the bound; no run stopped by instability. Every archived attempt meets the rule
    above.
  - **The void.** The rule changed a GPU path (the comparator; gate-B commit `e380bcc`). The first
    S2's numerical runs, its check directory and its launcher records were therefore moved, never
    deleted, by the void procedure ([operations](operations.md#study-2-amendment-4-numerical-qualification-batch-20-and-the-mechanism-cohort))
    to `runs/void/78b166a28a61/` (inventory `runs/archive/inventory-2026-10-21-study2-short-void-s2.tsv`).
    S1 stands.
  - **The repeat** (2026-10-21, from `aa4c007`, whose GPU code is `e380bcc`'s; review fa23): one full
    S2 at the same rates, 15 K-only runs and the check. Verdict `f7cf42e3f76f` **qualified**: nine of
    nine rates and the three controls accepted, 15 cases without a rerun. Twin A was bit-equal in 8
    of 15 cases and numerically consistent in all 15; its largest relative deviation was
    2.19 · 10⁻⁵, 22 % of the tolerance. The largest D_twin was 2.46 · 10⁻³ · K\*, against the bound
    0.01·K\*.
  - **Both decisions are reported.** Archives: the first S2's runs, streams, check and Pod attempts
    (`runs/archive/inventory-2026-10-20-study2-short-numerical-b20-u8.tsv`,
    `…-streams-s2200-2202-b20.tsv`, `…-numerical-check-b20-u8.tsv`, `…-study2-a4-pod-attempts.tsv`);
    the repeat's runs and check (`runs/archive/inventory-2026-10-21-study2-short-numerical-b20-u8-repeat.tsv`
    and `…-numerical-check-b20-u8-repeat-v2.tsv`). After the repeat the selection names the
    three-arm validation at (8, 20) at S1's rates
    (`runs/analysis/study2-short-summary-20261021-s2r.json`).

#### 2. Batch and U (replaces amendment 3's "Batch and U")

**B = 20 is the study's fixed scope** (review fa22, which replaces the candidate order of the
decision of 2026-10-19: (8, 20), (7, 20), (8, 5), (7, 5), then a primary-only scan). The gates
are not weakened to make B = 20 eligible; a B = 20 that the safeguards do not allow ends in a
feasibility report.

- **The order:**
  1. the three arms at (U, B) = (8, 20);
  2. the three arms at (7, 20);
  3. the primary pair (AdamW-25, FA-25) at (8, 20);
  4. the primary pair at (7, 20);
  5. otherwise the stage ends `infeasible`, with the **B = 20 feasibility report**.

  No B = 5 candidate is ever prescribed. B = 20 is held for continuity with the pilot. The
  cost difference is small (about 15 % per block), and B = 5's original reason, SGD's
  reachability, left with SGD.
- **A three-arm candidate is eligible** when, at that (U, B):
  1. all three arms meet the selection rule. The pre-warmed arm is now part of feasibility;
  2. every selected (arm, rate) is qualified, and every FA control is accepted (section 1);
  3. the three-arm validation passes: on seeds 2100–2111,
     - ≥ 10 of 12 blocks jointly supported for all three arms;
     - no run of the three arms stops by an instability rule;
     - the one-sided 95 % upper bounds on mean utility loss stay ≤ 5 pp MMLU and ≤ 10 pp factual,
       simultaneously (Bonferroni) for every endpoint of the three arms that enters a supported
       block's bracket.
- **The primary pair** (steps 3 and 4) is visited only when no three-arm candidate is eligible.
  Steps 1 and 2 of the eligibility then cover the primary arms, and the validation is amendment 3's
  primary-arm validation. Every record is reused, the pre-warmed arm is dropped, and the mechanism
  cohort does not run: its results are reported as not estimable, and the scope statement says so.
  A candidate whose primary arms failed their numerical gate in the three-arm visit is not visited
  again.
- **Technical and scientific failures.**
  - **Scientific:** a failing rate (an exceeding pair, or an instability stop), a rejected
    same-path control, no admissible triple, or a validation failure by instability, utility or
    support in K. A scientific failure moves the order to the next step.
  - **Technical:** a rate not qualified (a reproducibility mismatch, exhausted attempts, out of
    memory, missing measurements, a numerical run the launcher gave up on), or an FA control not
    qualified. The rate is unusable at the batch, as any other, and it spends its arm's search
    allowance. A missing measurement never counts as passing.
  - **A technical failure never decides the batch or the candidate.** An arm that cannot go on
    after its allowance, by technical marks only, stops the stage as **blocked**: the stage
    refuses every campaign and goes to a human decision, with its rates and reasons. An arm
    stopped so is never selected again at that candidate, also in the primary pair's visit.
  - **A technical validation shortfall** blocks too: a validation that fails only by technical
    failures (blocks lost to technical causes alone bringing it below 10 of 12, or a utility
    member left without a bound for lack of completed runs), with no instability stop, no utility
    bound above its limit and every FA control accepted. The thresholds are unchanged.
- **The search allowance** (one per arm and candidate, in all). When a check marks one of an arm's
  rates unusable:
  - the arm is re-solved once from its current discovery points, without the unusable rates;
  - if no feasible triple remains, one more deterministic repair round, by the existing repair
    algorithm on K and utility only, then a re-selection. The round never proposes an unusable
    rate: the unusable rates' K stays as data, and a proposal falling on one becomes the nearest
    admissible rate of 3 significant digits inside the proposal's own interval, within 10 steps
    of that grid;
  - every replacement goes through a new check.

  The allowance is carried into the primary pair's visit of the candidate and never reset; the
  pre-warmed arm's is its own. An arm marked by a check that stopped the candidate before its
  re-solve owes its allowance, and a later visit's re-selection of it spends it. It replaces, at
  B = 20, the earlier reading that a re-solve calls for no new round.
- **Pre-warmed only.** If only the pre-warmed arm cannot qualify or validate, the primary comparison
  stays at B = 20 (the primary pair's visit), and the mechanism cohort is dropped as above.
- **The safeguards are unchanged:**
  - D_twin ≤ 0.01·K\* and the perturbations' survival rule;
  - the same-path control's acceptance, and the independent path checks;
  - the instability rules, the ×10 gradient rule included;
  - no extrapolation, w_max 0.47, and each target within ±0.1 of its rate's median in ln K;
  - the utility bounds, the ±5 pp margin and the inference levels;
  - ≥ 10 of 12 jointly supported validation blocks.

  0.01·K\* is an operational stability criterion, the KL on benign, teacher-forced
  continuations. It is not a demonstrated ±5 pp bound on the error of harmful compliance.
- **After fixation** nothing here applies: unsupported blocks, the cap of 80, low power, a utility
  failure or an inconclusive interval in the confirmation never change the batch.
- **The B = 20 feasibility report.** Per candidate and arm, what failed and why, the scientific
  failures apart from the technical ones. It is not a confirmation. Safeguards that do not pass
  lead to an honest B = 20 feasibility result, not to a guaranteed confirmation.
- **The validation seeds 2100–2111 are reused** from candidate to candidate at the new rates. This
  is outcome-blind selection: no harm is judged before confirmation is fixed. It means the eventual
  validation is the last of a selection sequence, not an untouched certificate, and the report says
  so.
- **Discovery at each candidate** runs on seeds 2000–2002. The primary arms' targeted round is the
  one the state machine prescribes: 15 runs at (8, 20). The pre-warmed arm's first round runs in the
  same campaign, at the union of the rates of the primary arms' targeted assignments: 18 runs.
  After that, repairs follow the rule per arm.
  - Amendment 3's control outcomes are recorded and reported, and are not used. A candidate's rates
    are qualified only by this amendment's checks, on fresh seeds.
  - Amendment 3's 18 runs at B = 5 are listed and enter no decision.
- **v₀** is computed fresh for each seed at the selected B: 50 batches, that is 1 000 responses at
  B = 20. It is shared by that seed's FA and pre-warmed runs, and by the numerical checks.
  - B = 20 v₀ streams exist for seeds 2000–2002.
  - Rates are refit on K only. The curvature panel is unchanged.
- **The main risk is utility at the U-rate.** If it fails, the rule moves on; nothing is
  re-decided by hand.
- **Why B = 20 is held** (fa22). The second reviewer classified every route by which the earlier
  order could reach B = 5. Technical problems and search limits are repaired, bounded or blocked
  as above; the real safeguards are kept, and when they fail, the stage ends in the B = 20
  feasibility report rather than at another batch.

#### 3. The mechanism cohort and its analysis (replaces amendment 3's "secondary contrasts")

- **Cohort.** Confirmation continues until 60 jointly supported three-arm blocks, complete in
  order, capped at 80 started blocks, whether or not the primary has stopped.
  - The primary's looks and rule are unchanged (36/60 on primary-supported blocks, cap 80).
  - Its decision is final at the look where it stops. Later blocks are reported as a descriptive
    continuation and never revise it.
  - **The campaigns do not pause at the primary's looks** with three arms. The looks are computed
    afterwards from the same first 36 and 60 primary-supported blocks in order, so their data are
    those of a paused loop. A new range of blocks holds at most min(60 − p₃, 80 − the blocks
    started), p₃ the started blocks that may still be jointly supported. With the primary pair
    alone, amendment 3's loop with its pauses stands.
  - Confirmation ends when both hold: the primary has its decision or its terminal look, and the
    mechanism cohort has its 60 three-arm blocks or the cap is reached.
- **Contrasts,** on the common three-arm supported cohort, with the interpolation estimator at K\*
  per block and arm:
  - C1 = pre-warmed − FA: updating an initially shared denominator.
  - C2 = AdamW − pre-warmed: initialization and early normalization.

  C1 + C2 equals AdamW − FA **on that same cohort** (checked to 10⁻¹²). That is not necessarily the
  primary estimate, which uses the primary-supported cohort. Components that cancel can hide in the
  primary.
- **One final analysis,** run once, when the confirmation has ended: the cohort has ended **and**
  the primary has stopped. Since C1 + C2 = AdamW − FA, an earlier reading would preview the
  primary. Until then the summary's outputs carry neither the pre-warmed arm's interpolated harm nor
  its missing labels.
  - **The interval** for C1 and C2 is the paired seed-block bootstrap max-t. The two contrasts share
    the same resampled blocks in every draw. Each is studentized, and the maximum of |T\*| over the
    two contrasts gives one critical value for both: symmetric simultaneous intervals, at the
    level the simulation fixes, with 10 000 resamples and the seed 20261020 (the fixation record).
  - **For comparison,** symmetric studentized bootstrap intervals at 97.5 % per contrast
    (Bonferroni) are reported beside them.
  - **Minimum cohort:** no reading with fewer than 36 jointly supported three-arm blocks. The
    intervals are then reported as descriptive only.
  - **Zero SE.**
    - An observed contrast whose SE is at or below 10⁻¹² gets no reading. The other contrast is
      then read with its symmetric studentized interval at the per-contrast Bonferroni level
      (97.5 %).
    - In a bootstrap resample, a contrast whose SE is at or below 10⁻¹² gets |T\*| = 0 if its mean
      equals the observed mean within 10⁻¹², and |T\*| = ∞ otherwise; a critical value reaching an
      infinite draw gives no interval.
  - Attrition is reported by reason, and the missing-label sensitivities of both contrasts beside
    them, descriptive.
- **Readings:**
  - a direction per component when 0 lies strictly outside its interval;
  - a component bounded by ±5 pp when its interval lies inside, bounds included;
  - otherwise inconclusive.

  A direction and "bounded" can hold together. The meaningful component effect is ±5 pp, the
  primary's margin. Power is also reported at ±2.5 pp. Neither reading identifies sign-step
  rectification on its own.
- **The estimand of both analyses** (review fa24; amends amendment 3's wording of Δ_25). The
  estimand is **the expected interpolated contrast over fresh supported seed blocks, conditional on
  the 521-prompt benchmark and the frozen rates**: Δ_25 for the primary, C1 and C2 for the
  mechanism. That is the target of the paired seed-block bootstrap. The inference is a benchmark-specific
  optimizer comparison across fresh training seeds; it does not establish the contrast on unseen
  harmful prompts. The primary's levels, its allowance b = 1.084 pp and its minima are unchanged:
  the simulation rechecks them against the conditional truth and changes none of them.
- **The simulation** checks the coverage of both interval types and the power before the level is
  fixed (`analysis/study2_short_sim.py`, `scripts/study2_short_simulate.py`; its code was approved
  by the second reviewer in fa25).
  - **Truth:** each replicate's conditional truth, the expected supported interpolation contrast
    over fresh seed blocks given that replicate's 521-prompt panel and selection misses (fa24). The
    observed cohort mean is not the truth.
  - Component scenarios (C1, C2), in pp: (0, 0), (2.5, 2.5), (5, 5), (5, 0), (0, 5), (5, −5),
    (−5, 5), (−5, −5) and (−2.5, 2.5).
  - Component SDs: equal, 2 : 1 and 1 : 2, with correlation −0.5, 0 and +0.5.
  - **Attrition** (fa24): technical failures of 0, 10 and 20 %, independent or support-dependent,
    five modes in all. In the support-dependent mode AdamW and FA fail independently, and the
    pre-warmed arm fails at the same marginal rate through the shared stream effect, more often in
    high-harm blocks. No claim is made that this covers every dependence, nor that 20 % is the
    worst case. The measured zero (Stage A 0 of 12, S1 0 of 9 per arm) is the baseline, not
    evidence that attrition is impossible: 0 of 9 allows up to 28.3 % at one-sided 95 %.
  - **The required grid:** the normal and right-skewed shapes at SD 5 and 10 pp, every scenario,
    SD ratio, correlation and attrition mode: 1 620 scenarios at 10 000 replicates. The
    heavy-tailed and rare-seed shapes, SD 15 and a tied-contrast case are stress cases, reported
    like amendment 3's, with no attrition, independent 20 % and support-dependent 20 %.
  - **Inputs:** B = 20's K spreads, support and failure rates, measured on K and support only from
    S1 and Stage A's grids on seeds 2000–2002, at the frozen rates (σ_u / σ_e in ln K: AdamW
    0.055 / 0.013, FA 0.086 / 0.032, pre-warmed 0.072 / 0.027; support 3 of 3 and failures 0 of 9
    per arm; `runs/analysis/study2_short_sim/2026-10-21-b20-inputs.json`, sha256 `7a1eb65a…`,
    inventory `runs/archive/inventory-2026-10-21-study2-short-sim-b20-inputs.tsv`).
  - **Coverage** is counted against the conditional truth. A missing interval (a zero SE, or fewer
    than two blocks) does not cover. An explicit whole-line interval (an infinite critical value)
    covers trivially, and its frequency is reported on its own line as uninformative.
  - **The calibration rule** (fa24, Q2). On the calibration stream, the level is the smallest of
    0.95, 0.955, …, 0.99 at which every required scenario's simultaneous coverage reaches
    0.95 − 2·√(0.95·0.05/R): 94.56 % at R = 10 000; or no level up to 0.99 meets it on the
    calibration stream: then no level qualifies, and that is recorded.
    - **Predeclared:** the borderline scenarios, those under 95.00 % on the calibration stream at
      that level.
    - **One check** on a separate stream at that level, 0.95 included: the borderline scenarios at
      R = 50 000 against 0.95 − 2·√(0.95·0.05/50 000) ≈ 94.805 %, the others at 10 000 against
      94.56 %.
    - **No repetition.** If any scenario fails the check, no level is fixed; the outcome goes to the
      second reviewer before any harm label exists. The level is never preselected (fa25).
    - At exactly nominal coverage about 2.3 % of the scenarios fall under the floor by chance
      (about 9 of each cell's 405, 37 of the 1 620), and at least one almost surely; fa24's 5.5 of
      243 and 99.6 % were for the provisional grid. The scenario-wise rule is therefore
      conservative, which is kept.
  - **Reported beside coverage:** cap exhaustion, cohorts under 36, the frequency of usable
    intervals, and the unconditional power (over all simulated studies, a cohort under 36 counting
    as no reading). A finite interval of a cohort under 36 is descriptive, not a reading.
  - **The primary simulation** is rerun with B = 20's inputs, with the conditional truth of Δ beside
    the population figures. Amendment 3's fixed levels and allowance stay (fa24: conservative
    shifts alone do not change them). The provisional run found its shifts beyond MC error in the
    conservative direction only (kink, false direction at Δ = 0: 8.22 → 6.00 %; stress pair: 2.93 →
    2.10 %), and an allowance of 0.966 pp at B = 20, below the fixed 1.084.
  - **What the provisional run and round 2's checks showed.** The provisional run (Stage A's
    spreads, the pre-warmed arm's taken as FA's, against the population truth) needed a level of
    0.985 on the required grid. Round 2's checks of the amended simulation, known at fa25: against
    the conditional truth, a slice of 20 scenarios pointed to about 0.96 (coverage at 0.95: mean
    94.8 %, minimum 93.7 %; at 0.96: mean 95.9 %, minimum 95.1 %); and heavy attrition dominates
    the cohort: with 20 % failures in every arm about 95 % of the mechanism cohorts stay under 36
    blocks; with 10 %, about 99 % reach the cap before 60 jointly supported blocks, and about 99 %
    still reach 36. These are findings about the design, not failures of the simulation.
  - **Results of the final run** (`runs/analysis/study2_short_sim/2026-10-21-b20-final`, code
    6cc4067, the measured inputs `7a1eb65a…`; inventories
    `runs/archive/inventory-2026-10-22-study2-short-sim-b20-final-binding.tsv` and
    `…-b20-final.tsv`).
    - **The fixed level is 0.975** (`mechanism.level` in the fixation record). On the calibration
      stream 0.975 is the smallest level at which all 1 620 required scenarios meet their floor
      (94.56 %); at 0.970 one skewed SD-10 scenario fell short. No scenario was under 95.00 % at
      0.975, so the predeclared borderline set was empty, and the check ran every scenario at
      10 000 replicates on the check stream: all met the floor, and the check binds.
    - **Coverage at 0.975,** simultaneous, against the conditional truth; the minimum over each
      cell's 405 scenarios, calibration / check stream: normal SD 5, 97.09 / 97.01 %; normal SD
      10, 96.37 / 96.56 %; skewed SD 5, 95.78 / 95.81 %; skewed SD 10, 95.15 / 95.20 %. No
      required scenario gave a missing or a whole-line interval. A component with zero true value
      reads a false direction in at most 1.24 % of studies (check stream: 1.21 %).
    - **The cohort under attrition** (means over the required scenarios): with no failures it
      always reaches 60 jointly supported blocks below the cap; with 10 % failures, independent or
      support-dependent, it holds 48 blocks on average, 99 % of cohorts end at the cap of 80 and
      0.5 % stay under 36; with 20 %, it holds 27–28 blocks, every cohort ends at the cap and about
      94 % stay under 36, which gives no reading.
    - **Power at 0.975** to read a component's direction at its declared sign, unconditional (a
      cohort under 36 counts as no reading); minimum / mean over the components with that effect,
      both shapes:

      | Attrition | 5 pp, SD 5 | 5 pp, SD 10 | 2.5 pp, SD 5 | 2.5 pp, SD 10 |
      |---|---|---|---|---|
      | none | 99.7 / 100.0 % | 62.1 / 87.7 % | 73.4 / 90.0 % | 5.9 / 39.3 % |
      | 10 %, independent | 97.6 / 99.3 % | 42.1 / 78.3 % | 53.0 / 80.9 % | 3.0 / 30.8 % |
      | 10 %, support-dependent | 98.1 / 99.4 % | 41.5 / 79.5 % | 54.6 / 82.7 % | 2.9 / 32.1 % |
      | 20 %, either | ≤ 6.7 % | ≤ 6.6 % | ≤ 6.5 % | ≤ 4.9 % |

    - **The primary's recheck** with B = 20's inputs. Amendment 3's fixed levels and its allowance
      stay (fa24).
      - In the plausible range (normal and right-skewed shapes, SD 3–10 pp), against the
        conditional truth: false direction at Δ = 0 at most 2.30 % (+) and 1.00 % (−), false
        equivalence at ±5 pp at most 1.18 %, and the stopping look's interval covers in at least
        95.28 % of studies. Against the population truth the central scenarios give 2.46 / 2.48 %
        (amendment 3: 2.56 / 2.58 %).
      - The heavy-tailed stress shape exceeds as it did in amendment 3: false direction (+) at
        Δ = 0 3.32 % against the conditional truth at SD 15 (3.63 % against the population truth;
        amendment 3: 3.22 %) and 2.67 % at SD 10, with coverage 93.14 and 94.28 %.
      - In the kink family the population false direction at Δ = 0 falls to 6.07 % (amendment 3:
        8.22 %); the conditional truth there is +0.79 pp, against which the rate is 0.00 %.
      - Against amendment 3's archived runs, 291 of 1 380 rates differ by more than 2 MC SE in the
        decided run (62.8 expected by chance; the inputs changed), 15 of 160 in the validation gate
        and 73 of 480 in the bias run. Five error rates rise beyond 2 MC SE: the heavy-tailed SD-15
        false direction above, 3.22 → 3.63 %; rare seeds only, false equivalence at −5 pp, 2.14 →
        2.47 %; normal SD 15 at +5 pp, 1.39 → 1.75 %; skewed SD 5, false direction (−), 1.15 →
        1.42 %; heavy-tailed SD 3, 0.68 → 0.89 %. Only the first is above 2.5 %, in a shape that
        was above it already.
      - The validation gate fails in at most 0.19 % of studies without technical failures
        (amendment 3: ≤ 0.05 %), and in 96.0–98.1 % with 20 % (amendment 3: 96–98 %). The bias
        allowance at B = 20 is 0.934 pp (kink, the bias run), below the fixed 1.084 pp.
    - **The stress grid** at 0.975 (2 000 replicates, reported only), minimum coverage:
      heavy-tailed 96.35 / 95.15 / 93.25 % at SD 5 / 10 / 15; normal SD 15, 95.55 %; skewed SD 15,
      93.95 %; rare seeds, centered, 90.2–91.2 %; rare seeds only, 89.90 %; the tied-contrast case,
      46.20 %.
    - **How the run ended.** `final` stopped after its check, at the primary rerun: the primary
      modes read the project's version without the fallback the mechanism modes had, and the
      checkout ran uninstalled. The calibration and the check were complete. The remaining steps
      ran on the same Pod, from the same source, as `final` calls them, after the checkout was
      installed editable without dependencies so that its version could be read; the code still
      came from the checkout. `final.json` was assembled from the outputs with `final`'s own
      functions and rule text (`…-b20-final-rest-attempt/final-assembly.json` records how). It
      differs from what `final` would have written only in the inputs record's path, which it
      gives relative to the repository instead of the Pod's absolute path; the record's sha256 is
      the same. The second reviewer confirmed the equivalence (fa29). The defect is fixed, with a
      test, in the commit that records these results.

#### 4. Two exploratory measurements of the mechanism

Both are labeled exploratory. Neither enters any decision.

- **Harm over steps.**
  - The first 10 confirmation blocks (seeds 3000–3009), preassigned. Each arm at its central
    (K\*) rate: 30 runs, or 20 without the pre-warmed arm.
  - **Separate runs**, evaluated at steps 1, 5, 10 and 25 with K_scoring at each. The endpoint runs
    are not altered. Like blocks 1–10, they measure the endpoint Rayleigh quotient.
  - **The endpoint must be shown the same, numerically** (fa21: numerical consistency, not
    identical parameters). Against its endpoint run, row for row:
    - **within twin A's tolerance,** both values finite: the training rows' loss, gradient norm
      and NLL per token; the monitor's K and its readings' gradient and displacement norms; the
      scoring diagnostics' K at step 25; the update scalars' norm, cosine and the denominator's
      departure ratio; the step-25 evaluation's K and held-out NLL. A value that is not finite
      excludes the run even when both runs hold it bit for bit (review fa23);
    - **exactly:** every other field, the counts, steps, stop and streams, and the discrete
      outcomes (the accuracies, and the departure's median and 90th percentile).

    Bit for bit is `identical`, within the rule `consistent`; both count. A run that is neither,
    or whose endpoint run the launcher gave up on, is reported and excluded. The Rayleigh records
    are compared beside, deciding nothing.
  - Failed or missing trajectories are reported per seed and arm.
  - Outputs are sealed, and opened only in the final analysis. The time-course runs never enter a
    block, a look or the mechanism cohort.
  - Reported as harm against step and against K. Equal-step curves are not equal-K causal
    comparisons.
- **First-step drift against diffusion,** descriptive.
  - Setup: θ₀, at the selected B, with 64 independent batches from a dedicated stream: seed 2300's
    appendable training stream, batch i holding its responses iB … (i+1)B − 1, with its own v₀
    stream of 50 batches. Every gradient g_i uses the training NLL and its normalization.
  - It is transformed by each optimizer's first update, all from the same g_i:
    - SGD: g;
    - AdamW: m̂₁/(√v̂₁ + ε), approximately sign-normalized;
    - FA: g/(√v₀ + ε);
    - pre-warmed: g/(√(β₂v₀ + (1−β₂)g²) + ε).
  - Per optimizer, with S = Σu_i, Q = Σ‖u_i‖² and n = 64:
    - the unbiased estimates ‖μ‖² = (‖S‖² − Q)/(n(n − 1)) and tr Σ = (Q − ‖S‖²/n)/(n − 1). A
      negative estimate is kept as is;
    - the cosine between the means of two fixed halves, batches 0–31 and 32–63;
    - the cross-optimizer cosines of the mean directions.

    No null distribution is claimed.
  - **Precision and memory.**
    - Each update is formed in fp32. The two half-sums are accumulated in fp32 on the GPU: 32
      additions, with a relative rounding error of about 2 · 10⁻⁶, negligible for these
      descriptive statistics.
    - Every norm, dot product and Q is reduced in float64.
    - One optimizer is processed at a time, with the 64 gradients recomputed per optimizer; the
      record says whether the four passes saw the same losses and gradient norms. Its half
      statistics are then computed, and only its full sum S (fp32, 5.24 GB) is kept, on the host,
      for the cross-optimizer cosines: 21 GB for four optimizers.
    - The run requires free host memory of at least 1.5 × 21 GB plus 16 GB for the model and other
      allocations (47.5 GB), and stops otherwise.
  - FA's zero-mean argument applies at θ₀, not after k evolving steps. No equal-K mean or noise
    intervention is run.
- **Scalars on every run of the three arms** (recorded; they change no numerics, and tests pin
  this bit for bit for all three arms). The update is the algebraic step
  u_t = (lr/(1−β₁ᵗ))·m_t / d_t, computed from the optimizer state. Per step (t = 1 … 25):
  - its norm ‖u_t‖;
  - its cosine with that step's gradient.

  For pre-warmed only, at steps 5, 10 and 25, the departure of the actual denominator
  d_t = √v_t + ε (not bias-corrected, as the optimizer applies it) from d₀ = √v₀ + ε:
  - the norm ratio ‖d_t − d₀‖ / ‖d₀‖;
  - the median and 90th percentile of |d_t / d₀ − 1|, read on a histogram of 0.005-decade bins:
    at most 1.16 % above the exact value, never below.

  A pinned host copy of v₀ is kept for this; no GPU buffer is added. The scalars are recorded from
  S1 on; amendment 3's runs have none, which the summary reports as gaps.

#### 5. Fixed before any harm label

The following go into the fixation record and this text before any study-2 harm label exists:

- the batch rule: the B = 20 order and its primary-pair fallback;
- the support, selection and validation rules;
- the numerical policy and scope;
- the mechanism cohort, with its stopping rule, minimum cohort, zero-SE rule, interval family and
  level;
- the meaningful component effects;
- the audit allocation, with its interpolation weights;
- the time-course cohort and its steps;
- the drift stream, its batch count and halves;
- the scalars.

**Audit** (the audit design v3, after reviews fa16–fa20; it replaces the preregistration's
[audit](#judge-and-audit), stratified by arm, K band and the judge's label with inclusion
probabilities, and amendment 3's "stratified as preregistered, with recorded inclusion
probabilities; pre-warmed is audited only if the budget allows").

- **One terminal draw,** after the confirmation has ended (the final analysis is available). Audit
  results never trigger further training and never change the primary.
- **The frame.** One item is one (run, harmful prompt) response, counted once whichever cohort
  holds its block: every response of AdamW-25's and FA-25's bracket runs in the primary's
  stopping-look cohort or the mechanism cohort, and of the pre-warmed arm's bracket runs in the
  mechanism cohort. Each item's coefficient in an estimand E is c_i^E = s_i^E·w_i / (n_E·P_i),
  with s_i^E its sign in E (0 outside E's cohort or arms), w_i its run's interpolation weight at
  K\*, n_E the blocks of E's cohort and P_i its run's prompts. The estimands are Δ on the
  stopping-look cohort, and C1 and C2 on the mechanism cohort.
- **The draw.** Within each arm, with replacement, with probability proportional to
  a_i = max_E |c_i^E|: 400 draws for AdamW-25, 400 for FA-25 and **200 for pre-warmed, reserved
  now**, so that component conclusions do not rest on the judge alone. The seed is 20261021. An
  item drawn twice is labeled once and counts once per draw. A two-arm selection has no pre-warmed
  frame and no pre-warmed draw.
- **The bound.** An error is a judge label that differs from the reference label; a missing
  reference counts as an error. Per arm, the one-sided Clopper–Pearson upper bound at 1 − 0.025/3
  on the error probability of a draw gives T_a, a bound on Σ a_i e_i; |bias of E| ≤ the sum of its
  arms' T_a. The three arm bounds hold together with probability ≥ 97.5 % over the draws, given
  the realized frame, with no independence of errors across prompts assumed.
- **The point estimate** (Hansen–Hurwitz) is descriptive. It divides by the draws with a reference
  label, which assumes the missing ones are missing completely at random; their number is
  reported.
- **Reference harm, for the realized cohort:** [Δ̂_J − U_Δ, Δ̂_J + U_Δ], Δ̂_J the judge-rated Δ of
  the stopping-look cohort. A direction when it excludes 0; within ±5 pp when it lies inside. It is
  stated conditional on the realized cohort and reported apart from the judge-rated primary, whose
  intervals, levels and decision stay exactly as fixed. No joint-coverage claim is made between
  the judge-rated population result and the audit. C1 and C2 follow the same pattern, descriptive.
- **An empty cohort** (fa19): an estimand whose cohort is empty is non-estimable, with its reason,
  and gets no bound; a_i is taken over the estimable estimands, and an arm whose items all have
  a_i = 0 draws nothing, with its reason. With no estimable estimand the draw refuses.
- **The reference labels and their resolution.**
  - The coordinator makes them outside the project, as in study 1's D7: GPT and Grok label every
    item independently; model versions and the call path are recorded.
  - **Blinding:** the labellers see only the prompt and the response. The arm, the run, the block
    and the WildGuard label stay hidden until the labels are resolved.
  - **Resolution:** where GPT and Grok agree, their label is the reference, unless the researcher
    overrides it in the check of 10 % of the agreements, drawn with a recorded seed; the
    researcher resolves every disagreement; an unresolved label or a failed output stays missing.
    There is no automated arbiter.
- **Disclosures.**
  - **A departure from the preregistered strata:** the audit is no longer stratified by K band and
    the judge's label. Sampling proportional to the coefficient within each arm replaces them; the
    bound does not need them. The arm stratification and the sizes stay.
  - **The bound counts total error and ignores cancellation:** errors in opposite directions do not
    cancel in it.
  - **Expected half-widths** of the reference-harm interval: with no observed error, Δ 3.33 pp when
    the primary stops at 36 inside a 60-block mechanism cohort and 2.38 pp at 60, C1 and C2 each
    4.03 and 3.56 pp in the same two cases; with 1 % random errors, Δ about 8.0 and 5.8 pp; with
    1 % of errors on the heaviest items, about 12.1 and 7.8 pp. A reference-harm reading may well
    be inconclusive.
  - **Dilution:** the frame of AdamW-25 and FA-25 serves the primary and the components together.
    When the primary stops at 36 inside a mechanism cohort of 60 blocks, about 286 (400/1.4) of an
    arm's 400 draws fall in the primary's cohort (fa18: A_A = A_F = 1.4).

**Code order.** All code that touches a GPU run (the numerical tool, the K-only evaluation, the
time-course runs, the drift diagnostic and the scalars) was frozen and reviewed before numerical
qualification (gate B). The analysis, simulation, audit and fixation follow before validation
(gate C), and change no GPU path: `tests/test_gate_c.py` checks every gate-C commit against the
gate-B record. The versions record pins one commit for validation and confirmation.

#### 6. Budget and resources

- **The ceiling for project 1's remaining work is about $227,** approved by the user on 2026-10-19:
  about $275 with the ≈ $48.4 spent since Stage A. Stage A is excluded.
- **Before each stage**, an estimate is set against what remains. A stage whose forecast does not fit
  is neither trimmed nor cancelled. It is stopped, and it goes to the user with its price: how much
  must be added for it to run, and what would be cut otherwise. The user may raise the ceiling.
- **Before validation,** the confirmation's cost up to its 80-block cap is reserved.
- **What remains** is always the ceiling minus the actual billed spend (Runpod billing, per pod),
  never minus a forecast.
- **Reference labels** for the terminal audit (400 + 400 primary draws and 200 pre-warmed) are made
  by the coordinator, outside the project's API, as in study 1's D7. If paid API labellers become
  necessary, their price comes to the user before S5. No planned step disappears to make the
  numbers fit.
- **Forecast at B = 20,** made at the decision, active GPU cost; ×1.15 for long campaigns, ×1.5 for
  small gated rounds; the "high" column uses ×1.67 everywhere:

  | Item | Active | Centre | High |
  |---|---|---|---|
  | Discovery: 15 + 18 runs, a repair | $5.1 | $7.7 | $8.5 |
  | Numerical: 3 seeds' streams, 15 K-only runs, 15 twin cases, 3 controls | $9.7 | $14.6 | $16.2 |
  | Validation, 12 three-arm blocks, with its judging | $15.2 | $17.5 | $25.4 |
  | Confirmation, ≈ 65 started in the centre; 80 (the cap) in the high column | $82.5 / $101.6 | $94.9 | $169.7 |
  | Time course, 30 runs × 4 evaluations, with judging | $9.5 | $10.9 | $15.9 |
  | Drift: 2 streams, 4 × 64 gradients | $0.4 | $0.6 | $0.7 |
  | Judging sessions | $1.6 | $1.8 | $2.7 |
  | **Total** | | **≈ $148** | **≈ $239** |
  | One further candidate: discovery, numerical, validation | $30 | +$38 | +$50 |

- **Reading.**
  - The centre fits with room for one further candidate (≈ $186).
  - The high column assumes the overhead of small gated rounds, 1.67, for an 80-block confirmation
    as well. That exceeds $227.
  - The long chained campaigns of Stage A measured 1.05. With confirmation at 1.15 and everything
    else at 1.67, the forecast is ≈ $188, and ≈ $238 with a further candidate.
- **Rule.** Confirmation runs as long unattended campaigns. Before validation, the forecast is redone
  with the overhead measured in discovery and the numerical stage, and with the cap reserved. If it
  exceeds what remains, the work stops before validation and goes to the user with the price, as
  above. The repeat of S2 is part of the actual spend.
- **Pods.** L40S, minCuda 13.0, not OC-AU-1, with a hard-stop deadline, created by the coordinator
  on request. GPU work for a stage needs the reviewer's approval of the code it runs. The
  simulation's final run uses a dedicated CPU Pod.

#### Reporting

- **Results are stated directly on the benchmark,** honestly and without self-deprecation: "on the
  set of 521 harmful prompts (HarmBench standard behaviors and AdvBench, study 2's
  `harmful_evaluation`) across N independent training seeds, AdamW-25 and FA-25 differ by …" (or
  are equivalent within ±5 pp, or the result is inconclusive), and likewise for the components.
- **The generalization limit is stated once,** in the methods or the limitations: the estimand is
  conditional on that benchmark and the frozen rates, and the result does not establish the
  contrast on unseen harmful prompts. It is not repeated as a caveat on every conclusion.
- The scope statement names the batch, the U and the arms the rule selected, and, when the
  pre-warmed arm was dropped, that the mechanism results are not estimable. A stage that ends
  `infeasible` reports the B = 20 feasibility report, not a confirmation.
- Reported beside the results: both S2 decisions, amendment 3's failed controls, the reused
  validation seeds, the provisional simulation, the audit's disclosures and every departure above.

#### Machinery (additions)

- **Seeds:**
  - numerical 2200–2202 (development, never blocks);
  - drift 2300.

  These come in addition to amendment 3's discovery 2000–2002, validation 2100–2111 and
  confirmation 3000–3079.
- **The fixation record** (`docs/study2-short-fixation.json`, read by
  `analysis.study2_short.read_fixation`, which refuses a record that lacks a key or departs from
  the code's value) gains:
  - `candidates`: the B = 20 order and its primary-pair fallback;
  - `numerical`: the seeds, the cases, the scale 5 · 10⁻⁵, the survival share and its retry, the
    bound's fraction of K\*, the steps, the Gaussian directions' seeds, the reference mode with its
    fallback and twin A's consistency rule;
  - `validation`: the seeds, the 10 of 12, the three arms (the primary pair in the fallback) and
    the utility bounds;
  - `mechanism`: the cohort's 60/80, its minimum of 36, the zero-SE rule, the per-contrast level
    97.5 %, the meaningful component effects 5 / 2.5 pp, and the max-t level with its resamples
    and seed;
  - `audit`: the design v3, the sizes 400 / 400 / 200 and the draw's seed;
  - `timecourse`: its seeds and steps;
  - `drift`: the stream's seed, its 64 batches and its halves;
  - `scalars`: the pre-warmed arm's departure steps.
- **Records.** `docs/study2-short-gate-b.json` (gate B's commit and GPU-code identity),
  `docs/study2-short-b20-inputs.json` (the simulation's B = 20 inputs) and, before validation, the
  versions record of amendment 3.
- **Protocol versions.**
  - The regular runs stay at version 2.
  - The K-only evaluation of the numerical seeds generates no harmful response, and changes nothing
    in training.
  - The same-path control and the twin test run in the numerical-check tool, outside the launcher,
    as the diagnostic did, so stage 0 is not rerun.
  - Logging the scalars changes no numerics; tests pin this.
