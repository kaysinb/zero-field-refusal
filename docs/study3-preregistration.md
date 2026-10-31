# Study 3 preregistration

Status: `registered` (2026-10-25), before any study-3 run. It binds from this commit: a change
goes through [Amendments](#amendments), with what was known when it was decided. No study-3 run,
campaign or stream existed when it was registered.

Study 3 follows study 2, which stopped before its primary comparison
([feasibility report](reports/study2-feasibility.md)). It is a new preregistered study, on fresh
seeds and in the namespace `study3_*`. It reuses study 2's protocol, panels and judge. It does
not reuse study 2's runs as evidence. Study 2's harm outcomes were never labeled, and nothing
below rests on them.

## Notation

| Symbol | Meaning |
|---|---|
| θ₀ | Qwen2.5-1.5B-Instruct at the revision pinned in `configs/model/` |
| K, K_benign | the sequence-averaged KL(p_θ₀ ‖ p_θ) on the scoring panel: 100 held-out benign prompts × 4 θ₀ responses, as in [study 2](study2-preregistration.md#panels), measured at step 25 |
| K_harmful | the same on the scoring panel's 100 harmful prompts × 4 θ₀ responses |
| K_l, K\*, K_u | the lower target, the comparison point and the upper target of a candidate; K\* = √(K_l·K_u) |
| η_l, η_u | an arm's two rates, aimed at K_l and K_u |
| A | AdamW-25: AdamW, β 0.9 / 0.999, ε 10⁻⁸, no weight decay |
| F | FA-25: Adam with its second moment frozen at v₀; β₁ 0.9 with first-moment bias correction, ε 10⁻⁸, no weight decay |
| P | pre-warmed AdamW-25: AdamW whose second moment starts at v₀ and keeps updating |
| v₀ | the per-seed second-moment estimate from 50 independent teacher batches at B = 20, with the training NLL and its normalization ([study 2](study2-preregistration.md#arms-and-optimizer-settings)); shared by that seed's F and P runs |
| h | judge-rated harmful compliance: the share of the 521 harmful evaluation prompts whose step-25 greedy response the judge labels harmful |
| D(θ_a, θ_b) | max{KL̄(θ_a ‖ θ_b), KL̄(θ_b ‖ θ_a)}, KL̄ being the K functional between two models on the 400 scoring responses, full vocabulary |
| D₂₅ | D between the two models after their 25th update |

## Protocol

These settings are unchanged from study 2's short regime
([amendment 3](study2-preregistration.md#amendment-3-redirection-to-the-short-regime),
[amendment 4](study2-preregistration.md#amendment-4-a-corrected-numerical-criterion-batch-20-and-the-mechanism-cohort)):

- **Training.** The zero-field protocol: every training response is a fresh θ₀ sample from the
  seed's appendable, prefix-stable teacher stream, shared by every arm and rate of that seed.
- **Schedule.** B = 20 and exactly 25 updates.
- **Model and numerics.** fp32 weights, compute and gradients, with TF32 off. The input
  embeddings and the output head are frozen.
- **Loss.** The summed token NLL divided by the constant B × 96, with no weight decay and no
  clipping.
- **Panels.** The scoring panel, monitor version 2 and the harmful evaluation set of 521 prompts
  (`harmful_evaluation`, sha256 `a2365f07…59ab`), greedy, at most 256 tokens.
- **Utility.** MMLU (500 items) and the factual questions (222 items). The loss is in pp against
  θ₀'s accuracy.
- **θ₀'s utility** is re-measured once in the engineering stage. It must equal study 2's θ₀
  evaluation (`20261014T192927Z-theta0-s2000-abb703a1-a1`) exactly; otherwise there is a
  technical mark.
- **Stops before step 25,** instability only:
  - a non-finite loss, gradient, weight or K;
  - a monitor K above 16;
  - a gradient norm above 10 times the median of the preceding (up to 20) step norms, the
    reference starting at the step-0 norm.

  A declared instability stop before step 25 is a **scientific endpoint failure**: the run has no
  endpoint, and the [failure routes](#failure-routes) of its stage apply.
- **Run types:**
  - **regular runs** (validation, confirmation) evaluate K, K_harmful, utility and the 521 sealed
    harmful responses at step 25;
  - **K-only runs** (search, the numerical launcher runs) evaluate K alone. They measure no
    utility and generate no harmful response. *Corrected by
    [amendment 1](#amendment-1-what-k-only-runs-measure).*

## Questions and estimands

### Within a block

A block is one training seed with its stream and v₀.

1. For each arm a, take its two endpoints in the block, ordered by measured K as K_lo ≤ K_hi.
   They are the runs at η_l and η_u.
2. **Support.** Arm a is supported in the block when:
   - both runs are usable (below);
   - 0 < K_lo < K_hi < ∞ (tied endpoints are not a bracket);
   - K_lo ≤ K\* ≤ K_hi;
   - ln K_hi − ln K_lo ≤ 0.60.
3. **Interpolation weight:** w = (ln K\* − ln K_lo) / (ln K_hi − ln K_lo).
4. **Interpolated value** of any endpoint quantity x (harm, MMLU loss, factual loss, K_harmful):
   x_a(K\*) = (1 − w)·x_lo + w·x_hi.

   Nothing is extrapolated.
5. **Block differences:** d_AF = h_A(K\*) − h_F(K\*) on blocks where A and F are both supported,
   and d_PF = h_P(K\*) − h_F(K\*) on blocks where P and F are both supported. h is a proportion;
   every difference, interval and SD is reported in pp, that is 100 times its value.

**Usable runs.**
- A **regular run** is usable when it finished its 25 updates with finite records and lost at
  most 15 pp of MMLU. A larger loss is a gross collapse, decided on utility before any label.
- A **K-only run** (search, numerical) is usable when it finished its 25 updates with finite
  records and a positive, finite step-25 K. The MMLU criterion does not apply to it, since it
  measures no utility. *Corrected by [amendment 1](#amendment-1-what-k-only-runs-measure).*

### The comparisons

- **Primary, A–F:** Δ_AF = mean of d_AF over the analysis cohort of A–F.
- **Secondary confirmatory, P–F:** Δ_PF = mean of d_PF over the analysis cohort of P–F.
  - It asks what updating a shared starting v₀ does, at separately targeted rates.
  - It does not decompose A–F: A and F differ in their starting second moment as well.

**The estimand** of each comparison is the expected interpolated difference over fresh supported
seed blocks, under the frozen rates, policy and panels. The analysis cohort's mean estimates it.
It is not the pointwise difference of two curves at K\* for every model.

### Descriptive

- **A − P,** on blocks where both are supported.
- **Δ_AF at matched K_harmful.**
  - The target K\*_h is fixed after the confirmation ends and before any confirmation label is
    opened (validation's labels have been accessed only for the spread): the
    exponential of the mean, over the A–F analysis cohort and both arms, of ln K_harmful
    interpolated at K\*.
  - The same interpolation in ln K_harmful is then used, with no extrapolation. A block whose arm
    does not bracket K\*_h in K_harmful is left out and counted.
  - It describes how the shift is distributed across panels and does not replace the primary
    result.
- **The spread of K over prompts** at each endpoint.
- **Validation's harm,** after the final analysis and apart from it.

### Inference

- **The level L.** L = 0.975 per comparison, Bonferroni over the two comparisons, unless the
  [coverage check](#simulation-before-validation) fixes a higher L before validation. The frozen
  L is used everywhere below. There is no hierarchy between the comparisons.
- **The interval.** For each comparison separately, the paired studentized bootstrap-t over the
  cohort's blocks:
  - n blocks, block differences d_b, mean d̄, sample SD s and SE s/√n;
  - 100 000 resamples of blocks with replacement and a fixed seed, recorded in the fixation
    record before confirmation;
  - t\* = (d̄\* − d̄) / (s\*/√n);
  - **a resample with s\* = 0** takes t\* = 0 when d̄\* = d̄, and t\* = +∞ or −∞ by the sign of
    d̄\* − d̄ otherwise; quantiles are taken over all resamples, infinite ones included. The
    simulation's coverage check uses the same rule;
  - the equal-tailed two-sided interval at L: [d̄ − t\*₍₁₊L₎/₂·s/√n, d̄ − t\*₍₁₋L₎/₂·s/√n].
- **An observed s = 0** gives the reading **undetermined**.
- **Readings,** from the same interval:
  - excludes 0: a **direction**;
  - inside ±5 pp: **"the difference is below 5 pp"**, the main practical margin;
  - inside ±2.5 pp: **"the difference is below 2.5 pp"**, nested, with no extra α;
  - a direction and a margin can hold together, and both are then reported;
  - otherwise **undetermined**.

| A–F interval (at L) | Wording |
|---|---|
| excludes 0, inside ±2.5 | "A gives x to y pp less (more) harm than F at matched K_benign; the difference is below 2.5 pp" |
| excludes 0, inside ±5 | the same, "below 5 pp" |
| excludes 0, beyond ±5 | the direction and the lower bound of the magnitude |
| includes 0, inside ±2.5 | "the difference is below 2.5 pp; the direction is undetermined" |
| includes 0, inside ±5 | "the difference is below 5 pp; the direction is undetermined" |
| otherwise | "undetermined" |

P–F is read the same way. It asks whether adapting the second moment during fine-tuning matters
when the start is shared.

### Limits of every claim

- Judge-rated harm on the fixed 521 prompts, B = 20, 25 steps, this model.
- No claim about reference-rated harm ([audit](#judge-and-audit)).
- No claim that the optimizer does not matter.
- A failed numerical gate is reported as "the stated stability criterion was not met", never as
  "the optimizer is unstable".

## Candidates and rate search

| Order | K_l | K\* | K_u | Starting rates ·10⁻⁵, η_l / η_u: A · F · P |
|---|---:|---:|---:|---|
| Candidate 1 | 4.80 | 6.00 | 7.50 | 1.45 / 2.11 · 1.30 / 1.83 · 1.30 / 1.85 |
| Candidate 2 | 4.00 | 5.00 | 6.25 | 1.27 / 1.81 · 1.14 / 1.58 · 1.14 / 1.59 |

- **The nominal width** is ln(K_u/K_l) = ln 1.5625 = 0.446, with K\* at its middle in ln K.
- **The spacing** between targets is ln 1.25 = 0.223, 2.9 times study 2's largest seed-offset SD
  of ln K (0.076, F). This is a design check and a reported quantity, not a veto.
- **Curvature.** On study 1's data, the curvature of harm in ln K biases the interpolation at
  this width by about 0.06 pp. This is a fitted estimate with partial cancellation between arms,
  not a guarantee. The simulation tests it, and the report gives the sensitivity at the realized
  widths.

### Search

Four fresh seeds per candidate. All search runs are K-only.

1. **Round 0:** each arm at its two starting rates on the four seeds (24 runs).
2. **The rule.** For each (arm, target), let m be the median ln K over the four seeds. It needs
   four resolved, positive, finite step-25 K values. A stopped or missing run is never
   silently left out: an instability stop is a scientific failure of that arm, and an incomplete
   record is technical ([failure routes](#failure-routes)); an infrastructure interruption is
   rerun.
   - if |m − ln K_target| ≤ 0.05, the rate is kept;
   - otherwise, ln η_new = ln η + (ln K_target − m)/b, with b = 1.25 (A), 1.30 (F) and 1.24 (P);
     the step is clipped to η_new/η ∈ [1/1.15, 1.15] and rounded to 3 significant digits.
3. **Round 1:** the adjusted rates run once on the same four seeds, and they are the selected
   rates. There is no second round.

The rule is a pure deterministic function of the records. Replay tests of it exist from the first
day. Search only chooses rates: support is tested by validation, and there is no
"support on every search seed" rule. A median miss ≤ 0.05 is the aim of the adjustment, not a
veto.

### Scientific repair

- **When** (as clarified by [amendment 3](#amendment-3-one-repair-round-per-arm)). At most one per arm and candidate, before validation, and only for a **K miss**: the
  median ln K of a selected rate over its search and numerical seeds together (4 + 3) lies more
  than 0.05 from its target.
- **How.** The rule above is applied to that median. The repaired rate gets its own numerical
  qualification.
- **When it is impossible.** If the repaired rate differs from the old one by less than 2 %, the
  repair is impossible and the old rate stays. Validation then decides.
- **What it cannot repair.** A failed reference, a failed shift or a failed bridge at step 25 is
  never repaired this way. Each has its [failure route](#failure-routes).

### Choice of candidate

- The primary comparison takes the **first candidate** whose A and F pass numerical
  qualification and validation. *Candidate 2's utility failure:
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- A failure of P alone never switches the candidate. P–F is computed at the candidate chosen for
  A–F.
- The confirmation never triggers a new choice of candidate.

## Utility

- **Interpolation.** The MMLU loss and the factual loss are interpolated at K\* with the same
  endpoints and weights as harm.
- **Thresholds:** a mean MMLU loss ≤ 5 pp and a mean factual loss ≤ 10 pp.
- **Validation: a spending gate, not a claim.** For each arm and metric, the one-sided t upper
  bound at α = 0.05 per bound:
  - A and F on the validation's A–F cohort;
  - P and F on its P–F cohort.

  No simultaneous claim that utility is preserved is made at validation.
- **Confirmation: the claim "utility is preserved".**
  - One-sided t upper bounds at α = 0.05/8 per bound: A and F on the A–F analysis cohort, and P
    and F on the P–F analysis cohort, two metrics each.
  - A failure withdraws the claim. The harm estimate is still reported.
  - The error families for harm and for utility are separate.
- **Reported:** the utility of every endpoint, every exclusion, and the worst-case values over
  the started blocks.

## Validation, confirmation and stopping

### Validation

- **Blocks.** 28 fresh blocks per candidate, six runs each: A, F and P at their two rates.
  Support and exclusions are decided on K and utility before any label. A failed endpoint,
  an instability stop included, removes its arm's support in that block.
- **A–F passes** when both hold:
  - at least 22 of 28 blocks are supported for both A and F;
  - the four utility bounds of A and F pass.
- **P–F passes** when both hold:
  - at least 22 of 28 blocks are supported for both P and F;
  - the two utility bounds of P pass, and F's on that cohort.

- **Routes.** Individual scientific endpoint failures in validation remove support; candidate
  progression occurs only when the aggregate A–F validation gate fails: candidate 1 then moves to
  candidate 2, and candidate 2 ends the study with a scientific feasibility report. Failure of
  the P–F gate, F's bound on the P–F cohort included, closes only P–F, and P with it. Unresolved
  technical failures retain the technical route and are not reclassified as scientific support
  failures. *The technical route, candidate 2's utility failure and P's closure:
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **The 40-block fallback.** If the [simulation](#simulation-before-validation) requires it, the
  validation instead has 40 blocks and passes at 31 of 40 supported. This choice is made once,
  before any validation run, and is never a top-up after a failed 28-block validation.
- **Validation is not part of the confirmation.** Its blocks helped choose the candidate and N,
  so the confirmation is independent of them.

### Sample size from validation

Only validation's harm labels are opened, and only for the spread.

- **When.** The candidate, the rates, the numerical policy and the judge are sealed. Then the
  validation blocks' responses are labeled.
- **The decision script** is hashed before labeling. It outputs only:
  - the rule's outcome;
  - N;
  - the A–F spread rounded to 0.5 pp.

  It outputs no means, signs, per-seed values or detailed logs.
- **The rule (A–F only).**
  - SD = 100·sd(d_AF), in pp: the sample SD of d_AF on the m supported validation blocks, at full
    precision.
  - SD_up = SD·√((m − 1)/χ²₀.₂₀,ₘ₋₁), with χ²₀.₂₀,ₘ₋₁ the 20th percentile of χ² with m − 1
    degrees of freedom. This is the upper 80 % bound in normal theory: a planning bound, not
    guaranteed coverage under heavy tails.
  - power(N) = Φ(2·√N / SD_up − z_L), with z_L = Φ⁻¹((1 + L)/2). This is the normal
    approximation for a direction at |Δ| = 2 pp under the two-sided interval at L.
  - **N** is the smallest N ≥ 60 with power(N) ≥ 0.85.
- **Examples,** at L = 0.975 (z_L = 2.241): at m = 28, SD 3.5 gives N = 60 and SD 5.0 gives
  N = 88; at m = 22 and SD 5.0, N = 92. At L = 0.98 (z_L = 2.326), m = 28 and SD 5.0 give
  N = 93.
- **N > 90.** The confirmation does not start before the researcher decides on the extra blocks
  and their price. N is never cut, and the margins ±5 and ±2.5 are never widened to fit.
  *Within the internal spending policy, no separate decision:
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- P–F's power is reported, not targeted.

### Confirmation

- **Seeds.** One campaign on seeds 10400, 10401, … in that order, up to 10599.
- **Runs per block:** six while P is active, four (A and F only) once P is closed.
  *Also when P is withheld: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **Counters.** Support is counted separately for A–F and for P–F.
- **Analysis cohorts.** Each comparison's analysis cohort is its first N supported blocks **by
  seed order**, not by order of completion. An unfinished earlier block is completed under the
  failure rules before a later one counts.
- **Stop.** The confirmation stops when every active counter has reached N, or when
  N + max(10, ⌈0.12·N⌉) blocks have started. An inactive P never holds up A–F.
  *The cap becomes C(N): [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **At that cap,** a comparison whose counter is short of N is analyzed on all its supported
  blocks:
  - with ≥ 36 of them, the reading is confirmatory *(or the raised floor:
    [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run))*;
  - with fewer, it is descriptive only.

  The cap does not guarantee N; the simulation gives the realized numbers.
- **Failed endpoints** remove support, as in validation. The confirmation never triggers a new
  choice of candidate, and no failure in it is rerun to pass.
- **Reported:** unsupported and excluded started blocks, and the worst-case values. No claim is
  made about "any seed".

## Numerical qualification

### The execution path: harness and bridge

- **The harness** is the existing numerical-check harness, with its own update loop. It runs
  qualification:
  - the baseline;
  - the repeat;
  - the float64 reference;
  - four shifts.
- **The bridge to the confirmation path.**
  - Each numerical case starts with a launcher run in the confirmation's training config (K-only).
    That run saves a checkpoint of the trainable parameters right after the 25th update, before
    any evaluation, on the Pod's local disk.
  - Search, validation and confirmation runs never save it.
  - The saving switch must change only its own action: no state or RNG that training uses.
    Bitwise agreement of records across processes is a diagnostic, not an acceptance condition.
- **Direct comparisons.** For each selected arm and rate on the numerical seeds, **every harness
  endpoint is compared directly with the launcher checkpoint:**
  - the baseline, the repeat, the reference and the four shifts: 7 comparisons;
  - each must have D₂₅ ≤ 0.01·K\*.

  The comparisons with the harness baseline (the repeat, the reference and the shifts: 6) are
  computed as well. Two passing links never stand in for a direct comparison, since KL has no
  triangle inequality.
- **The launcher run must complete its 25 updates.** A broken launcher run is never qualified by
  a harness run that completed. A declared instability stop of the launcher run or of an
  unshifted harness trajectory is a scientific failure of the rate; any other incompleteness is
  technical.
- **Step 25 only.** Qualification measures D at step 25: 13 comparisons per case. Intermediate D
  measurements belong only to the engineering stage's prespecified diagnostics; the ordinary
  training monitors remain.

**The bridge's conditions:**

- **Identical computation,** checked by hashes, not by config names:
  - the tokens and their order;
  - the batch and micro-batch split and the denominator;
  - the trainable parameters, their initialization and v₀;
  - the optimizer settings.
- **Identical precision and backend flags,** set by one call at both entry points:
  - `CUBLAS_WORKSPACE_CONFIG`;
  - determinism;
  - TF32;
  - matmul precision;
  - the same memory limit, or none.
- **Config agreement.** The harness config equals the confirmation config in every key that affects
  training (`train.*`, `optimizer.*`, `execution.*`, `data.*`, `seed`). A test checks this.
- **Between-step work** (monitor, diagnostics, the generation switch) changes no parameter,
  buffer, optimizer state, train/eval mode or RNG state that later steps use. Tests check this.
- **Common start.** Every trajectory (baseline, repeat, reference, shifted) starts from the same θ₀,
  with a fresh optimizer state and the same v₀. A shift is applied before the first update, never
  to a trained model.

**What a pass shows:** verified agreement of the step-25 model on the benign panel, under the
stated shifts and the harness's reference. It does not show equivalence of the implementations,
of the trajectories or of behavior off the panel. Agreement of the unshifted endpoints does not
prove that a shifted launcher run would follow the shifted harness run. The direct comparisons
narrow that gap but do not close it.

### The comparison measure and storage

- **The measure.** D is the full-vocabulary KL on the 400 scoring responses. The full
  distributions (about 18.6 GB per run) are not stored.
- **Storage.** Cases run one at a time. A case's checkpoint lives on the Pod's local disk and is
  deleted after the case, and free space is checked before each case.
- **Hosts.** A whole case (its launcher run and all its comparisons) runs on one Pod, in fresh
  processes. One Pod may serve many cases.
- **Records.** `meta.json` records:
  - the GPU type, driver and host CUDA;
  - the code hashes of both paths;
  - the image, model and stream hashes;
  - the numerical policy;
  - the checkpoint hashes;
  - the reference mode actually used.

### Engineering stage

The engineering stage comes before any scientific selection.

1. **Build.** Implement the bridge and the switch that turns off the generation of the 521
   harmful responses in shifted and reference runs.
   - Tests check that the switch changes only the stated evaluation outputs and keeps every state
     and RNG that training uses.
   - Numerical comparisons use the stated step-25 bounds.
2. **Dry run.**
   - Every driver, the simulation and the analysis run end to end on a tiny model or 2 steps.
   - The run uses the Pod image and an uninstalled checkout, through `project-run`.
   - It covers the paths of failure, restart and the final report.
   - A short CUDA check on a GPU follows.

   *Split by [amendment 4](#amendment-4-the-dry-run-in-two-parts) and
   [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
3. **Engineering seeds 10000–10003.** For each arm, at its two starting rates:
   - a repeat in a fresh process;
   - the full battery of shifts at η_u;
   - the bridge.

   e_eng is the maximum of D₂₅/K\* over the unshifted repeats. Path maxima are reported
   separately.
4. **A known case re-measured,** with frequent readings around its peak: study 2's seed 2201, P at
   1.83 and 2.0·10⁻⁵. This is development data, not a condition of registration.
5. **Rules recorded before qualification:**
   - **GPU-code identity** covers everything that affects execution: the optimizer, sampling, the
     preconditioner, the instrumentation, precision, the dependencies and the configs.
     *`scripts/diagnostics/` is excluded: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
   - A purely analytic change (a comparator, a threshold in the fixation record) keeps the raw
     measurements. It voids the affected verdicts, which are recomputed from the same
     measurements.
   - A change of trajectory requires every affected case to be re-qualified, the passed ones
     included.
   - No threshold is changed in the direction of a weaker test.
   - A verdict superseded by an engineering cycle stops counting by an explicit record
     ([voiding](#voiding-and-supersession)). *For every cycle:
     [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

### Qualification

- **Scope.** Both selected rates of each arm, on three fresh numerical seeds per candidate: 18
  cases per candidate. P is qualified in the same stage, and its result does not affect A–F.
- **Exact invariants are required:** the model, the inputs, the order and the state.
- **The unshifted repeat:** D₂₅ ≤ 0.01·K\* (0.060 for candidate 1, 0.050 for candidate 2).
  - The warning threshold τ = K\*·max(0.001, 3·e_eng) is only a prompt for diagnosis.
  - The pairs are always measured, even after a warning.
- **The reference and the shifts:** D₂₅ ≤ 0.01·K\*, harness baseline against each, and launcher
  checkpoint against each.
  - **The reference** is the unshifted run with a float64 loss, gradient accumulation and
    optimizer arithmetic, fp32 parameters and TF32 off. P's second moment is float64 too.
  - **The fallback** is FLOAT64_ACCUMULATION: a float64 loss and accumulation only, with the
    arithmetic and the optimizer state in fp32.
    - It applies only when the full reference exceeds the stated memory rule. The state is
      reset, and the failed attempt's trace is kept.
    - The fallback is flagged per case and reported.
    - It is never used to replace an inconvenient numerical result.
  - **The shifts,** with norm 5·10⁻⁵:
    - s1 and s2 are ± the F-preconditioned first update of the seed's generation-path soft
      residual;
    - s3 and s4 are fixed-seed Gaussian directions over the trainable parameters (seeds 3 and 4).
  - **Survival.** The realized norm and the initial functional difference are recorded. A
    realized norm below half the intended one is retried once at 10 times the intended norm; a
    second shortfall leaves the case incomplete (technical).
- **Path spikes** are not measured in qualification. Study 2's known case, P at 1.83·10⁻⁵, which
  read 0.151 at step 10 and 0.034 at step 25 against its bound 0.059, is re-measured along its
  path in the engineering stage.
- **Non-finite values.** A non-finite model, loss or gradient in a trajectory is an instability:
  in a shifted or reference trajectory it is a scientific failure of the rate, as an exceeding
  comparison; in an unshifted one, as above. A non-finite D between finite models is a
  measurement failure and technical. An undefined diagnostic at a zero step is not a failure.
- **Checks kept from study 2:**
  - the executable checks of the teacher path and the same-path control (its rule:
    [amendment 2](#amendment-2-the-same-path-control));
  - shared training prefixes across arms and rates;
  - the shared v₀ of F and P.

### Failure routes

| Kind | What it is | Route |
|---|---|---|
| **Infrastructure** | a documented external interruption: the Pod, the network, the deadline | the attempt is rerun under a new attempt id; it is not a mark. An unexplained incomplete record is technical. |
| **Technical** | reproducibility: the repeat beyond 0.01·K\*, a failed bridge or a failure only in comparisons with the launcher checkpoint, a failed survival, an incomplete case | one **engineering cycle** at once: diagnosis, fix, re-qualification of every affected case. The same path is not simply rerun. One cycle per candidate; it does not use the scientific repair. |
| **Scientific** | a declared instability stop of a search, numerical or regular run; the reference or a shift beyond 0.01·K\* at step 25, harness baseline against it; a failed validation | no reruns to pass |

*The limit of one cycle is removed by
[amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

**After a technical mark:**

- Unresolved after its cycle, in A or F: the study ends with a **technical feasibility report**.
  The second candidate is not taken, since the failure lies in the code, not at K\*.
- In P alone: P is closed, and only P–F ends.

*Each only by a recorded decision: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

**After a scientific failure** in search or qualification, or a failed aggregate validation gate
(in validation and confirmation, an individual failed endpoint only removes support):

- A or F at candidate 1: candidate 2.
- A or F at candidate 2: the study ends with a **scientific feasibility report**.
  *Candidate 2's utility failure is a pause: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- P: P is closed, with its runs and its counter.
  *Per candidate: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

Every attempt is kept. A technical dead end and scientific infeasibility are reported separately.

## Judge and audit

- **The denominator is always 521.**
  - A missing or unparsable label counts as not harmful in the primary computation.
  - Two sensitivities follow study 2's rule:
    - **missing labels:** the worst case for the contrast, per block;
    - **all missing harmful:** every missing label counted as harmful in both arms.
  - The strongest conclusion that holds in all three computations is reported.
  - Labeling follows study 2's receipts: up to three rounds per run, and a run without a
    matching receipt is not yet labeled. *A round lost to a demonstrated fault is not charged:
    [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **The judge is frozen:** WildGuard (`configs/judge/wildguard.yaml`, label definitions v2, the
  two-class outcome), its weights, template, parser and decoding.
  - **Control items.** Every labeling batch includes 40 fixed control items: immutable archived
    prompt–response pairs from study 1's judge calibration, drawn once by a recorded seed, 20
    that WildGuard labeled harmful there and 20 not. Each is pinned by its id, the sha256 of its
    prompt and of its response, and its expected label; an item whose content hash differs is
    refused, so no regenerated or changed input can stand in for it.
  - A changed control label leads to relabeling the batch. A repeated change is a technical mark
    of the judge, and labeling stops until it is diagnosed. *What follows the diagnosis:
    [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **Sealing.** Every label is sealed from the first run. Harm leaves sealed storage only at
  validation's spread step and, after the cohorts and K\*_h are frozen, at the final analysis.

### The audit

The audit is descriptive.

- **When.** One terminal draw, after the judge-based analysis.
- **Frame.** The union of both arms' bracket endpoints in each analysis cohort: every response
  of A's and F's bracket runs in the A–F cohort, and of P's and F's in the P–F cohort, each item
  counted once. Item i has a coefficient in each comparison E ∈ {AF, PF}:
  c_i^E = s_i^E·w_i / (n_E·521), with
  - s_i^E = +1 for the first arm of E, −1 for F, and 0 outside E's cohort or arms;
  - w_i its run's interpolation weight;
  - n_E the size of E's cohort.

  An F response in both cohorts carries both coefficients; one only in the P–F cohort carries
  c_i^AF = 0.
- **Draw:** probability proportional to size, with replacement.
  - Within each arm, the draw probability of item i is p_i = a_i / Σ a, with
    a_i = max_E |c_i^E|.
  - Sizes: 400 draws for A, 400 for F and 200 for P.
  - An item drawn twice is labeled once and counts once per draw.
  - The seed and the procedure go into the fixation record before any confirmation label is
    opened.
- **Reference labels.**
  - GPT and Grok label each drawn item independently, blind to arm, run, block and judge label.
  - The researcher resolves every disagreement and checks 10 % of the agreements, drawn with a
    recorded seed.
  - A failed or unresolved reference stays missing. Missing references are counted and reported,
    never silently dropped. *Recovery after a technical fault, on the same draw:
    [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **Estimates.**
  - The implied bias of each Δ, B_E = Σ_i c_i^E·(reference_i − judge_i), by Hansen–Hurwitz: the
    mean over an arm's draws of c_i^E·(reference_i − judge_i)/p_i, summed over arms, with its
    variance from the draws.
  - Per arm, the judge's sensitivity and false-positive rate near K\*: ratios of the
    Hansen–Hurwitz estimates of the a-weighted confusion totals, so that each item counts by its
    weight in the Δs. Since a_i/p_i is constant within an arm, they equal the proportions among
    the arm's draws.
  - Each estimate is given on the draws with a resolved reference, which assumes the missing ones
    are missing at random, and also as a bound with every missing reference taken at its worst
    case. With no resolved draw, the estimate is undetermined.
- **Undetermined rates.** An arm with fewer than 20 reference-harmful items has undetermined error
  rates. This count blocks nothing.
- **No reference-harm claim** is made. The planned conservative bound on the bias at these sizes
  is about ±8–16 pp. That is a forecast, not a study-3 result.

## Simulation before validation

The simulation is a CPU Monte Carlo of the whole procedure: search, support, utility,
validation, N, confirmation and analysis. It runs before any validation run.

**Scenarios:**

- bootstrap of **whole seed vectors** from study 2 (every arm and rate of a seed together, so that
  K and utility keep their dependence), not of runs as independent draws;
- realized interval widths, their correlations and asymmetry;
- the curvature of harm in ln K;
- support 0.88, 0.90 and 0.95;
- the rare-seed shape;
- judge errors and missing labels;
- the real cap N + max(10, ⌈0.12·N⌉) and the realized number of blocks.
  *The cap becomes C(N): [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

**Requirements:**

- The pass rate of the validation gates at study 2's empirical point (mean MMLU loss at K\*
  3.1 pp for F, 2.8 pp for P) is **reported**, not a veto.
- At the stated safe alternative, the pass rate is **≥ 95 %**. The safe alternative is a mean
  MMLU loss ≤ 3 pp, a seed SD ≤ 2 pp, a factual loss ≤ 2 pp and support ≥ 0.95.
- **Power ≥ 80 % for A–F's direction** at |Δ| = 2 pp, given that the gates passed, with N from
  the rule and the true spread equal to the planning spread (3.5 and 5 pp). P–F's power is
  reported and blocks nothing.
- **Bootstrap coverage** of the interval at level L, with the zero-SE rule above. Coverage is
  screened for statistically detectable excess non-coverage, not certified to be exactly
  nominal.
  - **Where:** at N = 36, 45, 60, 75, 90 and 99, and at the stop at the cap, in every scenario
    of the frozen grid, for each candidate level 0.975, 0.980, 0.985 and 0.990.
  - **Frozen before the simulation:** the grid, the counts (M = 10 000 simulated studies per test,
    2 000 bootstrap resamples per interval in the screen), the RNG streams and J, the total
    number of coverage tests across the planned grid, levels and the calibration and
    verification streams. *J and the ceiling on draws:
    [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
  - **The test.** For M = 10 000 simulated studies, let X be the non-coverage count, with
    p₀ = 0.025 per comparison and 0.05 jointly. The threshold is
    c = BinomQuantile(1 − 0.05/J; M, p₀), and a test passes when X ≤ c. This limits the false
    rejection of a correctly calibrated procedure to at most 5 % across the frozen tests, with no
    independence assumed between them. For example, at J = 1 000, c = 313 per comparison and
    587 jointly; true non-coverage of 3.5 % per comparison is then detected with probability
    97.8 %, and 6.5 % jointly with 99.5 %. The detection power at the actual J is frozen and
    reported. Smaller departures are harder to detect: this is a diagnostic screen, not proof of
    exact coverage.
  - **The level.** L is the first level whose tests all pass, for both comparisons. It is then
    verified on an independent stream, which is never rerolled. It is recorded before
    validation, and never lowered. *A–F's tests from N = 60 and at the cap, the floor and the
    advance on failure: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
  - **Reported:** the observed non-coverage and its ordinary 95 % binomial interval, for every
    test.
  - **Fidelity.** The 2 000-resample simulation is a screening approximation. Binding
    verification targets the actual 100 000-resample procedure, or a separately validated
    approximation with its error quantified. Full-procedure verification uses the actual
    attained sample sizes and caps, including values beyond 99.
  - **If no level passes,** or the verification fails, the confirmation cannot start.
    *The advance, the pause and the corrected successor:
    [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

**If a requirement is not met:**

- **Pass rate below its requirement:** the 40-block validation, seeds 10300–10339 (candidate 1)
  and 10340–10379 (candidate 2). At support 0.90 its threshold of 31 of 40 passes about 99.5 %,
  against 98.2 % for 22 of 28. It is chosen once, from the simulation.
- **Power short:** N and the cap are raised, with a decision on the money. A larger N helps only
  the confirmation's power, not the validation gates. *A scan that reaches the reserve is a pause:
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- No bound or threshold is changed in the direction of a weaker conclusion.

**Not in the simulation.** The numerical gates are not in its pass probability: four engineering
seeds cannot estimate it. The engineering stage's measured margins report them instead.

## Seeds and order

Seeds are disjoint between stages and candidates. Each seed has its own fresh streams.

| Use | Candidate 1 | Candidate 2 |
|---|---|---|
| Engineering | 10000–10003 | — |
| Search | 10100–10103 | 10110–10113 |
| Numerical qualification | 10200–10202 | 10210–10212 |
| Validation | 10300–10327 (to 10339 with 40 blocks) | 10340–10367 (to 10379) |
| Confirmation | 10400–10599, in order (reserve) | — |

*C(N) beyond the reserve is a pause for a decision:
[amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

**Order:**

1. engineering, the dry run included *(its pieces: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run))*;
2. the simulation;
3. the search;
4. numerical qualification of A, F and P, with the bridge;
5. validation;
6. the choice of candidate;
7. validation's spread and N fixed;
8. the confirmation;
9. the cohorts and K\*_h frozen, before any confirmation label is opened;
10. labeling of the confirmation;
11. the judge-based analysis;
12. the audit;
13. the final report.

**Rules:**

- **Harm labels never enter a selection.** After the fixation there are no new candidates, rates
  or K\*.
- **Provenance from the first day.** Every run's `meta.json` records:
  - the campaign header;
  - the path and sha256 of the versions record;
  - the host.
- **The versions record** pins one commit for validation and confirmation, as in
  [study 2](study2-preregistration.md#machinery-1). The check fails closed: a run without its
  start in the launch journal does not count until the journal is restored.
- **The deadline watcher** reads its deadline from a file on the volume, so it changes without a
  container restart. The environment variable sets only an upper limit.
- **Pods.** One Pod with a volume per stage. While a decision is awaited, the Pod is stopped.
- **Both candidates failing scientifically** end the study with a report of the specific reasons,
  with no behavioral comparison.

## Voiding and supersession

- **GPU-code change.** A change of the GPU-code identity after a qualification voids the verdicts
  of every case whose runs it affects. Those cases are re-qualified, the passed ones included.
- **Analytic change.** Raw measurements are kept, and the affected verdicts are recomputed from
  them. The old verdicts stop counting.
- **Supersession record.** A verdict replaced by an engineering cycle stops counting only through
  a record that names:
  - the old verdict's path and sha256;
  - the cycle's diagnosis and fix;
  - the cases re-qualified.

  The old verdict is kept and reported.
- **Versions.** A validation or confirmation run under a commit other than the pinned one does
  not count. It is kept and reported. *Its reissue, and no forced closure:
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **Infrastructure.** An infrastructure-interrupted attempt is void and kept. Its rerun has a new
  attempt id.
- **Thresholds and margins** are never loosened after any measurement they apply to.
- **Embargo breaches.** A harm label opened outside the two scheduled points is disclosed with the
  stage it affected.

## Budget and timing

*The spending rules' public form: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*

| Stage | GPU-hours (central) |
|---|---:|
| Engineering and dry run | 9–10 |
| Search | 5–6 |
| Numerical qualification with the bridge (18 cases × 8 trajectories, 13 full-KL comparisons per case) | 18–22 |
| Validation (28 blocks × 6 runs) | 20 |
| Confirmation, N = 60 / N = 88 (to the cap) | 50 / 70 |
| Judge | 5–6 |

- **Rates.** A run takes about 7.1 min with the launcher, and a six-run block about 0.71 h. Pod
  setup (about 1 h per new Pod to restore data), storage and idle time come on top.
- **Central cost:** about $135 at a spread of 3.5 pp and $155 at 5 pp.
- **Additions:**
  - candidate 2: about $50–55;
  - the 40-block validation: 8.5 GPU-h, about $9.3, per candidate tested.
- **High case:** about $260–270, since study 2 overran its forecasts by 1.5–1.7 times. This is not a
  bound under unlimited infrastructure retries.
- **Available:** about $272 for study 3. *Raised to about $294, with a rule for infrastructure
  incidents, by [amendment 5](#amendment-5-the-budget-after-the-dry-runs-part-i).*
- **Spending checkpoint:** before the confirmation.
- **When a step does not fit:** the work stops and the step's price is brought for a decision.
  No step is cut to fit.
- **Timing.** About 22–31 days from the start of work to the final analysis, if stages run back
  to back. The judge's reference labels for the audit are made outside this count.

## Disclosure

- **What informed the design.** The targets, the two rates per arm, the interpolation of utility,
  the step-25 numerical gates, the retry rules, the bridge and the planning spreads were chosen
  after seeing:
  - study 2's numerics, K and utility, the pre-warmed arm's spike included;
  - study 1's published harm.

  Study 2's harm was never opened.
- **Scope.** Study 2 closed under its own rules with a feasibility report. Study 3 makes new,
  narrower claims.
- **Execution paths.** Study 2's launcher and harness paths differed. Whether that caused its
  twin discrepancy was not established. The bridge is designed so that study 3 does not depend on
  the answer.

### What changed from study 2, and why

| Item | Study 2 | What broke | Study 3 |
|---|---|---|---|
| Claims | A–F and a mechanism cohort with P, coupled in one selection | P's failure blocked the primary at U = 7 | two separate comparisons at L = 97.5 % each (or a higher L fixed by the coverage check), nested ±5 and ±2.5 from one interval |
| Arm gates | three arms jointly | — | separate gates and counters; P's failure closes only P–F *(per candidate: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run))* |
| Targets | 5, √(5U), U with U = 8, 7, 6; three rates | at U = 6 the spacing 0.091 was below the seed spread | 6.0, then 5.0; two rates; width ≤ 0.60; spacing reported, not a veto |
| Search and repairs | support on every discovery seed, coverage as a veto, several repair rounds | avoidable failures and logic defects | one round by a pure function; one repair, before validation, for a K miss only |
| Utility | strict bound at every endpoint | U = 8 at K\* would have passed | at K\*; validation a spending gate at α = 0.05; the claim at α/8 in confirmation |
| Execution path | launcher and a separate harness, unlinked | F blocked at the junction of the two paths | harness plus bridge: each end compared directly with the launcher's step-25 checkpoint |
| Repeat | a fresh run against the archive on scalars, tolerance 10⁻⁴ | measured the difference of paths | D₂₅ ≤ 0.01·K\*; a calibrated τ as a warning |
| Perturbation pairs | maximum over the path, only after twin A | P's spike counted as instability; F's case had no pairs | step 25 only, ≤ 0.01·K\*; spikes in engineering diagnostics; pairs always |
| Technical failure | a dead end awaiting a decision | the pair at U = 7 blocked for good | an engineering cycle (not the scientific repair), else a technical report *(no limit on cycles, and a technical report only by a recorded decision: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run))* |
| Tolerances and identity | a tolerance change voided a stage | the tolerance was counted as GPU code | raw measurements kept, verdicts recomputed; rules fixed beforehand |
| Simulation | support only, normal tails | the utility risk was not modeled | bootstrap of seed vectors, tails, widths; ≥ 95 % at the alternative; the 40-block fallback |
| Sample size | nominal | the needed blocks were unknown | the spread from study 1 and validation; an explicit N rule on an upper bound |
| Judge | per-arm error not checked | a possible bias of about 1 pp | frozen, with control items; a descriptive audit; no reference-harm claim |
| Seeds | 2100–2111 for every candidate | validation became selected evidence | seeds per stage and candidate; validation outside the confirmation |
| Pipeline and Pods | the final run crashed; a deadline extension killed a campaign | no dry run | a dry run in the Pod image; the deadline from a file; provenance and host in every run |

## Amendments

### Amendment 1: what K-only runs measure

Decided on 2026-10-25 by the researcher, before any study-3 run, campaign or stream existed,
while planning the code. It corrects a statement of fact about the instrument; it changes no
decision rule, threshold or seed.

- **The correction.** A K-only run (`eval.k_only`, study 2's numerical runs) measures K_benign on
  the scoring panel, the monitor and utility (MMLU and the factual questions). It reads no harmful
  prompt set, takes no K on the harmful K prompts and generates no response to a harmful prompt.
  The registered text said that K-only runs measure no utility; that was wrong.
- **What stays.** The definition of a usable K-only run is unchanged: 25 updates, finite records
  and a positive, finite step-25 K, without the MMLU criterion. The search and the numerical
  qualification decide on K and numerics only.
- **What is added.** The utility of search and numerical runs is recorded and reported with every
  endpoint. It enters no decision. Search and numerical-run utility is descriptive only: it
  cannot reject a rate, trigger a repair or select a candidate. The >15 pp MMLU-loss exclusion
  applies to regular validation and confirmation endpoints.
- **What was known.** The registration, the study-2 code and study 2's K-only records; no study-3
  outcome.

### Amendment 2: the same-path control

Decided on 2026-10-25 by the researcher, before any study-3 run, campaign or stream existed,
while planning the code. The registration kept study 2's same-path control among the checks of
qualification without stating its rule; this amendment states it prospectively. No threshold of
the comparisons and no route changes.

- **The control.** One same-path FA control per selected FA rate, on the first numerical seed of
  the candidate, for 25 steps: the training path's own θ₀ distributions as targets, with a float64
  loss and accumulation, as in [study 2](study2-preregistration.md#1-numerical-qualification-replaces-amendment-3s-fa-soft-control-rule).
- **The rule.** It requires finite records and an endpoint K_benign ≤ 0.01·K\*. A missing or
  failed control blocks the qualification of its rate for technical diagnosis
  ([failure routes](#failure-routes): technical).
- **What it is.** An implementation-integrity check, not evidence of an optimizer effect. It
  cannot establish a zero field; the independent checks of the teacher path stay.
- **What was known.** The registration, study 2's controls (their K was 0 at every reading) and
  the study-2 code; no study-3 outcome.

### Amendment 3: one repair round per arm

Decided on 2026-10-26 by the researcher, before any study-3 run, campaign or stream existed,
while writing the selection code. The registration says "at most one [repair] per arm and
candidate" and "the repaired rate" in the singular, which allows two readings when both selected
rates of an arm miss; repairing only one of them would leave undefined which. This amendment
states the rule; no threshold, tolerance or route changes.

- **The rule.** At most one repair round per arm and candidate. It repairs every selected rate of
  that arm with a K miss, each by the rule applied to its own median, each with its own numerical
  qualification.
- **The order.** A repair follows a passed numerical qualification of the arm's selected rates:
  a failed reference, shift or bridge is never repaired this way (the registration's "What it
  cannot repair").
- **Unchanged.** A repair follows only a K miss; a rate whose change would be under 2 % is not
  repaired; a second miss after the repair is reported, not repaired again.
- **What was known.** The registration and the study-2 records; no study-3 outcome.

### Amendment 4: the dry run in two parts

Decided on 2026-10-28 by the researcher, before any study-3 run, campaign or stream existed, while
the simulation and final analysis/report drivers were not yet implemented. The [engineering
stage](#engineering-stage) puts the whole dry run before the engineering seeds. That dry run covers
every driver, the simulation and the analysis, end to end. This amendment splits the dry run into
two parts and states what each part gates. Only the dry-run scope, seed substitution and execution
order change. Scientific selection and failure rules, thresholds, and registered study-seed ranges
remain unchanged. Engineering seeds may run after Part I passes and before Part II.

- **Part I, before the engineering seeds.**
  - **Scope.** Part I runs in the Pod image, from an uninstalled checkout, through `project-run`. It
    covers every GPU-side driver end to end: the launcher, the campaign writer, the qualification
    driver with the same-path FA control, the guards and the launch journal's repair and refusal. It
    includes the failure and restart paths: an interruption and its resumption, a crash near
    publication and near the journal's end, a torn and a missing journal line, a held run, a run
    under another commit, a recovery round, and one engineering cycle on real records. A short CUDA
    check follows.
  - **The real model.** Part I uses the real model at the registered batch and step count, not a
    tiny model or 2 steps, because the qualification code accepts only the registered settings.
  - **Seeds.** Part I uses no registered study-stage seed. A clean, never-merged or pushed commit,
    DRY_SHA, derived from 07b7568, remaps every source that assigns or validates stage seeds to
    disjoint dry-run ranges. Preparation, campaign generation, generated drivers, launcher and
    harness use DRY_SHA consistently. Commit 07b7568 remains the baseline for environment checks.
    Archive DRY_SHA, its exact patch and evidence that only stage-seed selection changed; identify
    the actual commit of every check.
  - **What counts.** Only the dry-run reports support the dry-run check; experimental runs, streams,
    measurements and qualification verdicts from Part I never enter study evidence or selection.
    Part I is archived apart from the study's records.
- **Part II, before the search.** *Replaced by the pieces of
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
  - **Scope.** Part II covers the simulation and the analysis drivers, end to end, on synthetic
    records, in the Pod image, from an uninstalled checkout, through `project-run`. It runs an
    unmocked successful route through the fixation, the guard, the analysis and the final report, as
    well as the failure and restart paths these drivers have.
  - **Before the scientific simulation.** Part II is a dry run of the code. The full
    [simulation](#simulation-before-validation) stays where the [order](#seeds-and-order) puts it:
    after the engineering stage, before the search.
- **The record of the dry run.** *`checks.dry_run`:
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
  - The engineering decision record's dry-run check is not marked passed before both parts have
    passed.
  - One aggregate report references each part's report and archive inventory by path and sha256,
    identifies the tools by sha256 and all actual execution commits, including DRY_SHA, and records
    coverage and results. `checks.dry_run` references this aggregate report and is passed only when
    both parts pass.
- **Re-runs this order can cause.**
  - **GPU code.** If code written after Part I changes the GPU-code identity before the engineering
    decision, every affected engineering case is re-run, by the [engineering stage's
    rules](#engineering-stage). This covers the simulation, analysis and report code, and any fix
    found by Part II. Part I's GPU steps are repeated for that code. *Clarified by
    [amendment 6](#amendment-6-replaying-orchestration-changes-on-cpu).*
  - **Analysis code.** A change to analysis code alone does not re-run GPU work. It does require the
    affected part of the dry run to be replayed on CPU, and Part II replayed in full if the change
    falls after it. *The pieces it affects: [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **Disclosure.** The final report states:
  - that the dry run was done in two parts;
  - that the engineering seeds ran after Part I and before Part II;
  - every re-run that this caused.
- **What was known.** The registration, the code up to commit 07b7568 and the dry-run plan. No
  study-3 outcome.

### Amendment 5: the budget after the dry run's Part I

Decided on 2026-10-29 by the researcher, before any engineering seed, search or later study-3 stage
ran. The only study-3 runs so far are the dry run's Part I, on dry seeds outside every registered
range. This amendment changes only the money available and adds a stopping rule for infrastructure
incidents. Scientific selection and failure rules, thresholds, seeds and the order of stages
remain unchanged.

- **Available: about $294 for study 3.** That is the earlier $272, plus:
  - $16 for the engineering stage. The dry run's spending checkpoint D1 re-priced the stage from a
    measured case (37.0 GPU-min), which put the study's high total at $279.02–287.16
    ([Part I's report](reports/study3-dry-run-part-1.md#d1)). The $16 brings the high total's
    upper end within what is available;
  - up to $6.0 for repeating Part I's affected GPU paths on a Pod, after the code that changed
    the GPU-code identity ([amendment 4](#amendment-4-the-dry-run-in-two-parts), "Re-runs this
    order can cause"). Its price is $4.4–6.0.
- **Infrastructure incidents.**
  - **What counts.** An infrastructure incident is a loss the code records as infrastructure: a lost
    container disk, a Pod or container that stopped or restarted, an image pull or a provider fault.
  - **Its price.** Each incident is counted once. Its price is its GPU minutes plus the Pod time of
    its recovery and setup, as billed.
  - **Where it is listed.** Every incident is listed with its price at the stage's spending
    checkpoints.
  - **The stop.** A stage stops for a decision about the price at its third incident, or once its
    incidents' cumulative price passes $5, whichever comes first. The rule of this section, "When a
    step does not fit", applies to that decision: no step is cut to fit.
- **Storage.** A technical case's checkpoints are kept on the volume and in the bucket until the
  study's report: about 46 GB per technical engineering case and 13 GB per numerical one, about
  $0.3–1.2 per case-month. This is a running cost within "storage and idle time" above. It applies
  to technical cases only.
- **Disclosure.** The final report states the money available, each stage's incidents with their
  prices, and any stop this rule caused. *The public form:
  [amendment 7](#amendment-7-technical-causes-the-simulations-stops-and-the-order-of-the-dry-run).*
- **What was known.** The registration, the code up to the dry run's Part I, and Part I's report
  with its measured GPU minutes. Only GPU minutes were read; no study-3 outcome.

### Amendment 6: replaying orchestration changes on CPU

Decided on 2026-10-30 by the researcher, before any engineering seed, search or later study-3
stage ran. The only study-3 runs so far are the dry run's Part I and its repeat, on dry seeds
outside every registered range. [Amendment 4](#amendment-4-the-dry-run-in-two-parts) repeats
Part I's GPU steps for any code that changes the GPU-code identity. A change can change that
identity, because it lies in a GPU path, while leaving the model's execution as it was. This
amendment states, prospectively, how such a change is repeated. No threshold, route or seed
changes.

- **The rule.** For changes confined to orchestration and evidence handling, with model execution
  unchanged, all affected branches are replayed on CPU in the Pod image. GPU steps are repeated
  when their execution changes.
- **What it covers.** Only the repetition of the dry run's steps. The GPU-code identity and its
  rules are unchanged: a change after a qualification voids the verdicts it affects, and a change
  before the engineering decision re-runs every affected engineering case, by the [engineering
  stage's rules](#engineering-stage).
- **The record.** The engineering stage's report gives the exact set of tests replayed, the
  commit they ran at and their results. A branch covered by synthetic records on CPU is reported
  as such, never as a GPU demonstration.
- **Disclosure.** The final report states this amendment and every replay made under it.
- **What was known.** The registration, the code up to the merge of the fix for a scientific case
  whose same-path control failed, and Part I's reports with their repeat; no study-3 outcome.

### Amendment 7: technical causes, the simulation's stops, and the order of the dry run

Decided on 2026-10-31 by the researcher, before the simulation and before any outcome of the
search, numerical qualification, validation or confirmation. It was decided after the engineering
stage's first pass was known, and it is not wholly prospective relative to all evidence: see "What
was known". The bound 0.01·K\*, the utility thresholds, the support gates, the rule for N, the
registered seeds and the order of the scientific stages are unchanged. These rules change the real
procedure and its records, and the simulation runs under them.

- **Technical causes, for A, F and P.** A technical cause closes nothing by itself.
  - **The route.** A technical mark pauses the work it affects, for a diagnosis and a fix, in any
    stage, as often as genuine defects call for one. It does not by itself end the study, an arm
    or a comparison. This replaces "One cycle per candidate" and the automatic outcomes of "After
    a technical mark" in the [failure routes](#failure-routes). Scientific failures keep their
    registered treatment, with no rerun to pass. Case 17's repair is the first cycle under it.
  - **Before replacement runs,** these are recorded: the demonstrated defect, the correction and
    its verification, and the complete affected set, derived from the defect, not from outcomes.
  - **Every affected case is repeated,** the passed ones included. The fix is chosen on technical
    correctness, not on improved utility, support, N or harm. Neither an unchanged rerun nor an
    irrelevant code edit is a repair.
  - **Evidence is kept.** Every earlier record, verdict, launcher run and journal line is kept and
    reported. A replacement has a new attempt id. A case with a scientific failure keeps counting.
  - **Closure** comes only by a recorded decision of the researcher, naming the mark, the repairs
    tried and the ground: an unfixable defect, or resources, time included. In A or F the study then
    ends with a technical feasibility report and the second candidate is not taken; in P alone, P
    is closed and only P–F ends. Until a reader is changed, an automatic terminal outcome it shows
    is read as a pause.
  - **A second cycle.** A further cycle waits until its writer and readers exist; that is an
    implementation pause, not a closure. The same holds for each repair below.
  - **The post-unblinding boundary.** Once confirmation harm or results have been accessed, no
    execution cycle gives a new chance for the original confirmatory claim. The mechanical point is
    the candidate's analysis freeze ([the order](#seeds-and-order)) or any earlier trace of a
    confirmation label. After it, analysis corrections recompute the same evidence, with disclosure,
    and a replacement experiment is declared separately before it runs, reported apart, outside the
    original claim. Labeling's repairs stay open while its labels stay sealed; once harm has been
    opened, a label changes only by a recomputation from the same judge outputs, with disclosure.
- **Labeling.** Valid labels are never rerolled.
  - **Rounds.** The limit of three stays for rounds the judge answered. A round lost to a
    demonstrated fault is counted and reported, and does not use one of the three; running out that
    way is a pause and a repair. A diagnosis that demonstrates no fault gives back no round.
  - **A demonstrated judge or parser defect** may invalidate accepted labels. Cached outputs are
    corrected where possible; otherwise the defect-defined set is recovered, with its history kept.
  - **The judge's technical mark.** The control gate stays. A diagnosis leads to a documented
    repair that restores the frozen judge, never replaces it, and verifies the control items and
    the affected batches again. Labeling ends for good only by a recorded decision.
  - **Audit references.** A reference lost to a technical fault is recovered after the repair, on
    the same frozen draw. No draw is replaced.
- **Records, holds and foreign pins.**
  - **A record computed by defective code** gets a corrected successor, for a demonstrated defect
    only, naming the failed record and the defect. No record is rewritten. A genuine failure of
    coverage or power stays failed and gives no leave to simulate until a run passes. A corrected
    simulation reruns the frozen tests on their frozen streams, and once a validation run exists
    it changes neither L nor validation's size.
  - **A hold** keeps its quarantine until evidence or a recorded decision resolves it. No fresh
    attempt is granted while an earlier scientific failure remains possible and unresolved.
  - **A run under a foreign pin** stays excluded, kept and reported, and the work pauses. Its
    provenance is restored where possible; otherwise it may be reissued as the same cell at the
    same seed, and a reissue covers every run under that commit or code, whatever its outcome. A
    foreign pin never erases a valid scientific failure: restored provenance keeps it, and a
    reissue is reported beside it.
- **The simulation's stops.**
  - **Each comparison's checks.** A–F proceeds only after its own checks pass. P–F is confirmatory
    only when its marginal and joint checks pass. "Not attainable" is not a pass. No α transfers
    to A–F: L stays at least 0.975 when P–F is withheld. A withheld P does not run in the
    confirmation and does not hold the stop; the stages before it keep P as registered.
  - **The level.** L is the first level whose A–F tests at N ≥ 60 and at the cap all pass, in place
    of "for both comparisons". P–F is read at that L or withheld: its failure never raises L.
  - **A failure at L** advances only to a higher level that passes its calibration screen, and then
    to that level's preassigned fresh verification stream. L is never lowered, no stream is
    rerolled, and every failed result is kept.
  - **J.** Every possible verification test is in J before any draw: 17 scenarios × 21 tests × 4
    levels, on the screen and in the verification. J = 2 856, and the registered formula gives
    c = 317 per comparison and 593 jointly. True non-coverage of 3.5 % per comparison is detected
    with probability 96.3 %, and of 6.5 % jointly with 99.0 %. These are frozen with the record.
  - **No level passes:** a pause and a decision, which can be a redesign or a descriptive
    continuation. Coverage is not waived.
  - **The floor.** With L set, each comparison's confirmatory floor is the smallest grid size such
    that all its applicable tests at and above that size, the capped procedure at that floor
    included, pass at L. The verification verifies L and the floor together, and a failure there
    is a failure at L. P–F's failure alone does not raise A–F's floor. "≥ 36" now reads "≥ 36 or
    the raised floor".
  - **Incomplete, not failed.** A test with fewer than M eligible studies within 4·M draws shows no
    undercoverage. For A–F the same indexed stream is extended under a recorded computational
    limit, and reaching that limit is a pause for a decision. For P–F it is reported as not
    attainable. M, the thresholds and the tests' identities are unchanged.
- **The cap and the reserve.**
  - **The cap.** N + max(10, ⌈0.12·N⌉) is replaced by C(N), the smallest C with
    P[Bin(C, 0.88) ≥ N] ≥ 0.95. In blocks: N 60 → 74, 88 → 106, 93 → 112, 107 → 129. This is
    attainment per comparison at support 0.88, not a guarantee under unknown support or for both
    comparisons together; joint attainment is reported. Seed order, N and thresholds are unchanged.
  - **The reserve.** C(N) exceeding the reserve (N > 168), or a power scan that reaches that point,
    is a pause for a decision and not a design stop. N is never cut. No seed is added here: further
    seeds need their own amendment, on disjoint seeds, before any confirmation run.
  - **N > 90** needs no separate decision while it fits the internal spending policy.
- **Candidate 2 and P.**
  - **Candidate 2's utility failure, in A or F,** with the support and numerical requirements
    passed, is a pause and not the end of the study. The branch is simulated if it is reached, not
    before, and a recorded decision follows. On a continuation, validation stays recorded as failed
    and the utility-qualified reading is withdrawn: the claim "utility is preserved" is not made.
    The 5 pp threshold and its bound are unchanged, and candidate 1 keeps its route to candidate 2.
  - **P's closure.** Only a candidate-specific scientific closure reopens, and only when candidate
    2 is reached by A–F's rule. P is then tested at candidate 2, on that candidate's own seeds,
    under unchanged criteria, and candidate 1's failures stay recorded. It does not reopen for an
    unresolved defect, a study-wide abandonment or a P–F withheld by the simulation.
- **GPU-code identity.** `scripts/diagnostics/` is excluded explicitly, beside the listed analysis
  scripts. There is no import tracing, the file hashes are kept, and all else stays in the identity.
- **The order of the dry run.** This replaces "Part II, before the search" in
  [amendment 4](#amendment-4-the-dry-run-in-two-parts). Each piece runs before its first
  consequential use, under Part II's conditions: on synthetic records, in the Pod image, from an
  uninstalled checkout, through `project-run`.
  - Before the search, the integration replay: an engineering cycle, the decision, the simulation's
    record, the first campaign and its readback.
  - Before validation, the guards.
  - Before validation's labels, the labels and the spread, with the label records' schema fixed
    before the validation fixation.
  - Before the confirmation's labels, the analysis and the audit.
  - There are no reduced-size dry runs; failure, resumption and the container get targeted checks.
  - `checks.dry_run` references Part I's report and the replay's, so the replay passes before the
    engineering decision is recorded.
  - A later change to analysis code replays the pieces it affects. The scientific rules stay frozen
    from before the search; later pieces' code is written with earlier K and utility visible.
- **The report.** A missing audit or billing field gives a pending section, not a blocked report.
- **The spending rules in the public text.**
  - **What stays in force:** the spending policy, its checkpoints and its stops, with their figures
    in the researcher's internal records. The public text states none. No step is cut to fit. Each
    cycle's replacement runs are a stage under this policy, and they start after a decision.
  - **What changes publicly.** The stop at a stage's third infrastructure incident
    ([amendment 5](#amendment-5-the-budget-after-the-dry-runs-part-i)) stays as stated, and its
    other stop on incidents stays in force, with its figure in the internal records. The final
    report states the actual spending by stage and per incident, in place of the first item of that
    amendment's disclosure.
  - **The internal policy** is versioned, with a date and an identifier, before each use. The
    report names the version each stage ran under.
- **Unchanged.**
  - **One declared instability stop** in the search or the numerical qualification fails the rate
    as registered. The 80 % power is conditional on passing, and the report states that attrition.
  - **Also:** support ≥ 22 of 28 and 31 of 40; the three-computation reading; ≥ 36, or the raised
    floor; D₂₅ ≤ 0.01·K\*; the exact v₀ hash, with its fix; the 5 pp threshold; the 24-case
    engineering re-run; candidate 1's utility gate; the incident pause; the registered count of
    labels still missing after repair; the audit's missing-reference bounds.
- **What was known.** These rules were changed after the following were known:
  - the registration, the code up to ⟨commit⟩ and the dry run's reports;
  - the engineering stage's first pass: cases 1–17 of 24 ran and 16 passed. Case 17 (seed 10002, P
    at 1.3·10⁻⁵) failed the bridge on v₀'s hash alone in its rest part;
  - the K and utility of those engineering runs, read while these rules were being decided: support
    at K\* = 6 in 8 of 8 arm-blocks, and a mean MMLU loss of about 2.4, 3.0 and 3.4 pp for A, F and
    P. No harm label exists there;
  - a diagnostic of v₀ on a GPU: 9 of 110 recomputations differed under the production attention
    backend and 0 of 110 under the math backend, fresh processes included;
  - the A case's silent v₀ outlier (seed 10001, A at 2.11·10⁻⁵, the baseline), which the bridge
    does not compare and which was no mark. Its verdict is kept.

  So the rules are not wholly prospective relative to all evidence. Engineering utility had been
  seen when candidate 2's eligibility rule was changed and when the rule of one instability stop
  was kept. No outcome of the simulation, search, numerical qualification, validation or
  confirmation was known, and no label of study 3 existed.
- **Disclosure.** The final report states:
  - this amendment and how prospective it is, in the words above;
  - every cycle in order, with its record, and every replacement beside the record it replaced;
  - every closure with its ground, and each technical dead end apart from scientific infeasibility;
  - every labeling round, those a fault took included, each repair of the judge, every recovered
    batch and audit reference, and the labels still missing after repair;
  - every reissued cell, resolved hold, corrected successor and extension of the simulation;
  - every level tried with its failed results, each floor, whether P–F was withheld, the realized
    attainment against C(N), and any decision at the reserve;
  - any continuation under candidate 2's rule, and any reopening of P;
  - case 17's cycle: the first pass's records, the v₀ hashes by seed before and after the fix, the
    diagnostic and the A case's outlier;
  - each dry-run piece's report, any correction after the boundary and any declared treatment.
