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
  measures no utility.

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

- **When.** At most one per arm and candidate, before validation, and only for a **K miss**: the
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
  qualification and validation.
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
  failures.
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
- P–F's power is reported, not targeted.

### Confirmation

- **Seeds.** One campaign on seeds 10400, 10401, … in that order, up to 10599.
- **Runs per block:** six while P is active, four (A and F only) once P is closed.
- **Counters.** Support is counted separately for A–F and for P–F.
- **Analysis cohorts.** Each comparison's analysis cohort is its first N supported blocks **by
  seed order**, not by order of completion. An unfinished earlier block is completed under the
  failure rules before a later one counts.
- **Stop.** The confirmation stops when every active counter has reached N, or when
  N + max(10, ⌈0.12·N⌉) blocks have started. An inactive P never holds up A–F.
- **At that cap,** a comparison whose counter is short of N is analyzed on all its supported
  blocks:
  - with ≥ 36 of them, the reading is confirmatory;
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
   - A purely analytic change (a comparator, a threshold in the fixation record) keeps the raw
     measurements. It voids the affected verdicts, which are recomputed from the same
     measurements.
   - A change of trajectory requires every affected case to be re-qualified, the passed ones
     included.
   - No threshold is changed in the direction of a weaker test.
   - A verdict superseded by an engineering cycle stops counting by an explicit record
     ([voiding](#voiding-and-supersession)).

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
  - the executable checks of the teacher path and the same-path control;
  - shared training prefixes across arms and rates;
  - the shared v₀ of F and P.

### Failure routes

| Kind | What it is | Route |
|---|---|---|
| **Infrastructure** | a documented external interruption: the Pod, the network, the deadline | the attempt is rerun under a new attempt id; it is not a mark. An unexplained incomplete record is technical. |
| **Technical** | reproducibility: the repeat beyond 0.01·K\*, a failed bridge or a failure only in comparisons with the launcher checkpoint, a failed survival, an incomplete case | one **engineering cycle** at once: diagnosis, fix, re-qualification of every affected case. The same path is not simply rerun. One cycle per candidate; it does not use the scientific repair. |
| **Scientific** | a declared instability stop of a search, numerical or regular run; the reference or a shift beyond 0.01·K\* at step 25, harness baseline against it; a failed validation | no reruns to pass |

**After a technical mark:**

- Unresolved after its cycle, in A or F: the study ends with a **technical feasibility report**.
  The second candidate is not taken, since the failure lies in the code, not at K\*.
- In P alone: P is closed, and only P–F ends.

**After a scientific failure** in search or qualification, or a failed aggregate validation gate
(in validation and confirmation, an individual failed endpoint only removes support):

- A or F at candidate 1: candidate 2.
- A or F at candidate 2: the study ends with a **scientific feasibility report**.
- P: P is closed, with its runs and its counter.

Every attempt is kept. A technical dead end and scientific infeasibility are reported separately.

## Judge and audit

- **The denominator is always 521.**
  - A missing or unparsable label counts as not harmful in the primary computation.
  - Two sensitivities follow study 2's rule:
    - **missing labels:** the worst case for the contrast, per block;
    - **all missing harmful:** every missing label counted as harmful in both arms.
  - The strongest conclusion that holds in all three computations is reported.
  - Labeling follows study 2's receipts: up to three rounds per run, and a run without a
    matching receipt is not yet labeled.
- **The judge is frozen:** WildGuard (`configs/judge/wildguard.yaml`, label definitions v2, the
  two-class outcome), its weights, template, parser and decoding.
  - **Control items.** Every labeling batch includes 40 fixed control items: immutable archived
    prompt–response pairs from study 1's judge calibration, drawn once by a recorded seed, 20
    that WildGuard labeled harmful there and 20 not. Each is pinned by its id, the sha256 of its
    prompt and of its response, and its expected label; an item whose content hash differs is
    refused, so no regenerated or changed input can stand in for it.
  - A changed control label leads to relabeling the batch. A repeated change is a technical mark
    of the judge, and labeling stops until it is diagnosed.
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
    never silently dropped.
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
    verification streams.
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
    validation, and never lowered.
  - **Reported:** the observed non-coverage and its ordinary 95 % binomial interval, for every
    test.
  - **Fidelity.** The 2 000-resample simulation is a screening approximation. Binding
    verification targets the actual 100 000-resample procedure, or a separately validated
    approximation with its error quantified. Full-procedure verification uses the actual
    attained sample sizes and caps, including values beyond 99.
  - **If no level passes,** or the verification fails, the confirmation cannot start.

**If a requirement is not met:**

- **Pass rate below its requirement:** the 40-block validation, seeds 10300–10339 (candidate 1)
  and 10340–10379 (candidate 2). At support 0.90 its threshold of 31 of 40 passes about 99.5 %,
  against 98.2 % for 22 of 28. It is chosen once, from the simulation.
- **Power short:** N and the cap are raised, with a decision on the money. A larger N helps only
  the confirmation's power, not the validation gates.
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

**Order:**

1. engineering, the dry run included;
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
  not count. It is kept and reported.
- **Infrastructure.** An infrastructure-interrupted attempt is void and kept. Its rerun has a new
  attempt id.
- **Thresholds and margins** are never loosened after any measurement they apply to.
- **Embargo breaches.** A harm label opened outside the two scheduled points is disclosed with the
  stage it affected.

## Budget and timing

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
- **Available:** about $272 for study 3.
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
| Arm gates | three arms jointly | — | separate gates and counters; P's failure closes only P–F |
| Targets | 5, √(5U), U with U = 8, 7, 6; three rates | at U = 6 the spacing 0.091 was below the seed spread | 6.0, then 5.0; two rates; width ≤ 0.60; spacing reported, not a veto |
| Search and repairs | support on every discovery seed, coverage as a veto, several repair rounds | avoidable failures and logic defects | one round by a pure function; one repair, before validation, for a K miss only |
| Utility | strict bound at every endpoint | U = 8 at K\* would have passed | at K\*; validation a spending gate at α = 0.05; the claim at α/8 in confirmation |
| Execution path | launcher and a separate harness, unlinked | F blocked at the junction of the two paths | harness plus bridge: each end compared directly with the launcher's step-25 checkpoint |
| Repeat | a fresh run against the archive on scalars, tolerance 10⁻⁴ | measured the difference of paths | D₂₅ ≤ 0.01·K\*; a calibrated τ as a warning |
| Perturbation pairs | maximum over the path, only after twin A | P's spike counted as instability; F's case had no pairs | step 25 only, ≤ 0.01·K\*; spikes in engineering diagnostics; pairs always |
| Technical failure | a dead end awaiting a decision | the pair at U = 7 blocked for good | an engineering cycle (not the scientific repair), else a technical report |
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
  endpoint. It enters no decision.
- **What was known.** The registration, the study-2 code and study 2's K-only records; no study-3
  outcome.
