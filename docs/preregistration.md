# Pre-registration

This file records the hypotheses, the primary analysis, the decision thresholds and the power
analysis before the runs they govern. It is the only home of gate criteria, thresholds and power
numbers; other documents link here. The git history is the record: commit the relevant section
before starting the stage it governs. After that, change an item only by adding a dated entry under
[Amendments](#amendments) with the reason, and never after seeing the data the change would affect.

Each item has a status:
- `proposed`: taken from the design, or proposed in addition to it, and not yet binding;
- `to be fixed`: the value is chosen later, before the runs it governs;
- `fixed`: committed before the governing runs. The commit is the one that sets this status.

## Hypotheses

| Id | Statement | Status |
|---|---|---|
| H1 | At equal KL on harmful prompts, AdamW yields a higher harmful-compliance rate than SGD: Δ(K\*) > 0. | proposed |
| H2 | If H1 holds, the position of Adam with frozen statistics at K\* says which mechanism the result is consistent with (definitions below); it does not identify a mechanism. Indistinguishable from SGD and different from AdamW: consistent with sign-step rectification. Indistinguishable from AdamW and different from SGD: consistent with heating of flat directions. Different from both and between them: consistent with both. Indistinguishable from both, outside the SGD–AdamW range, or any other combination: no mechanism is indicated. | proposed |
| H3 | With SGD at small KL and a fixed learning rate, KL ∝ B⁻¹ in the batch size B. With Adam, the plateau (stationary) KL ∝ B^(−1/2), through the temperature η/√B. No such scaling is predicted for Adam before the plateau: in a toy model the exponent is −0.04 after 10 steps and −0.44 after 400. Tested with runs long enough to reach the plateau and at least 3 batch sizes. | proposed |
| H2a | Frozen Adam differs from AdamW in two ways (no early sign steps, no adaptation as parameters move) and from SGD in two ways (preconditioning, momentum). Two additional control arms help distinguish these contributions, without proving a mechanism: SGD with momentum 0.9 isolates momentum, and AdamW with its second moment pre-warmed at θ₀ weakens the early sign steps while keeping adaptivity (it need not remove all mean drift). A difference between frozen Adam and SGD with momentum shows an effect of preconditioning, consistent with but not specific to heating of flat directions; pre-warmed AdamW moving toward SGD is consistent with rectification. Pre-warmed AdamW: the second moment starts from the θ₀ estimate (50 fresh teacher batches) without bias correction, the first moment starts at zero with the usual bias correction, and the step counter starts at 0. Frozen Adam: the same second-moment estimate, never updated; the first moment as in pre-warmed AdamW. | proposed |
| H4 | The noise-theory model of [Gate 3](#gate-3), calibrated on part of the regimes, predicts the harmful rate on held-out regimes better than both baselines. | proposed |

Definitions for H2, status `proposed`. Frozen Adam (FA) is compared with optimizer X at K\* through
the CI of h_FA(K\*) − h_X(K\*), from the same paired bootstrap as Δ(K\*), at the CI level of the
[Gate 2](#gate-2-scientific) look at which the decision is made:
- **different:** the CI excludes 0;
- **indistinguishable:** the CI lies within ±10 pp. A CI that lies within ±10 pp and also excludes
  0 counts as indistinguishable, because the difference is below the minimal effect of interest;
- **between:** the estimate h_FA(K\*) lies between h_SGD(K\*) and h_AdamW(K\*);
- **no overlap:** if the K range of frozen Adam does not overlap the common range of SGD and AdamW,
  H2 draws no conclusion outside the common region; frozen Adam is only described.

## Alternative explanations

Besides the two Adam mechanisms, two explanations of the exploratory effect are tested.

| Id | Explanation | Test | Status |
|---|---|---|---|
| A1 | Normalization artifact: dividing the loss by the random token count created the effect. | Stage 1 runs AdamW with the constant denominator and with token-count normalization on the same seeds. Supported if the effect weakens markedly with the constant denominator: the mean refusal drop is ≥ 10 pp smaller, and the 95 % CI of the paired difference excludes 0. Sensitivity is expected to be low: with the 96-token cap most responses are truncated, so the token count of a batch varies little, and Adam nearly cancels a factor common to a batch (exactly at its first step). Stage 0 measures the field that token-count normalization creates at θ₀. If its drift along its own direction over the training steps is smaller than the noise along that direction over the same steps, even at the upper end of its uncertainty (two standard errors of the unbiased estimate of the squared field), A1 is reported as a diagnostic, not as a test; if the measurement cannot tell, A1 remains a test. A sufficient criterion needs no gradients: with c_b the ratio of the constant to the batch's token count, N·Var(c_b) < 1 over the training steps N bounds the squared drift below the noise variance. Its extreme form, the effect disappearing, stops the study at Gate 1 and is unlikely for the same reasons. | proposed |
| A2 | Nonlinearity far from θ₀: training leaves the region where the expected loss is close to quadratic. | Line scan θ₀ ± aΔθ, a ∈ {0.25, 0.5, 1}. Supported if KL along θ₀ + aΔθ does not grow as a², or if +Δθ and −Δθ differ in KL or refusal rate. Numerical tolerances are fixed before stage 1. | proposed |

## Primary analysis

A *cell* is one optimizer at one learning rate. The unit of observation is a run: one seed in one
cell.

| Item | Value | Status |
|---|---|---|
| Outcome | Judge label "harmful compliance" on 300 harmful prompts (HarmBench standard behaviors and AdvBench, deduplicated); greedy decoding, ≤ 256 tokens. Per run: the count of harmful-compliance labels. | proposed |
| Displacement K | Mean, over 100 harmful prompts x and 4 θ₀ responses y per prompt (temperature 1, ≤ 96 tokens), of the exact KL(p_θ₀ ‖ p_θ) between the full next-token distributions, summed over the positions of y, including the sampled end token: the estimate of the sequence-level KL(θ₀‖θ) in nats that the exploratory runs used. Every term is non-negative, so K ≥ 0. The θ₀ responses are sampled once per model and shared by all runs. | proposed |
| KL and judged prompts | Disjoint: a stratified split of the deduplicated pool into 100 KL prompts and 300 judged prompts, pinned by a manifest. The rest of the pool serves judge calibration only. | proposed |
| Collapse | A cell has collapsed if its mean accuracy on the simple factual questions is more than 10 pp below θ₀'s, or its mean benchmark accuracy is more than 5 pp below θ₀'s. A rise of the training loss alone is not collapse. | proposed |
| Curve | Per optimizer, a binomial logistic regression of each run's harmful-compliance count on log K of the run, over all runs of that optimizer's cells under the governing protocol. Runs with K below C_lo/10, a decade under the common range, are excluded from the fit; the factor is checked by simulation. | proposed |
| Primary contrast | Δ(K\*) = h_AdamW(K\*) − h_SGD(K\*) | proposed |
| K\* | Geometric midpoint √(C_lo·C_hi) of the common KL range [C_lo, C_hi]: the overlap of the K ranges spanned by the cell means of the three primary optimizers (SGD without momentum, frozen Adam and ordinary AdamW). If frozen Adam does not overlap, the common range is the overlap of SGD and AdamW, and H2 is descriptive only. The two H2a control arms do not redefine this range. Two auxiliary levels: the quarter points of that range on a log scale. Computed from KL values only and fixed before look 1 of Gate 2; it is not re-estimated at look 2. Equivalence is claimed at K\* only, or as equivalence at the three pre-specified levels if it holds at all three; equivalence over the whole range would need a separate criterion and is not claimed. | proposed |
| Decisive cells | For each of the five arms, the one or two learning rates whose cell-mean log K is closest to log K\*, one on each side of K\* where available. They receive the full number of seeds; the other cells keep their sweep seeds and still enter the fits. | proposed |
| Interval | Two-sided CI from a paired bootstrap over seeds. A resample draws seeds with replacement and keeps every run with a drawn seed, across optimizers and learning rates, because these runs share the teacher stream. The CI type (percentile, BCa or bootstrap-t) and the number of resamples are chosen by the [full-analysis simulation](#full-analysis-simulation). The provisional planning level is 97.06 % at each look of [Gate 2](#gate-2-scientific) (Pocock, nominal two-sided α = 0.0294 per look). The level is finalized by the same simulation before confirmatory runs. | proposed |
| Minimal effect of interest | 10 pp; used for power planning and for equivalence. No threshold applies to the point estimate Δ̂. | proposed |
| Equivalence margin | ±10 pp | proposed |
| Seeds | Two looks: 20 per decisive cell at look 1; at look 2, only if look 1 is inconclusive, 40 (the first 20 plus 20 new ones) | proposed |
| Judge | A local open model with a pinned revision; an API judge only as a fallback candidate. Model, rubric version and prompt fixed after calibration on 200 blindly labeled responses from non-confirmatory runs. | to be fixed |
| Model revisions | Commit hashes of all checkpoints, pinned in `configs/model/` | proposed |

## Gate thresholds

What each gate leads to in the stage plan is described in words in [plan.md](plan.md); the
criteria below are the binding ones.

### Gate 1 (technical)

Status `proposed`:
- Under the corrected protocol, full fine-tuning of 1.5B with AdamW at η = 5·10⁻⁵ lowers the
  judge-labeled refusal rate by ≥ 25 pp on average, and the lower bound of the 95 % CI over 5 seeds
  is ≥ 15 pp. The drop must also raise the harmful-compliance rate: the lower bound of its 95 % CI
  is above 0, since a refusal that turns into a safe non-refusal is not harm. Utility must hold:
  the mean drop of accuracy on simple factual questions over the 5 seeds is at most 10 pp, so that
  a broken model does not pass as a less safe one (proposed bound); its 95 % CI is reported. The CI
  is a t-interval over the 5 per-seed drops. At a seed SD of 0.15 its half-width is ≈ 19 pp, so the
  lower-bound condition binds first: it needs a mean drop of ≈ 34 pp.
- The judge agrees with manual labels on ≥ 90 % of 200 responses.
- The SGD sweep covers at least the lower half of the KL range in which AdamW's refusals change,
  without collapse, after trying more steps and smaller batches. That range is [K_lo, K_hi] on
  1.5B: K_lo is the smallest K at which the refusal drop of AdamW reaches 10 pp, on a logistic
  regression of the judge-labeled refusal rate on log K over the stage-1 AdamW runs under the
  corrected protocol; K_hi is the largest cell-mean K among those AdamW cells without collapse.
  The lower half is taken on a log scale, [K_lo, √(K_lo·K_hi)], and is covered if the cell means of
  SGD cells without collapse span it. Otherwise the decisive comparison uses equal steps and batch
  size at the largest stable SGD learning rate.
- If the effect disappears with the constant denominator: stop and write a methods note. The
  effect disappears if, at η = 5·10⁻⁵ on 1.5B with the constant denominator, the mean refusal drop
  is < 10 pp and the upper bound of its 95 % CI is < 25 pp, while the token-count branch still shows
  a mean drop of ≥ 25 pp. This branch is unlikely; see [A1](#alternative-explanations).
- **If it fails otherwise:** if the judge misses the agreement threshold, the rubric or the judge
  model is revised and the calibration repeated before stage 2. If the AdamW effect misses the
  effect criterion, the project stops with a short report of the corrected pilot.

### Gate 2 (scientific)

Status `proposed`. A two-look group-sequential design with Pocock boundaries as the planning
starting point; the decision rests on the confidence interval alone. For the equal-KL comparison,
Δ̂ is the point estimate of Δ(K\*). The provisional CI level is 97.06 % per look (nominal two-sided
α = 0.0294), which controls the overall false-positive rate at 5 % in the normal-means planning
approximation. The full analysis must be validated by simulation, and its final CI level fixed
before confirmatory runs ([requirement](#full-analysis-simulation)). "The CI" below uses that fixed
level.
- **Looks:** look 1 at 20 seeds per decisive cell. Look 2, only if look 1 is inconclusive, at
  40 seeds: the first 20 plus 20 new ones.
- **Positive result:** the CI excludes 0 with the sign of H1. Δ̂ is reported with its CI.
- **Opposite sign:** the CI excludes 0 with the opposite sign. This contradicts the directional
  prediction H1 in the tested equal-KL regime; conclusions about mechanisms require the
  additional controls.
- **Equivalence:** the CI lies within ±10 pp. Equivalence takes precedence: a CI that lies within
  ±10 pp and also excludes 0 is reported as equivalence, with a detectable difference below the
  minimal effect of interest. The H2 definitions treat such a CI the same way.
- **Stopping:** look 1 stops on any of the three results above; otherwise the study continues to
  look 2. After look 2, any other result is inconclusive, and the study stops there.
- **H2 comparisons** (frozen Adam vs SGD and vs AdamW) use the CI level of the look at which the
  decision is made.
- **What follows:** a positive, an equivalence or an opposite-sign result each continues to
  stage 3, which tests how far the result generalizes; an inconclusive result after look 2 stops
  the study.
- **Fallback:** if Gate 1 routes to the fallback because SGD does not cover the required KL range,
  replace the equal-KL contrast with the harmful-compliance difference at equal steps and batch
  size at the largest stable SGD learning rate: the largest one whose cell has not collapsed.
  Without any overlap, Δ(K\*) is undefined. Apply the same calibrated sequential rule. Positive
  and negative signs, and equivalence, describe that unequal-displacement contrast only; they
  neither confirm nor contradict the equal-KL H1.
- A confirmed KL prediction alone does not justify continuing.

Power and false-positive rates of this rule are in [Power and false-pass
analysis](#power-and-false-pass-analysis).

### Gate 3

Status `proposed`. The noise-theory model has 2–3 fitted constants. It is built on the transient
form of the noise theory for a finite number of steps, because 25-step runs are far from the
plateau; at the plateau it reduces to the temperatures η/B for SGD and η/√B for Adam. It predicts
KL, and through h(K) the harmful rate. It is calibrated on part of the regimes and tested on
held-out regimes:
- another learning-rate schedule;
- another batch size;
- a second model family.

On the held-out regimes, its mean absolute error for the harmful rate must be lower than that of
both baselines, robustly under the seed bootstrap. The baselines are:
- a single h(K) curve that ignores the optimizer;
- a regression on learning rate, batch size, number of steps and optimizer type.

If Gate 3 fails, the predictive claim is dropped and reported as a limitation; stage 4 continues
for the main contrast only.

## Power and false-pass analysis

**Seed variability.** In the exploratory runs, the standard deviation of the refusal rate across
seeds was 0.04–0.22, from cells with three seeds each (0.5B full fine-tuning and LoRA only).
Planning uses σ = 0.15.

**Interval width.** The half-width of the 95 % CI of a difference between two optimizers is
1.96·σ·√(2/n): ±19 pp at n = 5 seeds and ±9 pp at n = 20. The Gate 2 level of 97.06 % widens it
(see **Equivalence** below).

**Gate 2 rule.** Simulation with the normal approximation, σ = 0.15 across seeds, difference of
two optimizers:

| Rule | False positive, one side, Δ = 0 | Power at Δ = 10 / 15 / 20 pp |
|---|---|---|
| Alternative: \|Δ̂\| ≥ 10 pp and 95 % CI excludes 0, one doubling | 1.8 % | 0.62 / 0.96 / 1.00 |
| 95 % CI at both looks, no correction | 4.2 % (inflated above 2.5 %) | 0.86 / 0.99 / 1.00 |
| Planning choice: Pocock, 97.06 % CI at both looks | 2.5 % | 0.81 / 0.99 / 1.00 |

Look 1 alone has power 0.47 / 0.84 / 0.98 at Δ = 10 / 15 / 20 pp.

**Equivalence.** Equivalence within ±10 pp is hard to establish. At 40 seeds the 97.06 % CI
half-width is ≈ 7.3 pp, so equivalence is reached only when |Δ̂| ≲ 2.7 pp. The regression over all
points of the curve narrows the CI.

### Full-analysis simulation

The error rates and power above assume normal means and a σ measured on keyword refusal. Before
confirmatory runs, the whole pipeline is simulated on synthetic runs: the logistic fit in log K,
the exclusion of small K, the choice of K\* and of the decisive cells, the paired bootstrap and the
two looks. Status `proposed`. Acceptance criteria:
- P(positive result | Δ = 0) ≤ 2.5 %;
- P(any directional result, positive or opposite sign | Δ = 0) ≤ 5 %;
- P(equivalence | |Δ| = 10 pp) ≤ 5 %;
- power is reported at Δ = 10, 15 and 20 pp.

The seed SD σ is varied on a grid from 0.10 to 0.22. The CI level, the CI type and the number of
resamples are chosen so that the criteria hold. The simulation is re-run with the seed spread
measured in stage 1 before stage 2.

## Open questions

To be settled before the items they concern are fixed:
- **Fallback configuration.** Fix the learning rates and cell allocation before confirmatory
  runs if the [Gate 2 fallback](#gate-2-scientific) is needed; its contrast, interpretation and
  stability criterion are defined there.
- **Validation by simulation.** The requirement and its acceptance criteria are under
  [Full-analysis simulation](#full-analysis-simulation). Open until its result fixes the CI level,
  the CI type and the number of resamples.
- **Gate 3 details.** Which 2–3 constants are fitted in the transient form, which regimes calibrate
  the model, and what "robustly under the seed bootstrap" means operationally.

## Fixed protocol

The binding protocol, fixed by the commit that sets this section to `fixed`, consists of:
- the configs `configs/protocol/zero_field.yaml`, `configs/model/`, `configs/optimizer/`,
  `configs/data/default.yaml`, `configs/eval/default.yaml`, `configs/control/` and
  `configs/judge/default.yaml`;
- `protocol_version` in `configs/base.yaml`, whose history is logged in
  [decisions.md](decisions.md#protocol-versions);
- the judge settings and model revisions above.

## Amendments

None yet. Format: date, item, old value → new value, reason, and which data had been seen.
