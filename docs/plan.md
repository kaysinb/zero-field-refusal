# Plan

The study runs in stages. A gate is a check, recorded in advance, that decides whether to continue.
This document says what each stage runs and produces, which gate closes it and what happens on
failure, in words. The criteria themselves are stated only in
[preregistration.md](preregistration.md#gate-thresholds); they bind once committed there before the
corresponding runs. Gate decisions are logged in [decisions.md](decisions.md), with a report in
[reports/](reports/README.md).

Stages 0–5 below are study 1, the pilot. It stopped at Gate 1
([decision](decisions.md#study-decisions)), and its design is not continued. Study 2 follows under
its own [preregistration](study2-preregistration.md) and is planned in [Study 2](#study-2).

## Stage 0 — Setup

Runs and produces:
- Repeat the prior-work search.
- Implement the pipeline with the protocol fixes: a constant loss denominator, shared prompts for
  all optimizers, responses of at most 96 tokens, and the θ₀ − Δθ control.
- Settle the [stage-0 design decisions](architecture.md#design-decisions-for-stage-0).
- Implement the [evidence contract](architecture.md#publication-evidence) and verify a saved
  fixture → CPU table/figure rebuild, including shared-seed pairing and missing/failed records.
  This engineering check does not replace scientific gates or analysis calibration.
- Set the campaign's [retention plan](operations.md#artifact-retention); revisit it before scaling up.
- On θ₀ samples, measure the fraction of responses that end with an end token within 96 tokens.
- Measure the field that token-count normalization creates at θ₀, by the mean-gradient method of
  [invariant 9](architecture.md#zero-field-invariants). Its size decides whether
  [A1](preregistration.md#alternative-explanations) is a test or a diagnostic.
- Measure the discrepancy between the generation path and the training path per execution
  environment and record whether its residual field meets the bound
  ([invariant 8](architecture.md#zero-field-invariants)); measure the step-0 teacher gap and set
  its thresholds before the first training run.
- Run non-confirmatory runs for judge calibration: θ₀ and AdamW at η = 5·10⁻⁵ on 0.5B and 1.5B.
  Their responses enter no analysis, and their prompts are disjoint from the evaluation sets.
- Calibrate the judge on 200 of these responses against reference labels: three model labelers
  (Claude Opus 5, Claude Sonnet 5 and GPT, gpt-6-astra through the Codex CLI; versions corrected
  on 2026-10-12 from the session logs) label the blind sample independently, a fresh Claude Opus
  4.8 arbiter resolves their disagreements, and the user checks every disputed item and 30 random
  unanimous ones (changed from labeling by hand on 2026-10-11;
  [amendment](preregistration.md#amendments)). Claude Fable 5.1 was planned as a labeler, but its
  safety filter stopped it on the harmful texts; Sonnet replaced it.
- Implement the full-analysis simulation and validate the sequential CI level before confirmatory
  stage-2 runs ([requirement](preregistration.md#full-analysis-simulation)).
- Commit the hypotheses, the sign of Δ and the thresholds in
  [preregistration.md](preregistration.md).

Gate: none.

## Stage 1 — Corrected protocol on 0.5B and 1.5B

Runs and produces:
- AdamW at η = 2·10⁻⁵ and 5·10⁻⁵ on both models, and also at 5·10⁻⁶ and 1·10⁻⁵ on 1.5B, 5 seeds
  each, all run whatever the outcomes. Next to them a branch with token-count normalization
  (AdamW 5·10⁻⁵, seeds 0–4, both models), outside the main curves. The Gate 1 rule "the effect
  disappears with the constant denominator" applies to it although A1 is a diagnostic. That rule
  needs a constant-denominator drop below 10 pp, so it is false whenever the effect criterion
  passes; a missing token-count branch then leaves an incomplete diagnostic of the normalization,
  reported as such, not an unmet Gate 1 criterion. A missing calibration result is different:
  the calibration criterion is then not shown, and Gate 1 cannot pass.
- A bridge cell on each model: AdamW at η = 5·10⁻⁵, 5 seeds, trained on the same teacher responses
  truncated to 48 tokens, next to the main 96-token cell. It shows whether the longer responses
  weakened the effect relative to the exploratory runs.
- SGD and frozen-Adam learning-rate sweeps chosen from seed-0 pilots by K, divergence and
  collapse alone, and SGD also at 100 steps × batch 5 on 1.5B, run in any case and used only by
  the Gate 1 coverage rule ([rules](preregistration.md#stage-1-sweeps)).
- The full soft-target control on each model, one seed, right after the SGD sweep at its largest
  stable learning rate: with soft targets the gradient at θ₀ is nearly zero in every batch, so
  the control is informative only at the largest step the study uses (decided on 2026-10-09).
- The θ₀ ± aΔθ line scans.
- Gradient and update statistics at θ₀, in a random projection.
- A provisional K\*, which places the decisive cells of stage 2. Since the amendment of
  2026-10-12 it comes from eligible cells only (seeds 0–4 completed, no divergence, no
  cell-mean collapse), as do the decisive cells and the primary curves; the original analysis
  over every cell is kept and reported beside it
  ([eligible cells](preregistration.md#stage-1-sweeps)).

Gate: [Gate 1 (technical)](preregistration.md#gate-1-technical). It checks that the corrected protocol
increases harmful compliance as refusals fall, that utility stays within its bound, that the judge
agrees with the reference labels, and that SGD reaches enough of AdamW's KL range.

On failure: stage 2 does not start, except where an action is defined.
- If the judge misses the agreement threshold, the rubric or the judge model is revised and the
  calibration repeated.
- If the AdamW effect does not survive the corrected protocol, the project stops with a short report
  of the corrected pilot.
- If SGD cannot cover enough of the KL range, even with more steps and smaller batches, the
  decisive comparison switches to the fallback: equal numbers of steps and equal batch size at the
  largest stable SGD learning rate. This is a comparison at unequal displacement, not a test of
  the equal-KL H1.
- If the effect disappears with the constant denominator, the normalization created it. Stop and
  write a short methods note. This branch is unlikely: most responses reach the length cap, and
  Adam nearly cancels a factor common to a batch ([A1](preregistration.md#alternative-explanations)).

## Stage 2 — Decisive comparison on 1.5B

Runs and produces: AdamW, SGD and Adam with frozen statistics, plus two additional control arms
(SGD with momentum, AdamW with a pre-warmed second moment). When Gate 1's KL-coverage criterion is
met, the two-look sequential design allocates seeds to cells near K\* and estimates Δ(K\*) with its confidence
interval. K\* is fixed before the first look and remains unchanged at the second, and so are the
eligible cells it was selected from. In the fallback, the same stopping rule applies to the
harmful-compliance difference at equal steps and batch size (25 × 20) between the largest stable
SGD and AdamW rates of the main condition, at unequal displacement; it does not test H1. The full analysis must pass the preregistered
simulation check before confirmatory runs.

Gate: [Gate 2 (scientific)](preregistration.md#gate-2-scientific). The definitions and
interpretations of positive, opposite-sign, equivalence and inconclusive outcomes live there.
At the first look an inconclusive result leads to the second look; after the final look it stops
the study. A decided result continues to stage 3 for generalization checks. The same progression
applies to the fallback, with its weaker interpretation.

A confirmed KL prediction alone is no reason to continue: it validates the instrument, not the
hypothesis.

Before look 1:
- K\* is frozen: the `k_star.json` of the 1.5B sweep build, passed to every Gate 2 build
  (`scripts/build_analysis.py --k-star`). It carries the eligible cells, so later instability
  neither excludes seeds nor reselects K\*.
- Each look has a frozen manifest of its runs (run ids and file hashes), created when the look
  closes and hashed into its results. The look-1 build refuses when the experiment holds runs
  outside that manifest, and the look-2 build requires the look-1 manifest.
- The judge qualifies on a check for error that differs between optimizers
  ([preregistration](preregistration.md#judge-check-across-optimizers)). If the stage-1 check is
  undetermined, look-1 runs may train, but look 1 is not analyzed before the check is repeated on
  its runs.
- The full-analysis simulation is re-run with the seed spread measured in stage 1, and includes
  the adaptive refinement of the stage-1 sweep rates if its runs enter the inference and the
  eligibility of cells.

## Stage 3 — Factors and scale

Runs and produces:
- Batch size, learning-rate schedule, LoRA.
- Calibration of the noise-theory predictor.
- Full fine-tuning of 3B.
- A second model family: Llama-3.2-1B or 3B.

Gate: [Gate 3](preregistration.md#gate-3). The calibrated model must predict the harmful rate on
held-out regimes better than both baselines. On failure, the predictive claim is dropped and
reported as a limitation, and stage 4 continues for the main contrast only. The Gate 3 details are
an [open question](preregistration.md#open-questions).

## Stage 4 — Large models

Qwen2.5-7B or Llama-3.1-8B:
- LoRA on one GPU;
- full fine-tuning sharded across GPUs with FSDP2;
- predictions on held-out regimes;
- a test of the noise theory at small KL.

Gate: none.

## Stage 5 — Write-up

Analysis, figures and a preprint; release of the code and the aggregated results, following the
[release checklist](operations.md#release-checklist).
Follow the [publication evidence contract](architecture.md#publication-evidence) for paper assembly.

Gate: none.

## Study 2

Study 2 compares long AdamW with long SGD at equal K_benign, under a common batch, a common
appendable stream prefix and a common step limit. Its rules, gates and outcomes are stated in
the [study-2 preregistration](study2-preregistration.md); this section gives the order of work
and its cost.

**Stage A (feasibility):** code first, then numerics, discovery and validation on fresh seeds:

1. **Code**, each piece reviewed before any GPU use:
   - appendable, prefix-stable teacher streams;
   - soft targets computed in bounded memory;
   - Hessian–vector products for the top curvature;
   - separate monitor and scoring panels;
   - the evaluation scheduler with its stopping rules;
   - trajectory records;
   - the expanded harmful evaluation set.

   Full resume checkpoints, the simulator and the sequential analysis follow after Stage A and
   before confirmation.
2. **Numerics:**
   - curvature at θ₀;
   - the 100-step soft-target screen of the whole SGD rate grid at both batches, on seed 2000;
   - full-length controls at the two candidate rates of each batch, the two largest that pass
     the screen;
   - the path drift/noise bounds.
3. **Discovery:** three seeds, both batches, every run evaluated at all candidate levels
   (5, √35, √40, 7 and 8):
   - SGD-long at the two candidate rates of each batch;
   - AdamW-long at {1, 3, 5, 10}·10⁻⁶;
   - AdamW-25 and frozen Adam-25 on their grid.

   The batch, rates, U and T_max follow the
   [selection rule](study2-preregistration.md#selection-rule). Harm labels stay closed until
   the selection is sealed.
4. **Validation:** twelve seeds at the selected batch and rates, with the instability rule
   frozen.

Planning cost of Stage A, on one L40S at $1.09/h. The estimates rest on the pilot's measured step
times (≈ 1.0 s per SGD step at batch 5, ≈ 1.8 s at batch 20), ≈ 110 s per full evaluation and
≈ 61 s per 1000 teacher responses. The monitor's cost is measured in Stage A.

| Step | Runs | GPU hours | Cost |
|---|---|---|---|
| Curvature at θ₀, both batches (≈ 1.4–1.9 h with the defaults of `scripts/curvature.py`) | 1 | ≈ 1.5–2 | ≈ $2 |
| Soft-target screen: 10 rates × 2 batches × 100 steps | 20 | ≈ 1–2 | ≈ $1–2 |
| Full-length soft-target controls: 2 candidate rates × 2 batches, seed 2000, ≤ 4000 steps | 4 | ≈ 5–12 | ≈ $5–13 |
| Teacher streams of the discovery seeds, generated as needed | — | ≈ 1–4 | ≈ $1–4 |
| SGD-long discovery: 2 rates × 3 seeds × 2 batches, ≤ 4000 steps | ≤ 12 | ≈ 10–30 | ≈ $11–33 |
| AdamW-long discovery: 4 rates × 3 seeds × 2 batches | 24 | ≈ 3–8 | ≈ $3–9 |
| AdamW-25 and frozen Adam-25 grids: 6 rates (up to 8 at batch 5) × 3 seeds × 2 batches × 2 arms | 72–84 | ≈ 5–6 | ≈ $5–7 |
| Validation: 12 seeds, both long arms and AdamW-25 at three rates | ≈ 60 | ≈ 12–25 | ≈ $13–27 |
| Judge for the discovery and validation labels, opened after the selection is sealed | — | ≈ 1–2 | ≈ $1–2 |
| **Stage A** | | **≈ 40–90** | **≈ $40–100** |

**Stage B (fixation):** no GPU beyond measurements already made; the simulation runs on CPU.

**Stage C (confirmation):** two looks on fresh seeds. After look 1 the pods stop until its
analysis is done; one terminal audit follows.

**Budget for the whole study:**

- ≈ $160 centrally, and up to $320 at batch 5;
- a slow branch at batch 20 above $320 needs a separate decision;
- the barrier pilot (≈ $10–20) runs only if the barrier branch fires.

**Hardware:**

- one or two L40S Pods in Stage A, about three in Stage C;
- each Pod's own volume disk plus the bucket, since no data center has both L40S capacity and
  network volumes ([storage](operations.md#storage));
- a deadline watcher on every Pod, and keys only from Runpod secrets.

## Compute

Hardware only; all computation runs on Runpod ([operations](operations.md#containers)). Why these
sizes fit is explained under [memory](architecture.md#memory).

| Stage | Hardware |
|---|---|
| 0–2 | One 48 GB L40S: every 0.5B–1.5B run in fp32, next to a frozen θ₀ copy (≈ 26 GB at 1.5B). The only GPU type measured in stage 0, which the path-field records and teacher-gap thresholds require. A 7–8B judge server also fits. |
| 3 | 80 GB GPUs (H100): runs up to 1.5B and full fine-tuning of 3B. |
| 4 | One 80 GB GPU for LoRA up to 8B; several 80 GB GPUs for full fine-tuning of 7–8B, sharded with FSDP2 ([distributed training](architecture.md#distributed-training)). The precision at this size is a [proposed decision](architecture.md#design-decisions-for-stage-0). |
| 5 | — |
