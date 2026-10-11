# Plan

The study runs in stages. A gate is a check, recorded in advance, that decides whether to continue.
This document says what each stage runs and produces, which gate closes it and what happens on
failure, in words. The criteria themselves are stated only in
[preregistration.md](preregistration.md#gate-thresholds); they bind once committed there before the
corresponding runs. Gate decisions are logged in [decisions.md](decisions.md), with a report in
[reports/](reports/README.md).

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
  (Claude Opus, Claude Fable, GPT) label the blind sample independently, an arbiter resolves
  their disagreements, and the user checks every disputed item and 30 random unanimous ones
  (changed from labeling by hand on 2026-10-11; [amendment](preregistration.md#amendments)).
- Implement the full-analysis simulation and validate the sequential CI level before confirmatory
  stage-2 runs ([requirement](preregistration.md#full-analysis-simulation)).
- Commit the hypotheses, the sign of Δ and the thresholds in
  [preregistration.md](preregistration.md).

Gate: none.

## Stage 1 — Corrected protocol on 0.5B and 1.5B

Runs and produces:
- AdamW at two learning rates × 5 seeds, next to a branch with token-count normalization.
- A bridge cell on each model: AdamW at η = 5·10⁻⁵, 5 seeds, trained on the same teacher responses
  truncated to 48 tokens, next to the main 96-token cell. It shows whether the longer responses
  weakened the effect relative to the exploratory runs.
- An SGD learning-rate sweep.
- The full soft-target control on each model, one seed, right after the SGD sweep at its largest
  stable learning rate: with soft targets the gradient at θ₀ is nearly zero in every batch, so
  the control is informative only at the largest step the study uses (decided on 2026-10-09).
- The θ₀ ± aΔθ line scans.
- Gradient and update statistics at θ₀, in a random projection.
- A provisional K\*, which places the decisive cells of stage 2.

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
interval. K\* is fixed before the first look and remains unchanged at the second. In the fallback,
the same stopping rule applies to the harmful-compliance difference at equal steps and batch size,
at unequal displacement; it does not test H1. The full analysis must pass the preregistered
simulation check before confirmatory runs.

Gate: [Gate 2 (scientific)](preregistration.md#gate-2-scientific). The definitions and
interpretations of positive, opposite-sign, equivalence and inconclusive outcomes live there.
At the first look an inconclusive result leads to the second look; after the final look it stops
the study. A decided result continues to stage 3 for generalization checks. The same progression
applies to the fallback, with its weaker interpretation.

A confirmed KL prediction alone is no reason to continue: it validates the instrument, not the
hypothesis.

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

## Compute

Hardware only; all computation runs on Runpod ([operations](operations.md#containers)). Why these
sizes fit is explained under [memory](architecture.md#memory).

| Stage | Hardware |
|---|---|
| 0–2 | One 48 GB L40S: every 0.5B–1.5B run in fp32, next to a frozen θ₀ copy (≈ 26 GB at 1.5B). The only GPU type measured in stage 0, which the path-field records and teacher-gap thresholds require. A 7–8B judge server also fits. |
| 3 | 80 GB GPUs (H100): runs up to 1.5B and full fine-tuning of 3B. |
| 4 | One 80 GB GPU for LoRA up to 8B; several 80 GB GPUs for full fine-tuning of 7–8B, sharded with FSDP2 ([distributed training](architecture.md#distributed-training)). The precision at this size is a [proposed decision](architecture.md#design-decisions-for-stage-0). |
| 5 | — |
