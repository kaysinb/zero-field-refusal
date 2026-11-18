# Study 5: the training answers along the refusal direction

Status: `registered` (2026-11-18). This analysis was planned after the harm of every run below was known.
Its predictors are computed from θ₀ and the training data alone. Its rules, test family, thresholds
and directions of test are fixed here before any forward pass. It is a bounded retrospective check,
and its readings say "retrospectively associated with harm" or "did not establish", never "predicts"
or "rules out".

## Question

The question is whether some streams simply hold answers that overlap more with the refusal
direction and push the model the other way. What is already reported bears on this only in part:

- An order effect holds the multiset of answers fixed. This rules out an unordered statistic of the
  data as the complete explanation, but not interactions of content, position and optimizer.
- Study 5's registered first-order score p_s (docs/study5-preregistration.md), one
  parameter-space predictor, ranks the 39 supported Qwen streams by harm at a Spearman correlation
  of 0.02. That does not exclude activation-space predictors.
- The stream's share of the variance of h\* across streams and orders (an intraclass correlation)
  is small in Qwen (0.06) and large in Llama (0.61). A small share does not exclude order-dependent
  effects of particular answers.

This analysis adds a refusal direction in the residual stream (difference of means, after Arditi et
al. 2024), the projection of the training answers on it, and an order score that weights each
batch's projection by Adam's position weights. How each outcome would be read:

- A specific association that survives the controls: refusal-related features of the data track the
  variation among the runs. This is a representation-level correlate, not evidence that training
  pushes along that direction.
- The order score associated with harm where the stream score is not: placement matters more than
  the stream's average.
- The raw stream score associated with harm but not the prompt-centered one: the prompt mixture,
  not unusually refusal-like answers, carries the association.
- No association: this score is not established as associated with harm, with its interval. Other
  representation-level accounts are not excluded.
- Random or control directions doing as well: the association may reflect broad representation
  geometry rather than refusal.

## Direction at θ₀ (one per model)

- **Prompts.**
  - Harmful: the 100 refusal-drift prompts (`harmful_kl`), disjoint from the 521 harm prompts. The
    two other harmful sets (`harmful_judged`, `harmful_calibration`) hold all of the 521 (260 and
    261 of them) and 75 prompts outside it. Those 75 are not used, since their relation to the 521
    (the removal of near-paraphrases) is not checked here.
  - Harmless: the 100 held-out Alpaca instructions (`alpaca_heldout`), disjoint from the training,
    benign-drift and monitor instructions.
  - Split: each set is split 60/20/20 into direction, selection and confirmation prompts by
    `numpy.random.default_rng(derive_seed(0, "study5", "refusal-direction", "split")).permutation(100)`
    (indices 0–59, 60–79 and 80–99 of the permutation).
  - The prompt ids of each part are written to a committed record before any forward pass.
- **Inputs.** Each prompt is rendered by the project's chat template, as in training and
  evaluation. The token ids are the ones the template gives, recorded with the prompt ids. There is
  no padding: sequences run singly or in batches of equal length.
- **Precision.** FP32, as in training.
- **Layers.** ℓ indexes the residual stream as `hidden_states[ℓ]`: ℓ = 0 is the embedding output,
  and ℓ ≥ 1 is the output of block ℓ − 1, the input of block ℓ.
  - Candidate layers: 1 ≤ ℓ < 0.8 L, i.e. Qwen 1–22 (L = 28) and Llama 1–12 (L = 16).
- **Positions.** The post-instruction tokens: those after the instruction up to the start of the
  answer, as the template renders them. Their ids and count are read from each tokenizer at θ₀ as
  the suffix that two one-token instructions share once rendered, every direction prompt must end
  with them, and they are recorded (five tokens are expected for both models).
  - Llama's template writes the current date into its system turn, so its direction, selection and
    confirmation prompts carry the date of the Pod's run; the training answers keep their archived
    token ids.
- **Candidates.** r(ℓ, i) = mean of `hidden_states[ℓ]` at post-instruction position i over the 60
  harmful direction prompts, minus the same over the 60 harmless ones. A candidate whose vector is
  zero or not finite is excluded and listed. Template tokens at low layers can give identical
  activations for every prompt.
- **Refusal score.** An adaptation of Arditi et al.'s refusal log-odds: the study's surrogate r, the
  log-mean-exp likelihood of the five refusal openings minus that of the six compliance openings,
  teacher-forced at the start of the answer (`configs/study5/surrogate-2.yaml`, eleven
  continuations per prompt).
- **Interventions.**
  - Ablation: r̂ = r / ‖r‖ is projected out of every write to the residual stream, at every
    position. These are the embedding output and the output of every attention and every MLP
    sublayer.
  - Addition: r is added to `hidden_states[ℓ]` (the input of block ℓ) at every position.
- **Selection** (θ₀ only, forward passes only, on the selection prompts).
  - Baselines: base_harmful and base_harmless, the mean r over the 20 harmful and the 20 harmless
    selection prompts without intervention.
  - Scores per candidate:
    - bypass(ℓ, i): mean r on the 20 harmful selection prompts under ablation;
    - induce(ℓ, i): mean r on the 20 harmless selection prompts under addition;
    - KL(ℓ, i): mean KL(p_θ₀ ‖ p_ablated) of the next-token distribution at the last prompt position
      over the 20 harmless selection prompts.
  - A candidate is **validated** when all hold: bypass < base_harmful; induce > base_harmless and
    induce > 0; KL < 0.1.
  - Chosen: the validated candidate with the lowest bypass, ties broken by the lower layer, then the
    earlier position.
  - **If no candidate is validated**, the model's direction is reported as unvalidated and its
    projection analysis is not run. There is no fallback.
- **Confirmation** (in-sample effects named). The selection is tuned on the selection prompts, so
  its scores there are in-sample. On the 20 + 20 confirmation prompts, θ₀'s mean r with and without
  ablation (harmful) and with and without addition (harmless) are reported for the chosen direction.
- Orientation: r̂ points from harmless to harmful. Its function, not its orientation, is what the
  validation checks.

## Predictors (θ₀ and the data only)

**Projections.** For an answer y to a training prompt x, p(x, y) is the mean of
⟨`hidden_states[ℓ*]`(t), r̂⟩ over the answer's token positions t. The tokens are the archived prompt
and response token ids of the stream, teacher-forced through θ₀ in FP32 without padding. ℓ\* is the
chosen layer. The answer's positions are those of its response tokens, the end token excluded
(explicit mask). A truncated answer (no end token) uses all its 96 tokens.

**Stream predictors.** Each stream holds 500 answers to prompts drawn with replacement from the 100
training instructions (`teacher.draw_prompt_indices`), so streams differ in their prompt mixture.

- Primary: P_s^c = (1/500) Σ_i [p(x_i, y_i) − b_{−s}(x_i)]. Here b_{−s}(x) is the mean of p over
  all answers to prompt x in the other streams of the same model (leave one stream out). It is the
  prompt-matched projection difference; its expectation is zero for every stream, because the
  baseline answers are samples from the same distribution (θ₀ under the teacher's sampling). An
  answer whose prompt has no answer in the other streams has no b and is left out, listed.
- Secondary: the raw P_s = (1/500) Σ_i p(x_i, y_i).
- Persona Vectors' projection difference uses greedy base answers, so its expectation need not be
  zero. Here the baseline is the same sampling distribution, which is what makes the centred score
  a pure sampling deviation.
- The hypothesis predicts more harm for a lower P (answers further toward compliance).

**Order predictors.** These are heuristic activation scores, not the optimizer's predicted update:
Adam's coefficients do not turn activation projections into gradients, and the scores ignore the
preconditioner (fixed in frozen Adam, adaptive in AdamW).

- Primary: W(s, o) = Σ_t c_t · mean_{i in batch t} [p(x_i, y_i) − b_{−s}(x_i)].
  - The batches under order o are those of `study5_options.training_order` / `order_edit`.
  - c_t are the first moment's position weights with bias correction
    (`linear_response.momentum_weights`, β₁ = 0.9, 25 updates), which frozen Adam and AdamW share.
- Secondary (token-weighted, as the training loss's constant denominator): W_tok, the same with each
  answer's projection summed over its positions and divided by 96 instead of averaged; P_tok the
  stream analogue.
- The hypothesis predicts more harm for the order with the lower W.

## Tests (one-sided, family of four, Holm at 0.05)

The harm measures are those already reported; no run is added and no outcome is recomputed.

| Test | Predictor | Outcome | Units | p |
|---|---|---|---|---|
| T1, Qwen streams | P^c | mean h\* over the three orders (own, E2's two) | the 37 streams with h\* in all three | permutation, 10 000, RNG seed 20261118 |
| T2, Llama streams | P^c | mean h\* over the three orders | the 10 streams | permutation, 10 000 |
| T3, Qwen orders | W | sign of h\*(order 1) − h\*(order 2) | one pair per stream, E2's, 37 streams | exact binomial sign test |
| T4, Llama orders | W | h\* of the three orders | 10 streams | joint within-stream permutation |

- T1 and T2 use Spearman ρ, predicted negative.
- T3: a stream agrees when the order with the lower W has the higher h\*. A pair with |Δh\*| < 10⁻⁹
  pp is a tie and is excluded and listed; a stream without h\* in both orders is missing and listed.
- T4: the statistic is the mean over streams of Spearman ρ between W and h\* across the stream's
  three orders, predicted negative. Its null permutes the three h\* within each stream
  (100 000 draws, RNG seed 20261118).
- The four one-sided p-values are Holm-adjusted together. A test whose adjusted p < 0.05 reads
  "retrospectively associated with harm"; otherwise "not established". Each is reported with its
  effect: ρ or the agreement count, with a 95 % bootstrap interval over streams (10 000 resamples,
  RNG seed 20261118): for T1 and T2 of ρ, for T3 of the agreement share, for T4 of the mean
  within-stream ρ.
- Power: Llama has 10 streams. A one-sided test at 0.05 is significant only from about ρ = −0.56,
  and 80 % power needs |ρ| ≈ 0.74 (Fisher approximation). A non-significant T2 or T4 is read only
  with its interval.

**Secondary, descriptive, outside the family:**
- T1–T4's statistics for the raw P and W (uncentred) and for P_tok and W_tok;
- Qwen's own order alone: ρ over the 39 supported streams, the gradient score's comparison;
- M6's 35 edits against their stream order at the fixed rate: agreement counts with a 95 % interval
  from resampling the six streams (cluster bootstrap), no exact p, identity permutations excluded by
  their permutation. Each edit's order is rebuilt and checked against the permutation's sha256 in
  extension 2's record; a mismatch stops the analysis.

## Controls (descriptive)

- **Random directions.** 1000 random unit directions at ℓ\* (Gaussian, RNG seed 20261118), each with
  the statistics of T1–T4. The chosen direction's statistics are reported as percentiles of these
  reference distributions, which are not calibrated null p-values.
- **Another layer.** The difference-of-means direction at the chosen position at layer ⌊L/4⌋
  (Qwen 7, Llama 4), with the same statistics.
- **Answer length.** The stream's mean answer length in tokens (its ρ with the outcome of T1 and
  T2, and the partial ρ of P^c given length) and the order analogue for T3 and T4 (W with each
  answer's number of positions in place of its projection, uncentred).

## Deferred

The projection of trained minus θ₀ activations on the θ₀ direction, on fixed held-out prompts and
positions, and steering of trained models back toward refusal would connect the data's projection to
the training's change. Neither is planned here. They would be registered separately, and only if T1
or T2 reads "associated" and the checkpoints exist.

## Computation

- Forward passes only, θ₀ of each model, FP32.
  - Selection, per candidate: 220 bypass rows (20 prompts × 11 continuations), 220 induce rows and
    20 KL rows, i.e. 460 rows of about 70 tokens.
    - Qwen: 22 layers × 5 positions = 110 candidates, ≈ 50 600 rows.
    - Llama: 12 × 5 = 60 candidates, ≈ 27 600 rows.
  - Direction and confirmation: about 1 000 rows per model.
  - Projections: Qwen 40 streams × 500 answers = 20 000 rows; Llama 10 × 500 = 5 000; each under
    about 160 tokens. The per-answer mean activation at ℓ\* and at the control layer is stored, so
    the random-direction controls run on a CPU.
  - Total: about 6.5 M tokens for Qwen (1.5 B parameters) and 2.7 M for Llama (1.2 B), ≈ 2.7·10¹⁶
    FLOPs.
- The first step on the Pod (`direction --bench`, once per model) times the model load, the
  baselines, one layer's candidates and the projection of 40 answers, and extrapolates to the
  model's whole run; if the two models' projected totals add up to more than 2.5 h, the Pod stops
  before the rest and the timing is reported. The Pod's cap is 3 h.
- **Data.** The answers are the training streams in the bucket: pod 2's archive for the 40 Qwen
  fresh seeds, and the Llama campaign's stream archive for the 10 Llama seeds. Both are fetched as
  the campaigns fetch them. No response to a harmful prompt is read; scripts print numbers only.
- **Where.** A separate short Pod after the β₁ = 0 runs, never beside training.
- The registration is committed `registered`, with the prompt-id record, before any forward pass of
  θ₀ for this analysis.

## Code and records

- `src/zero_field_refusal/refdir.py`: the post-instruction tokens, the hooks (block inputs,
  ablation from every residual write, addition), the refusal score on batches of equal length
  without padding, the candidates, the selection rule, the confirmation and the answer means.
- `scripts/study5_refdir.py`: `prompts` (CPU) writes `docs/study5-refdir-prompts.json`;
  `direction` and `project` run on the Pod and print numbers, ids and hashes only.
- `src/zero_field_refusal/analysis/study5_refdir.py` and `scripts/study5_refdir_analysis.py`: the
  centring, the scores, T1–T4 with Holm, the secondary statistics and the controls, on a CPU, from
  the projections and the study's harm records; the record is written once.
- Tests: `tests/test_refdir.py`, `tests/test_study5_refdir_script.py`,
  `tests/test_study5_refdir_analysis.py`.
