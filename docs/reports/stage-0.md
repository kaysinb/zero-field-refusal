# Stage 0: checks of the pipeline on the real checkpoints

Stage 0 has no gate. This report records the GPU measurements of
[plan.md, stage 0](../plan.md#stage-0--setup), the values set from them and the decisions taken on
2026-10-09. Aggregates only.

## Runs

All on one Runpod L40S (driver 595.91, CUDA 13.2 host), image
`ghcr.io/kaysinb/zero-field-refusal@sha256:9554679e…` (env-v1), torch 2.14.1, transformers 5.18.0,
`experiments/stage0_checks`, models at the revisions pinned in `configs/model/`.

| Run id | Source snapshot (commit) | Outcome |
|---|---|---|
| `stage0-qwen0.5b-l40s-20261009-01` | `b7bf058a` (7df9c75) | failed: the W&B mirror rejected integer keys (fixed in 1dfe138); streams kept |
| `stage0-qwen0.5b-l40s-20261009-02` | `8c76245b` (1dfe138) | completed |
| `stage0-qwen1.5b-l40s-20261009-01` | `8c76245b` (1dfe138) | failed: CUDA out of memory in the field at θ₀ (fixed in a8cd44c); streams kept |
| `stage0-qwen1.5b-l40s-20261009-02` | `153ee510` (a8cd44c) | completed; path-field record `path-field-3d0c94dfebfe` |
| `stage0-qwen0.5b-l40s-20261009-03` | `153ee510` (a8cd44c) | completed, bit-identical to attempt 02; record `path-field-0efb3469a979` |

The results below come from the two runs of snapshot `153ee510`.

## Measurements

| | 0.5B | 1.5B |
|---|---|---|
| Path field: upper bound N·X⁺ (criterion 0.01) | 1.2 × 10⁻⁶ | 2.4 × 10⁻⁶ |
| χ² per response, X̂ ± SE (160 responses) | (2.26 ± 0.07) × 10⁻⁹ | (4.34 ± 0.21) × 10⁻⁹ |
| χ² sums: skewness, largest response's share | −0.80, 1.1 % | 1.23, 2.1 % |
| N·\|f\|²/σ², Euclidean / Adam metric (descriptive) | 5.1 × 10⁻⁷ / 1.2 × 10⁻⁷ | 6.1 × 10⁻⁸ / 7.2 × 10⁻⁸ |
| Per-token KL between the paths, mean / max | 1.4 × 10⁻¹¹ / 3.3 × 10⁻¹⁰ | 2.7 × 10⁻¹¹ / 4.0 × 10⁻⁹ |
| Teacher gap, max \|Δ\| per token, seeds 0 / 1 / 2 | 6.0 / 4.4 / 4.8 × 10⁻⁵ | 2.0 / 1.0 / 1.4 × 10⁻⁴ |
| Teacher gap, mean signed Δ per response, seeds 0 / 1 / 2 | −2.5 / 0.5 / 1.2 × 10⁻⁶ | −3.4 / −0.9 / 0.8 × 10⁻⁶ |
| Custom Adam vs `torch.optim.AdamW`, 5 steps, every 4th tensor | max difference 0.0 | max difference 0.0 |
| Responses ending with an end token within 96 tokens | 22 % (mean length 83.9) | 28 % (79.8) |
| Token-count field: N·\|f\|²/σ² [± 2 SE] | 0.14 [−0.03, 0.32] | 0.04 [−0.17, 0.25] |
| Token-count field: N·Var(c_b) | 0.17 | 0.23 |
| Constant denominator: \|f\|² ± SE, σ² along its direction | 0.016 ± 0.021, 0.042 | 0.0085 ± 0.0074, 0.021 |
| Mean-gradient norm over 1 → 100 batches (noise only: ÷ 10) | 7.45 → 0.707 | 3.75 → 0.364 |
| Training step (batch 20, micro-batch 4) | 0.41 s | 1.20 s |
| Greedy batch, 32 prompts × 256 tokens | 3.5 s | 5.6 s |
| Peak GPU memory of a run (student, θ₀ copy, Adam, greedy batch) | 13.3 GB | 26.3 GB |
| The stage-0 check itself, streams reused / generated | 5.5 min / — | — / 14 min |

## Values and decisions

- **Path field.** Both models pass by three to four orders of magnitude; the per-response sums are
  not heavy-tailed, so the bootstrap-t bound keeps its level ([criterion](../architecture.md#design-decisions-for-stage-0)).
- **Teacher-gap thresholds** (10 × the largest gap over seeds 0–2): 0.5B 6.0 × 10⁻⁴ per token and
  2.6 × 10⁻⁵ per response, 1.5B 2.0 × 10⁻³ and 3.5 × 10⁻⁵; in `configs/model/` and
  [preregistration.md](../preregistration.md#primary-analysis), status `proposed`.
- **A1 is a diagnostic in stage 1.** The token-count field is at the noise level for both models,
  by the estimate's upper end and by the sufficient criterion N·Var(c_b) < 1, so by the rule of
  [A1](../preregistration.md#alternative-explanations) it is reported, not tested.
- **The constant denominator's field is `undetermined`, and that is not a gap.** The estimate is
  within 1.2 SE of zero, but showing N·|f|²/σ² < 1 against the noise along a single direction
  would take about a thousand times more batches. The zero field of the constant denominator
  rests on the identity E_p[∇ log p] = 0, whose only numerical remainder is the path discrepancy,
  bounded above at about 10⁻⁶; the mean-gradient norm also falls exactly as 1/√n, matching the
  noise-only value √(tr Σ / n) within 2–3 % at 100 batches.
- **End tokens.** Three quarters of the responses are truncated at 96 tokens. Truncation keeps the
  field at zero; the end token enters the loss in 22–28 % of the responses.
- **GPU.** Stages 0–2 run on the L40S. Other 48 GB cards would need their own checks and would
  enter the thresholds.
- **Soft-target distributions** of the path-field check are deleted from the Pod's volume
  ([retention](../operations.md#retention-plan-stages-0-and-1)); their hashes and the records stay.
