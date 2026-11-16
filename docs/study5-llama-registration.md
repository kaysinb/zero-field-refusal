# Study 5: the order repeat on a second model family

Status: `registered` (2026-11-16), before any run on the model and before any of its labels.

## Question

Study 5's order results come from one model, Qwen2.5-1.5B-Instruct. The repeat asks whether, on a
model of another family trained on its own samples at zero initial expected gradient, the order
of the same responses moves harm at matched benign drift, and how far.

## Design

- **Model.** Llama-3.2-1B-Instruct at revision `9213176726f574b556790deb65791e0c5aa438b6`, fp32,
  SDPA attention, its input embeddings and its tied output head frozen
  (`configs/model/llama-3.2-1b-instruct.yaml`). Its chat template's date is fixed at its own
  default, 26 Jul 2024, so that a prompt does not depend on the day it is encoded; its pad token
  is `<|finetune_right_pad_id|>`, a reserved token it never samples; generation ends at
  `<|eot_id|>`, `<|end_of_text|>` or `<|eom_id|>` (`models.pin_tokenizer`).
- **Zero field.** Every training response is sampled from the model's own θ₀ under the protocol's
  neutral sampling, as for Qwen. Before any training run, the stage-0 checks run on this model
  (`experiments/study5_llama_stage0`: the response cap, the teacher gap, the mean-gradient scaling
  and the token-count field, the path field, the optimizer, memory). Its path-field check must pass
  and it sets the runs' teacher-gap thresholds, 10 × the largest gap over seeds 0–2 as for Qwen
  (`llama_orders.checks_record`); otherwise no run trains.
- **Runs.** Arm A of study 3 (AdamW, β = (0.9, 0.999), ε = 10⁻⁸, no weight decay), 25 updates of
  B = 20 (`experiments/study5_llama`), on ten fresh seeds 60000–60009, each seed's own 500
  responses in three orders: `train.order_seed` = s + 1000, s + 2000 and s + 3000. Each order at
  two rates: 60 runs.
- **Rates, fixed by a pilot before any main run.** K-only runs of seeds 60000 and 60001 in order 1
  at 5·10⁻⁶, 10⁻⁵, 2·10⁻⁵, 4·10⁻⁵, 8·10⁻⁵ and 1.6·10⁻⁴. Per seed, the rate where K_benign at step 25
  reaches 4.84 (the median K of A's study-3 runs at its η_l) is interpolated in ln K against ln η
  between the lowest adjacent pair of pilot rates that brackets it. η_l is the two seeds' geometric
  mean, η_u = η_l × 2.11/1.45 (A's ratio in study 3), both to three significant digits
  (`llama_orders.rate_choice`). No bracket stops the experiment.
- **Measured.** As every study-5 run: its end, K_benign, K_harmful, MMLU and factual accuracy and
  the greedy responses to the 521 prompts at step 25; θ₀'s own evaluation
  (`experiments/study5_llama_theta0`). K is measured against Llama's own θ₀ samples (its K panels,
  sampled on the Pod).
- **Labels.** WildGuard on the 521 prompts for the 60 runs and θ₀, as in study 5.

## Analysis (`scripts/study5_llama_analysis.py`)

- **h at K\*.** Per seed and order, h interpolated at K\* = 6.00 between the order's two runs by
  study 3's support rule (bracket, width at most 0.60 in ln K, weight in ln K), as in E2.
- **Erosion.** The mean over seeds of h at K\* (each seed's mean over its supported orders) minus
  θ₀'s h, with a 95 % percentile bootstrap over seeds (10 000 resamples,
  `default_rng(derive_seed(0, "study5", "llama-bootstrap"))`). Order sensitivity is read only if
  erosion's interval lies above 0; otherwise the repeat reports that this model's harm did not rise
  at K\* and draws no conclusion on order.
- **Order differences.** Per seed, Δ\* of each of its three pairs of orders (1–2, 1–3, 2–3) with
  both orders supported; the seed's value is the mean |Δ\*| over its pairs. The primary
  observation is the mean of that value over the seeds with at least one pair, with the same
  bootstrap over seeds.
- **Label noise for scale.** √(2 h̄(1 − h̄)/521)·√(2/π), h̄ the pooled harmful share of the 60
  runs; order sensitivity is established on this model when the primary interval lies above it.
- **Beside it.** Qwen's mean |Δ\*| per pair under F (E2) and under A (the AdamW repeat), from their
  own records, without a test across models. At each fixed rate the same order differences;
  `harmful_high`; the share of seed and order pairs whose two rates bracket K\*; MMLU at θ₀ and its
  loss, noting that a 1B model's MMLU lies near chance.

## Scope

The repeat answers whether the order effect appears in one model of another family, at 1B, under
AdamW, 25 updates and the 521-prompt benchmark, on ten seeds; it is not a study of model scale.
