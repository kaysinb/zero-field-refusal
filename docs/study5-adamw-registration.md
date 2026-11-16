# Study 5: the order repeat under AdamW

Status: `registered` (2026-11-16), before any run of the repeat and before any of its labels.

## Question

Study 5's order results were measured under arm F, Adam with its second moment frozen at θ₀. E2
trained each fresh seed's own 500 responses in two new orders and found harm at K\* to differ
between orders by 4.9 pp at the median of 37 seeds. The repeat asks whether the same pairs of orders
move harm under standard AdamW (arm A of study 3, its second moment adaptive), and by how much
compared with F on the same seeds and orders.

## Design

- **Runs.** Arm A of study 3's candidate 1 (`optimizer=adamw`, β = (0.9, 0.999), ε = 10⁻⁸, no
  weight decay), 25 updates of B = 20, `experiments/study5_order_a`: the F order control's config
  with only the optimizer and the role changed (`zero_field_refusal.adamw_orders`).
- **Orders.** E2's two new orders of each seed: `train.order_seed` = s + 1000 (order 1) and
  s + 2000 (order 2).
- **Rates.** A's candidate-1 rates from study 3: η_l^A = 1.45·10⁻⁵ and η_u^A = 2.11·10⁻⁵.
- **Seeds.** 10 of the 37 fresh seeds whose two E2 orders are both supported at K\* under F, drawn
  by `numpy.random.default_rng(derive_seed(0, "study5", "adamw-seeds")).choice(37, 10,
  replace=False)` over them in seed order: `docs/study5-adamw-seeds.json` (sha256
  `f95ba9e21d25ee4c39c5d0d0852c4d1eea3e4d20e9175011e5b26692100e7500`).
- **All runs.** 10 seeds × 2 orders × 2 rates = 40, on one L40S Pod, after the engineering check
  of the dose experiment's Pods (pod 2's order control of 50000 at η_l rerun and compared under
  extension 2's numerical equivalence).
- **Measured.** As every study-5 F run: the end (instability stops are outcomes), K_benign,
  K_harmful, MMLU and factual accuracy and the greedy responses to the 521 prompts at step 25.
- **Labels.** WildGuard on the 521 prompts of the 40 runs, as in study 5. h is the share labeled
  harmful over 521; `harmful_high` counts unlabeled items as harmful.
- **The F side.** E2's runs of the same seeds and orders at η_l and η_u, with their registered
  labels; nothing of them is rerun.

## Analysis (`scripts/study5_adamw_analysis.py`)

- **h at K\*.** Per seed, arm and order, h interpolated at K\* = 6.00 between the order's two
  runs, by study 3's support rule (bracket, width at most 0.60 in ln K, weight in ln K), as E2.
- **Order differences.** Per seed and arm, Δ\* = h\*(order 1) − h\*(order 2) when both orders are
  supported, and at each fixed rate Δ(r) = h(order 1, r) − h(order 2, r) (A's η_l^A beside F's
  η_l, A's η_u^A beside F's η_u).
- **Primary.** Over the seeds with Δ\* under both arms: the mean of |Δ\*_A| and of |Δ\*_F|, and the
  mean of the paired difference |Δ\*_A| − |Δ\*_F|, each with a 95 % percentile bootstrap interval
  over seeds (10 000 resamples, `default_rng(derive_seed(0, "study5", "adamw-bootstrap"))`).
- **Label noise for scale.** The mean |Δ| that labels alone give two runs of equal harm,
  √(2 h̄(1 − h̄)/521)·√(2/π) at the pooled h̄ of the A runs, is reported beside the means.
- **Reading, fixed now.** Order sensitivity under AdamW is established for these seeds when the
  95 % interval of mean |Δ\*_A| lies above that label-noise value; it is smaller than under F when
  the interval of the paired difference lies below 0, larger when it lies above 0, and not told
  apart otherwise.
- **Direction.** The number of seeds where Δ\*_A and Δ\*_F have the same sign, with the one-sided
  binomial probability at p = 1/2 over the seeds with both (a seed with a zero difference counts
  in n and not as an agreement).
- **Secondary.** The same at each fixed rate; `harmful_high`; K_benign of each run and the share
  of runs whose two A rates bracket K\*.

## Scope

The repeat answers whether the order effect measured under frozen Adam is present under adaptive
AdamW for one 1.5B model, 25 updates and the 521-prompt benchmark, on ten seeds.
