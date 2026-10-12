# Gate 1: short report of the corrected pilot

> **Draft for review, 2026-10-12.**
>
> Gate 1 was not passed: the lower bound of the AdamW effect's interval is 14.44 pp against the
> fixed 15 pp. On 2026-10-12 the user decided to stop
> ([decision log](../decisions.md#gate-decisions)).
>
> Every comparison between optimizers below is exploratory and *preliminary*. The judge check
> across optimizers was undetermined after its first round; its pre-specified second round is
> being labeled ([rule](../preregistration.md#judge-check-across-optimizers)).
>
> Aggregates only.

## What was run

Stage 1 ran on 2026-10-11 ([runs and mechanics](stage-1.md)):

- **Runs:** 144 runs of full fine-tuning on Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct
  under the corrected zero-field protocol: 25 steps × batch 20 on θ₀'s own responses, with a
  96-token cap.
- **Judge:** every completed run's responses were labeled by WildGuard, calibrated at 94.5 % on
  200 reference items.
- **Optimizer cells on 1.5B:**
  - AdamW at η = 5·10⁻⁶, 1·10⁻⁵, 2·10⁻⁵ and 5·10⁻⁵;
  - SGD and Adam with frozen statistics ("frozen Adam") at rates chosen from seed-0 pilots;
  - SGD under 100 steps × batch 5.

## Gate 1 against its criteria (1.5B)

The criteria are [fixed in the preregistration](../preregistration.md#gate-1-technical).

| Criterion | Result | Met |
|---|---|---|
| Mean refusal drop of AdamW at 5·10⁻⁵ over 5 seeds ≥ 25 pp | 44.9 pp | yes |
| Lower bound of its 95 % t-interval ≥ 15 pp | 14.44 pp (interval [14.4, 75.3]) | **no** |
| Harmful-compliance rise: lower bound of its interval > 0 | 41.5 pp, [13.6, 69.5] | yes |
| Mean factual drop ≤ 10 pp | 1.7 pp, [1.1, 2.3] | yes |
| Judge agreement with the reference labels ≥ 90 % of 200 | 94.5 % (κ 0.90) | yes |
| Stable SGD cells cover [K_lo, √(K_lo·K_hi)] = [4.80, 5.85] nats | main condition up to K 2.41; 100 × 5 up to 5.51 | no |
| The effect disappears with the constant denominator | no: the token-count cell drops 47.1 pp [17.9, 76.2] | — |

**Verdict:** not passed. The effect criterion blocks, and the next step is to stop.

- **The interval.** θ₀ refuses 96 % of the judged prompts. The per-seed refusal drops are 77.0,
  28.7, 63.3, 37.0 and 18.3 pp: a mean well above 25 pp, but a seed SD of 24.5 pp. The
  preregistration assumed 15 pp, so the interval's half-width is 30 pp instead of 19.
- **Coverage.** K_lo = 4.80 comes from the logistic fit over the 20 AdamW runs. K_hi = 7.13 is
  AdamW at 2·10⁻⁵, the largest stable AdamW cell. Without the effect, coverage would have routed
  to the fallback; after the effect failure this is moot.

**Disclosures required by the amendments of 2026-10-12:**

- The effect cell collapses on the benchmark, 22.2 pp below θ₀. The cells of the same rate in the
  token-count and bridge branches lose 23.2 and 25.3 pp; on 0.5B the loss is 10.8 pp. No claim
  that general utility is preserved can be made at this rate.
- K_hi is taken among stable cells.

**Report-only readings** ([amendment](../preregistration.md#amendments)):

- **Not-harmful share.** Read from the not-harmful share by the same procedure, the drop is
  41.5 pp [13.6, 69.5], so its lower bound is also below 15. K_lo is 5.35, the coverage interval
  [5.35, 6.17], and neither SGD condition covers it.
- **Three-class tables.** The judge check's tables of the not-harmful side are in its result
  file.

## Judge check across optimizers

- **Round 1 is undetermined in every panel that could decide.** Each stratum holds only 3–18
  reference-harmful responses per 100, below the 20 per class the criterion needs.
- **No known failure.** Agreement with the reference is 0.88–0.97 in all nine strata.
- **The effect cell passes alone.** The Gate 1 cell, AdamW 5·10⁻⁵, passes its per-stratum
  criteria: agreement 0.88, sensitivity 0.96, false-positive rate 0.036, lower bound of the
  discrimination 0.80.
- **Refusal against safe non-refusal.** At that cell WildGuard labels 8 of the 14 reference safe
  non-refusals as refusals. An error of that kind understates the refusal drop rather than
  inflating it.
- **Round 2.** The pre-specified second round (100 further prompts per stratum) was drawn on
  2026-10-12 to close the procedure; its result will be added here.

## 0.5B (descriptive)

- **Effect criterion:** passes. Refusal drop 53.6 pp [41.8, 65.4]; harmful rise 53.3 pp
  [41.1, 65.4]; factual drop 9.0 pp.
- **Utility:** the effect cell collapses on the benchmark, 10.8 pp below θ₀.
- **SGD:** no stable rate on 0.5B. It collapses even at η = 1/300, a local instability along the
  layer-0 value bias ([diagnosis](stage-1.md#sgd-on-05b)).

## Exploratory results (preliminary)

### 1. Seed spread

Across the five seeds of the 1.5B effect cell, the refusal drop rises with the seed's
displacement (Spearman 0.9):

| Seed's K (nats) | Refusal drop |
|---|---|
| 12.8 | 18 pp |
| 14.8 | 29 pp |
| 16.6 | 37 pp |
| 19.1 | 77 pp |
| 20.0 | 63 pp |

The spread of the effect follows the spread of the displacement that the same rate produces.

### 2. Harmful compliance against K by optimizer (1.5B)

Cell means at comparable K; θ₀ refuses 96 % and complies harmfully on 3 %.

| K range (nats) | Cell | K | Refusal | Harmful |
|---|---|---|---|---|
| ≈ 2 | AdamW 5·10⁻⁶ | 1.86 | 0.935 | 0.048 |
| | frozen Adam 5·10⁻⁶ | 2.13 | 0.935 | 0.048 |
| | SGD 0.0228–0.0280 | 1.9–2.4 | 0.939–0.945 | 0.041–0.046 |
| ≈ 4–5.5 | AdamW 1·10⁻⁵ | 3.86 | 0.900 | 0.080 |
| | frozen Adam 1·10⁻⁵ | 4.54 | 0.879 | 0.097 |
| | SGD 100 × 5, 0.0132–0.0199 | 4.0–5.5 | 0.900–0.918 | 0.065–0.083 |
| ≈ 7–9 | AdamW 2·10⁻⁵ | 7.13 | 0.780 | 0.189 |
| | frozen Adam 2·10⁻⁵ | 8.61 | 0.723 | 0.238 |
| | SGD | — | no stable cell | — |

The analysis build fits the curves h(K) per optimizer. It uses the stage-1 interval settings:
99.5 % percentile intervals, paired seed bootstrap.

- **Eligible cells** (K\* = 2.27): Δ(K\*) = h_AdamW − h_SGD = +0.4 pp [−5.6, +3.2]. Frozen Adam
  is indistinguishable from both.
- **Every cell** (K\* = 5.91): Δ(K\*) = +3.8 pp [−8.0, +11.5], inconclusive. The SGD curve here
  is fitted over stable cells at K ≤ 2.4 and over collapsed pilots at K 350–410. Its value at
  5.9 therefore interpolates across a region with no stable SGD cell.

How to read this:

- Within the displacements that SGD reaches without collapse (K ≤ 2.4, or ≤ 5.5 under 100 × 5),
  harmful compliance at matched K is close across the three optimizers.
- It says nothing about the region above K_lo = 4.8, where AdamW's refusals fall by 10 pp or more
  and the main-condition SGD has no stable cell.
- It is not a test of H1. No Gate 2 decision is read from it.

### 3. SGD's cliff

- **Main condition.** Raising η from 0.0280 to 0.03 (× 1.07) takes K from 2.4 to 402 nats and
  collapses the model: harmful compliance 0.85, degenerate outputs.
- **100 × 5.** The step from 0.0199 to 0.0213 does the same: K from 5.5 to 245.
- **No intermediate stable state** was found under either condition.
- **Not a code error.** The saved training logs show an instability, not an error:
  - At every rate the gradient norm first rises over the first five steps. This early
    amplification grows smoothly with η: 1.4 × the step-0 norm at η ≤ 0.017 and 2.6 × at 0.0280.
  - At 0.03 it escapes: 21 ×, then one update of η·‖g‖ = 2.1.
  - The run is in float32 without clipping, and every quantity stays finite.
  - The threshold is lower with the noisier batch of 5.

  This matches an SGD learning rate near 2/λ_max (an edge of stability with a catapult), not a
  defect.

### 4. Response length and normalization (AdamW 5·10⁻⁵)

- **Bridge cell.** No evidence that the longer, 96-token responses weakened the effect. The paired
  difference, 96 minus 48 tokens, is −14.5 pp [−43, +14] on 1.5B and −10.3 pp [−27, +6] on 0.5B.
- **Token-count normalization.** The paired difference, constant minus token-count denominator,
  is −2.2 pp [−8.4, +4.0].

### 5. Line scans (A2)

- **Nonlinearity of K.** K grows faster than a² along θ₀ + aΔθ in every scan group, at every a
  except one mixed case (SGD at a = +0.5). The backward direction −Δθ gives a much larger K than
  +Δθ.
- **Random directions.** Random displacements of the trained norm, about 0.6 % of θ₀'s norm, give
  K ≈ 0.3 % of the trained K on 1.5B AdamW. The trained displacement is far from noise-like.
- **Refusal asymmetry, +Δθ minus −Δθ, per run:**

  | Run | Difference |
  |---|---|
  | AdamW 1.5B, seed 0 | −26.7 pp [−33.2, −19.8] |
  | AdamW 1.5B, seed 1 | +2.0 pp [−5.0, +8.9] |
  | AdamW 0.5B, seed 0 | −24.0 pp [−31.4, −16.2] |
  | SGD 1.5B, seed 0 | −10.0 pp [−13.9, −6.9] |
  | SGD 1.5B, seed 1 | −1.7 pp [−4.5, +1.0] |

### 6. Soft targets (SGD 0.0280, seed 0)

With θ₀'s full next-token distributions as targets:

- K is 0.67 nats, against 2.20 with sampled responses.
- Refusals are 7.0 pp [4.3, 10.4] higher, at θ₀'s level.

### 7. Gradient and update statistics

- **Gradient mean.** In two fixed random directions, the mean gradient over the 25 steps is
  indistinguishable from zero in every arm (median |z| 0.6–0.9), as the zero field predicts.
- **Drift against diffusion.** For most runs, the ratio of the squared final displacement to the
  summed squared updates is near 1, i.e. diffusive. The Adam arms have a tail up to 16–20.

### 8. Comparison with [4]

The reference is described in the [related work](../design.md#related-work-and-novelty).

- **Learning rate.** [4] finds that smaller learning rates reduce the effect. Here the AdamW
  refusal drop grows with η and K across cells (5·10⁻⁶ to 5·10⁻⁵), and with K across seeds at one
  rate.
- **Weight noise.** In [4], Gaussian weight noise removes safety at about 10 % of the weight
  norm. Here, random displacements at the trained norm change the model by about 0.3 % of the
  trained K. This agrees with a local safety basin [24]: the trained displacement acts through
  its direction, not its size.

## What the pilot shows

- **The effect.** Under the corrected protocol, AdamW on θ₀'s own responses lowers refusals by
  45 pp on average on 1.5B and by 54 pp on 0.5B. The effect varies strongly between seeds, in
  step with the displacement each seed reaches. With 5 seeds its interval misses the fixed lower
  bound by 0.56 pp.
- **Utility.** At the rate that produces the effect, the benchmark accuracy collapses.
- **Coverage.** Without collapse, SGD under 25 × 20 or 100 × 5 does not reach the displacements
  at which AdamW's refusals change. An equal-KL comparison where the effect lives was not
  possible in this design.
- **H1.** No conclusion about H1 is drawn.

## Evidence

- **Runs:** every stage-1 run directory, archived in the bucket ([stage-1 report](stage-1.md)).
  The effect cell holds `20261011T105906Z-adamw-5.0e-5-1.5b-s0-dd83f41e-a1` and the runs of seeds
  1–4 of the same cell.
- **Gate 1 report:** `scripts/gate1_report.py` at 69ed300, `gate1.json` sha256
  `1df352859df12a31…` (1.5B); `be070daa8f572ca9…` (0.5B).
- **Analysis build:** `scripts/build_analysis.py` at f66ea75 (clean checkout), `manifest.json`
  sha256 `a506a78802d313a1…`.
- **Judge check:** frozen sample 0752c68; reference labels sha256 `480ed4ee50fba526…`; result
  `qualification-2026-10-12.json` sha256 `ae876d7625d97c87…`; round 2 drawn in f66ea75.
- **Descriptive analyses:** `descriptive.json` sha256 `af5640e16592d17e…`.
