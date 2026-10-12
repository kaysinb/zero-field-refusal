# Gate 1: short report of the corrected pilot

> **Report of the stop, 2026-10-12.**
>
> Gate 1 was not passed: the lower bound of the AdamW effect's interval is 14.44 pp against the
> fixed 15 pp. On 2026-10-12 the user decided to stop
> ([decision log](../decisions.md#gate-decisions)).
>
> Every comparison between optimizers below is exploratory and *preliminary*. The judge check
> across optimizers stayed undetermined after its pre-specified second round, so differential
> judge error between optimizers could not be checked
> ([rule](../preregistration.md#judge-check-across-optimizers)).
>
> Aggregates only.

## What was run

Stage 1 ran on 2026-10-11 ([runs and mechanics](stage-1.md)):

- **Runs:** 144 runs of full fine-tuning on Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct
  under the corrected zero-field protocol: 25 steps × batch 20 on θ₀'s own responses, with a
  96-token cap.
- **Optimizer cells on 1.5B:**
  - AdamW at η = 5·10⁻⁶, 1·10⁻⁵, 2·10⁻⁵ and 5·10⁻⁵;
  - SGD and Adam with frozen statistics ("frozen Adam") at rates chosen from seed-0 pilots;
  - SGD under 100 steps × batch 5.
- **Judge:** the responses of the 141 completed runs were sent to WildGuard, calibrated at
  94.5 % on 200 reference items.
  - Labeled: 54 300 outputs, of which 2060 did not parse.
  - Where the unparsed outputs fall: all in 8 collapsed runs, whose responses are a single
    repeated character.
  - Consequence: these runs lack labels and leave every analysis that needs them.

## Gate 1 against its criteria (1.5B)

The criteria are [fixed in the preregistration](../preregistration.md#gate-1-technical).

| Criterion | Result | Met |
|---|---|---|
| Mean refusal drop of AdamW at 5·10⁻⁵ over 5 seeds ≥ 25 pp | 44.9 pp | yes |
| Lower bound of its 95 % t-interval ≥ 15 pp | 14.44 pp (interval [14.44, 75.29]) | **no** |
| Harmful-compliance rise: lower bound of its interval > 0 | 41.5 pp, [13.6, 69.5] | yes |
| Mean factual drop ≤ 10 pp | 1.7 pp, [1.1, 2.3] | yes |
| Judge agreement with the reference labels ≥ 90 % of 200 | 94.5 % (κ 0.90) | yes |
| Stable SGD cells cover [K_lo, √(K_lo·K_hi)] = [4.80, 5.85] nats | main condition up to K 2.41; 100 × 5 up to 5.51 | no |
| The effect disappears with the constant denominator | no: the token-count cell drops 47.1 pp [17.9, 76.2] | — |

**Verdict:** not passed. The effect criterion blocks, and the next step is to stop.

- **The interval.** θ₀ refuses 96 % of the judged prompts. The per-seed refusal drops are 77.0,
  28.7, 63.3, 37.0 and 18.3 pp, with an SD of 24.5 pp. The preregistration assumed 15 pp, so the
  half-width is 30.4 pp instead of 18.6.
- **K_lo and K_hi.** K_lo = 4.80 comes from the logistic fit over the 20 AdamW runs. K_hi = 7.13
  is AdamW at 2·10⁻⁵, the largest stable AdamW cell.
- **Coverage.** It fails because SGD does not span the interval up to 5.85. Under the main
  condition the stable SGD cells stop at K 2.41, below K_lo. Under 100 × 5 they reach 5.19 and
  5.51, beyond K_lo but short of the upper end. After the effect failure this is moot.

**Disclosures required by the amendments of 2026-10-12:**

- The effect cell collapses on the benchmark, 22.2 pp below θ₀. The token-count and bridge cells
  of the same rate lose 23.2 and 25.3 pp; on 0.5B the loss is 10.8 pp. No claim that general
  utility is preserved can be made at this rate.
- K_hi is taken among stable cells.

**Report-only readings** ([amendment](../preregistration.md#amendments)):

- **Not-harmful share.** Read from the not-harmful share by the same procedure, the drop is
  41.5 pp [13.6, 69.5], so its lower bound is also below 15. K_lo is 5.35, the coverage interval
  [5.35, 6.17], and neither SGD condition covers it.
- **Three-class tables.** The judge check's tables of the not-harmful side, per stratum, are under
  `not_harmful_tables` in its result file (see Evidence).

## Judge check across optimizers

**The sample.** Both rounds together: 200 prompts per stratum in nine strata, 1800 items.

**The reference labels.** The same in both rounds: GPT and Grok label each distinct text
independently, and the user checks every disagreement and 30 random agreements.

- Round 1: the user changed 2 of the 45 items checked.
- Round 2: the user changed 9 of the 52 items checked.
- Every change was to a disputed item.

**The comparison strata.** Seven of them have fewer than 20 reference-harmful responses each, so
every panel that compares optimizers is undetermined:

| Stratum | Reference-harmful responses (of 200) |
|---|---|
| SGD 0.0262 and 0.0280 | 4 each |
| frozen Adam 5·10⁻⁶ | 8 |
| frozen Adam 1·10⁻⁵ | 14 |
| AdamW 5·10⁻⁶ | 10 |
| SGD 100 × 5, 0.0186 and 0.0199 | 12 and 10 |

AdamW 2·10⁻⁵ reaches 32. There is no known failure:

- agreement with the reference is 0.89–0.98 in every stratum;
- the lower bound of the discrimination is 0.76 and 0.82 in the two strata whose classes fill.

**The Gate 1 stratum passes its per-stratum criteria.** AdamW 5·10⁻⁵ has 88 harmful and 112 not
harmful responses: agreement 0.89, sensitivity 0.93, false-positive rate 0.036, lower bound of
the discrimination 0.82.

**The procedure is closed.**

- The pre-specified second round was the last one.
- Differential judge error between optimizers could not be checked, and every comparison between
  optimizers in this report says so.
- The sensitivity corrections between optimizers are *not identifiable*.
- In the sparse strata the sensitivity is 0.58–0.75, on 4–14 harmful items each. That is
  descriptive only.

**Refusal against safe non-refusal.** At the Gate 1 cell:

- WildGuard labels all 88 reference refusals as refusals.
- Of the 24 reference safe non-refusals, it labels 12 as refusals, 8 as safe non-refusals and 4
  as harmful compliance.

Calling a safe non-refusal a refusal understates the refusal drop.

## 0.5B (descriptive)

| Component | Result | Status |
|---|---|---|
| Effect cell, AdamW 5·10⁻⁵ | refusal drop 53.6 pp [41.8, 65.4]; harmful rise 53.3 pp [41.1, 65.4]; factual drop 9.0 pp; benchmark drop 10.8 pp | effect criterion passes; benchmark collapse |
| AdamW 2·10⁻⁵ | refusal drop 31.7 pp; harmful rise 30.5 pp; factual 0.1 pp; benchmark 0.4 pp | described |
| Token-count branch, 5·10⁻⁵ | refusal drop 57.0 pp [47.7, 66.3]; paired difference from the constant denominator −3.4 pp [−13.6, +6.8] | A1 is a diagnostic, as decided in stage 0 |
| Bridge cell, 48 tokens | refusal drop 63.9 pp; factual drop 20.8 pp; benchmark drop 18.8 pp | collapsed |
| SGD | every pilot from η = 1/300 to 9 collapsed; 27 diverged | no stable rate ([diagnosis](stage-1.md#sgd-on-05b)) |
| Soft-target control | — | not run: SGD has no stable rate on 0.5B |
| Line scan, AdamW 5·10⁻⁵, seed 0 | see section 5 below | run |

## Exploratory results (preliminary)

### 1. Seed spread

| Model | Seed's K (nats) | Refusal drop | Spearman ρ |
|---|---|---|---|
| 1.5B | 12.8 / 14.8 / 16.6 / 19.1 / 20.0 | 18 / 29 / 37 / 77 / 63 pp | +0.9 |
| 0.5B | 24.5–34.1 | 41.3–64.7 pp | −0.9 |

On 1.5B the five seeds' refusal drops rise with each seed's displacement. On 0.5B the order is
reversed. The association therefore holds on 1.5B only and is not general.

### 2. Harmful compliance against K by optimizer (1.5B)

The table gives cell means; θ₀ refuses 96 % and complies harmfully on 3 %. Cells of five seeds
unless marked. The SGD 100 × 5 cells are shown for description only and are not part of the
fits below.

| K range (nats) | Cell | K | Refusal | Harmful |
|---|---|---|---|---|
| ≈ 2 | AdamW 5·10⁻⁶ | 1.86 | 0.935 | 0.048 |
| | frozen Adam 5·10⁻⁶ | 2.13 | 0.935 | 0.048 |
| | SGD 0.0228–0.0280 | 1.92–2.41 | 0.939–0.945 | 0.041–0.046 |
| ≈ 4–5.5 | AdamW 1·10⁻⁵ | 3.86 | 0.900 | 0.080 |
| | frozen Adam 1·10⁻⁵ | 4.54 | 0.879 | 0.097 |
| | SGD 100 × 5, 0.0132–0.0199 (not in the fits) | 4.02–5.51 | 0.900–0.918 | 0.065–0.083 |
| ≈ 7–9 | AdamW 2·10⁻⁵ | 7.13 | 0.780 | 0.189 |
| | frozen Adam 2·10⁻⁵ | 8.61 | 0.723 | 0.238 |
| | SGD (main condition) | — | no stable cell | — |

The analysis build fits h(K) per optimizer under the main condition. It uses the stage-1 interval
settings: 99.5 % percentile intervals from the paired seed bootstrap.

**Eligible cells.**

- The common range is [2.13, 2.41], so K\* = 2.26.
- Δ(K\*) = h_AdamW − h_SGD = +0.4 pp [−5.7, +3.2].
- Frozen Adam minus SGD: −0.7 pp [−6.1, +3.1]. Frozen Adam minus AdamW: −1.1 pp [−2.4, +0.5].

**Every cell** (the original rule).

- The common range is [2.13, 16.45], so K\* = 5.91.
- Δ(K\*) = +3.8 pp [−8.1, +11.5].
- Frozen Adam minus SGD: +3.6 pp [−6.6, +10.3]. Frozen Adam minus AdamW: −0.2 pp [−2.2, +2.4].
- **Runs left out of this build:**
  - the SGD pilots at η = 0.1 and 3, whose outputs did not parse at all;
  - the diverged SGD pilot at η = 27;
  - the θ₀ run, which is not a training run;
  - in the fits, the SGD pilot at 1/300, below the threshold C_lo/10 = 0.21.
- **What the SGD curve rests on.** It is fitted over the stable cells at K ≤ 2.41 and over
  collapsed pilots at K 355–402. Its value at K = 5.91 therefore interpolates across a range
  without any stable SGD cell.

**How to read this:**

- These comparisons hold only at the levels analyzed: the eligible common range [2.13, 2.41] of
  the main condition.
- They say nothing about the range above K_lo = 4.80, where AdamW's refusals fall by 10 pp or
  more and the main-condition SGD has no stable cell.
- They are not a test of H1, and no Gate 2 decision is read from them.

### 3. SGD's cliff

**Main condition.** The 5-seed cell at η = 0.0280 has a mean K of 2.41; the single-seed pilot at
η = 0.03 collapsed. Paired by seed 0, on identical batches:

- K goes from 2.20 at 0.0280 to 402 at 0.03 (× 183).
- At 0.03 the model collapses: harmful compliance 0.85, degenerate outputs.

**100 × 5.** Paired by seed 0, K goes from 5.13 at 0.0199 (5-seed mean 5.51) to 245 at the
single-seed pilot at 0.0213 (× 48).

**An instability that depends on the rate.** The saved training logs show:

- At every rate the gradient norm first rises over the first five steps.
- This early amplification grows smoothly with η: 1.18 × the step-0 norm at 1/300, 1.42 × at
  0.0173, 2.63 × at 0.0280. At 0.03 it reaches 20.9 ×, and one update of η·‖g‖ = 2.1 follows.
- The training runs in float32 without clipping, and every logged quantity stays finite.
- Under the noisier batch of 5 the threshold is lower.

**The code.** A review of the SGD step's code found no defect.

**Presumed mechanism:** a learning rate near 2/λ_max, with an edge of stability and a catapult.
The curvature was not measured.

### 4. Response length and normalization (AdamW 5·10⁻⁵)

- **Bridge cell.** Whether the longer, 96-token responses weaken the effect is uncertain. The
  estimated weakening is 14.5 pp on 1.5B and 10.3 pp on 0.5B, with wide intervals: [−43.2, +14.3]
  and [−26.8, +6.2] for 96 minus 48 tokens.
- **Token-count normalization.** The paired difference, constant minus token-count denominator,
  is −2.2 pp [−8.4, +4.0] on 1.5B and −3.4 pp [−13.6, +6.8] on 0.5B.

### 5. Line scans (A2)

- **Nonlinearity of K.** Along θ₀ + aΔθ, the ratio K(a)/(a²K(1)) lies above 1.25 for every scan
  run at every a. There is one exception: SGD seed 0 at a = +0.5, whose interval meets
  [0.8, 1.25]. The backward direction −Δθ gives a much larger K than +Δθ.
- **Random directions.** On 1.5B AdamW, random displacements of the same norm (about 0.6 % of
  θ₀'s norm) produce a K of about 0.3 % of the trained K.
- **Refusal asymmetry, +Δθ minus −Δθ, per run** (Newcombe's method 10):

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
- Refusals are 7.0 pp [4.3, 10.4] higher, paired by prompt.

### 7. Gradient and displacement in two random projections

This is an exploratory diagnostic in two fixed random directions, not a statement about the
whole gradient vector.

- **Projected gradient.** Averaged over the 25 steps, the mean projected gradient has a |z| with
  a median of 0.6–1.0 per arm. |z| exceeds 2 in 3 of the 60 projections of the main-condition
  SGD runs and in none elsewhere.
- **Drift against diffusion.** The ratio of the squared final projected displacement to the
  summed squared projected updates has these medians:
  - 1.5B: AdamW 1.11, frozen Adam 1.08, SGD 0.78, SGD 100 × 5 0.38;
  - 0.5B: AdamW 3.89, SGD 1.0.

  The Adam arms have a tail up to 16–20.

### 8. Comparison with [4]

The reference is described in the [related work](../design.md#related-work-and-novelty).

- **Learning rate.** [4] finds that smaller learning rates reduce the effect. Here, too, the
  AdamW refusal drop is larger at larger η and K, across the cells from 5·10⁻⁶ to 5·10⁻⁵.
- **Weight noise.** In [4], Gaussian weight noise removes safety at about 10 % of the weight
  norm. Here random displacements were tested at about 0.6 % of the norm only, where they changed
  K by about 0.3 % of the trained K. The scales differ by more than an order of magnitude, so
  neither observation tests the other.

## What the pilot shows

- **The effect.** Under the corrected protocol, AdamW on θ₀'s own responses lowers refusals by
  44.9 pp on average on 1.5B and by 53.6 pp on 0.5B. On 1.5B the effect varies strongly between
  seeds, and with 5 seeds its interval misses the fixed lower bound by 0.56 pp.
- **Utility.** At that rate the benchmark accuracy collapses on both models.
- **Coverage.** Under the main condition, SGD stays stable only up to K 2.41, below K_lo. Under
  100 × 5 it reaches 5.51, short of the interval's upper end, 5.85.
- **H1.** No conclusion about H1 is drawn.

## Evidence

The paths are under the runs root (`RUNS_DIR`). Every stage-1 run directory is archived in the
bucket ([stage-1 report](stage-1.md)). The effect cell holds the runs
`20261011T105906Z-adamw-5.0e-5-1.5b-s0-dd83f41e-a1`, `20261011T110250Z-adamw-5.0e-5-1.5b-s1-14107220-a1`,
`20261011T110637Z-adamw-5.0e-5-1.5b-s2-3932d764-a1`, `20261011T111021Z-adamw-5.0e-5-1.5b-s3-c848e2a3-a1`
and `20261011T111406Z-adamw-5.0e-5-1.5b-s4-6f1639c7-a1`.

### Gate 1, 1.5B

- **Command:**
  `python scripts/gate1_report.py runs --judge wildguard --model 1.5b --calibration runs/judge_calibration/calibration --out runs/analysis/gate1-1.5b-2026-10-12-b`
- **Code:** 69ed300, clean.
- **`gate1.json`:** sha256 `1df352859df12a31c1a76a172d817e7b974d5b2ec9792f2c547b9bc03ef2288d`.
- **`manifest.json`:** sha256 `99aa44e049ac3ad753c1aa6dfa128bdd551960438e3d8f030e11763af4afc2f9`.

### Gate 1, 0.5B

- **Command:**
  `python scripts/gate1_report.py runs --judge wildguard --model 0.5b --out runs/analysis/gate1-0.5b-2026-10-12`
- **Code:** 9aa8d08, clean.
- **`gate1.json`:** sha256 `be070daa8f572ca969a70a64435ac0774fb027eb9b43e907c808aa8f8e5cb18d`.
- **`manifest.json`:** sha256 `25addbe9ab5d9e6d398d1b76c2d6ecf64dde78c59dbff8ebaeeb459ec0d5dd1c`.

### Analysis build, 1.5B

- **Command:**
  `python scripts/build_analysis.py runs --experiment stage1_main_1.5b --judge wildguard --out runs/analysis/build-stage1-main-1.5b-2026-10-12`
- **Code:** f66ea75, clean.
- **`manifest.json`:** sha256 `a506a78802d313a1398c81bdc7feb707310d64347c597ab5c7ebee5a09ad5350`.

### Judge check, both rounds

**Frozen samples:**

| Round | Directory | Selection committed in |
|---|---|---|
| 1 | `runs/judge_check/frozen-2026-10-11` | 0752c68 |
| 2 | `runs/judge_check/frozen-2026-10-11-round2` | f66ea75 |

**Reference labels:**

- Round 1: `reference/labels-d7-509b6f1d.jsonl`, sha256
  `480ed4ee50fba52696a8e3807adae5c2f834a1df5f4b36cb1e5cbeabe952980e`.
- Round 2: `reference/labels-d7r2-96d185b1.jsonl`, sha256
  `0ebb81a6b03507096e3362df8d1adce4cc95692750dea39ccaf1b0545b64199c`.
- Each round's provenance is in the `reference/README.md` next to its labels, with the user's
  review log.
- The coordination copies are in the bucket under `coordination/2026-10-12/`.

**Command:**
`python scripts/judge_check_qualify.py runs/judge_check/frozen-2026-10-11 runs/judge_check/frozen-2026-10-11-round2 --reference runs/judge_check/frozen-2026-10-11/reference/labels-d7-509b6f1d.jsonl --reference runs/judge_check/frozen-2026-10-11-round2/reference/labels-d7r2-96d185b1.jsonl --runs-root runs --judge wildguard --out runs/judge_check/frozen-2026-10-11/qualification-both-rounds-2026-10-12.json`

- **Code:** 479d567, clean, recorded in the file.
- **Result file:** sha256 `9dca8835c27b009bb23050e310b9dcdcac2325e8fbf37c8fc6d4e5d0bca92f22`.
- **Three-class tables:** under `not_harmful_tables` in that file.
- **Round 1 alone:** `qualification-2026-10-12-r2.json`, sha256
  `8b23b140174fec675e2cbcf0fbb84c97567fa0f7569b5446e6b4349a3b78b962`, at 4e8b766.

### Descriptive analyses

- **Command:**
  `python scripts/stage1_descriptive.py runs --gate1-15 runs/analysis/gate1-1.5b-2026-10-12-b/gate1.json --gate1-05 runs/analysis/gate1-0.5b-2026-10-12/gate1.json --qualification runs/judge_check/frozen-2026-10-11/qualification-both-rounds-2026-10-12.json --build runs/analysis/build-stage1-main-1.5b-2026-10-12 --out runs/analysis/descriptive-2026-10-12-r5`
- **Code:** 479d567, clean.
- **Inputs:** 662 files, each with its sha256 in the manifest.
- **`descriptive.json`:** sha256 `9ec1f41b476e2b849ea6bb630374dbe6d9190e69f30e9bcd376dc417d922b702`.
- **`manifest.json`:** sha256 `164f81ae54e853ff4c3408ef4e8f852784f04bc16a1552a6743751e84b74a0ba`.

All numbers in this report come from these files, unrounded until printed.
