# Study 2: feasibility report

> **Report of the stop, 2026-10-25.**
>
> **The study stopped at a technical numerical-qualification block; the primary comparison was
> not performed.** FA at 1.75·10⁻⁵ failed the archived-run consistency check in both permitted
> attempts. Its perturbation tests were not reached. The prescribed re-selection and repair
> produced no admissible replacement triple.
>
> The selection's own status stays `blocked` (technical) at the primary pair at (U, B) = (7, 20)
> ([amendment 5](../study2-preregistration.md#amendment-5-the-candidate-order-at-batch-20-after-u-8)).
> On 2026-10-25 the researcher decided to end the study there, without a further attempt, and
> to start a new preregistered study on fresh seeds
> ([decision log](../decisions.md#study-decisions)).
>
> No study-2 harm label was made or opened. Aggregates only.

## What study 2 asked

The [preregistration](../study2-preregistration.md) asked whether the optimizer changes harmful
compliance at matched benign drift K = K_benign (the KL to θ₀ on a fixed panel of 100 benign
prompts) under the zero-field protocol: Qwen2.5-1.5B-Instruct fine-tuned on θ₀'s own responses,
with harm judged on 521 harmful prompts.

After Stage A, amendment 3 redirected the primary question to AdamW-25 against Adam with its
second moment frozen at v₀ (FA-25). Both run exactly 25 steps, and harm is interpolated to
K\* = √(5U) in ln K between three targeted rates per arm. Amendment 4 fixed B = 20 and added a
mechanism cohort with pre-warmed AdamW-25 (P: second moment started at the same v₀ and updated).
It also replaced the numerical acceptance with a twin test. Amendment 5 fixed the order of
candidates: the three arms at U = 8, 7 and 6, then the primary pair at U = 7 and 6, else this
report.

## Course of the study

| Stage | Outcome | Kind |
|---|---|---|
| Stage A, long regime (2026-10-14 to 16) | No accepted SGD-long rate reached U at either batch: `barrier` under amendment 2, `numerics_fail` under the preregistered rule. The SGD-momentum pilot also ended `barrier`. Both readings are kept ([K-only analyses](../study2-stage-a-analysis.md)). | scientific, on the tested policies |
| Amendment 3, B = 5, U = 8 (2026-10-16) | Both primary arms were feasible in the targeted round. FA's soft-target controls failed at 1.21 and 1.46·10⁻⁵ (contamination rule). The [outcome-blind diagnostic](../study2-fa-control-diagnostic.md) traced the residual gradient to targets and logits computed on different paths. The tested perturbations were not amplified. | the control did not measure what it was meant to; replaced in amendment 4 |
| S1, discovery at (8, 20) (2026-10-20) | 33 runs on seeds 2000–2002. A triple was found for each of the three arms. | — |
| S2, numerical qualification at (8, 20) | First check: two rates not qualified (technical) under the then bitwise twin-A gate. After the twin-A rule changed to a numerical tolerance, the first check was voided and repeated once: 9 of 9 rates qualified. | technical, then passed |
| Simulation and gate C (2026-10-21/22) | The analysis, simulation and audit were approved. The mechanism cohort's level was fixed at 0.975. | — |
| S3, three-arm validation at (8, 20) (2026-10-22) | 108 runs on seeds 2100–2111. Support held in 12 of 12 blocks, and no run stopped by an instability rule. **Utility failed** (below). | scientific (utility gate) |
| Three arms at (7, 20) (2026-10-22/23) | Discovery passed. At the numerical stage, **P at 1.83·10⁻⁵ failed** and **FA at 1.75·10⁻⁵ was not qualified (technical)** (both below). After the search allowance and its repair round, FA and P had no admissible triple. | scientific (P) and technical (FA) |
| Three arms at (6, 20) (2026-10-23) | **No admissible FA triple** after discovery and its repair round (below). | scientific (selection) |
| Primary pair at (7, 20) | `blocked`: the pair inherits the latest three-arm selection at U = 7. After exclusion and repair, FA had no admissible triple; the exhausted rules and the unresolved technical mark of FA 1.75·10⁻⁵ blocked further selection. The pair at (6, 20) is ordered after it and was not visited. | technical |

## The block: FA at 1.75·10⁻⁵, U = 7

- **The case.** FA, 1.75·10⁻⁵, B = 20, numerical seed 2201. K\* = √35 ≈ 5.916, and the
  perturbation bound is 0.01·K\* ≈ 0.0592.
- **The check.** Twin A is a fresh re-run of the archived K-only run of that seed and rate. It
  must agree with the archived run within |x − y| ≤ 10⁻⁸ + 10⁻⁴·max(|x|, |y|), field by field
  and step by step
  ([rule](../study2-preregistration.md#1-numerical-qualification-replaces-amendment-3s-fa-soft-control-rule)).
- **What failed.** The gradient norm at steps 21, 22 and 23 differed by 1.10, 3.50 and 1.27 times
  the tolerance. At step 21, for example, the rerun read 2.756686 and the archive 2.756989.
  - Every other compared value was within the tolerance: loss, NLL per token and the monitor's K
    at every step, and the gradient norm at steps 1–20 and 24–25.
  - The two permitted attempts gave bit-identical traces, so the case stayed incomplete after its
    rerun.
  - The perturbation pairs (p1–p4) and the float64 reference were therefore never run for this
    case. On seeds 2200 and 2202 the same rate's cases completed within the bound.
- **The consequence.** Under the rule the rate became unusable at B = 20, with a technical mark.
  The prescribed search allowance re-solved FA's selection without it. Its repair round, at
  1.96·10⁻⁵ on seeds 2000–2002 (step-25 K 7.43, 7.88 and 8.50), gave no triple that met the
  selection's rules jointly. The pair inherits the latest three-arm selection. After exclusion
  and repair, FA had no admissible triple; the exhausted rules and the unresolved technical mark
  blocked further selection, so the pair is `blocked`.
- **Execution paths.** The archived run came from the launcher, and twin A from the check's own
  training loop, the diagnostic's harness, which also serves the perturbation pairs. These paths
  differ in process and in loop code, though both accumulate gradients the same way over fixed
  micro-batches. Whether this difference caused the discrepancy was not established. The
  discrepancy shows neither amplification nor harmless rounding.

## Separate outcomes

### Utility at U = 8

S3 validated the three arms at S1's rates on 12 fresh blocks. The family of nine one-sided upper
bounds on mean MMLU loss against θ₀ (level 0.95, α ≈ 0.0056 per bound) failed at three rates:

| Arm | Rates (·10⁻⁵) | Upper bound on mean MMLU loss (pp) | Bound on factual loss (pp) |
|---|---|---|---|
| AdamW-25 | 1.49 / 1.83 / 2.24 | 3.07 / 3.22 / 4.12 | ≤ 1.27 |
| FA-25 | 1.34 / 1.62 / 2.0 | 3.42 / **5.79** / **6.59** | ≤ 1.51 |
| Pre-warmed AdamW-25 | 1.34 / 1.62 / 2.0 | 3.51 / **6.26** / 4.41 | ≤ 1.62 |

- The thresholds were 5 pp for MMLU and 10 pp for factual accuracy.
- Support held in 12 of 12 blocks for every arm.
- The pair alone, recomputed on the same runs over six bounds, also failed: FA 5.62 and 6.40 pp at
  1.62 and 2.0·10⁻⁵.
- The gate applied to every endpoint, including the upper rate above K\*. It is a validation gate
  on these rates, not an estimate of utility at K\*.

### Pre-warmed AdamW at U = 7: a transient sensitivity to one stress direction

- **The case.** P at 1.83·10⁻⁵, seed 2201. Its pair p2 is the negative of the FA-preconditioned
  first update of the seed's generation-path soft residual, at norm 5·10⁻⁵.
- **What it measured.** D_twin (the larger directional full-vocabulary KL on the 400 scoring
  responses) read:

  | Step | 1 | 5 | 10 | 15 | 20 | 25 |
  |---|---|---|---|---|---|---|
  | D_twin | 0.0003 | 0.0006 | **0.151** | 0.040 | 0.041 | 0.034 |

  - The bound was 0.0592.
  - In the same case, p1 stayed at or below 0.0033, the Gaussian pairs below 3·10⁻⁶ and the
    float64 reference below 10⁻⁷.
- **The verdict.** Study 2's criterion takes the maximum over the measured steps, so the rate was
  failing: a scientific mark that makes it unusable at B = 20.
- **What this shows.** The run is sensitive to that direction in the first half of training. The
  difference falls back below the bound by step 15 but stays about ten times p1's.

### U = 6: the selection was infeasible

- At U = 6 the targets are 5, √30 ≈ 5.48 and 6, about 0.09 apart in ln K, inside the selection's
  coverage tolerance of 0.1.
- AdamW found a triple: 1.49, 1.5 and 1.83·10⁻⁵.
- FA found none that met the rules jointly, after discovery and its repair round (1.11 and
  1.88·10⁻⁵). Its seed-to-seed spread of step-25 ln K on the discovery seeds, about 0.2 at a
  fixed rate, is about twice the spacing of the targets.
- P had no admissible triple either (support and width).

## What is not concluded

- **Not** that FA, or any arm, is numerically unstable. The FA mark is a technical
  reproducibility failure whose cause is unknown. P's mark records that a stated criterion was not
  met at one rate and one seed.
- **Not** that the optimizer has no effect on harmful compliance, and **not** that the arms are
  equivalent. No harm comparison was made.
- **Not** that the short regime is infeasible in general. The tested candidates at B = 20, under
  this preregistration's rules, did not pass its safeguards.

## What the study measured

These are K, utility and numerical records. They inform later designs and are not tests of the
preregistered question.

- **Rate to K at 25 steps, B = 20.** Step-25 ln K rises almost linearly in ln η, with a slope of
  about 1.25 (AdamW), 1.30 (FA) and 1.24 (P). The seed offset is shared across rates: an SD of
  0.06–0.08 in ln K.
- **Utility near the zone.** Mean MMLU loss rose with ln K faster for FA and P than for AdamW
  (fitted slopes about 2.2, 2.7 and 1.5 pp per unit ln K), with a heavy tail across seeds: 14 of
  232 runs lost more than 5 pp, none more than 15 pp.
- **Numerical reproducibility at fp32.**
  - In the S2 repeat, twin A was bit-equal in 8 of 15 cases and within the tolerance in all 15.
    The largest relative deviation was 2.2·10⁻⁵.
  - At U = 7, twin A was within the tolerance in 14 of 15 cases. The exception was the FA case
    above.
  - Apart from P's p2, every measured perturbation pair stayed at least four times below its
    bound.
- **Infrastructure.** A simulation of the selection and analysis with a conditional truth and a
  calibrated level, sealed labeling, an audit design drawn by coefficient, and versions records
  that fail closed.

## Harm outcomes

The regular runs of study 2 generated sealed greedy responses to the 521 harmful prompts, together
with their K_harmful. None was judged or labeled, and no response text was read in this work.
Every decision of study 2 used K, numerical records, utility and timings only. The sealed
responses are not part of this report.

## Records

- **Selection, after the last campaign:** `runs/analysis/study2-short-summary-20261023-u6-repair.json`.
- **Utility at U = 8:**
  - S3: `runs/analysis/study2-short-summary-20261022-s3.json`;
  - the pair's recomputation: `runs/analysis/study2-short-pair-u8-recheck-20261022.json`.
- **Numerical verdicts:**
  - U = 8 (repeat): `runs/numerical/b20-u8/verdicts/f7cf42e3f76f.json`;
  - U = 7: `runs/numerical/b20-u7/verdicts/146cd8b89969.json`, with its case records under
    `runs/numerical/b20-u7/twins/`.
- **The voided first S2:** `runs/void/78b166a28a61/`.
- **Archive.** Every run, stream, journal and check is archived with an inventory under
  `runs/archive/`. The last summaries and campaign files were added on 2026-10-25
  (`inventory-2026-10-25-study2-close.tsv`), and every inventory was then checked against the
  bucket's sizes and MD5.
- **Code:** stage0 at `028b953` (the versions record after amendment 5). The U = 7 candidate ran on
  `6fb20a9`.

**Scope.** Everything above concerns the tested arms, rates and seeds at 25 steps on
Qwen2.5-1.5B-Instruct, with harm judged on the 521-prompt benchmark. Whether it holds for other
models, prompt sets or regimes was not tested.
