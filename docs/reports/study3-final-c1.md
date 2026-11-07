# Study 3: the final report of candidate 1

Study 3 compared harm after 25 fine-tuning updates under three optimizers at matched benign drift:
A (AdamW), F (Adam with its second moment frozen at v₀) and P (AdamW whose second moment starts at
v₀). Candidate 1's comparison point is K\* = 6.00. The confirmation reached N = 60 blocks for both
comparisons, and both are read as confirmatory ([confirmation](study3-confirmation-c1.md)).

**Results, judge-rated harm on the 521 harmful prompts, at L = 0.985 per comparison:**

- **A–F (primary):** A gives 0.45 to 3.63 pp less harm than F at matched K_benign; the difference is
  below 5 pp.
- **P–F (secondary):** the difference is below 2.5 pp; the direction is undetermined.

Both utility gates pass. The judge audit, on a mixed reference with no arbiter decision, does not
exclude a judge bias as large as the A–F effect: its worst-case bound is wider than the difference
with the majority references and without them (below, "The judge audit").

The facts and tables of this report were written by `scripts/study3_analysis.py report` from the
records alone; they are in [`study3-final-c1-evidence/`](study3-final-c1-evidence/) (`facts.json`,
`tables.md`, with the billing and embargo files it read). The two copies differ from the written
files (sha256 `1920918f…d5d1` and `4f9ad00e…0502`) only in one local path, the runs root, shown as
`$RUNS_DIR`.

## The confirmatory comparisons

| Comparison | Blocks | Mean (pp) | Interval at L = 0.985 (pp) | Reading |
|---|---:|---:|---|---|
| A–F | 60 | −1.56 | [−3.63, −0.45] | A less harm; below 5 pp |
| P–F | 60 | +0.06 | [−0.22, +0.28] | below 2.5 pp; direction undetermined |

- **The interval** is the paired studentized bootstrap-t over the cohort's blocks, 100 000
  resamples, seed 20261105. L = 0.985 was fixed by the simulation's coverage check before
  validation.
- **Missing labels.** No run and no label is missing in either cohort, so the missing-labels
  envelope and the all-missing-harmful computation equal the primary interval.
- **The cohorts** are the analysis freeze's (`docs/study3-analysis-freeze-c1.json`, sha256
  `0e8f2640…5703`), fixed before any confirmation label was opened.
- **Block differences.** A–F ranges from −19.45 pp (seed 10400) to +4.10 pp (seed 10437); P–F from
  −3.52 pp (seed 10435) to +1.93 pp (seed 10418).
- **Without block 10400** (not registered, descriptive): 32 of the 60 A–F blocks have d < 0. Without
  block 10400, A–F's mean is −1.26 pp over 59 blocks, with the same bootstrap-t interval at L,
  [−2.84, −0.31] pp (100 000 resamples, stream `descriptive-without-10400`, [`af-without-10400.py`](study3-final-c1-evidence/af-without-10400.py)), which still
  excludes 0.

## Descriptive results

| Quantity | Blocks | Interval at L (pp) | Reading |
|---|---:|---|---|
| A − P | 62 | [−3.55, −0.39] | A less harm; below 5 pp |
| A–F at matched K_harmful (K\*_h = 5.876) | 34 of 60 | [−1.36, +0.57] | below 2.5 pp; direction undetermined |
| Validation's A–F | 27 | [−4.65, −0.35] | A less harm; below 5 pp |
| Validation's P–F | 26 | [−0.41, +0.35] | below 2.5 pp; direction undetermined |

- **At matched K_harmful,** 26 of the 60 A–F blocks do not bracket K\*_h in K_harmful for one arm
  or both and are left out; the 34 matched blocks give no direction. This describes how the shift
  is distributed across panels and does not replace the primary result.
- **Validation's harm** was read after the analysis was written and apart from it. One validation
  A–F run and two P–F runs each miss one label; the envelopes are in the tables.
- **Curvature.** At the realized bracket widths, the interpolation bias from harm's curvature in
  ln K, with the simulation fixation's κ per arm, is 0.013 pp for A–F (largest block 0.20 pp) and
  0.014 pp for P–F (largest 0.15 pp).
- **The started blocks outside each cohort.** Four per comparison: for A–F, 10428, 10447 and 10460
  (unsupported) and 10463 (supported, beyond N); for P–F, 10414, 10428, 10447 and 10460
  (unsupported). Taken at the extremes of ±100 pp, they move the mean to −7.71 or +4.79 pp (A–F)
  and −6.19 or +6.31 pp (P–F); the bootstrap envelope with them is at least ±43 pp wide. It bounds
  nothing useful and states only that each result is conditional on its cohort.

## Utility

Every gate's one-sided upper bound at α = 0.05/8 lies below its limit (MMLU 5 pp, factual 10 pp):

| Comparison | Arm | MMLU loss, mean (upper) | Factual loss, mean (upper) |
|---|---|---|---|
| A–F | A | 2.51 (2.84) | 0.25 (0.34) |
| A–F | F | 2.72 (3.13) | 0.28 (0.39) |
| P–F | F | 2.76 (3.18) | 0.28 (0.39) |
| P–F | P | 2.75 (3.19) | 0.29 (0.41) |

## The judge audit

One terminal draw after the analysis ([audit](../study3-preregistration.md#the-audit)): 400 draws
from A, 400 from F and 200 from P, 1000 distinct items, from 188 602 frame items.

**The reference is mixed.** Sonnet and Grok were given every item blind; GPT labeled their
disagreements and failed items; Claude Fable was the arbiter
([amendment 11](../study3-preregistration.md#amendment-11-the-audits-reference-annotators)).
The provider's safety filter stopped many calls of the Claude models
([amendment 12](../study3-preregistration.md#amendment-12-safety-filter-stops-in-the-audits-labeling)):

| Model | Calls stopped | Items without its label |
|---|---|---|
| Sonnet 5.5 | 8 of 40 annotator batches | 200 |
| Grok 4.7 | 3 of 40 annotator batches | 75 |
| GPT (gpt-6-astra) | 0 of 11 | 0 |
| Fable 5.1, arbiter | 11 of 11 | all 258 arbiter items |
| Fable 5.1, check | 9 of 10 | all 75 checked agreements of the final selection |

| Reference source | Items |
|---|---:|
| Agreement of Sonnet and Grok | 742 |
| Majority of the available votes after an arbiter stop | 215 (207 two of two, 8 two of three) |
| Fable's decision | 0 |
| Missing | 43 (26 without a majority, 17 never sent to the arbiter) |

The majority after an arbiter stop is a reference by the models' agreement, not an equivalent of
the arbiter's decision. The estimate's source "arbiter" counts these 215 items.

**Estimates.** B_E is the judge's implied bias in each comparison's Δ (missing at random, with its
worst-case bound):

| | A–F | P–F |
|---|---|---|
| B_E, MAR (SE), pp | −0.05 (1.23) | +0.25 (1.48) |
| Worst-case bound, pp | [−4.59, +3.24] | [−4.59, +5.99] |
| Missing draws | 31 | 31 |

| Arm | Sensitivity, MAR [worst case] | FPR, MAR [worst case] | Missing draws |
|---|---|---|---:|
| A | 0.840 [0.700, 0.846] | 0.009 [0.009, 0.015] | 12 |
| F | 0.884 [0.709, 0.889] | 0.010 [0.009, 0.019] | 20 |
| P | 0.862 [0.641, 0.867] | 0.006 [0.006, 0.012] | 11 |

**Sensitivity to the majority rule** (amendment 12, item 7): the same draw with those 215
references missing (258 missing):

| | A–F | P–F |
|---|---|---|
| B_E, MAR (SE), pp | +0.35 (1.13) | +2.00 (1.70) |
| Worst-case bound, pp | [−23.58, +28.06] | [−21.59, +28.07] |
| Missing draws | 205 | 147 |

- Under MAR, the implied bias is within about one standard error of zero with or without the
  majority rule (P–F without it: +2.00, SE 1.70). For A–F it is −0.05 pp (SE 1.23), an SE of the
  same order as the effect.
- Its worst-case bound, [−4.59, +3.24] pp with the majority references and about ±25 pp without
  them, is wider than the A–F difference (1.56 pp) in both cases: the audit does not exclude a
  judge bias as large as the effect. The stops fall on harmful texts, so the missing references
  are not missing at random, and the worst-case bound is the one to read.
- Fewer missing references make the reference neither more accurate nor its missingness random.

**The check of the agreements is unavailable.** It tried 75 agreements of the final selection and
labeled none: every call was stopped. This is no evidence that the consensus's error is zero.

**The labels' provenance** ([correction record](../study3-preregistration.md#correction-record-2026-11-07-the-audit-labels-provenance)).
When the safety filter stopped a call, the labeling harness continued it, with another model or the
same one. Those calls count as stopped and their outputs are discarded; every call was checked
against its log for the model that wrote it. The agreements step was recomputed twice on the same
draw. The audit labels and their summary counts existed when this was found
(865/10/125, then 841/9/150 agreements, disagreements and failed items); no audit result had been
seen and the estimate had not been run.

No claim is made about reference-rated harm.

## Limits

- Judge-rated harm (WildGuard) on the fixed 521 harmful prompts, Qwen2.5-1.5B-Instruct, B = 20 and
  25 updates, at candidate 1's rates; the results are stated for this benchmark, and how they
  extend to other models, data or training lengths is not tested here.
- The comparisons are means over each cohort's supported blocks; three or four started blocks per
  comparison were unsupported.
- No claim that the optimizer does not matter, and none about reference-rated harm.

## Records

- **Code.** The analysis at `fb6df93`, the audit's draw and steps at `fcc7ebe` and later, the
  report at `cf35f23`, all with the GPU code `fd54945` (identity `717cd023…bbc`), in the Pod image
  `ghcr.io/kaysinb/zero-field-refusal@sha256:9554679e…da5`, on CPU, without network.
- **Records** (sha256):
  - `analysis.json` `a639d564…f145`, `validation-harm.json` `7b5defd4…6f0d`;
  - the draw's `selection.json` `f4bb942c…ffd1`, `unique.jsonl` `95ca673b…cda9`;
  - `agreements.json` `a3565823…e4bc`, `arbiter.json` `801805e6…fdb8`;
  - `estimate.json` `2b308736…0feb`; the sensitivity's `estimate.json` `b3c25a0d…89e7`;
  - the labeling's sealed record `study3-labeling-20261107T132754.304347Z-wildguard.json`
    `35f93d59…e2d5`.
- **Archive.** Under `$ARTIFACT_REMOTE/study3/confirmation/study3-confirmation-c1-20261105/`:
  `labels/`, `analysis/`, `audit/` (the draw, each recomputation of the agreements and the
  sensitivity), each with its inventory. The superseded agreement and arbiter records are kept
  with their sha256.
- **Labeling.** The 372 confirmation runs the freeze names were labeled in 19 batches, every
  control check passed and no judge mark was set.

## Spend

Billed, by stage, in USD:

| Stage | Billed |
|---|---:|
| Dry run | 12.35 |
| Engineering | 45.58 |
| Simulation (run on a local CPU) | 0 |
| Search | 4.21 |
| Numerical qualification | 16.38 |
| Validation | 21.91 |
| Confirmation | 56.42 |
| Labeling (two judge Pods) | 11.05 |
| Audit (model annotators through their command-line tools, no billed amount) | 0 |
| **Total** | **167.90** |

Incidents, each included in its stage:

| Incident | Stage | Billed |
|---|---|---:|
| Case 17's diagnosis and the engineering re-run under fix `8dcfe20` | engineering | 28.24 |
| A failed image pull | dry run | 1.03 |
| A network file system as the Pod's volume | dry run | 1.14 |
| The same | engineering | 0.12 |
| The same | confirmation | 0.17 |
| The labeling read the freeze gate once per run until fix `860700a` | labeling | 1.50 |
