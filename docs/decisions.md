# Decision log

Append-only, newest entries last.

## Gate decisions

One entry per gate decision. Each entry names its evidence: the run ids and the gate report in
[reports/](reports/README.md), so the decision can be audited later against the criteria in
[preregistration.md](preregistration.md#gate-thresholds).

| Date | Gate | Evidence | Result against thresholds | Decision |
|---|---|---|---|---|
| 2026-10-12 | Gate 1 | The 1.5B AdamW 5·10⁻⁵ runs `20261011T105906Z-adamw-5.0e-5-1.5b-s0-dd83f41e-a1`, `…-s1-14107220-a1`, `…-s2-3932d764-a1`, `…-s3-c848e2a3-a1`, `…-s4-6f1639c7-a1` and every other stage-1 run directory (in the bucket); the Gate 1 report of `scripts/gate1_report.py` at 69ed300 (`gate1.json`, sha256 `1df352859df12a31…`), with the judge calibration of `runs/judge_calibration/calibration`; the [stage-1 report](reports/stage-1.md); the short report of the corrected pilot (`reports/gate-1.md`) follows | Effect criterion not met: mean refusal drop 44.9 pp (≥ 25 met), lower bound of the 95 % t-interval 14.44 pp < 15; harmful rise [13.6, 69.5] pp and factual drop 1.7 pp met. Judge calibration 0.945 ≥ 0.90. Coverage not met (route fallback, moot). The effect does not disappear with the constant denominator. Provisional on the judge check across optimizers, undetermined after its first round | Stop, by the user's decision of 2026-10-12, with the short report of the corrected pilot. The pre-specified second round of the judge check is drawn to close the procedure |

## Study decisions

One entry per decision that stops, keeps or starts a study, with its reasons and evidence.

| Date | Decision | Reasons | Evidence |
|---|---|---|---|
| 2026-10-14 | The pilot (study 1) stays stopped at Gate 1; the stop decision of 2026-10-12 stands. Its design was a poor one and did not yield a decisive, interpretable optimizer comparison. In this repository, study 2 begins under a new design, approved by the user on 2026-10-14, with its own [preregistration](study2-preregistration.md), fresh seeds and the namespace `study2_*`; the pilot's data serve only to plan it. | (1) A shared budget of 25 steps × batch 20 for optimizers whose step scales differ left SGD one lever, its learning rate, which met a stability cliff: stable SGD reached K_harmful 2.41, below AdamW's effect zone from 4.80. (2) Gate 1 sat in a collapsed cell (MMLU −22.2 pp) with 5 seeds and a lower-bound criterion. Under stated distributional assumptions, a simulation at the observed seed SD of 24.5 pp gives a real effect of the observed size a pass probability of ≈ 0.54. (3) The soft-target control at 0.028, the largest rate called stable, grew ≈ 9–14× per step and reached 30 % of the regular run's K_harmful (42 % of its K_benign): that rate was linearly unstable. The threshold was not measured, so whether every nominally stable rate exceeded it is unknown. (4) The K\* rule placed the comparison at K\* = 2.26, where observed harm was low in every arm and discrimination weak. (5) Matching on K_harmful is outcome-adjacent, since θ₀'s responses there are refusals; that it attenuated the contrast is a hypothesis. | The [short report](reports/gate-1.md) and the [stage-1 report](reports/stage-1.md); the Gate 1 decision above; the study-2 design record of 2026-10-14 (coordination notes and the reviews behind them) |
| 2026-10-15 | Study 2's monitor panel is replaced during Stage A's discovery, before fixation and with the harm labels closed ([amendment 1](study2-preregistration.md#amendment-1-the-monitor-panel)): fresh θ₀ responses to the scoring panel's own 100 prompts, one per prompt read every 5 steps (m = 1), a nested second as the declared fallback (m = 2), the scoring panel's K alone every 25 steps, θ₀'s logits cached and checked against uncached readings. m = 1 is chosen if it passes the agreement rule for both optimizers at the selected batch on the version-2 discovery runs, else m = 2, else the monitor is redesigned before validation. The long arms are searched again under version 2, rate by rate downward on all three discovery seeds (U = 8 first), audited by a conditional manifest; the version-1 runs stay observations with their measured K. | The 25-prompt monitor read 13.3 % below the scoring panel at the median (p95 17.5 %; bounds 5 % and 15 %), low at all 14 states, so that the upper level was evaluated near 1.15 U, outside the fit band. Its shorter responses explain about half; the rest is the composition of its prompts, which a separate panel keeps at any size through the scoring panel's own offset (≈ 3.5 %). The estimand is conditional on the scoring prompts, so sharing them with independent responses removes the mismatch without selecting on the covariate's own noise. | K only: the 14 full evaluations of the AdamW-long runs at B = 5 (seeds 2001, 2002) with the scoring panel's K per response, the panels' response lengths and the runs' step times (`scratchpad/analysis/stage-a-2026-10-15/`, archived with the stage); the second reviewer's reviews r18, r18b and r19 in the coordination notes |

## Protocol versions

`protocol_version` in `configs/base.yaml` is part of every run's config hash
([why](architecture.md#configuration)). Each bump adds a row.

| Version | Date | What changed | Why |
|---|---|---|---|
| 1 | 2026-10-11 | Initial protocol, fixed for stage 1 ([preregistration](preregistration.md#fixed-protocol)) | Stage 0 settled the thresholds, the judge and the stage-1 interval |
| 2 | 2026-10-14 | Study 2 (`experiments/study2_*`): appendable training streams read in order, soft targets replayed instead of stored, the monitor panel, K-level schedules and stops, the harmful evaluation set of 521 prompts | The [study-2 preregistration](study2-preregistration.md); study 1's configs stay at 1. The stage-0 path-field record is keyed by the protocol version, so the checks run again under 2 |
