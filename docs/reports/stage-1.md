# Stage 1: runs and mechanics

This report records how stage 1 ran on 2026-10-11: compute, runs, the choice of the sweep rates,
cell stability, the 0.5B SGD diagnosis, the judge labeling, incidents and costs. It summarizes no
judge label and no refusal outcome. Gate 1 and the outcomes wait for two decisions of the user:
the proposed amendment on eligible cells, and the reference ensemble of the
[judge check](../preregistration.md#judge-check-across-optimizers). Aggregates only.

## Compute and provenance

- **Training Pod** `32fjt8f6671w17`: one L40S (Secure Cloud, US-TX-4, driver 595.91.07, host
  CUDA 13.2), image `ghcr.io/kaysinb/zero-field-refusal@sha256:9554679e…` (env-v1), 120 GB
  volume disk; created 10:37 UTC, stopped 21:00 UTC, deleted with its volume after the archive
  was verified ([storage](../operations.md#storage)).
- **Code:** a clean checkout of `29f1872` from a git bundle. The campaign files were committed
  separately (`f76e891`, `37a984e`, `56790e8`, `6875ff6`, `69ce39f`) and run from copies with
  verified sha256. Later commits change analysis code only, not training.
- **Streams generated:**
  - training seeds 3–4 on both models (seeds 0–2 restored from stage 0, same spec);
  - preconditioner seeds 0–4 on 1.5B;
  - the full soft-target stream of 1.5B seed 0 (`73cbbd57f829`, 500 responses). Its ≈ 23 GB of
    distributions were deleted after its run; the manifest and samples are kept.

  Each stream was archived before the runs that read it.
- **Archive:**
  - Every run directory, stream (small files), launch log, attempt record and campaign file is
    in the bucket. Each object was checked against its local file by size and MD5; the
    inventories are in `runs/archive/`.
  - The copy on this machine was checked by sha256: 2440 files. Not copied: the 144
    `wandb/.../debug-core.log` symlinks; their content is in the bucket.
- **Hard stop:** an in-Pod deadline watcher ran throughout, deadline-only. Its deadline was
  extended several times, each extension verified.

## Runs

| Campaign | Window (UTC) | Runs | Ended |
|---|---|---|---|
| `stage1-a` (phase A: AdamW, branches, θ₀, pilots on the base grids) | 10:55–14:39 | 71 | 71 completed |
| `stage1-a2` (first outer extensions) | 14:39–14:57 | 6 | 5 completed, 1 diverged |
| `stage1-a3` (second outer extensions) | 15:01–15:13 | 5 | 3 completed, 2 diverged |
| `stage1-a4` (first refinement round) | 15:15–15:40 | 6 | 6 completed |
| `stage1-a5` (second refinement round) | 15:43–16:09 | 6 | 6 completed |
| `stage1-b` (phase B: sweeps at seeds 0–4, scans) | 16:22–20:45 | 49 | 49 completed |
| `stage1-b-soft` (soft-target control, 1.5B) | 20:52–20:56 | 1 | 1 completed |

144 runs: 141 completed and 3 diverged. The diverged runs are SGD at η = 27 on both models and
SGD under 100 × 5 at η = 3, all seed-0 pilots. Every run ended on its first attempt: there was
no technical failure and no recovery round.

| Experiment | 1.5B | 0.5B |
|---|---|---|
| main condition (θ₀, AdamW, SGD and frozen-Adam pilots and sweeps) | 68 + 1 diverged | 19 + 1 diverged |
| token-count normalization (AdamW 5·10⁻⁵) | 5 | 5 |
| bridge cell, 48 tokens (AdamW 5·10⁻⁵) | 5 | 5 |
| SGD 100 × 5 | 28 + 1 diverged | — |
| line scans and random directions | 4 | 1 |
| full soft-target control | 1 | — |

**Run times on the L40S:**

| Run | Time |
|---|---|
| 1.5B, main condition | 3.4 min |
| 1.5B, frozen Adam | 4.4 min |
| 1.5B, SGD 100 × 5 | 4.5 min |
| 0.5B | 1.9 min |
| Scan, 1.5B (nine response conditions) | 18.4 min |
| Scan, 0.5B | 12 min |
| Soft-target control | 4.0 min |

## Choice of the sweep rates

The rates were chosen by the [stage-1 sweeps rule](../preregistration.md#stage-1-sweeps), with
the refinement and the definition of "reach" added on 2026-10-11, from seed-0 pilots and from K,
divergence and collapse alone. A pilot is its own cell.

- **θ₀ utility:**

  | Model | Factual | Benchmark |
  |---|---|---|
  | 1.5B | 0.9955 | 0.568 |
  | 0.5B | 0.964 | 0.422 |

- **AdamW K range read by the rule** (cell means of the completed AdamW cells on 1.5B):
  1.86 (5·10⁻⁶) to 16.45 (5·10⁻⁵).

| Sweep | Stable pilots (K) | Collapsed or diverged pilots |
|---|---|---|
| SGD 25 × 20, 1.5B | 1/300 (0.12), 0.01 (0.63), 0.0132 (0.95), 0.0173 (1.41), 0.0228 (1.87), 0.0244 (2.00), 0.0262 (2.05), 0.0280 (2.20) | 0.03, 0.1, 0.3, 1, 3, 9 (K 333–411, factual collapse); 27 diverged |
| Frozen Adam, 1.5B | 5·10⁻⁶ (3.15), 1·10⁻⁵ (6.0), 2·10⁻⁵ (10.1) | 5·10⁻⁵, 1·10⁻⁴ (K 350, 483) |
| SGD 100 × 5, 1.5B | 0.01 (3.29), 0.0132 (4.05), 0.0173 (4.63), 0.0186 (5.13), 0.0199 (5.13) | 0.0213, 0.0228, 0.03, 0.1, 0.3, 1, 9 (K 245–312); 3 diverged |
| SGD 25 × 20, 0.5B | none | 1/300 to 9 (K 558–1490); 27 diverged |

**Rounds:**

- Phase A piloted the base grids.
- `stage1-a2` and `stage1-a3` extended the grids outward:
  - SGD and SGD 100 × 5 by factors of 3 at both ends;
  - frozen Adam by a factor of 2 downward.
- `stage1-a4` and `stage1-a5` refined SGD (both conditions) inside the cliff between the largest
  stable and the smallest collapsed pilot:
  - first round: 0.0132, 0.0173, 0.0228 between 0.01 and 0.03;
  - second round on SGD 25 × 20: 0.0244, 0.0262, 0.0280 between 0.0228 and 0.03;
  - second round on SGD 100 × 5: 0.0186, 0.0199, 0.0213 between 0.0173 and 0.0228.

**Chosen rates** (accepted by the coordinator before phase B):

- **SGD 25 × 20, 1.5B:** 0.0228, 0.0244, 0.0262 and 0.0280, the largest stable rate.
- **Frozen Adam:** 5·10⁻⁶, 1·10⁻⁵ and 2·10⁻⁵.
- **SGD 100 × 5:** 0.01, 0.0132, 0.0186 and 0.0199.
- **SGD on 0.5B:** no stable rate. So there is no SGD sweep and no soft-target control on 0.5B.
- **Line scans:** AdamW 5·10⁻⁵ on both models, and SGD 0.0280 on 1.5B.
- **Soft-target control:** 1.5B at 0.0280.

## Cell stability

Collapse is judged on cell means, by the [fixed definition](../preregistration.md#primary-analysis):
the factual drop must not exceed 10 pp and the benchmark drop must not exceed 5 pp below θ₀. Drops
are in pp below θ₀. A cell is eligible when seeds 0–4 completed, none diverged and it did not
collapse.

| Cell (1.5B unless noted) | K (geometric mean) | Factual drop | Benchmark drop | Status |
|---|---|---|---|---|
| SGD 25 × 20: 0.0228 / 0.0244 / 0.0262 / 0.0280 | 1.92 / 2.06 / 2.21 / 2.41 | 0.0–0.1 | 1.0–1.7 | eligible |
| Frozen Adam: 5·10⁻⁶ / 1·10⁻⁵ / 2·10⁻⁵ | 2.13 / 4.54 / 8.61 | 0.0–0.6 | 1.8 / 1.8 / 4.9 | eligible |
| AdamW: 5·10⁻⁶ / 1·10⁻⁵ / 2·10⁻⁵ | 1.86 / 3.86 / 7.13 | 0.0–0.4 | 1.7 / 2.0 / 3.6 | eligible |
| AdamW 5·10⁻⁵ (the Gate 1 effect cell) | 16.45 | 1.7 | 22.2 | collapsed (benchmark) |
| SGD 100 × 5: 0.0132 / 0.0186 / 0.0199 | 4.02 / 5.19 / 5.51 | ≤ 0 | 2.0 / 3.0 / 3.2 | eligible |
| SGD 100 × 5: 0.01 | 7.51 | 20.0 | 9.2 | collapsed (one seed at K ≈ 300) |
| Token count, AdamW 5·10⁻⁵ | 16.3 | 1.8 | 23.2 | collapsed (benchmark) |
| Bridge 48, AdamW 5·10⁻⁵ | 18.5 | 1.7 | 25.3 | collapsed (benchmark) |
| 0.5B AdamW 2·10⁻⁵ | 10.4 | 0.1 | 0.4 | eligible |
| 0.5B AdamW 5·10⁻⁵ | 27.2 | 9.0 | 10.8 | collapsed (benchmark) |
| 0.5B token count, AdamW 5·10⁻⁵ | 25.4 | 5.8 | 9.6 | collapsed (benchmark) |
| 0.5B bridge 48, AdamW 5·10⁻⁵ | 32.8 | 20.8 | 18.8 | collapsed (both) |

**What follows:**

- **The Gate 1 rate.** At η = 5·10⁻⁵, AdamW loses 9.6–25.3 pp of benchmark accuracy in every
  cell, on both models and in both branches. A Gate 1 result at this rate has to disclose this.
  It cannot support a claim that general utility is preserved.
- **K_hi.** By the fixed text, K_hi is taken among stable AdamW cells.
- **K\* from all runs vs. from eligible cells.** Over all main-condition runs, K\* is 5.91: the
  collapsed pilots and cells widen the common range. Over eligible cells only, K\* is 2.26, with
  common range [2.13, 2.41].
- **The proposed amendment.** It would base the primary common range, K\*, the decisive cells,
  the curves and the fallback comparator on eligible cells. It awaits the user's decision.

## SGD on 0.5B

SGD collapses on 0.5B even at η = 1/300, where 1.5B is stable. A gradient diagnosis on the local
GPU (θ₀, the seed-0 stream, no labels) found no defect in the code:

- **No double updates.** Input and output embeddings are tied and frozen (zero gradient), so
  nothing is updated or counted twice.
- **No clipping.** No gradient clipping is configured, as intended.
- **Larger gradients.** At θ₀ the 0.5B gradients are about twice those of 1.5B (attention 4.13
  vs. 1.78, MLP 5.90 vs. 2.82), on smaller parameter norms.
- **One tensor runs away.** Over 25 SGD steps at 1/300 the 0.5B loss rises from 1.47 to 13.7.
  The layer-0 value bias moves by 82 % of its norm and the layer-0 value weight by 14 %, while the
  total displacement is 0.35 %. On 1.5B the loss falls from 0.98 to 0.86 and no tensor moves by
  more than 0.07 %.

It is a local SGD instability along a high-curvature direction, which Adam's per-coordinate
scaling avoids.

## Judge labeling

- **Server:** WildGuard served by vLLM on a separate L40S Pod (`l6e8as7qj600qr`), as in
  [operations](../operations.md#judge-server).
- **Hard stop:** the Pod had no SSH, so it ran with the start-command deadline watcher. The
  watcher's key check and the deadline were verified at start. The Pod was stopped by hand after
  labeling and then deleted.
- **Labels:** the 141 completed runs give 54 300 labels, 300 per run and nine response conditions
  per scan. The token check passed (one BOS).
- **Unparseable outputs:** 2060.
  - Where: all in 8 collapsed runs, namely seven collapsed seed-0 pilots and the collapsed seed
    of SGD 100 × 5 at 0.01.
  - Why: their responses are a single repeated character. WildGuard answers outside its format,
    and a retry returns the same outputs.
  - Effect: none of these runs is in the judge-check sample.
- **Judge-check sample:** frozen in `0752c68` from K, divergence and utility before any label was
  opened. It has 9 strata, 900 items and 551 distinct texts
  ([selection](stage1-judge-check-selection.json)). Its reference labeling is pending.

## Incidents and records

- **Phase-B chain script.** The attempt shell lacked `DATA_DIR`, so the chain stopped after
  generating the soft-target stream. A second attempt archived the stream and ran the control;
  no run was affected.
- **Runpod proxy.** It answers 403 to the `Python-urllib` User-Agent. The judge client is not
  affected.
- **Leftover AdamW keys.** SGD runs under 100 × 5 and of the soft-target control carry AdamW keys
  (betas, eps, moment dtypes) in their optimizer section, left there by the defaults
  composition. SGD ignores them (momentum 0, weight decay 0). They are left as run, and Gate 1
  checks the composition as the campaigns make it.
- **Calibration record.** The recorded model versions of the calibration reference were corrected
  on 2026-10-12 (`a5440a2`): a correction of the record, not of the protocol.

## Costs

| Item | Cost | Source |
|---|---|---|
| Training Pod `32fjt8f6671w17` | $11.56 (GPU $11.31, disk $0.24) | measured, Runpod billing for 2026-10-11 |
| Judge Pod `l6e8as7qj600qr` | ≈ $0.70 (38.5 min at $1.09/h) | estimated; not yet in the billing records |
| Bucket storage | — | not measured |
