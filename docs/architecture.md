# Architecture

This document fixes the engineering design decisions that are expensive to change later. The
scientific design is in [design.md](design.md); procedures for running things are in
[operations.md](operations.md).

## Principles

- **One run, one directory.** Each run saves its config, provenance, measurements and outputs.
  Analysis traces reported values to these saved records.
- **Every experiment is a config.** Sweeps are lists of configs, not code edits. Record executed
  source, input and environment identities under [Publication evidence](#publication-evidence).
- **Protocol invariants live in code, with tests.** Violating one silently turns the zero field
  into a real field, and the results still look plausible. Stage 0 must implement a check in the
  code path and a unit test for each invariant below before experimental runs.
- **Measure in the same process; store weights only on request.** A 1.5B checkpoint is several
  gigabytes and the study has hundreds of runs. Evaluations, line scans and perturbation controls
  run right after training, while the weights are still in memory.

## Data flow

1. **Prompt sets.** Defined in `configs/data/`, each pinned by dataset revision (or source URL
   and file hash) and selected indices, with its manifest saved next to the data. The factual
   questions are a versioned resource in the package; an edited resource gets a new version name:
   - Alpaca instructions without an input field, split into three disjoint sets of 100: teacher
     prompts, benign KL prompts and held-out reference answers; study 2 (`data/study2`) adds 25
     monitor prompts (`alpaca_monitor`), taken next in the same permutation, so the other sets
     are unchanged;
   - harmful prompts: HarmBench standard behaviors and AdvBench, deduplicated, split into 100 KL
     prompts and 300 judged prompts; the rest is reserved for judge calibration. Study 2
     (`data/study2`) evaluates on `harmful_evaluation` instead: the judged and calibration
     prompts cleaned of near-paraphrases and of prompts near a KL prompt (word-set Jaccard
     ≥ 0.6, in source order), 521 prompts;
   - at least 200 simple factual questions and 500 MMLU questions, stratified by subject.
2. **Teacher samples.** θ₀ samples responses for one (model, prompt set, seed) and stores them in
   `data/teacher/`, one directory per stream and one record per response. A stream is written into
   a partial directory and published by renaming it when complete, so an interrupted generation
   never leaves a stream that looks complete; partial directories are removed explicitly. Each
   response's instruction is drawn uniformly with replacement from the teacher prompts. A record
   holds:
   - the prompt token ids, including the chat template's default system prompt, and the response
     token ids; never only text;
   - per-token log-probabilities under the teacher, from raw logits;
   - the effective sampling settings (see [generation settings](#generation-settings)).

   For the soft-target streams, the full next-token distributions are also recorded at sampling
   time, for the [path-field check and the soft-target control](#zero-field-invariants). An
   appendable stream stores none: its soft-target control recomputes them during the run
   ([soft-target storage](#design-decisions-for-stage-0)).

   Each seed has two disjoint streams of fresh responses:
   - **training stream:** 500 responses (steps × batch size) from `derive_seed(seed, "teacher")`,
     identical for all optimizers;
   - **preconditioner stream:** 50 × 20 = 1000 responses from `derive_seed(seed, "preconditioner")`.
     Adam with frozen statistics and AdamW with a pre-warmed second moment share this stream and
     the same estimate at θ₀. A separate stream keeps the training stream identical across all
     optimizers. Their initialization and bias correction follow
     [H2a](preregistration.md#hypotheses).

   **Appendable training streams** (study 2, `teacher.stream: appendable`) have no fixed length.
   A seed's stream is a sequence of chunks of `teacher.chunk_size` responses in
   `<stream>/chunk-00000/`, `chunk-00001/`, …, each published atomically like a whole stream,
   with the fingerprints of its distributions for the soft-target replay.
   Chunk j draws its instructions and samples from `derive_seed(seed, "teacher", "prompts",
   "chunk", j)` and `derive_seed(seed, "teacher", "sampling", "chunk", j)` alone, so appending
   never changes earlier chunks, and the stream's identity holds no length. Runs read the stream
   in order, without a permutation (the instructions are already iid draws): after t steps of
   batch B a run has used responses 0 … tB − 1, so runs of either batch size and of any length
   share one prefix. A run records the hashes of the chunks it read. A stream has one writer:
   runs that may extend it run on the machine that holds it, or it is generated before them.
3. **Reference samples for KL.** θ₀ samples responses to the KL prompts once per model and stores
   them the same way. All runs of that model measure K on these responses. Study 2's monitor
   panel (`kl_monitor`, `eval/study2`) is a third such stream, from its own seeds, read every few
   steps during training to place the full evaluations, which use the scoring panel
   (`kl_benign`, `kl_harmful`). Version 2 (amendment 1) answers the scoring panel's own prompts,
   2 responses each: the first of each prompt is read every few steps, the second only with the
   scoring panel's K alone at fixed diagnostic steps; θ₀'s logits for both panels are cached
   (`evaluation.ReferenceLogits`) and checked against uncached readings. Version 1 answered 25
   further prompts. θ₀'s next-token
   distributions come from a frozen θ₀ copy kept on the GPU next to the student and evaluated in the
   same forward path, so K is exactly zero at θ₀ and needs no stored distributions. The curvature
   panel of the endpoint Rayleigh quotient (`curvature_panel`, `eval/rayleigh`) is stored the
   same way: zero-field θ₀ responses to the teacher prompts, drawn iid from
   `derive_seed(0, "curvature_panel", "prompts" | "sampling")`, one panel per model that no run
   trains on.
4. **Training run.** A student starts from θ₀ and takes N optimizer steps over the training stream
   in a fixed order. The bridge cell truncates the same stream to 48 tokens, which is identical in
   distribution to sampling with a 48-token cap and keeps it paired with the main cell. The
   Alpaca-reference control uses 500 distinct Alpaca examples outside the reserved sets, each once.
   The run logs the loss and the gradient and displacement norms with their projections on fixed
   random directions (generated on the CPU, since CUDA random numbers of large tensors depend on
   the GPU model; consecutive displacement projections give the update projections), then
   evaluates in-process:
   - K on harmful and benign prompts;
   - greedy responses to harmful prompts and keyword refusal;
   - utility;
   - weight geometry and the requested controls.

   A study-2 run (`schedule/discovery`, `schedule/short`) reads its batches in stream order as it
   goes and checks each chunk before training on it. It reads K on the monitor panel every few
   steps and evaluates in full where K crosses its levels, or at fixed steps: the scoring panel's
   K, utility, and the greedy responses to the harmful prompts, stored for the judge but not
   scored. Refusal and harm are not computed during the run. It writes `monitor.jsonl`,
   `evaluations.jsonl`, `chunk_checks.jsonl`, under monitor version 2 `diagnostics.jsonl` and
   `cache_checks.jsonl`, and, atomically, `terminal.json`: the stop, its
   reason and what fired it, the batch refused before an update or the update that broke the
   weights (an applied update counts as a step), the levels reached (only by a successful
   evaluation) and missed, and a crossing whose evaluation failed. With `eval/rayleigh` (off
   unless an experiment adds it), the run then frees its optimizer state and panels and measures
   the endpoint Rayleigh quotient δᵀH₀δ/‖δ‖² of its displacement on the curvature panel, one
   pass of Hessian–vector products at θ₀ on the frozen copy (`curvature.endpoint_rayleigh`),
   into `rayleigh.json` and the summary; the student's weights become δ, since the run is over.
   A failed measurement is recorded there too (R null, `failed: …`) before the error goes on.
   Two switches of amendment 4 leave the training's numerics unchanged:
   - `eval.k_only` (`experiments/study2_short_k_only`, the numerical seeds' runs) evaluates K on
     the scoring panel's benign responses, the monitor and utility only: no harmful prompt set is
     read, no K on harmful prompts is measured and no harmful response is generated
     (`inputs.k_only` in `meta.json`).
   - `tracking.update_scalars` (on for `experiments/study2_short`) writes `update_scalars.jsonl`:
     after every update of an Adam arm, the norm of the algebraic step
     u_t = (lr/(1−β₁ᵗ))·m_t/d_t and its cosine with the step's gradient, read from the optimizer
     state (`schedule.UpdateScalars`; `step` counts the updates taken, 1–25, as in
     `monitor.jsonl`); for the pre-warmed arm at steps 5, 10 and 25 also `d_departure`, the
     departure of the denominator d_t = √v_t + ε from d₀ = √v₀ + ε (the norm ratio, and the
     median and 90th percentile of |d_t/d₀ − 1| read on a histogram of 0.005-decade bins, at most
     1.16 % above the exact value). The hook only reads; v₀ is kept in pinned host memory and no
     GPU buffer persists. Tests pin the parameters, the optimizer state and every other record
     bit for bit with the hook on and off.
5. **Judge.** The judge labels stored responses in a separate step, batched across runs. The judge
   model and the rubric version are recorded with every label. A run's labels from the judge
   `<name>` (`configs/judge/<name>.yaml`) live in `<run dir>/judge/<name>.jsonl`, in the record
   format of `judge.JudgeClient`: one record per response of `responses/<prompt set>.jsonl`,
   keyed by `<condition>/<prompt id>` (e.g. `trained/judged:17`; the in-process controls answer
   the same prompts under other conditions), each carrying the judge identity (backend, prompt
   version, judge model and revision, rubric version). A record with a null label (unparseable
   output) labels nothing.
6. **Analysis.** Fits of h(K), Δ(K\*) with a seed bootstrap (or the unequal-displacement fallback),
   the noise-theory predictor against its baselines, and gate reports
   ([reports/](reports/README.md)). The full sequential analysis is calibrated by simulation
   before confirmatory runs ([requirement](preregistration.md#full-analysis-simulation)). Analysis reads run
   directories only; `scripts/build_analysis.py` rebuilds an experiment's tables, results and
   figure from them ([publication evidence](#publication-evidence)).

## Modules

Implemented:

| Module | Responsibility |
|---|---|
| `paths` | repository paths, data and run roots (`DATA_DIR`, `RUNS_DIR`), loading `.env` |
| `config` | YAML composition, checked command-line overrides, config hash |
| `protocol` | loading run configs with checks of the zero-field settings, the declared control and pinned revisions |
| `tracking` | run directories, provenance including the source snapshot, metrics, summary, W&B mirror |
| `seeding` | run seed and named per-purpose streams |
| `data` | pinned prompt sets, deduplication and splits, manifests with hashes; the factual-question resource `resources/simple_facts_v1.jsonl` |
| `models` | loading models and tokenizers at pinned revisions, numerics, neutral generation config, end tokens, chat encoding, frozen input embeddings and output head |
| `teacher` | zero-field sampling from θ₀ under the invariants below; replaying stored responses on the generation path; stream identity, storage and validation, including the chunks of appendable streams |
| `runtime` | the run environment (tokenizer, model factory, prompt sets); generating and loading streams and soft-target distributions; extending appendable streams, reading their prefix and replaying their soft targets |
| `batches` | right-padded prompt + response rows with explicit response targets and truncation to the cap |
| `losses` | the training path; summed NLL over response tokens divided by a constant; the token-count control; the soft-target control loss |
| `optim` | Adam with fp32 state in adaptive, frozen and pre-warmed modes (H2a); the θ₀ second-moment estimate; SGD with and without momentum from PyTorch; read-only accessors of a parameter's step size and denominator, which a test pins to the step's own |
| `train` | one training run with the step-0 teacher gap, projections, in-process evaluation and controls; study 2's K-only evaluation (`eval.k_only`: K on the benign panels, the monitor and utility, no harmful prompt) |
| `evaluation` | K against a frozen θ₀ copy, greedy responses, utility, weight geometry, the line scan θ₀ ± aΔθ, random-direction controls |
| `refusal` | the versioned keyword refusal list |
| `diagnostics`, `checks` | stage-0 measurements: mean-gradient scaling, the token-count field, the path discrepancy, the Adam check, the step profile |
| `panels` | study 2: K on the monitor panel (version 2: nested panels on the scoring prompts, cached θ₀ logits), the scoring panel's K alone for diagnostics, the cache checks, and the Stage-A check that the monitor agrees with the scoring panel |
| `analysis.stage_a` | study 2, Stage A: run records checked for integrity (including every evaluation's answers to the recorded harmful prompts), summaries per arm, batch and rate (reachability and steps per level, utility, time, stops), soft-target screen and numerical acceptance, the gradient rule's margin and possible false alarms, monitor agreement, and the selection rule as one function, which first requires the whole prescribed search and otherwise returns `incomplete` |
| `schedule` | study 2 runs: batches read lazily from the appendable stream with each chunk checked when first read (teacher gap, the run's path-field bound for its maximum exposure), monitor readings, full evaluations where K levels are crossed, the stops (target, plateau, instability, T_max) and the terminal record; the update scalars of amendment 4 (`UpdateScalars`, logging only) |
| `curvature` | study 2, Stage A: top curvature of the training loss at θ₀ by Lanczos on Hessian–vector products (a fixed number of steps, convergence by the Ritz residual and agreement of independent starts, not a certified accuracy of the top eigenvalue), for single batches at each batch size and for the Hessian pooled over many responses. The pooled 2/λ_max describes the mean dynamics, not SGD's mean-square stability, and single-batch medians do not certify a rate: the curvature guides the SGD rate screen, the soft-target control accepts rates. Amendment 3: the endpoint Rayleigh quotient R = δᵀH₀δ/‖δ‖² of a study-2 run's displacement, H₀ the training loss's Hessian at θ₀ on the curvature panel, recorded signed (null with its reason for a zero or non-finite ‖δ‖ or a failed measurement), from one pass of products over the panel (δᵀH₀δ without a sum of products) after the run's terminal record |
| `fa_control_diagnostic` | study 2: the outcome-blind diagnostic of the FA soft-control failure ([specification](study2-fa-control-diagnostic.md)), K-only on benign panels: shared-state checks, the v₀ census and the residual's sources (D4, D1), preconditioned Lanczos and the linearized bias-corrected recurrence (D2), twin runs against perturbations and a float64 reference (D3), and the frozen decision branches. Amendment 4's numerical check, built on D1 and D3: its frozen scope (`NumericalScope`), the cases and the same-path controls, twin A's numerical consistency with its archived run (`compare_archive`, review fa21; the diagnostic keeps its bitwise gate), the float64 reference with its fallback, the records bound to the GPU code, the policy and the run located now, the run-environment check, the verdict per selection, and the void of a numerical stage |
| `code_identity` | study 2, amendment 4: the GPU-code identity, a sha256 over every GPU path of a commit (the code a GPU run executes: `src/` outside `analysis/` but with `analysis/stage_a.py`, `scripts/` but the analysis scripts, `configs/`, `docker/`, the dependency manifests and the files that build the image), the GPU paths two commits differ in, and the reuse check by which a qualification holds at a later commit |
| `drift` | study 2, amendment 4: the first-step drift diagnostic, descriptive: seed 2300's stream of 64 batches, each gradient transformed by the first update of SGD, AdamW, FA and pre-warmed AdamW (bit for bit `optim.Adam`'s first step), the unbiased mean and trace estimates, the cosine between two fixed halves and the cross-optimizer cosines, one optimizer at a time with float64 reductions; its streams prepared as the regular runs' |
| `judge` | three-label judge over an OpenAI-compatible endpoint (WildGuard and rubric formats), resumable label files, agreement with reference labels |
| `launch` | campaign files: cells checked against the protocol, expanded into runs, run in order in fresh processes under their own run ids; qualifying completed runs skipped, diverged runs left, technical failures retried up to three attempts |
| `judge_check` | the blind sample of the judge check across optimizers: strata chosen from K alone, prompts and seeds paired across optimizers |
| `run_judging` | judge items from a run's stored responses, labeled into the run's label file (`<run dir>/judge/<judge>.jsonl`), in at most three rounds per run and judge across reruns, each counted before its requests in `<judge>.rounds.json`, with a judging receipt once the labeling completed (`<judge>.receipt.json`: expected items, judge identity, rounds used in all, labels still missing, label file sha256), kept as it is on a rerun; labels changed outside a round (a receipt that no longer matches, or a label file changed after its receipt was deleted) are refused, nothing sent; a failed round stops the invocation, and a run another process is labeling is busy (a lock); the conditions of a study-2 run are its recorded evaluations `step-<N>`, and it is labelable once its terminal record was published with whole records, whatever its status |
| `analysis.study2_labels` | per-evaluation judge-label counts of a study-2 run, with the bounds for missing labels; counts no label without a completed receipt matching the label file (the run stays pending); refuses stale, foreign-judge or conflicting labels |
| `analysis.study2_short` | study 2's short regime (amendments 3 and 4), pure functions over run records: the selection's state machine (the B = 20 order and its primary-pair fallback, discovery rounds, the numerical gate reading the check's verdicts, the search allowance, technical and scientific failures, the three-arm validation, `infeasible` with the feasibility report, `blocked`), the stage's conformance, support and the interpolation estimator, the looks, the confirmation guard, the mechanism cohort and its final max-t analysis, the time course with its numerical consistency against the endpoint runs, the update scalars' coverage, and the fixation and versions records (`read_fixation`, `versions_problems`). It reads K, numerics and utility, and labels only in the looks and the final analyses |
| `analysis.study2_short_sim` | the simulation of the short regime's confirmation: amendment 3's primary procedure (levels, validation gate, bias allowance), and amendment 4's mechanism cohort (component scenarios, attrition modes, the max-t intervals of the analysis on shared draws, each replicate's conditional truth, the calibration rule with its single check), from the B = 20 inputs record |
| `analysis.study2_audit` | the terminal audit of the judge (amendment 4, design v3): the deduplicated frame and its coefficients, the draw with replacement proportional to the largest coefficient, the Clopper–Pearson bound on each arm's total error and each estimand's bias bound, the descriptive Hansen–Hurwitz estimate, the reference harm of the realized cohort, non-estimable estimands, and the 10 % agreement check |
| `analysis.study2_audit_cli` | the audit's command line (`python -m zero_field_refusal.analysis.study2_audit_cli draw / agreements / estimate`), in the analysis package so that it is no GPU path: the draw behind the summary's final analysis, with D7's blinded files, the labellers' agreements, and the estimate from the resolved reference labels |
| `labeling`, `calibration` | the pool of calibration responses, blind calibration samples, the terminal labeling session and the import of reference labels |
| `runindex` | one table of all runs from their `meta.json`, `config.yaml` and `summary.json` |
| `analysis` | h(K) fits, K\* and the decisive cells over eligible cells (`analysis.eligibility`), Δ(K\*) with the paired seed bootstrap, the H2 rule, the full-analysis simulation; the runs table from run directories with explicit exclusions (`analysis.evidence`) and versioned tables, results, figure and manifest (`analysis.build`, with `analysis.outputs`); the Gate 1 report from the stage-1 runs (`analysis.gate1`) |

Planned:

| Module | Responsibility |
|---|---|
| `models` | the LoRA variant |
| `distributed` | `torchrun` and FSDP2 setup, global-batch arithmetic, rank-aware data order |
| `analysis` | the noise-theory predictor and baselines, the Gate 2 and Gate 3 reports |
| `toy` | toy models for checking the theory: a categorical bias model, a 2-D quadratic bowl, a small network |

Entry points in `scripts/`:
- `prepare_data.py`: builds the prompt sets into `DATA_DIR/prompts/<config hash>/`, caching the
  pinned source CSVs in `DATA_DIR/sources/`;
- `generate_teacher.py`: samples a stream (training, preconditioner, K reference, soft targets,
  Alpaca reference, curvature panel) into `DATA_DIR/teacher/`, or extends an appendable training stream to
  `--responses N`;
- `train.py`: one run from existing streams; a config with a `schedule` (study 2) runs under it
  and may extend its seed's stream;
- `evaluate_theta0.py`: evaluates θ₀ itself with `run_type: theta0`, which keeps its config hash
  apart from the training runs of the same config; the baseline of every gate;
- `stage0_checks.py`: the stage-0 checks of the real pipeline on one model;
- `curvature.py`: the Stage-A curvature measurement of study 2 on a seed's appendable stream
  (`run_type: curvature`);
- `fa_control_diagnostic.py`: the parts of the FA soft-control diagnostic, its plan and its
  Pod driver, into `RUNS_DIR/diagnostics/fa_control/`
  ([operations](operations.md#study-2-fa-soft-control-diagnostic));
- `numerical_check.py`: amendment 4's numerical check of the selected rates (`plan`, `control`,
  `twins`, `verdict`, the Pod `driver`) into `RUNS_DIR/numerical/b{B}-u{U}/`, and the `void` of a
  numerical stage after a GPU-path change ([operations](operations.md#s2-the-numerical-stage));
- `drift_diagnostic.py`: amendment 4's first-step drift diagnostic (`prepare`, `plan`, `run`, the
  Pod `driver`) into `RUNS_DIR/diagnostics/drift/b{B}/`
  ([operations](operations.md#the-drift-diagnostic));
- `study2_short_campaigns.py`, `study2_short_summary.py`, `study2_short_simulate.py`: the short
  regime's campaigns, prerequisite drivers and numerical-check requests; its summary, looks, final
  analysis and versions record; its simulation and the final level-fixing run. They read records
  and write no GPU result (`code_identity.ANALYSIS_SCRIPTS`)
  ([operations](operations.md#study-2-short-regime-amendment-3)); the audit's command line is a
  module of the analysis package, `analysis.study2_audit_cli`;
- `harmful_pool_cleaning.py`: the counts and the blind-labeling pairs behind the cleaning of
  study 2's harmful evaluation set (the pairs file holds prompt texts and stays out of Git);
- `stage_a_campaigns.py`: the campaigns of study 2's Stage A, one seed each, the data-dependent
  ones (SGD candidates, grid extensions, full controls) from the run records;
- `stage_a_summary.py`: the Stage-A summary and selection of study 2 from the run records, with
  the code revision, written once; it reads no harm label;
- `calibration_pool.py`, `make_calibration_sample.py`, `label.py`, `judge.py`: the calibration
  pool from `experiments/judge_calibration` runs, the blind sample, manual labeling and judging;
- `launch.py`: runs the missing runs of a campaign file (cells of config, overrides and seeds)
  one after another, each in its own process, skipping any config whose hash already has a
  completed run in `RUNS_DIR/<experiment>/`; `--dry-run` lists them;
- `archive.py`, `pod_setup.sh`: archive files to the bucket without overwriting, with checks and
  an inventory; prepare a Pod (pinned Google Cloud CLI, archive key, `DATA_DIR` restored and
  verified) ([storage](operations.md#storage));
- `choose_rates.py`: chooses the stage-1 sweep rates from the seed-0 pilots by K, divergence and
  collapse alone, or names the extension rates to pilot first (`analysis.sweeps`);
- `make_judge_check_sample.py`: draws the blind sample of the judge check across optimizers from
  the 1.5B stage-1 runs (strata, sample, key, selection record);
- `label_runs.py`: labels every response of completed run directories (study-2 runs: published
  terminal records) with one judge into their label files, the input of the analysis build; the
  label file is the cache, and a judging receipt marks a completed labeling;
- `import_labels.py`: imports reference labels made outside the project (the ensemble) next to
  the sample, with their provenance folder and hashes;
- `index_runs.py`: flattens `meta.json`, `config.yaml` and `summary.json` of every run, failed ones
  included, into one table;
- `rescore_factual_qa.py`: rescores the stored factual answers of run directories with the current
  matcher into `observations/factual_qa_rescored.json`, without model runs;
- `simulate_analysis.py`: the full-analysis simulation of Gate 2 into
  `RUNS_DIR/analysis/simulation/<UTC timestamp>/`; `--confirm` reruns the binding nulls on a fresh
  seed, `--resummarize` recomputes the verdicts of an earlier run;
- `build_analysis.py`: rebuilds the analysis of one experiment on the CPU from its run directories
  and one judge's labels, with the stage-1 interval settings and K\* either selected from the
  analyzed runs (provisional, refused for Gate 2 runs) or frozen from an earlier build
  (`--k-star`), into a new directory outside the checkout or under an ignored path (never
  overwritten): runs and exclusions, cell, arm and estimate tables, the K\* record, results, a
  figure and a manifest ([publication evidence](#publication-evidence));
- `gate1_report.py`: the [Gate 1](preregistration.md#gate-1-technical) numbers of one model from its
  stage-1 run directories on the CPU, into a new directory like `build_analysis.py`; it refuses
  any run whose condition differs from the committed config of its experiment (model, training
  regime, evaluation volume). `--calibration` takes the judge-calibration directory: the report
  computes the judge's agreement with the reference labels itself and checks the fixed sample
  and reference hashes, the WildGuard identity of the labels and the 200 reference items; any
  departure makes the calibration invalid and Gate 1 pending.

Planned:
- `report.py`.

## Zero-field invariants

| # | Invariant | Why | Check |
|---|---|---|---|
| 1 | The teacher is θ₀: the same checkpoint revision, precision, attention implementation and chat template as the student at step 0. | Any difference between how the teacher samples and how the student scores creates a real field. | Blocking: before its first optimizer step, a run checks that the training path reproduces the stored teacher log-probabilities of its whole stream within the preregistered thresholds (`checks.teacher_gap`, per model in `configs/model/`, set from the stage-0 GPU measurements), and stops otherwise; see [precision](#precision). |
| 2 | Sampling at temperature 1 with top-k off, top-p 1, no repetition penalty, no minimum length, natural end of turn. | Any reshaping of the distribution is a real field (e.g. temperature 0.7 sharpens frequent tokens). | At every step the scores the sampler used equal a copy of the raw logits taken before `generate` processes them; see [generation settings](#generation-settings). |
| 3 | Truncation at `max_new_tokens` is allowed; filtering or resampling by length or content is not. | Truncation keeps the expected gradient zero; selection on the outcome does not. | No code path drops or resamples teacher responses. |
| 4 | Training uses the stored token ids, including the sampled end-of-sequence token; text is never decoded and re-tokenized. Sampling stops at either of the model's end tokens (`model.end_tokens`). | Re-tokenization can produce token sequences the teacher never sampled. | The loss reads token ids only. The response length comes from the first end-of-sequence token, never from a `!= pad` mask: Qwen2.5's pad id equals one of its end-of-sequence ids, so such a mask drops a sampled end token. |
| 5 | Loss = summed token NLL / constant (`train.loss.denominator` = batch size × `train.max_response_tokens`). | Dividing by the batch's token count weights short responses more and pushes toward shorter answers. | Unit test: the loss of a batch does not depend on the lengths of the other responses in it. |
| 6 | Every teacher response is used once; for a given seed all optimizers see the same responses in the same order. | Reuse turns noise into a fitted signal; a shared stream makes optimizers comparable. | The data order derives from `derive_seed(seed, "data_order")` alone, or, in an appendable stream, is the stream order; frozen and pre-warmed Adam share a separate preconditioner stream, never reused for training. |
| 7 | Weight decay is off; the input embeddings and the output head are frozen at every model size. | Decay is a systematic force toward zero. Freezing keeps the memory budget and is part of the fixed protocol. 0.5B and 1.5B tie the two modules, 7B does not, so freezing the input embeddings alone would leave the 7B head trainable. | Config validation; a check that neither module has a trainable parameter. |
| 8 | Soft-target control: the student trains on KL to θ₀'s full next-token distributions, recorded on the generation path (`generate` with the KV cache) while the teacher samples, or, for an appendable stream, recomputed by replaying the same generation batches with the sampled tokens forced. | The field at θ₀ is zero up to a measured residual: the student trains toward the generation path's distributions, so any discrepancy between the two paths is a residual field, the mean gradient of this loss at θ₀. Targets recomputed by the student's own function would match it bit for bit and measure nothing. | Blocking: the stage-0 checks bound the residual field per execution environment (model numerics, device, library versions) over several batches, N·\|f\|²/σ² ≤ the preregistered bound in every direction and metric, through the χ² bound of the two paths' distributions ([criterion and its limits](#design-decisions-for-stage-0)), and write a record; a training run without a passed record for its environment stops before its first optimizer step. Reported alongside: gradient estimates in the Euclidean and Adam metrics, the per-token KL and a one-batch gradient ratio. A training run of this control uses SGD: under Adam, tiny residual gradients become full-size sign steps, so the student moves however clean the pipeline is. |
| 9 | Zero mean gradient: the mean gradient over many teacher batches at θ₀ shrinks like 1/√n. | A direct test that the field is zero. | A stage-1 diagnostic and a unit test on a tiny model. |

Unit tests for these invariants run on CPU with a tiny randomly initialized model, so they need no
downloads.

### Generation settings

Hugging Face `generate` fills every sampling field the caller leaves unset from the model's
`generation_config.json`, then from library defaults. Qwen2.5-Instruct ships temperature 0.7,
top-p 0.8, top-k 20 and repetition penalty 1.1 (1.05 for 7B); the library default for top-k is 50.
A call that sets only `temperature=1.0` therefore samples from a reshaped distribution. Rules:
- replace the model's generation config with a neutral one that sets only the end-of-sequence and
  pad ids;
- pass every sampling field explicitly;
- assert at every step that the scores the sampler used equal a copy of the raw logits taken in a
  forward hook, before `generate` processes them, so no logits processor reshaped the
  distribution, even one that edits the logits in place (which would also alter
  `output_logits`);
- store the effective generation config with the samples.

A unit test loads a fake generation config carrying the Qwen values and checks that none of them
reaches sampling. The rules apply to teacher samples, KL reference samples and the greedy
evaluation: a repetition penalty would change greedy outputs too.

### Precision

- Stages 0–2 run entirely in fp32: weights, compute, gradients and both Adam moments; TF32 is off
  (`model.dtype`, `model.tf32`). Each larger model size fixes its storage, compute and
  gradient-reduction dtypes; FSDP2 reduces gradients in fp32 (`reduce_dtype=float32`). Provenance
  records all of them.
- Teacher log-probabilities come from the raw logits, not from scores after logits processors.
- The teacher–student tolerance is stated as statistics of the per-response summed log-probability
  difference: mean signed, mean absolute and maximum. It is checked in the real training
  configuration: FSDP2, padding and micro-batch size.
- bf16 compute is itself a small real field. On 0.5B, the maximum per-token difference between
  teacher and student log-probabilities was 2.6·10⁻⁵ in fp32 and 0.19 in bf16.

### Loss units

The loss is summed NLL / (B·L), with B the batch size and L = `train.max_response_tokens`: 96, or
48 in the bridge cell. The noise
formulas in [design.md](design.md#heating-of-flat-directions) use the mean per-sequence loss, so
for SGD their learning rate is η_config / L. Adam's update does not depend on the scale of the loss
(apart from ε), so for Adam it is η_config.

## Memory

- Full fine-tuning of 1.5B in fp32 needs ≈ 22 GB for weights, gradients and both Adam moments, plus
  ≈ 6 GB for the frozen θ₀ copy that K uses, and activations. It fits on one 48 GB GPU, with the
  first moment in fp32, so the AdamW arm is the standard optimizer. Measured on an L40S in stage 0:
  a training step with the θ₀ copy, Adam and a greedy batch peaks at ≈ 26 GB allocated.
- The stage-0 field measurements hold parameter-sized fp32 buffers on top of the weights and the
  gradient: two at a time for the field at θ₀ (one normalization after the other), and a
  soft-target copy plus two for the path field, with Adam's v̂ kept on the CPU. With four at once
  and the previous batch's gradient still alive, 1.5B exceeded 48 GB (activations of a micro-batch
  of 4 take ≈ 10 GB); `tests/test_gpu.py` checks the buffer count.
- The endpoint Rayleigh quotient of a study-2 run takes its product after the run, with the
  optimizer state, the panels and their caches and the student released and the student's
  trainable weights turned into δ in place. It takes δᵀH₀δ micro-batch by micro-batch without a
  sum of products, and releases the first-order gradients before each second backward, so it
  holds the frozen θ₀ copy (≈ 6.2 GB at 1.5B) and two trainable-sized fp32 vectors at a time
  (≈ 5.2 GB each: δ, and the gradients or the second backward's result), ≈ 16.7 GB, plus the
  double-backward graph of one micro-batch of `execution.micro_batch_size` (≈ 13–17 GB by count
  at 4): ≈ 30–34 GB. The Stage-A curvature run's Lanczos steps held two vectors more and ran on
  an L40S at micro-batch 4; `rayleigh.json` records the product's peak.
- Amendment 4's tools run under a cap of 46 GB reserved per process (1e9 bytes) and check the
  free host memory first. The numerical check holds one seed's setup at a time (θ₀, v₀, a second
  model, the p1 direction) and needs 52.5 GB of free host memory; its float64 reference of the
  AdamW and pre-warmed arms falls back to float64 loss and accumulation if it meets the cap. The
  drift diagnostic keeps the four sums S on the host (≈ 21 GB at 1.5B) and needs 47.5 GB. The
  update scalars keep v₀ pinned on the host for the pre-warmed arm (≈ 6.7 GB at 1.5B: the pinned
  allocator rounds each parameter's block up to a power of two).
- Pure bf16 training is ruled out: at typical weight magnitudes the bf16 grid spacing exceeds
  Adam's update, so the update rounds to zero.
- 8-bit Adam would change the optimizer under study. For the same reason the optimizers stay plain
  PyTorch, so a fused library implementation never replaces the object of study.
- Larger models are sharded; see [distributed training](#distributed-training). Hardware per stage
  is listed in [plan.md](plan.md#compute).

## Distributed training

- **When:** only when full fine-tuning does not fit on one GPU, i.e. 7–8B on 80 GB GPUs. Everything
  smaller runs as one process.
- **How:** `torchrun` with FSDP2 (`torch.distributed.fsdp.fully_shard`). Parameters, gradients and
  optimizer state are sharded per parameter as DTensors.
- **Why not DeepSpeed ZeRO:** ZeRO can wrap a client optimizer, but it hands the optimizer flattened
  partitions and owns the master weights and mixed precision. Here the optimizer is the object of
  study, so it should see ordinary per-parameter tensors. Adam-type updates are elementwise and act
  on local shards without communication. The frozen second-moment estimate of Adam with frozen
  statistics is optimizer state with the same sharding as its parameter.
- **Batch arithmetic:** the config fixes the global batch, which sets the noise temperature. The
  number of processes must not change it. Per-device micro-batch and accumulation steps are derived
  so that micro-batch × accumulation × world size = global batch, and the code fails if this cannot
  hold exactly. With a global batch of 20, the world size is therefore a divisor of 20. This is a
  convenience: with the constant denominator, uneven slices would also give the exact gradient, as
  long as every rank runs the same number of micro-steps.
- **Loss scaling:** data-parallel reduction averages gradients over processes. Each process
  therefore scales its local summed NLL by world size / denominator. The resulting gradient equals
  the single-process gradient of summed NLL / denominator.
  - Gradient accumulation adds micro-batch gradients; there is no 1/accumulation factor.
  - The logged loss is the unscaled summed NLL / denominator.
  - Alternative: `set_gradient_divide_factor(1.0)` on the FSDP2 module, and each process scales by
    1 / denominator.
- **Data order:** every process reads its slice of the same global batch order. A run on one GPU
  and a run on two GPUs see identical batches.
- **Global statistics:** gradient norms, weight geometry and random projections issue collectives,
  so every rank calls them, not only the main process.
- **Random directions:** random directions and projections derive from parameter names and full
  shapes, never from the local shard, so runs on one and on two GPUs give the same results.
- **Verification before use:** the custom AdamW must reproduce `torch.optim.AdamW` on 1 and on 2
  GPUs over a short run. Both use fp32 state, so the tolerance can be tight.

## Configuration

Container environments are built from the same dependency manifests. Source checkouts stay
separate from the image and are selected by the container launcher; runtime setup and storage
paths are defined in [operations](operations.md#containers).

- **Files:** YAML under `configs/`, composed through `defaults`; experiment files live in
  `configs/experiments/`.
- **Overrides:** a command-line override must name an existing key; `+key=value` adds a new one.
  This stops typos in sweeps from silently creating keys.
- **Group overrides:** `optimizer=sgd` replaces the `optimizer` section with
  `configs/optimizer/sgd.yaml`. This works for any top-level key with a matching `configs/<key>/`
  folder. Other sections the file sets are merged in rather than replaced: a model file carries
  its teacher-gap thresholds under `checks`, and `model=...` keeps the rest of `checks`. Later
  dotted overrides apply on top: `optimizer=sgd optimizer.lr=0.1`.
- **Composition:** an experiment composes only the groups its entry point reads. The judge, for
  example, has its own group, `configs/judge/`, read only by the judging step, so fixing the judge
  model does not change the hash of any training run.
- **Protocol checks:** entry points load run configs through `protocol.load_run_config`. It rejects
  any departure from the zero-field settings (invariants 2, 3 and 5–7, fp32 without TF32, pinned
  model revisions) unless a control declared under `control` explains it. Each control in
  `configs/control/` is composed through `defaults` and permits exactly its own deviation.
- **Required values:** `???` marks a value that must be set before a run, such as the learning rate
  of SGD and of Adam with frozen statistics. Loading works; computing the config hash, i.e.
  starting a run, fails until the value is set. `null` is kept where null is a real value.
- **Derived values:** `${mul:a,b}` returns the product of its arguments. The protocol derives
  `teacher.responses` and `train.loss.denominator` this way, so changing the steps, batch size or
  maximum length cannot leave them stale.
- **Identity:** the config hash excludes labels (`experiment`, `name`, `tracking`) and execution
  settings (`execution`: device and batch sizes for micro-batching, evaluation and generation, which
  change only the floating-point summation order) and identifies what a run computes. The launcher
  uses it to skip cells that already have a completed run.
  - `protocol_version` in `configs/base.yaml` is part of the hash. Bump it on any change to code or
    data that alters results, so the launcher never skips a cell computed by older code.
  - `tracking.update_scalars` records only, outside the hash like the rest of `tracking`;
    `eval.k_only` is inside it, since it changes what a run evaluates.
  - **The GPU-code identity** (`code_identity`) is a second identity, of code rather than of a
    config: amendment 4's numerical qualification holds for the GPU code its runs and check
    executed, and a later commit reuses it only after a recorded check that no GPU path changed.
  - `meta.json` records the config hash, the git commit, the hash of `uv.lock`, package versions,
    GPUs, and the source snapshot and container image passed by the launcher
    (`SOURCE_SNAPSHOT_ID`, `CONTAINER_IMAGE`). Input-manifest provenance still needs the
    implementation described under
    [Publication evidence](#publication-evidence). The history of `protocol_version` is logged in
    [decisions.md](decisions.md#protocol-versions).
- **Gate runs** set `tracking.require_clean_git: true` and run from a real checkout of the commit
  (a clone or a Git bundle), so every result maps to a commit. A checkout is clean without
  uncommitted changes, staged or not, and without untracked files that its `.gitignore` files do
  not ignore (`tracking.worktree_changes`, which the study-2 versions record reads too). Git
  configuration does not change this: `status.showUntrackedFiles`, `.git/info/exclude` and a
  user's excludes file. Gate runs also set `tracking.gate` to their gate (1 or 2; null
  otherwise), which the analysis build reads. Development runs may use source snapshots without
  Git metadata; their snapshot ID identifies the executed source.

## Experiment tracking and storage

| What | Where | Notes |
|---|---|---|
| Run outputs (config, provenance, metrics, responses, judge labels) | `runs/<experiment>/<run id>/` | Source of truth for every analysis and gate report. |
| Run index | one table built by `scripts/index_runs.py` | Flattens `meta.json`, `config.yaml` and `summary.json` of every run. |
| Analysis builds | a new directory per build of `scripts/build_analysis.py` (`--out`), outside the checkout or under an ignored path such as `RUNS_DIR` | Versioned, never overwritten; the manifest holds the hashes of the inputs read and whether the checkout is dirty, which an untracked output directory inside it would make it. |
| Metrics for browsing and comparison | Weights & Biases, private project `zero-field-refusal`, group = experiment | Entity, key and mode from `.env`; the W&B run id is the run id. |
| Prompt sets | `data/` | Regenerable from pinned dataset revisions and indices; manifests record revisions and hashes. |
| Teacher and reference samples | `data/` | The record of the experiment. GPU sampling is not bit-reproducible, so they cannot be regenerated exactly; back them up to `ARTIFACT_REMOTE`. |
| Large artifacts on rented machines | `DATA_DIR`, `RUNS_DIR` on the Pod's working storage at `/workspace`, copied from the Pod to `ARTIFACT_REMOTE` (a private, versioned GCS bucket) | Where each kind of data lives: [storage](operations.md#storage); archive and handoff procedure: [operations](operations.md#rented-machines). |
| Checkpoints | only when `train.save_checkpoint: true` | Line scans and perturbation controls run in-process instead. |
| Hypotheses, thresholds, gate decisions | `docs/preregistration.md`, `docs/decisions.md` | Committed before the runs they govern. |
| Gate reports | `docs/reports/` | Aggregates only; see the [convention](reports/README.md). |

- **W&B privacy.** Runs start with privacy settings: no hostname, paths, git metadata, code or
  console capture. A failure to start W&B is recorded in `meta.json` and does not stop the run,
  unless `tracking.wandb.required` is set.

TensorBoard is not used: W&B covers browsing, and the local files cover analysis. Procedures for
rented machines, merging runs, W&B in practice and public releases are in
[operations.md](operations.md).

## Publication evidence

Tracking already saves run/config/Git/lock/package/hardware metadata, metrics and summaries.
Input provenance and the observation/analysis formats below are stage-0 implementation work; the
rebuild of the primary analysis from run directories is implemented and checked on a saved
fixture (`tests/fixtures/evidence/`, regenerated by its `make_fixture.py`).

- **Provenance.** `meta.json` records under `inputs` the hash of the prompt-set manifest and the
  path, hash and size of every teacher, preconditioner and K-reference stream a run read; the model
  and tokenizer revision is pinned in the config, and the source snapshot ID and container image
  come from the launcher. The executed source of a development run
  is its retained snapshot, not a copy inside the run directory; gate runs require clean Git. Link
  downstream inputs to producing runs and hashes. Preserve actual generated samples as specified in
  [Data flow](#data-flow).
- **Observations.** Retain run/prompt/response IDs linked to seed, optimizer, learning rate, step
  and control condition; per-response KL contributions, refusal/judge labels, utility outcomes
  and geometry/control measurements. Preserve shared-teacher pairing across optimizers and rates,
  detector/rubric/judge-model versions, and private responses for re-scoring. Record missing/failed
  measurements and reasons explicitly.
- **Analysis.** Recompute from saved observations without training, generation or judge calls. Save run/artifact
  IDs, exclusions/reasons, protocol version, analysis revision/settings and RNG seed. Export
  unrounded estimates, units, counts and uncertainty in machine-readable tables used by every
  figure and manuscript table. Keep original measurements immutable and version derived results.
  Retain the runs used to choose K*, its frozen value and each sequential look's membership under
  [primary analysis](preregistration.md#primary-analysis).
  - *Implementation.* `scripts/build_analysis.py RUNS_ROOT --experiment NAME --judge NAME --out
    DIR` reads every run directory of the experiment, failed ones included (`analysis.evidence`):
    the arm is `optimizer.name` (`sgd_momentum` for SGD with momentum, `+<control>` for a
    control run), K is `K_harmful` of `summary.json`, and the counts come from the trained
    model's responses and their [judge labels](#data-flow). Each run left out keeps a reason
    code and a detail: diverged, not completed, a missing or malformed file (meta, config, summary,
    responses, labels), no K, an arm outside the analysis, incomplete labels, labels of another
    judge identity. The included runs must share one condition (every config key except the
    optimizer section, the seed and the keys outside the config hash) and one optimizer
    configuration per arm apart from the learning rate, hold at most one run per seed in a cell,
    share one training stream per seed and use different streams for different seeds
    (shared-teacher pairing); otherwise the build stops. A 48-token bridge cell therefore needs
    its own experiment. Known limitations: the condition check also compares evaluation-only
    settings such as `eval.controls`, so runs with and without line scans do not mix; when K\* is
    undefined (SGD and AdamW ranges do not overlap) the build writes nothing, as the
    unequal-displacement fallback is not implemented here.
  - The build runs the analysis with the stage-1 interval settings and writes `runs.csv`,
    `exclusions.csv`, `cells.csv` (seeds present and absent per cell), `arms.csv`,
    `estimates.csv`, `k_star.json` (the selection used, its status, the runs it was computed
    from with their gates, and the file a frozen K\* was read from with its sha256),
    `results.json`, `figure.svg` (drawn from those tables only) and
    `manifest.json` (the sha256 of every input file read, settings, RNG seed, protocol versions,
    judge identity, code revision, package versions, output hashes). A rebuild from the same
    inputs is byte-identical apart from the manifest's `created_at`.
  - *Provisional or frozen K\*.* Without `--k-star`, the build selects K\* from the analyzed
    runs themselves: a sweep-style selection that suits the stage-0 fixture check and the stage-1
    sweep, recorded as `k_star_status: provisional_from_analyzed_runs`. It is not the Gate 2 K\*,
    which is computed from KL values only, fixed before look 1 and not re-estimated at look 2
    ([primary analysis](preregistration.md#primary-analysis)), so the build refuses to run when
    any run directory of the experiment, a failed one included, sets `tracking.gate: 2`. A
    Gate 2 look passes `--k-star` with the `k_star.json` of the sweep build that fixed K\*; the
    build uses that selection unchanged (`k_star_status: frozen`), records the file's sha256, and
    refuses a selection computed from Gate 2 runs or one whose levels do not follow from its
    common range. Look 2 passes the same file. Per-run look membership is not an input yet: the
    look-1 analysis must be built before any look-2 run joins the experiment.
- **Paper.** Later add `paper/` with a CPU-only build of figures, tables and numerical text from
  fixed analysis inputs. Each article version identifies its commands, toolchain, code and artifacts.

See [retention](operations.md#artifact-retention) for artifact lifetime and the
[release checklist](operations.md#release-checklist) for public exports. Statistical rules remain
in preregistration.

## Design decisions for stage 0

Settled in stage 0 with status `proposed`; the ones that are part of the binding protocol become
`fixed` through [preregistration.md](preregistration.md#fixed-protocol).

- **Soft-target storage.** The targets are θ₀'s full next-token distributions recorded on the
  generation path ([invariant 8](#zero-field-invariants)); bf16 is not acceptable, because rounding
  the targets would itself create a gradient at θ₀. They are stored in one fp32 file per stream,
  hashed in its manifest. The path-field check uses 8 batches per environment, ≈ 9.6 GB; the full
  SGD control run needs 500 responses × 96 tokens × 151,936 entries ≈ 29 GB per model and seed.
  Generation copies each chunk into one preallocated buffer and saves the filled rows without a
  further copy, so host RAM peaks at ≈ 30 GB; training reads the targets lazily, one batch at a
  time. Decided on 2026-10-09 over two alternatives: choosing Pods by verified RAM ≥ 96 GB (host
  RAM varies per Pod: an RTX 6000 Ada Pod had 62 GB, not the listed 167 GB; a larger-RAM type such
  as H100 is a new execution environment that needs its own stage-0 checks), and sharding (more
  code for the same effect). The files are deleted after use, keeping hashes and summaries.
  - *Study 2: replay instead of storage.* A soft-target control over an appendable stream would
    need ≈ 1 TB for 4000 steps of batch 5. It stores nothing. The run's frozen θ₀ copy recomputes
    the distributions on the generation path:
    - each generation batch of a chunk is replayed through `generate` as it was sampled, with the
      same prompts, order, padding, sampling settings and ending, the stored tokens forced after
      the raw logits are captured, so every forward call repeats one of the sampling;
    - the replayed log-probabilities of the sampled tokens must match the stored ones within
      `REPLAY_MAX_ABS` = 10⁻⁵. Exact agreement is expected; the margin covers fp32 rounding
      alone, an order of magnitude below the 1.97·10⁻⁴ that two computation orders of the same
      weights differ by in stage 0;
    - the whole distributions must match too, since a check of the sampled token would miss a
      change elsewhere in the vector. While sampling, each generation batch of a chunk records
      a fingerprint (`fingerprints.safetensors`, hashed in the chunk manifest): the sha256 of the
      fp32 log-probabilities of all its response positions, and their fp64 projections on 8
      fixed unit-norm Gaussian directions over the vocabulary. Only an equal digest accepts a
      replayed batch. The projections are a diagnostic: a bound on them bounds the error
      vector's norm, not each entry, so a frequent token's log-probability could move by 10⁻³
      within it (review r8). Rows are copied to the host and projected one by one as they are
      computed; the digest of a generation batch is hashed in a background thread while the next
      one samples, and a failure there stops the chunk before it is published. On an RTX 5090
      at 1.5B (2026-10-14), four generation batches took 17.7 s with fingerprints against
      15.5 s without;
    - a stream for a soft-target control is therefore generated on the GPU model that replays
      it, the Stage-A Pod's, as the one-writer rule of appendable streams already requires. Each
      chunk records the device it was generated on, and the replay stops before computing
      anything on another device; a differing digest stops the run at that batch, before the
      first optimizer step when it is the first batch;
    - the run records, for every replayed batch, the sampled tokens' largest and mean deviation,
      whether the digest matched and the projections' largest deviation
      (`soft_targets_replay.jsonl`);
    - at most two generation batches stay in host memory (≈ 1.9 GB each at 32 responses of 96
      tokens).

    The training path cannot supply these targets: at θ₀ it equals the student's own function,
    so the control's gradient would vanish exactly and hide the path field it measures. Decided
    on 2026-10-14 (coordinator review).
- **Path-field criterion.** Training on samples of the generation path p with the training path
  q's scores s_t gives each token the conditional mean m_t = E_p[s_t] = E_p[(1 − q/p)·s_t]. By
  Cauchy–Schwarz, for any direction v in any linear metric, Euclidean or Adam's,
  N·(f·v)²/Var(n·v) ≤ N·X: f is the residual field, n = Σ_t (s_t − m_t) the martingale part of the
  gradient noise, and X the expected sum over a batch's response tokens of χ²(q ‖ p). The checks
  require an upper confidence bound on N·X to stay within the preregistered 0.01.
  - *Limitation: an approximation against the whole gradient noise.* The bound is exact against
    the martingale noise n. The whole noise g = n + M, with M = Σ_t m_t, has
    Var(g·v) = Var(n·v) + Var(M·v) + 2 Cov(n·v, M·v). Var(M·v) is second order in the discrepancy
    and at most ≈ X_r·Var(n·v) per response, where X_r is a response's expected χ² sum, if that
    sum is not strongly correlated with the response's conditional score variance; the covariance
    is then at most ≈ √X_r·Var(n·v). So the ratio against the whole noise is N·X·(1 + O(√X_r)),
    and the per-token second moment adds a factor 1/(1 − χ²_t). At the bound (X_r ≤ 2 × 10⁻⁵ for
    N = 25, B = 20) both corrections are below 1 %: the criterion holds to this approximation,
    not as a strict inequality against the whole noise.
  - *Upper bound.* Responses are independent (instructions drawn with replacement), so the
    per-response sums are the units: 160 in a stage-0 record (8 batches of the soft-target
    stream), N·B in each run's own check. The bound is the one-sided 97.5 % studentized-bootstrap
    (bootstrap-t) bound, 9,999 resamples with a fixed seed. Decided on 2026-10-09 after review,
    replacing the mean plus 2 SE over 8 batch sums, which undercovers for 8 units and for skewed
    sums. A Student quantile alone also undercovers right-skewed sums (≈ 0.93 for lognormal σ = 1
    at n = 160), bootstrap-t keeps ≈ 0.96 there, and no method keeps the level for very heavy
    tails (lognormal σ ≥ 1.5): the record therefore also stores the sums' skewness and the
    largest response's share of the total. Coverage is tested in `tests/test_prechecks.py`.
- **Displacement K.** The exact per-position KL along θ₀'s reference responses, with θ₀'s
  distributions from a frozen copy on the same GPU ([data flow](#data-flow)); defined in
  [preregistration.md](preregistration.md#primary-analysis).
- **Precision for 7–8B.** fp32 storage and compute, TF32 off, FSDP2 gradient reduction in fp32;
  fixed before stage 4 after measuring the teacher–student tolerance at that size. bf16 compute is
  excluded because it creates a real field ([precision](#precision)).
- **Judge.** A local open model with a pinned revision, served separately behind the
  OpenAI-compatible endpoint. The primary candidate is WildGuard-7B, whose refusal and
  harmful-response fields map onto the three labels; Llama-Guard-3-8B and an API judge with a rubric
  are fallback candidates. Chosen on 200 responses from the calibration runs against ensemble
  reference labels checked by the user;
  model, rubric and prompt are fixed in [preregistration.md](preregistration.md#primary-analysis).
- **KL and judged prompts.** Disjoint, from a stratified split of the deduplicated harmful pool
  (`configs/data/default.yaml`).
- **Utility.** 500 MMLU questions stratified by subject, scored zero-shot by the likelihood of the
  answer letter in the chat template, without generation or judge; at least 200 simple factual
  questions with answer aliases.
- **Teacher prompts.** Each response's instruction is drawn uniformly with replacement.
- **End tokens.** Sampling stops at either of the model's end tokens; the sampled end token is part
  of the loss.
