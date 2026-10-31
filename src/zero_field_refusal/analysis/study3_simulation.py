"""Study 3's simulation before validation: the scenario models, the exact truth, one simulated
study through the registered functions and the calibration through the selection.

docs/study3-preregistration.md, "Simulation before validation": a CPU Monte Carlo of the whole
procedure (search, support, utility, validation, N, confirmation and analysis) over a frozen grid
of scenarios, with the coverage of the interval at level L screened against the registered
estimand ("Questions and estimands", "The comparisons": the expected interpolated difference over
fresh supported seed blocks, under the frozen rates). This module holds the scenarios, the harm
and label models, the truth that the coverage check compares with, one study of the procedure
and the calibration of each scenario. Pure numpy and scipy on the CPU; it reads no run
directory, no harm label, response or judge output.

- ``SCENARIOS`` (``SCENARIOS_BY_NAME``): the grid, C01–C17 (coverage; C16 and C17 at
  Δ = +2 pp), P18–P21 (power: Δ = ±2 pp at the two spreads, missing labels at the planning rate,
  M6b) and G1–G2 (the validation gates), each ``Scenario`` with its parameters and one ``source``
  line per parameter (``SOURCES``).
- ``Calibration``: what is calibrated per (scenario, candidate) through the selection: the
  per-run failure f, the safe alternative's utility shifts, the block-effect SDs and offsets.
- ``harm_curve``, ``bracket_bias``, ``CURVATURE_PP``, ``SLOPE_PP``, ``NOMINAL_BIAS_AF_PP``: the
  harm curves in ln K and their curvature.
- ``draw_effects`` and ``expected_clipped``: the block effects' shapes and E[clip(μ + σ·e, 0, 1)]
  in closed form.
- ``endpoint_labels`` (with ``endpoint_counts``): one block's primary judged shares, from 521
  labels per endpoint, and each endpoint's missing labels.
- ``truth`` (with ``normal_piecewise`` and ``support_breakpoints``): the registered estimand of a
  study at its final rates.
- ``shifted_vectors``: the seed vectors with a calibration's utility shifts applied.
- ``simulate_study`` (with ``Setup``, ``StudyOutcome``, ``synthetic_endpoint``,
  ``final_direction``, ``outcome_json`` and ``outcome_from_json``): one study through the
  registered functions; ``power_report`` (with ``report_scenario``) a power study again at the
  denser missing mask of the reported power (M6b).
- ``calibrate_scenario`` (with ``realized_support``, ``calibrate_failure``, ``calibrate_safe``,
  ``calibrate_spread``, ``calibrate_offsets``, ``calibration_stream``, ``label_variance``):
  everything calibrated per (scenario, candidate), through the selection; ``calibrate_all``
  every pair over the workers, ``calibration_problems`` the recomputation's comparison with the
  frozen calibration.
- ``run_screen``, ``run_verification`` and ``results_record`` (with ``Context``,
  ``load_context``, ``Unit``, ``part_path``, ``header``, ``run_unit``, ``execute``, ``Workers``,
  ``existing_parts``, ``first_confirmed``, ``screen_summary``, ``verification_summary``,
  ``selected_level``): the engine, parts of studies written once, resumed, and aggregated
  from the parts read back into ``screen.json``, ``verification.json`` and ``results.json``.

**The harm model** (M1–M5). In pp, with x = ln(K/K*), arm a's harm curve is
H_a(x) = 15 + β_a·x + ½·g·κ_a·x², g the scenario's ``curvature_scale`` (1, or 3 in C11). β_a and
κ_a are study 1's local slope and curvature at K*, from study 1's logistic curves at baseline 0.15
(h′ = p(1−p)s, h″ = p(1−p)(1−2p)s² + 2p(1−p)q): β A 15.300, F and P 17.7225 pp per ln K; κ A
9.5370, F and P 11.1240 pp per (ln K)². The curve is parameterized by κ itself, so that g = 3
triples the curvature (scaling the logistic's negative q would lower h″ instead). A block's
effects are additive on the probability scale: δ_F = 0, δ_A = e_AF + o_A, δ_P = e_PF + o_P, with
e_AF and e_PF independent, of the scenario's shape, SDs σ_AF and σ_PF; the offsets set the truth
averaged over the selections to Δ_AF and Δ_PF. An endpoint's harm probability is
h = clip(H_a(x)/100 + δ_a, 0, 1). The harm comes only from the planning spreads (through σ) and
study 1's curvature: no study-2 harm is read.

**The shapes** (M3): SD 1 times σ: ``normal``; ``heavy``, a normal scaled ×3 with probability 0.1,
rescaled to E[S²] = 1; ``skewed``, (G − 2)/√2 with G ~ Gamma(2); ``rare``, (B − 0.1)/√0.09 with
B ~ Bernoulli(0.1).

**The labels** (M5, M6, M6b). Per prompt (521, exchangeable, no fixed prompt effects),
y = 1[Z < Φ⁻¹(h)] with the copula Z = √ρ_b·Z_{b,i} + √(ρ_a − ρ_b)·Z_{b,a,i} + √(1 − ρ_a)·Z_{b,a,r,i}
(ρ_a 0.7 within an arm's two endpoints, ρ_b 0.3 within a block). The judge calls y = 1 harmful
with probability Se_a and y = 0 with FPR_a. A label is missing at the scenario's rate (MCAR), or
among harmful responses only (MNAR): one uniform per label, the label missing when it is below
the rate, so that on the same stream a denser rate's mask contains a sparser one's. The primary h
is the share judged harmful and not missing, over 521; each endpoint's missing count is kept, for
the sensitivities of "Judge and audit" (h_high = (harmful + missing)/521, every missing label
harmful). Given h, its expectation is
p_a(h) = (1 − m_h)·Se_a·h + (1 − m_n)·FPR_a·(1 − h), with m_h and m_n the missing rates among
harmful and not harmful responses.

**The truth** (the registered estimand, judge-rated, a missing label counting as not harmful),
per study and comparison, at the study's final rates:
truth = Σ_v E[1_A·1_F·(I_A − I_F)] / Σ_v E[1_A·1_F] over the inputs' whole seed vectors v (each
fresh block draws one, with replacement), and the same for P–F. 1_a is arm a's support in a block
("Within a block": both runs usable, untied, K_lo ≤ K* ≤ K_hi, ln K_hi − ln K_lo ≤ 0.60, no gross
collapse, the MMLU loss above 15 pp) times (1 − f)², f the calibrated per-run failure;
I_a = (1 − w)·p_a(x_lo) + w·p_a(x_hi), with p_a(x) = p_a(E[h]) and
E[h] = E[clip(H_a(x)/100 + δ_a, 0, 1)] in closed form (``expected_clipped``: normal, heavy and
rare through the clipped-normal mean of their components, μ[Φ(β) − Φ(α)] + s[φ(α) − φ(β)] +
[1 − Φ(β)] with α = −μ/s and β = (1 − μ)/s, a two-point component at s = 0; skewed through the
exact truncated moments of Gamma(2), ``scipy.special.gammainc``). No quadrature crosses the
clip's kinks. Labels add no bias (p_a is linear in h), and failures are independent of harm.
Without jitter the truth is a finite sum over the vectors. Under jitter (C12: N(0, σ_e,arm) per
run in ln K), A and F are independent given v, so the expectations factorize per arm; each arm's
two per-run jitters are integrated as nested one-dimensional integrals over the normal CDF scale
(``normal_piecewise``), with breakpoints (``support_breakpoints``) where the bracket condition
changes (x = 0), where the 0.60 width binds, at a tie, and where a gross collapse starts (the
endpoint's MMLU loss at 15 pp, a point on the piecewise-linear utility curve): in the second
jitter for each first, and in the first where the inner breakpoints change order, at the same
thresholds. Gauss–Legendre on each smooth piece, from 16 nodes, doubling until n and 2n nodes
agree within 1e-6 on the proportion scale on every piece, up to 512: a truth that does not
converge raises ``TruthNotConverged``, at every evaluation, and no coverage is computed from it.
The pieces are also split at fixed points of the normal scale (``SPLITS``: 0, ±1, ±2, ±3, ±4,
±6, ±8), so that no piece but the two outer tails (mass 6·10⁻¹⁶ each) reaches into a tail, where
Φ⁻¹'s growth would slow the rule on the CDF scale. (A plain Gauss–Hermite rule misses the
support's discontinuities: a standard normal cut at 0.3125 gives 0.5 at 20 nodes instead of
0.62267.)

**Units.** h, block effects, offsets and σ are proportions; spreads, curvature, slopes and
utility losses are pp, as ``Endpoint`` holds them.

**The procedure** (``simulate_study``; "Candidates and rate search", "Validation",
"Sample size from validation", "Confirmation", "Inference"). A study is a pure function of its
stream, scenario and index: every draw comes from ``study3_inference.stream(base, "procedure",
scenario, index, *part)``, each part (``search``, ``numerical``, ``validation``,
``confirmation``, ``labels``, and the same under ``candidate-2`` for the fallback) its own
stream, and every block draws the same numbers whatever is later read of it (P's runs
included), so that nothing depends on the worker, the order or P's activity. Each seed the
procedure touches draws one whole vector, with replacement (the bootstrap of whole seed
vectors). The registered functions decide, on simulated ``Endpoint`` records labeled with the
registered seeds (``synthetic_endpoint``), never re-implemented:
1. *Search* (``search_state``): round 0 at the starting rates on the four search seeds' vectors,
   round 1 at the adjusted rates; K-only runs never fail (M13).
2. *Repair* (``repair_state``, amendment 3): the numerical runs on the three numerical seeds'
   vectors, the qualification passed (M13), the repaired rates' runs; the final rates.
3. *Validation* (``block_state``, ``utility_gate`` at α 0.05 per bound, 22 of 28 or 31 of 40):
   each run's K is its vector's at the final rate (plus the jitter, M8), its utility the
   vector's at that K (shifted for the safe alternative, M10; above 15 pp a gross collapse); it
   fails scientifically with the calibrated probability f (M7).
4. *Spread and N*: SD = 100·sd(d_AF) over the m A–F-supported validation blocks (judged primary
   harm, full precision); N = ``sample_size(SD, m, L)`` at each level (at least ``n_floor``),
   the cap ``cap_of(N)``; a cap beyond the 200 reserved seeds leaves the study unconfirmed at
   that level (``cap_beyond_seeds``); N above 90 proceeds (M17).
5. *Confirmation* (``confirmation_state``): blocks in seed order from 10400 until the largest
   cap is started and each comparison has 99 supported blocks, or the 200 seeds are used; per
   level the stop (every active counter at N, or the cap) and the analysis sets are the
   registered function's.
6. *Analysis*: per block d = interpolated(share_lo, share_hi, w) of the arm minus F's
   (``study3_blocks.interpolated``); per (N point, comparison) one ``bootstrap_t`` seed,
   ``stream(base, "bootstrap", scenario, index, comparison, point)``, for every level, chunk
   500; covered when lower ≤ truth ≤ upper. An undetermined interval, or an N point not reached
   within the 200 blocks, is a miss (M18); the coverage takes the primary interval alone (M24).
   In a power scenario the cap gives the registered final direction (the power, M20,
   ``final_direction``): the direction common to the three computations of "Judge and audit"
   (the primary; the missing labels, the envelope of d_low = X_low − F_high and d_high = X_high −
   F_low; all missing harmful, X_high − F_high), read as the final analysis reads them
   (``study3_estimates.three_way_reading``), the four intervals on the cap's one stream; it
   detects only with a confirmatory reading (``readings``). ``utility_gate`` at α 0.05/8 gives
   the utility claim, reported only.
The two modes. ``"screen"``: the scenario's candidate with no fallback (a failed A–F gate is
``af_gate``, M14), N from the rule alone, P–F computed in every confirmed study as if active
(M16), a joint miss when either comparison misses. ``"procedure"``: the realized route, candidate
1, then candidate 2 (fresh search, repair and validation on its seeds, with its calibration),
then the scientific feasibility report (``scientific_report``); N raised to ``n_min`` when the
screen found the power short; P–F counted only where its gate passed, the joint test then A–F's
(M27). A P–F closed scientifically at candidate 1 reopens at candidate 2 when A–F's rule reaches
it (amendment 7, "P's closure"), under unchanged criteria and on that candidate's own seeds: its
P–F gate is read there, and its P runs are drawn on every route, so that every stream stays paired.

**The calibration through the selection** (``calibrate_scenario``; M1, M7, M10, M15). On the
stream (calibration base, "support", scenario, candidate) (``calibration_stream``),
``study3_decisions.CALIBRATION_STUDIES`` (10 000) studies' search, repair and 28-block validation
(``simulate_study(..., validation_only=True)``) at the rates each selects, drawn once with every
run's failure uniform kept:
- *f* (M7): by bisection on [0, 1), the realized A–F support of the validation blocks, bracket
  failures and gross collapses included, equals the scenario's s within 1e-4; f = 0, flagged,
  when the natural failures alone leave it below s. The selection does not depend on f (M13),
  so every step reuses the same draws.
- *The safe shifts* (M10): per arm each vector's MMLU curve moved by c_v = 3 − m + (λ − 1)(ū_v −
  m), λ² = (4 − W)/B over the supported blocks' interpolated MMLU loss (W within, B between the
  vectors), so that it has mean 3 pp and SD 2 pp; λ = 0, flagged, when W > 4. The factual loss
  takes one shift to mean 2 pp. The shift moves gross collapses, so f and the shifts alternate
  until neither moves (at most 10 rounds, else ``CalibrationNotStable``).
- *σ_AF, σ_PF* (M1, M2): by a bracketing root-finder (Brent, to 10⁻¹⁴ in σ), the observed
  spread of d at Δ = 0 over the supported validation blocks at the selected rates is the
  planning spread: the square root of the mean within-study variance, the quantity the spread
  rule estimates. Δ = 0 is the truth's, not the offset's: the spread is taken at the offsets
  o⁰_A, o⁰_P at which the mean truth over the selections is 0 (the curvature and a differential
  judge make it nonzero at offset 0). σ given o⁰ (its bracket from the previous round's σ) and
  o⁰ given σ alternate until neither moves by more than its solver's precision (at most 10
  rounds, else ``CalibrationNotStable``). The labels enter through their exact conditional variance
  (``label_variance``: the prompts are independent given the block, two endpoints' harmful
  labels bivariate normal thresholds), the expectation of the observed sample variance given
  the blocks, so that every σ reuses the same draws; a test checks it, study by study, against
  simulated labels.
- *o_A, o_P* (M15): solved, from o⁰ and at the calibrated σ, so that the mean, over the
  selections, of the per-study truth at the study's selected rates is Δ_AF (Δ_PF); at Δ = 0 they
  are o⁰.
σ and the offsets depend on f and the shifts, not the other way round: they are computed in the
round in which nothing moved, with the attained spreads (at o⁰) and mean truths and their Monte
Carlo SEs. The truth's moments are remembered per (vector, arm, rates, harm model, the arm's σ and
offset), since the selected rates are rounded to 3 significant digits and repeat.

**The engine** (§"Simulation before validation"; the files under ``RUNS_DIR/study3/simulation/``,
reserved). A phase (``gates``, ``screen``, ``scan-<n>``, ``verification``) draws, per scenario,
units of ``UNIT`` (250) studies in index order, each written once as a part
(``parts/<phase>/<scenario>/<first>.json``: its header, the stream, the sizes, the fixation and
the commit, and one row per study). A scenario's units are issued while the contiguous finished
prefix holds fewer than M studies for some test (only as many as the shortfall needs), never past
4·M draws (the gates: exactly M); a test takes the first M studies by index that count for it
(confirmed at the level; in the verification's P–F test, with P–F active); a test short of M at
the bound is not attainable. Each study is a pure function of (stream, scenario, index), so the
results depend neither on the workers, nor on the unit size, nor on the order in which parts
finish, nor on an interruption: a run that stopped resumes from its parts (each part's header
must be the unit's, of this fixation and commit), and every summary is computed from the parts
read back, by one code path. The workers are spawned processes with single-threaded BLAS, each
holding the run's ``Context`` (the fixation, its scenarios and calibrations, the inputs'
vectors) and the truths' moments known already. M, the resamples and the calibration's
selections are ``study3_decisions``' four adjacent constants ``COVERAGE_RUNS``,
``SCREEN_RESAMPLES``, ``VERIFICATION_RESAMPLES`` and ``CALIBRATION_STUDIES``, each read when it
is used (the calibration's by the script, which passes it), and the fixation's reader
(``study3_decisions.fixation_problems``) checks a fixation against them on every path, so that
the dry run's Part II changes those four lines and nothing else, and its reader reads them (M25).
- *gates*: G1 and G2, M validations each (28 blocks, 40 reported), the safe pass rate setting
  the validation size (28 when it reaches 0.95, else 40);
- *screen*: the non-gate scenarios at that size, four levels, ``SCREEN_RESAMPLES``,
  ``"screen"`` mode; L the first level whose tests pass (``selected_level``); the powers at
  every level, and beside them the power at 1 % MCAR (``power_report``, M6b): each power study
  of the screen again on its own streams with the denser mask, in the same part, its A–F
  detections counted over the studies the power counts, at the N the rule chose (before any
  n_min increase: not the power of the final increased design);
- *scan-<n>*: when ``power_af`` at L (0.975 without an L) is short, the power scenarios again
  from the same streams with N at least n = 60, 61, …, to the first n whose smallest power
  reaches 0.80 or whose cap exceeds the 200 reserved seeds;
- *verification*: C01–C17 on the verification stream in ``"procedure"`` mode at L,
  ``VERIFICATION_RESAMPLES``, N at least ``n_min`` when the power was short. When the scan
  stopped at an n whose cap exceeds the 200 reserved seeds, no n is attainable
  (``unattainable_n``): the verification is not run, its tests are not due and the record fails
  (the design stop, ``NO_ATTAINABLE_N``).

**Rulings** (the study's frozen settings, each with its one-line source):
- M1, the harm model of block differences: additive on the probability scale, σ calibrated so
  that the observed SD of d at Δ = 0 over the supported validation blocks at the rates the
  procedure selects is the planning spread (study 2's spread calibration,
  ``study2_short_sim.calibrate_spread``; the registration's "true spread equal to the planning
  spread"); decided.
- M2, P–F's spread and Δ_PF: equal to A–F's spread; Δ_PF = Δ_AF wherever Δ_AF ≠ 0 (C16, C17 and
  the power scenarios), 0 elsewhere (none registered: P–F's power is reported; the symmetric
  choice); decided.
- M3, the shapes: heavy ×3 at 0.1, skew Gamma(2), rare π = 0.1 centered (study 2's
  ``study2_short_sim``: ``HEAVY_*``, ``GAMMA_SHAPE``, ``RARE``, used in study 2's registered
  checks); decided.
- M4, the curvature: a quadratic in ln K with study 1's local κ at K* and slopes, g = 1, and 3
  in C11 (``study2_short_sim.STUDY1`` at K*). At the nominal symmetric bracket (±0.223, w = ½)
  the bias is ½·κ·0.223²: A 0.2371 pp, F 0.2766 pp, A−F −0.0395 pp (``NOMINAL_BIAS_AF_PP``). The
  registration states about 0.06 pp, a fitted estimate with partial cancellation between arms; the
  fixation records both and no threshold depends on them; decided.
- M5, the label model: 521 exchangeable prompts, copula ρ_a 0.7, ρ_b 0.3, baseline h(K*) 0.15
  (``study2_short_sim.Scenario``'s defaults; fixed prompt effects would make the truth
  panel-conditional); decided.
- M6, the judge: Se 0.93, FPR 0.036 in every arm; C13 F and P at Se 0.75 (study 1's Gate-1
  stratum, docs/reports/gate-1.md; sparse strata 0.58–0.75); decided.
- M6b, missing labels, the researcher's decision: MCAR 1 % in every coverage scenario and MNAR 5 %
  among harmful responses in C14, as stress values (no record holds study 3's rate: study 2's
  labels were never made); MCAR 0.1 % in the power scenarios (``POWER_MISSING_RATE``), the
  planning rate of the labels still unresolved after the labeling rounds the registration allows
  ("Judge and audit": up to three rounds per run). The 0.1 % rests on study 1's labels with the
  same judge, WildGuard: 0 unparseable outputs in the 39 900 labels of its non-collapsed runs
  (docs/reports/stage-1.md, "Judge labeling"). It is a conservative planning assumption for
  supported runs, not a proven upper bound: the bound from 0 in 39 900 assumes independent labels
  and is computed on a selected subset, the non-collapsed runs. Study 1's rate over all its runs
  includes its collapsed runs, which are not representative, and is not used. The rate is MCAR.
  Run- or arm-level clustered label failures are not explicitly modeled. The stress scenarios cover
  higher MCAR missingness and harmful-response-dependent MNAR. It was decided after preliminary
  calculations showed the power gate failing at 1 % (the worst end of the missing labels' envelope
  lies about m/521 below the primary, m the missing labels per run, and the power counts the
  reading common to the three computations, M20); it changes the planning alternative of the power
  requirement; the registration sets no rate, and the required simulation itself stays prospective.
  The coverage tests keep 1 %, so C16 and C17 (Δ = +2 pp) are coverage scenarios and P20, P21 the
  power's positive Δ. The power at 1 % MCAR is reported beside the gate, not a gate
  (``power_report``): each power study of the screen again on its own streams and its scenario's
  calibration, its labels missing at 1 % through the same uniforms (one per label, missing below
  the rate: the 1 % mask contains the 0.1 % mask), through the registered procedure (its own
  validation spread, N, cap and confirmation) to the cap's reading common to the three
  computations; it counts over the studies the power counts, and the n_min scan does not read it.
  It uses the N the rule chose, before any n_min increase, and is not the power of the final
  increased design. Under MCAR, Δ scales by 0.99 / 0.999 (±2 pp becomes about ±1.982 pp), and the
  SD need not shrink by exactly 1 %. The fixation records the rates, the disclosure and that
  report's procedure (``missing_labels_record``), and its reader checks them and that its scenarios
  are the whole grid; decided.
- M7, support: the realized probability that a validation block is A–F-supported at the rates
  the procedure selects, failures of bracket and collapse included; f calibrated to it, 0 and
  flagged when the natural failures exceed 1 − s (the registration's "support 0.88, 0.90,
  0.95"); decided.
- M8, widths and jitter: the vectors' own widths; C12 adds N(0, σ_e) per run in ln K, σ_e A 0.013,
  F 0.032, P 0.027 (docs/study2-short-b20-inputs.json); decided.
- M10, the safe alternative: each vector's MMLU curve shifted by a constant so that the
  interpolated MMLU loss over the supported validation blocks has mean 3 pp and seed SD 2 pp,
  the factual loss mean 2 pp, support 0.95 (the registration's safe alternative, at its
  boundary); decided.
- M11, the empirical point: the vectors unchanged, support 0.90 (as in M7), the realized means
  at K* reported against 3.1 pp (F) and 2.8 pp (P) (the registration's "study 2's empirical
  point"); decided.
- M12, the utility of the coverage and power scenarios: the safe alternative (the coverage is
  conditional on the confirmation starting; the gates' rates are G1's and G2's); decided.
- M13, search and numerical runs never fail, the qualification passes, the repair is simulated
  ("Not in the simulation": the numerical gates); decided.
- M14, candidates: the screen runs the scenario's candidate with no chaining; the verification
  the realized route; C15 starts at candidate 2 (the tests are per candidate); decided.
- M15, Δ's scale: the judge-rated estimand's, the offsets solved so that the truth averaged over
  the selections is Δ ("h: judge-rated" in the notation); decided.
- M16, P–F with P closed: the screen computes it as if active, the verification counts P–F only
  where the gate of the study's final candidate passed (A–F's analysis set does not depend on P's
  activity); decided.
- M17, N above 90: the simulated confirmation proceeds (the budget decision is outside the
  model); decided.
- M18, misses: an undetermined interval, or an N point not reached within 200 blocks, is a miss;
  the cap's descriptive cases count in the cap test (conservative); decided.
- M19, ``safe_pass_rate``: the A–F gate's pass rate at the safe alternative, the P–F gate's
  rate reported beside it (the A–F gate is the one that moves candidates); decided.
- M20, the power: normal shape, Δ = ±2 pp, ``power_af`` the smallest correct-direction rate over
  the power scenarios P18–P21 (the two spreads and signs, missing labels at the planning rate,
  M6b) at L; ``n_min`` by a linear scan from 60 ("Power ≥ 80 % … at
  |Δ| = 2 pp … the planning spread"); a study detects when its cap's reading is confirmatory and
  the direction common to the three computations ("Judge and audit": "the strongest conclusion
  that holds in all three computations is reported") is Δ's sign; decided.
- M21, the binomial interval of each test: Clopper–Pearson 95 % ("ordinary 95 % binomial
  interval"); decided.
- M22, the grid: 17 coverage scenarios (J = 1 785), 4 power and 2 gate scenarios
  (``SCENARIOS``); decided.
- M23, the stream integers: calibration 20261101, verification 20261102 (``STREAMS``; any two
  distinct integers); decided.
- M24, the coverage test's interval: the primary computation's only (the registration's coverage
  concerns "the interval"); decided.
- M25, the sizes of the dry run's Part II (amendment 4): M = 200, 200 screen and 1 000
  verification resamples and 1 000 calibration selections per (scenario, candidate), set in
  that run's own commit through ``study3_decisions``' ``COVERAGE_RUNS``, ``SCREEN_RESAMPLES``,
  ``VERIFICATION_RESAMPLES`` and ``CALIBRATION_STUDIES`` alone, four adjacent lines (the fixation
  records the last in ``calibration_record``, and its reader checks all four); decided.
- M26, the bound on draws: each test draws at most 4·M studies; a test short of M confirmed
  studies at the bound is not attainable and fails; a power rate counts the missing studies as
  not detecting (a bounded failure route); decided.
- M27, the joint test in the verification: over the studies confirmed for A–F, a miss when A–F
  misses or when P–F is active and misses (the registration's joint p₀ 0.05 over the two
  comparisons); decided.
- The constants taken from study 2's simulation (``STUDY1``'s logistic slopes and curvatures,
  the shapes' constants, ρ_a, ρ_b and ``THREAD_VARIABLES``) are written out here so that this
  module loads no GPU path through study 2's analysis; a test compares them with
  ``study2_short_sim``.
- The bracket compares ln K with ln K* (the measured K's order), the width allows 10⁻⁹ in ln K and
  a usable run's MMLU loss 10⁻⁹ pp, as ``study3_blocks`` and ``study3_records`` do.
- The jitter quadrature (C12) is split so that the part over the runs' jitters, which depends on
  neither the block effects' σ nor the offset, is computed once per vector, arm and pair of rates
  and kept across the calibration's σ and offset steps (``_JitterGrid``); the outer rule, which
  carries the shares, is checked for convergence at every evaluation.
- ``run`` recomputes the calibration and compares it with the frozen one: integers, flags and
  identities exactly; floats (from the solvers and the quadrature) equal when they agree within
  a relative 10⁻⁹ or within an absolute floor at the solvers' own precision, 10⁻⁹ for a pp field
  (the attained spreads and Δs, the utility shifts), 10⁻¹¹ for a proportion (f, the support, σ,
  the offsets o and o⁰) and 10⁻¹⁰ for λ (``CALIBRATION_FLOORS``), so that a result near 0, such
  as the attained Δ at Δ = 0, is compared by its precision (``calibration_problems``); the
  fixation records the machine it was calibrated on (``machine_record``: the platform, the CPU
  model, numpy's version), and a different machine is reported, not refused.
- The reported quantiles of the realized blocks are type-1 order statistics (zero-based rank
  ⌈n·q⌉ − 1, ``study3_inference.tail_ranks``' rule; ``order_statistic``), never an interpolated
  quantile.
- The reported figures of a screened scenario count the studies its tests drew (``attempted``,
  the largest draws over its levels); a study unconfirmed for a reason counts once for it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, ndtr, ndtri

from . import study3_decisions as _decisions  # M and the gates read at call time: one definition
from .study3_blocks import (
    CONFIRMATION_ALPHA,
    VALIDATION_ALPHA,
    WIDTH,
    block_state,
    interpolated,
    utility_gate,
)
from .study3_blocks import SLACK as WIDTH_SLACK
from .study3_decisions import (
    LEVELS,
    MAX_DRAWS_FACTOR,
    MIN_N,
    N_POINTS,
    SIMULATION_FIXATION,
    SIMULATION_RESULTS,
    SUPPORTED_AT,
    TESTED,
    VALIDATION_BLOCKS,
    cap_of,
    coverage_threshold,
    detection_power,
    fixation_problems,
    identities_sha256,
    level_slot,
    sample_size,
    test_identities,
)
from .study3_estimates import envelope, three_way_reading
from .study3_inference import Interval, bootstrap_t, reading, stream
from .study3_progress import CONFIRMATORY, confirmation_state
from .study3_records import (
    EXPERIMENT,
    GROSS_COLLAPSE,
    GROSS_COLLAPSE_PP,
    INSTABILITY,
    SEEDS,
    STEP_ROLES,
    TOLERANCE_PP,
    UPDATES,
    Cell,
    Endpoint,
)
from .study3_records import SIMULATION as SIMULATION_FOLDER
from .study3_search import (
    ARM_ORDER,
    CANDIDATES,
    LABELS,
    QUALIFIED,
    REPAIR_STEP,
    repair_state,
    search_state,
)
from .study3_seed_vectors import Vector

# ------------------------------------------------------------------ constants

SHAPES = ("normal", "heavy", "skewed", "rare")
ROLES = ("coverage", "power", "gate")
UTILITIES = ("safe", "empirical")
MISSING = ("MCAR", "MNAR")
COMPARISONS = {"A-F": ("A", "F"), "P-F": ("P", "F")}
# M6: (Se, FPR) per arm
JUDGES: dict[str, dict[str, tuple[float, float]]] = {
    "equal": {"A": (0.93, 0.036), "F": (0.93, 0.036), "P": (0.93, 0.036)},
    "differential": {"A": (0.93, 0.036), "F": (0.75, 0.036), "P": (0.75, 0.036)},
}
RHO_ARM, RHO_BLOCK, BASELINE, PROMPTS = 0.7, 0.3, 0.15, 521  # M5
# M6b: the missing labels' rates, MCAR (C14: MNAR, 5 % of the harmful responses)
COVERAGE_MISSING_RATE = 0.01  # the coverage scenarios': a stress value
POWER_MISSING_RATE = 0.001  # the power scenarios': the planning rate of unresolved labels
REPORT_MISSING_RATE = 0.01  # the power reported beside the gate, not a gate: the coverage rate
# M8: σ_e of ln K per run (docs/study2-short-b20-inputs.json)
JITTER_SD = {"A": 0.013, "F": 0.032, "P": 0.027}
# M3: the shapes' constants (study2_short_sim's HEAVY_SHARE, HEAVY_RATIO, GAMMA_SHAPE, RARE)
HEAVY_SHARE, HEAVY_RATIO = 0.1, 3.0
HEAVY_SCALES = tuple(
    s / math.sqrt(HEAVY_SHARE * HEAVY_RATIO**2 + (1 - HEAVY_SHARE)) for s in (HEAVY_RATIO, 1.0)
)
GAMMA_SHAPE = 2.0
RARE = 0.1
RARE_POINTS = ((1 - RARE) / math.sqrt(RARE * (1 - RARE)), -RARE / math.sqrt(RARE * (1 - RARE)))
# M4: study 1's logistic curves at K* (study2_short_sim.STUDY1: slope s and curvature q per arm)
STUDY1_LOGISTIC = {"A": (1.20, -0.13), "F": (1.39, -0.24), "P": (1.39, -0.24)}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
NOMINAL_HALF_WIDTH = 0.223  # ln(7.5/6): the nominal symmetric bracket of candidate 1
# The truth's quadrature
TOLERANCE = 1e-6  # n and 2n nodes agree within this on every piece (proportion scale)
FIRST_NODES, MAX_NODES = 16, 512
# Fixed splits of the normal scale beside the breakpoints: a piece of u = Φ(z) that reached far
# into a tail would make Φ⁻¹'s growth at u = 0 or 1 a singularity of the integrand in u; split
# here, every piece but the two tails (mass 6·10⁻¹⁶ each) is a bounded interval of z
SPLITS = (-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
COLLAPSE_AT_PP = GROSS_COLLAPSE_PP + TOLERANCE_PP  # a usable run loses at most this


def _slope_pp(arm: str) -> float:
    s, _ = STUDY1_LOGISTIC[arm]
    return 100 * BASELINE * (1 - BASELINE) * s


def _curvature_pp(arm: str) -> float:
    s, q = STUDY1_LOGISTIC[arm]
    p = BASELINE
    return 100 * (p * (1 - p) * (1 - 2 * p) * s**2 + 2 * p * (1 - p) * q)


SLOPE_PP = {arm: _slope_pp(arm) for arm in ARM_ORDER}  # β_a, pp per ln K
CURVATURE_PP = {arm: _curvature_pp(arm) for arm in ARM_ORDER}  # κ_a, pp per (ln K)²
NOMINAL_BIAS_AF_PP = 0.5 * (CURVATURE_PP["A"] - CURVATURE_PP["F"]) * NOMINAL_HALF_WIDTH**2

# One source line per scenario parameter: where its value comes from, then its decision (M<n>)
SOURCES: dict[str, str] = {
    "candidate": (
        "the coverage and power tests are per candidate: the screen runs the scenario's "
        "candidate with no chaining, the verification the realized route (candidate 1, then "
        "candidate 2, then the scientific feasibility report); C15 starts at candidate 2; "
        "M14, decided"
    ),
    "spread_pp": (
        "the registration's 'true spread equal to the planning spread' (3.5 and 5 pp), the "
        "observed SD of d at delta 0 over the supported validation blocks at the rates the "
        "procedure selects, as study 2's spread calibration (study2_short_sim.calibrate_spread); "
        "M1, decided"
    ),
    "shape": (
        "study2_short_sim (HEAVY_*, GAMMA_SHAPE, RARE), used in study 2's registered checks: "
        "heavy x3 at 0.1, skew Gamma(2), rare pi = 0.1 centered; M3, decided"
    ),
    "delta_af_pp": (
        "'h: judge-rated' in the notation and the power requirement at |delta| = 2 pp: the "
        "population truth averaged over the selection distribution is delta_AF; M15, decided"
    ),
    "delta_pf_pp": (
        "none registered (P-F's power is reported): the symmetric choice, delta_PF = delta_AF "
        "wherever delta_AF is not 0 (C16, C17 and the power scenarios) and 0 elsewhere, the P-F "
        "spread equal to A-F's; M2, decided"
    ),
    "support": (
        "the registration's support 0.88, 0.90, 0.95, read as the realized probability that a "
        "validation block is A-F-supported, bracket and collapse failures included; f "
        "calibrated to it; M7, decided"
    ),
    "curvature_scale": (
        "study 1's local curvature at K* (study2_short_sim.STUDY1): a quadratic in ln K, g = 1, "
        "and 3 in C11; M4, decided"
    ),
    "jitter": (
        "the vectors' own interval widths; C12 adds N(0, sigma_e) per run in ln K, sigma_e A "
        "0.013, F 0.032, P 0.027 (docs/study2-short-b20-inputs.json); M8, decided"
    ),
    "judge": (
        "study 1's Gate-1 stratum (docs/reports/gate-1.md: Se 0.93, FPR 0.036) in every arm; "
        "C13 F and P at Se 0.75 (sparse strata 0.58-0.75); M6, decided"
    ),
    "missing": (
        "MCAR in every scenario but C14, MNAR among harmful responses there (the fixation's "
        "missing_labels). Run- or arm-level clustered label failures are not explicitly "
        "modeled. The stress scenarios cover higher MCAR missingness and "
        "harmful-response-dependent MNAR. M6b, decided"
    ),
    "missing_rate": (
        "1 % MCAR in the coverage scenarios and 5 % MNAR among harmful in C14, as stress values "
        "(no record holds study 3's rate: study 2's labels were never made); 0.1 % MCAR in the "
        "power scenarios, the planning rate of labels unresolved after the labeling rounds, from "
        "study 1 with the same judge (0 unparseable outputs in 39 900 labels of its "
        "non-collapsed runs, docs/reports/stage-1.md), a conservative planning assumption, not "
        "a proven bound (the fixation's missing_labels); M6b, decided"
    ),
    "utility": (
        "the registration's safe alternative at its boundary (interpolated MMLU loss mean 3 pp, "
        "seed SD 2 pp, factual 2 pp over the supported blocks; M10) in every coverage and power "
        "scenario and G1; study 2's empirical point in G2 (M11); M12, decided"
    ),
}


class TruthNotConverged(ArithmeticError):
    """A quadrature of the truth whose n and 2n nodes disagree at the largest node count."""


# ------------------------------------------------------------------ the scenarios


@dataclass(frozen=True)
class Scenario:
    """One scenario of the grid (module docstring; the registration's "Simulation before
    validation", "Scenarios"). ``None`` marks a parameter a gate scenario does not use."""

    name: str
    role: str
    candidate: int
    spread_pp: float | None
    shape: str | None
    delta_af_pp: float | None
    delta_pf_pp: float | None
    support: float
    curvature_scale: float | None
    jitter: bool
    judge: str | None
    missing: str | None
    missing_rate: float | None
    utility: str
    sources: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.sources:
            object.__setattr__(
                self, "sources", tuple((name, SOURCES[name]) for name in self.parameters())
            )
        problems = []
        if not (isinstance(self.name, str) and self.name):
            problems.append(f"name {self.name!r}")
        if self.role not in ROLES:
            problems.append(f"role {self.role!r}")
        if not isinstance(self.jitter, bool):
            problems.append(f"jitter {self.jitter!r}")
        if self.candidate not in CANDIDATES or isinstance(self.candidate, bool):
            problems.append(f"candidate {self.candidate!r}")
        if not (isinstance(self.support, float) and 0 < self.support < 1):
            problems.append(f"support {self.support!r}")
        if self.utility not in UTILITIES:
            problems.append(f"utility {self.utility!r}")
        if self.role != "gate":
            if not (isinstance(self.spread_pp, float) and self.spread_pp > 0):
                problems.append(f"spread {self.spread_pp!r}")
            if self.shape not in SHAPES:
                problems.append(f"shape {self.shape!r}")
            if self.judge not in JUDGES:
                problems.append(f"judge {self.judge!r}")
            if self.missing not in MISSING or not (
                isinstance(self.missing_rate, float) and 0 <= self.missing_rate < 1
            ):
                problems.append(f"missing labels {self.missing!r} at {self.missing_rate!r}")
            if not (isinstance(self.curvature_scale, float) and self.curvature_scale >= 0):
                problems.append(f"curvature scale {self.curvature_scale!r}")
            for delta in (self.delta_af_pp, self.delta_pf_pp):
                if not (isinstance(delta, float) and math.isfinite(delta)):
                    problems.append(f"Δ {delta!r}")
        if [name for name, _ in self.sources] != list(self.parameters()) or not all(
            isinstance(line, str) and line for _, line in self.sources
        ):
            problems.append("sources must give one line per parameter, in order")
        if problems:
            raise ValueError(f"scenario {self.name!r}: {', '.join(problems)}")

    @classmethod
    def parameters(cls) -> tuple[str, ...]:
        """The fields that carry a ``source`` line: every field but the name, the role and the
        sources themselves."""
        return tuple(f.name for f in fields(cls) if f.name not in ("name", "role", "sources"))

    @property
    def coverage(self) -> bool:
        """Whether the scenario's tests count in J (the "J" column)."""
        return self.role == "coverage"

    @property
    def power(self) -> bool:
        """Whether the scenario is one the power gate reads (M20)."""
        return self.role == "power"

    def missing_rates(self) -> tuple[float, float]:
        """(m_h, m_n): the missing rates among harmful and not harmful responses."""
        rate = self.missing_rate or 0.0
        return (rate, rate) if self.missing == "MCAR" else (rate, 0.0)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "sources"
        }
        out["sources"] = dict(self.sources)
        return out

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Scenario:
        names = {f.name for f in fields(cls)}
        if not isinstance(d, Mapping) or set(d) != names:
            raise ValueError(f"a scenario must hold exactly {sorted(names)}")
        sources = d["sources"]
        if not isinstance(sources, Mapping) or set(sources) != set(cls.parameters()):
            raise ValueError(f"a scenario's sources must name exactly {list(cls.parameters())}")
        values = {k: v for k, v in d.items() if k != "sources"}
        for key in (
            "spread_pp",
            "delta_af_pp",
            "delta_pf_pp",
            "support",
            "curvature_scale",
            "missing_rate",
        ):
            if isinstance(values[key], int) and not isinstance(values[key], bool):
                values[key] = float(values[key])
        return cls(**values, sources=tuple((p, sources[p]) for p in cls.parameters()))


def _scenario(name: str, role: str = "coverage", **change: Any) -> Scenario:
    common: dict[str, Any] = {
        "candidate": 1,
        "spread_pp": 5.0,
        "shape": "normal",
        "delta_af_pp": 0.0,
        "delta_pf_pp": 0.0,
        "support": 0.90,
        "curvature_scale": 1.0,
        "jitter": False,
        "judge": "equal",
        "missing": "MCAR",
        "missing_rate": COVERAGE_MISSING_RATE,
        "utility": "safe",
    }
    return Scenario(name=name, role=role, **{**common, **change})


def _power(name: str, spread_pp: float, delta_pp: float) -> Scenario:
    """A power scenario (M20): normal, Δ_AF = Δ_PF = ``delta_pp``, its missing labels MCAR at the
    planning rate (M6b)."""
    return _scenario(name, "power", spread_pp=spread_pp, delta_af_pp=delta_pp,
                     delta_pf_pp=delta_pp, missing_rate=POWER_MISSING_RATE)  # fmt: skip


_GATE = {
    "spread_pp": None,
    "shape": None,
    "delta_af_pp": None,
    "delta_pf_pp": None,
    "curvature_scale": None,
    "judge": None,
    "missing": None,
    "missing_rate": None,
}
SCENARIOS: tuple[Scenario, ...] = (
    _scenario("C01", spread_pp=3.5),
    _scenario("C02"),
    _scenario("C03", spread_pp=3.5, shape="heavy"),
    _scenario("C04", shape="heavy"),
    _scenario("C05", spread_pp=3.5, shape="skewed"),
    _scenario("C06", shape="skewed"),
    _scenario("C07", spread_pp=3.5, shape="rare"),
    _scenario("C08", shape="rare"),
    _scenario("C09", support=0.88),
    _scenario("C10", support=0.95),
    _scenario("C11", curvature_scale=3.0),
    _scenario("C12", jitter=True),
    _scenario("C13", judge="differential"),
    _scenario("C14", missing="MNAR", missing_rate=0.05),
    _scenario("C15", candidate=2),
    _scenario("C16", spread_pp=3.5, delta_af_pp=2.0, delta_pf_pp=2.0),
    _scenario("C17", delta_af_pp=2.0, delta_pf_pp=2.0),
    _power("P18", 3.5, -2.0),
    _power("P19", 5.0, -2.0),
    _power("P20", 3.5, 2.0),
    _power("P21", 5.0, 2.0),
    _scenario("G1", "gate", support=0.95, **_GATE),
    _scenario("G2", "gate", utility="empirical", **_GATE),
)
SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


@dataclass(frozen=True)
class Calibration:
    """What is calibrated for one (scenario, candidate), through the selection: ``failure``
    (f, the per-run scientific failure probability), ``realized_support`` and
    ``support_flagged`` (M7); the safe alternative's ``mmlu_shift`` (arm → vector seed → c_v,
    pp), ``factual_shift`` (arm → pp), ``utility_realized`` (arm → the interpolated MMLU mean,
    its seed SD and the factual mean over the supported blocks, pp) and ``utility_flagged``
    (M10; empty shifts for the empirical point, M11); ``sigma_af`` and ``sigma_pf`` (the block
    effects' SDs, proportions), ``offset_a`` and ``offset_p`` (proportions), the attained spreads
    ``attained_af_pp`` and ``attained_pf_pp`` (M1) and ``attained_delta_pp`` (comparison → the
    mean truth over the selections and its Monte Carlo SE, pp; M15); ``utility_lambda`` (arm →
    λ of the safe shifts, M10), ``attained_spread_se_pp`` (comparison → the attained spread's
    Monte Carlo SE, pp, M1) and ``null_offset_a``, ``null_offset_p`` (o⁰: the offsets at which
    the mean truth is 0, where σ is calibrated and the spread attained, proportions; M1)."""

    candidate: int
    failure: float
    realized_support: float
    support_flagged: bool
    mmlu_shift: Mapping[str, Mapping[int, float]]
    factual_shift: Mapping[str, float]
    utility_realized: Mapping[str, tuple[float, float, float]]
    utility_flagged: bool
    sigma_af: float
    sigma_pf: float
    offset_a: float
    offset_p: float
    attained_af_pp: float
    attained_pf_pp: float
    attained_delta_pp: Mapping[str, tuple[float, float]]
    utility_lambda: Mapping[str, float] = field(default_factory=dict)
    attained_spread_se_pp: Mapping[str, float] = field(default_factory=dict)
    null_offset_a: float = 0.0
    null_offset_p: float = 0.0

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES or isinstance(self.candidate, bool):
            raise ValueError(f"no candidate {self.candidate!r}")
        if not (math.isfinite(self.failure) and 0 <= self.failure < 1):
            raise ValueError(f"the per-run failure {self.failure!r} is not in [0, 1)")
        for name in ("sigma_af", "sigma_pf"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} {value!r} is not a finite SD")
        for name in ("offset_a", "offset_p", "null_offset_a", "null_offset_p"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} is not finite")

    def sigma(self, arm: str) -> float:
        """The SD of arm ``arm``'s block effect (F has none)."""
        return {"A": self.sigma_af, "F": 0.0, "P": self.sigma_pf}[arm]

    def offset(self, arm: str) -> float:
        return {"A": self.offset_a, "F": 0.0, "P": self.offset_p}[arm]

    def at_null(self) -> Calibration:
        """This calibration at the Δ = 0 offsets o⁰ (where σ is calibrated, M1)."""
        return replace(self, offset_a=self.null_offset_a, offset_p=self.null_offset_p)


# ------------------------------------------------------------------ the harm and label models


def harm_curve(arm: str, x: np.ndarray, curvature_scale: float) -> np.ndarray:
    """H_a(x)/100, x = ln(K/K*): (15 + β_a·x + ½·g·κ_a·x²)/100, in proportions."""
    x = np.asarray(x, dtype=float)
    return (
        100 * BASELINE + SLOPE_PP[arm] * x + 0.5 * curvature_scale * CURVATURE_PP[arm] * x**2
    ) / 100


def bracket_bias(arm: str, half_width: float, curvature_scale: float) -> float:
    """The interpolation bias at a symmetric bracket (±``half_width`` in ln K, w = ½):
    ½·(H(−h) + H(h)) − H(0) = ½·g·κ·h², in proportions."""
    ends = harm_curve(arm, np.array([-half_width, half_width]), curvature_scale)
    return float(0.5 * ends.sum() - harm_curve(arm, np.array(0.0), curvature_scale))


def draw_effects(shape: str, rng: np.random.Generator, size: int) -> np.ndarray:
    """``size`` block effects of ``shape``, each with mean 0 and SD 1 (M3)."""
    if shape == "normal":
        return rng.standard_normal(size)
    if shape == "heavy":
        scale = np.where(rng.random(size) < HEAVY_SHARE, HEAVY_SCALES[0], HEAVY_SCALES[1])
        return scale * rng.standard_normal(size)
    if shape == "skewed":
        return (rng.gamma(GAMMA_SHAPE, 1.0, size) - GAMMA_SHAPE) / math.sqrt(GAMMA_SHAPE)
    if shape == "rare":
        return np.where(rng.random(size) < RARE, RARE_POINTS[0], RARE_POINTS[1])
    raise ValueError(f"no shape {shape!r} (shapes: {SHAPES})")


def _clipped_normal(mean: np.ndarray, s: float) -> np.ndarray:
    """E[clip(μ + s·Z, 0, 1)], Z standard normal; clip(μ) at s = 0."""
    mean = np.asarray(mean, dtype=float)
    if s == 0:
        return np.clip(mean, 0.0, 1.0)
    a, b = -mean / s, (1 - mean) / s
    phi_a, phi_b = np.exp(-0.5 * a * a), np.exp(-0.5 * b * b)
    return mean * (ndtr(b) - ndtr(a)) + s * (phi_a - phi_b) / math.sqrt(2 * math.pi) + (1 - ndtr(b))


def expected_clipped(mean: np.ndarray, sigma: float, shape: str) -> np.ndarray:
    """E[clip(μ + σ·e, 0, 1)] for e of ``shape`` (mean 0, SD 1), elementwise in μ, in closed
    form (module docstring)."""
    mean = np.asarray(mean, dtype=float)
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError(f"σ must be finite and at or above 0, not {sigma!r}")
    if sigma == 0:
        return np.clip(mean, 0.0, 1.0)
    if shape == "normal":
        return _clipped_normal(mean, sigma)
    if shape == "heavy":
        return HEAVY_SHARE * _clipped_normal(mean, sigma * HEAVY_SCALES[0]) + (
            1 - HEAVY_SHARE
        ) * _clipped_normal(mean, sigma * HEAVY_SCALES[1])
    if shape == "rare":
        return RARE * np.clip(mean + sigma * RARE_POINTS[0], 0.0, 1.0) + (1 - RARE) * np.clip(
            mean + sigma * RARE_POINTS[1], 0.0, 1.0
        )
    if shape == "skewed":
        # Y = a + b·G, G ~ Gamma(2): the clip's kinks at G = −a/b and (1 − a)/b
        k = GAMMA_SHAPE
        b = sigma / math.sqrt(k)
        a = mean - sigma * math.sqrt(k)
        lo = np.maximum(-a / b, 0.0)
        hi = np.maximum((1 - a) / b, 0.0)
        mass = gammainc(k, hi) - gammainc(k, lo)
        first = k * (gammainc(k + 1, hi) - gammainc(k + 1, lo))  # E[G; lo < G < hi]
        return a * mass + b * first + (1 - gammainc(k, hi))
    raise ValueError(f"no shape {shape!r} (shapes: {SHAPES})")


def judged_share(arm: str, h: np.ndarray, scenario: Scenario) -> np.ndarray:
    """p_a(h) = (1 − m_h)·Se_a·h + (1 − m_n)·FPR_a·(1 − h): the expected primary judged share
    at harm probability h."""
    se, fpr = JUDGES[scenario.judge][arm]
    m_h, m_n = scenario.missing_rates()
    h = np.asarray(h, dtype=float)
    return (1 - m_h) * se * h + (1 - m_n) * fpr * (1 - h)


def endpoint_labels(
    h: np.ndarray, arms: Sequence[str], scenario: Scenario, rng: np.random.Generator
) -> np.ndarray:
    """The primary judged shares of one block's endpoints (``endpoint_counts``' harmful count
    over the 521 prompts)."""
    return endpoint_counts(h, arms, scenario, rng)[0] / PROMPTS


def endpoint_counts(
    h: np.ndarray, arms: Sequence[str], scenario: Scenario, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """The label counts of one block's endpoints (h of shape (6,), one per endpoint of ``arms``,
    each arm's two endpoints together; four once P is closed): per prompt the copula's harmful
    label, the judge's call and the missing label; per endpoint the count judged harmful and not
    missing, and the count missing, of the 521 prompts ("Judge and audit": a missing label is
    not harmful in the primary computation, harmful in its sensitivities)."""
    h = np.asarray(h, dtype=float)
    arms = list(arms)
    if h.shape != (len(arms),) or not set(arms) <= set(ARM_ORDER):
        raise ValueError(f"one harm probability per endpoint of {arms}, not {h.shape}")
    thresholds = ndtri(np.clip(h, 0.0, 1.0))
    block = rng.standard_normal(PROMPTS)
    by_arm = {arm: rng.standard_normal(PROMPTS) for arm in ARM_ORDER if arm in arms}
    own = rng.standard_normal((len(arms), PROMPTS))
    z = (
        math.sqrt(RHO_BLOCK) * block
        + math.sqrt(RHO_ARM - RHO_BLOCK) * np.stack([by_arm[a] for a in arms])
        + math.sqrt(1 - RHO_ARM) * own
    )
    harmful = z < thresholds[:, None]
    se = np.array([JUDGES[scenario.judge][a][0] for a in arms])[:, None]
    fpr = np.array([JUDGES[scenario.judge][a][1] for a in arms])[:, None]
    calls = rng.random((len(arms), PROMPTS))
    judged = np.where(harmful, calls < se, calls < fpr)
    m_h, m_n = scenario.missing_rates()
    lost = rng.random((len(arms), PROMPTS)) < np.where(harmful, m_h, m_n)
    return (judged & ~lost).sum(axis=1), lost.sum(axis=1)


# ------------------------------------------------------------------ the utility shifts


def shifted_vectors(vectors: Sequence[Vector], calibration: Calibration) -> list[Vector]:
    """The vectors with the calibration's utility shifts (M10): each arm's MMLU losses moved by
    the vector's c_v and its factual losses by the arm's common shift. Empty shifts (the
    empirical point, M11) leave the vectors unchanged. ``ValueError`` for shifts that do not
    name every arm, and every vector's seed for the MMLU."""
    if not calibration.mmlu_shift and not calibration.factual_shift:
        return list(vectors)
    out = []
    for v in vectors:
        arms = {}
        for arm, points in v.arms.items():
            try:
                c = calibration.mmlu_shift[arm][v.seed]
                f = calibration.factual_shift[arm]
            except KeyError as exc:
                raise ValueError(f"no utility shift for arm {arm} of seed {v.seed}") from exc
            arms[arm] = tuple(
                replace(p, mmlu_loss_pp=p.mmlu_loss_pp + c, factual_loss_pp=p.factual_loss_pp + f)
                for p in points
            )
        out.append(Vector(seed=v.seed, arms=arms))
    return out


# ------------------------------------------------------------------ the quadrature


@functools.cache
def _legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


_U_LOW, _U_HIGH = 1e-300, float(np.nextafter(1.0, 0.0))


def _piecewise(
    fn: Callable[[np.ndarray], np.ndarray],
    breaks: np.ndarray,
    *,
    tol: float,
    max_nodes: int,
) -> np.ndarray:
    """A batch of integrals E[fn(Z)], Z standard normal, Gauss–Legendre on u = Φ(z) between
    each row's breakpoints (and ``SPLITS``). ``breaks``: (B, K) breakpoints in z per row; ``fn``
    maps z of shape (B, P, n) to values of shape (C, B, P, n). Returns (C, B). From
    ``FIRST_NODES`` nodes per piece the count doubles until n and 2n agree within ``tol`` on
    every piece, up to ``max_nodes``; otherwise ``TruthNotConverged``."""
    breaks = np.asarray(breaks, dtype=float)
    splits = np.broadcast_to(np.array(SPLITS), (len(breaks), len(SPLITS)))
    u = np.sort(ndtr(np.concatenate([breaks, splits], axis=1)), axis=1)
    edges = np.concatenate([np.zeros((len(u), 1)), u, np.ones((len(u), 1))], axis=1)
    half = (edges[:, 1:] - edges[:, :-1]) / 2  # (B, P)
    mid = (edges[:, 1:] + edges[:, :-1]) / 2
    kept = np.any(half > 0, axis=0)  # a piece of length 0 in every row adds exactly nothing
    half, mid = half[:, kept], mid[:, kept]

    def rule(n: int) -> np.ndarray:
        t, w = _legendre(n)
        z = ndtri(np.clip(mid[..., None] + half[..., None] * t, _U_LOW, _U_HIGH))
        return (np.asarray(fn(z)) * w).sum(axis=-1) * half  # (C, B, P)

    n = FIRST_NODES
    if 2 * n > max_nodes:
        raise TruthNotConverged(f"{max_nodes} nodes cannot check {n} against {2 * n}")
    coarse = rule(n)
    while True:
        fine = rule(2 * n)
        if np.all(np.abs(fine - coarse) <= tol):
            return fine.sum(axis=-1)
        n *= 2
        if 2 * n > max_nodes:
            worst = float(np.max(np.abs(fine - coarse)))
            raise TruthNotConverged(
                f"{n} and {n // 2} nodes differ by {worst:.3g} (tolerance "
                f"{tol:g}, at most {max_nodes} nodes)"
            )
        coarse = fine


def normal_piecewise(
    fn: Callable[[np.ndarray], np.ndarray],
    breakpoints: Sequence[float],
    *,
    tol: float = TOLERANCE,
    max_nodes: int = MAX_NODES,
) -> float:
    """E[fn(Z)], Z standard normal, with Gauss–Legendre on u = Φ(z) between the sorted
    breakpoints (and the fixed ``SPLITS`` of the normal scale): nodes double from 16 until n and
    2n agree within ``tol`` on every piece, else ``TruthNotConverged``. ``fn`` maps an array of
    z to an array of values elementwise."""
    breaks = np.asarray(list(breakpoints), dtype=float).reshape(1, -1)
    return float(
        _piecewise(
            lambda z: np.asarray(fn(z), dtype=float)[None], breaks, tol=tol, max_nodes=max_nodes
        )[0, 0]
    )


# ------------------------------------------------------------------ support and the truth


def _support(
    x1: np.ndarray, x2: np.ndarray, usable1: np.ndarray, usable2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An arm's support from its two runs' x = ln(K/K*) and whether each is usable (no gross
    collapse), elementwise ("Within a block"): (supported, x_lo, x_hi, w), w = −x_lo/(x_hi −
    x_lo) where supported (0 elsewhere)."""
    lo, hi = np.minimum(x1, x2), np.maximum(x1, x2)
    span = hi - lo
    supported = (
        usable1 & usable2 & (span > 0) & (lo <= 0) & (hi >= 0) & (span <= WIDTH + WIDTH_SLACK)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(supported, -lo / np.where(span > 0, span, 1.0), 0.0)
    return supported, lo, hi, w


def _arm_terms(
    vector: Vector,
    arm: str,
    m: tuple[float, float],
    sigma_e: float,
    kstar_ln: float,
    slopes: Mapping[str, tuple[float, float]],
    scenario: Scenario,
    calibration: Calibration,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The integrand of an arm in one vector: (z1, z2) → (1_a, 1_a·I_a) before the failures,
    z the runs' standardized jitters."""
    effect_sd, offset = calibration.sigma(arm), calibration.offset(arm)

    def share(x: np.ndarray) -> np.ndarray:
        mean = harm_curve(arm, x, scenario.curvature_scale) + offset
        return judged_share(arm, expected_clipped(mean, effect_sd, scenario.shape), scenario)

    def usable(x: np.ndarray) -> np.ndarray:
        return vector.losses_at(arm, kstar_ln + x, slopes)[0] <= COLLAPSE_AT_PP

    def terms(z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x1, x2 = m[0] + sigma_e * np.asarray(z1), m[1] + sigma_e * np.asarray(z2)
        usable1, usable2 = usable(x1), usable(x2)  # the first run's before broadcasting
        x1, x2, usable1, usable2 = np.broadcast_arrays(x1, x2, usable1, usable2)
        supported, lo, hi, w = _support(x1, x2, usable1, usable2)
        value = np.zeros(supported.shape)
        if supported.any():  # the shares only where the arm is supported
            at, weight = supported, w[supported]
            value[at] = (1 - weight) * share(lo[at]) + weight * share(hi[at])
        return supported.astype(float), value

    return terms


def _crossings(
    vector: Vector, arm: str, slopes: Mapping[str, tuple[float, float]], kstar_ln: float
) -> list[float]:
    """The x = ln(K/K*) where the arm's MMLU loss reaches the gross collapse."""
    return [c - kstar_ln for c in vector.mmlu_crossings(arm, COLLAPSE_AT_PP, slopes)]


def _x_breaks_outer(crossings: Sequence[float]) -> list[float]:
    """The first run's x where the support of either run or the order of the inner breakpoints
    changes: the bracket (0), the width (±0.60), a collapse c and c ± 0.60."""
    width = WIDTH + WIDTH_SLACK
    return [
        0.0,
        -width,
        width,
        *crossings,
        *(c - width for c in crossings),
        *(c + width for c in crossings),
    ]


def _x_breaks_inner(x1: np.ndarray, crossings: Sequence[float]) -> np.ndarray:
    """For each first run's x, the second run's x where the support changes: the bracket (0),
    the width (x1 ± 0.60), a tie (x1) and a collapse; shape (len(x1), K)."""
    width = WIDTH + WIDTH_SLACK
    x1 = np.asarray(x1, dtype=float).reshape(-1, 1)
    fixed = np.broadcast_to(np.array([0.0, *crossings]), (len(x1), 1 + len(crossings)))
    return np.concatenate([fixed, x1 - width, x1 + width, x1], axis=1)


def _arm_means(
    vector: Vector, arm: str, rates: Mapping[str, float], kstar_ln: float
) -> tuple[float, float]:
    """x = ln(K/K*) of the arm's two runs (at η_l and η_u) without jitter."""
    return (vector.ln_k(arm, rates["l"]) - kstar_ln, vector.ln_k(arm, rates["u"]) - kstar_ln)


def support_breakpoints(
    vector: Vector,
    arm: str,
    rates: Mapping[str, Mapping[str, float]],
    scenario: Scenario,
    e_other: float | None,
    *,
    slopes: Mapping[str, tuple[float, float]],
    candidate: int | None = None,
) -> list[float]:
    """The standardized jitters (e/σ_e,arm) where an arm's support or a gross collapse changes
    (module docstring): with ``e_other`` None, those of the run at η_l (the outer integral's
    breakpoints); with ``e_other`` that run's standardized jitter, those of the run at η_u for
    it. ``slopes``: the utility slopes (a collapse may lie on an extension); ``candidate``: the
    candidate whose K* applies (the scenario's by default). The vector's utility is as given
    (shift it first: ``shifted_vectors``). Empty without jitter."""
    sigma_e = JITTER_SD[arm] if scenario.jitter else 0.0
    if sigma_e == 0:
        return []
    kstar_ln = math.log(CANDIDATES[candidate or scenario.candidate]["targets"]["star"])
    m1, m2 = _arm_means(vector, arm, rates[arm], kstar_ln)
    crossings = _crossings(vector, arm, slopes, kstar_ln)
    if e_other is None:
        return sorted((x - m1) / sigma_e for x in _x_breaks_outer(crossings))
    x1 = m1 + sigma_e * e_other
    return sorted(float((x - m2) / sigma_e) for x in _x_breaks_inner(np.array([x1]), crossings)[0])


def _arm_moments(
    vector: Vector,
    arm: str,
    rates: Mapping[str, float],
    scenario: Scenario,
    calibration: Calibration,
    slopes: Mapping[str, tuple[float, float]],
    kstar_ln: float,
    *,
    tol: float,
    max_nodes: int,
) -> tuple[float, float]:
    """(E[1_a], E[1_a·I_a]) of one arm in one vector over its runs' jitters, before the
    failures: in closed form without jitter, else through the arm's jitter quadrature
    (``_JitterGrid``, remembered: it depends on neither σ nor the offset), its outer rule
    checked for convergence at this σ and offset."""
    sigma_e = JITTER_SD[arm] if scenario.jitter else 0.0
    if sigma_e == 0:
        m = _arm_means(vector, arm, rates, kstar_ln)
        terms = _arm_terms(vector, arm, m, sigma_e, kstar_ln, slopes, scenario, calibration)
        s, t = terms(np.array(0.0), np.array(0.0))
        return float(s), float(t)
    grid = _grid(vector, arm, rates, slopes, kstar_ln, tol=tol, max_nodes=max_nodes)
    return grid.moments(_share(arm, scenario, calibration))


def _share(arm: str, scenario: Scenario, calibration: Calibration) -> Callable:
    """x → p_a(E[h](x)), the arm's expected primary judged share at x = ln(K/K*) under the
    calibration's block effect (σ and offset)."""
    effect_sd, offset = calibration.sigma(arm), calibration.offset(arm)

    def share(x: np.ndarray) -> np.ndarray:
        mean = harm_curve(arm, x, scenario.curvature_scale) + offset
        return judged_share(arm, expected_clipped(mean, effect_sd, scenario.shape), scenario)

    return share


class _JitterGrid:
    """The jitter quadrature of one arm in one vector at one pair of rates (M8), which depends
    on neither the block effects' σ nor their offset (only the shares do).

    With x₁, x₂ the two runs' x = ln(K/K*), 1_a their support and w the weight,
    I_a = c₁·p(x₁) + c₂·p(x₂), c₁ = 1 − w where run 1 is the lower and w where it is the upper
    (c₂ = 1 − c₁). So E[1_a·I_a] = E₁[p(x₁)·A₁(x₁)] + E₂[p(x₂)·A₂(x₂)] and E[1_a] = E₁[S₁(x₁)],
    with A_i(x_i) = E[1_a·c_i | x_i] and S₁(x₁) = E[1_a | x₁] integrals over the other run's
    jitter alone. Each part (run i outer) is the nested quadrature of the module docstring: the
    outer rule on run i's CDF scale between its breakpoints (the bracket, the width, a
    collapse and the collapse ± the width) and ``SPLITS``, the inner integrals over the other
    run between theirs (the bracket, the width, a tie, a collapse), each inner integral to
    convergence (n and 2n nodes within the tolerance on every piece). The inner integrals at
    each outer level are computed once and kept; the outer rule, with p, is checked for
    convergence at every σ and offset it is evaluated at (n and 2n nodes on every piece, from 16
    nodes, doubling up to the maximum; else ``TruthNotConverged``), refining with new inner
    integrals where an offset needs it."""

    def __init__(self, vector: Vector, arm: str, rates: Mapping[str, float],
                 slopes: Mapping[str, tuple[float, float]], kstar_ln: float, *, tol: float,
                 max_nodes: int) -> None:  # fmt: skip
        self.vector, self.arm, self.kstar_ln = vector, arm, kstar_ln
        self.slopes = {arm: tuple(slopes[arm])}
        self.tol, self.max_nodes = tol, max_nodes
        self.sigma_e = JITTER_SD[arm]
        self.means = _arm_means(vector, arm, rates, kstar_ln)
        self.crossings = _crossings(vector, arm, slopes, kstar_ln)
        self.parts = []
        for own in (0, 1):  # the outer run: run 1 (at η_l), then run 2 (at η_u)
            breaks = np.array([[(x - self.means[own]) / self.sigma_e
                                for x in _x_breaks_outer(self.crossings)]])  # fmt: skip
            splits = np.broadcast_to(np.array(SPLITS), (1, len(SPLITS)))
            u = np.sort(ndtr(np.concatenate([breaks, splits], axis=1)), axis=1)
            edges = np.concatenate([np.zeros((1, 1)), u, np.ones((1, 1))], axis=1)
            half = (edges[:, 1:] - edges[:, :-1]) / 2
            mid = (edges[:, 1:] + edges[:, :-1]) / 2
            kept = half[0] > 0  # a piece of length 0 adds exactly nothing
            self.parts.append({"half": half[0, kept], "mid": mid[0, kept], "levels": {}})

    def _usable(self, x: np.ndarray) -> np.ndarray:
        return self.vector.losses_at(self.arm, self.kstar_ln + x, self.slopes)[0] <= COLLAPSE_AT_PP

    def _level(self, own: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Run ``own``'s outer x at the n-node rule of each piece, (P, n), and the inner
        integrals there, (2, P, n): E[1_a | x] and E[1_a·c | x] (c: the weight on run
        ``own``'s share)."""
        part = self.parts[own]
        found = part["levels"].get(n)
        if found is not None:
            return found
        t, _ = _legendre(n)
        half, mid = part["half"][:, None], part["mid"][:, None]
        z = ndtri(np.clip(mid + half * t, _U_LOW, _U_HIGH))  # (P, n)
        x_own = self.means[own] + self.sigma_e * z
        flat = x_own.reshape(-1)
        other = self.means[1 - own]
        inner_breaks = (_x_breaks_inner(flat, self.crossings) - other) / self.sigma_e
        usable_own = self._usable(flat)[:, None, None]

        def inner(z2: np.ndarray) -> np.ndarray:
            x_other = other + self.sigma_e * np.asarray(z2)
            mine = flat[:, None, None]
            x1, x2 = (mine, x_other) if own == 0 else (x_other, mine)
            u1, u2 = (usable_own, self._usable(x_other)) if own == 0 else (
                self._usable(x_other), usable_own)  # fmt: skip
            x1, x2, u1, u2 = np.broadcast_arrays(x1, x2, u1, u2)
            supported, _, _, w = _support(x1, x2, u1, u2)
            lower = (x1 < x2) if own == 0 else (x2 < x1)  # run ``own`` is the lower
            weight = np.where(lower, 1 - w, w)
            return np.stack([supported.astype(float), np.where(supported, weight, 0.0)])

        values = _piecewise(inner, inner_breaks, tol=self.tol, max_nodes=self.max_nodes)
        found = part["levels"][n] = (x_own, values.reshape(2, *x_own.shape))
        return found

    def _rule(self, own: int, n: int, share: Callable) -> np.ndarray:
        """(E[S], E[p·A]) of run ``own``'s part per piece at the n-node outer rule, (2, P)."""
        x, inner = self._level(own, n)
        _, w = _legendre(n)
        f = np.stack([inner[0], share(x) * inner[1]])
        return (f * w).sum(axis=-1) * self.parts[own]["half"]

    def moments(self, share: Callable) -> tuple[float, float]:
        """(E[1_a], E[1_a·I_a]) at the shares ``share`` (x → p(x)): each part's outer rule from
        16 nodes, doubling until n and 2n agree within the tolerance on every piece."""
        out = []
        for own in (0, 1):
            n = FIRST_NODES
            if 2 * n > self.max_nodes:
                raise TruthNotConverged(f"{self.max_nodes} nodes cannot check {n} against {2 * n}")
            coarse = self._rule(own, n, share)
            while True:
                fine = self._rule(own, 2 * n, share)
                if np.all(np.abs(fine - coarse) <= self.tol):
                    break
                n *= 2
                if 2 * n > self.max_nodes:
                    worst = float(np.max(np.abs(fine - coarse)))
                    raise TruthNotConverged(
                        f"{n} and {n // 2} nodes differ by {worst:.3g} (tolerance "
                        f"{self.tol:g}, at most {self.max_nodes} nodes)"
                    )
                coarse = fine
            out.append(fine.sum(axis=-1))
        return float(out[0][0]), float(out[0][1] + out[1][1])


# The jitter quadratures by what they depend on: the vector's points of the arm (as shifted),
# the arm, the rates, K*, the slopes and the quadrature's settings
_GRIDS: dict[tuple, _JitterGrid] = {}
# A grid takes about 40 kB. One-off truths (a worker's studies) keep a few thousand; a jitter
# scenario's calibration keeps every grid its σ and offset steps need (its three arms' pairs of
# rates × the vectors, some 40 000 at 10 000 selections), and forgets them when it ends
GRIDS_KEPT, CALIBRATION_GRIDS = 4_000, 200_000
_GRIDS_LIMIT = GRIDS_KEPT


def _grid_key(vector: Vector, arm: str, rates: Mapping[str, float],
              slopes: Mapping[str, tuple[float, float]], kstar_ln: float, tol: float,
              max_nodes: int) -> tuple:  # fmt: skip
    return (vector.seed, vector.arms[arm], arm, float(rates["l"]), float(rates["u"]), kstar_ln,
            tuple(slopes[arm]), tol, max_nodes)  # fmt: skip


def _grid_task(args: tuple) -> _JitterGrid:
    """A worker's task: one arm's jitter quadrature, its outer levels settled at the shares of
    ``scenario`` and ``calibration``."""
    vector, arm, rates, slopes, kstar_ln, scenario, calibration = args
    grid = _JitterGrid(vector, arm, rates, slopes, kstar_ln, tol=TOLERANCE, max_nodes=MAX_NODES)
    grid.moments(_share(arm, scenario, calibration))
    return grid


def _build_grids(mapper: Callable, scenario: Scenario, calibration: Calibration,
                 slopes: Mapping[str, tuple[float, float]], kstar_ln: float,
                 wanted: Sequence[tuple[Vector, str, Mapping[str, float]]]) -> None:  # fmt: skip
    """The jitter quadratures of ``wanted`` (vector, arm, rates) not yet kept, built through
    ``mapper`` (each one a pure function of its inputs) and kept."""
    tasks, keys = [], []
    for vector, arm, rates in wanted:
        key = _grid_key(vector, arm, rates, slopes, kstar_ln, TOLERANCE, MAX_NODES)
        if key not in _GRIDS and key not in keys:
            keys.append(key)
            tasks.append((vector, arm, dict(rates), {arm: tuple(slopes[arm])}, kstar_ln,
                          scenario, calibration))  # fmt: skip
    if not tasks:
        return
    if len(_GRIDS) + len(tasks) > _GRIDS_LIMIT:
        _GRIDS.clear()
    for key, grid in zip(keys, mapper(_grid_task, tasks), strict=True):
        _GRIDS[key] = grid


def _grid(vector: Vector, arm: str, rates: Mapping[str, float],
          slopes: Mapping[str, tuple[float, float]], kstar_ln: float, *, tol: float,
          max_nodes: int) -> _JitterGrid:  # fmt: skip
    """The arm's jitter quadrature, remembered (``_JitterGrid``)."""
    key = _grid_key(vector, arm, rates, slopes, kstar_ln, tol, max_nodes)
    found = _GRIDS.get(key)
    if found is None:
        if len(_GRIDS) >= _GRIDS_LIMIT:
            _GRIDS.clear()
        found = _GRIDS[key] = _JitterGrid(vector, arm, rates, slopes, kstar_ln, tol=tol,
                                          max_nodes=max_nodes)  # fmt: skip
    return found


# (E[1_a], E[1_a·I_a]) per (vector, arm, rates, harm model, the arm's σ and offset, K*,
# slopes, quadrature): the selected rates are rounded to 3 significant digits, so studies share
# them, and the truth under jitter is costly
_MOMENTS: dict[tuple, tuple[float, float]] = {}
_MOMENTS_LIMIT = 2_000_000


def _cached_moments(
    vector: Vector,
    arm: str,
    rates: Mapping[str, float],
    scenario: Scenario,
    calibration: Calibration,
    slopes: Mapping[str, tuple[float, float]],
    kstar_ln: float,
    *,
    tol: float,
    max_nodes: int,
) -> tuple[float, float]:
    """``_arm_moments``, remembered by everything it depends on (the vector's points of the arm,
    as shifted, included)."""
    key = _moment_key(vector, arm, rates, scenario, calibration, slopes, kstar_ln, tol, max_nodes)
    found = _MOMENTS.get(key)
    if found is None:
        if len(_MOMENTS) >= _MOMENTS_LIMIT:
            _MOMENTS.clear()
        found = _arm_moments(
            vector, arm, rates, scenario, calibration, slopes, kstar_ln, tol=tol,
            max_nodes=max_nodes,
        )  # fmt: skip
        _MOMENTS[key] = found
    return found


def _moment_key(
    vector: Vector,
    arm: str,
    rates: Mapping[str, float],
    scenario: Scenario,
    calibration: Calibration,
    slopes: Mapping[str, tuple[float, float]],
    kstar_ln: float,
    tol: float,
    max_nodes: int,
) -> tuple:
    """What ``_arm_moments`` depends on: the vector's seed and points of the arm (as shifted),
    the arm, the rates, the jitter, the harm and label model, the arm's σ and offset, K*, the
    slopes and the quadrature's settings."""
    return (
        vector.seed,
        vector.arms[arm],
        arm,
        float(rates["l"]),
        float(rates["u"]),
        scenario.jitter,
        scenario.curvature_scale,
        scenario.shape,
        scenario.judge,
        scenario.missing,
        scenario.missing_rate,
        calibration.sigma(arm),
        calibration.offset(arm),
        kstar_ln,
        tuple(slopes[arm]),
        tol,
        max_nodes,
    )


def truth(
    scenario: Scenario,
    calibration: Calibration,
    vectors: Sequence[Vector],
    slopes: Mapping[str, tuple[float, float]],
    rates: Mapping[str, Mapping[str, float]],
    *,
    tol: float = TOLERANCE,
    max_nodes: int = MAX_NODES,
) -> dict[str, float]:
    """The registered estimand of each comparison at the final ``rates`` (arm → ``l``, ``u``),
    in proportions (module docstring): Σ_v E[1_a·1_F·(I_a − I_F)] / Σ_v E[1_a·1_F] over the
    inputs' ``vectors`` (as read: the calibration's utility shifts are applied here), at the
    calibration's candidate's K*. A comparison is given when ``rates`` holds both its arms; NaN
    when no vector can support it at these rates (no fresh block is supported: undefined).
    ``TruthNotConverged`` under jitter when the quadrature does not converge."""
    shifted = _shifted(vectors, calibration)
    kstar_ln = math.log(CANDIDATES[calibration.candidate]["targets"]["star"])
    arms = [arm for arm in ARM_ORDER if arm in rates]
    moments = {
        arm: [
            _cached_moments(
                v,
                arm,
                rates[arm],
                scenario,
                calibration,
                slopes,
                kstar_ln,
                tol=tol,
                max_nodes=max_nodes,
            )
            for v in shifted
        ]
        for arm in arms
    }
    return _combine(moments, calibration.failure)


def _combine(moments: Mapping[str, Sequence[tuple[float, float]]], failure: float) -> dict:
    """Σ_v E[1_a·1_F·(I_a − I_F)] / Σ_v E[1_a·1_F] of each comparison whose arms ``moments``
    holds (per arm, per vector, (E[1_a], E[1_a·I_a]) before the failures), each arm's support
    times (1 − f)²; NaN when no vector supports it."""
    survive = (1 - failure) ** 2
    out = {}
    for name, (a, b) in COMPARISONS.items():
        if a not in moments or b not in moments:
            continue
        num = den = 0.0
        for (s_a, t_a), (s_b, t_b) in zip(moments[a], moments[b], strict=True):
            num += survive**2 * (t_a * s_b - s_a * t_b)
            den += survive**2 * s_a * s_b
        out[name] = num / den if den > 0 else math.nan
    return out


_SHIFTED: dict[tuple, list[Vector]] = {}


def _shifted(vectors: Sequence[Vector], calibration: Calibration) -> list[Vector]:
    """``shifted_vectors``, remembered by the vectors' points and the shifts."""
    key = (
        tuple((v.seed, tuple(v.arms.get(a, ()) for a in ARM_ORDER)) for v in vectors),
        tuple(sorted((arm, tuple(sorted(v.items()))) for arm, v in calibration.mmlu_shift.items())),
        tuple(sorted(calibration.factual_shift.items())),
    )
    found = _SHIFTED.get(key)
    if found is None:
        if len(_SHIFTED) >= 64:
            _SHIFTED.clear()
        found = _SHIFTED[key] = shifted_vectors(vectors, calibration)
    return found


# ------------------------------------------------------------------ the procedure

# The sizes (M, the screen's and the verification's resamples, the calibration's selections) are
# study3_decisions' four adjacent constants, which the dry run's Part II changes alone (M25)
CHUNK = 500  # bootstrap_t's chunk: the result does not depend on it, the memory does
MODES = ("screen", "procedure")
REACH = 99  # the largest N point: each comparison is followed to 99 supported blocks
GATE_OF = {"A-F": "AF", "P-F": "PF"}  # a comparison's name in study3_blocks
UNCONFIRMED = ("af_gate", "cap_beyond_seeds", "scientific_report")  # why a study is not confirmed
NO_ATTAINABLE_N = "no_attainable_n"  # the n_min scan stopped at a cap beyond the reserved seeds
RUNS = tuple((arm, label) for arm in ARM_ORDER for label in LABELS)  # a block's six runs
RUN_ARMS = tuple(arm for arm, _ in RUNS)
FAILED_STOP = INSTABILITY[0]  # the stop a simulated scientific failure records
CALIBRATION_BLOCKS = 28  # the calibration's validations have 28 blocks
CALIBRATION_ROUNDS = 10  # f and the safe shifts alternate at most this often
SUPPORT_TOLERANCE = 1e-4  # |realized support − s| at the calibrated f
SAFE_MMLU_MEAN, SAFE_MMLU_SD, SAFE_FACTUAL_MEAN = 3.0, 2.0, 2.0  # M10, pp
SIGMA_PRECISION = 1e-14  # Brent's absolute tolerance on σ (proportions)
SIGMA_RTOL = 4 * float(np.finfo(float).eps)  # and its relative one, the smallest brentq takes
SIGMA_STEP = 1e-3  # a bracket from the previous σ starts this far from it, relative, then ×8
OFFSET_PRECISION = 1e-12  # |mean truth − Δ| at the solved offset (proportions)


class CalibrationNotStable(RuntimeError):
    """f and the safe shifts still moved after ``CALIBRATION_ROUNDS`` rounds."""


class SimulationError(RuntimeError):
    """A registered function did not take a simulated record as the procedure would: a fault of
    the simulation, never a result."""


@dataclass(frozen=True)
class Setup:
    """What one simulated study needs besides its scenario and stream: the inputs' ``vectors``
    and utility ``slopes``, the ``calibrations`` keyed by (scenario, candidate), ``M``, the
    ``levels``, the ``validation_blocks`` (28 or 40, from G1) and ``n_min`` (None unless the
    screen found the power short)."""

    vectors: tuple[Vector, ...]
    slopes: Mapping[str, tuple[float, float]]
    calibrations: Mapping[tuple[Scenario, int], Calibration]
    M: int
    levels: tuple[float, ...]
    validation_blocks: int
    n_min: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vectors", tuple(self.vectors))
        object.__setattr__(self, "levels", tuple(self.levels))
        if not self.vectors or not all(isinstance(v, Vector) for v in self.vectors):
            raise ValueError("a setup needs the inputs' whole seed vectors")
        if set(self.slopes) != set(ARM_ORDER):
            raise ValueError(f"utility slopes of every arm {ARM_ORDER} are needed")
        if not (_integer(self.M) and self.M >= 1):
            raise ValueError(f"M {self.M!r} is not a positive integer")
        if not self.levels or not set(self.levels) <= set(LEVELS):
            raise ValueError(f"levels {self.levels!r} are not among {LEVELS}")
        if self.validation_blocks not in VALIDATION_BLOCKS or not _integer(self.validation_blocks):
            raise ValueError(f"the validation has {VALIDATION_BLOCKS} blocks")
        if self.n_min is not None and not (_integer(self.n_min) and self.n_min >= MIN_N):
            raise ValueError(f"n_min {self.n_min!r} is not an N of at least {MIN_N}")
        # the vectors' points and the slopes, as one short key of the caches
        blob = repr((tuple((v.seed, tuple(v.arms[a] for a in ARM_ORDER)) for v in self.vectors),
                     tuple(sorted(self.slopes.items()))))  # fmt: skip
        object.__setattr__(self, "fingerprint", hashlib.sha256(blob.encode()).hexdigest())

    def calibration(self, scenario: Scenario, candidate: int) -> Calibration:
        try:
            return self.calibrations[(scenario, candidate)]
        except KeyError as exc:
            raise ValueError(
                f"no calibration of scenario {scenario.name} at candidate {candidate}"
            ) from exc


def _integer(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _with(setup: Setup, scenario: Scenario, candidate: int, calibration: Calibration) -> Setup:
    """``setup`` with ``calibration`` for (scenario, candidate)."""
    return replace(setup, calibrations={**setup.calibrations, (scenario, candidate): calibration})


@dataclass
class _Block:
    """One simulated block (in memory): its seed, the vector it drew (an index into the inputs),
    the six runs' x = ln(K/K*) (jitter included) and failure uniforms, the standardized block
    effects (e_AF, e_PF), its endpoints and ``block_state``, and the six runs' primary judged
    shares and their ``shares_high`` (every missing label counted as harmful: (harmful +
    missing)/521), None until labeled."""

    seed: int
    vector: int
    x: np.ndarray
    uniforms: np.ndarray
    effects: np.ndarray
    endpoints: list[Endpoint]
    state: dict[str, Any]
    shares: np.ndarray | None = None
    shares_high: np.ndarray | None = None


@dataclass(frozen=True)
class StudyOutcome:
    """One simulated study (§"Simulation before validation"): its ``index`` and ``mode``, the
    final ``candidate`` and the ``active`` comparisons; the ``levels`` it ran; ``rates`` per
    candidate that ran (arm → ``l``/``u``); the final candidate's validation ``gates`` (``AF``,
    ``PF``) and ``m``, the A–F-supported validation blocks; per level ``confirmed`` and ``why``
    (``UNCONFIRMED``),
    ``N``, ``cap`` and ``blocks`` (started at the stop); ``covered`` per (N point, level,
    comparison) (None: P–F not active in ``"procedure"`` mode); per (level, comparison) the
    cap's ``direction`` (a power scenario's registered final direction, ``final_direction``;
    None elsewhere, or with no common direction), the confirmation's ``readings`` and its
    ``utility_claim``; the
    ``truth`` per comparison (proportions; None when undefined); a power study of the screen's
    ``report`` (``power_report``: per level slot its ``N``, ``confirmed``, and A–F's ``readings``
    and ``direction`` at the denser mask; None elsewhere). In memory only, never
    serialized and not compared: ``spread_pp``, ``utility`` (per arm, (vector seed, interpolated
    MMLU loss, interpolated factual loss) of the supported validation blocks), ``point_seeds``,
    ``cohorts`` and ``analysis`` (the seeds behind each test) and ``validation`` (the final
    candidate's validation blocks)."""

    index: int
    mode: str
    candidate: int
    active: list[str]
    levels: tuple[float, ...]
    rates: Mapping[int, Mapping[str, Mapping[str, float]]]
    gates: Mapping[str, bool]
    m: int
    confirmed: Mapping[float, bool]
    why: Mapping[float, str | None]
    N: Mapping[float, int | None]
    cap: Mapping[float, int | None]
    blocks: Mapping[float, int | None]
    covered: Mapping[tuple[int | str, float, str], bool | None]
    direction: Mapping[tuple[float, str], str | None]
    readings: Mapping[tuple[float, str], str | None]
    utility_claim: Mapping[tuple[float, str], bool | None]
    truth: Mapping[str, float | None]
    report: Mapping[str, Mapping[str, Any]] | None = None
    spread_pp: float | None = field(default=None, compare=False, repr=False)
    utility: Mapping[str, tuple[tuple[int, float, float], ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    point_seeds: Mapping[tuple[int, str], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    cohorts: Mapping[tuple[float, str], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    analysis: Mapping[tuple[float, str], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    validation: tuple[_Block, ...] = field(default=(), compare=False, repr=False)

    def rates_at_candidate(self, candidate: int) -> dict[str, dict[str, float]]:
        """The final rates the study selected at ``candidate`` (empty when it did not run it)."""
        return {arm: dict(r) for arm, r in self.rates.get(candidate, {}).items()}


def synthetic_endpoint(
    candidate: int,
    step: str,
    round_: int | None,
    arm: str,
    rate: float,
    seed: int,
    *,
    K: float | None,
    mmlu: float | None,
    factual: float | None,
    state: str = "usable",
    k_only: bool,
) -> Endpoint:
    """A simulated run as ``study3_records.read_endpoints`` would give it: a ``Cell`` of the
    campaign ``sim-c<N>-<step>[-r<k>]-s<seed>`` that prescribed (arm, rate, seed), run id
    ``sim-…``, one attempt, in state ``usable``, ``unusable`` (a gross collapse) or
    ``scientific`` (an instability stop: no K, no utility)."""
    if step not in STEP_ROLES:
        raise ValueError(f"no step {step!r}")
    if state not in ("usable", "unusable", "scientific"):
        raise ValueError(f"a simulated run is usable, unusable or scientific, not {state!r}")
    tag = f"c{candidate}-{step}" + ("" if round_ is None else f"-r{round_}")
    run_id = f"sim-{tag}-{arm}-{rate:.6g}-s{seed}"
    label = f"{arm}-{rate:.6g}/s{seed}"
    cell = Cell(
        campaign=f"sim-{tag}-s{seed}",
        label=label,
        candidate=candidate,
        step=step,
        round=round_,
        arm=arm,
        rate=rate,
        seed=seed,
        config_hash="simulation",
        prescribed=(arm, rate, seed),
    )
    failed = state == "scientific"
    return Endpoint(
        run_id=run_id,
        run_dir=Path(run_id),
        cell=cell,
        arm=arm,
        lr=rate,
        seed=seed,
        role=STEP_ROLES[step],
        candidate=candidate,
        k_only=k_only,
        attempt=1,
        label=label,
        state=state,
        steps=None if failed else UPDATES,
        stop=FAILED_STOP if failed else "steps",
        K=None if failed else K,
        mmlu_loss_pp=None if failed else mmlu,
        factual_loss_pp=None if failed else factual,
        unusable_reason=GROSS_COLLAPSE if state == "unusable" else None,
    )


def _base(base: int | np.random.SeedSequence) -> int:
    """The integer every key of a study's draws starts from: ``base`` itself, or 128 bits a
    calibration's stream generates."""
    if isinstance(base, np.random.SeedSequence):
        words = base.generate_state(4, np.uint32).astype("<u4")
        return int.from_bytes(words.tobytes(), "little")
    if not (_integer(base) and base >= 0):
        raise ValueError(f"a stream's base is a non-negative integer, not {base!r}")
    return int(base)


def _rng(base: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(stream(base, *keys))


def _jitter(scenario: Scenario, arm: str) -> float:
    return JITTER_SD[arm] if scenario.jitter else 0.0


def calibration_stream(base: int, scenario: Scenario, candidate: int) -> np.random.SeedSequence:
    """The calibration's stream of (scenario, candidate): (calibration base, "support",
    scenario, candidate)."""
    return stream(base, "support", scenario.name, candidate)


# -- the search and the repair


_SELECTIONS: dict[tuple, dict[str, dict[str, float]]] = {}
_SELECTIONS_LIMIT = 50_000


def _select(
    scenario: Scenario, setup: Setup, base: int, index: int, candidate: int, prefix: tuple
) -> dict[str, dict[str, float]]:
    """``_selected``, remembered by everything it depends on (the scenario's name and jitter,
    the stream, the vectors' K): the calibration's passes and modes share a study's
    selection."""
    key = (scenario.name, scenario.jitter, base, index, candidate, prefix, setup.fingerprint)
    found = _SELECTIONS.get(key)
    if found is None:
        if len(_SELECTIONS) >= _SELECTIONS_LIMIT:
            _SELECTIONS.clear()
        found = _SELECTIONS[key] = _selected(scenario, setup, base, index, candidate, prefix)
    return {arm: dict(r) for arm, r in found.items()}


def _selected(
    scenario: Scenario, setup: Setup, base: int, index: int, candidate: int, prefix: tuple
) -> dict[str, dict[str, float]]:
    """The final rates of a candidate (§"Candidates and rate search"): round 0 at the starting
    rates on the four search seeds' vectors, round 1 at the adjusted rates (``search_state``),
    the numerical qualification passed (M13), the repair (``repair_state``) and its runs. K-only
    runs never fail (M13); each run's K carries the scenario's jitter."""
    vectors = setup.vectors
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "search")
    seeds = list(SEEDS["search"][candidate])
    picks = rng.integers(0, len(vectors), len(seeds))
    z = rng.standard_normal((2, len(seeds), len(ARM_ORDER), len(LABELS)))
    start = CANDIDATES[candidate]["start"]

    def k_only(step: str, round_: int | None, vector: Vector, seed: int, arm: str, rate: float,
               jitter: float) -> Endpoint:  # fmt: skip
        k = math.exp(vector.ln_k(arm, rate) + _jitter(scenario, arm) * jitter)
        return synthetic_endpoint(candidate, step, round_, arm, rate, seed, K=k, mmlu=None,
                                  factual=None, k_only=True)  # fmt: skip

    endpoints = [
        k_only("search", 0, vectors[picks[i]], seed, arm, start[arm][label], z[0, i, a, j])
        for i, seed in enumerate(seeds)
        for a, arm in enumerate(ARM_ORDER)
        for j, label in enumerate(LABELS)
    ]
    state = search_state(endpoints, candidate, problems=())
    if state["next"] == "round1":
        for a, arm in enumerate(ARM_ORDER):
            for j, label in enumerate(LABELS):
                decision = state["arms"][arm]["targets"][label]["decision"]
                if decision in (None, "kept"):
                    continue
                endpoints += [
                    k_only("search", 1, vectors[picks[i]], seed, arm, decision, z[1, i, a, j])
                    for i, seed in enumerate(seeds)
                ]
        state = search_state(endpoints, candidate, problems=())
    if state["next"] != "numerical":
        raise SimulationError(f"the simulated search did not select rates ({state['next']}: "
                              f"{state['problems'] or state['failures']})")  # fmt: skip
    selected = state["selected"]
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "numerical")
    seeds = list(SEEDS["numerical"][candidate])
    picks = rng.integers(0, len(vectors), len(seeds))
    z = rng.standard_normal((2, len(seeds), len(ARM_ORDER), len(LABELS)))
    endpoints += [
        k_only("numerical", None, vectors[picks[i]], seed, arm, selected[arm][label],
               z[0, i, a, j])
        for i, seed in enumerate(seeds)
        for a, arm in enumerate(ARM_ORDER)
        for j, label in enumerate(LABELS)
    ]  # fmt: skip
    qualified = {arm: {selected[arm][label]: QUALIFIED for label in LABELS} for arm in ARM_ORDER}
    repair = repair_state(endpoints, candidate, selected, qualified, problems=())
    if repair["next"] == "repair":
        for arm, repairs in repair["repairs"].items():
            a = ARM_ORDER.index(arm)
            for label, rate in repairs.items():
                j = LABELS.index(label)
                endpoints += [
                    k_only(REPAIR_STEP, None, vectors[picks[i]], seed, arm, rate, z[1, i, a, j])
                    for i, seed in enumerate(seeds)
                ]
        repair = repair_state(endpoints, candidate, selected, qualified, problems=())
    if repair["next"] not in ("repaired", "none"):
        raise SimulationError(f"the simulated repair did not finish ({repair['next']}: "
                              f"{repair['problems'] or repair['undecided']})")  # fmt: skip
    return {
        arm: {
            label: float(repair["repairs"].get(arm, {}).get(label, selected[arm][label]))
            for label in LABELS
        }
        for arm in ARM_ORDER
    }


# -- blocks


def _draw_block(
    scenario: Scenario,
    setup: Setup,
    calibration: Calibration,
    shifted: Sequence[Vector],
    candidate: int,
    rates: Mapping[str, Mapping[str, float]],
    step: str,
    seed: int,
    rng: np.random.Generator,
    arms: Sequence[str],
) -> _Block:
    """One block: a whole vector drawn with replacement; each of the six runs its ln K at its
    rate (plus the scenario's jitter, M8), its failure uniform (a scientific failure below f,
    M7) and its utility at that K (the safe shifts applied, M10; a loss above 15 pp is a gross
    collapse); the block effects. Every run is drawn whichever arms are ``arms``, so that P's
    activity moves no draw; the endpoints are those of ``arms``, and ``block_state`` reads them."""
    vector = int(rng.integers(0, len(setup.vectors)))
    z = rng.standard_normal((len(ARM_ORDER), len(LABELS)))
    uniforms = rng.random((len(ARM_ORDER), len(LABELS)))
    effects = draw_effects(scenario.shape, rng, 2) if scenario.shape else np.zeros(2)
    kstar_ln = math.log(CANDIDATES[candidate]["targets"]["star"])
    own = shifted[vector]
    x = np.empty((len(ARM_ORDER), len(LABELS)))
    endpoints = []
    for a, arm in enumerate(ARM_ORDER):
        for j, label in enumerate(LABELS):
            rate = rates[arm][label]
            ln_k = own.ln_k(arm, rate) + _jitter(scenario, arm) * z[a, j]
            x[a, j] = ln_k - kstar_ln
            if arm not in arms:
                continue
            if uniforms[a, j] < calibration.failure:
                endpoints.append(synthetic_endpoint(
                    candidate, step, None, arm, rate, seed, K=None, mmlu=None, factual=None,
                    state="scientific", k_only=False))  # fmt: skip
                continue
            mmlu, factual = own.losses(arm, ln_k, setup.slopes)
            state = "unusable" if mmlu > COLLAPSE_AT_PP else "usable"
            endpoints.append(synthetic_endpoint(
                candidate, step, None, arm, rate, seed, K=math.exp(ln_k), mmlu=mmlu,
                factual=factual, state=state, k_only=False))  # fmt: skip
    state = block_state(endpoints, seed, candidate, arms)
    return _Block(seed, vector, x, uniforms, effects, endpoints, state)


def _label(scenario: Scenario, calibration: Calibration, block: _Block,
           rng: np.random.Generator) -> None:  # fmt: skip
    """The six runs' primary judged shares and their ``shares_high`` (M5, M6, M6b):
    h = clip(H_a(x)/100 + δ_a, 0, 1) with δ_F = 0, δ_A = σ_AF·e_AF + o_A, δ_P = σ_PF·e_PF + o_P;
    P's runs are labeled whatever its activity, so that it moves no draw."""
    delta = {
        "A": calibration.sigma_af * block.effects[0] + calibration.offset_a,
        "F": 0.0,
        "P": calibration.sigma_pf * block.effects[1] + calibration.offset_p,
    }
    h = np.stack([
        np.clip(harm_curve(arm, block.x[a], scenario.curvature_scale) + delta[arm], 0.0, 1.0)
        for a, arm in enumerate(ARM_ORDER)
    ])  # fmt: skip
    harmful, missing = endpoint_counts(h.ravel(), RUN_ARMS, scenario, rng)
    block.shares = (harmful / PROMPTS).reshape(h.shape)
    block.shares_high = ((harmful + missing) / PROMPTS).reshape(h.shape)


def _at_kstar(block: _Block, arm: str, shares: np.ndarray) -> float:
    """interpolated(share_lo, share_hi, w) of an arm, its endpoints and weight ``block_state``'s."""
    a = ARM_ORDER.index(arm)
    lo = 0 if block.x[a, 0] < block.x[a, 1] else 1
    weight = block.state["arms"][arm]["weight"]
    return interpolated(shares[a, lo], shares[a, 1 - lo], weight)


def _difference(block: _Block, comparison: str) -> float:
    """d of a block supported for the comparison (the primary computation): the interpolated
    share of its arm minus F's (``_at_kstar``)."""
    first, second = COMPARISONS[comparison]
    return _at_kstar(block, first, block.shares) - _at_kstar(block, second, block.shares)


def _computations(block: _Block, comparison: str) -> dict[str, float]:
    """The block's difference in each computation of "Judge and audit" (``study3_estimates``'
    ``COMPUTATIONS``, its rule): for the first arm X and F, with h_low the primary share and h_high
    every missing label harmful, primary X_low − F_low, missing_low X_low − F_high, missing_high
    X_high − F_low and all_missing_harmful X_high − F_high."""
    first, second = COMPARISONS[comparison]
    x_low, f_low = (_at_kstar(block, arm, block.shares) for arm in (first, second))
    x_high, f_high = (_at_kstar(block, arm, block.shares_high) for arm in (first, second))
    return {"primary": x_low - f_low, "missing_low": x_low - f_high,
            "missing_high": x_high - f_low, "all_missing_harmful": x_high - f_high}  # fmt: skip


@dataclass
class _Validation:
    candidate: int
    rates: dict[str, dict[str, float]]
    blocks: list[_Block]
    gates: dict[str, bool]
    m: int
    spread_pp: float | None
    labels: np.random.Generator | None


def _validate(
    scenario: Scenario,
    setup: Setup,
    base: int,
    index: int,
    candidate: int,
    prefix: tuple,
    labels: bool,
) -> _Validation:
    """A candidate's search, repair and validation (§"Validation"): ``validation_blocks`` blocks
    of six runs; the A–F and P–F gates (22 of 28 or 31 of 40 supported and ``utility_gate`` at
    α 0.05 per bound); the spread over the A–F-supported blocks (judged primary harm, full
    precision) when ``labels``."""
    calibration = setup.calibration(scenario, candidate)
    rates = _select(scenario, setup, base, index, candidate, prefix)
    shifted = shifted_vectors(setup.vectors, calibration)
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "validation")
    seeds = SEEDS["validation"][candidate][: setup.validation_blocks]
    blocks = [_draw_block(scenario, setup, calibration, shifted, candidate, rates, "validation",
                          seed, rng, ARM_ORDER) for seed in seeds]  # fmt: skip
    states = [b.state for b in blocks]
    gates = {}
    for name in GATE_OF.values():
        count = sum(bool(s[name]["supported"]) for s in states)
        passes = utility_gate(states, name, VALIDATION_ALPHA)["passes"] is True
        gates[name] = count >= SUPPORTED_AT[setup.validation_blocks] and passes
    m = sum(bool(s["AF"]["supported"]) for s in states)
    label_rng, spread = None, None
    if labels and scenario.role != "gate":
        label_rng = _rng(base, "procedure", scenario.name, index, *prefix, "labels")
        for b in blocks:
            _label(scenario, calibration, b, label_rng)
        d = [_difference(b, "A-F") for b in blocks if b.state["AF"]["supported"]]
        spread = 100 * float(np.std(d, ddof=1)) if len(d) >= 2 else None
    return _Validation(candidate, rates, blocks, gates, m, spread, label_rng)


def _supported_utility(validation: _Validation) -> dict[str, tuple[tuple[int, float, float], ...]]:
    """Per arm, (vector seed, interpolated MMLU loss, interpolated factual loss) of the
    validation blocks supported for its comparison (A and F: A–F; P: P–F)."""
    out: dict[str, list[tuple[int, float, float]]] = {arm: [] for arm in ARM_ORDER}
    for b in validation.blocks:
        for arm in ARM_ORDER:
            if b.state["AF" if arm != "P" else "PF"]["supported"]:
                unit = b.state["arms"][arm]
                out[arm].append((b.seed, unit["mmlu_loss_pp"], unit["factual_loss_pp"]))
    return {arm: tuple(v) for arm, v in out.items()}


# -- the study


def simulate_study(
    scenario: Scenario,
    setup: Setup,
    base: int | np.random.SeedSequence,
    index: int,
    *,
    mode: str = "screen",
    resamples: int,
    levels: Sequence[float],
    n_floor: int = MIN_N,
    validation_only: bool = False,
    candidate: int | None = None,
    labels: bool = True,
    power_reading: bool = False,
) -> StudyOutcome:
    """One simulated study, a pure function of (``base``, ``scenario``, ``index``) and the
    setup (module docstring, "The procedure"). ``mode``: ``"screen"`` (the scenario's
    candidate alone, N from the rule, P–F computed as if active: M14, M16) or ``"procedure"``
    (the realized route: candidate 1, then 2, then the scientific feasibility report; N raised
    to ``setup.n_min``; P–F only where the final candidate's gate passed: amendment 7, "P's
    closure"). ``n_floor``: N is at least this (the n_min scan). ``validation_only``: the search, repair
    and validation of the first candidate alone (the gates and the calibration). ``candidate``:
    the first candidate (the scenario's by default). ``labels`` False skips the labels (no
    spread, no confirmation): the calibration's draws, whose label noise enters through its
    exact variance. ``power_reading``: of the confirmation only A–F's cap interval, reading and
    final direction (``power_report``): no interval at the N points, no truth, no utility claim
    (the stop, N and every draw unchanged). ``base``: an integer, or a calibration's stream
    (``calibration_stream``)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    if not (_integer(index) and index >= 0):
        raise ValueError(f"a study's index is a non-negative integer, not {index!r}")
    if not (_integer(n_floor) and n_floor >= MIN_N):
        raise ValueError(f"n_floor {n_floor!r} is not an N of at least {MIN_N}")
    levels = tuple(levels)
    if not levels or not set(levels) <= set(LEVELS):
        raise ValueError(f"levels {levels!r} are not among {LEVELS}")
    root = _base(base)
    first = scenario.candidate if candidate is None else candidate
    route = [(first, ())]
    if mode == "procedure" and first == 1 and not validation_only:
        route.append((2, ("candidate-2",)))
    rates: dict[int, dict[str, dict[str, float]]] = {}
    for c, prefix in route:
        validation = _validate(scenario, setup, root, index, c, prefix, labels)
        rates[c] = validation.rates
        if validation.gates["AF"] or mode == "screen" or validation_only:
            break
    c = validation.candidate
    gates = dict(validation.gates)
    common = {
        "index": int(index),
        "mode": mode,
        "candidate": c,
        "levels": levels,
        "rates": rates,
        "gates": gates,
        "m": validation.m,
        "spread_pp": validation.spread_pp,
        "utility": _supported_utility(validation),
        "validation": tuple(validation.blocks),
    }
    active = ["A-F", "P-F"] if mode == "screen" or gates["PF"] else ["A-F"]
    none = dict.fromkeys(levels)
    if validation_only or not labels or scenario.role == "gate":
        return StudyOutcome(**common, active=active, confirmed=dict.fromkeys(levels, False),
                            why=none, N=none, cap=none, blocks=none, covered={}, direction={},
                            readings={}, utility_claim={}, truth={})  # fmt: skip
    if not validation.gates["AF"]:
        why = "af_gate" if mode == "screen" else "scientific_report"
        return StudyOutcome(**common, active=active, confirmed=dict.fromkeys(levels, False),
                            why=dict.fromkeys(levels, why), N=none, cap=none, blocks=none,
                            covered={}, direction={}, readings={}, utility_claim={},
                            truth={})  # fmt: skip
    return _confirm(scenario, setup, root, index, mode, resamples, levels, n_floor, validation,
                    route[[r[0] for r in route].index(c)][1], active, common,
                    power_reading)  # fmt: skip


def _confirm(
    scenario: Scenario,
    setup: Setup,
    base: int,
    index: int,
    mode: str,
    resamples: int,
    levels: tuple[float, ...],
    n_floor: int,
    validation: _Validation,
    prefix: tuple,
    active: list[str],
    common: dict[str, Any],
    power_reading: bool = False,
) -> StudyOutcome:
    """N per level, the confirmation (``confirmation_state``) and the analysis (module
    docstring, "The procedure", steps 4–6); with ``power_reading``, of the analysis A–F's cap
    interval, reading and final direction alone."""
    c = validation.candidate
    calibration = setup.calibration(scenario, c)
    reserved = SEEDS["confirmation"][c]
    N, cap, why, confirmed = {}, {}, {}, {}
    for level in levels:
        n = max(sample_size(validation.spread_pp, validation.m, level), n_floor)
        if mode == "procedure" and setup.n_min is not None:
            n = max(n, setup.n_min)
        N[level], cap[level] = n, cap_of(n)
        confirmed[level] = cap[level] <= len(reserved)
        why[level] = None if confirmed[level] else "cap_beyond_seeds"
    blocks_n: dict[float, int | None] = dict.fromkeys(levels)
    out = {"covered": {}, "direction": {}, "readings": {}, "utility_claim": {}}
    memory: dict[str, dict] = {"point_seeds": {}, "cohorts": {}, "analysis": {}}
    if not any(confirmed.values()):
        return StudyOutcome(**common, active=active, confirmed=confirmed, why=why, N=N, cap=cap,
                            blocks=blocks_n, truth={}, **out, **memory)  # fmt: skip
    arms = [arm for arm in ARM_ORDER if arm != "P" or "P-F" in active]
    shifted = shifted_vectors(setup.vectors, calibration)
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "confirmation")
    target = max(cap[level] for level in levels if confirmed[level])
    blocks: list[_Block] = []
    supported: dict[str, list[int]] = {comparison: [] for comparison in active}
    for seed in reserved:
        block = _draw_block(scenario, setup, calibration, shifted, c, validation.rates,
                            "confirmation", seed, rng, arms)  # fmt: skip
        _label(scenario, calibration, block, validation.labels)
        blocks.append(block)
        for comparison in active:
            if block.state[GATE_OF[comparison]]["supported"]:
                supported[comparison].append(block.seed)
        if len(blocks) >= target and all(len(s) >= REACH for s in supported.values()):
            break
    by_seed = {b.seed: b for b in blocks}
    d = {comparison: {s: _difference(by_seed[s], comparison) for s in seeds}
         for comparison, seeds in supported.items()}  # fmt: skip
    found = {} if power_reading else truth(
        scenario, calibration, setup.vectors, setup.slopes,
        {arm: validation.rates[arm] for arm in arms})  # fmt: skip
    true = {k: (None if math.isnan(v) else float(v)) for k, v in found.items()}
    covered = out["covered"]
    for comparison in () if power_reading else active:
        seeds = supported[comparison]
        for n in N_POINTS[:-1]:
            reached = len(seeds) >= n
            if reached:
                memory["point_seeds"][(n, comparison)] = tuple(seeds[:n])
            seed = stream(base, "bootstrap", scenario.name, index, comparison, n)
            for level in levels:
                if not confirmed[level]:
                    continue
                interval = (_interval([d[comparison][s] for s in seeds[:n]], level, resamples,
                                      seed) if reached else None)  # fmt: skip
                covered[(n, level, comparison)] = _covers(interval, true.get(comparison))
    for level in levels:
        if not confirmed[level]:
            continue
        stop = _stop(blocks, active, N[level], cap[level])
        state = confirmation_state([e for b in blocks[:stop] for e in b.endpoints], c, N[level],
                                   active)  # fmt: skip
        if not state["stop"]["stopped"] or state["problems"] or len(state["started"]) != stop:
            raise SimulationError(f"the simulated confirmation did not stop as the registered "
                                  f"one ({state['stop']}, {state['problems']})")  # fmt: skip
        blocks_n[level] = len(state["started"])
        for comparison in active:
            seeds = list(state["analysis"][comparison] or ())
            memory["analysis"][(level, comparison)] = tuple(seeds)
            memory["cohorts"][(level, comparison)] = tuple(state["cohorts"][comparison])
            if power_reading and comparison != "A-F":
                continue
            seed = stream(base, "bootstrap", scenario.name, index, comparison, "cap")
            interval = _interval([d[comparison][s] for s in seeds], level, resamples, seed)
            out["direction"][(level, comparison)] = (
                final_direction(interval, [_computations(by_seed[s], comparison) for s in seeds],
                                level, resamples, seed) if scenario.power else None
            )  # fmt: skip
            out["readings"][(level, comparison)] = state["readings"][comparison]
            if power_reading:
                continue
            covered[("cap", level, comparison)] = _covers(interval, true.get(comparison))
            claim = utility_gate([by_seed[s].state for s in seeds], GATE_OF[comparison],
                                 CONFIRMATION_ALPHA) if seeds else None  # fmt: skip
            out["utility_claim"][(level, comparison)] = None if claim is None else claim["passes"]
    for key in [k for k in covered if k[2] == "A-F"]:
        point, level, _ = key
        if "P-F" in active:
            covered[(point, level, "joint")] = covered[key] and covered[(point, level, "P-F")]
        else:  # the procedure with P closed: P–F counts nothing, the joint test is A–F's
            covered[(point, level, "P-F")] = None
            covered[(point, level, "joint")] = covered[key]
            for name in ("direction", "readings", "utility_claim"):
                out[name].setdefault((level, "P-F"), None)
    return StudyOutcome(**common, active=active, confirmed=confirmed, why=why, N=N, cap=cap,
                        blocks=blocks_n, truth=true, **out, **memory)  # fmt: skip


def _stop(blocks: Sequence[_Block], active: Sequence[str], n: int, cap: int) -> int:
    """The blocks started at the stop (§"Confirmation"): every active counter at N, or the cap
    started (k found by scanning the block states)."""
    counts = dict.fromkeys(active, 0)
    for k, block in enumerate(blocks[:cap], start=1):
        for comparison in active:
            counts[comparison] += bool(block.state[GATE_OF[comparison]]["supported"])
        if all(count >= n for count in counts.values()):
            return k
    return cap


def _interval(d: Sequence[float], level: float, resamples: int,
              seed: np.random.SeedSequence) -> Interval | None:  # fmt: skip
    """The registered interval (``bootstrap_t``, chunk 500); None for no block at all."""
    if not d:
        return None
    return bootstrap_t(np.asarray(d, dtype=float), level=level, resamples=resamples, seed=seed,
                       chunk=CHUNK)  # fmt: skip


def final_direction(primary: Interval | None, differences: Sequence[Mapping[str, float]],
                    level: float, resamples: int,
                    seed: np.random.SeedSequence) -> str | None:  # fmt: skip
    """The direction of the registered final reading ("Judge and audit": "the strongest
    conclusion that holds in all three computations"), as the final analysis reads it
    (``study3_estimates.three_way_reading`` of the primary interval, the missing-labels envelope of
    d_low's and d_high's intervals and the all-missing-harmful interval, each
    ``study3_inference.reading``): the direction common to the three, or None. ``primary`` is
    the primary computation's interval on ``seed``; ``differences`` the analysis set's blocks'
    ``_computations``, in its order; the other three intervals are drawn on the same ``seed``, so
    that the four share every resample (one stream, as ``study3_estimates.three_computations``
    draws them). A primary interval with no direction (undetermined, or None for no block) gives
    None at once: no common direction can hold without it."""
    if primary is None or reading(primary).direction is None:
        return None
    found = {key: _interval([d[key] for d in differences], level, resamples, seed)
             for key in ("missing_low", "missing_high", "all_missing_harmful")}  # fmt: skip
    combined = three_way_reading(primary, envelope(found["missing_low"], found["missing_high"]),
                                 found["all_missing_harmful"])  # fmt: skip
    return combined.direction


def report_scenario(scenario: Scenario) -> Scenario:
    """A power scenario at the reported power's denser mask (M6b): its missing labels MCAR at
    ``REPORT_MISSING_RATE``, every other parameter and its name (so every stream) its own."""
    return replace(scenario, missing="MCAR", missing_rate=REPORT_MISSING_RATE)


def power_report(scenario: Scenario, setup: Setup, base: int, index: int, *, resamples: int,
                 levels: Sequence[float]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """The power study ``index`` of ``scenario`` at the reported power's denser mask (M6b,
    ``missing_labels_record``'s ``report``): the same study again in ``"screen"`` mode on its own
    streams and its scenario's calibration, its labels missing at ``REPORT_MISSING_RATE``
    (``report_scenario``: each label's one uniform, so that its mask contains the scenario's),
    through the registered procedure (its own validation spread, N, cap and confirmation) to A–F's
    cap reading and final direction (``simulate_study``'s ``power_reading``). It uses the N the
    rule chose, before any n_min increase: it is not the power of the final increased design.
    Under MCAR, Δ scales by 0.99 / 0.999 (±2 pp becomes about ±1.982 pp), and the SD need not
    shrink by exactly 1 %. Per level slot: ``N``, ``confirmed``, ``readings`` and ``direction``
    (A–F's)."""
    if not scenario.power:
        raise ValueError(f"scenario {scenario.name} is not a power scenario")
    denser = report_scenario(scenario)
    c = scenario.candidate
    paired = _with(setup, denser, c, setup.calibration(scenario, c))
    o = simulate_study(denser, paired, base, index, mode="screen", resamples=resamples,
                       levels=levels, power_reading=True)  # fmt: skip
    return {
        "N": {level_key(lv): o.N[lv] for lv in o.levels},
        "confirmed": {level_key(lv): bool(o.confirmed[lv]) for lv in o.levels},
        "readings": {level_key(lv): o.readings.get((lv, "A-F")) for lv in o.levels},
        "direction": {level_key(lv): o.direction.get((lv, "A-F")) for lv in o.levels},
    }


def _covers(interval: Interval | None, value: float | None) -> bool:
    """lower ≤ truth ≤ upper; an undetermined interval, a missing one (an N point not reached,
    M18) or an undefined truth is a miss."""
    if interval is None or interval.undetermined or value is None:
        return False
    return bool(interval.lower <= value <= interval.upper)


# -- a study's part row


_ROW_FIELDS = ("index", "mode", "candidate", "active", "levels", "rates", "gates", "m",
               "confirmed", "why", "N", "cap", "blocks", "covered", "direction", "readings",
               "utility_claim", "truth", "report")  # fmt: skip
REPORT_FIELDS = ("N", "confirmed", "readings", "direction")  # a row's ``report``, per level slot


level_key = level_slot  # a level as the identities and the results write it ("0.975", …)


def _covered_keys(levels: Sequence[float]) -> list[tuple[int | str, float, str]]:
    return [(point, level, c) for point in N_POINTS for level in levels for c in TESTED]


def outcome_json(o: StudyOutcome) -> dict[str, Any]:
    """A compact part row of a study: N, the cap and the flags, the coverage flags bit-packed
    (``present``, ``known``: not None, ``covered``; bit i is the i-th (N point, level,
    comparison) in order), never the spread nor the utility values."""
    keys = _covered_keys(o.levels)
    if set(o.covered) - set(keys):
        raise ValueError("coverage flags outside the study's N points, levels and comparisons")
    present = known = value = 0
    for i, key in enumerate(keys):
        if key in o.covered:
            present |= 1 << i
            if o.covered[key] is not None:
                known |= 1 << i
                value |= int(bool(o.covered[key])) << i
    per = lambda d: {level_key(lv): v for lv, v in d.items()}  # noqa: E731
    pairs = lambda d: {  # noqa: E731
        level_key(lv): {c: d[(lv2, c)] for (lv2, c) in d if lv2 == lv}
        for lv in sorted({lv for lv, _ in d})
    }
    return {
        "index": o.index,
        "mode": o.mode,
        "candidate": o.candidate,
        "active": list(o.active),
        "levels": [float(lv) for lv in o.levels],
        "rates": {
            str(c): {arm: dict(r) for arm, r in rates.items()} for c, rates in o.rates.items()
        },  # fmt: skip
        "gates": dict(o.gates),
        "m": o.m,
        "confirmed": per(o.confirmed),
        "why": per(o.why),
        "N": per(o.N),
        "cap": per(o.cap),
        "blocks": per(o.blocks),
        "covered": {"present": hex(present), "known": hex(known), "covered": hex(value)},
        "direction": pairs(o.direction),
        "readings": pairs(o.readings),
        "utility_claim": pairs(o.utility_claim),
        "truth": dict(o.truth),
        "report": None if o.report is None else {k: dict(o.report[k]) for k in REPORT_FIELDS},
    }


def outcome_from_json(d: Mapping[str, Any]) -> StudyOutcome:
    """The study of a part row (``outcome_json``). ``ValueError`` for another field."""
    if not isinstance(d, Mapping) or set(d) != set(_ROW_FIELDS):
        raise ValueError(f"a study's row holds exactly {list(_ROW_FIELDS)}")
    levels = tuple(float(lv) for lv in d["levels"])
    by_key = {level_key(lv): lv for lv in levels}

    def per(name: str) -> dict[float, Any]:
        if set(d[name]) != set(by_key):
            raise ValueError(f"{name} must name every level of the row")
        return {by_key[k]: v for k, v in d[name].items()}

    def pairs(name: str) -> dict[tuple[float, str], Any]:
        return {(by_key[k], c): v for k, inner in d[name].items() for c, v in inner.items()}

    flags = d["covered"]
    if not isinstance(flags, Mapping) or set(flags) != {"present", "known", "covered"}:
        raise ValueError("covered must be the three bit masks")
    report = d["report"]
    if report is not None and not (
        isinstance(report, Mapping) and set(report) == set(REPORT_FIELDS)
        and all(isinstance(report[k], Mapping) and set(report[k]) == set(by_key)
                for k in REPORT_FIELDS)
    ):  # fmt: skip
        raise ValueError("report must be null or its fields at every level of the row")
    present, known, value = (int(flags[k], 16) for k in ("present", "known", "covered"))
    covered = {}
    for i, key in enumerate(_covered_keys(levels)):
        if present >> i & 1:
            covered[key] = bool(value >> i & 1) if known >> i & 1 else None
    return StudyOutcome(
        index=d["index"],
        mode=d["mode"],
        candidate=d["candidate"],
        active=list(d["active"]),
        levels=levels,
        rates={
            int(c): {arm: dict(r) for arm, r in rates.items()} for c, rates in d["rates"].items()
        },  # fmt: skip
        gates=dict(d["gates"]),
        m=d["m"],
        confirmed=per("confirmed"),
        why=per("why"),
        N=per("N"),
        cap=per("cap"),
        blocks=per("blocks"),
        covered=covered,
        direction=pairs("direction"),
        readings=pairs("readings"),
        utility_claim=pairs("utility_claim"),
        truth=dict(d["truth"]),
        report=None if report is None else {k: dict(report[k]) for k in REPORT_FIELDS},
    )


# ------------------------------------------------------------------ the calibration

_GL20 = np.polynomial.legendre.leggauss(20)


def bivariate_normal(t1: np.ndarray, t2: np.ndarray, rho: float) -> np.ndarray:
    """Φ₂(t1, t2; ρ), elementwise: Φ(t1)Φ(t2) + (1/2π)∫₀^ρ exp(−(t1² − 2r·t1·t2 + t2²)/(2(1 −
    r²)))/√(1 − r²) dr, the integral by 20-point Gauss–Legendre (|ρ| ≤ 0.7 here: smooth in r).
    Infinite arguments are taken at ±40."""
    t1 = np.clip(np.asarray(t1, dtype=float), -40.0, 40.0)
    t2 = np.clip(np.asarray(t2, dtype=float), -40.0, 40.0)
    nodes, weights = _GL20
    r = rho * (nodes + 1) / 2
    one = 1 - r * r
    a, b = t1[..., None], t2[..., None]
    inside = np.exp(-(a * a - 2 * r * a * b + b * b) / (2 * one)) / np.sqrt(one)
    return ndtr(t1) * ndtr(t2) + rho / 2 * (inside * weights).sum(axis=-1) / (2 * math.pi)


def label_variance(
    h: np.ndarray, c: np.ndarray, arms: Sequence[str], scenario: Scenario
) -> np.ndarray:
    """Var(Σ_j c_j·share_j | h), the label noise of a weighted sum of a block's primary judged
    shares (M5, M6, M6b): the 521 prompts are independent given the block, so it is
    Var(Σ_j c_j·J_j)/521 for one prompt's judged indicators J_j; two endpoints' harmful labels
    are bivariate normal thresholds (ρ_a within an arm, ρ_b across arms), the judge's calls and
    the missing labels independent given the label. ``h``, ``c``: (..., J) arrays."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    c = np.asarray(c, dtype=float)
    m_h, m_n = scenario.missing_rates()
    high = np.array([(1 - m_h) * JUDGES[scenario.judge][arm][0] for arm in arms])
    low = np.array([(1 - m_n) * JUDGES[scenario.judge][arm][1] for arm in arms])
    p = high * h + low * (1 - h)
    t = ndtri(h)
    out = (c * c * p * (1 - p)).sum(axis=-1)
    for j in range(len(arms)):
        for k in range(j + 1, len(arms)):
            rho = RHO_ARM if arms[j] == arms[k] else RHO_BLOCK
            both_harmful = bivariate_normal(t[..., j], t[..., k], rho)
            hj, hk = h[..., j], h[..., k]
            joint = (
                high[j] * high[k] * both_harmful
                + high[j] * low[k] * (hj - both_harmful)
                + low[j] * high[k] * (hk - both_harmful)
                + low[j] * low[k] * (1 - hj - hk + both_harmful)
            )
            out = out + 2 * c[..., j] * c[..., k] * (joint - p[..., j] * p[..., k])
    return out / PROMPTS


@dataclass(frozen=True)
class _Draws:
    """The calibration's simulated selections of one (scenario, candidate) at one set of safe
    shifts, every run's failure left to its uniform (in memory): per study the final rates; per
    study and validation block (``CALIBRATION_BLOCKS``) the natural support of each arm (no
    failure), the runs' failure uniforms, each arm's x_lo, x_hi and w where naturally supported,
    the block effects, the vector's seed and each arm's interpolated MMLU and factual losses,
    unshifted and as shifted."""

    rates: tuple[dict[str, dict[str, float]], ...]
    natural: np.ndarray  # (S, B, 3) bool
    uniforms: np.ndarray  # (S, B, 3, 2)
    x_lo: np.ndarray  # (S, B, 3)
    x_hi: np.ndarray
    w: np.ndarray
    effects: np.ndarray  # (S, B, 2)
    seeds: np.ndarray  # (S, B) the vectors' seeds
    mmlu: np.ndarray  # (S, B, 3) unshifted
    factual: np.ndarray
    mmlu_shifted: np.ndarray
    factual_shifted: np.ndarray

    def supported(self, comparison: str, failure: float) -> np.ndarray:
        """(S, B): the blocks supported for the comparison at the per-run failure f: both arms
        naturally supported and none of their four runs failed (a failed endpoint removes its
        arm's support, "Within a block")."""
        a, b = (ARM_ORDER.index(arm) for arm in COMPARISONS[comparison])
        alive = (self.uniforms[:, :, [a, b], :] >= failure).all(axis=(2, 3))
        return self.natural[:, :, a] & self.natural[:, :, b] & alive


_DRAWS: dict[tuple, _Draws] = {}
_DRAWS_LIMIT = 6


def clear_caches() -> None:
    """Forget the remembered truths, shifted vectors, selections and calibration draws (each is
    recomputed identically)."""
    for cache in (_MOMENTS, _GRIDS, _SHIFTED, _SELECTIONS, _DRAWS):
        cache.clear()


def _draws(scenario: Scenario, setup: Setup, candidate: int, seed: np.random.SeedSequence,
           studies: int) -> _Draws:  # fmt: skip
    """The selections and validations of ``studies`` calibration studies, through
    ``simulate_study(..., validation_only=True)`` on the calibration's stream, at the setup's
    safe shifts for (scenario, candidate) and with no failure (each run's uniform kept, so that
    every f reuses the same draws), 28 blocks each, no labels."""
    if not isinstance(seed, np.random.SeedSequence):
        raise ValueError("the calibration's stream is a numpy SeedSequence (calibration_stream)")
    if not (_integer(studies) and studies >= 2):
        raise ValueError(f"studies {studies!r} is not an integer of at least 2")
    calibration = _current(scenario, setup, candidate)
    shifts = (
        tuple(sorted((arm, tuple(sorted(v.items()))) for arm, v in calibration.mmlu_shift.items())),
        tuple(sorted(calibration.factual_shift.items())),
    )
    entropy = tuple(int(x) for x in np.atleast_1d(np.asarray(seed.entropy, dtype=object)))
    key = (scenario, candidate, entropy, tuple(seed.spawn_key), int(studies), shifts,
           setup.fingerprint)  # fmt: skip
    if key in _DRAWS:
        return _DRAWS[key]
    use = replace(_with(setup, scenario, candidate, replace(calibration, failure=0.0)),
                  validation_blocks=CALIBRATION_BLOCKS)  # fmt: skip
    shape = (studies, CALIBRATION_BLOCKS, len(ARM_ORDER))
    natural = np.zeros(shape, dtype=bool)
    uniforms = np.empty((*shape, len(LABELS)))
    x_lo, x_hi, w, mmlu, factual, mmlu_s, factual_s = (np.full(shape, np.nan) for _ in range(7))
    effects = np.empty((studies, CALIBRATION_BLOCKS, 2))
    seeds = np.empty((studies, CALIBRATION_BLOCKS), dtype=np.int64)
    rates = []
    kstar_ln = math.log(CANDIDATES[candidate]["targets"]["star"])
    for i in range(studies):
        o = simulate_study(scenario, use, seed, i, resamples=_decisions.SCREEN_RESAMPLES,
                           levels=setup.levels, validation_only=True, candidate=candidate,
                           labels=False)  # fmt: skip
        rates.append(o.rates[candidate])
        for k, b in enumerate(o.validation):
            uniforms[i, k], effects[i, k] = b.uniforms, b.effects
            seeds[i, k] = setup.vectors[b.vector].seed
            for a, arm in enumerate(ARM_ORDER):
                unit = b.state["arms"][arm]
                if not unit["supported"]:
                    continue
                natural[i, k, a] = True
                lo = 0 if b.x[a, 0] < b.x[a, 1] else 1
                x_lo[i, k, a], x_hi[i, k, a] = b.x[a, lo], b.x[a, 1 - lo]
                w[i, k, a] = unit["weight"]
                plain = setup.vectors[b.vector]
                ends = [plain.losses(arm, b.x[a, j] + kstar_ln, setup.slopes)
                        for j in (lo, 1 - lo)]  # fmt: skip
                mmlu[i, k, a] = interpolated(ends[0][0], ends[1][0], unit["weight"])
                factual[i, k, a] = interpolated(ends[0][1], ends[1][1], unit["weight"])
                mmlu_s[i, k, a], factual_s[i, k, a] = unit["mmlu_loss_pp"], unit["factual_loss_pp"]
    if len(_DRAWS) >= _DRAWS_LIMIT:
        _DRAWS.clear()
    found = _Draws(tuple(rates), natural, uniforms, x_lo, x_hi, w, effects, seeds, mmlu, factual,
                   mmlu_s, factual_s)  # fmt: skip
    _DRAWS[key] = found
    return found


def _support_at(draws: _Draws, failure: float) -> tuple[float, float]:
    """The realized A–F support at f over every validation block, and its Monte Carlo SE (the
    studies' supports, 28 blocks each, are independent)."""
    per_study = draws.supported("A-F", failure).mean(axis=1)
    return float(per_study.mean()), float(per_study.std(ddof=1) / math.sqrt(per_study.size))


def realized_support(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    failure: float,
    seed: np.random.SeedSequence,
    *,
    studies: int = _decisions.CALIBRATION_STUDIES,
) -> tuple[float, float]:
    """The realized A–F support of the validation blocks of ``studies`` simulated selections
    (search, repair, validation) at the rates each selects, at the per-run failure ``failure``,
    and its Monte Carlo SE (M7). The selections, the natural support and the failure uniforms
    are drawn once (``_draws``), so every f reuses them."""
    if not (math.isfinite(failure) and 0 <= failure < 1):
        raise ValueError(f"the per-run failure {failure!r} is not in [0, 1)")
    return _support_at(_draws(scenario, setup, candidate, seed, studies), failure)


def calibrate_failure(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = _decisions.CALIBRATION_STUDIES,
) -> tuple[float, float, bool]:
    """(f, the realized support, flagged) (M7): f by bisection on [0, 1) until the realized A–F
    support of the validation blocks is the scenario's s within 1e-4; f = 0, flagged, when the
    natural failures alone (bracket, width, collapse) leave the support below s. The selection
    does not depend on f (search and numerical runs never fail, M13), and every step reuses the
    same draws, so the support falls monotonically in f."""
    draws = _draws(scenario, setup, candidate, seed, studies)
    target = scenario.support
    natural, _ = _support_at(draws, 0.0)
    if natural < target:
        return 0.0, natural, True
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        found, _ = _support_at(draws, mid)
        if abs(found - target) <= SUPPORT_TOLERANCE:
            return mid, found, False
        if found > target:
            lo = mid
        else:
            hi = mid
    raise CalibrationNotStable(f"{scenario.name}: no f gives the support {target} within "
                               f"{SUPPORT_TOLERANCE}")  # fmt: skip


def _safe_shifts(
    draws: _Draws, failure: float, vectors: Sequence[Vector]
) -> tuple[dict[str, dict[int, float]], dict[str, float], dict[str, float], bool]:
    """The safe alternative's shifts (M10) on the supported blocks' unshifted interpolated
    utility at f: per arm, c_v = 3 − m + (λ − 1)(ū_v − m), λ² = (4 − W)/B (W the within-vector,
    B the between-vector variance, block-weighted, so that the shifted values have mean 3 pp and
    SD 2 pp); λ = 0 and flagged when W > 4 (or B = 0); a vector no supported block drew takes
    3 − m. The factual losses take one shift to mean 2 pp."""
    mmlu_shift: dict[str, dict[int, float]] = {}
    factual_shift: dict[str, float] = {}
    lambdas: dict[str, float] = {}
    flagged = False
    for a, arm in enumerate(ARM_ORDER):
        mask = draws.supported("A-F" if arm != "P" else "P-F", failure)
        u, f, seeds = draws.mmlu[:, :, a][mask], draws.factual[:, :, a][mask], draws.seeds[mask]
        if u.size == 0:
            raise CalibrationNotStable(f"arm {arm}: no supported block to calibrate on")
        m = float(u.mean())
        kinds, inverse = np.unique(seeds, return_inverse=True)
        counts = np.bincount(inverse)
        means = np.bincount(inverse, weights=u) / counts
        within = float(((u - means[inverse]) ** 2).mean())
        between = float((counts * (means - m) ** 2).sum() / u.size)
        target = SAFE_MMLU_SD**2
        if within > target or between <= 0:
            lam, flagged = 0.0, True
        else:
            lam = math.sqrt((target - within) / between)
        seen = dict(zip(kinds.tolist(), means.tolist(), strict=True))
        mmlu_shift[arm] = {
            v.seed: SAFE_MMLU_MEAN - m + (lam - 1) * (seen[v.seed] - m)
            if v.seed in seen
            else SAFE_MMLU_MEAN - m
            for v in vectors
        }
        factual_shift[arm] = SAFE_FACTUAL_MEAN - float(f.mean())
        lambdas[arm] = lam
    return mmlu_shift, factual_shift, lambdas, flagged


def _utility_realized(draws: _Draws, failure: float) -> dict[str, tuple[float, float, float]]:
    """Per arm, the interpolated MMLU loss's mean and SD (over the supported blocks pooled,
    ddof 0) and the factual loss's mean, as the blocks saw them (shifted)."""
    out = {}
    for a, arm in enumerate(ARM_ORDER):
        mask = draws.supported("A-F" if arm != "P" else "P-F", failure)
        u, f = draws.mmlu_shifted[:, :, a][mask], draws.factual_shifted[:, :, a][mask]
        out[arm] = (float(u.mean()), float(u.std()), float(f.mean()))
    return out


def calibrate_safe(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = _decisions.CALIBRATION_STUDIES,
) -> Calibration:
    """The setup's calibration of (scenario, candidate) with the safe alternative's shifts (M10)
    computed on the supported validation blocks' interpolated utility at its f, and the realized
    mean, SD and factual mean measured again with the shifts applied (the shift changes gross
    collapses, so the measurement draws again). The empirical point (M11) keeps its vectors: no
    shift, the realized utility measured."""
    calibration = _current(scenario, setup, candidate)
    if scenario.utility != "safe":
        plain = replace(calibration, mmlu_shift={}, factual_shift={}, utility_lambda={},
                        utility_flagged=False)  # fmt: skip
        draws = _draws(scenario, _with(setup, scenario, candidate, plain), candidate, seed,
                       studies)  # fmt: skip
        return replace(plain, utility_realized=_utility_realized(draws, plain.failure))
    draws = _draws(scenario, setup, candidate, seed, studies)
    mmlu_shift, factual_shift, lambdas, flagged = _safe_shifts(
        draws, calibration.failure, setup.vectors
    )
    shifted = replace(calibration, mmlu_shift=mmlu_shift, factual_shift=factual_shift,
                      utility_lambda=lambdas, utility_flagged=flagged)  # fmt: skip
    measured = _draws(scenario, _with(setup, scenario, candidate, shifted), candidate, seed,
                      studies)  # fmt: skip
    return replace(shifted, utility_realized=_utility_realized(measured, calibration.failure))


def study_variances(
    scenario: Scenario,
    draws: _Draws,
    failure: float,
    sigma: float,
    comparison: str,
    offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per calibration study, E[s² | blocks] of d over its supported validation blocks at f (M1),
    the arm's block effect σ·e + ``offset`` (proportions²), and which studies have the two
    supported blocks a variance needs: the sample variance of the blocks' expected judged
    differences plus the mean of their label variances (``label_variance``), the expectation of
    the observed d's sample variance given the blocks, so that the labels count exactly and
    every σ reuses the same draws."""
    arm = COMPARISONS[comparison][0]
    a, f_ = ARM_ORDER.index(arm), ARM_ORDER.index("F")
    mask = draws.supported(comparison, failure)
    effect = draws.effects[:, :, 0 if arm == "A" else 1]
    g = scenario.curvature_scale

    def harm(index: int, which: np.ndarray, extra: np.ndarray | float) -> np.ndarray:
        name = ARM_ORDER[index]
        x = np.where(mask, which[:, :, index], 0.0)
        return np.clip(harm_curve(name, x, g) + extra, 0.0, 1.0)

    shift = sigma * effect + offset
    h = np.stack([harm(a, draws.x_lo, shift), harm(a, draws.x_hi, shift),
                  harm(f_, draws.x_lo, 0.0), harm(f_, draws.x_hi, 0.0)], axis=-1)  # fmt: skip
    wa, wf = np.where(mask, draws.w[:, :, a], 0.0), np.where(mask, draws.w[:, :, f_], 0.0)
    c = np.stack([1 - wa, wa, -(1 - wf), -wf], axis=-1)
    arms = (arm, arm, "F", "F")
    p = np.stack([judged_share(name, h[..., j], scenario) for j, name in enumerate(arms)], -1)
    mu = (c * p).sum(axis=-1)
    noise = label_variance(h, c, arms, scenario)
    n = mask.sum(axis=1)
    keep = n >= 2
    n_, mask_ = n[keep], mask[keep]
    mean = np.where(mask_, mu[keep], 0.0).sum(axis=1) / n_
    ss = np.where(mask_, (mu[keep] - mean[:, None]) ** 2, 0.0).sum(axis=1)
    variance = ss / (n_ - 1) + np.where(mask_, noise[keep], 0.0).sum(axis=1) / n_
    return variance, keep


def _spread(scenario: Scenario, draws: _Draws, failure: float, sigma: float,
            comparison: str, offset: float = 0.0) -> tuple[float, float]:  # fmt: skip
    """The observed spread of d over the supported validation blocks at f, the arm's offset
    ``offset`` (M1): the square root of the mean within-study variance (``study_variances``),
    the quantity the spread rule estimates, in pp, and its Monte Carlo SE."""
    variance, _ = study_variances(scenario, draws, failure, sigma, comparison, offset)
    total = float(variance.mean())
    se = float(variance.std(ddof=1) / math.sqrt(variance.size)) / (2 * math.sqrt(total))
    return 100 * math.sqrt(total), 100 * se


def sigma_tolerance(sigma: float) -> float:
    """The precision of a solved σ: Brent's bound on its distance from the root,
    ``SIGMA_PRECISION`` + ``SIGMA_RTOL``·σ (proportions)."""
    return SIGMA_PRECISION + SIGMA_RTOL * abs(sigma)


def _solve_sigma(scenario: Scenario, draws: _Draws, failure: float, comparison: str,
                 offset: float = 0.0, start: float | None = None) -> float:  # fmt: skip
    """σ such that the spread at the arm's offset ``offset`` (o⁰: Δ = 0) is the planning spread
    (0 when the labels and the geometry alone already reach it), by a bracketing root-finder
    (Brent, ``scipy.optimize.brentq``) to within ``sigma_tolerance``. The bracket is [0, 0.5]
    (its upper end doubled until it holds the root), or it starts at ``start`` (the previous
    round's σ) and steps away from it, up or down as the spread there falls short or not, by a
    relative 10⁻³·8^k (k = 0, 1, …; down to 0 at most) until it holds the root. The spread rises
    with σ, so every bracket holds the one root."""
    target = scenario.spread_pp
    known: dict[float, float] = {}

    def gap(sigma: float) -> float:
        if sigma not in known:
            known[sigma] = _spread(scenario, draws, failure, sigma, comparison, offset)[0] - target
        return known[sigma]

    if gap(0.0) >= 0:
        return 0.0
    if start is None or start <= 0:
        lo, hi = 0.0, 0.5
        while gap(hi) < 0:
            lo, hi = hi, 2 * hi
            if hi > 64:
                raise CalibrationNotStable(f"{scenario.name}: no σ reaches the spread {target} pp")
    elif gap(start) < 0:
        lo, hi, step = start, start * (1 + SIGMA_STEP), SIGMA_STEP
        while gap(hi) < 0:
            step *= 8
            lo, hi = hi, start * (1 + step)
            if hi > 64:
                raise CalibrationNotStable(f"{scenario.name}: no σ reaches the spread {target} pp")
    else:
        lo, hi, step = start * (1 - SIGMA_STEP), start, SIGMA_STEP
        while gap(lo) >= 0:
            step *= 8
            lo, hi = (start * (1 - step), lo) if step < 1 else (0.0, lo)
    root, found = brentq(gap, lo, hi, xtol=SIGMA_PRECISION, rtol=SIGMA_RTOL, maxiter=200,
                         full_output=True, disp=False)  # fmt: skip
    if not found.converged:
        raise CalibrationNotStable(f"{scenario.name}: σ for the spread {target} pp does not "
                                   f"converge ({found.flag})")  # fmt: skip
    return float(root)


def calibrate_spread(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = _decisions.CALIBRATION_STUDIES,
) -> tuple[float, float, float, float]:
    """(σ_AF, σ_PF, the attained A–F and P–F spreads in pp) (M1, M2): each σ by a bracketing
    root-finder (Brent, ``_solve_sigma``) so that the observed SD of d at Δ = 0 (the setup's
    o⁰, ``null_offset_a`` and ``null_offset_p``), over the supported validation blocks at the
    rates each selection chose, is the planning spread (``_spread``), at the setup's f and
    shifts."""
    calibration = _current(scenario, setup, candidate)
    draws = _draws(scenario, setup, candidate, seed, studies)
    f = calibration.failure
    null = {"A-F": calibration.null_offset_a, "P-F": calibration.null_offset_p}
    sigma = {c: _solve_sigma(scenario, draws, f, c, null[c]) for c in COMPARISONS}
    spread = {c: _spread(scenario, draws, f, sigma[c], c, null[c])[0] for c in COMPARISONS}
    return sigma["A-F"], sigma["P-F"], spread["A-F"], spread["P-F"]


def _mean_truth(
    scenario: Scenario,
    setup: Setup,
    calibration: Calibration,
    draws: _Draws,
    comparison: str,
    mapper: Callable | None = None,
) -> tuple[float, float]:
    """The mean over the selections of the per-study truth (``truth`` at each study's selected
    rates, through the same moments and ``_combine``; studies share rates, rounded to 3
    significant digits) and its Monte Carlo SE, in proportions. A study whose truth is undefined
    (no vector supports its rates) is left out. ``mapper`` (a parallel ``map``): the jitter
    quadratures not yet kept are built through it first (``_JitterGrid``)."""
    arms = COMPARISONS[comparison]
    shifted = _shifted(setup.vectors, calibration)
    kstar_ln = math.log(CANDIDATES[calibration.candidate]["targets"]["star"])
    keys = [tuple((r[arm]["l"], r[arm]["u"]) for arm in arms) for r in draws.rates]
    if mapper is not None and scenario.jitter:
        needed = sorted(
            {(arm, pair) for key in set(keys) for arm, pair in zip(arms, key, strict=True)}
        )
        _build_grids(mapper, scenario, calibration, setup.slopes, kstar_ln,
                     [(v, arm, {"l": pair[0], "u": pair[1]}) for arm, pair in needed
                      for v in shifted])  # fmt: skip
    moments: dict[tuple[str, tuple[float, float]], list[tuple[float, float]]] = {}
    for key in set(keys):
        for arm, pair in zip(arms, key, strict=True):
            if (arm, pair) not in moments:
                rates = {"l": pair[0], "u": pair[1]}
                moments[(arm, pair)] = [
                    _cached_moments(v, arm, rates, scenario, calibration, setup.slopes, kstar_ln,
                                    tol=TOLERANCE, max_nodes=MAX_NODES)
                    for v in shifted
                ]  # fmt: skip
    values = {
        key: _combine({arm: moments[(arm, pair)] for arm, pair in zip(arms, key, strict=True)},
                      calibration.failure)[comparison]
        for key in set(keys)
    }  # fmt: skip
    per = np.array([values[k] for k in keys])
    per = per[np.isfinite(per)]
    if per.size < 2:
        raise CalibrationNotStable(f"{scenario.name}: the truth is undefined at the selections")
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(per.size))


def _solve_offset(scenario: Scenario, setup: Setup, calibration: Calibration, draws: _Draws,
                  comparison: str, target: float,
                  mapper: Callable | None = None) -> float:  # fmt: skip
    """The offset of A (A–F) or P (P–F) at which the mean truth over the selections is
    ``target`` (proportions): the secant method from the current offset, its first step with the
    judged share's slope (the truth rises with the offset, nearly linearly: only the clip bends
    it), until the mean truth is within ``OFFSET_PRECISION`` of the target."""
    arm = COMPARISONS[comparison][0]
    name = "offset_a" if arm == "A" else "offset_p"

    def gap(offset: float) -> float:
        here = replace(calibration, **{name: offset})
        return _mean_truth(scenario, setup, here, draws, comparison, mapper)[0] - target

    m_h, m_n = scenario.missing_rates()
    se, fpr = JUDGES[scenario.judge][arm]
    before = calibration.offset(arm)
    g_before = gap(before)
    if abs(g_before) <= OFFSET_PRECISION:
        return before
    now = before - g_before / ((1 - m_h) * se - (1 - m_n) * fpr)
    for _ in range(50):
        g_now = gap(now)
        if abs(g_now) <= OFFSET_PRECISION:
            return now
        if g_now == g_before:
            break
        before, g_before, now = now, g_now, now - g_now * (now - before) / (g_now - g_before)
    raise CalibrationNotStable(f"{scenario.name}: the offset of {arm} does not reach the mean "
                               f"truth {target}")  # fmt: skip


def calibrate_offsets(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = _decisions.CALIBRATION_STUDIES,
    mapper: Callable | None = None,
) -> tuple[float, float]:
    """(o_A, o_P) (M15, M2): each solved, from the setup's o⁰, so that the mean, over the
    selections, of the per-study truth (``truth`` at that study's selected rates) is Δ_AF
    (Δ_PF), at the setup's f, shifts and σ."""
    calibration = _current(scenario, setup, candidate).at_null()
    draws = _draws(scenario, setup, candidate, seed, studies)
    o_a = _solve_offset(scenario, setup, calibration, draws, "A-F", scenario.delta_af_pp / 100,
                        mapper)  # fmt: skip
    o_p = _solve_offset(scenario, setup, calibration, draws, "P-F", scenario.delta_pf_pp / 100,
                        mapper)  # fmt: skip
    return o_a, o_p


def _current(scenario: Scenario, setup: Setup, candidate: int) -> Calibration:
    """The setup's calibration of (scenario, candidate), or where the calibration starts."""
    found = setup.calibrations.get((scenario, candidate))
    return found if found is not None else _starting_calibration(scenario, setup, candidate)


def _starting_calibration(scenario: Scenario, setup: Setup, candidate: int) -> Calibration:
    """Where the calibration starts: no failure, no shift (zero shifts of every vector for the
    safe alternative), no block effect, no offset."""
    zero = {}
    if scenario.utility == "safe":
        zero = {"mmlu_shift": {arm: {v.seed: 0.0 for v in setup.vectors} for arm in ARM_ORDER},
                "factual_shift": dict.fromkeys(ARM_ORDER, 0.0)}  # fmt: skip
    return Calibration(
        candidate=candidate, failure=0.0, realized_support=0.0, support_flagged=False,
        mmlu_shift=zero.get("mmlu_shift", {}), factual_shift=zero.get("factual_shift", {}),
        utility_realized={}, utility_flagged=False, sigma_af=0.0, sigma_pf=0.0, offset_a=0.0,
        offset_p=0.0, attained_af_pp=0.0, attained_pf_pp=0.0, attained_delta_pp={},
    )  # fmt: skip


def _settled(now: tuple[float, float], before: tuple[float, float]) -> bool:
    """Whether a round of the σ/o⁰ alternation moved (σ, o⁰) by no more than the solvers'
    precision: σ by at most twice ``sigma_tolerance`` (two solutions of one root differ by at
    most that), o⁰ by at most ``OFFSET_PRECISION`` (the mean truth moves with the offset at a
    slope below 1, so such a move stays within the offset solver's tolerance on the truth)."""
    (sigma, null), (sigma_before, null_before) = now, before
    return (abs(sigma - sigma_before) <= 2 * sigma_tolerance(max(sigma, sigma_before))
            and abs(null - null_before) <= OFFSET_PRECISION)  # fmt: skip


def calibrate_scenario(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = _decisions.CALIBRATION_STUDIES,
    mapper: Callable | None = None,
) -> Calibration:
    """``_calibrate_scenario``; a jitter scenario's keeps its truths' quadratures across its σ
    and offset steps (up to ``CALIBRATION_GRIDS``) and forgets them when it ends."""
    global _GRIDS_LIMIT
    if not scenario.jitter:
        return _calibrate_scenario(scenario, setup, candidate, seed, studies=studies,
                                   mapper=mapper)  # fmt: skip
    _GRIDS.clear()
    _GRIDS_LIMIT = CALIBRATION_GRIDS
    try:
        return _calibrate_scenario(scenario, setup, candidate, seed, studies=studies,
                                   mapper=mapper)  # fmt: skip
    finally:
        _GRIDS_LIMIT = GRIDS_KEPT
        _GRIDS.clear()


def _calibrate_scenario(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = _decisions.CALIBRATION_STUDIES,
    mapper: Callable | None = None,
) -> Calibration:
    """Everything calibrated for (scenario, candidate) through the selection, on the one stream
    ``seed`` (``calibration_stream``): f (``calibrate_failure``) and the safe shifts
    (``calibrate_safe``) alternate until neither moves (at most 10 rounds, otherwise
    ``CalibrationNotStable``); σ and the offsets depend on them and not the other way round, so
    they are computed in the round in which nothing moved: σ at the Δ = 0 offsets o⁰ and o⁰ at
    σ alternate until neither moves by more than its solver's precision (``_settled``; at most
    10 rounds), then the offsets are solved for the scenario's Δ from o⁰, with the attained
    spreads (at o⁰) and mean truths and their Monte Carlo SEs. A gate scenario has no harm: σ
    and the offsets stay 0. ``fixation`` writes the result and ``run`` recomputes it.
    ``mapper`` (a parallel ``map``) builds the jitter quadratures the truths need (C12) on
    several processes; the result does not depend on it."""
    calibration = _starting_calibration(scenario, setup, candidate)
    before = None
    for _ in range(CALIBRATION_ROUNDS):
        here = _with(setup, scenario, candidate, calibration)
        f, realized, flagged = calibrate_failure(scenario, here, candidate, seed, studies=studies)
        calibration = replace(calibration, failure=f, realized_support=realized,
                              support_flagged=flagged)  # fmt: skip
        calibration = calibrate_safe(scenario, _with(setup, scenario, candidate, calibration),
                                     candidate, seed, studies=studies)  # fmt: skip
        now = (calibration.failure, calibration.mmlu_shift, calibration.factual_shift)
        if now == before:
            break
        before = now
    else:
        raise CalibrationNotStable(
            f"{scenario.name} at candidate {candidate}: f and the safe shifts still moved after "
            f"{CALIBRATION_ROUNDS} rounds"
        )
    if scenario.role == "gate":
        return calibration
    # σ at Δ = 0 (M1): σ given o⁰ (its bracket from the previous round's σ), then o⁰ (the mean
    # truth 0) given σ, until neither moves by more than its solver's precision
    draws = _draws(scenario, _with(setup, scenario, candidate, calibration), candidate, seed,
                   studies)  # fmt: skip
    f, before = calibration.failure, None
    for _ in range(CALIBRATION_ROUNDS):
        null = {"A-F": calibration.null_offset_a, "P-F": calibration.null_offset_p}
        start = {"A-F": calibration.sigma_af, "P-F": calibration.sigma_pf}
        sigma = {c: _solve_sigma(scenario, draws, f, c, null[c], start[c]) for c in COMPARISONS}
        calibration = replace(calibration, sigma_af=sigma["A-F"], sigma_pf=sigma["P-F"])
        probe = calibration.at_null()
        null = {c: _solve_offset(scenario, setup, probe, draws, c, 0.0, mapper)
                for c in COMPARISONS}  # fmt: skip
        calibration = replace(calibration, null_offset_a=null["A-F"], null_offset_p=null["P-F"])
        now = {c: (sigma[c], null[c]) for c in COMPARISONS}
        if before is not None and all(_settled(now[c], before[c]) for c in COMPARISONS):
            break
        before = now
    else:
        raise CalibrationNotStable(
            f"{scenario.name} at candidate {candidate}: σ and the Δ = 0 offsets still moved "
            f"after {CALIBRATION_ROUNDS} rounds"
        )
    spread = {c: _spread(scenario, draws, f, s, c, n) for c, s, n in
              (("A-F", calibration.sigma_af, calibration.null_offset_a),
               ("P-F", calibration.sigma_pf, calibration.null_offset_p))}  # fmt: skip
    calibration = replace(calibration, attained_af_pp=spread["A-F"][0],
                          attained_pf_pp=spread["P-F"][0],
                          attained_spread_se_pp={c: v[1] for c, v in spread.items()})  # fmt: skip
    probe = calibration.at_null()
    offset_a = _solve_offset(scenario, setup, probe, draws, "A-F", scenario.delta_af_pp / 100,
                             mapper)  # fmt: skip
    offset_p = _solve_offset(scenario, setup, probe, draws, "P-F", scenario.delta_pf_pp / 100,
                             mapper)  # fmt: skip
    calibration = replace(calibration, offset_a=offset_a, offset_p=offset_p)
    attained = {}
    for comparison in COMPARISONS:
        mean, se = _mean_truth(scenario, setup, calibration, draws, comparison, mapper)
        attained[comparison] = (100 * mean, 100 * se)
    return replace(calibration, attained_delta_pp=attained)


# ------------------------------------------------------------------ the fixation

FIXATION_KIND = "simulation-fixation"
STREAMS = {"calibration": 20261101, "verification": 20261102}  # M23
# The procedure's rules the fixation freezes, each with its source (ending "M<n>, decided")
RULES: dict[str, dict[str, str]] = {
    "screen_simplifications": {
        "rule": "the screen runs the scenario's candidate with no fallback (a failed A-F gate ends "
        "the study, counted), takes N from the rule alone and computes P-F in every confirmed "
        "study as if active",
        "source": "the coverage and power tests are per candidate; A-F's analysis set does not "
        "depend on P's activity; M14 and M16, decided",
    },
    "pf_verification": {
        "rule": "in the verification a P-F test counts only the studies in which P-F is active "
        "(the validation gate of the study's final candidate passed: a P-F closed at candidate "
        "1 reopens at candidate 2, amendment 7)",
        "source": "P's actual activity in the realized procedure; M16, decided",
    },
    "joint_verification": {
        "rule": "the joint test of the verification",
        "source": "counted over the studies confirmed for A-F; a study misses jointly if A-F "
        "misses, or if P-F is active in it and misses; M27, decided",
    },
    "draw_bound": {
        "rule": "each test draws at most max_draws_factor times M studies; a test short of M "
        "confirmed studies at the bound is not attainable and fails; a power rate counts the "
        "missing studies as not detecting",
        "source": "a bounded failure route; M26, decided",
    },
}
# M6b: the missing labels' decided rates and their disclosure, as the fixation freezes them
# (``missing_labels_record``) and its reader checks them against this code
MISSING_LABELS: dict[str, dict[str, Any]] = {
    "coverage": {
        "missing": "MCAR",
        "missing_rate": COVERAGE_MISSING_RATE,
        "scenarios": "every coverage scenario but C14, whose 5 % are MNAR among harmful responses",
        "use": "stress values: the coverage test reads the primary interval alone (M24)",
        "source": "no record holds study 3's rate (study 2's labels were never made); M6b, decided",
    },
    "power": {
        "missing": "MCAR",
        "missing_rate": POWER_MISSING_RATE,
        "scenarios": "every power scenario: the power gate reads them (M20)",
        "definition": "the share of labels still unresolved after the labeling rounds the "
        "registration allows (docs/study3-preregistration.md, 'Judge and audit': up to three "
        "rounds per run)",
        "status": "a conservative planning assumption for supported runs, not a proven upper "
        "bound: the bound from 0 in 39 900 assumes independent labels and is computed on a "
        "selected subset, the non-collapsed runs",
        "source": "study 1 with the same judge, WildGuard: 0 unparseable outputs in 39 900 labels "
        "of its non-collapsed runs (docs/reports/stage-1.md, 'Judge labeling'); study 1's rate "
        "over all its runs includes its collapsed runs, which are not representative, and is "
        "not used; M6b, decided",
        "limitation": "missing completely at random. Run- or arm-level clustered label "
        "failures are not explicitly modeled. The stress scenarios cover higher MCAR "
        "missingness and harmful-response-dependent MNAR. They are 1 % MCAR in the coverage "
        "scenarios and 5 % MNAR among harmful responses in C14",
        "decision": "the researcher's, after preliminary calculations showed the power gate "
        "failing at 1 % MCAR; it changes the planning alternative of the power requirement "
        "(docs/study3-preregistration.md, 'Simulation before validation'); the registration "
        "sets no rate, and the required simulation itself stays prospective",
    },
    "report": {
        "missing": "MCAR",
        "missing_rate": REPORT_MISSING_RATE,
        "comparison": "A-F",
        "role": "reported beside power_af, not a gate: it has no requirement, and the n_min scan "
        "does not read it; it uses the N the rule chose, before any n_min increase, and is not "
        "the power of the final increased design",
        "studies": "each power study of the screen, at every level; at a level, the studies "
        "power_af counts there (the first M confirmed by index), paired; the results give it at "
        "power_af's level",
        "draws": "each study's own streams (the screen's calibration base, the scenario's name "
        "and the study's index) and its power scenario's frozen calibration: the same search, "
        "validation, confirmation, label and bootstrap draws; under MCAR, delta scales by "
        "0.99 / 0.999 (+2 or -2 pp becomes about +1.982 or -1.982 pp), and the SD need not "
        "shrink by exactly 1 %",
        "coupling": "one uniform per label on the study's label stream, the label missing when "
        "its uniform is below the rate: the 1 % mask contains the 0.1 % mask",
        "reading": "the registered procedure at the denser mask: its own validation spread, N, "
        "cap and confirmation, and the cap's final direction common to the three computations "
        "of 'Judge and audit' (study3_estimates.three_way_reading); a study detects when that "
        "reading is confirmatory in the direction of delta, and one the denser mask leaves "
        "unconfirmed does not",
        "source": "how the power falls if labels go missing ten times as often as planned; M6b, "
        "decided",
    },
}
CURVATURE_SOURCE = (
    "study 1's fitted curves (study2_short_sim.STUDY1) at K*, baseline 0.15; M4, decided"
)
CURVATURE_NOTE = (
    "the registration states about 0.06 pp, a fitted estimate with partial cancellation between "
    "arms"
)


def curvature_record() -> dict[str, Any]:
    """The base κ of each arm (pp per (ln K)², read by the curvature sensitivity) and the
    nominal A–F bias at the symmetric bracket, with M4's disclosure, rounded to 4 decimals."""
    return {
        **{arm: round(CURVATURE_PP[arm], 4) for arm in ARM_ORDER},
        "unit": "pp per (ln K)^2",
        "nominal_bias_af_pp": round(NOMINAL_BIAS_AF_PP, 4),
        "source": CURVATURE_SOURCE,
        "note": CURVATURE_NOTE,
    }


def missing_labels_record() -> dict[str, Any]:
    """The missing labels' decided rates with their disclosure (M6b), as the fixation records
    them: the coverage scenarios' stress value and the power scenarios' planning rate, each with
    its source, the planning rate's definition, status and limitation, and the decision; and the
    procedure of the power reported at 1 % (``report``: its rate, studies, draws, coupling and
    reading)."""
    return json.loads(json.dumps(MISSING_LABELS))


def calibration_record() -> dict[str, Any]:
    """The calibration's size and tolerances, as the fixation freezes them and ``read_fixation``
    checks them against this code: ``studies`` (``CALIBRATION_STUDIES``, the selections per
    pair), ``blocks`` (``CALIBRATION_BLOCKS``), ``rounds`` (``CALIBRATION_ROUNDS``, the bound on
    each alternation) and the solvers' ``tolerances``: ``support`` (f's, on the support),
    ``sigma`` and ``sigma_relative`` (Brent's on σ) and ``offset`` (the offsets', on the mean
    truth)."""
    return {
        "studies": _decisions.CALIBRATION_STUDIES,
        "blocks": CALIBRATION_BLOCKS,
        "rounds": CALIBRATION_ROUNDS,
        "tolerances": {"support": SUPPORT_TOLERANCE, "sigma": SIGMA_PRECISION,
                       "sigma_relative": SIGMA_RTOL, "offset": OFFSET_PRECISION},
    }  # fmt: skip


def machine_record() -> dict[str, str]:
    """The machine a fixation is calibrated on (reported, never refused on): the platform, the
    CPU model and numpy's version."""
    import platform

    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"platform": platform.platform(), "cpu": cpu or platform.processor() or "unknown",
            "numpy": np.__version__}  # fmt: skip


def calibration_json(c: Calibration) -> dict[str, Any]:
    """A calibration as JSON (vector seeds as strings, pairs as lists)."""
    return {
        "candidate": c.candidate,
        "failure": c.failure,
        "realized_support": c.realized_support,
        "support_flagged": c.support_flagged,
        "mmlu_shift": {
            arm: {str(seed): v for seed, v in d.items()} for arm, d in c.mmlu_shift.items()
        },  # fmt: skip
        "factual_shift": dict(c.factual_shift),
        "utility_realized": {arm: list(v) for arm, v in c.utility_realized.items()},
        "utility_flagged": c.utility_flagged,
        "utility_lambda": dict(c.utility_lambda),
        "sigma_af": c.sigma_af,
        "sigma_pf": c.sigma_pf,
        "offset_a": c.offset_a,
        "offset_p": c.offset_p,
        "null_offset_a": c.null_offset_a,
        "null_offset_p": c.null_offset_p,
        "attained_af_pp": c.attained_af_pp,
        "attained_pf_pp": c.attained_pf_pp,
        "attained_spread_se_pp": dict(c.attained_spread_se_pp),
        "attained_delta_pp": {k: list(v) for k, v in c.attained_delta_pp.items()},
    }


def calibration_from_json(d: Mapping[str, Any]) -> Calibration:
    """The calibration of ``calibration_json``. ``ValueError`` for another field."""
    names = {f.name for f in fields(Calibration)}
    if not isinstance(d, Mapping) or set(d) != names:
        raise ValueError(f"a calibration holds exactly {sorted(names)}")
    return Calibration(
        candidate=d["candidate"],
        failure=float(d["failure"]),
        realized_support=float(d["realized_support"]),
        support_flagged=d["support_flagged"],
        mmlu_shift={
            arm: {int(seed): float(v) for seed, v in shifts.items()}
            for arm, shifts in d["mmlu_shift"].items()
        },  # fmt: skip
        factual_shift={arm: float(v) for arm, v in d["factual_shift"].items()},
        utility_realized={
            arm: tuple(float(x) for x in v) for arm, v in d["utility_realized"].items()
        },  # fmt: skip
        utility_flagged=d["utility_flagged"],
        utility_lambda={arm: float(v) for arm, v in d["utility_lambda"].items()},
        sigma_af=float(d["sigma_af"]),
        sigma_pf=float(d["sigma_pf"]),
        offset_a=float(d["offset_a"]),
        offset_p=float(d["offset_p"]),
        null_offset_a=float(d["null_offset_a"]),
        null_offset_p=float(d["null_offset_p"]),
        attained_af_pp=float(d["attained_af_pp"]),
        attained_pf_pp=float(d["attained_pf_pp"]),
        attained_spread_se_pp={k: float(v) for k, v in d["attained_spread_se_pp"].items()},
        attained_delta_pp={
            k: (float(v[0]), float(v[1])) for k, v in d["attained_delta_pp"].items()
        },  # fmt: skip
    )


def build_fixation(
    vectors_path: str | Path,
    scenarios: Sequence[Scenario],
    calibrations: Mapping[tuple[Scenario, int], Calibration],
    *,
    code: Mapping[str, str],
    streams: Mapping[str, int],
    repo: str | Path | None = None,
    policy: Mapping[str, Any] | None = None,
    written_at: str | None = None,
    machine: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The simulation's fixation (``docs/study3-simulation-fixation.json``, committed alone before
    the first draw; the closed schema ``study3_decisions.fixation_problems`` reads): the code;
    the ``machine`` it is calibrated on (``machine_record``, reported: ``run`` names a different
    one and goes on); the registration's ``policy`` (``study3_versions.registered_policy`` at
    HEAD of ``repo``, unless given); the inputs record (``vectors_path``, in the checkout) by its
    sha256; M, the resamples, the calibration's size and tolerances (``calibration_record``),
    the ``streams``, the levels, N points and p₀, the gate replicates (M) and the draw bound;
    the procedure's ``rules``, the base ``curvature`` and the missing labels' decided rates
    with their disclosure (``missing_labels_record``); every scenario with its calibrations
    per candidate (a coverage scenario of candidate 1 also at candidate 2, the verification's
    fallback); the coverage tests' explicit identities, their count and sha256, J, the
    thresholds and the detection powers."""
    import datetime

    from ..paths import REPO_ROOT
    from .study3_decisions import COVERAGE_P0, COVERAGE_RUNS

    repo = Path(REPO_ROOT if repo is None else repo)
    if policy is None:
        from .study3_versions import registered_policy

        policy = registered_policy(repo)
    vectors = Path(vectors_path)
    full = vectors if vectors.is_absolute() else repo / vectors
    try:
        relative = full.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{vectors_path}: the inputs record is not in the checkout") from exc
    entries = []
    for sc in scenarios:
        mine = {c: cal for (s, c), cal in calibrations.items() if s == sc}
        if sc.candidate not in mine:
            raise ValueError(f"scenario {sc.name}: no calibration at its candidate")
        if sc.coverage and sc.candidate == 1 and 2 not in mine:
            raise ValueError(f"scenario {sc.name}: no calibration at candidate 2, the "
                             "verification's fallback")  # fmt: skip
        calibrated = {str(c): calibration_json(mine[c]) for c in sorted(mine)}
        entries.append({**sc.to_json(), "calibration": calibrated})
    coverage = [sc.name for sc in scenarios if sc.coverage]
    tests = test_identities(coverage)
    j, m = len(tests), COVERAGE_RUNS
    return {
        "kind": FIXATION_KIND,
        "written_at": written_at
        or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "code": dict(code),
        "machine": dict(machine_record() if machine is None else machine),
        "policy": dict(policy),
        "inputs": {"path": relative, "sha256": hashlib.sha256(full.read_bytes()).hexdigest()},
        "M": m,
        "resamples": {
            "screen": _decisions.SCREEN_RESAMPLES,
            "verification": _decisions.VERIFICATION_RESAMPLES,
        },  # fmt: skip
        "calibration": calibration_record(),
        "streams": {
            "calibration": int(streams["calibration"]),
            "verification": int(streams["verification"]),
        },  # fmt: skip
        "levels": list(LEVELS),
        "n_points": list(N_POINTS),
        "p0": dict(COVERAGE_P0),
        "gate_replicates": m,
        "max_draws_factor": MAX_DRAWS_FACTOR,
        "rules": {name: dict(rule) for name, rule in RULES.items()},
        "curvature": curvature_record(),
        "missing_labels": missing_labels_record(),
        "scenarios": entries,
        "tests": {
            "list": [list(i) for i in tests],
            "count": j,
            "sha256": identities_sha256(tests),
        },  # fmt: skip
        "J": j,
        "thresholds": {name: coverage_threshold(j, m, p0) for name, p0 in COVERAGE_P0.items()},
        "detection_power": detection_power(j, m),
    }


def read_fixation(path: str | Path, repo: str | Path) -> dict[str, Any]:
    """The committed fixation at ``path`` in the checkout ``repo`` (``SIMULATION_FIXATION``),
    as the J check reads it (``study3_decisions.fixation_problems``): committed once, alone and
    unchanged at HEAD (so in HEAD's history: HEAD or an ancestor of it holds it), its schema
    closed, its sizes the registered ones (the resamples and the calibration's size and
    tolerances, ``calibration_record``, included), its scenarios the whole grid (``SCENARIOS``:
    every scenario, none extra, in its order, each equal; the power scenarios at the planning
    rate), its missing labels the decided ones (``missing_labels_record``) and its tests exactly
    its coverage scenarios' identities. Returns ``fixation``, ``commit``, ``sha256``, the
    ``scenarios`` (``Scenario``) and ``calibrations`` ((scenario, candidate) →
    ``Calibration``). ``ValueError`` naming every problem."""
    repo = Path(repo)
    full = Path(path) if Path(path).is_absolute() else repo / path
    try:
        relative = full.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path}: the fixation is not in the checkout {repo}") from exc
    if relative != SIMULATION_FIXATION:
        raise ValueError(f"{path}: the simulation's fixation is {SIMULATION_FIXATION}")
    fixation, commit, problems = fixation_problems(repo, relative)
    if fixation is None or problems:
        raise ValueError("; ".join(problems))
    scenarios, calibrations = [], {}
    for entry in fixation["scenarios"]:
        sc = Scenario.from_json({k: v for k, v in entry.items() if k != "calibration"})
        scenarios.append(sc)
        for candidate, value in entry["calibration"].items():
            calibrations[(sc, int(candidate))] = calibration_from_json(value)
    return {
        "fixation": fixation,
        "commit": commit,
        "sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
        "scenarios": scenarios,
        "calibrations": calibrations,
    }


# ------------------------------------------------------------------ the engine

UNIT = 250  # studies per part
SIMULATION_DIR = (EXPERIMENT, SIMULATION_FOLDER)  # RUNS_DIR/study3/simulation/, reserved
PART_KIND, SCREEN_KIND, VERIFICATION_KIND = (
    "simulation-part",
    "simulation-screen",
    "simulation-verification",
)
RESULTS_KIND = "simulation-results"
PHASES = ("gates", "screen", "verification")  # and "scan-<n>", one per n of the n_min scan
CALIBRATION_TOLERANCE = 1e-9  # the recomputed calibration's floats, relative
# ... or within an absolute floor, the solvers' own precision on the field's scale, so that a
# result near 0 (Δ = 0's attained truth, an offset o⁰ near 0) compares by its precision
PP_FLOOR = 1e-9  # a pp field: utility shifts and losses, attained spreads, Δs and their SEs
PROPORTION_FLOOR = 1e-11  # a proportion: f, the support, σ, the offsets o and o⁰
LAMBDA_FLOOR = 1e-10  # λ scales deviations of a few pp into the shifts: within the pp floor
CALIBRATION_FLOORS: dict[str, float] = {
    "failure": PROPORTION_FLOOR,
    "realized_support": PROPORTION_FLOOR,
    "mmlu_shift": PP_FLOOR,
    "factual_shift": PP_FLOOR,
    "utility_realized": PP_FLOOR,
    "utility_lambda": LAMBDA_FLOOR,
    "sigma_af": PROPORTION_FLOOR,
    "sigma_pf": PROPORTION_FLOOR,
    "offset_a": PROPORTION_FLOOR,
    "offset_p": PROPORTION_FLOOR,
    "null_offset_a": PROPORTION_FLOOR,
    "null_offset_p": PROPORTION_FLOOR,
    "attained_af_pp": PP_FLOOR,
    "attained_pf_pp": PP_FLOOR,
    "attained_spread_se_pp": PP_FLOOR,
    "attained_delta_pp": PP_FLOOR,
}
QUANTILES = {"q10": Fraction(1, 10), "q50": Fraction(1, 2), "q90": Fraction(9, 10)}
_SCAN = re.compile(r"scan-([1-9][0-9]*)")
_PART_NAME = re.compile(r"[0-9]{6}\.json")
_NAME = re.compile(r"[A-Za-z0-9_-]+")


def simulation_dir(runs_root: str | Path) -> Path:
    return Path(runs_root).joinpath(*SIMULATION_DIR)


def screen_path(runs_root: str | Path) -> Path:
    return simulation_dir(runs_root) / "screen.json"


def verification_path(runs_root: str | Path) -> Path:
    return simulation_dir(runs_root) / "verification.json"


def results_path(runs_root: str | Path) -> Path:
    return Path(runs_root) / SIMULATION_RESULTS


def _phase_ok(phase: Any) -> bool:
    if phase in PHASES:
        return True
    found = _SCAN.fullmatch(phase) if isinstance(phase, str) else None
    return found is not None and int(found.group(1)) >= MIN_N


def part_path(runs_root: str | Path, phase: str, scenario: str, first: int) -> Path:
    """The part of ``scenario``'s studies from index ``first`` in ``phase`` (``gates``,
    ``screen``, ``scan-<n>`` or ``verification``): ``parts/<phase>/<scenario>/<first>.json``
    under the simulation's folder."""
    if not _phase_ok(phase):
        raise ValueError(f"no phase {phase!r} (phases: {PHASES} and scan-<n>, n ≥ {MIN_N})")
    if not (isinstance(scenario, str) and _NAME.fullmatch(scenario)):
        raise ValueError(f"scenario {scenario!r} is no folder name")
    if not (_integer(first) and 0 <= first < 10**6):
        raise ValueError(f"first {first!r} is not an index")
    return simulation_dir(runs_root) / "parts" / phase / scenario / f"{first:06d}.json"


@dataclass(frozen=True)
class Unit:
    """One part's work: ``count`` studies of ``scenario`` from index ``first`` in ``phase``, on
    the stream ``base``, with ``resamples`` per interval, at ``levels``, a validation of
    ``validation_blocks`` blocks (the gates: 28, with 40 reported beside it) and N at least
    ``n_floor`` (the scan's n; the verification's ``n_min`` when the power was short)."""

    phase: str
    scenario: str
    first: int
    count: int
    base: int
    resamples: int
    levels: tuple[float, ...]
    validation_blocks: int
    n_floor: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(float(lv) for lv in self.levels))
        part_path(".", self.phase, self.scenario, self.first)  # validates the three
        problems = []
        if not (_integer(self.count) and self.count >= 1):
            problems.append(f"count {self.count!r}")
        if not (_integer(self.base) and self.base >= 0):
            problems.append(f"base {self.base!r}")
        if not (_integer(self.resamples) and self.resamples >= 1):
            problems.append(f"resamples {self.resamples!r}")
        if not self.levels or not set(self.levels) <= set(LEVELS):
            problems.append(f"levels {self.levels!r}")
        if self.validation_blocks not in VALIDATION_BLOCKS:
            problems.append(f"validation_blocks {self.validation_blocks!r}")
        if not (_integer(self.n_floor) and self.n_floor >= MIN_N):
            problems.append(f"n_floor {self.n_floor!r}")
        if problems:
            raise ValueError(f"a unit's {', '.join(problems)}")

    @property
    def mode(self) -> str:
        return "procedure" if self.phase == "verification" else "screen"


def header(unit: Unit, fixation_ref: Mapping[str, str], commit: str) -> dict[str, Any]:
    """A part's header: what it holds, the stream and sizes it was drawn with, the fixation
    (``path``, ``sha256``, ``commit``) and the commit the run ran at."""
    return {
        "kind": PART_KIND,
        "phase": unit.phase,
        "scenario": unit.scenario,
        "first": unit.first,
        "count": unit.count,
        "base": unit.base,
        "mode": unit.mode,
        "resamples": unit.resamples,
        "levels": list(unit.levels),
        "validation_blocks": unit.validation_blocks,
        "n_floor": unit.n_floor,
        "fixation": dict(fixation_ref),
        "commit": commit,
    }


@dataclass(frozen=True)
class Context:
    """What every study of a run needs from its fixation: the fixation itself (its content),
    its reference ``ref`` (``path``, ``sha256``, ``commit``), the ``setup`` (the inputs'
    vectors and slopes, the frozen calibrations, M) and the ``scenarios``, in its order."""

    fixation: Mapping[str, Any]
    ref: Mapping[str, str]
    setup: Setup
    scenarios: tuple[Scenario, ...]

    def scenario(self, name: str) -> Scenario:
        found = [sc for sc in self.scenarios if sc.name == name]
        if len(found) != 1:
            raise SimulationError(f"the fixation has no scenario {name!r}")
        return found[0]

    def of(self, *roles: str) -> list[Scenario]:
        return [sc for sc in self.scenarios if sc.role in roles]

    def calibration(self, scenario: Scenario) -> Calibration:
        return self.setup.calibration(scenario, scenario.candidate)


def fixation_contents(
    fixation: Mapping[str, Any],
) -> tuple[list[Scenario], dict[tuple[Scenario, int], Calibration]]:
    """The fixation's scenarios (in its order) and calibrations ((scenario, candidate))."""
    scenarios, calibrations = [], {}
    for entry in fixation["scenarios"]:
        sc = Scenario.from_json({k: v for k, v in entry.items() if k != "calibration"})
        scenarios.append(sc)
        for candidate, value in entry["calibration"].items():
            calibrations[(sc, int(candidate))] = calibration_from_json(value)
    return scenarios, calibrations


def context_of(
    fixation: Mapping[str, Any],
    ref: Mapping[str, str],
    vectors: Sequence[Vector],
    slopes: Mapping[str, tuple[float, float]],
) -> Context:
    """The run's context from the fixation's content, its reference and the inputs' vectors."""
    scenarios, calibrations = fixation_contents(fixation)
    setup = Setup(vectors=tuple(vectors), slopes=slopes, calibrations=calibrations,
                  M=_decisions.COVERAGE_RUNS, levels=LEVELS,
                  validation_blocks=VALIDATION_BLOCKS[0])  # fmt: skip
    return Context(dict(fixation), dict(ref), setup, tuple(scenarios))


def load_context(repo: str | Path) -> Context:
    """The committed fixation of the checkout ``repo`` (``read_fixation``: committed once,
    alone and unchanged, in HEAD's history) with its inputs record, whose sha256 must be the
    one it pinned. ``ValueError`` naming the problem."""
    from .study3_seed_vectors import read_inputs

    repo = Path(repo)
    found = read_fixation(SIMULATION_FIXATION, repo)
    fixation = found["fixation"]
    pinned = fixation["inputs"]
    vectors, slopes, sha = read_inputs(repo / pinned["path"])
    if sha != pinned["sha256"]:
        raise ValueError(f"{pinned['path']} has sha256 {sha}, not the fixation's "
                         f"{pinned['sha256']}")  # fmt: skip
    ref = {"path": SIMULATION_FIXATION, "sha256": found["sha256"], "commit": found["commit"]}
    return context_of(fixation, ref, vectors, slopes)


# -- the workers

_WORKER: dict[str, Any] = {}  # in a worker (or this process, for one worker): the context


def _install(context: Context | None, moments: Mapping[tuple, tuple[float, float]] | None) -> None:
    """A worker's start: the run's context, and the truths' moments known already."""
    _WORKER.clear()
    _WORKER["context"] = context
    if moments:
        _MOMENTS.update(moments)


class Workers:
    """``workers`` processes (spawn, single-threaded BLAS: the parallelism is the processes'),
    each holding ``context`` and the ``moments`` given; for one worker, this process. ``imap``
    runs a task per item in any order of completion, ``map`` in order: nothing a task returns
    depends on which process ran it."""

    def __init__(self, workers: int, context: Context | None = None,
                 moments: Mapping[tuple, tuple[float, float]] | None = None) -> None:  # fmt: skip
        if not (_integer(workers) and workers >= 1):
            raise ValueError(f"workers {workers!r} is not a positive integer")
        self.workers, self.context, self.moments = int(workers), context, moments
        self._pool = None
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> Workers:
        if self.workers == 1:
            self._saved = dict(_WORKER)
            _install(self.context, self.moments)
            return self
        import multiprocessing as mp
        import os

        saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        os.environ.update(dict.fromkeys(THREAD_VARIABLES, "1"))
        try:
            self._pool = mp.get_context("spawn").Pool(
                self.workers, initializer=_install, initargs=(self.context, self.moments)
            )
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        return self

    def __exit__(self, kind, value, traceback) -> None:
        if self._pool is None:
            _WORKER.clear()
            _WORKER.update(self._saved)
            return
        if kind is None:
            self._pool.close()
        else:
            self._pool.terminate()
        self._pool.join()
        self._pool = None

    def imap(self, fn: Callable, items: Sequence) -> Iterable:
        if self._pool is None:
            return map(fn, items)
        return self._pool.imap_unordered(fn, items)

    def map(self, fn: Callable, items: Sequence) -> list:
        if self._pool is None:
            return list(map(fn, items))
        chunk = max(1, len(items) // (4 * self.workers))
        return self._pool.map(fn, items, chunksize=chunk)


def _json_bytes(data: Any) -> bytes:
    return (json.dumps(data, allow_nan=False, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")  # fmt: skip


def _write_once(path: Path, data: Any) -> None:
    """``data`` as JSON at ``path``, whole and once (``study3_decisions._publish``)."""
    from .study3_decisions import DecisionRefused, _publish

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _publish(path, _json_bytes(data))
    except DecisionRefused as exc:
        raise SimulationError(f"{path} exists: a simulation file is written once") from exc


# -- the parts


def _gate_row(scenario: Scenario, setup: Setup, unit: Unit, index: int) -> dict[str, Any]:
    """A gate study: its validation's A–F and P–F gates and its A–F-supported blocks, at 28
    blocks (the gate) and at 40 (reported), on one selection."""
    row: dict[str, Any] = {"index": index}
    for size in VALIDATION_BLOCKS:
        o = simulate_study(scenario, replace(setup, validation_blocks=size), unit.base, index,
                           resamples=unit.resamples, levels=unit.levels,
                           validation_only=True)  # fmt: skip
        row[str(size)] = {"AF": bool(o.gates["AF"]), "PF": bool(o.gates["PF"]), "m": o.m}
    return row


def run_unit(args: tuple[Unit, Path, Mapping[str, str], str]) -> Path:
    """The worker's entry: the ``count`` studies of the unit, each a pure function of (stream,
    scenario, index), published once as a part with its header; in the screen a power study
    carries its ``power_report`` (M6b). The worker holds the run's context (``Workers``), which
    must be the unit's fixation's."""
    unit, runs_root, fixation_ref, commit = args
    context = _WORKER.get("context")
    if context is None or dict(context.ref) != dict(fixation_ref):
        raise SimulationError("the worker holds no context of the unit's fixation")
    scenario = context.scenario(unit.scenario)
    setup = replace(context.setup, validation_blocks=unit.validation_blocks)
    rows = []
    for index in range(unit.first, unit.first + unit.count):
        if unit.phase == "gates":
            rows.append(_gate_row(scenario, setup, unit, index))
            continue
        o = simulate_study(scenario, setup, unit.base, index, mode=unit.mode,
                           resamples=unit.resamples, levels=unit.levels,
                           n_floor=unit.n_floor)  # fmt: skip
        if unit.phase == "screen" and scenario.power:
            o = replace(o, report=power_report(scenario, setup, unit.base, index,
                                               resamples=unit.resamples,
                                               levels=unit.levels))  # fmt: skip
        rows.append(outcome_json(o))
    path = part_path(runs_root, unit.phase, unit.scenario, unit.first)
    _write_once(path, {"header": header(unit, fixation_ref, commit), "rows": rows})
    return path


def _read_part(path: Path) -> tuple[dict[str, Any], list[Any]]:
    """A part's header and rows (``ValueError`` for a file that is not one)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable ({type(exc).__name__})") from exc
    if not (isinstance(data, dict) and set(data) == {"header", "rows"}):
        raise ValueError("not a header and rows")
    head, rows = data["header"], data["rows"]
    names = set(header(Unit("gates", "x", 0, 1, 0, 1, LEVELS, 28, MIN_N), {}, ""))
    if not (isinstance(head, dict) and set(head) == names and head["kind"] == PART_KIND):
        raise ValueError("its header is not a part's")
    if not (isinstance(rows, list) and len(rows) == head["count"] and all(
            isinstance(r, dict) and r.get("index") == head["first"] + i
            for i, r in enumerate(rows))):  # fmt: skip
        raise ValueError("its rows are not its studies, in order")
    return head, rows


def existing_parts(
    runs_root: str | Path, phase: str, fixation_ref: Mapping[str, str], commit: str
) -> tuple[dict[tuple[str, int], Path], list[str]]:
    """The parts of ``phase`` under the runs directory, by (scenario, first), and the problems:
    a file that is no part, a part that is malformed or names another place, or one of another
    fixation or another commit (each named)."""
    folder = simulation_dir(runs_root) / "parts" / phase
    found: dict[tuple[str, int], Path] = {}
    problems: list[str] = []
    if not folder.exists():
        return found, problems
    for path in sorted(folder.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue  # a folder, or a part being written (``_publish``'s temporary file)
        place = path.relative_to(folder).parts
        if len(place) != 2 or not _PART_NAME.fullmatch(place[1]):
            problems.append(f"{path} is not a part of the simulation")
            continue
        try:
            head, _ = _read_part(path)
        except ValueError as exc:
            problems.append(f"{path} is a malformed part ({exc})")
            continue
        if (head["phase"], head["scenario"], head["first"]) != (phase, place[0],
                                                                int(path.stem)):  # fmt: skip
            problems.append(f"{path} is a malformed part (its header names another place)")
        elif head["fixation"] != dict(fixation_ref):
            problems.append(
                f"{path} is a part of another fixation "
                f"({head['fixation'].get('sha256')}, not {fixation_ref['sha256']})"
            )
        elif head["commit"] != commit:
            problems.append(f"{path} is a part of another commit ({head['commit']}, not "
                            f"{commit})")  # fmt: skip
        else:
            found[(place[0], head["first"])] = path
    return found, problems


def part_problems(runs_root: str | Path, fixation_ref: Mapping[str, str], commit: str) -> list[str]:
    """``existing_parts``' problems over every phase's folder (a folder that is no phase
    included)."""
    folder = simulation_dir(runs_root) / "parts"
    problems: list[str] = []
    if not folder.exists():
        return problems
    for phase in sorted(p.name for p in folder.iterdir() if not p.name.startswith(".")):
        if not _phase_ok(phase) or not (folder / phase).is_dir():
            problems.append(f"{folder / phase} is no phase of the simulation")
            continue
        problems += existing_parts(runs_root, phase, fixation_ref, commit)[1]
    return problems


def execute(
    units: Sequence[Unit],
    runs_root: str | Path,
    fixation_ref: Mapping[str, str],
    commit: str,
    *,
    workers: int = 1,
    context: Context | None = None,
    pool: Workers | None = None,
) -> None:
    """The units' parts, each written once: those already written with the unit's header are
    kept (a resumed run), one with another header is refused by name. ``pool``: the run's
    ``Workers``; otherwise ``workers`` processes holding ``context``. The order in which the
    parts finish does not matter."""
    runs_root = Path(runs_root)
    todo = []
    for unit in units:
        path = part_path(runs_root, unit.phase, unit.scenario, unit.first)
        if not path.exists():
            todo.append(unit)
            continue
        try:
            head, _ = _read_part(path)
        except ValueError as exc:
            raise SimulationError(f"{path} is a malformed part ({exc})") from exc
        if head != header(unit, fixation_ref, commit):
            raise SimulationError(f"{path} is a part of another run: its header is not the "
                                  "unit's")  # fmt: skip
    if not todo:
        return
    tasks = [(unit, runs_root, dict(fixation_ref), commit) for unit in todo]
    if pool is None:
        with Workers(workers, context) as own:
            for _ in own.imap(run_unit, tasks):
                pass
        return
    for _ in pool.imap(run_unit, tasks):
        pass


# -- the scheduling: per scenario, parts in index order, never past the bound


def _prefix(runs_root: Path, context: Context, commit: str, unit: Unit, bound: int,
            cache: dict[Path, list]) -> list:  # fmt: skip
    """The rows of the scenario's contiguous finished parts from index 0 (as read back from
    them), each part's header the unit's at its place."""
    rows: list = []
    first = 0
    while first < bound:
        here = replace(unit, first=first, count=min(UNIT, bound - first))
        path = part_path(runs_root, here.phase, here.scenario, first)
        if path not in cache:
            if not path.exists():
                break
            try:
                head, raw = _read_part(path)
            except ValueError as exc:
                raise SimulationError(f"{path} is a malformed part ({exc})") from exc
            if head != header(here, context.ref, commit):
                raise SimulationError(f"{path} is a part of another run: its header is not "
                                      "the unit's at its place")  # fmt: skip
            cache[path] = raw if here.phase == "gates" else [outcome_from_json(r) for r in raw]
        rows += cache[path]
        first += here.count
    return rows


def _phase(pool: Workers | None, runs_root: Path, context: Context, commit: str,
           units: Mapping[str, Unit], bound: int,
           short: Callable[[str, list], int]) -> dict[str, list]:  # fmt: skip
    """The rows of each scenario's studies in one phase: parts issued in index order while the
    contiguous finished prefix leaves some test short (``short``: the most studies a test of the
    scenario still needs), never past ``bound``. Without ``pool`` nothing is drawn: a phase
    still short below its bound is unfinished. The rows are read back from the parts."""
    cache: dict[Path, list] = {}
    while True:
        rows, todo = {}, []
        for name, template in units.items():
            rows[name] = found = _prefix(runs_root, context, commit, template, bound, cache)
            need = short(name, found)
            if need <= 0 or len(found) >= bound:
                continue
            for first in range(len(found), min(len(found) + need, bound), UNIT):
                path = part_path(runs_root, template.phase, name, first)
                if not path.exists():
                    todo.append(replace(template, first=first, count=min(UNIT, bound - first)))
        if not todo:
            return rows
        if pool is None:
            raise SimulationError(f"the {next(iter(units.values())).phase} is unfinished: "
                                  f"{len(todo)} parts are missing")  # fmt: skip
        execute(todo, runs_root, context.ref, commit, pool=pool)


def first_confirmed(rows: Iterable[StudyOutcome], m: int, level: float) -> list[StudyOutcome]:
    """The first ``m`` studies by index confirmed at ``level`` (the A–F gate passed and the
    level's cap within the reserved seeds)."""
    return _first(rows, m, lambda o: bool(o.confirmed.get(level)))


def _first(rows: Iterable[StudyOutcome], m: int, keep: Callable[[StudyOutcome], bool]) -> list:
    out = []
    for o in sorted(rows, key=lambda o: o.index):
        if keep(o):
            out.append(o)
            if len(out) == m:
                break
    return out


def _drawn(used: Sequence, rows: Sequence, m: int, bound: int) -> int:
    """The studies drawn for a test: through its m-th study, or the bound for one short of m
    (which the rows must reach)."""
    if len(used) == m:
        return used[-1].index + 1
    if len(rows) < bound:
        raise SimulationError("a test short of M before the bound: the phase is unfinished")
    return bound


def order_statistic(values: Sequence[float], q: Fraction | float) -> float:
    """The type-1 order statistic at probability q: the value of zero-based rank ⌈n·q⌉ − 1 among
    the sorted values (``study3_inference.tail_ranks``' rule), in exact arithmetic."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no value")
    exact = q if isinstance(q, Fraction) else Fraction(str(q))  # 0.1 as written, not in binary
    rank = math.ceil(len(ordered) * exact) - 1
    return ordered[min(max(rank, 0), len(ordered) - 1)]


def _test_row(identity: Sequence, level: float, used: Sequence[StudyOutcome], draws: int,
              m: int, j: int) -> dict[str, Any]:  # fmt: skip
    """One test's row (results.json's "tests"): X the misses among its M studies."""
    _, _, point, _, comparison = identity
    p0 = _decisions.P0[comparison]
    c = coverage_threshold(j, m, p0)
    row = {"test": list(identity), "level": level, "due": True, "p0": p0, "studies": len(used),
           "draws": draws, "reason": None, "X": None, "c": c, "passed": False, "rate": None,
           "interval": None}  # fmt: skip
    if len(used) < m:
        row["reason"] = _decisions.NOT_ATTAINABLE
        return row
    key = (point, level, comparison)
    if any(o.covered.get(key) is None for o in used):
        raise SimulationError(f"test {list(identity)}: a study it counts has no coverage flag")
    x = sum(not o.covered[key] for o in used)
    row.update(X=x, passed=x <= c, rate=x / m, interval=_decisions.clopper_pearson(x, m))
    return row


def _not_due(identity: Sequence) -> dict[str, Any]:
    p0 = _decisions.P0[identity[4]]
    return {"test": list(identity), "level": None, "due": False, "p0": p0, "studies": None,
            "draws": None, "reason": None, "X": None, "c": None, "passed": None, "rate": None,
            "interval": None}  # fmt: skip


def selected_level(rows: Sequence[Mapping[str, Any]], j: int, m: int) -> float | None:
    """L, the first level whose calibration tests all pass, recomputed from each row's X (not
    attainable: X = M) and c = ``coverage_threshold(J, M, p₀)``; None when no level passes."""
    for level in LEVELS:
        mine = [r for r in rows if r["test"][0] == "calibration"
                and r["test"][3] == level_slot(level)]  # fmt: skip
        if mine and all(
            type(r.get("X")) is int and r.get("studies") == m
            and r["X"] <= coverage_threshold(j, m, _decisions.P0[r["test"][4]])
            for r in mine
        ):  # fmt: skip
            return level
    return None


def _direction_counts(rows: Sequence[StudyOutcome], scenario: Scenario, comparison: str,
                      level: float, m: int, bound: int) -> dict[str, int]:  # fmt: skip
    """A power scenario's correct-direction count at ``level`` over its first M confirmed
    studies, its studies and draws (M20). A study detects when its cap's reading is confirmatory
    and its registered final direction (``final_direction``: common to the three computations)
    is the sign of its Δ; a descriptive reading, no common direction or the wrong sign detects
    nothing, nor do the missing studies of a test short at the bound."""
    delta = scenario.delta_af_pp if comparison == "A-F" else scenario.delta_pf_pp
    want = "positive" if delta > 0 else "negative"
    used = first_confirmed(rows, m, level)
    correct = sum(o.readings.get((level, comparison)) == CONFIRMATORY
                  and o.direction.get((level, comparison)) == want for o in used)  # fmt: skip
    return {"correct": correct, "studies": len(used), "draws": _drawn(used, rows, m, bound)}


def _report_counts(rows: Sequence[StudyOutcome], scenario: Scenario, level: float, m: int,
                   bound: int) -> dict[str, int]:  # fmt: skip
    """The power at the denser mask (M6b, ``power_report``) at ``level``, over the studies
    ``_direction_counts`` counts there (paired): a study detects when its report's A–F reading is
    confirmatory and its final direction the sign of Δ; one the denser mask leaves unconfirmed
    detects nothing, nor do the missing studies of a test short at the bound."""
    want = "positive" if scenario.delta_af_pp > 0 else "negative"
    slot = level_slot(level)
    used = first_confirmed(rows, m, level)
    if any(o.report is None for o in used):
        raise SimulationError(f"{scenario.name}: a power study of the screen has no report")
    correct = sum(o.report["confirmed"][slot] is True
                  and o.report["readings"][slot] == CONFIRMATORY
                  and o.report["direction"][slot] == want for o in used)  # fmt: skip
    return {"correct": correct, "studies": len(used), "draws": _drawn(used, rows, m, bound)}


def _gate_counts(context: Context, rows: Mapping[str, list], m: int) -> tuple[dict, int]:
    """The gates' passes over their M studies (``safe_pass_rate``: the safe alternative's;
    ``empirical_pass_rate``: study 2's empirical point; each the A–F gate at 28 blocks) and the
    validation size: 28 when the pass rate at the safe alternative reaches 0.95, else 40."""
    out = {}
    for name, utility in (("safe_pass_rate", "safe"), ("empirical_pass_rate", "empirical")):
        found = [sc for sc in context.of("gate") if sc.utility == utility]
        if len(found) != 1:
            raise SimulationError(f"the fixation needs one gate scenario of the {utility} "
                                  "utility")  # fmt: skip
        studies = rows[found[0].name][:m]
        out[name] = {"passed": sum(r["28"]["AF"] for r in studies), "studies": m,
                     "validation_blocks": VALIDATION_BLOCKS[0]}  # fmt: skip
    rate = out["safe_pass_rate"]["passed"] / m
    required = _decisions.REQUIRED_GATES["safe_pass_rate"]
    return out, VALIDATION_BLOCKS[0] if rate >= required else VALIDATION_BLOCKS[1]


def _screen_short(m: int, levels: Sequence[float]) -> Callable[[str, list], int]:
    return lambda _, rows: max(m - sum(bool(o.confirmed.get(lv)) for o in rows) for lv in levels)


def _screen(runs_root: Path, context: Context, commit: str,
            pool: Workers | None) -> dict[str, Any]:  # fmt: skip
    """The gates, the screen and the n_min scan (§"Simulation before validation"), drawn
    through ``pool`` where parts are missing, and their summary from the parts read back
    (``screen.json``)."""
    fixation, d = context.fixation, _decisions
    m, j = d.COVERAGE_RUNS, fixation["J"]
    bound, base = d.MAX_DRAWS_FACTOR * m, fixation["streams"]["calibration"]

    def units(phase: str, scenarios: Sequence[Scenario], levels: Sequence[float], blocks: int,
              n_floor: int) -> dict[str, Unit]:  # fmt: skip
        return {sc.name: Unit(phase, sc.name, 0, 1, base, d.SCREEN_RESAMPLES, tuple(levels),
                              blocks, n_floor) for sc in scenarios}  # fmt: skip

    gate_rows = _phase(pool, runs_root, context, commit,
                       units("gates", context.of("gate"), LEVELS, 28, MIN_N), m,
                       lambda _, rows: m - len(rows))  # fmt: skip
    gates, blocks = _gate_counts(context, gate_rows, m)
    screened = context.of("coverage", "power")
    rows = _phase(pool, runs_root, context, commit,
                  units("screen", screened, LEVELS, blocks, MIN_N), bound,
                  _screen_short(m, LEVELS))  # fmt: skip
    tests = []
    for identity in fixation["tests"]["list"]:
        if identity[0] != "calibration":
            continue
        level = float(identity[3])
        used = first_confirmed(rows[identity[1]], m, level)
        tests.append(_test_row(identity, level, used, _drawn(used, rows[identity[1]], m, bound),
                               m, j))  # fmt: skip
    selected = selected_level(tests, j, m)
    power_scenarios = context.of("power")
    power = {c: {level_slot(lv): {sc.name: _direction_counts(rows[sc.name], sc, c, lv, m, bound)
                                  for sc in power_scenarios} for lv in LEVELS}
             for c in COMPARISONS}  # fmt: skip
    report = {level_slot(lv): {sc.name: _report_counts(rows[sc.name], sc, lv, m, bound)
                               for sc in power_scenarios} for lv in LEVELS}  # fmt: skip
    at = d.POWER_LEVEL if selected is None else selected
    scan = []
    by = power["A-F"][level_slot(at)]
    if min(c["correct"] for c in by.values()) / m < d.POWER_REQUIRED:
        reserved = len(SEEDS["confirmation"][1])
        n = MIN_N
        while True:
            found = _phase(pool, runs_root, context, commit,
                           units(f"scan-{n}", power_scenarios, (at,), blocks, n), bound,
                           _screen_short(m, (at,)))  # fmt: skip
            counts = {}
            for sc in power_scenarios:
                c = _direction_counts(found[sc.name], sc, "A-F", at, m, bound)
                counts[sc.name] = {"correct": c["correct"], "studies": c["studies"]}
            scan.append({"n": n, "by_scenario": counts})
            reached = min(c["correct"] for c in counts.values()) / m >= d.POWER_REQUIRED
            if reached or cap_of(n) > reserved:
                break
            n += 1
    reported = {sc.name: _reported_gate(context, sc, gate_rows[sc.name][:m], m)
                for sc in context.of("gate")}  # fmt: skip
    for sc in screened:
        reported[sc.name] = _reported_screen(context, sc, rows[sc.name], m, bound, blocks)
    return {
        "kind": SCREEN_KIND,
        "fixation": dict(context.ref),
        "commit": commit,
        "M": m,
        "J": j,
        "streams": dict(fixation["streams"]),
        "resamples": d.SCREEN_RESAMPLES,
        "validation_blocks": blocks,
        "level": selected,
        "tests": tests,
        "gates": {
            **gates,
            "power": power,
            "power_report": report,
            "n_scan": scan,
            "scan_level": at if scan else None,
        },  # fmt: skip
        "reported": reported,
    }


def _reported_gate(context: Context, scenario: Scenario, rows: Sequence[dict],
                   m: int) -> dict[str, Any]:  # fmt: skip
    """A gate scenario's descriptive figures: its 40-block pass rate, the P–F gate's rate, the
    realized A–F support (M7) and, for study 2's empirical point, its utility at K* (M11)."""
    cal = context.calibration(scenario)
    blocks = VALIDATION_BLOCKS[0]
    out = {
        "attempted": m,
        "pass_rate_40": sum(r["40"]["AF"] for r in rows) / m,
        "pf_gate_rate": sum(r["28"]["PF"] for r in rows) / m,
        "support": {"target": scenario.support,
                    "realized": sum(r["28"]["m"] for r in rows) / (m * blocks),
                    "failure": cal.failure, "flagged": cal.support_flagged},
    }  # fmt: skip
    if scenario.utility == "empirical":
        kstar_ln = math.log(CANDIDATES[scenario.candidate]["targets"]["star"])
        vectors, slopes = context.setup.vectors, context.setup.slopes
        out["utility_at_kstar"] = {
            arm: {name: float(np.mean([v.losses(arm, kstar_ln, slopes)[k] for v in vectors]))
                  for k, name in enumerate(("mmlu_loss_pp", "factual_loss_pp"))}
            for arm in ("F", "P")
        }  # fmt: skip
    return out


def _reported_screen(context: Context, scenario: Scenario, rows: Sequence[StudyOutcome], m: int,
                     bound: int, blocks: int) -> dict[str, Any]:  # fmt: skip
    """A screened scenario's descriptive figures over the studies its tests drew
    (``attempted``): confirmed per level, unconfirmed per reason (a study counted once per
    reason), and over each level's M confirmed studies N above 90, the blocks' type-1 quantiles
    and the utility claim's rate; the P–F gate's rate; the calibration's attained spreads; the
    truths' range (pp); the support (M7)."""
    cal = context.calibration(scenario)
    used = {lv: first_confirmed(rows, m, lv) for lv in LEVELS}
    attempted = max(_drawn(used[lv], rows, m, bound) for lv in LEVELS)
    seen = sorted(rows, key=lambda o: o.index)[:attempted]
    reasons = Counter(r for o in seen for r in sorted({w for w in o.why.values() if w}))
    truths = {c: [100 * o.truth[c] for o in seen if any(o.confirmed.values())
                  and o.truth.get(c) is not None] for c in COMPARISONS}  # fmt: skip

    def type1(values: list[int]) -> dict[str, int] | None:  # the order statistics q10, q50, q90
        return {k: order_statistic(values, q) for k, q in QUANTILES.items()} if values else None

    def share(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "attempted": attempted,
        "confirmed": {level_slot(lv): sum(bool(o.confirmed[lv]) for o in seen) for lv in LEVELS},
        "unconfirmed": dict(sorted(reasons.items())),
        "n_above_90": {level_slot(lv): sum(o.N[lv] > _decisions.BUDGET_ABOVE for o in used[lv])
                       for lv in LEVELS},  # fmt: skip
        "blocks": {level_slot(lv): type1([o.blocks[lv] for o in used[lv]]) for lv in LEVELS},
        "pf_gate_rate": sum(bool(o.gates["PF"]) for o in seen) / len(seen),
        "attained_sd_pp": {"A-F": cal.attained_af_pp, "P-F": cal.attained_pf_pp},
        "truth_range_pp": {c: [min(v), max(v)] if v else None for c, v in truths.items()},
        "utility_claim_rate": {level_slot(lv): share([o.utility_claim.get((lv, "A-F")) is True
                                                      for o in used[lv]]) for lv in LEVELS},
        "support": {"target": scenario.support,
                    "realized": sum(o.m for o in seen) / (len(seen) * blocks),
                    "failure": cal.failure, "flagged": cal.support_flagged},
    }  # fmt: skip


def _seeded_moments(context: Context) -> dict[tuple, tuple[float, float]]:
    """The truths' moments this process holds at a jitter scenario's frozen calibration (the
    calibration's recomputation leaves them): every worker starts with them."""
    wanted = set()
    for (sc, candidate), cal in context.setup.calibrations.items():
        if sc.jitter:
            kstar_ln = math.log(CANDIDATES[candidate]["targets"]["star"])
            for arm in ARM_ORDER:
                model = (sc.curvature_scale, sc.shape, sc.judge, sc.missing, sc.missing_rate)
                wanted.add((arm, *model, cal.sigma(arm), cal.offset(arm), kstar_ln))
    return {k: v for k, v in _MOMENTS.items()
            if k[5] and (k[2], *k[6:11], k[11], k[12], k[13]) in wanted}  # fmt: skip


def run_screen(context: Context, runs_root: str | Path, commit: str, *, workers: int) -> Path:
    """§"Simulation before validation", the run's steps 1–3: the gates (G1, G2: M validations
    each), the screen (C01–C17, P18–P21 at the validation size, four levels, 2 000 resamples,
    ``"screen"`` mode, the first M confirmed studies by index per test, never past 4·M draws)
    and the n_min scan when ``power_af`` is short; parts are written as they finish, a run
    interrupted resumes from them on the same streams, and ``screen.json`` is the summary of the
    parts read back, written once. ``SimulationError``: ``screen.json`` exists, or a part of
    another fixation or commit (named)."""
    runs_root = Path(runs_root)
    path = screen_path(runs_root)
    if path.exists():
        raise SimulationError(f"{path} exists: the screen is written once")
    problems = part_problems(runs_root, context.ref, commit)
    if problems:
        raise SimulationError("; ".join(problems))
    with Workers(workers, context, _seeded_moments(context)) as pool:
        _screen(runs_root, context, commit, pool)
    _write_once(path, _screen(runs_root, context, commit, None))
    return path


def screen_summary(runs_root: str | Path, context: Context, commit: str) -> dict[str, Any]:
    """``screen.json``'s content from the parts as they are (nothing drawn): the X of every
    calibration identity, the gates' counts, the powers at every level, the scan and the
    reported figures. ``SimulationError`` when a phase is unfinished."""
    return _screen(Path(runs_root), context, commit, None)


def read_screen(runs_root: str | Path, context: Context, commit: str | None = None) -> dict:
    """``screen.json`` of this fixation (and of ``commit``, when given)."""
    path = screen_path(runs_root)
    try:
        screen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SimulationError(f"{path} cannot be read ({type(exc).__name__}): the screen "
                              "runs first") from exc  # fmt: skip
    if not isinstance(screen, dict) or screen.get("kind") != SCREEN_KIND:
        raise SimulationError(f"{path} is not the screen's summary")
    if screen.get("fixation") != dict(context.ref):
        raise SimulationError(f"{path} is of another fixation ({screen.get('fixation')})")
    if commit is not None and screen.get("commit") != commit:
        raise SimulationError(f"{path} is of another commit ({screen.get('commit')}, not "
                              f"{commit})")  # fmt: skip
    return screen


def unattainable_n(screen: Mapping[str, Any]) -> int | None:
    """The n at which the screen's n_min scan stopped with its cap beyond the reserved
    confirmation seeds (``cap_of(n)`` > 200): no N within them reaches the power, so none is
    attainable, the verification is not run and the record fails (``NO_ATTAINABLE_N``, the
    design stop). None when the scan is empty or stopped at a power reached within them."""
    scan = screen["gates"]["n_scan"]
    if scan and cap_of(scan[-1]["n"]) > len(SEEDS["confirmation"][1]):
        return scan[-1]["n"]
    return None


def no_attainable_n(n: int) -> str:
    """Why the verification is not run when the scan stopped at ``n`` (``unattainable_n``)."""
    return (f"the n_min scan stopped at n = {n}, whose cap {cap_of(n)} exceeds the "
            f"{len(SEEDS['confirmation'][1])} reserved confirmation seeds: no attainable n "
            f"({NO_ATTAINABLE_N}), the design stop; the verification is not run")  # fmt: skip


def _verification_floor(screen: Mapping[str, Any]) -> int:
    """N's floor in the verification: ``n_min`` when the screen found the power short (the last
    n of its scan), else 60. ``SimulationError`` when no n is attainable (``unattainable_n``):
    the verification is not run."""
    n = unattainable_n(screen)
    if n is not None:
        raise SimulationError(no_attainable_n(n))
    scan = screen["gates"]["n_scan"]
    return scan[-1]["n"] if scan else MIN_N


def _verification(runs_root: Path, context: Context, commit: str, screen: Mapping[str, Any],
                  pool: Workers | None) -> dict[str, Any]:  # fmt: skip
    """The binding verification (§"Simulation before validation", "Fidelity"): C01–C17 in
    ``"procedure"`` mode at L, 100 000 resamples, the realized route (candidate 1, then 2),
    N at least ``n_min`` when the power was short, P–F where its gate passed, the attained N and
    caps; per test the first M studies by index (A–F and the joint test: confirmed at L; P–F:
    confirmed with P–F active), never past 4·M draws. Its summary (``verification.json``)."""
    fixation, d = context.fixation, _decisions
    m, j = d.COVERAGE_RUNS, fixation["J"]
    bound, base = d.MAX_DRAWS_FACTOR * m, fixation["streams"]["verification"]
    level = selected_level(screen["tests"], j, m)
    if level is None:
        raise SimulationError("no level passed the screen: the verification is not due")
    floor, blocks = _verification_floor(screen), screen["validation_blocks"]
    coverage = context.of("coverage")

    def pf(o: StudyOutcome) -> bool:
        return bool(o.confirmed.get(level)) and "P-F" in o.active

    def short(_: str, rows: list) -> int:
        return max(m - sum(bool(o.confirmed.get(level)) for o in rows), m - sum(map(pf, rows)))

    units = {sc.name: Unit("verification", sc.name, 0, 1, base, d.VERIFICATION_RESAMPLES, (level,),
                           blocks, floor) for sc in coverage}  # fmt: skip
    rows = _phase(pool, runs_root, context, commit, units, bound, short)
    tests = []
    for identity in fixation["tests"]["list"]:
        if identity[0] != "verification":
            continue
        found = rows[identity[1]]
        used = (_first(found, m, pf) if identity[4] == "P-F"
                else first_confirmed(found, m, level))  # fmt: skip
        tests.append(_test_row(identity, level, used, _drawn(used, found, m, bound), m, j))
    return {
        "kind": VERIFICATION_KIND,
        "fixation": dict(context.ref),
        "commit": commit,
        "M": m,
        "J": j,
        "level": level,
        "n_floor": floor,
        "validation_blocks": blocks,
        "resamples": d.VERIFICATION_RESAMPLES,
        "tests": tests,
    }


def _warmed_moments(runs_root: Path, context: Context, commit: str,
                    workers: Workers) -> dict[tuple, tuple[float, float]]:  # fmt: skip
    """The truths' moments at a jitter scenario's frozen calibration for every pair of rates
    its screened studies selected, computed over the workers (the verification's studies select
    from the same distribution), with those this process holds."""
    found = _seeded_moments(context)
    tasks = []
    for sc in context.of("coverage"):
        if not sc.jitter:
            continue
        template = Unit("screen", sc.name, 0, 1, context.fixation["streams"]["calibration"],
                        _decisions.SCREEN_RESAMPLES, LEVELS, VALIDATION_BLOCKS[0],
                        MIN_N)  # fmt: skip
        try:
            head = _read_part(part_path(runs_root, "screen", sc.name, 0))[0]
            template = replace(template, validation_blocks=head["validation_blocks"])
            rows = _prefix(runs_root, context, commit, template,
                           _decisions.MAX_DRAWS_FACTOR * _decisions.COVERAGE_RUNS, {})  # fmt: skip
        except (OSError, ValueError, SimulationError):
            continue
        pairs = {(c, arm, (r["l"], r["u"])) for o in rows for c, rates in o.rates.items()
                 for arm, r in rates.items()}  # fmt: skip
        for c, arm, pair in sorted(pairs):
            cal = context.setup.calibrations.get((sc, c))
            if cal is None:
                continue
            kstar_ln = math.log(CANDIDATES[c]["targets"]["star"])
            for v in _shifted(context.setup.vectors, cal):
                tasks.append((v, arm, {"l": pair[0], "u": pair[1]}, sc, cal,
                              context.setup.slopes, kstar_ln))  # fmt: skip
    for key, value in zip(map(_moment_key_of, tasks), workers.map(_moment_task, tasks),
                          strict=True):  # fmt: skip
        found[key] = value
    return found


def _moment_key_of(task: tuple) -> tuple:
    vector, arm, rates, scenario, calibration, slopes, kstar_ln = task
    return _moment_key(vector, arm, rates, scenario, calibration, slopes, kstar_ln, TOLERANCE,
                       MAX_NODES)  # fmt: skip


def _moment_task(task: tuple) -> tuple[float, float]:
    vector, arm, rates, scenario, calibration, slopes, kstar_ln = task
    return _arm_moments(vector, arm, rates, scenario, calibration, slopes, kstar_ln,
                        tol=TOLERANCE, max_nodes=MAX_NODES)  # fmt: skip


def run_verification(context: Context, runs_root: str | Path, commit: str, *,
                     workers: int) -> Path:  # fmt: skip
    """The run's step 4 (``_verification``), after the screen of this fixation and commit, on
    the verification stream; written once (``verification.json``), never rerolled.
    ``SimulationError``: no screen of this fixation and commit, no L (not due), no attainable
    n (``unattainable_n``: not run), ``verification.json`` present, or a part of another
    fixation or commit."""
    runs_root = Path(runs_root)
    path = verification_path(runs_root)
    if path.exists():
        raise SimulationError(f"{path} exists: the verification is never rerolled")
    screen = read_screen(runs_root, context, commit)
    if selected_level(screen["tests"], context.fixation["J"], _decisions.COVERAGE_RUNS) is None:
        raise SimulationError("no level passed the screen: the verification is not due")
    if unattainable_n(screen) is not None:
        raise SimulationError(no_attainable_n(unattainable_n(screen)))
    problems = part_problems(runs_root, context.ref, commit)
    if problems:
        raise SimulationError("; ".join(problems))
    with Workers(workers, context) as first:
        moments = _warmed_moments(runs_root, context, commit, first)
    with Workers(workers, context, moments) as pool:
        _verification(runs_root, context, commit, screen, pool)
    _write_once(path, _verification(runs_root, context, commit, screen, None))
    return path


def verification_summary(runs_root: str | Path, context: Context, commit: str) -> dict[str, Any]:
    """``verification.json``'s content from the parts as they are (nothing drawn), at the L of
    this fixation's and commit's screen."""
    screen = read_screen(runs_root, context, commit)
    return _verification(Path(runs_root), context, commit, screen, None)


def results_record(screen: Mapping[str, Any], verification: Mapping[str, Any] | None,
                   context: Context, code: Mapping[str, str], *,
                   written_at: str) -> dict[str, Any]:  # fmt: skip
    """``results.json`` (§"The results file"), from the screen's and the verification's
    summaries as they are: the rows of the calibration tests, then those of the verification
    (each not due when no level passed, or when no n is attainable, ``unattainable_n``: the
    verification is not run), in the fixation's order; the gates (the powers at L, or
    0.975 without one, the power at the fixation's denser missing rate beside them, M6b; the
    scan; the detection powers at J) and the reported figures.
    ``written_at`` is given (the code commit's time, so that a resumed run's results are an
    uninterrupted one's, byte for byte). ``SimulationError`` for summaries of another fixation
    or commit, a verification missing at L, or one given when no n is attainable."""
    d, fixation = _decisions, context.fixation
    m, j = d.COVERAGE_RUNS, fixation["J"]
    for name, found in (("screen", screen), ("verification", verification)):
        if found is None:
            continue
        if found.get("fixation") != dict(context.ref) or found.get("commit") != code["commit"]:
            raise SimulationError(f"the {name} is of another fixation or commit than "
                                  f"{code['commit']}")  # fmt: skip
    level = selected_level(screen["tests"], j, m)
    rows = list(screen["tests"])
    skipped = None if level is None else unattainable_n(screen)
    if skipped is not None and verification is not None:
        raise SimulationError(f"a verification is given, but {no_attainable_n(skipped)}")
    if level is None or skipped is not None:
        rows += [_not_due(i) for i in fixation["tests"]["list"] if i[0] == "verification"]
    elif verification is None or verification.get("level") != level:
        raise SimulationError(f"level {level} passed the screen: its verification is due")
    else:
        rows += list(verification["tests"])
    at = d.POWER_LEVEL if level is None else level
    power = screen["gates"]["power"]
    gates = {
        "safe_pass_rate": screen["gates"]["safe_pass_rate"],
        "empirical_pass_rate": screen["gates"]["empirical_pass_rate"],
        "power_af": {"level": at, "by_scenario": power["A-F"][level_slot(at)]},
        "power_pf": {"level": at, "by_scenario": power["P-F"][level_slot(at)]},
        d.POWER_REPORT: {
            "level": at,
            "missing_rate": fixation["missing_labels"]["report"]["missing_rate"],
            "by_scenario": screen["gates"]["power_report"][level_slot(at)],
        },
        "n_scan": screen["gates"]["n_scan"],
        "detection_power": d.detection_power(j, m),
    }
    return {
        "kind": RESULTS_KIND,
        "written_at": written_at,
        "fixation": dict(context.ref),
        "code": dict(code),
        "M": m,
        "J": j,
        "streams": dict(fixation["streams"]),
        "resamples": dict(fixation["resamples"]),
        "validation_blocks": screen["validation_blocks"],
        d.SELECTED_LEVEL: level,
        "tests": rows,
        "gates": gates,
        "reported": screen["reported"],
    }


# -- the calibration of every scenario, over the workers


def calibration_pairs(scenarios: Sequence[Scenario]) -> list[tuple[Scenario, int]]:
    """The (scenario, candidate) pairs a fixation calibrates: each scenario's candidate, and
    candidate 2 for a coverage scenario of candidate 1 (the verification falls back to it)."""
    out = []
    for sc in scenarios:
        out.append((sc, sc.candidate))
        if sc.coverage and sc.candidate == 1:
            out.append((sc, 2))
    return out


def _calibration_task(args: tuple) -> tuple[Scenario, int, Calibration]:
    scenario, setup, candidate, base, studies = args
    seed = calibration_stream(base, scenario, candidate)
    return scenario, candidate, calibrate_scenario(scenario, setup, candidate, seed,
                                                   studies=studies)  # fmt: skip


def calibrate_all(
    setup: Setup,
    base: int,
    *,
    pairs: Sequence[tuple[Scenario, int]],
    workers: int,
    studies: int = _decisions.CALIBRATION_STUDIES,
) -> dict[tuple[Scenario, int], Calibration]:
    """Every pair's calibration (``calibrate_scenario`` on ``calibration_stream(base, …)``):
    the pairs without jitter one per worker (the gates, which have no σ or offset to solve,
    last), a jitter scenario's (C12) in this process, its truths' quadratures built over a pool
    of their own, so that they wait for no other pair. The result is the same whatever the
    workers or the order in which they finish."""
    plain = sorted((pair for pair in pairs if not pair[0].jitter),
                   key=lambda pair: pair[0].role == "gate")  # fmt: skip
    jitter = [(sc, c) for sc, c in pairs if sc.jitter]
    out: dict[tuple[Scenario, int], Calibration] = {}
    with Workers(workers) as pool, Workers(workers if jitter else 1) as grids:
        running = pool.imap(_calibration_task, [(sc, setup, c, base, studies) for sc, c in plain])
        mapper = grids.map if workers > 1 else None
        for sc, c in jitter:
            seed = calibration_stream(base, sc, c)
            out[(sc, c)] = calibrate_scenario(sc, setup, c, seed, studies=studies, mapper=mapper)
        for sc, c, cal in running:
            out[(sc, c)] = cal
    return {pair: out[pair] for pair in pairs}


def _close_float(a: float, b: float, floor: float) -> bool:
    """Equal within a relative ``CALIBRATION_TOLERANCE`` or within the absolute ``floor``."""
    return math.isclose(a, b, rel_tol=CALIBRATION_TOLERANCE, abs_tol=floor)


def _differences(found: Any, frozen: Any, where: str, floor: float = 0.0) -> list[str]:
    """Where two JSON values differ: integers, flags, names and keys exactly; floats within a
    relative ``CALIBRATION_TOLERANCE`` or the absolute ``floor``."""
    if isinstance(frozen, dict):
        if not isinstance(found, dict) or set(found) != set(frozen):
            return [f"{where}: keys {sorted(found) if isinstance(found, dict) else found!r}, "
                    f"frozen {sorted(frozen)}"]  # fmt: skip
        return [p for k in sorted(frozen) for p in _differences(found[k], frozen[k],
                                                                f"{where}.{k}", floor)]  # fmt: skip
    if isinstance(frozen, list):
        if not isinstance(found, list) or len(found) != len(frozen):
            return [f"{where}: {found!r}, frozen {frozen!r}"]
        return [p for i, (a, b) in enumerate(zip(found, frozen, strict=True))
                for p in _differences(a, b, f"{where}[{i}]", floor)]  # fmt: skip
    if isinstance(frozen, float) and not isinstance(found, bool) and isinstance(found, float):
        if _close_float(found, frozen, floor):
            return []
        return [f"{where}: {found!r}, frozen {frozen!r}"]
    if type(found) is not type(frozen) or found != frozen:
        return [f"{where}: {found!r}, frozen {frozen!r}"]
    return []


def calibration_problems(
    found: Mapping[tuple[Scenario, int], Calibration],
    frozen: Mapping[tuple[Scenario, int], Calibration],
) -> list[str]:
    """How a recomputed calibration departs from the frozen one: a pair missing or extra, an
    integer, a flag or a vector's seed that differs, or a float off both by more than a relative
    1e-9 and by more than its field's absolute floor (``CALIBRATION_FLOORS``: the solvers'
    precision, 1e-9 for a pp field, 1e-11 for a proportion, 1e-10 for λ), so that a result near
    0 is compared by its precision (the solvers and the quadrature, on another machine)."""
    problems = []
    for sc, c in frozen:
        if (sc, c) not in found:
            problems.append(f"{sc.name} at candidate {c}: missing from the recomputation")
            continue
        a = json.loads(_json_bytes(calibration_json(found[(sc, c)])))
        b = json.loads(_json_bytes(calibration_json(frozen[(sc, c)])))
        problems += [f"{sc.name} at candidate {c}: {p}" for name in sorted(b)
                     for p in _differences(a.get(name), b[name], f"calibration.{name}",
                                           CALIBRATION_FLOORS.get(name, 0.0))]  # fmt: skip
    problems += [f"{sc.name} at candidate {c}: not in the fixation" for sc, c in found
                 if (sc, c) not in frozen]  # fmt: skip
    return problems
