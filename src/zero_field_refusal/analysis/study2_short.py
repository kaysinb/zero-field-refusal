"""Study 2's redirected short-regime stage, design A (amendment 3; its batch, U and rates selected
under amendment 4): selection, support, the estimator and the looks, as pure functions over
``stage_a.Trajectory`` objects, numerical-check verdicts and label rows.

AdamW-25 and FA-25 (Adam with its second moment frozen at θ₀) each run exactly 25 steps at three
rates targeted at the levels (5, K*, U), K* = √(5U); harm at K* is interpolated linearly in ln K
between the nearest bracketing endpoints of each seed block, and the paired block differences
enter two sequential looks. Selection, support and validation read K_benign, numerical records
and utility only, never a judge label or K_harmful; only the estimator, the looks, the mechanism
cohort's final analysis (amendment 4; amendment 3's secondary contrasts before it) and the time
course's final reading read labels. The values the companion simulation fixes (w_max, the look
levels, the SE floor, the bias allowance) are inputs, never constants here. Harm rates and
differences are fractions (1 pp = 0.01); utility losses are in percentage points.

- ``usable_rate``: an arm, batch and rate over the discovery seeds: every seed finished its 25
  steps with a finite K25 (K_benign at step 25), mean utility loss in bounds against θ₀.
- ``targeted_rates``: log–log interpolation of the rate against the median K25 between adjacent
  usable rates, for each target; no extrapolation.
- ``select_rates``: the rate-ordered triple of usable rates nearest the targets in ln K under the
  support, realized-width and coverage constraints, or the constraints no triple meets with the
  best unconstrained triple; ``repair_rates``: the one repair round from that triple.
- ``candidate_order``: amendment 4's candidates (U, B) = (8, 20), (7, 20), in that fixed order
  (B = 20 is the study's fixed scope, review fa22), with their costs for the report.
- ``numerical_cases``, ``numerical_policy``, ``numerical_verdict`` and ``read_numerical``: the
  numerical qualification's cases on seeds 2200-2202, its frozen policy, and the verdict records
  of the numerical-check tool (scripts/numerical_check.py) under $RUNS_DIR/numerical/.
- ``selection``: the state machine over every run so far and the verdicts: the next campaign
  (``discovery``, ``repair``, ``prewarmed``, ``targeted``, ``numerical``, the step
  ``numerical-check``, ``validation``, ``done`` with the frozen rates, ``infeasible`` when no
  candidate is eligible even for the primary arms alone, with its B = 20 feasibility report
  (``feasibility_report``), ``incomplete`` when the records are ambiguous, or with ``blocked``
  when a technical non-qualification stops the stage for a human decision), every attempt
  recorded.
- ``control_verdict``: an FA soft-target control under amendment 2's acceptance at 0.01 × K*
  (amendment 3's controls, reported and never used under amendment 4).
- ``block_support``: one arm in one block: supported, or the cause (bracketing, width,
  technical), with its bracket.
- ``harm``, ``interpolate``, ``arm_harm`` and ``block_difference``: the estimator on label rows
  (plain dicts ``condition, step, prompts, harmful, missing``, as ``analysis.study2_labels``
  reads them), given per run id, with the per-block bounds of the missing-label sensitivities.
- ``looks`` and ``finite_bounds``: the two sequential looks and the finite-cohort bounds;
  ``bootstrap_t_interval``: the looks' interval as decided (the bootstrap-t);
  ``sensitivity_looks``: a missing-label sensitivity (``SENSITIVITIES``) at the primary's looks.
- ``secondary_contrasts``: A − P and P − F on the common three-arm supported cohort, by the
  fixation record's secondary interval (amendment 3's; retired under amendment 4, whose summary
  does not call it, and kept with its fixation keys for amendment 3's record).
- ``continuation``: the blocks after the primary's stopping look, descriptive.
- ``mechanism_state``, ``mechanism_blocks`` and ``mechanism_analysis``: the mechanism cohort
  (amendment 4, section 3), its state on K and support only, its blocks with their labels, and
  its one final analysis at the cohort's end: C1 = P − F and C2 = A − P by ``max_t_interval``
  (the paired seed-block bootstrap max-t on shared draws), with the per-contrast
  ``symmetric_studentized`` intervals at 97.5 % beside them, read by ``component_readings``.
- ``scalars_coverage``: the update scalars' coverage per regular run, reported, never gating.
- ``utility_family`` and ``validation_verdict``: the simultaneous utility bounds and the verdict
  of the validation blocks, over the three arms (amendment 4) or the primary arms (amendment 3,
  the fallback).
- ``progress``: the stage as a whole for scripts/study2_short_campaigns.py and
  scripts/study2_short_summary.py: the records checked against the campaigns (``conformance``,
  one training stream per seed, one v₀ per seed and batch: ``preconditioner_pairing``), the
  selection's state machine with its numerical and validation gates, then the confirmation
  blocks with their support (K only) and the mechanism cohort's state, the runs the launcher gave
  up on (``exhausted_runs``, with the attempts abandoned by the operator: ``read_abandonment``)
  counted as technical failures; ``confirmation_guard`` decides whether a
  range of blocks may start; ``look_blocks`` turns the blocks into those of ``looks`` once their
  labels exist. ``read_fixation`` reads the values the simulation fixes.
- ``split_timecourse``, ``timecourse_identity``, ``timecourse_summary`` and
  ``timecourse_guard``: the time course (amendment 4, section 4, "Harm over steps";
  exploratory), separate runs of the first 10 confirmation blocks at each arm's central rate
  evaluated at steps 1, 5, 10 and 25: split from the other runs where they are loaded, each
  checked for numerical consistency with its endpoint run (review fa21), the trajectories with
  the failed and missing runs per
  seed and arm, harm only when the final analysis passes the labels; and when its campaign may
  be written.

Blocks given to ``looks`` and ``secondary_contrasts`` are dicts in their preassigned order:
``{"block": id, "complete": bool, "supported": bool, "delta": float | None, "h": {arm: float}}``,
``supported`` meaning AdamW-25 and FA-25 both supported, ``delta`` their difference at K*, and
``h`` the interpolated harm of every supported arm (the secondary contrasts read it);
``sensitivity_looks`` also reads the fields its sensitivity names (``block_difference``);
``mechanism_analysis`` reads them with the fields ``mechanism_blocks`` adds.

**Rulings** where the amendment leaves a detail open, fixed here before any data:
- *An endpoint* is a run that stopped as ``steps`` after its 25 steps with an evaluation at step
  25 whose K_benign and accuracies are finite; K25 is that K_benign. Any other run is a failure:
  an instability failure when its stop is one of Stage A's instability rules, otherwise technical
  (a missing run, another stop, no finite K25).
- *Discovery runs* of an arm at a batch are those on seeds 2000–2002 at Stage A's grid
  (``stage_a.SHORT_GRID``, the primary arms) and at the rates its rounds prescribed; any other
  run there is reported as unexpected and never selected. Numerical and validation runs are
  those a visit called for. A run at a batch the replay has not reached (amendment 3's targeted
  round at B = 5: no candidate visits B = 5 since B = 20 became the fixed scope, review fa22) is
  listed as ``not_reached``, not as a problem, and enters no decision.
- *Targeted rates*: a target lying in several increasing stretches takes the one of the smaller
  rates; a targeted or repair rate already run at that arm and batch is not run again.
- *Selection*: cost ties are compared at 12 decimals. "The constraints no triple meets" lists
  each of ``support``, ``width`` and ``coverage`` that no triple meets on its own, ``jointly``
  when each is met by some triple but none meets all, or ``three usable rates`` when there are
  fewer. The best unconstrained triple ignores support and width and prefers triples meeting
  coverage, else takes the order alone, so that the repair round can act on a target outside the
  usable range.
- *Repair*: its rates are rounded to 3 significant digits like the targeted ones; one midpoint
  per round, of the bracketing pair of the seed with the widest too-wide bracket (the first such
  seed on a tie); no repair without a triple (fewer than three usable rates). A coverage miss
  (|ln median K − ln t| > 0.1 in the best unconstrained triple) is repaired by re-targeting: the
  log–log interpolation for that target recomputed over every usable rate of the arm at the
  batch, the targeted round's included (coordinator, 2026-10-16). A target outside the usable
  range takes the outward extension on its side (r_5 / 1.21 below, r_U × 1.21 above), the
  brief's "the repair round handles it". When the repair rate is none or a rate already run,
  there is no repair and the candidate fails for that arm, before any of the round's rates is
  run.
- *Rounds* run only for an arm with no feasible triple, at most one targeted and one repair round
  per arm, batch and U (coordinator, 2026-10-16: U = 7's targets differ from U = 8's). Every run
  made at the batch enters the selection at either U, so a U = 7 round builds on U = 8's runs and
  proposes only rates not yet run. The primary arms go in step: the targeted round of every arm
  needing it, then the repair round; then the pre-warmed arm's rounds. A round spent at an
  earlier visit of the candidate is reused, never spent again. The search allowance's repair
  round (below) is apart from these: it comes only after a numerical verdict.
- *Joint discovery* (amendment 4): when the primary arms' targeted round is due at a candidate
  and the pre-warmed arm has no feasible triple there, the same ``discovery`` campaign holds the
  pre-warmed arm's first round at (B, U), at the union of the rates of the primary arms'
  targeted assignments (each target's rate, a grid rate included; a primary arm that needs no
  round contributes its selected triple). Without that campaign, the pre-warmed arm's first
  round, when it has no feasible triple, is at the union of the selected AdamW and FA rates
  (``prewarmed``). Only rates not yet run at the batch are run; one first round per (B, U).
- *Candidate order* (amendment 4, as amended by review fa22: B = 20 is the study's fixed scope):
  the three arms at (U, B) = (8, 20), then at (7, 20); then the primary pair (AdamW-25, FA-25) at
  (8, 20), then at (7, 20) (*Fallback*); if none qualifies, the stage ends ``infeasible`` with its
  B = 20 feasibility report, not a confirmation. No B = 5 candidate is ever prescribed. The
  costs below are kept for the report and decide nothing.
- *Candidate cost*: per arm and target, the usable grid rate (``stage_a.SHORT_GRID``) nearest the
  target in ln median K25 (ties to the smaller rate); per seed the sum of those runs'
  ``training_seconds`` (a rate nearest two targets counts twice: a block runs three per arm),
  plus the training stream, seconds per chunk × ⌈25 B / chunk size⌉, plus v₀'s preconditioner
  stream, seconds per chunk × 50 B / chunk size, counted once per block since FA, pre-warmed and
  their controls share v₀ (coordinator, 2026-10-16). The seconds per chunk are amendment 1's
  reference, ``stage_a.stream_time_reference`` over the inputs (the whole version-1 discovery
  set, checked, with its provenance; anything missing leaves the selection ``incomplete``). An
  unknown cost (an arm without a usable grid rate) is None, with its reason.
- *Amendment 3's FA controls* (experiments/study2_short_fa_controls) are listed under
  ``amendment3_controls`` with their verdicts under amendment 2's rule (``control_verdict``, at
  the bound they were configured with) and their departures from its prescription
  (``control_conformance``). They are never used: no rate is unusable by them. A second control
  of a key, or one off seed 2000, still leaves the records ambiguous (``incomplete``).
- *Pre-warmed AdamW-25* (amendment 4): part of a candidate's feasibility. Its first round is
  that of the joint discovery (or ``prewarmed``), its rounds follow the primary rule, it has no
  controls; still infeasible, the candidate fails for the three arms.
- *Numerical qualification* (amendment 4, review fa12): once every arm of the visit has its triple,
  the K-only regular runs of the check's cases (``numerical_cases``, role numerical,
  experiments/study2_short_k_only) are called for; once they all exist, the check of the
  numerical-check tool. Every selection's verdict is kept, under $RUNS_DIR, at
  numerical/b{B}-u{U}/verdicts/<digest>.json (``numerical_digest``; a record whose name is not the
  digest of its own rates is malformed), so none is overwritten; the request reads its own digest's
  verdict, else (the primary arms alone) a verdict of the candidate whose rates for the requested
  arms are exactly the request's, one marking a requested rate unusable first. A verdict at a batch
  the replay reached that it did not read and that marks a rate unusable leaves no conclusion. A
  verdict decides only when its batch and U are the candidate's, its policy is the frozen one
  (``numerical_policy``: norm 5e-05, survival 0.5, retry ×10, bound 0.01 × K*, steps 1, 5, 10, 15,
  20, 25, twin A's consistency rule ``twin_a_consistency`` {abs 1e-08, rel 1e-04} of review fa21,
  and the Gaussian directions' seeds and the reference's modes as the tool's module fixes them; a
  verdict without the rule departs from it), its GPU code (``policy.gpu_code``, review fa13; null
  is stale) is that of every commit its numerical runs recorded (each the verdict's
  ``gpu_code.commit`` or ``code_identity.reuse_check`` from it to that commit reusable, so that a
  replacement run made at an analysis-only descendant counts; a run on a dirty checkout or
  recording no commit leaves the selection ``incomplete``, a commit whose GPU code cannot be
  checked makes the verdict stale) and that of the running checkout
  (``reuse_check`` from its commit to HEAD reusable: analysis-only commits keep a qualification, a
  GPU-path change voids it; the checks are kept in the record), its ``tool_commit`` names the
  check's commits (a non-empty list, or one commit in an older record), every requested rate and FA
  control is in it and every rate's seeds cover its cases; otherwise it is stale and the next step
  runs the check again (review of Task 2, important 1 and 2). A rate or control ``incomplete`` is
  never a pass: the next step finishes the check. A rate ``failing`` (a pair exceeding, or its
  numerical run stopped by an instability rule) or ``not_qualified`` (technical, with its reason,
  never reported as instability), or an FA rate whose control is ``rejected`` or ``not_qualified``,
  is unusable at that batch for every U: excluded up front at every later visit of the batch. These
  marks persist when the verdict is stale (review of Task 2, minor 4): staleness voids a
  qualification, never a failure. They persist too when the tool writes a selection's verdict again:
  the earlier one, moved to verdicts/history/<digest>/, is read with it, and its marks hold whatever
  its staleness or unfinished entries (the coordinator, after the integration step); an earlier
  verdict the replay did not read is an unread verdict like any other. A re-solve is decided only
  from a current verdict with no incomplete entry, stale or not (the coordinator, after fix round
  1): while a requested rate or control is incomplete, missing, or tested on fewer seeds than its
  cases need (re-review 2), the next step finishes or reruns the check, and the re-solve is not
  spent. An arm holding unusable rates spends its search allowance (below); the replacement's
  cases are called for (a replacement rate on 2200, and on 2201 and 2202 if it is its arm's new
  U-rate; a replacement FA rate with its control) and checked in a new check. The verdict's own
  ``qualified`` must agree with its states, and an unreadable or malformed verdict leaves the
  selection ``incomplete``. A numerical run the launcher gave up on (``exhausted_runs``) lets the
  check go on (review of Task 2, minor 3): the check request names its case in ``given_up``
  (as the tool names a case, ``numerical_case_name``; the final review of gate B, I1), the tool's
  plan notes it instead of stopping, its parts record the case incomplete and, after the rerun,
  not qualified (technical), and its rate leaves the batch.
- *Technical and scientific failures* (review fa22: a technical non-qualification never decides
  the batch, nor the candidate). A rate is *technical* when a verdict marks it ``not_qualified``
  (a reproducibility mismatch, exhausted attempts, out of memory, missing measurements; a
  numerical run the launcher gave up on reaches the verdict so), and so is an FA rate whose
  control is ``not_qualified``: unusable like any other (a missing measurement never counts as
  passing), it spends its arm's search allowance. A rate ``failing`` (an exceeding pair, or an
  instability stop) and an FA rate whose control is ``rejected`` are *scientific* failures; a
  rate once marked scientific stays so. An arm that cannot go on is judged by why. Holding an
  unusable rate once its allowance is spent, it is stopped by a scientific failure when that
  verdict marks one of its rates scientific, otherwise by technical marks only. Left without a
  feasible triple (after its rounds, after its allowance's re-selection, or with its allowance
  spent and a new round due), it is stopped by technical marks only when lifting its technical
  exclusions, the scientific ones kept, gives it a feasible triple, whatever the marks of its
  last verdict (technical marks never decide; review of fa22, M2), and by a scientific failure
  otherwise (no admissible triple). An arm stopped by technical marks only at the numerical gate
  is never selected again at the candidate, even when another arm's scientific failure fails
  the candidate: the fallback's visit then blocks on it (review of fa22, I1). After a technical
  stop at discovery beside a scientific one, the fallback's visit runs the same discovery again:
  a primary arm stopped scientifically fails it again, and the arm stopped technically is judged
  again by the same rule. One allowance holds in all per arm, batch and U: an arm
  marked by a check that stopped the candidate before its re-solve owes its allowance, and a
  later visit's re-selection of it is that allowance (its second re-review, minor 1). The same
  reading holds at discovery,
  where marks made at the batch by an earlier candidate's check exclude rates up front: an arm left
  without a triple after its rounds by technical exclusions alone is stopped by them (this
  implementation's reading, accepted by the coordinator, C1). A scientific stop fails the
  candidate and the order moves on (so do a rejected same-path control, no admissible triple and
  a validation failure, except one that technical failures alone make: a shortfall of jointly
  supported blocks, or a utility member left without a bound: *Validation*); a technical-only
  stop ends the replay as ``blocked``:
  ``next`` is ``{"campaign": "incomplete", "blocked": "technical", ...}`` with the candidate,
  the gate, the arms, the rates with their reasons and the arms' allowances, for a human
  decision. It is no problem of the records, and no campaign is written. At one step a
  scientific stop of one arm outranks a technical stop of another (the scientific one alone
  would fail the candidate).
- *The search allowance* (review fa22; it replaces, at B = 20, the re-solve "with no new round"
  of Task 2's ruling): one per arm and candidate (U at B = 20), in all. When an arm's verdict
  marks one of its rates unusable and its allowance is unspent, the arm is re-solved once from
  its current discovery points (every rate run at the batch but the unusable ones); when no
  feasible triple remains, one more deterministic repair round (``repair_rates`` from the
  re-solve's best unconstrained triple, on K and utility only; the ``repair`` campaign, named
  after the check that spent it, whose cells are new rates of their arm and batch) and a
  re-selection follow. The round never proposes an unusable rate (the coordinator's ruling on
  C2): its interpolation and targeting read every usable rate of the arm at the batch, the
  excluded ones' K included (discovery measurements; only their numerics failed), and a
  proposal landing on an excluded rate becomes the nearest admissible rate: of 3 significant
  digits, nearest the unrounded proposal (ties to the smaller), neither run nor unusable,
  strictly inside the proposal's own interval (a re-targeted rate's bracketing pair, a
  midpoint's pair; below r_5, or above r_U, for an outward extension), within 10 steps of the
  3-digit grid (``SUBSTITUTE_STEPS``). Without one, or for any other coverage miss that
  re-targeting cannot repair, the round runs nothing, as at discovery. Every other part of the
  algorithm is unchanged, and the discovery rounds keep it whole. Every replacement goes through
  a new check. An arm whose allowance is spent
  stops at its next unusable rate, or when its re-selection has no feasible triple, and is
  judged as above.
  The replay keeps the allowance across the candidate's visits: spent at the three-arm visit, it
  stays spent at the primary pair's (never reset; the pre-warmed arm's is its own), where an arm
  whose allowance is spent and that unusable rates leave without a triple gets no new round.
- *A stale check with runs beyond it*: when the replay waits at a stale check while runs it
  called for after that check exist, the problems name the stale check first and list those
  runs, all of them under ``beyond_stale``; the replay cannot tell such a run from a stray one
  until the check is decided again, after which a stray run shows as unexpected again.
  scripts/study2_short_campaigns.py writes the ``numerical-check`` input when every problem is
  under ``beyond_stale``, and nothing else. A check that went stale under the same numerical
  runs (a policy departure, say) is run again. One stale by a GPU-path change cannot be: the
  tool refuses numerical runs whose GPU code is not its own. The voiding procedure (the final
  review of gate B, I2), ``scripts/numerical_check.py void --batch B --upper U`` on a clean
  checkout of the new commit, moves those numerical runs, the check's directory
  numerical/b{B}-u{U}/ and the launcher's record of their campaigns to
  $RUNS_DIR/void/<old GPU code>/ with an inventory; the replay then calls for the numerical
  runs again (``numerical``) and their new check, and the fresh campaign starts with all its
  attempts. ``void`` refuses once a run of the role validation, confirmation or timecourse
  exists at the batch (the coordinator's ruling on concern C1): a GPU-path change after S3 has
  started is outside this procedure and goes to the coordinator, since voiding those runs means
  spending on S3 again, a budget decision. It refuses too when a check directory of another U
  exists at the batch (the re-review's N1): the numerical runs at B serve that check as well,
  whose verdicts the replay would then never read; that case goes to the coordinator.
- *w_max* (the final review of gate B, I4): frozen since S1. Every campaign's header records
  the w_max it was written under, and the launcher logs that header with each start
  (``issued_w_max``); the replay under another w_max is refused (``incomplete``, with that
  reason), before anything is selected. Starts logged without a header record none.
- *The whole config* (the final review of gate B, M2): every regular run of this stage records
  the config hash of its campaign's config and overrides (``prescribed_hash``), checked by
  ``conformance``; the code that composes them (``run_prescription``) is outside the GPU-code
  identity, and this check is what shows a change of it after runs were made.
- *Fallback* (amendment 4, review fa12; review of Task 2, critical 1 and minor 2; review fa22):
  when no candidate is eligible with the three arms, (8, 20) and (7, 20) are visited again, in
  that order, for the primary pair alone, each selected over every record at its batch (rounds
  made at the other U included), except those whose primary arms failed at the numerical gate (a
  second check there would be a second allowance). A candidate whose primary arms failed at
  discovery in pass 1 is visited again: with no new round beyond each arm's once-only targeted
  and repair rounds per batch and U, it can only become feasible from rates run since; a
  three-arm validation failure does not exclude a candidate either, since the fallback's
  validation is amendment 3's primary-arm one; nor does the pre-warmed arm failing alone, in its
  qualification or its validation: the primary comparison stays at B = 20. A primary arm's
  allowance spent in pass 1 at the candidate stays spent (*The search allowance*); one spent by
  the pre-warmed arm is not the primary arms'. Every record is reused (rounds, runs, the checks
  whose rates hold the request); in a three-arm check read again, only the primary arms' rates
  and the FA controls count, whatever its pre-warmed entries say (review fa13), and nothing of
  the pre-warmed arm is called for. A selection ending on the primary pair has no mechanism
  cohort (the two-arm confirmation): the mechanism's results are not estimable. When no
  candidate is eligible even then, the stage ends ``infeasible``, with ``scope`` "B=20", every
  attempt's reasons and the ``feasibility`` report (``feasibility_report``): per candidate and
  arm, what failed and why, the scientific failures apart from the technical ones.
- *Support*: the nearest bracketing pair is the largest K25 below K* and the smallest at or above
  it. An unsupported arm's cause is ``technical`` whenever one of its endpoints failed (the
  failed run may have been the bracketing one), otherwise ``bracketing`` or ``width``. Beside
  it, ``k_cause`` gives what the surviving endpoints alone show and ``failure_kinds`` whether
  the failures were instability or technical, so that no count is lost. A second run at a
  rate, or a run at a rate outside the frozen ones, is a problem, not an endpoint.
- *Estimator and missing labels* (review fa4): the primary counts a missing label as not
  harmful. Two named sensitivities bound what missing labels can do, each evaluated at the
  primary's actual looks only (``sensitivity_looks``: the same blocks, level, interval, bootstrap
  stream and minima as the look; they have no stopping path of their own and never stop
  anything):
  - ``missing_labels``, the worst case for the contrast: per block the bounds
    [A_low − F_high, A_high − F_low], low counting an arm's missing labels as not harmful and
    high as harmful (both interpolated at K*); at a look, the interval runs from the lower bound
    of the interval over the blocks' lower bounds to the upper bound of that over their upper
    bounds;
  - ``all_missing_harmful``: every missing label counted as harmful in both arms, A_high −
    F_high, which can cancel between the arms (the earlier "worst case").
- *Looks*: ``levels`` are the looks' two-sided confidence levels, like the module's other
  levels; anything outside (0.5, 1), or a number of levels other than the number of looks, is
  refused. Blocks are read in order up to the first incomplete one. When the cap is reached
  without a stop, the terminal look happens there with the last level (also when look 1 fell on
  the cap and did not stop). With fewer than two supported blocks a look has no interval and is
  inconclusive. Equivalence: the interval within [−5, 5] pp, bounds included; direction: 0
  strictly outside it; ±8: within [−8, 8] pp, bounds included; inconclusive when none of the
  three holds. These are the simulation's conventions, on which the levels' calibration rests
  (coordinator, 2026-10-16; the bounds were strict before).
- *The looks' interval* (coordinator, 2026-10-16 and -17: looks 36 and 60, cap 80, α 0.040,
  levels 0.9920 / 0.9631, SE floor 0, w_max 0.47, equivalence minimum 36, direction minimum 10;
  the bias allowance is set in the fixation record): the paired seed-block
  bootstrap-t, the studentized bootstrap of the mean over the look's supported blocks exactly as
  analysis/study2_short_sim.py (revision 3) defines it. B resamples of the blocks with
  replacement, drawn at once; T* = (mean* − mean)/se*, se* = max(sd*, se_floor)/√n with the
  sample sd (ddof 1), the floor applying to se and se* alike; at level ℓ the interval
  [mean − q_{(1+ℓ)/2}·se, mean − q_{(1−ℓ)/2}·se], q np.quantile's default (linear) quantiles of
  T*. A resample with se* = 0 (at most ``ZERO_SE``) is dropped and counted; a zero se gives
  [mean, mean]; fewer than 2 blocks or 2 kept resamples give no interval. Look k's resamples come
  from ``numpy.random.default_rng(derive_seed(bootstrap_seed, "look", k))`` (k = 2 for a
  terminal look at the cap without look 1), the same streams for the sensitivities, so a look
  is reproducible from the fixation record alone. A bootstrap look reports numpy's mean, sd and se,
  those its interval is computed from (a t look reports ``statistics``' exact sums): an exact
  tie at a margin is read on numpy's mean, which can differ from the exact one in the last
  ulp. The t interval stays available (``interval`` ``t``).
- *Minimum supported blocks*: two gates on a look's readings, each 0 to disable it.
  - Equivalence is concluded only at a look with at least ``equivalence_min_supported``
    supported blocks, so not at a cap look below it. Direction and ±8 are unaffected. This is
    the simulation's ``eq_min36``.
  - Below ``direction_min_supported`` supported blocks no reading is concluded, neither
    direction nor equivalence nor ±8: the look is inconclusive (coordinator, 2026-10-17; at very
    few blocks the bootstrap-t collapses, e.g. to [mean, mean] at 2). The simulation's decision
    rule does not have this gate.
  - A look records whether each was met (``equivalence_allowed``, ``direction_allowed``).
- *Secondary contrasts* (coordinator, 2026-10-17; amendment 3's, retired by amendment 4's
  mechanism cohort, whose summary does not compute them): two-sided intervals at 98.75 % per
  contrast per look, without the SE floor, over the first N blocks started at the stopping look
  k that are supported in all three arms. The interval is the fixation record's
  ``secondary_interval``: the
  bootstrap-t of the looks (decided after the secondary simulation of 2026-10-17;
  ``bootstrap_t_interval``, the record's resamples) with its own stream per contrast and look,
  ``numpy.random.default_rng(derive_seed(bootstrap_seed, "secondary", contrast, k))``, or the t
  interval. The record's ``secondary_status`` (``secondary`` or ``exploratory``) is reported
  beside them.
- *Utility family*: blocks supported for the primary contrast (in a three-arm validation, for
  the three arms); for each member, an arm and rate of a supported block's bracket, the step-25
  losses of every completed run of that arm and rate among the given runs; one-sided t bounds at
  1 − 0.05 / (looks × family size) per metric: two looks in confirmation, one in validation, a
  single reading (coordinator, 2026-10-16). A member with fewer than two completed runs has no
  bound and fails; an empty family does not pass.
- *Validation*: each block's arm needs a run at every frozen rate (a missing one is a technical
  failure); the instability check covers every validation run of the arms read; the FA
  controls pass when every frozen FA rate has an accepted verdict (amendment 4: its same-path
  control in the check that qualified it). A three-arm candidate's validation (review fa12)
  reads the three arms: at least 10 of 12 blocks jointly supported for all three, no run of the
  three stopped by an instability rule, the utility family over all three; the fallback's is
  amendment 3's primary-arm validation. Ambiguous records (a second run of an arm, rate and
  seed, a run at an unfrozen rate) give no verdict (``passes`` None). The validation seeds are
  reused from candidate to candidate at the new batch and rates: a run of the same arm, batch,
  rate and seed is the same run. A failed validation fails the candidate, except a technical
  one (the coordinator's rulings on C3 and M3, review fa22; ``_technical_shortfall``; the 10 of
  12 and the utility bounds unchanged): no instability stop, no utility bound computed above
  its limit, every FA control accepted, and every remaining fault technical. A shortfall below
  10 jointly supported blocks is technical when at least 10 blocks are left once only the
  blocks with an arm unsupported by K (bracketing or width) are counted lost: the blocks lost by
  technical causes alone (runs given up, other stops, no finite K25) decide it. A utility
  member left without a bound (fewer than two completed runs) is technical: its runs failed
  technically or were given up. The stage is then ``blocked`` (technical) for a human decision;
  the same shortfall with a cause in K, or a bound above its limit, moves on.
- *The stage's records* (``progress``): a regular run of the three arms has the role of its seed
  (``study2_short.role``; Stage A's grids ``stage_a.role`` discovery; a time-course run
  timecourse, on seeds 3000-3009, with its own evaluation steps: *The time course*), the model,
  protocol version 2 and the 25-step schedule of experiments/study2_short (Stage A's), monitor
  version 2 (Stage A's own discovery runs read version 1), measures the Rayleigh quotient
  exactly on the first 10 confirmation blocks (seeds 3000-3009) and evaluates K only
  (``eval.k_only``, experiments/study2_short_k_only) exactly on the numerical seeds; an FA
  control has the role full_control, protocol version 3 and the schedule of
  experiments/study2_short_fa_controls but its bound; every run of a seed read that seed's one
  training stream, and every run that estimates v₀ (FA-25, pre-warmed AdamW-25, the FA
  controls, Stage A's FA grids) read one preconditioner stream per seed and batch, the same
  path with the same sha256 in its ``meta.json``. Any
  departure, and any problem the selection reports (a run it did not prescribe included), leaves
  the stage ``incomplete``: no campaign is written on records that are not whole. A confirmation
  run before the rates are frozen is a problem; once they are, every confirmation run must be at
  the chosen batch and a frozen rate of an arm still in the stage.
- *Validation and confirmation*: a candidate's validation blocks wait until every run of every
  arm at its rates on seeds 2100-2111 has its terminal record or was given up by the launcher; a
  run that is not there yet is not a failure. Then the verdict decides: ``confirmation`` when it
  passes, the next candidate otherwise, but ``blocked`` for a technical shortfall
  (*Validation*). A confirmation block is complete once every run of
  every arm at its frozen rates is there or given up; for the looks also once the bracket runs
  of every arm supported in it are labeled (``look_blocks``: a run without a label file, or
  without a completed judging receipt matching it (``study2_labels.LabelsPending``), is not
  labeled yet; a label file that does not read otherwise is an error).
- *Given up* (``exhausted_runs``, coordinator, 2026-10-16; review fa4): a validation or
  confirmation run whose cell had its recovery round and whose 2 × MAX_ATTEMPTS attempts in the
  launcher's journal all ended technical is a technical failure of its arm in its block, as a
  frozen rate without a run (``block_support``). A missing end record never makes a run given up
  by itself: the last attempt (the one after the recovery round) must have its end record, or be
  abandoned in the operator's abandonment record (``read_abandonment``); until then it may still
  be running and its run is pending. An earlier attempt without an end record was settled by the
  start of the next. A run with a terminal record is never given up, whatever the journal says.
  Discovery runs are not covered: the selection waits for them. A numerical run given up lets
  its check go on (*Numerical qualification*).
- *Confirmation blocks* (``confirmation_guard``, the review of Task 5): a block is started once
  it has runs or belongs to an issued range, which the launcher may not have reached yet (the
  review of Task 3, M1); ranges start at the first block not started and never overlap one
  another (re-issuing an issued range completes it); the looks fall at 36 and 60 supported
  blocks (coordinator, 2026-10-16; cap 80).
  - A three-arm selection (amendment 4, section 3; the coordinator, gate C): the campaigns do not
    pause at the primary's looks. The looks are computed afterwards from the same first 36 and
    60 primary-supported blocks, so their data are the same, and the confirmation ends with the
    mechanism cohort (below). A new range holds at most min(60 − p₃, 80 − the blocks before the
    first one not started) blocks (the review of Task 3, M3), p₃ counting every started block
    except the complete ones not jointly supported, so that no block is started past the
    cohort's 60; none starts once 60 jointly supported blocks are complete in order or every
    block up to the cap of 80 is started; ``after_look`` is refused, on a re-issue too (M2).
  - A two-arm selection (the pre-warmed arm dropped, or the primary-only fallback; no mechanism
    cohort), amendment 3's: a new range holds at most n_k − p blocks, p counting every started
    block except the complete unsupported ones, so that no block is started past a look that
    could stop; once look k is due, no block starts until the operator, having read it in the
    summary, passes ``after_look`` = k.
- *The mechanism cohort* (amendment 4, section 3; Task 3's brief): a block is jointly supported
  (``three_arm_supported``) when it is complete and its three arms are supported at K*, with a
  three-arm selection only. ``mechanism_state`` (K and support only) reads the blocks complete in
  order up to the 60th jointly supported one (its window); the cohort ends with 60 jointly
  supported blocks complete in order, or once all 80 blocks are started and complete, and its
  attrition is counted over the window by reason. The cohort is those jointly supported blocks
  of the window, whether or not the primary has stopped: the primary's decision is final at its
  stopping look (``looks`` reads its blocks only), and the blocks after it are reported as a
  descriptive ``continuation``. ``mechanism_analysis`` runs once, at the cohort's end: the
  summary computes it only when the cohort has ended, its blocks are labeled and the primary has
  stopped (the confirmation's end: both hold), and reports "not yet" before, reading no label
  for it. C1 = P − F and C2 = A − P per cohort block (P the pre-warmed arm), so that C1 + C2 =
  A − F on that cohort, checked to 1e-12; the max-t intervals share one set of resampled blocks
  per draw (``max_t_interval``), with the per-contrast ``symmetric_studentized`` intervals at
  97.5 % beside them, each on its own stream ``derive_seed(seed, "mechanism", ...)`` of the
  fixation record's ``mechanism`` seed. Zero SE: a resample's contrast with SE* ≤ 1e-12 has
  |T*| = 0 when its mean is the observed one within 1e-12, ∞ otherwise; the quantile reaching
  an ∞ draw is ∞ (no interval). Readings (``component_readings``) with ``_readings``'
  conventions: a direction (``positive``, ``negative``) when 0 is strictly outside the interval,
  ``bounded`` when it lies within ±5 pp, bounds included; none below 36 cohort blocks; an
  observed SE ≤ 1e-12 leaves its contrast unread and the other read on its 97.5 % interval. The
  missing-label sensitivities are computed for both contrasts as ``block_difference`` computes
  them for A − F, by the max-t on the same stream, descriptive.
- *The time course* (amendment 4, section 4; Task 5's brief): seeds 3000-3009 (blocks 1-10,
  preassigned), every arm of the frozen selection at its central rate (``central_rate``: the
  middle of its three, the K* rate), the endpoint run's config but for the role timecourse and
  ``schedule.evaluate_at`` [1, 5, 10, 25] (``TIMECOURSE_OVERRIDES``; blocks 1-10 measure the
  Rayleigh quotient, so these runs do too); 30 runs, 20 without the pre-warmed arm. The
  campaign is written only once the stage's next step is ``confirmation`` and blocks 1-10 are in
  issued confirmation ranges (``timecourse_guard``), with the versions record; a run whose
  endpoint run the launcher gave up on is left out and named in the campaign's header (its
  identity could never be shown). Its runs are split from the others by their role once, where
  the scripts load them (``split_timecourse``; review of Task 5, C1 and I1): the summary's looks,
  utility family, sensitivities and contrasts and the campaigns' bookkeeping never see them, and
  ``progress`` takes them apart (a time-course run among its other runs leaves the stage
  ``incomplete``). They are checked like the others (role, schedule, monitor, Rayleigh, one
  stream and one v₀ per seed) and never enter the selection, a confirmation block, a look or
  the mechanism cohort; one before the rates are frozen, a second run of a key, or a run at a
  key the time course does not prescribe leaves the stage ``incomplete``. The endpoint check,
  numerical consistency (``timecourse_identity``, review fa21: not identical parameters),
  compares row for row the training's metrics.jsonl rows (every row but those an evaluation
  logged), the rows of the endpoint's evaluation, monitor.jsonl, update_scalars.jsonl, the
  endpoint's evaluations against the time course's of the same steps (step 25: K and utility),
  the stop with the steps taken and the streams read; the timings and memory peaks are left out
  (not numerics), and so is K_harmful (outcome-blind until the final analysis). The named
  continuous fields (``TIMECOURSE_TOLERANCED``: the training rows' loss, gradient norm and NLL
  per token; the monitor's K and its readings' gradient and displacement norms; the scoring
  diagnostics' K_scoring and K_monitor_full at step 25, in monitor.jsonl and their metrics.jsonl
  rows; the update scalars' norm, cosine and the denominator's departure norm ratio; the step-25
  evaluation's K and held-out NLL) are compared within twin A's tolerance, |x − y| ≤ 1e-8 +
  1e-4·max(|x|, |y|) with both finite; every other field exactly, as canonical JSON: counts,
  steps, stops, metadata, the streams and the discrete outcomes (the benchmark and factual
  accuracies, k/N; the departure's median and p90, bin edges of a histogram; review fa21: no
  tolerance on discrete outcomes). Bit for bit everywhere is
  ``identical``, within the rule ``consistent`` (the bitwise flag kept as ``bitwise_equal``);
  both count. Bit-equal records go through the rule too (review fa23): a named continuous field
  that is not finite excludes the run, whatever the endpoint run holds. Any other difference, or
  a record that is missing or does not read (metrics, monitor, update scalars, meta.json: every
  regular run writes them), excludes the run with its reasons; the endpoint Rayleigh records
  (R, ‖δ‖, δᵀH₀δ) are compared beside, as ``rayleigh_equal``, deciding nothing; no endpoint
  run yet leaves it pending; an endpoint run the launcher gave up on leaves it excluded (the
  conservative reading: its consistency can never be shown). A time-course run that did not
  complete its four evaluations, or that the launcher gave up on (``exhausted_runs`` keys it
  with ``TIMECOURSE``, apart from its endpoint run), is a failure of its seed and arm, a run not
  there yet is missing.
  ``timecourse_summary`` reads labels only when the final analysis passes them (``labels``, None
  by default and in ``progress``), and then only for runs that count (``identical`` or
  ``consistent``: ``IDENTITY_COUNTED``): an excluded or pending run keeps its K and no harm.
- *The fixation record* (``read_fixation``): a JSON object with ``w_max``, ``levels`` (the
  looks' two-sided confidence levels), ``se_floor`` and ``bias_allowance``, the last two in the
  units of Δ (fractions, 1 pp = 0.01), ``interval`` (``bootstrap_t`` or ``t``),
  ``bootstrap_resamples`` (an integer ≥ 1000) and ``bootstrap_seed`` (an integer), ``looks`` and
  ``cap`` (equal to ``LOOK_SIZES`` and ``CAP``, which the looks and the confirmation guard use: a
  record made for other looks is refused), ``equivalence_min_supported`` and
  ``direction_min_supported`` (integers ≥ 0), ``secondary_interval`` (``t`` or ``bootstrap_t``)
  and ``secondary_status`` (``secondary`` or ``exploratory``), and ``mechanism`` (amendment 4):
  ``target`` 60, ``cap`` 80, ``minimum`` 36, ``zero_se`` 1e-12, ``comparison_level`` 0.975 and
  ``effects`` [0.05, 0.025], each equal to the code's, the max-t ``level`` in [0.95, 1), its
  ``resamples`` (≥ 1000) and ``seed``; and ``audit`` (amendment 4, the terminal audit of the
  judge, design v3, ``analysis.study2_audit``): ``sizes`` (400, 400 and 200), ``selection``
  (``pps_with_replacement``), ``weight`` (``max_abs_coefficient``), ``interval``
  (``clopper_pearson_total_error``), ``arm_level`` (1 − 0.025/3), ``bias_level`` (0.975),
  ``missing_reference`` (``error``) and ``agreement_check_share`` (0.1), each equal to the
  code's, and the draw's ``seed``; and amendment 4's other values fixed before any harm label
  (``FIXED_KEYS``, ``fixed_values``), each an object equal to the code's: ``candidates`` (the
  B = 20 order and its primary-pair fallback), ``numerical`` (the numerical policy),
  ``validation`` (the validation rule), ``timecourse`` (its seeds and steps), ``drift`` (its
  stream, batches and halves) and ``scalars`` (the departure steps). Each is required and none
  is defaulted, also when ``interval`` is ``t`` and the bootstrap's values go unused. A key
  enters the code and the committed record in the same commit, so that the record always reads.
- *The versions record* (``versions_problems``, review fa4 and its fix review): the validation
  and confirmation campaigns and the looks refuse without it committed (tracked and unchanged at
  HEAD), or when the running commit, the model, the judge's identity or the fixation record
  departs from it; the discovery and numerical campaigns (``discovery``, ``repair``,
  ``prewarmed``, ``targeted``, ``numerical``) precede it and do not read it. Abandonment
  records and summary outputs live under ``$RUNS_DIR`` (ignored by Git), archived with an
  inventory, never committed: a commit after the record's own makes HEAD a descendant that the
  record does not allow.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

from ..config import config_hash, load_config
from ..judge import IDENTITY_FIELDS, judge_identity
from ..paths import CONFIG_ROOT, REPO_ROOT
from ..protocol import load_run_config
from ..seeding import derive_seed
from ..tracking import code_changes, git_state, tracked
from .stage_a import (
    BATCHES,
    CONTAMINATION_FRACTION,
    CONTROL_SEED,
    DISCOVERY_SEEDS,
    FACTUAL_BOUND_PP,
    INSTABILITY,
    LOWER,
    MMLU_BOUND_PP,
    SHORT_GRID,
    Trajectory,
    _finite,
    arm_of,
    contamination,
    in_bounds,
    index,
    monitor_spec,
    rate,
    schedule_spec,
    stream_pairing,
    stream_time_reference,
    utility_loss,
)

ADAMW, FA, PREWARMED = "adamw_25", "frozen_adam_25", "prewarmed_adamw_25"
PRIMARY_ARMS = (ADAMW, FA)
FA_CONTROL_ARM = "soft_control_frozen_adam"
STEPS = 25
COVERAGE = 0.1  # |ln median K - ln target| for each target
EXTENSION = 1.21  # the outward extension of a repair round
PRECONDITIONER_BATCHES = 50  # v₀'s teacher batches at the selected B, per seed
NUMERICAL_SEEDS = (2200, 2201, 2202)  # amendment 4: the numerical acceptance's fresh seeds
VALIDATION_SEEDS = tuple(range(2100, 2112))
CONFIRMATION_SEEDS = tuple(range(3000, 3080))
# (U, B): amendment 4's fixed order, B = 20 only (review fa22: the study's fixed scope); each for
# the three arms, then (the fallback) for the primary pair
CANDIDATES = ((8.0, 20), (7.0, 20))
SCOPE = "B=20"  # an ``infeasible`` stage's record: a B = 20 feasibility report
# The unusable marks of a verdict (``_marks``' states) that are technical, never scientific
TECHNICAL_MARKS = ("not_qualified", "control not_qualified")
ALLOWANCE = "allowance_repair"  # the search allowance's repair round, in ``rounds``
SUBSTITUTE_STEPS = 10  # how far that round looks, on the 3-digit grid, for an admissible rate
LOOK_SIZES = (36, 60)  # supported blocks at looks 1 and 2 (decided 2026-10-16, coordinator)
CAP = 80  # started blocks
EQUIVALENCE = 0.05
NESTED = 0.08
SECONDARY_LEVEL = 0.9875  # per contrast per look: Bonferroni over 2 contrasts x 2 looks
UTILITY_LEVEL = 0.95  # simultaneous, per metric
VALIDATION_MINIMUM = 10  # blocks jointly supported for AdamW-25 and FA-25
ARMS = (*PRIMARY_ARMS, PREWARMED)
V0_ARMS = (FA, PREWARMED, FA_CONTROL_ARM)  # the runs that estimate v₀ from the preconditioner
# The stage's configs (scripts/study2_short_campaigns.py builds every run on them).
REGULAR = "experiments/study2_short"
RAYLEIGH = "experiments/study2_short_rayleigh"  # REGULAR measuring the Rayleigh quotient
FA_CONTROLS = "experiments/study2_short_fa_controls"  # amendment 3's, reported and not used
ROLES = ("discovery", "numerical", "validation", "confirmation", "timecourse")  # study2_short.role
# The time course (amendment 4, section 4, "Harm over steps"; exploratory): separate runs of the
# first 10 confirmation blocks, each arm at its central (K*) rate, the endpoint run's config but
# for the role and the evaluation steps (TIMECOURSE_OVERRIDES).
TIMECOURSE = "timecourse"
TIMECOURSE_SEEDS = CONFIRMATION_SEEDS[:10]
TIMECOURSE_STEPS = (1, 5, 10, 25)
TIMECOURSE_OVERRIDES = (
    f"study2_short.role={TIMECOURSE}",
    f"schedule.evaluate_at=[{','.join(map(str, TIMECOURSE_STEPS))}]",
)
# A time-course run against its endpoint run (review fa21): identical (bit for bit) and
# consistent (numerically consistent) count; excluded or pending do not
IDENTITY_STATES = ("identical", "consistent", "excluded", "pending")
# The numerical-check tool (scripts/numerical_check.py) and its verdicts under $RUNS_DIR: one
# directory per check, numerical/b{B}-u{U}/verdict.json for a candidate's first.
NUMERICAL_TOOL = "scripts/numerical_check.py"
NUMERICAL_DIR = "numerical"
VERDICTS_DIR = "verdicts"  # numerical/b{B}-u{U}/verdicts/<digest>.json, one per selection checked
HISTORY_DIR = "history"  # verdicts/history/<digest>/*.json: a selection's earlier verdicts
K_ONLY = "experiments/study2_short_k_only"  # REGULAR with the K-only evaluation: numerical runs
OPTIMIZERS = {ADAMW: "adamw", FA: "adam_frozen", PREWARMED: "adamw_prewarmed"}  # its arm names
# Each arm's optimizer override of the stage's configs (``run_prescription``)
ARM_OVERRIDES = {ADAMW: (), FA: ("optimizer=adam_frozen",),
                 PREWARMED: ("optimizer=adamw_prewarmed",)}  # fmt: skip
# The states of an (arm, rate) and of an FA rate's same-path control in a verdict (review fa12):
# "not_qualified" is technical (incomplete after its rerun), never instability; "incomplete"
# waits for the rerun.
RATE_STATES = ("qualified", "failing", "not_qualified", "incomplete")
CONTROL_STATES = ("accepted", "rejected", "not_qualified", "incomplete")
# The frozen numerical policy a verdict must carry (its bound is 0.01 × K* of the U checked),
# with twin A's consistency rule against its archived run (review fa21): |x − y| ≤ abs +
# rel·max(|x|, |y|) on the named numerical fields, both finite.
NUMERICAL_POLICY = {"norm": 5e-05, "survival_share": 0.5, "retry_scale": 10,
                    "steps": [1, 5, 10, 15, 20, 25],
                    "twin_a_consistency": {"abs": 1e-08, "rel": 0.0001}}  # fmt: skip
NUMERICAL_BOUND_FRACTION = 0.01
CONTROL_REASON = "control: "  # a verdict's reason of a same-path control, under its FA rate's key
RAYLEIGH_BLOCKS = 10  # the first confirmation blocks: every endpoint measures R
INTERVALS = ("bootstrap_t", "t")  # the primary interval at a look, and the secondary one
SECONDARY_STATUSES = ("secondary", "exploratory")  # how the secondary contrasts are reported
VERSIONS_RECORD = "docs/study2-short-versions.json"  # relative to the repository root
VERSIONS_CONFIG = ("experiments/study2_short", ["study2_short.role=confirmation"])  # its model
# The missing-label sensitivities: the block fields of their lower and upper bounds.
SENSITIVITIES = {
    "missing_labels": ("delta_missing_low", "delta_missing_high"),
    "all_missing_harmful": ("delta_all_missing_harmful", "delta_all_missing_harmful"),
}
MIN_RESAMPLES = 1000  # the fixation record's bootstrap resamples, at least
ZERO_SE = 1e-12  # an SE at or below this is zero (identical values up to rounding), as simulated
# The mechanism cohort (amendment 4, section 3): its stopping rule, minimum and interval family.
MECHANISM_TARGET = 60  # jointly supported three-arm blocks complete in order
MECHANISM_CAP = CAP  # started blocks, the primary's cap
MECHANISM_MINIMUM = 36  # jointly supported blocks below which nothing is read
COMPARISON_LEVEL = 0.975  # the per-contrast (Bonferroni) symmetric studentized intervals
MECHANISM_EFFECTS = (0.05, 0.025)  # the meaningful component effect (the margin), and power's
# The fixation record's ``mechanism`` values that must equal the code's, then those it sets.
MECHANISM_FIXED = {"target": MECHANISM_TARGET, "cap": MECHANISM_CAP, "minimum": MECHANISM_MINIMUM,
                   "zero_se": ZERO_SE, "comparison_level": COMPARISON_LEVEL,
                   "effects": list(MECHANISM_EFFECTS)}  # fmt: skip
MECHANISM_KEYS = (*MECHANISM_FIXED, "level", "resamples", "seed")
CONTRASTS = {"C1": (PREWARMED, FA), "C2": (ADAMW, PREWARMED)}  # name: (minuend, subtrahend)
# The terminal audit of the judge (amendment 4, the audit design v3 after review fa18;
# analysis.study2_audit): the fixation record's ``audit`` values that must equal the code's, then
# the one it sets (the draw's seed). ``sizes`` names the arms as the record does (``AUDIT_ARMS``);
# ``arm_level`` is 1 − 0.025/3, each arm's share of the 97.5 % ``bias_level``.
AUDIT_ARMS = {"adamw": ADAMW, "fa": FA, "prewarmed": PREWARMED}
AUDIT_FIXED = {"sizes": {"adamw": 400, "fa": 400, "prewarmed": 200},
               "selection": "pps_with_replacement", "weight": "max_abs_coefficient",
               "interval": "clopper_pearson_total_error", "arm_level": 1 - 0.025 / 3,
               "bias_level": 0.975, "missing_reference": "error",
               "agreement_check_share": 0.1}  # fmt: skip
AUDIT_KEYS = (*AUDIT_FIXED, "seed")
# Amendment 4's other values fixed before any harm label, each a fixation-record key whose every
# value must be the code's (``fixed_values``): the candidate order with its fallback (review
# fa22: B = 20 only), the numerical policy (section 1, with review fa21's rule), the validation
# rule, the time course, the drift stream and the update scalars.
FIXED_KEYS = ("candidates", "numerical", "validation", "timecourse", "drift", "scalars")
FIXATION_KEYS = (
    "w_max",
    "levels",
    "se_floor",
    "bias_allowance",
    "interval",
    "bootstrap_resamples",
    "bootstrap_seed",
    "looks",
    "cap",
    "equivalence_min_supported",
    "direction_min_supported",
    "secondary_interval",
    "secondary_status",
    "mechanism",
    "audit",
    *FIXED_KEYS,
)


def k_star(upper: float) -> float:
    return math.sqrt(LOWER * upper)


def targets(upper: float) -> tuple[float, float, float]:
    return (LOWER, k_star(upper), upper)


def _three(x: float) -> float:
    """A rate rounded to 3 significant digits."""
    return float(f"{x:.3g}")


def _endpoint(run: Trajectory) -> tuple[dict[str, Any] | None, str | None]:
    """The run's step-25 evaluation if it is an endpoint, else why not."""
    if run.stop != "steps" or run.steps != STEPS:
        return None, f"stopped by {run.stop} at step {run.steps}"
    row = next((r for r in run.evaluations if r["step"] == STEPS), None)
    if row is None:
        return None, "no step-25 evaluation"
    if not all(_finite(row.get(k)) for k in ("K_benign", "benchmark_accuracy", "factual_accuracy")):
        return None, "a non-finite K25 or accuracy at step 25"
    return row, None


def _failure_kind(run: Trajectory) -> str:
    return "instability" if run.stop in INSTABILITY else "technical"


# ------------------------------------------------------------------ selection


def usable_rate(
    runs: Sequence[Trajectory], theta0: Mapping[str, float], seeds: Sequence[int] = DISCOVERY_SEEDS
) -> dict[str, Any]:
    """One arm, batch and rate over the discovery seeds: usable when every seed has one run that
    finished its 25 steps with a finite K25, and the mean utility losses over the seeds are in
    bounds (5 pp MMLU, 10 pp factual, against θ₀)."""
    reasons, k25, losses = [], {}, []
    for seed in seeds:
        found = [run for run in runs if run.seed == seed]
        if len(found) != 1:
            reasons.append(f"seed {seed}: {len(found)} runs")
            continue
        row, why = _endpoint(found[0])
        if row is None:
            reasons.append(f"seed {seed}: {why}")
            continue
        k25[seed] = row["K_benign"]
        losses.append(utility_loss(row, theta0))
    summary = {
        "mmlu_loss_pp": statistics.fmean(m for m, _ in losses) if losses else None,
        "factual_loss_pp": statistics.fmean(f for _, f in losses) if losses else None,
    }
    if not reasons and not in_bounds(summary):
        reasons.append(f"utility out of bounds (MMLU {summary['mmlu_loss_pp']:.2f} pp, factual "
                       f"{summary['factual_loss_pp']:.2f} pp)")  # fmt: skip
    return {
        "usable": not reasons,
        "reasons": reasons,
        "K25": k25,
        "median_K25": statistics.median(k25.values()) if k25 else None,
        **summary,
    }


def targeted_rates(records: Mapping[float, Mapping[str, Any]], upper: float) -> dict[str, Any]:
    """For each target (5, K*, U): ln(rate) interpolated linearly against ln(median K25) between
    adjacent usable rates (sorted by rate), rounded to 3 significant digits. A pair whose median
    K does not increase with the rate is reported and skipped; a target outside the usable range
    gets no rate (the repair round handles it)."""
    usable = sorted((lr, rec["median_K25"]) for lr, rec in records.items() if rec["usable"])
    pairs, skipped = [], []
    for (r0, k0), (r1, k1) in itertools.pairwise(usable):
        if k1 > k0:
            pairs.append((r0, k0, r1, k1))
        else:
            skipped.append(f"median K25 does not increase from {r0:g} ({k0:.4g}) to {r1:g} "
                           f"({k1:.4g}): pair skipped")  # fmt: skip
    found = []
    for target in targets(upper):
        pair = next((p for p in pairs if p[1] <= target <= p[3]), None)
        if pair is None:
            if len(usable) < 2:
                why = "fewer than two usable rates"
            elif target < min(k for _, k in usable):
                why = "below the usable range"
            elif target > max(k for _, k in usable):
                why = "above the usable range"
            else:
                why = "inside a skipped pair"
            found.append({"target": target, "rate": None, "reason": why})
            continue
        r0, k0, r1, k1 = pair
        lr = _log_interpolated(target, r0, k0, r1, k1)
        found.append({"target": target, "rate": _three(lr), "between": [r0, r1]})
    return {"U": upper, "K_star": k_star(upper), "targets": found, "skipped": skipped}


def _log_interpolated(target: float, r0: float, k0: float, r1: float, k1: float) -> float:
    """The rate where ln K reaches ln ``target`` on the line through (ln r0, ln k0) and
    (ln r1, ln k1), unrounded."""
    fraction = (math.log(target) - math.log(k0)) / (math.log(k1) - math.log(k0))
    return math.exp(math.log(r0) + fraction * (math.log(r1) - math.log(r0)))


def _nearest_admissible(
    exact: float, lo: float | None, hi: float | None, taken: Collection[float]
) -> float | None:
    """The rate of 3 significant digits nearest ``exact`` (ties to the smaller) strictly between
    ``lo`` and ``hi`` (None: unbounded on that side) and not in ``taken``, among ``exact``
    rounded and the ``SUBSTITUTE_STEPS`` grid neighbours on either side of it, each step taken
    on the 3-digit grid of its own decade (below 1.00e-5 comes 9.99e-6; review of fa22, M1);
    None when there is none."""
    base = _three(exact)
    found = [base]
    for down in (False, True):
        lr = base
        for _ in range(SUBSTITUTE_STEPS):
            lr = _three(lr - _grid_step(lr, down=True) if down else lr + _grid_step(lr))
            found.append(lr)
    admissible = [lr for lr in found if lr > 0 and (lo is None or lr > lo)
                  and (hi is None or lr < hi) and rate(lr) not in taken]  # fmt: skip
    return min(admissible, key=lambda lr: (abs(lr - exact), lr), default=None)


def _grid_step(lr: float, *, down: bool = False) -> float:
    """The step of the 3-significant-digit grid at ``lr``: that of its decade, or, going
    ``down`` from the decade's first rate (1.00 × 10^e), that of the decade below."""
    mantissa, exponent = f"{lr:.2e}".split("e")
    return 10.0 ** (int(exponent) - (1 if down and mantissa == "1.00" else 0) - 2)


def _nearest_pair(points: Sequence[Mapping[str, Any]], kstar: float) -> dict[str, Any]:
    """The nearest bracketing pair of ``points`` ({rate, K, ...}): the largest K below K* and the
    smallest K at or above it, with its realized width ln K_hi − ln K_lo."""
    below = [p for p in points if p["K"] < kstar]
    above = [p for p in points if p["K"] >= kstar]
    lo = max(below, key=lambda p: (p["K"], p["rate"]), default=None)
    hi = min(above, key=lambda p: (p["K"], p["rate"]), default=None)
    width = math.log(hi["K"]) - math.log(lo["K"]) if lo and hi else None
    return {"lo": lo, "hi": hi, "width": width}


def _check_triple(
    triple: tuple[float, float, float],
    records: Mapping[float, Mapping[str, Any]],
    upper: float,
    w_max: float,
) -> dict[str, Any]:
    deviations = [
        math.log(records[lr]["median_K25"]) - math.log(t)
        for lr, t in zip(triple, targets(upper), strict=True)
    ]
    seeds = sorted(set.intersection(*(set(records[lr]["K25"]) for lr in triple)))
    per_seed = {
        seed: _nearest_pair([{"rate": lr, "K": records[lr]["K25"][seed]} for lr in triple],
                            k_star(upper))
        for seed in seeds
    }  # fmt: skip
    support = all(b["lo"] and b["hi"] for b in per_seed.values())
    return {
        "rates": list(triple),
        "cost": sum(abs(d) for d in deviations),
        "deviations": deviations,
        "per_seed": per_seed,
        "support": support,
        "width": support and all(b["width"] <= w_max for b in per_seed.values()),
        "coverage": all(abs(d) <= COVERAGE for d in deviations),
    }


def select_rates(
    records: Mapping[float, Mapping[str, Any]],
    upper: float,
    w_max: float,
    *,
    exclude: Sequence[float] = (),
) -> dict[str, Any]:
    """The rate-ordered triple (r_5 < r_K* < r_U) of usable rates, outside ``exclude``, with the
    least Σ_t |ln median K(r_t) − ln t|, subject to support (on every seed min K < K* ≤ max K),
    the realized width of every seed's nearest bracketing pair (≤ w_max) and coverage (each
    |ln median K(r_t) − ln t| ≤ 0.1); ties to the lexicographically smaller triple. Without one:
    the constraints no triple meets and the best unconstrained triple, for the repair round."""
    usable = sorted(lr for lr, rec in records.items() if rec["usable"] and lr not in exclude)
    checks = [_check_triple(t, records, upper, w_max) for t in itertools.combinations(usable, 3)]
    feasible = [c for c in checks if c["support"] and c["width"] and c["coverage"]]
    best = min(feasible, key=lambda c: (round(c["cost"], 12), c["rates"]), default=None)
    unconstrained = min(checks, key=lambda c: (not c["coverage"], round(c["cost"], 12),
                                               c["rates"]), default=None)  # fmt: skip
    if len(usable) < 3:
        unmet = ["three usable rates"]
    elif best is None:
        unmet = [name for name in ("support", "width", "coverage")
                 if not any(c[name] for c in checks)] or ["jointly"]  # fmt: skip
    else:
        unmet = []
    return {
        "U": upper,
        "K_star": k_star(upper),
        "w_max": w_max,
        "usable": usable,
        "excluded": sorted(exclude),
        "feasible": best is not None,
        "rates": best["rates"] if best else None,
        "selected": best,
        "unmet": unmet,
        "best_unconstrained": None if best else unconstrained,
    }


def repair_rates(
    triple: Mapping[str, Any] | None,
    records: Mapping[float, Mapping[str, Any]],
    upper: float,
    w_max: float,
    *,
    unusable: Collection[float] = (),
) -> dict[str, Any]:
    """The one repair round from the best unconstrained triple, over ``records`` (every rate of
    the arm at the batch): a seed with no K below K* adds r_5 / 1.21; a seed with no K at or above
    K* adds r_U × 1.21; a bracket wider than w_max adds the geometric midpoint of the widest one's
    pair; a target missed by more than 0.1 in ln K is re-targeted, its rate interpolated again
    over every usable rate (outside the usable range, the outward extension on its side). Rates
    rounded to 3 significant digits. A coverage miss whose repair rate is none or a rate already
    run is ``unrepairable``: the candidate fails.

    ``unusable`` (the search allowance's round; review fa22, the coordinator's ruling on C2): the
    arm's rates found unusable at the batch by a numerical check. Their K still enters the
    interpolation (only their numerics failed), but no proposal is one of them: a proposal
    landing on one becomes the nearest admissible rate (``_nearest_admissible``: 3 significant
    digits, nearest the unrounded proposal, neither run nor unusable, strictly inside the
    proposal's own interval: the bracketing pair of a re-targeted rate or of a midpoint, below
    r_5 or above r_U for an outward extension), listed under ``substituted``; with none within
    ``SUBSTITUTE_STEPS`` steps, the proposal is ``unrepairable``."""
    if triple is None:
        return {"from": None, "rates": [], "reasons": ["no triple to repair from"],
                "unrepairable": [], "substituted": []}  # fmt: skip
    r5, _, r_upper = triple["rates"]
    per_seed = triple["per_seed"]
    existing = {rate(lr) for lr in records}
    unusable = {rate(lr) for lr in unusable}
    proposals: list[dict[str, Any]] = []
    unrepairable: list[str] = []
    substituted: list[dict[str, Any]] = []

    def propose(exact: float, lo: float | None, hi: float | None, reason: str) -> None:
        """Propose the unrounded rate ``exact``, whose interval is (``lo``, ``hi``)."""
        lr = _three(exact)
        if rate(lr) not in unusable:
            proposals.append({"rate": lr, "reason": reason})
            return
        other = _nearest_admissible(exact, lo, hi, existing | unusable)
        substituted.append({"unusable": lr, "rate": other, "for": reason})
        if other is None:
            unrepairable.append(f"{reason}: {lr:g} is unusable at the batch and no admissible "
                                f"rate lies within {SUBSTITUTE_STEPS} steps")  # fmt: skip
        else:
            proposals.append({"rate": other, "reason": f"{reason}: {lr:g} is unusable at the "
                                                       f"batch, the nearest admissible rate is "
                                                       f"{other:g}"})  # fmt: skip

    low = [seed for seed, b in per_seed.items() if b["lo"] is None]
    high = [seed for seed, b in per_seed.items() if b["hi"] is None]
    wide = [seed for seed, b in per_seed.items() if b["width"] is not None and b["width"] > w_max]
    if low:
        propose(r5 / EXTENSION, None, r5, f"no K below K* on seeds {low}: r_5 / {EXTENSION}")
    if high:
        propose(r_upper * EXTENSION, r_upper, None,
                f"no K at or above K* on seeds {high}: r_U × {EXTENSION}")  # fmt: skip
    if wide:
        seed = max(wide, key=lambda s: per_seed[s]["width"])
        pair = per_seed[seed]
        lo, hi = pair["lo"]["rate"], pair["hi"]["rate"]
        propose(math.sqrt(lo * hi), lo, hi, f"bracket {pair['width']:.3f} wider than {w_max} on "
                f"seed {seed}: midpoint of {lo:g} and {hi:g}")  # fmt: skip
    retargeted = targeted_rates(records, upper)["targets"]
    for target, deviation, entry in zip(targets(upper), triple["deviations"], retargeted,
                                        strict=True):  # fmt: skip
        if abs(deviation) <= COVERAGE:
            continue
        missed = f"target {target:.4g} missed by {deviation:+.3f} in ln K"
        how = "re-targeted over every usable rate"
        if entry.get("reason") == "below the usable range":
            exact, lo, hi = r5 / EXTENSION, None, r5
            how = f"below the usable range: r_5 / {EXTENSION}"
        elif entry.get("reason") == "above the usable range":
            exact, lo, hi = r_upper * EXTENSION, r_upper, None
            how = f"above the usable range: r_U × {EXTENSION}"
        elif entry["rate"] is None:
            unrepairable.append(f"{missed}: {how} gives {entry['reason']}")
            continue
        else:
            lo, hi = entry["between"]
            exact = _log_interpolated(target, lo, records[lo]["median_K25"], hi,
                                      records[hi]["median_K25"])  # fmt: skip
        lr = _three(exact)
        if rate(lr) in existing and rate(lr) not in unusable:
            unrepairable.append(f"{missed}: {how} gives {lr:g}, already run")
        else:
            propose(exact, lo, hi, f"{missed}: {how}")
    return {
        "from": triple["rates"],
        "rates": list(dict.fromkeys(p["rate"] for p in proposals)),
        "reasons": [p["reason"] for p in proposals] + unrepairable or ["nothing to repair"],
        "proposals": proposals,
        "unrepairable": unrepairable,
        "substituted": substituted,
    }


def candidate_order(
    runs: Sequence[Trajectory],
    theta0: Mapping[str, float],
    seconds_per_chunk: float,
    *,
    grid: Sequence[float] = SHORT_GRID,
    seeds: Sequence[int] = DISCOVERY_SEEDS,
) -> list[dict[str, Any]]:
    """The candidates in amendment 4's fixed order (``CANDIDATES``: (U, B) = (8, 20), (7, 20);
    B = 20 only), each with its cost for the report, which no longer orders them: the median
    over seeds of the summed training seconds of both primary arms' runs at the usable grid
    rates nearest each target, plus the block's training stream and v₀'s preconditioner
    stream."""
    cells: dict[tuple[str, int, float], list[Trajectory]] = defaultdict(list)
    for run in runs:
        if run.arm in PRIMARY_ARMS and run.seed in seeds:
            cells[(run.arm, run.batch_size, rate(run.lr))].append(run)
    grid = [rate(lr) for lr in grid]
    order = []
    for upper, batch in CANDIDATES:
        nearest, reasons, used = {}, [], []
        for arm in PRIMARY_ARMS:
            records = {lr: usable_rate(cells.get((arm, batch, lr), []), theta0, seeds)
                       for lr in grid}  # fmt: skip
            usable = {lr: rec for lr, rec in records.items() if rec["usable"]}
            if not usable:
                reasons.append(f"{arm}: no usable grid rate at B={batch}")
                continue
            nearest[arm] = [
                min(usable, key=lambda lr, t=t: (abs(math.log(usable[lr]["median_K25"])
                                                     - math.log(t)), lr))
                for t in targets(upper)
            ]  # fmt: skip
            used += [(arm, lr) for lr in nearest[arm]]
        entry: dict[str, Any] = {"batch": batch, "U": upper, "K_star": k_star(upper),
                                 "nearest": nearest, "reasons": reasons}  # fmt: skip
        if reasons:
            entry.update(seconds=None, per_seed={}, stream_seconds=None,
                         preconditioner_seconds=None)  # fmt: skip
        else:
            chunk_size = cells[(used[0][0], batch, used[0][1])][0].chunk_size
            stream = seconds_per_chunk * -(-STEPS * batch // chunk_size)
            # v₀'s preconditioner stream, once per block (FA and pre-warmed share it).
            preconditioner = seconds_per_chunk * PRECONDITIONER_BATCHES * batch / chunk_size
            per_seed = {
                seed: stream + preconditioner + sum(
                    next(r for r in cells[(arm, batch, lr)] if r.seed == seed)
                    .training_seconds for arm, lr in used)
                for seed in seeds
            }  # fmt: skip
            entry.update(seconds=statistics.median(per_seed.values()), per_seed=per_seed,
                         stream_seconds=stream,
                         preconditioner_seconds=preconditioner)  # fmt: skip
        order.append(entry)
    return order


def control_verdict(control: Trajectory, kstar: float) -> dict[str, Any]:
    """Amendment 2's acceptance of an FA soft-target control at 0.01 × K*: its 25 steps run and
    every scoring read finite and at most the bound (amendment 3's controls, reported under
    amendment 4 and not used)."""
    bound = CONTAMINATION_FRACTION * kstar
    reasons = []
    if control.stop != "steps" or control.steps != STEPS:
        reasons.append(f"stopped by {control.stop} at step {control.steps}")
    c = contamination(control, every_read=True)
    if c is None or not c <= bound:
        reasons.append(f"contamination {c} above {bound:.4g}")
    return {"run": control.run_id, "contamination": c, "bound": bound, "accepted": not reasons,
            "reasons": reasons}  # fmt: skip


def control_conformance(control: Trajectory, twin: Trajectory | None) -> list[str]:
    """How an FA control departs from the prescribed one (amendment 2, as
    ``stage_a.numerical_acceptance`` checks it): the model and the stream prefix of its regular
    twin (the FA run of seed 2000 at its batch and rate), planned for 25 steps and evaluated at
    least at the twin's evaluation steps and step 25. A departure leaves no conclusion."""
    name = control.run_id
    if twin is None:
        return [f"control {name}: no FA run of seed {CONTROL_SEED} at its batch and rate"]
    problems = []
    chunks, twin_chunks = control.stream.get("chunks", []), twin.stream.get("chunks", [])
    common = min(len(chunks), len(twin_chunks))
    if (control.model != twin.model or control.stream.get("path") != twin.stream.get("path")
            or chunks[:common] != twin_chunks[:common]):  # fmt: skip
        problems.append(f"control {name} is not on the model and stream of {twin.run_id}")
    needed = {row["step"] for row in twin.evaluations} | {STEPS}
    if control.planned_steps != STEPS or not needed <= set(control.evaluate_at):
        problems.append(f"control {name} planned {control.planned_steps} steps evaluated at "
                        f"{sorted(control.evaluate_at)}; it needs {STEPS} at "
                        f"{sorted(needed)}")  # fmt: skip
    return problems


def _configured_bound(run: Trajectory) -> Any:
    return (run.spec.get("instability") or {}).get("contamination_k")


def _bound_key(bound: Any) -> float | None:
    return float(f"{float(bound):.6g}") if _finite(bound) else None


def _control_key(run: Trajectory) -> tuple[int, float, float | None]:
    return (run.batch_size, rate(run.lr), _bound_key(_configured_bound(run)))


def _amendment3_controls(
    controls: Mapping[tuple[int, float, float | None], Trajectory],
    found: Mapping[tuple, Trajectory],
) -> list[dict[str, Any]]:
    """Amendment 3's FA soft-target controls with their verdicts under its own rule and their
    departures from its prescription, for the report: amendment 4 never uses them."""
    out = []
    for key in sorted(controls, key=str):
        control = controls[key]
        bound = _configured_bound(control)
        verdict = control_verdict(control, bound / CONTAMINATION_FRACTION if _finite(bound)
                                  else math.nan)  # fmt: skip
        twin = found.get((FA, control.batch_size, rate(control.lr), CONTROL_SEED))
        out.append({"run": control.run_id, "batch": control.batch_size, "lr": rate(control.lr),
                    "bound": key[2], "contamination": verdict["contamination"],
                    "accepted": verdict["accepted"], "reasons": verdict["reasons"],
                    "conformance": control_conformance(control, twin),
                    "used": False})  # fmt: skip
    return out


# ------------------------------------------------------------------ the numerical acceptance


def numerical_cases(rates: Mapping[str, Sequence[float]]) -> list[tuple[str, float, int]]:
    """The numerical qualification's cases (amendment 4) for each arm's selected ``rates``, as
    (arm, rate, seed): every selected rate on seed 2200, then each arm's U-rate (its largest) on
    seeds 2201 and 2202; 15 cases for three arms. After a re-solve a replacement rate has its
    case on 2200, and on 2201 and 2202 when it is its arm's new U-rate."""
    first = [(arm, rate(lr), NUMERICAL_SEEDS[0]) for arm in ARMS
             for lr in sorted(rates.get(arm) or ())]  # fmt: skip
    upper = [(arm, rate(max(rates[arm])), seed) for seed in NUMERICAL_SEEDS[1:] for arm in ARMS
             if rates.get(arm)]  # fmt: skip
    return first + upper


def numerical_case_name(arm: str, lr: float, seed: int) -> str:
    """A numerical case as the numerical-check tool names it
    (``fa_control_diagnostic.NumericalCase.name``): ``s<seed>-<optimizer>-<lr:g>``."""
    return f"s{seed}-{OPTIMIZERS[arm]}-{lr:g}"


def _pair_key(arm: str, lr: float) -> str:
    """An (arm, rate) as the verdict record keys it: f"{optimizer}|{lr:g}"."""
    return f"{OPTIMIZERS[arm]}|{lr:g}"


def numerical_policy(upper: float) -> dict[str, Any]:
    """The frozen numerical policy of a check at U (amendment 4, section 1): the perturbation
    norm, the survival share and its retry scale, the twin bound 0.01 × K*, the steps and twin
    A's consistency rule (review fa21), with the Gaussian directions' seeds, the reference's
    modes and the consistency rule as the numerical-check tool fixes them (``_tool_policy``)."""
    return {**NUMERICAL_POLICY, "bound": NUMERICAL_BOUND_FRACTION * k_star(upper),
            **_tool_policy()}  # fmt: skip


@functools.cache
def _tool_policy() -> dict[str, Any]:
    """What the numerical-check tool fixes: the seeds of the Gaussian directions p3 and p4 and
    twin A's consistency rule (its tracked settings, configs/experiments/numerical_check.yaml,
    read through its own ``NumericalScope``) and the reference's modes
    (``NUMERICAL_REFERENCE``), all from ``fa_control_diagnostic`` (review of Task 2, minor 1).
    The consistency rule is restated in ``NUMERICAL_POLICY`` too (review fa21): a tool whose
    tracked rule departs from it makes every verdict stale (``_policy_departures``). Imported
    when first needed: the tool's module loads torch."""
    from .. import fa_control_diagnostic as tool
    from ..paths import CONFIG_ROOT

    settings = load_config(CONFIG_ROOT / "experiments" / "numerical_check.yaml")
    scope = tool.NumericalScope.of(settings)
    return {"gaussian": dict(scope.gaussian), "reference": tool.NUMERICAL_REFERENCE,
            "twin_a_consistency": scope.consistency_rule}  # fmt: skip


def _reuse_check(base: str, head: str) -> dict[str, Any]:
    from ..code_identity import reuse_check  # git, read only when a verdict is

    return reuse_check(base, head)


def _gpu_code(
    gpu: Mapping[str, Any], commits: Collection[str], reuse: Callable[[str, str], Mapping[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """The checks that the verdict's GPU code (``gpu``) is that of the numerical runs, made at
    ``commits`` (each one's GPU identity the verdict's), and that of the running checkout (review
    fa13), and how it departs from them."""
    found: dict[str, Any] = {"runs": {}, "head": None}
    departures = []

    def checked(base: str, head: str) -> Mapping[str, Any] | None:
        try:
            return reuse(base, head)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            departures.append(f"the GPU code of {base} cannot be checked against {head}: {exc}")
            return None

    for commit in sorted(set(commits) - {gpu["commit"]}):
        found["runs"][commit] = checked(commit, gpu["commit"])
        if found["runs"][commit] is not None and not found["runs"][commit].get("reusable"):
            departures.append(f"numerical runs ran another GPU code ({commit}) than the "
                              f"verdict's ({gpu['commit']}): "
                              f"{found['runs'][commit].get('changed')}")  # fmt: skip
    found["head"] = checked(gpu["commit"], "HEAD")
    if found["head"] is not None:
        base = found["head"].get("base") or {}
        if (base.get("files"), base.get("sha256")) != (gpu["files"], gpu["sha256"]):
            departures.append(f"the verdict's gpu_code is not the identity of {gpu['commit']}")
        if not found["head"].get("reusable"):
            departures.append(f"GPU paths changed since {gpu['commit']}: "
                              f"{found['head'].get('changed')}")  # fmt: skip
    return found, departures


def _policy_departures(policy: Mapping[str, Any], upper: float) -> list[str]:
    frozen = numerical_policy(upper)
    out = []
    for key in ("norm", "survival_share", "retry_scale"):
        if not (_number(policy.get(key)) and policy[key] == frozen[key]):
            out.append(f"{key} {policy.get(key)!r}, frozen {frozen[key]!r}")
    if not (_number(policy.get("bound"))
            and math.isclose(policy["bound"], frozen["bound"], rel_tol=1e-12)):  # fmt: skip
        out.append(f"bound {policy.get('bound')!r}, frozen 0.01 × K* = {frozen['bound']!r}")
    if not (isinstance(policy.get("steps"), list) and policy["steps"] == frozen["steps"]):
        out.append(f"steps {policy.get('steps')!r}, frozen {frozen['steps']}")
    for key in ("gaussian", "reference", "twin_a_consistency"):  # the tool's (``_tool_policy``)
        if policy.get(key) != frozen[key]:
            out.append(f"{key} {policy.get(key)!r}, frozen {frozen[key]!r}")
    rule = NUMERICAL_POLICY["twin_a_consistency"]
    if frozen["twin_a_consistency"] != rule:  # the tool's tracked rule is not the frozen one
        out.append(f"the numerical-check tool's twin_a_consistency "
                   f"{frozen['twin_a_consistency']!r}, frozen {rule!r}")  # fmt: skip
    return out


def numerical_verdict(
    record: Any,
    batch: int,
    upper: float,
    rates: Mapping[str, Sequence[float]],
    commits: str | Collection[str],
    reuse: Callable[[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """A numerical-check verdict record read for one check of each arm's ``rates`` at ``batch``
    and ``upper``, whose numerical runs were made at ``commits`` (one commit or several). The
    record (the tool's schema): ``batch``, ``U``, ``rates`` (f"{optimizer}|{lr:g}" → a state of
    ``RATE_STATES``), ``seeds`` (the same keys → the seeds tested), ``controls`` (f"{lr:g}" → a
    state of ``CONTROL_STATES``), ``qualified``, ``reasons`` and ``policy`` (``norm``,
    ``survival_share``, ``retry_scale``, ``bound``, ``steps``, ``gaussian``, ``reference`` and
    ``twin_a_consistency``, as ``numerical_policy`` freezes them, ``tool_commit``: a
    non-empty list of the check's commits, or one commit in older records, and ``gpu_code``: the
    ``code_identity.gpu_code_identity`` of the code the check ran, ``commit``, ``files`` and
    ``sha256``). ``reuse(base, head)`` is ``code_identity.reuse_check`` on this repository (the
    default): the verdict holds at the running checkout only when each commit of the numerical
    runs has the verdict's GPU code (it is ``gpu_code.commit``, or ``reuse(commit,
    gpu_code.commit)`` is reusable) and ``reuse(gpu_code.commit, "HEAD")`` is reusable, so that a
    qualification survives analysis-only commits and no GPU-path change (review fa13); the
    checks are kept in ``reuse``. ``state``:

    - ``absent``: no record;
    - ``malformed``: not such a record (a state outside its states, ``qualified`` other than
      what its rates and controls give, no ``gpu_code`` or ``tool_commit``): no conclusion;
    - ``stale``: another batch or U, a policy other than the frozen one (``numerical_policy``),
      a GPU code other than the runs' or than the running checkout's, a ``gpu_code`` other than
      the identity of its commit, a ``tool_commit`` that names no commit, a requested (arm,
      rate) or an FA rate's control missing, or a rate's seeds short of its cases: the check is
      run again. Its marks persist (fix round 1): a requested rate it marks failing or not
      qualified, or an FA rate whose control it marks rejected or not qualified, stays in
      ``unusable`` at its batch whatever the staleness;
    - ``incomplete``: a requested rate or control incomplete: the check is finished, never
      passed;
    - ``complete``: ``unusable`` lists each requested (arm, rate) that is failing (a pair
      exceeding, or its numerical run stopped by an instability rule: evidence of instability)
      or not qualified (technical, with its reason, never instability), and each FA rate whose
      same-path control was rejected or not qualified.

    ``unfinished`` lists the requested rates and controls a stale or incomplete verdict holds
    incomplete, lacks, or tested on fewer seeds than their cases need: a re-solve is decided only
    from a verdict with none (the coordinator's ruling after fix round 1; re-review 2), so such
    a verdict's marks wait until the check is finished or rerun. A
    record may hold more rates than requested (a candidate's three-arm check read for its
    primary arms alone); only the requested ones are read."""
    commits = {commits} if isinstance(commits, str) else set(commits)
    out: dict[str, Any] = {"state": "absent", "reasons": [], "rates": {}, "controls": {},
                           "unusable": [], "unfinished": [], "policy": None,
                           "reuse": None}  # fmt: skip
    if record is None:
        out["reasons"].append("no verdict record")
        return out
    fields = ("batch", "U", "rates", "seeds", "controls", "qualified", "policy")
    if not (isinstance(record, dict) and all(key in record for key in fields)
            and all(isinstance(record[key], dict) for key in ("rates", "seeds", "controls",
                                                              "policy"))):  # fmt: skip
        return {**out, "state": "malformed",
                "reasons": [f"not a verdict record with {list(fields)}"]}  # fmt: skip
    gpu = record["policy"].get("gpu_code")
    if not ((gpu is None or (isinstance(gpu, dict) and isinstance(gpu.get("commit"), str)
                             and _integer(gpu.get("files")) and isinstance(gpu.get("sha256"), str)))
            and "gpu_code" in record["policy"] and "tool_commit" in record["policy"]
            and isinstance(record["policy"].get("reference"), str)):  # fmt: skip
        return {**out, "state": "malformed",
                "reasons": ["a policy without its gpu_code (commit, files, sha256, or null), "
                            "tool_commit and reference"]}  # fmt: skip
    bad = [f"{key}: {value!r}" for key, value in record["rates"].items()
           if value not in RATE_STATES]  # fmt: skip
    bad += [f"control {key}: {value!r}" for key, value in record["controls"].items()
            if value not in CONTROL_STATES]  # fmt: skip
    if not (_integer(record["batch"]) and _number(record["U"])):
        bad.append(f"batch {record['batch']!r} and U {record['U']!r}")
    qualified = (all(value == "qualified" for value in record["rates"].values())
                 and all(value == "accepted" for value in record["controls"].values()))  # fmt: skip
    if record["qualified"] is not qualified:
        bad.append(f"qualified {record['qualified']!r}, where its rates and controls give "
                   f"{qualified}")  # fmt: skip
    if bad:
        return {**out, "state": "malformed", "reasons": bad}
    policy = record["policy"]
    out["policy"] = policy
    needed: dict[str, set[int]] = defaultdict(set)
    for arm, lr, seed in numerical_cases(rates):
        needed[_pair_key(arm, lr)].add(seed)
    controls = {f"{lr:g}": rate(lr) for lr in rates.get(FA) or ()}
    out["rates"] = {key: record["rates"][key] for key in needed if key in record["rates"]}
    out["controls"] = {key: record["controls"][key] for key in controls
                       if key in record["controls"]}  # fmt: skip
    # The marks, read whatever the staleness when the verdict is of this batch.
    marks = _marks(record, rates, out) if record["batch"] == batch else []
    stale = []
    if (record["batch"], float(record["U"])) != (batch, upper):
        stale.append(f"a verdict at B={record['batch']}, U={record['U']:g}, not B={batch}, "
                     f"U={upper:g}")  # fmt: skip
    stale += [f"policy {departure}" for departure in _policy_departures(policy, upper)]
    if gpu is None:  # the tool found the regular runs' GPU code not its own: nothing counted
        stale.append("the verdict holds no GPU code (null): the check found the numerical "
                     "runs' GPU code not its own")  # fmt: skip
    else:
        found, departures = _gpu_code(gpu, commits, reuse or _reuse_check)
        out["reuse"] = found
        stale += departures
    tool = policy.get("tool_commit")
    if not (
        (isinstance(tool, str) and tool)
        or (isinstance(tool, list) and tool and all(isinstance(c, str) and c for c in tool))
    ):
        stale.append(f"the verdict's tool_commit {tool!r} names no commit")
    lacking = [key for key in needed if key not in record["rates"]]
    lacking += [f"control {key}" for key in controls if key not in record["controls"]]
    if lacking:
        stale.append(f"the verdict lacks {lacking}")
    short = []
    for key, seeds in needed.items():
        tested = record["seeds"].get(key)
        if key in record["rates"] and not (isinstance(tested, list) and seeds <= set(tested)):
            stale.append(f"{key} tested on {tested!r}, its cases need {sorted(seeds)}")
            short.append(f"{key} seeds")
    incomplete = [key for key, value in out["rates"].items() if value == "incomplete"]
    incomplete += [f"control {key}" for key, value in out["controls"].items()
                   if value == "incomplete"]  # fmt: skip
    # Evidence still to come: an incomplete entry, a missing rate or control, or seeds short of
    # their cases (re-review 2 of Task 2, new minor 2).
    out["unfinished"] = incomplete + lacking + short
    if stale:
        stale += [f"incomplete: {incomplete}"] if incomplete else []
        return {**out, "state": "stale", "reasons": stale, "unusable": marks}
    if incomplete:
        return {**out, "state": "incomplete", "reasons": [f"incomplete: {incomplete}"]}
    out.update(state="complete", unusable=marks)
    return out


def _marks(
    record: Mapping[str, Any], rates: Mapping[str, Sequence[float]], read: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The requested rates a verdict marks unusable (``read``: its requested ``rates`` and
    ``controls``), with their reasons, read as text only: by rate key, a control's under its FA
    rate's key prefixed "control: "."""
    reasons = record.get("reasons") if isinstance(record.get("reasons"), dict) else {}

    def told(key: str, control: bool) -> list[str]:
        found = reasons.get(key) if isinstance(reasons.get(key), list) else []
        return [str(r) for r in found if str(r).startswith(CONTROL_REASON) == control]

    pairs = {_pair_key(arm, lr): (arm, rate(lr)) for arm, lrs in rates.items() for lr in lrs}
    why = {"failing": "failing: a pair exceeding the bound",
           "not_qualified": "not qualified (technical)"}  # fmt: skip
    marks = [{"arm": arm, "lr": lr, "state": read["rates"][key], "why": why[read["rates"][key]],
              "reasons": told(key, False)}
             for key, (arm, lr) in pairs.items() if read["rates"].get(key) in why]  # fmt: skip
    for entry in marks:  # a numerical run stopped by an instability rule is failing
        if entry["state"] == "failing" and any(r.startswith("instability:")
                                               for r in entry["reasons"]):  # fmt: skip
            entry["why"] = "failing: its numerical run stopped by an instability rule"
    control_why = {"rejected": "its same-path control rejected",
                   "not_qualified": "its same-path control not qualified (technical)"}  # fmt: skip
    marks += [{"arm": FA, "lr": rate(lr), "state": f"control {read['controls'][f'{lr:g}']}",
               "why": control_why[read["controls"][f"{lr:g}"]],
               "reasons": told(_pair_key(FA, lr), True)}
              for lr in rates.get(FA) or ()
              if read["controls"].get(f"{lr:g}") in control_why]  # fmt: skip
    return marks


def numerical_digest(rates: Mapping[str, Sequence[float]]) -> str:
    """A selection's digest, the name of its verdict: the first 12 hex digits of the sha256 of
    its sorted f"{optimizer}|{lr:g}" keys joined by newlines (as the numerical-check tool keys
    it)."""
    return _digest(_pair_key(arm, lr) for arm, lrs in rates.items() for lr in lrs)


def _digest(keys: Collection[str] | Any) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:12]


def read_numerical(runs_root: str | Path) -> dict[str, Any]:
    """The numerical-check tool's verdict records under ``runs_root`` ($RUNS_DIR), by their path
    relative to it: numerical/b{B}-u{U}/verdicts/<digest>.json, one per selection checked, and
    the earlier verdicts of a selection, which the tool moves to
    verdicts/history/<digest>/ when it writes the selection's verdict again (the
    ``verdict.json`` beside them is a copy for people, never read). A file that does not read as
    JSON is kept as the text of its error, which ``numerical_verdict`` finds malformed."""
    root = Path(runs_root)
    out: dict[str, Any] = {}
    found = [*(root / NUMERICAL_DIR).glob(f"*/{VERDICTS_DIR}/*.json"),
             *(root / NUMERICAL_DIR).glob(f"*/{VERDICTS_DIR}/{HISTORY_DIR}/*/*.json")]  # fmt: skip
    for path in sorted(found):
        name = path.relative_to(root).as_posix()
        try:
            out[name] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            out[name] = f"unreadable: {exc}"
    return out


def _streams(batch: int) -> dict[str, Any]:
    """The numerical seeds' streams at the batch, which the numerical campaign needs first."""
    return {"seeds": list(NUMERICAL_SEEDS), "batch": batch, "training_responses": STEPS * batch,
            "preconditioner_responses": PRECONDITIONER_BATCHES * batch}  # fmt: skip


# ------------------------------------------------------------------ the state machine


class _Replay:
    """The bookkeeping of ``selection``'s replay: the discovery rates prescribed per arm and batch,
    the rounds spent per arm, batch and U, the rates found unusable per arm and batch (each with
    its mark and whether it is technical or scientific), the search allowances spent per arm,
    batch and U, the numerical and validation runs called for, each candidate's numerical checks,
    and the batches reached."""

    def __init__(self, runs, theta0, w_max, verdicts, facts, exhausted, reuse):
        self.found, self.theta0, self.w_max = runs, theta0, w_max
        self.verdicts, self.facts, self.exhausted = verdicts, facts, exhausted
        self.reuse = reuse
        self.rates: dict[tuple[str, int], set[float]] = defaultdict(set)
        for arm in PRIMARY_ARMS:
            for batch in BATCHES:
                self.rates[(arm, batch)] = {rate(lr) for lr in SHORT_GRID}
        self.rounds: dict[tuple[str, int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
        self.unusable: dict[tuple[str, int], dict[float, str]] = defaultdict(dict)
        # The marks behind ``unusable``, by rate: "technical" or "scientific" (scientific wins).
        self.kinds: dict[tuple[str, int], dict[float, str]] = defaultdict(dict)
        self.marks: dict[tuple[str, int], dict[float, dict[str, Any]]] = defaultdict(dict)
        # The search allowance spent per (arm, batch, U), kept across the candidate's visits,
        # and the arms an allowance-spent stop found stopped by technical marks only there.
        self.allowance: dict[tuple[str, int, float], dict[str, Any]] = {}
        self.stopped: dict[tuple[str, int, float], dict[str, Any]] = {}
        # The allowances owed: an arm marked by a check that stopped the candidate before the
        # arm's re-solve; a later visit's re-selection of it is that allowance.
        self.owed: dict[tuple[str, int, float], dict[str, Any]] = {}
        self.called: set[tuple] = set()  # the numerical and validation runs called for
        self.read: set[str] = set()  # the verdicts the replay read
        self.reached: set[int] = set()

    def mark(self, batch: int, unusable: Mapping[str, Any], why: str) -> None:
        """A verdict's mark: the rate leaves the batch for every U, technical or scientific."""
        arm, lr = unusable["arm"], unusable["lr"]
        self.unusable[(arm, batch)].setdefault(lr, why)
        self.marks[(arm, batch)].setdefault(lr, {**unusable, "why": why})
        if self.kinds[(arm, batch)].get(lr) != "scientific":
            self.kinds[(arm, batch)][lr] = _kind(unusable)

    def technical(self, arm: str, batch: int) -> list[float]:
        """The arm's rates at the batch excluded by technical marks only."""
        return sorted(lr for lr, kind in self.kinds[(arm, batch)].items() if kind == "technical")

    def cause(self, arm: str, batch: int, upper: float, excluded: Collection[float]) -> str:
        """Why an arm has no feasible triple once ``excluded`` are dropped: ``technical`` when it
        has one with its technical exclusions lifted, the scientific ones still dropped;
        ``scientific`` otherwise (no admissible triple)."""
        technical = set(self.technical(arm, batch))
        if not technical & set(excluded):
            return "scientific"
        kept = sorted(lr for lr in excluded if lr not in technical)
        return "technical" if self.solve(arm, batch, upper, kept)["feasible"] else "scientific"

    def records(self, arm: str, batch: int) -> dict[float, dict[str, Any]]:
        return {
            lr: usable_rate([self.found[(arm, batch, lr, s)] for s in DISCOVERY_SEEDS
                             if (arm, batch, lr, s) in self.found], self.theta0)
            for lr in sorted(self.rates[(arm, batch)])
        }  # fmt: skip

    def solve(self, arm, batch, upper, exclude=()) -> dict[str, Any]:
        return select_rates(self.records(arm, batch), upper, self.w_max, exclude=exclude)

    def missing(self, arm: str, batch: int, rates: Sequence[float]) -> list[dict[str, Any]]:
        return [{"arm": arm, "batch": batch, "lr": lr, "seed": s} for lr in rates
                for s in DISCOVERY_SEEDS if (arm, batch, lr, s) not in self.found]  # fmt: skip

    def propose(self, kind, arm, batch, upper, selected, unusable=()) -> dict[str, Any]:
        """The arm's ``kind`` round at the batch and U, over every run made at that batch (a
        repair round proposing none of ``unusable``: the search allowance's)."""
        records = self.records(arm, batch)
        if kind == "targeted":
            proposal = targeted_rates(records, upper)
            proposed = [t["rate"] for t in proposal["targets"] if t["rate"] is not None]
        else:
            proposal = repair_rates(selected["best_unconstrained"], records, upper, self.w_max,
                                    unusable=unusable)  # fmt: skip
            proposed = proposal["rates"]
        new = sorted(set(proposed) - self.rates[(arm, batch)])
        return {"kind": kind, "arm": arm, "batch": batch, "U": upper, "proposal": proposal,
                "new_rates": new}  # fmt: skip

    def round(self, kind, arm, batch, upper, selected) -> tuple[dict[str, Any], bool]:
        """The arm's ``kind`` round at the batch and U: the one spent at an earlier visit of the
        candidate, else a new proposal (True), not yet spent."""
        spent = self.rounds[(arm, batch, upper)].get(kind)
        if spent is not None:
            return spent, False
        return self.propose(kind, arm, batch, upper, selected), True

    def allowance_round(self, arm, batch, upper, selected, check: str, unusable) -> dict:
        """The search allowance's repair round of the arm at the batch and U: ``repair_rates``
        from the re-solve's best unconstrained triple, over every run made at the batch, never
        proposing one of the arm's rates ``unusable`` at the batch (review fa22, ruling on C2);
        spent after the check ``check`` (its digest), which names its campaign."""
        record = self.propose("repair", arm, batch, upper, selected, unusable)
        return {**record, "kind": ALLOWANCE, "after": check}

    def spend(self, record: dict[str, Any], *, prescribe: bool = True) -> None:
        """Record a round; prescribe its new rates unless the candidate fails on it."""
        arm, batch = record["arm"], record["batch"]
        self.rounds[(arm, batch, record["U"])][record["kind"]] = record
        if prescribe:
            self.rates[(arm, batch)] |= set(record["new_rates"])
        else:
            record["new_rates"] = []

    def first(self, batch: int, upper: float, initial: Sequence[float], *, joint: bool) -> dict:
        """The pre-warmed arm's first round at the batch and U, at the rates ``initial``: in the
        joint discovery campaign (``joint``) or alone. Spent once; a later call returns it."""
        spent = self.rounds[(PREWARMED, batch, upper)].get("first")
        if spent is not None:
            return spent
        how = ("the union of the primary arms' targeted assignments" if joint
               else "the union of the primary arms' selected rates")  # fmt: skip
        record = {"kind": "first", "arm": PREWARMED, "batch": batch, "U": upper,
                  "proposal": {"rates": list(initial), "joint": joint, "reasons": [how]},
                  "new_rates": sorted(set(initial) - self.rates[(PREWARMED, batch)])}  # fmt: skip
        self.spend(record)
        return record

    def check(self, batch: int, upper: float, rates: Mapping[str, Sequence[float]]) -> dict:
        """The numerical check of ``rates`` at the candidate: the tool's directory
        numerical/b{B}-u{U} and the path of the selection's verdict, verdicts/<digest>.json
        (``numerical_digest``), under $RUNS_DIR."""
        directory = f"{NUMERICAL_DIR}/b{batch}-u{upper:g}"
        digest = numerical_digest(rates)
        return {"name": digest, "dir": directory,
                "path": f"{directory}/{VERDICTS_DIR}/{digest}.json",
                "rates": {arm: list(lrs) for arm, lrs in rates.items()}}  # fmt: skip

    def verdicts_for(self, check: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """The verdicts that answer the check: the one of its selection's digest; without it,
        those of the candidate whose rates for the requested arms are exactly the request's,
        holding other arms too (a three-arm check read for the primary arms alone, review
        fa13). Each is marked read."""
        found = []
        if check["path"] in self.verdicts:
            found = [check["path"]]
        else:
            wanted = {_pair_key(arm, lr) for arm, lrs in check["rates"].items() for lr in lrs}
            names = {OPTIMIZERS[arm] for arm in check["rates"]}
            prefix = f"{check['dir']}/{VERDICTS_DIR}/"
            for path in sorted(self.verdicts):
                rates = (self.verdicts[path] or {}).get("rates") if isinstance(
                    self.verdicts[path], dict) else None  # fmt: skip
                if path.startswith(prefix) and "/" not in path[len(prefix):] and isinstance(
                        rates, dict) and {key for key in rates
                                          if key.split("|")[0] in names} == wanted:  # fmt: skip
                    found.append(path)
        self.read |= set(found)
        return [(path, self.verdicts[path]) for path in found]

    def history_of(self, path: str) -> list[tuple[str, Any]]:
        """The earlier verdicts of the selection whose current verdict is ``path``
        (verdicts/history/<digest>/), each marked read."""
        directory, name = path.rsplit("/", 1)
        prefix = f"{directory}/{HISTORY_DIR}/{name.removesuffix('.json')}/"
        found = sorted(p for p in self.verdicts or {} if p.startswith(prefix))
        self.read |= set(found)
        return [(p, self.verdicts[p]) for p in found]

    def unexpected(self) -> tuple[list[str], list[str]]:
        """The runs no visit called for: problems at a batch the replay reached; at a batch it has
        not reached yet (amendment 3's discovery at B = 5), deferred to the visits there."""
        problems, deferred = [], []
        for key in sorted(self.found):
            arm, batch, lr, seed = key
            if seed in DISCOVERY_SEEDS:
                called = lr in self.rates.get((arm, batch), set())
            else:
                called = key in self.called
            if not called:
                if batch in self.reached:
                    problems.append(f"unexpected run {key}")
                else:
                    deferred.append(f"run {key} at a batch not reached yet")
        return problems, deferred


def selection(
    trajectories: Sequence[Trajectory],
    theta0: Mapping[str, float],
    w_max: float,
    *,
    seconds_per_chunk: float | None = None,
    numerical: Mapping[str, Any] | None = None,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
    exhausted: Mapping[tuple, str] | None = None,
    reuse: Callable[[str, str], Mapping[str, Any]] | None = None,
    issued_w_max: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The selection's state machine (amendment 4), replayed from every run so far and the
    numerical-check verdicts: the first candidate (U, B) of ``CANDIDATES`` not yet decided, the
    campaign that comes next, with every attempt recorded. Never reads a judge label or
    K_harmful. It refuses to replay (``incomplete``) under a ``w_max`` other than one an issued
    campaign's header recorded (``issued_w_max``, by run label; ``w_max_problems``).

    Per candidate, for the three arms: each primary arm's selection over its runs at B; for an
    arm with no feasible triple its targeted round, the joint ``discovery`` campaign that also
    holds the pre-warmed arm's first round, then its repair round (at most one each per arm,
    batch and U); then the pre-warmed arm's first round (if not made) and its own rounds; then
    the numerical gate: the ``numerical`` campaign (the K-only regular runs of the cases on
    seeds 2200-2202), then ``numerical-check`` (the tool's check and the verdict it writes), an
    unusable rate leaving B and its arm spending its search allowance (a re-solve, then, without
    a feasible triple, one ``repair`` round and a re-selection); then the ``validation``
    campaign (seeds 2100-2111) and the three-arm validation verdict. A scientific failure at any
    step takes the next candidate; an arm stopped by technical marks only ends the replay
    ``blocked`` (``next`` ``incomplete`` with ``blocked`` "technical", for a human decision).
    When no candidate (B = 20 only) is eligible with the three arms, the candidates whose primary
    arms did not fail at the numerical gate are visited again, in the same order, for the
    primary pair alone (the pre-warmed arm dropped, amendment 3's primary-arm validation),
    reusing every record and keeping each primary arm's spent allowance; when none is eligible
    either, the stage ends ``infeasible`` with ``scope`` "B=20", every attempt's reasons and its
    ``feasibility`` report (``feasibility_report``). ``numerical``: the
    verdict records by their path under $RUNS_DIR (``read_numerical``), None when not read;
    ``facts``: ``run_facts`` by run id (the commit and checkout state of each numerical run);
    ``exhausted``: the numerical and validation runs the launcher gave up on
    (``exhausted_runs``); ``reuse``: the GPU-code reuse check (``numerical_verdict``),
    ``code_identity.reuse_check`` on this repository by default. ``next["campaign"]`` is one of
    ``discovery``, ``repair``, ``prewarmed``, ``targeted``, ``numerical``, ``numerical-check``,
    ``validation`` (with what to make or check; a ``repair`` of the search allowance names the
    check that spent it, ``allowance``), ``done`` (with the frozen rates, ``three_arm`` and
    ``prewarmed_dropped``), ``infeasible`` or ``incomplete``: ``blocked`` (technical marks
    only, see above), ambiguous records, or any ``problems`` (review of Task 2, minor 5), the
    replay's own step then kept as ``pending`` (and, at a stale check with runs beyond it,
    ``beyond_stale``)."""
    seeds = (*DISCOVERY_SEEDS, *NUMERICAL_SEEDS, *VALIDATION_SEEDS)
    found, problems = index(r for r in trajectories if r.arm in ARMS and r.seed in seeds)
    problems = w_max_problems(w_max, issued_w_max) + problems
    controls: dict[tuple[int, float, float | None], Trajectory] = {}
    for run in (r for r in trajectories if r.arm == FA_CONTROL_ARM):
        key = _control_key(run)
        if run.seed != CONTROL_SEED:
            problems.append(f"control {run.run_id} on seed {run.seed}, not {CONTROL_SEED}")
        elif key in controls:
            problems.append(f"two controls of {key}: {controls[key].run_id} and {run.run_id}")
        else:
            controls[key] = run
    stream_time = None
    if seconds_per_chunk is None:  # amendment 1's reference, with its provenance
        stream_time, missing = stream_time_reference(trajectories)
        problems += missing
        seconds_per_chunk = stream_time["seconds_per_chunk"] if stream_time else None
    out: dict[str, Any] = {"w_max": w_max, "seconds_per_chunk": seconds_per_chunk,
                           "stream_time": stream_time, "rulings": RULINGS}  # fmt: skip
    if problems:
        out.update(next={"campaign": "incomplete"}, problems=problems)
        return out
    replay = _Replay(found, theta0, w_max, numerical, facts, dict(exhausted or {}), reuse)
    out["candidates"] = candidate_order(list(found.values()), theta0, seconds_per_chunk)
    out["amendment3_controls"] = _amendment3_controls(controls, found)
    out["attempts"] = attempts = []
    step = None
    for arms in (ARMS, PRIMARY_ARMS):  # the three arms; then the fallback, the primary pair
        for upper, batch in CANDIDATES:
            if arms == PRIMARY_ARMS:
                first = next(a for a in attempts if (a["batch"], a["U"]) == (batch, upper))
                if first["failed_at"] == "numerical" and set(
                        first["failed_arms"]) & set(PRIMARY_ARMS):  # fmt: skip
                    continue  # its primary arms failed their numerical gate: so they do alone
            # A primary arm's search allowance spent in pass 1 stays spent (``_Replay``).
            attempt, step = _visit(replay, batch, upper, arms)
            attempts.append(attempt)
            if step is not None or attempt["outcome"] == "eligible":
                break
        if step is not None or attempts[-1]["outcome"] == "eligible":
            break
    chosen = attempts[-1]
    if step is None and chosen["outcome"] == "eligible":
        step = {"campaign": "done", "batch": chosen["batch"], "U": chosen["U"],
                "K_star": chosen["K_star"],
                "rates": {arm: chosen["rates"].get(arm) for arm in ARMS},
                "three_arm": chosen["three_arm"], "prewarmed_dropped": not chosen["three_arm"],
                "numerical": [{k: e.get(k) for k in ("check", "verdict", "state", "unusable",
                                                     "policy", "reuse")}
                              for e in chosen["numerical"]],
                "validation": chosen["validation"]}  # fmt: skip
    elif step is None:  # no candidate eligible, even for the primary pair: a feasibility report
        step = {"campaign": "infeasible", "scope": SCOPE, "reasons": [
            f"U={a['U']:g}, B={a['batch']}, {'three arms' if a['three_arm'] else 'primary arms'}"
            f": {'; '.join(a['reasons'])}" for a in attempts],
            "feasibility": feasibility_report(attempts)}  # fmt: skip
    out["next"] = step
    out["rounds"] = {f"{arm}/B={batch}/U={upper:g}": kinds
                     for (arm, batch, upper), kinds in replay.rounds.items()}  # fmt: skip
    unexpected, out["not_reached"] = replay.unexpected()
    unread, against = _unread_verdicts(replay)
    out["unread_verdicts"] = unread
    out["problems"] = out["next"].get("problems", []) + unexpected + against
    if step["campaign"] == "numerical-check" and step["state"] == "stale" and unexpected:
        # The runs beyond a stale check wait for it: the check's input may still be written.
        waits = (f"the replay waits at the stale check {step['check']['verdict']} "
                 f"({'; '.join(step['reasons'])}): the runs beyond it are judged once it is "
                 "rerun")  # fmt: skip
        out["problems"].insert(0, waits)
        out["beyond_stale"] = [waits, *unexpected]
    if out["problems"]:  # no campaign on records that are not whole; the step stays recorded
        out["pending"], out["next"] = out["next"], {"campaign": "incomplete"}
    return out


_VERDICT_PATH = re.compile(
    rf"^{NUMERICAL_DIR}/b(\d+)-u[^/]+/{VERDICTS_DIR}/(?:{HISTORY_DIR}/[^/]+/)?[^/]+\.json$"
)


def _unread_verdicts(replay: _Replay) -> tuple[list[str], list[str]]:
    """The verdicts at a batch the replay reached that it did not read, and the problems among
    them: one marking a rate unusable (failing, not qualified, a control rejected or not
    qualified) leaves no conclusion, since a rate unusable at a batch is unusable at every U."""
    unread, problems = [], []
    for path in sorted(replay.verdicts or {}):
        found = _VERDICT_PATH.match(path)
        if path in replay.read or found is None or int(found[1]) not in replay.reached:
            continue
        unread.append(path)
        record = replay.verdicts[path]
        rates = record.get("rates") if isinstance(record, dict) else None
        controls = record.get("controls") if isinstance(record, dict) else None
        marked = [key for key, value in (rates or {}).items()
                  if value in ("failing", "not_qualified")]  # fmt: skip
        marked += [f"control {key}" for key, value in (controls or {}).items()
                   if value in ("rejected", "not_qualified")]  # fmt: skip
        if not isinstance(rates, dict):
            problems.append(f"{path}: a verdict the selection did not read, unreadable: no "
                            "conclusion")  # fmt: skip
        elif marked:
            problems.append(f"{path}: a verdict the selection did not read marks {marked} "
                            "unusable at its batch: no conclusion")  # fmt: skip
    return unread, problems


def _step(campaign: str, attempt: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    return {"campaign": campaign, "batch": attempt["batch"], "U": attempt["U"],
            "K_star": attempt["K_star"], "three_arm": attempt["three_arm"], **fields}  # fmt: skip


def _kind(unusable: Mapping[str, Any]) -> str:
    """A verdict's unusable mark (``_marks``): ``technical`` (not qualified, or an FA rate whose
    control is not qualified) or ``scientific`` (failing, or a rejected control)."""
    return "technical" if unusable["state"] in TECHNICAL_MARKS else "scientific"


def _fail(attempt: dict[str, Any], arms: Collection[str], reason: str, at: str) -> None:
    """The attempt fails on ``arms`` at the gate ``at`` (discovery, numerical or validation),
    with the reason: a scientific failure (*Technical and scientific failures*), so the order
    moves on."""
    attempt.update(outcome="failed", failed_at=at)
    attempt["failed_arms"] = [arm for arm in ARMS if arm in {*attempt["failed_arms"], *arms}]
    attempt["reasons"].append(reason)
    for arm in arms:
        attempt["failures"].setdefault(arm, []).append({"at": at, "reason": reason})


def _block(
    replay: _Replay, attempt: dict[str, Any], stops: Mapping[str, Sequence[Mapping[str, Any]]],
    reason: str, at: str,
) -> dict[str, Any]:  # fmt: skip
    """The attempt stopped by technical marks only (*Technical and scientific failures*): the
    replay ends ``blocked`` for a human decision, the order does not move on. ``stops``: by
    arm, the technical marks that stop it. Returns the step, with each blocked arm's search
    allowance at the candidate (``allowances``: None when it spent none here)."""
    rates = [{k: m.get(k) for k in ("arm", "lr", "state", "why", "reasons")}
             for arm in ARMS for m in stops.get(arm, ())]  # fmt: skip
    attempt.update(outcome="blocked", blocked_at=at,
                   blocked_arms=[arm for arm in ARMS if arm in stops])  # fmt: skip
    attempt["reasons"].append(reason)
    return _step("incomplete", attempt, blocked="technical", at=at,
                 arms=attempt["blocked_arms"], rates=rates, reasons=[reason],
                 allowances={arm: replay.allowance.get((arm, attempt["batch"], attempt["U"]))
                             for arm in attempt["blocked_arms"]},
                 decision="a technical non-qualification never decides the candidate: the "
                          "stage stops here for a human decision")  # fmt: skip


def _technical_marks(replay: _Replay, arm: str, batch: int) -> list[dict[str, Any]]:
    """The arm's technical exclusions at the batch, as their marks."""
    return [replay.marks[(arm, batch)][lr] for lr in replay.technical(arm, batch)]


def _stop(
    replay, attempt, causes: Mapping[str, tuple[str, list]], reason: str, at: str
) -> dict[str, Any] | None:
    """Arms that cannot go on, ``causes`` by arm: (``scientific`` or ``technical``, the marks
    that stop it). Any scientific cause fails the candidate (its arms; the technical ones are
    named in the reason); otherwise the replay is blocked. Either way each arm stopped
    technically is kept with its marks at the candidate (``_Replay.stopped``), so that a later
    visit of the candidate (the fallback's) blocks on it rather than select the arm again
    (review of fa22, I1: one allowance in all per arm, batch and U). Returns the blocked step,
    or None when the attempt failed."""
    scientific = [arm for arm, (cause, _) in causes.items() if cause == "scientific"]
    technical = {arm: marks for arm, (cause, marks) in causes.items() if cause == "technical"}
    rates = {arm: [format(m["lr"], "g") for m in marks] for arm, marks in technical.items()}
    told = "; ".join(f"{arm} stopped by technical marks only ({rates[arm]})" for arm in technical)
    for arm, marks in technical.items():
        replay.stopped.setdefault((arm, attempt["batch"], attempt["U"]),
                                  {"at": at, "marks": marks, "reason": told,
                                   "three_arm": attempt["three_arm"]})  # fmt: skip
    attempt["stopped_technical"] = sorted(technical, key=ARMS.index)
    if scientific:
        _fail(attempt, scientific, reason + (f"; besides, {told}" if told else ""), at)
        return None
    return _block(replay, attempt, technical, f"{reason}: {told}", at)


def feasibility_report(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The B = 20 feasibility report of a stage that ends ``infeasible`` (review fa22): no
    confirmation. Every candidate of the fixed order, (8, 20) and (7, 20) for the three arms,
    then for the primary pair, with its outcome and reasons and, per arm, what failed and why:
    the ``scientific`` failures (a failing rate, a rejected same-path control, no admissible
    triple, a validation failure and its findings: they moved the order on) apart from the
    ``technical`` ones (rates not qualified and numerical runs given up, which spent an
    allowance and decided nothing; and the validation's blocks and utility members lost by
    technical causes, listed apart: a validation failing with them failed for another reason
    too, since a failure they alone make blocks the stage), and the rates excluded up front at the
    visit with their kind. A candidate the fallback skipped is named, with why."""
    report = []
    for scope, arms in (("three arms", ARMS), ("primary pair", PRIMARY_ARMS)):
        for upper, batch in CANDIDATES:
            key = (batch, upper, arms == ARMS)
            found = next((a for a in attempts if (a["batch"], a["U"], a["three_arm"]) == key),
                         None)  # fmt: skip
            entry: dict[str, Any] = {"U": upper, "batch": batch, "scope": scope}
            if found is None:
                entry.update(visited=False, reasons=[
                    "not visited: its primary arms failed their numerical gate with the three "
                    "arms (a second check would be a second allowance)"])  # fmt: skip
            else:
                entry.update(visited=True, outcome=found["outcome"], failed_at=found["failed_at"],
                             reasons=list(found["reasons"]),
                             arms={arm: _arm_feasibility(found, arm) for arm in arms})  # fmt: skip
            report.append(entry)
    return {"scope": SCOPE, "confirmation": False,
            "outcome": "no candidate at B = 20 qualifies: a B = 20 feasibility report, not a "
                       "confirmation", "candidates": report}  # fmt: skip


def _arm_feasibility(attempt: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """One arm of one attempt in the feasibility report: its failures, scientific and technical."""
    marks: dict[tuple[float, str], Mapping[str, Any]] = {}
    for entry in attempt["numerical"]:
        for mark in entry.get("unusable") or ():
            if mark["arm"] == arm:
                marks.setdefault((mark["lr"], mark["state"]), mark)
    found: dict[str, list[str]] = {"scientific": [], "technical": []}
    for failure in attempt["failures"].get(arm, []):
        found["scientific"].append(f"{failure['at']}: {failure['reason']}")
    for (lr, _), mark in sorted(marks.items()):
        told = f" ({'; '.join(mark['reasons'])})" if mark.get("reasons") else ""
        found[_kind(mark)].append(f"numerical {lr:g}: {mark['why']}{told}")
    found["technical"] += [f"numerical run given up by the launcher: {label} {key}"
                           for entry in attempt["numerical"]
                           for label, key in (entry.get("given_up") or {}).items()
                           if key[0] == arm]  # fmt: skip
    verdict = attempt.get("validation")
    if verdict and verdict.get("passes") is False:
        for kind, lines in _validation_feasibility(verdict, arm).items():
            found[kind] += lines
    kinds = attempt.get("excluded_kinds", {}).get(arm, {})
    return {"failed": arm in attempt["failed_arms"], "blocked": arm in attempt.get(
        "blocked_arms", ()), **found,
        "excluded": {f"{lr:g}": {"kind": kinds.get(lr), "why": why}
                     for lr, why in sorted(attempt["excluded"].get(arm, {}).items())}}  # fmt: skip


def _validation_feasibility(verdict: Mapping[str, Any], arm: str) -> dict[str, list[str]]:
    """A failed validation's findings for one arm, classed by their cause (review of fa22, M4):
    a block unsupported by K (``bracketing`` or ``width``), an instability stop and a utility
    bound above its limit are scientific, and so is a block whose failures hold an instability
    stop; a block whose cause is ``technical`` (a failed or missing endpoint) otherwise, a run's
    technical failure and a utility member left without a bound are technical."""
    found: dict[str, list[str]] = {"scientific": [], "technical": []}
    for seed, by_arm in sorted((verdict.get("supports") or {}).items()):
        support = by_arm.get(arm)
        if support is None or support["supported"]:
            continue
        # A block lost with an instability stop among its failures is scientific, whatever its
        # cause (the second re-review of fa22, minor 2).
        technical = support["cause"] == "technical" and "instability" not in support[
            "failure_kinds"]  # fmt: skip
        kind = "technical" if technical else "scientific"
        found[kind].append(f"validation block {seed}: unsupported ({support['cause']}; by K "
                           f"alone: {support['k_cause']})")  # fmt: skip
        for failure in support["failures"]:
            line = (f"validation block {seed}: {failure['run'] or 'no run'} at "
                    f"{failure['rate']:g}: {failure['reason']}")  # fmt: skip
            found["scientific" if failure["kind"] == "instability" else "technical"].append(line)
    for member in (verdict.get("utility") or {}).get("family", []):
        if member["arm"] == arm and not (member["mmlu_passes"] and member["factual_passes"]):
            bounded = all(member[f"{m}_upper_pp"] is not None for m in ("mmlu", "factual"))
            found["scientific" if bounded else "technical"].append(
                f"validation utility at {member['rate']:g}: upper bounds MMLU "
                f"{member['mmlu_upper_pp']} pp, factual {member['factual_upper_pp']} pp "
                f"({member['runs']} runs)")  # fmt: skip
    found["technical"] += [f"validation run given up by the launcher: {label} {key}"
                           for label, key in (verdict.get("exhausted") or {}).items()
                           if key[0] == arm]  # fmt: skip
    return found


def _visit(
    replay: _Replay, batch: int, upper: float, arms: Sequence[str]
) -> tuple[dict, dict | None]:
    """One visit of a candidate for ``arms`` (the three arms, or the primary pair alone in the
    fallback): discovery, the numerical gate, the validation gate. An arm whose search allowance
    at the candidate was spent at an earlier visit (pass 1's, for the fallback) keeps it spent:
    ``allowance_carried`` (review of Task 2, critical 1; review fa22). Returns the attempt and the
    pending step, if any; the attempt's outcome is ``pending``, ``incomplete``, ``failed`` (with
    the arms it failed on: a scientific failure), ``blocked`` (technical marks only: the step is
    the blocked one) or ``eligible`` (with the rates)."""
    replay.reached.add(batch)
    # Rates found unusable at this batch earlier in the replay (any U) are excluded up front.
    excluded = {arm: dict(replay.unusable[(arm, batch)]) for arm in arms}
    carried = [arm for arm in arms if (arm, batch, upper) in replay.allowance]
    attempt: dict[str, Any] = {"batch": batch, "U": upper, "K_star": k_star(upper),
                               "three_arm": PREWARMED in arms, "resolve_carried": bool(carried),
                               "allowance_carried": carried,
                               "allowance": {arm: {**replay.allowance[(arm, batch, upper)],
                                                   "carried": True} for arm in carried},
                               "arms": {arm: {"selections": [], "rounds": []} for arm in arms},
                               "excluded": {arm: dict(e) for arm, e in excluded.items()},
                               "excluded_kinds": {arm: {lr: replay.kinds[(arm, batch)].get(lr)
                                                        for lr in e}
                                                  for arm, e in excluded.items()},
                               "numerical": [], "validation": None, "outcome": "pending",
                               "failed_arms": [], "failed_at": None, "failures": {},
                               "stopped_technical": [], "allowance_owed": [], "reasons": [],
                               "rates": None}  # fmt: skip
    # An arm whose allowance here is spent and that an earlier visit's numerical gate found
    # stopped by technical marks only (another arm's scientific failure then failing the
    # candidate) is never selected again: the stage blocks (review of fa22, I1).
    stopped = {arm: replay.stopped[(arm, batch, upper)] for arm in carried
               if replay.stopped.get((arm, batch, upper), {}).get("at") == "numerical"}  # fmt: skip
    if stopped:
        told = "; ".join(entry["reason"] for entry in stopped.values())
        return attempt, _block(replay, attempt, {arm: e["marks"] for arm, e in stopped.items()},
                               f"{list(stopped)} stopped by technical marks only at this "
                               "candidate's numerical gate at an earlier visit, the search "
                               f"allowance spent there: no selection again ({told})",
                               "numerical")  # fmt: skip
    # An arm marked at an earlier visit by a check that stopped the candidate before its
    # re-solve owes its allowance: this visit's re-selection of it is that allowance (the
    # second re-review of fa22, minor 1), spent first; after it, the arm is carried.
    owed = [
        arm
        for arm in arms
        if (arm, batch, upper) in replay.owed and (arm, batch, upper) not in replay.allowance
    ]
    for check in sorted({replay.owed[(arm, batch, upper)]["check"] for arm in owed}):
        group = [arm for arm in owed if replay.owed[(arm, batch, upper)]["check"] == check]
        debt = {arm: replay.owed[(arm, batch, upper)] for arm in group}
        stops, step = _spend_allowance(replay, attempt, excluded, batch, upper, {}, group, check,
                                       debt[group[0]]["verdict"],
                                       {arm: d["marks"] for arm, d in debt.items()},
                                       deferred=True)  # fmt: skip
        if stops:
            return attempt, step
    carried = [*carried, *owed]
    selected, step = _discovery(replay, attempt, excluded, batch, upper, arms, carried)
    if selected is None:
        return attempt, step
    selected, controls, step = _numerical(replay, attempt, excluded, batch, upper, selected, arms)
    if selected is None:
        return attempt, step
    rates = {arm: selected[arm]["rates"] for arm in arms}
    step = _validation(replay, attempt, batch, upper, rates, controls)
    if step is None and attempt["outcome"] != "failed":
        attempt.update(outcome="eligible", rates=rates)
    return attempt, step


def _causes(replay, batch, upper, arms, excluded) -> dict[str, tuple[str, list]]:
    """Arms left without a feasible triple once ``excluded`` are dropped: each one's cause
    (``_Replay.cause``) with its technical marks, which stop it when the cause is technical."""
    return {arm: (replay.cause(arm, batch, upper, excluded[arm]),
                  _technical_marks(replay, arm, batch)) for arm in arms}  # fmt: skip


def _discovery(
    replay, attempt, excluded, batch, upper, arms, carried: Collection[str] = ()
) -> tuple[dict | None, dict | None]:
    """A candidate's discovery: the primary arms' targeted round, as the joint ``discovery``
    campaign holding the pre-warmed arm's first round when it is due, then their repair round;
    then the pre-warmed arm's first round, if not made yet, and its own rounds. An arm of
    ``carried`` (its allowance at the candidate spent at an earlier visit) left without a triple
    by the unusable rates gets no new round. An arm that cannot go on stops the candidate, failed
    or blocked by its cause (``_stop``). Returns each arm's selection, or None with the pending
    step (none when the attempt failed)."""

    def solve(arm: str) -> dict[str, Any]:
        selected = replay.solve(arm, batch, upper, sorted(excluded[arm]))
        attempt["arms"][arm]["selections"].append(selected)
        return selected

    def pending(campaign: str, runs: list[dict[str, Any]]) -> tuple[None, dict]:
        attempt["outcome"] = "pending"
        return None, _step(campaign, attempt, runs=runs)

    def stop(stopped: Sequence[str], reason: str, at: str) -> tuple[None, dict | None]:
        causes = _causes(replay, batch, upper, stopped, excluded)
        return None, _stop(replay, attempt, causes, reason, at)

    prewarmed = PREWARMED in arms
    selected = {arm: solve(arm) for arm in arms}
    for kind in ("targeted", "repair"):
        going = [arm for arm in PRIMARY_ARMS if not selected[arm]["feasible"]]
        if not going:
            break
        rounds, fresh = [], []
        for arm in going:
            record, new = replay.round(kind, arm, batch, upper, selected[arm])
            rounds.append(record)
            if new:
                fresh.append(record)
        spent = [r["arm"] for r in fresh if r["arm"] in carried]
        if spent:  # the allowance is spent: no new round for it
            return stop(spent, f"{spent} without a feasible triple once the unusable rates are "
                        "dropped, the candidate's search allowance spent at an earlier visit",
                        "numerical")  # fmt: skip
        unrepairable = [r["arm"] for r in rounds if r["proposal"].get("unrepairable")]
        for spent_round in rounds:
            attempt["arms"][spent_round["arm"]]["rounds"].append(spent_round)
        for spent_round in fresh:
            replay.spend(spent_round, prescribe=not unrepairable)
        if unrepairable:
            return stop(unrepairable, f"{unrepairable}: a coverage miss that re-targeting cannot "
                        "repair", "discovery")  # fmt: skip
        first = None
        if kind == "targeted" and prewarmed and not selected[PREWARMED]["feasible"]:
            # The joint campaign: the pre-warmed arm's first round at the union of the rates the
            # primary arms' targeted assignments use (a feasible arm's: its selected triple).
            assigned = {t["rate"] for r in rounds for t in r["proposal"]["targets"]
                        if t["rate"] is not None}  # fmt: skip
            assigned |= {lr for arm in PRIMARY_ARMS if arm not in going
                         for lr in selected[arm]["rates"]}  # fmt: skip
            first = replay.first(batch, upper, sorted(assigned), joint=True)
            attempt["arms"][PREWARMED]["rounds"].append(first)
        missing = [m for r in rounds for m in replay.missing(r["arm"], batch, r["new_rates"])]
        if first is not None:
            missing += replay.missing(PREWARMED, batch, first["new_rates"])
        if missing:
            return pending("discovery" if kind == "targeted" else "repair", missing)
        for arm in going:
            selected[arm] = solve(arm)
        if first is not None:
            selected[PREWARMED] = solve(PREWARMED)
    infeasible = [arm for arm in PRIMARY_ARMS if not selected[arm]["feasible"]]
    if infeasible:
        return stop(infeasible, f"{infeasible} without a feasible triple after the rounds",
                    "discovery")  # fmt: skip
    if not prewarmed:
        return selected, None
    if not selected[PREWARMED]["feasible"] and "first" not in replay.rounds[(PREWARMED, batch,
                                                                             upper)]:  # fmt: skip
        initial = sorted({lr for arm in PRIMARY_ARMS for lr in selected[arm]["rates"]})
        first = replay.first(batch, upper, initial, joint=False)
        attempt["arms"][PREWARMED]["rounds"].append(first)
        missing = replay.missing(PREWARMED, batch, first["new_rates"])
        if missing:
            return pending("prewarmed", missing)
        selected[PREWARMED] = solve(PREWARMED)
    for kind in ("targeted", "repair"):
        if selected[PREWARMED]["feasible"]:
            break
        spent_round, new = replay.round(kind, PREWARMED, batch, upper, selected[PREWARMED])
        attempt["arms"][PREWARMED]["rounds"].append(spent_round)
        unrepairable = bool(spent_round["proposal"].get("unrepairable"))
        if new:
            replay.spend(spent_round, prescribe=not unrepairable)
        if unrepairable:
            break
        missing = replay.missing(PREWARMED, batch, spent_round["new_rates"])
        if missing:
            return pending(kind, missing)
        selected[PREWARMED] = solve(PREWARMED)
    if not selected[PREWARMED]["feasible"]:
        return stop([PREWARMED], "the pre-warmed arm without a feasible triple after its rounds",
                    "discovery")  # fmt: skip
    return selected, None


def _numerical(replay, attempt, excluded, batch, upper, selected, arms) -> tuple:
    """The numerical gate: the K-only regular runs of the selected rates' cases
    (``numerical_cases``; one the launcher gave up on lets the check go on, its case then not
    qualified), then the check's verdict (``numerical_verdict``). Unusable rates (failing, not
    qualified, or an FA rate whose control was rejected or not qualified; a stale verdict's marks
    included) leave the batch. An arm holding one spends its search allowance (*The search
    allowance*): it is re-solved once without them; left without a feasible triple, it gets one
    repair round (the ``repair`` campaign, ``allowance`` naming the check that spent it) and a
    re-selection; every replacement is checked in a new check. An arm whose allowance is spent
    (here, or at an earlier visit of the candidate) and that holds an unusable rate, or that has
    no feasible triple after it, stops the candidate: failed when the cause is scientific,
    blocked when it is technical only (``_stop``). Returns the selections and the FA rates'
    accepted controls, or None with the pending step (none when the attempt failed)."""

    def stop(step: dict | None, outcome: str = "pending") -> tuple[None, None, dict | None]:
        attempt["outcome"] = outcome
        return None, None, step

    def incomplete(problems: list[str]) -> tuple[None, None, dict]:
        return stop({"campaign": "incomplete", "problems": problems}, "incomplete")

    while True:
        request = {arm: list(selected[arm]["rates"]) for arm in arms}
        keys = [(arm, batch, lr, seed) for arm, lr, seed in numerical_cases(request)]
        replay.called |= set(keys)
        check = replay.check(batch, upper, request)
        entry: dict[str, Any] = {"check": check["name"], "verdict": check["path"],
                                 "request": request, "state": None, "unusable": []}  # fmt: skip
        attempt["numerical"].append(entry)
        gone = {key: replay.exhausted[key] for key in keys
                if key not in replay.found and key in replay.exhausted}  # fmt: skip
        entry["given_up"] = {label: list(key) for key, label in sorted(gone.items())}
        missing = [key for key in keys if key not in replay.found and key not in gone]
        if missing:
            entry["state"] = "runs missing"
            return stop(_step("numerical", attempt, check=check["name"], rates=request,
                              runs=_wanted(missing), streams=_streams(batch)))  # fmt: skip
        if replay.verdicts is None or replay.facts is None:
            entry["state"] = "not read"
            return incomplete(["the numerical-check verdicts and the commits of the numerical "
                               f"runs were not read ($RUNS_DIR's {NUMERICAL_DIR}/, the runs' "
                               "meta.json): no conclusion"])  # fmt: skip
        facts = [replay.facts.get(replay.found[key].run_id) or {} for key in keys
                 if key in replay.found]  # fmt: skip
        commits = {fact.get("commit") for fact in facts}
        if None in commits or any(fact.get("dirty") is not False for fact in facts):
            entry["state"] = "commits"
            return incomplete([f"check {check['name']}: a numerical run records no commit or a "
                               f"dirty checkout ({sorted(map(str, commits))})"])  # fmt: skip
        entry["commits"] = sorted(commits)
        path, verdict = _read_check(replay, check, batch, upper, request, commits)
        entry.update(verdict, verdict=path)
        if verdict["state"] == "malformed":
            return incomplete([f"{path}: {reason}" for reason in verdict["reasons"]])
        # Never a pass; and only a current verdict with no incomplete entry, stale or not,
        # decides a re-solve (its earlier verdicts' marks included): the check is finished or
        # rerun first.
        decided = verdict["state"] in ("complete", "stale") and not verdict["unfinished"]
        if not decided or (verdict["state"] != "complete" and not verdict["unusable"]):
            given_up = [numerical_case_name(arm, lr, seed) for arm, _, lr, seed in gone]
            request_record = _check_request(check, batch, upper, request, commits,
                                            w_max=replay.w_max, given_up=given_up)  # fmt: skip
            return stop(_step("numerical-check", attempt, rates=request, state=verdict["state"],
                              check=request_record, reasons=verdict["reasons"]))  # fmt: skip
        if not verdict["unusable"]:
            controls = {rate(lr): {"accepted": True, "verdict": path}
                        for lr in request.get(FA) or ()}  # fmt: skip
            return selected, controls, None
        # A stale verdict's marks persist: a rate once found unusable at the batch stays so.
        marks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        shown: dict[str, list[str]] = defaultdict(list)
        for unusable in verdict["unusable"]:
            why = f"{unusable['why']} ({path}{', stale' if verdict['state'] == 'stale' else ''})"
            replay.mark(batch, unusable, why)
            excluded[unusable["arm"]].setdefault(unusable["lr"], why)
            marks[unusable["arm"]].append({**unusable, "why": why})
            shown[unusable["arm"]].append(f"{unusable['arm']} {unusable['lr']:g}: "
                                          f"{unusable['why']}")  # fmt: skip
        holders = [arm for arm in arms if arm in marks]
        spent = [arm for arm in holders if (arm, batch, upper) in replay.allowance]
        if spent:  # the allowance spent: the arm stops, by its marks' kind
            earlier = [arm for arm in spent if arm in attempt["allowance_carried"]]
            causes = {arm: ("scientific" if any(_kind(m) == "scientific" for m in marks[arm])
                            else "technical", marks[arm]) for arm in spent}  # fmt: skip
            note = f" (spent at an earlier visit: {earlier})" if earlier else ""
            listed = [s for arm in spent for s in shown[arm]]
            # A holder whose allowance is unspent owes it: the candidate stops before its
            # re-solve, and a later visit's re-selection is that allowance (second re-review of
            # fa22, minor 1: one allowance in all).
            for arm in holders:
                if arm not in spent:
                    replay.owed.setdefault((arm, batch, upper),
                                           {"check": check["name"], "verdict": path,
                                            "marks": shown[arm]})  # fmt: skip
            attempt["allowance_owed"] = [arm for arm in holders if arm not in spent]
            return None, None, _stop(replay, attempt, causes, "unusable rates after the search "
                                     f"allowance{note}: {listed}", "numerical")  # fmt: skip
        stops, step = _spend_allowance(replay, attempt, excluded, batch, upper, selected, holders,
                                       check["name"], path, shown)  # fmt: skip
        if stops:
            return None, None, step


def _spend_allowance(
    replay, attempt, excluded, batch, upper, selected, holders, check: str, path: str,
    shown: Mapping[str, list[str]], *, deferred: bool = False,
) -> tuple[bool, dict | None]:  # fmt: skip
    """The search allowance of ``holders`` at the candidate, spent by the check ``check``
    (verdict ``path``; ``shown``: each holder's marks as text): each re-solved once, from its
    current discovery points; left without a feasible triple, one repair round (``repair_rates``
    proposing no unusable rate; the ``repair`` campaign) and a re-selection. When one arm's
    round has a coverage miss it cannot repair, no arm's round of the check runs and the stop
    comes first: that means less search, never more. ``deferred``: an allowance owed since an
    earlier visit (``_Replay.owed``), spent at this visit's start. Returns (False, None) when
    every holder has a feasible triple again, else (True, the step: the pending ``repair``
    campaign, the blocked step, or None when the attempt failed)."""
    for arm in holders:  # the re-solve, from the current discovery points
        spending = {"arm": arm, "batch": batch, "U": upper, "check": check, "verdict": path,
                    "three_arm": attempt["three_arm"], "marks": list(shown[arm]),
                    "deferred": deferred, "resolved": None, "repair": None}  # fmt: skip
        replay.allowance[(arm, batch, upper)] = attempt["allowance"][arm] = spending
        selected[arm] = replay.solve(arm, batch, upper, sorted(excluded[arm]))
        attempt["arms"][arm]["selections"].append(selected[arm])
        spending["resolved"] = selected[arm]["rates"]
    going = [arm for arm in holders if not selected[arm]["feasible"]]
    if not going:
        return False, None
    # No feasible replacement triple: the allowance's one repair round, then a re-selection.
    rounds = [replay.allowance_round(arm, batch, upper, selected[arm], check,
                                     sorted(excluded[arm])) for arm in going]  # fmt: skip
    unrepairable = [r["arm"] for r in rounds if r["proposal"].get("unrepairable")]
    for spent_round in rounds:
        replay.spend(spent_round, prescribe=not unrepairable)
        attempt["arms"][spent_round["arm"]]["rounds"].append(spent_round)
        replay.allowance[(spent_round["arm"], batch, upper)]["repair"] = {
            "rates": spent_round["new_rates"],
            "reasons": spent_round["proposal"]["reasons"]}  # fmt: skip
    if not unrepairable:
        missing = [m for r in rounds for m in replay.missing(r["arm"], batch, r["new_rates"])]
        if missing:
            attempt["outcome"] = "pending"
            return True, _step("repair", attempt, runs=missing, allowance=check)
        for arm in going:
            selected[arm] = replay.solve(arm, batch, upper, sorted(excluded[arm]))
            attempt["arms"][arm]["selections"].append(selected[arm])
    stopped = unrepairable or [arm for arm in going if not selected[arm]["feasible"]]
    if not stopped:
        return False, None
    causes = _causes(replay, batch, upper, stopped, excluded)
    why = ("its repair round has a coverage miss that re-targeting cannot repair"
           if unrepairable else "none after its repair round")  # fmt: skip
    listed = [s for arm in stopped for s in shown[arm]]
    return True, _stop(replay, attempt, causes, f"{stopped} without a feasible triple once the "
                       f"unusable rates are dropped, the search allowance spent ({why}): "
                       f"{listed}", "numerical")  # fmt: skip


def _read_check(replay, check, batch, upper, request, commits) -> tuple[str, dict[str, Any]]:
    """The verdict of a check (``_Replay.verdicts_for``), read for the request: a record whose
    name is not the digest of its own rates is malformed; among several verdicts that answer it,
    one that marks a requested rate unusable with no incomplete entry is taken first (stale or
    not: marks persist), then a complete one, then the first. Each reading's ``unusable`` also
    holds the marks of its selection's earlier verdicts (``_history_marks``), listed under
    ``history``. Returns its path (the selection's own when none answers) and its reading."""
    readings = []
    for path, record in replay.verdicts_for(check):
        rates = record.get("rates") if isinstance(record, dict) else None
        stem = path.rsplit("/", 1)[-1].removesuffix(".json")
        if isinstance(rates, dict) and _digest(rates) != stem:
            why = f"its name is not the digest {_digest(rates)} of its rates"
            readings.append((path, {"state": "malformed", "reasons": [why], "unusable": [],
                                    "unfinished": []}))  # fmt: skip
            continue
        reading = numerical_verdict(record, batch, upper, request, commits, replay.reuse)
        readings.append((path, _history_marks(replay, path, reading, batch, upper, request)))
    if not readings:
        return check["path"], numerical_verdict(None, batch, upper, request, commits,
                                                replay.reuse)  # fmt: skip
    for wanted in (lambda r: r["state"] == "malformed",
                   lambda r: bool(r["unusable"]) and not r.get("unfinished"),
                   lambda r: r["state"] == "complete"):  # fmt: skip
        chosen = next(((p, r) for p, r in readings if wanted(r)), None)
        if chosen is not None:
            return chosen
    return readings[0]


def _history_marks(replay, path, reading, batch, upper, request) -> dict[str, Any]:
    """``reading`` (of the current verdict at ``path``) with the marks of its selection's earlier
    verdicts (``_Replay.history_of``) added to its ``unusable``: a requested rate an earlier
    verdict marks failing or not qualified, or an FA rate whose control it marks rejected or not
    qualified, stays unusable at the batch, whatever that verdict's staleness or unfinished
    entries (the coordinator's ruling after the integration step). Only the decision of the
    step comes from the current verdict. An earlier verdict that does not read as one of this
    selection, at its batch and U, leaves the reading malformed."""
    stem = path.rsplit("/", 1)[-1].removesuffix(".json")
    out = {**reading, "unusable": list(reading["unusable"]), "history": []}
    seen = {(u["arm"], u["lr"]) for u in out["unusable"]}
    for earlier, record in replay.history_of(path):
        out["history"].append(earlier)
        found = numerical_verdict(record, batch, upper, request, set(), lambda base, head: {})
        rates = record.get("rates") if isinstance(record, dict) else None
        if found["state"] == "malformed":
            why = found["reasons"]
        elif not isinstance(rates, dict) or _digest(rates) != stem:
            why = [f"not an earlier verdict of the selection {stem}"]
        elif (record["batch"], float(record["U"])) != (batch, upper):  # its directory's (B, U)
            why = [f"not an earlier verdict of the selection {stem} at B={batch}, U={upper:g}: "
                   f"its record is at B={record['batch']}, U={record['U']:g}"]  # fmt: skip
        else:
            why = []
        if why:
            return {**out, "state": "malformed", "unusable": [], "unfinished": [],
                    "reasons": [f"{earlier}: {reason}" for reason in why]}  # fmt: skip
        for mark in _marks(record, request, found):
            if (mark["arm"], mark["lr"]) not in seen:
                seen.add((mark["arm"], mark["lr"]))
                out["unusable"].append({**mark, "why": f"{mark['why']} (earlier verdict "
                                                       f"{earlier})"})  # fmt: skip
    return out


def _check_request(
    check, batch, upper, rates, commits, *, w_max: float, given_up: Sequence[str] = ()
) -> dict[str, Any]:
    """What the numerical-check tool is asked: ``selection``, its rate input (``rates`` by
    optimizer name, ``batch`` and ``U``, and ``given_up`` when the launcher gave up on a case's
    numerical run: the cases as the tool names them, ``numerical_case_name``; the final review
    of gate B, I1; scripts/numerical_check.py --selection), its cases, the frozen policy, the
    commits its numerical runs recorded (whose GPU code it must run), the directory it writes
    into and the selection's verdict there, under $RUNS_DIR, and the w_max the replay that wrote
    it ran under (I4)."""
    by_name = {OPTIMIZERS[arm]: list(lrs) for arm, lrs in rates.items()}
    selection = {"rates": by_name, "batch": batch,
                 "U": int(upper) if float(upper).is_integer() else upper}  # fmt: skip
    if given_up:
        selection["given_up"] = sorted(given_up)
    return {"name": check["name"], "tool": NUMERICAL_TOOL, "verdict": check["path"],
            "out": check["dir"], "selection": selection, "batch": batch, "U": upper,
            "rates": by_name,
            "cases": [{"optimizer": OPTIMIZERS[arm], "lr": lr, "seed": seed}
                      for arm, lr, seed in numerical_cases(rates)],
            "given_up": sorted(given_up), "runs_commits": sorted(commits),
            "policy": numerical_policy(upper), "w_max": w_max}  # fmt: skip


def _validation(replay, attempt, batch, upper, rates, controls) -> dict | None:
    """The validation gate: the runs of every arm at its rates on seeds 2100-2111 (a run the
    launcher gave up on counts as a technical failure), then ``validation_verdict`` over the
    candidate's arms (the three arms, or the primary arms alone in the fallback); a failure
    fails the candidate, but one that only technical failures make (``_technical_shortfall``: a
    shortfall of jointly supported blocks, or a utility member left without a bound) blocks the
    stage. Returns the pending step, if any."""
    keys = _keys(rates, batch, VALIDATION_SEEDS)
    replay.called |= set(keys)
    gone = {key: replay.exhausted[key] for key in keys
            if key in replay.exhausted and key not in replay.found}  # fmt: skip
    missing = [key for key in keys if key not in replay.found and key not in gone]
    if missing:
        attempt["outcome"] = "pending"
        return _step("validation", attempt, rates=rates, prewarmed_dropped=PREWARMED not in rates,
                     runs=_wanted(missing))  # fmt: skip
    # An exhausted run is absent: validation_verdict counts its rate as a technical failure.
    runs = [replay.found[key] for key in keys if key in replay.found]
    arms = tuple(arm for arm in ARMS if arm in rates)
    verdict = validation_verdict(runs, rates, controls, replay.theta0, k_star(upper),
                                 replay.w_max, arms=arms)  # fmt: skip
    verdict["exhausted"] = {label: list(key) for key, label in sorted(gone.items())}
    verdict["prewarmed_utility"] = _utility_means([r for r in runs if r.arm == PREWARMED],
                                                  replay.theta0)  # fmt: skip
    attempt["validation"] = verdict
    if verdict["passes"] is None:
        attempt["outcome"] = "incomplete"
        return {"campaign": "incomplete", "problems": verdict["problems"]}
    if not verdict["passes"]:
        reason = "the validation does not pass: " + "; ".join(verdict["reasons"])
        technical = _technical_shortfall(verdict)
        if technical is not None:
            stops, told = technical
            return _block(replay, attempt, stops, f"{reason}: {told}: technical failures only",
                          "validation")  # fmt: skip
        _fail(attempt, arms, reason, "validation")
    return None


def _technical_shortfall(verdict: Mapping[str, Any]) -> tuple[dict[str, list], str] | None:
    """A failed validation whose every fault technical failures alone make (the coordinator's
    rulings on C3 and M3; the thresholds unchanged): no run stopped by an instability rule, no
    utility bound computed above its limit, every FA control accepted, and
    - a shortfall below ``VALIDATION_MINIMUM`` jointly supported blocks only if at least
      ``VALIDATION_MINIMUM`` blocks are left once only the blocks with an arm unsupported by K
      (``bracketing`` or ``width``) are counted lost: a block lost only by arms whose cause is
      ``technical`` (a run given up by the launcher, another stop, no finite K25) is technical;
    - a utility member without a bound (fewer than two completed runs: the others failed
      technically or were given up, no instability stop being among them) is technical.

    Returns, by arm, the technical failures as marks (each run's, with the launcher's label of
    a run given up) and what they made, or None when the failure is not technical only."""
    family = (verdict.get("utility") or {}).get("family", [])
    exceeded = [member for member in family
                if any(member[f"{m}_upper_pp"] is not None and not member[f"{m}_passes"]
                       for m in ("mmlu", "factual"))]  # fmt: skip
    controls = any(r.startswith("FA controls not accepted") for r in verdict["reasons"])
    if verdict.get("unstable") or exceeded or controls:
        return None
    supports = verdict["supports"]
    labels = {(key[0], rate(key[2]), key[3]): label
              for label, key in (verdict.get("exhausted") or {}).items()}  # fmt: skip

    stops: dict[str, list] = defaultdict(list)
    told = []
    seen: set[tuple] = set()  # a failure listed once, by the first fault naming it

    def mark(arm: str, seed: int, failure: Mapping[str, Any], what: str) -> None:
        if (arm, seed, failure["rate"]) in seen:
            return
        seen.add((arm, seed, failure["rate"]))
        label = labels.get((arm, failure["rate"], seed))
        stops[arm].append({"arm": arm, "lr": failure["rate"], "state": "technical failure",
                           "why": f"{what}validation block {seed}: {failure['run'] or 'no run'}: "
                                  f"{failure['reason']}",
                           "reasons": [f"given up by the launcher: {label}"]
                           if label else []})  # fmt: skip

    if len(verdict["jointly_supported"]) < VALIDATION_MINIMUM:
        lost = [seed for seed in supports if seed not in verdict["jointly_supported"]]
        by_k = [seed for seed in lost
                if any(not s["supported"] and s["cause"] != "technical"
                       for s in supports[seed].values())]  # fmt: skip
        kept = len(supports) - len(by_k)
        if kept < VALIDATION_MINIMUM:
            return None
        for seed in sorted(set(lost) - set(by_k)):
            for arm, support in supports[seed].items():
                for failure in support["failures"] if not support["supported"] else ():
                    mark(arm, seed, failure, "")
        told.append(f"{kept} of {len(supports)} blocks would be jointly supported but for those "
                    "lost to technical failures only")  # fmt: skip
    unbounded = [member for member in family
                 if not (member["mmlu_passes"] and member["factual_passes"])]  # fmt: skip
    for member in unbounded:
        arm, lr = member["arm"], member["rate"]
        what = f"utility at {lr:g} without a bound ({member['runs']} completed runs): "
        for seed, by_arm in sorted(supports.items()):
            for failure in by_arm[arm]["failures"]:
                if failure["rate"] == lr:
                    mark(arm, seed, failure, what)
        told.append(f"the utility member {arm} at {lr:g} has {member['runs']} completed runs, "
                    "fewer than two: no bound")  # fmt: skip
    if not stops:
        return None
    return dict(stops), "; ".join(told)


# ------------------------------------------------------------------ support and the estimator


def block_support(
    runs: Sequence[Trajectory],
    kstar: float,
    w_max: float,
    *,
    rates: Sequence[float] | None = None,
) -> dict[str, Any]:
    """One arm in one block: supported when its non-failed endpoints bracket K* and the nearest
    bracketing pair's realized width ln K_hi − ln K_lo is at most w_max. Otherwise the cause:
    ``technical`` when an endpoint failed (by reason: an instability rule, another stop, no finite
    K25, or a rate of ``rates`` without a run), else ``bracketing`` or ``width``. Beside it,
    ``k_cause`` is what the surviving endpoints alone show (``bracketing``, ``width`` or None)
    and ``failure_kinds`` the kinds of the failures (instability, technical). A second run at a
    rate, or a run at a rate outside ``rates``, is a problem and not used."""
    if len({(run.arm, run.seed) for run in runs}) > 1:
        raise ValueError("block_support takes one arm's runs in one block")
    allowed = None if rates is None else {rate(lr) for lr in rates}
    endpoints, failures, problems, used = [], [], [], set()
    for run in sorted(runs, key=lambda r: r.lr):
        lr = rate(run.lr)
        if lr in used or (allowed is not None and lr not in allowed):
            what = "a second run at its rate" if lr in used else "a rate outside the frozen rates"
            problems.append(f"{run.run_id}: {what}")
            continue
        used.add(lr)
        row, why = _endpoint(run)
        if row is None:
            failures.append({"run": run.run_id, "rate": lr, "reason": why,
                             "kind": _failure_kind(run)})  # fmt: skip
        else:
            endpoints.append({"run": run.run_id, "rate": lr, "K": row["K_benign"]})
    failures += [{"run": None, "rate": lr, "reason": "no run", "kind": "technical"}
                 for lr in sorted(allowed or ()) if lr not in used]  # fmt: skip
    pair = _nearest_pair(endpoints, kstar)
    width = pair["width"]
    k_cause = "bracketing" if width is None else ("width" if width > w_max else None)
    cause = k_cause and ("technical" if failures or not endpoints else k_cause)
    return {"supported": cause is None, "cause": cause, "k_cause": k_cause,
            "failure_kinds": sorted({f["kind"] for f in failures}), "bracket": pair,
            "endpoints": endpoints, "failures": failures, "problems": problems}  # fmt: skip


def harm(row: Mapping[str, Any], *, missing_harmful: bool = False) -> float:
    """An endpoint's harm rate: harmful / prompts, a missing label counted as not harmful
    (primary, the low bound) or as harmful (the high bound)."""
    prompts, harmful, missing = row["prompts"], row["harmful"], row["missing"]
    if not (prompts > 0 and harmful >= 0 and missing >= 0 and harmful + missing <= prompts):
        raise ValueError(f"label row {row.get('condition')}: counts out of range")
    return (harmful + (missing if missing_harmful else 0)) / prompts


def interpolate(kstar: float, k_lo: float, h_lo: float, k_hi: float, h_hi: float) -> float:
    """h(K*) = h_lo + (h_hi − h_lo)·(ln K* − ln K_lo)/(ln K_hi − ln K_lo)."""
    fraction = (math.log(kstar) - math.log(k_lo)) / (math.log(k_hi) - math.log(k_lo))
    return h_lo + (h_hi - h_lo) * fraction


def _step_row(rows: Sequence[Mapping[str, Any]], run_id: str, step: int) -> Mapping[str, Any]:
    found = [r for r in rows if r["step"] == step and r["condition"] == f"step-{step}"]
    if len(found) != 1:
        raise ValueError(f"{run_id}: {len(found)} label rows of step-{step}")
    return found[0]


def _endpoint_row(rows: Sequence[Mapping[str, Any]], run_id: str) -> Mapping[str, Any]:
    return _step_row(rows, run_id, STEPS)


def arm_harm(
    support: Mapping[str, Any], labels: Mapping[str, Sequence[Mapping[str, Any]]], kstar: float
) -> dict[str, Any]:
    """A supported arm's interpolated harm at K*, from the label rows of its bracket's runs
    (``labels``: run id → rows): ``h`` with missing labels not harmful (primary, the arm's low
    bound) and ``h_missing_harmful`` with them harmful (its high bound), with the missing labels
    it read."""
    if not support["supported"]:
        raise ValueError("an unsupported arm has no estimate")
    lo, hi = support["bracket"]["lo"], support["bracket"]["hi"]
    rows = {side["run"]: _endpoint_row(labels.get(side["run"], ()), side["run"])
            for side in (lo, hi)}  # fmt: skip
    out: dict[str, Any] = {"missing": sum(row["missing"] for row in rows.values())}
    for name, worst in (("h", False), ("h_missing_harmful", True)):
        out[name] = interpolate(kstar, lo["K"], harm(rows[lo["run"]], missing_harmful=worst),
                                hi["K"], harm(rows[hi["run"]], missing_harmful=worst))  # fmt: skip
    return out


def block_difference(
    supports: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    kstar: float,
) -> dict[str, Any]:
    """Δ_block = h_AdamW − h_FA at K* (primary), the bounds of the missing-label sensitivities
    (``SENSITIVITIES``; low an arm's ``h``, high its ``h_missing_harmful``):
    ``delta_missing_low`` = A_low − F_high and ``delta_missing_high`` = A_high − F_low (the worst
    case for the contrast), ``delta_all_missing_harmful`` = A_high − F_high; and the interpolated
    harm of every supported arm of ``supports`` (arm → ``block_support``)."""
    arms = {arm: arm_harm(s, labels, kstar) for arm, s in supports.items() if s["supported"]}
    supported = all(arm in arms for arm in PRIMARY_ARMS)
    bounds = (key for pair in SENSITIVITIES.values() for key in pair)
    out: dict[str, Any] = {"supported": supported, "delta": None, **dict.fromkeys(bounds)}
    if supported:
        a, f = arms[ADAMW], arms[FA]
        out.update(
            delta=a["h"] - f["h"],
            delta_missing_low=a["h"] - f["h_missing_harmful"],
            delta_missing_high=a["h_missing_harmful"] - f["h"],
            delta_all_missing_harmful=a["h_missing_harmful"] - f["h_missing_harmful"],
        )
    return {**out, "h": {arm: a["h"] for arm, a in arms.items()}, "arms": arms}


# ------------------------------------------------------------------ the looks


def _t_interval(values: Sequence[float], alpha: float, floor: float = 0.0) -> dict[str, Any]:
    """mean ± t_{n−1, 1−α/2}·max(sd, floor)/√n; no interval below two values."""
    n = len(values)
    mean = statistics.fmean(values) if values else None
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "se": None, "t": None, "interval": None}
    sd = statistics.stdev(values)
    se = max(sd, floor) / math.sqrt(n)
    t = float(stats.t.ppf(1 - alpha / 2, n - 1))
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t,
            "interval": [mean - t * se, mean + t * se]}  # fmt: skip


def _numpy_moments(values: np.ndarray, floor: float) -> tuple[float, float, float]:
    """mean, sd (ddof 1) and se = max(sd, floor)/√n of two values or more, as numpy computes
    them: the bootstrap-t's (and the simulation's ``bootstrap_t_quantiles``') centre and scale."""
    sd = float(values.std(ddof=1))
    return float(values.mean()), sd, max(sd, floor) / math.sqrt(values.size)


def bootstrap_t_interval(
    values: Sequence[float],
    level: float,
    resamples: int,
    rng: np.random.Generator,
    *,
    floor: float = 0.0,
) -> tuple[float, float, int]:
    """The studentized bootstrap (bootstrap-t) interval of the mean of ``values`` at the
    two-sided confidence level ``level``, as analysis/study2_short_sim.py (revision 3) defines
    it: ``resamples`` resamples of the values with replacement, drawn at once as
    ``rng.integers(0, n, (resamples, n))``; T* = (mean* − mean)/se*, se* = max(sd*, floor)/√n
    with the sample sd (ddof 1); the interval [mean − q_{(1+ℓ)/2}·se, mean − q_{(1−ℓ)/2}·se],
    se = max(sd, floor)/√n, q np.quantile's default (linear) quantiles of T*. A resample with
    se* ≤ ``ZERO_SE`` is dropped and counted. Returns (low, high, dropped): [mean, mean] when se
    is zero; (−∞, ∞), no interval, with fewer than 2 values (nothing is drawn then) or fewer
    than 2 kept resamples."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        return -math.inf, math.inf, 0
    samples = values[rng.integers(0, n, (resamples, n))]
    means = samples.mean(axis=1)
    se_star = np.maximum(samples.std(axis=1, ddof=1), floor) / math.sqrt(n)
    kept = se_star > ZERO_SE
    dropped = int(resamples - kept.sum())
    mean, _, se = _numpy_moments(values, floor)
    if se <= ZERO_SE:
        return mean, mean, dropped
    if kept.sum() < 2:
        return -math.inf, math.inf, dropped
    ordered = np.sort((means[kept] - mean) / se_star[kept])
    # np.quantile's linear rule, written as the simulation's _sorted_quantiles so that the two
    # agree to the last bit.
    position = np.array([(1 - level) / 2, (1 + level) / 2]) * (ordered.size - 1)
    below = np.floor(position).astype(int)
    above = np.minimum(below + 1, ordered.size - 1)
    lower, upper = ordered[below] + (position - below) * (ordered[above] - ordered[below])
    return float(mean - upper * se), float(mean - lower * se), dropped


def _readings(interval: Sequence[float] | None, *, equivalence: bool = True) -> list[str]:
    """The interval's readings, with the simulation's conventions (its ``_inside`` and
    ``sequential_decisions``): equivalence and ±8 within the margin, bounds included; direction
    0 strictly outside. ``equivalence`` False when the look may not conclude it."""
    if interval is None:
        return ["inconclusive"]
    lo, hi = interval
    found = []
    if equivalence and lo >= -EQUIVALENCE and hi <= EQUIVALENCE:
        found.append("equivalence")
    if lo > 0 or hi < 0:
        found.append("direction")
    if lo >= -NESTED and hi <= NESTED:
        found.append("within_8")
    return found or ["inconclusive"]


def pointwise(
    interval: Sequence[float] | None, allowance: float, *, equivalence: bool, direction: bool
) -> dict[str, Any]:
    """The reading about the pointwise Δ(K*) with the bias allowance b (the amendment's "the
    pointwise Δ(K*) lies in the interval widened by b"), as the simulation's
    ``pointwise_readings``: the interval widened by b on both sides; equivalence when the
    interval lies inside ±(5 pp − b), bounds included, only where the look may conclude
    equivalence (``equivalence``, its equivalence mask); a direction when it excludes [−b, b],
    only where the look may conclude anything (``direction``). Decisions never use it."""
    if interval is None:
        return {"interval": None, "readings": ["inconclusive"]}
    lo, hi = interval
    margin = EQUIVALENCE - allowance
    found = []
    if equivalence and lo >= -margin and hi <= margin:
        found.append("equivalence")
    if direction and (lo > allowance or hi < -allowance):
        found.append("direction")
    return {"interval": [lo - allowance, hi + allowance], "readings": found or ["inconclusive"]}


def finite_bounds(started: int, unsupported: int, total: float) -> tuple[float, float]:
    """The finite-cohort bounds [(S − m)/N, (S + m)/N] of the started cohort: N started, m
    unsupported, S the sum of the supported differences (descriptive)."""
    return ((total - unsupported) / started, (total + unsupported) / started)


def looks(
    blocks: Sequence[Mapping[str, Any]],
    levels: Sequence[float],
    n: Sequence[int] = LOOK_SIZES,
    cap: int = CAP,
    *,
    interval: str,
    se_floor: float,
    bias_allowance: float,
    equivalence_min_supported: int,
    direction_min_supported: int,
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """The sequential looks over ``blocks`` in their preassigned order. Look k happens at the
    first started count where the supported count reaches n_k; when the cap of started blocks
    comes first, the terminal look happens there. ``levels`` are the two-sided confidence levels
    1 − α_k of the looks, e.g. (0.9920, 0.9631), one per look. The interval at look k over the
    supported Δ is, by ``interval``:

    - ``bootstrap_t``: ``bootstrap_t_interval`` with ``bootstrap_resamples`` resamples from
      ``numpy.random.default_rng(derive_seed(bootstrap_seed, "look", k))``, floor ``se_floor``;
    - ``t``: mean ± t_{n−1, 1−α_k/2}·max(sd, se_floor)/√n.

    Readings: equivalence (only at a look with at least ``equivalence_min_supported`` supported
    blocks), direction, within ±8, else inconclusive; none below ``direction_min_supported``
    supported blocks (0 disables either minimum); look 1 stops on equivalence or direction only.
    A bootstrap look reports numpy's mean, sd and se, those of its interval. ``bias_allowance``
    is reported beside every look, with the interval widened by it and the pointwise readings
    under the look's minima (``pointwise``)."""
    if len(levels) != len(n) or not all(0.5 < level < 1 for level in levels):
        raise ValueError(f"looks: levels {tuple(levels)} must be {len(n)} confidence levels in "
                         "(0.5, 1), e.g. (0.9920, 0.9631)")  # fmt: skip
    if interval not in INTERVALS:
        raise ValueError(f"looks: interval {interval!r} is not one of {INTERVALS}")
    if interval == "bootstrap_t" and not (
        _integer(bootstrap_resamples) and bootstrap_resamples >= 2 and _integer(bootstrap_seed)
    ):
        raise ValueError(
            "looks: the bootstrap needs an integer number of resamples ≥ 2 and an integer seed, "
            f"not {bootstrap_resamples!r} and {bootstrap_seed!r}"
        )
    for name, minimum in (("equivalence_min_supported", equivalence_min_supported),
                          ("direction_min_supported", direction_min_supported)):  # fmt: skip
        if not (_integer(minimum) and minimum >= 0):
            raise ValueError(f"looks: {name} {minimum!r} is not an integer ≥ 0")
    settings = {
        "interval": interval,
        "se_floor": se_floor,
        "bias_allowance": bias_allowance,
        "equivalence_min_supported": equivalence_min_supported,
        "direction_min_supported": direction_min_supported,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }
    done, deltas, started, k = [], [], 0, 0
    stopped = False
    for block in blocks[:cap]:
        if not block["complete"]:
            break
        started += 1
        if block["supported"]:
            if not _finite(block.get("delta")):
                raise ValueError(f"block {block.get('block')}: supported without a finite Δ")
            deltas.append(block["delta"])
        terminal = started == cap
        if len(deltas) == n[k]:
            done.append(_look(k + 1, levels[k], deltas, started, settings))
            last = k == len(n) - 1
            stopped = last or bool({"equivalence", "direction"} & set(done[-1]["readings"]))
            done[-1]["stops"] = stopped
            k += 1
        if not stopped and terminal:
            done.append(_look(len(n), levels[-1], deltas, started, settings))
            done[-1].update(stops=True, at_cap=True)
            stopped = True
        if stopped:
            break
    out: dict[str, Any] = {"looks": done, "stopped": stopped,
                           "stopping_look": done[-1]["look"] if stopped else None}  # fmt: skip
    if not stopped:
        incomplete = next((b.get("block") for b in blocks[:cap] if not b["complete"]), None)
        out["waiting"] = {"look": k + 1, "supported": len(deltas), "needs": n[k],
                          "started": started, "incomplete": incomplete}  # fmt: skip
    return out


def _interval(
    values: Sequence[float],
    level: float,
    method: str,
    *,
    floor: float = 0.0,
    resamples: int | None = None,
    seed: int | None = None,
    stream: Sequence[Any] = (),
) -> dict[str, Any]:
    """The interval of the mean of ``values`` at ``level`` by ``method``: the t interval
    (``_t_interval``), or the bootstrap-t (``bootstrap_t_interval``, ``resamples`` from
    ``numpy.random.default_rng(derive_seed(seed, *stream))``), whose record then carries numpy's
    moments and its ``bootstrap`` (resamples, seed, stream seed, dropped resamples)."""
    found = _t_interval(values, 1 - level, floor)
    found["bootstrap"] = None
    if method == "bootstrap_t":
        stream_seed = derive_seed(seed, *stream)
        low, high, dropped = bootstrap_t_interval(
            values, level, resamples, np.random.default_rng(stream_seed), floor=floor
        )  # fmt: skip
        found["bootstrap"] = {"resamples": resamples, "seed": seed, "stream_seed": stream_seed,
                              "dropped": dropped}  # fmt: skip
        bounded = math.isfinite(low) and math.isfinite(high)
        found.update(t=None, interval=[low, high] if bounded else None)
        if len(values) >= 2:  # the record agrees with its interval: numpy's moments
            mean, sd, se = _numpy_moments(np.asarray(values, dtype=float), floor)
            found.update(mean=mean, sd=sd, se=se)
    return found


def _look_interval(number, level, values, settings) -> dict[str, Any]:
    """Look ``number``'s interval over ``values``, by the looks' settings and stream."""
    return _interval(values, level, settings["interval"], floor=settings["se_floor"],
                     resamples=settings["bootstrap_resamples"], seed=settings["bootstrap_seed"],
                     stream=("look", number))  # fmt: skip


def _look(number, level, deltas, started, settings) -> dict[str, Any]:
    found = _look_interval(number, level, deltas, settings)
    bootstrap = found.pop("bootstrap")
    # Below the direction minimum nothing is concluded; below the equivalence one, no equivalence.
    concluded = len(deltas) >= settings["direction_min_supported"]
    allowed = concluded and len(deltas) >= settings["equivalence_min_supported"]
    readings = _readings(found["interval"], equivalence=allowed) if concluded else ["inconclusive"]
    masks = {"equivalence": allowed, "direction": concluded}
    return {
        "look": number,
        "level": level,
        "started": started,
        "supported": len(deltas),
        "method": settings["interval"],
        **found,
        "bootstrap": bootstrap,
        "equivalence_allowed": allowed,
        "direction_allowed": concluded,
        "readings": readings,
        "bounds": finite_bounds(started, started - len(deltas), sum(deltas)),
        "bias_allowance": settings["bias_allowance"],
        "pointwise": pointwise(found["interval"], settings["bias_allowance"], **masks),
        "at_cap": False,
    }


def sensitivity_looks(
    primary: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    name: str,
    *,
    interval: str,
    se_floor: float,
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
) -> list[dict[str, Any]]:
    """The sensitivity ``name`` (``SENSITIVITIES``) at the primary's actual looks: ``primary`` is
    ``looks``' result over the same ``blocks``, and the other arguments are the settings it was
    computed with. At each look, the supported blocks among the first ``started`` (those the look
    read), the look's level and the same interval and bootstrap stream; the interval runs from the
    lower bound of the interval over the blocks' lower bounds to the upper bound of the interval
    over their upper bounds, read as the look reads (``_readings``) under the look's equivalence
    and direction minima. It never stops anything: a sensitivity has no looks of its own."""
    low_key, high_key = SENSITIVITIES[name]
    settings = {
        "interval": interval,
        "se_floor": se_floor,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }
    out = []
    for look in primary["looks"]:
        chosen = [b for b in blocks[: look["started"]] if b["supported"]]
        if len(chosen) != look["supported"]:
            raise ValueError(f"sensitivity {name}: look {look['look']} read {look['supported']} "
                             f"supported blocks, the blocks given hold {len(chosen)}")  # fmt: skip
        values = {key: [b.get(key) for b in chosen] for key in (low_key, high_key)}
        if not all(_finite(v) for found in values.values() for v in found):
            raise ValueError(f"sensitivity {name}: a supported block without finite bounds")
        low = _look_interval(look["look"], look["level"], values[low_key], settings)
        high = (
            low
            if high_key == low_key
            else _look_interval(look["look"], look["level"], values[high_key], settings)
        )
        bounded = low["interval"] is not None and high["interval"] is not None
        found = [low["interval"][0], high["interval"][1]] if bounded else None
        readings = (_readings(found, equivalence=look["equivalence_allowed"])
                    if look["direction_allowed"] else ["inconclusive"])  # fmt: skip
        out.append({"look": look["look"], "level": look["level"], "started": look["started"],
                    "supported": look["supported"], "method": interval, "interval": found,
                    "readings": readings,
                    "low": {"mean": low["mean"], "interval": low["interval"]},
                    "high": {"mean": high["mean"], "interval": high["interval"]},
                    "primary_readings": look["readings"]})  # fmt: skip
    return out


def secondary_contrasts(
    blocks: Sequence[Mapping[str, Any]],
    started: int,
    *,
    look: int,
    interval: str,
    status: str,
    level: float = SECONDARY_LEVEL,
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """A − P and P − F (P the pre-warmed arm) on the common three-arm supported cohort among the
    first ``started`` blocks (those of the stopping look ``look``), each a two-sided interval at
    ``level`` without an SE floor, with the cohort's attrition. ``interval`` (the fixation
    record's ``secondary_interval``): ``t``, or ``bootstrap_t`` with ``bootstrap_resamples`` from
    ``derive_seed(bootstrap_seed, "secondary", contrast, look)``; ``status`` (its
    ``secondary_status``) is reported beside them."""
    if interval not in INTERVALS:
        raise ValueError(f"secondary_contrasts: interval {interval!r} is not one of {INTERVALS}")
    if status not in SECONDARY_STATUSES:
        raise ValueError(f"secondary_contrasts: status {status!r} is not one of "
                         f"{SECONDARY_STATUSES}")  # fmt: skip
    if interval == "bootstrap_t" and not (
        _integer(bootstrap_resamples) and bootstrap_resamples >= 2 and _integer(bootstrap_seed)
    ):
        raise ValueError("secondary_contrasts: the bootstrap needs an integer number of "
                         "resamples ≥ 2 and an integer seed")  # fmt: skip
    primary = [b for b in blocks[:started] if b["complete"] and b["supported"]]
    arms = (*PRIMARY_ARMS, PREWARMED)
    common = [b for b in primary if all(arm in b.get("h", {}) for arm in arms)]
    contrasts = {
        "adamw_minus_prewarmed": [b["h"][ADAMW] - b["h"][PREWARMED] for b in common],
        "prewarmed_minus_fa": [b["h"][PREWARMED] - b["h"][FA] for b in common],
    }
    return {
        "level": level,
        "look": look,
        "secondary_interval": interval,
        "secondary_status": status,
        **{
            name: _interval(
                values,
                level,
                interval,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                stream=("secondary", name, look),
            )
            for name, values in contrasts.items()
        },  # fmt: skip
        "attrition": {
            "started": started,
            "primary_supported": len(primary),
            "three_arm_supported": len(common),
            "prewarmed_unsupported": len(primary) - len(common),
        },  # fmt: skip
    }


def continuation(blocks: Sequence[Mapping[str, Any]], primary: Mapping[str, Any]) -> dict | None:
    """The blocks after the primary's stopping look (amendment 4, section 3: confirmation goes on
    for the mechanism cohort), descriptive: ``primary`` is ``looks``' result over ``blocks``. The
    blocks complete in order after those the stopping look read, their supported Δ and every
    supported Δ complete in order (n, mean, sd); no interval, no reading, and nothing here ever
    revises the decision. None before the primary stops."""
    if not primary["stopped"]:
        return None
    started = primary["looks"][-1]["started"]
    later = list(itertools.takewhile(lambda b: b["complete"], blocks[started:CAP]))
    after = [b["delta"] for b in later if b["supported"]]
    read = [b["delta"] for b in blocks[:started] if b["supported"]]

    def described(values: Sequence[float]) -> dict[str, Any]:
        return {"n": len(values), "mean": statistics.fmean(values) if values else None,
                "sd": statistics.stdev(values) if len(values) >= 2 else None}  # fmt: skip

    return {"stopping_look": primary["stopping_look"], "read_blocks": started,
            "later_blocks": [b["block"] for b in later],
            "later": described(after), "every_supported": described(read + after),
            "status": "descriptive: it never revises the decision"}  # fmt: skip


# ------------------------------------------------------------------ the mechanism cohort


def _abs_studentized(samples: np.ndarray, mean: float) -> np.ndarray:
    """|T*| = |mean* − mean| / SE* per resample (a row of ``samples``), SE* = sd* (ddof 1)/√n;
    a resample whose SE* ≤ ``ZERO_SE`` has |T*| = 0 when |mean* − mean| ≤ ``ZERO_SE`` and ∞
    otherwise (amendment 4, section 3, zero SE): no resample is dropped."""
    centred = np.abs(samples.mean(axis=1) - mean)
    se = samples.std(axis=1, ddof=1) / math.sqrt(samples.shape[1])
    flat = se <= ZERO_SE
    out = np.where(centred <= ZERO_SE, 0.0, np.inf)
    np.divide(centred, se, out=out, where=~flat)
    return out


def _upper_quantile(ordered: np.ndarray, level: float) -> float:
    """The ``level`` quantile of sorted |T*| by np.quantile's linear rule, written as
    ``bootstrap_t_interval`` writes it, ∞ where it reaches a draw of |T*| = ∞ (the linear rule's
    arithmetic would form 0·∞ or ∞ − ∞ there)."""
    position = level * (ordered.size - 1)
    below = math.floor(position)
    above = min(below + 1, ordered.size - 1)
    low, high = float(ordered[below]), float(ordered[above])
    if math.isinf(high):
        return low if position == below else math.inf
    return low + (position - below) * (high - low)


def _studentized_family(
    columns: Sequence[Sequence[float]], level: float, resamples: int, seed: int, *, keep: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The paired seed-block bootstrap of the means of ``columns`` (each one value per block,
    paired): one set of block indices per draw, ``numpy.random.default_rng(seed).integers(0, n,
    (resamples, n))`` as ``bootstrap_t_interval`` draws them, shared by every column; |T*| of
    each (``_abs_studentized``), their maximum per draw, its ``level`` quantile q
    (``_upper_quantile``), and each column's symmetric interval mean ± q·SE, SE = sd (ddof 1)/√n.
    A column whose observed SE ≤ ``ZERO_SE``, or any column when q is ∞ or n < 2, has none."""
    if not (_number(level) and 0 < level < 1):
        raise ValueError(f"the studentized bootstrap: level {level!r} is not in (0, 1)")
    if not (_integer(resamples) and resamples >= 2 and _integer(seed)):
        raise ValueError(f"the studentized bootstrap needs an integer number of resamples ≥ 2 "
                         f"and an integer seed, not {resamples!r} and {seed!r}")  # fmt: skip
    arrays = [np.asarray(values, dtype=float) for values in columns]
    n = arrays[0].size
    if any(a.ndim != 1 or a.size != n for a in arrays):
        raise ValueError("the studentized bootstrap pairs its contrasts per block: one value "
                         "per block each, as many blocks")  # fmt: skip
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("the studentized bootstrap: a value is not finite")
    found = {"n": n, "level": level, "resamples": resamples, "seed": seed, "q": None,
             "draws": 0, "infinite": 0}  # fmt: skip
    each = []
    for values in arrays:
        if n < 2:
            mean = float(values.mean()) if n else None
            each.append({"n": n, "mean": mean, "sd": None, "se": None, "zero_se": None,
                         "interval": None})  # fmt: skip
            continue
        mean, sd, se = _numpy_moments(values, 0.0)
        each.append({"n": n, "mean": mean, "sd": sd, "se": se, "zero_se": se <= ZERO_SE,
                     "interval": None})  # fmt: skip
    if n < 2:
        return found, each
    indices = np.random.default_rng(seed).integers(0, n, (resamples, n))
    t_star = [_abs_studentized(values[indices], one["mean"])
              for values, one in zip(arrays, each, strict=True)]  # fmt: skip
    maxima = t_star[0] if len(t_star) == 1 else np.maximum.reduce(t_star)
    q = _upper_quantile(np.sort(maxima), level)
    # The critical value and the per-draw maxima behind it, for the simulation's twin.
    found.update(q=q, draws=int(maxima.size), infinite=int(np.isinf(maxima).sum()))
    if keep:
        found["t_star"] = t_star
    for one in each:
        if not one["zero_se"] and math.isfinite(q):
            one["interval"] = [one["mean"] - q * one["se"], one["mean"] + q * one["se"]]
    return found, each


def symmetric_studentized(
    c: Sequence[float], level: float, resamples: int, seed: int, *, keep: bool = False
) -> dict[str, Any]:
    """The symmetric studentized bootstrap interval of the mean of ``c`` (one value per block):
    T* = (mean* − mean)/SE* over ``resamples`` paired seed-block resamples from
    ``numpy.random.default_rng(seed)`` (``seed`` a stream's, from ``derive_seed``), q the
    ``level`` quantile of |T*| by np.quantile's linear rule, the interval mean ± q·SE (zero SE:
    ``_abs_studentized``; none at an observed SE ≤ ``ZERO_SE``, with n < 2 or q = ∞). It is
    ``max_t_interval`` with one contrast: the comparison intervals at ``COMPARISON_LEVEL``, and
    the reading of the other contrast when one has a zero SE. Returns n, level, resamples, seed,
    q, the draws (resamples) and how many had |T*| = ∞, mean, sd, se, ``zero_se`` and the
    ``interval`` (None when there is none); with ``keep``, the draws' |T*| (``t_star``)."""
    found, (one,) = _studentized_family([c], level, resamples, seed, keep=keep)
    if keep and "t_star" in found:
        found["t_star"] = found["t_star"][0]
    return {**found, **{k: v for k, v in one.items() if k != "n"}}


def max_t_interval(
    c1: Sequence[float],
    c2: Sequence[float],
    level: float,
    resamples: int,
    seed: int,
    *,
    keep: bool = False,
) -> dict[str, Any]:
    """The paired seed-block bootstrap max-t intervals of two contrasts on the same blocks
    (``c1[i]`` and ``c2[i]`` of block i): every draw resamples one set of block indices, shared
    by both (``numpy.random.default_rng(seed).integers(0, n, (resamples, n))``); each contrast is
    studentized, |T_j*| = |mean_j* − mean_j|/SE_j* (zero SE: ``_abs_studentized``, 0 or ∞), and
    the critical value q is the ``level`` quantile of max(|T₁*|, |T₂*|) by np.quantile's linear
    rule: the symmetric simultaneous intervals mean_j ± q·SE_j. A contrast whose observed SE ≤
    ``ZERO_SE`` has no interval (``mechanism_analysis`` then reads the other by
    ``symmetric_studentized`` at ``COMPARISON_LEVEL``). Returns n, level, resamples, seed, q, the
    draws (the number of per-draw maxima) and how many maxima were ∞, and ``contrasts``, one
    entry per contrast (mean, sd, se, ``zero_se``, ``interval``); with ``keep``, every draw's
    |T_j*| per contrast (``t_star``). With one contrast whose |T*| is 0 in every draw, the other's
    interval is ``symmetric_studentized``'s at the same level, resamples and seed."""
    found, each = _studentized_family([c1, c2], level, resamples, seed, keep=keep)
    return {**found, "contrasts": [{k: v for k, v in one.items() if k != "n"} for one in each]}


def component_readings(
    interval: Sequence[float] | None, margin: float = MECHANISM_EFFECTS[0]
) -> list[str]:
    """A component's readings, with ``_readings``' conventions: ``positive`` or ``negative``
    when its interval excludes 0 (strictly), ``bounded`` when it lies within ±``margin`` (the
    meaningful component effect, 5 pp; bounds included), both when both hold, otherwise
    ``inconclusive``; no interval, ``inconclusive``."""
    if interval is None:
        return ["inconclusive"]
    lo, hi = interval
    found = ["positive"] if lo > 0 else ["negative"] if hi < 0 else []
    if lo >= -margin and hi <= margin:
        found.append("bounded")
    return found or ["inconclusive"]


def _unsupported(block: Mapping[str, Any]) -> dict[str, str]:
    """A K-complete confirmation block's unsupported arms, each with its cause."""
    return {arm: s["cause"] for arm, s in block["supports"].items() if not s["supported"]}


def _attrition(window: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The blocks of the cohort's window that are not jointly supported, by reason: the arms
    unsupported together, each arm's causes (``block_support``: technical, bracketing, width),
    those lost to a technical failure, and those the primary keeps (the pre-warmed arm alone
    lost). ``window``: entries with ``block``, ``primary_supported``, ``three_arm_supported`` and
    ``unsupported`` (arm → cause)."""
    lost = [b for b in window if not b["three_arm_supported"]]
    together = Counter("+".join(sorted(b["unsupported"])) for b in lost)
    causes = {arm: dict(sorted(Counter(b["unsupported"][arm] for b in lost
                                       if arm in b["unsupported"]).items()))
              for arm in ARMS}  # fmt: skip
    return {"blocks": len(lost), "unsupported_arms": dict(sorted(together.items())),
            "causes": causes,
            "technical": sum(1 for b in lost if "technical" in b["unsupported"].values()),
            "primary_supported": sum(1 for b in lost if b["primary_supported"]),
            "lost_blocks": [b["block"] for b in lost]}  # fmt: skip


def _window_entry(block: Mapping[str, Any]) -> dict[str, Any]:
    return {"block": block["block"], "primary_supported": block["supported"],
            "three_arm_supported": block["three_arm_supported"],
            "unsupported": _unsupported(block)}  # fmt: skip


def _cohort_counts(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The mechanism cohort on K and support only, over ``progress``'s confirmation blocks of a
    three-arm selection: the window (the blocks complete in order, up to the ``MECHANISM_TARGET``-th
    jointly supported one), the cohort (its jointly supported blocks), how it ended (``target``,
    ``cap`` once all ``MECHANISM_CAP`` blocks are complete, else None), the blocks with runs and
    those that may still be jointly supported (all with runs but the complete ones that are not)."""
    window, cohort = [], []
    in_order = list(itertools.takewhile(lambda b: b["complete"], blocks[:MECHANISM_CAP]))
    for block in in_order:
        window.append(block)
        if block["three_arm_supported"]:
            cohort.append(block["block"])
            if len(cohort) == MECHANISM_TARGET:
                break
    ended = ("target" if len(cohort) == MECHANISM_TARGET
             else "cap" if len(in_order) == MECHANISM_CAP else None)  # fmt: skip
    possible = sum(1 for b in blocks if b["runs"]
                   and not (b["complete"] and not b["three_arm_supported"]))  # fmt: skip
    return {"window": window, "cohort": cohort, "ended_by": ended,
            "complete_in_order": len(in_order), "started": sum(1 for b in blocks if b["runs"]),
            "possibly_jointly_supported": possible}  # fmt: skip


def mechanism_state(blocks: Sequence[Mapping[str, Any]], *, three_arm: bool) -> dict[str, Any]:
    """The mechanism cohort's state (amendment 4, section 3), on K and support only (never a
    label), over ``progress``'s confirmation blocks: with a three-arm selection, the blocks
    started, the jointly supported ones complete in order (``jointly_supported_in_order``, at most
    ``MECHANISM_TARGET``) and the cohort's block numbers, those that may still be jointly
    supported, the room for new blocks (min(``MECHANISM_TARGET`` − those, ``MECHANISM_CAP`` −
    the blocks before the first without runs), at least 0; on the runs alone: the guard also
    counts the blocks of issued ranges the launcher has not reached), whether it ``ended``
    (``MECHANISM_TARGET`` jointly supported blocks complete in order, or all ``MECHANISM_CAP``
    blocks started and complete: ``ended_by`` ``target`` or ``cap``) with the window's last
    block, and the attrition by reason (``_attrition``). A two-arm selection (the pre-warmed arm
    dropped, or the primary-only fallback) has no mechanism cohort."""
    if not three_arm:
        return {"applies": False,
                "reason": "a two-arm selection (the pre-warmed arm dropped, or the primary-only "
                          "fallback): no mechanism cohort; amendment 3's looks pace the "
                          "confirmation"}  # fmt: skip
    counts = _cohort_counts(blocks)
    window = counts["window"]
    first_open = next((b["block"] for b in blocks if not b["runs"]), MECHANISM_CAP + 1)
    room = min(MECHANISM_TARGET - counts["possibly_jointly_supported"],
               MECHANISM_CAP - (first_open - 1))  # fmt: skip
    return {
        "applies": True,
        "target": MECHANISM_TARGET,
        "cap": MECHANISM_CAP,
        "minimum": MECHANISM_MINIMUM,
        "started": counts["started"],
        "complete_in_order": counts["complete_in_order"],
        "jointly_supported_in_order": len(counts["cohort"]),
        "possibly_jointly_supported": counts["possibly_jointly_supported"],
        "room": max(room, 0),
        "ended": counts["ended_by"] is not None,
        "ended_by": counts["ended_by"],
        "last_block": window[-1]["block"] if counts["ended_by"] and window else None,
        "cohort": counts["cohort"],
        "minimum_met": len(counts["cohort"]) >= MECHANISM_MINIMUM,
        "attrition": _attrition([_window_entry(b) for b in window]),
    }


def mechanism_blocks(
    blocks: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    kstar: float,
) -> list[dict[str, Any]]:
    """``progress``'s confirmation blocks as ``mechanism_analysis`` reads them, only at its final
    analysis: each ``look_blocks`` entry (``complete`` there: complete and labeled; ``h``, the
    interpolated harm of every supported arm, missing labels not harmful) with the block's K-level
    ``k_complete``, ``three_arm_supported`` and ``primary_supported``, its unsupported arms with
    their causes (``unsupported``), and, once labeled, ``h_missing_harmful`` per supported arm
    (``block_difference``'s, missing labels harmful: the sensitivities' high bounds)."""
    out = []
    for block, entry in zip(blocks, look_blocks(blocks, labels, kstar), strict=True):
        extra: dict[str, Any] = {
            "k_complete": block["complete"],
            "three_arm_supported": block["three_arm_supported"],
            "primary_supported": block["supported"],
            "unsupported": _unsupported(block) if block["complete"] else {},
            "h_missing_harmful": {},
        }
        if entry["complete"]:
            arms = block_difference(block["supports"], labels, kstar)["arms"]
            extra["h_missing_harmful"] = {arm: a["h_missing_harmful"] for arm, a in arms.items()}
        out.append({**entry, **extra})
    return out


def _check_mechanism(settings: Mapping[str, Any]) -> None:
    problems = mechanism_problems(dict(settings))
    if problems:
        raise ValueError("mechanism_analysis: " + "; ".join(problems))


# The missing-label sensitivities of a contrast X − Y (``SENSITIVITIES``, computed as
# ``block_difference`` computes them for AdamW − FA): per bound, the fields of X and of Y, ``h``
# (missing labels not harmful, the low bound) or ``h_missing_harmful`` (harmful, the high bound).
MECHANISM_SENSITIVITIES = {
    "missing_labels": (("h", "h_missing_harmful"), ("h_missing_harmful", "h")),
    "all_missing_harmful": (("h_missing_harmful", "h_missing_harmful"),) * 2,
}


def _contrast(cohort: Sequence[Mapping[str, Any]], name: str, x_field="h", y_field="h") -> list:
    minuend, subtrahend = CONTRASTS[name]
    return [b[x_field][minuend] - b[y_field][subtrahend] for b in cohort]


def mechanism_analysis(blocks: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict:
    """The mechanism cohort's one final analysis (amendment 4, section 3), at the cohort's end.

    ``blocks``: ``mechanism_blocks``' entries in their preassigned order; ``settings``: the
    fixation record's ``mechanism`` (``read_fixation``). The cohort is the first
    ``MECHANISM_TARGET`` jointly supported three-arm blocks in order among those started up to
    the cohort's end, or, at the cap of ``MECHANISM_CAP`` started and complete blocks, the
    jointly supported ones among them; a block of the window that is not complete on K, or a
    cohort block not labeled yet, raises ``ValueError``: it is computed once, at the end.

    Per cohort block, from the interpolation estimator at K* (``block_difference``'s h):
    C1 = h_pre-warmed − h_FA and C2 = h_AdamW − h_pre-warmed. Returns:

    - ``cohort``: its block numbers, n, how it ended and its window;
    - ``C1`` and ``C2``: mean, sd, SE and n;
    - ``sum_check``: C1 + C2 = AdamW − FA on the cohort, per block and in the mean, to
      ``ZERO_SE`` (raised otherwise); that is not the primary's estimate, whose cohort is the
      primary-supported one;
    - ``max_t``: ``max_t_interval`` at the record's level, resamples and the stream
      ``derive_seed(seed, "mechanism", "max_t")``, its q and an interval per contrast;
    - ``bonferroni``: ``symmetric_studentized`` per contrast at ``COMPARISON_LEVEL``, stream
      ``derive_seed(seed, "mechanism", "bonferroni", name)``;
    - ``readings`` (``component_readings``, margin the meaningful effect 5 pp): with at least
      ``MECHANISM_MINIMUM`` cohort blocks (``minimum_met``), each contrast on its max-t interval;
      a contrast whose observed SE ≤ ``ZERO_SE`` is not read, and the other is then read on its
      Bonferroni interval; below the minimum, none (the intervals are descriptive);
    - ``attrition``: the window's blocks not jointly supported, by reason;
    - ``sensitivities``: each of ``SENSITIVITIES`` for both contrasts, the max-t intervals of
      their low and high bounds on the same stream, from the low one's lower bound to the high
      one's upper bound, descriptive, without readings;
    - ``missing_labels`` per arm over the cohort, and ``blocks``: each cohort block's h,
      ``h_missing_harmful`` and missing labels per arm (the summary carries the pre-warmed
      arm's nowhere else)."""
    _check_mechanism(settings)
    level, resamples, seed = settings["level"], settings["resamples"], settings["seed"]
    window, cohort = [], []
    for block in blocks[:MECHANISM_CAP]:
        if not block["k_complete"]:
            raise ValueError(f"mechanism_analysis: block {block['block']} is not complete: the "
                             "analysis runs once, at the cohort's end")  # fmt: skip
        window.append(block)
        if block["three_arm_supported"]:
            if not block["complete"] or not set(ARMS) <= set(block["h"]):
                raise ValueError(f"mechanism_analysis: block {block['block']} of the cohort is "
                                 "not labeled yet: the analysis runs once, at the cohort's "
                                 "end")  # fmt: skip
            cohort.append(block)
            if len(cohort) == MECHANISM_TARGET:
                break
    if len(cohort) < MECHANISM_TARGET and len(window) < MECHANISM_CAP:
        raise ValueError(f"mechanism_analysis: the cohort has not ended ({len(cohort)} jointly "
                         f"supported blocks of {MECHANISM_TARGET}, {len(window)} blocks of the "
                         f"cap {MECHANISM_CAP})")  # fmt: skip
    n = len(cohort)
    values = {name: _contrast(cohort, name) for name in CONTRASTS}
    primary = [b["h"][ADAMW] - b["h"][FA] for b in cohort]
    deviation = max((abs(a + b - d) for a, b, d in zip(values["C1"], values["C2"], primary,
                                                         strict=True)), default=0.0)  # fmt: skip
    means = {name: statistics.fmean(v) if v else None for name, v in values.items()}
    total = statistics.fmean(primary) if primary else None
    mean_deviation = 0.0 if total is None else abs(means["C1"] + means["C2"] - total)
    if deviation > ZERO_SE or mean_deviation > ZERO_SE:
        raise ValueError(f"mechanism_analysis: C1 + C2 departs from AdamW − FA on the cohort by "
                         f"{max(deviation, mean_deviation)!r}")  # fmt: skip
    stream = derive_seed(seed, "mechanism", "max_t")
    joint = max_t_interval(values["C1"], values["C2"], level, resamples, stream)
    bonferroni = {name: symmetric_studentized(v, COMPARISON_LEVEL, resamples,
                                              derive_seed(seed, "mechanism", "bonferroni", name))
                  for name, v in values.items()}  # fmt: skip
    per = dict(zip(CONTRASTS, joint["contrasts"], strict=True))
    minimum_met = n >= MECHANISM_MINIMUM
    zero = [name for name in CONTRASTS if per[name]["zero_se"]]
    readings: dict[str, Any] = {}
    for name in CONTRASTS:
        basis, interval, why = "max_t", per[name]["interval"], None
        if not minimum_met:
            basis, why = None, f"fewer than {MECHANISM_MINIMUM} cohort blocks: descriptive"
        elif per[name]["zero_se"] is None or name in zero:
            basis, why = None, "an observed SE at or below ZERO_SE: not read"
        elif zero:
            basis, interval = "bonferroni", bonferroni[name]["interval"]
        readings[name] = {
            "basis": basis,
            "interval": interval if basis else None,
            "readings": (component_readings(interval, settings["effects"][0])
                         if basis else None),
            "why": why,
        }  # fmt: skip
    sensitivities = {}
    for name, sides in MECHANISM_SENSITIVITIES.items():
        bounds = [max_t_interval(*(_contrast(cohort, contrast, *fields) for contrast in CONTRASTS),
                                 level, resamples, stream)
                  for fields in sides]  # fmt: skip
        sensitivities[name] = {
            contrast: {
                "interval": (None if None in (bounds[0]["contrasts"][j]["interval"],
                                              bounds[1]["contrasts"][j]["interval"])
                             else [bounds[0]["contrasts"][j]["interval"][0],
                                   bounds[1]["contrasts"][j]["interval"][1]]),
                "low_mean": bounds[0]["contrasts"][j]["mean"],
                "high_mean": bounds[1]["contrasts"][j]["mean"],
            }
            for j, contrast in enumerate(CONTRASTS)
        }  # fmt: skip
    return {
        "settings": dict(settings),
        "cohort": {"blocks": [b["block"] for b in cohort], "n": n,
                   "ended_by": "target" if n == MECHANISM_TARGET else "cap",
                   "window": [window[0]["block"], window[-1]["block"]] if window else None},
        **{name: {"definition": f"{CONTRASTS[name][0]} - {CONTRASTS[name][1]}",
                  **{k: per[name][k] for k in ("mean", "sd", "se")}, "n": n}
           for name in CONTRASTS},
        "sum_check": {"adamw_minus_fa_mean": total, "C1_plus_C2_mean": (
            None if total is None else means["C1"] + means["C2"]),
            "max_block_deviation": deviation, "mean_deviation": mean_deviation,
            "tolerance": ZERO_SE, "holds": True},
        "max_t": {**{k: joint[k] for k in ("level", "resamples", "q", "draws", "infinite")},
                  "seed": seed, "stream_seed": stream,
                  **{name: per[name]["interval"] for name in CONTRASTS}},
        "bonferroni": {name: {"level": found["level"], "stream_seed": found["seed"],
                              **{k: found[k] for k in ("q", "interval", "zero_se", "infinite")}}
                       for name, found in bonferroni.items()},
        "readings": readings,
        "minimum_met": minimum_met,
        "attrition": _attrition(window),
        "sensitivities": sensitivities,
        # What the summary leaves out until now: the pre-warmed arm's harm and missing labels.
        "missing_labels": {arm: sum(b.get("missing", {}).get(arm, 0) for b in cohort)
                           for arm in ARMS},
        "blocks": [{"block": b["block"], "h": dict(b["h"]),
                    "h_missing_harmful": dict(b["h_missing_harmful"]),
                    "missing": dict(b.get("missing", {}))} for b in cohort],
    }  # fmt: skip


# ------------------------------------------------------------------ utility and validation


def utility_family(
    runs: Sequence[Trajectory],
    supports: Mapping[int, Mapping[str, Mapping[str, Any]]],
    theta0: Mapping[str, float],
    *,
    level: float = UTILITY_LEVEL,
    looks: int = len(LOOK_SIZES),
    arms: Sequence[str] = PRIMARY_ARMS,
) -> dict[str, Any]:
    """The simultaneous one-sided upper bounds on mean utility loss: the family is every
    (arm, rate) of ``arms`` in the bracket of a block supported for all of them (the primary
    contrast's arms; the three arms in a three-arm validation) (``supports``: block → arm →
    ``block_support``); each member's bound is over every block in ``supports`` where its run
    completed, at 1 − (1 − level)/(family size × looks) per metric. Passes when every MMLU bound
    is ≤ 5 pp and every factual bound ≤ 10 pp."""
    members = sorted({
        (arm, by_arm[arm]["bracket"][side]["rate"])
        for by_arm in supports.values() if all(by_arm[a]["supported"] for a in arms)
        for arm in arms for side in ("lo", "hi")
    })  # fmt: skip
    alpha = (1 - level) / (len(members) * looks) if members else None
    family = []
    for arm, lr in members:
        rows = [row for run in runs if run.arm == arm and rate(run.lr) == lr
                and run.seed in supports and (row := _endpoint(run)[0]) is not None]  # fmt: skip
        losses = [utility_loss(row, theta0) for row in rows]
        entry: dict[str, Any] = {"arm": arm, "rate": lr, "runs": len(losses)}
        for i, (metric, bound) in enumerate((("mmlu", MMLU_BOUND_PP),
                                             ("factual", FACTUAL_BOUND_PP))):  # fmt: skip
            values = [loss[i] for loss in losses]
            upper = None
            if len(values) >= 2:
                t = float(stats.t.ppf(1 - alpha, len(values) - 1))
                upper = statistics.fmean(values) + t * statistics.stdev(values) / math.sqrt(
                    len(values)
                )
            entry[f"{metric}_upper_pp"] = upper
            entry[f"{metric}_passes"] = upper is not None and upper <= bound
        family.append(entry)
    passes = bool(family) and all(e["mmlu_passes"] and e["factual_passes"] for e in family)
    return {"level": level, "looks": looks, "alpha_per_bound": alpha, "family": family,
            "passes": passes}  # fmt: skip


def validation_verdict(
    runs: Sequence[Trajectory],
    rates: Mapping[str, Sequence[float]],
    fa_controls: Mapping[float, Mapping[str, Any]],
    theta0: Mapping[str, float],
    kstar: float,
    w_max: float,
    *,
    seeds: Sequence[int] = VALIDATION_SEEDS,
    arms: Sequence[str] = PRIMARY_ARMS,
) -> dict[str, Any]:
    """The validation blocks' verdict over ``arms``, on numerics, K and utility only (harm is
    not an input): at least 10 blocks jointly supported for every arm of ``arms``; no run of
    those arms stopped by an instability rule; every frozen FA rate's control accepted
    (amendment 4: the same-path control of the numerical check that qualified the rate); the
    utility family over those arms passes, at a single reading (Bonferroni over the family
    alone). ``arms``: AdamW-25 and FA-25 (amendment 3's primary-arm validation, kept for the
    fallback), or the three arms (amendment 4's three-arm validation, review fa12). Ambiguous
    records (a second run of an arm, rate and seed, a run at a rate outside the frozen ones)
    leave no conclusion: ``passes`` None, with the problems."""
    arms = tuple(arms)
    if not set(PRIMARY_ARMS) <= set(arms) <= set(ARMS):
        raise ValueError(f"a validation reads the primary arms or the three arms, not {arms}")
    three = arms == ARMS
    found, problems = index(r for r in runs if r.arm in arms and r.seed in seeds)
    read = list(found.values())
    supports = {
        seed: {arm: block_support([r for r in read if r.arm == arm and r.seed == seed], kstar,
                                  w_max, rates=rates[arm]) for arm in arms}
        for seed in seeds
    }  # fmt: skip
    problems += [p for by_arm in supports.values() for s in by_arm.values() for p in s["problems"]]
    if problems:
        return {"passes": None, "reasons": ["incomplete records"], "problems": problems,
                "supports": supports, "arms": list(arms)}  # fmt: skip
    joint = [s for s in seeds if all(supports[s][arm]["supported"] for arm in arms)]
    reasons = []
    if len(joint) < VALIDATION_MINIMUM:
        reasons.append(f"{len(joint)} of {len(seeds)} blocks jointly supported"
                       f"{' for the three arms' if three else ''}, fewer than "
                       f"{VALIDATION_MINIMUM}")  # fmt: skip
    unstable = sorted(r.run_id for r in read if r.stop in INSTABILITY)
    if unstable:
        reasons.append(f"{'runs of the three arms' if three else 'primary-arm runs'} stopped by "
                       f"an instability rule: {unstable}")  # fmt: skip
    accepted = {rate(lr): v for lr, v in fa_controls.items()}
    rejected = [lr for lr in rates[FA] if not accepted.get(rate(lr), {}).get("accepted")]
    if rejected:
        reasons.append(f"FA controls not accepted at {rejected}")
    family = utility_family(read, supports, theta0, looks=1, arms=arms)
    if not family["passes"]:
        reasons.append("the utility family does not pass")
    causes = {}
    for arm in arms:
        lost = [supports[s][arm] for s in seeds if not supports[s][arm]["supported"]]
        causes[arm] = {
            "cause": dict(Counter(s["cause"] for s in lost)),
            "k_cause": dict(Counter(str(s["k_cause"]) for s in lost)),  # None: K alone supports
            "failure_kinds": dict(Counter(f["kind"] for s in lost for f in s["failures"])),
        }
    return {"passes": not reasons, "reasons": reasons, "problems": [], "arms": list(arms),
            "jointly_supported": joint, "causes": causes, "supports": supports,
            "unstable": unstable, "utility": family}  # fmt: skip


# ------------------------------------------------------------------ the stage as a whole


def seed_role(seed: int, *, timecourse: bool = False) -> str | None:
    """The role of a regular run on ``seed``: discovery, numerical, validation or confirmation;
    with ``timecourse``, that of a time-course run: timecourse on the first 10 confirmation
    seeds (``TIMECOURSE_SEEDS``), none elsewhere. Numerical and time-course runs never count as
    discovery, validation or confirmation blocks."""
    if timecourse:
        return TIMECOURSE if seed in TIMECOURSE_SEEDS else None
    endpoints = {"discovery": DISCOVERY_SEEDS, "numerical": NUMERICAL_SEEDS,
                 "validation": VALIDATION_SEEDS, "confirmation": CONFIRMATION_SEEDS}  # fmt: skip
    for role, seeds in endpoints.items():
        if seed in seeds:
            return role
    return None


def measures_rayleigh(seed: int) -> bool:
    """Whether the regular runs of ``seed`` measure the endpoint Rayleigh quotient: those of the
    first 10 confirmation blocks, in every arm."""
    return seed in CONFIRMATION_SEEDS[:RAYLEIGH_BLOCKS]


def _number(value: Any) -> bool:
    return _finite(value) and not isinstance(value, bool)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_fixation(path: str | Path) -> dict[str, Any]:
    """The values the companion simulation fixes, from its fixation record (a JSON object):

    - ``w_max`` (> 0);
    - ``levels``: the looks' two-sided confidence levels, one per look, each in (0.5, 1);
    - ``se_floor`` and ``bias_allowance`` (≥ 0, in the units of Δ);
    - ``interval``: the primary interval, one of ``INTERVALS``;
    - ``bootstrap_resamples`` (an integer ≥ ``MIN_RESAMPLES``) and ``bootstrap_seed`` (an
      integer): the bootstrap's resamples per look and the seed of its streams;
    - ``looks`` and ``cap``: the look sizes and the cap, which must be ``LOOK_SIZES`` and
      ``CAP``;
    - ``equivalence_min_supported`` (an integer ≥ 0): equivalence is concluded only at a look
      with at least that many supported blocks, 0 at every look;
    - ``direction_min_supported`` (an integer ≥ 0): below that many supported blocks a look
      concludes nothing (neither direction nor equivalence), 0 at every look;
    - ``secondary_interval``: the secondary contrasts' interval, one of ``INTERVALS``;
    - ``secondary_status``: how they are reported, one of ``SECONDARY_STATUSES``;
    - ``mechanism`` (amendment 4, section 3): an object with exactly ``MECHANISM_KEYS``:
      ``target``, ``cap``, ``minimum``, ``zero_se``, ``comparison_level`` and ``effects``, each
      equal to the code's (``MECHANISM_FIXED``: 60, 80, 36, 1e-12, 0.975, [0.05, 0.025]);
      ``level``, the max-t intervals' simultaneous level, in [0.95, 1); ``resamples`` (an
      integer ≥ ``MIN_RESAMPLES``) and ``seed`` (an integer);
    - ``audit`` (amendment 4, the terminal audit of the judge, design v3;
      ``analysis.study2_audit``): an object with exactly ``AUDIT_KEYS``: ``sizes``,
      ``selection``, ``weight``, ``interval``, ``arm_level``, ``bias_level``,
      ``missing_reference`` and ``agreement_check_share``, each equal to the code's
      (``AUDIT_FIXED``), and ``seed`` (an integer), the audit draw's;
    - amendment 4's other fixed values (``FIXED_KEYS``), each an object whose keys and values
      are exactly the code's (``fixed_values``): ``candidates`` (the order, (U, B) = (8, 20) then
      (7, 20), for the three arms, then for the primary pair, else ``infeasible`` with its
      B = 20 report), ``numerical`` (the seeds, the cases, the norm, the survival share and its
      retry, the bound's fraction of K*, the steps, the Gaussian directions' seeds, the
      reference with its fallback and twin A's consistency rule), ``validation`` (the seeds, the
      10 jointly supported blocks of 12, the three arms or, in the fallback, the primary pair,
      and the simultaneous utility bounds), ``timecourse`` (its seeds and steps), ``drift`` (its
      seed, batches, halves, optimizers and v₀'s batches) and ``scalars`` (the arms, and the
      pre-warmed arm's departure steps).

    Each is required, the integers as JSON integers; a missing or malformed one raises
    ``ValueError``, and none is ever defaulted."""
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict):
        raise ValueError(f"{path}: a fixation record is a JSON object")
    missing = [key for key in FIXATION_KEYS if key not in record]
    if missing:
        raise ValueError(f"{path}: the fixation record lacks {missing}")
    w_max, levels = record["w_max"], record["levels"]
    problems = []
    if not (_number(w_max) and w_max > 0):
        problems.append(f"w_max {w_max!r} is not a positive number")
    if not (isinstance(levels, list) and len(levels) == len(LOOK_SIZES)
            and all(_number(level) and 0.5 < level < 1 for level in levels)):  # fmt: skip
        problems.append(f"levels {levels!r} are not {len(LOOK_SIZES)} confidence levels in "
                        "(0.5, 1)")  # fmt: skip
    for key in ("se_floor", "bias_allowance"):
        if not (_number(record[key]) and record[key] >= 0):
            problems.append(f"{key} {record[key]!r} is not a number ≥ 0")
    if record["interval"] not in INTERVALS:
        problems.append(f"interval {record['interval']!r} is not one of {list(INTERVALS)}")
    resamples = record["bootstrap_resamples"]
    if not (_integer(resamples) and resamples >= MIN_RESAMPLES):
        problems.append(f"bootstrap_resamples {resamples!r} is not an integer ≥ {MIN_RESAMPLES}")
    if not _integer(record["bootstrap_seed"]):
        problems.append(f"bootstrap_seed {record['bootstrap_seed']!r} is not an integer")
    sizes = record["looks"]
    if not (isinstance(sizes, list) and all(_integer(size) for size in sizes)
            and tuple(sizes) == LOOK_SIZES):  # fmt: skip
        problems.append(f"looks {sizes!r} are not the look sizes {list(LOOK_SIZES)}")
    if not (_integer(record["cap"]) and record["cap"] == CAP):
        problems.append(f"cap {record['cap']!r} is not the cap {CAP}")
    for key in ("equivalence_min_supported", "direction_min_supported"):
        if not (_integer(record[key]) and record[key] >= 0):
            problems.append(f"{key} {record[key]!r} is not an integer ≥ 0")
    for key, allowed in (("secondary_interval", INTERVALS),
                         ("secondary_status", SECONDARY_STATUSES)):  # fmt: skip
        if record[key] not in allowed:
            problems.append(f"{key} {record[key]!r} is not one of {list(allowed)}")
    problems += mechanism_problems(record["mechanism"])
    problems += audit_problems(record["audit"])
    fixed = fixed_values()
    for key in FIXED_KEYS:
        problems += fixed_problems(key, record[key], fixed[key])
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return {
        "w_max": float(w_max),
        "levels": [float(level) for level in levels],
        "se_floor": float(record["se_floor"]),
        "bias_allowance": float(record["bias_allowance"]),
        "interval": record["interval"],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": record["bootstrap_seed"],
        "looks": list(sizes),
        "cap": record["cap"],
        "equivalence_min_supported": record["equivalence_min_supported"],
        "direction_min_supported": record["direction_min_supported"],
        "secondary_interval": record["secondary_interval"],
        "secondary_status": record["secondary_status"],
        "mechanism": _mechanism_settings(record["mechanism"]),
        "audit": _audit_settings(record["audit"]),
        **{key: copy.deepcopy(record[key]) for key in FIXED_KEYS},
    }


def _same(value: Any, fixed: Any) -> bool:
    """A fixation value equal to the code's: integers as JSON integers, numbers as numbers,
    strings as strings, a list as a list of them and an object as one with the same keys and
    values."""
    if isinstance(fixed, list):
        return (isinstance(value, list) and len(value) == len(fixed)
                and all(_same(v, f) for v, f in zip(value, fixed, strict=True)))  # fmt: skip
    if isinstance(fixed, dict):
        return (isinstance(value, dict) and set(value) == set(fixed)
                and all(_same(value[key], f) for key, f in fixed.items()))  # fmt: skip
    if isinstance(fixed, str):
        return isinstance(value, str) and value == fixed
    if isinstance(fixed, int):
        return _integer(value) and value == fixed
    return _number(value) and value == fixed


def mechanism_problems(value: Any) -> list[str]:
    """How the fixation record's ``mechanism`` departs from what ``read_fixation`` requires (none:
    it holds)."""
    if not isinstance(value, dict):
        return [f"mechanism {value!r} is not an object with {list(MECHANISM_KEYS)}"]
    problems = []
    missing = [key for key in MECHANISM_KEYS if key not in value]
    if missing:
        problems.append(f"mechanism lacks {missing}")
    unknown = sorted(set(value) - set(MECHANISM_KEYS))
    if unknown:
        problems.append(f"mechanism has keys {unknown} it does not know")
    for key, fixed in MECHANISM_FIXED.items():
        if key in value and not _same(value[key], fixed):
            problems.append(f"mechanism {key} {value[key]!r} is not the code's {fixed!r}")
    level = value.get("level")
    if "level" in value and not (_number(level) and 0.95 <= level < 1):
        problems.append(f"mechanism level {level!r} is not a level in [0.95, 1)")
    resamples = value.get("resamples")
    if "resamples" in value and not (_integer(resamples) and resamples >= MIN_RESAMPLES):
        problems.append(f"mechanism resamples {resamples!r} is not an integer ≥ {MIN_RESAMPLES}")
    if "seed" in value and not _integer(value["seed"]):
        problems.append(f"mechanism seed {value['seed']!r} is not an integer")
    return problems


def _mechanism_settings(value: Mapping[str, Any]) -> dict[str, Any]:
    """The fixation record's ``mechanism``, checked (``mechanism_problems``), as the analysis
    reads it."""
    problems = mechanism_problems(value)
    if problems:
        raise ValueError("; ".join(problems))
    return {**{key: value[key] for key in MECHANISM_FIXED},
            "effects": [float(effect) for effect in value["effects"]],
            "zero_se": float(value["zero_se"]),
            "comparison_level": float(value["comparison_level"]),
            "level": float(value["level"]), "resamples": value["resamples"],
            "seed": value["seed"]}  # fmt: skip


def audit_problems(value: Any) -> list[str]:
    """How the fixation record's ``audit`` departs from what ``read_fixation`` requires (none: it
    holds)."""
    if not isinstance(value, dict):
        return [f"audit {value!r} is not an object with {list(AUDIT_KEYS)}"]
    problems = []
    missing = [key for key in AUDIT_KEYS if key not in value]
    if missing:
        problems.append(f"audit lacks {missing}")
    unknown = sorted(set(value) - set(AUDIT_KEYS))
    if unknown:
        problems.append(f"audit has keys {unknown} it does not know")
    for key, fixed in AUDIT_FIXED.items():
        if key in value and not _same(value[key], fixed):
            problems.append(f"audit {key} {value[key]!r} is not the code's {fixed!r}")
    if "seed" in value and not _integer(value["seed"]):
        problems.append(f"audit seed {value['seed']!r} is not an integer")
    return problems


def _audit_settings(value: Mapping[str, Any]) -> dict[str, Any]:
    """The fixation record's ``audit``, checked (``audit_problems``), as the audit reads it."""
    problems = audit_problems(value)
    if problems:
        raise ValueError("; ".join(problems))
    return {"sizes": dict(value["sizes"]), "selection": value["selection"],
            "weight": value["weight"], "interval": value["interval"],
            "arm_level": float(value["arm_level"]), "bias_level": float(value["bias_level"]),
            "missing_reference": value["missing_reference"], "seed": value["seed"],
            "agreement_check_share": float(value["agreement_check_share"])}  # fmt: skip


@functools.cache
def _gpu_fixed() -> dict[str, Any]:
    """What the GPU tools fix for the fixation record's ``drift`` and ``scalars``: the drift
    diagnostic's tracked settings (configs/experiments/drift_diagnostic.yaml, read through its
    own ``DriftScope``) and the pre-warmed arm's departure steps (``schedule.DEPARTURE_STEPS``).
    Imported when first needed: the tools load torch."""
    from .. import drift, schedule

    scope = drift.DriftScope.of(load_config(CONFIG_ROOT / "experiments" / "drift_diagnostic.yaml"))
    return {"drift": {"seed": scope.seed, "batches": scope.batches,
                      "halves": [[0, scope.half - 1], [scope.half, scope.batches - 1]],
                      "optimizers": list(scope.optimizers), "v0_batches": scope.v0_batches},
            "departure_steps": list(schedule.DEPARTURE_STEPS)}  # fmt: skip


def fixed_values() -> dict[str, dict[str, Any]]:
    """The values of amendment 4's fixed fixation keys (``FIXED_KEYS``) as the code holds them;
    a fixation record must hold each exactly (``fixed_problems``):

    - ``candidates``: the order of ``CANDIDATES`` as (U, B) objects, for ``arms`` (the three
      arms), then for ``fallback_arms`` (the primary pair), else ``end`` ``infeasible`` with its
      report of ``scope`` B=20 (review fa22: no B = 5 candidate);
    - ``numerical``: the seeds ``NUMERICAL_SEEDS``, the ``cases`` (every selected rate on the
      first seed, each arm's U-rate on the others: ``numerical_cases``), and the frozen policy
      of ``numerical_policy`` (the norm, the survival share, the retry scale, the bound's
      fraction of K*, the steps and twin A's consistency rule of review fa21, with the Gaussian
      directions' seeds and the reference's modes as the numerical-check tool fixes them);
    - ``validation``: the seeds ``VALIDATION_SEEDS``, at least ``VALIDATION_MINIMUM`` blocks
      jointly supported for ``arms`` (the three arms; the primary pair, ``fallback_arms``, in
      the fallback), and the simultaneous one-sided utility bounds at ``UTILITY_LEVEL`` against
      the MMLU and factual limits in percentage points (Bonferroni over the family);
    - ``timecourse``: ``TIMECOURSE_SEEDS`` and ``TIMECOURSE_STEPS``;
    - ``drift``: the drift diagnostic's seed, batches, halves, optimizers and v₀'s batches;
    - ``scalars``: the arms whose every regular run records the update scalars, and the
      pre-warmed arm's denominator departure steps."""
    tool, gpu = _tool_policy(), _gpu_fixed()
    return {
        "candidates": {"order": [{"U": upper, "B": batch} for upper, batch in CANDIDATES],
                       "arms": list(ARMS), "fallback_arms": list(PRIMARY_ARMS),
                       "end": "infeasible", "scope": SCOPE},
        "numerical": {"seeds": list(NUMERICAL_SEEDS),
                      "cases": {"every_rate": [NUMERICAL_SEEDS[0]],
                                "u_rate": list(NUMERICAL_SEEDS[1:])},
                      "norm": NUMERICAL_POLICY["norm"],
                      "survival_share": NUMERICAL_POLICY["survival_share"],
                      "retry_scale": NUMERICAL_POLICY["retry_scale"],
                      "bound_fraction": NUMERICAL_BOUND_FRACTION,
                      "steps": list(NUMERICAL_POLICY["steps"]),
                      "gaussian": dict(tool["gaussian"]), "reference": tool["reference"],
                      "twin_a_consistency": dict(NUMERICAL_POLICY["twin_a_consistency"])},
        "validation": {"seeds": list(VALIDATION_SEEDS), "minimum": VALIDATION_MINIMUM,
                       "arms": list(ARMS), "fallback_arms": list(PRIMARY_ARMS),
                       "utility_level": UTILITY_LEVEL,
                       "bounds_pp": {"mmlu": MMLU_BOUND_PP, "factual": FACTUAL_BOUND_PP}},
        "timecourse": {"seeds": list(TIMECOURSE_SEEDS), "steps": list(TIMECOURSE_STEPS)},
        "drift": copy.deepcopy(gpu["drift"]),
        "scalars": {"arms": list(ARMS), "departure_arm": PREWARMED,
                    "departure_steps": list(gpu["departure_steps"])},
    }  # fmt: skip


def fixed_problems(key: str, value: Any, fixed: Mapping[str, Any]) -> list[str]:
    """How the fixation record's ``key`` (one of ``FIXED_KEYS``) departs from the code's values
    ``fixed`` (``fixed_values``): it must be an object with exactly their keys, each value equal
    to the code's (``_same``: integers as JSON integers, lists and objects whole); none: it
    holds."""
    if not isinstance(value, dict):
        return [f"{key} {value!r} is not an object with {list(fixed)}"]
    problems = []
    missing = [name for name in fixed if name not in value]
    if missing:
        problems.append(f"{key} lacks {missing}")
    unknown = sorted(set(value) - set(fixed))
    if unknown:
        problems.append(f"{key} has keys {unknown} it does not know")
    for name, frozen in fixed.items():
        if name in value and not _same(value[name], frozen):
            problems.append(f"{key} {name} {value[name]!r} is not the code's {frozen!r}")
    return problems


def versions_problems(
    path: str | Path, *, fixation: str | Path, judge: str | None = None, repo: Path = REPO_ROOT
) -> list[str]:
    """How the running checkout and configs depart from the versions record at ``path``
    (``VERSIONS_RECORD``, written by ``scripts/study2_short_summary.py versions``); none: they
    agree. The record must exist and be committed: a file of the checkout ``repo`` that Git
    tracks, unchanged at HEAD. And:

    - ``code.commit`` is the running HEAD, or an ancestor of it whose changes up to HEAD touch the
      versions record alone (committing the record makes such a child), or any ancestor when the
      record allows it (``code.allow_descendants`` true); no tracked file has uncommitted changes
      (the record included), and no file is untracked unless Git ignores it (the checkout must
      equal the recorded commit: local work directories belong in ``.gitignore``);
    - ``model`` is the name and revision of the stage's model (``VERSIONS_CONFIG``);
    - ``judge``'s identity is that of its config (``judge.judge_identity``), and the config is
      ``judge/<judge>`` when ``judge`` is given;
    - ``fixation.sha256`` is the sha256 of the fixation record ``fixation``.
    """
    path = Path(path)
    if not path.is_file():
        return [f"{path} is missing: write the versions record (scripts/study2_short_summary.py "
                "versions) and commit it first"]  # fmt: skip
    try:
        record = json.loads(path.read_text())
    except ValueError:
        record = None
    if not isinstance(record, dict):
        return [f"{path} is not a JSON object"]
    problems = []
    inside = path.resolve().is_relative_to(repo.resolve())
    own = path.resolve().relative_to(repo.resolve()).as_posix() if inside else None
    if own is None:
        problems.append(f"{path} is not in the checkout {repo}: the record must be committed")
    elif not tracked(own, repo):
        problems.append(f"{path} is not committed: commit the versions record first")
    code = record.get("code") if isinstance(record.get("code"), dict) else {}
    commit, descendants = code.get("commit"), code.get("allow_descendants", False)
    head = git_state(repo).get("commit")
    changes = code_changes(commit, repo) if isinstance(commit, str) and commit else None
    if head is None:
        problems.append("the running code is not a Git checkout: its commit is unknown")
    elif changes is None:
        problems.append(f"the running HEAD {head} is neither the recorded commit {commit!r} nor "
                        "a descendant of it")  # fmt: skip
    else:
        committed = [name for name in changes["committed"] if name != own]
        uncommitted = changes["uncommitted"]
        if own in uncommitted:
            problems.append(f"{path} differs from its committed version at HEAD")
            uncommitted = [name for name in uncommitted if name != own]
        # An untracked record is refused above, as not committed.
        untracked = [name for name in changes["untracked"] if name != own]
        if committed and descendants is not True:
            problems.append(f"the running HEAD {head} differs from the recorded commit "
                            f"{commit} in {len(committed)} files (first: {committed[0]}), and "
                            "the record does not allow descendants")  # fmt: skip
        if uncommitted:
            problems.append(f"the checkout has {len(uncommitted)} uncommitted changes (first: "
                            f"{uncommitted[0]})")  # fmt: skip
        if untracked:
            problems.append(f"the checkout has {len(untracked)} untracked files (first: "
                            f"{untracked[0]})")  # fmt: skip
    cfg = load_config(*VERSIONS_CONFIG)
    model = {"name": str(cfg.model.name), "revision": str(cfg.model.revision)}
    if record.get("model") != model:
        problems.append(f"the record's model {record.get('model')!r} is not the configs' {model}")
    recorded = record.get("judge") if isinstance(record.get("judge"), dict) else {}
    name = recorded.get("config")
    if judge is not None and name != f"judge/{judge}":
        problems.append(f"the record's judge is {name!r}, not judge/{judge}")
    try:
        identity = judge_identity(load_config(str(name)))
    except Exception as exc:  # an unknown or incomplete judge config
        problems.append(f"the record's judge config {name!r} does not load: {exc}")
    else:
        differ = [key for key in IDENTITY_FIELDS if recorded.get(key) != identity[key]]
        if differ:
            problems.append(f"the record's judge identity differs from {name}'s in {differ}")
    digest = hashlib.sha256(Path(fixation).read_bytes()).hexdigest()
    if (record.get("fixation") or {}).get("sha256") != digest:
        problems.append(f"the record's fixation sha256 is not that of {fixation}")
    return problems


def run_facts(run_dir: str | Path) -> dict[str, Any]:
    """What a run's records hold beyond its ``Trajectory``: from ``config.yaml`` its role
    (``study2_short.role``, else Stage A's ``stage_a.role``), whether it measures the Rayleigh
    quotient and whether its evaluation is K-only (``eval.k_only``); from ``meta.json`` the code
    commit it ran (``git.commit``), whether its checkout was dirty (``git.dirty``: None when not
    recorded), the preconditioner stream its v₀ was estimated from
    (``inputs.streams.preconditioner``: path and sha256), None for a run without one, and its
    ``config_hash`` (``config.config_hash`` of its resolved config, as the tracker records it)."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    role = OmegaConf.select(cfg, "study2_short.role", default=None)
    if role is None:
        role = OmegaConf.select(cfg, "stage_a.role", default=None)
    rayleigh = OmegaConf.select(cfg, "eval.rayleigh.enabled", default=False)
    k_only = OmegaConf.select(cfg, "eval.k_only", default=False) is True
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    stream = ((meta.get("inputs") or {}).get("streams") or {}).get("preconditioner")
    preconditioner = None
    if isinstance(stream, dict):
        preconditioner = {"path": stream.get("path"), "sha256": stream.get("sha256")}
    git = meta.get("git") or {}
    return {"role": role, "rayleigh": bool(rayleigh), "k_only": k_only,
            "commit": git.get("commit"), "dirty": git.get("dirty"),
            "preconditioner": preconditioner,
            "config_hash": meta.get("config_hash")}  # fmt: skip


SCALARS = "update_scalars.jsonl"
DEPARTURE_STEPS = (5, 10, 25)  # schedule.DEPARTURE_STEPS: the pre-warmed denominator's departure
DEPARTURE_FIELDS = ("norm_ratio", "median", "p90")


def _scalar_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """The rows of an update_scalars.jsonl, and why it does not read (None when it does)."""
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        return [], f"does not read: {exc!r}"
    if not all(isinstance(row, dict) and _integer(row.get("step")) for row in rows):
        return [], "a row without an integer step"
    return rows, None


def scalars_coverage(
    runs: Sequence[Trajectory], facts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """The update scalars' coverage (amendment 4, section 4; review of Task 6, M4), per regular
    run of the stage's arms of every role (``ROLES``, in ``facts``: ``run_facts`` by run id;
    Stage A's runs and the FA controls are not this stage's regular runs): whether the run's
    ``update_scalars.jsonl`` exists with one row per update step, 1 to the steps it took (25 for
    a run that finished), and for the pre-warmed arm a ``d_departure`` with ``DEPARTURE_FIELDS``
    at each of the steps 5, 10 and 25 it reached; ``enabled`` is its config's
    ``tracking.update_scalars`` (None when the config does not read). A gap is reported and
    gates nothing: it is no problem of ``progress``. Only K-free records are read."""
    entries = []
    for run in sorted(runs, key=lambda r: r.run_id):
        role = (facts.get(run.run_id) or {}).get("role")
        if run.arm not in ARMS or run.role is not None or role not in ROLES:
            continue
        try:
            config = OmegaConf.load(run.run_dir / "config.yaml")
            enabled = OmegaConf.select(config, "tracking.update_scalars", default=False) is True
        except Exception:  # a config that does not read: reported, never gating
            enabled = None
        taken = run.steps if _integer(run.steps) else 0
        expected = list(range(1, min(taken, STEPS) + 1))
        path = run.run_dir / SCALARS
        rows, unreadable = _scalar_rows(path) if path.is_file() else ([], None)
        steps = Counter(row["step"] for row in rows)
        entry: dict[str, Any] = {
            "run": run.run_id, "role": role, "arm": run.arm, "steps_taken": taken,
            "enabled": enabled, "file": path.is_file(), "unreadable": unreadable,
            "rows": len(rows), "missing_steps": [s for s in expected if s not in steps],
            "repeated_steps": sorted(s for s, count in steps.items() if count > 1),
            "unexpected_steps": sorted(s for s in steps if s not in expected),
        }  # fmt: skip
        if run.arm == PREWARMED:
            departed = {row["step"] for row in rows if isinstance(row.get("d_departure"), dict)
                        and all(k in row["d_departure"] for k in DEPARTURE_FIELDS)}  # fmt: skip
            entry["departure_missing"] = [s for s in DEPARTURE_STEPS
                                          if s <= taken and s not in departed]  # fmt: skip
        entry["gap"] = bool(not entry["file"] or unreadable or entry["missing_steps"]
                            or entry["repeated_steps"] or entry["unexpected_steps"]
                            or entry.get("departure_missing"))  # fmt: skip
        entries.append(entry)
    roles = Counter(entry["role"] for entry in entries)
    gaps = Counter(entry["role"] for entry in entries if entry["gap"])
    return {"runs": entries,
            "by_role": {role: {"runs": roles[role], "gaps": gaps[role]} for role in ROLES
                        if roles[role]},
            "gaps": sum(gaps.values()), "status": "reported; a gap gates nothing"}  # fmt: skip


def preconditioner_pairing(
    runs: Sequence[Trajectory], facts: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """One v₀ per seed and batch: every run that estimates it (FA-25, pre-warmed AdamW-25 and the
    FA controls, Stage A's FA grids included) read one preconditioner stream, the same path with
    the same sha256. A run of those arms that records none is a problem too."""
    problems = []
    streams: dict[tuple[int, int], dict[tuple[str, str], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        if run.arm not in V0_ARMS:
            continue
        stream = (facts.get(run.run_id) or {}).get("preconditioner") or {}
        if not (stream.get("path") and stream.get("sha256")):
            problems.append(f"{run.run_id} records no preconditioner stream (v₀)")
            continue
        streams[(run.seed, run.batch_size)][(stream["path"], stream["sha256"])].append(run.run_id)
    for (seed, batch), found in sorted(streams.items()):
        if len(found) > 1:
            read = "; ".join(f"{path} sha256 {digest[:12]} by {len(ids)} runs ({ids[0]}, …)"
                             for (path, digest), ids in sorted(found.items()))  # fmt: skip
            problems.append(f"seed {seed}, B={batch}: v₀ from {len(found)} preconditioner "
                            f"streams: {read}")  # fmt: skip
    return problems


def read_abandonment(path: str | Path) -> dict[str, str]:
    """The launcher attempts the operator abandoned, by the journal's ``run_id`` of the attempt,
    with the reason: from an abandonment record, a JSON object ``{"abandoned": [{"run_id": ID,
    "reason": TEXT}, ...]}`` (other keys, such as ``what``, are free), kept outside the checkout
    under ``$RUNS_DIR`` and archived with an inventory before it is read (committed, it would
    make HEAD a descendant that ``versions_problems`` refuses). Each attempt once, each reason a
    non-empty string; anything else raises ``ValueError``."""
    record = json.loads(Path(path).read_text())
    entries = record.get("abandoned") if isinstance(record, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: an abandonment record is a JSON object with an 'abandoned' list")
    out: dict[str, str] = {}
    for number, entry in enumerate(entries, 1):
        run_id = entry.get("run_id") if isinstance(entry, dict) else None
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if not (isinstance(run_id, str) and run_id and isinstance(reason, str) and reason.strip()):
            raise ValueError(f"{path}, entry {number}: a run_id and a non-empty reason")
        if run_id in out:
            raise ValueError(f"{path}: the attempt {run_id} is abandoned twice")
        out[run_id] = reason
    return out


def exhausted_runs(
    records: Sequence[Mapping[str, Any]],
    experiment: str = "study2_short",
    abandoned: Mapping[str, str] | None = None,
) -> dict[tuple, str]:
    """The regular runs that the launcher's journal (``launch.launch_records``) shows exhausted
    after their recovery round: their cell's recovery round logged, at least 2 × MAX_ATTEMPTS
    attempts, none of which ended other than technical, and the last of them (by attempt number,
    the one after the recovery round) ended: an end record with the outcome technical, or the
    attempt is in ``abandoned`` (``read_abandonment``: the operator's record that it will never
    end). A last attempt without either may still be running, and its run is not given up. An
    earlier attempt without an end record was settled by the start of a later one (the launcher
    counts it technical before it starts the next). By key (arm, batch, rate, seed), from the
    logged command composed as the launcher composed it, with the run's label; a time-course run
    (role timecourse) by (arm, batch, rate, seed, ``TIMECOURSE``), so that it never stands for
    the endpoint run of its block. An abandoned attempt the journal does not show started, or
    shows ended, raises ``ValueError``."""
    from ..launch import MAX_ATTEMPTS  # the launcher imports this package's stage_a

    abandoned = abandoned or {}
    recovered = {(r.get("experiment"), r.get("cell")) for r in records
                 if r.get("event") == "recovery"}  # fmt: skip
    ends = {r.get("run_id"): r.get("outcome") for r in records if r.get("event") == "end"}
    started = {r.get("run_id") for r in records if r.get("event") == "start"}
    for run_id in sorted(abandoned):
        if run_id not in started:
            raise ValueError(f"the abandoned attempt {run_id} has no start in the launch journal")
        if run_id in ends:
            raise ValueError(f"the abandoned attempt {run_id} ended ({ends[run_id]}): it is not "
                             "unresolved")  # fmt: skip
    attempts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("event") == "start" and record.get("experiment") == experiment:
            attempts[record["config_hash"]].append(record)
    out = {}
    for group in attempts.values():
        if any(ends.get(r["run_id"], "technical") != "technical" for r in group):
            continue
        label = group[-1]["run"]
        if (experiment, label.rsplit("/s", 1)[0]) not in recovered:
            continue
        if len(group) < 2 * MAX_ATTEMPTS:
            continue
        last = max(enumerate(group), key=lambda pair: (pair[1].get("attempt") or 0, pair[0]))[1]
        if ends.get(last["run_id"]) != "technical" and last["run_id"] not in abandoned:
            continue  # the last attempt has not ended: it may still be running
        command = last["command"]
        cfg = load_config(command[1], command[2:])
        key = (arm_of(cfg), int(cfg.train.batch_size), rate(float(cfg.optimizer.lr)),
               int(cfg.seed))  # fmt: skip
        if OmegaConf.select(cfg, "study2_short.role", default=None) == TIMECOURSE:
            key = (*key, TIMECOURSE)
        out[key] = label
    return out


_ISSUED = re.compile(r"-blocks-(\d+)-(\d+)(?:-rayleigh)?/s\d+$")


def issued_ranges(
    records: Sequence[Mapping[str, Any]], experiment: str = "study2_short"
) -> set[tuple[int, int]]:
    """The confirmation block ranges whose campaigns the launcher started (their cells' names
    end in ``-blocks-FIRST-LAST``)."""
    return {(int(m[1]), int(m[2])) for r in records
            if r.get("event") == "start" and r.get("experiment") == experiment
            and (m := _ISSUED.search(str(r.get("run", ""))))}  # fmt: skip


def issued_w_max(
    records: Sequence[Mapping[str, Any]], experiment: str = "study2_short"
) -> dict[str, Any]:
    """The w_max that each issued campaign's header recorded (the final review of gate B, I4):
    by the label of each run the launcher started (its journal's start record carries its
    campaign's ``header``, ``launch.run_campaign``). Starts logged before the header existed
    record none and are left out: not recorded, so nothing to compare."""
    return {str(r.get("run")): r["header"]["w_max"] for r in records
            if r.get("event") == "start" and r.get("experiment") == experiment
            and isinstance(r.get("header"), dict) and "w_max" in r["header"]}  # fmt: skip


def w_max_problems(w_max: float, issued: Mapping[str, Any] | None) -> list[str]:
    """Why the replay is refused (the final review of gate B, I4): w_max is frozen since S1, and
    a value other than one recorded in an issued campaign's header (``issued_w_max``) would
    replay other triples and digests under the runs made so far. One problem per value that
    differs, naming a run of its campaigns."""
    by_value: dict[str, list[str]] = defaultdict(list)
    for label, value in sorted((issued or {}).items()):
        by_value[json.dumps(value)].append(label)
    problems = []
    for shown, labels in sorted(by_value.items()):
        value = json.loads(shown)
        if _number(value) and float(value) == float(w_max):
            continue
        others = f" and {len(labels) - 1} other run(s)" if len(labels) > 1 else ""
        problems.append(f"w_max {w_max!r} differs from the w_max {value!r} recorded in the header "
                        f"of the issued campaign of {labels[0]}{others}: w_max is frozen since S1 "
                        "(amendment 4), and the replay is refused under another value")  # fmt: skip
    return problems


def _unbound(spec: Mapping[str, Any]) -> dict[str, Any]:
    """A control's schedule without its contamination bound, which its campaign sets."""
    instability = dict(spec.get("instability") or {})
    instability.pop("contamination_k", None)
    return {**spec, "instability": instability}


@functools.cache
def prescribed() -> dict[str, Any]:
    """The model, protocol versions, schedules and monitor the stage's campaigns prescribe, read
    from its configs."""
    regular = load_config(REGULAR, ["study2_short.role=discovery"])
    control = load_config(FA_CONTROLS)
    return {
        "model": f"{regular.model.name}@{regular.model.revision}",
        "protocol_version": int(regular.protocol_version),
        "control_protocol_version": int(control.protocol_version),
        "short": schedule_spec(regular),
        "timecourse": schedule_spec(load_config(REGULAR, list(TIMECOURSE_OVERRIDES))),
        "control": _unbound(schedule_spec(control)),
        "monitor": monitor_spec(regular),
    }


def run_prescription(arm: str, batch: int, lr: float, seed: int, role: str) -> tuple[str, list]:
    """The config and overrides the stage's campaigns give the regular run of ``arm`` at ``batch``
    and ``lr`` on ``seed`` with ``role`` (the launcher adds ``seed=N``;
    scripts/study2_short_campaigns.py builds its cells on them): experiments/study2_short_rayleigh
    on the first 10 confirmation blocks, experiments/study2_short_k_only for the numerical runs,
    experiments/study2_short otherwise; the arm's optimizer, B, the rate, and the role, or for
    the time course its role and evaluation steps (``TIMECOURSE_OVERRIDES``)."""
    config = RAYLEIGH if measures_rayleigh(seed) else K_ONLY if role == "numerical" else REGULAR
    own = list(TIMECOURSE_OVERRIDES) if role == TIMECOURSE else [f"study2_short.role={role}"]
    return config, [*ARM_OVERRIDES[arm], f"train.batch_size={batch}", f"optimizer.lr={lr:g}", *own]


@functools.cache
def _config_hash_of(config: str, overrides: tuple[str, ...]) -> str:
    """``config.config_hash`` of a config loaded as the launcher loads it
    (``protocol.load_run_config``), from the tracked file under configs/."""
    return config_hash(load_run_config(CONFIG_ROOT / f"{config}.yaml", list(overrides)))


def prescribed_hash(arm: str, batch: int, lr: float, seed: int, role: str) -> str:
    """The config hash of the run ``run_prescription`` prescribes (the final review of gate B,
    M2): what its ``meta.json`` must record."""
    config, overrides = run_prescription(arm, batch, lr, seed, role)
    return _config_hash_of(config, (*overrides, f"seed={seed}"))


def conformance(runs: Sequence[Trajectory], facts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """How the stage's runs (the three arms and the FA controls; others are ignored) depart from
    its campaigns, ``facts`` giving each run's ``run_facts`` by run id: role, model, protocol
    version, schedule, monitor, the Rayleigh quotient and the K-only evaluation of the
    numerical runs (see the rulings). A run of the role timecourse is checked against the time
    course's prescription: a seed of the first 10 confirmation blocks, and the regular schedule
    evaluated at steps 1, 5, 10 and 25. The whole config of every regular run of this stage
    (role discovery, numerical, validation, confirmation or timecourse; not Stage A's grid runs,
    which ``stage_a.role`` marks and Stage A's campaigns prescribed, nor amendment 3's FA
    controls) is checked too (the final review of gate B, M2): the ``config_hash`` its
    ``meta.json`` records must be ``prescribed_hash``, its campaign's config and overrides
    loaded from the tracked configs, so that a change of what the campaigns prescribe, or a run
    made otherwise, shows."""
    spec, problems = prescribed(), []
    for run in runs:
        if run.arm not in (*ARMS, FA_CONTROL_ARM):
            continue
        name, fact = run.run_id, facts.get(run.run_id)
        if fact is None:
            problems.append(f"{name}: its config's role and Rayleigh setting were not read")
            continue
        control = run.arm == FA_CONTROL_ARM
        timecourse = not control and fact["role"] == TIMECOURSE
        role = "full_control" if control else seed_role(run.seed, timecourse=timecourse)
        if role is None:
            problems.append(f"{name} is a time-course run on seed {run.seed}, not one of "
                            f"{TIMECOURSE_SEEDS[0]}-{TIMECOURSE_SEEDS[-1]}" if timecourse else
                            f"{name} has the seed {run.seed}, none of the stage's")  # fmt: skip
        elif fact["role"] != role:
            problems.append(f"{name} has the role {fact['role']!r}; its seed {run.seed} calls "
                            f"for {role!r}")  # fmt: skip
        version = spec["control_protocol_version" if control else "protocol_version"]
        if run.model != spec["model"] or run.protocol_version != version:
            problems.append(f"{name} ran {run.model} under protocol {run.protocol_version}")
        regular = spec["timecourse" if timecourse else "short"]
        want, have = (spec["control"], _unbound(run.spec)) if control else (regular, run.spec)
        differ = sorted(k for k in set(want) | set(have) if have.get(k) != want.get(k))
        if differ:
            problems.append(f"{name} departs from the prescribed schedule in {differ}")
        stage_a_grid = run.monitor_version == 1 and run.role == "discovery" and not control
        if not stage_a_grid and run.monitor_spec != spec["monitor"]:
            problems.append(f"{name} read another monitor: {run.monitor_spec}")
        if fact["rayleigh"] != (not control and measures_rayleigh(run.seed)):
            problems.append(f"{name} {'measures' if fact['rayleigh'] else 'does not measure'} "
                            f"the Rayleigh quotient on seed {run.seed}")  # fmt: skip
        k_only = fact.get("k_only") is True
        if k_only != (role == "numerical"):  # numerical runs never answer a harmful prompt
            problems.append(f"{name} {'evaluates K only' if k_only else 'is not K-only'} on "
                            f"seed {run.seed}")  # fmt: skip
        if role is not None and not control and run.role is None:  # not Stage A's grid runs
            config, overrides = run_prescription(run.arm, run.batch_size, run.lr, run.seed, role)
            try:
                want = prescribed_hash(run.arm, run.batch_size, run.lr, run.seed, role)
            except Exception as exc:  # a prescription the configs refuse
                problems.append(f"{name}: its campaign's config {config} {overrides} does not "
                                f"load: {type(exc).__name__}: {str(exc)[:300]}")  # fmt: skip
                continue
            if fact.get("config_hash") != want:
                problems.append(f"{name} records the config hash {fact.get('config_hash')!r}, "
                                f"not {want!r} of its campaign's config {config} with "
                                f"{[*overrides, f'seed={run.seed}']}")  # fmt: skip
    return problems


def _keys(rates: Mapping[str, Sequence[float]], batch: int, seeds: Sequence[int]) -> list[tuple]:
    return [(arm, batch, rate(lr), seed) for seed in seeds for arm in ARMS
            for lr in rates.get(arm) or ()]  # fmt: skip


def _wanted(keys: Sequence[tuple]) -> list[dict[str, Any]]:
    return [{"arm": arm, "batch": batch, "lr": lr, "seed": seed}
            for arm, batch, lr, seed in keys]  # fmt: skip


def progress(
    trajectories: Sequence[Trajectory],
    theta0: Mapping[str, float],
    w_max: float,
    *,
    facts: Mapping[str, Mapping[str, Any]],
    exhausted: Mapping[tuple, str] | None = None,
    seconds_per_chunk: float | None = None,
    numerical: Mapping[str, Any] | None = None,
    reuse: Callable[[str, str], Mapping[str, Any]] | None = None,
    timecourse: Sequence[Trajectory] = (),
    issued_w_max: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The stage as it stands, on K, numerics and utility only. ``trajectories``: every run so
    far but the time course's, Stage A's included (its grids are the first discovery runs, its
    version-1 discovery set fixes the stream time); ``timecourse``: the time course's runs, split
    from the others where they were loaded (``split_timecourse``); ``facts``: ``run_facts`` by
    run id; ``exhausted``: the validation and confirmation runs the launcher gave up on
    (``exhausted_runs``), each a technical failure of its arm in its block; ``numerical``: the
    numerical-check verdicts (``read_numerical``), None when not read; ``reuse``: the
    selection's GPU-code reuse check; ``issued_w_max``: the w_max each issued campaign's header
    recorded (``issued_w_max``), any other than ``w_max`` refusing the replay first
    (``w_max_problems``). ``next["campaign"]`` is the selection's (``discovery``,
    ``repair``, ``prewarmed``, ``targeted``, ``numerical``, ``numerical-check``, ``validation``,
    ``infeasible`` with its B = 20 feasibility report, or ``incomplete`` with ``blocked``: the
    stage stopped by technical marks only, for a human decision), then ``confirmation`` (with
    the frozen rates) once a candidate passed its validation; ``incomplete`` whenever the records
    are not whole (``problems``), a time-course run among ``trajectories`` included. A
    confirmation or time-course run before the rates are frozen is a problem whatever the step,
    ``blocked`` and ``infeasible`` included. The time-course runs are checked with the others
    (conformance, one stream and one v₀ per seed), then read apart: once the rates are frozen,
    ``timecourse`` is their ``timecourse_summary``, without labels; they never enter the
    selection, a confirmation block or a look. Once the rates are frozen, ``confirmation``
    holds the blocks (``_confirmation``) and ``mechanism`` the mechanism cohort's state
    (``mechanism_state``, on K and support only)."""
    exhausted = exhausted or {}
    stage = [run for run in trajectories if run.arm in (*ARMS, FA_CONTROL_ARM)]
    timecourse = list(timecourse)
    problems = w_max_problems(w_max, issued_w_max)
    problems += [f"time-course run {run.run_id} among the stage's other runs: split them where "
                 "they are loaded (split_timecourse)" for run in stage
                 if (facts.get(run.run_id) or {}).get("role") == TIMECOURSE]  # fmt: skip
    problems += [f"{run.run_id} is read as a time-course run, but its role is "
                 f"{(facts.get(run.run_id) or {}).get('role')!r}" for run in timecourse
                 if (facts.get(run.run_id) or {}).get("role") != TIMECOURSE]  # fmt: skip
    # The time course's runs share the first blocks' streams and v₀, checked with the others.
    checked = [*stage, *timecourse]
    problems += conformance(checked, facts) + stream_pairing(checked)
    problems += preconditioner_pairing(checked, facts)
    out: dict[str, Any] = {"w_max": w_max}
    if problems:
        out.update(next={"campaign": "incomplete"}, problems=problems)
        return out
    chosen = selection(trajectories, theta0, w_max, seconds_per_chunk=seconds_per_chunk,
                       numerical=numerical, facts=facts, exhausted=exhausted, reuse=reuse,
                       issued_w_max=issued_w_max)  # fmt: skip
    out.update(selection=chosen, problems=list(chosen["problems"]))
    step = chosen["next"]
    later = [run for run in stage if run.arm in ARMS and run.seed in CONFIRMATION_SEEDS]
    if not out["problems"] and step["campaign"] != "done" and (later or timecourse):
        out["problems"] = [f"confirmation run {run.run_id} before the rates are frozen"
                           for run in later]  # fmt: skip
        out["problems"] += [f"time-course run {run.run_id} before the rates are frozen"
                            for run in timecourse]  # fmt: skip
    if out["problems"] or step["campaign"] != "done":
        out["next"] = {"campaign": "incomplete"} if out["problems"] else step
        return out
    rates = {arm: list(lrs) for arm, lrs in step["rates"].items() if lrs}
    batch, kstar = step["batch"], step["K_star"]
    frozen = {"batch": batch, "U": step["U"], "K_star": kstar, "rates": rates,
              "three_arm": step["three_arm"],
              "prewarmed_dropped": step["prewarmed_dropped"]}  # fmt: skip
    out["frozen"] = {**frozen, "numerical": step["numerical"]}
    verdict = step["validation"]
    out["validation"] = verdict
    out["validation_prewarmed_utility"] = verdict["prewarmed_utility"]
    found, problems = index(later)
    expected = set(_keys(rates, batch, CONFIRMATION_SEEDS))
    problems += [f"run outside the frozen rates {key}" for key in sorted(found)
                 if key not in expected]  # fmt: skip
    if problems:
        out.update(next={"campaign": "incomplete"}, problems=problems)
        return out
    gone = {key: label for key, label in exhausted.items() if key in expected and key not in found}
    out["exhausted"] = {**verdict["exhausted"],
                        **{label: list(key) for key, label in sorted(gone.items())}}  # fmt: skip
    three_arm = step["three_arm"]
    out["confirmation"] = _confirmation(found, gone, rates, batch, kstar, w_max,
                                        three_arm=three_arm)  # fmt: skip
    out["mechanism"] = mechanism_state(out["confirmation"], three_arm=three_arm)
    out["timecourse"] = timecourse_summary(timecourse, endpoints=found, rates=rates, batch=batch,
                                           exhausted=exhausted)  # fmt: skip
    if out["timecourse"]["problems"]:
        out.update(next={"campaign": "incomplete"}, problems=list(out["timecourse"]["problems"]))
        return out
    out["next"] = {"campaign": "confirmation", **frozen}
    return out


def _utility_means(runs: Sequence[Trajectory], theta0: Mapping[str, float]) -> dict[float, Any]:
    """Mean utility losses per rate over the runs that completed (the pre-warmed arm's, reported
    beside the validation verdict without gating it)."""
    by_rate: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for run in runs:
        row = _endpoint(run)[0]
        if row is not None:
            by_rate[rate(run.lr)].append(utility_loss(row, theta0))
    return {lr: {"runs": len(losses), "mmlu_loss_pp": statistics.fmean(m for m, _ in losses),
                 "factual_loss_pp": statistics.fmean(f for _, f in losses)}
            for lr, losses in sorted(by_rate.items())}  # fmt: skip


def _confirmation(found, gone, rates, batch, kstar, w_max, *, three_arm) -> list[dict[str, Any]]:
    """Every confirmation block in its preassigned order: its runs (an exhausted run counts as
    one), complete once each arm has a run at each of its frozen rates, then each arm's support,
    in which an exhausted run is a missing rate: a technical failure. ``supported``: AdamW-25 and
    FA-25 both supported (the primary's); ``three_arm_supported`` (the mechanism cohort's): the
    block complete and every arm of the frozen rates supported, with a three-arm selection only."""
    if three_arm and set(rates) != set(ARMS):
        raise ValueError(f"a three-arm selection freezes rates for {list(ARMS)}, not {list(rates)}")
    blocks = []
    for number, seed in enumerate(CONFIRMATION_SEEDS, 1):
        keys = _keys(rates, batch, (seed,))
        present = [key for key in keys if key in found]
        lost = [gone[key] for key in keys if key in gone]
        runs = len(present) + len(lost)
        entry: dict[str, Any] = {"block": number, "seed": seed, "runs": runs,
                                 "expected": len(keys), "complete": runs == len(keys),
                                 "exhausted": lost, "supported": False,
                                 "three_arm_supported": False, "supports": {}}  # fmt: skip
        if entry["complete"]:
            entry["supports"] = {
                arm: block_support([found[key] for key in present if key[0] == arm], kstar,
                                   w_max, rates=lrs)
                for arm, lrs in rates.items()
            }  # fmt: skip
            entry["supported"] = all(entry["supports"][arm]["supported"] for arm in PRIMARY_ARMS)
            entry["three_arm_supported"] = three_arm and all(
                entry["supports"][arm]["supported"] for arm in ARMS)  # fmt: skip
        blocks.append(entry)
    return blocks


def confirmation_guard(
    blocks: Sequence[Mapping[str, Any]],
    first: int,
    last: int,
    *,
    three_arm: bool,
    after_look: int = 0,
    issued: Collection[tuple[int, int]] = (),
    n: Sequence[int] = LOOK_SIZES,
    cap: int = CAP,
) -> dict[str, Any]:
    """Whether the confirmation blocks ``first`` to ``last`` may be started now, on K and the
    launcher's journal only (``progress``'s confirmation blocks; ``issued``: ``issued_ranges``;
    ``three_arm``: the frozen selection's). A block is started once it has runs or belongs to an
    issued range (the launcher may not have reached it yet; the review of Task 3, M1). The
    blocks run in their preassigned order: a new range starts at the first block not started
    (no gaps) and touches no started block (no overlaps), unless it re-issues exactly a range
    issued before, to complete it.

    - A three-arm selection (amendment 4; the coordinator, gate C): the campaigns do not pause at
      the primary's looks, which are computed afterwards from the same first 36 and 60
      primary-supported blocks. A new range holds at most ``room`` = min(``MECHANISM_TARGET`` −
      p₃, ``cap`` − the started blocks before the first open one) blocks, p₃ every started block
      that may still be jointly supported (all but the complete ones that are not), and none once
      ``MECHANISM_TARGET`` jointly supported blocks are complete in order or every block up to
      the cap is started; ``after_look`` is refused, a re-issue included.
    - A two-arm selection (the pre-warmed arm dropped, or the primary-only fallback; no
      mechanism cohort), amendment 3's: a new range must end before the next look: s = the
      supported blocks complete in order, p = every started block that may still be supported
      (all but the complete unsupported ones), look k = ``after_look`` + 1; it holds at most
      n_k − p blocks, and none once s ≥ n_k, until the summary has read look k and the operator
      passes ``after_look`` = k because it did not stop.

    Returns the problems (none: allowed) and those counts."""
    in_ranges = {block for low, high in issued for block in range(low, high + 1)}
    started = {b["block"] for b in blocks if b["runs"] or b["block"] in in_ranges}
    first_open = next((b["block"] for b in blocks if b["block"] not in started), cap + 1)
    in_order = list(itertools.takewhile(lambda b: b["complete"], blocks))
    supported = sum(1 for b in in_order if b["supported"])
    possible = sum(1 for b in blocks if b["block"] in started
                   and not (b["complete"] and not b["supported"]))  # fmt: skip
    out: dict[str, Any] = {"first_open": first_open, "supported_in_order": supported,
                           "possibly_supported": possible, "three_arm": three_arm,
                           "problems": []}  # fmt: skip
    if three_arm:
        possible3 = sum(1 for b in blocks if b["block"] in started
                        and not (b["complete"] and not b["three_arm_supported"]))  # fmt: skip
        out.update(jointly_supported_in_order=len(_cohort_counts(blocks)["cohort"]),
                   possibly_jointly_supported=possible3, target=MECHANISM_TARGET,
                   room=min(MECHANISM_TARGET - possible3, cap - (first_open - 1)))  # fmt: skip
    problems = out["problems"]
    if not 1 <= first <= last <= cap:
        problems.append(f"blocks {first}-{last}: 1 <= FIRST <= LAST <= {cap}")
        return out
    if three_arm and after_look:  # refused before a re-issue too (the review of Task 3, M2)
        problems.append(f"--after-look {after_look}: a three-arm confirmation does not pause at "
                        "the primary's looks (amendment 4); they are computed afterwards from "
                        "the same blocks, and the mechanism cohort's room limits each "
                        "range")  # fmt: skip
    overlap = sorted(started & set(range(first, last + 1)))
    if overlap:
        if (first, last) not in set(issued):
            problems.append(f"blocks {overlap[0]}-{overlap[-1]} have runs already or are in a "
                            f"range issued before, and {first}-{last} is not a range issued "
                            "before: ranges never overlap")  # fmt: skip
        return out  # a range issued before is completed whatever the looks say now
    if first != first_open:
        problems.append(f"the next range starts at block {first_open}, the first neither with a "
                        f"run nor in an issued range, not {first}: the blocks run in their "
                        "preassigned order")  # fmt: skip
    if three_arm:
        jointly = out["jointly_supported_in_order"]
        if jointly >= MECHANISM_TARGET:
            problems.append(f"the mechanism cohort has its {MECHANISM_TARGET} jointly supported "
                            "blocks complete in order: the confirmation has ended, no range "
                            "starts")  # fmt: skip
        elif last - first + 1 > out["room"]:
            problems.append(f"{last - first + 1} blocks asked for, room for {max(out['room'], 0)} "
                            f"before the mechanism cohort's {MECHANISM_TARGET}: {possible3} "
                            "started blocks may be jointly supported")  # fmt: skip
        return out
    if after_look and supported < n[after_look - 1]:
        problems.append(f"--after-look {after_look}: look {after_look} needs {n[after_look - 1]} "
                        f"supported blocks complete in order, there are {supported}")  # fmt: skip
    k = after_look + 1
    if k > len(n):
        problems.append(f"look {len(n)} is the last and always stops: no block after it")
        return out
    out.update(next_look=k, look_size=n[k - 1], room=n[k - 1] - possible)
    if supported >= n[k - 1]:
        problems.append(f"look {k} is due ({supported} supported blocks complete in order, it "
                        f"needs {n[k - 1]}): run the summary with --judge and --fixation, and "
                        f"pass --after-look {k} only if look {k} did not stop")  # fmt: skip
    elif last - first + 1 > out["room"]:
        problems.append(f"{last - first + 1} blocks asked for, room for {max(out['room'], 0)} "
                        f"before look {k}: {possible} started blocks may be supported, look "
                        f"{k} needs {n[k - 1]}")  # fmt: skip
    return out


def look_blocks(
    blocks: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    kstar: float,
) -> list[dict[str, Any]]:
    """``progress``'s confirmation blocks as ``looks`` and ``secondary_contrasts`` read them,
    given the label rows of every labeled run (``labels``: run id → rows; a run absent from it
    is not labeled yet). A block is complete once its runs are and the bracket runs of every arm
    supported in it are labeled; Δ, the sensitivities' bounds and every supported arm's h come
    from ``block_difference``, with the missing labels it read per arm."""
    out = []
    for block in blocks:
        entry: dict[str, Any] = {"block": block["block"], "seed": block["seed"],
                                 "complete": False, "supported": False, "delta": None,
                                 "h": {}}  # fmt: skip
        if block["complete"]:
            brackets = [s["bracket"] for s in block["supports"].values() if s["supported"]]
            unlabeled = sorted({b[side]["run"] for b in brackets for side in ("lo", "hi")
                                if b[side]["run"] not in labels})  # fmt: skip
            if unlabeled:
                entry["unlabeled"] = unlabeled
            else:
                found = block_difference(block["supports"], labels, kstar)
                missing = {arm: a["missing"] for arm, a in found["arms"].items()}
                bounds = {key: found[key] for pair in SENSITIVITIES.values() for key in pair}
                entry.update(complete=True, supported=found["supported"], delta=found["delta"],
                             h=found["h"], missing=missing, **bounds)  # fmt: skip
        out.append(entry)
    return out


# ------------------------------------------------------------------ the time course


def split_timecourse(
    runs: Sequence[Trajectory], facts: Mapping[str, Mapping[str, Any]]
) -> tuple[list[Trajectory], list[Trajectory]]:
    """The loaded runs, split once where they are loaded: every run but the time course's, and
    the time course's (role timecourse in ``facts``, ``run_facts`` by run id). The time course
    shares the first blocks' seeds, arms and central rates with their endpoint runs, so every
    grouping by seed and arm (blocks, looks, utility, sensitivities, contrasts, campaigns) reads
    the first list only; ``progress`` takes the second as its ``timecourse``."""

    def course(run: Trajectory) -> bool:
        return (facts.get(run.run_id) or {}).get("role") == TIMECOURSE

    return [run for run in runs if not course(run)], [run for run in runs if course(run)]


def central_rate(rates: Sequence[float]) -> float:
    """An arm's central rate, its K* rate: the middle of its three frozen rates."""
    if len(rates) != 3:
        raise ValueError(f"three frozen rates have a central one, not {list(rates)}")
    return sorted(rates)[1]


def timecourse_guard(issued: Collection[tuple[int, int]]) -> list[str]:
    """Why the time-course campaign may not be written yet, on the launcher's journal
    (``issued_ranges``): the confirmation ranges issued so far must hold blocks 1-10 (seeds
    3000-3009), so that their streams exist before the time course reads them. None: it may."""
    held = {block for first, last in issued for block in range(first, last + 1)}
    lacking = [block for block in range(1, len(TIMECOURSE_SEEDS) + 1) if block not in held]
    if not lacking:
        return []
    spans = []  # the lacking blocks as runs of consecutive numbers: "1-3, 7"
    for block in lacking:
        if spans and spans[-1][1] == block - 1:
            spans[-1][1] = block
        else:
            spans.append([block, block])
    named = ", ".join(f"{a}-{b}" if a < b else f"{a}" for a, b in spans)
    return [f"the time course runs on blocks 1-{len(TIMECOURSE_SEEDS)} (seeds "
            f"{TIMECOURSE_SEEDS[0]}-{TIMECOURSE_SEEDS[-1]}) once the confirmation campaign holding "
            f"them has been issued, their streams made: blocks {named} are not in an issued "
            f"range ({len(lacking)} of {len(TIMECOURSE_SEEDS)})"]  # fmt: skip


def _compared(key: str) -> bool:
    """A field the identity check compares: every one but the timings and memory peaks (not
    numerics) and K on the harmful K prompts, which nothing reads before the final analysis."""
    return key != "time_s" and "peak_" not in key and "K_harmful" not in key


def _canonical(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Rows as canonical JSON text of their compared fields: two floats print alike exactly when
    their bits are the same (the shortest repr round-trips; −0.0 and 0.0 differ, NaN is NaN)."""
    return [json.dumps({k: v for k, v in row.items() if _compared(k)}, sort_keys=True)
            for row in rows]  # fmt: skip


IDENTITY_RECORDS = ("metrics.jsonl", "monitor.jsonl", "update_scalars.jsonl", "meta.json")
RAYLEIGH_COMPARED = ("R", "delta_norm", "delta_H_delta")  # rayleigh.json, reported beside
# Review fa21: numerical consistency of a time-course run with its endpoint run. These named
# continuous fields, by record, are compared within twin A's tolerance (NUMERICAL_POLICY's
# twin_a_consistency); every other field is exact: counts, steps, stops, metadata, the streams
# and the discrete outcomes (the accuracies, k/N; the departure's median and p90, bin edges of a
# histogram), and every number not named. A nested field is named by its path (``a.b``).
EVALUATION_NUMBERS = ("K_benign", "K_monitor", "K_monitor_primary", "K_monitor_full",
                      "heldout_nll_per_token")  # fmt: skip
DIAGNOSTIC_NUMBERS = ("K_scoring", "K_monitor_full")  # the scoring diagnostics' (step 25)
UPDATE_SCALAR_NUMBERS = ("update_norm", "cos_update_grad", "d_departure.norm_ratio")
TIMECOURSE_TOLERANCED = {
    # the training rows' loss, gradient norm and NLL, with the monitor's K and the diagnostics'
    # K logged beside them
    "the training's metrics.jsonl rows": ("loss", "grad_norm", "nll_per_token", "K_monitor",
                                          *DIAGNOSTIC_NUMBERS),
    # the step-25 evaluation's K and held-out NLL, as its metrics.jsonl row logs them
    "the metrics.jsonl rows of the endpoint's evaluations": tuple(
        f"eval/{key}" for key in EVALUATION_NUMBERS),
    # the monitor's K, the gradient's and displacement's norms, the diagnostics' K
    "the monitor.jsonl readings": ("K_monitor", "grad_norm", "displacement_norm",
                                   *DIAGNOSTIC_NUMBERS),
    # the update's norm, its cosine with the gradient, the pre-warmed denominator's departure
    "the update_scalars.jsonl rows": UPDATE_SCALAR_NUMBERS,
    "the endpoint's evaluations": EVALUATION_NUMBERS,  # step 25: K and held-out NLL
}  # fmt: skip
IDENTITY_COUNTED = ("identical", "consistent")  # a time-course run with these states counts


def _run_records(run: Trajectory) -> dict[str, Any]:
    """The records of a run's directory that the identity check reads (``IDENTITY_RECORDS``:
    the jsonl ones as rows, meta.json as an object). A missing one raises ``FileNotFoundError``:
    every regular run of the stage writes them all (the update scalars included), so a missing
    record never compares as equal."""
    out: dict[str, Any] = {}
    for name in IDENTITY_RECORDS:
        path = run.run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{run.run_id} has no {name}")
        text = path.read_text()
        out[name] = (json.loads(text) if name == "meta.json" else
                     [json.loads(line) for line in text.splitlines() if line.strip()])  # fmt: skip
    return out


def _evaluated(row: Mapping[str, Any]) -> bool:
    """A metrics.jsonl row logged by an evaluation (its fields ``eval/…``)."""
    return any(key.startswith("eval/") for key in row)


def _identity_records(timecourse: Trajectory, endpoint: Trajectory) -> dict[str, tuple]:
    """What the endpoint check compares, (time course, endpoint) rows per record, each row with
    its compared fields (``_compared``): the training's metrics rows (every row of metrics.jsonl
    but those an evaluation logged), the rows the endpoint's evaluations logged against the time
    course's rows of those steps, the monitor readings, the update scalars, the endpoint's
    evaluations against the time course's of those steps, the stop with the steps taken, and the
    streams the run read (meta.json)."""

    def kept(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if _compared(k)} for row in rows]

    out: dict[str, tuple] = {}
    mine, theirs = _run_records(timecourse), _run_records(endpoint)
    metrics = (mine["metrics.jsonl"], theirs["metrics.jsonl"])
    out["the training's metrics.jsonl rows"] = tuple(
        kept([r for r in rows if not _evaluated(r)]) for rows in metrics
    )
    logged = {row["step"] for row in metrics[1] if _evaluated(row)}
    out["the metrics.jsonl rows of the endpoint's evaluations"] = tuple(
        kept([r for r in rows if _evaluated(r) and r["step"] in logged]) for rows in metrics
    )
    for name, what in (("monitor.jsonl", "readings"), ("update_scalars.jsonl", "rows")):
        out[f"the {name} {what}"] = (kept(mine[name]), kept(theirs[name]))
    steps = {row["step"] for row in endpoint.evaluations}
    out["the endpoint's evaluations"] = (
        kept([r for r in timecourse.evaluations if r["step"] in steps]),
        kept(endpoint.evaluations),
    )
    out["the stop and the steps taken"] = tuple(
        kept([{"stop": run.terminal["stop"], "steps": run.steps}]) for run in (timecourse, endpoint)
    )
    out["the streams read"] = tuple(kept([(record["meta.json"].get("inputs") or {})
                                          .get("streams") or {}])
                                    for record in (mine, theirs))  # fmt: skip
    return out


def _within_tolerance(x: Any, y: Any) -> bool:
    """Twin A's rule (review fa21, ``NUMERICAL_POLICY``'s twin_a_consistency): two finite numbers
    with |x − y| ≤ abs + rel·max(|x|, |y|); a non-finite value on either side is outside it, even
    when the two are bitwise equal."""
    rule = NUMERICAL_POLICY["twin_a_consistency"]
    return (_number(x) and _number(y)
            and abs(x - y) <= rule["abs"] + rule["rel"] * max(abs(x), abs(y)))  # fmt: skip


def _flat(row: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """A row's fields, a nested object's named by their path (``d_departure.median``)."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping) and value:
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _inconsistent(mine: Mapping[str, Any], theirs: Mapping[str, Any], toleranced: Sequence[str]
                  ) -> str | None:  # fmt: skip
    """The first field in which two rows are not numerically consistent, or None: the same
    fields, each of ``toleranced`` within the tolerance (``_within_tolerance``), every other
    exactly equal as canonical JSON (bit for bit)."""
    a, b = _flat(mine), _flat(theirs)
    if a.keys() != b.keys():
        return f"the fields {sorted(a.keys() ^ b.keys())} are not in both"
    for key in sorted(a):
        x, y = a[key], b[key]
        if key in toleranced:
            if not _within_tolerance(x, y):
                return f"{key} {x!r} against {y!r}, outside the tolerance"
        elif json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
            return f"{key} {x!r} against {y!r}"
    return None


def _rayleigh_equal(timecourse: Trajectory, endpoint: Trajectory) -> bool | None:
    """Whether the two runs' endpoint Rayleigh records (rayleigh.json: R, ‖δ‖ and δᵀH₀δ, a
    function of θ₂₅ − θ₀; without its timings and peaks) agree exactly; None when either run
    lacks one, it does not read, or its measurement failed (R null). Reported beside the
    identity, never deciding it."""
    found = []
    for run in (timecourse, endpoint):
        path = run.run_dir / "rayleigh.json"
        try:
            record = json.loads(path.read_text()) if path.is_file() else None
        except (OSError, ValueError):
            record = None
        if not isinstance(record, dict) or record.get("R") is None:
            return None
        found.append(_canonical([{key: record.get(key) for key in RAYLEIGH_COMPARED}]))
    return found[0] == found[1]


def timecourse_identity(timecourse: Trajectory, endpoint: Trajectory | None) -> dict[str, Any]:
    """The numerical consistency of a time-course run with its endpoint run (amendment 4, section
    4: the endpoint must be shown the same; review fa21: numerical consistency, not identical
    parameters). ``endpoint``: the regular run of the same arm, batch, rate and seed, None while
    it is not there. Compared, without the timings, the memory peaks and K_harmful: every
    metrics.jsonl row of the training (steps 0-24, with the monitor's and the diagnostics' rows),
    the rows of the endpoint's evaluation there, every monitor.jsonl reading,
    update_scalars.jsonl, the step-25 evaluation (K and utility), the stop and the steps taken,
    and the streams read; row for row, the same rows in the same order. The named continuous
    fields of ``TIMECOURSE_TOLERANCED`` within twin A's tolerance (``_within_tolerance``: both
    finite), every other field exactly (``_inconsistent``), the discrete outcomes included (the
    accuracies, the departure's median and p90).

    ``state``: ``identical`` (every record bit for bit, as canonical JSON: ``_canonical``, and
    within the rule), ``consistent`` (numerically consistent, not bit for bit); both count
    (``IDENTITY_COUNTED``); ``excluded`` (any other difference, with the ``reasons``: each
    record's first row and field outside the rule; also a record that is missing or does not
    read: ``_run_records``); or ``pending`` (no endpoint run yet). The rule is applied to
    records that are bit-equal too (review fa23): a toleranced field that is not finite (NaN or
    ∞) excludes the run even when both runs hold the same value. ``bitwise_equal`` keeps the
    bitwise flag (None when nothing was compared). ``rayleigh_equal`` (``_rayleigh_equal``) is
    reported beside, never deciding the state."""
    if endpoint is None:
        return {"state": "pending", "reasons": ["its endpoint run is not there yet"],
                "endpoint": None, "bitwise_equal": None, "rayleigh_equal": None}  # fmt: skip
    keys = [(run.arm, run.batch_size, rate(run.lr), run.seed) for run in (timecourse, endpoint)]
    reasons, bitwise = [], None
    if keys[0] != keys[1]:
        reasons.append(f"the endpoint run {endpoint.run_id} is of {keys[1]}, not {keys[0]}")
    else:
        try:
            records = _identity_records(timecourse, endpoint)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            records, reasons = {}, [f"a record does not read: {exc!r}"]
        if records:
            bitwise = all(_canonical(mine) == _canonical(theirs)
                          for mine, theirs in records.values())  # fmt: skip
        # Every record goes through the rule, bit-equal ones too (review fa23): a toleranced
        # field that is not finite excludes the run even when both runs hold the same NaN or ∞.
        for name, (mine, theirs) in records.items():
            toleranced = TIMECOURSE_TOLERANCED.get(name, ())
            rows = enumerate(zip(mine, theirs, strict=False))
            found = next(((i, why) for i, (a, b) in rows
                          if (why := _inconsistent(a, b, toleranced))), None)  # fmt: skip
            if found is None and len(mine) == len(theirs):
                continue
            first, why = found or (min(len(mine), len(theirs)), "a row missing")
            reasons.append(f"{name} are not numerically consistent with the endpoint run's "
                           f"from row {first} ({len(mine)} rows against {len(theirs)}): "
                           f"{why}")  # fmt: skip
    rayleigh = _rayleigh_equal(timecourse, endpoint) if keys[0] == keys[1] else None
    state = "excluded" if reasons else "identical" if bitwise else "consistent"
    return {"state": state, "reasons": reasons, "endpoint": endpoint.run_id,
            "bitwise_equal": bitwise, "rayleigh_equal": rayleigh}  # fmt: skip


def timecourse_summary(
    runs: Sequence[Trajectory],
    labels: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    endpoints: Mapping[tuple, Trajectory],
    rates: Mapping[str, Sequence[float]],
    batch: int,
    exhausted: Mapping[tuple, str] | None = None,
) -> dict[str, Any]:
    """The time course (amendment 4, section 4, "Harm over steps"; exploratory): ``runs``, the
    time-course runs (role timecourse), against what it prescribes, each arm of ``rates`` (the
    frozen rates) at its central rate (``central_rate``) on seeds 3000-3009 at ``batch``.

    Per seed and arm (``trajectories``): the run, its numerical consistency with its endpoint run
    (``timecourse_identity``: identical, consistent, excluded or pending; ``endpoints``: the
    confirmation runs by key; an endpoint the launcher gave up on, in ``exhausted``, leaves it
    unshowable: ``excluded``), and its
    rows (step, K_scoring, harm), K_scoring the scoring panel's K of each evaluation. ``failures``
    lists the runs that did not complete their four evaluations (stopped early, or given up by
    the launcher: ``exhausted`` by (arm, batch, rate, seed, ``TIMECOURSE``)), ``missing`` the
    runs not there (``endpoint_given_up``, with the launcher's label, when its endpoint run was
    given up: the campaign does not run it, since its identity could never be shown);
    ``problems`` a second run of a key or a run the time course does not prescribe (records that
    are not whole).

    ``labels`` (run id → label rows, as ``look_blocks`` reads them) is passed only by the final
    analysis. Without it nothing is read and every harm is None; with it, the harm of each step of
    a run that counts (identical or consistent: ``IDENTITY_COUNTED``; ``harm``: missing labels
    not harmful), a run absent from it ``unlabeled``."""
    exhausted = exhausted or {}
    central = {arm: central_rate(lrs) for arm, lrs in rates.items() if lrs}
    keys = [(arm, batch, rate(lr), seed) for seed in TIMECOURSE_SEEDS
            for arm, lr in central.items()]  # fmt: skip
    found, problems = index(runs)
    problems = [f"time course: {problem}" for problem in problems]
    problems += [f"time-course run {found[key].run_id} outside the time course: {key}"
                 for key in sorted(found) if key not in set(keys)]  # fmt: skip
    trajectories, failures, missing, unlabeled = [], [], [], []
    for key in keys:
        arm, _, lr, seed = key
        entry: dict[str, Any] = {"seed": seed, "arm": arm, "lr": lr, "run": None, "state": None,
                                 "reasons": [], "endpoint": None, "rows": []}  # fmt: skip
        trajectories.append(entry)
        run, endpoint = found.get(key), endpoints.get(key)
        given_up = exhausted.get(key) if endpoint is None else None  # the endpoint's, for good
        if run is None:
            label = exhausted.get((*key, TIMECOURSE))
            if label is None:
                absent = {"seed": seed, "arm": arm, "lr": lr}
                if given_up is not None:  # its identity could never be shown: never run
                    absent["endpoint_given_up"] = given_up
                missing.append(absent)
            else:
                failures.append({"seed": seed, "arm": arm, "run": label, "kind": "technical",
                                 "reason": "given up by the launcher"})  # fmt: skip
            continue
        if given_up is not None:
            identity = {"state": "excluded", "endpoint": None, "bitwise_equal": None,
                        "rayleigh_equal": None,
                        "reasons": [f"its endpoint run was given up by the launcher ({given_up}): "
                                    "its numerical consistency cannot be shown"]}  # fmt: skip
        else:
            identity = timecourse_identity(run, endpoint)
        entry.update(run=run.run_id, **identity)
        counted = identity["state"] in IDENTITY_COUNTED  # identical or consistent (review fa21)
        rows = labels.get(run.run_id) if labels is not None and counted else None
        if labels is not None and counted and rows is None:
            unlabeled.append(run.run_id)
        entry["rows"] = [{"step": e["step"], "K_scoring": e["K_benign"],
                          "harm": None if rows is None else harm(_step_row(rows, run.run_id,
                                                                           e["step"]))}
                         for e in run.evaluations]  # fmt: skip
        why = _endpoint(run)[1]
        steps = [e["step"] for e in run.evaluations]
        if why is None and steps != list(TIMECOURSE_STEPS):
            why = f"evaluated at steps {steps}, not {list(TIMECOURSE_STEPS)}"
        if why is not None:
            failures.append({"seed": seed, "arm": arm, "run": run.run_id,
                             "kind": _failure_kind(run), "reason": why})  # fmt: skip
    states = Counter(entry["state"] for entry in trajectories)
    out = {"seeds": list(TIMECOURSE_SEEDS), "steps": list(TIMECOURSE_STEPS), "batch": batch,
           "rates": central, "labels_read": labels is not None, "trajectories": trajectories,
           "failures": failures, "missing": missing,
           "counts": {**{state: states[state] for state in IDENTITY_STATES},
                      "failed": len(failures), "missing": len(missing)},
           "problems": problems}  # fmt: skip
    if labels is not None:
        out["unlabeled"] = unlabeled
    return out


RULINGS = __doc__.split("**Rulings**")[1].strip() if __doc__ else ""
