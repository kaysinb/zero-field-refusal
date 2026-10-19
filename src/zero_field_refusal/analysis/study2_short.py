"""Study 2's redirected short-regime stage, design A (amendment 3; its batch, U and rates selected
under amendment 4): selection, support, the estimator and the looks, as pure functions over
``stage_a.Trajectory`` objects, numerical-check verdicts and label rows.

AdamW-25 and FA-25 (Adam with its second moment frozen at θ₀) each run exactly 25 steps at three
rates targeted at the levels (5, K*, U), K* = √(5U); harm at K* is interpolated linearly in ln K
between the nearest bracketing endpoints of each seed block, and the paired block differences
enter two sequential looks. Selection, support and validation read K_benign, numerical records
and utility only, never a judge label or K_harmful; only the estimator, the looks and the
secondary contrasts read labels. The values the companion simulation fixes (w_max, the look
levels, the SE floor, the bias allowance) are inputs, never constants here. Harm rates and
differences are fractions (1 pp = 0.01); utility losses are in percentage points.

- ``usable_rate``: an arm, batch and rate over the discovery seeds: every seed finished its 25
  steps with a finite K25 (K_benign at step 25), mean utility loss in bounds against θ₀.
- ``targeted_rates``: log–log interpolation of the rate against the median K25 between adjacent
  usable rates, for each target; no extrapolation.
- ``select_rates``: the rate-ordered triple of usable rates nearest the targets in ln K under the
  support, realized-width and coverage constraints, or the constraints no triple meets with the
  best unconstrained triple; ``repair_rates``: the one repair round from that triple.
- ``candidate_order``: amendment 4's candidates (U, B) = (8, 20), (7, 20), (8, 5), (7, 5), in
  that fixed order, with their costs for the report.
- ``numerical_cases``, ``numerical_policy``, ``numerical_verdict`` and ``read_numerical``: the
  numerical qualification's cases on seeds 2200-2202, its frozen policy, and the verdict records
  of the numerical-check tool (scripts/numerical_check.py) under $RUNS_DIR/numerical/.
- ``selection``: the state machine over every run so far and the verdicts: the next campaign
  (``discovery``, ``repair``, ``prewarmed``, ``targeted``, ``numerical``, the step
  ``numerical-check``, ``validation``, ``done`` with the frozen rates, ``infeasible`` when no
  candidate is eligible even for the primary arms alone, ``incomplete`` when the records are
  ambiguous), every attempt recorded.
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
  fixation record's secondary interval.
- ``utility_family`` and ``validation_verdict``: the simultaneous utility bounds and the verdict
  of the validation blocks, over the three arms (amendment 4) or the primary arms (amendment 3,
  the fallback).
- ``progress``: the stage as a whole for scripts/study2_short_campaigns.py and
  scripts/study2_short_summary.py: the records checked against the campaigns (``conformance``,
  one training stream per seed, one v₀ per seed and batch: ``preconditioner_pairing``), the
  selection's state machine with its numerical and validation gates, then the confirmation
  blocks with their support (K only), the runs the launcher gave up on
  (``exhausted_runs``, with the attempts abandoned by the operator: ``read_abandonment``) counted
  as technical failures; ``confirmation_guard`` decides whether a
  range of blocks may start; ``look_blocks`` turns the blocks into those of ``looks`` once their
  labels exist. ``read_fixation`` reads the values the simulation fixes.
- ``split_timecourse``, ``timecourse_identity``, ``timecourse_summary`` and
  ``timecourse_guard``: the time course (amendment 4, section 4, "Harm over steps";
  exploratory), separate runs of the first 10 confirmation blocks at each arm's central rate
  evaluated at steps 1, 5, 10 and 25: split from the other runs where they are loaded, each
  against its endpoint run bit for bit, the trajectories with the failed and missing runs per
  seed and arm, harm only when the final analysis passes the labels; and when its campaign may
  be written.

Blocks given to ``looks`` and ``secondary_contrasts`` are dicts in their preassigned order:
``{"block": id, "complete": bool, "supported": bool, "delta": float | None, "h": {arm: float}}``,
``supported`` meaning AdamW-25 and FA-25 both supported, ``delta`` their difference at K*, and
``h`` the interpolated harm of every supported arm (the secondary contrasts read it);
``sensitivity_looks`` also reads the fields its sensitivity names (``block_difference``).

**Rulings** where the amendment leaves a detail open, fixed here before any data:
- *An endpoint* is a run that stopped as ``steps`` after its 25 steps with an evaluation at step
  25 whose K_benign and accuracies are finite; K25 is that K_benign. Any other run is a failure:
  an instability failure when its stop is one of Stage A's instability rules, otherwise technical
  (a missing run, another stop, no finite K25).
- *Discovery runs* of an arm at a batch are those on seeds 2000–2002 at Stage A's grid
  (``stage_a.SHORT_GRID``, the primary arms) and at the rates its rounds prescribed; any other
  run there is reported as unexpected and never selected. Numerical and validation runs are
  those a visit called for. A run at a batch the replay has not reached yet (amendment 3's
  targeted round at B = 5 while B = 20 is decided) is listed as ``not_reached``, not as a
  problem: it enters no decision until a visit reaches its batch, where it is judged like any
  other (amendment 4's replay at (8, 5) prescribes that round again from the same grid).
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
  earlier visit of the candidate is reused, never spent again.
- *Joint discovery* (amendment 4): when the primary arms' targeted round is due at a candidate
  and the pre-warmed arm has no feasible triple there, the same ``discovery`` campaign holds the
  pre-warmed arm's first round at (B, U), at the union of the rates of the primary arms'
  targeted assignments (each target's rate, a grid rate included; a primary arm that needs no
  round contributes its selected triple). Without that campaign, the pre-warmed arm's first
  round, when it has no feasible triple, is at the union of the selected AdamW and FA rates
  (``prewarmed``). Only rates not yet run at the batch are run; one first round per (B, U).
- *Candidate order* (amendment 4): (U, B) = (8, 20), (7, 20), (8, 5), (7, 5), fixed. The costs
  below are kept for the report and decide nothing.
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
  20, 25, and the Gaussian directions' seeds and the reference's modes as the tool's module fixes
  them), its GPU code (``policy.gpu_code``, review fa13; null is stale) is that of every commit its
  numerical runs recorded (each the verdict's ``gpu_code.commit`` or ``code_identity.reuse_check``
  from it to that commit reusable, so that a replacement run made at an analysis-only descendant
  counts; a run on a dirty checkout or recording no commit leaves the selection ``incomplete``, a
  commit whose GPU code cannot be checked makes the verdict stale) and that of the running checkout
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
  spent. The arms holding unusable rates are re-solved once, from the same runs with no new round;
  the re-solve's cases are called for (a replacement rate on 2200, and on 2201 and 2202 if it is its
  arm's new U-rate; a replacement FA rate with its control) and checked in a new check; an unusable
  rate there, or no feasible triple after the re-solve, fails the candidate. The verdict's own
  ``qualified`` must agree with its states, and an unreadable or malformed verdict leaves the
  selection ``incomplete``. A numerical run the launcher gave up on (``exhausted_runs``) lets the
  check go on (review of Task 2, minor 3): the check request names its case in ``given_up``
  (as the tool names a case, ``numerical_case_name``; the final review of gate B, I1), the tool's
  plan notes it instead of stopping, its parts record the case incomplete and, after the rerun,
  not qualified (technical), and its rate leaves the batch.
- *A stale check with runs beyond it*: when the replay waits at a stale check while runs it
  called for after that check exist, the problems name the stale check first and list those
  runs, all of them under ``beyond_stale``; the replay cannot tell such a run from a stray one
  until the check is decided again, after which a stray run shows as unexpected again.
  scripts/study2_short_campaigns.py writes the ``numerical-check`` input when every problem is
  under ``beyond_stale``, and nothing else. A check that went stale under the same numerical
  runs (a policy departure, say) is run again. One stale by a GPU-path change cannot be: the
  tool refuses numerical runs whose GPU code is not its own. The voiding procedure (the final
  review of gate B, I2), ``scripts/numerical_check.py void --batch B --upper U`` on a clean
  checkout of the new commit, moves those numerical runs and the check's directory
  numerical/b{B}-u{U}/ to $RUNS_DIR/void/<old GPU code>/ with an inventory; the replay then
  calls for the numerical runs again (``numerical``) and their new check. Runs the replay
  called for after the voided check (validation runs) are then unexpected and leave the stage
  ``incomplete`` until the coordinator rules on them: under the versions record's one-commit
  rule, a GPU-path change after them voids them too.
- *w_max* (the final review of gate B, I4): frozen since S1. Every campaign's header records
  the w_max it was written under, and the launcher logs that header with each start
  (``issued_w_max``); the replay under another w_max is refused (``incomplete``, with that
  reason), before anything is selected. Starts logged without a header record none.
- *The whole config* (the final review of gate B, M2): every regular run of this stage records
  the config hash of its campaign's config and overrides (``prescribed_hash``), checked by
  ``conformance``; the code that composes them (``run_prescription``) is outside the GPU-code
  identity, and this check is what shows a change of it after runs were made.
- *Fallback* (amendment 4, review fa12; review of Task 2, critical 1 and minor 2): when no
  candidate is eligible with the three arms, the candidates are visited again in the same order
  for the primary arms alone, each selected over every record at its batch (rounds made at the
  other U included), except those whose primary arms failed at the numerical gate (a second
  chance there would break the one-re-solve rule). A candidate whose primary arms failed at
  discovery in pass 1 is visited again: with no new round beyond each arm's once-only targeted
  and repair rounds per batch and U, it can only become feasible from rates run since; a
  three-arm validation failure does not exclude a candidate either, since the fallback's
  validation is amendment 3's primary-arm one. A primary arm's re-solve in pass 1 at the
  candidate (a primary rate in an ``unusable`` entry of its checks) is spent: the fallback's
  visit starts re-solved, so an unusable rate in its check fails the candidate, and a primary
  arm left without a triple by the unusable rates gets no new round and fails it too (the
  conservative reading: a round would be a second chance). A re-solve caused by pre-warmed rates
  only is not carried. Every record is reused (rounds, runs, the checks whose rates hold the
  request); in a three-arm check read again, only the primary arms' rates and the FA controls
  count, whatever its pre-warmed entries say (review fa13), and nothing of the pre-warmed arm is
  called for. When no candidate is eligible even then, the stage ends ``infeasible`` with every
  attempt's reasons.
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
- *Secondary contrasts* (coordinator, 2026-10-17): two-sided intervals at 98.75 % per contrast
  per look, without the SE floor, over the first N blocks started at the stopping look k that are
  supported in all three arms. The interval is the fixation record's ``secondary_interval``: the
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
  rate and seed is the same run.
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
  passes, the next candidate otherwise. A confirmation block is complete once every run of
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
- *Confirmation blocks* (``confirmation_guard``, the review of Task 5): ranges start at the first
  block without a run and never overlap one another (re-issuing an issued range completes it);
  the looks fall at 36 and 60 supported blocks (coordinator, 2026-10-16; cap 80). A new range
  holds at most n_k − p blocks, p counting every block with runs except the complete unsupported
  ones, so that no block is started past a look that could stop; once look k is due, no block
  starts until the operator, having read it in the summary, passes ``after_look`` = k.
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
  key the time course does not prescribe leaves the stage ``incomplete``. Endpoint identity
  (``timecourse_identity``) compares exactly, field by field as canonical JSON, the training's
  metrics.jsonl rows (every row but those an evaluation logged), the rows of the endpoint's
  evaluation, monitor.jsonl, update_scalars.jsonl, the endpoint's evaluations against the time
  course's of the same steps (step 25: K and utility), the stop with the steps taken and the
  streams read; the timings and memory peaks are left out (not numerics), and so is K_harmful
  (outcome-blind until the final analysis). Any difference, or a record that is missing or does
  not read (metrics, monitor, update scalars, meta.json: every regular run writes them),
  excludes the run with its reasons; the endpoint Rayleigh records (R, ‖δ‖, δᵀH₀δ) are compared
  beside, as ``rayleigh_equal``, deciding nothing; no endpoint run yet leaves it pending; an
  endpoint run the
  launcher gave up on leaves it excluded (the conservative reading: its identity can never be
  shown). A time-course run that did not complete its four evaluations, or that the launcher
  gave up on (``exhausted_runs`` keys it with ``TIMECOURSE``, apart from its endpoint run), is a
  failure of its seed and arm, a run not there yet is missing. ``timecourse_summary`` reads
  labels only when the final analysis passes them (``labels``, None by default and in
  ``progress``), and then only for runs whose identity is ``identical``: an excluded or pending
  run keeps its K and no harm.
- *The fixation record* (``read_fixation``): a JSON object with ``w_max``, ``levels`` (the
  looks' two-sided confidence levels), ``se_floor`` and ``bias_allowance``, the last two in the
  units of Δ (fractions, 1 pp = 0.01), ``interval`` (``bootstrap_t`` or ``t``),
  ``bootstrap_resamples`` (an integer ≥ 1000) and ``bootstrap_seed`` (an integer), ``looks`` and
  ``cap`` (equal to ``LOOK_SIZES`` and ``CAP``, which the looks and the confirmation guard use: a
  record made for other looks is refused), ``equivalence_min_supported`` and
  ``direction_min_supported`` (integers ≥ 0), ``secondary_interval`` (``t`` or ``bootstrap_t``)
  and ``secondary_status`` (``secondary`` or ``exploratory``); each is required and none is
  defaulted, also when ``interval`` is ``t`` and the bootstrap's values go unused.
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
CANDIDATES = ((8.0, 20), (7.0, 20), (8.0, 5), (7.0, 5))  # (U, B): amendment 4's fixed order
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
IDENTITY_STATES = ("identical", "excluded", "pending")  # a time-course run against its endpoint
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
# The frozen numerical policy a verdict must carry (its bound is 0.01 × K* of the U checked).
NUMERICAL_POLICY = {"norm": 5e-05, "survival_share": 0.5, "retry_scale": 10,
                    "steps": [1, 5, 10, 15, 20, 25]}  # fmt: skip
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
        fraction = (math.log(target) - math.log(k0)) / (math.log(k1) - math.log(k0))
        lr = math.exp(math.log(r0) + fraction * (math.log(r1) - math.log(r0)))
        found.append({"target": target, "rate": _three(lr), "between": [r0, r1]})
    return {"U": upper, "K_star": k_star(upper), "targets": found, "skipped": skipped}


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
) -> dict[str, Any]:
    """The one repair round from the best unconstrained triple, over ``records`` (every rate of
    the arm at the batch): a seed with no K below K* adds r_5 / 1.21; a seed with no K at or above
    K* adds r_U × 1.21; a bracket wider than w_max adds the geometric midpoint of the widest one's
    pair; a target missed by more than 0.1 in ln K is re-targeted, its rate interpolated again
    over every usable rate (outside the usable range, the outward extension on its side). Rates
    rounded to 3 significant digits. A coverage miss whose repair rate is none or a rate already
    run is ``unrepairable``: the candidate fails."""
    if triple is None:
        return {"from": None, "rates": [], "reasons": ["no triple to repair from"],
                "unrepairable": []}  # fmt: skip
    r5, _, r_upper = triple["rates"]
    per_seed = triple["per_seed"]
    proposals: list[dict[str, Any]] = []
    low = [seed for seed, b in per_seed.items() if b["lo"] is None]
    high = [seed for seed, b in per_seed.items() if b["hi"] is None]
    wide = [seed for seed, b in per_seed.items() if b["width"] is not None and b["width"] > w_max]
    if low:
        proposals.append({"rate": _three(r5 / EXTENSION),
                          "reason": f"no K below K* on seeds {low}: "
                                    f"r_5 / {EXTENSION}"})  # fmt: skip
    if high:
        proposals.append({"rate": _three(r_upper * EXTENSION),
                          "reason": f"no K at or above K* on seeds {high}: "
                                    f"r_U × {EXTENSION}"})  # fmt: skip
    if wide:
        seed = max(wide, key=lambda s: per_seed[s]["width"])
        pair = per_seed[seed]
        lo, hi = pair["lo"]["rate"], pair["hi"]["rate"]
        proposals.append({"rate": _three(math.sqrt(lo * hi)),
                          "reason": f"bracket {pair['width']:.3f} wider than {w_max} on seed "
                                    f"{seed}: midpoint of {lo:g} and {hi:g}"})  # fmt: skip
    unrepairable = []
    retargeted = targeted_rates(records, upper)["targets"]
    existing = {rate(lr) for lr in records}
    for target, deviation, entry in zip(targets(upper), triple["deviations"], retargeted,
                                        strict=True):  # fmt: skip
        if abs(deviation) <= COVERAGE:
            continue
        lr, how = entry["rate"], "re-targeted over every usable rate"
        if entry.get("reason") == "below the usable range":
            lr, how = _three(r5 / EXTENSION), f"below the usable range: r_5 / {EXTENSION}"
        elif entry.get("reason") == "above the usable range":
            lr, how = _three(r_upper * EXTENSION), f"above the usable range: r_U × {EXTENSION}"
        missed = f"target {target:.4g} missed by {deviation:+.3f} in ln K"
        if lr is None or rate(lr) in existing:
            got = f"{entry['reason']}" if lr is None else f"{lr:g}, already run"
            unrepairable.append(f"{missed}: {how} gives {got}")
        else:
            proposals.append({"rate": lr, "reason": f"{missed}: {how}"})
    return {
        "from": triple["rates"],
        "rates": list(dict.fromkeys(p["rate"] for p in proposals)),
        "reasons": [p["reason"] for p in proposals] + unrepairable or ["nothing to repair"],
        "proposals": proposals,
        "unrepairable": unrepairable,
    }


def candidate_order(
    runs: Sequence[Trajectory],
    theta0: Mapping[str, float],
    seconds_per_chunk: float,
    *,
    grid: Sequence[float] = SHORT_GRID,
    seeds: Sequence[int] = DISCOVERY_SEEDS,
) -> list[dict[str, Any]]:
    """The candidates in amendment 4's fixed order (``CANDIDATES``: (U, B) = (8, 20), (7, 20),
    (8, 5), (7, 5)), each with its cost for the report, which no longer orders them: the median
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
    norm, the survival share and its retry scale, the twin bound 0.01 × K* and the steps, with
    the Gaussian directions' seeds and the reference's modes as the numerical-check tool fixes
    them (``_tool_policy``)."""
    return {**NUMERICAL_POLICY, "bound": NUMERICAL_BOUND_FRACTION * k_star(upper),
            **_tool_policy()}  # fmt: skip


@functools.cache
def _tool_policy() -> dict[str, Any]:
    """What the numerical-check tool fixes and this module does not restate: the seeds of the
    Gaussian directions p3 and p4 (its tracked settings, configs/experiments/
    numerical_check.yaml, read through its own ``NumericalScope``) and the reference's modes
    (``NUMERICAL_REFERENCE``), both from ``fa_control_diagnostic`` (review of Task 2, minor 1).
    Imported when first needed: the tool's module loads torch."""
    from .. import fa_control_diagnostic as tool
    from ..paths import CONFIG_ROOT

    settings = load_config(CONFIG_ROOT / "experiments" / "numerical_check.yaml")
    scope = tool.NumericalScope.of(settings)
    return {"gaussian": dict(scope.gaussian), "reference": tool.NUMERICAL_REFERENCE}


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
    for key in ("gaussian", "reference"):  # the tool's own values (``_tool_policy``)
        if policy.get(key) != frozen[key]:
            out.append(f"{key} {policy.get(key)!r}, frozen {frozen[key]!r}")
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
    ``survival_share``, ``retry_scale``, ``bound``, ``steps``, ``gaussian`` and ``reference``,
    as ``numerical_policy`` freezes them, ``tool_commit``: a
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
    the rounds spent per arm, batch and U, the rates found unusable per arm and batch, the
    numerical and validation runs called for, each candidate's numerical checks, and the
    batches reached."""

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
        self.called: set[tuple] = set()  # the numerical and validation runs called for
        self.read: set[str] = set()  # the verdicts the replay read
        self.reached: set[int] = set()

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

    def propose(self, kind, arm, batch, upper, selected) -> dict[str, Any]:
        """The arm's ``kind`` round at the batch and U, over every run made at that batch."""
        records = self.records(arm, batch)
        if kind == "targeted":
            proposal = targeted_rates(records, upper)
            proposed = [t["rate"] for t in proposal["targets"] if t["rate"] is not None]
        else:
            proposal = repair_rates(selected["best_unconstrained"], records, upper, self.w_max)
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
    unusable rate leaving B and the selection re-solved once; then the ``validation`` campaign
    (seeds 2100-2111) and the three-arm validation verdict. A failure at any step takes the next
    candidate. When no candidate is eligible with the three arms, the candidates whose primary
    arms did not fail at the numerical gate are visited again, in the same order, for the
    primary arms alone (the pre-warmed arm dropped, amendment 3's primary-arm validation),
    reusing every record and carrying a primary re-solve of pass 1; when none is eligible
    either, the stage ends ``infeasible`` with every attempt's reasons. ``numerical``: the
    verdict records by their path under $RUNS_DIR (``read_numerical``), None when not read;
    ``facts``: ``run_facts`` by run id (the commit and checkout state of each numerical run);
    ``exhausted``: the numerical and validation runs the launcher gave up on
    (``exhausted_runs``); ``reuse``: the GPU-code reuse check (``numerical_verdict``),
    ``code_identity.reuse_check`` on this repository by default. ``next["campaign"]`` is one of
    ``discovery``, ``repair``, ``prewarmed``, ``targeted``, ``numerical``, ``numerical-check``,
    ``validation`` (with what to make or check), ``done`` (with the frozen rates, ``three_arm``
    and ``prewarmed_dropped``), ``infeasible`` or ``incomplete``: ambiguous records, or any
    ``problems`` (review of Task 2, minor 5), the replay's own step then kept as ``pending``
    (and, at a stale check with runs beyond it, ``beyond_stale``)."""
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
    for arms in (ARMS, PRIMARY_ARMS):  # the three arms; then the fallback, the primary arms
        for upper, batch in CANDIDATES:
            carried = False
            if arms == PRIMARY_ARMS:
                first = next(a for a in attempts if (a["batch"], a["U"]) == (batch, upper))
                if first["failed_at"] == "numerical" and set(
                        first["failed_arms"]) & set(PRIMARY_ARMS):  # fmt: skip
                    continue  # its primary arms failed their numerical gate: so they do alone
                # A primary re-solve of pass 1 is spent; one forced by pre-warmed rates is not.
                carried = any(u["arm"] in PRIMARY_ARMS for e in first["numerical"]
                              for u in e.get("unusable") or ())  # fmt: skip
            attempt, step = _visit(replay, batch, upper, arms, carried)
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
    elif step is None:  # no candidate eligible, even for the primary arms alone
        step = {"campaign": "infeasible", "reasons": [
            f"U={a['U']:g}, B={a['batch']}, {'three arms' if a['three_arm'] else 'primary arms'}"
            f": {'; '.join(a['reasons'])}" for a in attempts]}  # fmt: skip
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


def _fail(attempt: dict[str, Any], arms: Collection[str], reason: str, at: str) -> None:
    """The attempt fails on ``arms`` at the gate ``at`` (discovery, numerical or validation),
    with the reason."""
    attempt.update(outcome="failed", failed_at=at)
    attempt["failed_arms"] = [arm for arm in ARMS if arm in {*attempt["failed_arms"], *arms}]
    attempt["reasons"].append(reason)


def _visit(
    replay: _Replay, batch: int, upper: float, arms: Sequence[str], carried: bool = False
) -> tuple[dict, dict | None]:
    """One visit of a candidate for ``arms`` (the three arms, or the primary arms alone in the
    fallback): discovery, the numerical gate, the validation gate. ``carried``: the fallback's
    visit of a candidate whose primary arms pass 1 re-solved already, so that its re-solve is
    spent (review of Task 2, critical 1). Returns the attempt and the pending step, if any; the
    attempt's outcome is ``pending``, ``incomplete``, ``failed`` (with the arms it failed on) or
    ``eligible`` (with the rates)."""
    replay.reached.add(batch)
    # Rates found unusable at this batch earlier in the replay (any U) are excluded up front.
    excluded = {arm: dict(replay.unusable[(arm, batch)]) for arm in arms}
    attempt: dict[str, Any] = {"batch": batch, "U": upper, "K_star": k_star(upper),
                               "three_arm": PREWARMED in arms, "resolve_carried": carried,
                               "arms": {arm: {"selections": [], "rounds": []} for arm in arms},
                               "excluded": {arm: dict(e) for arm, e in excluded.items()},
                               "numerical": [], "validation": None, "outcome": "pending",
                               "failed_arms": [], "failed_at": None, "reasons": [],
                               "rates": None}  # fmt: skip
    selected, step = _discovery(replay, attempt, excluded, batch, upper, arms, carried)
    if selected is None:
        return attempt, step
    selected, controls, step = _numerical(replay, attempt, excluded, batch, upper, selected, arms,
                                          carried)  # fmt: skip
    if selected is None:
        return attempt, step
    rates = {arm: selected[arm]["rates"] for arm in arms}
    step = _validation(replay, attempt, batch, upper, rates, controls)
    if step is None and attempt["outcome"] != "failed":
        attempt.update(outcome="eligible", rates=rates)
    return attempt, step


def _discovery(
    replay, attempt, excluded, batch, upper, arms, carried: bool = False
) -> tuple[dict | None, dict | None]:
    """A candidate's discovery: the primary arms' targeted round, as the joint ``discovery``
    campaign holding the pre-warmed arm's first round when it is due, then their repair round;
    then the pre-warmed arm's first round, if not made yet, and its own rounds. With
    ``carried`` (the candidate's re-solve spent in pass 1) a primary arm left without a triple by
    the unusable rates gets no new round: the candidate fails. Returns each arm's selection, or
    None with the pending step (none when the attempt failed)."""

    def solve(arm: str) -> dict[str, Any]:
        selected = replay.solve(arm, batch, upper, sorted(excluded[arm]))
        attempt["arms"][arm]["selections"].append(selected)
        return selected

    def pending(campaign: str, runs: list[dict[str, Any]]) -> tuple[None, dict]:
        attempt["outcome"] = "pending"
        return None, _step(campaign, attempt, runs=runs)

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
        if carried and fresh:  # the re-solve is spent: no new round for it
            _fail(attempt, [r["arm"] for r in fresh], f"{[r['arm'] for r in fresh]} without a "
                  "feasible triple once the unusable rates are dropped, the candidate's re-solve "
                  "spent in pass 1", "numerical")  # fmt: skip
            return None, None
        blocked = [r["arm"] for r in rounds if r["proposal"].get("unrepairable")]
        for spent in rounds:
            attempt["arms"][spent["arm"]]["rounds"].append(spent)
        for spent in fresh:
            replay.spend(spent, prescribe=not blocked)
        if blocked:
            _fail(attempt, blocked, f"{blocked}: a coverage miss that re-targeting cannot repair",
                  "discovery")  # fmt: skip
            return None, None
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
        _fail(attempt, infeasible, f"{infeasible} without a feasible triple after the rounds",
              "discovery")  # fmt: skip
        return None, None
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
        spent, new = replay.round(kind, PREWARMED, batch, upper, selected[PREWARMED])
        attempt["arms"][PREWARMED]["rounds"].append(spent)
        blocked = bool(spent["proposal"].get("unrepairable"))
        if new:
            replay.spend(spent, prescribe=not blocked)
        if blocked:
            break
        missing = replay.missing(PREWARMED, batch, spent["new_rates"])
        if missing:
            return pending(kind, missing)
        selected[PREWARMED] = solve(PREWARMED)
    if not selected[PREWARMED]["feasible"]:
        _fail(attempt, [PREWARMED], "the pre-warmed arm without a feasible triple after its "
              "rounds", "discovery")  # fmt: skip
        return None, None
    return selected, None


def _numerical(replay, attempt, excluded, batch, upper, selected, arms, carried=False) -> tuple:
    """The numerical gate: the K-only regular runs of the selected rates' cases
    (``numerical_cases``; one the launcher gave up on lets the check go on, its case then not
    qualified), then the check's verdict (``numerical_verdict``). Unusable rates (failing, not
    qualified, or an FA rate whose control was rejected or not qualified; a stale verdict's marks
    included) leave the batch, and the arms holding them are re-solved once without them (no new
    round); the new selection's cases are run and checked in turn (a new check), and an unusable
    rate there fails the candidate. ``carried``: the re-solve was spent in pass 1 (the fallback's
    visit), so a first unusable primary rate fails the candidate. Returns the selections and the
    FA rates' accepted controls, or None with the pending step (none when the attempt failed)."""

    def stop(step: dict | None, outcome: str = "pending") -> tuple[None, None, dict | None]:
        attempt["outcome"] = outcome
        return None, None, step

    def incomplete(problems: list[str]) -> tuple[None, None, dict]:
        return stop({"campaign": "incomplete", "problems": problems}, "incomplete")

    resolved = carried
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
        for unusable in verdict["unusable"]:
            why = f"{unusable['why']} ({path}{', stale' if verdict['state'] == 'stale' else ''})"
            replay.unusable[(unusable["arm"], batch)].setdefault(unusable["lr"], why)
            excluded[unusable["arm"]].setdefault(unusable["lr"], why)
        failed = [arm for arm in arms if any(u["arm"] == arm for u in verdict["unusable"])]
        shown = [f"{u['arm']} {u['lr']:g}: {u['why']}" for u in verdict["unusable"]]
        if resolved:
            _fail(attempt, failed, f"unusable rates after the re-solve"
                  f"{' (spent in pass 1)' if carried else ''}: {shown}", "numerical")  # fmt: skip
            return None, None, None
        resolved = True
        for arm in failed:
            selected[arm] = replay.solve(arm, batch, upper, sorted(excluded[arm]))
            attempt["arms"][arm]["selections"].append(selected[arm])
        infeasible = [arm for arm in failed if not selected[arm]["feasible"]]
        if infeasible:
            _fail(attempt, infeasible, f"{infeasible} without a feasible triple once the "
                                       f"unusable rates are dropped: {shown}",
                  "numerical")  # fmt: skip
            return None, None, None


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
    fails the candidate. Returns the pending step, if any."""
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
        _fail(attempt, arms, "the validation does not pass: " + "; ".join(verdict["reasons"]),
              "validation")  # fmt: skip
    return None


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
    - ``secondary_status``: how they are reported, one of ``SECONDARY_STATUSES``.

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
    }


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
    ``infeasible``), then ``confirmation`` (with the frozen rates) once a candidate passed its
    validation; ``incomplete`` whenever the records are not whole (``problems``), a time-course
    run among ``trajectories`` included. The time-course runs are checked with the others
    (conformance, one stream and one v₀ per seed), then read apart: once the rates are frozen,
    ``timecourse`` is their ``timecourse_summary``, without labels; they never enter the
    selection, a confirmation block or a look."""
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
    out["confirmation"] = _confirmation(found, gone, rates, batch, kstar, w_max)
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


def _confirmation(found, gone, rates, batch, kstar, w_max) -> list[dict[str, Any]]:
    """Every confirmation block in its preassigned order: its runs (an exhausted run counts as
    one), complete once each arm has a run at each of its frozen rates, then each arm's support,
    in which an exhausted run is a missing rate: a technical failure."""
    blocks = []
    for number, seed in enumerate(CONFIRMATION_SEEDS, 1):
        keys = _keys(rates, batch, (seed,))
        present = [key for key in keys if key in found]
        lost = [gone[key] for key in keys if key in gone]
        runs = len(present) + len(lost)
        entry: dict[str, Any] = {"block": number, "seed": seed, "runs": runs,
                                 "expected": len(keys), "complete": runs == len(keys),
                                 "exhausted": lost, "supported": False,
                                 "supports": {}}  # fmt: skip
        if entry["complete"]:
            entry["supports"] = {
                arm: block_support([found[key] for key in present if key[0] == arm], kstar,
                                   w_max, rates=lrs)
                for arm, lrs in rates.items()
            }  # fmt: skip
            entry["supported"] = all(entry["supports"][arm]["supported"] for arm in PRIMARY_ARMS)
        blocks.append(entry)
    return blocks


def confirmation_guard(
    blocks: Sequence[Mapping[str, Any]],
    first: int,
    last: int,
    *,
    after_look: int = 0,
    issued: Collection[tuple[int, int]] = (),
    n: Sequence[int] = LOOK_SIZES,
    cap: int = CAP,
) -> dict[str, Any]:
    """Whether the confirmation blocks ``first`` to ``last`` may be started now, on K and the
    launcher's journal only (``progress``'s confirmation blocks; ``issued``: ``issued_ranges``).
    The blocks run in their preassigned order: a new range starts at the first block without a
    run (no gaps) and touches no block with runs (no overlaps), unless it re-issues exactly a
    range issued before, to complete it. A new range must also end before the next look:
    s = the supported blocks complete in order, p = every block with runs that may still be
    supported (all but the complete unsupported ones), look k = ``after_look`` + 1; it holds at
    most n_k − p blocks, and none once s ≥ n_k, until the summary has read look k and the operator
    passes ``after_look`` = k because it did not stop. Returns the problems (none: allowed) and
    those counts."""
    started = {b["block"] for b in blocks if b["runs"]}
    first_open = next((b["block"] for b in blocks if not b["runs"]), cap + 1)
    in_order = list(itertools.takewhile(lambda b: b["complete"], blocks))
    supported = sum(1 for b in in_order if b["supported"])
    possible = sum(1 for b in blocks if b["runs"] and not (b["complete"] and not b["supported"]))
    out: dict[str, Any] = {"first_open": first_open, "supported_in_order": supported,
                           "possibly_supported": possible, "problems": []}  # fmt: skip
    problems = out["problems"]
    if not 1 <= first <= last <= cap:
        problems.append(f"blocks {first}-{last}: 1 <= FIRST <= LAST <= {cap}")
        return out
    overlap = sorted(started & set(range(first, last + 1)))
    if overlap:
        if (first, last) not in set(issued):
            problems.append(f"blocks {overlap[0]}-{overlap[-1]} have runs already, and "
                            f"{first}-{last} is not a range issued before: ranges never "
                            "overlap")  # fmt: skip
        return out  # a range issued before is completed whatever the looks say now
    if first != first_open:
        problems.append(f"the next range starts at block {first_open}, the first without a run, "
                        f"not {first}: the blocks run in their preassigned order")  # fmt: skip
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
                        f"before look {k}: {possible} blocks with runs may be supported, look "
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
    """What the identity check compares, (time course, endpoint) per record, canonical: the
    training's metrics rows (every row of metrics.jsonl but those an evaluation logged), the rows
    the endpoint's evaluations logged against the time course's rows of those steps, the monitor
    readings, the update scalars, the endpoint's evaluations against the time course's of those
    steps, the stop with the steps taken, and the streams the run read (meta.json)."""
    out: dict[str, tuple] = {}
    mine, theirs = _run_records(timecourse), _run_records(endpoint)
    metrics = (mine["metrics.jsonl"], theirs["metrics.jsonl"])
    out["the training's metrics.jsonl rows"] = tuple(
        _canonical([r for r in rows if not _evaluated(r)]) for rows in metrics
    )
    logged = {row["step"] for row in metrics[1] if _evaluated(row)}
    out["the metrics.jsonl rows of the endpoint's evaluations"] = tuple(
        _canonical([r for r in rows if _evaluated(r) and r["step"] in logged]) for rows in metrics
    )
    for name, what in (("monitor.jsonl", "readings"), ("update_scalars.jsonl", "rows")):
        out[f"the {name} {what}"] = (_canonical(mine[name]), _canonical(theirs[name]))
    steps = {row["step"] for row in endpoint.evaluations}
    out["the endpoint's evaluations"] = (
        _canonical([r for r in timecourse.evaluations if r["step"] in steps]),
        _canonical(endpoint.evaluations),
    )
    out["the stop and the steps taken"] = tuple(
        _canonical([{"stop": run.terminal["stop"], "steps": run.steps}])
        for run in (timecourse, endpoint)
    )
    out["the streams read"] = tuple(_canonical([(record["meta.json"].get("inputs") or {})
                                                .get("streams") or {}])
                                    for record in (mine, theirs))  # fmt: skip
    return out


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
    """Whether a time-course run trained as its endpoint run did, bit for bit (amendment 4,
    section 4: endpoint identity is required). ``endpoint``: the regular run of the same arm,
    batch, rate and seed, None while it is not there. Compared exactly (``_canonical``, without
    the timings, the memory peaks and K_harmful): every metrics.jsonl row of the training
    (loss, gradient norm and every other field, steps 0-24, and the monitor's and diagnostics'
    rows), the rows of the endpoint's evaluation there, every monitor.jsonl reading,
    update_scalars.jsonl, the step-25 evaluation (K and utility), the stop and the steps taken,
    and the streams read. ``state`` is ``identical``, ``excluded`` (any difference, with the
    ``reasons``; also a record that is missing or does not read: ``_run_records``) or
    ``pending`` (no endpoint run yet). ``rayleigh_equal`` (``_rayleigh_equal``) is reported
    beside, never deciding the state."""
    if endpoint is None:
        return {"state": "pending", "reasons": ["its endpoint run is not there yet"],
                "endpoint": None, "rayleigh_equal": None}  # fmt: skip
    keys = [(run.arm, run.batch_size, rate(run.lr), run.seed) for run in (timecourse, endpoint)]
    reasons = []
    if keys[0] != keys[1]:
        reasons.append(f"the endpoint run {endpoint.run_id} is of {keys[1]}, not {keys[0]}")
    else:
        try:
            records = _identity_records(timecourse, endpoint)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            records, reasons = {}, [f"a record does not read: {exc!r}"]
        for name, (mine, theirs) in records.items():
            if mine == theirs:
                continue
            first = next((i for i, (a, b) in enumerate(zip(mine, theirs, strict=False))
                          if a != b), min(len(mine), len(theirs)))  # fmt: skip
            reasons.append(f"{name} differ from the endpoint run's from row {first} "
                           f"({len(mine)} rows against {len(theirs)})")  # fmt: skip
    rayleigh = _rayleigh_equal(timecourse, endpoint) if keys[0] == keys[1] else None
    return {"state": "excluded" if reasons else "identical", "reasons": reasons,
            "endpoint": endpoint.run_id, "rayleigh_equal": rayleigh}  # fmt: skip


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

    Per seed and arm (``trajectories``): the run, its identity with its endpoint run
    (``timecourse_identity``; ``endpoints``: the confirmation runs by key; an endpoint the
    launcher gave up on, in ``exhausted``, leaves the identity unshowable: ``excluded``), and its
    rows (step, K_scoring, harm), K_scoring the scoring panel's K of each evaluation. ``failures``
    lists the runs that did not complete their four evaluations (stopped early, or given up by
    the launcher: ``exhausted`` by (arm, batch, rate, seed, ``TIMECOURSE``)), ``missing`` the
    runs not there (``endpoint_given_up``, with the launcher's label, when its endpoint run was
    given up: the campaign does not run it, since its identity could never be shown);
    ``problems`` a second run of a key or a run the time course does not prescribe (records that
    are not whole).

    ``labels`` (run id → label rows, as ``look_blocks`` reads them) is passed only by the final
    analysis. Without it nothing is read and every harm is None; with it, the harm of each step of
    an identical run (``harm``: missing labels not harmful), a run absent from it
    ``unlabeled``."""
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
            identity = {"state": "excluded", "endpoint": None, "rayleigh_equal": None,
                        "reasons": [f"its endpoint run was given up by the launcher ({given_up}): "
                                    "the identity cannot be shown"]}  # fmt: skip
        else:
            identity = timecourse_identity(run, endpoint)
        entry.update(run=run.run_id, **identity)
        rows = labels.get(run.run_id) if labels is not None and identity["state"] == (
            "identical") else None  # fmt: skip
        if labels is not None and identity["state"] == "identical" and rows is None:
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
