### E2 prediction note (2026-11-14): the prospective test on the order repeats

**Status.** Written after extension 2's measurements on the old seeds and before any outcome of
extension 1's order repeats (E2) is read: no label, K_harmful, K_monitor, utility or harm share of
an E2 run has been read. It fixes the predictor, the outcome and the test. The θ₀-based records it
lists by sha256 in the committed inputs file (below), each computed from θ₀ and its order alone
(E2's in extension 2's `sealed` block, the top-up's before its training), are kept as inputs; this
branch's predictor is measured on the E2 runs (below).

**The predictor**, chosen by extension 2's δ_lin validation criterion (M2). Its result is
extension 2's analysis record at commit `a129f47` (sha256
`b109cb3a298459492cb756f62da96d351eb1203e5b91046dceb0ffc21fa17d0e`). δ_lin exceeded the bound
10‖Δθ_i‖ on all 40 own orders, so no seed was left for ρ(q, Δr) or for mean q / mean Δr (n = 0;
neither was computed). The order-control pairs agree on 0 of 6, a pair whose q difference is
below 10 times the linear replay's repeat floor of q counting as a tie (fixed before the
criterion's numbers were read). Zero of six pairs counted as agreements because δ_lin diverged in
every pair; no q-sign comparison was available. The criterion failed.

Extension 2's fourth amendment then replaced the central difference in H_t δ with the exact
Hessian–vector product, and its fifth amendment fixed how that product is checked against the
attention kernel. On seeds 50000 and 50025, the probe met amendment 4's remaining gates and
amendment 5's replacement checks; the original default-gradient discrepancy remains recorded. The
validated exact recursion exceeded the bound on both seeds: at step 3 on seed 50000 and at step 5
on seed 50025. It did so in the same way with every production gradient under the math kernel
and, on seed 50000, in its repeat (archives `dlin-probe-20261115`, `dlin-probe-20261115-timing`
and `dlin-probe-20261115-kernel`). By amendment 4's stopping rule: probe stopped; corrected
criterion not evaluated. The criterion's result stays the record above. The correction followed
the original all-diverged result. No corrected E2 or top-up predictions were produced. The note
therefore retains its pre-specified association branch:
- the linear-response predictor did not pass its check; this says nothing against the
  mechanism. The predictor is then ln K_harmful at K\* of each order, measured on the E2 runs themselves
  with the interpolation of h at K\*; higher ln K_harmful goes with more harm. This is an
  association between two measurements of the same runs, not a prediction from θ₀.

**The outcome.** Primary: h at K\* of each order by the registered support rule and interpolation
over the two rates. Secondary, deciding nothing: the harm share at η_l of each order's η_l run,
without interpolation, on a paired cohort fixed now: the seeds whose two new orders' η_l runs both
have a harm share and a predicted sign (with ln K_harmful at K\* as the predictor, the predicted sign
itself needs support at K\*); support of h at K\* plays no other part. A technically missing η_l run
or label withholds the secondary result as well; an η_l run that stopped on an instability is a
failed endpoint by the registered rule and is excluded, not withheld. Δr at the endpoint, named as a secondary outcome when the test was
planned, cannot be computed for E2 runs: control runs save no checkpoint by the registration
(`experiments/study5_control_order`).

**The test.** For each seed, the two new orders (s+1000 and s+2000) are compared within the seed,
so the seed's own effect cancels; the original order is not part of the test. A_s = sign(predicted
difference) · sign(h₁ − h₂), the predicted difference being q₂ − q₁ (or ln K_h,1 − ln K_h,2). A seed
is left out and listed only for a scientific reason: a new order without support at K\* by the
registered rule; a declared δ_lin divergence (with q as the predictor only); with ln K_harmful at
K\* as the predictor, a K_harmful that was recorded but is not finite or not positive; or a zero in
either difference (a tie). A technically unresolved input (a missing journal, run, end, label,
prediction record, or a K_harmful that was not recorded) leaves no seed out: it withholds the
verdict until it is resolved. An established scientific exclusion of a seed takes precedence over
a technical gap in that seed when the gap cannot change the exclusion (for example an unsupported
order beside a missing label of the other order); then the seed is excluded, not withheld. With n seeds left and T of them agreeing, the one-sided p is P(Bin(n, ½) ≥ T), the exact
sign-flip null; the test succeeds at p < 0.05. Beside it, descriptive: the magnitude-weighted
S = Σ_s sign(predicted)·(h₁ − h₂) with its sign-flip p (exact up to 20 seeds, sampled from
10⁶ sign patterns above), and the per-seed table
(`python -m zero_field_refusal.analysis.study5_e2_test`).

**How the result is read**, fixed before the opening:
- With q as the predictor, a success means that the order of the data affects harm predictably, and that
  the effect is visible from θ₀ through the linear response of the training recursion. A failure means
  that the test did not establish the predicted association. With low power (below), a failure does
  not refute the linear response as a predictor.
- With ln K_harmful at K\* as the predictor, either result is only an association between two
  measurements of the same runs.
- The test does not choose between the two readings of the erosion, (D) a common drift and (E) an even
  readout. That choice rests on extension 2's sign interventions (M4), not on E2.

**Power**, conditional on keeping 10 (or 40) eligible, non-tied seeds. With the ten E2 seeds, success needs 9 of 10 agreements; the power is 15 % at a
predictor accuracy of 70 % and 74 % at 90 %. With the top-up, 40 seeds, it
needs 26 of 40, with a power of 81 % at 70 %.

**Top-up.** Seeds 50000–50005 and 50016–50039, all fresh seeds outside E2, each in the
two new orders `train.order_seed` = s+1000 and s+2000 at both rates (120 runs), with δ̂, δ_lin and q
of their 60 orders computed from θ₀ before training. Seeds 50000–50005 were seen in earlier
exploratory checks and in the δ_lin criterion, with their registered control order
(order_seed = s) only; none of the new orders was observed. The test compares two new orders within
a seed, so what was seen of the seed itself cancels. These seeds test the prediction of a new
order within the same stream of training responses; they do not test a generalization to new
streams. Everything about the top-up (seeds, orders, predictions, the joint test over 40 seeds) is
fixed before any outcome of the ten E2 seeds is opened.

**Inputs bound before the opening.** A committed inputs file (`docs/study5-e2-inputs.json`, sha256
`6374d57e6510da7d4bea85ab3e4c660c5f6529c02c381b066e0440856a7f3b35`), committed with this note, lists: the sha256 of extension 2's analysis record that holds
the criterion, whose part-3 multiplier is 10; the manifest of the prediction records used; and the
exact set of tested seeds. The opening record names this commit, this note and the inputs file
with their sha256; the test checks both files against the commit and every input against the
committed inputs file. The test (`python -m zero_field_refusal.analysis.study5_e2_test`) refuses
before reading any outcome if any of them differs.

**Opening.** The outcomes are opened only after this note is committed and its hash recorded, the
top-up decided, its 120 runs ended, and the analysis script reviewed.
