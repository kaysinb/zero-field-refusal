# Study 2: diagnostic of the FA soft-control failure

**Status: frozen before any diagnostic run** (2026-10-17). It is a specification outside the
confirmation: it changes nothing in [amendment 3](study2-preregistration.md#amendment-3-redirection-to-the-short-regime),
its selection or its records. File:line references are to commit 1e14ca1.

**Context.** Two of three FA soft-target controls (seed 2000, B 5, 25 steps, rates 1.21e-5 and
1.46e-5) stopped by the contamination check: their K grew exponentially from a small residual (×8–20
per 5 steps), with tiny displacement (K/‖δ‖² ≈ 35), while the regular FA runs at the same rates
look normal. The coordinator chose a diagnostic, without changing the protocol.

**Rules.**
- The diagnostic is outside the confirmation and **blind to outcomes**. Its outputs are only K,
  norms, eigenvalues and v₀ statistics.
- No harm, refusal or judge quantity is computed or written. No harmful-prompt generation runs at
  all: every evaluation in the diagnostic is K-only (the scoring panel and the monitor panels).
- Nothing changes in the protocol, the selection or the existing run records.
- New code lives in a new module and script. Its outputs go to `RUNS_DIR/diagnostics/fa_control/`,
  a new namespace never read by the stage's analysis.

## D1. The v₀ census

- Estimate v₀ exactly as FA does for seed 2000 at B 5 (`train._theta0_second_moment`, the
  preconditioner stream). Use the same code path, and record the stream's identity.
- Per module group (embedding / per-layer attention q,k,v,o / MLP / norms / lm_head if trainable),
  over the trainable parameters, report:
  - quantiles of √v₀;
  - the fractions with √v₀ < ε and < 10ε (ε = 1e-8);
  - the maximum of P = 1/(√v₀ + ε), and where it occurs (parameter name and index stats).
- Also report the distribution of P·|g_soft| for the soft control's first step, if cheap.

## D2. Preconditioned sharpness at θ₀

- The top eigenvalue of P^½ H P^½ at θ₀, with P = diag(1/(√v₀+ε)), by Lanczos on Hessian–vector
  products. Reuse `curvature.lanczos` and `hessian_vector_product`, and add a soft-target-loss
  HVP if none exists.
- Two Hessians, both on the soft control's first training batches (seed 2000's stream, B 5, the
  first k batches, with k fixed in advance, e.g. 4 pooled):
  - (a) the soft-target loss with its replayed θ₀ targets;
  - (b) the NLL loss.

  Use the same loss normalization as training.
- Report:
  - λ_max for each Hessian;
  - the module localization of the top eigenvector (norm share per module group);
  - the stability thresholds of the actual FA update at each rate η ∈ {1e-5, 1.21e-5, 1.46e-5}:
    - with the first moment as implemented (EMA β₁ 0.9, bias-corrected): derive the spectral
      radius of the linear map from the code's update and state the threshold;
    - the heavy-ball reference 2(1+β₁)/((1−β₁)η);
    - without momentum, 2/η.
- **Consistency check:**
  - the predicted per-step growth factor of δ along the top mode at each rate, from the derived
    linear map;
  - against the observed one, from the controls' monitor K: growth of δ per step = √(growth of K
    per step), with K ∝ ‖δ‖² (records under runs/study2_short_fa_controls/).

## D3. The twin test (does the regular run itself amplify fp noise?)

- **Determinism first.** Re-run the regular FA run of seed 2000 at 1e-5 (same config, stream and
  rate as the targeted round's run), K-only, and check that its per-step metrics (loss, grad norm)
  and monitor K equal the archived run's bit for bit. If they differ, report it and stop D3.
- **Twins**, for each FA rate in {1e-5, 1.21e-5, 1.46e-5}:
  - twin A is the regular run, unperturbed, saving the trainable parameters at steps 5, 10, 15,
    20 and 25 (fp32, to the Pod's volume);
  - twin B is the same run with θ₀ perturbed at the fp-noise level. The perturbation's norm
    equals the soft control's displacement norm after its first step at that rate (read it from
    the control's record, or recompute it). Its direction is a fixed-seed Gaussian over the
    trainable parameters.
  - At each of those steps, compute K(A_t ‖ B_t) on the scoring panel, the same KL definition as
    K_benign, with A_t as the reference.
  - Report the series against the bound 0.0632 and against K\* = 6.32.
- The same for AdamW-25 at 1.32e-5 (its selected middle rate), as a comparison.
- **Memory:** two parameter copies plus the optimizer state must fit within 46 GB reserved. Report
  the peaks.

## D4. Control path against regular path

Read the code and list every difference between the soft-control path and the regular path for
FA:
- token positions trained;
- the targets' and logits' dtype;
- the loss scale and denominator;
- ε and β₁;
- bias correction;
- gradient clipping;
- the gradient rule (logged vs stop);
- the replay;
- anything else.

Each difference gets file:line and a note on whether it can change the dynamics.

## Delivery

- **Code.**
  - A module, e.g. `src/zero_field_refusal/diagnostics/fa_control.py`.
  - A script `scripts/fa_control_diagnostic.py`, with subcommands d1, d2, d3 and a `--dry-run`.
  - A config if needed: `experiments/fa_control_diagnostic.yaml`, K-only evaluation and no harmful
    generation. Assert that in a test.
  - CPU tests on tiny models:
    - the soft-loss HVP against a dense Hessian;
    - the preconditioned Lanczos;
    - the twin K computation;
    - that no harmful generation happens.
- **A driver** for P1 at the new commit. Cost estimate per part: GPU minutes and peak memory.
- **D4** as a markdown table in the report.
- **Out of scope:** no protocol, selection or analysis code changes.

---

## Revision 1 (frozen)

After review fa5. It supersedes the text above where they differ.

Source: the external review fa5 of the plan ("C as planned — yes with changes").

### Order

D4 and D1 first, then D2, then D3.

### D4 + D1 (extended)

**Verify that the control and the regular run share, bit for bit:**
- the same v₀ (hash it);
- the same loss normalization;
- the same trainable parameter set;
- the same initial optimizer state (m₀, step counter).

**The initial residual gradient** (≈ 1.14e-4 in the records) is not assumed to be fp noise. The
control compares different forward paths (`src/zero_field_refusal/losses.py:62`): the targets come from the generation
path, the student's logits from the training path. Separate the two sources at θ₀ on the
control's first batch:
- (i) the gradient norm of the soft loss with the targets recomputed through the **training**
  path at θ₀, in fp32;
- (ii) the same with the loss and the gradient accumulation in float64, the forward still in fp32;
- (iii) the recorded control's first gradient.

Report the three norms and their pairwise cosine.

**The census of ε-dominated coordinates** (√v₀ < ε, and < 10ε) covers their count by module and
also their contribution:
- the share of ‖P·g‖² of the first step coming from those coordinates;
- their share of the top eigenvector of D2.

### D2 (on the actual recurrence)

**The recurrence** is the code's own update (`src/zero_field_refusal/optim.py:80`):
- m_t = β m_{t−1} + (1−β) g_t;
- θ_{t+1} = θ_t − η P m_t / (1 − β^t), with β = 0.9.

**Finite-horizon amplification over 25 steps** with changing batches. Take the control's first
4 batches (B 5, seed 2000's stream, fixed) and cycle them as the batch sequence for the 25 steps.
For each of η ∈ {1e-5, 1.21e-5, 1.46e-5}, compute the growth of ‖δ_t‖ and of δ_tᵀ F δ_t under the
linearized map with:
- (a) the soft-target Hessian (Fisher/Gauss–Newton) per batch;
- (b) the sampled-NLL Hessian per batch, which may be indefinite; report its most negative Ritz
  value too.

Start the map from the measured residual direction. Also report:
- the asymptotic heavy-ball reference 2(1+β)/((1−β)η);
- the first-step multiplier 1 − ηλ, beside the finite-horizon result;
- the top Ritz value of P^½ H P^½ per batch, for (a) and (b);
- its module localization.

Compare with the controls' observed per-step growth of δ = √(K growth per step).

### D3 (strengthened)

**Hold fixed:** tokens (the stream), RNG, v₀ and the optimizer initialization.

**Cases**, in this order:
- FA 1.21e-5 (failing);
- FA 1e-5 (passing);
- AdamW 1.32e-5.

FA 1.46e-5 is added only if the three cases give an ambiguous branch (below), and that rule is
fixed now.

**Perturbations** per case, each scaled to the norm of the control's first-step displacement at
that rate:
- (p1) the measured residual direction, i.e. the control's first update direction at that rate;
- (p2) its opposite;
- (p3, p4) two fixed-seed independent Gaussian directions over the trainable parameters.

Verify that θ₀ + p differs from θ₀ after fp32 rounding (count the changed coordinates); a
perturbation that does not survive rounding is reported and scaled up ×10 once.

**Reference:** the unperturbed regular NLL run in a numerically improved form with the same
mathematical optimizer:
- the loss, the gradient accumulation across micro-batches and the optimizer arithmetic in
  float64;
- the forward and backward in fp32;
- TF32 off.

If memory does not allow float64 optimizer state for all parameters, report that and use float64
for the loss and accumulation only.

**Measure:** D_twin(t) = max(KL(A_t‖B_t), KL(B_t‖A_t)), full vocabulary, averaged as K_benign is, on
the same fixed benign scoring continuations (the scoring panel's 400 responses), at steps 1, 5, 10,
15, 20 and 25. Compute it for:
- the unperturbed run against each perturbed twin;
- the unperturbed run against the reference.

Never use the difference of θ₀-relative K values.

**Bound:** 0.01·K\* = 0.06325, the same for every pair.

**Determinism** is checked first, as in the base spec.

### Decision branches

Frozen. All failures are kept and reported.

- **Branch F — fix the control (a path defect).** Applies when D4/D1 show that the residual comes
  from the path discrepancy, the same-path float64 soft control at FA 1.21e-5 keeps K below 0.06325
  through 25 steps, and D3's regular runs stay within the bound.

  Outcome: propose a corrected control. It is a new protocol, validated on fresh seeds, and this
  round's controls stay failed.
- **Branch B — a new composite criterion.** Applies when the same-path control still exceeds the
  bound at 1.21e-5, but D3's regular NLL runs at 1.21e-5 keep max_t D_twin ≤ 0.06325 for every
  perturbation and against the reference.

  Outcome: propose a composite criterion (teacher-path checks, perturbation sensitivity and
  reference agreement on NLL trajectories), validated on fresh seeds. The user decides.
- **Branch A — the regular runs are contaminated.** Applies when any D3 pair at FA 1.21e-5 exceeds
  0.06325.

  Outcome: the rates stay unusable as the rule says. The user decides on A (redesign) or on
  stabilizing the preconditioner. B = 20 stays untested, and is not presumed infeasible.
- **Ambiguous** (none of the above): add FA 1.46e-5 to D3 once, then report without choosing.

### For the report only (no edit to the frozen amendment text now)

- The amendment's statement that a fixed denominator keeps tiny errors tiny ("Why FA qualifies" in
  [its numerical acceptance](study2-preregistration.md#numerical-acceptance-1))
  is wrong: linearity of an update does not imply stability of repeated updates.
- B = 20 is untested.
- This round's controls remain failed. The diagnostic can only justify a new protocol, tested on
  fresh seeds.
