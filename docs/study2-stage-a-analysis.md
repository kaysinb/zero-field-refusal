# Study 2 Stage A: K-only analyses

Analyses of Stage A's discovery records that use only K, curvature and the numerical screen, never
harm. They do not enter the [selection rule](study2-preregistration.md#selection-rule). The harm
embargo of the [preregistration](study2-preregistration.md#outcome-embargo) applies unchanged.

## Recorded predictions

### SGD-long at 0.001, B = 20

Recorded on 2026-10-15, after the first search step's SGD-long runs at 0.002 (B = 20) and 0.0005
(B = 5) and before any run at 0.001 exists.

**The law.** In the zero field every training response is a fresh sample from θ₀. Near θ₀ the
expected loss is then quadratic, ½ δᵀHδ, and the gradient noise of a batch of B examples has
covariance H / B, since the Fisher information of θ₀'s own distribution equals its expected
Hessian. SGD without momentum, δ ← δ − η(Hδ + ξ), then gives in each eigendirection i of H

    Var_i(t) = (η / 2B) · (1 − (1 − ηλ_i)^(2t)) / (1 − ηλ_i / 2).

If K_benign ≈ ½ δᵀFδ with F proportional to H in the directions that move, then for ηλ_i ≪ 1

    K(t; η, B) ≈ (η / B) · g(η t),    g(x) = ¼ Σ_i λ_i (1 − exp(−2 λ_i x)).

g depends only on the spectrum, not on η or B. Two runs therefore map onto each other without a
free parameter: K at (η, B, t) equals (η B′ / η′ B) times K at (η′, B′, t η / η′).

**Already checked.** On seed 2002, SGD-long at 0.0005 and B = 5 against 0.002 and B = 20 at
equal ηt (the B = 5 run's step 4t against the B = 20 run's step t). The median ratio of their
monitor K is 0.85 over the B = 5 run's steps 100–300, 0.95 over 300–600, 0.97 over 600–1000 and
0.97 over 1000–1600.

**Prediction.** For 0.001 at B = 20, K_0.001(t) = ½ K_0.002(t / 2) on the same seed. Seed 2002's
0.002 run, which stopped by plateau at step 1400 with a median monitor K of 0.254 over its last
100 steps, predicts the following. The values are medians of the rescaled monitor readings
within 50 steps of t, and the plateau rule (from step 1000, six windows of 100 steps) is applied
to those readings.

| Step t | Predicted K |
|---|---|
| 1000 | 0.104 |
| 1400 | 0.116 |
| 2000 | 0.128 |
| 2800 | 0.129 |

- The run stops by plateau between steps 1000 and 2800; on the rescaled readings the rule fires at
  step 1200. It reaches no level.
- At the stop, its median monitor K over the last 100 steps lies between 0.10 and 0.13.
- The same holds on seeds 2000 and 2001, each scaled from its own run at 0.002.

**What would contradict it.** On any seed, a median monitor K over the last 100 steps outside
0.08–0.16 at the stop. Also, a median ratio K_0.001(t) / (½ K_0.002(t / 2)) outside 0.85–1.15
over steps 1000 to the stop on the same seed.
