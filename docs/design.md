# Scientific design

This document is the full scientific design. The question it answers is stated in the
[README](../README.md#the-question). Binding items (hypotheses, the primary analysis, gate criteria
and power) are in [preregistration.md](preregistration.md); stages and compute are in
[plan.md](plan.md); the engineering design is in [architecture.md](architecture.md).

## The zero-field experiment

The model is fine-tuned on fresh samples from its own frozen copy θ₀. Such data contain nothing new
on average: θ₀ minimizes the expected loss over its own responses, so the expected gradient there is
exactly zero. We call this setting the *zero field*, after physics experiments in which no external
field acts and only thermal noise remains. Any systematic change in behavior is then created by
minibatch noise and by the optimizer's response to it, not by the content of the data.

The field stays zero only under exact conditions:
- the teacher samples at temperature 1, with no top-k, top-p or repetition penalty;
- responses are never filtered. Truncation at a maximum length is harmless, but selecting
  responses by length creates a real field;
- the student trains on the sampled token ids, with the teacher's chat template and numerical
  precision;
- every response is used once;
- the batch loss is the summed token negative log-likelihood divided by a constant. Dividing by
  the random number of tokens in the batch would favor short responses and create a weak real field;
- weight decay is off, because it is a systematic force of its own.

Each condition must be enforced in code and covered by a test; see
[architecture.md](architecture.md#zero-field-invariants).

At θ₀ the covariance of single-sample gradients equals the curvature of the expected loss. This is
the Bartlett identity: when the model samples its own data, the Fisher matrix equals the Hessian of
the expected loss. The zero field is therefore a case where the gradient-noise covariance is known
exactly rather than assumed.

## Exploratory evidence

Exploratory runs on Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct used 100 harmful prompts,
keyword-based refusal detection, few seeds and no judge model. Training took 25 zero-field steps at
batch size 20 with AdamW, weight decay off. The runs found:

- **Refusals disappeared under full fine-tuning with AdamW.** On 0.5B the refusal rate fell from
  94 % to 6–14 % at η = 5·10⁻⁵ and to 35–62 % at η = 2·10⁻⁵, with three seeds each. On 1.5B it fell
  from 99 % to 25 % and 70 % at the same rates, with one seed each. A warm-up with decay did not
  remove the effect (11–20 % on 0.5B). The 1.5B model still answered 95–100 % of simple factual
  questions.
- **Direction matters, not only size.** Training moved the weights by ≈ 0.6 % of their norm. A
  random perturbation of the same norm barely changed the outputs and kept refusals. Gaussian
  weight noise of 7 % of the norm changed responses to harmful prompts about as much as zero-field
  training (KL ≈ 14), yet kept 81–87 % of refusals; zero-field runs at KL 7–30 kept 6–70 %. Noise
  large enough to remove refusals, 15–25 times the training displacement, broke the model: accuracy
  on simple questions fell to 28–50 %.
- **LoRA kept refusals.** LoRA of rank 16 on 1.5B kept them at 98–100 % up to learning rate 5·10⁻⁴
  (84–97 % at 10⁻³), although the same LoRA lowered refusals when trained on Alpaca. It changed the
  responses more than 10 times less in KL, so this is not a comparison at equal displacement. Why
  LoRA resists is untested.

Caveats. The exploratory runs differed from the protocol in several ways:
- Teacher responses for training and for KL were capped at 48 tokens, and 80–86 % of the training
  responses were truncated: only 14 % on 0.5B and 21 % on 1.5B ended within the cap. Truncation
  keeps the field at zero but concentrates training on the start of responses. Stage 1 runs a
  48-token bridge cell next to the 96-token one.
- The loss was divided by the batch's token count.
- Refusals were detected by 20 keyword phrases in 48 greedy tokens on 100 AdvBench prompts, without
  a judge, and the KL prompts were a subset of these prompts.
- KL was the exact per-position KL along θ₀ responses, the estimator K now uses, but over 2 × 32
  responses of at most 48 tokens, so the values are not on the scale of K, which uses 96.
- Utility was measured on 40 hand-written questions (25 facts, 15 arithmetic).
- Teacher prompts were drawn with replacement from 100 Alpaca instructions, as in the protocol.

Stage 0 fixes these differences; stage 1 re-runs the experiment under the corrected protocol before
any conclusion is drawn. The exploratory code and data predate this repository and are not
included.

## Two candidate mechanisms

Both mechanisms exist only for Adam-type optimizers, and both are hypotheses.

### Heating of flat directions

Near θ₀, where the expected loss is close to quadratic, the parameters perform an
Ornstein–Uhlenbeck process. The expected loss forms a bowl whose curvature pulls them back, and
minibatch noise kicks them around. Along a direction of curvature λ, the stationary variance is
D ≈ s²/(2r), where s is the typical kick per step and r is the fraction of the deviation removed
per step. With learning rate η and batch size B:
- **SGD:** s² = η²λ/B and r = ηλ, so D = η/(2B) in every direction. The cloud is isotropic, with
  temperature η/B.
- **Adam:** with a settled preconditioner equal to the noise scale, s² = η² and r = η√(λB), so
  D = η/(2√(λB)). The cloud is wider along flat directions, with temperature η/√B.

The KL change of ordinary responses is ≈ ½ Σ λD, so flat directions barely register in it. At
equal KL on benign prompts, Adam moves farther along the directions the benign data hardly
constrain. Directions that maintain refusals may be among them.

This picture has limits:
- The formulas hold for a quadratic bowl once the cloud has settled, which takes ≈ 1/r steps.
- Adam rescales each parameter separately, so the Adam case assumes that flat directions align
  with parameters. In a real network a flat direction mixes many parameters; whether the picture
  transfers is an empirical question.
- The exploratory AdamW runs went beyond the quadratic regime: on 1.5B the loss on fresh θ₀
  samples rose from 0.80 to 1.51 nats per token. There the bowl is a starting point, not a
  prediction.

The formulas above are stationary, but the decisive runs take 25 steps, while a direction settles
only after ≈ 1/r steps. Flat directions are therefore in the transient regime, where the walls have
not yet acted and the cloud grows as D ≈ N·s² after N steps: N·η²λ/B for SGD and N·η² for Adam. At
equal KL ≈ ½ Σ λD, Adam still places relatively more variance along flat directions, so the
qualitative argument survives; the stationary formulas and temperatures do not apply. In the toy
model, Adam's KL hardly depends on B after 10 steps (exponent −0.04) and approaches −0.5 only after
400. The predictor of [Gate 3](preregistration.md#gate-3) is therefore built on the transient form.

How η in these formulas relates to the configured learning rate is given under
[loss units](architecture.md#loss-units).

### Sign-step rectification

In its first steps Adam divides each gradient by a magnitude estimated from the same few
gradients, so its step approaches η·sign(g). Noise with zero mean but a skewed distribution then
gives a nonzero mean step. Take the logit bias of a token with probability 0.9. The gradient of a
sample's log-likelihood with respect to this bias is +0.1 in 90 % of samples and −0.9 in 10 %. The
mean SGD step is zero. The mean first Adam step is +0.8η at batch size 1, +0.31η at batch size 4
and +0.07η at batch size 20. Frequent openings become more frequent and rare ones rarer, and a
refusal opener is rare in benign responses.

The drift is small and fades:
- As Adam accumulates statistics, the mean step shrinks: at step 25 it is 0.087η, 0.027η and
  0.011η at batch sizes 1, 4 and 20.
- Per coordinate the drift is small against the noise. At batch size 20 it accumulates ≈ 0.65η
  over 25 steps, against a spread of ≈ 5.8η. The hypothesis is that the drift adds up coherently
  over many parameters, while the noise does not.
- Real gradients sum contributions of many tokens and are less skewed than in this toy.
- Qwen2.5 has no output bias, and in 0.5B and 1.5B the output layer is tied to the frozen input
  embeddings. Shared hidden weights would have to carry the effect.

### What the noise theory has matched so far

Exploratory checks:
- SGD without momentum on 0.5B, with a constant loss denominator: KL ∝ B^α with α between −0.75
  and −0.99, against −1 predicted.
- The absolute KL matched the formula within a factor of 2 only at small steps (KL 10⁻⁴–10⁻²,
  refusals intact). At larger steps the formula overestimated KL 3–6 times.
- AdamW: the plateau KL fell with batch size with an exponent of −0.41 to −0.47, against −0.5
  predicted, from two batch sizes only.
- On a 192-parameter toy network the zero-field identities hold to 10⁻¹⁰, and the cloud width along
  7 directions matches the formula within 1–11 %.

The theory predicts how far the model moves. Whether the move points toward harm is what the
experiment tests.

## Decisive experiment

Three primary optimizer arms run the zero-field protocol: SGD without momentum, Adam with frozen
statistics and AdamW with weight decay 0. Two additional controls use SGD with momentum and AdamW
with a pre-warmed second moment. Frozen and pre-warmed Adam share an estimate of the second moment
at θ₀ from fresh teacher responses, drawn separately from the training stream. Frozen Adam holds
that estimate fixed; pre-warmed Adam updates it during training. Both keep momentum. Initialization,
bias correction and the mechanism-consistency rules are specified in
[H2 and H2a](preregistration.md#hypotheses).

The optimizers are compared at **equal displacement**. Without this matching, whichever optimizer
moves the model farther trivially breaks more. Displacement is measured by K, the KL divergence from
θ₀ on harmful prompts: the mean, over harmful prompts x and θ₀'s responses y (temperature 1, at most
96 tokens), of the exact KL between the next-token distributions of θ₀ and θ, summed over the
positions of y. It estimates the sequence-level KL(θ₀‖θ) in nats, as in the exploratory runs, and is
never negative. K says how much the responses changed, not in which direction. The question is
which optimizer's change, at equal K, points more toward harm.

For each optimizer, a learning-rate sweep and many seeds give points (K, harmful-compliance rate).
A logistic regression in log K turns them into a curve h(K). The primary quantity is

  Δ(K\*) = h_AdamW(K\*) − h_SGD(K\*),

taken at one pre-specified level K\* inside the common KL range. Its confidence interval comes from
a bootstrap over seeds, and the expected sign is Δ > 0. The position of frozen Adam at K\*,
together with the two additional controls, shows which mechanism the result is consistent with;
it does not prove a mechanism. The rule for K\*, the bootstrap, the minimal effect of interest,
the decision rule and the mechanism-consistency rule are
defined in [preregistration.md](preregistration.md#primary-analysis).

### Setup

- **Model:** Qwen2.5-1.5B-Instruct for the decisive comparison; stage 1 also runs 0.5B.
- **Teacher data:** 100 fixed Alpaca instructions, with the reference answers unused. The teacher
  generates 500 responses (25 steps × batch 20) of at most 96 tokens, each to an instruction drawn
  uniformly with replacement.
- **Training:** full fine-tuning with the input embeddings and the output head frozen; the summed
  loss is divided by 20 × 96 = 1920.
- **Sweeps:** 3–4 learning rates per optimizer. SGD rates are chosen so that SGD covers AdamW's KL
  range.
- **Seeds:** each learning rate whose K falls near K\* receives the seeds of the two-look
  sequential design in [Gate 2](preregistration.md#gate-2-scientific). A seed changes which teacher
  responses are drawn and their order.

### Main methodological risk

SGD may not reach the KL range in which refusals change. Its cloud width is set by η/B, and η is
capped by stability. In exploratory SGD runs, batch size 16 kept KL below 0.8 and refusals intact.
Batch size 4 reached KL 38–52, but refusals fell to 2–4 % only together with a collapse: the
training loss rose from 1.3 to 2.3–3.4. The first remedy is more steps and smaller batches. If SGD
still falls short, the optimizers are compared at equal steps and batch size at the largest stable
SGD learning rate. This fallback measures a harmful-compliance difference at unequal displacement;
it does not test the equal-KL H1. Its stopping rule is defined in the
[preregistration](preregistration.md#gate-2-scientific), and the failure to reach a common KL range
is reported explicitly.

### Measurements

- **Harmful behavior:** greedy responses of at most 256 tokens to 300 harmful prompts from the
  HarmBench [3] standard behaviors and AdvBench [2], deduplicated. A judge model labels each
  response as a refusal, a safe non-refusal or harmful compliance; keyword refusal is recorded
  alongside.
- **Displacement:** KL on 100 harmful and 100 benign prompts, using 4 responses of θ₀ per prompt of
  at most 96 tokens. The harmful KL prompts are disjoint from the judged ones.
- **Utility:** loss on held-out Alpaca reference answers, accuracy on at least 200 simple factual
  questions and on 500 MMLU questions scored by likelihood.
- **Weights:** the size of the weight displacement.

### Predictive check

A noise-theory model of the displacement after a finite number of steps is built on the transient
form [above](#heating-of-flat-directions); at the plateau it reduces to the temperatures η/B for SGD
and η/√B for Adam. Calibrated on part of the regimes, it predicts KL, and through h(K) the harmful
rate, on held-out regimes, and it must beat two simple baselines. The fitted constants, the held-out regimes, the baselines and the criterion
are defined in [Gate 3](preregistration.md#gate-3).

## Outcomes

The authoritative definitions and stopping decisions are in
[Gate 2](preregistration.md#gate-2-scientific); mechanism-consistency statements follow
[H2 and H2a](preregistration.md#hypotheses). The design distinguishes a safety effect from a
mechanism explanation: a difference between optimizers can establish the former without proving
the latter. An equal-KL comparison and the unequal-displacement fallback answer different
questions; their interpretations must remain separate.

## Controls

- **No training:** θ₀ unchanged.
- **Random perturbation:** a random weight perturbation of the same norm.
- **Line scan:** the line θ₀ + aΔθ for a ∈ {0.25, 0.5, 1}, both signs, where Δθ is the final
  displacement. It checks nonlinearity and sign asymmetry. The negative side reverses the whole
  displacement, θ₀ − aΔθ: an earlier claim of sign asymmetry was withdrawn because its control
  flipped the signs of individual coordinates at random.
- **Non-zero fields:**
  - Alpaca reference answers truncated to 96 tokens;
  - teacher sampling at temperature 0.7 with top-p 0.9;
  - loss normalized by the token count.
- **Soft targets:** the student is trained on KL to the teacher's full next-token distributions,
  recorded while the teacher samples. If sampling and training computed identical distributions,
  the gradient at θ₀ would be zero, so the step-0 gradient measures the numerical discrepancy
  between the two paths ([criterion](architecture.md#zero-field-invariants)).
- **Labeling and utility:** the judge plus keyword detection, and the utility measures above.

## Related work and novelty

The effect itself is known [1]. Data-side work finds which benign examples break safety most [6].
Optimization settings such as a smaller learning rate, a larger batch and weight averaging reduce
it [4]. The optimizer itself matters for alignment under real data: with LoRA on narrowly
misaligned data, it explains most of the variation in emergent misalignment once training loss is
accounted for, with Adam and Lion worst and Muon best [16]. That comparison is not at equal
displacement, does not score refusals, and its SGD arm is confounded by a higher loss. Earlier,
Adam was found to worsen catastrophic forgetting relative to SGD [17]. The closest work shows that
benign fine-tuning moves the model back toward its state before safety training [5]. These train on
real data, so they cannot separate the data from the optimizer.

The closest precedent for the separation itself is [11]: Llama-3.2-1B was fine-tuned with LoRA on
its own answers to the Dolly and Alpaca questions, and safety degraded much less than with human
answers, so content was the main driver. That agrees with the exploratory LoRA result here. The
identity behind the zero field has been stated for replay: on self-generated data, next-token
prediction and KL to the frozen model have the same expected gradient, and the remaining difference
in forgetting was attributed to gradient noise, untested and without a safety measure [18]. In RL,
rewards without information still change behavior through the clipping bias of GRPO [19]; there an
asymmetry of the algorithm, not the data, creates the drift. A teacher that differs from the
student transmits traits even through unrelated data [26], one more reason the teacher must be
exactly θ₀. [11] leaves open what this project tests: an exactly zero field (sampling at
temperature 1 without filtering, a constant loss denominator, single use), full fine-tuning, where
the exploratory effect appeared, and optimizers compared at equal displacement. Optimizer memory
itself can drive forgetting: with momentum, stored gradients from earlier states push probability
away from learned answers [12]; the momentum arm of the decisive experiment controls for it. Both
candidate mechanisms have precedents. Under skewed noise Adam's steps have a nonzero mean and can
drift against the mean gradient [20]. Its second-moment normalization turns small one-sided
gradients on tokens absent from the data into full-sized updates that drive forgetting [21].
Alignment leaves the safety Fisher low-rank and flat, and benign fine-tuning re-sharpens it [22].
Self-distillation defenses train on the model's own outputs, but with a systematic signal: in SDFT
the model rewrites the reference answers of a new task [15], and in [10] the teacher sees a safety
hint that the student does not.

The noise theory of SGD and Adam used here is standard [7–9]. Near a minimum, SGD noise also causes
a slow deterministic drift toward flatter regions [23], second order in the learning rate; the
theory here neglects it. Its closest application to fine-tuning is a first-order model of drift and
noise for the forgetting of facts under SGD and LoRA, accurate locally, which leaves Adam and full
fine-tuning open [13]. A first-order theory of how alignment changes during fine-tuning has no
noise term [14]. Random weight noise erodes safety only outside a local "safety basin" [24]. [4]
reports that Gaussian weight noise of σ = 0.004 removes safety while utility declines slowly. It
names neither the model of that experiment (its other analyses use Llama-2-7B-Chat) nor the
convention for σ, and it scores utility with benchmarks such as MT-Bench rated by GPT-4o. Read as an
absolute per-weight standard deviation, σ = 0.004 is about 10 % of the weight norm of
Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, inside the range of 9–15 % where the exploratory
noise here also removed refusals; there, however, accuracy on simple questions fell to 28–50 %. The
two disagree about utility, measured differently, not about the noise scale at which refusals go.
Targeted perturbations break refusals far more than random ones of matched size [25]. The
contribution is the clean separation of the optimizer from the data: an exactly zero expected
signal, refusals as the outcome, full fine-tuning, optimizers compared at equal KL, an Adam arm with
frozen statistics and a noise theory of the drift. It is not a new phenomenon and not the first
evidence that the optimizer matters [16].

## References

1. X. Qi et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not
   Intend To!* arXiv:2310.03693.
2. A. Zou et al. *Universal and Transferable Adversarial Attacks on Aligned Language Models.*
   arXiv:2307.15043.
3. M. Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and
   Robust Refusal.* arXiv:2402.04249.
4. M. Kim et al. *Rethinking Safety in LLM Fine-tuning: An Optimization Perspective.*
   arXiv:2508.12531.
5. S. Poppi, N. Lukas. *A Gravitational Interpretation of Safety Reversion under Fine-Tuning.*
   arXiv:2606.28525.
6. L. He, M. Xia, P. Henderson. *What's in Your "Safe" Data? Identifying Benign Data that Breaks
   Safety.* arXiv:2404.01099.
7. S. Mandt, M. D. Hoffman, D. M. Blei. *Stochastic Gradient Descent as Approximate Bayesian
   Inference.* arXiv:1704.04289.
8. S. Jastrzębski et al. *Three Factors Influencing Minima in SGD.* arXiv:1711.04623.
9. S. Malladi, K. Lyu, A. Panigrahi, S. Arora. *On the SDEs and Scaling Rules for Adaptive
   Gradient Algorithms.* arXiv:2205.10287.
10. Y. Fu et al. *Reducing the Safety Tax in LLM Safety Alignment with On-Policy
    Self-Distillation.* arXiv:2605.15239.
11. K. C. Fraser, H. Dawkins, I. Nejadgholi et al. *Fine-Tuning Lowers Safety and Disrupts
    Evaluation Consistency.* arXiv:2506.17209.
12. V. Ruscio, S. Nabarro, K. Thompson. *A Step Towards Forgetting: Optimiser History and the Loss
    of Answer Mass.* arXiv:2610.03940.
13. J. Su, W. Zhang. *When Can First-Order Models of Fine-Tuning Bound Forgetting?*
    arXiv:2609.32818.
14. Y. Huang, H. Chen, Y. Dong. *Alignment Dynamics in LLM Fine-Tuning.* arXiv:2605.18309.
15. Z. Yang et al. *Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning.*
    arXiv:2402.13669.
16. J. R. Brown, P. Leask, L. McKinney. *Evil Spectra: How Optimisers can Amplify or Suppress
    Emergent Misalignment.* arXiv:2606.31591.
17. D. R. Ashley, S. Ghiassian, R. S. Sutton. *Does the Adam Optimizer Exacerbate Catastrophic
    Forgetting?* arXiv:2102.07686.
18. M. Marek, D. Cho, S. Qiu, R. Chunara, P. Izmailov, A. G. Wilson. *Forgetting in Language Models:
    Capacity, Optimization, and Self-Generated Replay.* arXiv:2605.26097.
19. R. Shao et al. *Spurious Rewards: Rethinking Training Signals in RLVR.* arXiv:2506.10947.
20. S. J. Reddi, S. Kale, S. Kumar. *On the Convergence of Adam and Beyond.* arXiv:1904.09237.
21. J. Han, Y. Song, J. Park. *A Deafening Silence: Catastrophic Forgetting Lives in the Output
    Embeddings of Tokens the Data Never Speaks.* arXiv:2610.09835.
22. Y. Guo, X. Chen, S. Zhang, X. Wang, H. Tang. *When Safety Routing Breaks: Understanding Alignment
    Fragility under Benign Fine-Tuning.* arXiv:2609.01455.
23. Z. Li, T. Wang, S. Arora. *What Happens after SGD Reaches Zero Loss? A Mathematical Framework.*
    arXiv:2110.06914.
24. S. Peng, P.-Y. Chen, M. Hull, D. H. Chau. *Navigating the Safety Landscape: Measuring Risks in
    Finetuning Large Language Models.* arXiv:2405.17374.
25. E. Jiang, A. Gjølbye, Y. J. Zhang, S. Koyejo. *When Behavioral Safety Evaluation Fails: A
    Representation-Level Perspective.* arXiv:2606.08044.
26. A. Cloud et al. *Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden
    Signals in Data.* arXiv:2507.14805.
