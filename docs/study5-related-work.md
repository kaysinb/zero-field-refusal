# Study 5: related work

Status: draft.

## What study 5 asks

Fine-tuning Qwen2.5-1.5B-Instruct on its own unfiltered samples (a zero field: no harmful
content, no selection, zero expected gradient at θ₀) for 25 steps of Adam with its second moment
frozen at v₀ produces very different harm in different seeds at the same benign drift K_benign = K\*.
In study 3, block means ranged from 1.0 % to 40.4 % judge-rated harm, and the block main effect was
93.5 % of the sum of squares. Study 5 asks whether one first-order quantity, computed at θ₀ before
any training from a seed's own streams, ranks fresh seeds by their harm after training:
p_s = −η Σ_t c_t ⟨D_s g_t(θ₀), ∇r(θ₀)⟩. Here g_t are the seed's batch gradients at θ₀, D_s is the
frozen Adam preconditioner from v₀, c_t are the momentum weights of the 25-step update and r is a
log-likelihood contrast between short refusal and compliance continuations on 100 harmful prompts.
The p_s of 40 fresh seeds are recorded and committed before their training runs. The test is a
one-sided Spearman correlation with judge-rated harm. Descriptive parts add a retrospective check on
study 3's 60 seeds, line scans along ±Δθ, crossed controls (order, preconditioner) and a ladder of
displacements matched in K.

## Related work

### Benign fine-tuning erodes safety

Qi et al. [1] showed that fine-tuning aligned models erodes safety even on benign data. On 330
harmful instructions (HEx-PHI, 11 categories), one epoch on Alpaca raised the harmfulness rate of
Llama-2-7b-Chat from 0.3 % to 16.1 %, on Dolly from 0.6 % to 12.1 %, and on LLaVA-Instruct from 0 %
to 18.8 %; for GPT-3.5 Turbo, Alpaca raised it from 5.5 % to 31.8 %. Larger learning rates and
smaller batches generally increased the degradation. Ten harmful examples were enough to break
GPT-3.5 Turbo's guardrails. A follow-up [2] argued that alignment is "shallow": the KL divergence
between aligned and base models is concentrated on the first few response tokens, and fine-tuning
gradients are largest there.

Several studies examine which data or settings drive the effect. Kim et al. [3] report that tuning
learning rate, batch size and steps cuts unsafe responses from 16 % to about 5 % (keyword matching)
and propose an exponential moving average of the weights. Hsiung et al. [4] find that guardrails
erode more when the downstream data are similar to the alignment data. Fraser et al. [5] fine-tuned
Llama-3.2-1B with LoRA on its own answers to the Dolly and Alpaca questions and found much lower
harm than with the human answers. They attribute the degradation to the newness of the content.
Peng et al. [6] map a "safety basin": random weight perturbations keep safety near the aligned
level within a local region and break it beyond. They also plot safety along the line from the
initial to the fine-tuned weights. Guo et al. [7] relate the fragility to a low-rank safety Fisher
and an output-routing pathway that benign fine-tuning on 100 examples re-sharpens. Poppi and
Lukas [8] find an activation-space "reversion direction" toward a helpful-only state; alignment
with it and harmfulness rise together (Spearman 0.958). Springer et al. [9] give a geometric
condition for alignment collapse in which the drift into a low-rank sensitive subspace is second
order; their first-order overlap score is computed from the weight change after fine-tuning.

Arditi et al. [10] show that refusal is mediated by one residual-stream direction in 13 chat
models. They score refusal by the log-odds of a set of refusal tokens at the first generated
position. Study 5's surrogate r is of the same kind: a teacher-forced log-likelihood of refusal
continuations against compliance continuations at the start of the response.

### Predicting or attributing harm from fine-tuning data

**He, Xia and Henderson [11]** is the closest precedent for the predictor. They score each benign
example z by ⟨g(z), g_harm⟩ − ⟨g(z), g_safe⟩, where g(z) = ∇θ ℓ(z; θ) and g_harm, g_safe are mean
gradients over two anchor sets: 10 harmful instructions (illegal activities, from the Pure-bad set)
with harmful completions, and the same 10 instructions with safe responses of Llama-2-7B-Chat. The
loss is taken over the first 10 response tokens. The score is motivated by a one-step first-order
Taylor expansion at the start of training. Gradient features are normalized before the inner
product, following LESS, because unnormalized gradients favored short sequences. Selection uses the
aligned model Llama-2-7B-Chat; fine-tuning is full, on 100 examples, batch size 20, 5 epochs, rate
5·10⁻⁵, five seeds. On the 520 AdvBench instructions with a GPT-3.5 judge (Table 1, mean (std)):

| Pool | Random 100 | Representation top 100 | Gradient top 100 |
|---|---|---|---|
| Alpaca | 13.0 (8.6) | 71.0 (2.0) | 66.5 (5.5) |
| Dolly | 8.2 (5.0) | 31.8 (6.0) | 53.3 (17.4) |

The random baseline averages over 5 random selections. Both anchors matter (Table 2): with harmful
anchors alone, the bottom-ranked Alpaca examples still gave 53.1 % ASR; with both, the top gave
66.5 % and the bottom 3.8 %. The selected examples were often lists, bullet points or math
questions. A representation-similarity variant selects by cosine similarity of final hidden states.

LESS [12] selects instruction-tuning data by the cosine between a validation gradient and the
Adam update direction of a training example, Γ(z, θ) = m̂/(√v̂ + ε), summed over warm-up
checkpoints weighted by the learning rate. It uses LoRA and random projection to 8192 dimensions,
and normalizes because raw inner products favor short sequences. Training on the 5 % it selects
often matches or beats the full set. TracIn [13] estimates influence as the change in a test loss
traced over the training steps, with checkpoint gradient inner products weighted by the learning
rate. Influence functions have been scaled to LLMs with EK-FAC [14]. TrackStar [15] divides
per-parameter gradients by the square root of the optimizer's second-moment estimate before
projection and unit-normalizes them. Datamodels [16] predict the output of a model trained on a
subset of the training set from which examples it contains; TRAK [17] evaluates attribution by the
linear datamodeling score, the Spearman correlation between predicted and actual outputs over
random training subsets, averaging five retrained models per subset.

Several recent works turn these tools on safety. Guan et al. [18] select 100 outlier benign
examples by a length-corrected self-influence score (no anchors) and report harmfulness on HEx-PHI
(1–5 scale, Llama-2-7B-Chat, Alpaca) of 3.47 ± 0.26 against 1.13 ± 0.08 for random selection and
3.52 ± 0.08 for He et al.'s bidirectional anchors. Wang et al. [19] score samples by projecting a
one-step LoRA update onto "danger" and "safety" parameter directions taken from separate
fine-tuning runs, and test the ranking by fine-tuning on five risk-ranked subsets of 1000 examples
(monotone in 10 of 12 settings). Reddy et al. [20] attribute safety degradation in Tulu 3 to
training documents by projected gradient similarity to unsafe responses and remove the most aligned
20 %. Bach et al. [21] filter high-gradient samples to preserve safety in continual fine-tuning.

### Variance across seeds and random subsets

Variance of fine-tuning outcomes over seeds is well known. Dodge et al. [22] ran thousands of BERT
fine-tuning trials varying only the seed and found that weight initialization and data order
contribute comparably to the variance. Bui et al. [23] report significant seed variance at the
level of aggregate metrics and of individual predictions. For safety, He et al.'s random 100-example
subsets gave 13.0 % ± 8.6 GPT-judged ASR on Alpaca [11]. Fraser et al. [5] report that at
temperature 0, five seeds of the same Llama-3.2-1B fine-tune on Dolly ranged from 0.73 to 0.82 in
harmfulness score in the most extreme case. For data attribution, Nguyen et al. [24] find that the
effect of a single training example is often hidden by noise from initialization and batch
composition. Mlodozeniec et al. [25] define distributional attribution, which predicts how the
distribution of outputs over training runs depends on the data, and show that influence functions
arise as a limit of unrolled differentiation.

### First-order and linearized accounts of fine-tuning

Malladi et al. [26] study when fine-tuning of language models follows kernel (NTK) dynamics and
extend the formalism to Adam. Ren and Sutherland [27] decompose the one-step change of a response's
log-likelihood under a gradient step on another example into an eNTK term and a remainder of order
η², and use it to explain effects of instruction and preference tuning; their update is plain
gradient descent. Su and Zhang [28] measure a first-order response model at the start of LoRA
fine-tuning with SGD; it predicts per-fact margin changes with correlation 0.974–0.998, but not
forgetting, which needs moves outside the validated region, and on violated facts the spread of
margins across test runs had a median 14.6 times the predicted change. Huang et al. [29] derive a
closed-form update of an alignment score during fine-tuning, split into a rebound force and a
driving force set by the data. Among the safety studies above, He et al. [11] use the first-order
expansion at the start of training explicitly, and LESS [12] includes Adam's preconditioner in the
training-side feature.

### Prospective and preregistered predictions

Su and Zhang [28] report two preregistered confirmatory studies on 43 new conditions. Williams et
al. [30] predicted post-deployment misbehavior rates of a production model from simulated
deployment, with registered, outcome-blinded predictions for one prospective deployment and three
retrospective ones. Jones et al. [31] forecast the risk of rare behaviors at larger query volumes
from small samples. Søgaard et al. [32] discuss the case for and against preregistration in NLP.
We found no prior study that records a first-order safety prediction for fine-tuning runs before
the runs are made.

## How study 5 differs

- **What is predicted.** Prior selection work [11, 18–21] ranks examples or subsets within a pool
  that differs in content, and tests the ranking by fine-tuning on the extremes (top versus random
  or bottom). Study 5 ranks unselected random streams drawn from one distribution, the model's own
  samples. The streams differ only by sampling noise, their order and the preconditioner's stream.
  The outcome is the spread among them, at a matched benign shift K_benign = K\*, not the gap
  between selected and random data. He et al.'s random baseline [11] and Fraser et al.'s seeds [5]
  show such spread exists; we found no work that tries to predict it per seed.
- **What step is linearized.** He et al. [11] use the raw, normalized gradient of one example. LESS
  [12] uses a normalized Adam direction with LoRA and projection. Study 5 uses the full-parameter
  update of the arm that is trained: F's frozen preconditioner D from v₀, its momentum and bias
  correction weights c_t, the learning rate, and the 25 batch gradients of the actual stream, summed
  without normalization. The quantity is the first-order change of r under the real step, so its
  size, not only its direction, enters. It is still a frozen-gradient approximation: all g_t are
  taken at θ₀.
- **When the prediction is fixed.** The scores of the 40 fresh seeds are recorded, copied and
  committed before any of their runs; the test, its direction and its exclusions are registered in
  advance. A retrospective association on 60 earlier seeds is reported but not tested.
- **What else is measured.** Line scans at ±a·Δθ separate the odd (first-order) and even parts of
  the harm change around θ₀. The ladder compares, at matched K, a Gaussian displacement with
  covariance D, the linear response without D and with D. The crossed controls swap order and
  preconditioner within seed. These are descriptive, on six seeds for the controls and ladder.

These are differences of setting and design, not of method. In scale study 5 is narrower than most
prior work: one 1.5B model, one optimizer arm, 25 steps, one judge and a fixed set of 521 prompts.

## What in study 5 is not new

- That benign fine-tuning erodes safety is known [1, 3–5], as is its dependence on learning rate
  and batch size [1, 3].
- The predictor's form is essentially He et al.'s bidirectional gradient score [11]. Their anchor
  difference g_harm − g_safe is the gradient of a safe-minus-harmful log-likelihood contrast, the
  same kind of contrast as r. Up to sign and scale, p_s is that score without normalization, in the
  metric D, weighted by c_t and summed over a stream's 25 batches, with short fixed refusal and
  compliance continuations on 100 prompts as anchors.
- Its first-order justification, at the start of training, is theirs [11] and TracIn's [13].
  Preconditioning the training gradient by Adam's second moment is LESS's [12] (and is used in
  TrackStar [15]).
- Scoring refusal by the likelihood of refusal tokens at the start of the response follows [10],
  and the focus on the first tokens matches [2].
- Rank-correlating predicted with actual outcomes over random training sets is how datamodels and
  TRAK are evaluated [16, 17]. Study 5's test is that idea for one outcome and one predictor.
- Seed variance of fine-tuning outcomes is known [22, 23], including in safety [5, 11], and its
  effect on attribution has been studied [24, 25].
- First-order response models measured at the start of fine-tuning, with preregistered
  confirmation, exist for forgetting [28]. Linearized accounts of fine-tuning, including Adam, exist
  [26, 27].
- Line scans along the fine-tuning direction resemble the one-dimensional safety landscapes of
  Peng et al. [6].

What study 5 adds, if its test passes, is narrower: evidence that this known first-order score,
with the real optimizer step, ranks unselected random streams of the model's own samples by
judge-rated harm at matched K, on predictions recorded before the runs.

## References

1. X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, P. Henderson. *Fine-tuning Aligned
   Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR 2024.
   arXiv:2310.03693.
2. X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, P. Henderson. *Safety Alignment
   Should Be Made More Than Just a Few Tokens Deep.* ICLR 2025. arXiv:2406.05946.
3. M. Kim, J. M. Kwak, L. Alssum, B. Ghanem, P. Torr, D. Krueger, F. Barez, A. Bibi. *Rethinking
   Safety in LLM Fine-tuning: An Optimization Perspective.* arXiv:2508.12531, 2025.
4. L. Hsiung, T. Pang et al. *Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity
   Analysis Between Alignment and Fine-tuning Datasets.* arXiv:2506.05346, 2025 (listed in the
   ACL 2026 Anthology; not checked further).
5. K. C. Fraser, H. Dawkins, I. Nejadgholi, S. Kiritchenko. *Fine-Tuning Lowers Safety and Disrupts
   Evaluation Consistency.* arXiv:2506.17209, 2025.
6. S. Peng, P.-Y. Chen, M. Hull, D. H. Chau. *Navigating the Safety Landscape: Measuring Risks in
   Finetuning Large Language Models.* NeurIPS 2024. arXiv:2405.17374.
7. Y. Guo, X. Chen, S. Zhang, X. Wang, H. Tang. *When Safety Routing Breaks: Understanding Alignment
   Fragility under Benign Fine-Tuning.* Findings of EMNLP 2026. arXiv:2609.01455.
8. S. Poppi, N. Lukas. *A Gravitational Interpretation of Safety Reversion under Fine-Tuning.*
   arXiv:2606.28525, 2026.
9. M. Springer, C. P. Lee, B. Metevier, J. Castleman, B. Turbal, H. Jung, Z. Shen, A. Korolova.
   *The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety.* arXiv:2602.15799, 2026.
10. A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, N. Nanda. *Refusal in
    Language Models Is Mediated by a Single Direction.* NeurIPS 2024. arXiv:2406.11717.
11. L. He, M. Xia, P. Henderson. *What is in Your Safe Data? Identifying Benign Data that Breaks
    Safety.* COLM 2024. arXiv:2404.01099.
12. M. Xia, S. Malladi, S. Gururangan, S. Arora, D. Chen. *LESS: Selecting Influential Data for
    Targeted Instruction Tuning.* ICML 2024. arXiv:2402.04333.
13. G. Pruthi, F. Liu, M. Sundararajan, S. Kale. *Estimating Training Data Influence by Tracing
    Gradient Descent.* NeurIPS 2020. arXiv:2002.08484.
14. R. Grosse, J. Bae, C. Anil, N. Elhage, A. Tamkin et al. *Studying Large Language Model
    Generalization with Influence Functions.* arXiv:2308.03296, 2023.
15. T. A. Chang, D. Rajagopal, T. Bolukbasi, L. Dixon, I. Tenney. *Scalable Influence and Fact
    Tracing for Large Language Model Pretraining.* arXiv:2410.17413, 2024.
16. A. Ilyas, S. M. Park, L. Engstrom, G. Leclerc, A. Madry. *Datamodels: Predicting Predictions
    from Training Data.* ICML 2022. arXiv:2202.00622.
17. S. M. Park, K. Georgiev, A. Ilyas, G. Leclerc, A. Madry. *TRAK: Attributing Model Behavior at
    Scale.* ICML 2023. arXiv:2303.14186.
18. Z. Guan, M. Hu, R. Zhu, S. Li, A. Vullikanti. *Benign Samples Matter! Fine-tuning On Outlier
    Benign Samples Severely Breaks Safety.* arXiv:2505.06843, 2025.
19. X. Wang, Y. Zhang, Y. Liu, X. Yang, Z. Wang, S. Feng, D. Wang. *From Parameter Dynamics to Risk
    Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning.* arXiv:2605.04572,
    2026.
20. S. Reddy, A. Abbas, A. Suri, A. Morcos, D. Schwab, P. Maini. *Follow the Gradient: Tracing
    Safety Degradation Back to Training Data.* arXiv:2610.37481, 2026.
21. T. Bach, D. Nguyen, T. M. Le, T. Tran. *Continual Safety Alignment via Gradient-Based Sample
    Selection.* Findings of ACL 2026. arXiv:2604.17215.
22. J. Dodge, G. Ilharco, R. Schwartz, A. Farhadi, H. Hajishirzi, N. Smith. *Fine-Tuning Pretrained
    Language Models: Weight Initializations, Data Orders, and Early Stopping.* arXiv:2002.06305,
    2020.
23. N. Bui, G. Savova, L. Wang. *Assessing the Macro and Micro Effects of Random Seeds on
    Fine-Tuning Large Language Models.* IJCNLP 2025. arXiv:2503.07329.
24. E. Nguyen, M. Seo, S. J. Oh. *A Bayesian Approach To Analysing Training Data Attribution In Deep
    Learning.* arXiv:2305.19765, 2023.
25. B. Mlodozeniec et al. *Distributional Training Data Attribution: What do Influence Functions
    Sample?* NeurIPS 2025. arXiv:2506.12965.
26. S. Malladi, A. Wettig, D. Yu, D. Chen, S. Arora. *A Kernel-Based View of Language Model
    Fine-Tuning.* ICML 2023. arXiv:2210.05643.
27. Y. Ren, D. J. Sutherland. *Learning Dynamics of LLM Finetuning.* ICLR 2025. arXiv:2407.10490.
28. J. Su, W. Zhang. *When Can First-Order Models of Fine-Tuning Bound Forgetting?*
    arXiv:2609.32818, 2026.
29. Y. Huang, H. Chen, Y. Dong. *Alignment Dynamics in LLM Fine-Tuning.* arXiv:2605.18309, 2026.
30. M. Williams, H. Sheahan, C. Raymond, T. Korbak et al. *Predicting LLM Safety Before Release by
    Simulating Deployment.* arXiv:2607.07184, 2026.
31. E. Jones et al. *Forecasting Rare Language Model Behaviors.* arXiv:2502.16797, 2025.
32. A. Søgaard, D. Hershcovich, M. de Lhoneux. *A Two-Sided Discussion of Preregistration of NLP
    Research.* EACL 2023. arXiv:2302.10086.

Verification note: the numbers above were read from the cited papers' arXiv pages or full text.
Venues were taken from the arXiv comments field or the proceedings listing. Items read only from
the abstract: [4] (beyond the method), [7], [8], [14], [21], [22], [23], [24], [25], [26], [29],
[30], [31]. The ACL 2026 venue of [4] was seen only in an Anthology preview listing.
