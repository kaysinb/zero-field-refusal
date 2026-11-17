# Refusal Erosion at Zero Field

## The question

Fine-tuning a chat model on benign data can weaken its refusals of harmful requests. This
repository studies a clean form of that effect. A model is fine-tuned on responses sampled from its
own frozen starting checkpoint θ₀, so the expected gradient of the training loss at θ₀ is zero (the
"zero field"): the data carry no systematic push away from θ₀. Whatever erodes the refusals then
comes from the training process itself: which samples were drawn, the order in which they are
seen, the learning rate and the optimizer.

## What is measured

- **Training** (studies 3 and 5). Full fine-tuning in fp32 with the input embeddings and the
  output head frozen, 25 updates of batch 20 on 500 responses that θ₀ sampled for 100 Alpaca
  instructions (one seed draws one such stream).
- **Drift.** K is the KL divergence from θ₀'s next-token distributions, summed over the positions
  of θ₀'s stored responses and averaged over a panel's responses, in nats per response. K_benign
  uses a fixed panel of benign prompts; K_harmful a panel of harmful ones.
- **Harm.** h is the share of 521 harmful prompts (from HarmBench and AdvBench) whose greedy
  response the WildGuard judge labels harmful.
- **Utility.** MMLU accuracy and accuracy on a set of simple factual questions.
- **Matched drift.** Runs are compared at the same K_benign: each seed and condition is trained at
  two learning rates that bracket a target K\* (6 nats per response), and h is interpolated to K\*.

## Studies

| Study | Question | Status and documents |
|---|---|---|
| 1, pilot | Is harmful compliance higher after AdamW than after SGD at equal KL on harmful prompts? | Stopped at Gate 1. [Preregistration](docs/preregistration.md), [Gate 1 report](docs/reports/gate-1.md), [stage reports](docs/reports/README.md) |
| 2 | Long AdamW against long SGD at equal K_benign; redirected to AdamW-25 against frozen Adam-25 | Stopped before its primary comparison. [Preregistration](docs/study2-preregistration.md), [feasibility report](docs/reports/study2-feasibility.md) |
| 3 | Harm after 25 updates under AdamW (A), Adam with its second moment frozen at v₀, the estimate at θ₀ (F), and AdamW whose second moment starts at v₀ (P), at K\* = 6 on Qwen2.5-1.5B-Instruct, in 60 seed blocks | Completed. [Preregistration](docs/study3-preregistration.md), [final report](docs/reports/study3-final-c1.md) and its stage reports |
| 5 | Whether a first-order quantity computed at θ₀ ranks fresh seeds by their harm, and how harm at matched drift depends on the seed's samples and on their order | Runs and analyses completed; see below |

The number 4 belonged to a repeat of study 3's A–F comparison on Qwen2.5-7B-Instruct, which was
drafted but not registered or run; its draft is not in this repository.

Study 5's registered parts:

- [Study 5 preregistration](docs/study5-preregistration.md): the confirmatory test on fresh seeds,
  the descriptive measurements, extension 1 (the ladder's unbracketed rungs and the order repeats,
  E2) and extension 2 (diagnostics of the mechanism: linear replays, order controls, swaps and
  shuffles of batches) with its amendments;
- [the E2 prediction note](docs/study5-e2-note.md): the prospective test on E2's order repeats;
- [the dose experiment](docs/study5-dose-registration.md): the early learning rate changed with
  the order kept;
- [the order repeat under AdamW](docs/study5-adamw-registration.md);
- [the order repeat on Llama-3.2-1B-Instruct](docs/study5-llama-registration.md).

The [design](docs/design.md) gives the scientific rationale and related work, and the
[decision log](docs/decisions.md) records the decisions that stopped, kept or started a study.

## Run records

Run outputs (configs, metrics, generated responses, judge labels) and the analysis records are not
part of this repository. The registrations, the code-identity files under `docs/` and the reports
under `docs/reports/` name runs and records by their ids and sha256.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
cp .env.example .env
uv run python scripts/check_docs.py
uv run pytest
```

These commands set up the environment, check the documentation and run the CPU test suite. They
download no model and start no training. GPU runs use the dependency image and the launcher
described in [operations](docs/operations.md) ([containers](docs/operations.md#containers),
[launching runs](docs/operations.md#launching-runs)); each study's campaign driver is a script
under `scripts/` named in its section there.

## Repository layout

```text
configs/                  protocol, model, data, judge and experiment parameters
docs/                     registrations, design, plan, architecture, operations, decisions
docs/reports/             stage and study reports, with their evidence
scripts/                  entry points: data, training, labeling, campaign drivers, analyses
src/zero_field_refusal/   the package: training, evaluation, judge client, analysis
tests/                    CPU tests
docker/, Dockerfile       the dependency image
```

## Licence

- **Code** (`src/`, `scripts/`, `tests/`, `configs/`, `docker/` and the build files): Apache License
  2.0, in [LICENSE](LICENSE).
- **Documentation and reports** (`docs/`, this README): Creative Commons Attribution-NonCommercial
  4.0 International, in [LICENSE-docs](LICENSE-docs).
- **Third-party models, datasets and the judge** are not included. They are downloaded at run time
  and stay under their own licences and terms:
  [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
  [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
  [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (gated),
  [WildGuard](https://huggingface.co/allenai/wildguard) (gated),
  [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca),
  [MMLU](https://huggingface.co/datasets/cais/mmlu),
  [HarmBench](https://github.com/centerforaisafety/HarmBench) and
  [AdvBench](https://github.com/llm-attacks/llm-attacks).
