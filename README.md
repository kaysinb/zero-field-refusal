# Refusal Collapse at Zero Field

How much does the optimizer alone erode safety during fine-tuning?

## The question

Fine-tuning on benign data can weaken a model's safety behavior. This project asks how much comes
from the training process when the expected gradient at initialization is exactly zero.
The student learns from fresh responses sampled by its own frozen starting checkpoint.

Self-distillation has prior precedent. This study combines a strictly zero-field NLL protocol,
full fine-tuning with the embeddings and output head frozen, and comparisons of optimizers at
matched KL displacement. The scientific design, exploratory evidence and limits of the mechanism
claims are in [design.md](docs/design.md), including [related work](docs/design.md#related-work-and-novelty).

## Status

Study 1, the pilot, ran stages 0 and 1 and stopped at Gate 1
([decision](docs/decisions.md#study-decisions), [short report](docs/reports/gate-1.md)). Its
preregistration stays in [preregistration.md](docs/preregistration.md).

Study 2 ([preregistration](docs/study2-preregistration.md)) compared optimizers at equal KL on
benign instructions, redirected after Stage A to AdamW-25 against frozen Adam-25. It stopped on
2026-10-25 at a technical numerical-qualification block, before its primary comparison
([feasibility report](docs/reports/study2-feasibility.md)).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
cp .env.example .env
uv run python scripts/check_docs.py
uv run pytest
```

These commands set up the environment and run documentation checks and CPU unit tests.
They do not download model weights or launch training. Environment and GPU setup are documented
in [operations.md](docs/operations.md); experimental entry points are still planned.

## Documentation

| Document | Purpose |
|---|---|
| [Design](docs/design.md) | Scientific rationale, evidence, controls and references |
| [Preregistration](docs/preregistration.md) | Authoritative hypotheses, analysis and decision rules of study 1 (the pilot) |
| [Study 2 preregistration](docs/study2-preregistration.md) | Study 2 question, arms, stages, gates and analysis |
| [Plan](docs/plan.md) | Stages, failure actions and compute |
| [Architecture](docs/architecture.md) | Engineering design and protocol implementation |
| [Operations](docs/operations.md) | Environment, runs, storage and release procedures |
| [Decisions](docs/decisions.md) and [reports](docs/reports/README.md) | Recorded decisions and their evidence |
| [Documentation index](docs/README.md) | Document responsibilities and maintenance |

Container build and runtime instructions are in [operations](docs/operations.md#containers).

## Repository layout

```text
configs/                  protocol and experiment parameters
docs/                     scientific and engineering documentation
scripts/                  documentation checker; experimental entry points planned
src/zero_field_refusal/    implemented infrastructure; pipeline modules planned
tests/                    CPU unit tests
```

Change the document that owns a rule first, then review its dependent descriptions and configs.
Run the documentation checker after edits; numerical gate criteria belong in preregistration.
