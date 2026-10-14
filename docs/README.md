# Documentation

The repository README is the overview. The document and topic tables below are generated from
[map.yaml](map.yaml); edit the map, not the generated region. A source owns the definition of a
topic. Review references identify related explanations, plans and configurations, not additional
authoritative copies.

## Documentation map

<!-- BEGIN GENERATED DOC MAP -->

| Document | Responsibility |
|---|---|
| [README.md](../README.md) | Repository overview and working quick start |
| [docs/README.md](README.md) | Generated document/topic index and maintenance workflow |
| [docs/design.md](design.md) | Scientific rationale, evidence, controls and references |
| [docs/preregistration.md](preregistration.md) | Authoritative hypotheses, analysis, gate criteria and amendments of study 1 (the pilot) |
| [docs/study2-preregistration.md](study2-preregistration.md) | Study 2 question, arms, stages, gates, analysis and amendments |
| [docs/plan.md](plan.md) | Stages, progression, failure paths and hardware |
| [docs/architecture.md](architecture.md) | Engineering design, implementation invariants and run identity |
| [docs/operations.md](operations.md) | Environment, configuration, execution, storage and release procedures |
| [docs/decisions.md](decisions.md) | Historical gate decisions and protocol-version changes |
| [docs/reports/README.md](reports/README.md) | Conventions for historical reports and their evidence |

| Topic | Authoritative source | Review after changes |
|---|---|---|
| Research question and prior-work positioning | [docs/design.md#related-work-and-novelty](design.md#related-work-and-novelty) | [README.md#the-question](../README.md#the-question); [docs/design.md#the-zero-field-experiment](design.md#the-zero-field-experiment); [docs/design.md#references](design.md#references) |
| Binding zero-field protocol and its implementation | [docs/preregistration.md#fixed-protocol](preregistration.md#fixed-protocol) | [docs/design.md#the-zero-field-experiment](design.md#the-zero-field-experiment); [docs/architecture.md#zero-field-invariants](architecture.md#zero-field-invariants); [docs/architecture.md#generation-settings](architecture.md#generation-settings); [docs/architecture.md#precision](architecture.md#precision); [docs/architecture.md#loss-units](architecture.md#loss-units); [configs/protocol/zero_field.yaml](../configs/protocol/zero_field.yaml); [configs/model/qwen2.5-0.5b-instruct.yaml](../configs/model/qwen2.5-0.5b-instruct.yaml); [configs/model/qwen2.5-1.5b-instruct.yaml](../configs/model/qwen2.5-1.5b-instruct.yaml); [configs/control/token_count.yaml](../configs/control/token_count.yaml); [configs/control/sharpened_sampling.yaml](../configs/control/sharpened_sampling.yaml); [configs/control/alpaca_reference.yaml](../configs/control/alpaca_reference.yaml); [configs/control/soft_targets.yaml](../configs/control/soft_targets.yaml); [src/zero_field_refusal/protocol.py](../src/zero_field_refusal/protocol.py) |
| Optimizer hypotheses, control arms and state initialization | [docs/preregistration.md#hypotheses](preregistration.md#hypotheses) | [docs/design.md#two-candidate-mechanisms](design.md#two-candidate-mechanisms); [docs/design.md#decisive-experiment](design.md#decisive-experiment); [docs/design.md#outcomes](design.md#outcomes); [docs/plan.md#stage-2--decisive-comparison-on-15b](plan.md#stage-2--decisive-comparison-on-15b); [docs/architecture.md#data-flow](architecture.md#data-flow); [docs/architecture.md#modules](architecture.md#modules); [configs/optimizer/sgd.yaml](../configs/optimizer/sgd.yaml); [configs/optimizer/sgd_momentum.yaml](../configs/optimizer/sgd_momentum.yaml); [configs/optimizer/adamw.yaml](../configs/optimizer/adamw.yaml); [configs/optimizer/adam_frozen.yaml](../configs/optimizer/adam_frozen.yaml); [configs/optimizer/adamw_prewarmed.yaml](../configs/optimizer/adamw_prewarmed.yaml) |
| Harmful compliance, KL matching and the primary contrast | [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis) | [docs/design.md#decisive-experiment](design.md#decisive-experiment); [docs/design.md#measurements](design.md#measurements); [docs/plan.md#stage-2--decisive-comparison-on-15b](plan.md#stage-2--decisive-comparison-on-15b); [docs/architecture.md#data-flow](architecture.md#data-flow); [configs/data/default.yaml](../configs/data/default.yaml); [configs/eval/default.yaml](../configs/eval/default.yaml); [configs/judge/default.yaml](../configs/judge/default.yaml) |
| Corrected-pilot gate, utility and judge calibration | [docs/preregistration.md#gate-1-technical](preregistration.md#gate-1-technical) | [docs/preregistration.md#alternative-explanations](preregistration.md#alternative-explanations); [docs/plan.md#stage-1--corrected-protocol-on-05b-and-15b](plan.md#stage-1--corrected-protocol-on-05b-and-15b); [docs/design.md#controls](design.md#controls); [docs/design.md#main-methodological-risk](design.md#main-methodological-risk); [configs/eval/default.yaml](../configs/eval/default.yaml); [configs/judge/default.yaml](../configs/judge/default.yaml); [docs/decisions.md#gate-decisions](decisions.md#gate-decisions) |
| Sequential decisions, local equivalence and unequal-KL fallback | [docs/preregistration.md#gate-2-scientific](preregistration.md#gate-2-scientific) | [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis); [docs/preregistration.md#open-questions](preregistration.md#open-questions); [docs/plan.md#stage-1--corrected-protocol-on-05b-and-15b](plan.md#stage-1--corrected-protocol-on-05b-and-15b); [docs/plan.md#stage-2--decisive-comparison-on-15b](plan.md#stage-2--decisive-comparison-on-15b); [docs/design.md#main-methodological-risk](design.md#main-methodological-risk); [docs/design.md#outcomes](design.md#outcomes); [docs/architecture.md#data-flow](architecture.md#data-flow); [docs/decisions.md#gate-decisions](decisions.md#gate-decisions) |
| Planning power and mandatory full-analysis simulation | [docs/preregistration.md#power-and-false-pass-analysis](preregistration.md#power-and-false-pass-analysis) | [docs/preregistration.md#open-questions](preregistration.md#open-questions); [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis); [docs/preregistration.md#gate-2-scientific](preregistration.md#gate-2-scientific); [docs/plan.md#stage-0--setup](plan.md#stage-0--setup); [docs/plan.md#stage-2--decisive-comparison-on-15b](plan.md#stage-2--decisive-comparison-on-15b); [docs/architecture.md#data-flow](architecture.md#data-flow) |
| Noise-theory prediction and held-out validation | [docs/preregistration.md#gate-3](preregistration.md#gate-3) | [docs/preregistration.md#hypotheses](preregistration.md#hypotheses); [docs/preregistration.md#open-questions](preregistration.md#open-questions); [docs/design.md#predictive-check](design.md#predictive-check); [docs/plan.md#stage-3--factors-and-scale](plan.md#stage-3--factors-and-scale); [docs/plan.md#stage-4--large-models](plan.md#stage-4--large-models); [docs/decisions.md#gate-decisions](decisions.md#gate-decisions) |
| Configuration composition and protocol-version identity | [docs/architecture.md#configuration](architecture.md#configuration) | [docs/preregistration.md#fixed-protocol](preregistration.md#fixed-protocol); [docs/operations.md#configs-on-the-command-line](operations.md#configs-on-the-command-line); [docs/operations.md#launching-runs](operations.md#launching-runs); [docs/decisions.md#protocol-versions](decisions.md#protocol-versions); [configs/base.yaml](../configs/base.yaml) |
| Study 2 question, arms, matching variable and analysis | [docs/study2-preregistration.md#question-and-estimand](study2-preregistration.md#question-and-estimand) | [docs/study2-preregistration.md#primary-analysis](study2-preregistration.md#primary-analysis); [docs/study2-preregistration.md#judge-and-audit](study2-preregistration.md#judge-and-audit); [docs/plan.md#study-2](plan.md#study-2); [docs/decisions.md#study-decisions](decisions.md#study-decisions); [docs/preregistration.md#fixed-protocol](preregistration.md#fixed-protocol); [configs/protocol/zero_field.yaml](../configs/protocol/zero_field.yaml) |
| Study 2 numerics, monitoring, stopping and Stage-A selection | [docs/study2-preregistration.md#stage-a--feasibility](study2-preregistration.md#stage-a--feasibility) | [docs/study2-preregistration.md#panels-monitoring-and-stopping](study2-preregistration.md#panels-monitoring-and-stopping); [docs/study2-preregistration.md#numerical-acceptance](study2-preregistration.md#numerical-acceptance); [docs/study2-preregistration.md#zone-and-levels](study2-preregistration.md#zone-and-levels); [docs/plan.md#study-2](plan.md#study-2) |
| Run provenance, storage, reports and release | [docs/architecture.md#experiment-tracking-and-storage](architecture.md#experiment-tracking-and-storage) | [docs/operations.md#rented-machines](operations.md#rented-machines); [docs/operations.md#weights--biases-in-practice](operations.md#weights--biases-in-practice); [docs/operations.md#release-checklist](operations.md#release-checklist); [docs/decisions.md#gate-decisions](decisions.md#gate-decisions); [docs/reports/README.md#gate-reports](reports/README.md#gate-reports); [docs/README.md#maintenance-rules](README.md#maintenance-rules) |
| Documentation ownership and checks | [docs/README.md#maintenance-rules](README.md#maintenance-rules) | [README.md#quick-start](../README.md#quick-start); [docs/operations.md#setup](operations.md#setup) |
| Container dependencies, source checkouts and runtime checks | [docs/operations.md#containers](operations.md#containers) | [README.md](../README.md); [docs/architecture.md#configuration](architecture.md#configuration); [Dockerfile](../Dockerfile); [.dockerignore](../.dockerignore); [docker/project-run.py](../docker/project-run.py); [scripts/check_runtime.py](../scripts/check_runtime.py) |
| Saved observations, analysis provenance and manuscript reproduction | [docs/architecture.md#publication-evidence](architecture.md#publication-evidence) | [docs/architecture.md#configuration](architecture.md#configuration); [docs/architecture.md#experiment-tracking-and-storage](architecture.md#experiment-tracking-and-storage); [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis); [docs/plan.md#stage-0--setup](plan.md#stage-0--setup); [docs/plan.md#stage-5--write-up](plan.md#stage-5--write-up); [docs/reports/README.md#gate-reports](reports/README.md#gate-reports); [docs/operations.md#artifact-retention](operations.md#artifact-retention) |
| Evidence archives, selected weights and disposable artifacts | [docs/operations.md#artifact-retention](operations.md#artifact-retention) | [docs/architecture.md#publication-evidence](architecture.md#publication-evidence); [docs/architecture.md#experiment-tracking-and-storage](architecture.md#experiment-tracking-and-storage); [docs/operations.md#rented-machines](operations.md#rented-machines); [docs/operations.md#release-checklist](operations.md#release-checklist); [docs/plan.md#stage-0--setup](plan.md#stage-0--setup); [configs/protocol/zero_field.yaml](../configs/protocol/zero_field.yaml) |

<!-- END GENERATED DOC MAP -->

## Maintenance rules

- For a substantive change, find its topic in the map and read the authoritative section first.
  Update that source, then inspect the listed dependent sections. A typo fix does not require
  reading unrelated documents.
- Keep formal decision rules in preregistration.md. Design explains the reasoning, plan describes
  stage progression, architecture describes implementation, and operations owns run procedures.
  Link to formal rules instead of maintaining a second outcomes table. Config files remain the
  executable parameter templates; a protocol change must be checked against them.
- Review dependencies in both directions: a changed config or implementation may require an
  update to its documented specification. When adding or renaming a document, update map.yaml.
  Keep watch patterns aligned with code as planned modules are implemented.
- Run the checker after changes, including work that is not being committed. Inspect its impact
  report and state in the handoff which topics were reviewed. A dependent file need not change
  if its description remains correct; do not make cosmetic edits merely to mark it reviewed.
- Fixed preregistrations and historical reports keep the protocol version and evidence they
  describe. Change a fixed rule only through the documented amendment process. Never silently
  rewrite an old result to match the current protocol.
- Every gate decision is logged in decisions.md with its run evidence and a report in reports/.
  Changes of code or data that alter results bump protocol_version as described in architecture.md.
- Keep documentation focused on research and implementation; exclude personal identifiers and credentials.

## Running the checks

From the repository root after `uv sync`:

```bash
uv run python scripts/check_docs.py
```

The default command is read-only and offline. It checks local Markdown links and heading/HTML
anchors, registered documents and topic references, the generated index, and the configuration
examples declared in map.yaml. Configs are loaded without model downloads or training. An example
marked `allow_missing: true` is a draft template: mandatory `???` values may remain, but invalid
config names, defaults, interpolations and overrides still fail. Shell examples are never executed.

In a Git checkout, ignored Markdown files are excluded from the document scan. Public links and
map entries must not depend on ignored files, even when those files exist locally. Without Git
metadata, the checker validates local files and links but cannot apply Git ignore rules.

Refresh only the generated region after editing the map:

```bash
uv run python scripts/check_docs.py --write-index
```

Request a focused impact report, or compare with a Git revision:

```bash
uv run python scripts/check_docs.py --changed docs/preregistration.md
uv run python scripts/check_docs.py --base HEAD
```

By default, changed paths include staged, unstaged and untracked files relative to HEAD. Before the
first commit, all non-ignored files are treated as new. Outside Git, checks still run; use
`--changed` for impact reporting. Watch patterns may name future modules, but source and review
references must already exist. Impact matching is file-level, even when references name sections.

A nonzero exit means a mechanical check failed. REVIEW lines are an advisory list, not proof of
scientific consistency and not a demand to change every linked file. Semantic agreement still
requires reading the affected sections. The checker never rewrites preregistration or reports.
