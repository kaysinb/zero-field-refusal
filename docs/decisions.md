# Decision log

Append-only, newest entries last.

## Gate decisions

One entry per gate decision. Each entry names its evidence: the run ids and the gate report in
[reports/](reports/README.md), so the decision can be audited later against the criteria in
[preregistration.md](preregistration.md#gate-thresholds).

| Date | Gate | Evidence | Result against thresholds | Decision |
|---|---|---|---|---|

## Protocol versions

`protocol_version` in `configs/base.yaml` is part of every run's config hash
([why](architecture.md#configuration)). Each bump adds a row.

| Version | Date | What changed | Why |
|---|---|---|---|
| 1 | | Initial protocol | |
