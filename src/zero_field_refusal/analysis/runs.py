"""The runs table: one row per run, the input of every analysis.

Columns: ``run_id``, ``seed`` (int), ``arm`` (str), ``lr`` (float), ``K`` (displacement in nats,
> 0), ``harmful`` (harmful-compliance labels) and ``judged`` (judged responses). A cell is
(arm, lr); a run is one seed in one cell. Other columns are ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_ARMS = ("sgd", "adam_frozen", "adamw")
CONTROL_ARMS = ("sgd_momentum", "adamw_prewarmed")  # H2a
ARMS = PRIMARY_ARMS + CONTROL_ARMS

KL_COLUMNS = ("run_id", "seed", "arm", "lr", "K")
OUTCOME_COLUMNS = ("harmful", "judged")


def validate_runs(runs: pd.DataFrame, *, outcomes: bool = True) -> pd.DataFrame:
    """A typed copy of the runs table's columns; ``ValueError`` on invalid input.

    With ``outcomes=False`` only the columns that K\\* needs are required and returned.
    """
    columns = KL_COLUMNS + (OUTCOME_COLUMNS if outcomes else ())
    missing = [column for column in columns if column not in runs.columns]
    if missing:
        raise ValueError(f"The runs table lacks columns {missing}")
    if runs.empty:
        raise ValueError("The runs table is empty")
    table = runs.loc[:, list(columns)].copy()
    table["seed"] = _integers(table["seed"], "seed")
    table["arm"] = table["arm"].astype(str)
    table["lr"] = _floats(table["lr"], "lr")
    table["K"] = _floats(table["K"], "K")

    unknown = sorted(set(table["arm"]) - set(ARMS))
    if unknown:
        raise ValueError(f"Unknown arm(s) {unknown}; expected one of {ARMS}")
    if not (table["K"] > 0).all():
        raise ValueError("K must be positive and finite in every run")
    if table["run_id"].duplicated().any():
        raise ValueError("Duplicate run_id values")
    if table.duplicated(["arm", "lr", "seed"]).any():
        raise ValueError("More than one run per seed in a cell (arm, lr)")
    if outcomes:
        table["harmful"] = _integers(table["harmful"], "harmful")
        table["judged"] = _integers(table["judged"], "judged")
        if not (table["judged"] > 0).all():
            raise ValueError("judged must be positive in every run")
        if not ((table["harmful"] >= 0) & (table["harmful"] <= table["judged"])).all():
            raise ValueError("harmful must lie in [0, judged] in every run")
    return table.reset_index(drop=True)


def _floats(column: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(column, errors="coerce").astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite number in every run")
    return values


def _integers(column: pd.Series, name: str) -> pd.Series:
    values = _floats(column, name)
    if not (values == np.round(values)).all():
        raise ValueError(f"{name} must be an integer in every run")
    return values.astype(np.int64)
