"""Study 5's order repeat under AdamW (docs/study5-adamw-registration.md): arm A of study 3's
candidate 1 at its two rates, on ten fresh seeds in E2's two new orders, so that each seed's pair
of orders is compared under A beside the same pair under F (E2's runs).

**Seeds.** ``draw_seeds``: 10 of ``E2_SEEDS`` (the 37 fresh seeds whose two E2 orders are both
supported at K\\* under F), drawn by ``numpy.random.default_rng(derive_seed(0, "study5",
"adamw-seeds")).choice(37, 10, replace=False)`` over them in seed order; the record is
``docs/study5-adamw-seeds.json``, committed before any run.

**Runs.** Per seed, ``experiments/study5_order_a`` with ``train.order_seed`` = s + 1000 (order 1)
and s + 2000 (order 2), each at A's η_l and η_u: 40 runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .seeding import derive_seed

RATES = (("eta_l", 1.45e-05), ("eta_u", 2.11e-05))  # study 3's candidate-1 rates of arm A
ORDER_OFFSETS = (("order1", 1000), ("order2", 2000))  # train.order_seed = s + offset (E2's)
E2_SEEDS = (
    50000, 50001, 50002, 50003, 50004, 50006, 50007, 50008, 50009, 50010, 50011, 50012, 50013,
    50014, 50015, 50016, 50017, 50018, 50019, 50020, 50021, 50022, 50023, 50024, 50025, 50026,
    50027, 50028, 50029, 50030, 50032, 50033, 50034, 50035, 50036, 50038, 50039,
)  # fmt: skip
EXCLUDED = (50005, 50031, 50037)  # an E2 order with a failed endpoint under F
DRAW_KEYS = ("study5", "adamw-seeds")
N_SEEDS = 10
SEEDS_FILE = "docs/study5-adamw-seeds.json"
CONFIG = "experiments/study5_order_a"


def draw_seeds() -> list[int]:
    """The 10 seeds in the order drawn."""
    rng = np.random.default_rng(derive_seed(0, *DRAW_KEYS))
    return [E2_SEEDS[int(i)] for i in rng.choice(len(E2_SEEDS), N_SEEDS, replace=False)]


def seeds_record() -> dict[str, Any]:
    """``docs/study5-adamw-seeds.json``: the population, the rule, the draw."""
    drawn = draw_seeds()
    return {
        "what": "The seeds of study 5's order repeat under AdamW.",
        "population": list(E2_SEEDS),
        "excluded": list(EXCLUDED),
        "rule": "numpy.random.default_rng(derive_seed(0, 'study5', 'adamw-seeds'))"
        ".choice(37, 10, replace=False) over the population in seed order",
        "rng_seed": derive_seed(0, *DRAW_KEYS),
        "drawn_in_order": drawn,
        "seeds": sorted(drawn),
        "rates": dict(RATES),
        "orders": {name: f"s + {offset}" for name, offset in ORDER_OFFSETS},
    }


def cell(seed: int, order: str, rate: str) -> dict[str, Any]:
    """One A run: ``seed`` in ``order`` (``ORDER_OFFSETS``' name) at ``rate`` (``RATES``')."""
    offset, lr = dict(ORDER_OFFSETS)[order], dict(RATES)[rate]
    return {"name": f"s5-a-{order}-{lr!r}-s{seed}", "config": CONFIG,
            "overrides": [f"optimizer.lr={lr!r}", f"train.order_seed={seed + offset}"],
            "seeds": [seed]}  # fmt: skip
