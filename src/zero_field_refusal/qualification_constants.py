"""Study 3's numerical qualification: the constants that decide what a part executes and what its
records say, with the record hash (docs/study3-preregistration.md, "Numerical qualification").

They live here, inside the GPU-code identity (``code_identity``): the protocol's batch and steps,
the shifts and the directions they read, the rest part's trajectories and their order, the
comparisons at step 25, the failure kinds and states a record carries, and ``record_sha256``,
which binds a rest record to its baseline record. ``analysis.qualification_verdicts``, outside
the identity, imports them from here and holds only functions of the records; ``qualification``
re-exports every name.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

BATCH, STEPS = 20, 25  # "Protocol": B = 20 and exactly 25 updates
SCIENTIFIC, TECHNICAL = "scientific", "technical"
QUALIFIED, FAILING, INCOMPLETE = "qualified", "failing", "incomplete"  # a case, a rate
NOT_QUALIFIED = "not_qualified"  # technical
ACCEPTED = "accepted"  # a same-path control
SHIFTS = {"s1": "p1", "s2": "p2", "s3": "s3", "s4": "s4"}  # a shift and its direction's key
UNSHIFTED = ("baseline", "repeat")  # the trajectories the declared instability rules apply to
REST = ("repeat", "reference", *SHIFTS)  # the rest part's trajectories, in order
COMPARISONS = ("launcher~baseline",  # the 13 comparisons at step 25
               *(f"{t}~{name}" for name in REST for t in ("launcher", "baseline")))  # fmt: skip
FAMILIES = ("launcher~baseline", "launcher~repeat", "baseline~repeat", "launcher~reference",
            "baseline~reference", "launcher~shift", "baseline~shift")  # fmt: skip
PARTS = ("baseline", "rest", "control")


class QualificationError(ValueError):
    """The qualification cannot run as registered (a scope or a request outside it)."""


def record_sha256(record: Mapping[str, Any] | None) -> str | None:
    """The sha256 of a record file as ``qualification.publish`` writes it (so of the file read
    back, too); None for no record."""
    if record is None:
        return None
    return hashlib.sha256((json.dumps(record, indent=1) + "\n").encode()).hexdigest()
