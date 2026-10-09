"""Random seeds.

Each source of randomness gets its own stream, derived from the run seed and a name (e.g.
``derive_seed(seed, "teacher")``, ``derive_seed(seed, "data_order")``). Changing how one stream is
consumed then does not shift the others, and every process of a distributed job can rebuild any
stream without communication.
"""

from __future__ import annotations

import hashlib
import json
import os
import random


def derive_seed(base: int, *keys: object) -> int:
    """Deterministic 31-bit seed for the stream named by ``keys`` under the run seed ``base``.

    Keys are hashed as a JSON list, so ``("a:b",)`` and ``("a", "b")``, or ``1`` and ``"1"``, name
    different streams.
    """
    if not isinstance(base, int) or isinstance(base, bool):
        raise TypeError(f"The run seed must be an int, got {base!r}")
    blob = json.dumps([base, *keys]).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little") & 0x7FFFFFFF


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (all devices); optionally force deterministic kernels."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
