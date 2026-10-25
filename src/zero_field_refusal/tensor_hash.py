"""The content hash of named tensors, shared by the launcher and the qualification harness.

One definition, so a run's recorded v₀ and θ₀ hashes and the harness's are the same function of
the same bytes (docs/study3-preregistration.md, "Numerical qualification").
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch


def tensors_sha256(named: Sequence[tuple[str, torch.Tensor]]) -> str:
    """sha256 over names, shapes, dtypes and bytes, in order."""
    digest = hashlib.sha256()
    for name, tensor in named:
        t = tensor.detach().contiguous().cpu()
        digest.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        digest.update(t.view(torch.uint8).numpy() if t.numel() else b"")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, read in pieces (checkpoints are gigabytes)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while piece := f.read(1 << 24):
            digest.update(piece)
    return digest.hexdigest()
