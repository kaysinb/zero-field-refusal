"""Parameter-space vectors over the trainable set, kept as files (docs/study5-preregistration.md,
"Extension 2"): a displacement Δθ = θ − θ₀, the linearized displacements δ̂ and δ_lin, the mean
μ̂ of displacements and its mixtures, and the surrogate's gradient ∇r(θ₀).

**The file.** ``torch.save`` of ``{name: tensor}`` over exactly the trainable set's names, in its
order, every tensor float32 on the host, written atomically (a hidden ``.partial`` file and
``os.replace``; a failed write removes it). Beside it, ``<file>.json`` holds the record:

- ``kind`` (``displacement``, ``delta_hat``, ``delta_lin``, ``mean``, ``mixture``, ``removal``,
  ``grad_r``, …) and free-form ``provenance`` (seeds, run ids, rates, orders, weights);
- ``names_sha256`` (``line_scan.names_sha256`` of the names, in order), ``theta0_sha256`` (θ₀'s
  trainable tensors, ``tensor_hash.tensors_sha256``, as the runs record it);
- ``sha256`` of the file's bytes, its ``norm`` (float64) and ``count`` of elements.

``load_vector`` checks the names against the trainable set, the bytes against the recorded (or
the caller's) sha256 and θ₀'s hash when the caller gives one; any difference is a ``ValueError``.
Arithmetic is float64 per tensor on the host: ``dot``, ``norm``, ``combine``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .tensor_hash import file_sha256

Named = Mapping[str, torch.Tensor]


def names_sha256(names: Sequence[str]) -> str:
    """As ``line_scan.names_sha256``: sha256 of the JSON list of names."""
    return hashlib.sha256(json.dumps(list(names)).encode()).hexdigest()


def record_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".json")


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        write(partial)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def dot(a: Named, b: Named) -> float:
    """⟨a, b⟩ in float64, tensor by tensor over a's names (b must hold them all)."""
    return math.fsum(
        torch.dot(t.reshape(-1).double(), b[name].reshape(-1).to(t.device).double()).item()
        for name, t in a.items()
    )


def norm(a: Named) -> float:
    return math.sqrt(math.fsum(t.double().square().sum().item() for t in a.values()))


def combine(terms: Iterable[tuple[float, Named]], names: Sequence[str]) -> dict[str, torch.Tensor]:
    """Σ_k w_k · v_k over ``names``, accumulated in float64, returned float32 on the host."""
    terms = list(terms)
    out = {}
    for name in names:
        acc = None
        for w, v in terms:
            t = v[name].detach().to("cpu").double() * float(w)
            acc = t if acc is None else acc.add_(t)
        if acc is None:
            raise ValueError("combine needs at least one term")
        out[name] = acc.float()
    return out


def displacement(state: Named, theta0: Named, names: Sequence[str]) -> dict[str, torch.Tensor]:
    """state − θ₀ over ``names``, in float64 then float32 on the host."""
    return {name: (state[name].detach().to("cpu").double()
                   - theta0[name].detach().to("cpu").double()).float()
            for name in names}  # fmt: skip


def save_vector(
    path: Path,
    tensors: Named,
    names: Sequence[str],
    *,
    kind: str,
    theta0_sha256: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``tensors`` (exactly ``names``, in that order) and its record; returns the record.
    An existing file is never overwritten."""
    path = Path(path)
    if path.exists() or record_path(path).exists():
        raise FileExistsError(f"{path} exists; a vector file is never overwritten")
    if sorted(tensors) != sorted(names):
        raise ValueError(
            f"the vector holds {len(tensors)} tensors, not exactly the {len(names)} names"
        )
    ordered = {name: tensors[name].detach().to("cpu", torch.float32).contiguous() for name in names}
    if not all(torch.isfinite(t).all() for t in ordered.values()):
        raise ValueError(f"{path.name}: a non-finite value")
    _atomic(path, lambda p: torch.save(ordered, p))
    record = {
        "kind": kind,
        "names_sha256": names_sha256(names),
        "theta0_sha256": theta0_sha256,
        "sha256": file_sha256(path),
        "norm": norm(ordered),
        "count": sum(t.numel() for t in ordered.values()),
        "provenance": dict(provenance or {}),
    }
    blob = json.dumps(record, indent=1, sort_keys=True) + "\n"
    _atomic(record_path(path), lambda p: Path(p).write_text(blob))
    return record


def read_record(path: Path) -> dict[str, Any]:
    return json.loads(record_path(path).read_text())


def load_vector(
    path: Path,
    names: Sequence[str],
    *,
    sha256: str | None = None,
    theta0_sha256: str | None = None,
    mmap: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The vector and its record; ``ValueError`` unless its names are ``names`` (in order), its
    bytes hash to the record's sha256 (and to ``sha256`` if given) and its θ₀ is ``theta0_sha256``
    (if given). With ``mmap`` the tensors stay memory-mapped (read-only use)."""
    path = Path(path)
    record = read_record(path)
    if record["names_sha256"] != names_sha256(names):
        raise ValueError(f"{path.name}: its names are not the trainable set's")
    if theta0_sha256 is not None and record["theta0_sha256"] != theta0_sha256:
        raise ValueError(f"{path.name}: built on another θ₀")
    actual = file_sha256(path)
    if actual != record["sha256"] or (sha256 is not None and actual != sha256):
        raise ValueError(f"{path.name}: its bytes hash to {actual}, not the recorded sha256")
    tensors = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if list(tensors) != list(names):
        raise ValueError(f"{path.name}: its tensors are not exactly the names, in order")
    return tensors, record
