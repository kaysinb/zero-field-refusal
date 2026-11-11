"""The vector files of extension 2 (``vectors``)."""

from __future__ import annotations

import math

import pytest
import torch

from zero_field_refusal import vectors

NAMES = ["a.weight", "b.bias"]


def named(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"a.weight": torch.randn(3, 4, generator=g), "b.bias": torch.randn(5, generator=g)}


def test_a_vector_round_trips_with_its_record(tmp_path) -> None:
    v = named(0)
    record = vectors.save_vector(tmp_path / "v.pt", v, NAMES, kind="mean", theta0_sha256="t0",
                                 provenance={"seeds": [1, 2]})  # fmt: skip
    assert record["kind"] == "mean" and record["provenance"] == {"seeds": [1, 2]}
    assert record["count"] == 17
    assert math.isclose(record["norm"], vectors.norm(v), rel_tol=1e-12)
    back, rec = vectors.load_vector(tmp_path / "v.pt", NAMES, theta0_sha256="t0")
    assert rec == record and list(back) == NAMES
    assert all(torch.equal(back[n], v[n]) for n in NAMES)


def test_a_vector_is_never_overwritten(tmp_path) -> None:
    vectors.save_vector(tmp_path / "v.pt", named(0), NAMES, kind="mean", theta0_sha256="t0")
    with pytest.raises(FileExistsError):
        vectors.save_vector(tmp_path / "v.pt", named(1), NAMES, kind="mean", theta0_sha256="t0")


@pytest.mark.parametrize("problem", ["names", "theta0", "bytes", "sha"])
def test_a_vector_that_is_not_the_recorded_one_is_refused(tmp_path, problem) -> None:
    path = tmp_path / "v.pt"
    vectors.save_vector(path, named(0), NAMES, kind="mean", theta0_sha256="t0")
    kwargs = {}
    names = NAMES
    if problem == "names":
        names = list(reversed(NAMES))
    elif problem == "theta0":
        kwargs["theta0_sha256"] = "other"
    elif problem == "bytes":
        torch.save(named(1), path)
    else:
        kwargs["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        vectors.load_vector(path, names, **kwargs)


def test_names_and_finiteness_are_required(tmp_path) -> None:
    with pytest.raises(ValueError):
        vectors.save_vector(tmp_path / "v.pt", {"a.weight": torch.ones(2)}, NAMES, kind="x",
                            theta0_sha256="t0")  # fmt: skip
    bad = named(0)
    bad["b.bias"][0] = math.nan
    with pytest.raises(ValueError):
        vectors.save_vector(tmp_path / "w.pt", bad, NAMES, kind="x", theta0_sha256="t0")
    assert not (tmp_path / "w.pt").exists() and not list(tmp_path.glob(".*partial"))


def test_arithmetic_is_float64() -> None:
    a, b = named(0), named(1)
    flat = lambda v: torch.cat([v[n].reshape(-1).double() for n in NAMES])  # noqa: E731
    assert math.isclose(vectors.dot(a, b), float(flat(a) @ flat(b)), rel_tol=1e-12)
    c = vectors.combine([(0.5, a), (-2.0, b)], NAMES)
    assert torch.allclose(flat(c), (0.5 * flat(a) - 2.0 * flat(b)).float().double(), atol=1e-6)
    d = vectors.displacement(a, b, NAMES)
    assert torch.allclose(flat(d), (flat(a) - flat(b)).float().double())
