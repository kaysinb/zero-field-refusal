import pytest

from zero_field_refusal.seeding import derive_seed


def test_derived_seeds_are_stable_and_distinct() -> None:
    assert derive_seed(0, "teacher") == derive_seed(0, "teacher")
    assert derive_seed(0, "teacher") != derive_seed(0, "data_order")
    assert derive_seed(0, "teacher") != derive_seed(1, "teacher")
    assert 0 <= derive_seed(123, "rank", 7) < 2**31


def test_keys_do_not_collide_through_formatting() -> None:
    assert derive_seed(0, "a:b") != derive_seed(0, "a", "b")
    assert derive_seed(0, 1) != derive_seed(0, "1")
    assert derive_seed(0, "1:x") != derive_seed(1, "x")


@pytest.mark.parametrize("base", ["0", 0.0, True, None])
def test_base_must_be_an_int(base: object) -> None:
    with pytest.raises(TypeError):
        derive_seed(base, "teacher")  # type: ignore[arg-type]
