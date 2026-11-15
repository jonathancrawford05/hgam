"""``hgam.engine.compress_df``: the port of ``mgcv:::compress.df`` (moved from polaris-re's
``test_bam_binning_conformance.py``; the comparison-harness tests stayed behind)."""

import numpy as np
import pytest

from hgam.engine.compress_df import compress_covariates, default_m
from hgam.errors import ValidationError


@pytest.mark.parametrize(
    ("d", "m", "expected"),
    [(1, None, 1000), (2, None, 100), (3, None, 25), (5, None, 25), (1, 500, 500), (2, 10000, 101)],
)
def test_default_m_matches_the_source(d: int, m: int | None, expected: int) -> None:
    assert default_m(d, m) == expected


def test_default_m_three_columns_given_m_is_round_cube_root_plus_one() -> None:
    assert default_m(3, 1000) == round(1000 ** (1 / 3)) + 1 == 11


def test_integer_covariates_are_not_binned() -> None:
    """The ILEC case: far fewer than 1000 unique values means the cells ARE the unique rows."""
    age = np.tile(np.arange(30, 91, dtype=np.float64), 20)
    term = compress_covariates({"age": age})
    assert not term.binned
    assert term.nr == 61
    np.testing.assert_array_equal(term.row_values("age"), age)


def test_all_unique_below_threshold_is_the_identity_index() -> None:
    x = np.random.default_rng(0).uniform(size=800)
    term = compress_covariates({"x": x})
    assert not term.binned
    np.testing.assert_array_equal(term.index, np.arange(800))
    np.testing.assert_array_equal(term.row_values("x"), x)


def test_threshold_is_strictly_greater_than() -> None:
    """1000 unique values are not binned, 1001 are (``nrow(xu) > mm * mf``)."""
    at = compress_covariates({"x": np.arange(1, 1001, dtype=np.float64)})
    over = compress_covariates({"x": np.arange(1, 1002, dtype=np.float64)})
    assert not at.binned
    assert over.binned
    assert over.nr <= 1000


def test_binning_snaps_to_the_closed_form_grid() -> None:
    """x on [0, 1] with 2000 points, m = 1000: grid points are i/999 and no value moves more
    than half a grid step (closed form: the rule is nearest-grid-point)."""
    x = np.linspace(0.0, 1.0, 2000)
    term = compress_covariates({"x": x})
    assert term.binned
    dx = 1.0 / 999.0
    snapped = term.row_values("x")
    assert np.max(np.abs(snapped - x)) <= dx / 2 + 1e-12
    np.testing.assert_allclose(snapped * 999.0, np.round(snapped * 999.0), atol=1e-9)
    assert term.values["x"].min() == 0.0
    np.testing.assert_allclose(term.values["x"].max(), 1.0, atol=1e-15)


def test_a_factor_multiplies_the_threshold_and_is_never_binned() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(size=6000)
    f = rng.integers(0, 3, size=6000).astype(np.float64)
    term = compress_covariates({"x": x, "f": f}, frozenset({"f"}))
    # d = 2 so m = 100; threshold = 100 * 3 = 300; 6000 unique rows exceed it.
    assert term.binned
    assert term.nr <= 100 * 3
    np.testing.assert_array_equal(term.row_values("f"), f)


def test_rows_in_one_cell_share_every_discretised_value() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(size=5000)
    term = compress_covariates({"z": z})
    v = term.row_values("z")
    for cell in np.unique(term.index)[:20]:
        assert np.unique(v[term.index == cell]).shape[0] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_covariates_are_refused(bad: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        compress_covariates({"x": np.array([1.0, bad, 3.0])})


def test_ragged_or_empty_input_is_refused() -> None:
    with pytest.raises(ValidationError):
        compress_covariates({})
    with pytest.raises(ValidationError, match="equal length"):
        compress_covariates({"a": np.zeros(3), "b": np.zeros(4)})
