"""The ``sz`` factor-smooth basis with a cr base smooth.

mgcv counterpart: ``smooth.construct.sz.smooth.spec`` (smooth.r, mgcv 1.9-4), with the sum-to-zero
contrast that ``smoothCon`` applies through ``XZKr`` (smooth.r).

The absorbed design and per-level penalties are deterministic pieces (1e-8 relative, ``smoothcon_sz_cr``,
section 7). A fitted model with an sz term is held to its gates in ``tests/test_gam_reml_fixtures.py``
(section 7).
"""

import numpy as np
import pytest

from hgam.engine.gam_basis_cr import sz_basis
from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

DETERMINISTIC_RTOL = 1e-8


def _rel(got: np.ndarray, ref: list) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(got - ref_arr)) / np.max(np.abs(ref_arr)))


@pytest.mark.parametrize("case", ["two_levels", "three_levels"])
def test_sz_basis_matches_smoothcon(load_fixture, case: str) -> None:
    fx = load_fixture("smoothcon_sz_cr")["cases"][case]
    design, penalties = sz_basis(
        np.asarray(fx["x"], dtype=np.float64),
        np.asarray(fx["group"], dtype=np.int64),
        fx["n_levels"],
        np.asarray(fx["knots"], dtype=np.float64),
    )
    # Structure: exact.
    assert design.shape[1] == fx["bs_dim"]
    assert len(penalties) == len(fx["S"]) == fx["n_levels"]
    assert [int(np.linalg.matrix_rank(s)) for s in penalties] == fx["rank"]
    # Deterministic pieces: 1e-8 relative.
    assert _rel(design, fx["X"]) < DETERMINISTIC_RTOL
    for got, ref in zip(penalties, fx["S"], strict=True):
        assert _rel(got, ref) < DETERMINISTIC_RTOL


@pytest.fixture
def frame() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    f = np.array(["a", "b", "c"])[rng.integers(0, 3, 200)]
    return {"x": x, "f": f, "y": np.sin(x) + 0.5 * (f == "b") * np.sin(x / 3) + rng.normal(0, 0.5, 200)}


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        ('y ~ f + s(f, x, bs="sz", k=5)', "without xt="),
        ('y ~ f + s(f, x, bs="sz", k=5, xt=list(bs="tp"))', "xt="),
        ('y ~ f + s(f, x, bs="sz", k=5, xt="cr")', "xt="),
        ('y ~ f + s(f, x, bs="sz", k=5, xt=list(bs="cr"), by=x)', "argument 'by'"),
        ('y ~ f + s(x, f, bs="sz", k=5, xt=list(bs="cr"))', "factor written first"),
        ('y ~ f + s(f, bs="sz", xt=list(bs="cr"))', "1 variable"),
        ('y ~ s(x, bs="cr", k=5, xt=list(bs="cr"))', "xt="),
    ],
)
def test_unported_sz_forms_are_refused(frame, formula: str, match: str) -> None:
    with pytest.raises(HgamNotImplementedError, match=match):
        gam(formula, frame, "gaussian")


def test_sz_with_select_is_refused_until_measured(frame) -> None:
    with pytest.raises(HgamNotImplementedError, match="select=True"):
        gam('y ~ f + s(f, x, bs="sz", k=5, xt=list(bs="cr"))', frame, "gaussian", select=True)


def test_sz_uses_supplied_knots(frame) -> None:
    knots = [0.0, 1.0, 3.0, 6.0, 10.0]
    formula = 'y ~ s(x, bs="cr", k=5) + s(f, x, bs="sz", k=5, xt=list(bs="cr"))'
    fit = gam(formula, frame, "gaussian", knots={"x": knots})
    state = next(s for s in fit.term_states if s.knots)
    np.testing.assert_array_equal(state.knots[0], knots)
    assert np.max(np.abs(fit.eta - gam(formula, frame, "gaussian").eta)) > 1e-3
