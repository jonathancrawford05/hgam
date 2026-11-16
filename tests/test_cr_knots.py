"""User-supplied knots for ``bs="cr"``.

mgcv counterpart: ``smooth.construct.cr.smooth.spec`` (smooth.r, mgcv 1.9-4).

The basis at supplied knots is a deterministic piece (1e-8 relative, ``smoothcon_cr_k6_knots``, section 1). A
fitted model with supplied knots is held to its gates in ``tests/test_gam_reml_fixtures.py`` (section 6).
"""

import numpy as np
import pytest

from hgam.engine.gam_basis_cr import absorb_sum_to_zero_constraint, cr_basis
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.api import gam

DETERMINISTIC_RTOL = 1e-8


def _rel(got: np.ndarray, ref: list) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(got - ref_arr)) / np.max(np.abs(ref_arr)))


def test_cr_basis_at_supplied_knots_matches_smoothcon(load_fixture) -> None:
    fx = load_fixture("smoothcon_cr_k6_knots")
    x = np.asarray(fx["x"], dtype=np.float64)
    design, penalty = cr_basis(x, np.asarray(fx["knots"], dtype=np.float64))
    assert _rel(design, fx["X"]) < DETERMINISTIC_RTOL
    assert _rel(penalty, fx["S"]) < DETERMINISTIC_RTOL
    design_c, penalty_c = absorb_sum_to_zero_constraint(design, penalty)
    assert _rel(design_c, fx["X_absorbed"]) < DETERMINISTIC_RTOL
    assert _rel(penalty_c, fx["S_absorbed"]) < DETERMINISTIC_RTOL


@pytest.fixture
def frame() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    return {"x": x, "z": rng.uniform(0, 5, 200), "y": np.sin(x) + rng.normal(0, 0.3, 200)}


def test_knot_count_must_equal_k(frame) -> None:
    with pytest.raises(HgamValidationError, match="!= k"):
        gam('y ~ s(x, bs="cr", k=6)', frame, "gaussian", knots={"x": [0, 2, 5, 10]})


def test_knots_for_a_ti_margin_must_match_its_k(frame) -> None:
    with pytest.raises(HgamValidationError, match="!= k"):
        gam('y ~ ti(x, z, bs="cr", k=c(5, 4))', frame, "gaussian", knots={"x": [0, 2, 4, 6, 8, 10]})


def test_unused_knots_are_refused_not_ignored(frame) -> None:
    with pytest.raises(HgamValidationError, match="no cr smooth"):
        gam('y ~ s(x, bs="cr", k=4)', frame, "gaussian", knots={"x": [0, 3, 6, 10], "w": [1, 2, 3]})


def test_knots_on_a_tp_smooth_are_not_ported(frame) -> None:
    with pytest.raises(HgamNotImplementedError, match="knots="):
        gam("y ~ s(x, k=5)", frame, "gaussian", knots={"x": [0, 2, 5, 8, 10]})
