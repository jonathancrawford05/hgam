"""The reference fixtures are present, self-describing and internally consistent."""

import numpy as np
import pytest

NAMES = ["smoothcon_cr_k6_knots", "gam_poisson_offset_reml", "exit_hgam_formula_structure"]


@pytest.mark.parametrize("name", NAMES)
def test_fixture_records_its_producer(load_fixture, name):
    meta = load_fixture(name)["meta"]
    assert meta["mgcv"] and meta["r"]
    assert meta["tier"] in (1, 3)
    assert meta["tier"] == 1 or meta["oracle_digest"].startswith("sha256:")


def test_cr_basis_fixture_shapes(load_fixture):
    fx = load_fixture("smoothcon_cr_k6_knots")
    x_mat, pen = np.array(fx["X"]), np.array(fx["S"])
    assert x_mat.shape == (40, 6) and pen.shape == (6, 6)
    np.testing.assert_allclose(pen, pen.T, atol=1e-12)
    np.testing.assert_allclose(x_mat.sum(axis=1), 1.0, atol=1e-10)  # cr basis functions sum to one
    assert np.array(fx["X_absorbed"]).shape == (40, 5)


def test_target_structure_is_recorded(load_fixture):
    fx = load_fixture("exit_hgam_formula_structure")
    assert fx["n_smooth"] == 8 and fx["n_sp"] == 21
    assert sum(fx["term_coef"]) + 4 == fx["n_coef"]
