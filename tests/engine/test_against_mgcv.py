"""The moved engine held to mgcv fixtures from ``oracle/make_fixtures.R``, at the gates of
``docs/PRIORITIES.md``.

Measured on the tier 1 fixtures (mgcv 1.9-1) when the engine moved: cr basis and penalty 1e-14
relative; Poisson REML fit: linear predictor 1e-6 absolute, standard errors 6e-6 relative,
per-term edf 3e-5 absolute.
"""

import numpy as np
import pytest

from hgam.engine.api import gam
from hgam.engine.gam_basis_cr import absorb_sum_to_zero_constraint, cr_basis


def _rel(actual: np.ndarray, expected: list) -> float:
    ref = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(actual - ref)) / np.max(np.abs(ref)))


def test_cr_basis_and_penalty_match_smoothcon(load_fixture) -> None:
    fx = load_fixture("smoothcon_cr_k6_knots")
    x_mat, s_mat = cr_basis(np.array(fx["x"]), np.array(fx["knots"]))
    assert _rel(x_mat, fx["X"]) < 1e-8
    assert _rel(s_mat, fx["S"]) < 1e-8
    x_abs, s_abs = absorb_sum_to_zero_constraint(x_mat, s_mat)
    assert _rel(x_abs, fx["X_absorbed"]) < 1e-8
    assert _rel(s_abs, fx["S_absorbed"]) < 1e-8


@pytest.fixture(scope="module")
def poisson_case(load_fixture):
    fx = load_fixture("gam_poisson_offset_reml")
    rows = fx["data"]
    data = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    data["logexpo"] = np.log(data["expo"])
    fit = gam("deaths ~ s(age, bs='cr', k=8) + s(dur, bs='cr', k=5) + offset(logexpo)", data, "poisson")
    return fx, data, fit


def test_poisson_reml_structure(poisson_case) -> None:
    fx, _, fit = poisson_case
    assert fit.coef.shape[0] == fx["n_coef"]
    assert fit.smooth_labels == ("s(age)", "s(dur)")


def test_poisson_reml_linear_predictor_within_gate(poisson_case) -> None:
    fx, _, fit = poisson_case
    assert float(np.max(np.abs(fit.eta - np.asarray(fx["eta"])))) < 1e-3


def test_poisson_reml_edf_per_term_within_gate(poisson_case) -> None:
    fx, _, fit = poisson_case
    for label, ref in zip(fit.smooth_labels, fx["edf"], strict=True):
        assert abs(fit.edf_per_term[label] - ref) <= max(0.05, 0.01 * ref), label


def test_poisson_reml_standard_errors_within_gate(poisson_case) -> None:
    fx, data, fit = poisson_case
    _, se = fit.predict(data, se_fit=True)
    assert float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0))) < 0.01
