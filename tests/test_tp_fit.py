"""The one-covariate thin-plate regression spline, ``s(x)`` / ``s(x, bs="tp", k=)``, held to mgcv
(``make_fixtures.R`` section 28).

mgcv counterparts: ``smooth.construct.tp.smooth.spec`` (smooth.r) and its ``construct_tprs`` (tprs.c), with
``smoothCon``'s sum-to-zero constraint, mgcv 1.9-4. hgam's basis is :func:`hgam.engine.gam_basis_tp.tp_basis`
(the eigen-truncation recipe in its docstring; LAPACK's ``syevd`` replaces mgcv's Lanczos routine), moved with
M0. PLAN step 3's rule is to measure first and port only what misses a gate; nothing did, so this module is
the measurement.

The basis is compared up to its gauge: the range-space coordinates are fixed only up to an invertible matrix
(eigenvector sign and order), so the deterministic gate (1e-8 relative) is applied to the column space of the
constrained design and to the penalty in the basis that maps hgam's design onto mgcv's. Fitted quantities are
gauge invariant and are gated directly (``docs/PRIORITIES.md``): structure exact; eta 1e-3 absolute; per-term
edf 0.05 absolute or 1%, whichever is larger; se of eta 1% relative, on the data and on new data (x outside
its range included).
"""

import numpy as np
import pytest

from hgam.engine.gam_basis_cr import absorb_sum_to_zero_constraint
from hgam.engine.gam_basis_tp import tp_basis
from hgam.frontend.api import gam

DETERMINISTIC_RTOL = 1e-8
ETA_ATOL = 1e-3
SE_RTOL = 1e-2

CASES = {
    # 28b: mgcv's default smooth, s(x) (tp, k = 10), beside a cr smooth.
    "gam_binomial_cloglog_tp_reml": dict(family='binomial(link="cloglog")', weights="n", select=False),
    # 28c: two tp smooths at set k, Poisson with an exposure offset.
    "gam_poisson_tp_reml": dict(family="poisson", weights=None, select=False),
    # 28d: 28b with select=TRUE (a null-space penalty on the tp term's linear function).
    "gam_binomial_cloglog_tp_select_reml": dict(family='binomial(link="cloglog")', weights="n", select=True),
}


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    return {name: np.asarray([r[name] for r in rows], dtype=np.float64) for name in rows[0]}


def test_tp_basis_matches_smoothcon_up_to_its_gauge(load_fixture, margin) -> None:
    """28a: ``smoothCon(s(x, bs="tp", k=8), absorb.cons=TRUE)`` on 1200 rows with repeated values.
    hgam's constrained design spans mgcv's column space, and the penalty agrees in the basis mapping one onto
    the other."""
    fx = load_fixture("smoothcon_tp_k8")
    x = np.asarray(fx["x"], dtype=np.float64)
    x_ref = np.asarray(fx["X"], dtype=np.float64)
    s_ref = np.asarray(fx["S"], dtype=np.float64)
    design, s, _ = tp_basis(x, fx["k"])
    design_c, s_c = absorb_sum_to_zero_constraint(design, s)
    assert design_c.shape == x_ref.shape == (x.size, fx["k"] - 1)

    # x_ref = design_c @ t: the map is square and invertible, so both span one space.
    t, *_ = np.linalg.lstsq(design_c, x_ref, rcond=None)
    assert np.linalg.matrix_rank(t) == fx["k"] - 1
    design_err = np.max(np.abs(design_c @ t - x_ref)) / np.max(np.abs(x_ref))
    penalty_err = np.max(np.abs(t.T @ s_c @ t - s_ref)) / np.max(np.abs(s_ref))
    margin("tp basis (28a)", "design, in mgcv's basis", design_err, "1e-8 relative", "smoothcon_tp_k8")
    margin("tp basis (28a)", "penalty, in mgcv's basis", penalty_err, "1e-8 relative", "smoothcon_tp_k8")
    assert design_err < DETERMINISTIC_RTOL
    assert penalty_err < DETERMINISTIC_RTOL
    # The penalty's rank: the k - 1 constrained coefficients less the null space left after the constraint
    # (the linear function), as smoothCon reports them.
    assert fx["null_space_dim"] == 1
    assert np.linalg.matrix_rank(s_c) == fx["rank"] == fx["k"] - 1 - fx["null_space_dim"]


@pytest.mark.parametrize("name", list(CASES))
def test_tp_reml_fit_meets_the_gates(name: str, load_fixture, margin) -> None:
    fx = load_fixture(name)
    case = CASES[name]
    data = _columns(fx["data"])
    fit = gam(
        fx["formula"].replace("'", '"'),
        data,
        case["family"],
        weights=case["weights"],
        method="REML",
        select=case["select"],
    )
    assert fit.converged
    labels = fx["labels"]
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(fx["sp"])

    eta, se = fit.predict(data, se_fit=True)
    eta_err = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    se_err = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(fx["edf"], dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    edf_ratio = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))

    newdata = _columns(fx["newdata"])
    if "log_expo" in fx["formula"]:
        newdata["log_expo"] = np.zeros(newdata["x"].size)  # the fixture predicts at log_expo = 0
    eta_new, se_new = fit.predict(newdata, se_fit=True)
    eta_new_err = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    se_new_err = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))

    margin(name, "eta", eta_err, "1e-3 absolute", name)
    margin(name, "se of eta (relative)", se_err, "1e-2 relative", name)
    margin(name, "edf per term (fraction of gate)", edf_ratio, "1.0", name)
    margin(name, "eta, new data", eta_new_err, "1e-3 absolute", name)
    margin(name, "se of eta, new data (relative)", se_new_err, "1e-2 relative", name)
    assert eta_err < ETA_ATOL
    assert se_err < SE_RTOL
    assert edf_ratio <= 1.0, (got_edf, ref_edf)
    assert eta_new_err < ETA_ATOL
    assert se_new_err < SE_RTOL
