"""``gam(method="fREML", discrete=True)`` held to ``mgcv::bam(method="fREML", discrete=TRUE)`` fixtures
(``make_fixtures.R`` section 12: 12a cr + ti, 12b cr + sz, 12c and 12d the same with ``select=TRUE``,
12e a numeric ``by``; section 13: 13a a factor beside its sz smooth and 13b the numeric ``by``, both with
``select=TRUE``).

The fit is ``bgam.fitd``'s performance iteration (``hgam.frontend.bam_fitd``) on covariates discretised
as ``discrete.mf`` does (``hgam.frontend.discrete.discretise_frame``). Gates (``docs/PRIORITIES.md``):
structure exact, the linear predictor to 1e-3 absolute, per-term edf to 0.05 or 1%, standard errors of
the linear predictor to 1% relative. Smoothing parameters are not a key result and are not gated.

Each fixture also records mgcv's ``bam`` at ``discrete = FALSE`` ("exact"); the last test shows the
discretisation step is what moves hgam from that fit onto mgcv's discrete one.
"""

import numpy as np
import pytest

from hgam.frontend.api import gam
from hgam.frontend.discrete import discretise_frame

ETA_ATOL = 1e-3
SE_RTOL = 1e-2

TI = 'p ~ s(age, bs="cr", k=8) + s(dur, bs="cr", k=5) + ti(age, dur, bs="cr", k=c(5, 4))'
SZ = 'p ~ s(x, bs="cr", k=6) + s(f, x, bs="sz", k=6, xt=list(bs="cr"))'
BY = 'p ~ s(x, bs="cr", k=6) + s(x, by=z, bs="cr", k=6)'
F_SZ = 'p ~ f + s(x, bs="cr", k=6) + s(f, x, bs="sz", k=6, xt=list(bs="cr"))'


def _columns(fx: dict) -> dict[str, np.ndarray]:
    """Numeric columns as float64; factor columns (written by jsonlite as strings) as string arrays."""
    rows = fx["data"]
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


CASES = {
    "bam_discrete_binomial_cloglog_ti": (TI, False),
    "bam_discrete_binomial_cloglog_sz": (SZ, False),
    "bam_discrete_binomial_cloglog_ti_select": (TI, True),
    "bam_discrete_binomial_cloglog_sz_select": (SZ, True),
    # 12e: a numeric by, binned on its own before the margin (discrete.mf).
    "bam_discrete_binomial_cloglog_by_numeric": (BY, False),
    # 13a: a factor main effect beside its sz smooth (the exit formula's shape), full rank with select=TRUE.
    "bam_discrete_binomial_cloglog_f_sz_select": (F_SZ, True),
    # 13b: the numeric by with select=TRUE.
    "bam_discrete_binomial_cloglog_by_numeric_select": (BY, True),
}


def _fit(fx: dict, name: str):
    formula, select = CASES[name]
    data = _columns(fx)
    fit = gam(
        formula,
        data,
        'binomial(link="cloglog")',
        weights="n",
        method="fREML",
        discrete=True,
        knots=fx.get("knots"),
        select=select,
    )
    return fit, data


@pytest.mark.parametrize("name", list(CASES))
def test_bam_discrete_meets_the_gates(load_fixture, name: str) -> None:
    fx = load_fixture(name)
    assert fx["select"] if CASES[name][1] else "select" not in fx
    fit, data = _fit(fx, name)

    assert fit.converged
    # Structure: exact (labels after tero's margin reordering, coefficients, smoothing parameters).
    labels = fx["labels"]
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"]
    assert fit.log_lambda.size == len(fx["sp"])

    # Linear predictor at the estimated smoothing parameters: 1e-3 absolute, largest difference.
    assert np.max(np.abs(fit.eta - np.asarray(fx["eta"]))) < ETA_ATOL

    # Per-term edf: 0.05 absolute or 1%, whichever is larger.
    # 13a's factor is a term too: select=TRUE makes mgcv's fit full rank, so its edf is identified.
    ref = dict(zip(labels, fx["edf"], strict=True))
    if "rank" in fx:
        assert fx["rank"] == fx["n_coef"]
        assert len(fit.pivoted_columns) == 0
        ref.update(
            zip(
                np.atleast_1d(fx["param_labels"]).tolist(),
                np.atleast_1d(fx["edf_param"]).tolist(),
                strict=True,
            )
        )
    for label, value in ref.items():
        assert abs(fit.edf_per_term[label] - value) <= max(0.05, 0.01 * value), (
            label,
            fit.edf_per_term[label],
            value,
        )

    # Standard errors of the linear predictor, at the discretised covariates as mgcv's predict.bam
    # computes them (it re-runs discrete.mf on the model frame): 1% relative.
    _, se = fit.predict(discretise_frame(fit.model, data), se_fit=True)
    np.testing.assert_allclose(se, fx["se"], rtol=SE_RTOL)


def test_discretisation_is_what_moves_the_fit_onto_mgcvs_discrete_fit(load_fixture) -> None:
    """12a: mgcv's own discrete and exact bam differ by more than the eta gate; hgam's discrete fit sits
    on mgcv's discrete side of that gap, so the binning (not the iteration) is what is matched."""
    name = "bam_discrete_binomial_cloglog_ti"
    fx = load_fixture(name)
    fit, _ = _fit(fx, name)
    mgcv_gap = np.max(np.abs(np.asarray(fx["eta"]) - np.asarray(fx["exact"]["eta"])))
    assert mgcv_gap > ETA_ATOL
    assert (
        np.max(np.abs(fit.eta - np.asarray(fx["eta"])))
        < ETA_ATOL
        < np.max(np.abs(fit.eta - np.asarray(fx["exact"]["eta"])))
    )
