"""The ``re`` random-effect term beside a cr smooth and beside the fs term, in a REML gam fit and in a
discrete bam fit, held to mgcv (``make_fixtures.R`` section 25).

mgcv counterpart: ``smooth.construct.re.smooth.spec`` (smooth.r, mgcv 1.9-4): the design is
``model.matrix(~ g - 1)``, one identity penalty, no centring constraint, ``null.space.dim`` 0; prediction
through ``Predict.matrix.random.effect``. In ``bam(discrete = TRUE)`` the term is marked ``tensor.possible``
and ``discrete.mf`` compresses ``g`` as a one-variable margin; a factor compresses exactly, so no binning
error enters. hgam builds the term in ``api._build_model`` (``TermSpec(basis="re")``) and needed no change
for these fits: this module measures it.

Gates (``docs/PRIORITIES.md``): structure exact; the linear predictor to 1e-3 absolute; standard errors of the
linear predictor 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger. The same gates hold for
prediction on new data. Smoothing parameters are not gated (fs's null-space pair is ordered by rounding).
"""

import numpy as np
import pytest

from hgam.frontend.api import gam

FAMILY = 'binomial(link="cloglog")'

CASES = {
    "gam_binomial_cloglog_cr_re_reml": {"method": "REML"},
    "bam_discrete_binomial_cloglog_cr_re": {"method": "fREML", "discrete": True},
    "gam_binomial_cloglog_fs_re_reml": {"method": "REML"},
    "bam_discrete_binomial_cloglog_fs_re": {"method": "fREML", "discrete": True},
}


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_re_meets_the_gates(name: str, load_fixture, margin) -> None:
    """Section 25: ``s(x, bs="cr") + s(g, bs="re")`` and ``s(x, f, bs="fs") + s(g, bs="re")`` with an
    eight-level ``g``, as a REML gam and a discrete bam. Structure (labels, coefficient count, full rank,
    coefficient names, one smoothing parameter per penalty) is exact; eta, se and edf meet their gates on the
    data and on new data."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    formula = fx["formula"].replace("'", '"')
    fit = gam(formula, data, FAMILY, weights="n", knots=fx["knots"], **CASES[name])
    assert fit.converged
    assert list(fit.smooth_labels) == fx["labels"]
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(fx["sp"])

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(fx["edf"], dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in fx["labels"]])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    margin("re fit", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("re fit", "se, max rel", d_se, "1e-2 rel", name)
    margin("re fit", "edf, fraction of gate", d_edf, "0.05 abs or 1%", name)
    margin("re predict", "eta, max abs", d_new, "1e-3 abs", name)
    margin("re predict", "se, max rel", d_se_new, "1e-2 rel", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
    assert d_new < 1e-3
    assert d_se_new < 1e-2
