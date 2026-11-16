"""``gam(method="REML")`` fits held to ``mgcv::gam`` fixtures (``make_fixtures.R`` sections 2, 5 and 6).

The outer smoothing-parameter search here is the moved engine's safeguarded Newton
(``hgam.engine.gam_reml_newton``), which re-derives rather than ports mgcv's ``newton`` (gam.fit3.r).
Under ``docs/PRIORITIES.md`` ("reuse before porting") it is kept while these fits meet the gates:
structure exact, the linear predictor to 1e-3 absolute, per-term edf to 0.05 or 1%, standard errors
of the linear predictor to 1% relative. Smoothing parameters are not a key result and are not gated.
"""

import numpy as np
import pytest

from hgam.frontend.api import gam

ETA_ATOL = 1e-3
SE_RTOL = 1e-2


def _columns(fx: dict) -> dict[str, np.ndarray]:
    rows = fx["data"]
    return {name: np.asarray([r[name] for r in rows], dtype=np.float64) for name in rows[0]}


CASES = {
    # section 2: mgcv's formula has offset(log(expo)); the front end takes offset(col), so log it here.
    "gam_poisson_offset_reml": dict(
        formula='deaths ~ s(age, bs="cr", k=8) + s(dur, bs="cr", k=5) + offset(log_expo)',
        family="poisson",
        weights=None,
        derived={"log_expo": lambda d: np.log(d["expo"])},
    ),
    # section 5: cr + ti, binomial cloglog with prior weights.
    "gam_binomial_cloglog_ti_reml": dict(
        formula='p ~ s(age, bs="cr", k=8) + s(dur, bs="cr", k=5) + ti(age, dur, bs="cr", k=c(5, 4))',
        family='binomial(link="cloglog")',
        weights="n",
        derived={},
    ),
    # section 6: the same model shape with user-supplied knots shared by s() and the ti() margins.
    "gam_binomial_cloglog_knots_reml": dict(
        formula='p ~ s(age, bs="cr", k=8) + s(dur, bs="cr", k=5) + ti(age, dur, bs="cr", k=c(8, 5))',
        family='binomial(link="cloglog")',
        weights="n",
        derived={},
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gam_reml_meets_the_gates(load_fixture, name: str) -> None:
    fx = load_fixture(name)
    case = CASES[name]
    data = _columns(fx)
    for col, make in case["derived"].items():
        data[col] = make(data)
    fit = gam(
        case["formula"], data, case["family"], weights=case["weights"], method="REML", knots=fx.get("knots")
    )

    assert fit.converged
    # Structure: exact.
    labels = fx["labels"]
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"]
    assert fit.log_lambda.size == len(fx["sp"])

    # Linear predictor at the estimated smoothing parameters: 1e-3 absolute, largest difference.
    eta, se = fit.predict(data, se_fit=True)
    assert np.max(np.abs(eta - np.asarray(fx["eta"]))) < ETA_ATOL

    # Per-term edf: 0.05 absolute or 1%, whichever is larger.
    ref_edf = np.asarray(fx["edf"], dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    assert np.all(np.abs(got_edf - ref_edf) <= np.maximum(0.05, 0.01 * ref_edf)), (got_edf, ref_edf)

    # Standard errors of the linear predictor: 1% relative.
    np.testing.assert_allclose(se, fx["se"], rtol=SE_RTOL)
