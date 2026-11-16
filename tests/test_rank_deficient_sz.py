"""A factor main effect beside sz smooths of that factor, held to ``mgcv::gam`` (make_fixtures.R section 8).

The exit formula's shape is structurally rank deficient: a cr basis sums to one, so every level's constant
lies in the null space of an sz smooth of that factor, and the factor's contrast columns are redundant with it
(8a: ``f + s(f, x, bs="sz")``; 8b: ``f`` with two sz smooths of one factor). mgcv fits these by dropping the
unidentified directions in ``gdiPK`` (gdi.c, mgcv 1.9-4), choosing them by a pivoted QR in the basis of
``totalPenaltySpace`` (gam.fit3.r) at the working weights. hgam pivots them out structurally
(:mod:`hgam.engine.gam_rank_pivot`); the fitted model is the same, so the key results that do not depend on
which direction is dropped are held to their gates here:

- structure (labels, coefficient count, rank, number of smoothing parameters): exact;
- the linear predictor (1e-3 absolute) and its standard errors (1% relative);
- per-term edf of every term no unidentified direction runs through;
- the summed edf of the terms the unidentified directions run through, which does not depend on the choice.

How that sum splits between those terms is decided by which direction mgcv drops, and that depends on the
arbitrary basis LAPACK returns for a degenerate eigenspace (see ``docs/DEFERRED.md``); it is not asserted
until Jonathan decides how to gate it (``docs/routines/CONTINUATION.md``).
"""

import warnings

import numpy as np
import pytest

from hgam.errors import HgamRankDeficiencyWarning
from hgam.frontend.api import gam

ETA_ATOL = 1e-3
SE_RTOL = 1e-2
FAMILY = 'binomial(link="cloglog")'

CASES = {
    # 8a: section 7's model with the factor main effect added.
    "gam_binomial_cloglog_f_sz_reml": {
        "formula": 'p ~ f + s(x, bs="cr", k=6) + s(f, x, bs="sz", k=6, xt=list(bs="cr"))',
        "involved": ("f", "s(f,x)"),
    },
    # 8b: two sz smooths of one two-level factor beside its main effect, as the exit formula has for FaceSize.
    "gam_binomial_cloglog_two_sz_reml": {
        "formula": (
            'p ~ f + s(x, bs="cr", k=6) + s(z, bs="cr", k=6) + s(f, x, bs="sz", k=6, xt=list(bs="cr")) '
            '+ s(f, z, bs="sz", k=6, xt=list(bs="cr"))'
        ),
        "involved": ("f", "s(f,x)", "s(f,z)"),
    },
}


def _columns(fx: dict) -> dict[str, np.ndarray]:
    """Numeric columns as float64; factor columns (written by jsonlite as strings) as string arrays."""
    rows = fx["data"]
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _within_edf_gate(got: float, ref: float) -> bool:
    return abs(got - ref) <= max(0.05, 0.01 * abs(ref))


@pytest.mark.parametrize("name", list(CASES))
def test_rank_deficient_sz_meets_the_gates(load_fixture, name: str) -> None:
    fx = load_fixture(name)
    case = CASES[name]
    data = _columns(fx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = gam(case["formula"], data, FAMILY, weights="n", method="REML", knots=fx["knots"])
    assert fit.converged
    assert any(issubclass(w.category, HgamRankDeficiencyWarning) for w in caught)

    # Structure: exact. mgcv keeps a zero for each dropped coefficient, as coef_full does.
    assert list(fit.smooth_labels) == fx["labels"]
    assert fit.coef_full.size == fx["n_coef"]
    assert fit.coef_full.size - len(fit.pivoted_columns) == fx["rank"]
    assert fit.log_lambda.size == len(fx["sp"])

    eta, se = fit.predict(data, se_fit=True)
    assert np.max(np.abs(eta - np.asarray(fx["eta"]))) < ETA_ATOL
    np.testing.assert_allclose(se, fx["se"], rtol=SE_RTOL)

    ref = dict(zip(fx["labels"], fx["edf"], strict=True))
    # jsonlite writes a length-one vector as a scalar.
    param_labels = np.atleast_1d(fx["param_labels"]).tolist()
    ref.update(zip(param_labels, np.atleast_1d(fx["edf_param"]).tolist(), strict=True))
    got = fit.edf_per_term
    involved = case["involved"]
    for label, value in ref.items():
        if label not in involved:
            assert _within_edf_gate(got[label], value), (label, got[label], value)
    got_sum = sum(got[label] for label in involved)
    ref_sum = sum(ref[label] for label in involved)
    assert _within_edf_gate(got_sum, ref_sum), (got_sum, ref_sum)
