"""``vcov`` of the exit fit, held to ``vcov.gam`` (mgcv.r) on mgcv's ``b14``.

The fit is the exit test's (``tests/test_exit.py``, fixture ``bam_discrete_exit_formula``), with the
factors given mgcv's level order. ``make_fixtures.R`` section 20 exports ``vcov(b14)`` (Vp),
``vcov(b14, freq = TRUE)`` (Ve), ``vcov(b14, unconditional = TRUE)`` (Vc), the ``dispersion = 2``
rescalings and the dimnames. Gates: names exact (structure); the diagonal 1% relative (the se gate:
the diagonal is the squared se of each coefficient). The off-diagonal is reported, not gated.
``unconditional=True`` is refused for a discrete fit: mgcv's Vc there comes from ``Sl.postproc``'s
smoothing-parameter correction (bam.r), not ported, and differs from Vp by up to 3x on the diagonal."""

import json
from pathlib import Path

import numpy as np
import pytest

import hgam
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.api import Factor, GamFit

FIXTURES = Path(__file__).parent / "fixtures"

HGAM_FORMULA = (
    "DthCnt / ExposCnt ~ FaceSize + Smoke + FaceSize:Smoke"
    " + s(AttdAge, k=13, bs='cr') + s(PolYear, k=6, bs='cr')"
    " + ti(AttdAge, PolYear, k=c(13, 6), bs='cr')"
    " + s(FaceSize, AttdAge, bs='sz', k=13, xt=list(bs='cr'))"
    " + s(Smoke, AttdAge, bs='sz', k=13, xt=list(bs='cr'))"
    " + s(FaceSize, PolYear, bs='sz', k=6, xt=list(bs='cr'))"
    " + s(Smoke, PolYear, bs='sz', k=6, xt=list(bs='cr'))"
    " + s(AttdAge, by=StudyYear_C, k=13, bs='cr')"
)


def _load(name: str) -> dict:
    with (FIXTURES / f"{name}.json").open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def ref() -> dict:
    return _load("bam_discrete_exit_vcov")


@pytest.fixture(scope="module")
def exit_fit() -> GamFit:
    rows = _load("bam_discrete_exit_formula")
    levels = _load("bam_discrete_exit_summary")["levels"]
    data: dict[str, object] = {}
    for name in rows["data"][0]:
        if name == "p":
            continue
        values = [r[name] for r in rows["data"]]
        data[name] = Factor(values, levels[name]) if name in levels else np.asarray(values, dtype=np.float64)
    return hgam.bam(
        HGAM_FORMULA,
        data=data,
        family="binomial",
        link="cloglog",
        weights="ExposCnt",
        knots=rows["knots"],
        discrete=True,
        select=True,
    )


def test_names_are_mgcvs(exit_fit, ref):
    assert list(exit_fit.coefficient_names) == ref["names"]
    assert len(ref["names"]) == exit_fit.coef_full.size == 132


def test_vp_diagonal_meets_the_se_gate(exit_fit, ref):
    vp = exit_fit.vcov()
    expected = np.asarray(ref["Vp"])
    assert vp.shape == expected.shape
    np.testing.assert_allclose(np.diag(vp), np.diag(expected), rtol=1e-2)
    np.testing.assert_allclose(vp, vp.T, rtol=0, atol=1e-14)  # symmetric to rounding


def test_ve_diagonal_meets_the_se_gate(exit_fit, ref):
    ve = exit_fit.vcov(freq=True)
    np.testing.assert_allclose(np.diag(ve), np.diag(np.asarray(ref["Ve"])), rtol=1e-2)
    # Ve is no larger than Vp on the diagonal (penalisation only shrinks it).
    assert np.all(np.diag(ve) <= np.diag(exit_fit.vcov()) * (1 + 1e-12))


def test_dispersion_rescales_as_mgcv(exit_fit, ref):
    assert ref["sig2"] == 1.0
    np.testing.assert_allclose(
        np.diag(exit_fit.vcov(dispersion=2.0)), np.diag(np.asarray(ref["Vp_dispersion_2"])), rtol=1e-2
    )
    np.testing.assert_allclose(
        np.diag(exit_fit.vcov(freq=True, dispersion=2.0)),
        np.diag(np.asarray(ref["Ve_dispersion_2"])),
        rtol=1e-2,
    )
    np.testing.assert_allclose(exit_fit.vcov(dispersion=2.0), 2.0 * exit_fit.vcov(), rtol=1e-15)


def test_unconditional_is_refused_for_a_discrete_fit(exit_fit, ref):
    # mgcv sets Vc here (Sl.postproc); it is not ported, so it is refused rather than returned as Vp.
    assert ref["vc_is_null"] is False
    with pytest.raises(HgamNotImplementedError, match="unconditional=True"):
        exit_fit.vcov(unconditional=True)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"sandwich": True}, HgamNotImplementedError, "gam.sandwich"),
        ({"freq": True, "unconditional": True}, HgamNotImplementedError, "freq=True with unconditional"),
        ({"dispersion": 0.0}, HgamValidationError, "positive finite"),
        ({"dispersion": float("nan")}, HgamValidationError, "positive finite"),
    ],
)
def test_refusals(exit_fit, kwargs, error, match):
    with pytest.raises(error, match=match):
        exit_fit.vcov(**kwargs)


def test_off_diagonal_is_reported(exit_fit, ref):
    """Not gated: the off-diagonal's largest difference, relative to the diagonal's scale."""
    for key, v in (("Vp", exit_fit.vcov()), ("Ve", exit_fit.vcov(freq=True))):
        expected = np.asarray(ref[key])
        d = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        corr_diff = np.max(np.abs(v - expected) / d)
        print(f"{key}: max |difference| / sqrt(v_ii v_jj) = {corr_diff:.2e}")
        assert np.isfinite(corr_diff)
