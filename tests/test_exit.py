"""The exit test: hgam_formula fits as written. Strict xfail until the port reaches it."""

import pytest

import hgam

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


@pytest.mark.xfail(raises=NotImplementedError, reason="exit test: delivered at the end of milestone M3")
def test_hgam_formula_fits_as_written(load_fixture):
    ref = load_fixture("exit_hgam_formula_structure")
    fit = hgam.bam(
        HGAM_FORMULA,
        data=None,
        family="binomial",
        link="cloglog",
        weights="ExposCnt",
        discrete=True,
        select=True,
    )
    assert len(fit.coefficients) == ref["n_coef"]
