"""The exit test: hgam_formula fits as written, through ``hgam.bam``, and meets every gate in
``docs/PRIORITIES.md`` against ``mgcv::bam(discrete = TRUE, select = TRUE)`` on the same model
(``make_fixtures.R`` section 3's frame and knots, recorded in section 14 as ``bam_discrete_exit_formula``)."""

import numpy as np

import hgam
from hgam.frontend.discrete import discretise_frame

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


def test_hgam_formula_fits_as_written(load_fixture, margin):
    ref = load_fixture("bam_discrete_exit_formula")
    structure = load_fixture("exit_hgam_formula_structure")
    rows = ref["data"]
    data = {
        name: np.asarray([r[name] for r in rows])
        if isinstance(rows[0][name], str)
        else np.asarray([r[name] for r in rows], dtype=np.float64)
        for name in rows[0]
        if name != "p"  # the response is DthCnt / ExposCnt, evaluated from the formula
    }
    fit = hgam.bam(
        HGAM_FORMULA,
        data=data,
        family="binomial",
        link="cloglog",
        weights="ExposCnt",
        knots=ref["knots"],
        discrete=True,
        select=True,
    )

    assert fit.converged
    # Structure: exact, against both section 3 and section 14.
    assert list(fit.smooth_labels) == ref["labels"] == structure["labels"]
    assert fit.coef.size == ref["n_coef"] == structure["n_coef"]
    assert fit.log_lambda.size == len(ref["sp"]) == structure["n_sp"]

    # Linear predictor at the estimated smoothing parameters: 1e-3 absolute.
    eta_diff = np.max(np.abs(fit.eta - np.asarray(ref["eta"])))
    margin("exit fit", "eta, max abs", eta_diff, "1e-3 abs", "bam_discrete_exit_formula")
    assert eta_diff < 1e-3

    # Per-term edf (mgcv's fit is full rank, so the three parametric terms are identified): 0.05 or 1%.
    assert ref["rank"] == ref["n_coef"] and len(fit.pivoted_columns) == 0
    edf = dict(zip(ref["labels"], ref["edf"], strict=True))
    edf.update(zip(ref["param_labels"], ref["edf_param"], strict=True))
    edf_diff = max(abs(fit.edf_per_term[label] - value) for label, value in edf.items())
    margin("exit fit", "per-term edf, max abs", edf_diff, "0.05 abs or 1%", "bam_discrete_exit_formula")
    for label, value in edf.items():
        assert abs(fit.edf_per_term[label] - value) <= max(0.05, 0.01 * value), (label, value)

    # Standard errors of the linear predictor at the discretised covariates (as predict.bam): 1% relative.
    _, se = fit.predict(discretise_frame(fit.model, data), se_fit=True)
    se_rel = np.max(np.abs(se / np.asarray(ref["se"]) - 1))
    margin("exit fit", "se, max rel", se_rel, "1% rel", "bam_discrete_exit_formula")
    np.testing.assert_allclose(se, ref["se"], rtol=1e-2)
