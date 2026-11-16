"""``predict`` on new data from a ``discrete=True`` fit, held to ``predict.bam`` / ``predict.bamd`` (bam.r).

The fit is the exit test's (``tests/test_exit.py``, fixture ``bam_discrete_exit_formula``).
``make_fixtures.R`` section 15 predicts mgcv's fit ``b14`` on two new frames: 15a (1500 rows, binned by
``discrete.mf`` over the new frame's own range) and 15b (40 rows, not binned). Gates: eta 1e-3 absolute,
se 1% relative."""

import json
from pathlib import Path

import numpy as np
import pytest

import hgam
from hgam.errors import HgamValidationError
from hgam.frontend.api import GamFit

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


def _columns(rows: list[dict], skip: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([r[name] for r in rows])
        if isinstance(rows[0][name], str)
        else np.asarray([r[name] for r in rows], dtype=np.float64)
        for name in rows[0]
        if name not in skip
    }


@pytest.fixture(scope="module")
def exit_fit() -> GamFit:
    ref = _load("bam_discrete_exit_formula")
    return hgam.bam(
        HGAM_FORMULA,
        data=_columns(ref["data"], skip=("p",)),
        family="binomial",
        link="cloglog",
        weights="ExposCnt",
        knots=ref["knots"],
        discrete=True,
        select=True,
    )


@pytest.mark.parametrize("case", ["a", "b"])
def test_predict_new_data_rebins_as_predict_bamd(exit_fit, case):
    ref = _load(f"bam_discrete_exit_predict_{case}")
    eta, se = exit_fit.predict(_columns(ref["newdata"]), se_fit=True)
    assert np.max(np.abs(eta - np.asarray(ref["eta"]))) < 1e-3
    np.testing.assert_allclose(se, ref["se"], rtol=1e-2)


def test_predict_discrete_false_is_predict_gam_at_raw_covariates(exit_fit):
    ref = _load("bam_discrete_exit_predict_a")
    # The fixture discriminates: on 15a mgcv's binned and raw-covariate predictions differ by more
    # than the gate.
    assert np.max(np.abs(np.asarray(ref["eta"]) - np.asarray(ref["eta_exact"]))) > 1e-3
    eta, se = exit_fit.predict(_columns(ref["newdata"]), se_fit=True, discrete=False)
    assert np.max(np.abs(eta - np.asarray(ref["eta_exact"]))) < 1e-3
    np.testing.assert_allclose(se, ref["se_exact"], rtol=1e-2)


def test_predict_unseen_factor_level_is_refused(exit_fit):
    newdata = _columns(_load("bam_discrete_exit_predict_b")["newdata"])
    newdata["Smoke"] = newdata["Smoke"].astype(object)
    newdata["Smoke"][0] = "Unknown"
    with pytest.raises(HgamValidationError):
        exit_fit.predict(newdata)
