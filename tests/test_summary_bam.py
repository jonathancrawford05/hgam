"""``summary`` of the exit fit, held to ``summary.gam`` (mgcv.r) on mgcv's ``b14``.

The fit is the exit test's (``tests/test_exit.py``, fixture ``bam_discrete_exit_formula``), with the
factors given mgcv's level order (``Low, High``; R would sort the strings ``High, Low``).
``make_fixtures.R`` section 16 exports ``summary(b14)``. Gates: names exact (structure); Estimate 1e-3
absolute (the eta gate); Std. Error 1% relative (the se gate); per-term edf 0.05 absolute or 1%.
z and p are functions of those two and are not gated. Smooth-term tests (reTest, Jonathan's PR #21 gates):
Chi.sq 1% relative, Ref.df exact except on three sz terms where mgcv's rank is set by rounding (option a,
PR #23); the p-value (psum.chisq) 1e-3 absolute or 1% relative, whichever is larger.

The intercept is the level of ``s(AttdAge)``, so it meets the estimate gate only because each
sum-to-zero constraint is taken over ``discrete.mf``'s padded frame, as bam(discrete = TRUE) takes it
(``tests/test_discrete_constraint_rows.py``, section 17).
Over one row per distinct value it missed by 3.6e-3."""

import json
from pathlib import Path

import numpy as np
import pytest

import hgam
from hgam.engine.gam_term_spec import TermSpec
from hgam.frontend.api import Factor, GamFit, parametric_coefficient_names

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
    return _load("bam_discrete_exit_summary")


@pytest.fixture(scope="module")
def exit_fit(ref) -> GamFit:
    rows = _load("bam_discrete_exit_formula")
    data: dict[str, object] = {}
    for name in rows["data"][0]:
        if name == "p":
            continue
        values = [r[name] for r in rows["data"]]
        if name in ref["levels"]:
            data[name] = Factor(values, ref["levels"][name])
        else:
            data[name] = np.asarray(values, dtype=np.float64)
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


def test_parametric_table_meets_the_gates(exit_fit, ref):
    s = exit_fit.summary()
    assert ref["p_columns"] == ["Estimate", "Std. Error", "z value", "Pr(>|z|)"]
    assert s.parametric is not None
    assert [r.name for r in s.parametric] == ref["p_names"]
    table = np.asarray(ref["p_table"])
    est = np.array([r.estimate for r in s.parametric])
    se = np.array([r.std_error for r in s.parametric])
    assert np.max(np.abs(est - table[:, 0])) < 1e-3
    np.testing.assert_allclose(se, table[:, 1], rtol=1e-2)
    # z and p are est / se and 2 * pnorm(-|z|), as summary.gam computes them.
    for r in s.parametric:
        np.testing.assert_allclose(r.z, r.estimate / r.std_error, rtol=1e-14)


def test_intercept_meets_the_estimate_gate(exit_fit, ref):
    s = exit_fit.summary()
    assert s.parametric is not None
    assert abs(s.parametric[0].estimate - ref["p_table"][0][0]) < 1e-3


def test_smooth_rows_and_scalars(exit_fit, ref):
    s = exit_fit.summary()
    assert [t.label for t in s.smooths] == ref["s_names"]
    edf = np.array([t.edf for t in s.smooths])
    ref_edf = np.asarray(ref["s_table"])[:, 0]
    assert np.all(np.abs(edf - ref_edf) <= np.maximum(0.05, 0.01 * np.abs(ref_edf)))
    assert s.n == ref["n"]
    assert s.scale == ref["scale"] == 1.0
    assert not s.scale_estimated
    assert s.outer == "fREML performance iteration"
    text = str(s)
    assert "FaceSizeHigh:SmokeS" in text and "Pr(>|z|)" in text
    assert "psum.chisq is not ported" in text and "fREML score is not ported" in text


def test_coefficient_names_follow_model_matrix():
    terms = (
        TermSpec(label="A", variables=("A",), basis="parametric", levels=(3,)),
        TermSpec(label="B", variables=("B",), basis="parametric", levels=(2,)),
        TermSpec(label="A:B", variables=("A", "B"), basis="parametric", levels=(3, 2)),
    )
    names = parametric_coefficient_names(terms, {"A": ("a0", "a1", "a2"), "B": ("b0", "b1")})
    # colnames(model.matrix(~A + B + A:B)) in R (gam_basis_parametric's docstring)
    assert names == ("(Intercept)", "Aa1", "Aa2", "Bb1", "Aa1:Bb1", "Aa2:Bb1")


# Jonathan's decision on PR #23 (option a): Ref.df is gated exact only where mgcv's rank is not set
# by rounding.
# On these three sz terms it counts eigenvalues of crossprod(Rm B), and B = mroot(Ve[m, m]) takes its rank
# from dpstrf stopping at the first non-positive pivot of a matrix whose spectrum falls to 1e-17. Their Ref.df
# is reported, not gated; Chi.sq and the p-value (below) are gated for them.
SZ_REF_DF = ("s(FaceSize,AttdAge)", "s(Smoke,AttdAge)", "s(FaceSize,PolYear)")
S_NAMES = json.loads((FIXTURES / "bam_discrete_exit_summary.json").read_text())["s_names"]


def test_smooth_null_space_and_chi_sq(exit_fit, ref):
    """Every exit smooth is fully penalised (select=TRUE), so summary.gam tests it by reTest."""
    s = exit_fit.summary()
    assert [t.null_space_dim for t in s.smooths] == ref["null_space_dim"]
    chi = np.array([t.chi_sq for t in s.smooths])
    np.testing.assert_allclose(chi, np.asarray(ref["s_table"])[:, 2], rtol=1e-2)


def test_smooth_p_values_meet_the_gate(exit_fit, ref):
    """Q2 (PR #21): p-value 1e-3 absolute or 1% relative, whichever is larger, on all eight exit smooths,
    the three sz terms whose Ref.df is not gated included (option (a), PR #23)."""
    got = np.array([t.p_value for t in exit_fit.summary().smooths])
    want = np.asarray(ref["s_table"])[:, 3]
    assert np.all(np.abs(got - want) <= np.maximum(1e-3, 1e-2 * np.abs(want)))


@pytest.mark.parametrize("label", [lab for lab in S_NAMES if lab not in SZ_REF_DF])
def test_smooth_ref_df_is_exact(exit_fit, ref, label):
    s = exit_fit.summary()
    row = next(t for t in s.smooths if t.label == label)
    assert row.ref_df == np.asarray(ref["s_table"])[ref["s_names"].index(label), 1]


@pytest.mark.parametrize("label", SZ_REF_DF)
def test_sz_ref_df_is_reported_not_gated(exit_fit, label):
    """Option (a): a rank in 1..n_coef is reported; its value is not compared with mgcv's."""
    row = next(t for t in exit_fit.summary().smooths if t.label == label)
    assert row.ref_df is not None and 1 <= row.ref_df <= row.n_coef
