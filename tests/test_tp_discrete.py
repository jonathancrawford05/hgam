"""The one-covariate thin-plate spline, ``s(x)`` / ``s(x, bs="tp", k=)``, in ``bam(method="fREML",
discrete=TRUE)``, held to mgcv (``make_fixtures.R`` section 30).

mgcv counterparts: ``discrete.mf`` (bam.r) bins the tp covariate as one group, like a cr covariate (a
one-covariate tp is not ``tensor.possible``); ``gam.setup`` builds the smooth on the padded frame ``mf0``
(``smooth.construct.tp.smooth.spec`` and ``construct_tprs``: the shift is ``colMeans`` over ``mf0``, the
column norms are over its ``nrow(mf0)`` rows; then ``smoothCon``'s ``scale.penalty`` and ``colMeans``
constraint over the same rows), and the design at the binned data rows is ``Predict.matrix.tprs.smooth``,
mgcv 1.9-4. hgam: :func:`hgam.engine.gam_term_build.tp_term_basis` inside ``constraint_frame(mf0)``.

Gates (``docs/PRIORITIES.md``): structure exact; eta 1e-3 absolute; se of eta 1% relative; per-term edf 0.05
absolute or 1%, whichever is larger, on the data and on new data (x outside its range included; the Poisson
case predicts at a nonzero offset). The frame the basis is built on is a deterministic piece (1e-8): the
stored shift and unique values (``smooth$shift``, ``smooth$Xu``) carry no eigenvector gauge and are compared
directly. Built over the data rows instead, the shift misses by 7.2e-2 and the smoothing parameters by 1.2e-2
relative while eta still agrees (the column space is the same), so the shift test, not the eta gate, holds
the frame.
"""

import numpy as np
import pytest

from hgam.frontend.api import gam

BINOMIAL = 'binomial(link="cloglog")'
CASES = {
    # 30a: section 28b's p ~ s(x) + s(z, bs="cr", k=6).
    "bam_discrete_binomial_cloglog_tp": dict(family=BINOMIAL, weights="n", select=False),
    # 30b: section 28c's two tp smooths, Poisson with an exposure offset.
    "bam_discrete_poisson_tp": dict(family="poisson", weights=None, select=False),
    # 30c: 30a with select=TRUE (the tp term's null-space penalty).
    "bam_discrete_binomial_cloglog_tp_select": dict(family=BINOMIAL, weights="n", select=True),
}


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    return {name: np.asarray([r[name] for r in rows], dtype=np.float64) for name in rows[0]}


def _fit(fx: dict, case: dict):
    return gam(
        fx["formula"].replace("'", '"'),
        _columns(fx["data"]),
        case["family"],
        weights=case["weights"],
        method="fREML",
        discrete=True,
        select=case["select"],
    )


@pytest.mark.parametrize("name", list(CASES))
def test_discrete_tp_meets_the_gates(name: str, load_fixture, margin) -> None:
    fx = load_fixture(name)
    fit = _fit(fx, CASES[name])
    data = _columns(fx["data"])
    assert fit.converged
    labels = fx["labels"]
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(fx["sp"])

    eta, se = fit.predict(data, se_fit=True)
    eta_err = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    se_err = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(fx["edf"], dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    edf_ratio = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    eta_new_err = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    se_new_err = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    sp_rel = float(np.max(np.abs(np.asarray(fit.lambda_) / np.asarray(fx["sp"]) - 1.0)))

    margin("tp discrete", "eta, max abs", eta_err, "1e-3 abs", name)
    margin("tp discrete", "se, max rel", se_err, "1e-2 rel", name)
    margin("tp discrete", "edf, fraction of gate", edf_ratio, "0.05 abs or 1%", name)
    margin("tp discrete predict", "eta, max abs", eta_new_err, "1e-3 abs", name)
    margin("tp discrete predict", "se, max rel", se_new_err, "1e-2 rel", name)
    margin("tp discrete", "sp, max rel (reported)", sp_rel, "not gated", name)
    assert eta_err < 1e-3
    assert se_err < 1e-2
    assert edf_ratio <= 1.0, (got_edf, ref_edf)
    assert eta_new_err < 1e-3
    assert se_new_err < 1e-2


@pytest.mark.parametrize("name", list(CASES))
def test_discrete_tp_is_built_on_the_padded_frame(name: str, load_fixture, margin) -> None:
    """The first tp term's ``shift`` and unique shifted values ``Xu`` are mgcv's, taken over ``mf0``."""
    fx = load_fixture(name)
    fit = _fit(fx, CASES[name])
    state = next(s for s in fit.term_states if s.term.basis == "tp")
    assert state.term.label == fx["labels"][0]
    want_shift = float(np.asarray(fx["tp_shift"]).reshape(-1)[0])
    shift_err = abs(state.tp.shift - want_shift) / abs(want_shift)
    want_xu = np.sort(np.asarray(fx["tp_xu"], dtype=np.float64))
    assert state.tp.xu.shape == want_xu.shape
    xu_err = float(np.max(np.abs(np.sort(state.tp.xu) - want_xu)) / np.max(np.abs(want_xu)))
    margin("tp discrete frame", "shift, rel", shift_err, "1e-8 rel", name)
    margin("tp discrete frame", "Xu, max rel", xu_err, "1e-8 rel", name)
    assert shift_err < 1e-8
    assert xu_err < 1e-8


def test_discrete_tp_select_summary_meets_the_smooth_test_gates(load_fixture, margin) -> None:
    """30c: with select=TRUE every smooth is fully penalised, so ``summary`` gives ``reTest``'s Ref.df,
    Chi.sq and p-value (gates from PR #21: Ref.df exact; Chi.sq 1% relative; p 1e-3 absolute or 1% relative,
    whichever is larger). Without select the tp and cr terms have a null space and mgcv's ``testStat`` is
    not ported, so those entries stay ``None``."""
    name = "bam_discrete_binomial_cloglog_tp_select"
    fx = load_fixture(name)
    fit = _fit(fx, CASES[name])
    table = np.asarray(fx["s_table"], dtype=np.float64)
    for row, smooth in zip(table, fit.summary().smooths, strict=True):
        assert smooth.ref_df == int(round(row[1]))
        chi_err = abs(smooth.chi_sq / row[2] - 1.0)
        p_err = abs(smooth.p_value - row[3])
        margin("tp discrete summary", "Chi.sq, rel", chi_err, "1e-2 rel", name)
        margin("tp discrete summary", "p-value, abs", p_err, "1e-3 abs or 1%", name)
        assert chi_err < 1e-2
        assert p_err <= max(1e-3, 0.01 * row[3])
