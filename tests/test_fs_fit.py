"""The ``fs`` factor-smooth term in a REML gam fit and in a discrete bam fit, held to mgcv
(``make_fixtures.R`` sections 24b to 24e).

mgcv counterpart: ``smooth.construct.fs.smooth.spec`` with ``xt = "cr"`` (smooth.r, mgcv 1.9-4), the
non-``gamm``, non-tensor branch, fitted by ``gam(method = "REML")``; prediction through
``Predict.matrix.fs.interaction``. hgam builds the term with ``build_python_fs_term``
(``hgam.engine.gam_basis_fs``). The term is not centred and ``side.constrain`` is ``FALSE``, so the intercept
lies in the span of the per-level null-space columns; the null-space penalties make the fit identifiable,
and mgcv's rank is the full column count.

Gates (``docs/PRIORITIES.md``): structure exact; the linear predictor to 1e-3 absolute; standard errors of the
linear predictor 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger. Smoothing parameters are
not gated: the two null-space directions are ordered by rounding (``test_fs_basis.py``), so their smoothing
parameters may come out swapped.
"""

import numpy as np
import pytest

from hgam import bam
from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

FORMULA = 'p ~ s(x, f, bs="fs", xt="cr", k=6)'
FAMILY = 'binomial(link="cloglog")'
NAME = "gam_binomial_cloglog_fs_reml"


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


@pytest.mark.parametrize("formula", [FORMULA, 'p ~ s(f, x, bs="fs", xt=list(bs="cr"), k=6)'])
def test_fs_fit_meets_the_gates(formula: str, load_fixture, margin) -> None:
    fx = load_fixture(NAME)
    data = _columns(fx["data"])
    fit = gam(formula, data, FAMILY, weights="n", method="REML", knots=fx["knots"])
    assert fit.converged

    # Structure: one term of k * nf columns under mgcv's label (written order) and names, full rank.
    label = "s(x,f)" if formula == FORMULA else "s(f,x)"
    assert list(fit.smooth_labels) == [label]
    assert fit.smooth_bs_dim == (18,)
    assert fit.coef.size == fx["n_coef"] == fx["rank"] == 19
    assert not fit.pivoted_columns
    names = fx["coef_names"] if formula == FORMULA else [n.replace("s(x,f)", label) for n in fx["coef_names"]]
    assert list(fit.coefficient_names) == names
    assert len(fit.lambda_) == len(fx["sp"]) == 3

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = float(fx["edf"])
    d_edf = abs(fit.edf_per_term[label] - ref_edf) / max(0.05, 0.01 * ref_edf)
    margin("fs fit", "eta, max abs", d_eta, "1e-3 abs", NAME)
    margin("fs fit", "se, max rel", d_se, "1e-2 rel", NAME)
    margin("fs fit", "edf, fraction of gate", d_edf, "0.05 abs or 1%", NAME)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (fit.edf_per_term[label], ref_edf)

    # predict.gam on new data, including covariate values outside the knot range.
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    margin("fs predict", "eta, max abs", d_new, "1e-3 abs", NAME)
    margin("fs predict", "se, max rel", d_se_new, "1e-2 rel", NAME)
    assert d_new < 1e-3
    assert d_se_new < 1e-2


BESIDE = {
    "gam_binomial_cloglog_cr_fs_reml": 'p ~ s(x, bs="cr", k=6) + s(x, f, bs="fs", xt="cr", k=6)',
    "gam_binomial_cloglog_factor_fs_reml": 'p ~ f + s(x, f, bs="fs", xt="cr", k=6)',
}


@pytest.mark.parametrize("name", list(BESIDE))
def test_fs_beside_its_null_space_meets_the_gates(name: str, load_fixture, margin) -> None:
    """Section 24c: a cr smooth of the same covariate (the global-plus-group-curves form) or a factor main
    effect beside the fs term. Both lie in the span of the fs null-space columns and fs has
    ``side.constrain = FALSE``, so only the penalties identify the fit; mgcv's rank is the full column count.
    The two null-space smoothing parameters come out swapped against mgcv here, as DEFERRED records; they are
    not gated."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    fit = gam(BESIDE[name], data, FAMILY, weights="n", method="REML", knots=fx["knots"])
    assert fit.converged
    labels = fx["labels"] if isinstance(fx["labels"], list) else [fx["labels"]]
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.atleast_1d(np.asarray(fx["edf"], dtype=np.float64))
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    margin("fs beside its null space", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("fs beside its null space", "se, max rel", d_se, "1e-2 rel", name)
    margin("fs beside its null space", "edf, fraction of gate", d_edf, "0.05 abs or 1%", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)


def test_fs_k_check_row_and_null_space_dim(load_fixture) -> None:
    """mgcv's ``k.check`` (plots.r) gives NA for a smooth with a factor covariate, and
    ``smooth.construct.fs.smooth.spec`` sets ``null.space.dim <- 0``. Not key results; read from the
    source."""
    fx = load_fixture(NAME)
    data = _columns(fx["data"])
    fit = gam(FORMULA, data, FAMILY, weights="n", method="REML", knots=fx["knots"])
    (row,) = fit.k_check(data, seed=1)
    assert (row.label, row.k_prime, row.k_index, row.p_value, row.status) == (
        "s(x,f)",
        18,
        None,
        None,
        "factor",
    )
    assert [(sm.label, sm.null_space_dim) for sm in fit.summary().smooths] == [("s(x,f)", 0)]


@pytest.fixture(scope="module")
def small() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(24)
    return {
        "x": rng.uniform(1, 21, 200),
        "z": rng.uniform(1, 21, 200),
        "f": rng.choice(["a", "b"], 200),
        "g": rng.choice(["u", "v"], 200),
        "p": rng.uniform(0.01, 0.1, 200),
        "n": np.full(200, 50.0),
    }


@pytest.mark.parametrize(
    ("formula", "kwargs", "match"),
    [
        ('p ~ s(x, f, bs="fs", k=6)', {}, "default fs base smooth is tp"),
        ('p ~ s(x, f, bs="fs", xt="tp", k=6)', {}, 'only xt="cr"'),
        ('p ~ s(x, f, bs="fs", xt="cr", m=1)', {}, "argument 'm'"),
        ('p ~ s(x, f, bs="fs", xt="cr", by=z)', {}, "argument 'by'"),
        ('p ~ s(x, z, bs="fs", xt="cr")', {}, "exactly one factor"),
        ('p ~ s(f, g, bs="fs", xt="cr")', {}, "exactly one factor"),
    ],
)
def test_fs_refusals(small, formula: str, kwargs: dict, match: str) -> None:
    opts = {"method": "REML", **kwargs}
    with pytest.raises(HgamNotImplementedError, match=match):
        gam(formula, small, FAMILY, weights="n", **opts)


DISCRETE = ("bam_discrete_binomial_cloglog_fs", "bam_discrete_binomial_cloglog_fs_factor_first")


@pytest.mark.parametrize("name", DISCRETE)
def test_fs_discrete_bam_meets_the_gates(name: str, load_fixture, margin) -> None:
    """Section 24d: ``bam(method = "fREML", discrete = TRUE)``. mgcv builds the term on ``discrete.mf``'s
    padded frame ``mf0`` (margins compressed in written order), so ``P`` and the penalty scaling come from
    ``mf0``'s rows; built on the data rows instead, eta misses by 8.1e-3. ``P`` is a deterministic piece
    (1e-8 relative, under the column-sign and null-order map of ``test_fs_basis.py``)."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    formula = fx["formula"].replace("'", '"')
    fit = bam(formula, data, FAMILY, weights="n", method="fREML", discrete=True, knots=fx["knots"])
    assert fit.converged
    assert list(fit.smooth_labels) == [fx["labels"]]
    assert fit.coef.size == fx["n_coef"] == fx["rank"] == 19
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]

    (state,) = fit.term_states
    assert state.fs_p is not None
    ref_p = np.asarray(fx["P"])
    rel_p = min(
        float(
            np.max(
                np.abs(state.fs_p[:, order] * np.sign(np.sum(state.fs_p[:, order] * ref_p, axis=0)) - ref_p)
            )
        )
        for order in ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4])
    ) / float(np.max(np.abs(ref_p)))

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = float(fx["edf"])
    d_edf = abs(fit.edf_per_term[fx["labels"]] - ref_edf) / max(0.05, 0.01 * ref_edf)
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    margin("fs discrete bam", "P, max rel", rel_p, "1e-8 rel", name)
    margin("fs discrete bam", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("fs discrete bam", "se, max rel", d_se, "1e-2 rel", name)
    margin("fs discrete bam", "edf, fraction of gate", d_edf, "0.05 abs or 1%", name)
    margin("fs discrete predict", "eta, max abs", d_new, "1e-3 abs", name)
    margin("fs discrete predict", "se, max rel", d_se_new, "1e-2 rel", name)
    # summary.gam's s.table: fs has null.space.dim 0, so its row comes from reTest. Gates (PR #21, Q2): Ref.df
    # exact, Chi.sq 1% relative, p-value 1e-3 absolute or 1% relative, whichever is larger.
    (sm,) = fit.summary().smooths
    _, ref_df, ref_chi, ref_pv = fx["s_table"][0]
    assert sm.ref_df == ref_df
    assert sm.chi_sq is not None and sm.p_value is not None
    d_chi = abs(sm.chi_sq / ref_chi - 1.0)
    margin("fs discrete summary", "Chi.sq, rel", d_chi, "1e-2 rel", name)
    assert d_chi < 1e-2
    assert abs(sm.p_value - ref_pv) <= max(1e-3, 1e-2 * ref_pv)
    assert rel_p < 1e-8
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0
    assert d_new < 1e-3
    assert d_se_new < 1e-2


SELECT = {
    "gam_binomial_cloglog_fs_select_reml": {"method": "REML"},
    "bam_discrete_binomial_cloglog_fs_select": {"method": "fREML", "discrete": True},
    "gam_binomial_cloglog_cr_fs_select_reml": {"method": "REML"},
}


@pytest.mark.parametrize("name", list(SELECT))
def test_fs_select_meets_the_gates(name: str, load_fixture, margin) -> None:
    """Section 24e: ``select=TRUE``. smoothCon's ``null.space.penalty`` branch takes ``need.full`` for
    fs (three penalties), and their sum is full rank, so mgcv appends no penalty to the fs term: its
    fit equals the ``select = FALSE`` one (``eta_vs_noselect`` is 0 in the fixture) and
    ``null_space_penalty`` adds nothing either. Beside ``s(x, bs="cr")`` the cr smooth gets its
    null-space penalty (five smoothing parameters)."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    formula = fx["formula"].replace("'", '"')
    fit = gam(formula, data, FAMILY, weights="n", select=True, knots=fx["knots"], **SELECT[name])
    assert fit.converged
    labels = fx["labels"] if isinstance(fx["labels"], list) else [fx["labels"]]
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    # Structure: the number of smoothing parameters is mgcv's (fs keeps its three; a cr smooth
    # gets a second). sp values are not gated (the two fs null-space directions are ordered by rounding).
    assert len(fit.lambda_) == len(fx["sp"]) == int(np.sum(fx["n_penalties"]))
    if "eta_vs_noselect" in fx:
        # mgcv's fs fit is its select=FALSE fit; hgam's must be too (PR #39 review nit).
        assert fx["eta_vs_noselect"] == 0.0
        plain = gam(formula, data, FAMILY, weights="n", knots=fx["knots"], **SELECT[name])
        np.testing.assert_array_equal(fit.predict(data), plain.predict(data))

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.atleast_1d(np.asarray(fx["edf"], dtype=np.float64))
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    margin("fs select", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("fs select", "se, max rel", d_se, "1e-2 rel", name)
    margin("fs select", "edf, fraction of gate", d_edf, "0.05 abs or 1%", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
