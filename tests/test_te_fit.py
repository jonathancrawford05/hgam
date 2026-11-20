"""The ``te`` tensor product on cr margins in a REML gam fit and a discrete bam fit, held to mgcv
(``make_fixtures.R`` sections 26b and 26c).

mgcv counterparts: ``te()`` and ``smooth.construct.tensor.smooth.spec`` with ``smoothCon``'s constraint
(smooth.r, mgcv 1.9-4; the basis is held to ``smoothCon`` in ``tests/test_te_basis.py``), and
``Predict.matrix.tensor.smooth`` for prediction. hgam parses ``te(...)`` as a ``TensorCall(kind="te")`` and
builds ``TermSpec(basis="te")`` through ``build_python_te_term``. In ``bam(discrete = TRUE)`` (26c) ``tero``
(bam.r) puts the margin with the largest ``bs.dim`` last (``te(x, z, k=c(5, 4))`` is fitted as ``te(z,x)``),
``discrete.mf`` compresses each margin covariate on its own, and the term is constructed on the padded frame
``mf0`` (penalty scaling and constraint over its rows; ``te_basis(frame=)``).

Gates (``docs/PRIORITIES.md``): structure exact; the linear predictor to 1e-3 absolute; standard errors of the
linear predictor 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger. The same gates hold for
prediction on new data, which includes points outside both margins' knot ranges.
"""

import numpy as np
import pytest

from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

FAMILY = 'binomial(link="cloglog")'

CASES = {
    "gam_binomial_cloglog_te_reml": {"method": "REML"},
    "gam_binomial_cloglog_factor_te_reml": {"method": "REML"},
    "bam_discrete_binomial_cloglog_te": {"method": "fREML", "discrete": True},
    "bam_discrete_binomial_cloglog_factor_te": {"method": "fREML", "discrete": True},
    "gam_binomial_cloglog_te_select_reml": {"method": "REML", "select": True},
    "gam_binomial_cloglog_factor_te_select_reml": {"method": "REML", "select": True},
    "bam_discrete_binomial_cloglog_te_select": {"method": "fREML", "discrete": True, "select": True},
    "bam_discrete_binomial_cloglog_factor_te_select": {"method": "fREML", "discrete": True, "select": True},
}


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


@pytest.mark.parametrize("name", CASES)
def test_te_meets_the_gates(name: str, load_fixture, margin) -> None:
    """Sections 26b (REML gam) and 26c (discrete bam): ``p ~ te(x, z, bs="cr", k=c(5, 4))`` with x's knots
    supplied, and ``p ~ f + te(x, z)`` at te's defaults. Structure (label, coefficient count, full rank,
    coefficient names, one smoothing parameter per margin) is exact; eta, se and edf meet their gates on
    the data and on new data."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    formula = fx["formula"].replace("'", '"')
    fit = gam(formula, data, FAMILY, weights="n", knots=fx["knots"] or None, **CASES[name])
    labels = _as_list(fx["labels"])
    assert fit.converged
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    # One sp per margin, and with select=TRUE a third for the null-space penalty (smoothCon's need.full).
    assert len(fit.lambda_) == len(fx["sp"]) == (3 if CASES[name].get("select") else 2)

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(_as_list(fx["edf"]), dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    margin("te fit", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("te fit", "se, max rel", d_se, "1e-2 rel", name)
    margin("te fit", "edf, fraction of gate", d_edf, "0.05 abs or 1%", name)
    margin("te predict", "eta, max abs", d_new, "1e-3 abs", name)
    margin("te predict", "se, max rel", d_se_new, "1e-2 rel", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
    assert d_new < 1e-3
    assert d_se_new < 1e-2


@pytest.mark.parametrize("name", [n for n, kw in CASES.items() if kw.get("discrete")])
def test_discrete_te_constraint_is_mgcvs(name: str, load_fixture, margin) -> None:
    """Section 26c: the te term's absorbed constraint (the fit's ``Z``, used for the design and
    prediction) satisfies mgcv's, taken over ``discrete.mf``'s padded frame ``mf0``: ``C Z = 0`` for
    mgcv's unit ``C``, at the deterministic-pieces gate (1e-8). Built over the data rows instead,
    ``C Z`` misses by 5.1e-2 and 2.3e-1 while eta still agrees to 1.5e-8 (the constraint mostly trades
    against the intercept), so this test, not the eta gate, holds the frame."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    fit = gam(
        fx["formula"].replace("'", '"'), data, FAMILY, weights="n", knots=fx["knots"] or None, **CASES[name]
    )
    state = next(s for s in fit.term_states if s.term.basis == "te")
    assert state.term.label == fx["labels"]
    want = np.asarray(fx["constraint"], dtype=np.float64)
    miss = float(np.max(np.abs(want @ state.constraint[0])))
    margin("te discrete", "C Z, max abs", miss, "1e-8 rel", name)
    assert miss < 1e-8


@pytest.mark.parametrize(
    ("formula", "kwargs", "needle"),
    [
        ('p ~ s(x, bs="cr") + te(x, z, bs="cr")', {}, "not verified"),
        ('p ~ te(x, z, bs="cr") + s(z, f, bs="fs", xt="cr")', {}, "not verified"),
    ],
)
def test_te_refusals(formula: str, kwargs: dict, needle: str, load_fixture) -> None:
    """What te does not port yet is refused by name: a te beside another smooth
    of one of its covariates, which is not measured (mgcv's gam.side constrains nested columns for some such
    smooths, e.g. a bare s(x), and leaves others alone, e.g. fs; gam.side is not ported)."""
    data = _columns(load_fixture("gam_binomial_cloglog_factor_te_reml")["data"])
    with pytest.raises(HgamNotImplementedError, match=needle.replace("(", r"\(").replace(")", r"\)")):
        gam(formula, data, FAMILY, weights="n", **{"method": "REML", **kwargs})


def test_k_check_refuses_te(load_fixture) -> None:
    fx = load_fixture("gam_binomial_cloglog_te_reml")
    data = _columns(fx["data"])
    fit = gam(fx["formula"].replace("'", '"'), data, FAMILY, weights="n", knots=fx["knots"], method="REML")
    with pytest.raises(HgamNotImplementedError, match="te term"):
        fit.k_check(data, seed=1)
