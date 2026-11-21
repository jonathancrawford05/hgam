"""The ``t2`` tensor product on cr margins in a REML gam fit, held to mgcv (``make_fixtures.R`` section 27b).

mgcv counterparts: ``t2()`` and ``smooth.construct.t2.smooth.spec`` with ``smoothCon``'s constraints
(smooth.r, mgcv 1.9-4; the basis is held to ``smoothCon`` in ``tests/test_t2_basis.py``). hgam parses
``t2(...)`` as a ``TensorCall(kind="t2")`` and builds ``TermSpec(basis="t2")`` through
``build_python_t2_term``, in smoothCon's fit parameterisation. mgcv fits in that parameterisation too and only
afterwards maps coefficients, ``Vp`` and ``Ve`` to the prediction one with ``gam.setup``'s ``G$P``, which is
not ported. edf, eta and se on the data come from before that map. Prediction on new data agrees because the
t2 term's unpenalised block holds the constant, so ``[1, X]`` and ``[1, Xp]`` span the same space
(:func:`test_fit_and_prediction_parameterisations_span_the_same_space`); coefficient-level output is refused.

Gates (``docs/PRIORITIES.md``): structure exact; the linear predictor to 1e-3 absolute; standard errors of the
linear predictor 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger. The same gates hold for
prediction on new data, which includes points outside both margins' knot ranges.

On ``p ~ f + t2(x, z)`` mgcv's default search stops short of its own REML optimum: the rn block's smoothing
parameter lies on a flat ridge towards infinity (log sp 5.45 by default, 19.4 at a tight ``newton`` conv.tol,
whose REML score is lower), and mgcv's two fits differ by 1.2e-3 on eta. Following ``docs/PRIORITIES.md`` (the
PR #12 rule), the quantities at estimated sp are gated against the tight fit there; the difference from the
default fit is reported, not gated. The fit at mgcv's own smoothing parameters is a deterministic piece, held
at 1e-8 on both cases.
"""

import numpy as np
import pytest

from hgam.engine.gam_basis_t2 import t2_basis
from hgam.engine.gam_model import resolve_family
from hgam.engine.gam_reml_optimize import penalized_fit_and_score
from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

FAMILY = 'binomial(link="cloglog")'

CASES = ("gam_binomial_cloglog_t2_reml", "gam_binomial_cloglog_factor_t2_reml")


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


def _fit(fx: dict):
    return gam(
        fx["formula"].replace("'", '"'),
        _columns(fx["data"]),
        FAMILY,
        weights="n",
        knots=fx["knots"] or None,
        method="REML",
    )


@pytest.mark.parametrize("name", CASES)
def test_t2_meets_the_gates(name: str, load_fixture, margin) -> None:
    """``p ~ t2(x, z, bs="cr", k=c(5, 4))`` with x's knots supplied, and ``p ~ f + t2(x, z)`` at t2's
    defaults.
    Structure (label, coefficient count, full rank, coefficient names, one smoothing parameter per penalised
    block: rr, nr, rn) is exact; eta, se and edf meet their gates on the data and on new data."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    fit = _fit(fx)
    labels = _as_list(fx["labels"])
    assert fit.converged
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(fx["sp"]) == 3

    ref = fx.get("tight")
    if ref is not None:
        assert ref["reml"] <= fx["reml"]  # the rule's condition: the tight fit is not worse
        eta_default = fit.predict(data)
        margin(
            "t2 fit",
            "eta vs mgcv default (not gated)",
            float(np.max(np.abs(eta_default - fx["eta"]))),
            "reported",
            name,
        )
    else:
        ref = fx
    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(ref["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(ref["se"]) - 1.0)))
    ref_edf = np.asarray(_as_list(ref["edf"]), dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_new = float(np.max(np.abs(eta_new - np.asarray(ref["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(ref["se_new"]) - 1.0)))
    margin("t2 fit", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("t2 fit", "se, max rel", d_se, "1e-2 rel", name)
    margin("t2 fit", "edf, fraction of gate", d_edf, "0.05 abs or 1%", name)
    margin("t2 predict", "eta, max abs", d_new, "1e-3 abs", name)
    margin("t2 predict", "se, max rel", d_se_new, "1e-2 rel", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
    assert d_new < 1e-3
    assert d_se_new < 1e-2


@pytest.mark.parametrize("name", CASES)
def test_t2_fit_at_mgcvs_sp(name: str, load_fixture, margin) -> None:
    """A deterministic piece: hgam's penalised fit at mgcv's own (default) smoothing parameters reproduces
    mgcv's eta to 1e-8 relative to its largest value, so the design and the three penalties are mgcv's."""
    fx = load_fixture(name)
    fit = _fit(fx)
    data = _columns(fx["data"])
    family = resolve_family(fit.model.family, fit.model.link)
    coef, _ = penalized_fit_and_score(
        data["p"],
        fit.design["x"],
        family,
        fit.design["penalty_blocks"],
        np.log10(np.asarray(fx["sp"])),
        weights=data["n"],
    )
    ref = np.asarray(fx["eta"])
    rel = float(np.max(np.abs(fit.design["x"] @ coef - ref)) / np.max(np.abs(ref)))
    margin("t2 fit", "eta at mgcv's sp, max rel", rel, "1e-8 rel", name)
    assert rel < 1e-8


def test_fit_and_prediction_parameterisations_span_the_same_space(load_fixture, margin) -> None:
    """Why hgam may predict in the fit parameterisation: on section 27's t2 terms, mgcv's fit design ``X``
    (constraint ``C``) lies in the span of ``[1, Xp]`` (constraint ``Cp``) and back, at rounding level, so
    ``gam.setup``'s ``G$P`` (``Xp P = X`` with the intercept column) is exact and predictions coincide."""
    for name in ("smoothcon_t2_cr", "smoothcon_t2_cr_three"):
        fx = load_fixture(name)
        x, xp = np.asarray(fx["X"]), np.asarray(fx["Xp"])
        one = np.ones((x.shape[0], 1))
        for a, b in (
            (np.hstack([one, xp]), np.hstack([one, x])),
            (np.hstack([one, x]), np.hstack([one, xp])),
        ):
            p_map = np.linalg.lstsq(a, b, rcond=None)[0]
            rel = float(np.max(np.abs(a @ p_map - b)) / np.max(np.abs(b)))
            margin("t2 G$P", "[1, Xp] P - [1, X], max rel", rel, "1e-8 rel", name)
            assert rel < 1e-8


def test_t2_prediction_matches_the_basis(load_fixture) -> None:
    """``predict`` at the training rows rebuilds the fit's own design (the stored knots, margin ``P`` and the
    fit constraint), so it reproduces the fitted linear predictor exactly."""
    fx = load_fixture("gam_binomial_cloglog_t2_reml")
    fit = _fit(fx)
    np.testing.assert_allclose(fit.predict(_columns(fx["data"])), fit.eta, rtol=0, atol=1e-10)
    state = next(s for s in fit.term_states if s.term.basis == "t2")
    data = _columns(fx["data"])
    built = t2_basis([data["x"], data["z"]], (5, 4), [np.asarray(fx["knots"]["x"]), None])
    np.testing.assert_allclose(state.constraint[0], built.z_fit, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("formula", "kwargs", "needle"),
    [
        ('p ~ t2(x, z, bs="cr")', {"select": True}, "select=True"),
        ('p ~ s(x, bs="cr") + t2(x, z, bs="cr")', {}, "not verified"),
        ('p ~ t2(x, z, bs="cr", full=TRUE)', {}, "full"),
        ('p ~ t2(x, z, bs="cr", k=2)', {}, "k="),
        ('p ~ t2(x, z, bs="tp")', {}, "bs="),
    ],
)
def test_t2_refusals(formula: str, kwargs: dict, needle: str, load_fixture) -> None:
    """What t2 does not port yet is refused by name: select=TRUE (not measured), a t2
    beside another smooth of its covariates (gam.side), and t2()'s other arguments, small k and other
    margin bases."""
    data = _columns(load_fixture("gam_binomial_cloglog_factor_t2_reml")["data"])
    with pytest.raises(HgamNotImplementedError, match=needle.replace("(", r"\(").replace(")", r"\)")):
        gam(formula, data, FAMILY, weights="n", **{"method": "REML", **kwargs})


def test_coefficient_level_output_refuses_t2(load_fixture) -> None:
    """mgcv reports coefficients, Vp and Ve of a t2 fit in the prediction parameterisation (``G$P``, not
    ported), so ``vcov``, ``summary``, ``k_check`` and the unconditional se refuse rather than report the fit
    parameterisation's."""
    fx = load_fixture("gam_binomial_cloglog_t2_reml")
    fit = _fit(fx)
    with pytest.raises(HgamNotImplementedError, match=r"G\$P"):
        fit.vcov()
    with pytest.raises(HgamNotImplementedError, match=r"G\$P"):
        fit.summary()
    with pytest.raises(HgamNotImplementedError, match="t2 term"):
        fit.k_check(_columns(fx["data"]), seed=1)
    # mgcv's estimate.gam maps Vp and Ve through G$P but not Vc, so its unconditional se differs from
    # hgam's fit-parameterisation one (PR #46 review): refused. The conditional se is held above.
    new = _columns(fx["newdata"])
    with pytest.raises(HgamNotImplementedError, match=r"G\$P"):
        fit.predict(new, se_fit=True, unconditional=True)
    fit.predict(new, se_fit=True)
