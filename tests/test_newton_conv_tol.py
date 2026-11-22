"""``gam.control(newton = list(conv.tol = ))`` on ``gam(method="REML")``, held to mgcv's tight refits
(``make_fixtures.R`` sections 27b and 27d).

mgcv counterpart: ``gam.control`` (mgcv.r, mgcv 1.9-4), whose ``newton$conv.tol`` defaults to ``1e-6`` and
is the relative gradient tolerance of ``newton`` (gam.fit3.r). hgam takes it as
``control={"newton": {"conv.tol": value}}`` and hands it to the outer search, by default the port of mgcv's
``newton`` (``tests/test_mgcv_newton.py``). The default tolerance is unchanged.

On ``p ~ f + t2(x, z)`` (27b2) and the same model with ``select=TRUE`` (27d3) the rn block's smoothing
parameter lies on a flat REML ridge, so where a search stops depends on its tolerance. The fixtures record
mgcv refitted at ``conv.tol = 1e-12``. With the same option hgam meets every gate against that refit (like
for like, Jonathan on PR #51 Q1, option (a)).

Gates (``docs/PRIORITIES.md``): the linear predictor to 1e-3 absolute; standard errors of the linear
predictor 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger; the same on new data.
"""

import numpy as np
import pytest

import hgam
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend import api as frontend_api

FAMILY = 'binomial(link="cloglog")'


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


def _hold_to_tight(fit, fx: dict, name: str, margin) -> None:
    ref = fx["tight"]
    assert ref["reml"] <= fx["reml"]  # mgcv's tight fit is not worse than its default (PR #12 rule)
    data = _columns(fx["data"])
    labels = _as_list(fx["labels"])
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(ref["sp"])
    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(ref["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(ref["se"]) - 1.0)))
    ref_edf = np.asarray(_as_list(ref["edf"]), dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_new = float(np.max(np.abs(eta_new - np.asarray(ref["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(ref["se_new"]) - 1.0)))
    margin(f"newton conv.tol, {name}", "eta vs mgcv tight, max abs", d_eta, "1e-3 abs", name)
    margin(f"newton conv.tol, {name}", "se vs mgcv tight, max rel", d_se, "1e-2 rel", name)
    margin(f"newton conv.tol, {name}", "edf vs mgcv tight, fraction of gate", d_edf, "0.05 abs or 1%", name)
    margin(f"newton conv.tol, {name}", "predict eta vs mgcv tight, max abs", d_new, "1e-3 abs", name)
    margin(f"newton conv.tol, {name}", "predict se vs mgcv tight, max rel", d_se_new, "1e-2 rel", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
    assert d_new < 1e-3
    assert d_se_new < 1e-2


def test_tight_conv_tol_meets_mgcvs_tight_refit(load_fixture, margin) -> None:
    """27b2, ``p ~ f + t2(x, z)``, through the public ``hgam.gam`` with mgcv's tight tolerance: the search
    converges and meets every gate against mgcv's tight refit."""
    name = "gam_binomial_cloglog_factor_t2_reml"
    fx = load_fixture(name)
    control = {"newton": {"conv.tol": fx["tight"]["conv_tol"]}}
    fit = hgam.gam(
        fx["formula"].replace("'", '"'),
        _columns(fx["data"]),
        FAMILY,
        weights="n",
        method="REML",
        control=control,
    )
    assert fit.converged
    _hold_to_tight(fit, fx, name, margin)


def test_tight_conv_tol_on_the_select_ridge(load_fixture, margin) -> None:
    """27d3, ``p ~ f + t2(x, z)`` with ``select=TRUE``, through the public ``hgam.gam`` at mgcv's tight
    tolerance: the port reports full convergence, as mgcv's ``newton`` does, and every gate is met against
    mgcv's tight refit."""
    name = "gam_binomial_cloglog_factor_t2_select_reml"
    fx = load_fixture(name)
    assert fx["tight"]["outer_conv"] == "full convergence"
    fit = hgam.gam(
        fx["formula"].replace("'", '"'),
        _columns(fx["data"]),
        FAMILY,
        weights="n",
        method="REML",
        select=True,
        control={"newton": {"conv.tol": fx["tight"]["conv_tol"]}},
    )
    assert fit.converged
    _hold_to_tight(fit, fx, name, margin)


def test_default_conv_tol_is_unchanged(load_fixture) -> None:
    """No ``control``, an empty ``control`` (mgcv's ``control = list()``), an empty ``newton`` list and an
    explicit ``conv.tol = 1e-6`` are one fit."""
    fx = load_fixture("gam_binomial_cloglog_factor_t2_reml")
    data = _columns(fx["data"])
    formula = fx["formula"].replace("'", '"')
    fits = [
        hgam.gam(formula, data, FAMILY, weights="n", method="REML", control=control)
        for control in (None, {}, {"newton": {}}, {"newton": {"conv.tol": 1e-6}})
    ]
    for fit in fits[1:]:
        np.testing.assert_array_equal(fit.lambda_, fits[0].lambda_)
        np.testing.assert_array_equal(fit.coef, fits[0].coef)


def test_summary_reports_the_tolerance_used(load_fixture) -> None:
    """``summary()``'s convergence line reports the tolerance the search ran with (t2 fits refuse
    ``summary``, so a te fit is used)."""
    fx = load_fixture("gam_binomial_cloglog_te_reml")
    data = _columns(fx["data"])
    formula = fx["formula"].replace("'", '"')
    for tol in (None, 1e-9):
        control = None if tol is None else {"newton": {"conv.tol": tol}}
        fit = hgam.gam(formula, data, FAMILY, weights="n", knots=fx["knots"], method="REML", control=control)
        assert fit.summary().epsilon_rel == pytest.approx(1e-6 if tol is None else tol, rel=1e-9)


@pytest.mark.parametrize(
    ("control", "kwargs", "error", "needle"),
    [
        ({"epsilon": 1e-8}, {}, HgamNotImplementedError, "control entries"),
        ({"newton": {"maxHalf": 10}}, {}, HgamNotImplementedError, "maxHalf"),
        (
            {"newton": {"conv.tol": 1e-8}},
            {"method": "fREML", "discrete": True},
            HgamNotImplementedError,
            "bam",
        ),
        ({"newton": {"conv.tol": 0.0}}, {}, HgamValidationError, "positive"),
        ({"newton": {"conv.tol": "1e-8"}}, {}, HgamValidationError, "number"),
        ({"newton": 1e-8}, {}, HgamValidationError, "mapping"),
    ],
)
def test_control_refusals(control, kwargs, error, needle, load_fixture) -> None:
    """Every ``gam.control`` entry but ``newton$conv.tol`` is refused by name, and so is the option on
    discrete bam (its performance iteration does not run ``newton``)."""
    fx = load_fixture("gam_binomial_cloglog_factor_t2_reml")
    with pytest.raises(error, match=needle):
        frontend_api.gam(
            fx["formula"].replace("'", '"'),
            _columns(fx["data"]),
            FAMILY,
            weights="n",
            control=control,
            **{"method": "REML", **kwargs},
        )


def test_empty_control_on_discrete_bam_is_the_default(load_fixture) -> None:
    """An empty ``control`` (mgcv's ``control = list()``) sets nothing, so discrete bam accepts it and fits as
    with no ``control``; only ``control={"newton": ...}`` is refused there (``test_control_refusals``)."""
    fx = load_fixture("gam_binomial_cloglog_factor_t2_reml")
    data = _columns(fx["data"])
    formula = fx["formula"].replace("'", '"')
    fits = [
        frontend_api.gam(formula, data, FAMILY, weights="n", method="fREML", discrete=True, control=control)
        for control in (None, {})
    ]
    np.testing.assert_array_equal(fits[1].lambda_, fits[0].lambda_)
    np.testing.assert_array_equal(fits[1].coef, fits[0].coef)
