"""The port of mgcv's ``newton`` (gam.fit3.r) as the REML outer search, held to mgcv's **default** fits
(``make_fixtures.R`` section 27e and the fixtures it names).

mgcv counterpart: ``newton`` (gam.fit3.r), mgcv 1.9-4, as
:func:`~hgam.engine.gam_reml_newton.newton_select_lambdas` with ``iteration="mgcv"``; its ``score.scale``
uses :func:`~hgam.engine.gam_reml.mgcv_reml_score_offset` (gam.fit3's REML constant, ``fix.family.ls``).

The five cases are REML gam fits whose stop depends on the search: on 27b2 and 27d3 (``p ~ f + t2(x, z)``,
without and with ``select=TRUE``) and on the cr-by-numeric fit an sp lies on a flat REML ridge, and on the
``sz`` and ``fs`` select fits the default search has several near-optima. hgam's own search (the default,
``iteration="hgam"``) is gated against mgcv's tight refits on the ridge cases (``docs/PRIORITIES.md``); the
port follows mgcv's path and is gated against mgcv's default fit, the reference Jonathan set for 27d3
(PR #48).
``gam()`` still refuses ``select=True`` with t2 in a REML fit; the refusal is bypassed here, and only here.

Gates (``docs/PRIORITIES.md``): the linear predictor to 1e-3 absolute; its standard errors 1% relative;
per-term edf 0.05 absolute or 1%, whichever is larger; the same on new data where the fixture has it. The
iteration count and the final REML score against mgcv's are reported, not gated.
"""

import functools

import numpy as np
import pytest

import hgam
from hgam.engine import gam_model
from hgam.engine.gam_family import binomial_cloglog, poisson_log
from hgam.engine.gam_reml import mgcv_reml_score_offset, saturated_log_likelihood
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend import api as frontend_api

FAMILY = 'binomial(link="cloglog")'
TRAJECTORIES = "gam_newton_trajectories"


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


def _case(load_fixture, name: str) -> dict:
    cases = [c for c in load_fixture(TRAJECTORIES)["cases"] if c["fixture"] == name]
    assert len(cases) == 1
    return cases[0]


def _mgcv_newton_fit(fx: dict, monkeypatch, control: dict | None = None) -> tuple:
    """``gam()`` on the fixture's model with ``newton_iteration="mgcv"``; also returns the selection."""
    seen = []
    search = gam_model.newton_select_lambdas

    def spy(*args, **kwargs):
        seen.append(search(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(gam_model, "newton_select_lambdas", spy)
    monkeypatch.setattr(
        frontend_api, "fit_engine_gam", functools.partial(gam_model.fit_engine_gam, newton_iteration="mgcv")
    )
    refuse = frontend_api._refuse_t2
    monkeypatch.setattr(
        frontend_api,
        "_refuse_t2",
        lambda call, parsed, **kw: refuse(call, parsed, select=False, discrete=False),
    )
    fit = hgam.gam(
        fx["formula"].replace("'", '"'),
        _columns(fx["data"]),
        FAMILY,
        weights="n",
        method="REML",
        select=bool(fx.get("select", False)),
        knots=fx["knots"] or None,
        control=control,
    )
    assert len(seen) == 1
    return fit, seen[0]


CASES = [
    "gam_binomial_cloglog_factor_t2_reml",
    "gam_binomial_cloglog_factor_t2_select_reml",
    "gam_binomial_cloglog_sz_select_reml",
    "gam_binomial_cloglog_cr_fs_select_reml",
    "gam_binomial_cloglog_by_numeric_reml",
]


@pytest.mark.parametrize("name", CASES)
def test_mgcv_newton_meets_mgcvs_default_fit(name: str, load_fixture, margin, monkeypatch) -> None:
    fx = load_fixture(name)
    traj = _case(load_fixture, name)
    assert traj["conv"] == fx["outer_conv"] == "full convergence"
    fit, selection = _mgcv_newton_fit(fx, monkeypatch)
    assert fit.converged
    assert selection.message == "full convergence"
    labels = _as_list(fx["labels"])
    assert list(fit.smooth_labels) == labels
    assert len(fit.lambda_) == len(fx["sp"] if isinstance(fx["sp"], list) else [fx["sp"]])

    data = _columns(fx["data"])
    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(_as_list(fx["edf"]), dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    margin(f"mgcv newton, {name}", "eta vs mgcv default, max abs", d_eta, "1e-3 abs", name)
    margin(f"mgcv newton, {name}", "se vs mgcv default, max rel", d_se, "1e-2 rel", name)
    margin(f"mgcv newton, {name}", "edf vs mgcv default, fraction of gate", d_edf, "0.05 abs or 1%", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
    if "newdata" in fx:
        eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
        d_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
        d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
        margin(f"mgcv newton, {name}", "predict eta vs mgcv default, max abs", d_new, "1e-3 abs", name)
        margin(f"mgcv newton, {name}", "predict se vs mgcv default, max rel", d_se_new, "1e-2 rel", name)
        assert d_new < 1e-3
        assert d_se_new < 1e-2

    # Reported, not gated: how closely the search followed mgcv's.
    margin(
        f"mgcv newton, {name}",
        "iterations, abs(hgam - mgcv) (reported)",
        abs(selection.n_iterations - traj["iter"]),
        "-",
        name,
    )
    mgcv_score = fit.reml_score + mgcv_reml_score_offset(
        binomial_cloglog(), data["p"], selection_blocks(fit), weights=data["n"]
    )
    margin(
        f"mgcv newton, {name}",
        "final REML score vs mgcv, abs (reported)",
        abs(mgcv_score - traj["score_hist"][-1]),
        "-",
        name,
    )


TIGHT = [
    "gam_binomial_cloglog_factor_t2_reml",
    "gam_binomial_cloglog_factor_t2_select_reml",
    "gam_binomial_cloglog_by_numeric_reml",
]


@pytest.mark.parametrize("name", TIGHT)
def test_mgcv_newton_at_a_tight_conv_tol_meets_mgcvs_tight_refit(
    name: str, load_fixture, margin, monkeypatch
) -> None:
    """The ridge cases at ``conv.tol = 1e-12``, like for like with mgcv's tight refit. The port reports full
    convergence where hgam's own search stalls on 27d3 (``tests/test_newton_conv_tol.py``)."""
    fx = load_fixture(name)
    tight = fx["tight"]
    assert tight["outer_conv"] == "full convergence"
    fit, selection = _mgcv_newton_fit(fx, monkeypatch, control={"newton": {"conv.tol": tight["conv_tol"]}})
    assert fit.converged
    assert selection.message == "full convergence"
    data = _columns(fx["data"])
    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(tight["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(tight["se"]) - 1.0)))
    labels = _as_list(fx["labels"])
    ref_edf = np.asarray(_as_list(tight["edf"]), dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    margin(f"mgcv newton tight, {name}", "eta vs mgcv tight, max abs", d_eta, "1e-3 abs", name)
    margin(f"mgcv newton tight, {name}", "se vs mgcv tight, max rel", d_se, "1e-2 rel", name)
    margin(f"mgcv newton tight, {name}", "edf vs mgcv tight, fraction of gate", d_edf, "0.05 abs or 1%", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
    if "fit_new" in tight:
        eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
        d_new = float(np.max(np.abs(eta_new - np.asarray(tight["fit_new"]))))
        d_se_new = float(np.max(np.abs(se_new / np.asarray(tight["se_new"]) - 1.0)))
        margin(f"mgcv newton tight, {name}", "predict eta vs mgcv tight, max abs", d_new, "1e-3 abs", name)
        margin(f"mgcv newton tight, {name}", "predict se vs mgcv tight, max rel", d_se_new, "1e-2 rel", name)
        assert d_new < 1e-3
        assert d_se_new < 1e-2


def selection_blocks(fit) -> tuple[np.ndarray, ...]:
    return tuple(fit.design["penalty_blocks"])


def test_mgcv_reml_score_offset_reproduces_gcv_ubre(load_fixture, monkeypatch) -> None:
    """hgam's REML score plus the offset is mgcv's ``m$gcv.ubre`` at mgcv's own sp (27b2, tier 1): the offset
    is ``-ls - Mp/2 log(2 pi)`` with ``Mp = 6`` here."""
    name = "gam_binomial_cloglog_factor_t2_reml"
    fx = load_fixture(name)
    fit, _ = _mgcv_newton_fit(fx, monkeypatch)
    data = _columns(fx["data"])
    offset = mgcv_reml_score_offset(binomial_cloglog(), data["p"], selection_blocks(fit), weights=data["n"])
    ls = saturated_log_likelihood(binomial_cloglog(), data["p"], data["n"])
    np.testing.assert_allclose(offset, -ls - 3.0 * np.log(2.0 * np.pi), rtol=1e-14)
    # The port stops at mgcv's sp, so the scores agree to the fit's own precision.
    np.testing.assert_allclose(fit.reml_score + offset, fx["reml"], rtol=1e-8)


def test_saturated_log_likelihood_poisson_and_refusal() -> None:
    """``fix.family.ls`` for poisson is ``sum(dpois(y, y, log = TRUE) * w)``; a free-scale family is
    refused."""
    y = np.array([0.0, 1.0, 3.0, 7.0])
    w = np.array([1.0, 2.0, 1.0, 0.5])
    expected = np.sum(
        (
            np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0)), 0.0)
            - y
            - np.array([0.0, 0.0, np.log(6.0), np.log(5040.0)])
        )
        * w
    )
    np.testing.assert_allclose(saturated_log_likelihood(poisson_log(), y, w), expected, rtol=1e-14)
    with pytest.raises(HgamNotImplementedError, match="gaussian"):
        saturated_log_likelihood(gam_model.resolve_family("gaussian", "identity"), y, w)


def test_newton_iteration_is_validated() -> None:
    with pytest.raises(HgamValidationError, match="iteration must be"):
        gam_model.newton_select_lambdas(
            np.ones(3), np.eye(3), poisson_log(), (np.eye(3),), x0=np.zeros(1), iteration="bfgs"
        )
