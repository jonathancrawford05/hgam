"""Closed-form verification of :mod:`hgam.engine.gam_dispersion` and its
exposure on :class:`~hgam.engine.gam_model.EngineGAMFit` (ladder slice 3c)."""

import numpy as np
import pytest

from hgam.engine.gam_dispersion import dispersion_estimates
from hgam.engine.gam_family import gaussian_identity, poisson_log
from hgam.errors import HgamValidationError


def test_poisson_pearson_and_fletcher_closed_form() -> None:
    y = np.array([0.0, 2.0, 5.0, 3.0])
    mu = np.array([1.0, 2.0, 4.0, 4.0])
    d = dispersion_estimates(y, mu, poisson_log(), edf_total=1.0)
    # Pearson stat = (0-1)^2/1 + 0 + (5-4)^2/4 + (3-4)^2/4 = 1.5 ; df = 3
    np.testing.assert_allclose(d.pearson, 1.5 / 3.0, rtol=1e-14)
    # s_bar = mean(1 * (y - mu) / mu) = mean(-1, 0, .25, -.25) = -0.25
    np.testing.assert_allclose(d.s_bar, -0.25, rtol=1e-14)
    np.testing.assert_allclose(d.fletcher, 0.5 / 0.75, rtol=1e-14)
    assert d.residual_df == 3.0


def test_poisson_deviance_closed_form() -> None:
    y = np.array([1.0, 3.0])
    mu = np.array([2.0, 2.0])
    d = dispersion_estimates(y, mu, poisson_log(), edf_total=0.5)
    dev = 2.0 * (1.0 * np.log(1.0 / 2.0) - (1.0 - 2.0) + 3.0 * np.log(3.0 / 2.0) - (3.0 - 2.0))
    np.testing.assert_allclose(d.deviance, dev / 1.5, rtol=1e-12)


def test_gaussian_fletcher_equals_pearson_equals_rss_over_df() -> None:
    """V' = 0 so s_bar = 0 and Fletcher collapses to Pearson = RSS / (n - edf)."""
    rng = np.random.default_rng(3)
    y = rng.normal(size=20)
    mu = np.zeros(20)
    d = dispersion_estimates(y, mu, gaussian_identity(), edf_total=4.0)
    rss = float(np.sum(y**2))
    assert d.s_bar == 0.0
    np.testing.assert_allclose(d.pearson, rss / 16.0, rtol=1e-14)
    np.testing.assert_allclose(d.fletcher, d.pearson, rtol=1e-14)
    np.testing.assert_allclose(d.deviance, d.pearson, rtol=1e-12)


def test_weights_enter_the_pearson_sum_only() -> None:
    y = np.array([0.0, 2.0, 5.0, 3.0])
    mu = np.array([1.0, 2.0, 4.0, 4.0])
    w = np.array([2.0, 1.0, 1.0, 1.0])
    d = dispersion_estimates(y, mu, poisson_log(), edf_total=1.0, weights=w)
    np.testing.assert_allclose(d.pearson, (2.0 * 1.0 + 0.25 + 0.25) / 3.0, rtol=1e-14)
    np.testing.assert_allclose(d.s_bar, -0.25, rtol=1e-14)  # unweighted mean, as gam.fit3


def test_s_bar_is_floored_at_minus_point_nine() -> None:
    y = np.zeros(4)
    mu = np.full(4, 5.0)  # (y - mu)/mu = -1 -> floored
    d = dispersion_estimates(y, mu, poisson_log(), edf_total=1.0)
    assert d.s_bar == -0.9
    np.testing.assert_allclose(d.fletcher, d.pearson / 0.1, rtol=1e-12)


@pytest.mark.parametrize("edf", [4.0, 5.0])
def test_no_residual_df_raises(edf: float) -> None:
    with pytest.raises(HgamValidationError, match="residual degrees"):
        dispersion_estimates(np.ones(4), np.ones(4), poisson_log(), edf_total=edf)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(HgamValidationError, match="equal-length"):
        dispersion_estimates(np.ones(4), np.ones(3), poisson_log(), edf_total=1.0)


def test_fit_dispersion_is_nan_not_an_error_when_no_residual_df() -> None:
    """The fit's diagnostic must never block the fit (PR #248 review [P2])."""
    from hgam.engine.gam_model import _dispersion_or_nan

    d = _dispersion_or_nan(np.ones(4), np.ones(4), poisson_log(), 5.0, None)
    assert np.isnan(d.pearson) and np.isnan(d.fletcher) and np.isnan(d.deviance)
    # saturated mu (V = 0 under a binomial-like variance) does not raise either
    ok = _dispersion_or_nan(np.ones(4), np.ones(4), poisson_log(), 1.0, None)
    assert np.isfinite(ok.pearson)
