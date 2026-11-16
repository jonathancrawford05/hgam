"""Closed-form and R-parity checks for :mod:`hgam.engine.gam_initial_sp`
(``mgcv``'s ``initial.sp`` recipe, ladder slice 3d)."""

import numpy as np
import pytest

from hgam.engine.gam_family import binomial_logit, gaussian_identity, poisson_log
from hgam.engine.gam_initial_sp import initial_log10_lambda
from hgam.errors import HgamValidationError


def test_single_block_closed_form() -> None:
    # y = 0.9 -> mu = 1.0 (y + 0.1) -> W = mu = 1; X = 2 I -> diag(X'WX) = 4.
    y = np.full(3, 0.9)
    x = 2.0 * np.eye(3)
    for scale in (1.0, 100.0):
        s = scale * np.eye(3)
        got = initial_log10_lambda(y, x, poisson_log(), (s,))
        # def_sp = 4/scale; ldss = def_sp*diag(S) = 4 -> ratio 0.5, already >= 0.4
        np.testing.assert_allclose(got, [np.log10(4.0 / scale)], rtol=1e-12)


def test_balance_loop_moves_lambda_by_decades() -> None:
    # Two blocks on disjoint columns with different data size: ratio ends in [0.4, 4].
    y = np.full(4, 0.9)
    x = np.diag([1.0, 1.0, 5.0, 5.0])
    s1 = np.zeros((4, 4))
    s1[:2, :2] = np.eye(2)
    s2 = np.zeros((4, 4))
    s2[2:, 2:] = np.eye(2)
    got = initial_log10_lambda(y, x, poisson_log(), (s1, s2))
    lam = 10.0**got
    ldxx = np.array([1.0, 1.0, 25.0, 25.0])
    ldss = np.array([lam[0], lam[0], lam[1], lam[1]])
    assert np.mean(ldxx / (ldxx + ldss)) >= 0.4
    assert np.mean(ldxx / (ldxx + 10.0 * ldss)) < 0.4  # one decade more would undershoot
    # relative spacing before rescaling is sizeXX/sizeS per block: 1 vs 25
    np.testing.assert_allclose(got[1] - got[0], np.log10(25.0), rtol=1e-12)


def test_gaussian_and_binomial_initialisation_run() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    s = np.eye(4)
    assert initial_log10_lambda(rng.normal(size=30), x, gaussian_identity(), (s,)).shape == (1,)
    yb = (rng.uniform(size=30) > 0.5).astype(float)
    assert initial_log10_lambda(yb, x, binomial_logit(), (s,)).shape == (1,)


def test_zero_block_raises() -> None:
    with pytest.raises(HgamValidationError, match="all zero"):
        initial_log10_lambda(np.ones(3), np.eye(3), poisson_log(), (np.zeros((3, 3)),))


def test_fit_engine_gam_rejects_initial_sp_start_with_x0_or_multistart() -> None:
    from dataclasses import replace

    from multiterm_spec import multiterm_model_spec

    from hgam.engine.gam_model import fit_engine_gam

    model = replace(
        multiterm_model_spec((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 3.0)), family="quasipoisson", link="log"
    )
    with pytest.raises(HgamValidationError, match="mutually exclusive"):
        fit_engine_gam(model, {}, np.ones(3), initial_sp_start=True, x0=np.zeros(1))
    with pytest.raises(HgamValidationError, match="mutually exclusive"):
        fit_engine_gam(model, {}, np.ones(3), initial_sp_start=True, multistart=True)
