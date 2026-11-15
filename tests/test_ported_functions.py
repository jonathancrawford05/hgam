"""The six functions ported from mgcv source, held to fixtures from ``oracle/make_fixtures.R`` (section 4).

Gates are those of ``docs/PRIORITIES.md``: structure exact, deterministic pieces and a fit at fixed
smoothing parameters to 1e-8 relative, a fitted linear predictor to 1e-3 absolute, per-term edf to 0.05
or 1%, standard errors of the linear predictor to 1% relative.
"""

import numpy as np
import pytest

from hgam.engine.gam_family import binomial_cloglog, poisson_log
from hgam.engine.gam_initial_sp import initial_log10_lambda_from_xx_diag
from hgam.frontend.api import _tero
from hgam.frontend.bam_fitd import _fit_chol, bam_fitd, penalty_block_structure
from hgam.frontend.discrete import compress_covariates
from hgam.frontend.formula import TensorCall

DETERMINISTIC_RTOL = 1e-8


def _blocks(fx: dict) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(s, dtype=np.float64) for s in fx["penalty_blocks"])


def test_tero_reorders_margins_as_mgcv(load_fixture) -> None:
    """mgcv counterpart: ``tero`` (bam.r). Structure, so exact."""
    for case in load_fixture("tero_margin_order")["cases"]:
        out = _tero(TensorCall(variables=tuple(case["variables"]), k=tuple(case["k"])))
        assert list(out.variables) == case["term"], case
        assert list(out.k) == case["k_out"], case
        assert out.text == case["label"], case


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when the two labellings group the rows identically."""
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    return pairs.shape[0] == np.unique(a).size == np.unique(b).size


@pytest.mark.parametrize(
    "case",
    [
        "integer_identity",
        "one_d_at_threshold",
        "one_d_binned",
        "two_d_binned",
        "numeric_and_factor",
        "two_d_given_m",
    ],
)
def test_compress_covariates_matches_compress_df(load_fixture, case: str) -> None:
    """mgcv counterpart: ``compress.df`` (bam.r), including ``default_m``'s grid sizes. The cell
    count is structure (exact); each row's discretised value is deterministic (1e-8). mgcv shuffles the
    unique rows with ``sample()``, so cell labels are compared only as a partition."""
    fx = load_fixture("compress_df_cases")["cases"][case]
    columns = {name: np.asarray(v, dtype=np.float64) for name, v in fx["columns"].items()}
    factors = frozenset(fx["factors"] if isinstance(fx["factors"], list) else [fx["factors"]])
    m = fx["m"] if isinstance(fx["m"], int) else None
    term = compress_covariates(columns, factors, m=m)
    assert term.nr == fx["nr"]
    for name, expected in fx["row_values"].items():
        np.testing.assert_allclose(term.row_values(name), expected, rtol=DETERMINISTIC_RTOL, atol=1e-12)
    assert _same_partition(term.index, np.asarray(fx["index"]))


def test_initial_sp_from_xx_diag_matches_initial_sp(load_fixture) -> None:
    """mgcv counterpart: ``initial.sp`` with ``XX = TRUE`` (mgcv.r). Deterministic: 1e-8 relative."""
    fx = load_fixture("initial_sp_xx")
    got = initial_log10_lambda_from_xx_diag(np.asarray(fx["xx_diag"]), _blocks(fx))
    np.testing.assert_allclose(10.0**got, fx["sp"], rtol=DETERMINISTIC_RTOL)


def test_fit_chol_matches_sl_fitchol_at_fixed_sp(load_fixture) -> None:
    """mgcv counterpart: ``Sl.fitChol`` (fast-REML.r). A fit at fixed smoothing parameters: 1e-8
    relative on the coefficients, the REML gradient, the Newton step and ``PP``."""
    fx = load_fixture("sl_fitchol")
    blocks = _blocks(fx)
    got = _fit_chol(
        np.asarray(fx["xx"]),
        np.asarray(fx["f"]),
        blocks,
        penalty_block_structure(blocks),
        np.asarray(fx["rho"]),
        tol=0.0,
    )
    for name, value, expected in [
        ("beta", got.beta, fx["beta"]),
        ("grad", got.grad, fx["grad"]),
        ("step", got.step, fx["step"]),
        ("PP", got.pp, fx["PP"]),
    ]:
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(value, expected, rtol=0, atol=DETERMINISTIC_RTOL * scale, err_msg=name)


@pytest.mark.parametrize(
    ("name", "family"),
    [("bgam_fitd_poisson", poisson_log), ("bgam_fitd_binomial_cloglog", binomial_cloglog)],
)
def test_bam_fitd_matches_bgam_fitd_on_bams_design(load_fixture, name: str, family) -> None:  # type: ignore[no-untyped-def]
    """mgcv counterpart: ``bgam.fitd`` (bam.r), run on bam's own design and penalties. Key results at
    their gates: linear predictor 1e-3 absolute, per-term edf 0.05 or 1%, se of eta 1% relative."""
    fx = load_fixture(name)
    x = np.asarray(fx["X"], dtype=np.float64)
    fit = bam_fitd(
        x,
        _blocks(fx),
        np.asarray(fx["y"], dtype=np.float64),
        family(),
        offset=np.asarray(fx["offset"], dtype=np.float64),
        prior_weights=np.asarray(fx["weights"], dtype=np.float64),
    )
    assert fit.converged
    eta_diff = float(np.max(np.abs(fit.eta - np.asarray(fx["eta"]))))
    assert eta_diff < 1e-3, eta_diff
    for first, last, expected in zip(fx["term_first"], fx["term_last"], fx["edf"], strict=True):
        got = float(np.sum(fit.edf[first - 1 : last]))
        assert abs(got - expected) <= max(0.05, 0.01 * abs(expected)), (first, got, expected)
    se = np.sqrt(np.einsum("ij,jk,ik->i", x, fit.vp, x))
    np.testing.assert_allclose(se, fx["se"], rtol=0.01)
