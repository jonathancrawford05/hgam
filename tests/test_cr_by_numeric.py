"""A numeric ``by`` on a cr smooth, held to mgcv (``make_fixtures.R`` section 10).

mgcv counterpart: ``smoothCon`` (smooth.r), mgcv 1.9-4, the numeric-``by`` branch. The penalty is scaled on
the un-multiplied ``X``, then ``X`` is multiplied row-wise by the by variable, and the sum-to-zero constraint
is dropped because the by variable is not constant, so the term keeps ``k`` coefficients. The exit formula
uses this as ``s(AttdAge, by = StudyYear_C, k = 13, bs = "cr")``.

The ``select=TRUE`` fit of section 10 is gated in ``tests/test_gam_reml_fixtures.py``. The fit without
``select`` is gated here on structure, per-term edf and standard errors against mgcv's default fit, and on the
fit at mgcv's smoothing parameters. Its linear predictor at the estimated smoothing parameters is gated
against mgcv refitted at a tight newton ``conv.tol``: mgcv's default search stops short of its own REML
optimum on that fixture (Jonathan's decision on PR #12, option B; ``docs/PRIORITIES.md``).
"""

import numpy as np
import pytest

from hgam.engine.gam_model import resolve_family
from hgam.engine.gam_reml_optimize import penalized_fit_and_score
from hgam.engine.gam_term_build import build_python_cr_term
from hgam.engine.gam_term_spec import TermSpec
from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

FORMULA = 'p ~ s(x, bs="cr", k=6) + s(x, by=z, bs="cr", k=6)'
FAMILY = 'binomial(link="cloglog")'


def _rel(got: np.ndarray, ref: list) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(got - ref_arr)) / np.max(np.abs(ref_arr)))


def _columns(fx: dict) -> dict[str, np.ndarray]:
    return {name: np.asarray([r[name] for r in fx["data"]], dtype=np.float64) for name in fx["data"][0]}


def test_by_numeric_basis_matches_smoothcon(load_fixture) -> None:
    fx = load_fixture("smoothcon_cr_by_numeric")
    term = TermSpec(
        label=fx["label"], variables=("x",), basis="cr", k=(6,), knots=(("x", tuple(fx["knots"])),), by="z"
    )
    got = build_python_cr_term(np.asarray(fx["x"]), term, by=np.asarray(fx["by"]))
    assert fx["C_rows"] == 0  # mgcv drops the constraint: the term keeps k columns
    assert got.design.shape == (len(fx["x"]), fx["bs_dim"])
    assert _rel(got.design, fx["X"]) < 1e-8
    assert _rel(got.s[0], fx["S"]) < 1e-8
    assert got.rank == (fx["rank"],)


def test_by_numeric_fit_meets_the_gates(load_fixture) -> None:
    fx = load_fixture("gam_binomial_cloglog_by_numeric_reml")
    data = _columns(fx)
    fit = gam(FORMULA, data, FAMILY, weights="n", method="REML", knots=fx["knots"])
    assert fit.converged

    # Structure: exact. mgcv labels the by smooth s(x):z and keeps its k coefficients.
    assert list(fit.smooth_labels) == fx["labels"] == ["s(x)", "s(x):z"]
    assert fit.coef.size == fx["n_coef"]
    assert fit.log_lambda.size == len(fx["sp"])

    # Per-term edf: 0.05 absolute or 1%, whichever is larger.
    ref_edf = np.asarray(fx["edf"], dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in fx["labels"]])
    assert np.all(np.abs(got_edf - ref_edf) <= np.maximum(0.05, 0.01 * ref_edf)), (got_edf, ref_edf)

    # Standard errors of the linear predictor: 1% relative.
    _, se = fit.predict(data, se_fit=True)
    np.testing.assert_allclose(se, fx["se"], rtol=1e-2)

    # Linear predictor at the estimated smoothing parameters: 1e-3 absolute, against mgcv refitted at a
    # tight newton conv.tol, because mgcv's default search stops short of its own REML optimum here
    # (docs/PRIORITIES.md). The fixture's own justification first: mgcv's tight REML score is not higher
    # than its default score.
    tight = fx["tight"]
    assert tight["reml"] <= fx["reml"]
    eta, se_tight = fit.predict(data, se_fit=True)
    assert np.max(np.abs(eta - np.asarray(tight["eta"]))) < 1e-3
    np.testing.assert_allclose(se_tight, tight["se"], rtol=1e-2)
    tight_edf = np.asarray(tight["edf"], dtype=np.float64)
    assert np.all(np.abs(got_edf - tight_edf) <= np.maximum(0.05, 0.01 * tight_edf)), (got_edf, tight_edf)

    # The fit at mgcv's own smoothing parameters (a deterministic piece): 1e-8, relative to the largest |eta|.
    family = resolve_family(fit.model.family, fit.model.link)
    coef, _ = penalized_fit_and_score(
        data["p"],
        fit.design["x"],
        family,
        fit.design["penalty_blocks"],
        np.log10(np.asarray(fx["sp"])),
        weights=data["n"],
    )
    assert _rel(fit.design["x"] @ coef, fx["eta"]) < 1e-8


def test_constant_numeric_by_is_refused() -> None:
    rng = np.random.default_rng(3)
    data = {"x": rng.uniform(1, 21, 200), "z": np.full(200, 2.0), "p": rng.uniform(0.01, 0.1, 200)}
    data["n"] = np.full(200, 50.0)
    with pytest.raises(HgamNotImplementedError, match="constant"):
        gam(FORMULA, data, FAMILY, weights="n", method="REML")
