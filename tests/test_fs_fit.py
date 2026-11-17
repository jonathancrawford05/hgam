"""The ``fs`` factor-smooth term in a REML gam fit, held to mgcv (``make_fixtures.R`` section 24b).

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
        ('p ~ s(x, f, bs="fs", xt="cr", k=6)', {"select": True}, "select=True"),
        ('p ~ s(x, f, bs="fs", xt="cr", k=6)', {"method": "fREML", "discrete": True}, "discrete=True"),
    ],
)
def test_fs_refusals(small, formula: str, kwargs: dict, match: str) -> None:
    opts = {"method": "REML", **kwargs}
    with pytest.raises(HgamNotImplementedError, match=match):
        gam(formula, small, FAMILY, weights="n", **opts)


def test_fs_refused_by_bam(small) -> None:
    with pytest.raises(HgamNotImplementedError, match="discrete=True"):
        bam('p ~ s(x, f, bs="fs", xt="cr", k=6)', small, FAMILY, weights="n", method="fREML", discrete=True)
