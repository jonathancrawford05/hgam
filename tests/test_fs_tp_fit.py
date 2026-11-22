"""The ``fs`` factor-smooth term on mgcv's default tp base in REML gam fits, held to mgcv (``make_fixtures.R``
section 29).

mgcv counterpart: ``smooth.construct.fs.smooth.spec`` (smooth.r, mgcv 1.9-4) with ``xt`` absent, so
``base.bs`` is ``"tp"``: the base is ``smooth.construct.tp.smooth.spec`` on every row
(:func:`~hgam.engine.gam_basis_tp.tp_construct`), then ``nat.param(type = 1)`` as on a cr base
(``tests/test_fs_basis.py``). Prediction is ``Predict.matrix.fs.interaction`` over
``Predict.matrix.tprs.smooth``.

Gates (``docs/PRIORITIES.md``): structure exact; the linear predictor to 1e-3 absolute; standard errors of the
linear predictor 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger; on the data and on new
data with x outside its range. Smoothing parameters are not gated: the two null-space directions are ordered
by rounding, so their smoothing parameters may come out swapped.
"""

import numpy as np
import pytest

from hgam import bam
from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

FAMILY = 'binomial(link="cloglog")'
ETA_ATOL = 1e-3
SE_RTOL = 1e-2

CASES = (
    # 29b: section 24b's model on the default tp base.
    "gam_binomial_cloglog_fs_tp_reml",
    # 29c: the global-plus-group-curves form, s(x) + s(x, f, bs="fs"), both tp.
    "gam_binomial_cloglog_tp_fs_tp_reml",
    # 29d: 29c with select=TRUE (a null-space penalty for s(x); none for fs, whose penalties sum to full rank).
    "gam_binomial_cloglog_tp_fs_tp_select_reml",
)


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


@pytest.mark.parametrize("name", CASES)
def test_fs_tp_fit_meets_the_gates(name: str, load_fixture, margin) -> None:
    fx = load_fixture(name)
    data = _columns(fx["data"])
    select = bool(fx.get("select", False))
    fit = gam(fx["formula"].replace("'", '"'), data, FAMILY, weights="n", method="REML", select=select)
    assert fit.converged

    labels = _as_list(fx["labels"])
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(fx["sp"])

    eta, se = fit.predict(data, se_fit=True)
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    d_eta_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    ref_edf = np.asarray(_as_list(fx["edf"]), dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))

    margin(name, "eta, max abs", d_eta, "1e-3 abs", name)
    margin(name, "se, max rel", d_se, "1e-2 rel", name)
    margin(name, "eta new data, max abs", d_eta_new, "1e-3 abs", name)
    margin(name, "se new data, max rel", d_se_new, "1e-2 rel", name)
    margin(name, "edf per term (fraction of gate)", d_edf, "1.0", name)
    assert d_eta < ETA_ATOL
    assert d_se < SE_RTOL
    assert d_eta_new < ETA_ATOL
    assert d_se_new < SE_RTOL
    assert d_edf <= 1.0, (got_edf, ref_edf)


@pytest.mark.parametrize(
    "formula", ['p ~ s(x, f, bs="fs", k=6)', 'p ~ s(x, f, bs="fs", xt=list(bs="tp"), k=6)']
)
def test_written_tp_base_is_the_default(formula: str, load_fixture) -> None:
    """``xt="tp"`` and ``xt=list(bs="tp")`` name the base mgcv takes when ``xt`` is absent."""
    fx = load_fixture("gam_binomial_cloglog_fs_tp_reml")
    data = _columns(fx["data"])
    default = gam('p ~ s(x, f, bs="fs", xt="tp", k=6)', data, FAMILY, weights="n", method="REML")
    written = gam(formula, data, FAMILY, weights="n", method="REML")
    np.testing.assert_allclose(written.predict(data), default.predict(data), rtol=0, atol=1e-12)


def test_fs_tp_refusals(load_fixture) -> None:
    """Not ported for a tp base: supplied knots (mgcv's tp places its own) and discrete bam (tp is refused
    there)."""
    data = _columns(load_fixture("gam_binomial_cloglog_fs_tp_reml")["data"])
    with pytest.raises(HgamNotImplementedError, match="knots= for 'x'"):
        gam(
            'p ~ s(x, f, bs="fs", k=6)',
            data,
            FAMILY,
            weights="n",
            method="REML",
            knots={"x": [1, 5, 9, 13, 17, 21]},
        )
    with pytest.raises(HgamNotImplementedError, match="fs term on its default tp base"):
        bam('p ~ s(x, f, bs="fs", k=6)', data, FAMILY, weights="n", method="fREML", discrete=True)
