"""``by=`` on the one-covariate thin-plate spline, ``s(x, by=z)`` and ``s(x, by=f)``, held to mgcv
(``make_fixtures.R`` section 31).

mgcv counterparts: ``smoothCon`` (smooth.r, mgcv 1.9-4) handles ``by`` alike for every basis. The tp basis
(``smooth.construct.tp.smooth.spec``, ``construct_tprs``) is built on the whole covariate column, and
``scale.penalty`` acts on the un-multiplied design. A numeric ``by`` drops the sum-to-zero constraint
(``sd(by) > 0``) and multiplies each row by it; a factor ``by`` keeps the constraint on the whole design and
makes one copy per level, labelled ``s(x):f<level>``, each zero outside its level. In ``bam(discrete = TRUE)``
the basis is built on ``discrete.mf``'s ``mf0`` (section 30). hgam: ``build_python_tp_term(by=, group=)``
through :func:`hgam.engine.gam_term_build.tp_term_basis`.

Gates (``docs/PRIORITIES.md``): structure exact (labels, coefficient count, full rank, names, number of sp);
eta 1e-3 absolute; se of eta 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger; the same on
new data (x outside its range included).
"""

import numpy as np
import pytest

from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

FAMILY = 'binomial(link="cloglog")'
CASES = {
    # 31a: p ~ s(x, by=z2, k=8) + s(z, bs="cr", k=6)
    "gam_binomial_cloglog_tp_by_numeric_reml": {"method": "REML"},
    "gam_binomial_cloglog_tp_by_numeric_select_reml": {"method": "REML", "select": True},
    "bam_discrete_binomial_cloglog_tp_by_numeric": {"method": "fREML", "discrete": True},
    "bam_discrete_binomial_cloglog_tp_by_numeric_select": {
        "method": "fREML",
        "discrete": True,
        "select": True,
    },
    # 31b: p ~ g + s(x, by=g, k=6)
    "gam_binomial_cloglog_tp_by_factor_reml": {"method": "REML"},
    "gam_binomial_cloglog_tp_by_factor_select_reml": {"method": "REML", "select": True},
    "bam_discrete_binomial_cloglog_tp_by_factor": {"method": "fREML", "discrete": True},
    "bam_discrete_binomial_cloglog_tp_by_factor_select": {
        "method": "fREML",
        "discrete": True,
        "select": True,
    },
}


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_tp_by_meets_the_gates(name: str, load_fixture, margin) -> None:
    fx = load_fixture(name)
    data = _columns(fx["data"])
    fit = gam(fx["formula"].replace("'", '"'), data, FAMILY, weights="n", **CASES[name])
    assert fit.converged
    labels = fx["labels"]
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(fx["sp"])

    eta, se = fit.predict(data, se_fit=True)
    eta_err = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    se_err = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(fx["edf"], dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    edf_ratio = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    eta_new_err = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    se_new_err = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    sp_rel = float(np.max(np.abs(np.asarray(fit.lambda_) / np.asarray(fx["sp"]) - 1.0)))

    margin("tp by", "eta, max abs", eta_err, "1e-3 abs", name)
    margin("tp by", "se, max rel", se_err, "1e-2 rel", name)
    margin("tp by", "edf, fraction of gate", edf_ratio, "0.05 abs or 1%", name)
    margin("tp by predict", "eta, max abs", eta_new_err, "1e-3 abs", name)
    margin("tp by predict", "se, max rel", se_new_err, "1e-2 rel", name)
    margin("tp by", "sp, max rel (reported)", sp_rel, "not gated", name)
    assert eta_err < 1e-3
    assert se_err < 1e-2
    assert edf_ratio <= 1.0, (got_edf, ref_edf)
    assert eta_new_err < 1e-3
    assert se_new_err < 1e-2


def test_numeric_by_on_tp_drops_the_constraint(load_fixture) -> None:
    """smoothCon drops the sum-to-zero constraint for a varying numeric by: the term keeps all k columns
    (31a: k = 8), as mgcv's coefficient names show; a factor-by copy keeps it (k - 1 = 5 per level)."""
    fx = load_fixture("gam_binomial_cloglog_tp_by_numeric_reml")
    assert sum(name.startswith("s(x):z2.") for name in fx["coef_names"]) == 8
    fx = load_fixture("gam_binomial_cloglog_tp_by_factor_reml")
    assert sum(name.startswith("s(x):ga.") for name in fx["coef_names"]) == 5


def test_a_constant_numeric_by_on_tp_is_refused(load_fixture) -> None:
    """A constant by, for which smoothCon keeps the constraint, is not ported (as for cr)."""
    data = _columns(load_fixture("gam_binomial_cloglog_tp_by_numeric_reml")["data"])
    data["one"] = np.ones_like(data["x"])
    with pytest.raises(HgamNotImplementedError, match="constant"):
        gam('p ~ s(x, by=one, k=8) + s(z, bs="cr", k=6)', data, FAMILY, weights="n", method="REML")
