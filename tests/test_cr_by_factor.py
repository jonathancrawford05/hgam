"""A factor ``by`` on a cr smooth, held to mgcv (``make_fixtures.R`` section 23).

mgcv counterpart: ``smoothCon`` (smooth.r), mgcv 1.9-4, the factor-``by`` branch. The smooth is built and
its sum-to-zero constraint taken on the whole, un-multiplied ``X``; then one copy is made per level of the
factor, ``X`` multiplied by that level's dummy, labelled ``s(x):f<level>``, and the constraint absorbed in
each copy, so every copy keeps ``k - 1`` coefficients. hgam builds these through ``factor_by_terms`` and
``build_python_cr_by_factor_term``.

Gates (``docs/PRIORITIES.md``): the basis to 1e-8 relative (a deterministic piece); structure exact; the
linear predictor to 1e-3 absolute; standard errors of the linear predictor 1% relative; per-term edf 0.05
absolute or 1%, whichever is larger. Fits: 23b ``gam(method="REML")``, 23c ``bam(method="fREML",
discrete=TRUE)``, 23d the same with ``select=TRUE`` and ``predict`` on new data.
"""

import numpy as np
import pytest

from hgam.engine.gam_term_build import build_python_cr_by_factor_term
from hgam.engine.gam_term_spec import factor_by_terms
from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam

FORMULA = 'p ~ f + s(x, by=f, bs="cr", k=6)'
FAMILY = 'binomial(link="cloglog")'
EDF_GATE = "0.05 abs or 1%"


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _rel(got: np.ndarray, ref: list) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(got - ref_arr)) / np.max(np.abs(ref_arr)))


def test_by_factor_basis_matches_smoothcon(load_fixture, margin) -> None:
    fx = load_fixture("smoothcon_cr_by_factor")
    specs = factor_by_terms(
        base_label="s(x)",
        variable="x",
        k=6,
        by_factor="f",
        n_levels=len(fx["by_levels"]),
        knots=tuple(fx["knots"]),
        level_labels=tuple(f"f{lv}" for lv in fx["by_levels"]),
    )
    assert [s.label for s in specs] == fx["labels"] == ["s(x):fa", "s(x):fb", "s(x):fc"]
    for j, spec in enumerate(specs):
        got = build_python_cr_by_factor_term(np.asarray(fx["x"]), np.asarray(fx["group"]), spec)
        assert got.design.shape == np.asarray(fx["X"][j]).shape  # k - 1 columns: the constraint is kept
        rel_x, rel_s = _rel(got.design, fx["X"][j]), _rel(got.s[0], fx["S"][j])
        margin("factor by basis", "X and S, max rel", max(rel_x, rel_s), "1e-8 rel", "smoothcon_cr_by_factor")
        assert rel_x < 1e-8
        assert rel_s < 1e-8
        assert got.rank == (fx["rank"][j],)


CASES = {
    "gam_binomial_cloglog_by_factor_reml": {"method": "REML"},
    "bam_discrete_binomial_cloglog_by_factor": {"method": "fREML", "discrete": True},
    "bam_discrete_binomial_cloglog_by_factor_select": {"method": "fREML", "discrete": True, "select": True},
}


@pytest.mark.parametrize("name", list(CASES))
def test_by_factor_fit_meets_the_gates(name: str, load_fixture, margin) -> None:
    fx = load_fixture(name)
    data = _columns(fx["data"])
    fit = gam(FORMULA, data, FAMILY, weights="n", knots=fx["knots"], **CASES[name])
    assert fit.converged

    # Structure: exact, including mgcv's coefficient names and full rank.
    assert list(fit.smooth_labels) == fx["labels"]
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert list(fit.coefficient_names) == fx["coef_names"]

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(fx["edf"], dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in fx["labels"]])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    margin("factor by fit", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("factor by fit", "se, max rel", d_se, "1e-2 rel", name)
    margin("factor by fit", "edf, fraction of gate", d_edf, EDF_GATE, name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)

    if "newdata" in fx:  # 23d: predict.bam on new data (predict.bamd re-runs discrete.mf)
        eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
        d_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
        d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
        margin("factor by predict", "eta, max abs", d_new, "1e-3 abs", name)
        margin("factor by predict", "se, max rel", d_se_new, "1e-2 rel", name)
        assert d_new < 1e-3
        assert d_se_new < 1e-2


class _OrderedCategorical:
    """The two attributes hgam reads from an ordered categorical (pandas' ``Categorical(ordered=True)``)."""

    ordered = True

    def __init__(self, values: list[str]) -> None:
        self._values = np.asarray(values)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return self._values


def test_ordered_factor_is_refused(load_fixture) -> None:
    fx = load_fixture("bam_discrete_binomial_cloglog_by_factor")
    data: dict[str, object] = dict(_columns(fx["data"]))
    data["f"] = _OrderedCategorical(list(data["f"]))
    with pytest.raises(HgamNotImplementedError, match="ordered factor"):
        gam(FORMULA, data, FAMILY, weights="n", knots=fx["knots"], method="REML")
