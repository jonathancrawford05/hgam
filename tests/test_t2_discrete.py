"""The ``t2`` tensor product on cr margins in ``bam(method="fREML", discrete=TRUE)``, held to mgcv
(``make_fixtures.R`` section 27c).

mgcv counterparts (mgcv 1.9-4): ``bam`` calls ``gam.setup(..., modCon = 2)`` (bam.r), so ``smoothCon``
(smooth.r) replaces t2's sparse fit constraint by the prediction one (``sm$C <- sm$Cp``, ``colSums(X)``) and
fit and prediction parameterisations are one; bam applies no ``G$P``. A ``t2.smooth.spec`` is not a
``tensor.smooth.spec``: ``tero`` leaves it alone and ``discrete.mf`` (``nmarg = 1``) compresses its
covariates jointly, one ``compress.df`` call, whose ``uniquecombs(ordered = TRUE)`` orders multi-column rows
by their text (``paste0(x, "*", z)``, C locale) before ``sample()`` shuffles them. ``gam.setup`` builds the
term on the padded frame ``mf0``: default knots, each margin's ``nat.param`` ``P``, the penalty scaling and
the constraint come from its rows (``t2_basis(frame=, mod_con=2)``).

Gates (``docs/PRIORITIES.md``): structure exact; the linear predictor to 1e-3 absolute; standard errors of the
linear predictor 1% relative; per-term edf 0.05 absolute or 1%, whichever is larger; the same on new data
(points outside both knot ranges too). Deterministic pieces (compress.df's rows, the constraint) at 1e-8.
"""

import numpy as np
import pytest

from hgam.errors import HgamNotImplementedError
from hgam.frontend.api import gam
from hgam.frontend.discrete import compress_covariates, r_double_string
from hgam.frontend.r_rng import RRandom

FAMILY = 'binomial(link="cloglog")'

CASES = (
    "bam_discrete_binomial_cloglog_t2",
    "bam_discrete_binomial_cloglog_factor_t2",
    "bam_discrete_binomial_cloglog_t2_repeated",
    "bam_discrete_binomial_cloglog_t2_binned",
)


def _columns(rows: list[dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = np.asarray(values) if isinstance(values[0], str) else np.asarray(values, dtype=np.float64)
    return out


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


def _fit(fx: dict):
    return gam(
        fx["formula"].replace("'", '"'),
        _columns(fx["data"]),
        FAMILY,
        weights="n",
        knots=fx["knots"] or None,
        method="fREML",
        discrete=True,
    )


@pytest.mark.parametrize("name", CASES)
def test_discrete_t2_meets_the_gates(name: str, load_fixture, margin) -> None:
    """27c1 ``p ~ t2(x, z, k=c(5, 4))`` (x's knots supplied), 27c2 ``p ~ f + t2(x, z)``, and 27c3
    ``p ~ t2(x, z, k=c(6, 5))`` on data whose (x, z) pairs repeat (the unique pairs are shuffled and
    padded), and 27c5 with more than 100^2 unique pairs, so compress.df snaps both columns to its grid.
    Structure (label in written order, coefficient count, full rank, names, three sp) is exact; eta, se and
    edf meet their gates on the data and on new data."""
    fx = load_fixture(name)
    data = _columns(fx["data"])
    fit = _fit(fx)
    labels = _as_list(fx["labels"])
    assert fit.converged
    assert list(fit.smooth_labels) == labels
    assert fit.coef.size == fx["n_coef"] == fx["rank"]
    assert not fit.pivoted_columns
    assert list(fit.coefficient_names) == fx["coef_names"]
    assert len(fit.lambda_) == len(fx["sp"]) == 3

    eta, se = fit.predict(data, se_fit=True)
    d_eta = float(np.max(np.abs(eta - np.asarray(fx["eta"]))))
    d_se = float(np.max(np.abs(se / np.asarray(fx["se"]) - 1.0)))
    ref_edf = np.asarray(_as_list(fx["edf"]), dtype=np.float64)
    got_edf = np.array([fit.edf_per_term[label] for label in labels])
    d_edf = float(np.max(np.abs(got_edf - ref_edf) / np.maximum(0.05, 0.01 * ref_edf)))
    eta_new, se_new = fit.predict(_columns(fx["newdata"]), se_fit=True)
    d_new = float(np.max(np.abs(eta_new - np.asarray(fx["fit_new"]))))
    d_se_new = float(np.max(np.abs(se_new / np.asarray(fx["se_new"]) - 1.0)))
    margin("t2 discrete", "eta, max abs", d_eta, "1e-3 abs", name)
    margin("t2 discrete", "se, max rel", d_se, "1e-2 rel", name)
    margin("t2 discrete", "edf, fraction of gate", d_edf, "0.05 abs or 1%", name)
    margin("t2 discrete predict", "eta, max abs", d_new, "1e-3 abs", name)
    margin("t2 discrete predict", "se, max rel", d_se_new, "1e-2 rel", name)
    assert d_eta < 1e-3
    assert d_se < 1e-2
    assert d_edf <= 1.0, (got_edf, ref_edf)
    assert d_new < 1e-3
    assert d_se_new < 1e-2


@pytest.mark.parametrize("name", CASES)
def test_discrete_t2_constraint_is_mgcvs(name: str, load_fixture, margin) -> None:
    """The t2 term's constraint is mgcv's ``Cp`` (``modCon = 2``) taken over ``mf0``'s rows: mgcv's unit
    constraint direction (``qr.Q(attr(sm, "qrc"))[, 1]``) equals ``colSums`` of hgam's raw design over the
    frame, up to sign, at 1e-8. The comparison is on absolute values because each margin's ``nat.param``
    eigenvectors carry a sign that rounding sets (``tests/test_t2_basis.py``); on 27c3 one margin column
    has the opposite sign to mgcv's, which flips the matching raw columns and leaves the fit unchanged.
    Over the data rows instead, 27c3 misses by 1.9e-2."""
    fx = load_fixture(name)
    fit = _fit(fx)
    state = next(s for s in fit.term_states if s.term.basis == "t2")
    want = np.asarray(fx["constraint"], dtype=np.float64)
    z = state.constraint[0]
    # Z's columns span the null space of hgam's unit Cp: recover it as the orthogonal complement.
    q, _ = np.linalg.qr(z, mode="complete")
    got = q[:, -1]
    miss = float(np.max(np.abs(np.abs(got) - np.abs(want))))
    margin("t2 discrete", "|Cp| vs mgcv, max abs", miss, "1e-8", name)
    assert miss < 1e-8


def test_discrete_t2_coefficient_output_is_refused(load_fixture) -> None:
    """Coefficient-level output for a t2 term in a discrete bam is not measured yet: refused by name."""
    fit = _fit(load_fixture("bam_discrete_binomial_cloglog_t2"))
    with pytest.raises(HgamNotImplementedError, match="bam\\(discrete=True\\)"):
        fit.vcov()
    with pytest.raises(HgamNotImplementedError, match="bam\\(discrete=True\\)"):
        fit.summary()


def test_joint_compress_df_rows_are_mgcvs(load_fixture, margin) -> None:
    """mgcv counterpart: ``compress.df`` on two columns (bam.r) under ``set.seed(8547)``, section 27c4. The
    values repeat and are written in mixed notation (``1e+05``, ``1e-04``, ``0.001``, ``123456``), so
    ``uniquecombs``' text order differs from numeric order; the shuffled unique rows and each data row's
    index match mgcv's exactly (structure)."""
    fx = load_fixture("compress_df_joint_order")
    columns = {k: np.asarray(v, dtype=np.float64) for k, v in fx["columns"].items()}
    term = compress_covariates(columns, rng=RRandom(8547))
    assert term.nr == fx["nr"]
    np.testing.assert_array_equal(term.values["x"], fx["x"])
    np.testing.assert_array_equal(term.values["z"], fx["z"])
    np.testing.assert_array_equal(term.index + 1, fx["index"])
    sorted_numeric = np.unique(np.column_stack([columns["x"], columns["z"]]), axis=0)
    text = sorted(set(fx["text"]))
    assert [f"{r_double_string(a)}*{r_double_string(b)}" for a, b in sorted_numeric] != text


def test_joint_compress_df_grid_branch_is_mgcvs(load_fixture, margin) -> None:
    """mgcv counterpart: ``compress.df`` on t2's two covariates with more than ``100^2`` unique pairs
    (27c5, 12000 continuous rows, PR #47 review): each column is snapped to 100 points over its range, then
    ``uniquecombs``' text order and ``sample()``'s shuffle. The row count and every index match exactly
    (structure); the grid values agree to the deterministic gate (they differ from R's ``seq`` in the last
    bit)."""
    fx = load_fixture("bam_discrete_binomial_cloglog_t2_binned")
    data = _columns(fx["data"])
    term = compress_covariates({"x": data["x"], "z": data["z"]}, rng=RRandom(8547))
    assert term.binned
    assert term.nr == fx["compress_nr"]
    np.testing.assert_array_equal(term.index + 1, fx["compress_index"])
    for name in ("x", "z"):
        want = np.asarray(fx[f"compress_{name}"], dtype=np.float64)
        rel = float(np.max(np.abs(term.values[name] - want)) / np.max(np.abs(want)))
        margin(
            "t2 discrete",
            "compress.df grid rows, max rel",
            rel,
            "1e-8 rel",
            "bam_discrete_binomial_cloglog_t2_binned",
        )
        assert rel < 1e-8


def test_r_double_string_is_paste0(load_fixture) -> None:
    """R counterpart: ``as.character`` on a double as ``paste0`` writes it (R >= 4.3.0), on 27c4's values."""
    fx = load_fixture("compress_df_joint_order")
    got = [
        f"{r_double_string(a)}*{r_double_string(b)}"
        for a, b in zip(fx["columns"]["x"], fx["columns"]["z"], strict=True)
    ]
    assert got == fx["text"]
    assert [r_double_string(v) for v in (1e5, 123456.0, 1e-4, 0.001, 0.1 + 0.2, -2.5, 0.0)] == [
        "1e+05",
        "123456",
        "1e-04",
        "0.001",
        "0.3",
        "-2.5",
        "0",
    ]
