"""The ``t2`` tensor product basis on cr margins, held to ``smoothCon`` (``make_fixtures.R`` section 27).

mgcv counterparts: ``smooth.construct.t2.smooth.spec``, ``t2.model.matrix``, ``Predict.matrix.t2.smooth``,
``nat.param`` (type 3) and ``smoothCon``'s scaling and constraints (smooth.r, mgcv 1.9-4), ported in
``hgam.engine.gam_basis_t2``. The basis is a deterministic piece: structure exact (rank, null.space.dim,
penalty labels, column count), X, S and PredictMat at 1e-8 relative.

Each margin's ``nat.param`` comes from eigenvectors, whose signs a LAPACK build may set differently (measured:
on section 27's three-margin case one column of the k = 3 margin has the opposite sign to mgcv 1.9-1's, with
both runs using ``dsyevr``). A sign flip of a column is an orthogonal re-parameterisation, which leaves
every fitted quantity unchanged. So the raw quantities (margin ``P``, X before absorption, ``C`` and ``Cp``)
are held element for element up to one sign per column, and the absorbed ones (X, S under the fit
constraint; Xp, Sp and PredictMat under ``Cp``) up to an orthogonal map ``R`` that the test solves for and
checks.
"""

import numpy as np
import pytest

from hgam.engine.gam_basis_fs import nat_param
from hgam.engine.gam_basis_t2 import t2_basis, t2_model_matrix, t2_predict_design
from hgam.engine.gam_select_penalty import null_space_penalty
from hgam.errors import HgamNotImplementedError

GATE = 1e-8

CASES = {
    "smoothcon_t2_cr": (("x", "z"), (5, 4), ([1.0, 3.0, 7.0, 12.0, 21.0], None)),
    "smoothcon_t2_cr_three": (("x", "z", "w"), (4, 4, 3), (None, None, None)),
}


def _column(frame: list[dict], variable: str) -> np.ndarray:
    return np.asarray([row[variable] for row in frame], dtype=np.float64)


def _rel(got: np.ndarray, ref: np.ndarray) -> float:
    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


def _signs(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """One sign per column, aligning ``got`` to ``ref``."""
    signs = np.sign(np.sum(np.asarray(got) * np.asarray(ref), axis=0))
    assert np.all(signs != 0.0)
    return signs


def _orthogonal_map(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``R`` with ``got @ R = ref`` (least squares over the rows); the caller checks it."""
    return np.linalg.lstsq(got, np.asarray(ref, dtype=np.float64), rcond=None)[0]


@pytest.mark.parametrize("name", sorted(CASES))
def test_t2_basis_matches_smoothcon(name, load_fixture, margin) -> None:
    fx = load_fixture(name)
    variables, ks, knots = CASES[name]
    xs = [_column(fx["data"], v) for v in variables]
    got = t2_basis(xs, ks, [None if kn is None else np.asarray(kn) for kn in knots])

    p = int(np.prod(ks))
    assert fx["label"] == f"t2({','.join(variables)})"
    assert got.design.shape == np.asarray(fx["X"]).shape == (len(xs[0]), p - 1)
    assert got.design_p.shape == np.asarray(fx["Xp"]).shape
    assert fx["n_cons"] == 1
    assert got.rank == tuple(fx["rank"]) == tuple(fx["rank_raw"])
    assert got.null_space_dim == fx["null_space_dim"] == fx["null_space_dim_raw"] - 1
    assert got.p_lab == tuple(fx["p_lab"])
    assert len(got.penalties) == len(fx["S"]) == len(fx["Sp"]) == len(fx["rank"])

    rel_knots = max(_rel(g, r) for g, r in zip(got.knots, fx["margin_knots"], strict=True))
    # Raw pieces, element for element up to one sign per column.
    rel_p = max(_rel(g * _signs(g, r), r) for g, r in zip(got.margin_p, fx["margin_P"], strict=True))
    x_raw = np.asarray(fx["X_raw"])
    got_raw = got.design_raw
    signs = _signs(got_raw, x_raw)
    rel_x_raw = _rel(got_raw * signs, x_raw)
    # The column signs are those of the margins' P columns, multiplied: t2.model.matrix forms row products.
    margin_signs = [_signs(g, r) for g, r in zip(got.margin_p, fx["margin_P"], strict=True)]
    ranks = [len(s) - 2 for s in margin_signs]
    product_signs, _, _ = t2_model_matrix([s.reshape(1, -1) for s in margin_signs], ranks)
    np.testing.assert_array_equal(product_signs[0], signs)
    nup = sum(got.rank)
    got_cp = got_raw.sum(axis=0)  # smooth.construct.t2.smooth.spec: Cp = colSums(X)
    got_c = np.concatenate([np.zeros(nup), got_cp[nup:]])  # C: colSums of the unpenalised block only
    rel_c = _rel(got_c * signs, np.asarray(fx["C_raw"]).ravel())
    rel_cp = _rel(got_cp * signs, np.asarray(fx["Cp_raw"]).ravel())
    np.testing.assert_allclose(got_c @ got.z_fit, 0.0, rtol=0, atol=1e-10 * np.abs(got_c).max())
    np.testing.assert_allclose(got_cp @ got.z_pred, 0.0, rtol=0, atol=1e-10 * np.abs(got_cp).max())

    # Absorbed pieces, up to an orthogonal re-parameterisation R.
    rel_abs = {}
    for label, design, penalties, ref_x, ref_s in (
        ("X, S (fit constraint)", got.design, got.penalties, fx["X"], fx["S"]),
        ("Xp, Sp (Cp)", got.design_p, got.penalties_p, fx["Xp"], fx["Sp"]),
    ):
        r = _orthogonal_map(design, ref_x)
        np.testing.assert_allclose(r.T @ r, np.eye(r.shape[0]), rtol=0, atol=1e-10)
        rel_abs[label] = max(
            _rel(design @ r, ref_x),
            *(_rel(r.T @ s @ r, sr) for s, sr in zip(penalties, ref_s, strict=True)),
        )
        if label.startswith("Xp"):
            new = [_column(fx["newdata"], v) for v in variables]
            rel_abs["PredictMat"] = _rel(t2_predict_design(new, got) @ r, fx["X_new"])

    values = {
        "knots": rel_knots,
        "margin P (signs)": rel_p,
        "X raw (signs)": rel_x_raw,
        "C": rel_c,
        "Cp": rel_cp,
    }
    values.update({f"{k} (orthogonal map)": v for k, v in rel_abs.items()})
    for quantity, value in values.items():
        margin("t2 basis", f"{quantity}, max rel", value, "1e-8 rel", name)
        assert value < GATE, quantity


@pytest.mark.parametrize("mod_con", [0, 2])
def test_t2_null_space_penalty_matches_smoothcon(mod_con: int, load_fixture, margin) -> None:
    """Section 27d1: ``smoothCon(t2(x, z, bs = "cr", k = c(5, 4)), null.space.penalty = TRUE, modCon = )``.
    smoothCon's ``need.full`` branch appends ``U U'`` over the null space of the summed (scaled, constrained)
    penalties: rank 3, the 2 x 2 all-null block less the constraint, and ``null.space.dim`` becomes 0. It is
    built in the fit parameterisation: gam's sparse ``C`` (``modCon = 0``) and bam's ``Cp`` (``modCon = 2``).
    hgam's :func:`null_space_penalty` on ``t2_basis``'s penalties gives the same fourth penalty, held with the
    first three up to the orthogonal map ``R`` that takes hgam's design to mgcv's (eigenvector signs; module
    docstring)."""
    fx = load_fixture("smoothcon_t2_cr_select")[f"mod_con_{mod_con}"]
    xs = [_column(fx["data"], v) for v in ("x", "z")]
    got = t2_basis(xs, (5, 4), [np.asarray([1.0, 3.0, 7.0, 12.0, 21.0]), None], mod_con=mod_con)
    extra = null_space_penalty(got.penalties)
    assert extra is not None
    s_null, null_rank = extra
    assert len(fx["S"]) == 4
    assert list(fx["rank"]) == [*got.rank, null_rank] == [6, 4, 6, 3]
    assert fx["null_space_dim"] == 0
    r = _orthogonal_map(got.design, fx["X"])
    np.testing.assert_allclose(r.T @ r, np.eye(r.shape[0]), rtol=0, atol=1e-10)
    rel_x = _rel(got.design @ r, fx["X"])
    rel_s = max(_rel(r.T @ g @ r, sr) for g, sr in zip(got.penalties, fx["S"][:3], strict=True))
    rel_null = _rel(r.T @ s_null @ r, fx["S"][3])
    name, tag = "smoothcon_t2_cr_select", f"modCon {mod_con}"
    margin("t2 select", f"X (orthogonal map), {tag}, max rel", rel_x, "1e-8 rel", name)
    margin("t2 select", f"S, {tag}, max rel", rel_s, "1e-8 rel", name)
    margin("t2 select", f"null-space penalty, {tag}, max rel", rel_null, "1e-8 rel", name)
    assert rel_x < GATE
    assert rel_s < GATE
    assert rel_null < GATE


def test_t2_model_matrix_blocks() -> None:
    """``t2.model.matrix`` on two margins with ranks 2 and 1: blocks rr (2 x 1), nr (1 x 1), rn (2 x 1), nn
    (1 x 1, unpenalised, last and unlabelled), each a row tensor product with the earlier margin slowest."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(7, 3)), rng.normal(size=(7, 2))
    x, sub_cols, labels = t2_model_matrix([a, b], [2, 1])
    assert sub_cols == (2, 1, 2)
    assert labels == ("rr", "nr", "rn")
    expect = np.column_stack(
        [
            a[:, 0] * b[:, 0],
            a[:, 1] * b[:, 0],
            a[:, 2] * b[:, 0],
            a[:, 0] * b[:, 1],
            a[:, 1] * b[:, 1],
            a[:, 2] * b[:, 1],
        ]
    )
    np.testing.assert_allclose(x, expect, rtol=0, atol=0)
    with pytest.raises(HgamNotImplementedError, match="full"):
        t2_model_matrix([a, b], [2, 1], full=True)
    with pytest.raises(HgamNotImplementedError, match="ord"):
        t2_model_matrix([a, b], [2, 1], ord=[1, 2])


def test_nat_param_type3_constant_first_null_column() -> None:
    """``nat.param(type = 3)``: ``X = design @ P``, range penalty ``scale^2 I``, and the first null column
    is constant (mgcv writes the centred cross-product's eigenvectors in reverse, ``rind``)."""
    from hgam.engine.gam_basis_cr import _cr_basis_raw

    x = np.random.default_rng(4).uniform(0.0, 10.0, 200)
    raw, penalty = _cr_basis_raw(x, np.linspace(0.0, 10.0, 6))
    x_np, d, p_mat = nat_param(raw, penalty, 4, type=3)
    np.testing.assert_allclose(x_np, raw @ p_mat, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        p_mat.T @ penalty @ p_mat, np.diag(np.concatenate([d, [0.0, 0.0]])), atol=1e-10
    )
    assert np.ptp(x_np[:, 4]) < 1e-12 * np.max(np.abs(x_np[:, 4]))
    with pytest.raises(HgamNotImplementedError, match="type = 2"):
        nat_param(raw, penalty, 4, type=2)
