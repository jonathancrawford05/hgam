"""The ``fs`` factor-smooth basis on cr, held to ``smoothCon`` (``make_fixtures.R`` section 24).

mgcv counterparts: ``smooth.construct.fs.smooth.spec`` and ``nat.param`` (type 1),
``Predict.matrix.fs.interaction`` and ``smoothCon``'s penalty scaling (smooth.r, mgcv 1.9-4), ported in
``hgam.engine.gam_basis_fs``.

The gate is the deterministic-piece gate, 1e-8 relative. Eigenvectors are defined up to sign, and the two
null-space directions of the cr base have eigenvalues at rounding level, so their order is set by rounding as
well. The test finds one column map (a sign per column, and the order of the two null columns) from the first
level's block, and holds the design of every level, ``P``, the prediction matrix and the penalty list to mgcv
under that one map. An mgcv fit is unchanged by such a map: its two null-space smoothing parameters swap. The
fitted term is held to a REML gam fit in ``tests/test_fs_fit.py`` (section 24b).
"""

import itertools

import numpy as np

from hgam.engine.gam_basis_fs import fs_basis, fs_predict_design

GATE = 1e-8


def _rel(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


def _column_map(got: np.ndarray, ref: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """The order and signs that carry hgam's ``k`` columns onto mgcv's: range columns in place, the null
    pair in either order; each column's sign from its inner product with mgcv's."""
    k = got.shape[1]
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for null_order in itertools.permutations(range(rank, k)):
        order = np.array([*range(rank), *null_order])
        signs = np.sign(np.sum(got[:, order] * ref, axis=0))
        err = _rel(got[:, order] * signs, ref)
        if best is None or err < best[0]:
            best = (err, order, signs)
    assert best is not None
    return best[1], best[2]


def test_fs_basis_matches_smoothcon(load_fixture, margin) -> None:
    fx = load_fixture("smoothcon_fs_cr")
    n_levels, knots = fx["n_levels"], np.asarray(fx["knots"])
    got = fs_basis(np.asarray(fx["x"]), np.asarray(fx["group"]), n_levels, knots)
    ref_x = np.asarray(fx["X"])
    k = len(knots)
    base_rank = k - 2

    # Structure: k columns per level, no constraint, mgcv's rank vector.
    assert got.design.shape == ref_x.shape == (len(fx["x"]), k * n_levels)
    assert fx["C_rows"] == 0 and fx["null_space_dim"] == 0
    assert got.rank == tuple(fx["rank"]) == (base_rank * n_levels, n_levels, n_levels)
    assert len(got.penalties) == len(fx["S"]) == 1 + k - base_rank

    order, signs = _column_map(got.design[:, :k], ref_x[:, :k], base_rank)
    full_order = np.concatenate([order + level * k for level in range(n_levels)])
    full_signs = np.tile(signs, n_levels)
    rel_x = _rel(got.design[:, full_order] * full_signs, ref_x)
    rel_p = _rel(got.p_mat[:, order] * signs, np.asarray(fx["P"]))

    # Penalties: the map's order applied to rows and columns; the null penalties follow their columns.
    penalty_order = [0, *(1 + order[base_rank:] - base_rank)]
    rel_s = max(
        _rel(got.penalties[penalty_order[i]][np.ix_(full_order, full_order)], np.asarray(fx["S"][i]))
        for i in range(len(fx["S"]))
    )

    x_new = fs_predict_design(
        np.asarray(fx["new_x"]), np.asarray(fx["new_group"]), n_levels, knots, got.p_mat
    )
    rel_new = _rel(x_new[:, full_order] * full_signs, np.asarray(fx["X_new"]))

    for quantity, value in (("X", rel_x), ("P", rel_p), ("S", rel_s), ("PredictMat", rel_new)):
        margin("fs basis", f"{quantity}, max rel", value, "1e-8 rel", "smoothcon_fs_cr")
    assert rel_x < GATE
    assert rel_p < GATE
    assert rel_s < GATE
    assert rel_new < GATE
