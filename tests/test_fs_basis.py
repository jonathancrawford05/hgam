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
from hgam.engine.gam_basis_tp import subspace_distance

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


def test_fs_tp_basis_matches_smoothcon_up_to_the_null_pair(load_fixture, margin) -> None:
    """Section 29a: ``smoothCon(s(x, f, bs="fs", k=6))``, mgcv's default tp base, against
    ``fs_basis(base="tp")``.

    The range columns are held column by column (up to sign), like cr's. The two null-space directions of
    ``R^-T S R^-1`` have eigenvalues at rounding level (about 1e-20), so LAPACK may return any rotation of
    that pair, not only a swap: they are held as a subspace (the sine of the largest principal angle), with
    the null-space penalties' sum, which is the identity on that subspace whatever the rotation. ``P`` is not
    compared: it maps the tp base's own coordinates, which carry tp's eigenvector gauge
    (``tests/test_tp_fit.py``). An mgcv fit is not invariant to a rotation of the null pair (measured: 0.1 rad
    moves eta by 1.2e-3 on both bases), and on a tp base rounding does rotate it (x perturbed at 1e-15 moves the
    fit by up to 3e-2; the cr fit does not move), so a fit on this base is refused until Jonathan decides
    (PR #55)."""
    fx = load_fixture("smoothcon_fs_tp")
    n_levels, k = fx["n_levels"], fx["k"]
    base_rank = k - 2
    got = fs_basis(np.asarray(fx["x"]), np.asarray(fx["group"]), n_levels, None, base="tp", k=k)
    ref_x = np.asarray(fx["X"])
    assert got.design.shape == ref_x.shape == (len(fx["x"]), k * n_levels)
    assert fx["C_rows"] == 0 and fx["null_space_dim"] == 0
    assert got.rank == tuple(fx["rank"]) == (base_rank * n_levels, n_levels, n_levels)
    assert len(got.penalties) == len(fx["S"]) == 3

    x_new = fs_predict_design(
        np.asarray(fx["new_x"]), np.asarray(fx["new_group"]), n_levels, None, got.p_mat, got.tp_state
    )
    ref_new = np.asarray(fx["X_new"])
    rng = np.arange(base_rank)
    null = np.arange(base_rank, k)
    rel_range, rel_new, d_null, d_null_new = 0.0, 0.0, 0.0, 0.0
    for level in range(n_levels):
        cols, ncols = rng + level * k, null + level * k
        rows = np.asarray(fx["group"]) == level
        signs = np.sign(np.sum(got.design[:, cols] * ref_x[:, cols], axis=0))
        rel_range = max(rel_range, _rel(got.design[:, cols] * signs, ref_x[:, cols]))
        rel_new = max(rel_new, _rel(x_new[:, cols] * signs, ref_new[:, cols]))
        d_null = max(d_null, subspace_distance(got.design[rows][:, ncols], ref_x[rows][:, ncols]))
        new_rows = np.asarray(fx["new_group"]) == level
        d_null_new = max(
            d_null_new, subspace_distance(x_new[new_rows][:, ncols], ref_new[new_rows][:, ncols])
        )
    ref_s = [np.asarray(s) for s in fx["S"]]
    rel_s_range = _rel(got.penalties[0], ref_s[0])
    rel_s_null = _rel(got.penalties[1] + got.penalties[2], ref_s[1] + ref_s[2])

    for quantity, value in (
        ("X range columns, max rel", rel_range),
        ("X null pair, subspace sine", d_null),
        ("range penalty, max rel", rel_s_range),
        ("summed null penalties, max rel", rel_s_null),
        ("PredictMat range columns, max rel", rel_new),
        ("PredictMat null pair, subspace sine", d_null_new),
    ):
        margin("fs basis (smoothcon_fs_tp)", quantity, value, "1e-8 rel", "smoothcon_fs_tp")
    assert rel_range < GATE
    assert d_null < GATE
    assert rel_s_range < GATE
    assert rel_s_null < GATE
    assert rel_new < GATE
    assert d_null_new < GATE
