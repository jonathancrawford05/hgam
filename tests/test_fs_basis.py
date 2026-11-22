"""The ``fs`` factor-smooth basis on cr and on tp, held to ``smoothCon`` (``make_fixtures.R`` sections 24
and 29).

mgcv counterparts: ``smooth.construct.fs.smooth.spec`` and ``nat.param`` (type 1),
``Predict.matrix.fs.interaction`` and ``smoothCon``'s penalty scaling (smooth.r, mgcv 1.9-4), ported in
``hgam.engine.gam_basis_fs``.

The gate is the deterministic-piece gate, 1e-8 relative. Eigenvectors are defined up to sign, and the two
null-space directions of the cr or tp base have eigenvalues at rounding level, so their order is set by
rounding as well. The test finds one column map (a sign per column, and the order of the two null columns)
from the first level's block, and holds the design of every level, ``P`` (cr only: a tp base's own coordinates
carry its eigenvector gauge), the prediction matrix and the penalty list to mgcv under that one map. An mgcv
fit is unchanged by such a map: its two null-space smoothing parameters swap. The fitted term is held to REML
gam fits in ``tests/test_fs_fit.py`` (section 24b) and ``tests/test_fs_tp_fit.py`` (section 29).
"""

import itertools

import numpy as np
import pytest

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


@pytest.mark.parametrize("name", ["smoothcon_fs_cr", "smoothcon_fs_tp"])
def test_fs_basis_matches_smoothcon(name: str, load_fixture, margin) -> None:
    """On a cr base at supplied knots (section 24a), and on mgcv's default tp base (section 29a)."""
    fx = load_fixture(name)
    n_levels = fx["n_levels"]
    if name == "smoothcon_fs_tp":
        knots = None
        k = fx["k"]
        got = fs_basis(np.asarray(fx["x"]), np.asarray(fx["group"]), n_levels, None, base="tp", k=k)
    else:
        knots = np.asarray(fx["knots"])
        k = len(knots)
        got = fs_basis(np.asarray(fx["x"]), np.asarray(fx["group"]), n_levels, knots)
    ref_x = np.asarray(fx["X"])
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
    # P maps the base smooth's coordinates. A tp base's are fixed only up to its eigenvector gauge (Lanczos in
    # mgcv, LAPACK here; tests/test_tp_fit.py), so P is compared for cr only; for tp the gauge-invariant
    # product, base design times P, is the design and the prediction matrix held below.
    rel_p = _rel(got.p_mat[:, order] * signs, np.asarray(fx["P"])) if knots is not None else 0.0

    # Penalties: the map's order applied to rows and columns; the null penalties follow their columns.
    penalty_order = [0, *(1 + order[base_rank:] - base_rank)]
    rel_s = max(
        _rel(got.penalties[penalty_order[i]][np.ix_(full_order, full_order)], np.asarray(fx["S"][i]))
        for i in range(len(fx["S"]))
    )

    x_new = fs_predict_design(
        np.asarray(fx["new_x"]), np.asarray(fx["new_group"]), n_levels, knots, got.p_mat, got.tp_state
    )
    rel_new = _rel(x_new[:, full_order] * full_signs, np.asarray(fx["X_new"]))

    checks = [("X", rel_x), ("S", rel_s), ("PredictMat", rel_new)] + (
        [("P", rel_p)] if knots is not None else []
    )
    for quantity, value in checks:
        margin(f"fs basis ({name})", f"{quantity}, max rel", value, "1e-8 rel", name)
    assert rel_x < GATE
    assert rel_p < GATE
    assert rel_s < GATE
    assert rel_new < GATE
