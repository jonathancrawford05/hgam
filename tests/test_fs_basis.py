"""The ``fs`` factor-smooth basis on cr and on its default tp base, held to ``smoothCon`` (``make_fixtures.R``
sections 24 and 29).

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
import pytest

from hgam.engine.gam_basis_fs import fs_basis, fs_predict_design
from hgam.engine.gam_basis_tp import subspace_distance
from hgam.errors import HgamNotImplementedError, HgamValidationError

GATE = 1e-8
NAME_29E = "gam_fs_tp_rounding_spread"


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


def test_fs_tp_basis_matches_smoothcon(load_fixture, margin) -> None:
    """Section 29a: ``s(x, f, bs="fs", k=5)`` with no ``xt``, mgcv's default tp base. On a tp base the two
    null-space eigenvalues of ``nat.param``'s ``R^-T S R^-1`` are rounding noise, so rounding sets not just
    the order of the null pair but its rotation (on mgcv's side too: section 29e). What is determined is
    held at the deterministic-piece gate: each range column up to sign, the null pair's span (sine of the
    largest principal angle), the range penalty, and the sum of the null penalties (the identity on the
    pair, invariant under its rotation). ``P`` maps from the tp base's gauge, so it is not compared."""
    fx = load_fixture("smoothcon_fs_tp")
    assert fx["base_bs"] == "tprs.smooth"
    n_levels, k = fx["n_levels"], fx["k"]
    base_rank = k - 2
    got = fs_basis(np.asarray(fx["x"]), np.asarray(fx["group"]), n_levels, None, base="tp", k=k)
    ref_x = np.asarray(fx["X"])
    assert got.design.shape == ref_x.shape == (len(fx["x"]), k * n_levels)
    assert fx["C_rows"] == 0 and fx["null_space_dim"] == 0
    assert got.rank == tuple(fx["rank"]) == (base_rank * n_levels, n_levels, n_levels)
    assert len(got.penalties) == len(fx["S"]) == 1 + k - base_rank
    assert got.tp_state is not None
    x_new = fs_predict_design(
        np.asarray(fx["new_x"]), np.asarray(fx["new_group"]), n_levels, None, got.p_mat, got.tp_state
    )
    ref_new = np.asarray(fx["X_new"])

    rng_idx = np.concatenate([np.arange(base_rank) + level * k for level in range(n_levels)])
    signs = np.sign(np.sum(got.design[:, rng_idx] * ref_x[:, rng_idx], axis=0))
    rel_range = _rel(got.design[:, rng_idx] * signs, ref_x[:, rng_idx])
    rel_range_new = _rel(x_new[:, rng_idx] * signs, ref_new[:, rng_idx])
    null_angle = 0.0
    for level in range(n_levels):
        cols = np.arange(base_rank, k) + level * k
        for a, b in ((got.design, ref_x), (x_new, ref_new)):
            rows = np.any(b[:, cols] != 0.0, axis=1)
            null_angle = max(null_angle, subspace_distance(a[np.ix_(rows, cols)], b[np.ix_(rows, cols)]))
    rel_s_range = _rel(got.penalties[0], np.asarray(fx["S"][0]))
    rel_s_null = _rel(sum(got.penalties[1:]), sum(np.asarray(s) for s in fx["S"][1:]))
    for quantity, value in (
        ("X range columns", rel_range),
        ("X null pair, sine of angle", null_angle),
        ("range penalty", rel_s_range),
        ("summed null penalties", rel_s_null),
        ("PredictMat range columns", rel_range_new),
    ):
        margin("fs tp basis", f"{quantity}, max rel", value, "1e-8 rel", "smoothcon_fs_tp")
    assert rel_range < GATE
    assert null_angle < GATE
    assert rel_s_range < GATE
    assert rel_s_null < GATE
    assert rel_range_new < GATE


def test_fs_tp_rounding_spread_is_reported(load_fixture, margin) -> None:
    """Section 29e, the reason a tp-based fs stays refused in ``gam()``: mgcv's own fit moves under a constant
    shift of x that changes nothing in exact arithmetic, on a tp base and not on cr. Reported, not gated (it
    measures mgcv, not hgam)."""
    fx = load_fixture(NAME_29E)
    margin(
        "fs tp, mgcv's own spread",
        "eta under x + 1e-12..1e-9, tp base",
        max(fx["eta_shift_tp"]),
        "-",
        NAME_29E,
    )
    margin(
        "fs tp, mgcv's own spread",
        "eta under x + 1e-12..1e-9, cr base",
        max(fx["eta_shift_cr"]),
        "-",
        NAME_29E,
    )
    assert len(fx["eta_shift_tp"]) == len(fx["eta_shift_cr"]) == len(fx["shifts"]) == 4


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"base": "tp", "k": None}, HgamValidationError),
        ({"base": "ps", "k": 5}, HgamNotImplementedError),
    ],
)
def test_fs_basis_refuses_a_bad_base(kwargs: dict, error: type) -> None:
    x = np.linspace(0.0, 1.0, 30)
    with pytest.raises(error):
        fs_basis(x, np.arange(30) % 2, 2, None, **kwargs)


def test_fs_tp_basis_in_a_frame_is_refused() -> None:
    """tp in ``bam(discrete = TRUE)`` (built on ``mf0``) is not ported."""
    x = np.linspace(0.0, 1.0, 30)
    g = np.arange(30) % 2
    with pytest.raises(HgamNotImplementedError, match="discrete"):
        fs_basis(x, g, 2, None, frame=(x, g), base="tp", k=5)
