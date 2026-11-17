"""The ``fs`` factor-smooth interaction basis on a ``cr`` base (``s(x, f, bs="fs", xt="cr")``).

mgcv counterparts (smooth.r, mgcv 1.9-4): ``nat.param`` (type 1, the QR branch, ``unit.fnorm = TRUE``),
``smooth.construct.fs.smooth.spec`` (the non-``gamm`` branch), ``Predict.matrix.fs.interaction``, and the
penalty scaling of ``smoothCon``. Held to ``smoothcon_fs_cr`` (``make_fixtures.R`` section 24) in
``tests/test_fs_basis.py``.

The construction, as mgcv runs it:

1. The base smooth is built by ``smooth.construct`` (not ``smoothCon``), so its penalty is the unscaled
   ``cr`` penalty (:func:`~hgam.engine.gam_basis_cr._cr_basis_raw`), on every row and with no constraint.
2. ``nat.param(X, S, rank, type = 1)`` re-parameterises it: ``X = QR``; the eigen-decomposition of
   ``R^-T S R^-1`` (``eigen(symmetric = TRUE)``, LAPACK ``dsyevr``; SciPy's ``driver="evr"`` is the same
   routine) gives ``X Q V`` and ``P = R^-1 V`` with the range columns first; type 1 divides each range column
   by ``sqrt`` of its eigenvalue, so the range penalty is the identity; then the range and the null columns
   are each rescaled to unit mean square, and the range penalty becomes ``scale^2 I``.
3. The design is that ``k``-column basis repeated once per factor level, each copy multiplied by the level's
   dummy. Penalties: one range penalty ``diag(rep(c(D, 0, 0), nf))`` and one penalty per null-space
   direction, the unit diagonal on that direction in every level. ``rank`` is
   ``c(rank * nf, rep(nf, null.d))``. No identifiability constraint (``C`` has no rows).
4. ``smoothCon`` divides each penalty by ``norm(S) / norm(X, "I")^2``.

What is not reproducible element for element: the null space of ``R^-T S R^-1`` has two eigenvalues at
rounding level (about 1e-19), so their order and signs are set by rounding. With ``dsyevr`` the two
directions agree with mgcv's up to order and sign (measured on section 24), and an mgcv fit is unchanged
by that swap (its two null-space smoothing parameters swap). Another eigen routine (``dsyevd``, NumPy's
``eigh``) rotates the pair, which changes the fit; it is not used.
"""

import numpy as np
import scipy.linalg

from hgam.engine.gam_basis_cr import _cr_basis_raw, _r_norm_inf, _r_norm_one
from hgam.errors import HgamComputationError, HgamValidationError

__all__ = ["FsBasis", "fs_basis", "fs_predict_design", "nat_param"]

_NAT_PARAM_TOL = float(np.finfo(np.float64).eps ** 0.8)


def nat_param(
    design: np.ndarray, penalty: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mgcv counterpart: ``nat.param`` (smooth.r), mgcv 1.9-4, ``type = 1``, ``unit.fnorm = TRUE``.

    R's ``qr()`` is LINPACK's ``dqrdc2``; NumPy's is LAPACK's ``dgeqrf``. They may differ in the sign of
    rows of ``R``, which leaves ``X`` and ``P`` unchanged up to the sign of eigenvectors.

    Args:
        design: The base model matrix ``X``, ``(n, p)``, full column rank.
        penalty: Its penalty ``S``, ``(p, p)``.
        rank: The penalty rank (the base smooth's ``rank``).

    Returns:
        ``(X, D, P)``: the re-parameterised ``(n, p)`` design with the ``rank`` range columns first, the
        ``rank`` diagonal entries of the range penalty, and ``P`` with ``X = design @ P``.
    """
    design = np.asarray(design, dtype=np.float64)
    q, r = np.linalg.qr(design)
    diag = np.abs(np.diag(r))
    if diag.min() <= diag.max() * _NAT_PARAM_TOL:
        # mgcv warns "smooth model matrix not full rank" and continues; that path is not ported.
        raise HgamComputationError("nat_param: the smooth model matrix is not full rank.")
    p = design.shape[1]
    if not 1 <= rank <= p:
        raise HgamValidationError(f"nat_param: rank must lie in [1, {p}]; got {rank}.")
    # forwardsolve(t(R), t(forwardsolve(t(R), t(S)))): R^-T S R^-1.
    inner = scipy.linalg.solve_triangular(r.T, np.asarray(penalty, dtype=np.float64).T, lower=True)
    rsr = scipy.linalg.solve_triangular(r.T, inner.T, lower=True)
    values, vectors = scipy.linalg.eigh(rsr, driver="evr")
    values, vectors = values[::-1], vectors[:, ::-1]  # eigen() returns decreasing values
    d = values[:rank]
    x_new = q @ vectors
    p_mat = scipy.linalg.solve_triangular(r, vectors)
    # type 1: the range penalty becomes the identity.
    e = np.concatenate([np.sqrt(d), np.ones(p - rank)])
    p_mat = p_mat / e
    x_new = x_new / e
    d = np.ones(rank)
    # unit.fnorm: range and null columns each to unit mean square.
    scale = 1.0 / np.sqrt(np.mean(x_new[:, :rank] ** 2))
    x_new[:, :rank] *= scale
    p_mat[:, :rank] *= scale
    d = d * scale**2
    if rank < p:
        scale_f = 1.0 / np.sqrt(np.mean(x_new[:, rank:] ** 2))
        x_new[:, rank:] *= scale_f
        p_mat[:, rank:] *= scale_f
    return x_new, d, p_mat


class FsBasis:
    """An ``fs`` term's model matrix, penalties and re-parameterisation, as ``smoothCon`` returns them."""

    def __init__(
        self, design: np.ndarray, penalties: tuple[np.ndarray, ...], rank: tuple[int, ...], p_mat: np.ndarray
    ) -> None:
        self.design = design
        """``(n, k * n_levels)``: level blocks of ``k`` columns, range columns first in each."""
        self.penalties = penalties
        """``1 + null_dim`` penalties: the range penalty, then one per null-space direction."""
        self.rank = rank
        """``(base_rank * n_levels, n_levels, ...)``, mgcv's ``rank``."""
        self.p_mat = p_mat
        """``(k, k)``: ``nat.param``'s ``P``, kept for prediction."""


def _expand(base: np.ndarray, group: np.ndarray, n_levels: int) -> np.ndarray:
    n, p = base.shape
    out = np.zeros((n, p * n_levels), dtype=np.float64)
    for level in range(n_levels):
        out[:, level * p : (level + 1) * p] = base * (group == level)[:, np.newaxis]
    return out


def _check_group(x: np.ndarray, group: np.ndarray, n_levels: int) -> None:
    if group.shape != x.shape:
        raise HgamValidationError(
            f"fs basis: x has shape {x.shape} but group has shape {group.shape}; "
            "one level code per row is needed."
        )
    if group.size and (group.min() < 0 or group.max() >= n_levels):
        raise HgamValidationError(f"fs basis: group codes must lie in [0, {n_levels}).")


def fs_basis(x: np.ndarray, group: np.ndarray, n_levels: int, knots: np.ndarray) -> FsBasis:
    """mgcv counterpart: ``smooth.construct.fs.smooth.spec`` (smooth.r) with ``xt = "cr"``, then
    ``smoothCon``'s penalty scaling, mgcv 1.9-4 (module docstring).

    Args:
        x: The smoothed covariate, ``(n,)``.
        group: 0-indexed level code per row (``as.integer(fac) - 1``), ``(n,)``.
        n_levels: ``length(levels(fac))``; every level gets a copy, observed or not, as in mgcv.
        knots: The cr knots, ``(k,)``.
    """
    x = np.asarray(x, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    _check_group(x, group, n_levels)
    base_x, base_s = _cr_basis_raw(x, np.asarray(knots, dtype=np.float64))
    k = base_x.shape[1]
    base_rank = k - 2  # smooth.construct.cr.smooth.spec: rank = k - 2, null.space.dim = 2
    xb, d, p_mat = nat_param(base_x, base_s, base_rank)
    null_d = k - base_rank

    design = _expand(xb, group, n_levels)
    unscaled = [np.diag(np.tile(np.concatenate([d, np.zeros(null_d)]), n_levels))]
    for i in range(null_d):
        unit = np.zeros(k)
        unit[base_rank + i] = 1.0
        unscaled.append(np.diag(np.tile(unit, n_levels)))
    # smoothCon: S[[i]] / (norm(S[[i]]) / norm(X, "I")^2).
    max_xx = _r_norm_inf(design) ** 2
    if max_xx == 0.0:
        raise HgamComputationError("fs_basis: the design matrix is identically zero.")
    penalties = tuple(s / (_r_norm_one(s) / max_xx) for s in unscaled)
    rank = (base_rank * n_levels, *([n_levels] * null_d))
    return FsBasis(design, penalties, rank, p_mat)


def fs_predict_design(
    x: np.ndarray, group: np.ndarray, n_levels: int, knots: np.ndarray, p_mat: np.ndarray
) -> np.ndarray:
    """mgcv counterpart: ``Predict.matrix.fs.interaction`` (smooth.r), mgcv 1.9-4: the base cr prediction
    matrix at new ``x`` times the fit's ``P``, one copy per level."""
    x = np.asarray(x, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    _check_group(x, group, n_levels)
    base_x, _ = _cr_basis_raw(x, np.asarray(knots, dtype=np.float64))
    return _expand(base_x @ np.asarray(p_mat, dtype=np.float64), group, n_levels)
