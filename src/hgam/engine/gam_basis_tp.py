"""Wood's thin-plate regression spline basis, one covariate (``bs="tp"``).

mgcv counterpart: ``smooth.construct.tp.smooth.spec`` (smooth.r) and its C routine ``construct_tprs``
(tprs.c), mgcv 1.9-4; LAPACK's symmetric eigensolver replaces mgcv's Lanczos routine. Held to ``smoothCon``
and to REML gam fits in ``tests/test_tp_fit.py`` (``make_fixtures.R`` section 28). ``mgcv``'s default smooth:
a bare ``s(x)`` is ``bs="tp"``, ``k = 10`` (``M + 8`` with the null-space dimension
``M = 2``).

What this module computes: the
thin-plate regression spline ``design_X`` and ``penalty_S`` from the covariate and
``k`` alone (the eigen-truncation recipe below); ``mgcv`` computes them via
``smoothCon(s(x, bs="tp", k=k), absorb.cons=TRUE)`` (its ``construct_tprs`` C routine);
compared on the **column space of the design** and on **the penalty in a common basis**
(:func:`compare_tp_basis`), not element-wise — see "The gauge" below.

The recipe, pinned by reading ``mgcv:::smooth.construct.tp.smooth.spec`` and then
measuring ``smoothCon`` output against it, not guessed:

1. ``shift = mean(x)``; work with ``x - shift``. The distinct shifted values ``xu``
   (``nu`` of them) carry the basis; the data rows only evaluate it.
2. ``E[i, j] = c · |xu_i - xu_j|³`` with ``c = Γ(-3/2) / (2⁴ √π) = 1/12`` (the
   ``m = 2``, ``d = 1`` thin-plate kernel, ``η(r) ∝ r^(2m-d)``). The null space is
   ``T = [1, xu]`` (``M = 2``).
3. **Truncate first, constrain second** (Wood 2003, §2.2; the name ``UZ`` is "U times Z"):
   eigendecompose the full ``E``, keep the ``k`` eigenpairs of largest ``|λ|``, then
   impose ``Tᵀδ = 0`` on ``δ = U_k δ_k`` by ``Z`` = the null space of ``T'U_k`` (full
   Householder QR of its transpose). The range basis is ``U_k Z`` and its penalty is
   ``Zᵀ diag(λ_k) Z`` (NOT diagonal). Truncating ``Z'EZ`` instead gives a different
   subspace (measured sine of the principal angle 0.10-0.17, not rounding).
4. Every column of the design ``[E(x, xu), T(x)] · UZ`` is rescaled to Euclidean norm
   ``√n`` (``n`` = number of data rows, not ``nu``), the penalty by the same factors.
5. ``smoothCon``'s ``scale.penalty`` (the same rescaling as :func:`cr_basis`), then the
   sum-to-zero identifiability constraint (:func:`absorb_sum_to_zero_constraint`).

The gauge. The range-space coordinates are determined only up to an invertible ``(k-2)²``
matrix: the sign and the order of the eigenvectors come out of ``mgcv``'s Lanczos routine
and of LAPACK differently, and the null space ``Z`` of ``T'U_k`` depends on both. Any
two such parameterisations give the same penalised fit (``η``, ``edf``, REML score), so
they are one model, but their ``X`` and ``S`` do not agree element by element. The
comparison is therefore gauge-invariant: the sine of the largest principal angle
between the two column spaces, and ``S`` in the basis that maps one design to the other.
``mgcv``'s truncated eigenvectors are themselves a Lanczos result, accurate to about
``1e-8`` on some cases (measured); the tolerance is derived from that, not chosen
(:func:`derived_tolerance`).

**Refused, by name** (:class:`~hgam.errors.HgamNotImplementedError`): ``mgcv`` randomly subsamples
``max.knots = 2000`` of the distinct covariate values when there are more than 2000 data rows *and* more
than 2000 distinct values (``temp.seed(1)`` and ``sample()``); that stream is not ported, so :func:`tp_basis`
raises rather than return a different basis.
"""

from dataclasses import dataclass
from math import gamma, pi, sqrt

import numpy as np

from hgam.engine.gam_basis_cr import _r_norm_inf, _r_norm_one
from hgam.errors import HgamComputationError, HgamNotImplementedError, HgamValidationError

__all__ = [
    "TP_MAX_KNOTS",
    "TP_NULL_DIM",
    "TprsState",
    "eigenspace_rounding_bound",
    "subspace_distance",
    "tp_basis",
    "tp_construct",
    "tp_predict",
]

TP_NULL_DIM = 2
"""Null-space dimension ``M`` of the default ``m = 2``, ``d = 1`` thin-plate penalty
(constants and linear functions are unpenalised)."""

TP_MAX_KNOTS = 2000
"""``mgcv``'s ``max.knots`` default: above it (rows AND distinct values) it subsamples."""

_ETA_CONST = gamma(-1.5) / (2.0**4 * sqrt(pi))
"""``Γ(d/2 - m) / (2^{2m} π^{d/2} (m-1)!)`` at ``m = 2, d = 1`` (``mgcv``'s ``eta.const``
for odd ``d``); equals ``1/12``."""

_EVAL_CHUNK = 2048


@dataclass(frozen=True)
class TprsState:
    """What a ``tp`` term must remember to be evaluated at new covariate values — the
    analogue of ``smooth$shift``, ``$Xu`` and ``$UZ``."""

    shift: float
    xu: np.ndarray
    uz: np.ndarray


def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(_ETA_CONST * np.abs(a[:, None] - b[None, :]) ** 3, dtype=np.float64)


def _eval_unique(xs: np.ndarray, state: TprsState) -> np.ndarray:
    """``[E(xs, xu), 1, xs] · UZ`` for distinct shifted values ``xs``."""
    out = np.empty((xs.shape[0], state.uz.shape[1]), dtype=np.float64)
    for start in range(0, xs.shape[0], _EVAL_CHUNK):
        chunk = xs[start : start + _EVAL_CHUNK]
        full = np.hstack([_kernel(chunk, state.xu), np.ones((chunk.shape[0], 1)), chunk[:, None]])
        out[start : start + _EVAL_CHUNK] = full @ state.uz
    return out


def tp_predict(x: np.ndarray, state: TprsState) -> np.ndarray:
    """mgcv counterpart: ``Predict.matrix.tprs.smooth`` (smooth.r), which calls ``predict_tprs`` (tprs.c),
    mgcv 1.9-4.

    The **unconstrained** ``tp`` design at ``x`` (``mgcv``'s ``Predict.matrix``): the
    caller applies the training sum-to-zero ``Z``. Evaluating at the training rows
    reproduces the training design exactly (same arithmetic)."""
    xs = np.asarray(x, dtype=np.float64) - state.shift
    xs_u, inverse = np.unique(xs, return_inverse=True)
    return _eval_unique(xs_u, state)[inverse.reshape(-1)]


def _eigen_truncation(xu: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(E, λ_k, U_k, order-info)`` — the ``k`` eigenpairs of the full kernel matrix
    ``E`` with the largest ``|λ|``, negatives (ascending) first then positives
    (descending), each eigenvector's largest-magnitude entry made positive."""
    e = _kernel(xu, xu)
    values, vectors = np.linalg.eigh(e)
    keep = np.argsort(-np.abs(values), kind="stable")[:k]
    lam = values[keep]
    neg = sorted((i for i in range(k) if lam[i] < 0), key=lambda i: lam[i])
    pos = sorted((i for i in range(k) if lam[i] >= 0), key=lambda i: -lam[i])
    order = np.asarray(neg + pos, dtype=np.int64)
    u = vectors[:, keep[order]]
    lam = lam[order]
    lead = np.argmax(np.abs(u), axis=0)
    u = u * np.sign(u[lead, np.arange(k)])
    return e, lam, u, values


def _check_inputs(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] == 0 or not np.all(np.isfinite(x)):
        raise HgamValidationError("tp_basis: x must be a non-empty finite 1-D array.")
    if k < TP_NULL_DIM + 1:
        raise HgamValidationError(
            f"tp_basis: k={k} is below the minimum {TP_NULL_DIM + 1} (null space {TP_NULL_DIM} "
            "plus one wiggly column); mgcv would silently raise it, so it is refused instead."
        )
    xs = x - float(np.mean(x))
    xu, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
    if x.shape[0] > TP_MAX_KNOTS and xu.shape[0] > TP_MAX_KNOTS:
        raise HgamNotImplementedError(
            f"tp_basis: {x.shape[0]} rows and {xu.shape[0]} distinct values both exceed mgcv's "
            f"max.knots={TP_MAX_KNOTS}, where mgcv draws a RANDOM subsample of the distinct "
            "values (set.seed(1) then sample()); that stream is not reproduced, so a different "
            "basis would be returned silently. Aggregate the covariate to <= 2000 distinct "
            "values (e.g. a banded or integer age) or use bs='cr'."
        )
    if xu.shape[0] < k:
        raise HgamValidationError(
            f"tp_basis: only {xu.shape[0]} distinct covariate value(s) for k={k}; mgcv refuses "
            "this too ('a term has fewer unique covariate combinations than specified maximum "
            "degrees of freedom')."
        )
    return xs, xu, inverse.reshape(-1), counts


def tp_construct(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, TprsState]:
    """mgcv counterpart: ``smooth.construct.tp.smooth.spec`` (smooth.r), ``construct_tprs`` (tprs.c),
    mgcv 1.9-4, as ``smooth.construct`` returns it: before ``smoothCon``'s ``scale.penalty``.

    The unconstrained design ``(n, k)`` with ``construct_tprs``'s column scaling, its penalty ``(k, k)``
    and the :class:`TprsState`. ``fs`` builds its base smooth this way (``smooth.construct.fs.smooth.spec``
    calls ``smooth.construct``, not ``smoothCon``); :func:`tp_basis` adds the penalty scaling.

    Raises:
        HgamValidationError: ``k < 3``; fewer than ``k`` distinct values.
        HgamNotImplementedError: the ``max.knots`` subsampling regime (module docstring).
        HgamComputationError: a degenerate (zero) column.
    """
    xs, xu, inverse, counts = _check_inputs(x, k)
    n = xs.shape[0]
    nu = xu.shape[0]
    m = TP_NULL_DIM
    _, lam, u, _ = _eigen_truncation(xu, k)
    t = np.column_stack([np.ones(nu), xu])
    q, _ = np.linalg.qr((t.T @ u).T, mode="complete")
    z = q[:, m:]
    uz = np.zeros((nu + m, k), dtype=np.float64)
    uz[:nu, : k - m] = u @ z
    uz[nu:, k - m :] = np.eye(m)
    s_raw = np.zeros((k, k), dtype=np.float64)
    s_raw[: k - m, : k - m] = z.T @ (lam[:, None] * z)
    state = TprsState(shift=float(np.mean(np.asarray(x, dtype=np.float64))), xu=xu, uz=uz)

    design_u = _eval_unique(xu, state)
    column_norm = np.sqrt(counts @ design_u**2)
    if np.any(column_norm == 0.0):
        raise HgamComputationError("tp_basis: a design column is identically zero.")
    col_scale = sqrt(n) / column_norm
    design_u = design_u * col_scale
    uz = uz * col_scale
    s_scaled = s_raw * np.outer(col_scale, col_scale)
    s_scaled = (s_scaled + s_scaled.T) / 2.0
    return design_u[inverse], s_scaled, TprsState(shift=state.shift, xu=xu, uz=uz)


def tp_basis(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, TprsState]:
    """mgcv counterpart: ``smooth.construct.tp.smooth.spec`` (smooth.r), ``construct_tprs`` (tprs.c),
    mgcv 1.9-4.

    The **unconstrained** ``tp`` design ``(n, k)``, the (``scale.penalty``-rescaled)
    penalty ``(k, k)`` and the :class:`TprsState` — ``smoothCon(..., absorb.cons=FALSE)``.

    Raises:
        HgamValidationError: ``k < 3``; fewer than ``k`` distinct values.
        HgamNotImplementedError: the ``max.knots`` subsampling regime (smooth.r: more than 2000 rows and
            more than 2000 distinct values), which is not ported.
        HgamComputationError: a degenerate (zero) column or penalty.
    """
    design, s_scaled, state = tp_construct(x, k)
    x_inf_sq = _r_norm_inf(design) ** 2
    scale = _r_norm_one(s_scaled) / x_inf_sq
    if scale == 0.0:
        raise HgamComputationError("tp_basis: the unscaled penalty is identically zero.")
    return design, s_scaled / scale, state


def subspace_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sine of the largest principal angle between ``col(a)`` and ``col(b)`` (a metric on
    subspaces, so it satisfies the triangle inequality)."""
    qa = np.linalg.qr(a)[0]
    qb = np.linalg.qr(b)[0]
    resid = qb - qa @ (qa.T @ qb)
    return float(np.linalg.svd(resid, compute_uv=False)[0])


def eigenspace_rounding_bound(x: np.ndarray, k: int) -> float:
    """A-priori bound on how far a floating-point eigensolver can move the kept ``k``-dimensional
    eigenspace of the kernel matrix ``E``: ``nu · eps · ‖E‖₂ / gap`` (Davis-Kahan, with the LAPACK
    backward-error constant ``nu``), ``gap`` the smallest distance between a kept and a dropped
    eigenvalue. Computed from Polaris's own spectrum; it reads no ``mgcv`` output."""
    _, xu, _, _ = _check_inputs(x, k)
    values = np.linalg.eigvalsh(_kernel(xu, xu))
    keep = np.zeros(values.shape[0], dtype=bool)
    keep[np.argsort(-np.abs(values), kind="stable")[:k]] = True
    if keep.all():
        return float(xu.shape[0] * np.finfo(np.float64).eps)
    gap = float(np.min(np.abs(values[keep][:, None] - values[~keep][None, :])))
    return float(xu.shape[0] * np.finfo(np.float64).eps * np.max(np.abs(values)) / gap)
