"""The ``te`` tensor product smooth on ``cr`` margins (``te(x, z, bs="cr", k=c(k1, k2))``).

mgcv counterparts (smooth.r, mgcv 1.9-4): ``smooth.construct.tensor.smooth.spec`` (the
``inter = FALSE`` branch that ``te()`` takes), ``tensor.prod.model.matrix`` (C ``mgcv_tmm``),
``tensor.prod.penalties``, ``Predict.matrix.tensor.smooth``, and ``smoothCon``'s penalty scaling and
constraint absorption. Held to ``smoothcon_te_cr`` and ``smoothcon_te_cr_three`` (``make_fixtures.R``
section 26) in ``tests/test_te_basis.py``.

The construction, as mgcv runs it for cr margins:

1. Each margin is built by ``smooth.construct`` (``object$mc`` is all ``FALSE`` for ``te``), so it
   carries no constraint and its penalty is the unscaled cr penalty
   (:func:`~hgam.engine.gam_basis_cr._cr_basis_raw`). Margin knots are the supplied ones, else
   ``smooth.construct.cr.smooth.spec``'s default (:func:`~hgam.engine.gam_basis_cr.cr_default_knots`).
2. ``np = TRUE`` asks for a re-parameterisation of each one-dimensional margin, but only where the
   margin does not set ``noterp``; ``smooth.construct.cr.smooth.spec`` sets it, so nothing is done for
   cr margins. A margin class without ``noterp`` is not ported.
3. Each margin penalty is divided by its largest eigenvalue.
4. ``X`` is the row tensor product of the margin designs, the first margin varying slowest; the
   penalties are ``S_1 (x) I (x) ...``, ``I (x) S_2 (x) ...``, each symmetrised; ``rank`` is
   ``prod(d) * r_i / d_i`` and ``null.space.dim`` is ``prod(nr_i)``.
5. ``smoothCon`` divides each penalty by ``norm(S) / norm(X, "I")^2``, takes ``C = colMeans(X)`` over
   the whole product (``te`` leaves ``C`` unset) and absorbs it: ``X Z`` and ``Z' S Z`` with ``Z`` the
   trailing columns of the QR of ``t(C)``, ``rank`` capped at ``ncol - 1``, ``null.space.dim`` reduced
   by one.

Prediction (``PredictMat`` -> ``Predict.matrix.tensor.smooth``) evaluates each margin's raw cr design at
the new values (linear beyond the end knots, as ``crspl``), forms the row tensor product and applies the
same ``Z``.
"""

from collections.abc import Sequence

import numpy as np

from hgam.engine.gam_basis_cr import (
    _cr_basis_raw,
    _leading_eigenvalue,
    _r_norm_inf,
    _r_norm_one,
    cr_default_knots,
    sum_to_zero_null_space,
)
from hgam.errors import HgamComputationError, HgamValidationError

__all__ = [
    "TeBasis",
    "te_basis",
    "te_predict_design",
    "tensor_prod_model_matrix",
    "tensor_prod_penalties",
]

_CR_NULL_SPACE_DIM = 2  # smooth.construct.cr.smooth.spec: rank = k - 2, null.space.dim = 2


def tensor_prod_model_matrix(designs: Sequence[np.ndarray]) -> np.ndarray:
    """mgcv counterpart: ``tensor.prod.model.matrix`` (smooth.r; C ``mgcv_tmm``), mgcv 1.9-4.

    Row ``i`` is ``X_1[i, ] %x% X_2[i, ] %x% ...``: the first margin varies slowest. NumPy's ``einsum``
    replaces mgcv's C loop.
    """
    if not designs:
        raise HgamValidationError("tensor_prod_model_matrix needs at least one margin.")
    out = np.asarray(designs[0], dtype=np.float64)
    n = out.shape[0]
    for design in designs[1:]:
        design = np.asarray(design, dtype=np.float64)
        if design.shape[0] != n:
            raise HgamValidationError(
                f"tensor_prod_model_matrix: margins have {n} and {design.shape[0]} rows; they must agree."
            )
        out = np.einsum("ij,ik->ijk", out, design).reshape(n, out.shape[1] * design.shape[1])
    return out


def tensor_prod_penalties(penalties: Sequence[np.ndarray]) -> list[np.ndarray]:
    """mgcv counterpart: ``tensor.prod.penalties`` (smooth.r), mgcv 1.9-4.

    Penalty ``i`` is the Kronecker product of identities with ``S_i`` in place ``i``, then ``(M + t(M)) / 2``.
    """
    penalties = [np.asarray(s, dtype=np.float64) for s in penalties]
    if len(penalties) == 1:
        return [penalties[0]]
    eyes = [np.eye(s.shape[1], dtype=np.float64) for s in penalties]
    out = []
    for i in range(len(penalties)):
        full = penalties[0] if i == 0 else eyes[0]
        for j in range(1, len(penalties)):
            full = np.kron(full, penalties[j] if j == i else eyes[j])
        out.append((full + full.T) / 2.0)
    return out


class TeBasis:
    """A ``te`` term as ``smoothCon(..., absorb.cons = TRUE)`` returns it, with what prediction needs."""

    def __init__(
        self,
        design: np.ndarray,
        penalties: tuple[np.ndarray, ...],
        rank: tuple[int, ...],
        null_space_dim: int,
        knots: tuple[np.ndarray, ...],
        z: np.ndarray,
    ) -> None:
        self.design = design
        """``(n, prod(k) - 1)``: the constrained tensor design."""
        self.penalties = penalties
        """One penalty per margin, in margin order, constrained and scaled."""
        self.rank = rank
        """mgcv's ``rank``, one per penalty."""
        self.null_space_dim = null_space_dim
        """mgcv's ``null.space.dim`` after the constraint."""
        self.knots = knots
        """Each margin's knots (mgcv's ``margin[[i]]$xp``)."""
        self.z = z
        """``(prod(k), prod(k) - 1)``: the constraint null space, kept for prediction."""


def _margin_knots(x: np.ndarray, k: int, knots: np.ndarray | None) -> np.ndarray:
    if knots is None:
        return cr_default_knots(x, k)
    knots = np.asarray(knots, dtype=np.float64)
    if knots.shape != (k,):
        # smooth.construct.cr.smooth.spec stops: "number of supplied knots != k for a cr smooth".
        raise HgamValidationError(f"te basis: {knots.shape[0]} knots supplied for a cr margin with k={k}.")
    return knots


def _check_margins(xs: Sequence[np.ndarray], ks: Sequence[int]) -> list[np.ndarray]:
    if len(xs) < 2:
        raise HgamValidationError("te basis: a tensor product needs at least two margins.")
    if len(ks) != len(xs):
        raise HgamValidationError(f"te basis: {len(xs)} margins but {len(ks)} values of k.")
    xs = [np.asarray(x, dtype=np.float64) for x in xs]
    if any(x.shape != xs[0].shape or x.ndim != 1 for x in xs):
        raise HgamValidationError("te basis: every margin needs one value per row.")
    return xs


def te_basis(
    xs: Sequence[np.ndarray],
    ks: Sequence[int],
    knots: Sequence[np.ndarray | None] | None = None,
    *,
    constraint_rows: np.ndarray | None = None,
) -> TeBasis:
    """mgcv counterpart: ``smoothCon(te(..., bs = "cr"), absorb.cons = TRUE)``, i.e.
    ``smooth.construct.tensor.smooth.spec`` (smooth.r) then ``smoothCon``'s scaling and constraint, mgcv 1.9-4
    (module docstring).

    Args:
        xs: One covariate per margin, each ``(n,)``, in written order.
        ks: Each margin's ``k``.
        knots: Each margin's supplied knots, or ``None`` for mgcv's default; ``None`` for all defaults.
        constraint_rows: The tensor design rows to take ``colMeans`` over instead of the data's.
    """
    xs = _check_margins(xs, ks)
    if knots is None:
        knots = [None] * len(xs)
    if len(knots) != len(xs):
        raise HgamValidationError(f"te basis: {len(xs)} margins but {len(knots)} knot entries.")
    margin_knots = tuple(_margin_knots(x, int(k), kn) for x, k, kn in zip(xs, ks, knots, strict=True))

    designs, margin_penalties = [], []
    for x, kn in zip(xs, margin_knots, strict=True):
        design, penalty = _cr_basis_raw(x, kn)
        designs.append(design)
        margin_penalties.append(penalty / _leading_eigenvalue(penalty))
    d = [design.shape[1] for design in designs]
    max_rank = int(np.prod(d))
    rank = [max_rank * (di - _CR_NULL_SPACE_DIM) // di for di in d]

    design = tensor_prod_model_matrix(designs)
    penalties = tensor_prod_penalties(margin_penalties)

    max_xx = _r_norm_inf(design) ** 2
    if max_xx == 0.0:
        raise HgamComputationError("te_basis: the tensor design matrix is identically zero.")
    penalties = [s / (_r_norm_one(s) / max_xx) for s in penalties]

    z = sum_to_zero_null_space(design, constraint_rows)
    constrained = []
    for s in penalties:
        s_c = z.T @ s @ z
        constrained.append((s_c + s_c.T) / 2.0)
    n_cons = 1
    return TeBasis(
        design @ z,
        tuple(constrained),
        tuple(min(r, max_rank - n_cons) for r in rank),
        max(0, _CR_NULL_SPACE_DIM ** len(d) - n_cons),
        margin_knots,
        z,
    )


def te_predict_design(xs: Sequence[np.ndarray], basis: TeBasis) -> np.ndarray:
    """mgcv counterpart: ``PredictMat`` -> ``Predict.matrix.tensor.smooth`` (smooth.r), mgcv 1.9-4, for cr
    margins: each margin's cr design at the new values, the row tensor product, then the fit's constraint."""
    xs = _check_margins(xs, [len(kn) for kn in basis.knots])
    if len(xs) != len(basis.knots):
        raise HgamValidationError(f"te_predict_design: {len(xs)} margins for a {len(basis.knots)}-margin te.")
    designs = [_cr_basis_raw(x, kn)[0] for x, kn in zip(xs, basis.knots, strict=True)]
    return tensor_prod_model_matrix(designs) @ basis.z
