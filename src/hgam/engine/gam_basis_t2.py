"""The ``t2`` tensor product smooth on ``cr`` margins (``t2(x, z, bs="cr", k=c(k1, k2))``), the basis.

mgcv counterparts (smooth.r, mgcv 1.9-4): ``smooth.construct.t2.smooth.spec``, ``t2.model.matrix``
(``full = FALSE``, ``ord = NULL``), ``Predict.matrix.t2.smooth``, ``nat.param`` (type 3,
:func:`~hgam.engine.gam_basis_fs.nat_param`), and ``smoothCon``'s penalty scaling and constraint absorption.
Held to ``smoothcon_t2_cr`` and ``smoothcon_t2_cr_three`` (``make_fixtures.R`` section 27) in
``tests/test_t2_basis.py``.

The construction, as mgcv runs it for cr margins:

1. Each margin is built by ``smooth.construct`` (unconstrained; the unscaled cr penalty,
   :func:`~hgam.engine.gam_basis_cr._cr_basis_raw`), rank ``k - 2``, then re-parameterised by
   ``nat.param(X, S, rank, type = 3, unit.fnorm = TRUE)``: the first ``rank`` columns ("r") carry an
   identity penalty, the last two ("n") are unpenalised, the constant first among them. ``P`` maps the raw
   margin design to the re-parameterised one and is kept for prediction.
2. ``t2.model.matrix`` forms row tensor products of the parts, margin by margin: after margin ``i`` every
   earlier block is multiplied by margin ``i``'s range part (label + "r") and then by its null part
   (label + "n"). Each block but the last ("nn...n", unpenalised) gets its own identity penalty;
   ``names(S)`` are the labels and ``rank`` the block widths (``sub.cols``).
3. ``null.space.dim`` is the width of the unpenalised block. The fit constraint ``C`` is zero on the
   penalised columns and ``colSums`` of ``X`` on the unpenalised ones; ``Cp = colSums(X)`` is the
   prediction constraint.
4. ``smoothCon`` divides each penalty by ``norm(S) / norm(X, "I")^2``. ``Cp`` is absorbed first, into ``Xp``
   and ``Sp`` (``Zp`` from the QR of ``t(Cp)``). ``C`` is then absorbed by the sparse branch (``drop = -1``
   and ``C`` has zero columns): only its nonzero columns ``indi`` are transformed, the range columns are
   untouched, and the last of ``indi`` is dropped. Last, the object's ``qrc`` is replaced by ``Cp``'s QR, so
   ``PredictMat`` applies ``Zp``, not the fit constraint: prediction is in the ``Xp`` parameterisation
   (``gam.setup``'s ``G$P`` maps fit coefficients to it; not in this module).

Prediction (``PredictMat`` -> ``Predict.matrix.t2.smooth``) evaluates each margin's raw cr design at the new
values, multiplies by its ``P``, forms ``t2.model.matrix`` and applies ``Zp``.

What is not reproducible element for element: each margin's re-parameterisation comes from eigenvectors, so
a column of ``X`` may differ from mgcv's in sign (the product of its margins' column signs). That is an
orthogonal re-parameterisation, which leaves a fit unchanged; the test holds the raw pieces up to column signs
and the absorbed ones up to an orthogonal map.
"""

from collections.abc import Sequence

import numpy as np

from hgam.engine.gam_basis_cr import _cr_basis_raw, _r_norm_inf, _r_norm_one, cr_default_knots
from hgam.engine.gam_basis_fs import nat_param
from hgam.engine.gam_basis_te import tensor_prod_model_matrix
from hgam.errors import HgamComputationError, HgamNotImplementedError, HgamValidationError

__all__ = ["T2Basis", "t2_basis", "t2_model_matrix", "t2_predict_design"]

_CR_NULL_SPACE_DIM = 2  # smooth.construct.cr.smooth.spec: rank = k - 2, null.space.dim = 2


def t2_model_matrix(
    designs: Sequence[np.ndarray],
    rank: Sequence[int],
    *,
    full: bool = False,
    ord: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    """mgcv counterpart: ``t2.model.matrix`` (smooth.r), mgcv 1.9-4, ``full = FALSE`` and ``ord = NULL``.

    Args:
        designs: The re-parameterised margin designs; the first ``rank[i]`` columns of margin ``i`` are its
            range part, the rest its null part.
        rank: Each margin's penalty rank.
        full: ``t2(full = TRUE)``; not ported.
        ord: ``t2(ord = )``; not ported.

    Returns:
        ``(X, sub_cols, p_lab)``: the model matrix, the width of each penalised block (mgcv's ``sub.cols``)
        and its label (``p.lab``). The unpenalised block, when there is one, is last and has no entry.
    """
    if full:
        raise HgamNotImplementedError("t2_model_matrix: full = TRUE is not ported.")
    if ord is not None:
        raise HgamNotImplementedError("t2_model_matrix: ord = is not ported.")
    designs = [np.asarray(d, dtype=np.float64) for d in designs]
    if len(designs) != len(rank):
        raise HgamValidationError(f"t2_model_matrix: {len(designs)} margins but {len(rank)} ranks.")

    def parts(i: int) -> tuple[np.ndarray, np.ndarray | None]:
        r = int(rank[i])
        if not 1 <= r <= designs[i].shape[1]:
            raise HgamValidationError(
                f"t2_model_matrix: margin {i + 1} has rank {r} of {designs[i].shape[1]}."
            )
        null = designs[i][:, r:] if r < designs[i].shape[1] else None
        return designs[i][:, :r], null

    z1, n1 = parts(0)
    blocks, labels = [z1], ["r"]
    no_null = n1 is None
    if n1 is not None:
        blocks.append(n1)
        labels.append("n")
    for i in range(1, len(designs)):
        zi, ni = parts(i)
        if ni is None:
            no_null = True
        new_blocks = [tensor_prod_model_matrix([b, zi]) for b in blocks]
        new_labels = [lab + "r" for lab in labels]
        if ni is not None:
            new_blocks += [tensor_prod_model_matrix([b, ni]) for b in blocks]
            new_labels += [lab + "n" for lab in labels]
        blocks, labels = new_blocks, new_labels
    widths = [b.shape[1] for b in blocks]
    x = np.concatenate(blocks, axis=1)
    if not no_null:
        widths, labels = widths[:-1], labels[:-1]
    return x, tuple(widths), tuple(labels)


class T2Basis:
    """A ``t2`` term as ``smoothCon(..., absorb.cons = TRUE)`` returns it, with what prediction needs."""

    def __init__(
        self,
        design: np.ndarray,
        penalties: tuple[np.ndarray, ...],
        design_p: np.ndarray,
        penalties_p: tuple[np.ndarray, ...],
        rank: tuple[int, ...],
        null_space_dim: int,
        p_lab: tuple[str, ...],
        knots: tuple[np.ndarray, ...],
        margin_p: tuple[np.ndarray, ...],
        z_fit: np.ndarray,
        z_pred: np.ndarray,
        design_raw: np.ndarray,
    ) -> None:
        self.design = design
        """``(n, prod(k) - 1)``: the design under the fit constraint (mgcv's ``X``)."""
        self.penalties = penalties
        """One identity penalty per penalised block, scaled, under the fit constraint (``S``)."""
        self.design_p = design_p
        """The design under the prediction constraint ``Cp`` (``Xp``)."""
        self.penalties_p = penalties_p
        """The penalties under ``Cp`` (``Sp``)."""
        self.rank = rank
        """mgcv's ``rank``: the width of each penalised block."""
        self.null_space_dim = null_space_dim
        """mgcv's ``null.space.dim`` after the constraint."""
        self.p_lab = p_lab
        """``names(S)``: each penalty's block label, "r" range and "n" null per margin."""
        self.knots = knots
        """Each margin's knots (``margin[[i]]$xp``)."""
        self.margin_p = margin_p
        """Each margin's ``nat.param`` ``P`` (mgcv's ``object$P``)."""
        self.z_fit = z_fit
        """``(prod(k), prod(k) - 1)``: ``X = X_raw @ z_fit``."""
        self.z_pred = z_pred
        """``(prod(k), prod(k) - 1)``: ``Cp``'s null space, what ``PredictMat`` applies."""
        self.design_raw = design_raw
        """``(n, prod(k))``: ``t2.model.matrix`` before any constraint (``absorb.cons = FALSE``'s ``X``)."""


def _margin_knots(x: np.ndarray, k: int, knots: np.ndarray | None) -> np.ndarray:
    if knots is None:
        return cr_default_knots(x, k)
    knots = np.asarray(knots, dtype=np.float64)
    if knots.shape != (k,):
        # smooth.construct.cr.smooth.spec stops: "number of supplied knots != k for a cr smooth".
        raise HgamValidationError(f"t2 basis: {knots.shape[0]} knots supplied for a cr margin with k={k}.")
    return knots


def _check_margins(xs: Sequence[np.ndarray], ks: Sequence[int]) -> list[np.ndarray]:
    if len(xs) < 2:
        raise HgamValidationError("t2 basis: a tensor product needs at least two margins.")
    if len(ks) != len(xs):
        raise HgamValidationError(f"t2 basis: {len(xs)} margins but {len(ks)} values of k.")
    xs = [np.asarray(x, dtype=np.float64) for x in xs]
    if any(x.shape != xs[0].shape or x.ndim != 1 for x in xs):
        raise HgamValidationError("t2 basis: every margin needs one value per row.")
    return xs


def _qr_null_space(constraint: np.ndarray) -> np.ndarray:
    """``qr(t(C))`` for one constraint row: the trailing columns of the full ``Q``."""
    if np.allclose(constraint, 0.0):
        raise HgamComputationError("t2 basis: the constraint row is (numerically) zero.")
    q, _ = np.linalg.qr(constraint.reshape(-1, 1), mode="complete")
    return np.asarray(q[:, 1:], dtype=np.float64)


def _fit_constraint_null_space(c: np.ndarray) -> np.ndarray:
    """smoothCon's sparse constraint branch (``nx < ncol(C)``, ``drop < 0``) as a matrix ``Z``: columns of the
    identity at the zero columns of ``C``; at ``indi`` (the nonzero ones) the first ``nx - 1`` take
    ``qr(t(C[, indi]))``'s trailing columns in place, and ``indi[nx]`` is dropped."""
    p = c.shape[0]
    indi = np.flatnonzero(c != 0.0)
    q = _qr_null_space(c[indi])
    z = np.eye(p, dtype=np.float64)
    z[np.ix_(indi, indi[:-1])] = q
    return np.delete(z, indi[-1], axis=1)


def t2_basis(
    xs: Sequence[np.ndarray], ks: Sequence[int], knots: Sequence[np.ndarray | None] | None = None
) -> T2Basis:
    """mgcv counterpart: ``smoothCon(t2(..., bs = "cr"), absorb.cons = TRUE)``, i.e.
    ``smooth.construct.t2.smooth.spec`` (smooth.r) then ``smoothCon``'s scaling and constraints, mgcv 1.9-4
    (module docstring).

    Args:
        xs: One covariate per margin, each ``(n,)``, in written order.
        ks: Each margin's ``k``.
        knots: Each margin's supplied knots, or ``None`` for mgcv's default; ``None`` for all defaults.
    """
    xs = _check_margins(xs, ks)
    if knots is None:
        knots = [None] * len(xs)
    if len(knots) != len(xs):
        raise HgamValidationError(f"t2 basis: {len(xs)} margins but {len(knots)} knot entries.")
    margin_knots = tuple(_margin_knots(x, int(k), kn) for x, k, kn in zip(xs, ks, knots, strict=True))

    designs, margin_p, ranks = [], [], []
    for x, kn in zip(xs, margin_knots, strict=True):
        raw, penalty = _cr_basis_raw(x, kn)
        r = raw.shape[1] - _CR_NULL_SPACE_DIM
        x_np, _, p_mat = nat_param(raw, penalty, r, type=3)
        designs.append(x_np)
        margin_p.append(p_mat)
        ranks.append(r)
    x_raw, sub_cols, p_lab = t2_model_matrix(designs, ranks)

    p = x_raw.shape[1]
    edges = np.concatenate([[0], np.cumsum(sub_cols)])
    penalties = []
    for j in range(len(sub_cols)):
        dd = np.zeros(p)
        dd[edges[j] : edges[j + 1]] = 1.0
        penalties.append(np.diag(dd))
    null_space_dim = p - int(edges[-1])
    if null_space_dim < 2:
        # cr margins always leave a 2^m-column unpenalised block; mgcv's null.space.dim 0 or 1 branches
        # (no constraint, set-to-zero) are not reached and not ported.
        raise HgamNotImplementedError(f"t2 basis: null.space.dim = {null_space_dim} is not ported.")
    nup = int(edges[-1])
    c = np.concatenate([np.zeros(nup), x_raw[:, nup:].sum(axis=0)])
    cp = x_raw.sum(axis=0)

    max_xx = _r_norm_inf(x_raw) ** 2
    if max_xx == 0.0:
        raise HgamComputationError("t2_basis: the tensor design matrix is identically zero.")
    penalties = [s / (_r_norm_one(s) / max_xx) for s in penalties]

    z_pred = _qr_null_space(cp)
    z_fit = _fit_constraint_null_space(c)
    n_cons = 1
    return T2Basis(
        x_raw @ z_fit,
        tuple(z_fit.T @ s @ z_fit for s in penalties),
        x_raw @ z_pred,
        tuple(z_pred.T @ s @ z_pred for s in penalties),
        tuple(min(r, p - n_cons) for r in sub_cols),
        max(0, null_space_dim - n_cons),
        p_lab,
        margin_knots,
        tuple(margin_p),
        z_fit,
        z_pred,
        x_raw,
    )


def t2_predict_design(xs: Sequence[np.ndarray], basis: T2Basis) -> np.ndarray:
    """mgcv counterpart: ``PredictMat`` -> ``Predict.matrix.t2.smooth`` (smooth.r), mgcv 1.9-4, for cr
    margins: each margin's raw cr design at the new values times its ``P``, ``t2.model.matrix``, then
    ``Cp``'s ``Zp`` (``PredictMat`` uses ``Cp``'s QR, so the result is in the ``Xp`` parameterisation)."""
    xs = _check_margins(xs, [len(kn) for kn in basis.knots])
    if len(xs) != len(basis.knots):
        raise HgamValidationError(f"t2_predict_design: {len(xs)} margins for a {len(basis.knots)}-margin t2.")
    designs = [_cr_basis_raw(x, kn)[0] @ p for x, kn, p in zip(xs, basis.knots, basis.margin_p, strict=True)]
    ranks = [kn.shape[0] - _CR_NULL_SPACE_DIM for kn in basis.knots]
    x, _, _ = t2_model_matrix(designs, ranks)
    return x @ basis.z_pred
