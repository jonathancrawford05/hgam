"""``summary.gam``'s smooth-term test for fully penalised terms: ``reTest`` and ``recov`` (mgcv.r).

A smooth with ``null.space.dim == 0`` (every term of a ``select=TRUE`` fit) is tested by ``reTest``
(Wood 2013, *Biometrika* 100, 221-228, section 3): the statistic is ``||Rm b_m||^2 / sig2``, where
``Rm`` is the ``m`` block of the QR factor of ``[R; sqrt(S_lambda)']`` with term ``m``'s columns moved
last (``recov``), and the reference distribution is a weighted sum of chi-squared(1) variables whose
weights are the eigenvalues of ``crossprod(Rm B) / sig2`` with ``B B' = Ve[m, m]``. ``Ref.df`` is the
number of those eigenvalues above ``max * eps^0.8``.

Ported here: the statistic (``Chi.sq``) and ``Ref.df``. The p-value (``psum.chisq``: Davies' algorithm,
mgcv's ``davies.c``) is not ported yet; terms with a null space (``testStat``) and random-effect
terms (``recov``'s ``re`` branch) are not ported either.
"""

import numpy as np
from scipy.linalg import lapack

from hgam.errors import HgamComputationError, HgamValidationError

__all__ = ["mroot", "re_test"]

_EPS = float(np.finfo(np.float64).eps)


def mroot(a: np.ndarray) -> np.ndarray:
    """mgcv counterpart: ``mroot`` (mgcv.r), ``method = "chol"``, mgcv 1.9-1 (unchanged in 1.9-4).

    ``B`` with ``B B' = A`` for a symmetric non-negative definite ``A``, ``rank(A)`` columns.
    ``chol(A, pivot = TRUE, tol = 0)`` is LAPACK's ``dpstrf`` with ``TOL = 0``, called here
    through SciPy: the pivoted Cholesky stops at the first non-positive pivot, which sets the
    rank. As in mgcv the trailing block is zeroed, the columns are put back in their original
    order and the first ``rank`` rows are transposed."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise HgamValidationError(f"mroot: a square matrix is required, got shape {a.shape}.")
    # all.equal(A, t(A)): mean relative difference against tolerance 1.5e-8 (PR #23 review nit).
    if np.sum(np.abs(a - a.T)) > 1.5e-8 * np.sum(np.abs(a)):
        raise HgamValidationError("mroot: supplied matrix not symmetric.")
    p = a.shape[0]
    u, piv, rank, info = lapack.dpstrf(a, tol=0.0, lower=0)
    if info < 0:
        raise HgamComputationError(f"mroot: dpstrf argument {-info} was illegal.")
    u = np.triu(u)
    if rank < p:
        u[rank:, rank:] = 0.0
    order = np.argsort(piv - 1)
    return np.ascontiguousarray(u[:, order][:rank, :].T)


def _recov_rm(r: np.ndarray, s_total: np.ndarray, block: slice) -> np.ndarray:
    """mgcv counterpart: ``recov(b, re = NULL, m)`` (mgcv.r), mgcv 1.9-1: the ``m > 0`` branch
    with no random terms. ``Rm`` is the trailing block of the unpivoted QR of ``[R; mroot(S)']``
    with term ``m``'s columns moved last. NumPy's Householder QR replaces LINPACK's ``dqrdc2``
    (``tol = 0``: no pivoting); the rows of ``Rm`` may differ in sign, which no use of it sees."""
    lrb = np.vstack([r, mroot(s_total).T])
    cols = np.arange(lrb.shape[1])
    inside = cols[block]
    outside = np.setdiff1d(cols, inside)
    lrb = lrb[:, np.concatenate([outside, inside])]
    rr = np.linalg.qr(lrb, mode="r")
    if rr.shape[0] < rr.shape[1]:
        raise HgamComputationError("recov: [R; mroot(S)'] has fewer rows than columns.")
    k = inside.size
    return rr[-k:, -k:]


def re_test(
    coef: np.ndarray,
    xwx: np.ndarray,
    s_total: np.ndarray,
    vp: np.ndarray,
    block: slice,
    sig2: float = 1.0,
) -> tuple[float, int]:
    """mgcv counterpart: ``reTest(b, m)`` (mgcv.r), mgcv 1.9-1, with no random-effect terms
    beside ``m``: returns ``(stat, rank)``, ``summary.gam``'s ``Chi.sq`` and ``Ref.df``.

    Args:
        coef: all coefficients (``coef(b)``).
        xwx: ``X'WX`` at the fit's final working weights (``crossprod(b$R)``); ``R`` is taken as
            ``t(mroot(xwx))``, any factor with ``R'R = X'WX`` giving the same statistic and rank.
        s_total: ``sum_j sp_j S_j`` over every smooth, full size.
        vp: the Bayesian covariance ``Vp``; ``Ve = Vp X'WX Vp`` (``b$Ve`` for a fixed scale).
        block: term ``m``'s coefficients (``first.para:last.para``).
        sig2: the scale (1 for the fixed-scale families ported).
    """
    if sig2 != 1.0:
        raise NotImplementedError("re_test: an estimated scale (sig2 != 1) is not ported.")
    r = mroot(xwx).T
    rm = _recov_rm(r, s_total, block)
    ve = vp @ xwx @ vp
    ve = (ve + ve.T) / 2.0
    b_mat = mroot(ve[block, block])
    d = rm @ coef[block]
    stat = float(d @ d) / sig2
    rb = rm @ b_mat
    ev = np.linalg.eigvalsh(rb.T @ rb / sig2)
    ev[ev < 0.0] = 0.0
    rank = int(np.sum(ev > np.max(ev) * _EPS**0.8)) if ev.size else 0
    return stat, rank
