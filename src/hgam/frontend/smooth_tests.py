"""``summary.gam``'s smooth-term test for fully penalised terms: ``reTest`` and ``recov`` (mgcv.r).

A smooth with ``null.space.dim == 0`` (every term of a ``select=TRUE`` fit) is tested by ``reTest``
(Wood 2013, *Biometrika* 100, 221-228, section 3): the statistic is ``||Rm b_m||^2 / sig2``, where
``Rm`` is the ``m`` block of the QR factor of ``[R; sqrt(S_lambda)']`` with term ``m``'s columns moved
last (``recov``), and the reference distribution is a weighted sum of chi-squared(1) variables whose
weights are the eigenvalues of ``crossprod(Rm B) / sig2`` with ``B B' = Ve[m, m]``. ``Ref.df`` is the
number of those eigenvalues above ``max * eps^0.8``.

Ported here: the statistic (``Chi.sq``), ``Ref.df`` and the p-value (``psum.chisq(stat, ev)``, Davies'
algorithm in ``hgam.frontend.davies``, with the ``liu2`` fallback). Terms with a null space
(``testStat``), random-effect terms (``recov``'s ``re`` branch) and the estimated-scale p-value are not
ported.
"""

import warnings

import numpy as np
from scipy.linalg import lapack
from scipy.stats import chi2, ncx2

from hgam.errors import HgamComputationError, HgamValidationError
from hgam.frontend.davies import davies

__all__ = ["liu2", "mroot", "psum_chisq", "re_test"]

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
) -> tuple[float, int, float]:
    """mgcv counterpart: ``reTest(b, m)`` (mgcv.r), mgcv 1.9-4 (unchanged from 1.9-1), with no
    random-effect terms beside ``m``: returns ``(stat, rank, pval)``, ``summary.gam``'s ``Chi.sq``,
    ``Ref.df`` and p-value. For a fixed scale ``pval = psum.chisq(stat, ev)``; the estimated-scale
    branch (``psum.chisq(0, c(ev, -stat/k), ...)``) is refused with ``sig2 != 1``.

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
    pval = float(psum_chisq(stat, ev)[0])
    return stat, rank, pval


def psum_chisq(
    q: float | np.ndarray,
    lb: np.ndarray,
    df: np.ndarray | int | None = None,
    nc: np.ndarray | float | None = None,
    sigz: float = 0.0,
    lower_tail: bool = False,
    tol: float = 2e-5,
    nlim: int = 100000,
) -> np.ndarray:
    """mgcv counterpart: ``psum.chisq`` (mgcv.r), mgcv 1.9-4 (unchanged from 1.9-1).

    ``Pr(q > sum_j lb[j] X_j + sigz Z)`` (``lower_tail=True``) or its complement (the default), with
    ``X_j`` chi-squared on ``df[j]`` degrees of freedom and non-centrality ``nc[j]``, ``Z`` standard
    normal, by Davies' algorithm (``davies``). On ``ifault`` 2 mgcv warns and keeps Davies' value; on any
    other fault it warns and falls back to ``liu2`` when every ``nc`` is 0 (else ``NaN``, R's ``NA``).
    As in mgcv, the fallback is ``liu2``'s upper tail whatever ``lower_tail`` is."""
    qv = np.atleast_1d(np.asarray(q, dtype=np.float64))
    lbv = np.atleast_1d(np.asarray(lb, dtype=np.float64))
    r = lbv.size
    dfv = np.ones(r) if df is None else np.atleast_1d(np.asarray(df, dtype=np.float64))
    ncv = np.zeros(r) if nc is None else np.atleast_1d(np.asarray(nc, dtype=np.float64))
    if dfv.size == 1:
        dfv = np.repeat(dfv, r)
    if ncv.size == 1:
        ncv = np.repeat(ncv, r)
    if dfv.size != r or ncv.size != r:
        raise HgamValidationError("psum_chisq: lengths of lb, df and nc must match.")
    dfv = np.round(dfv)  # R's round: half to even, as np.round
    if np.any(dfv < 1):
        raise HgamValidationError("psum_chisq: df must be positive integers.")
    if np.all(lbv == 0):
        raise HgamValidationError("psum_chisq: at least one element of lb must be non-zero.")
    if sigz < 0:
        sigz = 0.0
    dfi = dfv.astype(np.int64)
    p = qv.copy()
    for i, qi in enumerate(qv):
        res = davies(lbv, ncv, dfi, sigz, float(qi), nlim, tol)
        if res.ifault != 0:
            if res.ifault == 2:
                warnings.warn("psum_chisq: danger of round-off error", RuntimeWarning, stacklevel=2)
                p[i] = res.c if lower_tail else 1 - res.c
            else:
                warnings.warn(
                    "psum_chisq: failure of Davies method, falling back on Liu et al approximation",
                    RuntimeWarning,
                    stacklevel=2,
                )
                p[i] = liu2(np.array([qi]), lbv, h=dfv)[0] if np.all(ncv == 0) else np.nan
        else:
            p[i] = res.c if lower_tail else 1 - res.c
    return p


def liu2(
    x: float | np.ndarray, lam: np.ndarray, h: np.ndarray | None = None, lower_tail: bool = False
) -> np.ndarray:
    """mgcv counterpart: ``liu2`` (mgcv.r), mgcv 1.9-4: Liu, Tang and Zhang's (2009) approximation to
    ``Pr[sum_i lam_i chisq_{h_i} > x]`` (``lower_tail=False``). ``pchisq(..., ncp = delta)`` is SciPy's
    ``ncx2`` (``chi2`` where ``delta`` is 0), in place of R's ``pnchisq``."""
    xv = np.atleast_1d(np.asarray(x, dtype=np.float64))
    lamv = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    hv = np.ones(lamv.size) if h is None else np.atleast_1d(np.asarray(h, dtype=np.float64))
    if hv.size != lamv.size:
        raise HgamValidationError("liu2: lambda and h should have the same length!")
    lh = lamv * hv
    mu_q = np.sum(lh)
    lh = lh * lamv
    c2 = np.sum(lh)
    lh = lh * lamv
    c3 = np.sum(lh)
    xpos = xv > 0
    res = 1 + 0 * xv
    if np.sum(xpos) == 0 or c2 <= 0:
        return res
    s1 = c3 / c2**1.5
    s2 = np.sum(lh * lamv) / c2**2
    sig_q = np.sqrt(2 * c2)
    t = (xv[xpos] - mu_q) / sig_q
    if s1**2 > s2:
        a = 1 / (s1 - np.sqrt(s1**2 - s2))
        delta = s1 * a**3 - a**2
        l_ = a**2 - 2 * delta
    else:
        a = 1 / s1
        delta = 0.0
        if c3 == 0:
            return res
        l_ = c2**3 / c3**2
    mu_x = l_ + delta
    sig_x = np.sqrt(2) * a
    z = t * sig_x + mu_x
    if delta == 0:
        res[xpos] = chi2.cdf(z, l_) if lower_tail else chi2.sf(z, l_)
    else:
        res[xpos] = ncx2.cdf(z, l_, delta) if lower_tail else ncx2.sf(z, l_, delta)
    return res
