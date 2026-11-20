"""``k.check`` (plots.r), the basis-dimension test ``gam.check`` prints, for a fitted :class:`GamFit`.

For each smooth, mgcv compares the mean squared difference of deviance residuals between neighbouring
covariate values with the same statistic over ``n.rep`` random shuffles of the residuals; the k-index is
the first over ``mean(rsd^2)`` and the p-value the share of shuffles below it. The shuffles come from R's
RNG (:class:`~hgam.frontend.r_rng.RRandom`), so under one seed the draws are R's own.

Ported: one-covariate smooths (cr, tp, a numeric ``by`` dropped as mgcv drops it), the tensor-product
branch for ti (covariates rescaled by the term's mean absolute finite-difference gradient, then each row's
3 nearest neighbours), and the NA mgcv gives a smooth with a factor covariate (sz, re). ``nearest``'s
kd-tree (``k_nn``, sparse-smooth.c) is replaced by SciPy's ``cKDTree``: both return the exact Euclidean
3 nearest neighbours other than the row itself, so the sets agree wherever there are no distance ties
(mgcv breaks ties "arbitrarily"); a tie at the third neighbour is refused rather than broken differently."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial import cKDTree

from hgam.engine.gam_model import resolve_family
from hgam.engine.gam_predict import predict_term_design
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.r_rng import RRandom

if TYPE_CHECKING:
    from hgam.frontend.api import GamFit

__all__ = ["GamCheckReport", "KCheckRow", "deviance_residuals", "gam_check", "k_check"]


@dataclass(frozen=True)
class KCheckRow:
    """One row of ``k.check``'s table."""

    label: str
    k_prime: int
    """``k'``: the smooth's coefficient count (``length(first.para:last.para)``)."""
    edf: float
    k_index: float | None
    p_value: float | None
    status: Literal["tested", "factor"]
    """``"factor"``: mgcv's NA (a factor covariate)."""


def deviance_residuals(fit: "GamFit") -> np.ndarray:
    """mgcv counterpart: ``residuals.gam(object, type = "deviance")`` (mgcv.r), mgcv 1.9-4:
    ``sqrt(pmax(dev.resids(y, mu, wt), 0)) * sign(y - mu)``."""
    if fit.y is None:
        raise HgamValidationError("deviance_residuals: this fit carries no training response.")
    family = resolve_family(fit.model.family, fit.model.link)
    mu = family.link.linkinv(fit.eta)
    w = np.ones_like(fit.y) if fit.prior_weights is None else fit.prior_weights
    return np.sqrt(np.maximum(family.dev_resids(fit.y, mu, w), 0.0)) * np.sign(fit.y - mu)


def _covariate(data: Mapping[str, object], name: str, label: str, n: int) -> np.ndarray:
    if name not in data:
        raise HgamValidationError(f"k_check: data has no column {name!r} ({label}).")
    x = np.asarray(data[name], dtype=np.float64)
    if x.shape != (n,):
        raise HgamValidationError(f"k_check: column {name!r} has {x.size} rows; the fit has {n}.")
    return x


def _ti_neighbours(
    fit: "GamFit", index: int, block: Mapping[str, object], dat: dict[str, np.ndarray], label: str
) -> np.ndarray:
    """``k.check``'s tensor branch (plots.r, mgcv 1.9-4) up to ``nearest(3, as.matrix(dat))$ni``.

    ``f0 = PredictMat(smooth, dat) %*% beta``; for each covariate, ``dx = diff(range)/1000``,
    ``gr.f = mean(abs(fp - f0)) / dx`` with that covariate shifted by ``dx``; each covariate is then
    shifted to min 0 and scaled to ``gr.f * x / max``. Returns the 0-based ``ni`` (rows by 3).

    Raises:
        HgamNotImplementedError: a distance tie at the third neighbour (mgcv's kd-tree breaks it in its
            own order, which is not ported), or a constant covariate (mgcv divides by zero).
    """
    state = fit.term_states[index]
    beta = fit.coef_full[int(block["start"]) : int(block["end"])]
    f0 = predict_term_design(state, dat) @ beta
    columns = []
    for name in state.term.variables:
        x = dat[name]
        dx = float(np.max(x) - np.min(x)) / 1000.0
        if dx == 0.0:
            raise HgamNotImplementedError(f"k_check: {label}'s covariate {name!r} is constant.")
        fp = predict_term_design(state, {**dat, name: x + dx}) @ beta
        gr = float(np.mean(np.abs(fp - f0))) / dx
        shifted = x - np.min(x)
        columns.append(gr * shifted / np.max(shifted))
    points = np.column_stack(columns)
    nn = 3
    if points.shape[0] <= nn:
        raise HgamValidationError(f"k_check: {label} needs more than {nn} rows for nearest().")
    dist, idx = cKDTree(points).query(points, k=nn + 2)
    is_self = idx == np.arange(points.shape[0])[:, None]
    if np.any(is_self.sum(axis=1) != 1):
        raise HgamNotImplementedError(
            f"k_check: {label} has {nn + 2} or more rows at one point; mgcv's kd-tree (k_nn) picks among "
            "them in its own order, which is not ported."
        )
    keep = ~is_self
    idx = idx[keep].reshape(-1, nn + 1)
    dist = dist[keep].reshape(-1, nn + 1)
    # A tie in exact arithmetic can differ by an ulp after rounding; treat a relative gap of 1e-12 as a tie.
    if np.any(dist[:, nn] - dist[:, nn - 1] <= _TIE_RTOL * dist[:, nn]):
        raise HgamNotImplementedError(
            f"k_check: {label} has a distance tie at the third nearest neighbour; mgcv's kd-tree (k_nn) "
            "breaks it in its own order, which is not ported."
        )
    ni = idx[:, :nn]
    return ni


def k_check(
    fit: "GamFit", data: Mapping[str, object], *, seed: int, subsample: int = 5000, n_rep: int = 400
) -> tuple[KCheckRow, ...]:
    """mgcv counterpart: ``k.check`` (plots.r), mgcv 1.9-4, run after ``set.seed(seed)``.

    ``data`` is the training frame (mgcv reads ``b$model``): the test orders residuals by the raw
    covariate, not the binned one a ``discrete=True`` fit used. Smooths are taken in mgcv's order
    (coefficient order). ``order()`` is a stable sort, as R's.

    Raises:
        HgamNotImplementedError: a factor ``by`` smooth (mgcv drops the by and tests the covariate over
            all rows; not ported), a smooth basis other than cr, tp, ti, sz and re, or a ti term whose
            nearest neighbours tie (see :func:`_ti_neighbours`).
        HgamValidationError: ``data`` does not have the fit's row count, or a covariate is missing.
    """
    return _k_check(fit, data, RRandom(seed), subsample=subsample, n_rep=n_rep)


def _k_check(
    fit: "GamFit", data: Mapping[str, object], rng: RRandom, *, subsample: int, n_rep: int
) -> tuple[KCheckRow, ...]:
    """:func:`k_check` on an RNG stream already in use (``gam.check`` draws before it)."""
    rsd = deviance_residuals(fit)
    n = rsd.size
    rows_index = np.arange(n)
    if n > subsample:
        rows_index = rng.sample_int(n, subsample) - 1
    rsd = rsd[rows_index]
    nr = rsd.size
    blocks = {b["label"]: b for b in fit.design["term_blocks"]}
    terms = {t.label: t for t in fit.model.terms}
    term_index = {t.label: i for i, t in enumerate(fit.model.terms)}
    out: list[KCheckRow] = []
    for label in sorted(fit.smooth_labels, key=lambda lab: int(blocks[lab]["start"])):
        term = terms[label]
        block = blocks[label]
        k_prime = int(block["end"]) - int(block["start"])
        edf = float(fit.edf_per_term[label])
        if getattr(term, "by_factor", None) is not None:
            raise HgamNotImplementedError(
                f"k_check: {label} is a factor-by smooth; k.check drops the by and tests the covariate "
                "over all rows, which is not ported."
            )
        if term.basis in ("sz", "fs", "re"):
            out.append(KCheckRow(label, k_prime, edf, None, None, "factor"))
            continue
        if term.basis in ("te", "t2"):
            raise HgamNotImplementedError(
                f"k_check: {label} is a {term.basis} term; k.check's tensor branch is ported for ti only."
            )
        if term.basis == "ti":
            dat = {v: _covariate(data, v, label, n)[rows_index] for v in term.variables}
            ni = _ti_neighbours(fit, term_index[label], block, dat, label)
            v_obs = float(np.mean((rsd[:, None] - rsd[ni]) ** 2) / 2.0)
            ve = np.empty(n_rep, dtype=np.float64)
            for i in range(n_rep):
                rsdr = rsd[rng.sample_int(nr, nr) - 1]
                ve[i] = np.mean((rsdr[:, None] - rsdr[ni]) ** 2) / 2.0
            p_value = float(np.mean(ve < v_obs))
            out.append(KCheckRow(label, k_prime, edf, v_obs / float(np.mean(rsd**2)), p_value, "tested"))
            continue
        if term.basis not in ("cr", "tp") or len(term.variables) != 1:
            raise HgamNotImplementedError(f"k_check: basis {term.basis!r} ({label}) is not ported.")
        x = _covariate(data, term.variables[0], label, n)[rows_index]
        e = np.diff(rsd[np.argsort(x, kind="stable")])
        v_obs = float(np.mean(e**2) / 2.0)
        ve = np.empty(n_rep, dtype=np.float64)
        for i in range(n_rep):
            e = np.diff(rsd[rng.sample_int(nr, nr) - 1])
            ve[i] = np.mean(e**2) / 2.0
        p_value = float(np.mean(ve < v_obs))
        out.append(KCheckRow(label, k_prime, edf, v_obs / float(np.mean(rsd**2)), p_value, "tested"))
    return tuple(out)


_TIE_RTOL = 1e-12
"""Relative gap between the third and fourth neighbour distances below which ``_ti_neighbours`` refuses."""

_QQ_S_REP = 10
"""``qq.gam``'s ``s.rep`` default: the number of ``sample(U, n)`` draws it makes at ``rep = 0``."""

_K_FOOTNOTE = (
    "Basis dimension (k) checking results. Low p-value (k-index<1) may\n"
    "indicate that k is too low, especially if edf is close to k'."
)


@dataclass(frozen=True)
class GamCheckReport:
    """What ``gam.check`` prints for a ``bam(method = "fREML", discrete = TRUE)`` fit (no plots)."""

    method: str
    """``b$method``: ``"fREML"``."""
    optimizer: tuple[str, str]
    """``b$optimizer``: ``("perf", "chol")`` for ``bgam.fitd``."""
    grad: np.ndarray
    """``b$outer.info$grad``, printed by ``print(b$outer.info)`` (optimizer[2] is not newton/bfgs)."""
    hess: np.ndarray
    """``b$outer.info$hess``."""
    rank: int
    """``b$rank``: the rank of the penalised Cholesky factor (``Sl.fitChol``'s ``r``)."""
    n_coef: int
    """``length(b$coefficients)``."""
    k_table: tuple[KCheckRow, ...]
    """``k.check(b, subsample = k.sample, n.rep = k.rep)``."""

    def __str__(self) -> str:
        """The text ``gam.check`` prints. The method, rank and footnote lines are mgcv's. ``outer.info``
        is formatted by NumPy, not R's ``print.default``. The k table uses fixed widths (3 decimals
        for k' and edf, 2 for k-index and p-value), which is how ``printCoefmat(digits = 3)`` lays out
        the exit fit's table; ``printCoefmat``'s general digit rules are not ported."""
        lines = [f"Method: {self.method}   Optimizer: {' '.join(self.optimizer)}", "$grad"]
        with np.printoptions(precision=6, floatmode="maxprec_equal", linewidth=80):
            lines += [np.array2string(self.grad, formatter={"float_kind": "{: .6e}".format}), "", "$hess"]
            lines += [np.array2string(self.hess, formatter={"float_kind": "{: .6e}".format}), ""]
        lines += [f"Model rank =  {self.rank} / {self.n_coef} ", "", _K_FOOTNOTE, ""]
        width = max(len(r.label) for r in self.k_table) if self.k_table else 0
        k_prime = "k'"
        lines.append(f"{'':<{width}} {k_prime:>6} {'edf':>6} {'k-index':>7} {'p-value':>7}")

        def num(v: float | None, fmt: str, w: int) -> str:
            return f"{'NA':>{w}}" if v is None else f"{v:{w}{fmt}}"

        for r in self.k_table:
            lines.append(
                f"{r.label:<{width}} {r.k_prime:6.3f} {r.edf:6.3f} "
                f"{num(r.k_index, '.2f', 7)} {num(r.p_value, '.2f', 7)}"
            )
        return "\n".join(lines)


def gam_check(
    fit: "GamFit",
    data: Mapping[str, object],
    *,
    seed: int,
    type: str = "deviance",
    k_sample: int = 5000,
    k_rep: int = 200,
) -> GamCheckReport:
    """mgcv counterpart: ``gam.check`` (plots.r), mgcv 1.9-4, without its four plots, run after
    ``set.seed(seed)``, for a ``bam(method = "fREML", discrete = TRUE)`` fit.

    ``bgam.fitd`` sets ``optimizer = c("perf", "chol")`` and ``outer.info = list(grad, hess)``, so
    ``gam.check`` takes its ``print(b$outer.info)`` branch; then ``Model rank``; then ``k.check`` at
    ``k.sample`` and ``k.rep``. Before that, ``qq.gam`` (``rep = 0``, the family has a quantile function)
    draws ``s.rep = 10`` permutations ``sample(U, n)``; the plot is not ported, but the draws are made so
    ``k.check`` sees R's stream. ``data`` is the training frame (see :func:`k_check`).

    Raises:
        HgamNotImplementedError: a fit not made by ``discrete=True`` (the REML gam path's ``outer.info``
            and convergence lines come from mgcv's ``newton``, which hgam does not run), or ``type``
            other than ``"deviance"`` (``k.check`` always uses deviance residuals; only the plots read
            ``type``, and they are not ported).
    """
    if type != "deviance":
        raise HgamNotImplementedError(
            f"gam_check: type={type!r} is not ported; only the plots read it "
            "(k.check uses deviance residuals)."
        )
    if fit.fit_method != "fREML-discrete" or fit.outer_grad is None or fit.outer_hess is None:
        raise HgamNotImplementedError(
            "gam_check: only a bam(method='fREML', discrete=True) fit is ported; the REML gam path's "
            "convergence report (mgcv's newton outer.info) is not."
        )
    rng = RRandom(seed)
    n = int(np.asarray(fit.eta).size)
    for _ in range(_QQ_S_REP):
        rng.sample_int(n, n)
    return GamCheckReport(
        method="fREML",
        optimizer=("perf", "chol"),
        grad=fit.outer_grad.copy(),
        hess=fit.outer_hess.copy(),
        rank=int(fit.coef.size),
        n_coef=int(fit.coef_full.size),
        k_table=_k_check(fit, data, rng, subsample=k_sample, n_rep=k_rep),
    )
