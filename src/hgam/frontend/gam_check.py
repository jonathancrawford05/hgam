"""``k.check`` (plots.r), the basis-dimension test ``gam.check`` prints, for a fitted :class:`GamFit`.

For each smooth, mgcv compares the mean squared difference of deviance residuals between neighbouring
covariate values with the same statistic over ``n.rep`` random shuffles of the residuals; the k-index is
the first over ``mean(rsd^2)`` and the p-value the share of shuffles below it. The shuffles come from R's
RNG (:class:`~hgam.frontend.r_rng.RRandom`), so under one seed the draws are R's own.

Ported: one-covariate smooths (cr, tp, a numeric ``by`` dropped as mgcv drops it), and the NA mgcv gives a
smooth with a factor covariate (sz, re). Not ported: the tensor-product branch (``PredictMat`` finite
differences and ``nearest``'s kd-tree, ``k_nn`` in mgcv's C): a ti term gets k' and edf, no k-index or
p-value, but its ``n.rep`` shuffles are drawn so the terms after it see R's stream."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from hgam.engine.gam_model import resolve_family
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.r_rng import RRandom

if TYPE_CHECKING:
    from hgam.frontend.api import GamFit

__all__ = ["KCheckRow", "deviance_residuals", "k_check"]


@dataclass(frozen=True)
class KCheckRow:
    """One row of ``k.check``'s table."""

    label: str
    k_prime: int
    """``k'``: the smooth's coefficient count (``length(first.para:last.para)``)."""
    edf: float
    k_index: float | None
    p_value: float | None
    status: Literal["tested", "factor", "not_ported"]
    """``"factor"``: mgcv's NA (a factor covariate). ``"not_ported"``: mgcv tests it, hgam does not (ti)."""


def deviance_residuals(fit: "GamFit") -> np.ndarray:
    """mgcv counterpart: ``residuals.gam(object, type = "deviance")`` (mgcv.r), mgcv 1.9-4:
    ``sqrt(pmax(dev.resids(y, mu, wt), 0)) * sign(y - mu)``."""
    if fit.y is None:
        raise HgamValidationError("deviance_residuals: this fit carries no training response.")
    family = resolve_family(fit.model.family, fit.model.link)
    mu = family.link.linkinv(fit.eta)
    w = np.ones_like(fit.y) if fit.prior_weights is None else fit.prior_weights
    return np.sqrt(np.maximum(family.dev_resids(fit.y, mu, w), 0.0)) * np.sign(fit.y - mu)


def k_check(
    fit: "GamFit", data: Mapping[str, object], *, seed: int, subsample: int = 5000, n_rep: int = 400
) -> tuple[KCheckRow, ...]:
    """mgcv counterpart: ``k.check`` (plots.r), mgcv 1.9-4, run after ``set.seed(seed)``.

    ``data`` is the training frame (mgcv reads ``b$model``): the test orders residuals by the raw
    covariate, not the binned one a ``discrete=True`` fit used. Smooths are taken in mgcv's order
    (coefficient order). ``order()`` is a stable sort, as R's.

    Raises:
        HgamNotImplementedError: a factor ``by`` smooth (mgcv drops the by and tests the covariate over
            all rows; not ported), or a smooth basis other than cr, tp, ti, sz and re.
        HgamValidationError: ``data`` does not have the fit's row count, or a covariate is missing.
    """
    rsd = deviance_residuals(fit)
    n = rsd.size
    rng = RRandom(seed)
    rows_index = np.arange(n)
    if n > subsample:
        rows_index = rng.sample_int(n, subsample) - 1
    rsd = rsd[rows_index]
    nr = rsd.size
    blocks = {b["label"]: b for b in fit.design["term_blocks"]}
    terms = {t.label: t for t in fit.model.terms}
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
        if term.basis in ("sz", "re"):
            out.append(KCheckRow(label, k_prime, edf, None, None, "factor"))
            continue
        if term.basis == "ti":
            for _ in range(n_rep):
                rng.sample_int(nr, nr)
            out.append(KCheckRow(label, k_prime, edf, None, None, "not_ported"))
            continue
        if term.basis not in ("cr", "tp") or len(term.variables) != 1:
            raise HgamNotImplementedError(f"k_check: basis {term.basis!r} ({label}) is not ported.")
        name = term.variables[0]
        if name not in data:
            raise HgamValidationError(f"k_check: data has no column {name!r} ({label}).")
        x = np.asarray(data[name], dtype=np.float64)
        if x.shape != (n,):
            raise HgamValidationError(f"k_check: column {name!r} has {x.size} rows; the fit has {n}.")
        x = x[rows_index]
        e = np.diff(rsd[np.argsort(x, kind="stable")])
        v_obs = float(np.mean(e**2) / 2.0)
        ve = np.empty(n_rep, dtype=np.float64)
        for i in range(n_rep):
            e = np.diff(rsd[rng.sample_int(nr, nr) - 1])
            ve[i] = np.mean(e**2) / 2.0
        p_value = float(np.mean(ve < v_obs))
        out.append(KCheckRow(label, k_prime, edf, v_obs / float(np.mean(rsd**2)), p_value, "tested"))
    return tuple(out)
