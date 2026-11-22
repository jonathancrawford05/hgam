"""``GamFit.summary()`` — the fit report (preview epic Slice P4, ADR-253).

Mirrors the parts of ``mgcv``'s ``summary.gam`` that the verified engine produces
independently: the parametric coefficient table (``p.table``, :func:`parametric_table`),
per-smooth ``edf``, the smoothing parameters, the scale, the REML score, the deviance
explained and ``n`` — plus the Newton search's own convergence report (iterations, the final
relative projected gradient against ``epsilon_rel``, the search's ``conv.tol``: 1e-6 by default, ADR-248;
for the port of mgcv's ``newton`` the gradient is relative to ``5 * score.scale``, mgcv's own test).

**The smooth-term tests, in part.** For a ``discrete=True`` fit (fixed scale, no pivoted
coefficients), every fully penalised smooth (``null.space.dim == 0``, as under ``select=TRUE``) gets
``reTest``'s ``Chi.sq`` and ``Ref.df`` (:mod:`hgam.frontend.smooth_tests`). The p-value
(``psum.chisq``), ``testStat`` (smooths with a null space) and the ``gam()`` path are not ported;
those entries are ``None`` and the printed report says so.

``deviance_explained`` follows ``mgcv``: ``(null.deviance - deviance) / null.deviance``,
the null model being the intercept-only fit **with the offset** when there is one
(:func:`null_deviance`).
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from hgam.engine.gam_family import Family
from hgam.engine.gam_model import EngineGAMFit, resolve_family
from hgam.engine.gam_reml_newton import MGCV_NEWTON_CONV_TOL
from hgam.errors import HgamValidationError
from hgam.frontend.smooth_tests import re_test

__all__ = [
    "EPSILON_REL",
    "SATURATION_THRESHOLD",
    "GamSummary",
    "ParametricRow",
    "SmoothSummary",
    "build_summary",
    "null_deviance",
    "parametric_table",
]

EPSILON_REL: float = MGCV_NEWTON_CONV_TOL
"""The Newton search's relative gradient tolerance (``gam.control()$newton$conv.tol``,
ADR-248), imported rather than redeclared."""


SATURATION_THRESHOLD: float = 0.95
"""A smooth is saturated when ``edf >= SATURATION_THRESHOLD * n_coef`` (maintainer, 2026-11-08;
ADR-265). Defined here so the fit report and polaris-re's ``gam.fit_report`` share one value."""


@dataclass(frozen=True)
class SmoothSummary:
    """One smooth term's row."""

    label: str
    edf: float
    log10_sp: tuple[float, ...]
    """``log10`` of each of the term's smoothing parameters (``ti`` and ``select=TRUE``
    terms carry more than one), in penalty-block order."""
    n_coef: int
    null_space_dim: int | None = None
    """``smooth$null.space.dim``; ``None`` where the constructor's value is not ported."""
    ref_df: int | None = None
    """``summary.gam``'s ``Ref.df`` (``reTest``'s rank); ``None`` where not ported."""
    chi_sq: float | None = None
    """``summary.gam``'s ``Chi.sq`` (``reTest``'s statistic); ``None`` where not ported."""
    p_value: float | None = None
    """``summary.gam``'s p-value (``reTest``: ``psum.chisq(Chi.sq, ev)``); ``None`` where not ported."""

    @property
    def saturated(self) -> bool:
        """``edf >= 0.95 x n_coef`` (ADR-265): the basis may be limiting the term."""
        return self.n_coef > 0 and self.edf >= SATURATION_THRESHOLD * self.n_coef


@dataclass(frozen=True)
class ParametricRow:
    """One row of ``summary.gam``'s ``p.table``: a parametric coefficient."""

    name: str
    """``mgcv``'s coefficient name: ``(Intercept)``, ``FaceSizeHigh``, ``FaceSizeHigh:SmokeS``."""
    estimate: float
    std_error: float
    """From the Bayesian covariance ``Vp`` (``summary.gam``'s default ``freq = FALSE``)."""
    z: float
    p_value: float
    """``2 * pnorm(-|z|)``: the scale is fixed by the family."""


@dataclass(frozen=True)
class GamSummary:
    """What :meth:`hgam.frontend.api.GamFit.summary` returns. Plain data; ``str()`` prints
    the report. Field meanings follow ``summary.gam`` where it has the counterpart."""

    formula: str
    family: str
    link: str
    n: int
    n_parametric: int
    smooths: tuple[SmoothSummary, ...]
    parametric: tuple[ParametricRow, ...] | None
    """``summary.gam``'s ``p.table``; ``None`` when the scale is estimated (that branch, t values
    on ``residual.df``, is not ported)."""
    edf_total: float
    scale: float
    scale_estimated: bool
    reml_score: float
    deviance: float
    null_deviance: float
    deviance_explained: float
    converged: bool
    outer: str
    n_iterations: int | None
    max_abs_projected_gradient: float | None
    rel_projected_gradient: float | None
    epsilon_rel: float
    n_function_evals: int
    at_bound_blocks: tuple[str, ...]

    def __str__(self) -> str:
        lines = [
            f"Family: {self.family}  Link function: {self.link}",
            f"Formula: {self.formula}",
            "",
            "Parametric coefficients:",
        ]
        if self.parametric is None:
            lines.append(
                f"  {self.n_parametric} (intercept included); the table for an estimated scale "
                "(t on residual.df) is not ported"
            )
        else:
            lines.append(f"  {'':<28}{'Estimate':>12}{'Std. Error':>12}{'z value':>10}{'Pr(>|z|)':>12}")
            for r in self.parametric:
                lines.append(
                    f"  {r.name:<28}{r.estimate:>12.5g}{r.std_error:>12.5g}{r.z:>10.3f}{r.p_value:>12.3g}"
                )
        lines += [
            "",
            "Smooth terms:",
            f"  {'term':<28}{'coef':>6}{'edf':>10}{'Ref.df':>8}{'Chi.sq':>12}{'p-value':>12}   log10(sp)",
        ]
        for s in self.smooths:
            sps = ", ".join(f"{v:.3f}" for v in s.log10_sp) or "—"
            ref_df = "—" if s.ref_df is None else str(s.ref_df)
            chi = "—" if s.chi_sq is None else f"{s.chi_sq:.5g}"
            pv = "—" if s.p_value is None else f"{s.p_value:.3g}"
            lines.append(f"  {s.label:<28}{s.n_coef:>6}{s.edf:>10.4f}{ref_df:>8}{chi:>12}{pv:>12}   {sps}")
        lines += [
            "",
            f"n = {self.n}   total edf = {self.edf_total:.4f}",
            f"scale est. = {self.scale:.6g}" + ("" if self.scale_estimated else "  (fixed by the family)"),
            (
                f"REML score = {self.reml_score:.6f}"
                if np.isfinite(self.reml_score)
                else "REML score: not computed (bam's fREML score is not ported)"
            ),
            f"Deviance explained = {100.0 * self.deviance_explained:.3f}%"
            f"  (deviance {self.deviance:.6g}, null {self.null_deviance:.6g})",
        ]
        if self.rel_projected_gradient is not None:
            lines.append(
                f"Newton REML search: {self.n_iterations} iterations, "
                f"{self.n_function_evals} penalised fits, relative projected gradient "
                f"{self.rel_projected_gradient:.3e} vs epsilon_rel {self.epsilon_rel:.0e} — "
                + ("converged" if self.converged else "NOT CONVERGED")
            )
        else:
            lines.append(
                f"Outer search: {self.outer}, {self.n_function_evals} penalised fits — "
                + ("converged" if self.converged else "NOT CONVERGED")
            )
        sat = [t for t in self.smooths if t.saturated]
        if sat:
            lines.append(
                f"Basis saturation (edf >= {SATURATION_THRESHOLD:.0%} of coefficients; consider "
                f"a larger k): " + ", ".join(f"{t.label} ({t.edf:.2f} of {t.n_coef})" for t in sat)
            )
        if self.at_bound_blocks:
            lines.append(
                f"Smoothed to the null space (upper bound of the search): {', '.join(self.at_bound_blocks)}"
            )
        lines.append("")
        if any(t.p_value is None for t in self.smooths):
            lines.append(
                "Smooth-term tests ('—') are reported only for fully penalised smooths of a discrete fit "
                "with a fixed scale (reTest); testStat (terms with a null space), recov's random-effect "
                "branch and the estimated-scale p-value are not ported."
            )
        else:
            lines.append("Ref.df, Chi.sq and p-value are reTest's (p from psum.chisq, Davies' algorithm).")
        return "\n".join(lines)


def null_deviance(
    y: np.ndarray, family: Family, weights: np.ndarray | None, offset: np.ndarray | None
) -> float:
    """The intercept-only deviance, with the offset when there is one.

    ``mgcv`` evaluates the null model at the weighted mean when there is no offset and
    refits the intercept with the offset otherwise; both are the root ``b`` of the
    intercept score ``sum(w (y - mu) mu'/V) = 0`` at ``eta = offset + b``, which is what
    is solved here (the score is monotone in ``b``). Returns NaN when the score has no
    root (e.g. an all-zero count response).
    """
    w = np.ones_like(y) if weights is None else weights
    off = np.zeros_like(y) if offset is None else offset

    def score(b: float) -> float:
        eta = off + b
        mu = family.link.linkinv(eta)
        return float(np.sum(w * (y - mu) * family.link.mu_eta(eta) / family.variance(mu)))

    lo, hi = -1.0, 1.0
    for _ in range(60):
        if score(lo) > 0.0 > score(hi):
            break
        lo, hi = lo * 2.0, hi * 2.0
    else:
        return float("nan")
    b = brentq(score, lo, hi, xtol=1e-14, rtol=1e-14)
    mu0 = family.link.linkinv(off + b)
    return family.deviance(y, mu0, w)


def parametric_table(
    coef: np.ndarray, vp: np.ndarray, names: tuple[str, ...], scale_estimated: bool
) -> tuple[ParametricRow, ...] | None:
    """mgcv counterpart: the parametric coefficient table of ``summary.gam`` (mgcv.r), mgcv 1.9-4.

    ``p.coeff <- coefficients[1:nsdf]``, ``p.se <- sqrt(diag(Vp))[1:nsdf]`` (``freq = FALSE``),
    ``p.t <- p.coeff / p.se`` and, for a fixed scale, ``p.pv <- 2 * pnorm(abs(p.t), lower.tail =
    FALSE)``. ``coef`` and ``vp`` are the full coefficient vector and ``Vp``; the first
    ``len(names)`` entries are the parametric block. Returns ``None`` for an estimated scale:
    that branch (``pt`` on ``residual.df``, column "t value") is not ported, and the report says so.
    A coefficient pivoted out of a rank-deficient design has zero estimate and variance, so its
    z and p are NaN."""
    if scale_estimated:
        return None
    n = len(names)
    est = np.asarray(coef[:n], dtype=np.float64)
    se = np.sqrt(np.diag(vp)[:n])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = est / se
    p = 2.0 * norm.sf(np.abs(z))
    return tuple(
        ParametricRow(name=nm, estimate=float(e), std_error=float(s), z=float(t), p_value=float(pv))
        for nm, e, s, t, pv in zip(names, est, se, z, p, strict=True)
    )


def build_summary(
    fit: EngineGAMFit,
    formula: str,
    smooth_labels: tuple[str, ...],
    n_parametric: int,
    parametric_names: tuple[str, ...] = (),
) -> GamSummary:
    """Assemble the :class:`GamSummary` of ``fit``. ``smooth_labels`` are the smooth
    terms; the parametric block (``n_parametric`` = ``m$nsdf`` columns) is tabulated under
    ``parametric_names`` (mgcv's coefficient names) as ``summary.gam``'s ``p.table``."""
    if fit.y is None:
        raise HgamValidationError("summary(): this fit carries no training response.")
    family = resolve_family(fit.model.family, fit.model.link)
    w = np.ones_like(fit.y) if fit.prior_weights is None else fit.prior_weights
    mu = np.asarray(family.link.linkinv(fit.eta), dtype=np.float64)
    deviance = family.deviance(fit.y, mu, w)
    null_dev = null_deviance(fit.y, family, fit.prior_weights, fit.offset)
    explained = (null_dev - deviance) / null_dev if np.isfinite(null_dev) and null_dev > 0 else (float("nan"))
    scale = 1.0 if family.dispersion_fixed else float(fit.dispersion.fletcher)

    if len(parametric_names) != n_parametric:
        raise HgamValidationError(
            f"summary(): {len(parametric_names)} parametric coefficient name(s) for {n_parametric} "
            "parametric coefficients."
        )
    parametric = parametric_table(fit.coef_full, fit.vcov(), parametric_names, not family.dispersion_fixed)

    blocks = {b["label"]: b for b in fit.design["term_blocks"]}
    offsets = {}
    cursor = 0
    for b in fit.design["term_blocks"]:
        offsets[b["label"]] = cursor
        cursor += b["n_penalties"]
    smooths = tuple(
        SmoothSummary(
            label=label,
            edf=float(fit.edf_per_term[label]),
            log10_sp=tuple(
                float(v)
                for v in fit.log_lambda[offsets[label] : offsets[label] + blocks[label]["n_penalties"]]
            ),
            n_coef=int(blocks[label]["end"] - blocks[label]["start"]),
            null_space_dim=_null_space_dim(fit, label),
        )
        for label in smooth_labels
    )
    smooths = _with_smooth_tests(fit, smooths, blocks, family)
    # The search's own test, as a relative gradient against conv.tol: max|g| <= gradient_tolerance, where
    # gradient_tolerance = conv.tol * (1 + |score|) for hgam's iteration and 5 * score.scale * conv.tol for
    # the port of mgcv's newton (gam.fit3.r, mgcv 1.9-4; its score-change condition is not shown here).
    gradient_scale = (
        None
        if fit.gradient_tolerance is None or fit.conv_tol is None
        else fit.gradient_tolerance / fit.conv_tol
    )
    rel_grad = (
        None
        if fit.max_abs_projected_gradient is None
        else fit.max_abs_projected_gradient
        / (1.0 + abs(fit.reml_score) if gradient_scale is None else gradient_scale)
    )
    return GamSummary(
        formula=formula,
        family=fit.model.family,
        link=fit.model.link,
        n=int(fit.y.size),
        n_parametric=n_parametric,
        smooths=smooths,
        parametric=parametric,
        edf_total=float(fit.edf_total),
        scale=scale,
        scale_estimated=not family.dispersion_fixed,
        reml_score=float(fit.reml_score),
        deviance=float(deviance),
        null_deviance=float(null_dev),
        deviance_explained=float(explained),
        converged=bool(fit.converged),
        outer=(
            "fREML performance iteration"
            if fit.fit_method == "fREML-discrete"
            else "newton"
            if fit.n_iterations is not None
            else "lbfgsb"
        ),
        n_iterations=fit.n_iterations,
        max_abs_projected_gradient=fit.max_abs_projected_gradient,
        rel_projected_gradient=rel_grad,
        epsilon_rel=EPSILON_REL if fit.conv_tol is None else float(fit.conv_tol),
        n_function_evals=int(fit.n_function_evals),
        at_bound_blocks=fit.at_bound_blocks,
    )


def _null_space_dim(fit: EngineGAMFit, label: str) -> int | None:
    """mgcv counterpart: ``smooth$null.space.dim`` as ``smoothCon`` (smooth.r) leaves it, mgcv 1.9-1,
    where it is known without the constructors' own values: 0 under ``select=TRUE`` (``gam.setup``
    passes ``null.space.penalty = select``; ``smoothCon`` adds a penalty on any unpenalised space and
    sets the field to 0), and 0 for a ``bs="re"`` term (``smooth.construct.re.smooth.spec``) and a
    ``bs="fs"`` term (``smooth.construct.fs.smooth.spec``). Other
    terms carry their constructor's value, which is not ported: ``None``."""
    term = next(t for t in fit.model.terms if t.label == label)
    if fit.model.select or term.basis in ("re", "fs"):
        return 0
    return None


def _with_smooth_tests(
    fit: EngineGAMFit, smooths: tuple[SmoothSummary, ...], blocks: dict, family: Family
) -> tuple[SmoothSummary, ...]:
    """mgcv counterpart: the smooth loop of ``summary.gam`` (mgcv.r), mgcv 1.9-1: a term with
    ``null.space.dim == 0`` goes to ``reTest``. Ported only where ``X'WX`` is carried (the bam
    path), the scale is fixed and no coefficient was pivoted out, and only for a term with no
    random-effect smooth beside it: ``reTest`` collects every other smooth with ``$random`` set
    (``bs="re"``, ``smooth.construct.re.smooth.spec``) and then takes ``recov``'s ``re`` branch,
    which is not ported. Those terms keep ``None`` (PR #23 review, blocking 1)."""
    if fit.xwx is None or fit.vp_override is None or not family.dispersion_fixed or fit.pivoted_columns:
        return smooths
    s_total = np.zeros_like(fit.xwx)
    for lam, s_j in zip(fit.lambda_, fit.design["penalty_blocks"], strict=True):
        s_total += lam * s_j
    random = {t.label for t in fit.model.terms if t.basis == "re"}
    out = []
    for sm in smooths:
        if sm.null_space_dim == 0 and not (random - {sm.label}):
            b = blocks[sm.label]
            stat, rank, pval = re_test(
                fit.coef, fit.xwx, s_total, fit.vp_override, slice(b["start"], b["end"])
            )
            sm = replace(sm, ref_df=rank, chi_sq=stat, p_value=pval)
        out.append(sm)
    return tuple(out)
