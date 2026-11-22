"""Safeguarded Newton outer search for the REML smoothing parameters —
outer-solver epic (``docs/PLAN_wood_outer_solver.md``), Slice 1.

**Why this exists (ADR-241).** The REML criterion is right; the L-BFGS-B search
over it is not reliable. Slice 0 measured two defects, neither of them the
start: an UNCAPPED first quasi-Newton step that was accepted 11.8 decades onto
the ``lambda -> infinity`` plateau (gaussian L1), and a termination test
(``factr``, relative function reduction) that is not a stationarity test and
declared ``converged=True`` with the gradient still ~2 per decade (quasipoisson
3c). This module replaces the search, not the start and not the criterion:
every start strategy that existed is one more sample for a search whose defect
is what it does *after* the start.

**What it does.** Works in natural-log ``rho = log(lambda)`` (the units
``mgcv``'s controls are stated in) and, per iteration:

1. evaluates the score and its ANALYTIC gradient
   (:func:`~hgam.engine.gam_reml_gradient.reml_score_gradient`, or
   :func:`~hgam.engine.gam_reml_gradient.reml_score_gradient_profiled`
   for a free-scale family);
2. stops if the projected gradient is below ``conv_tol`` times the score's own
   scale — a STATIONARITY test, never a function-reduction test;
3. drops already-converged directions (``|g_j|`` below that same tolerance, or
   pinned at a bound with the gradient pushing outward) from the step;
4. forms the Hessian of the remaining directions — EXACT by default
   (:func:`~hgam.engine.gam_reml_hessian.reml_score_hessian`, Slice 2,
   ADR-243), or ``hessian="difference"``: Slice 1's central difference of the
   analytic gradient, kept so the two can be compared — forces it positive definite by
   eigendecomposition (negative eigenvalues flipped, tiny ones floored), and
   takes the Newton step;
5. caps the step at ``max_step`` natural-log units, clips it to the bounds, and
   halves it until the score decreases (``max_half`` times);
6. if no halving improves the score, retries along the steepest-descent
   direction capped at ``max_sd_step``; if that also fails the search ends
   ``converged=False`` and says so.

**Constants.** ``max_step = 5``, ``max_sd_step = 2``, ``max_half = 30`` and
``conv_tol = 1e-6`` are ``mgcv``'s documented ``gam.control()$newton`` defaults
(``maxNstep``, ``maxSstep``, ``maxHalf``, ``conv.tol``; read from
``mgcv:::newton`` 1.9.1 on tier 1 — behaviour and constants, not transcribed
code, ``mgcv`` being GPL and this project MIT). They are the method's own
numbers, named as such, and not tuned to any cell. The Hessian floor
``sqrt(machine epsilon) * max|eigenvalue|`` is the standard relative cut for
"numerically zero curvature". The central-difference step ``hessian_step`` is
this module's own (a stable-step measurement, see its docstring).

**``iteration="mgcv"``: the port of mgcv's ``newton`` (gam.fit3.r, mgcv 1.9-4), opt in (PLAN step 3,
follow-up (b)).** The steps above are hgam's own; with ``iteration="mgcv"`` the same evaluation, exact
Hessian and bounds run mgcv's iteration instead (step logic, ``qerror``, steepest-descent trials and
convergence test; :func:`newton_select_lambdas` lists what is ported and what differs). Measured on
section 27e (tier 1): the five fits' iteration counts equal mgcv's and the linear predictor agrees with
mgcv's default fit to 6.6e-5 or better (``tests/test_mgcv_newton.py``). It is not the default yet: that
changes which mgcv fit the ridge cases are gated against (``docs/PRIORITIES.md``), Jonathan's decision.
"""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from hgam.engine.gam_dispersion import dispersion_estimates
from hgam.engine.gam_family import Family
from hgam.engine.gam_fit import effective_degrees_of_freedom
from hgam.engine.gam_reml import mgcv_reml_score_offset, penalty_block_square_roots
from hgam.engine.gam_reml_gradient import (
    reml_score_gradient,
    reml_score_gradient_profiled,
)
from hgam.engine.gam_reml_hessian import (
    reml_score_hessian,
    reml_score_hessian_profiled,
)
from hgam.engine.gam_reml_optimize import penalized_fit_and_score
from hgam.errors import HgamComputationError, HgamValidationError

__all__ = [
    "MGCV_NEWTON_CONV_TOL",
    "MGCV_NEWTON_MAX_HALF",
    "MGCV_NEWTON_MAX_NSTEP",
    "MGCV_NEWTON_MAX_SSTEP",
    "NewtonLambdaSelection",
    "newton_select_lambdas",
]

MGCV_NEWTON_MAX_NSTEP = 5.0
"""``gam.control()$newton$maxNstep`` — the cap on any Newton step, in
natural-log ``lambda`` units (about 2.17 decades)."""

MGCV_NEWTON_MAX_SSTEP = 2.0
"""``gam.control()$newton$maxSstep`` — the cap on a steepest-descent step."""

MGCV_NEWTON_MAX_HALF = 30
"""``gam.control()$newton$maxHalf`` — step halvings before a direction is
declared exhausted."""

MGCV_NEWTON_CONV_TOL = 1.0e-6
"""``gam.control()$newton$conv.tol`` — gradient tolerance relative to the
score's own scale."""

_HESSIAN_STEP = 1.0e-3
"""Central-difference step (natural-log units) for the Hessian of the analytic
gradient. The gradient is a smooth function of ``rho`` evaluated at a
converged inner fit, so unlike a difference of the SCORE (ADR-212's noise
floor) a difference of the gradient at ``1e-3`` has truncation error
``O(h^2) ~ 1e-6`` relative and noise ``~ 1e-10 / h`` — both far below the
curvature being estimated. Verified against the Hessian at a doubled step in
the tests."""

_LN10 = float(np.log(10.0))

_QERROR_THRESH = 0.8
"""``newton``'s ``qerror.thresh``: the quadratic-approximation error a step may carry and be accepted."""

_EPS = float(np.finfo(np.float64).eps)
"""``.Machine$double.eps``, in mgcv's eigenvalue cuts (``eps^.5`` indefiniteness, ``eps^.7`` floor)."""


@dataclass(frozen=True)
class NewtonLambdaSelection:
    """What :func:`newton_select_lambdas` returns. Field names match
    :class:`~hgam.engine.gam_reml_optimize.ContinuousLambdaSelection`
    so ``fit_engine_gam`` consumes either; the rest is what the search did."""

    log_lambda: np.ndarray
    """``log10(lambda)`` at the stop, one entry per penalty block."""
    lambda_: np.ndarray
    coef: np.ndarray
    reml_score: float
    edf_total: float
    n_function_evals: int
    """Penalized fits spent: score+gradient evaluations at trial points, plus
    (``hessian="difference"`` only) the Hessian's central-difference gradient
    evaluations. The exact Hessian costs no extra fit."""
    n_rejected: int
    """Trial points whose inner fit did not converge (treated as a failed step)."""
    converged: bool
    """``iteration="hgam"``: ``True`` ONLY when the projected gradient met the score-scaled
    tolerance — the stationarity test. A stall (no step improves) is ``False``, whatever the
    gradient. ``iteration="mgcv"``: mgcv ``newton``'s ``conv``, ``True`` for "full convergence" and for
    "step failed" (warned, as mgcv does), ``False`` only when ``maxiter`` is reached."""
    at_bound: bool
    message: str
    n_iterations: int = 0
    max_abs_projected_gradient: float | None = None
    """``max |g^P|`` at the stop, natural-log units."""
    gradient_tolerance: float | None = None
    """The bound ``max |g^P|`` was tested against: ``conv_tol * (1 + |score|)`` for
    ``iteration="hgam"``; ``5 * score.scale * conv_tol``, mgcv's gradient bound, for ``"mgcv"``."""
    score_scale: float = float("nan")
    """``iteration="mgcv"`` only (else NaN): mgcv's ``score.scale`` at the stop,
    ``abs(log(scale.est)) + abs(score)``, with mgcv's REML score (:attr:`reml_score` plus
    :func:`~hgam.engine.gam_reml.mgcv_reml_score_offset`)."""
    accepted_log10: list[np.ndarray] = field(default_factory=list)
    """The start, then every ACCEPTED iterate, in ``log10(lambda)``."""
    max_accepted_step_decades: float = 0.0
    """Largest single-coordinate move of any accepted iterate, in decades."""
    hessian_condition: float = float("nan")
    """``cond(X'WX + S)`` at the stop. DIAGNOSTIC ONLY — never gates anything."""
    gradient_precision_floor: float = float("nan")
    """``eps * hessian_condition``: the order of the analytic gradient's own
    rounding error at the stop (the ``tr(H^-1 lambda S_j)`` terms are computed
    through ``H``). Reported beside :attr:`max_abs_projected_gradient` so a
    stall can be read as "gradient above tolerance but within what this
    arithmetic can resolve" (Wood 2011 §3.1's motivation, Slice 3) or not.
    It is NOT a tolerance and no constant multiplies it."""


def _project(g: np.ndarray, rho: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = g.copy()
    out[(rho <= lo) & (g > 0.0)] = 0.0  # at lower bound, descent would go below it
    out[(rho >= hi) & (g < 0.0)] = 0.0  # at upper bound, descent would go above it
    return out


def newton_select_lambdas(
    y: np.ndarray,
    x: np.ndarray,
    family: Family,
    penalty_blocks: tuple[np.ndarray, ...],
    *,
    x0: np.ndarray,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    gamma: float = 1.0,
    bounds: tuple[float, float] = (-2.0, 12.0),
    maxiter: int = 200,
    conv_tol: float = MGCV_NEWTON_CONV_TOL,
    max_step: float = MGCV_NEWTON_MAX_NSTEP,
    max_sd_step: float = MGCV_NEWTON_MAX_SSTEP,
    max_half: int = MGCV_NEWTON_MAX_HALF,
    hessian_step: float = _HESSIAN_STEP,
    step_halving: bool = False,
    hessian: Literal["exact", "difference"] = "exact",
    iteration: Literal["hgam", "mgcv"] = "hgam",
) -> NewtonLambdaSelection:
    """Select ``log10(lambda)`` for every block by safeguarded Newton from ``x0``.

    ``x0`` is REQUIRED (``log10(lambda)``): this module has no start strategy
    and must not grow one (ADR-241). ``bounds`` are in ``log10(lambda)`` like
    every other search in this package; the search itself runs in natural-log.

    ``iteration="hgam"`` (the default) is the safeguarded search the module docstring describes.
    ``iteration="mgcv"`` is the port of mgcv's ``newton``:

    mgcv counterpart: ``newton`` (gam.fit3.r), mgcv 1.9-4. Ported: the start-up ``uconv.ind`` (gradients
    above ``score.scale * conv.tol``); per iteration, the Hessian of the unconverged coordinates, its
    indefiniteness test (``eps^.5`` against the largest eigenvalue; against ``score.scale`` for one
    parameter), absolute eigenvalues floored at ``max * eps^.7``, the Newton step capped at ``maxNstep``;
    immediate acceptance only when the Hessian is positive definite, the score falls and ``qerror`` (the
    quadratic model's relative error) is below ``qerror.thresh = 0.8``; step halving, switching to the
    steepest-descent direction at the fourth halving in the first nine iterations (length at most
    ``maxSstep``) and ignoring ``qerror`` after the fifth; for an indefinite Hessian, the steepest-descent
    trial (up to 40 halvings) and the better of the two steps; and the convergence test after each
    iteration (not indefinite, every ``|g_j| <= 5 * score.scale * conv.tol``, the score change within
    ``score.scale * conv.tol``; ``uconv.ind`` refreshed at ``0.1 * score.scale * conv.tol`` on the
    gradient and the Hessian's diagonal; a step failure after ``maxHalf`` halvings ends the search as
    converged, with mgcv's warning). ``score.scale = abs(log(scale.est)) + abs(score)`` uses mgcv's REML
    score (:func:`~hgam.engine.gam_reml.mgcv_reml_score_offset`) and the Fletcher ``scale.est``.
    Differences: the inner fit and the derivatives are hgam's (NumPy/LAPACK, not ``gam.fit3``'s C); the
    inner P-IRLS tolerance is not tightened to ``conv.tol/100`` (``docs/DEFERRED.md``); hgam's ``bounds``
    clip each trial point (the Taylor prediction uses the clipped step) and hold a coordinate pinned at a
    bound with its gradient pushing out, where mgcv has no bounds; a trial whose inner fit fails counts as a
    non-finite score; ``edge.correct``, the ``lsp.max`` reparameterisation (switched off in mgcv), ``L``
    and ``lsp0`` are not ported (one coordinate per penalty block).

    Raises:
        HgamValidationError: shape mismatch, ``gamma`` with a free-scale
            family, ``lo >= hi``, or an unknown ``iteration``.
        HgamComputationError: if the start point itself cannot be fitted.
    """
    m = len(penalty_blocks)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (m,):
        raise HgamValidationError(f"newton_select_lambdas: x0 has shape {x0.shape}, expected ({m},).")
    if bounds[0] >= bounds[1]:
        raise HgamValidationError(f"newton_select_lambdas: bounds {bounds} are not lo < hi.")
    if hessian not in ("exact", "difference"):
        raise HgamValidationError(
            f"newton_select_lambdas: hessian must be 'exact' or 'difference', got {hessian!r}."
        )
    if not family.dispersion_fixed and gamma != 1.0:
        raise HgamValidationError("newton_select_lambdas: gamma is undefined for free scale.")
    if iteration not in ("hgam", "mgcv"):
        raise HgamValidationError(
            f"newton_select_lambdas: iteration must be 'hgam' or 'mgcv', got {iteration!r}."
        )
    lo, hi = bounds[0] * _LN10, bounds[1] * _LN10
    sqrt_blocks = penalty_block_square_roots(penalty_blocks)
    tally = {"fits": 0, "rejected": 0}

    def evaluate(rho: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
        """``(coef, score, gradient)`` at ``rho``, or ``None`` if the inner fit fails."""
        tally["fits"] += 1
        log10 = rho / _LN10
        try:
            coef, score = penalized_fit_and_score(
                y,
                x,
                family,
                penalty_blocks,
                log10,
                offset=offset,
                weights=weights,
                gamma=gamma,
                penalty_sqrt_blocks=sqrt_blocks,
                step_halving=step_halving,
            )
            lambdas = np.exp(rho)
            if family.dispersion_fixed:
                grad = reml_score_gradient(
                    y,
                    x,
                    family,
                    coef,
                    penalty_blocks,
                    lambdas,
                    offset=offset,
                    weights=weights,
                    gamma=gamma,
                    penalty_sqrt_blocks=sqrt_blocks,
                )
            else:
                grad = reml_score_gradient_profiled(
                    y,
                    x,
                    family,
                    coef,
                    penalty_blocks,
                    lambdas,
                    offset=offset,
                    weights=weights,
                    penalty_sqrt_blocks=sqrt_blocks,
                )
        except (HgamComputationError, np.linalg.LinAlgError):
            tally["rejected"] += 1
            return None
        if not (np.isfinite(score) and np.all(np.isfinite(grad))):
            tally["rejected"] += 1
            return None
        return coef, float(score), grad

    def exact_hessian(rho: np.ndarray, coef: np.ndarray, free: np.ndarray) -> np.ndarray | None:
        """The analytic Hessian restricted to the ``free`` coordinates, from the
        fit already in hand (no extra penalized fit). ``None`` if it cannot be
        formed (a singular ``H``)."""
        lambdas = np.exp(rho)
        try:
            if family.dispersion_fixed:
                full = reml_score_hessian(
                    y,
                    x,
                    family,
                    coef,
                    penalty_blocks,
                    lambdas,
                    offset=offset,
                    weights=weights,
                    gamma=gamma,
                    penalty_sqrt_blocks=sqrt_blocks,
                )
            else:
                full = reml_score_hessian_profiled(
                    y,
                    x,
                    family,
                    coef,
                    penalty_blocks,
                    lambdas,
                    offset=offset,
                    weights=weights,
                    penalty_sqrt_blocks=sqrt_blocks,
                )
        except (HgamComputationError, np.linalg.LinAlgError):
            return None
        if not np.all(np.isfinite(full)):
            return None
        return np.asarray(full[np.ix_(free, free)], dtype=np.float64)

    def difference_hessian(rho: np.ndarray, free: np.ndarray) -> np.ndarray | None:
        """Central difference of the analytic gradient over the ``free``
        coordinates (one-sided at a bound). ``None`` if any probe fails."""
        idx = np.flatnonzero(free)
        h_mat = np.zeros((idx.size, idx.size), dtype=np.float64)
        for col, j in enumerate(idx):
            up = rho.copy()
            dn = rho.copy()
            up[j] = min(rho[j] + hessian_step, hi)
            dn[j] = max(rho[j] - hessian_step, lo)
            ev_up, ev_dn = evaluate(up), evaluate(dn)
            if ev_up is None or ev_dn is None:
                return None
            h_mat[:, col] = (ev_up[2][idx] - ev_dn[2][idx]) / (up[j] - dn[j])
        return 0.5 * (h_mat + h_mat.T)

    def full_hessian(rho: np.ndarray, coef: np.ndarray) -> np.ndarray | None:
        everything = np.ones(m, dtype=bool)
        return (
            exact_hessian(rho, coef, everything)
            if hessian == "exact"
            else difference_hessian(rho, everything)
        )

    def score_scale(rho: np.ndarray, coef: np.ndarray, score: float) -> float:
        """``newton``'s ``score.scale`` under REML: ``abs(log(b$scale.est)) + abs(score)``, with mgcv's
        score (this module's plus :func:`mgcv_reml_score_offset`) and ``scale.est`` the Fletcher estimate
        ``gam.fit3`` returns whatever the family (``gam.control()$scale.est``)."""
        eta = base_offset + x @ coef
        mu = family.link.linkinv(eta)
        penalty = sum(lam * block for lam, block in zip(np.exp(rho), penalty_blocks, strict=True))
        trace_a = effective_degrees_of_freedom(x, family, eta, mu, penalty, weights)
        scale_est = dispersion_estimates(y, mu, family, trace_a, weights=weights).fletcher
        return float(abs(np.log(scale_est)) + abs(score + score_offset))

    base_offset = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=np.float64)
    score_offset = (
        mgcv_reml_score_offset(family, y, penalty_blocks, weights=weights, gamma=gamma)
        if iteration == "mgcv"
        else 0.0
    )
    rho = np.clip(x0 * _LN10, lo, hi)
    current = evaluate(rho)
    if current is None:
        raise HgamComputationError(
            "newton_select_lambdas: the penalized fit at the starting point did not converge."
        )
    coef, score, grad = current
    accepted = [rho / _LN10]
    max_move = 0.0
    if iteration == "mgcv":
        hess = full_hessian(rho, coef)
        converged = False
        message = "iteration limit reached"
        n_iter = 0
        scale_now = score_scale(rho, coef, score)
        uconv = np.abs(grad) > scale_now * conv_tol
        if not np.any(uconv):  # "check for all converged too soon, and undo"
            uconv[:] = True
        ii = 0
        for n_iter in range(1, maxiter + 1):
            g = _project(grad, rho, lo, hi)
            pinned = g != grad  # hgam's bounds (mgcv has none): held where descent would leave them
            if not np.any(uconv & ~pinned):
                uconv = uconv.copy()
                uconv[np.argmax(np.abs(g))] = True  # "need at least 1 to update"
            free = uconv & ~pinned
            if not np.any(free):
                # Every coordinate is held at a bound with its gradient pushing out (hgam's bounds only):
                # the projected gradient is zero and no step exists.
                converged = True
                message = "full convergence"
                break
            if hess is None:
                raise HgamComputationError(
                    "newton_select_lambdas: the REML Hessian could not be formed at an accepted point."
                )
            evals, evecs = np.linalg.eigh(hess[np.ix_(free, free)])
            evals, evecs = evals[::-1], evecs[:, ::-1]  # R's eigen() order: decreasing
            indef = bool(np.sum(-evals > abs(evals[0]) * np.sqrt(_EPS)) > 0)
            if evals.size == 1 and indef:  # one sp: an infinite sp must not count as always indefinite
                indef = bool(evals[0] < -scale_now * np.sqrt(_EPS))
            d = evals.copy()
            pdef = not bool(np.any(d < 0.0))
            d[d < 0.0] = -d[d < 0.0]  # "see Gill Murray and Wright p107/8"
            low_d = float(np.max(d)) * _EPS**0.7
            if np.any(d < low_d):
                pdef = False  # "not certain it is positive definite"
            d[d < low_d] = low_d
            d = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0.0)
            n_step = np.zeros(m, dtype=np.float64)
            n_step[free] = -(evecs @ (d * (evecs.T @ g[free])))  # (modified) Newton direction
            s_step = -g / float(np.max(np.abs(g)))  # steepest descent direction
            ms = float(np.max(np.abs(n_step)))
            if ms > max_step:
                n_step = max_step * n_step / ms
            sd_unused = True

            def trial(
                step: np.ndarray,
                rho: np.ndarray = rho,
                g: np.ndarray = g,
                hess: np.ndarray = hess,
                score: float = score,
                scale_now: float = scale_now,
            ) -> tuple[np.ndarray, tuple[np.ndarray, float, np.ndarray] | None, float, float]:
                """A trial point: ``(lsp1, fit, score1, qerror)``; the Taylor prediction uses the step taken
                after hgam's bound clip."""
                lsp1 = np.clip(rho + step, lo, hi)
                taken = lsp1 - rho
                fit = evaluate(lsp1)
                score1 = np.inf if fit is None else fit[1]
                pred_change = float(g @ taken + 0.5 * taken @ hess @ taken)
                change = score1 - score
                qerr = (
                    np.inf
                    if not np.isfinite(score1)
                    else abs(pred_change - change)
                    / (max(abs(pred_change), abs(change)) + scale_now * conv_tol)
                )
                return lsp1, fit, score1, qerr

            lsp1, fit1, score1, qerror = trial(n_step)
            new_point = None  # the accepted (lsp, fit), when derivatives are to be taken there
            ii = 0
            if np.isfinite(score1) and score1 < score and pdef and qerror < _QERROR_THRESH:
                new_point = (lsp1, fit1)  # immediately accept: it worked and the Hessian is positive definite
            elif not np.isfinite(score1) or score1 >= score or qerror >= _QERROR_THRESH:
                step = n_step  # step halving, from the (pseudo) Newton direction
                stored = None
                while (
                    not np.isfinite(score1) or score1 >= score or qerror >= _QERROR_THRESH
                ) and ii < max_half:
                    if ii == 3 and n_iter < 10:  # "Newton really not working": steepest descent, same length
                        s_length = min(float(np.sqrt(np.sum(step**2))), max_sd_step)
                        step = s_step * s_length / float(np.sqrt(np.sum(s_step**2)))
                        sd_unused = False
                    else:
                        step = step / 2.0
                    lsp1, fit1, score1, qerror = trial(step)
                    if ii > min(
                        4, max_half / 2
                    ):  # "don't allow step to fail altogether just because of qerror"
                        qerror = _QERROR_THRESH / 2.0
                    if np.isfinite(score1) and score1 < score and qerror < _QERROR_THRESH:
                        if pdef or not sd_unused:
                            new_point = (lsp1, fit1)
                        else:
                            stored = (lsp1, fit1, score1)  # the steepest descent trial below may do better
                        break
                    ii += 1
                if not pdef and sd_unused and ii < max_half and stored is not None:
                    lsp1, fit1, score1 = stored

            if not pdef and sd_unused:
                # Indefinite and steepest descent not yet tried: try it, and keep the better of the two.
                step = s_step * 2.0
                best = None  # (lsp2, fit2, score2)
                for _ in range(40):
                    step = step / 2.0
                    lsp3, fit3, score3, q3 = trial(step)
                    if (
                        best is None
                        or not np.isfinite(best[2])
                        or (np.isfinite(score3) and score3 <= best[2] and q3 < _QERROR_THRESH)
                    ):
                        best = (lsp3, fit3, score3)
                    if np.isfinite(best[2]) and np.isfinite(score3) and best[2] < score and score3 > best[2]:
                        break  # "stop when improvement found, and shorter step is worse"
                if best is not None and np.isfinite(best[2]) and best[2] < score1:
                    lsp1, fit1, score1 = best
                if fit1 is not None:
                    new_point = (lsp1, fit1)  # mgcv takes derivatives at the better step, whatever its score

            old_score = score
            if new_point is not None:
                move = float(np.max(np.abs(new_point[0] - rho))) / _LN10
                max_move = max(max_move, move)
                rho = new_point[0]
                coef, score, grad = new_point[1]
                hess = full_hessian(rho, coef)
                accepted.append(rho / _LN10)

            # mgcv's convergence test (gam.fit3.r, newton, end of each iteration).
            converged = not indef
            scale_now = score_scale(rho, coef, score)
            g = _project(grad, rho, lo, hi)
            grad2 = np.zeros(m, dtype=np.float64) if hess is None else np.diag(hess)
            uconv = (np.abs(g) > scale_now * conv_tol * 0.1) | (np.abs(grad2) > scale_now * conv_tol * 0.1)
            if np.any(np.abs(g) > scale_now * conv_tol * 5.0):
                converged = False
            if abs(old_score - score) > scale_now * conv_tol:
                if converged:
                    uconv[:] = True  # "otherwise can't progress"
                converged = False
            if ii == max_half:
                converged = True  # step failure
            if converged:
                break

        if ii == max_half:
            message = "step failed"
            warnings.warn(
                "newton_select_lambdas: fitting terminated with step failure - check results carefully",
                UserWarning,
                stacklevel=2,
            )
        elif not converged:
            message = "iteration limit reached"
            warnings.warn(
                "newton_select_lambdas: iteration limit reached without full convergence - check carefully",
                UserWarning,
                stacklevel=2,
            )
        else:
            message = "full convergence"

        gradient_tolerance = 5.0 * conv_tol * scale_now
    else:
        scale_now = float("nan")
        converged = False
        message = f"maxiter={maxiter} reached"
        n_iter = 0
        tol = conv_tol * (1.0 + abs(score))
        for n_iter in range(1, maxiter + 1):
            tol = conv_tol * (1.0 + abs(score))
            g_proj = _project(grad, rho, lo, hi)
            if float(np.max(np.abs(g_proj))) <= tol:
                converged = True
                message = "gradient test: max abs projected gradient <= conv_tol * (1 + abs(score))"
                n_iter -= 1
                break
            free = np.abs(g_proj) > tol  # converged directions are dropped from the step
            h_free = exact_hessian(rho, coef, free) if hessian == "exact" else difference_hessian(rho, free)
            step = np.zeros(m, dtype=np.float64)
            if h_free is not None:
                evals, evecs = np.linalg.eigh(h_free)
                floor = float(np.sqrt(np.finfo(np.float64).eps)) * float(np.max(np.abs(evals)))
                mag = np.maximum(np.abs(evals), max(floor, np.finfo(np.float64).tiny))
                step[free] = -(evecs @ ((evecs.T @ g_proj[free]) / mag))
            sd = np.zeros(m, dtype=np.float64)
            sd[free] = -g_proj[free]
            directions = [(step, max_step)] if h_free is not None else []
            directions.append((sd, max_sd_step))

            moved = None
            for direction, cap in directions:
                size = float(np.max(np.abs(direction)))
                if size == 0.0:
                    continue
                trial_step = direction * min(1.0, cap / size)
                for _ in range(max_half + 1):
                    trial = np.clip(rho + trial_step, lo, hi)
                    result = evaluate(trial)
                    if result is not None and result[1] < score:
                        moved = (trial, result)
                        break
                    trial_step = 0.5 * trial_step
                if moved is not None:
                    break
            if moved is None:
                message = "stalled: no Newton or steepest-descent step improved the score"
                break
            move = float(np.max(np.abs(moved[0] - rho))) / _LN10
            max_move = max(max_move, move)
            rho = moved[0]
            coef, score, grad = moved[1]
            accepted.append(rho / _LN10)

        gradient_tolerance = conv_tol * (1.0 + abs(score))

    log_lambda = rho / _LN10
    penalty = np.zeros_like(penalty_blocks[0])
    for lam, block in zip(np.exp(rho), penalty_blocks, strict=True):
        penalty = penalty + lam * block
    eta = (np.zeros_like(y) if offset is None else np.asarray(offset)) + x @ coef
    mu = family.link.linkinv(eta)
    edf_total = effective_degrees_of_freedom(x, family, eta, mu, penalty, weights)
    w_obs = family.observed_information_weight(
        y, eta, np.ones_like(y) if weights is None else np.asarray(weights)
    )
    kappa = float(np.linalg.cond(x.T @ (w_obs[:, None] * x) + penalty))
    return NewtonLambdaSelection(
        log_lambda=log_lambda,
        lambda_=10.0**log_lambda,
        coef=coef,
        reml_score=score,
        edf_total=edf_total,
        n_function_evals=tally["fits"],
        n_rejected=tally["rejected"],
        converged=converged,
        at_bound=bool(np.any(np.isclose(log_lambda, bounds[0])) or np.any(np.isclose(log_lambda, bounds[1]))),
        message=message,
        n_iterations=n_iter,
        max_abs_projected_gradient=float(np.max(np.abs(_project(grad, rho, lo, hi)))),
        gradient_tolerance=gradient_tolerance,
        score_scale=scale_now,
        accepted_log10=accepted,
        max_accepted_step_decades=max_move,
        hessian_condition=kappa,
        gradient_precision_floor=float(np.finfo(np.float64).eps) * kappa,
    )
