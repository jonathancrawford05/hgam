"""``fit_engine_bam`` — a :class:`~hgam.engine.gam_model.EngineGAMFit` from
``bam(method="fREML", discrete=TRUE)``'s performance iteration (real-data readiness epic,
Slice R3 Stage B, ADR-260 amendments 2-4).

The design and penalties come from the SAME producers ``fit_engine_gam`` uses
(:func:`~hgam.engine.gam_model.assemble_model_design`); only the smoothing-parameter
selection differs: :func:`~hgam.frontend.bam_fitd.bam_fitd` replaces the Laplace-REML Newton
search. The smoothing parameters are therefore the *iteration's end state*, which is what
``bam`` reports and why exact-REML ``gam`` sits ~1e-2 from it on weak-signal cells.

The covariates are first discretised as ``discrete.mf`` (bam.r) does
(:func:`~hgam.frontend.discrete.discretise_frame`), and every sum-to-zero constraint is taken over
``discrete.mf``'s padded frame (:func:`~hgam.frontend.discrete.discrete_model_frame`), as ``gam.setup``
takes it there. Without the binning the fit is mgcv's
``bam(discrete=FALSE)`` instead (``make_fixtures.R`` section 12 records both).

**Not ported (refused by name).** Free-scale and quasi families, ``gamma != 1``, and anything
needing the exact-REML Hessian (``vcov(unconditional=True)``); ``summary()`` reports no REML score.
"""

from collections.abc import Mapping

import numpy as np

from hgam.engine.gam_basis_cr import constraint_frame
from hgam.engine.gam_model import (
    EngineGAMFit,
    _dispersion_or_nan,
    _per_term_edf,
    _warn_if_rank_deficient,
    assemble_model_design,
    pivot_design,
    resolve_family,
)
from hgam.engine.gam_predict import fit_term_state
from hgam.engine.gam_term_spec import ModelSpec
from hgam.errors import HgamNotImplementedError
from hgam.frontend.bam_fitd import bam_fitd
from hgam.frontend.discrete import discrete_model_frame, discretise_frame

__all__ = ["fit_engine_bam"]


def fit_engine_bam(model: ModelSpec, data: Mapping[str, np.ndarray], y: np.ndarray) -> EngineGAMFit:
    """Fit ``model`` by the ``bgam.fitd`` performance iteration.

    Args:
        model: Poisson or binomial; terms ``cr`` / ``ti`` / ``sz`` / ``re`` / parametric (``select``
            as set on the model).
        data: covariate arrays keyed by name, plus the weights/offset columns if named.
        y: the response.

    Raises:
        HgamValidationError: an unsupported family, term basis or a structurally
            rank-deficient design.
        HgamComputationError: propagated from the iteration.
    """
    family = resolve_family(model.family, model.link)
    if not family.dispersion_fixed or family.name not in ("poisson", "binomial"):
        raise HgamNotImplementedError(
            f"gam(method='fREML', discrete=True): family {model.family!r} is not supported — "
            "only poisson and binomial (the fixed-scale branch of bgam.fitd). Free-scale, "
            "quasi and extended families are different builds (ADR-260 amendment 2)."
        )
    if any(term.basis == "tp" for term in model.terms):
        raise HgamNotImplementedError(
            "gam(method='fREML', discrete=True): tp terms (including a bare s(x)) are not "
            'supported — bam builds them on its compressed frame; write bs="cr" '
            "(ADR-261)."
        )
    # discrete.mf (bam.r): every numeric covariate is binned before the design is built, so the
    # fit, its edf and its term states (prediction) are at the discretised covariates.
    # gam.setup builds every smooth on discrete.mf's padded frame, so smoothCon's sum-to-zero
    # constraints are colMeans over its rows (discrete_model_frame), not over the data rows.
    mf0 = discrete_model_frame(model, data, y)
    data = discretise_frame(model, data)
    with constraint_frame(mf0):
        full_design = assemble_model_design(model, data)
    design = pivot_design(full_design)
    # A structurally redundant coefficient (a bare s(x) beside s(x, by=f), ADR-256) is pivoted
    # out exactly as the REML path does, with the same warning: eta, edf_total and the Vp se are
    # pivot-invariant. bgam.fitd pivots its own Cholesky for the same direction, so the fitted
    # model is the same one; which coefficient is eliminated (per-term edf) is not identified.
    _warn_if_rank_deficient(full_design, design, stacklevel=3)
    weights = (
        None if model.weights_column is None else np.asarray(data[model.weights_column], dtype=np.float64)
    )
    offset = None if model.offset_column is None else np.asarray(data[model.offset_column], dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    res = bam_fitd(design["x"], design["penalty_blocks"], y, family, offset=offset, prior_weights=weights)
    eta = (np.zeros_like(y) if offset is None else offset) + design["x"] @ res.coef
    mu = family.link.linkinv(eta)
    lam = np.exp(res.log_sp)
    penalty = np.zeros_like(design["penalty_blocks"][0])
    for lj, block in zip(lam, design["penalty_blocks"], strict=True):
        penalty = penalty + lj * block
    edf_total = float(np.sum(res.edf))
    with constraint_frame(mf0):
        term_states = tuple(fit_term_state(term, data) for term in model.terms)
    return EngineGAMFit(
        model=model,
        design=design,
        coef=res.coef,
        eta=eta,
        log_lambda=res.log_sp / np.log(10.0),
        lambda_=lam,
        edf_total=edf_total,
        edf_per_term=_per_term_edf(design, family, eta, mu, penalty, weights),
        reml_score=float("nan"),
        converged=res.converged,
        n_function_evals=res.n_iter,
        n_rejected=sum(it.n_halvings for it in res.trace),
        at_bound=False,
        at_bound_blocks=(),
        dispersion=_dispersion_or_nan(y, mu, family, edf_total, weights),
        term_states=term_states,
        y=y,
        prior_weights=weights,
        offset=offset,
        n_iterations=res.n_iter,
        vp_override=res.vp,
        xwx=res.xwx,
        outer_grad=res.grad,
        outer_hess=res.hess,
        fit_method="fREML-discrete",
    )
