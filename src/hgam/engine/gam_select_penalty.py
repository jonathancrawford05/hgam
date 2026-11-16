"""``select = TRUE``'s double penalty — mgcv-parity engine, PLAN slice 7
(``docs/PLAN_mgcv_parity_engine.md``).

``select = TRUE`` is what takes the target formula's smoothing-parameter count
from 13 to 21 (PLAN §1): Marra & Wood's null-space shrinkage adds one extra
penalty per smooth term, penalising exactly the null space its own existing
penalty block(s) leave unpenalised, with its own independent smoothing
parameter — so a term with no signal can be driven all the way to zero under
REML, not merely made smoother.

**The rule, read off ``mgcv``'s own setup path at tier 1 before any code was
written here** (never from ``mgcv``'s R source — Anchor 8's companion
licensing rule): let ``S_1, ..., S_m`` be a term's own already-existing
penalty blocks, each at its natural (unscaled, ``lambda = 1``) magnitude.
Eigendecompose ``S_combined = S_1 + ... + S_m`` and let ``U0`` span the
eigenvectors whose eigenvalue falls below :func:`numpy.linalg.matrix_rank`'s
own tolerance — the null space ``S_combined`` leaves unpenalised. The extra
penalty is ``S_null = U0 @ U0.T``.

This rule is **basis-agnostic** — measured, not assumed, against ``mgcv``'s
own ``gam(..., select=TRUE, fit=FALSE)$smooth[[i]]$S`` (the last entry, which
``select=TRUE`` appends) across every term archetype the target formula
uses: a single-block ``cr`` term, a two-block numeric-``by`` ``cr`` term, a
three-block ``ti`` term and a three-block ``sz`` term (two of its three
blocks are the term's own existing per-level penalties). All four agreed to
float round-trip precision with no per-basis special-casing, which is why
this module takes a term's penalty blocks generically rather than
dispatching on ``TermSpec.basis``.

``fit = FALSE`` is exact for this purpose, not an approximation of fitting:
``S`` depends only on the model's structure (knots, basis, ``by``/factor
variables), never on ``y`` — see ``scripts/gam_select_penalty_probe.R``'s
own header for the measurement this module is built from.

**Parity claim (per ``docs/VERIFICATION_STANDARD.md``, written before the
code):** hgam's :func:`null_space_penalty` computes the extra
penalty from a term's own already-independently-verified penalty block(s)
(ADR-194/200/205/215's ``cr``/``by``/``ti``/``sz`` producers) via NumPy's own
eigendecomposition and ``matrix_rank`` tolerance; ``mgcv`` computes it via
``gam(formula, family, data, knots, select = TRUE, fit = FALSE)$smooth[[i]]$S``
(the last entry) — the setup-only path that builds every smooth's own
null-space penalty without fitting a model; compared on the null-space
penalty matrix ``S_null``, for each of the six target-formula-shaped cases
:data:`~hgam.engine.gam_select_penalty.SELECT_PENALTY_CLAIM`'s own
comparator reads off ``scripts/gam_select_penalty_probe.R``.

This module is Stage A only. :func:`~hgam.engine.gam_model.assemble_model_design`
now dispatches on ``ModelSpec.select`` and appends each term's own extra
block built here (PLAN slice 7's Stage B,
:mod:`~hgam.engine.gam_select_multiterm_conformance`, ADR-217).
**Not yet built:** extending
:func:`~hgam.engine.gam_reml_optimize.select_lambdas_continuous` /
:func:`~hgam.engine.gam_model.fit_engine_gam`'s own free-``sp``
search to the doubled block count ``select = True`` produces — every case
this module and its Stage-B counterpart verify uses a fixed, externally-
supplied ``sp``. See ``docs/PLAN_mgcv_parity_engine.md`` slice 7 for what
remains.
"""

import numpy as np

from hgam.errors import HgamValidationError

__all__ = [
    "null_space_penalty",
]


def null_space_penalty(s_blocks: tuple[np.ndarray, ...]) -> tuple[np.ndarray, int] | None:
    """The extra penalty ``select = TRUE`` adds for one term — see the module
    docstring for the rule and how it was measured.

    Args:
        s_blocks: A term's own already-existing penalty block(s), each
            ``(p, p)`` — e.g. :attr:`~hgam.engine.gam_term_build.TermExtract.s`.
            Never a value read from ``mgcv``'s own output (the ADR-193
            mechanical test): every block here comes from an
            already-independently-verified Python producer
            (:func:`~hgam.engine.gam_basis_cr.cr_basis` and friends).

    Returns:
        ``(s_null, null_dim)`` — the extra penalty and its rank — or ``None``
        if the combined block is already full rank (no null space left to
        penalise, so ``select = TRUE`` would add nothing for this term).
        ``s_null`` is symmetrised (``(s + s.T) / 2``) to absorb the
        floating-point asymmetry an eigendecomposition-then-outer-product
        can introduce, matching
        :func:`~hgam.engine.gam_basis_cr.absorb_sum_to_zero_constraint`'s
        own convention.

    Raises:
        HgamValidationError: if ``s_blocks`` is empty, or its blocks are
            not all the same shape.
    """
    if not s_blocks:
        raise HgamValidationError("null_space_penalty needs at least one penalty block.")
    shapes = {block.shape for block in s_blocks}
    if len(shapes) != 1:
        raise HgamValidationError(
            f"null_space_penalty: penalty blocks have mismatched shapes {shapes} — "
            "they must all belong to the same term's design."
        )
    (p, p2) = next(iter(shapes))
    if p != p2:
        raise HgamValidationError(f"null_space_penalty: penalty blocks are {p}x{p2}, not square.")

    combined = np.zeros((p, p), dtype=np.float64)
    for block in s_blocks:
        combined = combined + block
    combined = (combined + combined.T) / 2.0

    rank = int(np.linalg.matrix_rank(combined))
    null_dim = p - rank
    if null_dim == 0:
        return None

    evals, evecs = np.linalg.eigh(combined)
    order = np.argsort(evals)  # ascending — the smallest `null_dim` are the null space.
    u0 = evecs[:, order[:null_dim]]
    s_null = u0 @ u0.T
    return (s_null + s_null.T) / 2.0, null_dim
