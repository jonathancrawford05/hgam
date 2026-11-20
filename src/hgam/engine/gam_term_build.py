"""Per-term design and penalty builders: the step from a ``TermSpec`` to one term's columns.

Moved from polaris-re's ``analytics/gam_stage_a.py``, which also held the conformance harness
(the R-side extraction readers and comparators). Only the builders the fitting path calls came
across; the harness and its ``VerificationClaim`` objects stayed behind. Each builder takes only
the covariate values and the term's own spec, never an mgcv payload.

Not a port: re-derived in polaris-re. What it reproduces is the per-term part of mgcv's
``smoothCon`` (smooth.r) with ``absorb.cons=TRUE``, through :mod:`hgam.engine.gam_basis_cr` and
its siblings.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hgam.engine.gam_basis_cr import (
    absorb_sum_to_zero_constraint,
    by_factor_mask_design,
    by_scale_design,
    constraint_rows,
    cr_basis,
    cr_default_knots,
    frame_column,
    sz_basis,
    ti_basis,
)
from hgam.engine.gam_basis_fs import fs_basis
from hgam.engine.gam_basis_parametric import parametric_design
from hgam.engine.gam_basis_re import re_basis
from hgam.engine.gam_basis_t2 import t2_basis
from hgam.engine.gam_basis_te import te_basis
from hgam.engine.gam_basis_tp import tp_basis
from hgam.engine.gam_term_spec import TermSpec
from hgam.errors import HgamNotImplementedError, HgamValidationError

__all__ = [
    "TermExtract",
    "build_python_cr_by_factor_term",
    "build_python_cr_term",
    "build_python_parametric_term",
    "build_python_re_term",
    "build_python_sz_term",
    "build_python_fs_term",
    "fs_frame",
    "te_frame",
    "build_python_t2_term",
    "build_python_te_term",
    "build_python_ti_term",
    "build_python_tp_term",
]


@dataclass(frozen=True)
class TermExtract:
    """One term's Stage-A artefacts, mirroring what ``gam_term_extract.R`` emits.

    Args:
        label: Matches the R side's per-term key, so a comparison keys off the same
            string on both sides rather than positional order.
        index_start: First coefficient column this term owns (inclusive).
        index_end: One past the last coefficient column this term owns (exclusive) —
            Python slice convention, since both sides of this comparison are Python
            or JSON-via-Python.
        design: This term's own columns of the fitted design, ``(n_rows, width)``.
        s: Every penalty this term carries, each ``(width, width)``. Empty for an
            unpenalized term (e.g. a factor block).
        rank: One rank per entry of :attr:`s`, same order.
        knots: Knot locations actually used, or ``None`` — always ``None`` for
            ``basis="raw"``, which has no knot recipe (:class:`TermSpec`'s own
            validation forbids supplying knots for it). Populated for mgcv-native
            terms.
    """

    label: str
    index_start: int
    index_end: int
    design: np.ndarray
    s: tuple[np.ndarray, ...]
    rank: tuple[int, ...]
    knots: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise HgamValidationError("A TermExtract needs a non-empty label.")
        if self.index_end <= self.index_start:
            raise HgamValidationError(
                f"TermExtract {self.label!r} has index range "
                f"[{self.index_start}, {self.index_end}) — end must exceed start."
            )
        width = self.index_end - self.index_start
        if self.design.shape[1] != width:
            raise HgamValidationError(
                f"TermExtract {self.label!r} carries a design block with "
                f"{self.design.shape[1]} column(s) but its index range spans {width}."
            )
        for j, block in enumerate(self.s):
            if block.shape != (width, width):
                raise HgamValidationError(
                    f"TermExtract {self.label!r} penalty S[{j}] is {block.shape} but "
                    f"the term's own width is {width}x{width}."
                )
        if len(self.rank) != len(self.s):
            raise HgamValidationError(
                f"TermExtract {self.label!r} carries {len(self.s)} penalty block(s) "
                f"but {len(self.rank)} rank value(s) — one rank per S_j is required."
            )


def build_python_cr_term(x: np.ndarray, term: TermSpec, by: np.ndarray | None = None) -> TermExtract:
    """The independent Python producer for a ``bs="cr"`` term (slice 2; slice 5 for
    ``term.by is not None``).

    Builds ``design_X``/``penalty_S`` from ``x`` and ``term``'s own recipe (``k``,
    and supplied knots if any) via :mod:`hgam.engine.gam_basis_cr` — never
    from ``mgcv``'s output: its signature takes only ``x``/``by`` (the covariate
    values) and ``term`` (the spec), not an R payload.

    When ``term.by`` is set (the MI term, slice 5), ``mgcv`` does not absorb any
    identifiability constraint on a numeric-``by`` smooth (measured, see
    :func:`~hgam.engine.gam_basis_cr.by_scale_design`'s docstring) — the
    design is the **unconstrained** ``k``-column basis with each row scaled by
    ``by``, and the penalty is that same unconstrained, un-rescaled-by-``by`` ``S``.
    Slice 2's constrained ``k-1`` path runs only when ``term.by is None``.

    Args:
        x: The covariate values the comparison is evaluated at — read off the R
            payload's own ``"x"`` field by the *caller*, not by this function, so
            the mechanical test sees a signature with no R payload in it.
        term: Must have ``basis="cr"`` and exactly one variable.
        by: The by-variable values, read off the R payload's own ``"by"`` field by
            the caller — required when ``term.by is not None``, and must be
            ``None`` otherwise (a term not declaring a ``by`` recipe has nothing
            for this array to scale).
    """
    if term.basis != "cr":
        raise HgamValidationError(
            f"build_python_cr_term only handles basis='cr'; {term.label!r} is basis={term.basis!r}."
        )
    if len(term.variables) != 1 or len(term.k) != 1:
        raise HgamValidationError(
            f"build_python_cr_term: {term.label!r} must name exactly one variable "
            f"and one k; got variables={term.variables!r}, k={term.k!r}."
        )
    if (term.by is not None) != (by is not None):
        raise HgamValidationError(
            f"build_python_cr_term: {term.label!r} has term.by={term.by!r} but the "
            f"by array is {'absent' if by is None else 'present'} — a numeric-by "
            f"term needs both or neither."
        )
    x = np.asarray(x, dtype=np.float64)
    k = term.k[0]
    supplied = term.knots_by_variable().get(term.variables[0])
    knots = np.asarray(supplied, dtype=np.float64) if supplied is not None else cr_default_knots(x, k)

    design_unc, s_unc = cr_basis(x, knots)
    if by is not None:
        by = np.asarray(by, dtype=np.float64)
        # mgcv counterpart: smoothCon (smooth.r), mgcv 1.9-4, the numeric-by branch: the sum-to-zero
        # constraint is dropped only when ``sd(by) > mean(by) * .Machine$double.eps * 1000``; otherwise
        # mgcv keeps it. That branch (a constant by) is not ported.
        if not np.std(by, ddof=1) > np.mean(by) * np.finfo(np.float64).eps * 1000.0:
            raise HgamNotImplementedError(
                f"{term.label!r}: a numeric by variable {term.by!r} that is constant, for which "
                "mgcv's smoothCon keeps the sum-to-zero constraint, is not ported."
            )
        design = by_scale_design(design_unc, by)
        s = s_unc
    else:
        rows = constraint_rows(term.variables[0], knots)
        design, s = absorb_sum_to_zero_constraint(design_unc, s_unc, rows)
    rank = int(np.linalg.matrix_rank(s))

    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=design.shape[1],
        design=design,
        s=(s,),
        rank=(rank,),
        knots=tuple(float(v) for v in knots),
    )


def build_python_tp_term(x: np.ndarray, term: TermSpec) -> TermExtract:
    """The independent Python producer for a ``bs="tp"`` term (Slice R4, ADR-261).

    Builds ``design_X`` / ``penalty_S`` from ``x`` and ``term.k`` alone via
    :mod:`hgam.engine.gam_basis_tp`; the signature takes no R payload
    (ADR-193's mechanical test). The coefficient gauge is not unique, so
    :func:`~hgam.engine.gam_basis_tp.compare_tp_basis` compares gauge-free.
    """
    if term.basis != "tp" or len(term.variables) != 1 or len(term.k) != 1:
        raise HgamValidationError(
            f"build_python_tp_term only handles a one-covariate basis='tp'; got {term.label!r} "
            f"(basis={term.basis!r}, variables={term.variables!r}, k={term.k!r})."
        )
    design_u, s_u, _ = tp_basis(np.asarray(x, dtype=np.float64), term.k[0])
    design, s = absorb_sum_to_zero_constraint(design_u, s_u)
    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=design.shape[1],
        design=design,
        s=(s,),
        rank=(int(np.linalg.matrix_rank(s)),),
    )


def build_python_ti_term(x1: np.ndarray, x2: np.ndarray, term: TermSpec) -> TermExtract:
    """The independent Python producer for a ``bs="ti"`` two-margin term (slice 5,
    ``ti(AttdAge, PolYear)``).

    Builds ``design_X`` and both penalty blocks from ``x1``/``x2`` and ``term``'s
    own recipe (``k`` per margin, and supplied knots per margin if any) via
    :func:`~hgam.engine.gam_basis_cr.ti_basis` — never from ``mgcv``'s
    output. Same mechanical-test shape as :func:`build_python_cr_term`: the
    signature takes only ``x1``/``x2`` (the shared per-margin covariate recipe)
    and ``term`` (the shared spec), not an R payload.

    Args:
        x1: Margin-1 covariate values, read off the R payload's own ``"x1"``
            field by the *caller*.
        x2: Margin-2 covariate values, read off the R payload's own ``"x2"``
            field by the *caller*.
        term: Must have ``basis="ti"`` and exactly two variables (this project's
            target formula has no ``ti()`` term over more than two — a third
            margin is out of scope until one is needed).
    """
    if term.basis != "ti":
        raise HgamValidationError(
            f"build_python_ti_term only handles basis='ti'; {term.label!r} is basis={term.basis!r}."
        )
    if len(term.variables) != 2 or len(term.k) != 2:
        raise HgamValidationError(
            f"build_python_ti_term: {term.label!r} must name exactly two variables "
            f"and two k values; got variables={term.variables!r}, k={term.k!r}."
        )
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    knots_by_var = term.knots_by_variable()
    var1, var2 = term.variables
    k1, k2 = term.k
    supplied1 = knots_by_var.get(var1)
    supplied2 = knots_by_var.get(var2)
    knots1 = np.asarray(supplied1, dtype=np.float64) if supplied1 is not None else cr_default_knots(x1, k1)
    knots2 = np.asarray(supplied2, dtype=np.float64) if supplied2 is not None else cr_default_knots(x2, k2)

    design, s1, s2 = ti_basis(
        x1,
        x2,
        knots1,
        knots2,
        constraint_rows1=constraint_rows(var1, knots1),
        constraint_rows2=constraint_rows(var2, knots2),
    )
    rank1 = int(np.linalg.matrix_rank(s1))
    rank2 = int(np.linalg.matrix_rank(s2))

    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=design.shape[1],
        design=design,
        s=(s1, s2),
        rank=(rank1, rank2),
        knots=None,
    )


def build_python_te_term(xs: Sequence[np.ndarray], term: TermSpec) -> TermExtract:
    """mgcv counterpart: ``smoothCon(te(..., bs = "cr"), absorb.cons = TRUE)`` (smooth.r), mgcv 1.9-4, through
    :func:`~hgam.engine.gam_basis_te.te_basis`: one penalty per margin, mgcv's ``rank`` vector, and the one
    sum-to-zero constraint on the whole product taken over the data rows, or, inside a
    :func:`~hgam.engine.gam_basis_cr.constraint_frame`, the term built on that frame's rows
    (:func:`te_frame`).

    Args:
        xs: One covariate per margin, in ``term.variables`` order.
        term: ``basis="te"`` with one ``k`` per margin; supplied knots per variable if any.
    """
    if term.basis != "te":
        raise HgamValidationError(
            f"build_python_te_term only handles basis='te'; {term.label!r} is basis={term.basis!r}."
        )
    if len(xs) != len(term.variables) or len(term.k) != len(term.variables):
        raise HgamValidationError(
            f"build_python_te_term: {term.label!r} needs one covariate and one k per margin; got "
            f"{len(xs)} covariate(s), variables={term.variables!r}, k={term.k!r}."
        )
    knots_by_var = term.knots_by_variable()
    supplied = [knots_by_var.get(v) for v in term.variables]
    basis = te_basis(
        [np.asarray(x, dtype=np.float64) for x in xs],
        term.k,
        [None if kn is None else np.asarray(kn, dtype=np.float64) for kn in supplied],
        frame=te_frame(term),
    )
    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=basis.design.shape[1],
        design=basis.design,
        s=basis.penalties,
        rank=basis.rank,
        knots=None,
    )


def te_frame(term: TermSpec) -> list[np.ndarray] | None:
    """Each margin's column of an active :func:`~hgam.engine.gam_basis_cr.constraint_frame`
    (``discrete.mf``'s ``mf0``, which ``gam.setup`` builds a ``te`` term on in ``bam(discrete = TRUE)``),
    or ``None`` outside one."""
    columns = [frame_column(v) for v in term.variables]
    if all(c is None for c in columns):
        return None
    return [np.asarray(c, dtype=np.float64) for c in columns]


def build_python_t2_term(xs: Sequence[np.ndarray], term: TermSpec) -> TermExtract:
    """mgcv counterpart: ``smoothCon(t2(..., bs = "cr"), absorb.cons = TRUE)`` (smooth.r), mgcv 1.9-4, through
    :func:`~hgam.engine.gam_basis_t2.t2_basis`: one identity penalty per penalised block, mgcv's ``rank``
    vector, in the **fit** parameterisation (``X``, smoothCon's sparse ``C`` constraint).

    mgcv fits in that parameterisation too, and only afterwards maps coefficients, ``Vp`` and ``Ve`` to
    the ``Cp`` one with ``gam.setup``'s ``G$P`` (``estimate.gam``, mgcv.r); edf, the linear predictor and
    its se are computed before that map. ``G$P`` is not ported: hgam keeps the fit parameterisation and
    predicts in it
    (``gam_predict``). Because the term's unpenalised block holds the constant, ``[1, X]`` and ``[1, Xp]``
    span the same space, so predictions agree (``tests/test_t2_fit.py`` measures it); coefficient-level
    outputs (``vcov``, ``summary``, ``k_check``) refuse a t2 fit.
    """
    if term.basis != "t2":
        raise HgamValidationError(
            f"build_python_t2_term only handles basis='t2'; {term.label!r} is basis={term.basis!r}."
        )
    if len(xs) != len(term.variables) or len(term.k) != len(term.variables):
        raise HgamValidationError(
            f"build_python_t2_term: {term.label!r} needs one covariate and one k per margin; got "
            f"{len(xs)} covariate(s), variables={term.variables!r}, k={term.k!r}."
        )
    supplied = [term.knots_by_variable().get(v) for v in term.variables]
    basis = t2_basis(
        [np.asarray(x, dtype=np.float64) for x in xs],
        term.k,
        [None if kn is None else np.asarray(kn, dtype=np.float64) for kn in supplied],
    )
    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=basis.design.shape[1],
        design=basis.design,
        s=basis.penalties,
        rank=basis.rank,
        knots=None,
    )


def fs_frame(term: TermSpec) -> tuple[np.ndarray, np.ndarray] | None:
    """The ``(x0, group0)`` rows an ``fs`` term is constructed on inside an active
    :func:`~hgam.engine.gam_basis_cr.constraint_frame` (``discrete.mf``'s ``mf0``), or ``None``."""
    factor_name, smoothed_name = term.variables
    x0 = frame_column(smoothed_name)
    if x0 is None:
        return None
    g0 = frame_column(factor_name)
    if g0 is None:
        raise HgamValidationError(f"fs_frame: the frame carries {smoothed_name!r} but not {factor_name!r}.")
    return np.asarray(x0, dtype=np.float64), np.asarray(g0, dtype=np.int64)


def build_python_fs_term(x: np.ndarray, group: np.ndarray, term: TermSpec) -> TermExtract:
    """mgcv counterpart: ``smooth.construct.fs.smooth.spec`` with ``xt = "cr"`` and ``smoothCon``'s penalty
    scaling (smooth.r), mgcv 1.9-4, through :func:`~hgam.engine.gam_basis_fs.fs_basis`.

    ``term.variables`` is ``(factor_name, smoothed_name)``. The ``1 + null_dim`` penalties keep mgcv's
    ``rank`` vector (``rank * nf``, then ``nf`` per null-space direction). No constraint is absorbed.
    Inside a :func:`~hgam.engine.gam_basis_cr.constraint_frame` (``bam(discrete = TRUE)``), ``P`` and the
    penalty scaling are taken over that frame's rows (:func:`fs_frame`).
    """
    if term.basis != "fs" or term.n_levels is None:
        raise HgamValidationError(
            f"build_python_fs_term only handles basis='fs' with n_levels set; {term.label!r} is "
            f"basis={term.basis!r}."
        )
    x = np.asarray(x, dtype=np.float64)
    supplied = term.knots_by_variable().get(term.variables[1])
    k = term.k[0]
    knots = np.asarray(supplied, dtype=np.float64) if supplied is not None else cr_default_knots(x, k)
    basis = fs_basis(x, np.asarray(group, dtype=np.int64), term.n_levels, knots, frame=fs_frame(term))
    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=basis.design.shape[1],
        design=basis.design,
        s=basis.penalties,
        rank=basis.rank,
        knots=tuple(float(v) for v in knots),
    )


def build_python_sz_term(x: np.ndarray, group: np.ndarray, n_levels: int, term: TermSpec) -> TermExtract:
    """The independent Python producer for a single-factor ``bs="sz"`` term
    (slice 6, e.g. ``s(FaceSize, AttdAge, bs="sz", xt=list(bs="cr"))``).

    Builds ``design_X`` and every penalty block from ``x``/``group``/``n_levels``
    and ``term``'s own recipe (``k``, and supplied knots if any) via
    :func:`~hgam.engine.gam_basis_cr.sz_basis` — never from ``mgcv``'s
    output. Same mechanical-test shape as :func:`build_python_cr_term` and
    :func:`build_python_ti_term`: the signature takes only the shared covariate
    recipe and ``term``, not an R payload.

    Args:
        x: The smoothed margin's covariate values, read off the R payload's own
            ``"x"`` field by the *caller*.
        group: 0-indexed factor-level code per row, read off the R payload's own
            ``"group"`` field by the *caller*.
        n_levels: Number of factor levels, read off the R payload's own
            ``"n_levels"`` field by the *caller*.
        term: Must have ``basis="sz"``. ``term.variables`` is
            ``(factor_name, smoothed_name)`` (:class:`TermSpec`'s own documented
            order, matching the target formula's own ``s(FaceSize, AttdAge, ...)``
            argument order) and ``term.k`` is the smoothed margin's single ``k``.
    """
    if term.basis != "sz":
        raise HgamValidationError(
            f"build_python_sz_term only handles basis='sz'; {term.label!r} is basis={term.basis!r}."
        )
    # TermSpec.__post_init__ already guarantees exactly two variables and one k
    # for basis="sz" (gam_term_spec.py) — no redundant re-check here.
    x = np.asarray(x, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    k = term.k[0]
    smoothed_var = term.variables[1]
    supplied = term.knots_by_variable().get(smoothed_var)
    knots = np.asarray(supplied, dtype=np.float64) if supplied is not None else cr_default_knots(x, k)

    design, s_blocks = sz_basis(x, group, n_levels, knots)
    rank = tuple(int(np.linalg.matrix_rank(s)) for s in s_blocks)

    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=design.shape[1],
        design=design,
        s=s_blocks,
        rank=rank,
        knots=tuple(float(v) for v in knots),
    )


def build_python_re_term(group: np.ndarray, n_levels: int, term: TermSpec) -> TermExtract:
    """The independent Python producer for a ``bs="re"`` term (capability ladder
    rung L2, ``docs/PLAN_mgcv_capability_ladder.md`` slice 2).

    Builds ``design_X``/``penalty_S`` from ``group``/``n_levels`` alone via
    :func:`~hgam.engine.gam_basis_re.re_basis` — never from ``mgcv``'s
    output. Same mechanical-test shape as :func:`build_python_sz_term`: the
    signature takes only the shared recipe (``group``, ``n_levels``) and
    ``term`` (the shared spec), not an R payload.

    Args:
        group: 0-indexed factor-level code per row, read off the R payload's
            own ``"group"`` field by the *caller*.
        n_levels: Number of factor levels, read off the R payload's own
            ``"n_levels"`` field by the *caller*.
        term: Must have ``basis="re"``. ``TermSpec.__post_init__`` already
            guarantees exactly one variable, no ``k``, and ``n_levels`` set
            and ``>= 2`` — no redundant re-check here.
    """
    if term.basis != "re":
        raise HgamValidationError(
            f"build_python_re_term only handles basis='re'; {term.label!r} is basis={term.basis!r}."
        )
    design, s = re_basis(group, n_levels)
    rank = int(np.linalg.matrix_rank(s))

    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=design.shape[1],
        design=design,
        s=(s,),
        rank=(rank,),
        knots=None,
    )


def build_python_parametric_term(groups: tuple[np.ndarray, ...], term: TermSpec) -> TermExtract:
    """The independent Python producer for a ``basis="parametric"`` term
    (capability ladder rung L4, ``docs/PLAN_mgcv_capability_ladder.md`` slice 5) —
    an unpenalized factor main effect (one variable) or interaction (two or more).

    Builds ``design_X`` from ``groups``/``term.levels`` alone via
    :func:`~hgam.engine.gam_basis_parametric.parametric_design` — never
    from ``mgcv``'s output. Same mechanical-test shape as
    :func:`build_python_re_term`: the signature takes only the shared recipe
    (one 0-indexed group array per variable) and ``term`` (the shared spec),
    not an R payload.

    Args:
        groups: One 0-indexed factor-level-code array per entry of
            ``term.variables``, in that order.
        term: Must have ``basis="parametric"``. ``TermSpec.__post_init__``
            already guarantees ``levels`` is set with one entry per variable,
            each ``>= 2`` — no redundant re-check here.
    """
    if term.basis != "parametric":
        raise HgamValidationError(
            f"build_python_parametric_term only handles basis='parametric'; "
            f"{term.label!r} is basis={term.basis!r}."
        )
    assert term.levels is not None  # TermSpec guarantees this for basis='parametric'
    design = parametric_design(groups, term.levels)

    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=design.shape[1],
        design=design,
        s=(),
        rank=(),
        knots=None,
    )


def build_python_cr_by_factor_term(x: np.ndarray, group: np.ndarray, term: TermSpec) -> TermExtract:
    """The independent Python producer for ONE LEVEL of a factor-``by`` ``bs="cr"``
    term (capability ladder rung L3, ``docs/PLAN_mgcv_capability_ladder.md`` slice 4,
    e.g. ``s(x, by=fac, bs="cr")``).

    Builds ``design_X``/``penalty_S`` from ``x`` (the WHOLE covariate column, every
    row) and ``term``'s own recipe (``k``, supplied knots if any, ``term.by_level``)
    via :mod:`hgam.engine.gam_basis_cr` — never from ``mgcv``'s output. Same
    mechanical-test shape as :func:`build_python_cr_term`: the signature takes only
    ``x``/``group`` (the shared covariate recipe) and ``term`` (the shared spec), not
    an R payload.

    ``term.by_factor``/``term.by_level`` select the masking (module docstring of
    :mod:`~hgam.engine.gam_basis_cr`): the shared no-``by`` constrained
    basis is computed on ALL of ``x`` (not restricted to this level — mgcv's own
    default-knot placement and identifiability constraint use the whole column),
    then every row outside ``group == term.by_level`` is zeroed.

    Args:
        x: The covariate values (the WHOLE column, every level), read off the R
            payload's own ``"x"`` field by the *caller*.
        group: 0-indexed factor-level code per row, read off the R payload's own
            ``"group"`` field by the *caller*.
        term: Must have ``basis="cr"`` with ``by_factor``/``by_level`` both set
            (:class:`TermSpec`'s own validation guarantees ``by_level`` lies in
            ``[0, n_levels)``).
    """
    if term.basis != "cr" or term.by_factor is None:
        raise HgamValidationError(
            f"build_python_cr_by_factor_term needs basis='cr' with by_factor set; "
            f"{term.label!r} has basis={term.basis!r}, by_factor={term.by_factor!r}."
        )
    assert term.by_level is not None  # TermSpec guarantees this alongside by_factor
    x = np.asarray(x, dtype=np.float64)
    group = np.asarray(group, dtype=np.int64)
    k = term.k[0]
    supplied = term.knots_by_variable().get(term.variables[0])
    knots = np.asarray(supplied, dtype=np.float64) if supplied is not None else cr_default_knots(x, k)

    design_unc, s_unc = cr_basis(x, knots)
    rows = constraint_rows(term.variables[0], knots)
    design_shared, s = absorb_sum_to_zero_constraint(design_unc, s_unc, rows)
    design = by_factor_mask_design(design_shared, group, term.by_level)
    rank = int(np.linalg.matrix_rank(s))

    return TermExtract(
        label=term.label,
        index_start=0,
        index_end=design.shape[1],
        design=design,
        s=(s,),
        rank=(rank,),
        knots=tuple(float(v) for v in knots),
    )
