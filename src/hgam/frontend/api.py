"""``gam()`` — the formula front end to the moved GAM engine (from polaris-re's ``gam/api.py``).

A thin typed facade: parse a formula **string** from the verified subset
(:mod:`hgam.frontend.formula`), encode a mapping of columns the way R does, build the
:class:`~hgam.engine.gam_term_spec.ModelSpec` the verified engine takes, and
call :func:`~hgam.engine.gam_model.fit_engine_gam` with **no solver
options** (its default is the deterministic Newton search, ADR-248). Nothing here is a
new numeric formula.

**Refusal is part of the contract.** Every construct outside the verified subset
raises :class:`~hgam.errors.HgamValidationError` naming the
construct and the ``MGCV_FEATURE_COVERAGE.md`` row that says why. Two refusals are
not about syntax:

* ``select=True`` is refused only when a bare smooth of a covariate sits beside a
  factor-``by`` smooth of the same covariate (the free-``sp`` search stops at a point
  ``mgcv`` scores lower on that shape, ADR-258); ``cr``, numeric-``by``, factor-``by``
  (``f + s(x, by=f)`` and ``s(x, by=f)``), ``ti``, ``re`` and the parametric block are
  accepted (ADR-217/218, ADR-252, ADR-258);
* a design whose **unpenalised null space is not identified by the data** (the
  structural condition behind ``rank(X) < p``, ADR-250) — most commonly
  ``s(x) + s(x, by=f)`` on one covariate — is **accepted** since parity-engine
  Slice 9: the unidentified coefficient is pivoted out of the fit as ``mgcv`` does
  (:mod:`~hgam.engine.gam_rank_pivot`; :attr:`GamFit.pivoted_columns`).
  A rank loss that depends on the weights is not structural and is not handled.

**Factor coding reproduces R's** (``factor()``, ``contr.treatment``): levels sorted
as R sorts them, the first level the reference. R sorts strings by the session's
``LC_COLLATE``; :data:`ORACLE_COLLATION` pins the rule for the oracle image and
``scripts/gam_formula_probe.R`` reports the image's own ordering so a mismatch is a
measured result. A :class:`Factor` column keeps its declared order (``factor(x,
levels=...)``). Only string arrays and :class:`Factor` columns are factors; a numeric
column used as a factor is refused rather than guessed at.

**Data.** ``data`` is a mapping from column name to a one-dimensional array (a ``dict`` of
NumPy arrays, or anything with ``name in data`` and ``data[name]``). polars was the data
frame in polaris-re; it is not a core dependency here (core stays at NumPy and SciPy).
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, overload

import numpy as np

from hgam.engine.gam_model import EngineGAMFit, fit_engine_gam
from hgam.engine.gam_rank_pivot import unidentified_directions
from hgam.engine.gam_term_spec import ModelSpec, TermSpec, factor_by_terms
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.bam_fit import fit_engine_bam
from hgam.frontend.discrete import discretise_frame
from hgam.frontend.formula import (
    ParametricCall,
    ParsedFormula,
    SzCall,
    TensorCall,
    parse_formula,
)
from hgam.frontend.summary import GamSummary, build_summary

__all__ = [
    "ORACLE_COLLATION",
    "Factor",
    "GamFit",
    "gam",
    "parametric_coefficient_names",
    "r_factor_levels",
    "structural_rank_deficiency",
]

ORACLE_COLLATION: Literal["C", "en_US"] = "en_US"
"""``LC_COLLATE`` rule used to sort string levels. Pinned to what the oracle image
reports (``scripts/gam_formula_probe.R`` -> ``collate``); see ADR-251."""

_COVERAGE = "docs/MGCV_FEATURE_COVERAGE.md"

_DEFAULT_LINK = {
    "gaussian": "identity",
    "poisson": "log",
    "quasipoisson": "log",
    "binomial": "logit",
}


def _collation_key(value: str, collation: str) -> tuple[object, ...]:
    if collation == "C":
        return (value,)
    if collation == "en_US":
        if not all(ch.isascii() and (ch.isalnum() or ch == "_") for ch in value):
            raise HgamNotImplementedError(
                f"gam(): level {value!r} has characters outside [A-Za-z0-9_]; its sort "
                "position under en_US collation is not reproduced here. Use a Factor "
                "column to state the level order explicitly."
            )

        # Measured on the oracle image (ADR-251, tier 3): '_' < digits < letters;
        # letters compare case-insensitively, lowercase before uppercase on a tie.
        def weight(ch: str) -> tuple[int, str]:
            if ch == "_":
                return (0, ch)
            return (1, ch) if ch.isdigit() else (2, ch.lower())

        return (
            tuple(weight(ch) for ch in value),
            tuple(ch.isupper() for ch in value),
        )
    raise HgamValidationError(f"gam(): unknown collation {collation!r}.")


def r_factor_levels(values: Sequence[str], collation: Literal["C", "en_US"] | None = None) -> tuple[str, ...]:
    """The level order R's ``factor(values)`` gives: ``sort(unique(values))`` under
    ``collation`` (default :data:`ORACLE_COLLATION`)."""
    rule = ORACLE_COLLATION if collation is None else collation
    unique = sorted(set(values), key=lambda v: _collation_key(v, rule))
    return tuple(unique)


@dataclass(frozen=True)
class Factor:
    """A factor column with a stated level order: R's ``factor(values, levels=levels)``.

    A plain string array is also a factor, with levels sorted as R sorts them
    (:func:`r_factor_levels`); use this type only to fix the order yourself."""

    values: Sequence[str] | np.ndarray
    levels: Sequence[str]


Column = np.ndarray | Factor
"""One column as :func:`_column` returns it: a 1-D array, or a :class:`Factor`."""


def _strings(col: Column, name: str) -> list[str]:
    values = col.values if isinstance(col, Factor) else col
    out = list(np.asarray(values, dtype=object).ravel())
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in out):
        raise HgamValidationError(f"gam(): column {name!r} contains nulls.")
    return [str(v) for v in out]


@dataclass(frozen=True)
class _FactorCoding:
    levels: tuple[str, ...]

    def encode(self, column: Column, name: str) -> np.ndarray:
        text = _strings(column, name)
        unseen = sorted(set(text) - set(self.levels))
        if unseen:
            raise HgamValidationError(
                f"gam(): column {name!r} has level {unseen[0]!r}, not seen when the model "
                f"was fitted ({list(self.levels)}); mgcv refuses an unseen level too."
            )
        index = {lv: i for i, lv in enumerate(self.levels)}
        return np.array([index[v] for v in text], dtype=np.int64)


@dataclass(frozen=True)
class GamFit(EngineGAMFit):
    """A :class:`~hgam.engine.gam_model.EngineGAMFit` that also remembers
    the formula and the factor coding, so :meth:`predict` takes the same kind of data as :func:`gam`."""

    formula: str = ""
    response: str = ""
    factor_levels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    numeric_columns: tuple[str, ...] = ()
    smooth_labels: tuple[str, ...] = ()
    """``mgcv``-style labels of the smooth terms (``s(x)``, ``s(x):w``, ``s(x):fa``,
    ``ti(x,z)``, ``s(f)``), in coefficient order after the parametric block."""
    smooth_bs_dim: tuple[int, ...] = ()
    """``mgcv``'s ``smooth$bs.dim`` per smooth: ``k`` for ``cr``, ``prod(k)`` for
    ``ti``, the level count for ``re``."""
    n_parametric: int = 0
    """``mgcv``'s ``m$nsdf``: the intercept plus the parametric block's columns."""
    parametric_names: tuple[str, ...] = ()
    """``mgcv``'s ``names(coef(m))[1:nsdf]``: ``(Intercept)``, then each parametric term's
    ``model.matrix`` column names (:func:`parametric_coefficient_names`)."""

    def summary(self) -> GamSummary:
        """mgcv counterpart: ``summary.gam`` (mgcv.r; ``bam`` fits use it too), mgcv 1.9-4, in part.

        The parametric table (``p.table``), per-smooth edf and ``log10(sp)``, scale, REML score
        (not for a ``discrete=True`` fit: bam's fREML score is not ported), deviance explained,
        ``n`` and the outer search's convergence report. The smooth-term tests (``Ref.df``,
        ``Chi.sq``, p-value) are not ported yet. ``print(fit.summary())`` renders it."""
        return build_summary(self, self.formula, self.smooth_labels, self.n_parametric, self.parametric_names)

    def encode(self, newdata: Mapping[str, object]) -> dict[str, np.ndarray]:
        """``newdata`` as the arrays :meth:`EngineGAMFit.predict` reads."""
        out: dict[str, np.ndarray] = {}
        for name in self.numeric_columns:
            out[name] = _numeric(newdata, name)
        for name, levels in self.factor_levels.items():
            out[name] = _FactorCoding(levels).encode(_column(newdata, name), name)
        return out

    @overload
    def predict(
        self,
        newdata: Mapping[str, np.ndarray],
        type: Literal["link", "response"] = "link",
        *,
        se_fit: Literal[False] = False,
        unconditional: bool = False,
        encoded: bool = False,
        discrete: bool = True,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        newdata: Mapping[str, np.ndarray],
        type: Literal["link", "response"] = "link",
        *,
        se_fit: Literal[True],
        unconditional: bool = False,
        encoded: bool = False,
        discrete: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(
        self,
        newdata: Mapping[str, object],
        type: Literal["link", "response"] = "link",
        *,
        se_fit: bool = False,
        unconditional: bool = False,
        encoded: bool = False,
        discrete: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """mgcv counterpart: ``predict.gam`` (mgcv.r); for a ``discrete=True`` fit, ``predict.bam`` and
        ``predict.bamd`` (bam.r), mgcv 1.9-4.

        ``newdata`` is encoded with the fitted factor coding (string levels in, offset column
        included), unless ``encoded=True`` says it already holds the integer codes and floats
        :meth:`EngineGAMFit.predict` takes.

        For a fit made with ``discrete=True`` and ``discrete=True`` here (``predict.bam``'s default),
        ``predict.bamd`` runs ``discrete.mf`` on ``newdata`` itself: each smooth covariate (and numeric
        ``by``) of ``newdata`` is compressed on its own by ``compress.df`` at ``m = NULL``, so above 1000
        unique values it is snapped to 1000 points over ``newdata``'s own range, and the prediction is
        made at those values (:func:`~hgam.frontend.discrete.discretise_frame`). ``predict.bamd``'s
        ``Xbd`` / ``diagXVXd`` products over the marginal matrices are replaced by the model matrix at
        the discretised rows, which gives the same eta and se. ``discrete=False`` predicts at the
        covariates as given (``predict.bam`` falling through to ``predict.gam``). Parametric covariates
        need no binning: only factors are accepted there. A factor level the fit did not see is refused
        (``predict.bamd`` sets its dummies to zero with a warning)."""
        data = newdata if encoded else self.encode(newdata)
        if discrete and self.fit_method == "fREML-discrete":
            data = discretise_frame(self.model, data)
        if se_fit:
            return super().predict(data, type, se_fit=True, unconditional=unconditional)
        return super().predict(data, type, se_fit=False, unconditional=unconditional)


def parametric_coefficient_names(
    terms: Sequence[TermSpec], factor_levels: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """The names R's ``model.matrix`` gives the parametric block under ``contr.treatment``:
    ``(Intercept)``, then per factor term its non-reference levels pasted to the variable name
    (``FaceSizeHigh``), and per interaction ``a:b`` the products, first-named variable fastest
    (``FaceSizeHigh:SmokeS``), as :func:`~hgam.engine.gam_basis_parametric.parametric_design`
    orders the columns."""
    names = ["(Intercept)"]
    for term in terms:
        if term.basis != "parametric":
            continue
        combos: list[list[str]] = [[]]
        for v in term.variables:
            combos = [[*c, f"{v}{lv}"] for lv in factor_levels[v][1:] for c in combos]
        names.extend(":".join(c) for c in combos)
    return tuple(names)


def _column(df: Mapping[str, object], name: str) -> Column:
    if name not in df:
        raise HgamValidationError(f"gam(): the data has no column {name!r} (columns: {list(df)}).")
    col = df[name]
    if isinstance(col, Factor):
        return col
    arr = np.asarray(col)
    if arr.ndim != 1:
        raise HgamValidationError(f"gam(): column {name!r} must be one-dimensional, got shape {arr.shape}.")
    return arr


def _dtype_name(col: Column) -> str:
    return "Factor" if isinstance(col, Factor) else str(col.dtype)


def _is_factor(col: Column) -> bool:
    return isinstance(col, Factor) or col.dtype.kind in "USO"


def _numeric(df: Mapping[str, object], name: str) -> np.ndarray:
    col = _column(df, name)
    if isinstance(col, Factor) or col.dtype.kind not in "iuf":
        raise HgamValidationError(
            f"gam(): column {name!r} has dtype {_dtype_name(col)}, but the formula uses it as a number."
        )
    values = col.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise HgamValidationError(f"gam(): column {name!r} contains nulls or non-finite values.")
    return values


def _factor_coding(df: Mapping[str, object], name: str) -> _FactorCoding:
    col = _column(df, name)
    if not _is_factor(col):
        raise HgamValidationError(
            f"gam(): column {name!r} has dtype {_dtype_name(col)}; factors must be string "
            "arrays or Factor columns (a numeric or Boolean column is never silently "
            "treated as a factor — convert it to strings, or to a Factor to fix the level order)."
        )
    text = _strings(col, name)
    if isinstance(col, Factor):
        levels = tuple(str(lv) for lv in col.levels)
        if len(set(levels)) != len(levels):
            raise HgamValidationError(f"gam(): Factor {name!r} lists a level twice: {list(levels)}.")
    else:
        levels = r_factor_levels(text)
    if len(levels) < 2:
        raise HgamValidationError(
            f"gam(): factor {name!r} has {len(levels)} level(s); mgcv needs at least 2."
        )
    return _FactorCoding(levels)


def _family(family: str) -> tuple[str, str]:
    text = family.strip()
    link = None
    if "(" in text:
        name, _, rest = text.partition("(")
        inner = rest.rstrip(")").strip()
        text = name.strip()
        if inner:
            key, _, value = inner.partition("=")
            if key.strip() != "link":
                raise HgamNotImplementedError(
                    f"gam(): family argument {inner!r} is not supported; "
                    "only link=... is (e.g. 'binomial(link=\"cloglog\")')."
                )
            link = value.strip().strip("'\"")
    if text not in _DEFAULT_LINK:
        raise HgamNotImplementedError(
            f"gam(): family {text!r} is not supported — verified families are "
            f"{sorted(_DEFAULT_LINK)} ({_COVERAGE} §2.2; Gamma, nb, tw and the extended "
            "families are not verified)."
        )
    return text, link or _DEFAULT_LINK[text]


def structural_rank_deficiency(x: np.ndarray, penalty_blocks: Sequence[np.ndarray]) -> int:
    """Number of unpenalised directions the data do not identify.

    Takes the null space ``N`` of ``S = sum_j S_j`` (the directions no penalty
    touches) and counts the rank deficiency of ``X N``. A coefficient direction in
    that null space that ``X`` cannot see is not estimable at *any* smoothing
    parameter — the structural form of ``rank(X) < p`` (ADR-250). It does not depend
    on ``y`` or on the fitted smoothing parameters.
    """
    return int(unidentified_directions(x, penalty_blocks).shape[1])


def _refuse_select_with_bare_and_factor_by(smooth: Sequence[TermSpec]) -> None:
    """Allow ``select=TRUE`` with a factor-``by`` smooth only when it is the ONLY smooth of its
    covariate (Slice R1, ADR-258, pre-registered handling R1-d; an allowlist, review P1).

    Measured and agreeing with ``mgcv``: ``f + s(x, by=f)`` and ``s(x, by=f)`` (120 of 120
    draws over two independent sets). A bare smooth beside the ``by`` smooth disagreed in 17 of
    120 draws, and a second ``by`` smooth (poisson) in 11 of 120. ``select=TRUE`` penalises the
    null space of every smooth, the null spaces overlap in one direction, and the data separate
    the four-or-more penalties that share it only weakly, so the Newton search stops, converged
    by its own test, at a point ``mgcv``'s REML scores lower (a plateau, mechanism class iii).
    A ``ti`` involving the covariate beside a factor ``by`` agreed in 59 of 60 draws but the one
    miss (a flat ridge, ``eta`` 3.5e-02) is a recorded disagreement, so it is refused to the same
    standard (zero misses in the accepted forms). A numeric ``by`` beside a factor ``by`` agreed
    in 60 of 60 draws and is accepted. A recorded limitation, not something to tune around.
    """
    groups: dict[str, set[str]] = {}
    for t in smooth:
        if t.basis == "ti":
            # a tensor interaction involving the covariate counts as another smooth of it
            for v in t.variables:
                groups.setdefault(v, set()).add(f"smooth:{t.label}")
            continue
        if len(t.variables) != 1 or t.basis not in ("cr", "tp"):
            continue
        if t.by is not None and t.by_factor is None:
            continue  # a numeric by smooth beside a factor by: 60 of 60 draws agree (ADR-258)
        key = f"by={t.by_factor}" if t.by_factor is not None else f"smooth:{t.label}"
        groups.setdefault(t.variables[0], set()).add(key)
    for var, keys in sorted(groups.items()):
        has_factor_by = any(k.startswith("by=") for k in keys)
        if has_factor_by and len(keys) > 1:
            raise HgamNotImplementedError(
                f"gam(): select=True with a factor-by smooth of {var!r} and another smooth of "
                f"{var!r} (a bare smooth, a second by-smooth or a ti term) "
                "is not "
                "supported (a thin-plate s(x) counts as a bare smooth, ADR-261 amendment 2) — "
                "measured against mgcv, the free-sp search stops at a point its REML "
                "scores lower in about 1 draw in 7 with a bare smooth and 1 in 11 with a second "
                "by-smooth (poisson), and for a ti term an equal-score ridge with a different "
                "eta (ADR-258, R1-d). Use one smooth of "
                f'{var!r}: f + s({var}, by=f, bs="cr"), or fit without select=True '
                f"({_COVERAGE} §2.3 select row)."
            )


def _tero(call: TensorCall) -> TensorCall:
    """mgcv counterpart: ``tero`` (bam.r), read from mgcv 1.9-1.

    ``mgcv:::tero`` (applied by ``bam(discrete=TRUE)``): swap the margin with the largest
    ``bs.dim`` (the last one on ties) into the last position. For two margins that is a swap
    exactly when ``k[0] > k[1]``. Deterministic, no RNG; it changes the coefficient order, the
    penalty order and the label (``ti(age,dur)`` with ``k=c(13,6)`` becomes ``ti(dur,age)``), and
    through the path-dependent iteration, the answer."""
    ks = call.k
    maxi, maxd = 0, -1
    for i, kk in enumerate(ks):
        if kk >= maxd:
            maxi, maxd = i, kk
    last = len(ks) - 1
    if maxi >= last:
        return call
    order = list(range(len(ks)))
    order[maxi], order[last] = order[last], order[maxi]
    return TensorCall(variables=tuple(call.variables[i] for i in order), k=tuple(ks[i] for i in order))


def _supplied_knots(knots: Mapping[str, Sequence[float]] | None) -> dict[str, tuple[float, ...]]:
    """``knots=`` as ``{variable: locations}``, mgcv's ``knots = list(x = c(...))``."""
    if knots is None:
        return {}
    if not isinstance(knots, Mapping):
        raise HgamValidationError("gam(): knots= must be a mapping from variable name to knot locations.")
    out: dict[str, tuple[float, ...]] = {}
    for name, locations in knots.items():
        try:
            values = np.asarray(locations, dtype=np.float64)
        except (TypeError, ValueError) as err:
            raise HgamValidationError(
                f"gam(): knots[{name!r}] must be a 1-D sequence of finite numbers."
            ) from err
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise HgamValidationError(f"gam(): knots[{name!r}] must be a 1-D sequence of finite numbers.")
        out[str(name)] = tuple(float(v) for v in values)
    return out


def _build_model(
    parsed: ParsedFormula,
    df: Mapping[str, object],
    family: str,
    link: str,
    weights: str | None,
    offset: str | None,
    select: bool,
    allow_unverified_select: bool = False,
    tero: bool = False,
    knots: Mapping[str, Sequence[float]] | None = None,
) -> tuple[ModelSpec, dict[str, np.ndarray], dict[str, _FactorCoding], list[tuple[str, int]]]:
    data: dict[str, np.ndarray] = {}
    supplied = _supplied_knots(knots)
    used_knots: set[str] = set()

    def cr_knots(variable: str, k: int, where: str) -> tuple[tuple[str, tuple[float, ...]], ...] | None:
        """mgcv's ``smooth.construct.cr.smooth.spec`` reads ``knots[[term]]`` and stops when its
        length is not ``k``."""
        if variable not in supplied:
            return None
        used_knots.add(variable)
        if len(supplied[variable]) != k:
            raise HgamValidationError(
                f"gam(): {where}: number of supplied knots for {variable!r} "
                f"({len(supplied[variable])}) != k ({k}) for a cr smooth (mgcv stops here too)."
            )
        return ((variable, supplied[variable]),)

    def refuse_knots(variable: str, where: str) -> None:
        if variable in supplied:
            raise HgamNotImplementedError(
                f"gam(): knots= for {variable!r} reaches {where}; supplied knots are ported for "
                'bs="cr" smooths and ti(bs="cr") margins only.'
            )

    codings: dict[str, _FactorCoding] = {}

    def need_numeric(name: str, where: str) -> None:
        col = _column(df, name)
        if _is_factor(col):
            raise HgamValidationError(
                f"gam(): {where} uses {name!r} as a number, but it is a factor column "
                '(write s(f, bs="re") for a random effect).'
            )
        data[name] = _numeric(df, name)

    def need_factor(name: str) -> _FactorCoding:
        if name not in codings:
            codings[name] = _factor_coding(df, name)
            data[name] = codings[name].encode(_column(df, name), name)
        return codings[name]

    parametric: list[TermSpec] = []
    smooth: list[TermSpec] = []
    smooth_meta: list[tuple[str, int]] = []
    parametric_vars = [t for t in parsed.terms if isinstance(t, ParametricCall)]
    mains = {t.variables[0] for t in parametric_vars if len(t.variables) == 1}
    for pt in parametric_vars:
        for v in pt.variables:
            if not _is_factor(_column(df, v)):
                raise HgamNotImplementedError(
                    f"gam(): the bare term {pt.text!r} is a numeric column ({v!r}, dtype "
                    f"{_dtype_name(_column(df, v))}); parametric numeric terms are not in the verified "
                    f"subset — only factor main effects and a:b ({_COVERAGE} §2.1 parametric "
                    'row). Model it with s(x, bs="cr") or convert it to strings if it is a factor.'
                )
        coding = [need_factor(v) for v in pt.variables]
        if len(pt.variables) == 2 and not set(pt.variables) <= mains:
            raise HgamNotImplementedError(
                f"gam(): the interaction {pt.text} needs both main effects "
                f"({pt.variables[0]} and {pt.variables[1]}) in the formula; R's default "
                "coding of an interaction without them is not reproduced "
                f"({_COVERAGE} §2.1 parametric row)."
            )
        parametric.append(
            TermSpec(
                label=pt.text,
                variables=pt.variables,
                basis="parametric",
                levels=tuple(len(c.levels) for c in coding),
            )
        )

    for call in parsed.terms:
        if isinstance(call, ParametricCall):
            continue
        if isinstance(call, TensorCall):
            for v in call.variables:
                need_numeric(v, call.text)
            call = _tero(call) if tero else call
            margin_knots = tuple(
                pair
                for v, k in zip(call.variables, call.k, strict=True)
                for pair in (cr_knots(v, k, call.text) or ())
            )
            smooth.append(
                TermSpec(
                    label=call.text,
                    variables=call.variables,
                    basis="ti",
                    k=call.k,
                    knots=margin_knots or None,
                )
            )
            smooth_meta.append((call.text, int(np.prod(call.k))))
        elif isinstance(call, SzCall):
            # mgcv counterpart: smooth.construct.sz.smooth.spec (smooth.r, mgcv 1.9-4), cr base. The factor
            # columns of the design come first whatever the written order, but only s(f, x) is measured.
            factor_name, smoothed = call.variables
            if not _is_factor(_column(df, factor_name)):
                raise HgamNotImplementedError(
                    f"gam(): {call.text}: bs='sz' is ported with the factor written first, s(f, x, ...); "
                    f"{factor_name!r} is not a factor column."
                )
            need_numeric(smoothed, call.text)
            levels = need_factor(factor_name).levels
            # select=True: smoothCon's null.space.penalty need.full branch (smooth.r, mgcv 1.9-4) appends U U'
            # for the null space of the summed per-level penalties; the engine's null_space_penalty builds it.
            smooth.append(
                TermSpec(
                    label=call.text,
                    variables=(factor_name, smoothed),
                    basis="sz",
                    k=(call.k,),
                    knots=cr_knots(smoothed, call.k, call.text),
                    n_levels=len(levels),
                )
            )
            # mgcv's bs.dim for sz: prod(nf - 1) * the base smooth's bs.dim.
            smooth_meta.append((call.text, call.k * (len(levels) - 1)))
        elif call.basis == "re":
            refuse_knots(call.variable, call.text)
            levels = need_factor(call.variable).levels
            smooth.append(
                TermSpec(label=call.text, variables=(call.variable,), basis="re", n_levels=len(levels))
            )
            smooth_meta.append((call.text, len(levels)))
        else:
            assert call.k is not None
            need_numeric(call.variable, call.text)
            if call.basis == "tp":
                refuse_knots(call.variable, call.text)
                smooth.append(TermSpec(label=call.text, variables=(call.variable,), basis="tp", k=(call.k,)))
                smooth_meta.append((call.text, call.k))
            elif call.by is None:
                smooth.append(
                    TermSpec(
                        label=call.text,
                        variables=(call.variable,),
                        basis="cr",
                        k=(call.k,),
                        knots=cr_knots(call.variable, call.k, call.text),
                    )
                )
                smooth_meta.append((call.text, call.k))
            elif _is_factor(_column(df, call.by)):
                levels = need_factor(call.by).levels
                for spec in factor_by_terms(
                    base_label=call.text,
                    variable=call.variable,
                    k=call.k,
                    by_factor=call.by,
                    n_levels=len(levels),
                    knots=dict(cr_knots(call.variable, call.k, call.text) or ()).get(call.variable),
                    level_labels=tuple(f"{call.by}{lv}" for lv in levels),
                ):
                    smooth.append(spec)
                    smooth_meta.append((spec.label, call.k))
            else:
                need_numeric(call.by, call.text)
                label = f"{call.text}:{call.by}"
                smooth.append(
                    TermSpec(
                        label=label,
                        variables=(call.variable,),
                        basis="cr",
                        k=(call.k,),
                        knots=cr_knots(call.variable, call.k, call.text),
                        by=call.by,
                    )
                )
                smooth_meta.append((label, call.k))

    unused = sorted(set(supplied) - used_knots)
    if unused:
        raise HgamValidationError(
            f"gam(): knots= names {unused}, which no cr smooth or ti margin in the formula uses. "
            "mgcv ignores such entries; hgam refuses rather than ignore an argument."
        )
    if select and not allow_unverified_select:
        _refuse_select_with_bare_and_factor_by(smooth)
    if weights is not None:
        need_numeric(weights, "weights=")
    if offset is not None:
        need_numeric(offset, "offset")
    model = ModelSpec(
        family=family,
        link=link,
        terms=(*parametric, *smooth),
        weights_column=weights,
        offset_column=offset,
        select=select,
    )
    return model, data, codings, smooth_meta


def gam(
    formula: str,
    data: Mapping[str, object],
    family: str,
    *,
    weights: str | None = None,
    offset: str | None = None,
    select: bool = False,
    scale: float | None = None,
    allow_unverified_select: bool = False,
    method: str = "REML",
    discrete: bool = False,
    knots: Mapping[str, Sequence[float]] | None = None,
) -> GamFit:
    """Fit an ``mgcv`` GAM formula from the verified subset by REML (or fREML, ``discrete=True``).

    Args:
        formula: e.g. ``'y ~ g + s(x, bs="cr", k=8) + ti(x, z, bs="cr", k=c(6, 5))'``.
            Supported: a bare ``s(x, k=)`` / ``s(x, bs="tp", k=)`` (thin-plate, ``mgcv``'s
            default, ADR-261), ``s(x, bs="cr", k=, by=)`` (numeric or factor ``by``),
            ``s(f, bs="re")``, ``s(f, x, bs="sz", k=, xt=list(bs="cr"))``,
            ``ti(x, z, bs="cr", k=)``, factors ``a``, ``a:b``
            (with both main effects), ``offset(col)``. Anything else is refused by
            name (:class:`~hgam.errors.HgamValidationError`).
        data: a mapping from column name to a 1-D array. String arrays and
            :class:`Factor` columns are factors; no nulls.
        family: ``"gaussian"``, ``"poisson"``, ``"quasipoisson"``, ``"binomial"``,
            optionally with ``link=`` (``'binomial(link="cloglog")'``).
        weights: name of a prior-weights column.
        offset: name of an offset column (alternative to ``offset(col)`` in the
            formula; not both).
        select: ``mgcv``'s ``select=TRUE``. Refused only when a bare smooth of the same covariate
            sits beside a factor-``by`` smooth (ADR-258). Accepted with ``sz`` smooths (one null-space
            penalty per sz term, ``make_fixtures.R`` section 9).
        allow_unverified_select: a DIAGNOSTIC switch. Skips the ADR-258 refusal of ``select=True``
            with a factor-``by`` smooth beside another smooth of its covariate, so the fit runs
            although ~1 draw in 7 disagrees with ``mgcv`` (outer-search plateau). For measuring
            that limitation; never for results anyone relies on.
        scale: a fixed dispersion. **Refused:** it is verified only inside the
            conformance module (ADR-236/237), not through the production fitter.
        method, discrete: ``method="REML"`` (the default) is the exact Laplace REML search.
            ``method="fREML", discrete=True`` is ``bam(method="fREML", discrete=TRUE)``'s
            performance iteration (``hgam.frontend.bam_fitd``, ADR-260 amendments 2-4) on the
            covariates binned as ``discrete.mf`` bins them (``make_fixtures.R`` section 12; ``predict``
            re-bins new data as ``predict.bam`` does, section 15),
            for poisson and binomial only; the smoothing parameters are the iteration's end
            state, so they differ from ``method="REML"``'s by design. The pair is accepted
            only together; every other combination is refused by name.
        knots: ``mgcv``'s ``knots = list(x = c(...))``: knot locations per variable, used as given by
            every ``bs="cr"`` smooth of that variable, every ``ti`` margin on it and every ``sz`` smooth
            of it (mgcv counterpart: ``smooth.construct.cr.smooth.spec``, smooth.r, mgcv 1.9-4, which
            stops when the count is not ``k``). Knots for a variable reached only by a ``tp`` or ``re``
            term are refused, and so are names no smooth uses (mgcv ignores those).

    Returns:
        a :class:`GamFit` (a :class:`~hgam.engine.gam_model.EngineGAMFit`
        whose ``predict`` encodes new data with the fitted factor coding). The smoothing parameters come
        from the deterministic Newton REML search; there are no solver options.
    """
    if scale is not None:
        raise HgamNotImplementedError(
            "gam(): scale= (a fixed dispersion) is not supported — it is verified only in "
            f"the conformance module, not through fit_engine_gam ({_COVERAGE} §2.3, "
            "ADR-236/237; production wiring is ladder slice 7c)."
        )
    if (method, discrete) not in (("REML", False), ("fREML", True)):
        raise HgamNotImplementedError(
            f"gam(): method={method!r}, discrete={discrete!r} is not a supported pair. "
            "Supported: method='REML' with discrete=False (exact Laplace REML), or "
            "method='fREML' with discrete=True (bam's performance iteration, ADR-260)."
        )
    if not (isinstance(data, Mapping) or hasattr(data, "columns")):
        raise HgamValidationError("gam(): data must be a mapping from column name to array.")
    parsed = parse_formula(formula)
    if offset is not None and parsed.offset is not None:
        raise HgamValidationError("gam(): both offset= and offset(.) in the formula were given; use one.")
    family_name, link = _family(family)
    model, arrays, codings, smooth_meta = _build_model(
        parsed,
        data,
        family_name,
        link,
        weights,
        offset or parsed.offset,
        select,
        allow_unverified_select,
        tero=discrete,
        knots=knots,
    )
    y = _numeric(data, parsed.response)
    if parsed.response_denominator is not None:
        denominator = _numeric(data, parsed.response_denominator)
        if np.any(denominator == 0):
            raise HgamValidationError(
                f"gam(): the response divides by column {parsed.response_denominator!r}, which has zeros."
            )
        y = y / denominator
    if discrete:
        fit = fit_engine_bam(model, arrays, y)
    else:
        fit = fit_engine_gam(model, arrays, y, warn_stacklevel=3)
    if not fit.converged:
        warnings.warn(
            "gam(): the "
            + ("performance iteration" if discrete else "REML search")
            + " did not meet its convergence criterion; treat the fit "
            "with caution (fit.converged is False).",
            stacklevel=2,
        )
    nsdf = 1 + sum(
        b["end"] - b["start"]
        for b in fit.design["term_blocks"]
        if any(t.label == b["label"] and t.basis == "parametric" for t in model.terms)
    )
    used = {v for t in model.terms for v in (*t.variables, t.by) if v is not None}
    if model.offset_column is not None:
        used.add(model.offset_column)
    numeric_columns = tuple(sorted(used - set(codings)))
    return GamFit(
        **{f: getattr(fit, f) for f in EngineGAMFit.__dataclass_fields__},
        formula=formula,
        response=parsed.response
        + ("" if parsed.response_denominator is None else f" / {parsed.response_denominator}"),
        factor_levels={name: c.levels for name, c in codings.items()},
        numeric_columns=numeric_columns,
        smooth_labels=tuple(label for label, _ in smooth_meta),
        smooth_bs_dim=tuple(dim for _, dim in smooth_meta),
        n_parametric=nsdf,
        parametric_names=parametric_coefficient_names(
            model.terms, {name: c.levels for name, c in codings.items()}
        ),
    )
