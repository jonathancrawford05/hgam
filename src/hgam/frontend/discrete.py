"""``bam(discrete=TRUE)``'s covariate compression: mgcv's ``compress.df``.

Moved from polaris-re's ``gam/bam_binning_conformance.py``; the comparison harness and its
claim objects stayed behind. Ported from the mgcv 1.9-1 source; polaris-re checked the pinned
1.9-4 by behaviour only. :func:`discretise_frame` is the subset of ``discrete.mf`` the
supported terms reach; ``fit_engine_bam`` fits on the frame it returns.

What the source says: for one covariate ``m = 1000``; two jointly ``m = 100``; more, ``25``.
If the covariate(s) have no more than ``m ** n_numeric * prod(n_factor_levels)`` unique rows,
nothing is binned and the cells are the unique rows. Only above that are numeric columns snapped
to ``m`` evenly spaced points over their range.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from hgam.engine.gam_term_spec import ModelSpec, TermSpec
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.r_rng import RRandom

__all__ = [
    "CompressedTerm",
    "compress_covariates",
    "default_m",
    "r_double_string",
    "discrete_model_frame",
    "discretise_frame",
]

_DISCRETE_SEED = 8547  # discrete.mf's temp.seed(8547), bam.r

_M_ONE_D = 1000
_M_TWO_D = 100
_M_MORE_D = 25


def default_m(n_columns: int, m: int | None = None) -> int:
    """mgcv counterpart: the ``m`` default in ``compress.df`` (bam.r), read from mgcv 1.9-1.

    ``compress.df``'s per-dimension grid size.

    ``m = NULL``: 1000 / 100 / 25 for one / two / more columns. ``m`` supplied and more than
    one column: ``round(m ** (1 / d)) + 1`` (R's round is half-to-even, as ``np.rint``)."""
    if m is None:
        return _M_ONE_D if n_columns == 1 else _M_TWO_D if n_columns == 2 else _M_MORE_D
    if n_columns > 1:
        return int(np.rint(m ** (1.0 / n_columns))) + 1
    return m


@dataclass(frozen=True)
class CompressedTerm:
    """One ``compress.df`` call's result: the unique covariate rows and each data row's cell.

    ``values[name][index]`` is the discretised covariate for every data row. Factor columns
    are carried as integer codes (sorted-unique order); they are never binned. ``index`` is
    0-based. ``binned`` records whether the grid rule fired."""

    names: tuple[str, ...]
    values: dict[str, np.ndarray]
    index: np.ndarray
    binned: bool

    @property
    def nr(self) -> int:
        return int(next(iter(self.values.values())).shape[0])

    def row_values(self, name: str) -> np.ndarray:
        return self.values[name][self.index]


def r_double_string(x: float) -> str:
    """R counterpart: ``as.character`` on a double (``formatReal``, R >= 4.3.0), as ``paste0`` calls it.

    R gives the shortest of up to 15 significant digits that equals ``x`` rounded to 15 significant
    digits, then writes it fixed or scientific, whichever is narrower (fixed on a tie; ``scipen = 0``):
    ``1e+05`` but ``123456``, ``1e-04`` but ``0.001``. The exponent has at least two digits. Read from
    R 4.3.3 by behaviour (``sprintf("%.17g")`` beside ``as.character`` on 46 values, including the
    width ties); the oracle image runs R 4.6.1, after the same 4.3.0 change.

    Raises:
        HgamValidationError: a non-finite value (R writes ``NaN``, ``Inf``; not reached here).
        HgamNotImplementedError: ``|x| >= 1e15`` written in fixed notation (R prints more digits).
    """
    if not np.isfinite(x):
        raise HgamValidationError(f"r_double_string: {x!r} is not finite.")
    if x == 0.0:
        return "0"
    sign = "-" if x < 0 else ""
    mantissa, exp_text = f"{abs(x):.14e}".split("e")
    digits = mantissa.replace(".", "").rstrip("0")
    e = int(exp_text)
    nsig = len(digits)
    sci = digits[0] + ("." + digits[1:] if nsig > 1 else "")
    sci += "e" + ("-" if e < 0 else "+") + f"{abs(e):02d}"
    if e >= 0:
        whole = digits[: e + 1].ljust(e + 1, "0")
        frac = digits[e + 1 :]
    else:
        whole = "0"
        frac = "0" * (-e - 1) + digits
    fixed = whole + ("." + frac if frac else "")
    if e >= 15 and len(fixed) <= len(sci):
        # R writes more than 15 digits here (123456789012345680 for 1.2345678901234568e17, but 2^60 exactly);
        # that rule is not ported.
        raise HgamNotImplementedError(
            f"r_double_string: {x!r} (fixed notation at 1e15 or above) is not ported."
        )
    return sign + (fixed if len(fixed) <= len(sci) else sci)


def _unique_rows(
    columns: list[np.ndarray],
    factors: tuple[bool, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """mgcv counterpart: ``uniquecombs(x, ordered = TRUE)`` (smooth.r), mgcv 1.9-4.

    The unique rows and each row's 0-based unique index. One column: ``sort(unique(x))``, numeric order,
    as ``np.unique``. More than one: each row's text ``paste0(x1, "*", x2, ...)`` (R's ``as.character``,
    :func:`r_double_string`; a factor as its 1-based level code), duplicates removed on that text with the
    first occurrence kept, and the unique rows ordered by their text in the C locale (byte order, as
    Python compares ASCII strings). So two doubles that agree to 15 significant digits are one row.
    ``factors`` flags factor columns, carried as 0-based codes."""
    stacked = np.column_stack(columns).astype(np.float64)
    if stacked.shape[1] == 1:
        uniq, inverse = np.unique(stacked, axis=0, return_inverse=True)
        return uniq, inverse.reshape(-1).astype(np.int64)
    flags = factors if factors is not None else (False,) * stacked.shape[1]
    text_columns = []
    for j, is_factor in enumerate(flags):
        col = stacked[:, j]
        values, inv = np.unique(col, return_inverse=True)
        if is_factor:
            strings = [str(int(v) + 1) for v in values]
        else:
            strings = [r_double_string(float(v)) for v in values]
        text_columns.append(np.asarray(strings, dtype=object)[inv.reshape(-1)])
    keys = ["*".join(parts) for parts in zip(*text_columns, strict=True)]
    first: dict[str, int] = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    ordered = sorted(first)
    position = {key: j for j, key in enumerate(ordered)}
    uniq = stacked[[first[key] for key in ordered]]
    inverse = np.fromiter((position[key] for key in keys), dtype=np.int64, count=len(keys))
    return uniq, inverse


def compress_covariates(
    columns: dict[str, np.ndarray],
    factors: frozenset[str] = frozenset(),
    *,
    m: int | None = None,
    rng: RRandom | None = None,
) -> CompressedTerm:
    """mgcv counterpart: ``compress.df`` (bam.r), read from mgcv 1.9-1.

    Port of ``mgcv:::compress.df`` for one term's covariate columns.

    ``uniquecombs(dat, TRUE)`` returns the unique rows sorted (:func:`_unique_rows`: numerically for one
    column, by their text for several, as for a ``t2`` term's jointly compressed covariates). mgcv then
    shuffles them (``ii <- sample(1:nrow(xu), nrow(xu)); xu[ii, ] <- xu``) and remaps the index.
    With ``rng`` (R's stream, :class:`~hgam.frontend.r_rng.RRandom`) the shuffle is drawn as
    mgcv draws it; without it the rows stay sorted. Either way ``row_values`` is unchanged.

    Args:
        columns: name to 1-d float array (numeric) or integer codes (factor), equal length.
        factors: names in ``columns`` that are factors (never binned; they multiply the
            threshold by their number of levels).
        m: ``bam``'s ``m`` argument (``None`` = the defaults).
        rng: R's stream to draw the shuffle from (``None``: no shuffle, no draw).

    Raises:
        HgamValidationError: empty input, ragged columns, or non-finite values.
    """
    if not columns:
        raise HgamValidationError("compress_covariates: no covariate columns given.")
    names = tuple(columns)
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}
    n = next(iter(arrays.values())).shape[0]
    if any(a.shape != (n,) for a in arrays.values()):
        raise HgamValidationError("compress_covariates: columns must be 1-d of equal length.")
    if any(not np.all(np.isfinite(a)) for a in arrays.values()):
        raise HgamValidationError(
            "compress_covariates: non-finite covariate values (NaN/Inf) are not discretised."
        )
    d = len(names)
    m_eff = default_m(d, m)
    factor_levels = 1
    numeric = [k for k in names if k not in factors]
    for k in names:
        if k in factors:
            factor_levels *= int(np.unique(arrays[k]).shape[0])
    threshold = (m_eff ** len(numeric)) * factor_levels

    flags = tuple(k in factors for k in names)
    uniq, _ = _unique_rows([arrays[k] for k in names], flags)
    binned = False
    if uniq.shape[0] > threshold:
        binned = True
        for k in numeric:
            lo, hi = float(arrays[k].min()), float(arrays[k].max())
            # R: xu <- seq(xl[1], xl[2], length = m); dx <- xu[2] - xu[1]
            by = (hi - lo) / (m_eff - 1)
            grid = lo + np.arange(m_eff, dtype=np.float64) * by
            grid[-1] = hi
            dx = grid[1] - grid[0]
            kx = np.rint((arrays[k] - lo) / dx).astype(np.int64)
            arrays[k] = grid[kx]
        uniq, _ = _unique_rows([arrays[k] for k in names], flags)

    if uniq.shape[0] == n:
        # R returns the (possibly binned) data itself with index 1:n.
        return CompressedTerm(
            names=names,
            values={k: arrays[k] for k in names},
            index=np.arange(n, dtype=np.int64),
            binned=binned,
        )
    uniq, inverse = _unique_rows([arrays[k] for k in names], flags)
    if rng is not None:
        ii = rng.sample_int(uniq.shape[0], uniq.shape[0]) - 1
        shuffled = np.empty_like(uniq)
        shuffled[ii] = uniq
        uniq, inverse = shuffled, ii[inverse]
    return CompressedTerm(
        names=names,
        values={k: uniq[:, i] for i, k in enumerate(names)},
        index=inverse,
        binned=binned,
    )


def _smooth_groups(term: TermSpec) -> list[tuple[str, ...]]:
    """The covariate groups ``discrete.mf`` compresses for one smooth, in its order: the ``by`` variable
    (numeric or factor) first, then each margin. A ``tensor.smooth.spec`` (``ti``, ``te``, after ``tero``)
    has one one-covariate margin per variable; a one-covariate smooth (``cr``, ``tp``) is one group.
    ``sz`` is a tensor of one-covariate margins there
    (``tensor.possible``, bam.r), factor first; ``re`` is its factor. ``fs`` is ``tensor.possible`` too,
    with one margin per written term (``TermSpec.margin_order``). A ``t2`` term is not a
    ``tensor.smooth.spec`` (``nmarg <- 1``), so its covariates are one group, compressed jointly."""
    if term.basis == "parametric":
        return []
    if term.basis not in ("cr", "tp", "ti", "te", "t2", "sz", "fs", "re"):
        raise HgamNotImplementedError(
            f"discrete_model_frame: {term.basis!r} terms are not discretised "
            "(only cr, tp, ti, te, t2, sz, fs, re and factors)."
        )
    by = term.by if term.by is not None else term.by_factor
    head = [(by,)] if by is not None else []
    if term.basis == "t2":
        return head + [term.variables]
    if term.basis == "fs":
        return [(v,) for v in (term.margin_order if term.margin_order is not None else term.variables)]
    return head + [(v,) for v in term.variables]


def _new_groups(model: ModelSpec) -> list[tuple[str, ...]]:
    """mgcv counterpart: ``check.term`` (bam.r), mgcv 1.9-4, over every smooth's groups in term order.

    A group whose variables were all compressed before, as a group of the same size, reuses that
    compression; a group that shares a variable with an earlier group of another size stops mgcv
    ("bam can not discretize with this nesting structure"). Returns the groups that need compressing."""
    seen: dict[str, tuple[str, ...]] = {}
    out: list[tuple[str, ...]] = []
    for term in model.terms:
        for group in _smooth_groups(term):
            prev = [seen[v] for v in group if v in seen]
            if not prev:
                out.append(group)
                for v in group:
                    seen[v] = group
                continue
            if len(group) != len(prev[0]) or any(v not in prev[0] for v in group):
                raise HgamValidationError(
                    f"discrete_model_frame: {term.label!r} compresses {group!r} beside an earlier group "
                    f"{prev[0]!r}; mgcv stops: 'bam can not discretize with this nesting structure'."
                )
    return out


def _factor_names(model: ModelSpec) -> frozenset[str]:
    names: set[str] = set()
    for term in model.terms:
        if term.basis in ("parametric", "re"):
            names.update(term.variables)
        elif term.basis in ("sz", "fs"):
            names.add(term.variables[0])
        if term.by_factor is not None:
            names.add(term.by_factor)
    return frozenset(names)


def discrete_model_frame(
    model: ModelSpec, data: Mapping[str, np.ndarray], response: np.ndarray
) -> dict[str, np.ndarray]:
    """mgcv counterpart: ``discrete.mf`` (bam.r) with ``full = TRUE``, mgcv 1.9-1 (unchanged in
    1.9-4): the padded frame ``mf0`` that ``bam(discrete=TRUE)`` hands ``gam.setup``.

    Under ``temp.seed(8547)`` each covariate is compressed once (``compress.df``, shuffled): first
    every smooth's ``by`` and margins in term order, then the parametric frame's columns not yet
    done, in ``model.frame``'s order: the response, the factors of the parametric terms (R's
    ``terms`` order: by degree, then as written), ``(weights)``, the offset. Every column is then
    padded to the longest with ``sample(mf0[[i]], maxr - me, replace = TRUE)``, in that order.
    ``smoothCon`` takes each sum-to-zero constraint as ``colMeans(X)`` over these rows.

    Returns each compressed column under its name (the response under ``"(response)"``),
    ``maxr`` rows each. An offset beside prior weights is refused: whether the offset sits before
    ``(weights)`` (written in the formula) or after it (``offset=``) is not carried by the model.
    """
    if model.offset_column is not None and model.weights_column is not None:
        raise HgamNotImplementedError(
            "discrete_model_frame: an offset together with prior weights is not ported "
            "(their order in discrete.mf's parametric frame depends on how the offset was given)."
        )
    factors = _factor_names(model)
    groups = _new_groups(model)
    done = {v for g in groups for v in g}
    parametric = sorted((t for t in model.terms if t.basis == "parametric"), key=lambda t: len(t.variables))
    pmf: dict[str, np.ndarray] = {"(response)": np.asarray(response, dtype=np.float64)}
    for term in parametric:
        for name in term.variables:
            pmf.setdefault(name, np.asarray(data[name], dtype=np.float64))
    for extra in (model.weights_column, model.offset_column):
        if extra is not None:
            pmf.setdefault(extra, np.asarray(data[extra], dtype=np.float64))
    groups += [(n,) for n in pmf if n not in done]
    rng = RRandom(_DISCRETE_SEED)
    mf0: dict[str, np.ndarray] = {}
    for group in groups:
        values = {
            n: pmf[n] if n in pmf and n not in done else np.asarray(data[n], dtype=np.float64) for n in group
        }
        compressed = compress_covariates(values, frozenset(n for n in group if n in factors), rng=rng)
        for n in group:
            mf0[n] = compressed.values[n]
    maxr = max(v.shape[0] for v in mf0.values())
    for name, values in mf0.items():
        me = values.shape[0]
        if me < maxr:
            if me == 1 and name not in factors:
                raise HgamNotImplementedError(
                    f"discrete_model_frame: {name!r} has one distinct value; R's sample() then "
                    "draws from 1:x, which is not ported."
                )
            pads = values[rng.sample_int(me, maxr - me, replace=True) - 1]
            mf0[name] = np.concatenate([values, pads])
    return mf0


def discretise_frame(model: ModelSpec, data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """mgcv counterpart: ``discrete.mf`` (bam.r), mgcv 1.9-4, the index side.

    Each covariate group a smooth reaches (:func:`_smooth_groups`; ``check.term`` reuses an earlier
    one) is compressed once by :func:`compress_covariates` at ``m = NULL``: one covariate on its own
    (``m = 1000``), a ``t2`` term's covariates jointly (``m = 100`` per numeric column when the grid rule
    fires); factors are never binned. The returned frame has every numeric column replaced by its
    discretised value per data row, so a design built on it equals mgcv's ``Xd[[j]][k[, j], ]`` row
    for row. mgcv builds its bases on the padded frame of unique values instead; basis placement
    (``place.knots`` takes unique values) is the same, and the sum-to-zero constraints are taken over
    that frame (:func:`discrete_model_frame`).
    """
    factors = _factor_names(model)
    out = dict(data)
    for group in _new_groups(model):
        if all(n in factors for n in group):
            continue
        compressed = compress_covariates(
            {n: np.asarray(data[n], dtype=np.float64) for n in group},
            frozenset(n for n in group if n in factors),
        )
        for n in group:
            if n not in factors:
                out[n] = compressed.row_values(n)
    return out
