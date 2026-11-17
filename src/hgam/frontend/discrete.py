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


def _unique_rows(
    columns: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows (``np.unique`` lexicographic order) and each row's 0-based unique index."""
    stacked = np.column_stack(columns).astype(np.float64)
    uniq, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return uniq, inverse.reshape(-1).astype(np.int64)


def compress_covariates(
    columns: dict[str, np.ndarray],
    factors: frozenset[str] = frozenset(),
    *,
    m: int | None = None,
    rng: RRandom | None = None,
) -> CompressedTerm:
    """mgcv counterpart: ``compress.df`` (bam.r), read from mgcv 1.9-1.

    Port of ``mgcv:::compress.df`` for one term's covariate columns.

    ``uniquecombs(dat, TRUE)`` returns the unique rows sorted, as ``np.unique`` does. mgcv then
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

    uniq, _ = _unique_rows([arrays[k] for k in names])
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
        uniq, _ = _unique_rows([arrays[k] for k in names])

    if uniq.shape[0] == n:
        # R returns the (possibly binned) data itself with index 1:n.
        return CompressedTerm(
            names=names,
            values={k: arrays[k] for k in names},
            index=np.arange(n, dtype=np.int64),
            binned=binned,
        )
    uniq, inverse = _unique_rows([arrays[k] for k in names])
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


def _smooth_margins(term: TermSpec) -> tuple[str, ...]:
    """Every covariate ``discrete.mf`` compresses for one smooth, in its order: the ``by``
    variable (numeric or factor) first, then each one-covariate margin. ``sz`` is a tensor of
    one-covariate margins there (``tensor.possible``, bam.r), factor first; ``re`` is its factor. ``fs`` is
    ``tensor.possible`` too, with one margin per written term (``TermSpec.margin_order``)."""
    if term.basis == "parametric":
        return ()
    if term.basis not in ("cr", "ti", "sz", "fs", "re"):
        raise HgamNotImplementedError(
            f"discrete_model_frame: {term.basis!r} terms are not discretised "
            "(only cr, ti, sz, fs, re and factors)."
        )
    if term.basis == "fs":
        return term.margin_order if term.margin_order is not None else term.variables
    by = term.by if term.by is not None else term.by_factor
    return ((by,) if by is not None else ()) + term.variables


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
    columns: dict[str, np.ndarray] = {}
    for term in model.terms:
        for name in _smooth_margins(term):
            if name not in columns:
                columns[name] = np.asarray(data[name], dtype=np.float64)
    smooth_names = list(columns)
    parametric = sorted((t for t in model.terms if t.basis == "parametric"), key=lambda t: len(t.variables))
    pmf: dict[str, np.ndarray] = {"(response)": np.asarray(response, dtype=np.float64)}
    for term in parametric:
        for name in term.variables:
            pmf.setdefault(name, np.asarray(data[name], dtype=np.float64))
    for extra in (model.weights_column, model.offset_column):
        if extra is not None:
            pmf.setdefault(extra, np.asarray(data[extra], dtype=np.float64))
    rng = RRandom(_DISCRETE_SEED)
    mf0: dict[str, np.ndarray] = {}
    for name in smooth_names + [n for n in pmf if n not in columns]:
        values = columns[name] if name in columns else pmf[name]
        flag = frozenset({name}) if name in factors else frozenset()
        mf0[name] = compress_covariates({name: values}, flag, rng=rng).values[name]
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


def _numeric_margins(term: TermSpec) -> tuple[str, ...]:
    """The numeric covariates ``discrete.mf`` compresses for one term, in its order: a numeric
    ``by`` first, then each margin. Every supported margin is one covariate: ``ti`` margins are
    marginal smooths, and ``bam(discrete=TRUE)`` gives ``sz`` (``tensor.possible``) one margin per
    variable (bam.r), of which the first is the factor and is not binned."""
    if term.basis in ("parametric", "re"):
        return ()
    if term.basis in ("cr", "ti"):
        margins: tuple[str, ...] = term.variables
    elif term.basis in ("sz", "fs"):
        margins = term.variables[1:]
    else:
        raise HgamNotImplementedError(
            f"discretise_frame: {term.basis!r} terms are not discretised "
            "(only cr, ti, sz, fs, re and factors)."
        )
    return ((term.by,) if term.by is not None else ()) + margins


def discretise_frame(model: ModelSpec, data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """mgcv counterpart: ``discrete.mf`` (bam.r), mgcv 1.9-4, for one-covariate margins only.

    Each numeric covariate a smooth reaches is compressed once, on its own, by
    :func:`compress_covariates` at ``m = NULL`` (``check.term`` reuses an earlier
    discretisation of the same covariate); factors are never binned. The returned frame has
    every such column replaced by its discretised value per data row, so a design built on it
    equals mgcv's ``Xd[[j]][k[, j], ]`` row for row. mgcv builds its bases on the padded frame
    of unique values instead; basis placement (``place.knots`` takes unique values) is the same,
    and the sum-to-zero constraints are taken over that frame (:func:`discrete_model_frame`).
    """
    out = dict(data)
    done: set[str] = set()
    for term in model.terms:
        for name in _numeric_margins(term):
            if name not in done:
                out[name] = compress_covariates({name: np.asarray(data[name], dtype=np.float64)}).row_values(
                    name
                )
                done.add(name)
    return out
