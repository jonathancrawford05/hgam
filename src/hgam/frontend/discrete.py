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

__all__ = ["CompressedTerm", "compress_covariates", "default_m", "discretise_frame"]

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
) -> CompressedTerm:
    """mgcv counterpart: ``compress.df`` (bam.r), read from mgcv 1.9-1.

    Port of ``mgcv:::compress.df`` for one term's covariate columns.

    One departure: mgcv shuffles the unique rows (``ii <- sample(1:nrow(xu))``) and remaps the
    index; this port returns them in ``np.unique`` lexicographic order, so the cell partition
    matches mgcv's but the cell labels do not. Reproducing R's random number stream is M3 work.

    Args:
        columns: name to 1-d float array (numeric) or integer codes (factor), equal length.
        factors: names in ``columns`` that are factors (never binned; they multiply the
            threshold by their number of levels).
        m: ``bam``'s ``m`` argument (``None`` = the defaults).

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
    return CompressedTerm(
        names=names,
        values={k: uniq[:, i] for i, k in enumerate(names)},
        index=inverse,
        binned=binned,
    )


def _numeric_margins(term: TermSpec) -> tuple[str, ...]:
    """The numeric covariates ``discrete.mf`` compresses for one term, in its order: a numeric
    ``by`` first, then each margin. Every supported margin is one covariate: ``ti`` margins are
    marginal smooths, and ``bam(discrete=TRUE)`` gives ``sz`` (``tensor.possible``) one margin per
    variable (bam.r), of which the first is the factor and is not binned."""
    if term.basis in ("parametric", "re"):
        return ()
    if term.basis in ("cr", "ti"):
        margins: tuple[str, ...] = term.variables
    elif term.basis == "sz":
        margins = term.variables[1:]
    else:
        raise HgamNotImplementedError(
            f"discretise_frame: {term.basis!r} terms are not discretised (only cr, ti, sz, re and factors)."
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
    the sum-to-zero constraint is taken over the data rows here rather than over mgcv's padded
    frame (``docs/DEFERRED.md``).
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
