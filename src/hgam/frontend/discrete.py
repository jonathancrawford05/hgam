"""``bam(discrete=TRUE)``'s covariate compression: mgcv's ``compress.df``.

Moved from polaris-re's ``gam/bam_binning_conformance.py``; the comparison harness and its
claim objects stayed behind. Ported from the mgcv 1.9-1 source; polaris-re checked the pinned
1.9-4 by behaviour only. Nothing calls this yet: ``discrete.mf`` and the padded frame that
``bam`` builds its smooths on arrive in M3.

What the source says: for one covariate ``m = 1000``; two jointly ``m = 100``; more, ``25``.
If the covariate(s) have no more than ``m ** n_numeric * prod(n_factor_levels)`` unique rows,
nothing is binned and the cells are the unique rows. Only above that are numeric columns snapped
to ``m`` evenly spaced points over their range.
"""

from dataclasses import dataclass

import numpy as np

from hgam.errors import HgamValidationError

__all__ = ["CompressedTerm", "compress_covariates", "default_m"]

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
