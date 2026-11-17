"""``bam``'s ``nthreads`` argument, mapped onto the BLAS thread pool.

mgcv counterpart: the ``nthreads`` argument of ``bam`` (bam.r), mgcv 1.9-4. This is a mapping onto
mgcv's argument name, not a port of its threading. In mgcv, ``nthreads`` (default 1) sets the number
of OpenMP threads of mgcv's own C code (``XWXd``, ``pchol``, ``Sl.initial.repara``'s ``nt`` and so
on); a scalar is expanded to ``rep(nthreads, 2)``, and a non-finite value or one below 1 becomes
``max(1, length(cluster))``, which is 1 without a cluster. hgam has no C of its own: its products run
in the BLAS that NumPy and SciPy load, so ``nthreads`` caps that BLAS's thread pool for the length of
the fit (``threadpoolctl``'s ``threadpool_limits(nthreads, user_api="blas")``).

The cap is process-wide while it holds: other threads of the process that call BLAS during the fit
see it too. It is restored when the fit returns or raises. Without the optional extra
``hgam[threads]`` (``threadpoolctl``), ``nthreads=1`` leaves BLAS as the process set it (see the README
on ``OPENBLAS_NUM_THREADS=1``), and any other value is refused, so the argument is never accepted and
ignored.
"""

import contextlib
import importlib
import math
import numbers
from collections.abc import Iterator

from hgam.errors import HgamNotImplementedError, HgamValidationError

__all__ = ["blas_thread_limit", "resolve_nthreads"]


def resolve_nthreads(nthreads: object) -> int:
    """``bam``'s ``nthreads[1]`` for a scalar argument: an integer of at least 1 is kept; a non-finite
    value or one below 1 becomes 1, as ``bam`` sets it to ``max(1, length(cluster))`` with no cluster.

    The two-element form ``c(n1, n2)`` (``n2`` for ``pchol`` and ``bam``'s ``nt`` vectors) is refused:
    hgam has one thread pool, so the second element would be accepted and ignored."""
    if isinstance(nthreads, (bool, str, bytes)):
        raise HgamValidationError(f"bam(): nthreads must be a whole number, got {nthreads!r}.")
    if hasattr(nthreads, "__len__"):
        raise HgamNotImplementedError(
            f"bam(): nthreads={nthreads!r}: the two-element form c(n1, n2) of mgcv's nthreads is not "
            "ported; hgam caps one BLAS thread pool, so pass a single number."
        )
    if isinstance(nthreads, numbers.Real) and not isinstance(nthreads, numbers.Integral):
        value = float(nthreads)
        if not math.isfinite(value):
            return 1
        if value.is_integer():
            nthreads = int(value)
    if not isinstance(nthreads, numbers.Integral):
        raise HgamValidationError(f"bam(): nthreads must be a whole number, got {nthreads!r}.")
    return max(1, int(nthreads))


@contextlib.contextmanager
def blas_thread_limit(nthreads: int) -> Iterator[None]:
    """Cap the BLAS thread pool at ``nthreads`` while the block runs, and restore it afterwards,
    also when the block raises. Without ``threadpoolctl``, ``nthreads == 1`` runs the block with BLAS
    as it is, and ``nthreads > 1`` is refused, naming the extra."""
    try:
        threadpoolctl = importlib.import_module("threadpoolctl")
    except ImportError:
        if nthreads == 1:
            yield
            return
        raise HgamNotImplementedError(
            f"bam(): nthreads={nthreads} needs the optional extra hgam[threads] (threadpoolctl), which is "
            "not installed; without it only nthreads=1 is accepted, and BLAS keeps its own thread count."
        ) from None
    with threadpoolctl.threadpool_limits(limits=nthreads, user_api="blas"):
        yield
