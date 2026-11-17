"""Davies' algorithm for the distribution of a linear combination of chi-squared variables.

mgcv counterpart: ``davies.c`` (mgcv's C, Simon Wood's translation of Davies 1980, *J. R. Statist. Soc. C*
29, 323-333, Algol 60), mgcv 1.9-4 (the file is unchanged since 1.8-32). ``psum.chisq`` (mgcv.r) calls it
through ``.C(C_davies, ...)`` for ``summary.gam``'s smooth-term p-values; the R side is ported in
``hgam.frontend.smooth_tests``.

The port keeps the C's function boundaries (``errbd``, ``ctff``, ``truncation``, ``findu``, ``integrate``,
``cfe``, ``davies``), its constants, its iteration order and its call counter (``trace[6]``). Two helpers
come from R itself: ``revsort`` (R's ``src/main/sort.c``, a heapsort into decreasing order, ported so ties
are ordered as R orders them) and ``log1pmx`` (R's ``src/nmath/pgamma.c``, with ``logcf``). The one change
of form: ``integrate``'s inner loop over the ``r`` weights runs over every ``k`` at once with NumPy (the
same per-``k`` operations in the same order), and the sums into ``intl`` and ``ersm`` are then taken in
the C's order, ``k = nterm`` down to ``0``.
"""

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["DaviesResult", "davies", "log1pmx", "revsort"]

_SCALEFACTOR = 4294967296.0**8  # SQR(SQR(SQR(2^32))) = 2^256


def _logcf(x: float, i: float, d: float, eps: float) -> float:
    """R counterpart: ``logcf`` (src/nmath/pgamma.c): the continued fraction for
    ``sum_k x^k / (i + k d)``."""
    c1 = 2.0 * d
    c2 = i + d
    c4 = c2 + d
    a1 = c2
    b1 = i * (c2 - i * x)
    b2 = d * d * x
    a2 = c4 * c2 - b2
    b2 = c4 * b1 - i * b2
    while abs(a2 * b1 - a1 * b2) > abs(eps * b1 * b2):
        c3 = c2 * c2 * x
        c2 += d
        c4 += d
        a1 = c4 * a2 - c3 * a1
        b1 = c4 * b2 - c3 * b1
        c3 = c1 * c1 * x
        c1 += d
        c4 += d
        a2 = c4 * a1 - c3 * a2
        b2 = c4 * b1 - c3 * b2
        if abs(b2) > _SCALEFACTOR:
            a1 /= _SCALEFACTOR
            b1 /= _SCALEFACTOR
            a2 /= _SCALEFACTOR
            b2 /= _SCALEFACTOR
        elif abs(b2) < 1.0 / _SCALEFACTOR:
            a1 *= _SCALEFACTOR
            b1 *= _SCALEFACTOR
            a2 *= _SCALEFACTOR
            b2 *= _SCALEFACTOR
    return a2 / b2


def log1pmx(x: float) -> float:
    """R counterpart: ``log1pmx`` (src/nmath/pgamma.c): ``log(1 + x) - x``, accurate for small ``x``."""
    if x > 1.0 or x < -0.79149064:
        return math.log1p(x) - x
    r = x / (2.0 + x)
    y = r * r
    if abs(x) < 1e-2:
        two = 2.0
        return r * ((((two / 9 * y + two / 7) * y + two / 5) * y + two / 3) * y - x)
    return r * (2.0 * y * _logcf(y, 3.0, 2.0, 1e-14) - x)


def revsort(a: list[float], ib: list[int]) -> None:
    """R counterpart: ``revsort`` (src/main/sort.c): sort ``a`` into decreasing order by heapsort, in
    place, permuting ``ib`` alongside. Ported so that ties come out in R's order."""
    n = len(a)
    if n <= 1:
        return
    # one-based views, as the C's a--, ib--
    a.insert(0, 0.0)
    ib.insert(0, 0)
    l_ = (n >> 1) + 1
    ir = n
    try:
        while True:
            if l_ > 1:
                l_ -= 1
                ra = a[l_]
                ii = ib[l_]
            else:
                ra = a[ir]
                ii = ib[ir]
                a[ir] = a[1]
                ib[ir] = ib[1]
                ir -= 1
                if ir == 1:
                    a[1] = ra
                    ib[1] = ii
                    return
            i = l_
            j = l_ << 1
            while j <= ir:
                if j < ir and a[j] > a[j + 1]:
                    j += 1
                if ra > a[j]:
                    a[i] = a[j]
                    ib[i] = ib[j]
                    i = j
                    j += j
                else:
                    j = ir + 1
            a[i] = ra
            ib[i] = ii
    finally:
        del a[0]
        del ib[0]


def _ln1(x: float, first: bool) -> float:
    """mgcv counterpart: ``ln1`` (davies.c): ``log1p(x)`` if ``first``, else ``log1pmx(x)``."""
    return math.log1p(x) if first else log1pmx(x)


@dataclass
class _Q:
    """The quadratic form's parameters and the C's static call counter."""

    lb: list[float]
    nc: list[float]
    n: list[int]
    r: int
    count: int = 0
    th: list[int] = field(default_factory=list)


def _errbd(q: _Q, u: float, sigsq: float) -> tuple[float, float]:
    """mgcv counterpart: ``errbd`` (davies.c): a bound on the tail probability; returns ``(bound, cx)``."""
    q.count += 1
    cx = u * sigsq
    sum1 = u * cx
    u = u * 2
    for j in range(q.r - 1, -1, -1):
        nj = q.n[j]
        lj = q.lb[j]
        ncj = q.nc[j]
        x = u * lj
        y = 1.0 - x
        cx += lj * (ncj / y + nj) / y
        xy = x / y
        sum1 = sum1 + ncj * xy * xy + nj * (x * xy + _ln1(-x, False))
    return math.exp(-0.5 * sum1), cx


def _ctff(
    q: _Q, accx: float, upn: float, mean: float, lmin: float, lmax: float, sigsq: float
) -> tuple[float, float]:
    """mgcv counterpart: ``ctff`` (davies.c): a cut-off with ``Pr(Q > ctff) < accx`` if ``upn > 0``,
    ``Pr(Q < ctff) < accx`` otherwise. Returns ``(ctff, upn)``, ``upn`` as the C updates it."""
    u2 = upn
    u1 = 0.0
    c1 = mean
    rb = 2 * lmax if u2 > 0 else 2 * lmin
    while True:
        bound, c2 = _errbd(q, u2 / (1 + u2 * rb), sigsq)
        if bound <= accx:
            break
        u1 = u2
        c1 = c2
        u2 = u2 * 2
    while True:
        if (c1 - mean) / (c2 - mean) >= 0.9:
            break
        u = (u1 + u2) * 0.5
        bound, cst = _errbd(q, u / (1 + u * rb), sigsq)
        if bound > accx:
            u1 = u
            c1 = cst
        else:
            u2 = u
            c2 = cst
    return c2, u2


def _truncation(q: _Q, u: float, tausq: float, sigsq: float) -> float:
    """mgcv counterpart: ``truncation`` (davies.c): a bound on the integration error from truncation."""
    q.count += 1
    pi = math.asin(1.0) * 2
    sum1 = prod2 = prod3 = 0.0
    s = 0
    sum2 = (sigsq + tausq) * u * u
    prod1 = 2 * sum2
    u = 2 * u
    for j in range(q.r):
        lj = q.lb[j]
        ncj = q.nc[j]
        nj = q.n[j]
        x = u * lj
        x = x * x
        sum1 += ncj * x / (1 + x)
        if x > 1:
            prod2 += nj * math.log(x)
            prod3 += nj * _ln1(x, True)
            s = s + nj
        else:
            prod1 += nj * _ln1(x, True)
    sum1 = sum1 * 0.5
    prod2 += prod1
    prod3 += prod1
    x = math.exp(-sum1 - 0.25 * prod2) / pi
    y = math.exp(-sum1 - 0.25 * prod3) / pi
    err1 = 1.0 if s == 0 else 2.0 * x / s
    err2 = 2.5 * y if prod3 > 1 else 1.0
    if err2 < err1:
        err1 = err2
    x = 0.5 * sum2
    err2 = 1.0 if x <= y else y / x
    return err1 if err1 < err2 else err2


def _findu(q: _Q, utx: float, accx: float, sigsq: float) -> float:
    """mgcv counterpart: ``findu`` (davies.c): ``u`` with ``truncation(u) < accx`` and
    ``truncation(u / 1.2) > accx``."""
    a = (2.0, 1.4, 1.2, 1.1)
    ut = utx
    u = ut * 0.25
    if _truncation(q, u, 0.0, sigsq) > accx:
        while _truncation(q, ut, 0.0, sigsq) > accx:
            ut = ut * 4
    else:
        ut = u
        u = u / 4
        while _truncation(q, u, 0.0, sigsq) <= accx:
            ut = u
            u = u / 4
    for ai in a:
        u = ut / ai
        if _truncation(q, u, 0.0, sigsq) <= accx:
            ut = u
    return ut


def _integrate(
    q: _Q,
    nterm: int,
    interv: float,
    tausq: float,
    main: bool,
    c: float,
    intl: float,
    ersm: float,
    sigsq: float,
) -> tuple[float, float]:
    """mgcv counterpart: ``integrate`` (davies.c): ``nterm + 1`` terms at step ``interv``; unless ``main``
    the integrand is multiplied by ``1 - exp(-0.5 tausq u^2)``. Returns the updated ``(intl, ersm)``.
    The ``j`` loop runs over every ``k`` at once (NumPy); the sums run ``k = nterm`` down to ``0``."""
    pi = math.asin(1.0) * 2
    inpi = interv / pi
    k = np.arange(nterm, -1, -1, dtype=np.float64)
    u = (k + 0.5) * interv
    sum1 = -2 * u * c
    sum2 = np.abs(sum1)
    sum3 = -0.5 * sigsq * u * u
    for j in range(q.r - 1, -1, -1):
        nj = q.n[j]
        x = 2 * q.lb[j] * u
        y = x * x
        sum3 = sum3 - 0.25 * nj * np.log1p(y)
        y = q.nc[j] * x / (1 + y)
        z = nj * np.arctan(x) + y
        sum1 = sum1 + z
        sum2 = sum2 + np.abs(z)
        sum3 = sum3 + -0.5 * x * y
    x = inpi * np.exp(sum3) / u
    if not main:
        x = x * (1 - np.exp(-0.5 * tausq * u * u))
    t1 = np.sin(0.5 * sum1) * x
    t2 = 0.5 * sum2 * x
    for v1, v2 in zip(t1.tolist(), t2.tolist(), strict=True):
        intl += v1
        ersm += v2
    return intl, ersm


def _cfe(q: _Q, x: float, ln28: float) -> tuple[float, bool]:
    """mgcv counterpart: ``cfe`` (davies.c): the coefficient of ``tausq`` in the error when the
    convergence factor ``exp(-0.5 tausq u^2)`` is used at ``x``. Returns ``(value, fail)``."""
    q.count += 1
    pi = math.asin(1.0) * 2
    axl = abs(x)
    sxl = -1 if x < 0 else 1
    sum1 = 0.0
    th, lb, n, nc = q.th, q.lb, q.n, q.nc
    for j in range(q.r - 1, -1, -1):
        t = th[j]
        if lb[t] * sxl > 0.0:
            lj = abs(lb[t])
            axl1 = axl - lj * (n[t] + nc[t])
            axl2 = lj / ln28
            if axl1 > axl2:
                axl = axl1
            else:
                if axl > axl2:
                    axl = axl2
                sum1 = (axl - axl1) / lj
                for kk in range(j - 1, -1, -1):
                    sum1 += n[th[kk]] + nc[th[kk]]
                break
    if sum1 > 100.0:
        return 1.0, True
    return 2.0 ** (sum1 * 0.25) / (pi * axl * axl), False


@dataclass(frozen=True)
class DaviesResult:
    """What ``.C(C_davies, ...)`` hands back: ``c`` (``Pr(Q < c)``, or ``-1`` when ``lim`` is reached),
    ``trace`` (the seven-vector documented in davies.c) and ``ifault`` (0 fine; 1 accuracy not obtained;
    2 round-off possibly significant; 3 invalid parameters; 4 integration parameters not located)."""

    c: float
    trace: tuple[float, ...]
    ifault: int


def davies(
    lb: np.ndarray, nc: np.ndarray, n: np.ndarray, sigma: float, c: float, lim: int, acc: float
) -> DaviesResult:
    """mgcv counterpart: ``davies`` (davies.c), mgcv 1.9-4.

    ``Pr(Q < c)`` for ``Q = sum_j lb[j] X_j + sigma X_0``, ``X_j`` chi-squared on ``n[j]`` degrees of
    freedom with non-centrality ``nc[j]``, ``X_0`` standard normal. ``lim`` bounds the number of
    integration terms and ``acc`` is the error bound. As in the C, the early exits return ``c = 1`` or
    ``c = 0`` with ``ifault = 0`` when ``c`` is outside the distribution's range, and ``c = -1`` when
    ``lim`` is reached (``ifault = 1`` only in the main integration's check, as the C sets it)."""
    lbl = [float(v) for v in np.asarray(lb, dtype=np.float64)]
    ncl = [float(v) for v in np.asarray(nc, dtype=np.float64)]
    nl = [int(v) for v in np.asarray(n)]
    r = len(lbl)
    q = _Q(lb=lbl, nc=ncl, n=nl, r=r)
    trace = [0.0] * 7
    ln28 = math.log(2.0) / 8
    pi = math.asin(1.0) * 2
    ifault = 0
    intl = ersm = 0.0
    acc1 = acc
    alb = [abs(v) for v in lbl]
    th = list(range(r))
    revsort(alb, th)
    q.th = th

    def done(cv: float, fault: int = 0, count: bool = True) -> DaviesResult:
        if count:
            trace[6] = float(q.count)
        return DaviesResult(c=cv, trace=tuple(trace), ifault=fault)

    sd = sigsq = sigma * sigma
    lmax = lmin = 0.0
    mean = 0.0
    for j in range(r):
        nj = nl[j]
        lj = lbl[j]
        ncj = ncl[j]
        if nj < 0 or ncj < 0:
            return done(c, 3, count=False)
        sd += lj * lj * (2.0 * nj + 4.0 * ncj)
        mean += lj * (nj + ncj)
        if lmax < lj:
            lmax = lj
        elif lmin > lj:
            lmin = lj
    if sd == 0.0:
        return done(1.0 if c > 0.0 else 0.0, 0, count=False)
    if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
        return done(c, 3, count=False)
    sd = math.sqrt(sd)
    almx = -lmin if lmax < -lmin else lmax

    utx = 16.0 / sd
    up = 4.5 / sd
    un = -up
    utx = _findu(q, utx, 0.5 * acc1, sigsq)

    if c != 0.0 and almx > 0.07 * sd:
        cf, fail = _cfe(q, c, ln28)
        tausq = 0.25 * acc1 / cf
        if not fail and _truncation(q, utx, tausq, sigsq) < 0.2 * acc1:
            sigsq = sigsq + tausq
            utx = _findu(q, utx, 0.25 * acc1, sigsq)
            trace[5] = math.sqrt(tausq)
    trace[4] = utx
    acc1 = 0.5 * acc1

    while True:
        cut, up = _ctff(q, acc1, up, mean, lmin, lmax, sigsq)
        d1 = cut - c
        if d1 < 0.0:
            return done(1.0)
        cut, un = _ctff(q, acc1, un, mean, lmin, lmax, sigsq)
        d2 = c - cut
        if d2 < 0.0:
            return done(0.0)
        intv = 2 * pi / d1 if d1 > d2 else 2 * pi / d2
        x = utx / intv
        nt = math.floor(x)
        if x - nt > 0.5:
            nt += 1
        x = 3.0 / math.sqrt(acc1)
        ntm = math.floor(x)
        if x - ntm > 0.5:
            ntm += 1
        if nt > ntm * 1.5:
            intv1 = utx / ntm
            x = 2 * pi / intv1
            if x <= abs(c):
                break
            cf_lo, _ = _cfe(q, c - x, ln28)
            cf_hi, fail = _cfe(q, c + x, ln28)  # C leaves the order unspecified; GCC's left to right is taken
            tausq = 0.33 * acc1 / (1.1 * (cf_lo + cf_hi))
            if fail:
                break
            acc1 = acc1 * 0.67
            if ntm > lim:
                return done(-1.0)
            intl, ersm = _integrate(q, ntm, intv1, tausq, False, c, intl, ersm, sigsq)
            lim = lim - ntm
            sigsq = sigsq + tausq
            trace[2] += 1.0
            trace[1] += ntm + 1
            utx = _findu(q, utx, 0.25 * acc1, sigsq)
            acc1 = 0.75 * acc1
        else:
            break

    trace[3] = intv
    if nt > lim:
        return done(-1.0, 1)
    intl, ersm = _integrate(q, nt, intv, 0.0, True, c, intl, ersm, sigsq)
    trace[2] += 1
    trace[1] += nt + 1
    cv = 0.5 - intl
    trace[0] = ersm
    x = ersm + acc / 10.0
    jj = 1
    for _ in range(4):
        if jj * x == jj * ersm:
            ifault = 2
        jj = jj * 2
    return done(cv, ifault)
