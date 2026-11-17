# Finding 0002: exit-model run time against mgcv (M3)

Date: 2026-11-16. The M3 benchmark from `docs/PRIORITIES.md`: the exit-test model (`hgam_formula`, section 3's
frame, n = 4000, `my_knots`) fitted by `hgam.bam(..., discrete=True, select=True)` and by `mgcv::bam(discrete = TRUE,
select = TRUE)`, on one machine (4 vCPUs), median of 15 fits after one warm-up. Scripts: `exit_benchmark/`.
mgcv 1.9-1 (R 4.3.3, Debian reference BLAS, single-threaded); hgam with NumPy 2.5.4 and SciPy's OpenBLAS 0.3.34.
Measurements, tier 1. mgcv and the one-thread row vary by about 10% between runs; the default-threads row varies far
more (a review rerun of 15 fits gave medians of 2.69 s and 3.28 s, 2.7 to 3.4 times slower than mgcv).

| Fitter | Median (s) | Min (s) | Max (s) | vs mgcv |
|---|---|---|---|---|
| mgcv `bam` | 0.97 to 0.98 | 0.92 | 1.12 | 1 |
| hgam, OpenBLAS default threads (4) | 2.19 to 2.40 | 0.98 | 4.01 | about 2.3 times slower |
| hgam, `OPENBLAS_NUM_THREADS=1` | 0.42 | 0.41 | 0.52 | about 2.3 times faster |

hgam is inside the ten-times rule, so nothing is tuned now. The default-threads row is slower and erratic because
OpenBLAS spreads small matrix products (a few hundred columns) over four threads; on one thread hgam is faster than
mgcv. mgcv here runs on reference BLAS, so the single-thread comparison is not like for like on BLAS either.

## Where the threaded time goes (2026-11-17)

`cProfile` over three fits on this machine (median 2.2 s on four threads, 0.69 s on one). In the fit,
`_weighted_cross_products` (X'WX, 4000 x 132) takes 23 ms a call on four threads and 5 ms on one. `np.linalg.eigh` in
`_fit_chol` (penalty blocks of at most 24 columns) takes 2.0 ms and 0.10 ms. The four-thread figures vary about twofold
between runs (the PR #31 review's rerun: X'WX 11 ms against 3.8 ms, `eigh` 0.93 ms against 0.07 ms); across both runs
X'WX is 11 to 23 ms a call on four threads against 4 to 5 ms on one, and `eigh` 0.9 to 2.0 ms against 0.07 to 0.10 ms. The same calls timed alone in a loop run
as fast or faster on four threads (X'WX 2.4 ms against 3.6 ms), so the cost is OpenBLAS waking its threads between
small calls inside the fit, not the products themselves. A thread cap scoped to `bam_fitd` would remove it. NumPy and
SciPy share one OpenBLAS here (`scipy_openblas64`).
