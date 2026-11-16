# Finding 0002: exit-model run time against mgcv (M3)

Date: 2026-11-16. The M3 benchmark from `docs/PRIORITIES.md`: the exit-test model (`hgam_formula`, section 3's
frame, n = 4000, `my_knots`) fitted by `hgam.bam(..., discrete=True, select=True)` and by `mgcv::bam(discrete = TRUE,
select = TRUE)`, on one machine (4 vCPUs), median of 15 fits after one warm-up. Scripts: `exit_benchmark/`.
mgcv 1.9-1 (R 4.3.3, Debian reference BLAS, single-threaded); hgam with NumPy 2.5.4 and SciPy's OpenBLAS 0.3.34.
Measurements, tier 1; wall-clock times vary by about 10% between runs.

| Fitter | Median (s) | Min (s) | Max (s) | vs mgcv |
|---|---|---|---|---|
| mgcv `bam` | 0.97 to 0.98 | 0.92 | 1.12 | 1 |
| hgam, OpenBLAS default threads (4) | 2.19 to 2.40 | 0.98 | 4.01 | about 2.3 times slower |
| hgam, `OPENBLAS_NUM_THREADS=1` | 0.42 | 0.41 | 0.52 | about 2.3 times faster |

hgam is inside the ten-times rule, so nothing is tuned now. The default-threads row is slower and erratic because
OpenBLAS spreads small matrix products (a few hundred columns) over four threads; on one thread hgam is faster than
mgcv. mgcv here runs on reference BLAS, so the single-thread comparison is not like for like on BLAS either.
