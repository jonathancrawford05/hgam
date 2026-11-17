# ADR 0002: discrete bam's constraints are taken over mgcv's padded frame, through R's own RNG

Status: accepted, 2026-11-17. Reopens polaris-re ADR-260 amendment 9 (cited in the moved code), which took
the constraints over one row per distinct covariate value and did not reproduce the padding.

## Context

`bam(discrete = TRUE)` calls `gam.setup` on `discrete.mf`'s frame `mf0` (bam.r), and `smoothCon`
(smooth.r, `sparse.cons = 0`) takes each sum-to-zero constraint as `colMeans(X)` over its rows. `mf0` holds each
compressed covariate's unique values, shuffled, then padded to the longest column with
`sample(mf0[[i]], maxr - me, replace = TRUE)`, all under `temp.seed(8547)`. The padding rows count in the
constraint. On the exit fit this moved the intercept of `summary()` by 3.6e-3 (gate 1e-3). Jonathan chose to
port the rows (PR #21, Q1 option A).

Measured in R on section 14: mgcv's stored constraint equals `colMeans` over the 985-row `mf0` to 0, and
misses one row per distinct value by 3.0e-3. The padding values depend on every `sample()` call `discrete.mf`
makes before them, so the stream has to be reproduced draw for draw.

## Decision

- `hgam.frontend.r_rng.RRandom` ports R's `set.seed` (Mersenne-Twister) and `sample.int` with Rejection
  sampling, held exactly to R's draws (fixture `r_rng_set_seed_8547`). It is R's code, not mgcv's. NumPy only.
- `hgam.frontend.discrete.discrete_model_frame` rebuilds `mf0` in `discrete.mf`'s order: every smooth's `by` and
  margins in term order, then the parametric frame (response, factors, weights, offset), then the padding.
  It is held row for row to mgcv's `mf0` (fixture `bam_discrete_constraint_rows`).
- `fit_engine_bam` takes every cr and ti-margin constraint over that frame (`constraint_frame`).
  `unique_row_constraint` is removed.

## Consequences

- On the exit fit the intercept is 4.8e-7 from mgcv and eta 7.4e-8 (was 1.1e-4). The constraint is held at the
  deterministic-pieces gate (1e-8 relative).
- An offset beside prior weights is refused in a discrete fit. Its place in `discrete.mf`'s parametric frame
  depends on whether it was written in the formula or passed as `offset=`, and `ModelSpec` does not carry that.
- Covariates reaching the frame by a path not listed above (a numeric parametric term, a matrix argument)
  are not ported. Any change to `discrete.mf`'s call order in a later mgcv changes the pads; CI's tier-3
  fixture shows it.
