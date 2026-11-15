# Deferred

One line each: what differs or is missing, where it was seen, and why it did not block. Revisit after the exit test.

- Data-frame input: `hgam.frontend.gam` takes a mapping of NumPy columns and a `Factor` type where polaris-re took a polars DataFrame (polars is not a core dependency). pandas/polars input waits for an optional-extra decision (M4 or packaging).
- Moved engine tests left in polaris-re (43 test functions): ones that call the conformance modules, the old `experience_gam_penalized` engine, R probes or `VerificationClaim` objects; the statsmodels binomial cross-check (statsmodels is not a dependency); and three regression tests that read polaris-re's synthetic-data JSON files, which `oracle/make_fixtures.R` did not produce.
- Top-level `hgam.gam` / `hgam.bam` remain stubs; the moved front end is `hgam.frontend.gam`. Wiring waits for the port to reach the exit test.
- The six ported functions name mgcv 1.9-1, the version they were read from; the pinned oracle's 1.9-4 was checked by behaviour only in polaris-re. Their own hgam fixtures come with the M1/M3 slices that re-port them.
- Docstrings in the moved code cite polaris-re ADR numbers (ADR-189 to ADR-265) and `docs/PLAN_*` files that live in polaris-re, not here.
- `compress_covariates` returns unique rows in lexicographic order; mgcv's `compress.df` shuffles them with `sample()`. Same cell partition, different labels. Waits for R's random number stream in M3 (PR #3 review).
- `initial_log10_lambda_from_xx_diag` trims each penalty block to the span from its first to its last non-zero column, while mgcv's `initial.sp` uses the whole stored block. The row and column means behind the `eps^0.8` threshold can therefore differ when a block has zero edge columns. This affects starting values only (PR #3 review).
