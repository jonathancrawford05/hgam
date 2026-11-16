# Deferred

One line each: what differs or is missing, where it was seen, and why it did not block. Revisit after the exit test.

- Data-frame input: `hgam.frontend.gam` takes a mapping of NumPy columns and a `Factor` type where polaris-re took a polars DataFrame (polars is not a core dependency). pandas/polars input waits for an optional-extra decision (M4 or packaging).
- Moved engine tests left in polaris-re (43 test functions): ones that call the conformance modules, the old `experience_gam_penalized` engine, R probes or `VerificationClaim` objects; the statsmodels binomial cross-check (statsmodels is not a dependency); and three regression tests that read polaris-re's synthetic-data JSON files, which `oracle/make_fixtures.R` did not produce.
- Top-level `hgam.gam` / `hgam.bam` remain stubs; the moved front end is `hgam.frontend.gam`. Wiring waits for the port to reach the exit test.
- The six ported functions name mgcv 1.9-1, the version they were read from; their fixtures (`oracle/make_fixtures.R` section 4) are tier 1 from mgcv 1.9-1 until CI's 1.9-4 oracle copies are committed.
- Docstrings in the moved code cite polaris-re ADR numbers (ADR-189 to ADR-265) and `docs/PLAN_*` files that live in polaris-re, not here.
- `compress_covariates` returns unique rows in lexicographic order; mgcv's `compress.df` shuffles them with `sample()`. Same cell partition, different labels. Waits for R's random number stream in M3 (PR #3 review).
- `initial_log10_lambda_from_xx_diag` trims each penalty block to the span from its first to its last non-zero column, while mgcv's `initial.sp` uses the whole stored block. The row and column means behind the `eps^0.8` threshold can therefore differ when a block has zero edge columns. This affects starting values only (PR #3 review).
- REML score: `GamFit.reml_score` differs from mgcv's `m$gcv.ubre` by a normalising constant (782.5 vs 2525.5 on `gam_binomial_cloglog_ti_reml`, 227.0 vs 1087.8 on `gam_poisson_offset_reml`); smoothing parameters agree to 1e-8 relative on the ti fixture, 7e-5 on the Poisson and 4e-4 on `gam_binomial_cloglog_knots_reml`, so the optimum is the same (corrected after the PR #5 review). Not a key result.
- Standard errors of the linear predictor on `gam_binomial_cloglog_ti_reml` differ by up to 8.6e-4 relative (gate 1%); on the Poisson fixture by 6e-6. Likely the weights at which `Vp` is formed for a non-canonical link. Inside the gate.
- `knots=` names that no cr smooth or ti margin uses are refused; mgcv ignores them. Knots reaching a `tp` or `re` term are refused (not ported).
- cr extrapolation outside supplied knots: `_cr_basis_raw` extrapolates linearly as `crspl` does (checked by the PR #6 review), but no fixture holds data outside the supplied knots to mgcv; `gam_binomial_cloglog_knots_reml` keeps all data inside them.
