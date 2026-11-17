# Development session log

## 2026-11-17 (sixteenth slice): summary.gam's smooth-term table, Chi.sq and Ref.df

PR #22 auto-merged (all six SETTINGS conditions checked and posted). Ported `reTest`, `recov` (no random terms) and `mroot` (mgcv.r) into `hgam.frontend.smooth_tests`; `summary()` of a discrete fit reports Chi.sq and Ref.df for fully penalised smooths; `bam_fitd` now carries X'WX. New fixture section 18 (`mroot_chol`); `tests/test_smooth_tests.py`; smooth-table tests in `tests/test_summary_bam.py`. PR #22's review nit (r_rng docstring) fixed.
Tests: 665 pass before; 675 pass and 3 strict xfail after (review fix: smooths beside a `bs="re"` term keep None, `recov`'s re branch not ported; null.space.dim by mgcv's select/re rule). `make lint` is clean.
Tier 1, mgcv 1.9-1: Chi.sq 1.7e-6 relative on all eight exit smooths; null.space.dim exact; Ref.df exact on five, the three sz terms 7/6/5 vs 11/7/6 (rounding-set rank; p with hgam's eigenvalues 3e-6 from mgcv's): question on the PR. mroot 5.8e-15.
Deferred: p-value (`psum.chisq`, Davies), `testStat`, random-effect terms in `recov`, the `gam()` path.

## 2026-11-17 (fifteenth slice): discrete bam's constraint rows

Ported R's RNG (`set.seed`, Mersenne-Twister, `sample.int` Rejection: RNG.c, random.c) and `discrete.mf`'s padded frame `mf0` (bam.r, `discrete_model_frame`); cr and ti-margin constraints in `fit_engine_bam` are `colMeans` over it, as `smoothCon` takes them (ADR 0002). New fixture section 17, `tests/test_discrete_constraint_rows.py`; strict xfail lifted.
Tests: 647 pass and 1 strict xfail before; 665 pass, 0 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: R draws exact; `mf0` 5.7e-14; constraint directions 1.2e-15 (gate 1e-8). Exit fit: intercept 4.8e-7 (was 3.6e-3), eta 7.4e-8 (was 1.1e-4), se 7.0e-7, edf 1.3e-6.
Deferred: an offset beside weights is refused in a discrete fit; 12d and 13a eta moves 1e-9 to 3e-6 with the summation order alone; 12a's ti sp still 1.11 times mgcv's (eta 2e-13).

## 2026-11-16 (fourteenth slice): summary of a discrete fit, the parametric table

Ported `summary.gam`'s parametric table (mgcv.r): `GamSummary.parametric` (Estimate, Std. Error from Vp, z, 2 * pnorm), mgcv's coefficient names (`GamFit.parametric_names`); `summary()` now accepts a `discrete=True` fit. New fixture section 16 (`bam_discrete_exit_summary`), `tests/test_summary_bam.py`.
Tests: 644 pass before; 647 pass and 1 strict xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: contrasts 8.1e-7, se 1.8e-5 relative, edf 2.2e-3, names exact. Intercept 3.6e-3 (gate 1e-3): it is s(AttdAge)'s level, the constraint row set of discrete bam (strict xfail; question on the PR).
Deferred: smooth-term tests (Ref.df, Chi.sq, p: testStat / reTest / psum.chisq) to the next slice; estimated-scale p.table not ported.

## 2026-11-16 (thirteenth slice): predict on new data from a discrete fit

Ported `predict.bamd`'s re-discretisation (bam.r): `GamFit.predict` re-bins new data with `discretise_frame` for a `discrete=True` fit; `discrete=False` predicts at raw covariates. New fixture section 15 (`bam_discrete_exit_predict_a`, `_b`), `tests/test_predict_bam.py`. Also fixed review nit 1 of PR #19 in finding 0002.
Tests: 640 pass before; 644 pass after. `make lint` is clean.
Tier 1, mgcv 1.9-1: 15a (binned) eta 8.4e-5, se 2.3e-4 relative; 15b eta 4.0e-5, se 2.2e-4. Without re-binning 15a missed by 3.4e-3.
Deferred: unseen factor levels refused (mgcv zeros them); `type="terms"` / `"lpmatrix"` not ported.

## 2026-11-16 (twelfth slice): the M3 benchmark of the exit model

Timed the exit model, `hgam.bam` vs `mgcv::bam(discrete = TRUE, select = TRUE)`, median of 15 after a warm-up (`docs/findings/0002-exit-benchmark.md`, scripts in `docs/findings/exit_benchmark/`). No engine change.
Tests: 640 pass before and after. `make lint` is clean.
mgcv 1.9-1 0.98 s; hgam 2.2 to 2.4 s at OpenBLAS's default 4 threads, 0.42 s on one thread. Inside the ten-times rule; nothing tuned.
Deferred: BLAS thread oversubscription on small problems, after M4.

## 2026-11-16 (eleventh slice): hgam.bam / hgam.gam wired; the exit test passes

`hgam.gam` (mgcv.r) and `hgam.bam` (bam.r) forward to the front end (`family=` + `link=`, unported methods refused); the response `a / b` (R's `model.frame`) is parsed and evaluated. `tests/test_exit.py` fits `hgam_formula` as written on section 14's data and knots, held to every gate; strict xfail lifted. New `tests/test_entry_points.py`.
Tests: 628 pass and 1 xfail before; 640 pass and 0 xfail after (review fix: hgam.gam defaults to mgcv's method="GCV.Cp" and refuses it). `make lint` is clean.
Tier 1, mgcv 1.9-1: the exit fit equals section 14's (eta 1.1e-4, se 2.6e-4 relative, edf 2.2e-3, structure exact vs sections 3 and 14). Also fixed PR #17's review nit (sp attribution in DEFERRED and the log).
Deferred: nothing new.

## 2026-11-16 (tenth slice): the exit formula on bam(discrete=TRUE, select=TRUE)

New fixture section 14 (`bam_discrete_exit_formula`: section 3's `b` refitted with response column p = DthCnt / ExposCnt, exact bam, rank and parametric edf beside it) held in `tests/test_bam_discrete_fixtures.py`; no engine change.
Tests: 627 pass and 1 xfail before; 628 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: eta 1.1e-4 (mgcv's own discrete vs exact gap 7.0e-3), se 2.6e-4 relative, per-term edf 2.2e-3 (FaceSize, Smoke, FaceSize:Smoke included), rank 132 of 132. Structure exact.
Deferred: sp of the ti term (6-14%), of s(Smoke,PolYear) (x1.96) and of s(FaceSize,PolYear) (x0.62; attribution corrected after the PR #17 review), not gated.

## 2026-11-16 (ninth slice): bam(discrete=TRUE, select=TRUE) on f + sz and on a numeric by

New fixture section 13 (13a: section 11a's `f + s(x) + s(f, x, sz)`, 13b: section 10's numeric by, both `bam(discrete = TRUE, select = TRUE)`, exact bam beside them) held in `tests/test_bam_discrete_fixtures.py`; no engine change.
Tests: 625 pass and 1 xfail before; 627 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: 13a eta 1.0e-9, se 1.6e-7, edf (factor included) 3.1e-8, rank 20 of 20; 13b eta 4.7e-14, se 1.3e-13, edf 3.2e-7. Structure exact.
Deferred: mgcv's discrete = FALSE reference fits for 12d and 13a warn "algorithm did not converge" (informational only).

## 2026-11-16 (eighth slice): bam(discrete=TRUE) on cr + ti and on sz

Ported the one-covariate-margin subset of `discrete.mf` (bam.r) as `discretise_frame`; `fit_engine_bam` now fits on binned covariates and accepts sz. New fixture section 12 (12a cr + ti, 12b sz, 12c/12d with select=TRUE, 12e numeric by (review), each with mgcv's discrete = FALSE bam beside it), `tests/test_bam_discrete_fixtures.py`.
Tests: 619 pass and 1 xfail before; 625 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1, vs mgcv's discrete bam: 12e eta 1e-13; 12a eta 3.7e-4, se 1.2e-3, edf 7.1e-3; 12b eta 2e-6; 12c eta 3.6e-4, se 1.3e-3, edf 6.6e-3; 12d eta 1.5e-9. Structure exact. Before binning, 12a matched mgcv's exact bam (8e-9) and missed its discrete bam (4.0e-3).
Deferred: 12a ti sp ~10% off (constraints over data rows vs mgcv's padded frame); predict on new data does not re-bin.

## 2026-11-16 (seventh slice): select=TRUE on the factor + sz shapes

New fixture section 11 (8a and 8b with `select=TRUE`) and a test in `tests/test_rank_deficient_sz.py`; no engine change. The null-space penalties identify the shared directions: mgcv's rank equals the coefficient count and hgam pivots nothing.
Tests: 617 pass and 1 xfail before; 619 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: 11a eta 8e-5, se 3.6e-4, per-term edf 8.5e-4; 11b eta 3.0e-4, se 9.2e-4, per-term edf 3.6e-3. Structure exact.
Deferred: 11a's near-linear s(x) sp (log10 7.40 vs 7.83, flat ridge); 11b's mgcv convergence message differs on the JSON round trip.

## 2026-11-16 (sixth slice): numeric by on a cr smooth

Held `s(x, by=z, bs="cr")` to mgcv (`smoothCon`'s numeric-by branch, smooth.r); constant `by` (mgcv keeps the constraint) is now refused. New fixture section 10 (10a smoothCon, 10b fit, 10c fit with select=TRUE), `tests/test_cr_by_numeric.py`.
Tests: 613 pass and 1 xfail before; 617 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: basis X 1.2e-14, S 2.1e-15 relative, C 0 rows, labels and 12 coef exact. 10c: eta 2.9e-5, edf 4e-4, se 2.1e-4. 10b: edf 0.029, se 0.79%, eta at mgcv's sp 1e-13, but eta at estimated sp 1.24e-3 (gate 1e-3): mgcv's Newton stops short of its own optimum (hgam is 1.9e-4 from mgcv at conv.tol 1e-12).
Jonathan chose option B: 10b now records mgcv at newton conv.tol 1e-12 (REML 2441.46465 vs default 2441.46481) and eta is gated against it, 1.9e-4 (se 9.7e-4, edf 3.5e-3).

## 2026-11-16 (fifth slice): select=TRUE with sz smooths

Accepted `select=TRUE` with sz (smoothCon's `null.space.penalty` need.full branch, smooth.r); the engine's `null_space_penalty` already built it, so only the front-end refusal went. New fixture section 9 (9a smoothCon S with null.space.penalty, 9b section 7's fit with select=TRUE).
Tests: 610 pass and 1 xfail before; 613 pass and 1 xfail after (the refusal test is replaced by an acceptance test). `make lint` is clean.
Tier 1, mgcv 1.9-1: sz null-space penalty 1.1e-14 / 5.7e-14 relative, ranks exact; fit: 6 sp and 18 coef exact, eta 9.2e-5, edf 1.5e-3, se 3.8e-4 relative.
Deferred: the near-linear s(x) penalty's sp (log10 7.29 vs 8.16, flat ridge).

## 2026-11-16 (fourth slice): a factor main effect beside sz smooths

New fixture section 8 (8a: `f + s(x) + s(f, x, sz)`; 8b: `f` with two sz smooths of a two-level factor) and `tests/test_rank_deficient_sz.py`; no engine change.
Tests: 608 pass and 1 xfail before; 610 pass and 1 xfail after. `make lint` is clean.
Tier 1, mgcv 1.9-1: eta 1.0e-4 / 3.6e-5, se 5.8e-4 / 3.7e-4 relative; edf of the involved terms summed 2.1e-3 / 1.2e-3; uninvolved terms inside the gate.
Not met: the per-term split inside the involved terms. mgcv's `gdiPK` choice depends on an arbitrary eigenvector basis (39 of 200 rotations change it): question for Jonathan, DEFERRED.

## 2026-11-16 (third slice): the sz basis

Accepted `s(f, x, bs="sz", k=, xt=list(bs="cr"))` in the front end (mgcv `smooth.construct.sz.smooth.spec` + `smoothCon`'s `XZKr`), knots shared as for cr; other xt, x-first order, `by=` and `select=TRUE` with sz are refused. New fixture section 7.
Tests: 596 pass and 1 xfail before; 608 pass and 1 xfail after (one refusal case in `test_formula.py` now names the still-refused no-xt form). `make lint` is clean.
sz basis vs `smoothCon` (2 and 3 levels): X 1.5e-14, S 1.6e-15 relative, ranks and bs.dim exact. sz fit (tier 1, mgcv 1.9-1): eta 1.0e-4, edf 2.1e-3, se 5.8e-4 relative; default knots would miss eta (1.7e-2).
Deferred: sp differences (log10 units; corrected after review); `f + sz` and two-sz-per-factor rank deficiency (per-term edf split differs from mgcv, next slice); the -2 lower bound on log10 lambda.

## 2026-11-16 (second slice): user-supplied cr knots

Added `knots=` to `hgam.frontend.gam` (mgcv `smooth.construct.cr.smooth.spec`): shared by a variable's cr smooths and ti margins, count must equal k, refused for tp/re and for unused names. New fixture section 6; section 2 now records `labels` (PR #5 nits).
Tests: 586 pass and 1 xfail before; 596 pass and 1 xfail after (incl. review follow-up: by-form knot tests). `make lint` is clean.
cr basis at supplied knots vs `smoothCon`: 1e-14 relative. Knots fit (tier 1): eta 3.3e-5, edf 2.8e-4, se 2.4e-4 relative; the same fit with default knots misses the eta gate (7.6e-3), so the knots are used.
Corrected PR #5's sp claim: 7e-5 on the Poisson fixture, not 1e-7.

## 2026-11-16: M1, outer REML search held to mgcv

Measured the moved Newton REML search against two `gam(method="REML")` fixtures (new: cr + ti binomial-cloglog with weights, section 5; existing: Poisson offset, section 2) instead of porting `newton`: it meets every gate, so it stays ("reuse before porting").
Tests: 583 pass and 1 xfail before; 586 pass and 1 xfail after. `make lint` is clean.
Largest differences (tier 1, mgcv 1.9-1): eta 5.7e-10 / 9.6e-7, edf 1.1e-8 / 2.9e-5, se 8.6e-4 / 5.9e-6 relative; sp agree to 1e-7.
Deferred: the REML score's normalising constant (not a key result) and the 8.6e-4 se difference (inside its gate).

## 2026-11-15: M0, move the engine

Moved 28 engine and front-end modules (about 10,800 lines) from polaris-re into `hgam.engine` and `hgam.frontend`, with no `polaris_re` imports left; dropped polars for NumPy columns; added mgcv counterpart lines to the six ported functions.
Tests: 5 pass and 1 xfail before; 564 pass and 1 xfail after. `make lint` is clean.
No key result was measured: this slice adds no fixture. 43 harness-bound tests stayed behind (`docs/DEFERRED.md`).
The tier-3 fixture artifact could not be downloaded: the network policy blocks its host.
After review: unported refusals raise `HgamNotImplementedError`; the six ports gained tier 1 fixtures (`tero`, `compress.df`, `initial.sp`, `Sl.fitChol`, `bgam.fitd` x2), all inside their gates (eta within 3e-9, edf within 3e-7). 583 pass.

## 2026-11-15: routines and handoff

Added the daily-development and review routines and their settings file, split the engine move into three
steps, and recorded in ADR 0001 that the project is the maintainer's own, with no employer sign-off pending.

## 2026-11-14: scaffold

Created the repository skeleton: package stub, GPL-2.0 licence text, CI, the fixture generator and
three tier 1 fixtures, the exit test as a strict xfail, scope, plan and ADR 0001.
Recorded the mgcv-rust check under `docs/findings/`.
