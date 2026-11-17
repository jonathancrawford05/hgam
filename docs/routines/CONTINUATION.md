# Continuation

Last updated: 2026-11-16, by the development run's fourteenth slice (M4: `summary` of a discrete fit, parametric table). Read `docs/PRIORITIES.md` before
choosing work.

## State

M0 is done (engine in `src/hgam/engine/`, front end in `src/hgam/frontend/`; errors in `hgam.errors`). The six
mgcv-ported functions are held to fixtures in `tests/test_ported_functions.py`.

M1, outer smoothing-parameter search: measured, not ported (PR #5). `tests/test_gam_reml_fixtures.py` holds
`hgam.frontend.gam(method="REML")` to four `mgcv::gam` REML fixtures (sections 2, 5, 6 and 7 of
`oracle/make_fixtures.R`); all meet every gate, so mgcv's `newton` is not ported unless a later fixture fails a
gate on the search.

User-supplied knots: `knots={"x": [...]}` reach every cr smooth, ti margin and sz smooth of that variable
(`tests/test_cr_knots.py`, section 6).

The sz basis: `s(f, x, bs="sz", k=, xt=list(bs="cr"))` is accepted (factor first; other xt, `by=`, x-first order
and `select=TRUE` with sz are refused). Basis vs `smoothCon` at 1.5e-14; the section 7 fit
`p ~ s(x, bs="cr") + s(f, x, bs="sz", ...)` meets every gate (`tests/test_sz_basis.py`, section 7).

A factor main effect beside sz smooths of that factor (section 8, `tests/test_rank_deficient_sz.py`): structure,
eta, se, the uninvolved terms' edf and the summed edf of the terms sharing the redundant directions meet their gates.
The split of that sum between the factor and its sz smooths is not identified and not portable (DEFERRED). Jonathan
decided on PR #8 (option A): those terms are gated on their summed edf, at the per-term gate (0.05 absolute or 1%,
whichever is larger); the split is reported as not identified and is not gated. `gdiPK`'s rank decision is not
ported for this and the exit formula stays as it is. The matching `docs/PRIORITIES.md` line is Jonathan's change.

`select=TRUE` with sz smooths is accepted (section 9, `tests/test_sz_basis.py` and the section 9 case in
`tests/test_gam_reml_fixtures.py`): the null-space penalty agrees with `smoothCon(..., null.space.penalty=TRUE)` to
6e-14, and section 7's model with `select=TRUE` meets every gate. With a factor main effect beside its sz smooths
(section 11: sections 8a and 8b with `select=TRUE`, `tests/test_rank_deficient_sz.py`) the null-space penalties make
the fit full rank in mgcv and hgam, and every term meets the per-term edf gate.

A numeric `by` on cr, `s(x, by=z, bs="cr")` (section 10, `tests/test_cr_by_numeric.py`): basis vs `smoothCon` at
1.2e-14, no constraint (k coefficients, label `s(x):z`); a constant `by`, for which mgcv keeps the constraint, is
refused. 10c (with `select=TRUE`) meets every gate. 10b meets structure, edf and se against mgcv's default fit, and eta
at the estimated sp against mgcv refitted at `newton` `conv.tol = 1e-12` (1.9e-4; option B, Jonathan's decision on
PR #12): mgcv's default stops short of its own REML optimum there.

`bam(method="fREML", discrete=TRUE)` (section 12, `tests/test_bam_discrete_fixtures.py`): `gam(..., method="fREML",
discrete=True)` bins every numeric covariate a smooth reaches with `compress.df`, one covariate at a time, as
`discrete.mf` does for one-covariate margins (`hgam.frontend.discrete.discretise_frame`), then runs `bam_fitd`. sz is
accepted there. On section 5's cr + ti model and section 7's sz model, with and without `select=TRUE`, every key result
meets its gate (eta at most 3.7e-4); so does section 10's numeric-by model (12e, eta 1e-13). Section 13 adds
`select=TRUE` on a factor beside its sz smooth (13a, full rank, factor edf gated) and on the numeric by (13b): eta 1e-9
and 5e-14. Without binning hgam matched mgcv's `discrete = FALSE` bam instead (8e-9).

The exit formula itself (section 14, `bam_discrete_exit_formula`, section 3's frame with n = 4000 and `my_knots`,
response column p = DthCnt / ExposCnt, weights ExposCnt) meets every gate through
`hgam.frontend.gam(..., method="fREML", discrete=True, select=True)`: eta 1.1e-4, se 2.6e-4 relative, per-term edf
2.2e-3 including FaceSize, Smoke and FaceSize:Smoke, full rank (132 of 132) in mgcv and hgam. Structure exact.

The public entry points are wired: `hgam.gam` (`method="REML"` written out; mgcv's default `"GCV.Cp"` is refused) and `hgam.bam` (`method="fREML", discrete=True` only; bam's
default `discrete=FALSE` is refused by name) forward to `hgam.frontend.api.gam`, with `family=` as mgcv writes it or
`family="binomial", link="cloglog"`. A response `a / b` is evaluated as R's `model.frame` does (other left-hand
expressions are refused; a zero denominator is refused). **The exit test passes**: `tests/test_exit.py` fits
`hgam_formula` as written on section 14's data and knots through `hgam.bam` and meets every gate. Tests: 640 pass,
no xfail. What differs is in `docs/DEFERRED.md`.

Run time (M3's one benchmark, `docs/findings/0002-exit-benchmark.md`): on the exit model mgcv's `bam` takes a median
0.98 s; `hgam.bam` 2.2 to 2.4 s with OpenBLAS's default four threads and 0.42 s with `OPENBLAS_NUM_THREADS=1`. Inside
the ten-times rule, so nothing was tuned. Candidate for after M4 (Priorities item 3): cap BLAS threads for small
problems, or document it; this needs no new dependency only if done through an environment variable, so ask first if
it would need `threadpoolctl`.

`predict` on new data from a discrete fit (section 15, `tests/test_predict_bam.py`): `GamFit.predict` re-runs
`discretise_frame` on the new data, as `predict.bamd` re-runs `discrete.mf` (`discrete=False` predicts at the raw
covariates, as `predict.bam` falling through to `predict.gam`). On the exit fit, 15a (1500 rows, binned) eta 8.4e-5
and se 2.3e-4 relative; 15b (40 rows, not binned) eta 4.0e-5, se 2.2e-4. Without re-binning 15a missed by 3.4e-3.

`summary` of a discrete fit, first step (section 16, `tests/test_summary_bam.py`, PR open): `GamSummary.parametric` is
`summary.gam`'s `p.table` for a fixed scale (Estimate, Std. Error from Vp, z, `2 * pnorm`), under mgcv's coefficient
names (`GamFit.parametric_names`, from `model.matrix`'s rule); `summary()` accepts a `discrete=True` fit (no REML
score). On the exit fit, with mgcv's level order (Low, High): contrasts 8.1e-7, se 1.8e-5 relative, names exact. The
intercept is 3.6e-3 off (gate 1e-3, strict xfail `test_intercept_meets_the_estimate_gate`): the gap is all the level
of `s(AttdAge)` (shape agrees to 8e-6). `discrete.mf` bins AttdAge to 981 points and the others to 985, so mgcv pads
AttdAge with 4 seeded `sample()` draws before `gam.setup` takes `colSums(X)` over `mf0`. But mgcv's stored constraint
direction (`attr(b14$smooth[[1]], "qrc")`) matches neither the 981 distinct points (1 - cos 8e-6) nor those plus the 4
pads (2.3e-5), so the row set is not pinned yet. Scripts that showed this were scratch (not committed): rebuild
`discrete.mf`'s input as bam does (sz specs get one-variable margins first, bam.r ~2265) to call it directly.

## Next slice: pin and port discrete bam's constraint rows (Q1, option A)

Pin the row set of `s(AttdAge)`'s constraint in `bam(discrete=TRUE)` (read `gam.setup` with `modCon=2`,
`smoothCon` with `sparse.cons=0` on `mf0`, and the padding in `discrete.mf`), port it into `unique_row_constraint`
(ADR-260 amendment 9 chose not to; this reopens it, so write the ADR amendment), and lift the strict xfail. It should
also move the ti sp gap in `docs/DEFERRED.md`. Re-check every bam fixture's gates afterwards.

Then: the smooth-term table of `summary.gam` on the exit fit, at the Q2 gates above. Every exit smooth
has `null.space.dim == 0` (select=TRUE), so all go through `reTest` / `recov` (mgcv.r), not `testStat`, and the p-value
through `psum.chisq` (Davies' algorithm, C code `davies.c`; `liu2` is only its fallback). Section 16's fixture already
holds `s_table` (edf, Ref.df, Chi.sq, p-value). Ref.df here is 12, 5, 60, 11, 7, 6, 5, 13.

## Open items for Jonathan

None.

Decided on PR #21 (2026-11-16): Q1 option (A): port mgcv's constraint rows for discrete bam, held to the
"Deterministic pieces ... constraints" gate, so the intercept meets the estimate gate. The strict xfail
`test_intercept_meets_the_estimate_gate` stays until that port lifts it. Q2 option (a): smooth-term tests are gated
at Ref.df exact; Chi.sq 1% relative; p-value 1e-3 absolute or 1% relative, whichever is larger. The matching
`docs/PRIORITIES.md` row is Jonathan's change (a routine slice does not edit that file).

Decided on PR #12 (option B): where mgcv's default search stops short of its own optimum, a fitted quantity at
estimated smoothing parameters is gated against mgcv refitted at a tight `newton` `conv.tol`, at the unchanged gate.
The fixture records both mgcv fits and their REML scores, and the tight fit is the reference only where mgcv's own
tight score is not higher than its default score. mgcv's `newton` is not ported for this (option A not taken); the
moved Newton search stays. The rule is in `docs/PRIORITIES.md`.

(PR #10's question on SETTINGS condition 5 is decided: option (a). A refusal test may be retired when a
fixture-backed acceptance test of the same input replaces it; the wording is in `docs/routines/SETTINGS.md`.)

## Open questions

None.

PR #15 review nit (not done, predates it): an sz term's factor is assumed to be `variables[0]`; `s(x, f, bs="sz")`
would fail with a bare `ValueError`, not a named refusal. Add the refusal when the sz parser is next touched.
