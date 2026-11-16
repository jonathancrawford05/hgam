# Continuation

Last updated: 2026-11-16, by the development run's sixth slice (numeric by on cr). Read `docs/PRIORITIES.md` before
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
6e-14, and section 7's model with `select=TRUE` meets every gate. Not yet measured: `select=TRUE` together with a
factor main effect beside its sz smooths (section 8's rank-deficient shape).

A numeric `by` on cr, `s(x, by=z, bs="cr")` (section 10, `tests/test_cr_by_numeric.py`): basis vs `smoothCon` at
1.2e-14, no constraint (k coefficients, label `s(x):z`); a constant `by`, for which mgcv keeps the constraint, is
refused. 10c (`s(x) + s(x, by=z)` with `select=TRUE`) meets every gate. 10b (the same without select) meets structure,
edf and se, and the fit at mgcv's sp to 1e-13, but its eta at the estimated sp is 1.24e-3 from mgcv's default fit (gate
1e-3) and is not gated: mgcv's Newton stops short of its own REML optimum (question below, `docs/DEFERRED.md`).

The top-level `hgam.gam` / `hgam.bam` are still stubs; the exit test stays a strict xfail. Tests: 617 pass,
1 strict xfail. What differs is in `docs/DEFERRED.md`.

## Next slice

If Jonathan has answered the 10b question on the open pull request, apply the answer there first (rule 1 of
`DAILY_DEV.md`). Otherwise the pull request waits.

After it merges: `select=TRUE` on the rank-deficient factor + sz shape. Fixture section 11: section 8a's model
`p ~ f + s(x, bs="cr", k=6) + s(f, x, bs="sz", k=6, xt=list(bs="cr"))` with `select=TRUE`, held at the gates in
`tests/test_rank_deficient_sz.py` (summed edf of `f` and the sz term). Then `bam(discrete=TRUE)` on `ti` and sz. Where the
exit test compares per-term edf, it compares the summed edf of the terms that share a redundant direction (Jonathan's
decision on PR #8).

## Open items for Jonathan

- **How to gate eta on section 10b** (`gam_binomial_cloglog_by_numeric_reml`), where mgcv's default Newton search
  stops short of its own REML optimum on a flat ridge (mgcv's score is lower at hgam's sp; mgcv at conv.tol 1e-12 moves
  1.4e-3 from its default and lands 1.9e-4 from hgam). Options: (A) port mgcv's `newton` (gam.fit3.r) so hgam stops
  where mgcv stops: several slices, and the exit model may need it anyway; (B) gate eta at estimated sp against
  mgcv refitted at a tight `conv.tol` whenever mgcv's own score is lower at hgam's point (fixture records both fits);
  (C) leave 10b's eta ungated, as now, and rely on 10c and the exit test. Recommendation: B, it holds hgam to mgcv's
  own optimum at the existing gate and costs one fixture line; switch to A only if the exit model misses on the search.

(PR #10's question on SETTINGS condition 5 is decided: option (a). A refusal test may be retired when a
fixture-backed acceptance test of the same input replaces it; the wording is in `docs/routines/SETTINGS.md`.)

## Open questions

None.
