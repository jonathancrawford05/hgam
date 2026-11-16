# Continuation

Last updated: 2026-11-16, by the development run's fifth slice (select=TRUE with sz). Read `docs/PRIORITIES.md` before
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

The top-level `hgam.gam` / `hgam.bam` are still stubs; the exit test stays a strict xfail. Tests: 613 pass,
1 strict xfail. What differs is in `docs/DEFERRED.md`.

## Next slice: numeric `by` on a cr smooth, held to mgcv

The exit formula has `s(AttdAge, by = StudyYear_C, k = 13, bs = "cr")`. The front end accepts a numeric `by` on cr
(`api.py`, label `s(x):z`), but no oracle fixture in this repository holds it.
1. Read how mgcv applies a numeric `by` in `smoothCon` (smooth.r: the by multiplies X; with a numeric by the sum-to-zero
   constraint is not applied, so the term keeps k coefficients) and check the label and coefficient count.
2. Fixture (section 10): `p ~ s(x, bs="cr", k=6) + s(x, by=z, bs="cr", k=6)` binomial cloglog with weights, REML,
   shared knots; also with `select=TRUE` if the slice stays small.
3. Add the case to `tests/test_gam_reml_fixtures.py` and meet every gate.

After that the exit test still needs: `select=TRUE` on the rank-deficient factor + sz shape (section 8 with select), and
`bam(discrete=TRUE)` on `ti` and sz. Where the exit test compares per-term edf, it compares the summed edf of the terms
that share a redundant direction (Jonathan's decision on PR #8).

## Open items for Jonathan

None. (PR #10's question on SETTINGS condition 5 is decided: option (a). A refusal test may be retired when a
fixture-backed acceptance test of the same input replaces it; the wording is in `docs/routines/SETTINGS.md`.)

## Open questions

None.
