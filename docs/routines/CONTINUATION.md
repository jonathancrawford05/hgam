# Continuation

Last updated: 2026-11-16, by the development run's fourth slice (rank-deficient sz fixtures). Read `docs/PRIORITIES.md` before
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

The top-level `hgam.gam` / `hgam.bam` are still stubs; the exit test stays a strict xfail. Tests: 610 pass,
1 strict xfail. What differs is in `docs/DEFERRED.md`.

## Next slice: `select=TRUE` with sz terms

The exit formula uses `select=TRUE`; sz with `select=TRUE` is refused today (`test_sz_with_select_is_refused_until_measured`).
1. Read how mgcv adds the null-space penalty for sz (`smoothCon` with `null.space.penalty`, smooth.r; the sz constructor
   sets its own null space per level).
2. Fixture (section 9): section 7's model with `select=TRUE`, plus `smoothCon(..., null.space.penalty=TRUE)` S for sz.
3. Accept `select=TRUE` with sz, hold the penalties to 1e-8 and the fit to the gates.

After that the exit test still needs: numeric `by` and `bam(discrete=TRUE)` on `ti` and sz. Where the exit test
compares per-term edf, it compares the summed edf of the terms that share a redundant direction (decision above).

## Open items for Jonathan

None.

## Open questions

None.
