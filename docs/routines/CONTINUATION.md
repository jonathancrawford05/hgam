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
The split of that sum between the factor and its sz smooths does not, and is not portable (DEFERRED; question below).

The top-level `hgam.gam` / `hgam.bam` are still stubs; the exit test stays a strict xfail. Tests: 610 pass,
1 strict xfail. What differs is in `docs/DEFERRED.md`.

## Next slice: `select=TRUE` with sz terms

The exit formula uses `select=TRUE`; sz with `select=TRUE` is refused today (`test_sz_with_select_is_refused_until_measured`).
1. Read how mgcv adds the null-space penalty for sz (`smoothCon` with `null.space.penalty`, smooth.r; the sz constructor
   sets its own null space per level).
2. Fixture (section 9): section 7's model with `select=TRUE`, plus `smoothCon(..., null.space.penalty=TRUE)` S for sz.
3. Accept `select=TRUE` with sz, hold the penalties to 1e-8 and the fit to the gates.

After that the exit test still needs: numeric `by`, `bam(discrete=TRUE)` on `ti` and sz, and the answer below on
per-term edf in the rank-deficient sz shape.

## Open items for Jonathan

**Per-term edf where a factor's contrasts are redundant with its sz smooths** (the exit formula's `FaceSize + ... +
s(FaceSize, AttdAge, bs="sz") + s(FaceSize, PolYear, bs="sz")`). mgcv drops the redundant directions in `gdiPK`
(gdi.c) by a pivoted QR in the `totalPenaltySpace` basis, an arbitrary basis of a degenerate eigenspace. Under 200
equally valid rotations of that basis mgcv's own rule changes the split 39 times (8a: f edf 1 or 2), so no port can
promise to reproduce it, and the pinned image's LAPACK may itself split differently from a local R. The sum over the
involved terms is invariant and already meets the gate. Options:
- A. Gate the summed edf of the terms an unidentified direction runs through (and every other term as now); report
  the split as not identified. Needs a line in `docs/PRIORITIES.md` (yours to change).
- B. Port `totalPenaltySpace` + `gdiPK`'s rank decision at the converged working weights, refitting if it changes,
  and accept that agreement depends on matching LAPACK's eigenvector basis (it does not match scipy's today).
- C. Change the exit formula so it is not rank deficient (drop `FaceSize` or one sz term): a scope change.
Recommendation: A. It is what the data identify, it is reproducible, and it costs no further slices.

## Open questions

None.
