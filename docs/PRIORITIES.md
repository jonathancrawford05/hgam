# Priorities

What this build optimises for, in order. When two rules pull against each other, the higher one wins.

1. **Alignment on key results.** hgam must reproduce what a user reads off an mgcv fit. Nothing else is traded
   for this.
2. **Build time and attention.** Reach the exit test in as few slices and as little of Jonathan's time as
   possible. This outranks everything below.
3. **Run time.** Ultimately at least as fast as mgcv on the same model. Measured from M3; not a gate before then.
4. **Everything else**: breadth beyond `docs/SCOPE.md`, internal elegance, exhaustive agreement on quantities
   nobody reads. Revisit after the exit test.

The aim is effective replication, not parity for its own sake.

## Key results and their gates

These are the quantities that gate a merge. Differences are against mgcv from the pinned oracle image.

| Quantity | Gate |
|---|---|
| Structure: term labels, coefficient counts, number of smoothing parameters | exact |
| Deterministic pieces: basis and penalty matrices, constraints, a fit at fixed smoothing parameters | 1e-8 relative |
| Linear predictor of a fitted model, smoothing parameters estimated | 1e-3 absolute, largest difference |
| Effective degrees of freedom, per term | 0.05 absolute or 1%, whichever is larger |
| Standard errors of the linear predictor | 1% relative |

The 1e-3 gate reflects mgcv itself: on the same model its own `gam`, `bam` and `bam(discrete = TRUE)` differ by
1e-3 to 4e-2 on the linear predictor (`docs/findings/0001-mgcv-rust-check.md`). Tighter agreement than mgcv has
with itself is not a goal. Changing a gate is an "Always ask" item.

## What this means in practice

- **Stop at the gate.** Once a slice meets its gates, merge and move on. Do not spend a run closing a difference
  that is already inside its gate.
- **Record, do not chase.** A difference in a quantity that is not a key result (iteration counts, the REML score
  to the last digit, an intermediate matrix in a different but equivalent parameterisation) goes in
  `docs/DEFERRED.md` in one line and does not block.
- **Port the shortest path first.** Take the mgcv code path the exit test actually runs. Branches it does not
  reach raise `NotImplementedError` and wait.
- **Reuse before porting.** Where the moved engine already meets the gates, keep it. Re-port a piece only when it
  fails a gate.
- **Little ceremony.** A session log entry is five lines. An ADR is only for a decision a later session might
  reopen. No measurement reports, no restating of earlier results.
- **One benchmark, late.** From M3, time the exit-test model against mgcv once per milestone and record it. Tune
  only after the exit test passes, unless a slice is more than ten times slower than mgcv.
