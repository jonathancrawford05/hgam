# Continuation

Last updated: 2026-11-14, by the scaffold session.

## State

Scaffold only. `hgam.gam` and `hgam.bam` raise `NotImplementedError`. Tests: 5 pass, 1 strict xfail
(the exit test). Fixtures are tier 1 (local R 4.3.3, mgcv 1.9-1): provisional.

## Next slice: M0, move the engine

See `docs/PLAN.md`, "M0, second slice". One pull request. Done when the moved tests pass here with no
import from `polaris_re`, and the six ported functions carry their mgcv counterpart lines.

## Open items for the maintainer

1. GPL sign-off with the employer's open-source policy owner (ADR 0001). The repository stays private
   until then.
2. Run the `oracle-fixtures` workflow once and commit its tier 3 fixtures in place of the tier 1 ones.
   The image is pinned to R-Gam-base build 10 (R 4.6.1, mgcv 1.9.4).
3. Set up the daily-dev and review routines as Claude Code scheduled tasks bound to sessions with this
   repository attached.
