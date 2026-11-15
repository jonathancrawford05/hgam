# Routine settings

Read by `DAILY_DEV.md` and `PR_REVIEW.md`. Only Jonathan edits this file.

- `auto_merge: on`
  The development routine squash-merges its own pull request when **all** of these hold, and then starts the next
  slice in the same run:
  1. Every CI check on the head commit passed, including `oracle`, which regenerates the fixtures inside the
     pinned image and runs the tests against them. A missing or skipped `oracle` check counts as a failure.
  2. A routine review of the head has the verdict "Ready for a human to merge".
  3. The pull request has no "Questions for Jonathan".
  4. The slice touches nothing under "Always ask" in `DAILY_DEV.md`, and does not change `.github/`,
     `docs/routines/`, `docs/PRIORITIES.md` or `oracle/compare_fixtures.py`.
  5. No test was removed, skipped or newly marked xfail, and no gate in `docs/PRIORITIES.md` was loosened.
  If any one fails, the pull request waits for Jonathan.
- `max_slices_per_run: 2`
