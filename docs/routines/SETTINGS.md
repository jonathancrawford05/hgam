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
     `docs/PRIORITIES.md`, `oracle/compare_fixtures.py`, or anything in `docs/routines/` other than
     `CONTINUATION.md` and `DEV_SESSION_LOG.md`, which every run must update.
  5. No test was removed, skipped or newly marked xfail, and no gate in `docs/PRIORITIES.md` was loosened.
  6. The "Ready" review ends with a `claude.ai/code/session_...` link, and that link differs from the session
     link at the foot of the pull request body. A review with no session link does not count.
  If any one fails, the pull request waits for Jonathan.
- `max_slices_per_run: 2`
