# Routine settings

Read by `DAILY_DEV.md` and `PR_REVIEW.md`. Only Jonathan edits this file.

- `auto_merge: off`
  - `off`: Jonathan merges every pull request.
  - `on`: the development routine may squash-merge its own pull request when all of these hold: CI is
    green on the head; a routine review of the head has the verdict "Ready for a human to merge"; the
    pull request has no "Questions for Jonathan"; and the slice touches nothing under "Always ask" in
    `DAILY_DEV.md`. It then starts the next slice in the same run.
- `max_slices_per_run: 1`
