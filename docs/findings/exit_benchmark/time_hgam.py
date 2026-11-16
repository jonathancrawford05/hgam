"""Time hgam.bam on the exit-test model: tests/test_exit.py's call on fixture bam_discrete_exit_formula's data
(oracle/make_fixtures.R section 3's frame, n = 4000) and knots. Median of `reps` runs after one warm-up.

Usage: uv run python docs/findings/exit_benchmark/time_hgam.py [reps]
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import hgam

HGAM_FORMULA = (
    "DthCnt / ExposCnt ~ FaceSize + Smoke + FaceSize:Smoke"
    " + s(AttdAge, k=13, bs='cr') + s(PolYear, k=6, bs='cr')"
    " + ti(AttdAge, PolYear, k=c(13, 6), bs='cr')"
    " + s(FaceSize, AttdAge, bs='sz', k=13, xt=list(bs='cr'))"
    " + s(Smoke, AttdAge, bs='sz', k=13, xt=list(bs='cr'))"
    " + s(FaceSize, PolYear, bs='sz', k=6, xt=list(bs='cr'))"
    " + s(Smoke, PolYear, bs='sz', k=6, xt=list(bs='cr'))"
    " + s(AttdAge, by=StudyYear_C, k=13, bs='cr')"
)

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 15
root = Path(__file__).resolve().parents[3]
ref = json.loads((root / "tests/fixtures/bam_discrete_exit_formula.json").read_text())
rows = ref["data"]
data = {
    name: np.asarray([r[name] for r in rows])
    if isinstance(rows[0][name], str)
    else np.asarray([r[name] for r in rows], dtype=np.float64)
    for name in rows[0]
    if name != "p"
}


def fit() -> object:
    return hgam.bam(
        HGAM_FORMULA,
        data=data,
        family="binomial",
        link="cloglog",
        weights="ExposCnt",
        knots=ref["knots"],
        discrete=True,
        select=True,
    )


fit()
times = []
for _ in range(reps):
    t0 = time.perf_counter()
    fit()
    times.append(time.perf_counter() - t0)
print(
    f"hgam (numpy {np.__version__}): median {statistics.median(times):.4f} s, "
    f"min {min(times):.4f} s, max {max(times):.4f} s over {reps} runs"
)
