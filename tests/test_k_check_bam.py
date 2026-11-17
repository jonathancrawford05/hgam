"""``k.check`` of the exit fit, held to ``k.check`` (plots.r) on mgcv's ``b14`` under ``set.seed(21)``.

``make_fixtures.R`` section 21. Gates: labels and k' exact (structure); edf at the per-term edf gate. The
deviance residuals the test is built on are held at the eta gate (1e-3 absolute; they are a function of eta).
The k-index and p-value are not key results in ``docs/PRIORITIES.md`` (no gate); they are asserted at what was
measured here (k-index 6.8e-10 relative, p-value exact under the seed) so a change is seen. The ti term's
k-index and p-value are not ported (kd-tree ``nearest``); the three sz terms are NA in mgcv as here."""

import json
from pathlib import Path

import numpy as np
import pytest

import hgam
from hgam.frontend.api import Factor, GamFit
from hgam.frontend.gam_check import deviance_residuals

FIXTURES = Path(__file__).parent / "fixtures"

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


def _load(name: str) -> dict:
    with (FIXTURES / f"{name}.json").open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def ref() -> dict:
    return _load("bam_discrete_exit_k_check")


@pytest.fixture(scope="module")
def frame() -> dict[str, object]:
    rows = _load("bam_discrete_exit_formula")
    levels = _load("bam_discrete_exit_summary")["levels"]
    data: dict[str, object] = {}
    for name in rows["data"][0]:
        if name == "p":
            continue
        values = [r[name] for r in rows["data"]]
        data[name] = Factor(values, levels[name]) if name in levels else np.asarray(values, dtype=np.float64)
    return data


@pytest.fixture(scope="module")
def exit_fit(frame) -> GamFit:
    knots = _load("bam_discrete_exit_formula")["knots"]
    return hgam.bam(
        HGAM_FORMULA,
        data=frame,
        family="binomial",
        link="cloglog",
        weights="ExposCnt",
        knots=knots,
        discrete=True,
        select=True,
    )


@pytest.fixture(scope="module")
def table(exit_fit, frame, ref):
    return exit_fit.k_check(frame, seed=ref["seed"], subsample=ref["subsample"], n_rep=ref["n_rep"])


def test_deviance_residuals_meet_the_eta_gate(exit_fit, ref):
    rsd = deviance_residuals(exit_fit)
    diff = np.max(np.abs(rsd - np.asarray(ref["residuals"])))
    print(f"deviance residuals: max |difference| {diff:.2e}")
    assert diff < 1e-3


def test_structure_and_edf(table, ref):
    assert ref["columns"] == ["k'", "edf", "k-index", "p-value"]
    assert [r.label for r in table] == ref["names"]
    k_table = ref["k_table"]
    assert [r.k_prime for r in table] == [int(row[0]) for row in k_table]
    edf = np.array([r.edf for r in table])
    ref_edf = np.array([row[1] for row in k_table], dtype=np.float64)
    assert np.all(np.abs(edf - ref_edf) <= np.maximum(0.05, 0.01 * np.abs(ref_edf)))


def test_na_where_mgcv_has_na(table, ref):
    for r, row in zip(table, ref["k_table"], strict=True):
        if row[2] == "NA":
            assert r.status == "factor" and r.k_index is None and r.p_value is None
        else:
            assert r.status in ("tested", "not_ported")


def test_k_index_and_p_value(table, ref):
    """Not key results: asserted at what was measured (k-index 6.8e-10 relative; p exact, R's draws)."""
    tested = [(r, row) for r, row in zip(table, ref["k_table"], strict=True) if r.status == "tested"]
    assert [r.label for r, _ in tested] == ["s(AttdAge)", "s(PolYear)", "s(AttdAge):StudyYear_C"]
    for r, row in tested:
        print(f"{r.label}: k-index {r.k_index:.6f} vs {row[2]:.6f}; p {r.p_value} vs {row[3]}")
        assert r.k_index is not None and r.p_value is not None
        np.testing.assert_allclose(r.k_index, row[2], rtol=1e-6)
        assert r.p_value == row[3]


def test_ti_is_not_ported(table):
    ti = [r for r in table if r.label.startswith("ti(")]
    assert len(ti) == 1 and ti[0].status == "not_ported" and ti[0].k_index is None


def test_factor_by_is_refused():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(size=n)
    g = Factor(list(rng.choice(["a", "b"], size=n)), ("a", "b"))
    y = np.sin(3 * x) + rng.normal(scale=0.3, size=n)
    fit = hgam.gam("y ~ g + s(x, by=g, k=6, bs='cr')", data={"x": x, "g": g, "y": y}, method="REML")
    with pytest.raises(hgam.errors.HgamNotImplementedError, match="factor-by"):
        fit.k_check({"x": x}, seed=1)


def test_subsample_branch(exit_fit, frame, ref):
    """``n > subsample``: ``sample(1:n, subsample)`` picks the rows first (PR #28 review 7c35e5c)."""
    sub = ref["subsampled"]
    table = exit_fit.k_check(frame, seed=ref["seed"], subsample=sub["subsample"], n_rep=sub["n_rep"])
    for r, row in zip(table, sub["k_table"], strict=True):
        if r.status != "tested":
            continue
        print(f"subsampled {r.label}: k-index {r.k_index:.8f} vs {row[2]:.8f}; p {r.p_value} vs {row[3]}")
        assert r.k_index is not None
        np.testing.assert_allclose(r.k_index, row[2], rtol=1e-6)
        assert r.p_value == row[3]
