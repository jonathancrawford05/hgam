"""``gam.check`` of the exit fit, held to ``gam.check(b14)`` (plots.r) under ``set.seed(22)``, plots off.

``make_fixtures.R`` section 22. Gates: method, optimizer, model rank and coefficient count exact
(structure); the k table's labels and k' exact and edf at the per-term edf gate. The k-index and p-value
are not key results (no gate in ``docs/PRIORITIES.md``); they are asserted at what was measured (k-index
1.3e-10 relative, p exact, which needs ``qq.gam``'s ten ``sample(U, n)`` draws before ``k.check``).
``outer.info``'s gradient and Hessian are not key results either: they are taken at each fit's own
smoothing parameters (``log(sp)`` up to 0.058 apart), so they are asserted at the measured difference
(grad 1.1e-5, Hessian 6.2e-6 absolute)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import hgam

sys.path.insert(0, str(Path(__file__).parent))
from test_k_check_bam import exit_fit, frame  # noqa: E402, F401  (module-scoped fixtures, shared)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ref() -> dict:
    with (FIXTURES / "bam_discrete_exit_gam_check.json").open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def report(exit_fit, frame, ref):  # noqa: F811
    return exit_fit.gam_check(frame, seed=ref["seed"], k_sample=ref["k_sample"], k_rep=ref["k_rep"])


def test_structure(report, ref):
    assert report.method == ref["method"] == "fREML"
    assert list(report.optimizer) == ref["optimizer"]
    assert report.rank == ref["rank"]
    assert report.n_coef == ref["n_coef"]


def test_outer_info(report, ref):
    """Not key results: asserted at the measured difference (each fit at its own sp)."""
    grad, hess = np.asarray(ref["grad"]), np.asarray(ref["hess"])
    assert report.grad.shape == grad.shape and report.hess.shape == hess.shape
    dg, dh = np.max(np.abs(report.grad - grad)), np.max(np.abs(report.hess - hess))
    print(f"outer.info: grad max |difference| {dg:.2e}, hess {dh:.2e}")
    assert dg < 2e-5
    assert dh < 1e-5


def test_k_table(report, ref):
    assert [r.label for r in report.k_table] == ref["names"]
    for r, row in zip(report.k_table, ref["k_table"], strict=True):
        assert r.k_prime == int(row[0])
        assert abs(r.edf - row[1]) <= max(0.05, 0.01 * abs(row[1]))
        if row[2] == "NA":
            assert r.k_index is None and r.p_value is None
            continue
        assert r.k_index is not None
        np.testing.assert_allclose(r.k_index, row[2], rtol=1e-6)
        assert r.p_value == row[3]


def test_printed_lines(report, ref):
    """The method line, rank line, footnote and k table print as mgcv prints them on this fit."""
    ours = str(report).splitlines()
    theirs = ref["output"]
    assert ours[0] == next(line for line in theirs if line != "")
    rank_line = next(line for line in theirs if line.startswith("Model rank"))
    assert rank_line in ours
    start = theirs.index(next(line for line in theirs if line.startswith("Basis dimension")))
    k_block = [line for line in theirs[start:] if line != ""]
    assert k_block == [line for line in ours[ours.index(theirs[start]) :] if line != ""]


def test_reml_fit_is_refused():
    rng = np.random.default_rng(4)
    x = rng.uniform(size=150)
    y = np.sin(4 * x) + rng.normal(scale=0.3, size=x.size)
    fit = hgam.gam("y ~ s(x, k=6, bs='cr')", data={"x": x, "y": y}, method="REML")
    with pytest.raises(hgam.errors.HgamNotImplementedError, match="discrete=True"):
        fit.gam_check({"x": x}, seed=1)


def test_type_other_than_deviance_is_refused(exit_fit, frame):  # noqa: F811
    with pytest.raises(hgam.errors.HgamNotImplementedError, match="pearson"):
        exit_fit.gam_check(frame, seed=1, type="pearson")
