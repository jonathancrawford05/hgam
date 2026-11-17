"""``hgam.bam(..., nthreads=)``: bam's ``nthreads`` mapped onto the BLAS thread pool
(:mod:`hgam.frontend.threads`).

The exit fit's key results are unchanged by the cap (they meet their gates against
``bam_discrete_exit_formula`` at ``nthreads=1`` and ``nthreads=4``), the cap holds during the fit and
is restored after it, also when the fit raises, and without ``threadpoolctl`` only ``nthreads=1``
is accepted."""

import sys

import numpy as np
import pytest
import threadpoolctl

import hgam
import hgam.api as hgam_api
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.api import GamFit
from hgam.frontend.threads import blas_thread_limit, resolve_nthreads

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


def _blas_threads() -> list[int]:
    return [info["num_threads"] for info in threadpoolctl.threadpool_info() if info["user_api"] == "blas"]


def _exit_data(ref: dict) -> dict[str, np.ndarray]:
    rows = ref["data"]
    return {
        name: np.asarray([r[name] for r in rows])
        if isinstance(rows[0][name], str)
        else np.asarray([r[name] for r in rows], dtype=np.float64)
        for name in rows[0]
        if name != "p"
    }


def _exit_fit(ref: dict, nthreads: int) -> GamFit:
    return hgam.bam(
        HGAM_FORMULA,
        data=_exit_data(ref),
        family="binomial",
        link="cloglog",
        weights="ExposCnt",
        knots=ref["knots"],
        discrete=True,
        select=True,
        nthreads=nthreads,
    )


def test_exit_fit_meets_its_gates_with_and_without_the_cap(load_fixture, margin):
    ref = load_fixture("bam_discrete_exit_formula")
    capped = _exit_fit(ref, nthreads=1)
    wide = _exit_fit(ref, nthreads=4)
    edf_ref = dict(zip(ref["labels"], ref["edf"], strict=True))
    edf_ref.update(zip(ref["param_labels"], ref["edf_param"], strict=True))
    for label, fit in (("nthreads=1", capped), ("nthreads=4", wide)):
        assert fit.converged
        eta_diff = float(np.max(np.abs(fit.eta - np.asarray(ref["eta"]))))
        margin(f"exit fit, {label}", "eta, max abs", eta_diff, "1e-3 abs", "bam_discrete_exit_formula")
        assert eta_diff < 1e-3
        for term, value in edf_ref.items():
            assert abs(fit.edf_per_term[term] - value) <= max(0.05, 0.01 * value), (label, term)
    # The cap changes only BLAS's summation order, which moves where the outer search stops: the two
    # fits differ by 3.3e-8 here (measured 2026-11-17), far inside the 1e-3 gate. Not a gate itself.
    assert np.max(np.abs(capped.eta - wide.eta)) < 1e-6


@pytest.mark.parametrize("nthreads", [1, 2])
def test_cap_holds_during_the_fit_and_is_restored(monkeypatch, nthreads):
    before = _blas_threads()
    assert before, "no BLAS thread pool found"
    seen: list[list[int]] = []
    monkeypatch.setattr(hgam_api, "_frontend_gam", lambda *a, **k: seen.append(_blas_threads()))
    hgam.bam("y ~ s(x)", {}, discrete=True, nthreads=nthreads)
    assert seen == [[nthreads] * len(before)]
    assert _blas_threads() == before


def test_cap_is_restored_when_the_fit_raises(monkeypatch):
    before = _blas_threads()

    def boom(*args, **kwargs):
        assert _blas_threads() == [1] * len(before)
        raise RuntimeError("fit failed")

    monkeypatch.setattr(hgam_api, "_frontend_gam", boom)
    with pytest.raises(RuntimeError, match="fit failed"):
        hgam.bam("y ~ s(x)", {}, discrete=True)
    assert _blas_threads() == before


def test_without_threadpoolctl_only_one_thread_is_accepted(monkeypatch):
    before = _blas_threads()
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)  # import now raises ImportError
    seen: list[list[int]] = []
    with blas_thread_limit(1):
        seen.append(_blas_threads())  # the module imported above still reads the pools
    assert seen == [before]  # BLAS left as the process set it
    with pytest.raises(HgamNotImplementedError, match=r"hgam\[threads\]"), blas_thread_limit(2):
        pass


def test_resolve_nthreads_follows_bam():
    assert resolve_nthreads(1) == 1
    assert resolve_nthreads(3) == 3
    assert resolve_nthreads(2.0) == 2
    assert resolve_nthreads(np.int64(2)) == 2
    # bam: a non-finite value or one below 1 becomes max(1, length(cluster)), 1 with no cluster
    assert resolve_nthreads(0) == 1
    assert resolve_nthreads(-2) == 1
    assert resolve_nthreads(float("nan")) == 1
    with pytest.raises(HgamNotImplementedError, match="two-element"):
        resolve_nthreads((2, 1))
    with pytest.raises(HgamNotImplementedError, match="two-element"):
        resolve_nthreads(np.array([2, 1]))
    for bad in (True, 2.5, "2"):
        with pytest.raises(HgamValidationError, match="whole number"):
            resolve_nthreads(bad)
