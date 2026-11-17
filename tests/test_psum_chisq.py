"""``psum.chisq`` (mgcv.r) and Davies' algorithm (mgcv's ``davies.c``) held to mgcv (fixture
``psum_chisq_davies``, section 19 of ``oracle/make_fixtures.R``).

Gate (Q2, PR #21): the smooth-term p-value, 1e-3 absolute or 1% relative, whichever is larger. The port
follows the C line for line, so the tests also hold its path to mgcv's: ``ifault`` and the integer parts
of ``trace`` (terms, integrations, the call counter) exact, the real parts of ``trace`` at 1e-8 relative
(deterministic pieces). The p-value itself is asserted at 1e-8 here, well inside its gate.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from hgam.errors import HgamValidationError
from hgam.frontend.davies import davies, log1pmx, revsort
from hgam.frontend.smooth_tests import liu2, psum_chisq

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "psum_chisq_davies.json").read_text())
CASES = FIXTURE["cases"]


def _args(case: dict) -> dict:
    return {
        "lb": np.asarray(case["lb"], dtype=np.float64),
        "df": np.asarray(case["df"], dtype=np.float64),
        "nc": np.asarray(case["nc"], dtype=np.float64),
        "sigz": float(case["sigz"][0]),
        "nlim": int(case["nlim"][0]),
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_psum_chisq_matches_mgcv(name: str) -> None:
    case = CASES[name]
    if case["ifault"] != 0:
        with pytest.warns(RuntimeWarning):
            got = psum_chisq(case["q"][0], **_args(case))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = psum_chisq(case["q"][0], **_args(case))
    np.testing.assert_allclose(got[0], case["p"], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("name", sorted(CASES))
def test_davies_follows_mgcv_path(name: str) -> None:
    case = CASES[name]
    a = _args(case)
    res = davies(a["lb"], a["nc"], np.round(a["df"]).astype(int), a["sigz"], case["q"][0], a["nlim"], 2e-5)
    want = np.asarray(case["trace"], dtype=np.float64)
    assert res.ifault == case["ifault"]
    np.testing.assert_array_equal(np.asarray(res.trace)[[1, 2, 6]], want[[1, 2, 6]])
    np.testing.assert_allclose(np.asarray(res.trace), want, rtol=1e-8, atol=1e-12)


def test_liu2_matches_mgcv() -> None:
    f = FIXTURE["liu2"]
    got = liu2(np.asarray(f["x"]), np.asarray(f["lambda"]), h=np.asarray(f["h"]))
    np.testing.assert_allclose(got, f["p"], rtol=1e-8, atol=1e-12)


def test_revsort_orders_as_a_decreasing_heapsort() -> None:
    a = [1.0, 1.0, 0.5, 1.0, 1.0, 2.0]
    ib = list(range(6))
    revsort(a, ib)
    assert a == sorted(a, reverse=True)
    assert sorted(ib) == list(range(6)) and ib[0] == 5 and ib[-1] == 2


@pytest.mark.parametrize("x", [-0.5, -1e-3, 1e-6, 5e-3, 0.3, 0.99, 3.0])
def test_log1pmx_is_log1p_minus_x(x: float) -> None:
    # where log1p(x) - x cancels (|x| < 0.3) the reference is its series, sum_{k>=2} (-1)^(k+1) x^k / k
    want = sum((-1) ** (k + 1) * x**k / k for k in range(200, 1, -1)) if abs(x) < 0.3 else np.log1p(x) - x
    np.testing.assert_allclose(log1pmx(x), want, rtol=1e-12)


def test_psum_chisq_refuses_what_mgcv_stops_on() -> None:
    with pytest.raises(HgamValidationError, match="must match"):
        psum_chisq(1.0, np.array([1.0, 2.0]), df=np.array([1.0, 1.0, 1.0]))
    with pytest.raises(HgamValidationError, match="positive integers"):
        psum_chisq(1.0, np.array([1.0]), df=np.array([0.4]))
    with pytest.raises(HgamValidationError, match="non-zero"):
        psum_chisq(1.0, np.array([0.0, 0.0]))
