"""``mroot`` (mgcv.r, ``method = "chol"``) held to mgcv (fixture ``mroot_chol``, section 18): the square root
``reTest`` takes of ``Ve[m, m]`` and ``recov`` of the total penalty. Gate: deterministic pieces, 1e-8
relative; the rank (its column count) exact. ``reTest`` itself is held through ``summary()`` in
``tests/test_summary_bam.py``."""

import json

import numpy as np
import pytest
from conftest import FIXTURES  # honours HGAM_FIXTURE_DIR

from hgam.errors import HgamValidationError
from hgam.frontend.smooth_tests import mroot

CASES = json.loads((FIXTURES / "mroot_chol.json").read_text())["cases"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_mroot_matches_mgcv(name: str) -> None:
    case = CASES[name]
    a = np.asarray(case["A"], dtype=np.float64)
    want = np.asarray(case["B"], dtype=np.float64).reshape(a.shape[0], case["rank"])
    got = mroot(a)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-8 * np.max(np.abs(want)))
    np.testing.assert_allclose(got @ got.T, a, rtol=1e-8, atol=1e-8 * np.max(np.abs(a)))


def test_mroot_refuses_an_asymmetric_matrix() -> None:
    with pytest.raises(HgamValidationError, match="not symmetric"):
        mroot(np.array([[1.0, 0.5], [0.0, 1.0]]))
