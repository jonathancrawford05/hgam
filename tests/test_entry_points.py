"""``hgam.gam`` / ``hgam.bam`` (mgcv's ``gam`` and ``bam``) forward to the front end, and refuse by
name what is not ported. The ratio response ``a / b`` is held to mgcv by the exit test
(``tests/test_exit.py``, section 14)."""

import numpy as np
import pytest

import hgam
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend.formula import parse_formula


def test_a_ratio_response_parses_into_numerator_and_denominator() -> None:
    parsed = parse_formula("DthCnt / ExposCnt ~ s(x, bs='cr', k=5)")
    assert (parsed.response, parsed.response_denominator) == ("DthCnt", "ExposCnt")
    assert parse_formula("y ~ s(x, bs='cr', k=5)").response_denominator is None


@pytest.mark.parametrize("lhs", ["log(y)", "cbind(y, n - y)", "y / n / m", "y * n", "(y / n)"])
def test_other_response_expressions_are_refused(lhs: str) -> None:
    with pytest.raises(HgamNotImplementedError, match="response"):
        parse_formula(f"{lhs} ~ s(x, bs='cr', k=5)")


def test_a_zero_denominator_is_refused() -> None:
    rng = np.random.default_rng(0)
    data = {"x": rng.uniform(size=50), "y": rng.uniform(size=50), "n": np.r_[0.0, np.ones(49)]}
    with pytest.raises(HgamValidationError, match="zeros"):
        hgam.gam("y / n ~ s(x, bs='cr', k=5)", data)


def test_unported_methods_are_refused() -> None:
    data = {"x": np.linspace(0, 1, 20), "y": np.zeros(20)}
    with pytest.raises(HgamNotImplementedError, match="discrete=False"):
        hgam.bam("y ~ s(x, bs='cr', k=5)", data)
    with pytest.raises(HgamNotImplementedError, match="method='fREML'"):
        hgam.gam("y ~ s(x, bs='cr', k=5)", data, method="fREML")


def test_link_beside_a_family_with_arguments_is_refused() -> None:
    data = {"x": np.linspace(0, 1, 20), "y": np.zeros(20)}
    with pytest.raises(HgamValidationError, match="use one"):
        hgam.gam("y ~ s(x, bs='cr', k=5)", data, family='binomial(link="logit")', link="cloglog")


def test_gam_matches_the_front_end() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(size=200)
    data = {"x": x, "y": np.sin(6 * x) + rng.normal(scale=0.3, size=200)}
    from hgam.frontend.api import gam as frontend_gam

    a = hgam.gam("y ~ s(x, bs='cr', k=8)", data)
    b = frontend_gam("y ~ s(x, bs='cr', k=8)", data, "gaussian")
    np.testing.assert_allclose(a.eta, b.eta, rtol=0, atol=0)
