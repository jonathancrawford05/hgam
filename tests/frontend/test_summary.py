"""``GamFit.summary()`` — preview Slice P4 (ADR-253).

Closed-form checks (a Gaussian null deviance is the weighted TSS; a Poisson intercept with an
offset has the closed-form MLE ``log(sum y / sum(w exp(off)))``). The ``summary.gam``
comparison itself stayed in polaris-re; these are own-criterion checks, not parity.
"""

import numpy as np

from hgam.engine.gam_family import gaussian_identity, poisson_log
from hgam.frontend import gam
from hgam.frontend.summary import EPSILON_REL, null_deviance


def _frame(n: int = 300, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return {
        "x": x,
        "f": rng.choice(["a", "b", "c"], n),
        "y": np.sin(x) + rng.normal(0, 0.3, n),
        "wt": rng.uniform(0.5, 2.0, n),
    }


def test_gaussian_null_deviance_is_the_weighted_total_sum_of_squares() -> None:
    y = np.array([1.0, 2.0, 4.0, 7.0])
    w = np.array([1.0, 2.0, 1.0, 3.0])
    expected = float(np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2))
    np.testing.assert_allclose(null_deviance(y, gaussian_identity(), w, None), expected, rtol=1e-12)


def test_poisson_null_deviance_with_an_offset_matches_the_closed_form_intercept() -> None:
    rng = np.random.default_rng(5)
    off = rng.normal(0.0, 0.5, 200)
    y = rng.poisson(np.exp(0.3 + off)).astype(np.float64)
    mu0 = np.exp(off) * y.sum() / np.exp(off).sum()
    expected = poisson_log().deviance(y, mu0, np.ones_like(y))
    np.testing.assert_allclose(null_deviance(y, poisson_log(), None, off), expected, rtol=1e-10)


def test_gaussian_deviance_explained_is_one_minus_rss_over_tss() -> None:
    df = _frame()
    fit = gam('y ~ s(x, k=8, bs="cr")', df, "gaussian")
    s = fit.summary()
    y = df["y"]
    rss = float(np.sum((y - fit.eta) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    np.testing.assert_allclose(s.deviance_explained, 1.0 - rss / tss, rtol=1e-10)
    np.testing.assert_allclose(s.deviance, rss, rtol=1e-10)
    assert s.n == len(df["y"])
    assert s.scale_estimated


def test_edf_total_is_the_intercept_plus_the_per_term_edf() -> None:
    fit = gam('y ~ f + s(x, k=8, bs="cr")', _frame(), "gaussian")
    s = fit.summary()
    smooth_edf = sum(t.edf for t in s.smooths)
    parametric_edf = fit.edf_per_term["f"]
    np.testing.assert_allclose(s.edf_total, 1.0 + smooth_edf + parametric_edf, rtol=1e-9)
    assert [t.label for t in s.smooths] == ["s(x)"]
    assert s.n_parametric == 3


def test_a_fixed_scale_family_reports_scale_one_and_the_newton_report() -> None:
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 10, 400)
    df = {"x": x, "y": rng.poisson(np.exp(0.5 + 0.3 * np.sin(x))).astype(float)}
    fit = gam('y ~ s(x, k=8, bs="cr")', df, "poisson")
    s = fit.summary()
    assert s.scale == 1.0
    assert not s.scale_estimated
    assert s.converged and s.outer == "newton"
    assert s.n_iterations is not None and s.n_iterations >= 1
    assert s.rel_projected_gradient is not None
    assert s.rel_projected_gradient <= s.epsilon_rel == EPSILON_REL == 1e-6


def test_select_true_reports_two_log_sp_for_a_penalised_smooth() -> None:
    fit = gam('y ~ s(x, k=8, bs="cr")', _frame(), "gaussian", select=True)
    (term,) = fit.summary().smooths
    assert len(term.log10_sp) == 2


def test_the_text_report_names_what_it_omits_and_what_it_found() -> None:
    text = str(gam('y ~ s(x, k=8, bs="cr")', _frame(), "gaussian").summary())
    assert "Deviance explained" in text and "log10(sp)" in text
    assert "converged" in text and "epsilon_rel" in text
    assert "Smooth-term tests ('—') are reported only for" in text and "Pr(>|z|)" not in text


def test_smooth_tests_are_left_out_beside_a_random_effect() -> None:
    """reTest takes recov's random-effect branch for any smooth with a bs="re" term beside it (not
    ported): those rows stay None. The re term alone has no other random smooth, so it is tested."""
    import hgam

    rng = np.random.default_rng(23)
    n = 2000
    g = np.array(list("abcdefgh"))[rng.integers(0, 8, n)]
    x = rng.uniform(0, 1, n)
    effect = dict(zip("abcdefgh", rng.normal(0, 0.3, 8), strict=True))
    y = rng.poisson(np.exp(1.0 + np.sin(2 * np.pi * x) + np.array([effect[v] for v in g]))).astype(float)
    fit = hgam.bam(
        'y ~ s(x, bs="cr", k=8) + s(g, bs="re")',
        data={"y": y, "x": x, "g": g},
        family="poisson",
        method="fREML",
        discrete=True,
        select=True,
    )
    rows = {t.label: t for t in fit.summary().smooths}
    assert rows["s(x)"].null_space_dim == 0
    assert rows["s(x)"].chi_sq is None and rows["s(x)"].ref_df is None
    assert rows["s(g)"].chi_sq is not None and rows["s(g)"].ref_df is not None
