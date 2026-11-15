"""``hgam.frontend.gam``, the formula front end (moved from polaris-re's ``tests/test_gam/test_api.py``).

Own-criterion checks, not parity with mgcv: the front end reproduces the engine it wraps, and each
refusal fires. Data are dicts of NumPy arrays; polaris-re used polars DataFrames."""

import numpy as np
import pytest

from hgam.engine.gam_model import fit_engine_gam
from hgam.engine.gam_term_spec import ModelSpec, TermSpec
from hgam.errors import HgamRankDeficiencyWarning, HgamValidationError
from hgam.frontend import Factor, GamFit, gam, r_factor_levels, structural_rank_deficiency
from hgam.frontend import api as gam_api

Frame = dict[str, np.ndarray]


def _head(frame: Frame, n: int) -> Frame:
    return {k: v[:n] for k, v in frame.items()}


def _with(frame: Frame, **cols: object) -> Frame:
    return {**frame, **cols}


@pytest.fixture(scope="module")
def frame() -> Frame:
    rng = np.random.default_rng(7)
    n = 260
    x = rng.uniform(0, 10, n)
    z = rng.uniform(0, 5, n)
    g = rng.choice(["b", "a", "c"], n)
    off = rng.uniform(-0.2, 0.2, n)
    y = np.sin(x) + 0.1 * z + np.where(g == "b", 0.4, 0.0) + rng.normal(0, 0.3, n)
    h = np.random.default_rng(8).choice(["p", "q"], n)
    return {"x": x, "z": z, "g": g, "h": h, "off": off, "y": y, "wt": rng.uniform(1, 2, n)}


def test_facade_reproduces_the_engine_it_wraps(frame: Frame) -> None:
    fit = gam('y ~ g + s(x, bs="cr", k=8)', frame, "gaussian")
    levels = r_factor_levels(list(frame["g"]))
    codes = np.array([levels.index(v) for v in frame["g"]], dtype=np.int64)
    model = ModelSpec(
        family="gaussian",
        link="identity",
        terms=(
            TermSpec(label="g", variables=("g",), basis="parametric", levels=(3,)),
            TermSpec(label="s(x)", variables=("x",), basis="cr", k=(8,)),
        ),
    )
    ref = fit_engine_gam(model, {"g": codes, "x": frame["x"]}, frame["y"])
    np.testing.assert_allclose(fit.eta, ref.eta, rtol=0, atol=1e-12)
    assert isinstance(fit, GamFit)
    assert fit.converged


def test_structure_matches_mgcv_conventions(frame: Frame) -> None:
    fit = gam(
        'y ~ g + s(x, bs="cr", k=8) + s(x, by=off, bs="cr", k=6) + ti(x, z, bs="cr", k=c(5, 4))',
        frame,
        "gaussian",
    )
    assert fit.smooth_labels == ("s(x)", "s(x):off", "ti(x,z)")
    assert fit.smooth_bs_dim == (8, 6, 20)
    assert fit.n_parametric == 3  # intercept + two dummy columns for the 3-level factor
    assert fit.factor_levels == {"g": ("a", "b", "c")}


def test_factor_by_labels_and_term_multiplier(frame: Frame) -> None:
    fit = gam('y ~ g + s(x, by=g, bs="cr", k=6)', frame, "gaussian")
    assert fit.smooth_labels == ("s(x):ga", "s(x):gb", "s(x):gc")
    assert len(fit.log_lambda) == 3  # one smoothing parameter per level


def test_predict_encodes_new_data_and_matches_codes(frame: Frame) -> None:
    fit = gam('y ~ g + s(x, bs="cr", k=8)', frame, "gaussian")
    new = _head(frame, 11)
    eta = fit.predict(new)
    np.testing.assert_allclose(eta, fit.eta[:11], rtol=0, atol=1e-10)
    mu, se = fit.predict(new, "response", se_fit=True)
    assert mu.shape == se.shape == (11,)
    assert np.all(se > 0)
    codes = {"g": np.array([("a", "b", "c").index(v) for v in new["g"]]), "x": new["x"]}
    np.testing.assert_allclose(fit.predict(codes, encoded=True), eta, rtol=0, atol=1e-12)


def test_unseen_level_in_newdata_is_refused(frame: Frame) -> None:
    fit = gam('y ~ g + s(x, bs="cr", k=8)', frame, "gaussian")
    bad = _with(_head(frame, 3), g=np.array(["zzz"] * 3))
    with pytest.raises(HgamValidationError, match="level 'zzz'"):
        fit.predict(bad)


def test_offset_in_formula_and_as_argument_agree(frame: Frame) -> None:
    a = gam('y ~ s(x, bs="cr", k=6) + offset(off)', frame, "gaussian")
    b = gam('y ~ s(x, bs="cr", k=6)', frame, "gaussian", offset="off")
    np.testing.assert_allclose(a.eta, b.eta, rtol=0, atol=1e-12)
    with pytest.raises(HgamValidationError, match="both offset="):
        gam('y ~ s(x, bs="cr", k=6) + offset(off)', frame, "gaussian", offset="off")
    # predict needs the offset column, as predict.gam does
    with pytest.raises(HgamValidationError, match="no column 'off'"):
        a.predict({k: v for k, v in _head(frame, 2).items() if k != "off"})


def test_weights_column_reaches_the_fit(frame: Frame) -> None:
    plain = gam('y ~ s(x, bs="cr", k=6)', frame, "gaussian")
    weighted = gam('y ~ s(x, bs="cr", k=6)', frame, "gaussian", weights="wt")
    assert float(np.max(np.abs(plain.eta - weighted.eta))) > 1e-6


def test_the_rank_deficient_smooth_plus_factor_by_is_fitted_with_a_pivot(
    frame: Frame,
) -> None:
    """Slice 9: the unidentified coefficient is pivoted out (it was refused before)."""
    fit = gam('y ~ s(x, bs="cr", k=6) + s(x, by=g, bs="cr", k=6)', frame, "gaussian")
    assert len(fit.pivoted_columns) == 1
    assert fit.design["x"].shape[1] == fit.design["full_width"] - 1
    assert fit.vcov().shape == (fit.design["full_width"],) * 2
    # the identified spelling of the same model carries no pivot
    assert gam('y ~ g + s(x, by=g, bs="cr", k=6)', frame, "gaussian").pivoted_columns == ()


def test_structural_rank_deficiency_closed_form() -> None:
    n = 20
    t = np.linspace(0, 1, n)
    x = np.column_stack([np.ones(n), t, t])  # duplicate column, no penalty
    assert structural_rank_deficiency(x, []) == 1
    pen = np.zeros((3, 3))
    pen[2, 2] = 1.0  # penalising one copy identifies the pair
    assert structural_rank_deficiency(x, [pen]) == 0
    assert structural_rank_deficiency(np.column_stack([np.ones(n), t]), []) == 0


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"scale": 2.0}, "scale="),
    ],
)
def test_unverified_options_are_refused(frame: Frame, kwargs: dict, needle: str) -> None:
    with pytest.raises(HgamValidationError, match=needle):
        gam('y ~ s(x, bs="cr", k=6)', frame, "gaussian", **kwargs)


def test_select_is_accepted_for_cr_ti_re_and_parametric_terms(frame: Frame) -> None:
    fit = gam('y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=5)', frame, "gaussian", select=True)
    assert len(fit.log_lambda) == 4  # each cr term gains a null-space penalty
    # re's identity penalty is full rank: nothing to double; parametric has no penalty
    mixed = gam('y ~ g + s(x, bs="cr", k=6) + s(g, bs="re")', frame, "gaussian", select=True)
    assert len(mixed.log_lambda) == 3  # cr (2) + re (1)


def test_select_with_a_factor_by_smooth_is_accepted_without_a_bare_smooth(
    frame: Frame,
) -> None:
    """Slice R1 (ADR-258) lifted the old blanket refusal for the two forms that agree with mgcv."""
    for formula in ('y ~ g + s(x, by=g, bs="cr", k=6)', 'y ~ s(x, by=g, bs="cr", k=6)'):
        fit = gam(formula, frame, "gaussian", select=True)
        assert fit.converged and fit.pivoted_columns == (), formula


def test_select_with_a_bare_smooth_beside_a_factor_by_smooth_is_refused_by_name(
    frame: Frame,
) -> None:
    for formula in (
        'y ~ s(x, bs="cr", k=6) + s(x, by=g, bs="cr", k=6)',
        'y ~ g + s(x, bs="cr", k=6) + s(x, by=g, bs="cr", k=6)',
        # measured disagreements (second by smooth) and the ti form (one miss in 60) are refused
        'y ~ g + h + s(x, by=g, bs="cr", k=6) + s(x, by=h, bs="cr", k=6)',
        'y ~ g + s(x, by=g, bs="cr", k=6) + ti(x, z, bs="cr", k=c(5, 4))',
    ):
        with pytest.raises(HgamValidationError, match="another smooth of 'x'"):
            gam(formula, frame, "gaussian", select=True)


def test_a_numeric_by_beside_a_factor_by_is_accepted_under_select(frame: Frame) -> None:
    """Measured at 60 of 60 draws agreeing with mgcv (ADR-258), so it is not refused."""
    fit = gam(
        'y ~ g + s(x, by=g, bs="cr", k=6) + s(x, by=z, bs="cr", k=6)',
        frame,
        "gaussian",
        select=True,
    )
    assert fit.converged and fit.pivoted_columns == ()


def test_the_bare_and_factor_by_refusal_is_narrow(frame: Frame) -> None:
    """Without select the bare form is still fitted (pivoted, Slice 9); a bare smooth of a
    DIFFERENT covariate, or beside a numeric-by smooth, is not caught by the refusal."""
    bare = 'y ~ s(x, bs="cr", k=6) + s(x, by=g, bs="cr", k=6)'
    assert gam(bare, frame, "gaussian").converged
    other = gam('y ~ s(z, bs="cr", k=5) + s(x, by=g, bs="cr", k=6)', frame, "gaussian", select=True)
    assert other.converged
    numeric_by = gam('y ~ s(x, bs="cr", k=6) + s(x, by=z, bs="cr", k=6)', frame, "gaussian", select=True)
    assert numeric_by.converged


@pytest.mark.parametrize(
    "family",
    ["Gamma", "nb", "tweedie", "binomial(link='probit')", "poisson(link='sqrt')", "gaussian(x=1)"],
)
def test_unsupported_families_are_refused(frame: Frame, family: str) -> None:
    with pytest.raises(HgamValidationError):
        gam('y ~ s(x, bs="cr", k=6)', frame, family)


def test_dtype_and_null_guards(frame: Frame) -> None:
    with pytest.raises(HgamValidationError, match="numeric column"):
        gam("y ~ x", frame, "gaussian")
    with pytest.raises(HgamValidationError, match="as a number"):
        gam('y ~ s(g, bs="cr", k=4)', frame, "gaussian")
    with pytest.raises(HgamValidationError, match="factors must be"):
        gam('y ~ s(x, bs="re")', frame, "gaussian")
    with pytest.raises(HgamValidationError, match="nulls"):
        gam(
            'y ~ s(x, bs="cr", k=6)',
            _with(frame, x=np.where(frame["x"] > 9, np.nan, frame["x"])),
            "gaussian",
        )
    with pytest.raises(HgamValidationError, match="no column 'nope'"):
        gam('y ~ s(nope, bs="cr", k=6)', frame, "gaussian")
    with pytest.raises(HgamValidationError, match="mapping from column name"):
        gam('y ~ s(x, bs="cr", k=6)', [1.0], "gaussian")  # type: ignore[arg-type]
    with pytest.raises(HgamValidationError, match="main effects"):
        gam('y ~ g:g2 + s(x, bs="cr", k=6)', _with(frame, g2=frame["g"]), "gaussian")


def test_factor_levels_follow_r_rules(frame: Frame) -> None:
    mixed = ["b", "B", "a", "A", "10", "9", "_z", "Z"]
    assert r_factor_levels(mixed, "C") == ("10", "9", "A", "B", "Z", "_z", "a", "b")
    assert r_factor_levels(mixed, "en_US") == (
        "_z",
        "10",
        "9",
        "a",
        "A",
        "b",
        "B",
        "Z",
    )  # tier-3 order
    assert r_factor_levels(mixed) == r_factor_levels(mixed, gam_api.ORACLE_COLLATION)
    with pytest.raises(HgamValidationError, match="Factor"):
        r_factor_levels(["é", "e"], "en_US")
    fit = gam(
        'y ~ g + s(x, bs="cr", k=6)',
        _with(frame, g=Factor(["z", "a", "m"] * 86 + ["z", "a"], levels=["z", "a", "m"])),
        "gaussian",
    )
    assert fit.factor_levels["g"] == ("z", "a", "m")  # a declared Factor order is kept
    with pytest.raises(HgamValidationError, match="at least 2"):
        gam('y ~ g + s(x, bs="cr", k=6)', _with(frame, g=np.array(["only"] * len(frame["x"]))), "gaussian")
    with pytest.raises(HgamValidationError, match="numeric column"):
        gam('y ~ g + s(x, bs="cr", k=6)', _with(frame, g=frame["x"] > 5), "gaussian")


def test_family_with_link_and_fitted_binomial_cloglog(frame: Frame) -> None:
    rng = np.random.default_rng(3)
    d = _with(frame, y=rng.binomial(1, 1 / (1 + np.exp(-np.sin(frame["x"])))))
    fit = gam('y ~ s(x, bs="cr", k=6)', d, 'binomial(link="cloglog")')
    assert fit.model.link == "cloglog"
    assert gam('y ~ s(x, bs="cr", k=6)', d, "binomial").model.link == "logit"


def test_the_rank_deficiency_warning_points_at_the_gam_caller(frame: Frame) -> None:
    """ADR-256 amendment 1: through the public entry point the warning is attributed to the
    user's line, not to library code."""
    with pytest.warns(HgamRankDeficiencyWarning) as rec:
        gam('y ~ s(x, bs="cr", k=6) + s(x, by=g, bs="cr", k=6)', frame, "gaussian")
    assert rec[0].filename == __file__


# --- Slice R4 (ADR-261): a bare s(x) is thin-plate, as in mgcv ---------------------------------


def test_a_bare_smooth_is_the_thin_plate_smooth_with_default_k(frame: Frame) -> None:
    bare = gam("y ~ s(x)", frame, "gaussian")
    explicit = gam('y ~ s(x, bs="tp", k=10)', frame, "gaussian")
    assert bare.smooth_labels == ("s(x)",)
    assert bare.smooth_bs_dim == (10,)
    np.testing.assert_allclose(bare.eta, explicit.eta, rtol=0, atol=1e-12)
    assert bare.converged


def test_the_thin_plate_fit_has_a_linear_trend_unpenalised_and_a_sensible_edf(
    frame: Frame,
) -> None:
    fit = gam("y ~ s(x, k=8) + s(z, bs='tp', k=6)", frame, "gaussian")
    assert fit.smooth_bs_dim == (8, 6)
    assert 2.0 < fit.edf_total < 14.0
    assert fit.converged


def test_a_thin_plate_smooth_is_predictable_at_new_rows(frame: Frame) -> None:
    fit = gam("y ~ s(x) + s(z, k=6)", frame, "gaussian")
    at_train = fit.predict({"x": frame["x"], "z": frame["z"]}, "link")
    np.testing.assert_allclose(at_train, fit.eta, rtol=0, atol=1e-9)
    new = {"x": np.array([1.5, 4.0, 11.0]), "z": np.array([0.5, 2.5, 6.0])}
    out = fit.predict(new, "link")
    assert out.shape == (3,) and np.all(np.isfinite(out))


def test_thin_plate_beside_cr_and_ti_and_a_factor_fits(frame: Frame) -> None:
    fit = gam('y ~ g + s(x) + s(z, bs="cr", k=6) + ti(x, z, bs="cr", k=c(5, 4))', frame, "gaussian")
    assert fit.smooth_labels == ("s(x)", "s(z)", "ti(x,z)")
    assert fit.converged


def test_select_true_with_thin_plate_smooths_fits_and_shrinks_a_noise_smooth(
    frame: Frame,
) -> None:
    """ADR-261 amendment 1: the tp null space after the constraint is one linear direction, so the
    null-space penalty is gauge-invariant up to a scalar and select=TRUE is accepted."""
    plain = gam("y ~ s(x) + s(z, k=6)", frame, "gaussian")
    shrunk = gam("y ~ s(x) + s(z, k=6)", frame, "gaussian", select=True)
    assert shrunk.converged
    assert shrunk.edf_total <= plain.edf_total + 1e-6


def test_the_discrete_bam_path_refuses_a_thin_plate_smooth(frame: Frame) -> None:
    counts = _with(frame, c=np.round(np.abs(frame["y"]) * 3).astype(np.int64))
    with pytest.raises(HgamValidationError, match=r"tp terms"):
        gam("c ~ s(x)", counts, "poisson", method="fREML", discrete=True)


def test_a_thin_plate_smooth_of_a_factor_column_is_refused(frame: Frame) -> None:
    with pytest.raises(HgamValidationError, match="factor column"):
        gam("y ~ s(g)", frame, "gaussian")


def test_select_true_refuses_a_thin_plate_smooth_beside_a_factor_by_smooth(
    frame: Frame,
) -> None:
    """ADR-261 amendment 2: measured (tier 1) 14 of 120 draws disagree, the ADR-258 plateau."""
    for formula in (
        'y ~ s(x) + s(x, by=g, bs="cr", k=6)',
        'y ~ g + s(x) + s(x, by=g, bs="cr", k=6)',
    ):
        with pytest.raises(HgamValidationError, match="thin-plate s"):
            gam(formula, frame, "gaussian", select=True)
    # a tp smooth of a DIFFERENT covariate is fine
    assert gam('y ~ s(z) + s(x, by=g, bs="cr", k=6)', frame, "gaussian", select=True).converged
