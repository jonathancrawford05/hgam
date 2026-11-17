"""The rows ``bam(discrete=TRUE)`` takes each sum-to-zero constraint over (section 17).

``discrete.mf`` (bam.r) compresses and shuffles each covariate under ``temp.seed(8547)`` and pads every
column to the longest with ``sample()``; ``gam.setup`` builds the smooths on that padded frame, so
``smoothCon``'s ``colMeans(X)`` is over its rows. Held here: R's stream (``RRandom``) to mgcv's own draws
exactly; the padded frame of the exit fit (section 14) and the constraint directions of every
constrained cr smooth and ti margin to the deterministic-pieces gate (1e-8 relative). Fixtures
``r_rng_set_seed_8547`` and ``bam_discrete_constraint_rows``, tier 1 locally, tier 3 in CI."""

import json
from pathlib import Path

import numpy as np
import pytest

import hgam.frontend.api as api
from hgam.engine.gam_basis_cr import cr_basis
from hgam.frontend.discrete import discrete_model_frame
from hgam.frontend.r_rng import RRandom

FIXTURES = Path(__file__).parent / "fixtures"
GATE = 1e-8


def _load(name: str) -> dict:
    with (FIXTURES / f"{name}.json").open() as fh:
        return json.load(fh)


def test_r_stream_reproduces_set_seed_8547() -> None:
    ref = _load("r_rng_set_seed_8547")
    r = RRandom(ref["seed"])
    np.testing.assert_allclose([r.unif_rand() for _ in range(5)], ref["unif"], rtol=1e-15, atol=0)
    assert r.sample_int(981, 981).tolist() == ref["perm_981"]
    assert r.sample_int(985, 50, replace=True).tolist() == ref["rep_985_50"]
    r = RRandom(ref["seed"])
    mixed = ref["mixed"]
    assert r.sample_int(2, 2).tolist() == mixed["perm_2"]
    assert r.sample_int(70000, 5, replace=True).tolist() == mixed["rep_70000_5"]
    assert r.sample_int(1, 3, replace=True).tolist() == mixed["rep_1_3"]


@pytest.fixture(scope="module")
def captured() -> dict:
    """The model, raw data and response the exit fit hands ``fit_engine_bam``."""
    import test_summary_bam as summary_tests

    out: dict = {}
    original = api.fit_engine_bam

    def spy(model, data, y):
        out.update(model=model, data=data, y=y)
        out["fit"] = original(model, data, y)
        return out["fit"]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(api, "fit_engine_bam", spy)
        summary_tests.exit_fit.__wrapped__(_load("bam_discrete_exit_summary"))
    return out


@pytest.fixture(scope="module")
def ref() -> dict:
    return _load("bam_discrete_constraint_rows")


@pytest.fixture(scope="module")
def frame(captured) -> dict[str, np.ndarray]:
    return discrete_model_frame(captured["model"], captured["data"], captured["y"])


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        ("AttdAge", "AttdAge"),
        ("PolYear", "PolYear"),
        ("StudyYear_C", "StudyYear_C"),
        ("(response)", "p"),
        ("ExposCnt", "weights"),
        ("FaceSize", "FaceSize"),
        ("Smoke", "Smoke"),
    ],
)
def test_padded_frame_is_mgcvs_row_for_row(frame, ref, ours, theirs) -> None:
    assert len(frame) == len(ref["columns"])
    got = frame[ours]
    want = np.asarray(ref[theirs], dtype=np.float64)
    assert got.shape == (ref["maxr"],)
    np.testing.assert_allclose(got, want, rtol=GATE, atol=0)


@pytest.mark.parametrize(
    ("variable", "key"),
    [
        ("AttdAge", "s_AttdAge"),
        ("PolYear", "s_PolYear"),
        ("PolYear", "ti_PolYear"),
        ("AttdAge", "ti_AttdAge"),
    ],
)
def test_constraint_direction_meets_the_deterministic_gate(frame, captured, ref, variable, key) -> None:
    term = next(
        t for t in captured["model"].terms if t.basis == "cr" and t.variables == (variable,) and t.by is None
    )
    knots = np.asarray(term.knots_by_variable()[variable], dtype=np.float64)
    c = cr_basis(frame[variable], knots)[0].mean(axis=0)
    c = c / np.linalg.norm(c)
    want = np.asarray(ref["constraint"][key], dtype=np.float64)
    sign = np.sign(c @ want)
    np.testing.assert_allclose(sign * c, want, rtol=GATE, atol=GATE * np.max(np.abs(want)))


@pytest.mark.parametrize(
    ("label", "margin", "key"),
    [
        ("s(AttdAge)", 0, "s_AttdAge"),
        ("s(PolYear)", 0, "s_PolYear"),
        ("ti(PolYear,AttdAge)", 0, "ti_PolYear"),
        ("ti(PolYear,AttdAge)", 1, "ti_AttdAge"),
    ],
)
def test_fitted_null_space_is_orthogonal_to_mgcvs_constraint(captured, ref, label, margin, key) -> None:
    """The fit's absorbed constraint (the term state ``Z``, used for the design and prediction)
    satisfies mgcv's: ``C Z = 0`` for mgcv's unit ``C``. One row per distinct value missed by
    3.0e-3 (AttdAge), the data rows by 2.7e-2."""
    state = next(s for s in captured["fit"].term_states if s.term.label == label)
    want = np.asarray(ref["constraint"][key], dtype=np.float64)
    assert ref["margin_terms"] == ["PolYear", "AttdAge"]
    assert np.max(np.abs(want @ state.constraint[margin])) < GATE


def test_an_offset_beside_prior_weights_is_refused_in_a_discrete_fit() -> None:
    from hgam.errors import HgamNotImplementedError
    from hgam.frontend.api import gam

    rng = np.random.default_rng(17)
    n = 300
    data = {
        "x": rng.uniform(0, 1, n),
        "w": rng.integers(20, 200, n).astype(np.float64),
        "lo": rng.normal(0, 0.1, n),
    }
    data["p"] = rng.binomial(data["w"].astype(int), 0.05) / data["w"]
    with pytest.raises(HgamNotImplementedError, match="offset together with prior weights"):
        gam(
            'p ~ s(x, bs="cr", k=6) + offset(lo)',
            data,
            'binomial(link="cloglog")',
            weights="w",
            method="fREML",
            discrete=True,
        )
