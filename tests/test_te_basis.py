"""The ``te`` tensor product basis on cr margins, held to ``smoothCon`` (``make_fixtures.R`` section 26).

mgcv counterparts: ``smooth.construct.tensor.smooth.spec``, ``tensor.prod.model.matrix``,
``tensor.prod.penalties``, ``Predict.matrix.tensor.smooth`` and ``smoothCon``'s scaling and constraint
(smooth.r, mgcv 1.9-4), ported in ``hgam.engine.gam_basis_te``. The basis is a deterministic piece: structure
exact, X, S and PredictMat at 1e-8 relative. ``gam()`` still refuses ``te`` (the fit is a later slice).
"""

import numpy as np
import pytest

from hgam.engine.gam_basis_cr import _cr_basis_raw
from hgam.engine.gam_basis_te import te_basis, te_predict_design, tensor_prod_model_matrix

GATE = 1e-8

CASES = {
    "smoothcon_te_cr": (("x", "z"), (5, 4), ([1.0, 3.0, 7.0, 12.0, 21.0], None)),
    "smoothcon_te_cr_three": (("x", "z", "w"), (4, 4, 3), (None, None, None)),
}


def _column(frame: list[dict], variable: str) -> np.ndarray:
    return np.asarray([row[variable] for row in frame], dtype=np.float64)


def _rel(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_te_basis_matches_smoothcon(name, load_fixture, margin) -> None:
    fx = load_fixture(name)
    variables, ks, knots = CASES[name]
    xs = [_column(fx["data"], v) for v in variables]
    knots_in = [None if kn is None else np.asarray(kn) for kn in knots]
    got = te_basis(xs, ks, knots_in)

    ref_x = np.asarray(fx["X"])
    assert fx["label"] == f"te({','.join(variables)})"
    assert got.design.shape == ref_x.shape == (len(xs[0]), int(np.prod(ks)) - 1)
    assert fx["n_cons"] == 1
    assert got.rank == tuple(fx["rank"])
    assert got.null_space_dim == fx["null_space_dim"]
    assert len(got.penalties) == len(fx["S"]) == len(ks)

    rel_knots = max(_rel(g, np.asarray(r)) for g, r in zip(got.knots, fx["margin_knots"], strict=True))
    rel_x = _rel(got.design, ref_x)
    rel_s = max(_rel(g, np.asarray(r)) for g, r in zip(got.penalties, fx["S"], strict=True))
    new = [_column(fx["newdata"], v) for v in variables]
    rel_new = _rel(te_predict_design(new, got), np.asarray(fx["X_new"]))

    for quantity, value in (("knots", rel_knots), ("X", rel_x), ("S", rel_s), ("PredictMat", rel_new)):
        margin("te basis", f"{quantity}, max rel", value, "1e-8 rel", name)
    assert rel_knots < GATE
    assert rel_x < GATE
    assert rel_s < GATE
    assert rel_new < GATE


def test_tensor_prod_model_matrix_row_kronecker() -> None:
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(6, 3)), rng.normal(size=(6, 2)), rng.normal(size=(6, 4))
    got = tensor_prod_model_matrix([a, b, c])
    for i in range(6):
        np.testing.assert_allclose(got[i], np.kron(np.kron(a[i], b[i]), c[i]), rtol=0, atol=1e-15)


def test_te_basis_constraint_rows_are_forwarded() -> None:
    """``constraint_rows`` replaces the rows ``colMeans`` is taken over (as ``bam(discrete=TRUE)``'s ``mf0``
    will need): the data's own tensor rows reproduce the default, other rows give a constraint orthogonal to
    their own column means."""
    rng = np.random.default_rng(1)
    x, z = rng.uniform(0.0, 1.0, 120), rng.uniform(0.0, 1.0, 120)
    default = te_basis([x, z], [5, 4])
    raw = tensor_prod_model_matrix(
        [_cr_basis_raw(v, kn)[0] for v, kn in zip((x, z), default.knots, strict=True)]
    )
    same = te_basis([x, z], [5, 4], constraint_rows=raw)
    np.testing.assert_allclose(same.design, default.design, rtol=0, atol=1e-13)
    other_rows = raw[:40]
    other = te_basis([x, z], [5, 4], constraint_rows=other_rows)
    np.testing.assert_allclose(other_rows.mean(axis=0) @ other.z, 0.0, rtol=0, atol=1e-14)
    assert not np.allclose(other.design, default.design)


def test_te_basis_frame_builds_on_the_frame_and_predicts_at_the_data() -> None:
    """``frame=`` constructs the term on other rows (``bam(discrete=TRUE)``'s ``mf0``): knots, penalties and
    ``Z`` are the frame's own basis, and the design is ``te_predict_design`` at the data rows. The data as
    its own frame reproduces the default; ``frame=`` and ``constraint_rows=`` together are refused."""
    rng = np.random.default_rng(2)
    x, z = rng.uniform(0.0, 1.0, 150), rng.uniform(0.0, 1.0, 150)
    x0, z0 = rng.uniform(0.0, 1.0, 90), rng.uniform(0.0, 1.0, 90)
    default = te_basis([x, z], [5, 4])
    same = te_basis([x, z], [5, 4], frame=[x, z])
    np.testing.assert_allclose(same.design, default.design, rtol=0, atol=1e-13)
    on_frame = te_basis([x0, z0], [5, 4])
    built = te_basis([x, z], [5, 4], frame=[x0, z0])
    np.testing.assert_allclose(built.z, on_frame.z, rtol=0, atol=0)
    for got, want in zip(built.penalties, on_frame.penalties, strict=True):
        np.testing.assert_allclose(got, want, rtol=0, atol=0)
    np.testing.assert_allclose(built.design, te_predict_design([x, z], on_frame), rtol=0, atol=0)
    with pytest.raises(ValueError, match="not both"):
        te_basis([x, z], [5, 4], frame=[x0, z0], constraint_rows=np.ones((3, 20)))


def test_te_basis_refuses_one_margin_and_wrong_knots() -> None:
    x = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError, match="at least two margins"):
        te_basis([x], [5])
    with pytest.raises(ValueError, match="knots supplied"):
        te_basis([x, x[::-1]], [5, 4], [np.linspace(0.0, 1.0, 4), None])
