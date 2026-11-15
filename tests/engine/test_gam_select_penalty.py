"""``docs/PLAN_mgcv_parity_engine.md`` slice 7 — ``select = TRUE``'s double
penalty, Stage A.

R-free tests assert :func:`~hgam.engine.gam_select_penalty.null_space_penalty`'s
own algebraic invariants (it is a genuine orthogonal projector onto a known
null space, for both a single block and several summed together) — checkable
without ``mgcv``, the same "an oracle outage delays the measurement without
idling the session" framing ``test_gam_model.py`` uses. The real parity
check — Python's independently-computed ``S_null`` against ``mgcv``'s own
``gam(..., select=TRUE, fit=FALSE)$smooth[[i]]$S`` — is the R-gated test at
the bottom, covering every term archetype the target formula uses (``cr``,
numeric-``by`` ``cr``, ``ti``, ``sz``), at the target's own knot vectors.
"""

from pathlib import Path

import numpy as np
import pytest

from hgam.engine.gam_select_penalty import (
    null_space_penalty,
)
from hgam.errors import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]

_AGE_KNOTS = (1.0, 2, 4, 7, 14, 18, 24, 35, 50, 70, 85, 90, 95)
_YEAR_KNOTS = (1.0, 2, 3, 5, 10, 21)


def _random_orthonormal(p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    return q


def test_null_space_penalty_is_a_projector_onto_the_known_null_space() -> None:
    """A single block with a KNOWN rank deficiency: eigenvalues
    ``[5, 3, 1, 0, 0]`` in a random orthonormal basis, so the null space is
    exactly the span of the last two eigenvectors."""
    p = 5
    v = _random_orthonormal(p, seed=1)
    eigenvalues = np.array([5.0, 3.0, 1.0, 0.0, 0.0])
    s = v @ np.diag(eigenvalues) @ v.T
    s = (s + s.T) / 2.0

    result = null_space_penalty((s,))
    assert result is not None
    s_null, null_dim = result
    assert null_dim == 2
    assert s_null.shape == (p, p)

    # Symmetric.
    np.testing.assert_allclose(s_null, s_null.T, atol=1e-12)
    # Idempotent -- a genuine orthogonal projector (U0 orthonormal).
    np.testing.assert_allclose(s_null @ s_null, s_null, atol=1e-10)
    # Rank equals the null dimension.
    assert np.linalg.matrix_rank(s_null) == 2
    # Projects the known null-space eigenvectors onto themselves...
    null_vectors = v[:, 3:]
    np.testing.assert_allclose(s_null @ null_vectors, null_vectors, atol=1e-10)
    # ...and every range-space eigenvector to zero.
    range_vectors = v[:, :3]
    np.testing.assert_allclose(s_null @ range_vectors, 0.0, atol=1e-10)
    # S_null and the original S share no non-null direction (orthogonal
    # projectors onto complementary spaces).
    np.testing.assert_allclose(s_null @ s, 0.0, atol=1e-8)


def test_null_space_penalty_combines_several_blocks_before_the_null_space() -> None:
    """Two blocks whose INDIVIDUAL null spaces differ, but whose SUM has a
    known, smaller null space — the same "combine first" rule measured
    against ``mgcv`` for ``ti``/``sz`` (module docstring): the null space is
    of the blocks' unscaled SUM, not of any one block alone."""
    p = 4
    v = _random_orthonormal(p, seed=2)
    # Block 1 penalizes directions 0,1; block 2 penalizes 0,2. Their sum
    # penalizes 0,1,2, leaving only direction 3 unpenalized.
    s1 = v @ np.diag([1.0, 1.0, 0.0, 0.0]) @ v.T
    s2 = v @ np.diag([1.0, 0.0, 1.0, 0.0]) @ v.T
    s1, s2 = (s1 + s1.T) / 2.0, (s2 + s2.T) / 2.0

    result = null_space_penalty((s1, s2))
    assert result is not None
    s_null, null_dim = result
    assert null_dim == 1
    null_vector = v[:, 3:4]
    np.testing.assert_allclose(s_null @ null_vector, null_vector, atol=1e-10)
    range_vectors = v[:, :3]
    np.testing.assert_allclose(s_null @ range_vectors, 0.0, atol=1e-10)


def test_null_space_penalty_returns_none_for_a_full_rank_block() -> None:
    """A full-rank penalty has nothing left to penalize -- `select = TRUE`
    would add no extra block for a term shaped like this."""
    p = 4
    v = _random_orthonormal(p, seed=3)
    s = v @ np.diag([5.0, 3.0, 1.0, 0.5]) @ v.T
    assert null_space_penalty(((s + s.T) / 2.0,)) is None


def test_null_space_penalty_rejects_empty_blocks() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        null_space_penalty(())


def test_null_space_penalty_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValidationError, match="mismatched shapes"):
        null_space_penalty((np.eye(3), np.eye(4)))


def test_null_space_penalty_rejects_non_square_blocks() -> None:
    with pytest.raises(ValidationError, match="not square"):
        null_space_penalty((np.zeros((3, 4)),))
