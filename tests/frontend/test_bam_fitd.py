"""``hgam.frontend.bam_fitd`` / ``bam_fit`` (moved from polaris-re's ``tests/test_gam/test_bam_fitd.py``).

Derivative formulas against finite differences of an independently coded criterion, and the
supported subset and its refusals. The R-gated probe and the provenance declarations stayed in
polaris-re with its conformance module."""

import dataclasses

import numpy as np
import pytest

from hgam.engine.gam_family import gaussian_identity, poisson_log, quasipoisson_log
from hgam.engine.gam_initial_sp import (
    initial_log10_lambda,
    initial_log10_lambda_from_xx_diag,
)
from hgam.errors import HgamNotImplementedError, HgamValidationError
from hgam.frontend import gam
from hgam.frontend.api import _tero
from hgam.frontend.bam_fitd import (
    _fit_chol,
    bam_fitd,
    penalty_block_structure,
)
from hgam.frontend.formula import TensorCall


def _two_term_cell(seed: int = 0, n: int = 600) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Truncated-line spline design (an identified one: 1, t, u unpenalised; 6 ridge-penalised
    truncated lines per covariate) with two penalties, each of rank 6."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(size=n)
    u = rng.uniform(size=n)
    knots = np.linspace(0.1, 0.9, 6)
    x = np.hstack(
        [
            np.ones((n, 1)),
            t[:, None],
            u[:, None],
            np.maximum(0.0, t[:, None] - knots[None, :]),
            np.maximum(0.0, u[:, None] - knots[None, :]),
        ]
    )
    p = x.shape[1]
    s1 = np.zeros((p, p))
    s1[3:9, 3:9] = np.eye(6)
    s2 = np.zeros((p, p))
    s2[9:15, 9:15] = np.eye(6)
    mu = 3.0 * np.exp(0.8 * np.sin(2 * np.pi * t) - 0.5 * u)
    return x, (s1, s2), rng.poisson(mu).astype(np.float64)


class TestFixedWeightCriterionDerivatives:
    """``_fit_chol``'s gradient/Hessian against finite differences of the criterion coded
    independently in this test (closed-form verification of the derivative formulas)."""

    @staticmethod
    def _criterion(
        xx: np.ndarray,
        f: np.ndarray,
        yy: float,
        blocks: tuple[np.ndarray, ...],
        rho: np.ndarray,
        ranks: tuple[int, ...],
    ) -> float:
        lam = np.exp(rho)
        s_lam = sum(lj * s for lj, s in zip(lam, blocks, strict=True))
        a = xx + s_lam
        beta = np.linalg.solve(a, f)
        rss_plus_bsb = yy - beta @ f
        ldet_a = np.linalg.slogdet(a)[1]
        ldet_s = 0.0
        for lj, s, rk in zip(lam, blocks, ranks, strict=True):
            ev = np.sort(np.linalg.eigvalsh(lj * s))[::-1][:rk]
            ldet_s += float(np.sum(np.log(ev)))
        return float((rss_plus_bsb + ldet_a - ldet_s) / 2.0)

    def test_gradient_and_hessian_match_finite_differences(self) -> None:
        x, blocks, y = _two_term_cell()
        w = y + 0.1
        z = np.log(y + 0.1) + (y - (y + 0.1)) / (y + 0.1)
        xx = (x * w[:, None]).T @ x
        f = x.T @ (w * z)
        yy = float(np.sum(w * z * z))
        st = penalty_block_structure(blocks)
        rho = np.array([1.0, 2.0])
        fc = _fit_chol(xx, f, blocks, st, rho, tol=0.0)
        ranks = (6, 6)
        h = 1.0e-4
        grad_fd = np.zeros(2)
        hess_fd = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            grad_fd[j] = (
                self._criterion(xx, f, yy, blocks, rho + e, ranks)
                - self._criterion(xx, f, yy, blocks, rho - e, ranks)
            ) / (2 * h)
        for j in range(2):
            for k in range(2):
                ej, ek = np.zeros(2), np.zeros(2)
                ej[j], ek[k] = h, h
                hess_fd[j, k] = (
                    self._criterion(xx, f, yy, blocks, rho + ej + ek, ranks)
                    - self._criterion(xx, f, yy, blocks, rho + ej - ek, ranks)
                    - self._criterion(xx, f, yy, blocks, rho - ej + ek, ranks)
                    + self._criterion(xx, f, yy, blocks, rho - ej - ek, ranks)
                ) / (4 * h * h)
        np.testing.assert_allclose(fc.grad, grad_fd, rtol=1e-6, atol=1e-7)
        # the Newton step solves hess @ step = -grad (before the eigenvalue floor/cap)
        np.testing.assert_allclose(hess_fd, hess_fd.T, rtol=1e-4)
        assert np.all(np.isfinite(fc.step))

    def test_overlapping_penalties_form_one_block_with_the_rank_of_their_sum(self) -> None:
        d2 = np.diff(np.eye(4), n=2, axis=0)
        a = np.zeros((6, 6))
        a[1:5, 1:5] = d2.T @ d2
        b = np.zeros((6, 6))
        b[1:5, 1:5] = np.eye(4)
        c = np.zeros((6, 6))
        c[5, 5] = 1.0
        st = penalty_block_structure((a, b, c))
        assert st.groups == ((0, 1), (2,))
        assert st.ranks == (4, 1)


class TestInitialSpRefactor:
    def test_xx_diag_form_equals_the_data_form(self) -> None:
        x, blocks, y = _two_term_cell()
        fam = poisson_log()
        mu = y + 0.1
        w = mu  # poisson/log working weight
        ldxx = np.sum((np.sqrt(w)[:, None] * x) ** 2, axis=0)
        np.testing.assert_array_equal(
            initial_log10_lambda(y, x, fam, blocks),
            initial_log10_lambda_from_xx_diag(ldxx, blocks),
        )


class TestBamFitd:
    def test_converges_and_is_stationary_for_the_penalised_working_model(self) -> None:
        x, blocks, y = _two_term_cell()
        res = bam_fitd(x, blocks, y, poisson_log())
        assert res.converged
        assert res.n_iter <= 40
        lam = np.exp(res.log_sp)
        s_lam = sum(lj * s for lj, s in zip(lam, blocks, strict=True))
        mu = res.mu
        w = mu
        z = (res.eta) + (y - mu) / mu
        score = x.T @ (w * (z - x @ res.coef)) - s_lam @ res.coef
        # stationarity of the penalised log-likelihood at the converged iterate
        assert np.max(np.abs(score)) / np.max(np.abs(x.T @ (w * z))) < 1e-4

    def test_edf_is_the_trace_of_the_hat_matrix_and_below_the_column_count(self) -> None:
        x, blocks, y = _two_term_cell()
        res = bam_fitd(x, blocks, y, poisson_log())
        assert 2.0 < float(np.sum(res.edf)) < x.shape[1]
        assert np.all(res.edf > -1e-8)

    def test_trace_records_every_iteration(self) -> None:
        x, blocks, y = _two_term_cell()
        res = bam_fitd(x, blocks, y, poisson_log())
        # the converging iteration stops at the TOP of its loop, before a smoothing step
        assert len(res.trace) == res.n_iter - 1
        assert [t.iteration for t in res.trace] == list(range(1, res.n_iter))

    @pytest.mark.parametrize("family", [gaussian_identity(), quasipoisson_log()])
    def test_unsupported_families_are_refused_by_name(self, family) -> None:  # type: ignore[no-untyped-def]
        x, blocks, y = _two_term_cell()
        with pytest.raises(HgamNotImplementedError, match=r"scale|supported"):
            bam_fitd(x, blocks, y, family)

    def test_no_penalty_is_refused(self) -> None:
        x, _, y = _two_term_cell()
        with pytest.raises(HgamValidationError, match="no penalty"):
            bam_fitd(x, (), y, poisson_log())

    def test_the_producer_signature_takes_a_design_and_no_mgcv_fit(self) -> None:
        import inspect

        params = set(inspect.signature(bam_fitd).parameters)
        assert params == {
            "x",
            "penalty_blocks",
            "y",
            "family",
            "offset",
            "prior_weights",
            "maxit",
            "epsilon",
            "structure",
        }


def _frame(n_rep: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    age = np.repeat(np.arange(40, 70), 10 * n_rep)
    dur = np.tile(np.repeat(np.arange(1, 11), n_rep), 30)
    expo = rng.uniform(5, 40, age.size)
    lin = -5.5 + 0.05 * (age - 40) - 0.04 * dur
    y = rng.poisson(expo * np.exp(lin))
    return {"age": age, "dur": dur, "expo": expo, "y": y, "logexpo": np.log(expo)}


FORMULA = 'y ~ s(age, bs="cr", k=8) + s(dur, bs="cr", k=5) + offset(logexpo)'


class TestGamEntryPoint:
    def test_bam_pair_runs_and_is_labelled(self) -> None:
        df = _frame()
        fit = gam(FORMULA, df, "poisson", method="fREML", discrete=True)
        assert fit.fit_method == "fREML-discrete"
        assert fit.converged
        assert fit.n_iterations is not None
        ref = gam(FORMULA, df, "poisson")
        # a different algorithm, not a different model: close, and not asserted identical
        assert float(np.max(np.abs(fit.eta - ref.eta))) < 5e-2

    @pytest.mark.parametrize(("method", "discrete"), [("fREML", False), ("REML", True), ("GCV", False)])
    def test_other_pairs_are_refused_by_name(self, method: str, discrete: bool) -> None:
        with pytest.raises(HgamNotImplementedError, match="not a supported pair"):
            gam(FORMULA, _frame(), "poisson", method=method, discrete=discrete)

    def test_unconditional_covariance_is_refused(self) -> None:
        # summary() of a discrete fit is ported (tests/test_summary_bam.py, section 16).
        fit = gam(FORMULA, _frame(), "poisson", method="fREML", discrete=True)
        with pytest.raises(HgamNotImplementedError, match="unconditional"):
            fit.vcov(unconditional=True)
        _, se = fit.predict(_frame(), se_fit=True)
        assert np.all(np.isfinite(se)) and np.all(se > 0)

    def test_gaussian_is_refused_by_name(self) -> None:
        df = {**_frame(), "y": _frame()["y"].astype(np.float64)}
        with pytest.raises(HgamNotImplementedError, match="not supported"):
            gam('y ~ s(age, bs="cr", k=8)', df, "gaussian", method="fREML", discrete=True)


class TestConstraintFrame:
    """``constraint_frame``: the sum-to-zero constraint over a supplied frame's rows."""

    def test_default_is_data_rows_and_the_context_switches_to_the_frame(self) -> None:
        from hgam.engine.gam_basis_cr import (
            constraint_frame,
            constraint_rows,
            cr_basis,
            cr_default_knots,
            sum_to_zero_null_space,
        )
        from hgam.errors import HgamNotImplementedError

        x = np.concatenate(
            [np.full(50, 1.0), np.full(5, 2.0), np.full(5, 3.0), np.full(5, 4.0), np.full(5, 5.0)]
        )
        knots = cr_default_knots(x, 4)
        design = cr_basis(x, knots)[0]
        assert constraint_rows("x", knots) is None
        z_data = sum_to_zero_null_space(design, constraint_rows("x", knots))
        frame_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
        with constraint_frame({"x": frame_x}):
            z_frame = sum_to_zero_null_space(design, constraint_rows("x", knots))
            with pytest.raises(HgamNotImplementedError, match="no column 'w'"):
                constraint_rows("w", knots)
        assert np.allclose(design.mean(axis=0) @ z_data, 0.0, atol=1e-12)
        assert np.allclose(cr_basis(frame_x, knots)[0].mean(axis=0) @ z_frame, 0.0, atol=1e-12)
        assert not np.allclose(z_data, z_frame)  # unbalanced data: the two constraints differ
        # the switch is scoped: outside the context the default is restored
        assert constraint_rows("x", knots) is None

    def test_bam_fit_predicts_its_own_training_rows_in_the_same_basis(self) -> None:
        df = _frame()
        fit = gam(FORMULA, df, "poisson", method="fREML", discrete=True)
        pred = fit.predict(df, "link")
        np.testing.assert_allclose(pred, fit.eta, atol=1e-10)


class TestBamVpUsesFisherWeights:
    """``bgam.fitd``'s ``Vp = PP * scale`` uses the expected (Fisher) working weights. For a
    non-canonical link (cloglog) these differ from the observed-information weights the exact-REML
    path rebuilds, so the bam path must serve ITS OWN ``Vp``."""

    def _binomial_frame(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(11)
        age = np.repeat(np.arange(40, 80), 6).astype(float)
        n = rng.integers(20, 200, age.size).astype(float)
        p = 1.0 - np.exp(-np.exp(-1.0 + 0.04 * (age - 40)))
        y = rng.binomial(n.astype(int), p) / n
        return {"age": age, "n": n, "y": y}

    def test_the_served_vp_is_the_iterations_own_pp(self) -> None:
        df = self._binomial_frame()
        fit = gam(
            'y ~ s(age, bs="cr", k=8)', df, 'binomial(link="cloglog")', weights="n",
            method="fREML", discrete=True,
        )  # fmt: skip
        assert fit.vp_override is not None
        np.testing.assert_allclose(fit.vcov(), fit.vp_override)
        # and it differs from the observed-information rebuild at the same coefficients (the
        # defect amendment 17 fixed): same fit, vp_override dropped
        rebuilt = dataclasses.replace(fit, vp_override=None).vcov()
        se_bam = np.sqrt(np.diag(fit.vcov()))
        se_obs = np.sqrt(np.diag(rebuilt))
        assert np.max(np.abs(se_bam / se_obs - 1.0)) > 1e-4
        reml = gam('y ~ s(age, bs="cr", k=8)', df, 'binomial(link="cloglog")', weights="n")
        assert reml.vp_override is None

    def test_a_canonical_link_fit_still_serves_its_own_vp(self) -> None:
        fit = gam(FORMULA, _frame(), "poisson", method="fREML", discrete=True)
        _, se = fit.predict(_frame(), se_fit=True)
        assert np.all(np.isfinite(se)) and np.all(se > 0)


class TestTero:
    """``mgcv:::tero``: the margin with the largest bs.dim goes last (the last one on ties)."""

    @pytest.mark.parametrize(
        ("variables", "k", "expected_vars", "expected_k"),
        [
            (("age", "dur"), (13, 6), ("dur", "age"), (6, 13)),
            (("age", "dur"), (6, 13), ("age", "dur"), (6, 13)),
            (("age", "dur"), (8, 8), ("age", "dur"), (8, 8)),
        ],
    )
    def test_swap_iff_first_margin_is_strictly_larger(
        self,
        variables: tuple[str, ...],
        k: tuple[int, ...],
        expected_vars: tuple[str, ...],
        expected_k: tuple[int, ...],
    ) -> None:
        out = _tero(TensorCall(variables=variables, k=k))
        assert (out.variables, out.k) == (expected_vars, expected_k)
        assert out.text == f"ti({','.join(expected_vars)})"

    def test_gam_discrete_labels_the_ti_term_as_bam_does_and_reml_does_not(self) -> None:
        df = _frame()
        f = 'y ~ s(age, bs="cr", k=6) + ti(age, dur, bs="cr", k=c(7,4)) + offset(logexpo)'
        assert gam(f, df, "poisson", method="fREML", discrete=True).smooth_labels[-1] == "ti(dur,age)"
        assert gam(f, df, "poisson").smooth_labels[-1] == "ti(age,dur)"
