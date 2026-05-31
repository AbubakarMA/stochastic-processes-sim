"""
Unit tests for stochastic process models.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck


class TestBrownianMotion:

    def setup_method(self):
        self.bm = BrownianMotion(T=1.0, N=1000, n_paths=2000, seed=0)
        self.W  = self.bm.simulate()

    def test_output_shape(self):
        assert self.W.shape == (2000, 1001)

    def test_starts_at_zero(self):
        assert np.all(self.W[:, 0] == 0.0)

    def test_mean_near_zero(self):
        """E[W(t)] = 0 for all t."""
        mean_terminal = np.mean(self.W[:, -1])
        assert abs(mean_terminal) < 0.05, f"Mean too far from 0: {mean_terminal:.4f}"

    def test_variance_equals_t(self):
        """Var[W(T)] = T."""
        var_terminal = np.var(self.W[:, -1])
        assert abs(var_terminal - self.bm.T) < 0.05, f"Var={var_terminal:.4f}, expected T={self.bm.T}"

    def test_theoretical_std(self):
        std = self.bm.theoretical_std()
        assert std[0] == 0.0
        assert abs(std[-1] - np.sqrt(self.bm.T)) < 1e-10


class TestGeometricBrownianMotion:

    def setup_method(self):
        self.gbm = GeometricBrownianMotion(S0=100, mu=0.08, sigma=0.20,
                                           T=1.0, N=252, n_paths=5000, seed=0)
        self.S = self.gbm.simulate()

    def test_output_shape(self):
        assert self.S.shape == (5000, 253)

    def test_starts_at_S0(self):
        assert np.all(self.S[:, 0] == self.gbm.S0)

    def test_positive_prices(self):
        """GBM prices must always be positive."""
        assert np.all(self.S > 0), "Found non-positive prices"

    def test_mean_near_theoretical(self):
        """E[S(T)] ≈ S0 * exp(mu * T)"""
        sim_mean  = np.mean(self.S[:, -1])
        theo_mean = self.gbm.theoretical_mean()[-1]
        rel_err   = abs(sim_mean - theo_mean) / theo_mean
        assert rel_err < 0.05, f"Relative error {rel_err:.2%} too large"

    def test_log_returns_normal(self):
        """Log returns should be approximately normally distributed."""
        from scipy.stats import normaltest
        log_rets = np.diff(np.log(self.S[0]))
        _, p = normaltest(log_rets)
        assert p > 0.01, f"Log returns fail normality test (p={p:.4f})"

    def test_terminal_distribution_keys(self):
        stats = self.gbm.terminal_distribution(self.S)
        for key in ["mean", "median", "std", "5th_pct", "95th_pct", "prob_above_S0"]:
            assert key in stats

    def test_prob_above_S0_reasonable(self):
        """With positive drift, P(S(T) > S0) should be > 0.5."""
        stats = self.gbm.terminal_distribution(self.S)
        assert stats["prob_above_S0"] > 0.5


class TestOrnsteinUhlenbeck:

    def setup_method(self):
        self.ou = OrnsteinUhlenbeck(X0=0.08, theta=3.0, mu=0.05,
                                    sigma=0.02, T=2.0, N=504, n_paths=3000, seed=0)
        self.X = self.ou.simulate()

    def test_output_shape(self):
        assert self.X.shape == (3000, 505)

    def test_starts_at_X0(self):
        assert np.all(self.X[:, 0] == self.ou.X0)

    def test_mean_reverts(self):
        """
        With high theta, the mean should be close to mu by end.
        E[X(T)] = mu + (X0-mu)*exp(-theta*T) ≈ mu when theta*T >> 1
        """
        sim_mean  = np.mean(self.X[:, -1])
        theo_mean = self.ou.theoretical_mean()[-1]
        assert abs(sim_mean - theo_mean) < 0.005, \
            f"sim={sim_mean:.4f}  theory={theo_mean:.4f}"

    def test_variance_converges(self):
        """Long-run variance should approach sigma^2 / (2*theta)."""
        long_run_var = self.ou.sigma**2 / (2 * self.ou.theta)
        sim_var = np.var(self.X[:, -1])
        rel_err = abs(sim_var - long_run_var) / long_run_var
        assert rel_err < 0.1, f"Variance rel error {rel_err:.2%}"

    def test_theoretical_var_nonneg(self):
        assert np.all(self.ou.theoretical_var() >= 0)
