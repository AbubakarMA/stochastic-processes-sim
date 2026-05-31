"""
Geometric Brownian Motion (GBM)

The Black-Scholes model for stock prices. Ensures S(t) > 0 always.

SDE:  dS = μ·S·dt + σ·S·dW

Exact solution (Itô's lemma):
    S(t) = S₀ · exp( (μ - σ²/2)·t + σ·W(t) )

The term (μ - σ²/2) is the *drift adjustment* — a key result from
Itô calculus. Without it, E[S(t)] ≠ S₀·exp(μt).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeometricBrownianMotion:
    """
    Geometric Brownian Motion for asset price simulation.

    Parameters
    ----------
    S0      : float — initial asset price
    mu      : float — annualised drift (expected return)
    sigma   : float — annualised volatility
    T       : float — time horizon in years
    N       : int   — number of time steps
    n_paths : int   — number of Monte Carlo paths
    seed    : int   — random seed
    """
    S0: float = 100.0
    mu: float = 0.08        # 8% annual return
    sigma: float = 0.20     # 20% annual volatility
    T: float = 1.0
    N: int = 252
    n_paths: int = 100
    seed: Optional[int] = 42

    dt: float = field(init=False)
    t: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dt = self.T / self.N
        self.t = np.linspace(0, self.T, self.N + 1)
        if self.seed is not None:
            np.random.seed(self.seed)

    def simulate(self) -> np.ndarray:
        """
        Simulate GBM paths using the exact analytical solution.
        Avoids discretisation error present in Euler-Maruyama.

        Returns
        -------
        S : ndarray, shape (n_paths, N+1)
            Simulated asset price paths.
        """
        Z = np.random.standard_normal((self.n_paths, self.N))
        # Itô-corrected drift
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        log_returns = drift + diffusion

        S = np.zeros((self.n_paths, self.N + 1))
        S[:, 0] = self.S0
        S[:, 1:] = self.S0 * np.exp(np.cumsum(log_returns, axis=1))
        return S

    def theoretical_mean(self) -> np.ndarray:
        """E[S(t)] = S0 * exp(mu * t)"""
        return self.S0 * np.exp(self.mu * self.t)

    def theoretical_std(self) -> np.ndarray:
        """std[S(t)] = S0*exp(mu*t)*sqrt(exp(sigma^2*t)-1)"""
        mean = self.theoretical_mean()
        return mean * np.sqrt(np.exp(self.sigma**2 * self.t) - 1)

    def terminal_distribution(self, S: np.ndarray) -> dict:
        """Summary statistics of terminal prices S(T)."""
        terminal = S[:, -1]
        return {
            "mean":       float(np.mean(terminal)),
            "median":     float(np.median(terminal)),
            "std":        float(np.std(terminal)),
            "5th_pct":    float(np.percentile(terminal, 5)),
            "95th_pct":   float(np.percentile(terminal, 95)),
            "prob_above_S0": float(np.mean(terminal > self.S0)),
        }

    def log_returns(self, S: np.ndarray) -> np.ndarray:
        """Compute log returns from simulated price paths."""
        return np.diff(np.log(S), axis=1)
