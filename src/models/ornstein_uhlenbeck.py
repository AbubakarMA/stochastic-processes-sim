"""
Ornstein-Uhlenbeck (OU) Process — Mean-Reverting

Models interest rates (Vasicek), commodity prices, pair-trading spreads.

SDE:  dX = θ·(μ - X)·dt + σ·dW

Parameters
----------
θ (theta) : speed of mean reversion (how fast X returns to μ)
μ (mu)    : long-run mean
σ (sigma) : volatility

Exact discretisation (conditional distribution):
    X(t+dt) = X(t)·e^(-θdt) + μ(1 - e^(-θdt)) + σ·√((1 - e^(-2θdt))/2θ) · Z

This is the exact solution — not an approximation like Euler-Maruyama.
It is valid for any step size dt and introduces no discretisation error.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck mean-reverting process.

    Parameters
    ----------
    X0      : float — initial value
    theta   : float — speed of mean reversion
    mu      : float — long-run mean
    sigma   : float — volatility
    T, N, n_paths, seed as in other models
    """
    X0: float = 0.05        # e.g. starting interest rate 5%
    theta: float = 2.0      # moderate mean reversion speed
    mu: float = 0.05        # long-run mean (e.g. target rate)
    sigma: float = 0.02
    T: float = 1.0
    N: int = 252
    n_paths: int = 50
    seed: Optional[int] = 42

    dt: float = field(init=False)
    t: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.dt = self.T / self.N
        self.t = np.linspace(0, self.T, self.N + 1)
        self._rng = np.random.default_rng(self.seed)   # ✅ modern, isolated, thread-safe

    def simulate(self) -> np.ndarray:
        """
        Exact discretisation using the conditional distribution of the OU process.

        At each step, X(t+dt) | X(t) is normally distributed with:
            mean = X(t)·e^(-θdt) + μ·(1 - e^(-θdt))
            std  = σ·√((1 - e^(-2θdt)) / 2θ)

        This is exact for any step size — no approximation error.

        Returns
        -------
        X : ndarray, shape (n_paths, N+1)
        """
        Z = self._rng.standard_normal((self.n_paths, self.N))   # ✅ modern RNG
        X = np.zeros((self.n_paths, self.N + 1))
        X[:, 0] = self.X0

        # Pre-compute constants (same at every step since dt is fixed)
        decay      = np.exp(-self.theta * self.dt)                            # e^(-θdt)
        mean_shift = self.mu * (1 - decay)                                    # μ(1 - e^(-θdt))
        std_exact  = self.sigma * np.sqrt((1 - decay**2) / (2 * self.theta)) # exact conditional std

        for i in range(self.N):
            X[:, i + 1] = X[:, i] * decay + mean_shift + std_exact * Z[:, i]

        return X

    def theoretical_mean(self) -> np.ndarray:
        """E[X(t)] = mu + (X0 - mu)*exp(-theta*t)"""
        return self.mu + (self.X0 - self.mu) * np.exp(-self.theta * self.t)

    def theoretical_var(self) -> np.ndarray:
        """Var[X(t)] = (sigma^2 / 2*theta) * (1 - exp(-2*theta*t))"""
        return (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * self.t))
