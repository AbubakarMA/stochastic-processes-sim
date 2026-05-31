"""
Ornstein-Uhlenbeck (OU) Process — Mean-Reverting

Models interest rates (Vasicek), commodity prices, pair-trading spreads.

SDE:  dX = θ·(μ - X)·dt + σ·dW

Parameters
----------
θ (theta) : speed of mean reversion (how fast X returns to μ)
μ (mu)    : long-run mean
σ (sigma) : volatility

Exact discretisation (Euler-Maruyama):
    X(t+dt) = X(t) + θ·(μ - X(t))·dt + σ·√dt·Z
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

    def __post_init__(self):
        self.dt = self.T / self.N
        self.t = np.linspace(0, self.T, self.N + 1)
        if self.seed is not None:
            np.random.seed(self.seed)

    def simulate(self) -> np.ndarray:
        """
        Euler-Maruyama discretisation.

        Returns
        -------
        X : ndarray, shape (n_paths, N+1)
        """
        Z = np.random.standard_normal((self.n_paths, self.N))
        X = np.zeros((self.n_paths, self.N + 1))
        X[:, 0] = self.X0

        for i in range(self.N):
            drift = self.theta * (self.mu - X[:, i]) * self.dt
            noise = self.sigma * np.sqrt(self.dt) * Z[:, i]
            X[:, i + 1] = X[:, i] + drift + noise

        return X

    def theoretical_mean(self) -> np.ndarray:
        """E[X(t)] = mu + (X0 - mu)*exp(-theta*t)"""
        return self.mu + (self.X0 - self.mu) * np.exp(-self.theta * self.t)

    def theoretical_var(self) -> np.ndarray:
        """Var[X(t)] = (sigma^2 / 2*theta) * (1 - exp(-2*theta*t))"""
        return (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * self.t))
