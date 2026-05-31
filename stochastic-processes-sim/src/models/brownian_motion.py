"""
Brownian Motion (Wiener Process)

The foundation of all continuous-time stochastic processes in finance.
W(t) ~ N(0, t)  — normally distributed with mean 0 and variance t.

Key properties:
  - W(0) = 0
  - Independent increments: W(t) - W(s) ⊥ W(s) - W(r) for r < s < t
  - Continuous paths (almost surely)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrownianMotion:
    """
    Standard Brownian Motion (Wiener Process).

    Parameters
    ----------
    T       : float  — total time horizon (years)
    N       : int    — number of time steps
    n_paths : int    — number of simulated paths
    seed    : int    — random seed for reproducibility
    """
    T: float = 1.0
    N: int = 252          # trading days in a year
    n_paths: int = 10
    seed: Optional[int] = 42

    # computed after __post_init__
    dt: float = field(init=False)
    t: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dt = self.T / self.N
        self.t = np.linspace(0, self.T, self.N + 1)
        if self.seed is not None:
            np.random.seed(self.seed)

    def simulate(self) -> np.ndarray:
        """
        Simulate paths using the standard construction:
            dW = sqrt(dt) * Z,  Z ~ N(0,1)

        Returns
        -------
        W : ndarray, shape (n_paths, N+1)
            Each row is one simulated path starting at W(0)=0.
        """
        Z = np.random.standard_normal((self.n_paths, self.N))
        dW = np.sqrt(self.dt) * Z                        # increments
        W = np.zeros((self.n_paths, self.N + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)                 # cumulative sum
        return W

    def theoretical_std(self) -> np.ndarray:
        """Theoretical standard deviation: std(W(t)) = sqrt(t)."""
        return np.sqrt(self.t)
