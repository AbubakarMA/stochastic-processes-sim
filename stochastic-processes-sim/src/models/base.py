"""
base.py
-------
Abstract base class for all stochastic process simulators.
Every model inherits from StochasticProcess and must implement simulate().
"""

from abc import ABC, abstractmethod
import numpy as np


class StochasticProcess(ABC):
    """
    Base class for stochastic process simulators.

    All subclasses must implement simulate(), which returns a 2D array
    of shape (n_paths, n_steps + 1) where column 0 is the initial value.
    """

    def __init__(self, T: float = 1.0, n_steps: int = 252, n_paths: int = 100, seed: int = None):
        """
        Parameters
        ----------
        T        : float  — total time horizon in years
        n_steps  : int    — number of time steps (252 = trading days in a year)
        n_paths  : int    — number of independent simulation paths
        seed     : int    — random seed for reproducibility (None = random)
        """
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")

        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.seed = seed
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)

        self._rng = np.random.default_rng(seed)

    def _sample_increments(self) -> np.ndarray:
        """
        Draw standard normal increments of shape (n_paths, n_steps).
        Scaled by sqrt(dt) to give Wiener increments.
        """
        return self._rng.standard_normal((self.n_paths, self.n_steps)) * np.sqrt(self.dt)

    @abstractmethod
    def simulate(self) -> np.ndarray:
        """
        Run the simulation.

        Returns
        -------
        paths : np.ndarray, shape (n_paths, n_steps + 1)
            Each row is one simulated path.
        """
        pass

    def summary_stats(self, paths: np.ndarray) -> dict:
        """
        Compute summary statistics across all paths at final time T.

        Parameters
        ----------
        paths : np.ndarray — simulation output from simulate()

        Returns
        -------
        dict with mean, std, min, 5th percentile, median, 95th percentile, max
        """
        final = paths[:, -1]
        return {
            "mean":   float(np.mean(final)),
            "std":    float(np.std(final)),
            "min":    float(np.min(final)),
            "p5":     float(np.percentile(final, 5)),
            "median": float(np.median(final)),
            "p95":    float(np.percentile(final, 95)),
            "max":    float(np.max(final)),
        }
