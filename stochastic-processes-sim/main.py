"""
main.py — Run all stochastic process simulations and save plots.

Usage:
    python main.py
    python main.py --n_paths 200 --T 2.0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.models import BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck
from src.visualization import plot_brownian_motion, plot_gbm, plot_ou

OUTPUT_DIR = "data"


def run_brownian_motion(n_paths: int, T: float):
    print("\n── Brownian Motion ──────────────────────────────")
    bm = BrownianMotion(T=T, N=252, n_paths=n_paths, seed=42)
    W  = bm.simulate()
    print(f"  Paths : {W.shape[0]}  |  Steps : {W.shape[1]-1}  |  T = {T}")
    print(f"  E[W(T)] simulated  : {W[:, -1].mean():.4f}  (theory = 0)")
    print(f"  std[W(T)] simulated: {W[:, -1].std():.4f}  (theory = {T**0.5:.4f})")
    fig = plot_brownian_motion(bm, W, save_path=f"{OUTPUT_DIR}/brownian_motion.png")
    return fig


def run_gbm(n_paths: int, T: float):
    print("\n── Geometric Brownian Motion ────────────────────")
    gbm = GeometricBrownianMotion(S0=100, mu=0.08, sigma=0.20, T=T, N=252, n_paths=n_paths, seed=42)
    S   = gbm.simulate()
    stats = gbm.terminal_distribution(S)
    print(f"  Paths : {S.shape[0]}  |  T = {T}yr  |  S0 = {gbm.S0}")
    print(f"  Terminal mean    : {stats['mean']:.2f}  (theory = {gbm.theoretical_mean()[-1]:.2f})")
    print(f"  Terminal median  : {stats['median']:.2f}")
    print(f"  Terminal std     : {stats['std']:.2f}")
    print(f"  5th / 95th pct   : {stats['5th_pct']:.2f} / {stats['95th_pct']:.2f}")
    print(f"  P(S(T) > S0)     : {stats['prob_above_S0']:.2%}")
    fig = plot_gbm(gbm, S, save_path=f"{OUTPUT_DIR}/gbm.png")
    return fig


def run_ou(n_paths: int, T: float):
    print("\n── Ornstein-Uhlenbeck ───────────────────────────")
    ou = OrnsteinUhlenbeck(X0=0.08, theta=3.0, mu=0.05, sigma=0.02, T=T, N=252, n_paths=n_paths, seed=42)
    X  = ou.simulate()
    print(f"  Paths : {X.shape[0]}  |  T = {T}yr")
    print(f"  Final mean  simulated : {X[:, -1].mean():.4f}  (theory = {ou.theoretical_mean()[-1]:.4f})")
    print(f"  Final std   simulated : {X[:, -1].std():.4f}  (theory = {ou.theoretical_var()[-1]**0.5:.4f})")
    fig = plot_ou(ou, X, save_path=f"{OUTPUT_DIR}/ornstein_uhlenbeck.png")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Stochastic Processes Simulator")
    parser.add_argument("--n_paths", type=int, default=500)
    parser.add_argument("--T",       type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_brownian_motion(args.n_paths, args.T)
    run_gbm(args.n_paths, args.T)
    run_ou(args.n_paths, args.T)

    print(f"\nAll plots saved to ./{OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
