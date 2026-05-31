"""
Visualization module for stochastic process simulations.
All plots use a clean, publication-quality style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ── Style ────────────────────────────────────────────────────────────────────
COLORS = {
    "paths":       "#85B7EB",   # light blue for individual paths
    "mean":        "#185FA5",   # strong blue for theoretical mean
    "std_band":    "#B5D4F4",   # very light blue for ±1σ band
    "highlight":   "#D85A30",   # coral for one highlighted path
    "neutral":     "#888780",   # gray for secondary elements
    "green":       "#1D9E75",
}

def _base_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "font.family":       "sans-serif",
        "font.size":         11,
    })


# ── Brownian Motion ───────────────────────────────────────────────────────────

def plot_brownian_motion(bm, W: np.ndarray, save_path: str = None):
    """
    Plot BM paths with theoretical ±1σ and ±2σ envelopes.
    """
    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Standard Brownian Motion (Wiener Process)", fontsize=14, fontweight="normal", y=1.01)

    # ── Left: Path plot ──
    ax = axes[0]
    ax.set_title("Simulated paths", fontsize=12, pad=8)

    for i in range(min(W.shape[0], 30)):
        ax.plot(bm.t, W[i], color=COLORS["paths"], alpha=0.4, linewidth=0.8)

    # Highlight one path
    ax.plot(bm.t, W[0], color=COLORS["highlight"], linewidth=1.5, label="Sample path", zorder=5)

    # Theoretical envelopes
    std = bm.theoretical_std()
    ax.fill_between(bm.t, -2*std, 2*std, color=COLORS["std_band"], alpha=0.4, label="±2σ band")
    ax.fill_between(bm.t, -std, std, color=COLORS["std_band"], alpha=0.6, label="±1σ band")
    ax.axhline(0, color=COLORS["neutral"], linewidth=1, linestyle="--", alpha=0.6)

    ax.set_xlabel("Time t")
    ax.set_ylabel("W(t)")
    ax.legend(fontsize=9)

    # ── Right: Distribution of W(T) ──
    ax2 = axes[1]
    ax2.set_title(f"Distribution of W(T={bm.T:.1f})", fontsize=12, pad=8)

    terminal = W[:, -1]
    ax2.hist(terminal, bins=40, color=COLORS["paths"], edgecolor="white", linewidth=0.5, density=True, alpha=0.8)

    # Overlay N(0, T) theoretical density
    x = np.linspace(terminal.min(), terminal.max(), 200)
    from scipy.stats import norm
    ax2.plot(x, norm.pdf(x, 0, np.sqrt(bm.T)), color=COLORS["mean"], linewidth=2,
             label=f"N(0, {bm.T:.1f}) theoretical")
    ax2.axvline(np.mean(terminal), color=COLORS["highlight"], linestyle="--", linewidth=1.5,
                label=f"Simulated mean = {np.mean(terminal):.3f}")

    ax2.set_xlabel("W(T)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── Geometric Brownian Motion ─────────────────────────────────────────────────

def plot_gbm(gbm, S: np.ndarray, save_path: str = None):
    """
    4-panel GBM figure:
      1. Price paths with theoretical mean
      2. Terminal price distribution vs log-normal theory
      3. Log-return distribution
      4. Cumulative return fan (5th / 50th / 95th percentiles)
    """
    _base_style()
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle(
        f"Geometric Brownian Motion  |  S₀={gbm.S0}  μ={gbm.mu:.0%}  σ={gbm.sigma:.0%}  T={gbm.T}yr",
        fontsize=13, fontweight="normal", y=1.01
    )

    # ── Panel 1: Price paths ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Simulated price paths", fontsize=11)
    n_show = min(S.shape[0], 50)
    for i in range(n_show):
        ax1.plot(gbm.t, S[i], color=COLORS["paths"], alpha=0.25, linewidth=0.7)
    ax1.plot(gbm.t, S[0], color=COLORS["highlight"], linewidth=1.5, label="Sample path", zorder=5)
    ax1.plot(gbm.t, gbm.theoretical_mean(), color=COLORS["mean"], linewidth=2.5,
             label=f"E[S(t)] = S₀·e^(μt)", zorder=6)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Price S(t)")
    ax1.legend(fontsize=9)

    # ── Panel 2: Terminal distribution ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f"Terminal price S(T={gbm.T})", fontsize=11)
    terminal = S[:, -1]
    ax2.hist(terminal, bins=50, color=COLORS["paths"], edgecolor="white", linewidth=0.3,
             density=True, alpha=0.8)

    # Log-normal theoretical
    from scipy.stats import lognorm
    log_mean = np.log(gbm.S0) + (gbm.mu - 0.5*gbm.sigma**2)*gbm.T
    log_std  = gbm.sigma * np.sqrt(gbm.T)
    x = np.linspace(terminal.min(), terminal.max(), 300)
    ax2.plot(x, lognorm.pdf(x, s=log_std, scale=np.exp(log_mean)),
             color=COLORS["mean"], linewidth=2, label="Log-normal theory")
    ax2.axvline(np.median(terminal), color=COLORS["highlight"], linestyle="--",
                linewidth=1.5, label=f"Median = {np.median(terminal):.1f}")
    ax2.set_xlabel("S(T)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)

    # ── Panel 3: Log-return distribution ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Daily log-return distribution", fontsize=11)
    log_rets = np.diff(np.log(S), axis=1).flatten()
    ax3.hist(log_rets, bins=80, color=COLORS["green"], edgecolor="white",
             linewidth=0.3, density=True, alpha=0.8)

    from scipy.stats import norm
    x2 = np.linspace(log_rets.min(), log_rets.max(), 300)
    ax3.plot(x2, norm.pdf(x2, loc=(gbm.mu - 0.5*gbm.sigma**2)*gbm.dt,
                          scale=gbm.sigma*np.sqrt(gbm.dt)),
             color=COLORS["mean"], linewidth=2, label="N(μdt, σ²dt) theory")
    ax3.set_xlabel("log(S(t+dt)/S(t))")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=9)

    # ── Panel 4: Percentile fan ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Price fan (5th / 50th / 95th pct)", fontsize=11)
    pct5  = np.percentile(S, 5,  axis=0)
    pct50 = np.percentile(S, 50, axis=0)
    pct95 = np.percentile(S, 95, axis=0)
    ax4.fill_between(gbm.t, pct5, pct95, color=COLORS["std_band"], alpha=0.5, label="5th–95th pct")
    ax4.plot(gbm.t, pct50, color=COLORS["mean"], linewidth=2, label="Median path")
    ax4.plot(gbm.t, gbm.theoretical_mean(), color=COLORS["highlight"],
             linestyle="--", linewidth=1.5, label="Theoretical mean")
    ax4.axhline(gbm.S0, color=COLORS["neutral"], linestyle=":", linewidth=1)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax4.set_xlabel("Time (years)")
    ax4.set_ylabel("Price S(t)")
    ax4.legend(fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── Ornstein-Uhlenbeck ────────────────────────────────────────────────────────

def plot_ou(ou, X: np.ndarray, save_path: str = None):
    """
    2-panel OU plot: paths + mean reversion evidence.
    """
    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Ornstein-Uhlenbeck Process  |  θ={ou.theta}  μ={ou.mu:.2%}  σ={ou.sigma:.2%}",
        fontsize=13, fontweight="normal", y=1.01
    )

    ax = axes[0]
    ax.set_title("Mean-reverting paths", fontsize=11)
    for i in range(min(X.shape[0], 30)):
        ax.plot(ou.t, X[i], color=COLORS["paths"], alpha=0.35, linewidth=0.8)
    ax.plot(ou.t, X[0], color=COLORS["highlight"], linewidth=1.5, label="Sample path")
    ax.plot(ou.t, ou.theoretical_mean(), color=COLORS["mean"], linewidth=2.5,
            label="Theoretical mean E[X(t)]")
    ax.axhline(ou.mu, color=COLORS["neutral"], linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Long-run mean μ = {ou.mu:.2%}")

    std = np.sqrt(ou.theoretical_var())
    ax.fill_between(ou.t, ou.theoretical_mean()-std, ou.theoretical_mean()+std,
                    color=COLORS["std_band"], alpha=0.4, label="±1σ band")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("X(t)")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.set_title("Simulated vs theoretical mean & std", fontsize=11)
    sim_mean = np.mean(X, axis=0)
    sim_std  = np.std(X, axis=0)
    ax2.plot(ou.t, sim_mean,            color=COLORS["highlight"], linewidth=2, label="Simulated mean")
    ax2.plot(ou.t, ou.theoretical_mean(), color=COLORS["mean"], linewidth=2, linestyle="--",
             label="Theoretical mean")
    ax2.fill_between(ou.t, sim_mean - sim_std, sim_mean + sim_std,
                     color=COLORS["std_band"], alpha=0.4, label="Simulated ±1σ")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax2.set_xlabel("Time (years)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig
