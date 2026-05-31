# Stochastic Processes Simulator

> **Module 01** of the [Quantitative Finance Learning Track](https://github.com/AbubakarMA)
> Part of a 12-project GitHub portfolio in quantitative finance, data science, and AgriTech.

Implementations of three foundational stochastic processes used in quantitative finance, built from first principles in Python. Each model includes the exact analytical solution, Monte Carlo simulation, theoretical validation, and publication-quality visualisations.

---

## Processes implemented

| Process | SDE | Key use |
|---|---|---|
| Brownian Motion | dW = √dt · Z | Foundation of all continuous-time finance |
| Geometric Brownian Motion | dS = μS dt + σS dW | Stock prices (Black-Scholes model) |
| Ornstein-Uhlenbeck | dX = θ(μ−X)dt + σ dW | Interest rates, commodity prices, pair trading |

---

## Key results

**Brownian Motion** — E[W(T)] = 0.024 (theory: 0), std[W(T)] = 0.984 (theory: 1.000)

**GBM** — Terminal mean: 108.77 (theory: 108.33), P(S(T) > S₀) = 64.2% with μ=8%

**Ornstein-Uhlenbeck** — Final mean: 0.0516 (theory: 0.0515), confirms mean reversion to μ=5%

---

## Project structure

```
stochastic-processes-sim/
├── src/
│   ├── models/
│   │   ├── brownian_motion.py     # Standard Wiener process
│   │   ├── gbm.py                 # Geometric Brownian Motion (exact solution)
│   │   └── ornstein_uhlenbeck.py  # Mean-reverting OU process
│   └── visualization/
│       └── plots.py               # Publication-quality matplotlib figures
├── tests/
│   └── test_models.py             # 17 unit tests (pytest)
├── data/                          # Generated plots
├── main.py                        # Run all simulations
├── requirements.txt
└── setup.py
```

---

## Quickstart

```bash
git clone https://github.com/AbubakarMA/stochastic-processes-sim
cd stochastic-processes-sim
pip install -r requirements.txt

# Run all simulations and generate plots
python main.py

# Custom parameters
python main.py --n_paths 1000 --T 2.0

# Run tests
pytest tests/ -v
```

---

## Usage in Python

```python
from src.models import GeometricBrownianMotion
from src.visualization import plot_gbm

# Simulate GBM for a stock with 8% return, 20% volatility
gbm = GeometricBrownianMotion(S0=100, mu=0.08, sigma=0.20, T=1.0, n_paths=500)
S   = gbm.simulate()

# Terminal distribution statistics
print(gbm.terminal_distribution(S))
# {'mean': 108.77, 'median': 107.85, 'std': 21.44, 'prob_above_S0': 0.642, ...}

# Plot
fig = plot_gbm(gbm, S, save_path="data/gbm.png")
```

---

## Mathematical notes

### Itô's Lemma — why GBM has a drift correction

Naively you might expect: S(t) = S₀ · exp(μt + σW(t))

But Itô's Lemma shows that for f(W) = exp(W):
```
df = f'dW + ½f''dt  →  drift correction of −½σ²
```

So the correct formula is: **S(t) = S₀ · exp((μ − σ²/2)·t + σ·W(t))**

This is implemented in `gbm.py` and verified against the log-normal distribution.

---

## Tests

```
17 passed in 5.72s

TestBrownianMotion        (5 tests) — shape, zero-start, mean≈0, var≈T, theoretical std
TestGeometricBrownianMotion (7 tests) — shape, S0-start, positivity, mean, normality, stats, P(S>S0)
TestOrnsteinUhlenbeck     (5 tests) — shape, X0-start, mean reversion, variance, non-negativity
```

---

## Next modules

- **Module 02**: [Portfolio Optimiser](../portfolio-optimiser) — Markowitz efficient frontier, VaR, CVaR
- **Module 03**: [Financial Forecaster](../financial-forecaster) — ARIMA, GARCH, LSTM

---

## Author

**Abubakar Mamudu Alutiba**
MSc Financial Engineering, WorldQuant University
Data Analyst, Broad Spectrum Limited, Ghana
[github.com/AbubakarMA](https://github.com/AbubakarMA) · [linkedin.com/in/abubakarma](https://linkedin.com/in/abubakarma)
