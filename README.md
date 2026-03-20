# Regime-Conditional Tail Risk Modeling for Portfolio Risk Management

**Nathan Hoang** | AlgoGators Investment Fund | March 2026

## Research Question

Do regime-aware risk models that combine Bayesian changepoint detection with dynamic dependence and extreme-value methods produce more reliable CVaR forecasts than traditional static or rolling-window risk models?

## Summary

This project investigates whether conditioning tail risk estimates on detected market regimes improves CVaR forecast accuracy for a diversified equity portfolio. Using daily returns from the S&P 500 and nine sector SPDR ETFs (2005–2024), I compare five models across a 2018–2024 out-of-sample test period that covers the COVID 2020 crash and the 2022 rate-hike cycle.

**Key finding:** Regime conditioning reveals economically meaningful differences in tail behavior (high-vol CVaR is ~2.8x larger than low-vol CVaR), but GARCH(1,1) outperforms regime models on standard backtesting metrics due to the HMM regime detection lag. Regime models significantly outperform during sustained stress (2022: 0.4% violations vs. 3–4% for baselines).

## Models Compared

| Model | Type |
|---|---|
| Static Historical CVaR | Baseline — full-sample, constant |
| Rolling-Window CVaR (252d) | Baseline — sliding window |
| GARCH(1,1) CVaR | Baseline — conditional variance |
| Regime-Conditional CVaR (Historical) | Proposed |
| Regime-Conditional CVaR (EVT/GPD) | Proposed — main model |

## Methods

- **Regime detection:** Hidden Markov Model (2-state) + Bayesian changepoint detection (PELT)
- **Tail modeling:** Peaks-over-Threshold with Generalized Pareto Distribution (EVT)
- **Dependence:** Gaussian, Student-t, and Clayton copulas
- **Backtesting:** Kupiec POF test, Christoffersen CC test, Diebold-Mariano pairwise comparison

## Results (Test Period 2018–2024)

| Model | Violation Rate | Expected | MAE |
|---|---|---|---|
| Static-Hist | 1.48% | 5.0% | 0.0300 |
| Rolling-252 | 2.67% | 5.0% | 0.0274 |
| GARCH | **3.58%** | 5.0% | **0.0205** |
| Regime-Hist | 3.30% | 5.0% | 0.0280 |
| Regime-EVT | 3.24% | 5.0% | 0.0282 |

**Stress period violations:**

| Model | COVID 2020 | Rate Hike 2022 |
|---|---|---|
| GARCH | 11.54% | 3.59% |
| Regime-EVT | 28.85% | **0.40%** |

## Project Structure

```
├── analysis.py              # Main analysis script (VS Code Jupyter cells)
├── generate_pdf.py          # PDF report generator
├── rough_draft.pdf          # AlgoGators-formatted rough draft
├── requirements.txt
├── src/
│   ├── data_loader.py       # Yahoo Finance data pipeline
│   ├── regime_detection.py  # HMM + Bayesian changepoint detection
│   ├── risk_models.py       # CVaR models (static, rolling, GARCH, EVT, regime)
│   ├── copula_models.py     # Gaussian, Student-t, Clayton copulas
│   └── evaluation.py        # Kupiec, Christoffersen, Diebold-Mariano tests
├── fig_*.png                # Output figures
└── results_*.csv            # Backtest and stress-period results
```

## Getting Started

```bash
pip install -r requirements.txt

# Run full analysis (VS Code: open analysis.py and Run All Cells)
python analysis.py

# Regenerate PDF
python generate_pdf.py
```

## Data

Downloaded automatically via `yfinance` on first run. No API key required.

- **Assets:** SPY, XLF, XLE, XLV, XLK, XLI, XLP, XLU, XLY, XLB
- **Period:** January 2005 – December 2024
- **Stress events covered:** GFC 2008–09, COVID 2020, Rate Hike 2022
