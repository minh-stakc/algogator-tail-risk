# Regime-Conditional Tail Risk Modeling for Portfolio Risk Management

**Nathan Hoang**
AlgoGators Investment Fund
March 2026

---

## Abstract

This paper investigates whether regime-aware risk models that combine Bayesian changepoint
detection with dynamic dependence and extreme-value methods produce more reliable
Conditional Value-at-Risk (CVaR) forecasts than traditional static or rolling-window
models. Using daily return data from the S&P 500 and nine sector SPDR ETFs over 2005–2024,
I detect market regimes via a Hidden Markov Model (HMM) and estimate tail risk separately
within each regime using Peaks-over-Threshold extreme value theory (EVT). Results show that
regime conditioning reveals economically meaningful differences in tail behavior — the
high-volatility regime exhibits CVaR nearly three times larger and excess kurtosis over four
times greater than the low-volatility regime — and that cross-asset correlations increase
substantially during stress periods, undermining diversification. However, on standard
out-of-sample backtesting metrics, a GARCH(1,1) model outperforms the regime-conditional
approach in terms of both violation rate accuracy and forecast error, primarily because
regime detection lags sudden market shocks. Regime-conditional models do excel during
sustained stress regimes (2022 rate-hike cycle: 0.4% violations vs. 3–4% for baselines).
The findings suggest that regime-aware modeling is most valuable as a risk-monitoring and
capital-allocation overlay rather than as a standalone CVaR forecaster.

---

## 1. Introduction

Portfolio risk management is especially challenging during periods of acute market stress.
Standard risk models — such as historical simulation and parametric Value-at-Risk — assume
that return distributions and cross-asset correlations remain roughly stable over time. In
practice, financial markets move through distinct regimes: calm, low-volatility environments
where diversification works as expected, and turbulent, high-volatility regimes where asset
correlations spike, losses cluster, and tail risks are severely underestimated by models
calibrated on mixed data.

The central measurement used to evaluate extreme downside risk in this paper is the
**Conditional Value-at-Risk (CVaR)**, also called Expected Shortfall, defined as the
expected portfolio loss conditional on exceeding the Value-at-Risk threshold. CVaR is
preferred over VaR because it captures the full severity of tail losses rather than just
the threshold.

**Research question:** Do regime-aware risk models that combine Bayesian changepoint
detection with dynamic dependence and extreme-value methods produce more reliable CVaR
forecasts than traditional static or rolling-window risk models?

**Hypothesis:** Financial return distributions and cross-asset dependence structures behave
differently across regimes. Pooling observations from different regimes into a single model
leads to systematic underestimation of joint tail risk during stress periods. Separating
the estimation by regime will improve the accuracy and stability of CVaR forecasts.

The approach is as follows: detect market regimes using a Gaussian Hidden Markov Model
and Bayesian changepoint detection (PELT algorithm); estimate regime-specific tail
distributions using Generalized Pareto Distribution (GPD) fitting; and compare out-of-sample
CVaR forecasting accuracy against three baselines — static historical simulation,
rolling-window historical simulation, and GARCH(1,1).

---

## 2. Methodology

### 2.1 Data

Daily adjusted closing prices were downloaded from Yahoo Finance for ten equity instruments:
the S&P 500 ETF (SPY) and nine sector SPDR ETFs (XLF, XLE, XLV, XLK, XLI, XLP, XLU, XLY,
XLB), covering January 2005 through December 2024 — a total of **5,031 trading days**.
The sample intentionally spans multiple market stress events: the 2008 Global Financial
Crisis (GFC), the 2020 COVID market crash, and the 2022 Federal Reserve tightening cycle.
The CBOE Volatility Index (VIX) was also downloaded as a supplementary stress indicator.

Daily log returns were computed as $r_t = \ln(P_t / P_{t-1})$. An equal-weight portfolio
was constructed by averaging returns across all ten assets. The data were split into a
**training set (2005–2017, 3,271 days)** and a **test set (2018–2024, 1,760 days)**.

**Table 1: Portfolio return summary statistics (full sample)**

| Statistic | Value |
|---|---|
| Mean daily return | 0.0356% |
| Std deviation | 1.169% |
| Minimum | −12.25% |
| Maximum | +10.62% |
| Excess kurtosis | **14.14** |
| Skewness | −0.59 |

The extreme kurtosis (14.14 vs. 3.0 for a normal distribution) and negative skewness
confirm pronounced heavy tails and asymmetric loss behavior, providing strong motivation
for extreme value theory over Gaussian assumptions.

### 2.2 Regime Detection

**Bayesian Changepoint Detection (PELT)**

Structural breaks in the time series were identified using the Pruned Exact Linear Time
(PELT) algorithm with a Bayesian Information Criterion (BIC) penalty. Applied to the
rolling 21-day realized volatility signal, PELT detected **36 structural breakpoints**,
producing 37 distinct volatility regimes. Key detected dates align closely with known
market events: September 2008 (Lehman collapse), February 2020 (COVID onset), May 2022
(peak rate-hike volatility). This method provides a deterministic, interpretable segmentation
of the time series.

**Hidden Markov Model (HMM)**

A 2-state Gaussian HMM was fitted to a standardized feature matrix consisting of rolling
21-day realized volatility, rolling mean return, and VIX level. The HMM produces soft
posterior probabilities over latent states and a Viterbi sequence of most-likely state
assignments. States are labeled by ascending mean volatility: state 0 = low-vol, state 1
= high-vol.

**Table 2: HMM transition matrix**

| | To: Low-vol | To: High-vol |
|---|---|---|
| From: Low-vol | 0.9911 | 0.0089 |
| From: High-vol | 0.0187 | 0.9813 |

Both states are highly persistent (average regime duration ~112 days for low-vol, ~53 days
for high-vol), consistent with the literature on market regime stickiness.

### 2.3 Tail Risk Modeling

**Extreme Value Theory (EVT) — Peaks-over-Threshold**

For each regime, a Generalized Pareto Distribution (GPD) was fitted to loss exceedances
above the 85th-percentile threshold using maximum likelihood estimation. The GPD is
characterized by shape parameter $\xi$ (tail heaviness) and scale parameter $\beta$.
CVaR is computed analytically:

$$\text{CVaR}_\alpha = \frac{\text{VaR}_\alpha + \beta - \xi \cdot u}{1 - \xi}, \quad \xi < 1$$

where $u$ is the threshold and $\text{VaR}_\alpha = u + \frac{\beta}{\xi}\left[\left(\frac{p_u}{\alpha}\right)^\xi - 1\right]$.

**Copula Analysis**

To model multivariate tail dependence, three copula families were fitted:
(1) **Gaussian copula** — zero tail dependence baseline;
(2) **Student-t copula** — symmetric tail dependence parameterized by degrees of freedom $\nu$;
(3) **Clayton copula** — asymmetric lower tail dependence $\lambda_L = 2^{-1/\theta}$.

Pseudo-observations were constructed via empirical rank transformation, and copula parameters
were estimated by maximum likelihood.

### 2.4 Baseline Models

Three baselines were compared against the regime-conditional approach:

1. **Static Historical CVaR:** Full training-sample historical simulation; constant forecast.
2. **Rolling-Window CVaR:** 252-day sliding-window historical simulation; updated daily.
3. **GARCH(1,1) CVaR:** Conditional variance from GARCH(1,1) with normal innovations;
   CVaR derived analytically as $\mu_t + \sigma_t \cdot \phi(z_\alpha)/\alpha$.

### 2.5 Walk-Forward Backtesting

Regime-conditional models were evaluated via an **expanding-window walk-forward backtest**:
models were re-estimated every quarter (63 trading days) using all data up to the
forecast origin, with a minimum 756-day (3-year) training burn-in. The forecast-day CVaR
was set equal to the regime-specific estimate for the most recently observed regime.
Statistical accuracy was assessed using:

- **Kupiec (1995) POF test:** Likelihood ratio test of whether the observed violation rate
  equals the expected $\alpha = 5\%$.
- **Christoffersen (1998) Conditional Coverage test:** Joint test of coverage and
  independence of violations (violations should not cluster).
- **Diebold-Mariano (1995) test:** Pairwise comparison of squared forecast errors.

---

## 3. Results

### 3.1 Regime Characterisation

The HMM assigns 67.8% of trading days to the low-volatility regime and 32.2% to the
high-volatility regime. The economic distinction between the two is stark.

**Table 3: Regime characteristics (full sample)**

| Statistic | Low-vol Regime | High-vol Regime |
|---|---|---|
| Days (% of sample) | 3,396 (67.8%) | 1,615 (32.2%) |
| Ann. mean return | +16.54% | −6.81% |
| Ann. volatility | 10.76% | **28.77%** |
| Skewness | −0.345 | −0.369 |
| Excess kurtosis | 1.39 | **6.05** |
| CVaR (95%) | 1.56% | **4.42%** |

The high-vol regime produces CVaR approximately **2.8 times larger** than the low-vol
regime and excess kurtosis more than four times greater, confirming that tail risk is
substantially regime-dependent.

*(See Figure 1: EDA plot; Figure 4: Loss distributions by regime)*

### 3.2 Correlation Breakdown in Stress

A key finding is the dramatic increase in cross-asset correlations during the high-vol regime.
For example, the SPY-XLF correlation rises from 0.815 in the low-vol regime to 0.857 in
the high-vol regime. More strikingly, XLU (Utilities) — typically a low-correlation
defensive sector — sees its correlation with SPY jump from 0.428 to 0.756.

**Table 4: Selected pairwise correlations by regime**

| Asset Pair | Low-vol | High-vol | Change |
|---|---|---|---|
| SPY – XLF | 0.815 | 0.857 | +0.042 |
| SPY – XLE | 0.580 | 0.790 | +0.210 |
| SPY – XLU | 0.428 | 0.756 | **+0.328** |
| SPY – XLV | 0.741 | 0.859 | +0.118 |

This correlation convergence directly undermines portfolio diversification during crises —
a risk that static models calibrated on pooled data will systematically miss.

*(See Figure 2: HMM posterior probabilities; Figure 3: BCP regime segmentation)*

### 3.3 Extreme Value Theory Results

**Table 5: GPD parameters by regime (training set)**

| Regime | n | xi (shape) | beta (scale) | CVaR 95% |
|---|---|---|---|---|
| Full sample | 3,271 | 0.181 | 0.00847 | 2.916% |
| Low-vol | 2,219 | −0.073 | 0.00537 | 1.558% |
| High-vol | 1,052 | **+0.116** | 0.01237 | **4.474%** |

The shape parameter $\xi > 0$ in the high-vol regime confirms a **heavy-tailed**
Pareto distribution, while $\xi < 0$ in the low-vol regime indicates a bounded,
thin-tailed distribution. Pooling both regimes (full sample $\xi = 0.181$) produces a
shape parameter intermediate between the two, diluting the signal — and likely
overstating tail risk in calm periods while understating it in stress.

*(See Figure 7: Mean excess plot — linear upward trend validates GPD fit)*

### 3.4 Copula Analysis

Fitting a Student-t copula to the full 10-asset universe yields **estimated degrees of
freedom $\nu = 4.30$**, well below the Gaussian limit, indicating substantial joint
tail dependence. The bivariate tail dependence coefficient between SPY and XLF is
$\lambda = 0.54$ on the full sample.

**Table 6: Tail dependence by stress period (Student-t copula)**

| Period | Estimated nu | Bivariate lam |
|---|---|---|
| Full sample | 4.30 | 0.543 |
| GFC 2008–09 | 7.98 | 0.458 |
| COVID 2020 | **6.69** | **0.694** |
| Rate Hike 2022 | 10.85 | 0.429 |

The COVID 2020 period shows the highest tail dependence ($\lambda = 0.694$), meaning
joint extreme losses are far more likely than a Gaussian model would predict. The Clayton
copula fitted to the SPY–XLF pair yields $\theta = 2.66$ and lower tail dependence
$\lambda_L = 0.770$, confirming that joint crash risk between the broad market and
financials is economically large.

*(See Figure 5: Tail dependence heatmap)*

### 3.5 Out-of-Sample CVaR Backtesting

**Table 7: CVaR model comparison — test period 2018–2024**

| Model | Viol Rate | Expected | Kupiec p | CC p | MAE | Quantile Score |
|---|---|---|---|---|---|---|
| Static-Hist | 1.48% | 5.0% | <0.001* | <0.001* | 0.0300 | −0.00173 |
| Rolling-252 | 2.67% | 5.0% | <0.001* | <0.001* | 0.0274 | −0.00163 |
| GARCH | **3.58%** | 5.0% | 0.004* | 0.009* | **0.0205** | **−0.00123** |
| Regime-Hist | 3.30% | 5.0% | <0.001* | <0.001* | 0.0280 | −0.00180 |
| Regime-EVT | 3.24% | 5.0% | <0.001* | <0.001* | 0.0282 | −0.00181 |

*\* Rejects null hypothesis at 5% significance.*

**No model achieves a well-calibrated 5% violation rate.** GARCH comes closest (3.58%),
followed by Regime-Hist (3.30%) and Regime-EVT (3.24%). Static-Hist is the most
over-conservative, producing only 1.48% violations — meaning it consistently overstates
risk and would require unnecessary capital reserves.

The Christoffersen conditional coverage test rejects all models, indicating that violations
cluster in time for every approach — a persistent challenge in risk modeling.

Diebold-Mariano pairwise tests confirm that GARCH dominates regime models:
Regime-EVT vs. GARCH: DM = 5.71, p < 0.001. Regime-EVT does not statistically
outperform any baseline on squared forecast error.

*(See Figure 6: CVaR forecast comparison; Figure 8: Violation plots by model)*

### 3.6 Stress-Period Analysis

**Table 8: Violation rates during stress periods**

| Model | Full Sample | COVID 2020 | Rate Hike 2022 |
|---|---|---|---|
| Static-Hist | 1.48% | 25.00% | 3.19% |
| Rolling-252 | 2.67% | 17.31% | 4.38% |
| GARCH | 3.58% | **11.54%** | 3.59% |
| Regime-Hist | 3.30% | 28.85% | **0.40%** |
| Regime-EVT | 3.24% | 28.85% | **0.40%** |

This table reveals a critical split in regime model performance:

- **COVID 2020 (sudden shock):** Regime models have the *worst* violation rates (28.85%),
  even worse than Static-Hist (25%). The HMM has not yet transitioned to the high-vol
  regime when losses begin, so the forecast remains anchored to the low-vol CVaR estimate.
- **Rate Hike 2022 (sustained regime):** Regime models have the *best* violation rates
  (0.40%), dramatically outperforming all baselines (3–4%). Once the HMM identifies the
  high-vol regime, the elevated CVaR estimate is maintained throughout the entire cycle.

---

## 4. Discussion

### 4.1 Interpretation

The results provide **partial support** for the hypothesis. Regime conditioning does reveal
meaningful, economically important differences in tail behavior: the high-vol regime has
nearly three times the CVaR of the low-vol regime, GPD shape parameters differ
significantly, and copula tail dependence rises sharply during crises. These findings
confirm that pooling regimes in a single model obscures important risk dynamics.

However, on standard out-of-sample CVaR accuracy metrics, GARCH outperforms the
regime-conditional approach. The key reason is **regime detection lag**: the HMM updates
its state estimate gradually as evidence accumulates. During sudden shocks like the COVID
crash, the model is still in "low-vol mode" precisely when losses are worst, producing
catastrophic underprediction of risk in the first weeks of the crisis.

GARCH avoids this problem because it conditions on the *previous day's* realized
return — it essentially responds to stress at a one-day lag rather than a multi-week lag.

### 4.2 Implications for the Fund

These findings suggest a **two-layer risk management framework**:

1. **GARCH for daily CVaR forecasting:** Best calibrated for day-to-day risk limits and
   position sizing, especially during abrupt market shifts.
2. **HMM regime monitoring as an overlay:** Use the posterior probability of the high-vol
   state as a leading indicator for strategic risk reduction — scaling back exposure when
   P(high-vol) exceeds a threshold (e.g., 40%) *before* the crisis is fully reflected
   in realized volatility.

The finding that regime-conditional models dramatically outperform during sustained stress
(2022) but fail during sudden crashes (2020) is itself actionable: the fund could use
GARCH-based limits for normal operations and regime-conditional limits for slow-moving
macro risks (credit cycles, rate cycles).

### 4.3 Limitations

Several limitations should be acknowledged:

1. **Regime detection lag** is the principal limitation, as discussed above. A forward-looking
   regime indicator (e.g., incorporating option-implied volatility surfaces or credit spreads)
   could partially address this.
2. **Equal-weight portfolio assumption** simplifies the analysis. Real portfolios have
   time-varying weights, which would affect both the return series and the optimal regime-
   conditional CVaR estimate.
3. **HMM identifiability:** The 2-state HMM may be too coarse; the 3-state extension
   (bull/transition/bear) could better capture intermediate regimes, though it reduces
   the data available per state for tail estimation.
4. **Copula CVaR** was estimated in-sample only. Walk-forward copula backtesting would
   require considerably more computation and is reserved for future work.
5. **Look-ahead bias in PELT:** The Bayesian changepoint detection was applied to the full
   sample for regime visualization; a real-time implementation would need a recursive
   online variant (e.g., Bayesian Online Changepoint Detection, BOCPD).

### 4.4 Next Steps

- Implement BOCPD for truly real-time regime detection to reduce the detection lag.
- Incorporate credit spreads (FRED: ICE BofA OAS) and options-derived measures as
  forward-looking HMM features.
- Extend the copula analysis to regime-specific copulas (fit separate copulas in each regime).
- Test alternative copula families (Gumbel, Frank) for robustness.
- Apply the framework to other asset classes (fixed income, commodities) to test generalizability.

---

## 5. Conclusion

This paper examined whether regime-conditional tail risk models improve CVaR forecasting
accuracy for a diversified equity portfolio. The empirical results are nuanced. On one hand,
the high-volatility regime identified by the HMM exhibits fundamentally different tail
characteristics — heavier GPD tails ($\xi = +0.116$ vs. $-0.073$), nearly three times
the CVaR, and substantially elevated cross-asset correlations — demonstrating that regime
conditioning provides genuine economic insight. On the other hand, a simple GARCH(1,1)
model outperforms regime-conditional models on standard backtesting metrics because it
adapts to volatility changes one day at a time, while HMM-based regime detection lags
sudden shocks by days to weeks.

The main contribution of this paper is to identify the conditions under which each
approach is preferable: GARCH for rapid-onset crises; regime-conditional models for
sustained stress cycles. This suggests a hybrid framework — GARCH for daily risk limits,
with HMM regime probability as a strategic overlay — as the most practical application
for the AlgoGators Investment Fund.

---

## References

Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic
Review, 39*(4), 841–862.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of
Business & Economic Statistics, 13*(3), 253–263.

Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling extremal events for
insurance and finance*. Springer.

Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the
variance of United Kingdom inflation. *Econometrica, 50*(4), 987–1007.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time
series and the business cycle. *Econometrica, 57*(2), 357–384.

Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models.
*Journal of Derivatives, 3*(2), 73–84.

McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures for
heteroscedastic financial time series: An extreme value approach. *Journal of Empirical
Finance, 7*(3–4), 271–300.

McNeil, A. J., Frey, R., & Embrechts, P. (2005). *Quantitative risk management: Concepts,
techniques and tools*. Princeton University Press.

Nelsen, R. B. (2006). *An introduction to copulas* (2nd ed.). Springer.

Rockafeller, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk.
*Journal of Risk, 2*(3), 21–41.

---

## Appendix: Figure List

| Figure | File | Description |
|---|---|---|
| Figure 1 | fig_eda.png | Portfolio returns, rolling vol, VIX with stress shading |
| Figure 2 | fig_hmm_states.png | HMM posterior state probabilities over full sample |
| Figure 3 | fig_bcp_regimes.png | Bayesian changepoint (PELT) regime segmentation |
| Figure 4 | fig_loss_distributions.png | Loss histograms by regime + QQ-plot vs. normal |
| Figure 5 | fig_tail_dependence.png | Pairwise tail dependence heatmap (Student-t copula) |
| Figure 6 | fig_cvar_comparison.png | CVaR forecast time series — all models, test period |
| Figure 7 | fig_mean_excess.png | Mean excess plot — GPD validity diagnostic |
| Figure 8 | fig_violations.png | CVaR violations (red dots) by model, test period |
| Table 7 | results_backtest.csv | Full backtesting metrics (also in Table 7 above) |
| Table 8 | results_stress_violations.csv | Stress-period violation rates (also in Table 8 above) |
