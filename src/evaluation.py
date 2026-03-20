"""
evaluation.py
-------------
Backtesting and statistical evaluation of CVaR/VaR forecasts.

Functions
---------
compute_violations(realized, forecasts, alpha)
    Flag days where the realized loss exceeded the CVaR forecast.

kupiec_pof_test(violations, alpha)
    Kupiec (1995) Proportion-of-Failures likelihood ratio test.
    H0: observed violation rate == α.

christoffersen_cc_test(violations, alpha)
    Christoffersen (1998) Conditional Coverage test.
    Combines unconditional coverage + independence of violations.
    H0: violations are i.i.d. Bernoulli(α).

diebold_mariano_test(e1, e2)
    Diebold-Mariano (1995) equal predictive accuracy test.
    H0: E[L(ê1)] == E[L(ê2)]  where L is squared error.

compare_models(model_dict, realized, alpha)
    Run all tests for a dict of {name: cvar_series} and return a summary table.

analyse_stress_periods(violations_dict, stress_periods, index)
    Violation rates during each named stress period.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ===========================================================================
# Violation analysis
# ===========================================================================

def compute_violations(
    realized: pd.Series,
    forecasts: pd.Series,
    alpha: float = 0.05,
) -> tuple[pd.Series, float]:
    """
    Parameters
    ----------
    realized  : portfolio loss series (positive = loss)
    forecasts : CVaR forecast series (positive)
    alpha     : tail probability (CVaR level)

    Returns
    -------
    violations   : boolean pd.Series – True on exceedance days
    viol_rate    : empirical violation rate
    """
    common = realized.index.intersection(forecasts.dropna().index)
    r = realized.loc[common]
    f = forecasts.loc[common]
    viol = r > f
    return viol, float(viol.mean())


# ===========================================================================
# Kupiec (1995) Proportion-of-Failures test
# ===========================================================================

def kupiec_pof_test(
    violations: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Tests H0: p = alpha  (observed == expected violation rate).

    Test statistic:
        LR_uc = -2 ln[(α^x (1-α)^{T-x}) / (p̂^x (1-p̂)^{T-x})]
        ~ χ²(1) under H0.

    Returns dict with: statistic, p_value, violations, total, viol_rate,
                       expected_rate, reject_H0.
    """
    v = np.asarray(violations, dtype=bool)
    T = len(v)
    x = v.sum()

    if T == 0:
        return _empty_test(alpha)

    p_hat = x / T

    if p_hat in (0.0, 1.0):
        return {
            "statistic": np.nan, "p_value": np.nan,
            "violations": int(x), "total": T,
            "viol_rate": p_hat, "expected_rate": alpha,
            "reject_H0": np.nan,
        }

    lr = -2 * (
        x * np.log(alpha / p_hat)
        + (T - x) * np.log((1 - alpha) / (1 - p_hat))
    )
    p_val = float(1 - stats.chi2.cdf(lr, df=1))

    return {
        "statistic": float(lr),
        "p_value": p_val,
        "violations": int(x),
        "total": T,
        "viol_rate": float(p_hat),
        "expected_rate": float(alpha),
        "reject_H0": p_val < 0.05,
    }


# ===========================================================================
# Christoffersen (1998) Conditional Coverage test
# ===========================================================================

def christoffersen_cc_test(
    violations: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Joint test of unconditional coverage and independence of violations.

    Transition counts:
        n_{ij} = # days where I_{t-1}=i, I_t=j  (i,j ∈ {0,1})

    Independence LR:
        LR_ind = -2 ln[L_null / L_alt]  ~ χ²(1) under independence
    Unconditional coverage LR (Kupiec):
        LR_uc  ~ χ²(1)
    Conditional coverage:
        LR_cc = LR_uc + LR_ind  ~ χ²(2)

    Returns dict with CC statistic, p_value, and reject_H0.
    """
    v = np.asarray(violations, dtype=int)
    T = len(v)

    if T < 2:
        return _empty_test(alpha)

    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0:
        return _empty_test(alpha)

    p01 = n01 / n0
    p11 = n11 / n1 if n1 > 0 else 0.0
    p_hat = (n01 + n11) / T

    # Guard against log(0)
    if any(x in (0.0, 1.0) for x in [p01, p11, p_hat]):
        return _empty_test(alpha)

    ll_alt = (
        n00 * np.log(1 - p01)
        + n01 * np.log(p01)
        + n10 * np.log(1 - p11)
        + n11 * np.log(p11)
    )
    ll_null = n0 * np.log(1 - p_hat) + n1 * np.log(p_hat)
    lr_ind = -2 * (ll_null - ll_alt)

    # Unconditional coverage (Kupiec)
    x = v.sum()
    lr_uc = -2 * (
        x * np.log(alpha / p_hat) + (T - x) * np.log((1 - alpha) / (1 - p_hat))
    ) if p_hat not in (0.0, 1.0) else np.nan

    lr_cc = float(lr_ind + (lr_uc if not np.isnan(lr_uc) else 0))
    p_val = float(1 - stats.chi2.cdf(lr_cc, df=2))

    return {
        "statistic_cc": lr_cc,
        "statistic_ind": float(lr_ind),
        "p_value": p_val,
        "p01": float(p01),         # P(violation today | no violation yesterday)
        "p11": float(p11),         # P(violation today | violation yesterday)
        "reject_H0": p_val < 0.05,
    }


# ===========================================================================
# Diebold-Mariano predictive accuracy test
# ===========================================================================

def diebold_mariano_test(
    e1: np.ndarray | pd.Series,
    e2: np.ndarray | pd.Series,
    loss: str = "squared",
) -> dict:
    """
    Diebold-Mariano (1995) test of equal predictive accuracy.

    H0: E[d_t] = 0  where d_t = L(ê1_t) − L(ê2_t).

    Negative DM statistic → model 1 is more accurate (lower loss).

    Parameters
    ----------
    e1, e2 : forecast error arrays  (realized − forecast)
    loss   : 'squared' (MSE) or 'absolute' (MAE)
    """
    a1, a2 = np.asarray(e1), np.asarray(e2)
    if loss == "squared":
        d = a1 ** 2 - a2 ** 2
    else:
        d = np.abs(a1) - np.abs(a2)

    T = len(d)
    d_bar = float(np.mean(d))

    # Newey-West HAC variance with 1 lag
    gamma0 = float(np.var(d, ddof=1))
    gamma1 = float(np.cov(d[:-1], d[1:])[0, 1]) if T > 2 else 0.0
    var_d = (gamma0 + 2 * gamma1) / T

    if var_d <= 0:
        return {"statistic": np.nan, "p_value": np.nan, "reject_H0": np.nan}

    dm = d_bar / np.sqrt(var_d)
    p_val = float(2 * (1 - stats.norm.cdf(abs(dm))))

    return {
        "statistic": float(dm),
        "p_value": p_val,
        "reject_H0": p_val < 0.05,
        "direction": "model1_better" if dm < 0 else "model2_better",
    }


# ===========================================================================
# Full per-model evaluation
# ===========================================================================

def evaluate_model(
    name: str,
    realized: pd.Series,
    forecasts: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Run all backtesting metrics for one model.

    Returns a dict suitable for building a summary DataFrame.
    """
    viol, vrate = compute_violations(realized, forecasts, alpha)
    kupiec  = kupiec_pof_test(viol, alpha)
    chris   = christoffersen_cc_test(viol, alpha)

    # Forecast errors on common sample
    common = realized.index.intersection(forecasts.dropna().index)
    r = realized.loc[common]
    f = forecasts.loc[common]
    errors = (r - f).values

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Quantile score (tick loss):  QL = (α − 1{r > f})·(r − f)
    ql = float(np.mean((alpha - (r > f).astype(float)) * (r - f)))

    return {
        "model":           name,
        "n_obs":           len(common),
        "viol_rate":       round(vrate, 4),
        "expected_rate":   alpha,
        "kupiec_stat":     round(kupiec["statistic"], 3) if not np.isnan(kupiec["statistic"]) else np.nan,
        "kupiec_pval":     round(kupiec["p_value"], 4) if not np.isnan(kupiec["p_value"]) else np.nan,
        "kupiec_reject":   kupiec["reject_H0"],
        "cc_stat":         round(chris.get("statistic_cc", np.nan), 3),
        "cc_pval":         round(chris.get("p_value", np.nan), 4),
        "cc_reject":       chris.get("reject_H0", np.nan),
        "p11_cluster":     round(chris.get("p11", np.nan), 4),
        "mae":             round(mae, 6),
        "rmse":            round(rmse, 6),
        "quantile_score":  round(ql, 6),
    }


def compare_models(
    model_dict: dict,
    realized: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluate all models and return a summary DataFrame.

    Parameters
    ----------
    model_dict : {model_name: cvar_forecast_series}
    realized   : portfolio loss series (positive = loss)
    """
    rows = []
    for name, forecasts in model_dict.items():
        rows.append(evaluate_model(name, realized, forecasts, alpha))
    return pd.DataFrame(rows).set_index("model")


# ===========================================================================
# Stress-period violation analysis
# ===========================================================================

def analyse_stress_periods(
    violations_dict: dict,
    stress_periods: dict,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute violation rate during each stress period for each model.

    Parameters
    ----------
    violations_dict : {model_name: boolean pd.Series}
    stress_periods  : {period_name: (start_date, end_date)}
    index           : DatetimeIndex of the full sample

    Returns
    -------
    pd.DataFrame  rows=models, columns=stress period names + 'full_sample'
    """
    rows = {}
    for model_name, viol in violations_dict.items():
        row = {"full_sample": float(viol.mean())}
        for period_name, (s, e) in stress_periods.items():
            mask = (viol.index >= s) & (viol.index <= e)
            sub = viol.loc[mask]
            row[period_name] = float(sub.mean()) if len(sub) > 0 else np.nan
        rows[model_name] = row

    return pd.DataFrame(rows).T


# ===========================================================================
# Internal helper
# ===========================================================================

def _empty_test(alpha: float) -> dict:
    return {
        "statistic": np.nan, "p_value": np.nan,
        "violations": 0, "total": 0,
        "viol_rate": np.nan, "expected_rate": alpha,
        "reject_H0": np.nan,
    }
