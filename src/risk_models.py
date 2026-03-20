"""
risk_models.py
--------------
CVaR estimation methods compared in the paper:

Baselines
  1. StaticHistoricalCVaR      – full-sample historical simulation
  2. RollingHistoricalCVaR     – sliding-window historical simulation
  3. GARCHCVaR                 – GARCH(1,1) conditional variance + normal tail

Proposed
  4. EVTModel                  – Peaks-over-Threshold GPD tail fit
  5. RegimeConditionalCVaR     – per-regime historical or EVT CVaR
     (the main methodological contribution of the paper)

All CVaR estimates are expressed as positive numbers (loss convention).
CVaR at level α means: expected loss given loss exceeds the (1-α)-VaR.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

ALPHA_DEFAULT = 0.05      # 95 % CVaR


# ===========================================================================
# Utility
# ===========================================================================

def losses(returns: pd.Series) -> pd.Series:
    """Flip signs: losses are positive."""
    return -returns


# ===========================================================================
# 1. Static Full-Sample Historical CVaR
# ===========================================================================

class StaticHistoricalCVaR:
    """
    Fit once on the full in-sample period; produce a constant CVaR estimate.
    Represents the 'naive' baseline that ignores regime shifts entirely.
    """

    def __init__(self, alpha: float = ALPHA_DEFAULT):
        self.alpha = alpha
        self.cvar_: float | None = None
        self.var_: float | None = None

    def fit(self, returns: pd.Series):
        L = losses(returns).values
        q = np.quantile(L, 1 - self.alpha)
        self.var_ = q
        self.cvar_ = L[L >= q].mean()
        return self

    def forecast(self, index: pd.DatetimeIndex) -> pd.Series:
        """Return constant CVaR broadcast over the given index."""
        return pd.Series(self.cvar_, index=index, name="static_cvar")


# ===========================================================================
# 2. Rolling-Window Historical CVaR
# ===========================================================================

class RollingHistoricalCVaR:
    """
    Sliding-window historical simulation. Updates parameters as new data
    arrives but does not explicitly model regime structure.

    Parameters
    ----------
    window : int   trading days in look-back window (default 252 = 1 year)
    alpha  : float tail probability
    """

    def __init__(self, window: int = 252, alpha: float = ALPHA_DEFAULT):
        self.window = window
        self.alpha = alpha
        self.forecasts_: pd.Series | None = None

    def fit(self, returns: pd.Series):
        L = losses(returns)

        def _cvar(x: np.ndarray) -> float:
            q = np.quantile(x, 1 - self.alpha)
            tail = x[x >= q]
            return float(tail.mean()) if len(tail) > 0 else float(q)

        self.forecasts_ = L.rolling(self.window).apply(_cvar, raw=True)
        self.forecasts_.name = "rolling_cvar"
        return self

    def forecast(self) -> pd.Series:
        return self.forecasts_


# ===========================================================================
# 3. GARCH(1,1) CVaR
# ===========================================================================

class GARCHCVaR:
    """
    GARCH(1,1) with normal innovations.
    Conditional volatility from ARCH library; CVaR derived analytically.

    CVaR_{α} = μ_t + σ_t * φ(z_α) / α   (normal innovations)
    where z_α = Φ^{-1}(1-α), φ is the standard normal PDF.
    """

    def __init__(self, alpha: float = ALPHA_DEFAULT):
        self.alpha = alpha
        self.forecasts_: pd.Series | None = None
        self._res = None

    def fit(self, returns: pd.Series):
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("pip install arch")

        # scale for numerical stability
        L = losses(returns) * 100

        am = arch_model(L, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        self._res = am.fit(disp="off", show_warning=False)

        cond_vol = self._res.conditional_volatility / 100   # unscale
        mu = self._res.params.get("mu", 0.0) / 100

        z_alpha = stats.norm.ppf(1 - self.alpha)
        phi_z   = stats.norm.pdf(z_alpha)
        multiplier = phi_z / self.alpha                      # E[Z | Z > z_α]

        self.forecasts_ = pd.Series(
            mu + cond_vol.values * multiplier,
            index=returns.index,
            name="garch_cvar",
        )
        return self

    def forecast(self) -> pd.Series:
        return self.forecasts_


# ===========================================================================
# 4. EVT Model — Peaks-over-Threshold with GPD
# ===========================================================================

class EVTModel:
    """
    Peaks-over-Threshold (POT) method.

    Models the tail of the loss distribution beyond a high threshold u
    with a Generalised Pareto Distribution (GPD):

        F_u(x) = 1 − (1 + ξ·x/β)^{−1/ξ}    x > 0, ξ ≠ 0

    GPD parameters (ξ, β) estimated by MLE.

    CVaR formula (McNeil & Frey 2000), valid for ξ < 1:
        VaR_p  = u + (β/ξ)·[(p_u/α)^ξ − 1]
        CVaR_p = (VaR_p + β − ξ·u) / (1 − ξ)

    where p_u = n_u / n  (empirical exceedance probability above u).
    """

    def __init__(self, threshold_q: float = 0.90):
        """
        threshold_q : quantile used to set threshold u
                      (e.g. 0.90 → u = 90th-percentile of losses)
        """
        self.threshold_q = threshold_q
        self.xi_: float | None = None      # shape
        self.beta_: float | None = None    # scale
        self.u_: float | None = None       # threshold level
        self.p_u_: float | None = None     # P(L > u)

    # ------------------------------------------------------------------
    def fit(self, L: np.ndarray | pd.Series):
        """Fit GPD to loss exceedances above threshold."""
        L = np.asarray(L, dtype=float)
        self.u_ = float(np.quantile(L, self.threshold_q))
        exceed = L[L > self.u_] - self.u_
        n_u = len(exceed)
        n   = len(L)
        self.p_u_ = n_u / n

        if n_u < 10:
            # Not enough tail data — fall back to empirical
            self.xi_, self.beta_ = 0.0, float(np.std(exceed)) + 1e-8
            return self

        def _neg_ll(params):
            xi, beta = params
            if beta <= 0:
                return 1e10
            if xi == 0:
                return n_u * np.log(beta) + np.sum(exceed) / beta
            arg = 1 + xi * exceed / beta
            if np.any(arg <= 0):
                return 1e10
            return n_u * np.log(beta) + (1 + 1 / xi) * np.sum(np.log(arg))

        res = minimize(
            _neg_ll,
            x0=[0.1, np.std(exceed)],
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000},
        )
        self.xi_, self.beta_ = float(res.x[0]), float(res.x[1])
        return self

    # ------------------------------------------------------------------
    def var(self, alpha: float = ALPHA_DEFAULT) -> float:
        """VaR at confidence level 1-α."""
        if self.xi_ == 0:
            return self.u_ - self.beta_ * np.log(alpha / self.p_u_)
        return self.u_ + (self.beta_ / self.xi_) * ((self.p_u_ / alpha) ** self.xi_ - 1)

    def cvar(self, alpha: float = ALPHA_DEFAULT) -> float:
        """CVaR at confidence level 1-α (expected shortfall)."""
        v = self.var(alpha)
        if self.xi_ >= 1:
            return np.inf
        return (v + self.beta_ - self.xi_ * self.u_) / (1 - self.xi_)

    def tail_index(self) -> float:
        """Tail index 1/ξ; larger = lighter tail."""
        return 1 / self.xi_ if self.xi_ > 0 else np.inf


# ===========================================================================
# 5. Regime-Conditional CVaR  (main model)
# ===========================================================================

class RegimeConditionalCVaR:
    """
    Estimates CVaR separately within each detected regime.

    Approach
    --------
    For each regime r:
      - Collect all returns assigned to regime r in the training set.
      - Fit either historical simulation ('hist') or EVT/GPD ('evt').
      - Store the regime-specific CVaR estimate.

    At forecast time, look up the estimate for the current regime label.

    Parameters
    ----------
    method  : 'hist' or 'evt'
    alpha   : tail probability
    min_obs : minimum regime observations to attempt fitting;
              if fewer, fall back to pooled historical CVaR.
    """

    def __init__(self, method: str = "evt", alpha: float = ALPHA_DEFAULT,
                 min_obs: int = 63):
        self.method  = method
        self.alpha   = alpha
        self.min_obs = min_obs
        self._regime_cvar: dict[int, float] = {}
        self._regime_var:  dict[int, float] = {}
        self._evt_models:  dict[int, EVTModel] = {}
        self._pooled_cvar: float | None = None

    # ------------------------------------------------------------------
    def fit(self, returns: pd.Series, regime_labels: np.ndarray):
        """
        Parameters
        ----------
        returns       : pd.Series  portfolio returns (in-sample)
        regime_labels : np.ndarray integer regime for each return observation
        """
        L_all = losses(returns).values
        labels = np.asarray(regime_labels, dtype=float)

        # Drop NaN-labelled observations
        valid = ~np.isnan(labels)
        L_all = L_all[valid]
        labels = labels[valid].astype(int)

        # Pooled fallback
        q = np.quantile(L_all, 1 - self.alpha)
        self._pooled_cvar = float(L_all[L_all >= q].mean())

        for r in np.unique(labels):
            mask = labels == r
            L_r = L_all[mask]

            if len(L_r) < self.min_obs:
                self._regime_cvar[int(r)] = self._pooled_cvar
                self._regime_var[int(r)]  = float(np.quantile(L_all, 1 - self.alpha))
                continue

            if self.method == "hist":
                q_r = np.quantile(L_r, 1 - self.alpha)
                self._regime_var[int(r)]  = float(q_r)
                self._regime_cvar[int(r)] = float(L_r[L_r >= q_r].mean())

            elif self.method == "evt":
                try:
                    evt = EVTModel(threshold_q=0.85)
                    evt.fit(L_r)
                    self._evt_models[int(r)] = evt
                    self._regime_var[int(r)]  = evt.var(self.alpha)
                    self._regime_cvar[int(r)] = evt.cvar(self.alpha)
                except Exception:
                    q_r = np.quantile(L_r, 1 - self.alpha)
                    self._regime_var[int(r)]  = float(q_r)
                    self._regime_cvar[int(r)] = float(L_r[L_r >= q_r].mean())

        return self

    # ------------------------------------------------------------------
    def forecast(self, regime_labels, index: pd.DatetimeIndex | None = None) -> pd.Series:
        """
        Produce CVaR forecast for each observation given its regime label.

        Parameters
        ----------
        regime_labels : array-like of int
        index         : optional DatetimeIndex for the returned Series
        """
        labels = np.asarray(regime_labels)
        out = np.array(
            [self._regime_cvar.get(int(r), self._pooled_cvar) for r in labels],
            dtype=float,
        )
        return pd.Series(out, index=index, name=f"regime_{self.method}_cvar")

    def regime_summary(self) -> pd.DataFrame:
        """Return a table of per-regime CVaR (and VaR)."""
        rows = [
            {"regime": r,
             "var":    self._regime_var.get(r, np.nan),
             "cvar":   self._regime_cvar.get(r, np.nan)}
            for r in sorted(self._regime_cvar)
        ]
        return pd.DataFrame(rows).set_index("regime")

    def evt_shape_params(self) -> pd.DataFrame:
        """Return GPD ξ and β for each regime (EVT method only)."""
        rows = [
            {"regime": r,
             "xi":     m.xi_,
             "beta":   m.beta_,
             "threshold": m.u_,
             "tail_index": m.tail_index()}
            for r, m in self._evt_models.items()
        ]
        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


# ===========================================================================
# Walk-forward backtesting helper
# ===========================================================================

def walk_forward_regime_cvar(
    returns: pd.Series,
    regime_labels: pd.Series,
    train_size: int = 756,
    step: int = 63,
    method: str = "evt",
    alpha: float = ALPHA_DEFAULT,
) -> pd.Series:
    """
    Expanding-window walk-forward CVaR forecast.

    At each step, fit RegimeConditionalCVaR on the expanding training window
    [0, t) and forecast CVaR for observations in [t, t+step).  The regime
    label assigned to each forecast day is the *last* regime label seen in
    the training window (i.e., the current regime at the forecast origin).

    Returns
    -------
    pd.Series of out-of-sample CVaR forecasts aligned to returns.index.
    """
    n = len(returns)
    forecasts = pd.Series(np.nan, index=returns.index, name=f"wf_{method}_cvar")

    for t in range(train_size, n, step):
        train_ret = returns.iloc[:t]
        train_lab = regime_labels.iloc[:t]

        # Drop NaN labels and align returns
        valid_mask = train_lab.notna()
        train_ret_v = train_ret.loc[valid_mask.index[valid_mask]]
        train_lab_v = train_lab.loc[valid_mask.index[valid_mask]].values

        if len(train_ret_v) < 30:
            continue

        mdl = RegimeConditionalCVaR(method=method, alpha=alpha)
        mdl.fit(train_ret_v, train_lab_v)

        # Current regime = last non-NaN label seen
        current_regime = int(train_lab_v[-1])
        cvar_est = mdl._regime_cvar.get(current_regime, mdl._pooled_cvar)

        end = min(t + step, n)
        forecasts.iloc[t:end] = cvar_est

    return forecasts
