"""
copula_models.py
----------------
Copula-based multivariate tail-dependence modelling for the paper.

Classes
-------
GaussianCopula
    Standard normal copula — zero tail dependence; useful as a benchmark.

StudentTCopula
    Symmetric tail dependence governed by degrees-of-freedom ν and correlation ρ.
    Preferred over Gaussian when joint extreme losses are of interest.

ClaytonCopula  (bivariate)
    Strong lower-tail dependence (λ_L = 2^{-1/θ}), zero upper-tail dependence.
    Captures joint crash risk between two assets.

CopulaCVaR
    Multivariate portfolio CVaR via Monte Carlo:
      1. Fit EVT marginals for each asset.
      2. Fit a copula to the pseudo-observations.
      3. Simulate joint scenarios → compute portfolio losses → CVaR.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ===========================================================================
# Probability Integral Transform  →  pseudo-observations in (0,1)
# ===========================================================================

def pseudo_observations(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Transform each marginal to uniform (0,1) using the empirical CDF rank.
    Scaled by n/(n+1) to avoid boundary values 0 and 1.
    """
    n = len(returns)
    u = returns.rank(method="average") / (n + 1)
    return u


def _normal_scores(u: np.ndarray) -> np.ndarray:
    """Φ^{-1}(u)  — clipped to avoid ±∞."""
    return stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


def _t_scores(u: np.ndarray, nu: float) -> np.ndarray:
    """t_{ν}^{-1}(u) — clipped."""
    return stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=nu)


# ===========================================================================
# 1. Gaussian Copula
# ===========================================================================

class GaussianCopula:
    """
    Gaussian copula C(u) = Φ_Σ(Φ^{-1}(u_1), …, Φ^{-1}(u_d)).

    Properties
    ----------
    - Zero upper and lower tail dependence.
    - Fully characterised by the linear correlation matrix Σ.
    """

    def __init__(self):
        self.corr_: np.ndarray | None = None
        self._d: int = 0

    def fit(self, u: pd.DataFrame | np.ndarray):
        arr = u.values if isinstance(u, pd.DataFrame) else np.asarray(u)
        z = _normal_scores(arr)
        self.corr_ = np.corrcoef(z.T)
        self._d = arr.shape[1]
        return self

    def sample(self, n: int) -> np.ndarray:
        """Return (n, d) array of uniform samples."""
        z = np.random.multivariate_normal(np.zeros(self._d), self.corr_, size=n)
        return stats.norm.cdf(z)

    def log_density(self, u: np.ndarray) -> np.ndarray:
        """Log copula density at each row of u."""
        z = _normal_scores(u)
        d = z.shape[1]
        corr_inv = np.linalg.inv(self.corr_)
        sign, logdet = np.linalg.slogdet(self.corr_)
        quad = np.einsum("ti,ij,tj->t", z, corr_inv - np.eye(d), z)
        return -0.5 * (logdet + quad)

    @staticmethod
    def tail_dependence() -> dict:
        return {"lower": 0.0, "upper": 0.0}


# ===========================================================================
# 2. Student-t Copula
# ===========================================================================

class StudentTCopula:
    """
    Student-t copula with ν degrees of freedom and correlation matrix Σ.

    Properties
    ----------
    - Symmetric upper and lower tail dependence:
        λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))
    - As ν → ∞, reduces to Gaussian copula.
    - Calibrated via maximum likelihood (profile MLE over ν, then estimate Σ).
    """

    def __init__(self, nu: float = 5.0):
        self.nu_: float = nu
        self.corr_: np.ndarray | None = None
        self._d: int = 0

    # ------------------------------------------------------------------
    def fit(self, u: pd.DataFrame | np.ndarray, estimate_nu: bool = True):
        arr = u.values if isinstance(u, pd.DataFrame) else np.asarray(u)
        T, d = arr.shape
        self._d = d

        if estimate_nu:
            self.nu_ = self._estimate_nu(arr)

        t = _t_scores(arr, self.nu_)
        self.corr_ = np.corrcoef(t.T)
        return self

    def _estimate_nu(self, u: np.ndarray, nu_grid: np.ndarray | None = None) -> float:
        """Profile MLE: grid search over ν ∈ [2.5, 40]."""
        if nu_grid is None:
            nu_grid = np.array([2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40])

        d = u.shape[1]
        best_nu, best_ll = 5.0, -np.inf

        for nu in nu_grid:
            t = _t_scores(u, nu)
            corr = np.corrcoef(t.T)
            try:
                ll = self._log_likelihood(u, corr, nu)
                if ll > best_ll:
                    best_ll, best_nu = ll, nu
            except np.linalg.LinAlgError:
                continue

        # Fine-tune with bounded optimisation around best grid point
        res = minimize(
            lambda x: -self._log_likelihood(u, np.corrcoef(_t_scores(u, x[0]).T), x[0]),
            x0=[best_nu],
            bounds=[(2.1, 50.0)],
            method="L-BFGS-B",
        )
        return float(res.x[0])

    def _log_likelihood(self, u: np.ndarray, corr: np.ndarray, nu: float) -> float:
        T, d = u.shape
        t = _t_scores(u, nu)
        try:
            corr_inv = np.linalg.inv(corr)
            sign, logdet = np.linalg.slogdet(corr)
            if sign <= 0:
                return -np.inf
        except np.linalg.LinAlgError:
            return -np.inf

        quad = np.einsum("ti,ij,tj->t", t, corr_inv, t)

        # Multivariate t copula log-density (per observation)
        ll_mv = (
            special.gammaln((nu + d) / 2)
            - special.gammaln(nu / 2)
            - (d / 2) * np.log(nu * np.pi)
            - 0.5 * logdet
            - ((nu + d) / 2) * np.log(1 + quad / nu)
        )
        # Subtract marginal t log-densities
        ll_marg = np.sum(stats.t.logpdf(t, df=nu), axis=1)
        return float(np.sum(ll_mv - ll_marg))

    # ------------------------------------------------------------------
    def sample(self, n: int) -> np.ndarray:
        """Return (n, d) uniform samples from the t-copula."""
        z = np.random.multivariate_normal(np.zeros(self._d), self.corr_, size=n)
        chi2 = np.random.chisquare(self.nu_, size=n)
        t = z / np.sqrt(chi2[:, None] / self.nu_)
        return stats.t.cdf(t, df=self.nu_)

    def tail_dependence(self, rho: float | None = None) -> dict:
        """
        Bivariate tail dependence coefficient.
        Uses off-diagonal rho from corr_ if not specified.
        """
        if rho is None:
            rho = self.corr_[0, 1] if self._d >= 2 else 0.0
        arg = -np.sqrt((self.nu_ + 1) * (1 - rho) / (1 + rho + 1e-8))
        lam = 2 * stats.t.cdf(arg, df=self.nu_ + 1)
        return {"lower": float(lam), "upper": float(lam)}

    def pairwise_tail_dependence(self) -> pd.DataFrame:
        """Return (d x d) matrix of bivariate tail dependence coefficients."""
        d = self._d
        td = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i == j:
                    td[i, j] = 1.0
                else:
                    rho = self.corr_[i, j]
                    td[i, j] = self.tail_dependence(rho)["lower"]
        return pd.DataFrame(td)


# ===========================================================================
# 3. Clayton Copula  (bivariate, lower-tail dependence)
# ===========================================================================

class ClaytonCopula:
    """
    Bivariate Clayton copula:
        C(u,v;θ) = (u^{-θ} + v^{-θ} − 1)^{-1/θ},  θ > 0

    Lower tail dependence: λ_L = 2^{-1/θ}
    Upper tail dependence: λ_U = 0
    """

    def __init__(self, theta: float = 2.0):
        self.theta_: float = theta

    def fit(self, u: pd.DataFrame | np.ndarray):
        arr = u.values if isinstance(u, pd.DataFrame) else np.asarray(u)
        if arr.shape[1] != 2:
            raise ValueError("ClaytonCopula is bivariate only.")

        u1, u2 = arr[:, 0], arr[:, 1]

        def _neg_ll(theta):
            th = float(theta[0])
            if th <= 0:
                return 1e10
            # Bivariate Clayton log-density
            ll = (
                np.log(1 + th)
                - (1 + th) * (np.log(u1) + np.log(u2))
                - (2 + 1 / th) * np.log(u1 ** (-th) + u2 ** (-th) - 1)
            )
            return float(-np.sum(ll[np.isfinite(ll)]))

        res = minimize(_neg_ll, x0=[self.theta_], bounds=[(0.01, 50)], method="L-BFGS-B")
        self.theta_ = float(res.x[0])
        return self

    def sample(self, n: int) -> np.ndarray:
        """Conditional inversion sampling."""
        th = self.theta_
        u1 = np.random.uniform(size=n)
        v  = np.random.uniform(size=n)
        # Conditional quantile  F_{U2|U1}(u2|u1) = v
        u2 = (v ** (-th / (1 + th)) - 1 + u1 ** (-th)) ** (-1 / th)
        u2 = np.clip(u2, 1e-6, 1 - 1e-6)
        return np.column_stack([u1, u2])

    def tail_dependence(self) -> dict:
        return {"lower": float(2 ** (-1 / self.theta_)), "upper": 0.0}

    def kendall_tau(self) -> float:
        """Kendall's τ = θ / (θ + 2)."""
        return self.theta_ / (self.theta_ + 2)


# ===========================================================================
# 4. Copula-based Portfolio CVaR via Monte Carlo
# ===========================================================================

class CopulaCVaR:
    """
    Multivariate portfolio CVaR combining EVT marginals with a copula.

    Pipeline
    --------
    fit(returns):
        1. Fit EVT marginal to each asset's loss distribution.
        2. Build pseudo-observations (uniform marginals).
        3. Fit the chosen copula to the uniform data.

    cvar(weights, alpha):
        1. Simulate N joint scenarios from the copula.
        2. Invert each marginal CDF to obtain simulated losses.
        3. Compute portfolio loss = weights @ asset_losses.
        4. Return CVaR = E[portfolio_loss | loss > VaR_α].

    Parameters
    ----------
    copula_type : 'gaussian', 't', 'clayton'
    alpha       : tail probability (default 0.05 → 95% CVaR)
    n_sim       : Monte Carlo sample size
    """

    def __init__(self, copula_type: str = "t", alpha: float = 0.05,
                 n_sim: int = 100_000):
        self.copula_type = copula_type
        self.alpha = alpha
        self.n_sim = n_sim
        self._copula = None
        self._evt: dict = {}
        self._emp_losses: dict = {}
        self._assets: list[str] = []

    # ------------------------------------------------------------------
    def fit(self, returns: pd.DataFrame):
        from src.risk_models import EVTModel

        self._assets = list(returns.columns)
        d = len(self._assets)

        # --- Fit EVT marginals ---
        for col in self._assets:
            L = -returns[col].values
            self._emp_losses[col] = np.sort(L)
            evt = EVTModel(threshold_q=0.85)
            evt.fit(L)
            self._evt[col] = evt

        # --- Pseudo-observations ---
        u = pseudo_observations(returns)

        # --- Fit copula ---
        if self.copula_type == "gaussian":
            self._copula = GaussianCopula()
            self._copula.fit(u)
        elif self.copula_type == "t":
            self._copula = StudentTCopula()
            self._copula.fit(u)
        elif self.copula_type == "clayton":
            if d != 2:
                # Fall back to Gaussian for high-dim
                self._copula = GaussianCopula()
                self._copula.fit(u)
            else:
                self._copula = ClaytonCopula()
                self._copula.fit(u.values)
        else:
            raise ValueError(f"Unknown copula_type: {self.copula_type}")

        return self

    # ------------------------------------------------------------------
    def simulate_losses(self, weights: np.ndarray | None = None) -> np.ndarray:
        """
        Return simulated portfolio loss vector of length n_sim.
        """
        d = len(self._assets)
        if weights is None:
            weights = np.ones(d) / d
        weights = np.asarray(weights)

        # Sample copula
        u_sim = self._copula.sample(self.n_sim)     # (n_sim, d)

        # Invert marginals
        asset_losses = np.zeros((self.n_sim, d))
        for j, col in enumerate(self._assets):
            evt = self._evt[col]
            uj = u_sim[:, j % u_sim.shape[1]]        # handle bivariate copula

            # P(L > u) = p_u  →  tail region: uj > 1 - p_u
            tail_mask = uj > (1 - evt.p_u_)

            lj = np.zeros(self.n_sim)

            # Tail: invert GPD via VaR formula
            if np.any(tail_mask):
                alpha_j = 1 - uj[tail_mask]          # small tail prob
                alpha_j = np.clip(alpha_j, 1e-8, evt.p_u_ - 1e-8)
                if abs(evt.xi_) < 1e-6:               # ξ ≈ 0 → Exponential
                    lj[tail_mask] = evt.u_ - evt.beta_ * np.log(alpha_j / evt.p_u_)
                else:
                    lj[tail_mask] = (
                        evt.u_
                        + (evt.beta_ / evt.xi_)
                        * ((evt.p_u_ / alpha_j) ** evt.xi_ - 1)
                    )

            # Body: empirical quantile
            if np.any(~tail_mask):
                emp = self._emp_losses[col]
                q_idx = np.clip(
                    (uj[~tail_mask] * len(emp)).astype(int), 0, len(emp) - 1
                )
                lj[~tail_mask] = emp[q_idx]

            asset_losses[:, j] = lj

        return asset_losses @ weights

    # ------------------------------------------------------------------
    def cvar(self, weights: np.ndarray | None = None) -> float:
        """Portfolio CVaR at level α."""
        port_losses = self.simulate_losses(weights)
        q = np.quantile(port_losses, 1 - self.alpha)
        return float(port_losses[port_losses >= q].mean())

    def var(self, weights: np.ndarray | None = None) -> float:
        """Portfolio VaR at level α."""
        port_losses = self.simulate_losses(weights)
        return float(np.quantile(port_losses, 1 - self.alpha))

    def tail_dependence_summary(self) -> dict:
        """Return tail dependence dict from the fitted copula if available."""
        if hasattr(self._copula, "tail_dependence"):
            return self._copula.tail_dependence()
        return {}
