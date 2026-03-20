"""
regime_detection.py
-------------------
Two complementary regime-detection methods used in the paper:

1. BayesianChangepointDetector
   Wraps the `ruptures` library's PELT algorithm with a BIC-style penalty.
   Produces a deterministic segmentation of the time series into structural blocks.

2. HMMRegimeDetector
   Gaussian Hidden Markov Model (via hmmlearn) fitted to volatility/return features.
   Produces soft posterior probabilities over latent states and a Viterbi state sequence.
   States are ordered by ascending mean volatility: state 0 = low-vol, state 1 = high-vol.

Helper
------
build_hmm_features(returns, vix, window)
   Constructs a standardised feature matrix suitable for HMM fitting.
"""

import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Bayesian Changepoint Detection via PELT
# ---------------------------------------------------------------------------

class BayesianChangepointDetector:
    """
    Pruned Exact Linear Time (PELT) changepoint detector with BIC penalty.

    Detects structural breaks in mean and/or variance of a univariate or
    multivariate signal.  A higher penalty → fewer breakpoints.

    Parameters
    ----------
    model : str
        Ruptures cost model: 'rbf' (default), 'l2', or 'normal'.
    min_size : int
        Minimum number of samples between breakpoints (~1 trading month).
    pen_scale : float
        Multiplier on the BIC log(n)*d penalty term.
    """

    def __init__(self, model: str = "rbf", min_size: int = 21, pen_scale: float = 1.0):
        self.model = model
        self.min_size = min_size
        self.pen_scale = pen_scale
        self.breakpoints_: list[int] = []          # indices (1-based, includes len)
        self._n: int = 0

    # ------------------------------------------------------------------
    def fit(self, signal, pen: float | None = None):
        """
        Detect changepoints.

        Parameters
        ----------
        signal : array-like (T,) or (T, d)
        pen    : penalty override; if None, uses BIC-style pen_scale * log(T) * d
        """
        if isinstance(signal, (pd.Series, pd.DataFrame)):
            arr = signal.values
        else:
            arr = np.asarray(signal)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        T, d = arr.shape
        self._n = T

        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=1)
        algo.fit(arr)

        if pen is None:
            pen = self.pen_scale * np.log(T) * d

        self.breakpoints_ = algo.predict(pen=pen)   # last element == T
        return self

    # ------------------------------------------------------------------
    def regime_labels(self) -> np.ndarray:
        """Integer regime labels 0, 1, 2, … for each time step."""
        labels = np.empty(self._n, dtype=int)
        prev = 0
        for r, bp in enumerate(self.breakpoints_[:-1]):
            labels[prev:bp] = r
            prev = bp
        labels[prev:] = len(self.breakpoints_) - 1
        return labels

    def n_regimes(self) -> int:
        return len(self.breakpoints_)

    def breakpoint_dates(self, index: pd.DatetimeIndex) -> list:
        """Return the date immediately before each detected breakpoint."""
        return [index[bp - 1] for bp in self.breakpoints_[:-1]]


# ---------------------------------------------------------------------------
# 2. Hidden Markov Model Regime Detector
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """
    Gaussian HMM with n_states latent states.

    States are relabelled so that state 0 = lowest mean volatility
    (low-vol / bull) and state n_states-1 = highest (high-vol / bear).

    Parameters
    ----------
    n_states : int
        Number of latent regimes (2 = bull/bear, 3 = bull/transition/bear).
    n_iter : int
        Maximum EM iterations.
    covariance_type : str
        'full' (default), 'diag', 'tied', or 'spherical'.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200,
                 covariance_type: str = "full"):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self._model: hmm.GaussianHMM | None = None
        self._order: np.ndarray | None = None    # internal → ordered state map

    # ------------------------------------------------------------------
    def fit(self, features):
        """
        Fit HMM to feature matrix.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray  shape (T, n_features)
        """
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)

        self._model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
        )
        self._model.fit(X)

        # Order states by the first feature's mean (assumed to be volatility)
        self._order = np.argsort(self._model.means_[:, 0])
        return self

    # ------------------------------------------------------------------
    def predict(self, features) -> np.ndarray:
        """Viterbi state sequence, ordered by volatility (0=low)."""
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        raw = self._model.predict(X)
        return self._remap(raw)

    def predict_proba(self, features) -> np.ndarray:
        """Posterior state probabilities, columns ordered by volatility."""
        X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        prob = self._model.predict_proba(X)        # (T, n_states)
        return prob[:, self._order]

    def _remap(self, raw: np.ndarray) -> np.ndarray:
        out = np.empty_like(raw)
        for new, old in enumerate(self._order):
            out[raw == old] = new
        return out

    # ------------------------------------------------------------------
    def transition_matrix(self) -> np.ndarray:
        """Transition matrix rows/cols ordered by volatility."""
        T = self._model.transmat_
        idx = self._order
        return T[np.ix_(idx, idx)]

    def regime_stats(self, features, labels) -> pd.DataFrame:
        """Mean and std of each feature per regime."""
        df = pd.DataFrame(features) if not isinstance(features, pd.DataFrame) else features
        df = df.copy()
        df["regime"] = labels
        return df.groupby("regime").agg(["mean", "std"])


# ---------------------------------------------------------------------------
# Feature engineering for HMM
# ---------------------------------------------------------------------------

def build_hmm_features(
    returns: pd.DataFrame,
    vix: pd.Series | None = None,
    window: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Build a standardised feature matrix for HMM fitting.

    Features
    --------
    roll_vol  : annualised rolling realised volatility of eq-weight portfolio
    roll_ret  : annualised rolling mean return
    vix       : VIX level (if provided)

    Returns
    -------
    features_raw    : pd.DataFrame  (unstandardised)
    features_scaled : pd.DataFrame  (zero-mean, unit-variance)
    scaler          : fitted StandardScaler
    """
    port = returns.mean(axis=1)

    raw = pd.DataFrame(index=returns.index)
    raw["roll_vol"] = port.rolling(window).std() * np.sqrt(252)
    raw["roll_ret"] = port.rolling(window).mean() * 252

    if vix is not None:
        raw["vix"] = vix.reindex(returns.index).ffill()

    raw = raw.dropna()

    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(raw.values)
    features_scaled = pd.DataFrame(scaled_vals, index=raw.index, columns=raw.columns)

    return raw, features_scaled, scaler
