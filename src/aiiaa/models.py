"""Tweedie (power 1.5) models of claim cost per unit exposure, all with the same fit/predict interface.

fit(x, rate, weight, valid=None): rate = cost / exposure, weight = exposure (C010).
valid = (x, rate, weight) enables early stopping for the tree models.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

POWER = 1.5
SEED = 42


def _bools_to_int(x):
    return x.astype({c: "int8" for c in x.columns if x[c].dtype == bool})


class Balanced:
    """C013: rescale predictions so total predicted cost equals total actual cost on the training data.

    Log-link Tweedie fits with power 1.5 do not preserve this balance; boosting drifts below it
    (motor LightGBM: 0.83 of actual on its own training rows). The factor uses training data only.
    """

    def fit(self, x, y, w, valid=None):
        self._fit(x, y, w, valid)
        self.balance_ = float(np.sum(y * w) / np.sum(self._raw(x) * w))
        return self

    def predict(self, x):
        return self._raw(x) * self.balance_


class GLM(Balanced):
    """Actuarial-style GLM: numeric columns banded into deciles (missing = own band), all one-hot, log link."""

    def __init__(self, n_bins=10, alpha=1e-4, max_iter=400, **_):
        self.n_bins, self.alpha, self.max_iter = n_bins, alpha, max_iter

    def _band(self, x):
        out = pd.DataFrame(index=x.index)
        for col in x.columns:
            if col in self.edges_:
                band = np.digitize(x[col].astype(float), self.edges_[col])
                out[col] = np.where(x[col].isna(), -1, band).astype(str)
            else:
                out[col] = x[col].astype(str)
        return out

    def _fit(self, x, y, w, valid=None):
        num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and x[c].dtype != bool]
        self.edges_ = {c: np.unique(np.nanquantile(x[c].astype(float), np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                       for c in num}
        self.enc_ = OneHotEncoder(handle_unknown="ignore", min_frequency=100)
        xb = self.enc_.fit_transform(self._band(x))
        self.model_ = TweedieRegressor(power=POWER, link="log", alpha=self.alpha, max_iter=self.max_iter)
        self.model_.fit(xb, y, sample_weight=w)

    def _raw(self, x):
        return self.model_.predict(self.enc_.transform(self._band(x)))


class GAM(Balanced):
    """C011: GAM-style Tweedie model: cubic B-splines (quantile knots) on numeric columns, one-hot categories.

    Smoothness is controlled by a single L2 penalty (alpha) rather than per-term smoothing selection.
    Missing numeric values are imputed with the median and flagged by an indicator column.
    """

    def __init__(self, n_knots=8, alpha=1e-4, max_iter=400, **_):
        self.n_knots, self.alpha, self.max_iter = n_knots, alpha, max_iter

    def _design(self, x, fit=False):
        num = x[self.num_].astype(float)
        missing = num.isna().to_numpy(dtype=float)[:, self.has_missing_]
        num = num.fillna(self.medians_)
        if fit:
            self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3, knots="quantile",
                                             extrapolation="constant", sparse_output=True)
            self.enc_ = OneHotEncoder(handle_unknown="ignore", min_frequency=100)
            s, c = self.spline_.fit_transform(num), self.enc_.fit_transform(x[self.cat_].astype(str))
        else:
            s, c = self.spline_.transform(num), self.enc_.transform(x[self.cat_].astype(str))
        return sparse.hstack([s, c, sparse.csr_matrix(missing)]).tocsr()

    def _fit(self, x, y, w, valid=None):
        self.num_ = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c]) and x[c].dtype != bool
                     and x[c].nunique() > self.n_knots]
        self.cat_ = [c for c in x.columns if c not in self.num_]
        self.medians_ = x[self.num_].astype(float).median()
        self.has_missing_ = x[self.num_].isna().any().to_numpy()
        self.model_ = TweedieRegressor(power=POWER, link="log", alpha=self.alpha, max_iter=self.max_iter)
        self.model_.fit(self._design(x, fit=True), y, sample_weight=w)

    def _raw(self, x):
        return self.model_.predict(self._design(x))


class LGBM(Balanced):
    # C015: shallow trees (7 leaves, >= 3000 rows per leaf); deeper trees early-stopped after ~50 rounds
    def __init__(self, n_estimators=5000, learning_rate=0.05, **params):
        self.model = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=POWER, n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=7, min_child_samples=3000, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, **params)

    def _fit(self, x, y, w, valid=None):
        kw = {}
        if valid is not None:
            xv, yv, wv = valid
            kw = dict(eval_set=[(_bools_to_int(xv), yv)], eval_sample_weight=[wv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        self.model.fit(_bools_to_int(x), y, sample_weight=w, **kw)
        self.best_iteration = self.model.best_iteration_ or self.model.n_estimators

    def _raw(self, x):
        return self.model.predict(_bools_to_int(x))


class XGB(Balanced):
    # C015: depth 3 (up to 8 leaves) to match the shallow LightGBM setting
    def __init__(self, n_estimators=5000, learning_rate=0.05, **params):
        self.params = dict(
            objective="reg:tweedie", tweedie_variance_power=POWER, n_estimators=n_estimators,
            learning_rate=learning_rate, max_depth=3, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, random_state=SEED,
            **params)

    def _fit(self, x, y, w, valid=None):
        kw = {}
        params = dict(self.params)
        if valid is not None:
            xv, yv, wv = valid
            params["early_stopping_rounds"] = 100
            kw = dict(eval_set=[(_bools_to_int(xv), yv)], sample_weight_eval_set=[wv], verbose=False)
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(_bools_to_int(x), y, sample_weight=w, **kw)
        self.best_iteration = (self.model.best_iteration + 1) if valid is not None else params["n_estimators"]

    def _raw(self, x):
        return self.model.predict(_bools_to_int(x))


def make(name, **kw):
    return {"glm": GLM, "gam": GAM, "lgbm": LGBM, "xgb": XGB}[name](**kw)
