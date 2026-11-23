"""Feature engineering transformers (sklearn-compatible, fit on training folds only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config as C

AIR, PROC, RPM, TORQUE, WEAR = C.AI4I_NUMERIC
AI4I_ENGINEERED = ["temp_diff [K]", "power [W]", "strain [Nm*min]"]


class AI4IFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds physically motivated features.

    temp_diff = process - air temperature   (heat dissipation)
    power     = torque * angular velocity   (power failure)
    strain    = torque * tool wear          (overstrain)
    All are row-wise formulas, so nothing is learned from data.
    """

    def __init__(self, add_features: bool = True):
        self.add_features = add_features

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        if self.add_features:
            X[AI4I_ENGINEERED[0]] = X[PROC] - X[AIR]
            X[AI4I_ENGINEERED[1]] = X[TORQUE] * X[RPM] * 2 * np.pi / 60
            X[AI4I_ENGINEERED[2]] = X[TORQUE] * X[WEAR]
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        return np.asarray(names + (AI4I_ENGINEERED if self.add_features else []), dtype=object)


class APSCleaner(BaseEstimator, TransformerMixin):
    """Column-level cleaning for APS, learned from the training fold only.

    - drops constant columns
    - optionally drops columns whose missing fraction exceeds `max_missing`
    - flags the ac_000 sentinel code (~0x7F000000) and converts it to NaN
    - optionally adds a per-row count of missing values
    - optionally adds per-histogram totals (sum of the 10 bins)
    """

    def __init__(self, sentinel_flag: bool = True, missing_count: bool = True,
                 max_missing: float | None = None, hist_totals: bool = False):
        self.sentinel_flag = sentinel_flag
        self.missing_count = missing_count
        self.max_missing = max_missing
        self.hist_totals = hist_totals

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        nun = X.nunique(dropna=True)
        drop = set(nun[nun <= 1].index)
        if self.max_missing is not None:
            miss = X.isna().mean()
            drop |= set(miss[miss > self.max_missing].index)
        self.dropped_ = sorted(drop)
        self.keep_ = [c for c in X.columns if c not in drop]
        prefixes = pd.Series([c.split("_")[0] for c in X.columns])
        counts = prefixes.value_counts()
        self.hist_groups_ = {
            g: [c for c in X.columns if c.startswith(g + "_")] for g in counts[counts > 1].index
        }
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        extra = {}
        if self.missing_count:
            extra["n_missing"] = X.isna().sum(axis=1).astype(float)
        col = C.APS_SENTINEL_COL
        is_sent = X[col] >= C.APS_SENTINEL_MIN
        if self.sentinel_flag:
            extra[f"{col}_is_sentinel"] = is_sent.astype(float)
        X.loc[is_sent, col] = np.nan
        if self.hist_totals:
            for g, cols in self.hist_groups_.items():
                extra[f"{g}_total"] = X[cols].sum(axis=1, min_count=1)
        out = X[self.keep_]
        if extra:
            out = pd.concat([out, pd.DataFrame(extra, index=X.index)], axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.keep_)
        if self.missing_count:
            names.append("n_missing")
        if self.sentinel_flag:
            names.append(f"{C.APS_SENTINEL_COL}_is_sentinel")
        if self.hist_totals:
            names += [f"{g}_total" for g in self.hist_groups_]
        return np.asarray(names, dtype=object)
