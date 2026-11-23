"""Feature importance and explainability: impurity, permutation and SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config as C
from .models import transform_for_model


def impurity_importance(pipe, X_sample) -> pd.Series | None:
    clf = pipe.steps[-1][1]
    if not hasattr(clf, "feature_importances_"):
        return None
    cols = transform_for_model(pipe, X_sample.head(5)).columns
    return pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)


def permutation(pipe, X, y, n_repeats=5, max_rows=None) -> pd.DataFrame:
    """Permutation importance of the *raw input* columns on held-out data, scored by
    average precision. Engineered features are not permuted separately: permuting a raw
    sensor also changes every feature derived from it."""
    if max_rows and len(X) > max_rows:
        rng = np.random.default_rng(C.RANDOM_STATE)
        pos = np.flatnonzero(np.asarray(y) == 1)
        neg = rng.choice(np.flatnonzero(np.asarray(y) == 0), max_rows - len(pos), replace=False)
        idx = np.sort(np.concatenate([pos, neg]))
        X, y = X.iloc[idx], np.asarray(y)[idx]
    r = permutation_importance(pipe, X, y, scoring="average_precision", n_repeats=n_repeats,
                               random_state=C.RANDOM_STATE, n_jobs=1)
    return pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std},
                        index=X.columns).sort_values("mean", ascending=False)


def shap_values(pipe, X_sample):
    """TreeSHAP on the fitted tree model, computed on model-input features.
    Returns (values for the failure class, transformed X)."""
    import shap
    Xt = transform_for_model(pipe, X_sample)
    explainer = shap.TreeExplainer(pipe.steps[-1][1])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xt
