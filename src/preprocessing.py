"""Dataset preparation and preprocessing pipelines.

Leakage rules enforced here:
* AI4I failure-subtype columns and identifiers never reach the model.
* Every learned transformation (imputation, scaling, column dropping) is an
  sklearn step, so it is re-fit inside each CV fold and on the training set only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config as C
from .data_loader import load_ai4i, load_aps
from .feature_engineering import AI4IFeatureEngineer, APSCleaner


# ---------------------------------------------------------------- AI4I
def ai4i_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=C.AI4I_ID_COLS + C.AI4I_LEAKAGE_COLS + [C.AI4I_TARGET])
    return X, df[C.AI4I_TARGET].astype(int)


def ai4i_split():
    """Stratified 80/20 hold-out split (AI4I has no official split)."""
    X, y = ai4i_xy(load_ai4i())
    return train_test_split(X, y, test_size=C.AI4I_TEST_SIZE, stratify=y,
                            random_state=C.RANDOM_STATE)


def ai4i_preprocessor(add_features: bool = True) -> Pipeline:
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.AI4I_CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("fe", AI4IFeatureEngineer(add_features)), ("ct", ct)]).set_output(transform="pandas")


# ---------------------------------------------------------------- APS
def aps_xy():
    tr, te = load_aps()
    return (tr.drop(columns=C.APS_TARGET), tr[C.APS_TARGET].astype(int),
            te.drop(columns=C.APS_TARGET), te[C.APS_TARGET].astype(int))


# Final settings, chosen by the imputation comparison (results/full/aps_imputation_comparison.csv)
APS_DEFAULTS = dict(imputer="median", sentinel_flag=True, missing_count=True,
                    max_missing=None, hist_totals=False)


def aps_preprocessor(imputer: str = "median", sentinel_flag: bool = True,
                     missing_count: bool = True, max_missing: float | None = None,
                     hist_totals: bool = False, impute: bool = True) -> Pipeline:
    """APS cleaning -> imputation -> log1p (heavy-tailed counters) -> standardisation.

    Trees are invariant to log/scale; LR and KNN need them. `impute=False` keeps NaN
    for models with native missing-value support (HistGradientBoosting)."""
    steps = [("clean", APSCleaner(sentinel_flag, missing_count, max_missing, hist_totals))]
    if impute:
        steps.append(("impute", SimpleImputer(strategy=imputer)))
    steps += [
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ]
    return Pipeline(steps).set_output(transform="pandas")
