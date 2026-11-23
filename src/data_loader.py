"""Raw-data loaders. These only parse files; all preprocessing lives in
preprocessing.py so it can be fit inside CV folds without leakage."""
from __future__ import annotations

import pandas as pd

import config as C


def load_ai4i(path=None) -> pd.DataFrame:
    path = path or C.AI4I_RAW  # resolved at call time (sample vs full mode)
    # File starts with a UTF-8 BOM; utf-8-sig strips it from the first column name.
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_aps(path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=C.APS_HEADER_LINES, na_values=C.APS_NA_TOKEN)
    df[C.APS_TARGET] = (df[C.APS_TARGET] == "pos").astype("int8")
    return df


def load_aps() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the official (train, test) split. The test set is never merged
    with training data and is only used for final evaluation."""
    return _load_aps(C.APS_TRAIN_RAW), _load_aps(C.APS_TEST_RAW)
