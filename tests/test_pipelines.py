"""Leakage and pipeline-correctness tests."""
import numpy as np
import pandas as pd
import pytest

import config as C
from src.feature_engineering import APSCleaner
from src.models import ALGOS, STRATEGIES, build_pipeline, supports
from src.preprocessing import ai4i_split, aps_preprocessor


@pytest.fixture(autouse=True)
def sample_mode():
    C.use_mode("sample")  # bundled data: tests run without downloads
    yield
    C.use_mode("full")


def test_ai4i_inputs_exclude_leakage():
    X_tr, X_te, y_tr, y_te = ai4i_split()
    banned = set(C.AI4I_LEAKAGE_COLS + C.AI4I_ID_COLS + [C.AI4I_TARGET])
    assert not banned & set(X_tr.columns)
    assert len(X_te) == 2000 and abs(y_tr.mean() - y_te.mean()) < 0.002  # stratified
    assert not set(X_tr.index) & set(X_te.index)


def test_aps_cleaner_learns_from_fit_data_only():
    tr = pd.DataFrame({"ac_000": [1.0, C.APS_SENTINEL_MIN, 3.0], "b_000": [np.nan, np.nan, 1.0],
                       "k_000": [5.0, 5.0, 5.0]})
    cl = APSCleaner(max_missing=0.5).fit(tr)
    assert cl.dropped_ == ["b_000", "k_000"]
    out = cl.transform(tr)
    assert np.isnan(out.loc[1, "ac_000"]) and out["ac_000_is_sentinel"].tolist() == [0, 1, 0]
    assert out["n_missing"].tolist() == [1, 1, 0]


def test_aps_imputer_statistics_come_from_training_rows():
    tr = pd.DataFrame({"ac_000": [1.0, 2.0, np.nan], "x_000": [1.0, 3.0, np.nan]})
    te = pd.DataFrame({"ac_000": [np.nan], "x_000": [1000.0]})
    pre = aps_preprocessor().fit(tr)
    assert pre.named_steps["impute"].statistics_[1] == 2.0  # median of train x_000
    assert pre.transform(te).shape[0] == 1


@pytest.mark.parametrize("strategy", ["ros", "smote"])
def test_resampling_only_affects_fit_data(strategy):
    X_tr, X_te, y_tr, _ = ai4i_split()
    pipe = build_pipeline("ai4i", "dtree", strategy).fit(X_tr.iloc[:3000], y_tr.iloc[:3000])
    # predictions are produced for exactly the rows passed in: samplers are fit-only
    assert len(pipe.predict_proba(X_te)) == len(X_te)
    Xs, ys = pipe.named_steps["sampler"].fit_resample(X_tr.iloc[:3000], y_tr.iloc[:3000])
    assert ys.mean() == pytest.approx(0.5)


def test_all_pipelines_build():
    for ds in ["ai4i", "aps"]:
        for a in ALGOS:
            for s in STRATEGIES:
                if supports(a, s):
                    build_pipeline(ds, a, s)
