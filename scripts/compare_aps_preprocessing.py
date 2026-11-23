"""Phase 3: compare APS imputation / cleaning variants with 5-fold CV on the
training set only (test set untouched). Output: results/full/aps_imputation_comparison.csv

    python scripts/compare_aps_preprocessing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import config as C  # noqa: E402
from src.evaluate import compute_metrics, cv_evaluate, select_threshold  # noqa: E402
from src.preprocessing import aps_preprocessor, aps_xy  # noqa: E402

VARIANTS = {
    "mean":                         dict(imputer="mean", sentinel_flag=False, missing_count=False),
    "median":                       dict(imputer="median", sentinel_flag=False, missing_count=False),
    "median+flags":                 dict(imputer="median"),
    "median+flags, drop >70% miss": dict(imputer="median", max_missing=0.7),
    "median+flags+hist totals":     dict(imputer="median", hist_totals=True),
    "native NaN (HistGB only)+flags": dict(impute=False),
}
MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=3000, class_weight="balanced",
                                                      random_state=C.RANDOM_STATE),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        class_weight="balanced", early_stopping=False, random_state=C.RANDOM_STATE),
}

if __name__ == "__main__":
    X, y, _, _ = aps_xy()
    y = y.to_numpy()
    rows = []
    for vname, kw in VARIANTS.items():
        for mname, mk in MODELS.items():
            if not kw.get("impute", True) and mname != "HistGradientBoosting":
                continue
            pipe = ImbPipeline(list(aps_preprocessor(**kw).steps) + [("clf", mk())])
            r = cv_evaluate(pipe, X, y)
            oof = r.pop("oof")
            thr = select_threshold(y, oof, "cost")
            m = compute_metrics(y, oof, thr)
            rows.append({"variant": vname, "model": mname, "oof_pr_auc": r["oof_pr_auc"],
                         "oof_roc_auc": r["oof_roc_auc"], "oof_cost@best_thr": m["cost"],
                         "best_thr": thr, "recall@best_thr": m["recall"],
                         "fit_time_s": r["fit_time_s"]})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "aps_imputation_comparison.csv", index=False)
