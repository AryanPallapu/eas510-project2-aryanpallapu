"""Ablation: AI4I models with vs without the engineered features (5-fold CV, training set).
Output: results/full/ai4i_feature_ablation.csv

    python scripts/ablation_ai4i_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from src.evaluate import compute_metrics, cv_evaluate, select_threshold  # noqa: E402
from src.models import build_pipeline  # noqa: E402
from src.preprocessing import ai4i_split  # noqa: E402

if __name__ == "__main__":
    X, _, y, _ = ai4i_split()
    y = y.to_numpy()
    rows = []
    for algo in ["logreg", "dtree", "rf", "gb"]:
        for add in [False, True]:
            pipe = build_pipeline("ai4i", algo, "none", {"fe__add_features": add})
            r = cv_evaluate(pipe, X, y)
            oof = r.pop("oof")
            m = compute_metrics(y, oof, select_threshold(y, oof, "f1"))
            rows.append({"algo": algo, "engineered_features": add, "oof_pr_auc": r["oof_pr_auc"],
                         "cv_pr_auc_std": r["cv_pr_auc_std"], "oof_f1@best_thr": m["f1"],
                         "oof_recall@best_thr": m["recall"]})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "ai4i_feature_ablation.csv", index=False)
