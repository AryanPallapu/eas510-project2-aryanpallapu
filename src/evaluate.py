"""Metrics, the Scania cost function, threshold selection and CV evaluation."""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

import config as C


def scania_cost(fp: int, fn: int) -> int:
    return int(C.APS_COST_FP * fp + C.APS_COST_FN * fn)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": scania_cost(fp, fn),
    }


# Finer at the low end: with ~2% positives and a 50:1 cost ratio, cost-optimal thresholds
# for APS sit well below 0.01 (a 0.005-step grid clipped them at its lower edge).
THRESHOLD_GRID = np.unique(np.round(np.concatenate(
    [np.arange(0.0005, 0.01, 0.0005), np.arange(0.01, 1.0, 0.005)]), 4))


def select_threshold(y_true, proba, objective: str = "f1") -> float:
    """Pick the threshold on out-of-fold training predictions (never on test)."""
    y_true = np.asarray(y_true)
    pred = proba[None, :] >= THRESHOLD_GRID[:, None]
    tp = (pred & (y_true == 1)).sum(1)
    fp = (pred & (y_true == 0)).sum(1)
    fn = ((~pred) & (y_true == 1)).sum(1)
    if objective == "f1":
        score = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        return float(THRESHOLD_GRID[np.argmax(score)])
    if objective == "cost":
        cost = C.APS_COST_FP * fp + C.APS_COST_FN * fn
        return float(THRESHOLD_GRID[np.argmin(cost)])
    raise ValueError(objective)


def predict_proba_timed(model, X):
    t = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t


def cv_evaluate(pipeline, X, y, folds: int | None = None) -> dict:
    """Stratified K-fold CV with a fresh clone per fold.

    Returns fold-mean metrics at threshold 0.5, out-of-fold probabilities,
    and mean fit / predict times."""
    folds = folds or C.CV_FOLDS
    skf = StratifiedKFold(folds, shuffle=True, random_state=C.RANDOM_STATE)
    y = np.asarray(y)
    oof = np.zeros(len(y))
    fold_metrics, fit_t, pred_t, pred_n = [], [], [], []
    for tr, va in skf.split(X, y):
        model = clone(pipeline)
        t = time.perf_counter()
        model.fit(X.iloc[tr], y[tr])
        fit_t.append(time.perf_counter() - t)
        p, dt = predict_proba_timed(model, X.iloc[va])
        pred_t.append(dt); pred_n.append(len(va))
        oof[va] = p
        fold_metrics.append(compute_metrics(y[va], p))
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr"]
    out = {f"cv_{k}": float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    out.update({f"cv_{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in ["recall", "f1", "pr_auc"]})
    agg = compute_metrics(y, oof)
    out.update({"oof_pr_auc": agg["pr_auc"], "oof_roc_auc": agg["roc_auc"],
                "oof_cost@0.5": agg["cost"], "oof_fn@0.5": agg["fn"], "oof_fp@0.5": agg["fp"]})
    out["fit_time_s"] = float(np.mean(fit_t))
    out["predict_ms_per_1k"] = float(1000 * np.sum(pred_t) / np.sum(pred_n) * 1000)
    out["oof"] = oof
    return out
