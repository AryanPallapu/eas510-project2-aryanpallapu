"""Training pipeline (resumable, results checkpointed to results/<mode>/).

    Normally invoked through main.py. Direct use:
    python -m src.train --dataset ai4i --mode full   # all stages
    python -m src.train --dataset aps --stage strategies
    stages: strategies -> tune -> final

strategies : 5 algorithms x 4 imbalance strategies, default hyperparameters, 5-fold CV.
             The `none` rows are the baselines.
tune       : RandomizedSearchCV (scoring = average precision) using each algorithm's best
             strategy; then 5-fold CV of the best config -> OOF probabilities -> threshold.
final      : refit baseline and tuned models on the full training set, evaluate once on
             the held-out test set, save models.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import config as C
from .evaluate import compute_metrics, cv_evaluate, predict_proba_timed, select_threshold
from .models import ALGOS, STRATEGIES, build_pipeline, search_space, supports
from .preprocessing import ai4i_split, aps_xy

OBJECTIVE = {"ai4i": "f1", "aps": "cost"}


def load(dataset):
    if dataset == "ai4i":
        X_tr, X_te, y_tr, y_te = ai4i_split()
    else:
        X_tr, y_tr, X_te, y_te = aps_xy()
    return X_tr, y_tr.to_numpy(), X_te, y_te.to_numpy()


def _csv(path, rows_df):
    rows_df.to_csv(path, index=False)


def _npz_update(path, **arrays):
    old = dict(np.load(path)) if path.exists() else {}
    old.update(arrays)
    np.savez_compressed(path, **old)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- stage 1
def run_strategies(dataset, X, y, algos):
    path = C.RESULTS_DIR / f"{dataset}_strategy_cv.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    oof_path = C.RESULTS_DIR / f"{dataset}_oof_strategies.npz"
    for algo in algos:
        for strat in STRATEGIES:
            if not supports(algo, strat):
                continue
            if len(df) and ((df.algo == algo) & (df.strategy == strat)).any():
                continue
            log(f"[{dataset}] CV {algo} / {strat}")
            res = cv_evaluate(build_pipeline(dataset, algo, strat), X, y)
            oof = res.pop("oof")
            _npz_update(oof_path, **{f"{algo}__{strat}": oof})
            df = pd.concat([df, pd.DataFrame([{"algo": algo, "strategy": strat, **res}])],
                           ignore_index=True)
            _csv(path, df)
            log(f"    PR-AUC {res['oof_pr_auc']:.3f}  recall {res['cv_recall']:.3f}  "
                f"F1 {res['cv_f1']:.3f}  fit {res['fit_time_s']:.1f}s")
    return df


# ---------------------------------------------------------------- stage 2
def best_strategy(dataset, algo):
    df = pd.read_csv(C.RESULTS_DIR / f"{dataset}_strategy_cv.csv")
    d = df[df.algo == algo]
    return d.loc[d.oof_pr_auc.idxmax(), "strategy"]


def run_tuning(dataset, X, y, algos):
    path = C.RESULTS_DIR / f"{dataset}_tuned_cv.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    oof_path = C.RESULTS_DIR / f"{dataset}_oof_tuned.npz"
    for algo in algos:
        if len(df) and (df.algo == algo).any():
            continue
        strat = best_strategy(dataset, algo)
        n_iter, folds = C.SEARCH_BUDGET[dataset][algo]
        log(f"[{dataset}] tune {algo} (strategy={strat}, n_iter={n_iter}, cv={folds})")
        search = RandomizedSearchCV(
            build_pipeline(dataset, algo, strat, n_jobs=1), search_space(dataset, algo),
            n_iter=n_iter, scoring="average_precision", n_jobs=2, refit=False,
            cv=StratifiedKFold(folds, shuffle=True, random_state=C.RANDOM_STATE),
            random_state=C.RANDOM_STATE, error_score="raise")
        t = time.perf_counter()
        search.fit(X, y)
        search_time = time.perf_counter() - t
        pd.DataFrame(search.cv_results_).drop(columns="params").sort_values("rank_test_score") \
            .to_csv(C.RESULTS_DIR / f"{dataset}_search_{algo}.csv", index=False)
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        log(f"    best search AP {search.best_score_:.3f}  {params}  ({search_time:.0f}s)")
        res = cv_evaluate(build_pipeline(dataset, algo, strat, params), X, y)
        oof = res.pop("oof")
        thr = select_threshold(y, oof, OBJECTIVE[dataset])
        at_thr = compute_metrics(y, oof, thr)
        _npz_update(oof_path, **{algo: oof})
        row = {"algo": algo, "strategy": strat, "params": json.dumps(params),
               "search_best_ap": search.best_score_, "search_time_s": search_time,
               "search_n_iter": n_iter, "search_folds": folds, **res, "threshold": thr,
               **{f"oof_{k}@thr": at_thr[k] for k in
                  ["precision", "recall", "f1", "fpr", "fnr", "fp", "fn", "cost"]}}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _csv(path, df)
        log(f"    tuned OOF PR-AUC {res['oof_pr_auc']:.3f}; thr {thr} -> F1 {at_thr['f1']:.3f} "
            f"recall {at_thr['recall']:.3f} cost {at_thr['cost']}")
    return df


# ---------------------------------------------------------------- stage 3
def _fit_eval(dataset, name, pipe, X, y, X_te, y_te, thresholds):
    t = time.perf_counter()
    pipe.fit(X, y)
    fit_t = time.perf_counter() - t
    p, pred_t = predict_proba_timed(pipe, X_te)
    joblib.dump(pipe, C.MODELS_DIR / f"{dataset}_{name}.joblib", compress=3)
    rows = []
    for label, thr in thresholds.items():
        rows.append({"model": name, "threshold_rule": label, **compute_metrics(y_te, p, thr),
                     "fit_time_s": fit_t, "predict_ms_per_1k": 1000 * pred_t / len(y_te) * 1000})
    return rows, p


def run_final(dataset, X, y, X_te, y_te, algos, classic_gb_reference=False):
    tuned = pd.read_csv(C.RESULTS_DIR / f"{dataset}_tuned_cv.csv")
    path = C.RESULTS_DIR / f"{dataset}_test_results.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    proba_path = C.RESULTS_DIR / f"{dataset}_test_proba.npz"
    jobs = []
    for algo in algos:
        jobs.append((f"{algo}_baseline", build_pipeline(dataset, algo, "none"), {"default_0.5": 0.5}))
        r = tuned[tuned.algo == algo].iloc[0]
        jobs.append((f"{algo}_tuned",
                     build_pipeline(dataset, algo, r.strategy, json.loads(r.params)),
                     {"default_0.5": 0.5, f"oof_{OBJECTIVE[dataset]}": float(r.threshold)}))
    if classic_gb_reference:
        jobs.append(("gb_classic_reference", build_pipeline(dataset, "gb", "none", classic_gb=True),
                     {"default_0.5": 0.5}))
    for name, pipe, thrs in jobs:
        if len(df) and (df.model == name).any():
            continue
        log(f"[{dataset}] final fit {name}")
        rows, p = _fit_eval(dataset, name, pipe, X, y, X_te, y_te, thrs)
        _npz_update(proba_path, **{name: p})
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        _csv(path, df)
        for rr in rows:
            log(f"    {rr['threshold_rule']}: F1 {rr['f1']:.3f} recall {rr['recall']:.3f} "
                f"precision {rr['precision']:.3f} cost {rr['cost']}")
    np.save(C.RESULTS_DIR / f"{dataset}_y_test.npy", y_te)
    write_best(dataset)
    return df


def rethreshold(dataset, y, y_te):
    """Re-select thresholds from saved OOF probabilities and re-score saved test
    probabilities (no refitting). Used after the threshold grid was refined."""
    tpath = C.RESULTS_DIR / f"{dataset}_tuned_cv.csv"
    tuned = pd.read_csv(tpath)
    oof = dict(np.load(C.RESULTS_DIR / f"{dataset}_oof_tuned.npz"))
    for i, r in tuned.iterrows():
        thr = select_threshold(y, oof[r.algo], OBJECTIVE[dataset])
        m = compute_metrics(y, oof[r.algo], thr)
        tuned.loc[i, "threshold"] = thr
        for k in ["precision", "recall", "f1", "fpr", "fnr", "fp", "fn", "cost"]:
            tuned.loc[i, f"oof_{k}@thr"] = m[k]
    _csv(tpath, tuned)
    test_path = C.RESULTS_DIR / f"{dataset}_test_results.csv"
    test = pd.read_csv(test_path)
    proba = dict(np.load(C.RESULTS_DIR / f"{dataset}_test_proba.npz"))
    for i, r in test.iterrows():
        if r.threshold_rule.startswith("oof_"):
            thr = float(tuned.set_index("algo").loc[r.model.split("_")[0], "threshold"])
            for k, v in compute_metrics(y_te, proba[r.model], thr).items():
                test.loc[i, k] = v
    _csv(test_path, test)
    log(f"[{dataset}] re-thresholded: " + ", ".join(f"{a}={t}" for a, t in zip(tuned.algo, tuned.threshold)))
    return write_best(dataset)


def write_best(dataset):
    """Best model = chosen on CV (OOF) results only, never on test."""
    t = pd.read_csv(C.RESULTS_DIR / f"{dataset}_tuned_cv.csv")
    if dataset == "ai4i":
        best = t.sort_values(["oof_f1@thr", "oof_pr_auc"], ascending=False).iloc[0]
        crit = "highest OOF F1 at the OOF-selected threshold (tie-break: OOF PR-AUC)"
    else:
        best = t.sort_values(["oof_cost@thr", "oof_pr_auc"], ascending=[True, False]).iloc[0]
        crit = "lowest OOF Scania cost at the OOF-selected threshold (tie-break: OOF PR-AUC)"
    info = {"dataset": dataset, "algo": best.algo, "strategy": best.strategy,
            "params": json.loads(best.params), "threshold": float(best.threshold),
            "model_file": f"models/{dataset}_{best.algo}_tuned.joblib", "criterion": crit}
    (C.RESULTS_DIR / f"{dataset}_best_model.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ai4i", "aps"], required=True)
    ap.add_argument("--stage", choices=["strategies", "tune", "final", "rethreshold", "all"], default="all")
    ap.add_argument("--algos", nargs="*", default=ALGOS)
    ap.add_argument("--mode", choices=["sample", "full"], default="full")
    a = ap.parse_args()
    C.use_mode(a.mode)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True); C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, X_te, y_te = load(a.dataset)
    if a.stage in ("strategies", "all"):
        run_strategies(a.dataset, X, y, a.algos)
    if a.stage in ("tune", "all"):
        run_tuning(a.dataset, X, y, a.algos)
    if a.stage in ("final", "all"):
        run_final(a.dataset, X, y, X_te, y_te, a.algos, classic_gb_reference=a.dataset == "aps")
    if a.stage == "rethreshold":
        rethreshold(a.dataset, y, y_te)


if __name__ == "__main__":
    main()
