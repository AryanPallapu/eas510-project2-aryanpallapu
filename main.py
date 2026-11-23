"""Single entry point.

    python main.py                 # sample run on bundled data (data/sample/) -> results/sample/
    python main.py --full          # full experiment (run `python data/download.py` first) -> results/full/
    python main.py --full --fresh  # full experiment from scratch (ignore checkpointed results)
    python main.py --dataset ai4i  # one dataset only

Workflow per dataset (identical code in both modes; only data and budgets differ, see config.py):
  1. strategies: 5 algorithms x 4 imbalance strategies (none / class weight / random oversampling /
     SMOTE), default hyperparameters, stratified k-fold CV on the training split.
  2. tune:  RandomizedSearchCV (average precision) for each algorithm with its best strategy, then
     k-fold out-of-fold (OOF) predictions -> decision threshold (AI4I: max F1; APS: min 10*FP+500*FN).
  3. final: refit baseline and tuned pipelines on the full training split, evaluate once on the test split.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402

OBJECTIVE_RULE = {"ai4i": "oof_f1", "aps": "oof_cost"}


def check_data(mode: str) -> None:
    missing = [p for p in (C.AI4I_RAW, C.APS_TRAIN_RAW, C.APS_TEST_RAW) if not Path(p).exists()]
    if missing:
        hint = "python data/download.py" if mode == "full" else "python scripts/make_sample.py"
        sys.exit(f"Missing data files: {[str(m) for m in missing]}\nRun: {hint}")


def run_dataset(ds: str, stage: str) -> None:
    from src import train as T
    X, y, X_te, y_te = T.load(ds)
    print(f"\n=== {ds.upper()} [{C.MODE}] train {X.shape}, positives {int(y.sum())} | "
          f"test {X_te.shape}, positives {int(y_te.sum())} ===", flush=True)
    if stage in ("strategies", "all"):
        T.run_strategies(ds, X, y, T.ALGOS)
    if stage in ("tune", "all"):
        T.run_tuning(ds, X, y, T.ALGOS)
    if stage in ("final", "all"):
        T.run_final(ds, X, y, X_te, y_te, T.ALGOS, classic_gb_reference=(ds == "aps"))


def summarize(datasets) -> str:
    """Print key metrics and save labelled tables/figures for this run."""
    import numpy as np
    import pandas as pd
    from src import visualization as V
    from src.models import ALGO_NAMES

    out = [f"# Run summary — {C.MODE} mode\n",
           "Test-split metrics. `baseline` = default hyperparameters, no imbalance handling, threshold 0.5. "
           "`tuned` = best imbalance strategy + RandomizedSearchCV, threshold chosen on OOF training "
           "predictions (AI4I: max F1; APS: min cost 10*FP + 500*FN). Higher is better for all metrics "
           "except FP, FN and cost.\n"]
    if C.MODE == "sample":
        out.append("**Sample mode is a functionality check** (APS: 5%/10% stratified subsample; 3-fold CV; "
                   "4 search iterations per model). Conclusions about model quality rest on results/full/.\n")
    fig_dir = C.RESULTS_DIR / "figures"
    for ds in datasets:
        t = pd.read_csv(C.RESULTS_DIR / f"{ds}_test_results.csv")
        base = t[(t.model.str.endswith("_baseline")) & (t.threshold_rule == "default_0.5")]
        tuned = t[(t.model.str.endswith("_tuned")) & (t.threshold_rule == OBJECTIVE_RULE[ds])]
        tab = pd.concat([base, tuned])
        tab = tab.assign(algorithm=tab.model.str.split("_").str[0].map(ALGO_NAMES),
                         variant=tab.model.str.split("_").str[1])
        cols = ["algorithm", "variant", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "fp", "fn"]
        if ds == "aps":
            cols.append("cost")
        tab = tab[cols].sort_values(["algorithm", "variant"]).reset_index(drop=True)
        tab.to_csv(C.RESULTS_DIR / f"{ds}_summary_table.csv", index=False)
        best = json.loads((C.RESULTS_DIR / f"{ds}_best_model.json").read_text())
        y_te = np.load(C.RESULTS_DIR / f"{ds}_y_test.npy")
        print(f"\n{ds.upper()} test split ({len(y_te)} rows, {int(y_te.sum())} positives):")
        print(tab.round(3).to_string(index=False))
        print(f"Selected on CV: {ALGO_NAMES[best['algo']]} ({best['strategy']}), threshold {best['threshold']}")
        out += [f"\n## {ds.upper()} — test split: {len(y_te)} rows, {int(y_te.sum())} positives\n",
                tab.round(3).to_markdown(index=False),
                f"\nModel selected on cross-validation (before test scoring): **{ALGO_NAMES[best['algo']]}**, "
                f"strategy `{best['strategy']}`, threshold {best['threshold']}.\n",
                f"\n![{ds} ROC and PR curves](figures/{ds}_roc_pr.png)\n"]
        proba = dict(np.load(C.RESULTS_DIR / f"{ds}_test_proba.npz"))
        fig = V.roc_pr_curves(y_te, {f"{a}_tuned": proba[f"{a}_tuned"] for a in ALGO_NAMES},
                              f"{ds.upper()} ({C.MODE} mode): tuned models, test split")
        V.save(fig, fig_dir / f"{ds}_roc_pr.png")
        V.plt.close("all")
    (C.RESULTS_DIR / "summary.md").write_text("\n".join(out))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="full experiment on data/full/ (default: bundled sample)")
    ap.add_argument("--dataset", choices=["ai4i", "aps", "both"], default="both")
    ap.add_argument("--stage", choices=["strategies", "tune", "final", "all"], default="all")
    ap.add_argument("--fresh", action="store_true",
                    help="delete existing results for this mode first (sample mode always starts fresh)")
    a = ap.parse_args()

    mode = "full" if a.full else "sample"
    C.use_mode(mode)
    check_data(mode)
    if (mode == "sample" or a.fresh) and a.stage == "all":
        for d in (C.RESULTS_DIR, C.MODELS_DIR):
            shutil.rmtree(d, ignore_errors=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["ai4i", "aps"] if a.dataset == "both" else [a.dataset]
    t0 = time.perf_counter()
    for ds in datasets:
        run_dataset(ds, a.stage)
    elapsed = time.perf_counter() - t0
    if a.stage in ("final", "all"):
        summarize(datasets)
    info = {"mode": mode, "datasets": datasets, "stage": a.stage, "runtime_s": round(elapsed, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "python": platform.python_version(),
            "platform": platform.platform(), "random_state": C.RANDOM_STATE, "cv_folds": C.CV_FOLDS,
            "search_budget": C.SEARCH_BUDGET,
            "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "imbalanced-learn",
                                                 "matplotlib", "seaborn", "joblib", "scipy"]}}
    (C.RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDone in {elapsed:.0f} s. Outputs: {C.RESULTS_DIR.relative_to(ROOT)}/ "
          f"(summary.md, *_test_results.csv, figures/), models: {C.MODELS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
