"""Cell definitions for modeling notebooks (02, 04) and the comparative notebook (05).

The modeling notebooks read the checkpointed outputs of `python main.py --full` (results/full/,
models/). Set RUN_TRAINING = True to re-run training from inside the notebook."""
try:
    from nb_findings import FINDINGS
except ImportError:
    FINDINGS = {}

from nb_eda import SETUP

INTRO = {
"ai4i": """# 02 · AI4I 2020 — Modeling, Evaluation and Explainability

**Setup:** stratified 80/20 hold-out split (8,000 train / 2,000 test, seed 42). All model
selection uses 5-fold stratified CV on the 8,000 training rows. The test set is scored once,
at the end.

**Inputs:** Type (one-hot) + 5 sensor readings + 3 engineered features (ΔT, power,
strain), standardized. Failure subtypes and identifiers are excluded.

**Decision threshold:** chosen to maximize F1 on out-of-fold (OOF) training predictions.""",
"aps": """# 04 · APS Scania — Modeling, Evaluation and Explainability

**Setup:** official split kept as is (60,000 train / 16,000 test). Model selection uses CV on
the training set only; the test set is scored once, at the end.

**Preprocessing:** drop the constant `cd_000` → flag and convert the `ac_000` sentinel to NaN →
add the per-row missing count → median imputation → log1p → standardization. Every step is
fit inside each training fold.

**Gradient Boosting:** `HistGradientBoostingClassifier` (early stopping on 10% of the
training fold). The classic `GradientBoostingClassifier` is too slow to tune on this machine
(measured: 183 s per fit), so it is fit once, untuned, as a reference.

**Decision threshold and headline metric:** official challenge cost = 10·FP + 500·FN; the
threshold minimizes cost on OOF training predictions.""",
}


def cells(ds):
    obj = "f1" if ds == "ai4i" else "cost"
    c = [("md", INTRO[ds]),
         ("code", SETUP + f"""
import json, joblib
from src.models import ALGO_NAMES, STRATEGY_NAMES
from src.evaluate import compute_metrics, THRESHOLD_GRID
from src import explain as E
DS = "{ds}"; R = C.RESULTS_DIR; FIGD = FIG / ("ai4i" if DS == "ai4i" else "aps")
RUN_TRAINING = False   # True -> re-run the full pipeline (APS: ~2 h on 2 cores)
if RUN_TRAINING:
    import subprocess; subprocess.run([sys.executable, "-m", "src.train", "--dataset", DS], cwd=ROOT, check=True)
strat = pd.read_csv(R / f"{{DS}}_strategy_cv.csv"); tuned = pd.read_csv(R / f"{{DS}}_tuned_cv.csv")
test = pd.read_csv(R / f"{{DS}}_test_results.csv"); proba = dict(np.load(R / f"{{DS}}_test_proba.npz"))
y_te = np.load(R / f"{{DS}}_y_test.npy"); best = json.loads((R / f"{{DS}}_best_model.json").read_text())
oof_t = dict(np.load(R / f"{{DS}}_oof_tuned.npz"))
name = lambda a: ALGO_NAMES[a]""")]
    if ds == "aps":
        c += [("md", "## 1. Preprocessing choice: imputation and cleaning variants (5-fold CV, training set)"),
              ("code", """imp = pd.read_csv(R / "aps_imputation_comparison.csv")
display(imp.round(4))""")]
    c += [
        ("md", "## 2. Baseline models (default hyperparameters, no imbalance handling, threshold 0.5)"),
        ("code", """cols = ["algo","cv_accuracy","cv_precision","cv_recall","cv_f1","cv_roc_auc","cv_pr_auc","cv_fpr","cv_fnr","fit_time_s","predict_ms_per_1k"]
base = strat[strat.strategy=="none"][cols].assign(algo=lambda d: d.algo.map(ALGO_NAMES))
display(base.round(3))
print("accuracy of the trivial 'never fails' rule:", round(1 - y_te.mean(), 4), "(on test) -> accuracy alone is uninformative")"""),
        ("md", "## 3. Class-imbalance strategies (5-fold CV, default hyperparameters)"),
        ("code", """for metric, lab, fn in [("oof_pr_auc","OOF PR-AUC (threshold-free)","08"),("cv_recall","Recall at threshold 0.5","09"),("cv_f1","F1 at threshold 0.5","10")]:
    pv = strat.pivot(index="algo", columns="strategy", values=metric).reindex(index=list(ALGO_NAMES), columns=list(STRATEGY_NAMES))
    pv.index = pv.index.map(ALGO_NAMES); pv.columns = pv.columns.map(STRATEGY_NAMES)
    fig = V.strategy_heatmap(pv, f"{DS.upper()}: {lab} by imbalance strategy", cbar_label=lab.split(" (")[0])
    V.save(fig, FIGD / f"{fn}_imbalance_{metric}.png"); plt.show()
display(strat[["algo","strategy","oof_pr_auc","cv_precision","cv_recall","cv_f1","cv_fpr","fit_time_s"]].round(3))"""),
        ("md", "## 4. Hyperparameter tuning (RandomizedSearchCV, scored on average precision)"),
        ("code", f"""tcols = ["algo","strategy","params","search_n_iter","search_folds","search_best_ap","search_time_s","oof_pr_auc","threshold","oof_precision@thr","oof_recall@thr","oof_f1@thr","oof_fp@thr","oof_fn@thr","oof_cost@thr"]
display(tuned[tcols].assign(algo=lambda d: d.algo.map(ALGO_NAMES)).round(4))
cmp = strat[strat.strategy=="none"][["algo","oof_pr_auc"]].rename(columns={{"oof_pr_auc":"baseline"}}).merge(
      tuned[["algo","oof_pr_auc"]].rename(columns={{"oof_pr_auc":"tuned (best strategy)"}}), on="algo")
display(cmp.assign(gain=lambda d: d["tuned (best strategy)"]-d.baseline, algo=lambda d: d.algo.map(ALGO_NAMES)).round(4))
fig, ax = plt.subplots(figsize=(8,3.4)); yy = np.arange(len(cmp))
ax.barh(yy-0.18, cmp.baseline, 0.34, color=V.SERIES[0], label="Baseline"); ax.barh(yy+0.18, cmp["tuned (best strategy)"], 0.34, color=V.SERIES[1], label="Tuned + imbalance handling")
for i,(b,t) in enumerate(zip(cmp.baseline, cmp["tuned (best strategy)"])): ax.text(b, i-0.18, f" {{b:.3f}}", va="center", fontsize=8); ax.text(t, i+0.18, f" {{t:.3f}}", va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(cmp.algo.map(ALGO_NAMES)); ax.invert_yaxis(); ax.set_xlim(0,1.12); ax.set_xlabel("OOF PR-AUC (5-fold CV, training set)")
ax.set_title("{ds.upper()}: baseline vs optimized (cross-validation)"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
V.save(fig, FIGD / "11_baseline_vs_tuned_cv.png"); plt.show()"""),
        ("md", "## 5. Final evaluation on the held-out test set (scored once)"),
        ("code", f"""tt = test.copy(); tt["algo"] = tt.model.str.split("_").str[0]
show = ["model","threshold_rule","threshold","accuracy","precision","recall","f1","roc_auc","pr_auc","fpr","fnr","fp","fn"] + (["cost"] if DS == "aps" else [])
display(tt[show].round(4))
final = tt[tt.threshold_rule == "oof_{obj}"].copy()
fig = V.metric_bars(final, ["precision","recall","f1","pr_auc","roc_auc"], "{ds.upper()}: tuned models on the test set (OOF-selected threshold)")
V.save(fig, FIGD / "12_test_model_comparison.png"); plt.show()"""),
        ("code", """tuned_keys = [f"{a}_tuned" for a in ALGO_NAMES]
V.save(V.roc_pr_curves(y_te, {k: proba[k] for k in tuned_keys}, f"{DS.upper()}: tuned models, test set"), FIGD / "13_roc_pr_curves.png"); plt.show()
thr = {a: float(tuned.set_index("algo").loc[a, "threshold"]) for a in ALGO_NAMES}
preds = {f"{name(a)} (t={thr[a]:.3f})": (proba[f"{a}_tuned"] >= thr[a]).astype(int) for a in ALGO_NAMES}
V.save(V.confusion_grid(y_te, preds, f"{DS.upper()}: confusion matrices, tuned models at OOF threshold"), FIGD / "14_confusion_matrices.png"); plt.show()"""),
    ]
    if ds == "ai4i":
        c += [
        ("md", "### 5b. Which failures are missed? (test set, best model, by failure subtype)"),
        ("code", """from src.data_loader import load_ai4i
from src.preprocessing import ai4i_split
_, X_te_idx, _, _ = ai4i_split()
raw = load_ai4i().loc[X_te_idx.index]
pb = proba[f"{best['algo']}_tuned"] >= best["threshold"]
fails = raw[raw[C.AI4I_TARGET] == 1].copy(); fails["detected"] = pb[raw[C.AI4I_TARGET].to_numpy() == 1]
def mode(r):
    m = [s for s in C.AI4I_LEAKAGE_COLS if r[s] == 1]; return "+".join(m) if m else "no subtype flagged"
fails["mode"] = fails.apply(mode, axis=1)
display(fails.groupby("mode").detected.agg(failures="size", detected="sum").assign(recall=lambda d: (d.detected/d.failures).round(3)).sort_values("failures", ascending=False))"""),
        ("md", "### 5c. Feature-engineering ablation (5-fold CV, training set)"),
        ("code", """abl = pd.read_csv(R / "ai4i_feature_ablation.csv"); display(abl.assign(algo=abl.algo.map(ALGO_NAMES)).round(3))"""),
        ]
    c += [
        ("md", "## 6. Trade-off: false alarms vs missed failures"),
        ("code", f"""a = best["algo"]; p_oof = oof_t[a]
from src.preprocessing import ai4i_split, aps_xy
y_tr = (ai4i_split()[2] if DS == "ai4i" else aps_xy()[1]).to_numpy()
rows = [compute_metrics(y_tr, p_oof, t) for t in THRESHOLD_GRID]
tr_df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8,3.8))
ax.plot(tr_df.threshold, tr_df.fp, color=V.SERIES[0], label="False alarms (FP)"); ax.plot(tr_df.threshold, tr_df.fn, color=V.SERIES[1], label="Missed failures (FN)")
ax.axvline(best["threshold"], color=V.MUTED, ls="--", lw=1); ax.text(best["threshold"], ax.get_ylim()[1]*0.92, f"  chosen t={{best['threshold']:.3f}}", fontsize=8, color=V.INK2)
ax.set_yscale("symlog"); ax.set(xlabel="Decision threshold", ylabel="count (symlog)", title=f"{{name(a)}}: errors vs threshold (OOF, training set)"); ax.legend()
V.save(fig, FIGD / "15_threshold_tradeoff.png"); plt.show()
if DS == "aps":
    fig, ax = plt.subplots(figsize=(8,3.4)); ax.plot(tr_df.threshold, tr_df.cost, color=V.SERIES[0])
    ax.axvline(best["threshold"], color=V.MUTED, ls="--", lw=1); ax.set_xscale("log"); ax.set(xlabel="Decision threshold (log scale)", ylabel="10·FP + 500·FN", title="Scania cost vs threshold (OOF, best model)")
    V.save(fig, FIGD / "16_cost_vs_threshold.png"); plt.show()
display(tr_df.loc[tr_df.threshold.isin([0.05,0.1,0.2,0.3,0.5,0.7, best['threshold']]), ["threshold","precision","recall","f1","fp","fn"] + (["cost"] if DS == "aps" else [])].round(3))"""),
        ("md", "## 7. Training and inference time"),
        ("code", f"""tm = final[["algo","fit_time_s","predict_ms_per_1k"]]
V.save(V.time_bars(tm, "{ds.upper()}: final tuned models, full-training-set fit and test inference"), FIGD / "17_time_comparison.png"); plt.show()
display(tt.drop_duplicates("model")[["model","fit_time_s","predict_ms_per_1k"]].round(3))"""),
        ("md", "## 8. Explainability"),
        ("code", f"""from src.preprocessing import ai4i_split, aps_xy
if DS == "ai4i":
    X_tr, X_te, _, _ = ai4i_split()
else:
    X_tr, _, X_te, _ = aps_xy()
models = {{a: joblib.load(C.MODELS_DIR / f"{{DS}}_{{a}}_tuned.joblib") for a in ["rf", "gb"]}}
imp = {{}}
rf_imp = E.impurity_importance(models["rf"], X_te); imp["RF impurity"] = rf_imp
gb_imp = E.impurity_importance(models["gb"], X_te)
if gb_imp is None:   # HistGB has no impurity importances -> use the classic GB reference model
    gb_imp = E.impurity_importance(joblib.load(C.MODELS_DIR / f"{{DS}}_gb_classic_reference.joblib"), X_te); gb_lab = "GB (classic reference) impurity"
else:
    gb_lab = "GB impurity"
imp[gb_lab] = gb_imp
imp_df = pd.DataFrame(imp).fillna(0); imp_df = imp_df / imp_df.max()
V.save(V.importance_bars(imp_df, f"{{DS.upper()}}: model-based (impurity) importance", top=15), FIGD / "18_impurity_importance.png"); plt.show()
display(imp_df.sort_values("RF impurity", ascending=False).head(12).round(3))"""),
        ("code", f"""best_pipe = joblib.load(ROOT / best["model_file"])
perm = {{}}
for lab, pipe in [(f"Best: {{name(best['algo'])}}", best_pipe), ("Random Forest", models["rf"])]:
    if lab.endswith("Random Forest") and best["algo"] == "rf":
        continue
    perm[lab] = E.permutation(pipe, X_te, y_te, n_repeats={5 if ds == "ai4i" else 3}, max_rows={None if ds == "ai4i" else 6000})
perm_df = pd.DataFrame({{k: v["mean"] for k, v in perm.items()}}).fillna(0)
for k, v in perm.items(): print(k); display(v.head(10).round(4))
V.save(V.importance_bars(perm_df.clip(lower=0) / perm_df.max(), f"{{DS.upper()}}: permutation importance on test data (drop in average precision)", top=15), FIGD / "19_permutation_importance.png"); plt.show()
pd.concat({{**{{"impurity_"+k: v for k, v in imp.items()}}, **{{"perm_"+k: v["mean"] for k, v in perm.items()}}}}, axis=1).to_csv(R / f"{{DS}}_feature_importance.csv")"""),
        ("code", f"""import shap
shap_algo = best["algo"] if best["algo"] in ("rf","gb","dtree") else "gb"
shap_pipe = joblib.load(C.MODELS_DIR / f"{{DS}}_{{shap_algo}}_tuned.joblib")
sample = X_te.sample(min(len(X_te), {2000 if ds == "ai4i" else 1500}), random_state=C.RANDOM_STATE)
sv, Xt = E.shap_values(shap_pipe, sample)
print("SHAP model:", name(shap_algo), "| sample:", len(sample))
plt.figure(); shap.summary_plot(sv, Xt, max_display=15, show=False, plot_size=(9, 6))
plt.title(f"{{DS.upper()}}: SHAP summary ({{name(shap_algo)}}, test sample)", loc="left", fontsize=12, fontweight="bold")
V.save(plt.gcf(), FIGD / "20_shap_summary.png"); plt.show()
ms = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)
ms.to_csv(R / f"{{DS}}_shap_mean_abs.csv", header=["mean_abs_shap"]); display(ms.head(12).round(4).to_frame("mean |SHAP|"))"""),
        ("md", "## 9. Selected model"),
        ("code", """print(json.dumps(best, indent=2))
display(tt[(tt.model == f"{best['algo']}_tuned")][show].round(4))"""),
    ]
    if FINDINGS.get(ds):
        c.append(("md", FINDINGS[ds]))
    return c


COMPARE = [
("md", """# 05 · Comparative Analysis: AI4I 2020 vs APS Scania

The two studies were modeled **independently**. Feature spaces, failure definitions and
data-generating processes differ, so no model is transferred across datasets and no claim
of cross-dataset generalization is made. What we compare is *how the same methodology
behaves* on two very different predictive-maintenance problems."""),
("code", SETUP + """
import json
R = C.RESULTS_DIR; FIGD = FIG / "comparison"
from src.models import ALGO_NAMES
from src.data_loader import load_ai4i, load_aps
ai = load_ai4i(); aps_tr, aps_te = load_aps()"""),
("md", "## 1. Dataset characteristics"),
("code", """Xa = aps_tr.drop(columns="class")
chars = pd.DataFrame({
 "AI4I 2020": ["Milling machine process (synthetic)", "10,000 (80/20 split)", 6, 3, f"{ai['Machine failure'].mean():.2%}",
               f"{(ai['Machine failure']==0).sum()/ai['Machine failure'].sum():.0f}:1", "0%", "None", "Named physical sensors", "Any of 5 failure modes"],
 "APS Scania": ["Truck air pressure system (real fleet)", "60,000 train / 16,000 test (official)", 170, 2, f"{aps_tr['class'].mean():.2%} train / {aps_te['class'].mean():.2%} test",
                f"{(aps_tr['class']==0).sum()/aps_tr['class'].sum():.0f}:1", f"{Xa.isna().values.mean():.1%} of cells", "ac_000 code 0x7F000000; 1 constant column", "Anonymized counters + 7 histograms", "Failure attributed to APS component vs other components"]},
 index=["Domain","Records","Raw input features","Engineered features","Positive rate","Imbalance ratio","Missing values","Sentinels / quirks","Feature semantics","Positive class"])
display(chars)"""),
("md", "## 2. Model performance (test set, tuned models at OOF-selected thresholds)"),
("code", """rows = []
for ds, rule in [("ai4i","oof_f1"),("aps","oof_cost")]:
    t = pd.read_csv(R / f"{ds}_test_results.csv"); t = t[t.threshold_rule == rule].copy()
    t["dataset"] = ds.upper(); t["algo"] = t.model.str.split("_").str[0]; rows.append(t)
perf = pd.concat(rows)
display(perf.pivot(index="algo", columns="dataset", values=["recall","precision","f1","pr_auc"]).rename(index=ALGO_NAMES).round(3))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, m in zip(axes, ["recall", "f1", "pr_auc"]):
    pv = perf.pivot(index="algo", columns="dataset", values=m).reindex(list(ALGO_NAMES))
    yy = np.arange(len(pv))
    for j, (ds, col) in enumerate([("AI4I", V.SERIES[0]), ("APS", V.SERIES[1])]):
        ax.barh(yy + (j - 0.5) * 0.36, pv[ds], 0.34, color=col, label=ds)
        for i, v in enumerate(pv[ds]): ax.text(v, i + (j - 0.5) * 0.36, f" {v:.2f}", va="center", fontsize=7.5)
    ax.set_title(m, fontsize=10); ax.set_xlim(0, 1.15); ax.grid(axis="y", visible=False)
axes[0].set_yticks(yy); axes[0].set_yticklabels(pv.index.map(ALGO_NAMES)); axes[0].invert_yaxis(); fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False)
fig.suptitle("Test-set performance by dataset (tuned models, OOF thresholds)", x=0.01, ha="left", fontweight="bold"); fig.tight_layout()
V.save(fig, FIGD / "01_performance_by_dataset.png"); plt.show()"""),
("md", "## 3. Effect of imbalance handling (CV, default hyperparameters)"),
("code", """rows = []
for ds in ["ai4i","aps"]:
    s = pd.read_csv(R / f"{ds}_strategy_cv.csv"); s["dataset"] = ds.upper(); rows.append(s)
st = pd.concat(rows, ignore_index=True)
# KNN has no class-weight variant, so averages use the 4 algorithms that support every strategy
g = st[st.algo != "knn"].groupby(["dataset","strategy"])[["oof_pr_auc","cv_recall","cv_f1"]].mean().round(3)
print("mean over LR, DT, RF, GB (KNN excluded: no class-weight variant)"); display(g)
print("best strategy per algorithm (by OOF PR-AUC):")
display(st.loc[st.groupby(["dataset","algo"]).oof_pr_auc.idxmax(), ["dataset","algo","strategy","oof_pr_auc"]].round(3))"""),
("md", "## 4. Computational requirements"),
("code", """tm = perf.pivot(index="algo", columns="dataset", values=["fit_time_s","predict_ms_per_1k"]).rename(index=ALGO_NAMES).round(3)
display(tm)
srch = pd.concat([pd.read_csv(R / f"{ds}_tuned_cv.csv").assign(dataset=ds.upper()) for ds in ["ai4i","aps"]])
display(srch.pivot(index="algo", columns="dataset", values="search_time_s").rename(index=ALGO_NAMES).round(0).rename(columns=lambda c: f"search time (s) {c}"))
fig, ax = plt.subplots(figsize=(8, 3.6)); pv = perf.pivot(index="algo", columns="dataset", values="fit_time_s").reindex(list(ALGO_NAMES)); yy = np.arange(len(pv))
for j, (ds, col) in enumerate([("AI4I", V.SERIES[0]), ("APS", V.SERIES[1])]):
    ax.barh(yy + (j - 0.5) * 0.36, pv[ds], 0.34, color=col, label=ds)
ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels(pv.index.map(ALGO_NAMES)); ax.invert_yaxis(); ax.legend()
ax.set(xlabel="final model fit time (s, log scale)", title="Training cost grows with data size and dimensionality"); ax.grid(axis="y", visible=False)
V.save(fig, FIGD / "02_training_time_by_dataset.png"); plt.show()"""),
("md", "## 5. Most important features"),
("code", """for ds in ["ai4i","aps"]:
    s = pd.read_csv(R / f"{ds}_shap_mean_abs.csv", index_col=0)
    print(ds.upper(), "- top 8 by mean |SHAP|:"); display(s.head(8).round(4))"""),
("md", "## 6. Best models"),
("code", """for ds in ["ai4i","aps"]:
    b = json.loads((R / f"{ds}_best_model.json").read_text()); print(ds.upper(), "->", ALGO_NAMES[b["algo"]], "| strategy:", b["strategy"], "| threshold:", b["threshold"], "|", b["criterion"])"""),
("md", """## 7. Summary

- **Same winners, different reasons.** Gradient Boosting was selected on both datasets, with
  Random Forest effectively tied. On AI4I, the ensembles win because failures are threshold and
  interaction rules. On APS, they win because they handle 170 skewed, redundant features with
  informative missingness.
- **Linear models split sharply:** Logistic Regression reaches PR-AUC 0.865 on APS but only 0.464
  on AI4I, where no single monotonic combination of sensors describes the failure modes.
- **Imbalance:** resampling helped only APS Gradient Boosting. On AI4I, every strategy lowered
  PR-AUC (SMOTE most of all).
- **Thresholds:** AI4I's F1-optimal thresholds sit at 0.19–0.58. APS's cost-optimal thresholds
  are 0.0025–0.04 because one missed failure costs as much as 50 false alarms. The business
  cost, not the model, sets the operating point.
- **Data quality** dominates APS preprocessing (missingness, a hidden sentinel, skew). On AI4I,
  domain feature engineering matters most (+0.13 PR-AUC for GB).
- **Compute:** APS models take 10–1,000× longer to fit. KNN inference (3 s per 1,000 trucks)
  and classic gradient boosting training (187 s) become the practical bottlenecks.
- **No cross-dataset claims:** different inputs, labels and generating processes. What transfers
  is the methodology."""),
]
