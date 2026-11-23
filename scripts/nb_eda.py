"""Cell definitions for the EDA notebooks (01, 03). Built by scripts/build_notebooks.py."""

SETUP = """import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import config as C
from src import visualization as V
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
FIG = C.FIGURES_DIR"""

AI4I = [
("md", """# 01 · AI4I 2020 — Exploratory Data Analysis

**Goal:** understand the sensor readings, the failure target and its subtypes, and pick
preprocessing that avoids leakage.

The dataset is **synthetic**: UCI describes how each variable and failure mode was generated.
That matters for interpreting results (see §8). EDA here runs on all 10,000 rows because it
is descriptive. No preprocessing parameter is fit on the data at this stage, and the
engineered features are fixed physical formulas, so nothing from the hold-out set reaches a
model."""),
("code", SETUP + "\nfrom src.data_loader import load_ai4i\nfrom src.feature_engineering import AI4IFeatureEngineer\ndf = load_ai4i()\nFIGD = FIG / 'ai4i'"),
("md", "## 1. Dimensions, columns and data types"),
("code", """print("shape:", df.shape)
desc = {
 "UDI": "Unique row identifier 1..10000 (dropped: no signal)",
 "Product ID": "Quality letter + serial number (dropped: identifier)",
 "Type": "Product quality variant L / M / H (50/30/20% by design)",
 "Air temperature [K]": "Random walk, sd 2 K around 300 K",
 "Process temperature [K]": "Air temperature + 10 K + random walk (sd 1 K)",
 "Rotational speed [rpm]": "From 2860 W power + normal noise",
 "Torque [Nm]": "Normal around 40 Nm, sd 10, no negatives",
 "Tool wear [min]": "Increases by 5/3/2 min per process for H/M/L",
 "Machine failure": "TARGET: 1 if any failure mode occurred",
 "TWF": "Tool wear failure (subtype, LEAKAGE: excluded from inputs)",
 "HDF": "Heat dissipation failure (subtype, excluded)",
 "PWF": "Power failure (subtype, excluded)",
 "OSF": "Overstrain failure (subtype, excluded)",
 "RNF": "Random failure, 0.1% chance (subtype, excluded)",
}
display(pd.DataFrame({"dtype": df.dtypes.astype(str), "n_unique": df.nunique(), "description": pd.Series(desc)}))"""),
("md", "## 2. Missing values and duplicates"),
("code", """print("missing cells:", int(df.isna().sum().sum()))
print("duplicate rows:", int(df.duplicated().sum()), "| duplicate Product IDs:", int(df['Product ID'].duplicated().sum()))
print("duplicate rows ignoring identifiers:", int(df.drop(columns=C.AI4I_ID_COLS).duplicated().sum()))"""),
("md", "## 3. Statistical summary"),
("code", "display(df[C.AI4I_NUMERIC].describe().T.round(2))\ndisplay(df.groupby(C.AI4I_TARGET)[C.AI4I_NUMERIC].mean().T.round(2).rename(columns={0:'mean | no failure',1:'mean | failure'}))"),
("md", "## 4. Target distribution, class imbalance and failure subtypes"),
("code", """y = df[C.AI4I_TARGET]
print(f"failure rate {y.mean():.2%}; imbalance ratio {(y==0).sum()/(y==1).sum():.1f} : 1")
V.save(V.target_distribution(y, "AI4I: machine failure target"), FIGD / "01_target_distribution.png"); plt.show()
sub = C.AI4I_LEAKAGE_COLS
display(df[sub].sum().rename("count").to_frame().assign(share_of_failures=lambda d: (d['count']/y.sum()).round(3)))
anysub = df[sub].any(axis=1)
incons = pd.DataFrame({
  "failure=1, no subtype": [int(((y==1)&~anysub).sum())],
  "subtype=1, failure=0": [int(((y==0)&anysub).sum())],
  "rows with >1 subtype": [int((df[sub].sum(axis=1)>1).sum())]})
display(incons)
display(df.loc[(y==0)&anysub, sub].sum().rename("subtype flags on failure=0 rows").to_frame().T)"""),
("md", "## 5. Product type vs failure"),
("code", """ct = df.groupby("Type")[C.AI4I_TARGET].agg(records="size", failures="sum", failure_rate="mean").loc[["L","M","H"]]
display(ct.assign(failure_rate=lambda d: (d.failure_rate*100).round(2).astype(str)+"%"))
fig, ax = plt.subplots(figsize=(5,3)); ax.bar(ct.index, ct.failure_rate*100, color=V.SERIES[0], width=0.5)
for i,v in enumerate(ct.failure_rate*100): ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("failure rate (%)"); ax.set_title("Failure rate by product type"); ax.grid(axis="x", visible=False)
V.save(fig, FIGD / "02_failure_rate_by_type.png"); plt.show()"""),
("md", "## 6. Feature distributions by class (raw and engineered)"),
("code", """fe = AI4IFeatureEngineer().fit(df).transform(df)
ENG = ["temp_diff [K]", "power [W]", "strain [Nm*min]"]
cols = C.AI4I_NUMERIC + ENG
V.save(V.class_conditional_hist(fe, cols, C.AI4I_TARGET), FIGD / "03_feature_distributions.png"); plt.show()"""),
("md", "## 7. Correlation analysis"),
("code", """cm = fe[cols + [C.AI4I_TARGET] + sub]
V.save(V.correlation_heatmap(cm, "AI4I: correlations (features, target, subtypes)", figsize=(10,8)), FIGD / "04_correlation_heatmap.png"); plt.show()
print(cm.corr()[C.AI4I_TARGET].drop(C.AI4I_TARGET).sort_values(key=abs, ascending=False).round(3).to_string())"""),
("md", "## 8. Outlier detection (IQR rule)"),
("code", """rows = []
for c in cols:
    q1, q3 = fe[c].quantile([.25,.75]); iqr = q3-q1
    out = (fe[c] < q1-1.5*iqr) | (fe[c] > q3+1.5*iqr)
    rows.append({"feature": c, "outliers": int(out.sum()), "share": round(out.mean()*100,2),
                 "failure_rate_in_outliers_%": round(fe.loc[out, C.AI4I_TARGET].mean()*100,2) if out.any() else np.nan})
out_df = pd.DataFrame(rows); display(out_df)
print(f"overall failure rate: {fe[C.AI4I_TARGET].mean()*100:.2f}%")
V.save(V.boxplots(fe, cols, C.AI4I_TARGET, ncols=4), FIGD / "05_outlier_boxplots.png"); plt.show()"""),
("md", "## 9. Temperature difference investigation (heat dissipation)"),
("code", """fe["temp_diff_bin"] = pd.cut(fe["temp_diff [K]"], bins=[7,8,8.6,9,9.5,10,10.5,11,12,13])
t = fe.groupby("temp_diff_bin", observed=True).agg(records=(C.AI4I_TARGET,"size"), failure_rate=(C.AI4I_TARGET,"mean"), HDF_rate=("HDF","mean"))
display((t.assign(failure_rate=t.failure_rate*100, HDF_rate=t.HDF_rate*100)).round(2))
fig, ax = plt.subplots(figsize=(7,4.2))
for k,lab,col,s in [(0,"No HDF",V.SERIES[0],4),(1,"HDF",V.SERIES[1],14)]:
    d = fe[fe.HDF==k]; ax.scatter(d["Rotational speed [rpm]"], d["temp_diff [K]"], s=s, alpha=.35 if k==0 else .9, color=col, label=lab, linewidths=0)
ax.axhline(8.6, color=V.MUTED, ls="--", lw=1); ax.axvline(1380, color=V.MUTED, ls="--", lw=1)
ax.text(2800, 8.65, "ΔT = 8.6 K", fontsize=8, color=V.INK2, ha="right"); ax.text(1390, 13.3, "1380 rpm", fontsize=8, color=V.INK2)
ax.set(xlabel="Rotational speed [rpm]", ylabel="Process − air temperature [K]", title="Heat dissipation failures cluster at low ΔT and low speed"); ax.legend()
V.save(fig, FIGD / "06_temp_diff_vs_speed_hdf.png"); plt.show()
print("HDF rows satisfying ΔT<8.6 & rpm<1380:", int(((fe["temp_diff [K]"]<8.6)&(fe["Rotational speed [rpm]"]<1380)&(fe.HDF==1)).sum()), "of", int(fe.HDF.sum()))"""),
("md", "## 10. Torque vs rotational speed (power)"),
("code", """print("corr(torque, rpm) =", round(fe["Torque [Nm]"].corr(fe["Rotational speed [rpm]"]),3))
fig, axes = plt.subplots(1,2, figsize=(12,4.3))
ax = axes[0]
for k,col,s,a in [(0,V.SERIES[0],4,.3),(1,V.SERIES[1],14,.9)]:
    d = fe[fe[C.AI4I_TARGET]==k]; ax.scatter(d["Rotational speed [rpm]"], d["Torque [Nm]"], s=s, alpha=a, color=col, label=V.CLASS_NAMES[k], linewidths=0)
ax.set(xlabel="Rotational speed [rpm]", ylabel="Torque [Nm]", title="Torque vs speed: failures sit at both extremes"); ax.legend()
ax = axes[1]
for k,lab,col in [(0,"No PWF",V.SERIES[0]),(1,"PWF",V.SERIES[1])]:
    ax.hist(fe.loc[fe.PWF==k,"power [W]"], bins=60, density=True, alpha=.6, color=col, label=lab, edgecolor=V.SURFACE, linewidth=.4)
for x in (3500, 9000): ax.axvline(x, color=V.MUTED, ls="--", lw=1)
ax.set(xlabel="Mechanical power = torque × ω [W]", title="Power failures occur outside 3.5–9 kW"); ax.set_yticks([]); ax.legend()
fig.tight_layout(); V.save(fig, FIGD / "07_torque_speed_power.png"); plt.show()
pw = fe["power [W]"]; print("PWF rows with power <3500 or >9000 W:", int((((pw<3500)|(pw>9000)) & (fe.PWF==1)).sum()), "of", int(fe.PWF.sum()))"""),
("md", "## 11. Tool wear × torque (overstrain) by product type"),
("code", """fig, ax = plt.subplots(figsize=(7.5,4))
for i,tp in enumerate(["L","M","H"]):
    d = fe[fe.Type==tp]
    ax.hist(d.loc[d.OSF==1,"strain [Nm*min]"], bins=25, alpha=.7, color=V.SERIES[i], label=f"OSF · type {tp} (n={int(d.OSF.sum())})", edgecolor=V.SURFACE)
for i,(tp,th) in enumerate([("L",11000),("M",12000),("H",13000)]): ax.axvline(th, color=V.SERIES[i], ls="--", lw=1)
ax.set(xlabel="Tool wear × torque [Nm·min]", ylabel="OSF records", title="Overstrain failures begin near type-specific limits (11k / 12k / 13k)"); ax.legend()
V.save(fig, FIGD / "08_overstrain_by_type.png"); plt.show()
print(fe.groupby("Type").apply(lambda d: pd.Series({"min strain at OSF": d.loc[d.OSF==1,"strain [Nm*min]"].min(), "OSF count": int(d.OSF.sum())})).loc[["L","M","H"]])"""),
("md", "## 12. Tool wear and tool-wear failures"),
("code", """print(fe.loc[fe.TWF==1, "Tool wear [min]"].describe().round(1).to_string())
print("failure rate when tool wear >= 200 min:", f"{fe.loc[fe['Tool wear [min]']>=200, C.AI4I_TARGET].mean():.2%}", "| < 200:", f"{fe.loc[fe['Tool wear [min]']<200, C.AI4I_TARGET].mean():.2%}")"""),
]

APS = [
("md", """# 03 · APS Failure at Scania Trucks — Exploratory Data Analysis

**Goal:** understand the 170 anonymized operational features, the missing-value and sentinel
patterns, and the class structure. **All analysis here uses the official training set
only**; the test set is loaded just to confirm its shape and class balance.

Reminder (dataset documentation): the negative class is *trucks whose failures were in
components unrelated to the APS*. The task is attributing a failure to the APS, not
telling failed trucks from healthy ones."""),
("code", SETUP + """
from src.data_loader import load_aps
from sklearn.metrics import roc_auc_score
tr, te = load_aps()
X, y = tr.drop(columns=C.APS_TARGET), tr[C.APS_TARGET]
FIGD = FIG / 'aps'"""),
("md", "## 1. Dimensions, structure and data types"),
("code", """print("train:", tr.shape, "| test:", te.shape)
print("dtypes:", X.dtypes.value_counts().to_dict())
prefix = pd.Series([c.split('_')[0] for c in X.columns], index=X.columns)
groups = prefix.value_counts()
hist_groups = groups[groups > 1]
print(f"{len(hist_groups)} histogram groups x 10 bins = {hist_groups.sum()} features; {(groups==1).sum()} single counters")
print("histogram groups:", ", ".join(hist_groups.index))"""),
("md", "## 2. Target distribution and class imbalance"),
("code", """for name, d in [("train", tr), ("test", te)]:
    print(f"{name}: {int(d['class'].sum())} pos / {int((d['class']==0).sum())} neg  ({d['class'].mean():.2%}); ratio {(d['class']==0).sum()/d['class'].sum():.0f}:1")
V.save(V.target_distribution(y, "APS (train): APS-component failure vs other failures"), FIGD / "01_target_distribution.png"); plt.show()"""),
("md", "## 3. Missing values"),
("code", """miss = X.isna().mean()
print(f"missing cells: {X.isna().values.mean():.2%} | rows with any missing: {X.isna().any(axis=1).mean():.1%} | complete rows: {int((~X.isna().any(axis=1)).sum())}")
bands = pd.cut(miss, [-0.001, 0, .01, .05, .2, .5, .7, 1.0], labels=["0%","0-1%","1-5%","5-20%","20-50%","50-70%",">70%"])
display(bands.value_counts().sort_index().rename("columns").to_frame().T)
V.save(V.missing_bar(miss, "APS (train): top 40 columns by missing share"), FIGD / "02_missing_by_column.png"); plt.show()"""),
("code", """# Is missingness informative? failure rate when a column is missing vs present
rows = []
for c in miss[miss > 0.05].index:
    m = X[c].isna()
    rows.append({"feature": c, "missing_%": round(m.mean()*100,1),
                 "fail_rate_if_missing_%": round(y[m].mean()*100,2), "fail_rate_if_present_%": round(y[~m].mean()*100,2)})
mi = pd.DataFrame(rows).sort_values("missing_%", ascending=False); display(mi.head(15))
nm = X.isna().sum(axis=1)
print("missing values per row - mean by class:", nm.groupby(y).mean().round(1).to_dict(), "| median:", nm.groupby(y).median().to_dict())
print("single-feature ROC-AUC of row missing-count:", round(roc_auc_score(y, nm), 3))
fig, ax = plt.subplots(figsize=(7,3.4))
for k in [0,1]: ax.hist(nm[y==k], bins=range(0, int(nm.max())+2, 2), density=True, alpha=.6, color=V.CLASS_COLORS[k], label=V.CLASS_NAMES[k], edgecolor=V.SURFACE, linewidth=.4)
ax.set(xlabel="missing values per row", title="Missing-value count per row, by class"); ax.set_yticks([]); ax.legend()
V.save(fig, FIGD / "03_missing_per_row.png"); plt.show()"""),
("md", "## 4. Sentinel values (−1 and others)"),
("code", """print("cells equal to -1:", int((X == -1).sum().sum()), "| columns with any negative value:", int((X < 0).any().sum()))
hi = (X >= 2.13e9) & (X < 2**31)
print("columns with values in [2.13e9, 2^31):", hi.sum()[hi.sum() > 0].to_dict())
ac = X[C.APS_SENTINEL_COL]
print("ac_000 most frequent values:"); print(ac.value_counts().head(6).to_string())
s = ac >= C.APS_SENTINEL_MIN
print(f"sentinel share {s.mean():.2%}; failure rate | sentinel {y[s].mean():.2%} vs non-sentinel {y[~s & ac.notna()].mean():.2%} vs ac_000 missing {y[ac.isna()].mean():.2%}")
print("2130706432 == 0x7F000000:", 2130706432 == 0x7F000000)
print("next-largest ac_000 value below the sentinel band:", ac[ac < 2.13e9].max())"""),
("md", "## 5. Constant columns and duplicates"),
("code", """nun = X.nunique(dropna=True)
print("constant columns:", list(nun[nun <= 1].index), "| duplicate rows:", int(tr.duplicated().sum()), "| duplicate feature rows:", int(X.duplicated().sum()))"""),
("md", "## 6. Statistical summary and skewness"),
("code", """d = X.describe().T
d["skew"] = X.skew()
display(d.sort_values("skew", ascending=False).head(8).round(1))
print("median |skew| across features:", round(X.skew().abs().median(), 1), "| features with |skew| > 10:", int((X.skew().abs() > 10).sum()))
print("share of zero values (median across features):", f"{(X == 0).mean().median():.1%}")"""),
("md", """## 7. Feature relationships with the label

Features are anonymized, so I rank them by **single-feature ROC-AUC** (each feature on its
own, missing values filled with the training median). This is a screening statistic only:
it measures association, not causation."""),
("code", """Xm = X.fillna(X.median())
auc = Xm.apply(lambda c: roc_auc_score(y, c)).rename("auc")
rank = (auc - 0.5).abs().sort_values(ascending=False)
top = rank.index[:20]
display(pd.DataFrame({"single-feature AUC": auc[top].round(3), "missing %": (miss[top]*100).round(1)}).head(12))
print(f"features with |AUC-0.5| > 0.4: {int((rank>0.4).sum())}, > 0.3: {int((rank>0.3).sum())}")
V.save(V.class_conditional_hist(tr, list(top[:9]), C.APS_TARGET, log_x=True), FIGD / "04_top_feature_distributions.png"); plt.show()"""),
("md", "## 8. Correlation analysis"),
("code", """corr = X[top].corr()
V.save(V.correlation_heatmap(X[top], "APS (train): correlation among the 20 most label-associated features", annot=False, figsize=(10,8)), FIGD / "05_correlation_top20.png"); plt.show()
full = X.corr().abs(); iu = np.triu_indices_from(full, 1); vals = full.values[iu]
print(f"feature pairs: {len(vals):,} | |r|>0.9: {(vals>0.9).sum()} | |r|>0.99: {(vals>0.99).sum()}")"""),
("md", "## 9. Outlier detection"),
("code", """L = np.log1p(X)
q1, q3 = L.quantile(.25), L.quantile(.75); iqr = q3 - q1
outl = ((L < q1 - 1.5*iqr) | (L > q3 + 1.5*iqr)) & (iqr > 0)
share = outl.mean()
print(f"IQR outliers on log1p scale (features with IQR>0): median share {share[iqr>0].median():.1%}, max {share.max():.1%}")
rows_out = outl.sum(axis=1)
print("failure rate among rows with >=10 outlying features:", f"{y[rows_out>=10].mean():.2%}", f"(n={int((rows_out>=10).sum())})", "vs overall", f"{y.mean():.2%}")
V.save(V.boxplots(L.assign(**{C.APS_TARGET: y}), list(top[:6]), C.APS_TARGET), FIGD / "06_outlier_boxplots_log.png"); plt.show()"""),
("md", "## 10. Histogram feature profiles"),
("code", """fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, g in zip(axes, ["ag", "ay", "cs"]):
    cols = [c for c in X.columns if c.startswith(g + "_")]
    shares = X[cols].div(X[cols].sum(axis=1), axis=0)
    for k in [0, 1]:
        ax.plot(range(10), shares[y == k].mean().values, marker="o", ms=4, color=V.CLASS_COLORS[k], label=V.CLASS_NAMES[k])
    ax.set(title=f"histogram '{g}': mean bin share", xlabel="bin"); ax.set_xticks(range(10))
axes[0].set_ylabel("share of total count"); axes[0].legend()
fig.tight_layout(); V.save(fig, FIGD / "07_histogram_profiles.png"); plt.show()"""),
]

AI4I.append(("md", """## 13. EDA summary → preprocessing decisions

- **Clean data:** 10,000 rows, no missing values, no duplicates. `UDI` and `Product ID` are
  pure identifiers → dropped.
- **Imbalance:** 339 failures (3.39%, about 28:1). Accuracy is uninformative (always predicting
  "no failure" scores 96.6%), so evaluation centers on recall, F1 and PR-AUC.
- **Leakage:** the five subtype flags *define* the target (r with the target 0.36–0.58 for TWF/HDF/PWF/OSF; RNF ≈ 0 because 18 of its 19 rows are labeled 0)
  → excluded from inputs, kept only for error analysis.
- **Label quirks:** 9 failures have no subtype; 18 of 19 RNF rows are labeled "no failure."
  The target is used as published.
- **Failure physics is recoverable from the data:** all 115 HDF rows have ΔT < 8.6 K and speed
  < 1380 rpm; all 95 PWF rows have power outside 3.5–9 kW; OSF begins at type-specific strain
  limits (11k / 12k / 13k Nm·min); TWF appears at 198–253 min tool wear. → Engineer
  **ΔT, power, strain**.
- **Torque and speed** are strongly anti-correlated (r = −0.875) by construction (constant-power
  generation), so their product (power) carries information neither has alone.
- **Outliers** in torque, power and strain are almost all failures (90–100% failure rate) →
  they are signal, not noise, so none are removed.
- **Product type:** failure rate L 3.92% > M 2.77% > H 2.09% → one-hot encoded.
- **Scaling:** standardization for LR and KNN; it has no effect on trees."""))

APS.append(("md", """## 11. EDA summary → preprocessing decisions

- **Structure:** 170 numeric features: 100 counters + 7 histograms × 10 bins. Train 60,000
  (1.67% positive), test 16,000 (2.34% positive). The official split is preserved.
- **No −1 sentinels** anywhere and no negative values. Missing values are `"na"` only: 8.3% of
  cells, 99% of rows. Only 591 training rows are complete, so dropping rows is not an option →
  **impute** (median; alternatives compared in notebook 04).
- **Missingness is informative:** positives average 29 missing values per row vs 14 for
  negatives (single-feature AUC 0.68). Rows missing `ac_000` fail at 13.9%, and the
  `bm–br` block fails at only 0.17–0.18% when missing → add a **per-row missing count** and keep
  heavily-missing columns (dropping them is evaluated, not assumed).
- **`ac_000` sentinel:** 15.8% of values sit at 0x7F000000 (2130706432) and two neighbors,
  far above the next real value (265,766). Failure rate is 0.16% there → **flag it, then treat the
  raw value as missing**.
- **`cd_000` is constant** → dropped (learned from training data inside the pipeline).
- **Extreme skew** (median |skew| 17.5) → **log1p** before scaling for LR and KNN.
- **Strong univariate signal and redundancy:** many single features reach AUC ≈ 0.97, and 137
  feature pairs have |r| > 0.9, so models with built-in feature selection (trees, regularized LR)
  are appropriate.
- **Outliers** (IQR on the log scale) are rare per feature, but rows with many outlying features
  fail at 2× the base rate → kept."""))
