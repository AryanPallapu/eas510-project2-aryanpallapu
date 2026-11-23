"""Plotting helpers with one consistent visual system.

Palette: validated categorical order (blue, orange, aqua, yellow, magenta) checked for
colour-vision-deficiency separation; identity is never colour-only (legends/labels).
Sequential = one hue (blue) light->dark; diverging = blue <-> orange with a neutral midpoint.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALGO_COLORS = {"logreg": SERIES[0], "knn": SERIES[1], "dtree": SERIES[2], "rf": SERIES[3], "gb": SERIES[4]}
ALGO_NAMES = {"logreg": "Logistic Regression", "knn": "KNN", "dtree": "Decision Tree",
              "rf": "Random Forest", "gb": "Gradient Boosting"}
CLASS_COLORS = {0: SERIES[0], 1: SERIES[1]}
CLASS_NAMES = {0: "No failure", 1: "Failure"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a481", "#b8461b"])


def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 9, "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    })


set_style()


def save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


# ------------------------------------------------------------------ EDA
def target_distribution(y: pd.Series, title: str):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, width=0.55,
                  color=[CLASS_COLORS[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v:,}  ({v / counts.sum():.2%})", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, color=INK, xytext=(0, 3),
                    textcoords="offset points")
    ax.set_ylabel("Records"); ax.set_title(title)
    ax.set_ylim(0, counts.max() * 1.15); ax.grid(axis="x", visible=False)
    return fig


def class_conditional_hist(df: pd.DataFrame, cols, target: str, ncols=3, bins=40, log_x=False):
    n = len(cols); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        for k in [0, 1]:
            v = df.loc[df[target] == k, c].dropna()
            if log_x:
                v = np.log1p(v)
            ax.hist(v, bins=bins, density=True, alpha=0.55, color=CLASS_COLORS[k],
                    label=CLASS_NAMES[k], edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(("log1p " if log_x else "") + c, fontsize=10)
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Feature distributions by class (density, each class normalised)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str, annot=True, figsize=(8, 6.5)):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap=DIV, vmin=-1, vmax=1, center=0, annot=annot, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor=SURFACE, ax=ax,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    ax.set_title(title)
    return fig


def boxplots(df: pd.DataFrame, cols, target: str, ncols=3):
    n = len(cols); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    pal = {CLASS_NAMES[k]: v for k, v in CLASS_COLORS.items()}
    d = df.assign(_cls=df[target].map(CLASS_NAMES))
    for ax, c in zip(axes.ravel(), cols):
        sns.boxplot(data=d, x="_cls", y=c, hue="_cls", palette=pal, ax=ax, width=0.5,
                    fliersize=2, linewidth=1, legend=False)
        ax.set_xlabel(""); ax.set_title(c, fontsize=10); ax.set_ylabel("")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Outlier view: box plots by class (whiskers = 1.5 x IQR)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def missing_bar(miss_frac: pd.Series, title: str, top=40):
    s = miss_frac.sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(s.index, s.values * 100, color=SERIES[0], width=0.7)
    ax.axhline(70, color=MUTED, lw=1, ls="--"); ax.text(len(s) - 0.5, 71, "70%", ha="right",
                                                        color=INK2, fontsize=8)
    ax.set_ylabel("% missing"); ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=7); ax.grid(axis="x", visible=False)
    return fig


# ------------------------------------------------------------------ evaluation
def roc_pr_curves(y, probas: dict, title: str, labels: dict | None = None):
    labels = labels or {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, p in probas.items():
        algo = key.split("_")[0]
        col = ALGO_COLORS.get(algo, MUTED)
        name = labels.get(key, ALGO_NAMES.get(algo, key))
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=col, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=col, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves")
    base = float(np.mean(y))
    a2.axhline(base, color=MUTED, lw=1, ls="--"); a2.text(0.01, base + 0.015, f"prevalence {base:.3f}",
                                                         ha="left", fontsize=8, color=INK2)
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves", ylim=(0, 1.02))
    a1.legend(loc="lower right"); a2.legend(loc="lower left", bbox_to_anchor=(0, 0.06))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


def confusion_grid(y, preds: dict, title: str, ncols=5):
    n = len(preds); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3 * nrows), squeeze=False)
    for ax, (name, pr) in zip(axes.ravel(), preds.items()):
        cm = confusion_matrix(y, pr, labels=[0, 1])
        ax.grid(False)
        sns.heatmap(np.log1p(cm), annot=cm, fmt=",d", cmap=SEQ, cbar=False, ax=ax,
                    linewidths=2, linecolor=SURFACE, annot_kws={"size": 11})
        ax.set_xticklabels(["No fail", "Fail"]); ax.set_yticklabels(["No fail", "Fail"], rotation=0)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def metric_bars(df: pd.DataFrame, metrics, title: str, algo_col="algo", group_col=None):
    """Grouped horizontal bars: one panel per metric, one bar per algorithm."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 0.45 * len(df) + 1.4),
                             sharey=True, squeeze=False)
    labels = df[algo_col].map(ALGO_NAMES) + ("" if group_col is None else " · " + df[group_col])
    ypos = np.arange(len(df))
    for ax, m in zip(axes[0], metrics):
        ax.barh(ypos, df[m], color=[ALGO_COLORS[a] for a in df[algo_col]], height=0.62)
        for yp, v in zip(ypos, df[m]):
            ax.text(v, yp, f" {v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(m, fontsize=10); ax.set_xlim(0, max(1.0, df[m].max() * 1.18))
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(ypos); axes[0, 0].set_yticklabels(labels); axes[0, 0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def strategy_heatmap(pivot: pd.DataFrame, title: str, fmt=".3f", cbar_label="OOF PR-AUC"):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=SEQ, linewidths=2, linecolor=SURFACE, ax=ax,
                cbar_kws={"label": cbar_label}, mask=pivot.isna())
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(title); ax.grid(False)
    for i, j in zip(*np.where(pivot.isna().to_numpy())):
        ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=9, color=MUTED)
    ax.tick_params(axis="y", rotation=0)
    return fig


def time_bars(df: pd.DataFrame, title: str, algo_col="algo"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.45 * len(df) + 1.5), sharey=True)
    ypos = np.arange(len(df)); cols = [ALGO_COLORS[a] for a in df[algo_col]]
    for ax, col, lab in [(a1, "fit_time_s", "Training time (s, log scale)"),
                         (a2, "predict_ms_per_1k", "Inference (ms per 1,000 rows, log scale)")]:
        ax.barh(ypos, df[col], color=cols, height=0.62); ax.set_xscale("log")
        for yp, v in zip(ypos, df[col]):
            ax.text(v, yp, f" {v:,.2f}", va="center", fontsize=8, color=INK)
        ax.set_title(lab, fontsize=10); ax.grid(axis="y", visible=False)
    a1.set_yticks(ypos); a1.set_yticklabels(df[algo_col].map(ALGO_NAMES)); a1.invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def importance_bars(imp: pd.DataFrame, title: str, top=15):
    """imp: index = feature, columns = importance methods (each normalised to max 1)."""
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index[:top]]
    methods = list(imp.columns)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(imp) * max(1, len(methods) / 2) + 1.5))
    h = 0.8 / len(methods); ypos = np.arange(len(imp))
    for i, m in enumerate(methods):
        ax.barh(ypos + i * h, imp[m], height=h * 0.9, color=SERIES[i], label=m)
    ax.set_yticks(ypos + h * (len(methods) - 1) / 2); ax.set_yticklabels(imp.index)
    ax.invert_yaxis(); ax.set_xlabel("Relative importance (each method scaled to max = 1)")
    ax.legend(loc="lower right"); ax.set_title(title); ax.grid(axis="y", visible=False)
    return fig
