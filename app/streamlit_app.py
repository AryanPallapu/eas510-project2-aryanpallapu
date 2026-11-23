"""Interactive demo: failure prediction, model performance and feature importance.

    streamlit run app/streamlit_app.py

Requires trained models (python main.py --full) and executed
notebooks 02/04 for figures and importance files."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from src.feature_engineering import AI4IFeatureEngineer  # noqa: E402
from src.models import ALGO_NAMES  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
R, M, F = C.RESULTS_DIR, C.MODELS_DIR, C.FIGURES_DIR


@st.cache_resource
def load_model(ds, algo, variant):
    return joblib.load(M / f"{ds}_{algo}_{variant}.joblib")


@st.cache_data
def tuned_table(ds):
    return pd.read_csv(R / f"{ds}_tuned_cv.csv").set_index("algo")


@st.cache_data
def best_info(ds):
    return json.loads((R / f"{ds}_best_model.json").read_text())


@st.cache_data
def aps_test():
    from src.data_loader import load_aps
    return load_aps()[1]


ds_label = st.sidebar.radio("Dataset", ["AI4I 2020 (milling machine)", "APS Scania (truck air system)"])
ds = "ai4i" if ds_label.startswith("AI4I") else "aps"
best = best_info(ds)
tuned = tuned_table(ds)
algos = list(ALGO_NAMES)
algo = st.sidebar.selectbox("Model", algos, index=algos.index(best["algo"]),
                            format_func=lambda a: ALGO_NAMES[a] + ("  (selected best)" if a == best["algo"] else ""))
variant = st.sidebar.radio("Variant", ["tuned", "baseline"], horizontal=True)
default_thr = float(tuned.loc[algo, "threshold"]) if variant == "tuned" else 0.5
thr = st.sidebar.slider("Decision threshold", 0.005, 0.995, default_thr, 0.005,
                        help="Default = threshold chosen on out-of-fold training predictions "
                             "(max F1 for AI4I, min Scania cost for APS); 0.5 for baselines.")
model = load_model(ds, algo, variant)

st.title("Machine Failure Prediction")
st.caption("Independent ML project: AI4I 2020 and APS Scania, modeled separately. "
           "Predictions are statistical estimates, not maintenance decisions.")
tab_pred, tab_perf, tab_imp = st.tabs(["Predict", "Model performance", "Feature importance"])

with tab_pred:
    if ds == "ai4i":
        st.subheader("Enter machine sensor readings")
        c1, c2, c3 = st.columns(3)
        typ = c1.selectbox("Product quality type", ["L", "M", "H"])
        air = c1.slider("Air temperature [K]", 295.0, 305.0, 300.0, 0.1)
        proc = c2.slider("Process temperature [K]", 305.0, 314.0, 310.0, 0.1)
        rpm = c2.slider("Rotational speed [rpm]", 1150, 2900, 1500, 10)
        torque = c3.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.5)
        wear = c3.slider("Tool wear [min]", 0, 255, 100, 1)
        row = pd.DataFrame([{"Type": typ, "Air temperature [K]": air, "Process temperature [K]": proc,
                             "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear}])
        fe = AI4IFeatureEngineer().fit(row).transform(row)
        st.write("Derived features:", fe[["temp_diff [K]", "power [W]", "strain [Nm*min]"]].round(1))
    else:
        st.subheader("Score trucks from the official APS test set")
        st.caption("170 anonymized features can't be typed in by hand: pick a test record, or "
                   "upload a CSV with the same raw columns (missing values blank or 'na').")
        te = aps_test()
        up = st.file_uploader("Optional CSV upload", type="csv")
        if up is not None:
            row = pd.read_csv(up, na_values="na")
            row = row[[c for c in te.columns if c != "class"]]
        else:
            idx = st.number_input("Test record index", 0, len(te) - 1, int(np.flatnonzero(te["class"] == 1)[0]))
            row = te.drop(columns="class").iloc[[idx]]
            st.write(f"Actual label: **{'APS failure' if te['class'].iloc[idx] else 'other failure'}**")
    proba = model.predict_proba(row)[:, 1]
    for i, p in enumerate(proba):
        flag = p >= thr
        st.metric(f"Failure probability{'' if len(proba) == 1 else f' (row {i})'}", f"{p:.1%}",
                  "ALERT: predicted failure" if flag else "below threshold",
                  delta_color="inverse" if flag else "off")
    st.progress(float(min(proba[0], 1.0)))

with tab_perf:
    test = pd.read_csv(R / f"{ds}_test_results.csv")
    st.subheader("Held-out test results")
    st.dataframe(test[["model", "threshold_rule", "threshold", "accuracy", "precision", "recall",
                       "f1", "roc_auc", "pr_auc", "fp", "fn", "cost", "fit_time_s"]].round(4),
                 width="stretch")
    sub = "ai4i" if ds == "ai4i" else "aps"
    for fname in ["12_test_model_comparison.png", "13_roc_pr_curves.png", "14_confusion_matrices.png",
                  "15_threshold_tradeoff.png"]:
        p = F / sub / fname
        if p.exists():
            st.image(str(p), width="stretch")

with tab_imp:
    st.subheader("Which inputs drive the predictions")
    st.caption("Importance shows statistical association learned by the model, not causation.")
    sp = R / f"{ds}_shap_mean_abs.csv"
    if sp.exists():
        s = pd.read_csv(sp, index_col=0).head(15)
        st.bar_chart(s, horizontal=True)
    for fname in ["20_shap_summary.png", "19_permutation_importance.png", "18_impurity_importance.png"]:
        p = F / ("ai4i" if ds == "ai4i" else "aps") / fname
        if p.exists():
            st.image(str(p), width="stretch")
