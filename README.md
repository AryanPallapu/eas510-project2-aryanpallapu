# Machine Failure Prediction for Predictive Maintenance

**Sri Veda Aryan Pallapu** · EAS 510: The Basics of AI (Fall 2026) · Project 2: Independent ML Project

This project asks: **how reliably can standard supervised classifiers detect rare equipment
failures from operational sensor data, and which modeling decisions matter most?** The decisions
studied are features, class-imbalance handling, hyperparameters and the decision threshold.

The same leakage-safe pipeline is applied **independently** to two public industrial datasets,
with no pooling and no cross-dataset transfer:

| | AI4I 2020 | APS Failure at Scania Trucks |
|---|---|---|
| One observation | one milling-process run (synthetic, modeled on a real process) | one truck with a failure (real fleet data) |
| Size | 10,000 rows | 60,000 train + 16,000 test (official split) |
| Inputs | product type + 5 sensors / settings | 170 anonymized counters and histogram bins |
| Target | machine failure (3.4% positive) | failure in the air-pressure system vs another component (1.7% / 2.3%) |

The comparison covers five algorithms: Logistic Regression (the linear reference), KNN, Decision
Tree, Random Forest and Gradient Boosting. Each runs as a default **baseline**, then under four
imbalance strategies (none, class weights, random oversampling, SMOTE), then **tuned** with
`RandomizedSearchCV`, with a decision threshold chosen on out-of-fold training predictions.
Test sets are scored once.

📄 **Report:** the graded final report `report.md` is due 2026-12-07 and will be linked here. The
current technical write-up of the full experiment is in
[`reports/technical_report_draft.md`](reports/technical_report_draft.md).

## Results summary (full data, `results/full/`)

Held-out test sets, tuned models at their OOF-selected thresholds. Higher is better except for cost.

| Model | AI4I F1 | AI4I recall | AI4I PR-AUC | APS recall | APS PR-AUC | APS cost (10·FP + 500·FN) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.496 | 0.471 | 0.464 | 0.955 | 0.865 | 14,570 |
| KNN | 0.647 | 0.662 | 0.685 | 0.939 | 0.877 | 15,230 |
| Decision Tree | 0.846 | 0.765 | 0.870 | 0.947 | 0.814 | 17,400 |
| Random Forest | **0.889** | **0.824** | 0.898 | **0.979** | 0.921 | **10,440** |
| Gradient Boosting ★ | 0.885 | 0.794 | **0.905** | 0.965 | **0.928** | 11,100 |

★ Selected on cross-validation (before the test set was scored) on both datasets.

Main findings so far:
- **Engineered physics features matter most on AI4I.** Temperature difference, mechanical power
  and tool wear × torque raised Gradient Boosting's CV PR-AUC from 0.78 to 0.91.
- **On APS, a cost-based threshold matters most.** It cut the selected model's test cost from
  27,460 at 0.5 to 11,100.
- **Resampling rarely helped.** It improved PR-AUC in only 1 of 10 algorithm/dataset pairs.

Full tables: `results/full/*_test_results.csv`, `*_strategy_cv.csv`, `*_tuned_cv.csv`.
Figures: `results/full/figures/`.

## Environment

- **Tested on:** Python 3.13.18, Ubuntu 24.04 (Linux x86_64), CPU only, 2 cores
- **Dependencies:** pinned in [`requirements.txt`](requirements.txt). No system packages are needed.
- **Randomness:** seed `RANDOM_STATE = 42` in [`config.py`](config.py), used for every split, CV
  shuffle, sampler, estimator and search.

## How to run

```bash
git clone https://github.com/AryanPallapu/eas510-project2-aryanpallapu.git
cd eas510-project2-aryanpallapu
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

`python main.py` runs the complete workflow on the bundled sample (`data/sample/`): loading,
preprocessing, CV over 5 algorithms × 4 imbalance strategies, tuning, threshold selection,
final test evaluation, saved models and figures. It prints the metric tables and writes them
to [`results/sample/`](results/sample/). **Measured: 133 s and ~330 MB peak memory** on the
test machine above. It needs no network access, downloads or GPU.

**Full experiment (the evidence in the report):**

```bash
python data/download.py        # fetches both datasets from UCI into data/full/ and verifies them
python main.py --full --fresh  # writes results/full/ and models/full/
```

The recorded full run took about 2.5 hours on 2 CPU cores (AI4I ~12 min, APS ~2 h 23 min, mostly tuning). Other jobs were competing for CPU at the time, so treat this as an upper estimate. Results are checkpointed
after every model, so an interrupted run resumes when relaunched without `--fresh`. Analysis
notebooks: `python scripts/build_notebooks.py` rebuilds and executes `notebooks/01–05`
(EDA, modeling, explainability, comparison).

Supporting experiments:
- APS imputation comparison: `python scripts/compare_aps_preprocessing.py`
- AI4I feature ablation: `python scripts/ablation_ai4i_features.py`

Tests: `python -m pytest -q`. Optional demo: `streamlit run app/streamlit_app.py`.

### Expected sample output

[`results/sample/summary.md`](results/sample/summary.md) holds the full tables. Key lines:

| Sample run (test split) | Selected model | Key metrics |
|---|---|---|
| AI4I (2,000 rows, 68 failures) | Random Forest, threshold 0.44 | F1 0.880, recall 0.809, PR-AUC 0.899 |
| APS sample (1,600 rows, 38 positives) | Random Forest, threshold 0.095 | recall 0.921, PR-AUC 0.832, cost 1,920 |

With the pinned versions, reruns on the same platform reproduce these numbers. On other CPUs
or BLAS/OpenMP builds, expect PR-AUC/F1 differences up to about ±0.01. FP/FN counts may shift
by one or two, because the APS sample test set has only 38 positives (one missed failure = 500
cost). Sample results check that the pipeline works; they use much smaller search budgets than
the full run, so they don't match `results/full/`.

## Data, permissions and resources

- **AI4I 2020**, UCI ML Repository, https://doi.org/10.24432/C5HS5C, CC BY 4.0. Bundled whole.
- **APS Failure at Scania Trucks**, Scania CV AB (2016), UCI #421, GPL v3. A stratified
  5% / 10% sample of the official train / test splits is bundled. The full data is downloaded.
- Selection procedure and differences: [`data/sample/README.md`](data/sample/README.md).
  Retrieved 2026-11-16.
- **Resources:** no GPU, paid services or credentials are needed. Full run: about 2.5 h on 2
  CPU cores. Peak memory of the full run was not measured; the sample run peaks at ~330 MB.
  Full-mode APS KNN models (~50 MB each) are regenerated rather than committed.

## Repository structure

```
├── main.py                 # single entry point (sample default, --full)
├── config.py               # seeds, split settings, CV/search budgets, paths per mode
├── requirements.txt
├── src/                    # data loading, preprocessing, feature engineering, models,
│                           # training (CV, tuning, thresholds), evaluation, explainability, plots
├── data/sample/            # bundled runnable sample (+ README)
├── data/download.py        # full-data download + verification
├── results/sample/         # documented sample-run outputs
├── results/full/           # full-experiment metrics, OOF/test predictions, figures
├── models/full/            # saved full-experiment pipelines (joblib)
├── notebooks/              # executed EDA / modeling / comparison notebooks
├── reports/                # execution plan, technical report draft
├── scripts/                # sample builder, supporting experiments, notebook builder
├── tests/                  # data, leakage and pipeline tests (pytest)
└── app/                    # optional Streamlit demo
```

## AI assistance

Claude (Anthropic) was used substantially in this project: for code generation, running and
debugging the experiments, and drafting documentation. The full disclosure, covering which
parts, how outputs were checked and what was changed, will be in the *Contributions and AI
Assistance* section of `report.md`.
