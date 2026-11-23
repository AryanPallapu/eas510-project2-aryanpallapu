# Execution Plan — Machine Failure Prediction for Predictive Maintenance

Phase 1 deliverable. Everything below about the data was measured on the raw files
(`scripts/download_data.py` re-downloads and re-verifies them).

## 1. Data acquisition — status

| | AI4I 2020 | APS Failure at Scania Trucks |
|---|---|---|
| Source | UCI #601 (official zip) | UCI #421 (official zip) |
| Files | `ai4i2020.csv` | `aps_failure_training_set.csv`, `aps_failure_test_set.csv`, description |
| Shape | 10,000 × 14 | train 60,000 × 171 · test 16,000 × 171 |
| Positives | 339 (3.39%) | train 1,000 (1.67%) · test 375 (2.34%) |
| Imbalance ratio | ~28 : 1 | ~59 : 1 (train) |
| Missing | none | 8.3% of cells; 99% of rows have ≥1 missing |
| Duplicates | none (rows or Product IDs) | none |
| Parsing quirks | UTF-8 BOM on first column | 20-line GPL header before CSV header; missing = `"na"` |
| Origin | **Synthetic** (generated to mirror real milling data) | Real fleet data, features anonymized |
| License | CC BY 4.0 | GPL v3 |

Both match the official specifications. Loader tests: `tests/test_data_loader.py` (passing).

## 2. Findings that change the original brief

**F1 — APS has no `-1` sentinel values.** Zero cells equal −1 and no column has negative
values in either split. Missing values are encoded only as `"na"`. The brief's "−1" check
is done and documented as a negative result.

**F2 — APS has a real sentinel, in `ac_000`.** About 9,468 training values (16%) sit in
[2130706432, 2³¹), mainly 2130706432 (= `0x7F000000`), 2130706434 and 2130706438. No other
column has values in this range. The failure rate in those rows is 0.16% against 1.67%
overall, so the code carries information (probably "not applicable / not recorded").
Plan: add a binary indicator `ac_000_is_sentinel`, set the raw value to NaN, and impute
it with the rest. Compare against leaving it as is in Phase 3.

**F3 — APS: constant column.** `cd_000` has a single value in both splits → drop.

**F4 — APS: heavy missingness in a few columns.** 8 columns are >50% missing and 7 are >70%.
The threshold for dropping a column is chosen on training data only. Missing-indicator
features are evaluated because missingness itself may relate to the class.

**F5 — APS: 7 histogram groups** (`ag, ay, az, ba, cn, cs, ee`, 10 bins each = 70 of the
170 features). The other 100 are single counters. Values are heavy-tailed (max ≈ 8.6e9),
so log1p is needed for the distance and linear models (LR, KNN). Candidate engineered
features: per-histogram totals and normalized bin shares.

**F6 — AI4I label inconsistencies.** 9 rows have `Machine failure = 1` with no subtype
flagged. 18 of the 19 RNF (random failure) rows have `Machine failure = 0`. The target is
used as published. These rows are documented as label noise, not "fixed."

**F7 — AI4I is synthetic, with deterministic failure rules.** The UCI documentation defines
the failure modes as threshold rules: heat dissipation (air–process temperature difference
plus speed), power (torque × angular speed), overstrain (tool wear × torque, threshold by
product type) and a tool-wear window. Engineered features that reproduce these quantities
will push scores very high. That is a property of the dataset, not evidence that the model
would work in a real plant, and the report will say so. RNF (0.1% random) sets a floor
that no model can predict.

**F8 — The APS task is fault attribution, not failure vs. healthy.** According to the
dataset description, the negative class is trucks with failures in components *unrelated*
to the APS. The model therefore answers "is this failure in the APS component?"
for trucks that have already failed. The README and report will frame it that way and
won't present it as detecting failures in healthy trucks.

## 3. Key methodological decisions

| Decision | Choice | Rationale |
|---|---|---|
| AI4I split | Stratified 80/20, seed 42; test set held out | No official split exists |
| APS split | Official train/test kept as is; test touched **once** at the end | Comparable to published results; no leakage |
| CV | StratifiedKFold(5, shuffle, seed 42) on training data only | Keeps the class ratio in each fold |
| Leakage control | All imputation, scaling, encoding and resampling inside an `imblearn.Pipeline` fit per fold | SMOTE/oversampling never sees validation or test rows |
| AI4I inputs | Drop `UDI`, `Product ID`, all 5 subtypes; one-hot `Type` (ordinal also tested) | IDs carry no signal; subtypes define the target |
| AI4I features | Raw 5 sensors + ΔT, mechanical power (W), torque × wear | Physically motivated (see F7); ablation with and without them |
| Gradient Boosting | `GradientBoostingClassifier` on AI4I; `HistGradientBoostingClassifier` on APS | Measured: classic GB = 183 s per fit on APS vs 3 s for HistGB; a full search with the classic version isn't feasible on this machine. Classic GB on APS gets one untuned run for reference |
| Tuning | `RandomizedSearchCV`, scored on average precision (PR-AUC) | Threshold-independent and robust to imbalance; accuracy is ignored |
| Decision threshold | Chosen on out-of-fold training predictions (max F1 for AI4I; min Scania cost for APS) | Never tuned on the test set |
| APS headline metric | Official cost = 10·FP + 500·FN, reported with recall, F1, PR-AUC | Lets results be compared with the IDA 2016 challenge (winning score 9,920) |
| Seeds | `RANDOM_STATE = 42` everywhere (`src/config.py`) | Reproducibility |

## 4. Experiment matrix (per dataset)

5 algorithms × 4 imbalance strategies (none, class weight, RandomOverSampler, SMOTE)
→ baseline (default hyperparameters) → tuned (best strategy per algorithm).

Notes:
- KNN and plain GB have no `class_weight`. For GB, use `sample_weight`; for KNN that cell is
  "not applicable" (distance weighting is a different thing).
- HistGB supports `class_weight` and native NaN handling; both are used.
- SMOTE on APS runs after imputation (it can't handle NaN), and on 170 log-scaled
  dimensions. Its benefit is an open question, not an assumption.

Recorded for every run: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR,
confusion matrix, fit time, predict time; for APS, also the Scania cost.
Results go to `results/{dataset}_cv_results.csv` and `results/{dataset}_test_results.csv`.
Models are saved to `models/{dataset}_{algo}_{variant}.joblib`.

## 5. Compute budget (2 CPU cores, 7 GB RAM — measured)

| Fit on full APS train | Time |
|---|---|
| HistGB, 100 iterations | 3 s |
| Random Forest, 200 trees | 72 s |
| GradientBoosting, 100 trees | 183 s |
| KNN predict on 16k rows | 4 s |

Search budgets: AI4I ≈ 30 configurations × 5 folds per algorithm (cheap). APS: HistGB and
LR ≈ 30 configurations, RF ≈ 10 (100 trees during the search, refit larger), KNN ≈ 8.
Expected APS modeling run: roughly 1.5–3 hours, run in chunks with checkpointed results.
SHAP: TreeExplainer on RF/HistGB using a 2,000-row sample for APS.

## 6. Architecture

```
predictive-maintenance-project/
├── datasets/{ai4i_2020,aps_scania}/raw/   # git-ignored; fetched by scripts/download_data.py
├── notebooks/01–05_*.ipynb                # narrative + figures; call into src/
├── src/
│   ├── config.py            # paths, seeds, constants            ✅
│   ├── data_loader.py       # raw parsing only                    ✅
│   ├── preprocessing.py     # per-dataset sklearn transformers / ColumnTransformers
│   ├── feature_engineering.py
│   ├── models.py            # algorithm registry + search spaces
│   ├── train.py             # CV, tuning, imbalance strategies, saving (CLI)
│   ├── evaluate.py          # metrics, cost, threshold selection
│   ├── explain.py           # impurity / permutation / SHAP importance
│   └── visualization.py
├── scripts/download_data.py                                         ✅
├── tests/                   # loader + leakage tests                ✅ (loaders)
├── models/ results/ figures/{ai4i,aps,comparison}/ reports/
├── app/streamlit_app.py     # Phase 9 (optional)
├── requirements.txt  .gitignore  README.md
```
Changes from the brief: added `config.py`, `data_loader.py`, `models.py`, `explain.py`,
`scripts/`, `tests/` and `app/`. Notebooks stay thin and call `src/`, so the pipeline also runs
headless (`python -m src.train --dataset aps`).

## 7. Phases

| Phase | Output | Exit check | Status |
|---|---|---|---|
| 1 Acquisition & plan | This doc, loaders, download script | Data verified | ✅ |
| 2 EDA | Notebooks 01 and 03 executed; figures in `figures/` | All brief EDA items covered for both datasets | ✅ |
| 3 Preprocessing & FE | `preprocessing.py`, `feature_engineering.py`, leakage tests | Pipelines fit on training data only; tests pass | ✅ |
| 4 Baselines | 5 models × no imbalance handling, CV metrics and times | `*_strategy_cv.csv` | ✅ |
| 5 Imbalance + tuning | Strategy comparison; tuned models; OOF thresholds | Baseline vs tuned table | ✅ |
| 6 Evaluation & explainability | Single test-set evaluation; ROC/PR/CM plots; importance + SHAP | `*_test_results.csv` | ✅ |
| 7 Comparative analysis | Notebook 05 | Covers every comparison point in the brief; no cross-dataset generalization claims | ✅ |
| 8 Docs | `reports/technical_report.md`, `README.md` | Every number traces to a results CSV | ✅ |
| 9 Streamlit (optional) | `app/streamlit_app.py` (AI4I inputs; APS performance view) | Runs locally | ✅ |

## 8. References (to cite in the report)
- Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. *Third International Conference on Artificial Intelligence for Industries (AI4I)*. AI4I dataset: UCI ML Repository, doi:10.24432/C5HS5C.
- Scania CV AB (2016). APS Failure at Scania Trucks. UCI ML Repository, dataset #421, https://archive.ics.uci.edu/dataset/421. The same data is also listed as "IDA2016Challenge" (#414, doi:10.24432/C5V60Q). Used in the IDA 2016 Industrial Challenge.
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR* 16.
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12.


> **Status (end of project):** all phases completed. Deviations from this plan are documented in
> `technical_report.md` (finer threshold grid, reduced APS Random Forest search space, uniform
> median imputation for all APS models).
