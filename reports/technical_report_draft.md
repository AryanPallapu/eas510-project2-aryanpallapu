# Machine Failure Prediction for Predictive Maintenance — Technical Report

> **Draft (2026-11-16).** Working technical write-up of the full experiment. The graded final
> report will be `report.md` in the repository root (due 2026-12-07).


*Independent project · AI4I 2020 and APS Failure at Scania Trucks · all results from
executed experiments (`results/full/`), seed 42.*

---

## 1. Project overview

This project builds and evaluates supervised classifiers that flag equipment failures from
operational sensor data. One methodology is applied **independently** to two public
datasets with different feature spaces and failure definitions: a synthetic milling-machine
process (AI4I 2020) and real air-pressure-system data from Scania trucks (APS). For each
dataset, five algorithms are compared. The pipeline covers exploratory analysis,
leakage-safe preprocessing, imbalance handling, hyperparameter tuning, cost- or F1-driven
threshold selection, a single held-out test evaluation, and explainability.

## 2. Problem statement

Unplanned failures cause downtime and repair cost. A predictive-maintenance model should:

1. **detect as many real failures as possible**: a missed failure is usually far more
   expensive than an unnecessary inspection;
2. keep false alarms at a level maintenance teams can act on;
3. be explainable enough for engineers to trust.

Formally, each dataset is a binary classification task with a rare positive class (1.7–3.4%).
Accuracy is not a useful objective: predicting "no failure" for every row scores 96.6%
(AI4I) and 97.7% (APS test). Model selection therefore uses **PR-AUC** (threshold-free) during
tuning, then a decision threshold chosen for **F1** (AI4I) or the **official Scania
misclassification cost** (APS).

## 3. Datasets

| | AI4I 2020 | APS Failure at Scania Trucks |
|---|---|---|
| Source | UCI #601, doi:10.24432/C5HS5C (CC BY 4.0) | UCI #421, Scania CV AB (GPL v3) |
| Nature | Synthetic, modeled on a real milling process | Real trucks in everyday use |
| Size | 10,000 rows | 60,000 train + 16,000 test (official split) |
| Inputs | Type (L/M/H), air & process temperature, speed, torque, tool wear | 170 anonymized features: 100 counters + 7 histograms × 10 bins |
| Target | `Machine failure` (any of TWF, HDF, PWF, OSF, RNF) | `pos` = failure of a specific APS component; `neg` = failure of *another* component |
| Positives | 339 (3.39%) | 1,000 (1.67%) train · 375 (2.34%) test |
| Missing | none | 8.3% of cells; 99% of rows |

Parsing quirks handled in `src/data_loader.py`: a UTF-8 BOM on the AI4I header; a 20-line
license header and `"na"` missing tokens in the APS CSVs.

**Framing note (APS).** Every APS record is a truck that had *some* failure. The model
attributes the failure to the APS component or not; it does not separate failing trucks from
healthy ones.

## 4. Exploratory data analysis

Full analysis: `notebooks/01_ai4i_eda.ipynb`, `notebooks/03_aps_eda.ipynb`; figures in `results/full/figures/`.

### 4.1 AI4I 2020
- **Failure physics is visible in the data.** All 115 heat-dissipation failures have
  process−air ΔT < 8.6 K *and* speed < 1,380 rpm. All 95 power failures have mechanical power
  (torque × ω) outside 3.5–9 kW. Overstrain failures start at type-specific limits of tool wear
  × torque (first occurrences: 11,003 / 12,337 / 13,235 Nm·min for L/M/H). Tool-wear failures
  occur at 198–253 min of tool wear.
- **Torque and speed are strongly anti-correlated** (r = −0.875, a result of constant-power
  generation). Their product carries information that neither has alone.
- **Outliers are signal:** IQR outliers in torque, power and strain fail at 90–100%, so none are
  removed.
- **Product type:** failure rate L 3.92%, M 2.77%, H 2.09%.
- **Label quirks:** 9 failures have no subtype flag; 18 of 19 RNF rows are labeled "no failure."
  Labels are used as published and the quirks are documented.

### 4.2 APS Scania (training set only)
- **No −1 sentinel values exist** (zero cells; no negative values anywhere), contrary to a common
  assumption about this dataset.
- **A different sentinel does exist:** 15.8% of `ac_000` values are 2,130,706,432 (= 0x7F000000)
  or two adjacent codes, far above the next real value (265,766). Failure rate in those rows: 0.16%.
- **Missingness is informative.** Positives average 29 missing values per row vs 14 for negatives
  (the count alone has AUC 0.68). Rows missing `ac_000` fail at 13.9%; the `bm_000–br_000`
  block fails at ≈0.18% when missing vs 4.5–8.5% when present.
- **Extreme skew** (median |skew| 17.5; 114 features above 10) and strong redundancy (137 feature
  pairs with |r| > 0.9). Several single features reach a univariate AUC of ≈0.97 (`ci_000`,
  `ck_000`, `aa_000`, `bt_000`).
- `cd_000` is constant. There are no duplicate rows.

## 5. Methodology

```
raw CSV ──► loader ──► [AI4I] stratified 80/20 split      [APS] official train/test
                              │                                    │
             5-fold stratified CV on TRAIN only  ◄─────────────────┘
             ├─ stage 1: 5 algorithms × 4 imbalance strategies (defaults)
             ├─ stage 2: RandomizedSearchCV (avg. precision) with each algorithm's best strategy
             │           → 5-fold OOF probabilities → threshold (max F1 | min Scania cost)
             └─ stage 3: refit on full TRAIN → score TEST once → save pipeline
```

Each model is a single `imblearn.Pipeline`. Every learned step (cleaning, imputation, scaling,
resampling) is therefore fit only on the data the pipeline is trained on: a CV training fold,
or the full training set at the end. The test sets play no part in any selection decision.

## 6. Preprocessing and feature engineering

### AI4I
| Step | Decision | Reason |
|---|---|---|
| Identifiers | drop `UDI`, `Product ID` | no predictive meaning |
| Leakage | drop TWF/HDF/PWF/OSF/RNF from inputs | they define the target |
| Features | add ΔT = process − air temperature; power = torque·rpm·2π/60; strain = torque × tool wear | the documented failure mechanisms (§4.1) |
| Encoding | one-hot `Type` | 3 levels; also lets SMOTENC treat it as categorical |
| Scaling | standardize | needed by LR and KNN; no effect on trees |

**Ablation (5-fold CV, OOF PR-AUC, without → with engineered features):** Gradient Boosting
0.782 → **0.909**; Random Forest 0.760 → **0.895**; Decision Tree 0.483 → 0.688; Logistic
Regression 0.459 → 0.496.

### APS
Pipeline: `APSCleaner` (drop constant columns, learned on the training fold · flag `ac_000`
sentinel then set it to NaN · add `n_missing`) → `SimpleImputer(median)` → `log1p` →
`StandardScaler`.

**Imputation comparison** (5-fold CV, training set; `results/full/aps_imputation_comparison.csv`):

| Variant | LR PR-AUC | HistGB PR-AUC | HistGB cost* |
|---|---|---|---|
| mean | 0.763 | 0.885 | 40,800 |
| median | 0.766 | 0.880 | 42,950 |
| median + sentinel flag + missing count (**chosen**) | 0.762 | 0.881 | 42,000 |
| … + drop columns >70% missing | 0.765 | 0.881 | 42,290 |
| … + histogram totals | 0.762 | 0.879 | 41,700 |
| HistGB native NaN + flags | — | 0.888 | 42,090 |

\*OOF cost at its own best threshold, on the 0.005-step grid used at the time.

All variants are within about 0.01, inside the fold-to-fold standard deviation (≈0.01–0.02).
The chosen variant is the most defensible rather than the measurably best: the median is
robust to the extreme skew, and the flags keep the documented sentinel. One complete
imputation shared by all models keeps the comparison like-for-like and is required by SMOTE.

## 7. Algorithms

| Algorithm | Implementation | Search space (key parameters) |
|---|---|---|
| Logistic Regression | `LogisticRegression` (L2, lbfgs) | C ∈ [1e-3, 1e2] (log) |
| KNN | `KNeighborsClassifier` | k ∈ {3…31}, uniform/distance weights, p ∈ {1, 2} |
| Decision Tree | `DecisionTreeClassifier` | depth, min_samples_leaf, criterion |
| Random Forest | `RandomForestClassifier` | trees, depth, min_samples_leaf, max_features |
| Gradient Boosting | AI4I: `GradientBoostingClassifier` · APS: `HistGradientBoostingClassifier` (early stopping on 10% of the training fold) | learning rate, depth/leaves, iterations, subsample/L2 |

The classic `GradientBoostingClassifier` took 183 s per fit on APS on the 2-core machine, which
made a 5-fold search infeasible. HistGB was used for APS tuning; the classic version was fit
once, untuned, as a reference. `GradientBoostingClassifier` has no `class_weight`, so a thin
subclass applies balanced sample weights computed from each training fold.

**Imbalance strategies:** none · class weights (`balanced`; not applicable to KNN) · random
oversampling · SMOTE (SMOTENC on raw AI4I rows so `Type` stays categorical; on APS, applied
after imputation because SMOTE cannot handle NaN).

## 8. Experimental setup

| | AI4I | APS |
|---|---|---|
| Hold-out | stratified 20% (2,000 rows, 68 failures) | official test (16,000 rows, 375 positives) |
| CV | StratifiedKFold(5, shuffle, seed 42) | same |
| Search | RandomizedSearchCV, 15–30 iterations, 5 folds | 8–25 iterations; 3 folds for KNN and RF (compute) |
| Search metric | average precision | average precision |
| Threshold | max F1 on OOF predictions | min 10·FP + 500·FN on OOF predictions |
| Threshold grid | 0.0005–0.0095 (step 0.0005), 0.01–0.995 (step 0.005) | same |
| Final pick | best OOF F1 (tie-break OOF PR-AUC) | lowest OOF cost (tie-break OOF PR-AUC) |
| Hardware | 2 CPU cores, 7 GB RAM | same |

Reported metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, confusion matrix,
fit time and inference time (ms per 1,000 rows), plus the Scania cost for APS.

## 9. Results

### 9.1 Baselines (5-fold CV, defaults, no imbalance handling, threshold 0.5)

| | AI4I PR-AUC | AI4I recall | AI4I F1 | APS PR-AUC | APS recall | APS F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.496 | 0.251 | 0.378 | 0.819 | 0.667 | 0.738 |
| KNN | 0.515 | 0.332 | 0.461 | 0.650 | 0.549 | 0.653 |
| Decision Tree | 0.688 | 0.830 | 0.826 | 0.500 | 0.708 | 0.703 |
| Random Forest | 0.895 | 0.819 | 0.889 | 0.875 | 0.698 | 0.789 |
| Gradient Boosting | 0.909 | 0.830 | 0.889 | 0.871 | 0.702 | 0.782 |

### 9.2 Imbalance handling (5-fold CV, OOF PR-AUC; full tables in `results/full/*_strategy_cv.csv`)

| | AI4I: none / class weight / ROS / SMOTE | APS: none / class weight / ROS / SMOTE |
|---|---|---|
| Logistic Regression | **0.496** / 0.455 / 0.455 / 0.451 | **0.819** / 0.762 / 0.763 / 0.754 |
| KNN | **0.515** / n/a / 0.393 / 0.362 | **0.650** / n/a / 0.585 / 0.521 |
| Decision Tree | **0.688** / 0.637 / 0.648 / 0.325 | **0.500** / 0.381 / 0.385 / 0.330 |
| Random Forest | **0.895** / 0.884 / 0.885 / 0.772 | **0.875** / 0.839 / 0.846 / 0.817 |
| Gradient Boosting | **0.909** / 0.901 / 0.902 / 0.850 | 0.871 / 0.840 / **0.896** / 0.889 |

Resampling and class weights reliably **raise recall at the default 0.5 threshold** (e.g. AI4I GB
0.83 → 0.89 with class weights; APS LR 0.67 → 0.93), but they mostly **move the operating point**
rather than improve ranking quality. In 9 of 10 cases they lowered PR-AUC. The one exception is
APS Gradient Boosting with random oversampling (+0.025). SMOTE was the weakest strategy on
AI4I: interpolating between rule-generated failure points creates synthetic rows that don't
follow the rules. Explicit threshold selection gives the same recall gain more transparently.

### 9.3 Tuning (5-fold CV, OOF on the training set)

| | AI4I tuned PR-AUC | Δ vs baseline | thr | OOF F1 | APS tuned PR-AUC | Δ vs baseline | thr | OOF cost |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.496 | 0.000 | 0.24 | 0.497 | 0.821 | +0.002 | 0.015 | 47,760 |
| KNN | 0.691 | +0.176 | 0.19 | 0.676 | 0.817 | +0.167 | 0.04 | 57,940 |
| Decision Tree | 0.894 | +0.206 | 0.41 | 0.867 | 0.731 | +0.231 | 0.003 | 65,990 |
| Random Forest | 0.901 | +0.006 | 0.42 | 0.888 | 0.876 | +0.001 | 0.02 | 43,610 |
| Gradient Boosting | **0.909** | 0.000 | 0.58 | **0.890** | **0.899** | +0.028 | 0.0025 | **39,360** |

Tuning mainly rescued the high-variance learners. KNN improved through more neighbors and
distance weighting; Decision Tree through depth and leaf-size limits. The ensembles were
already near their ceiling at default settings.

### 9.4 Held-out test results (scored once; tuned models at OOF-selected thresholds)

**AI4I** (2,000 rows, 68 failures)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **Gradient Boosting** ★ | **1.000** | 0.794 | 0.885 | **0.905** | 0.968 | 0.000 | 0 | 14 |
| Random Forest | 0.966 | **0.824** | **0.889** | 0.898 | **0.981** | 0.001 | 2 | 12 |
| Decision Tree | 0.945 | 0.765 | 0.846 | 0.870 | 0.971 | 0.002 | 3 | 16 |
| KNN | 0.634 | 0.662 | 0.647 | 0.685 | 0.931 | 0.013 | 26 | 23 |
| Logistic Regression | 0.525 | 0.471 | 0.496 | 0.464 | 0.926 | 0.015 | 29 | 36 |

**APS** (16,000 rows, 375 positives; cost = 10·FP + 500·FN)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FN | **Cost** |
|---|---|---|---|---|---|---|---|---|
| **Gradient Boosting (HistGB + ROS)** ★ | 0.440 | 0.965 | **0.605** | **0.928** | 0.993 | 460 | 13 | 11,100 |
| Random Forest | 0.363 | **0.979** | 0.530 | 0.921 | **0.996** | 644 | **8** | **10,440** |
| Logistic Regression | 0.371 | 0.955 | 0.534 | 0.865 | 0.990 | 607 | 17 | 14,570 |
| KNN | 0.486 | 0.939 | 0.640 | 0.877 | 0.968 | 373 | 23 | 15,230 |
| Decision Tree | 0.324 | 0.947 | 0.483 | 0.814 | 0.978 | 740 | 20 | 17,400 |
| *GB classic, untuned, threshold 0.5 (reference)* | 0.899 | 0.691 | 0.781 | 0.870 | 0.992 | 29 | 116 | 58,290 |

★ = selected on cross-validation before the test set was scored.

**Effect of the decision threshold (APS, selected model):** at 0.5 the cost is 27,460 (54 missed
failures, 46 false alarms); at 0.0025 it is 11,100 (13 missed, 460 false alarms). Because one
missed failure costs as much as 50 inspections, the optimum flags 822 of 16,000 trucks (5.1%)
to catch 96.5% of APS failures.

**Selection vs test outcome.** On APS, tuned Random Forest scored a lower test cost (10,440)
than the selected Gradient Boosting (11,100). Gradient Boosting was chosen because its OOF
cost was lower (39,360 vs 43,610); switching after seeing test results would be test-set
selection. The gap (660, roughly one missed failure) is within the variation one would expect
from a test set with 375 positives. On AI4I, Random Forest and Gradient Boosting are likewise
within a failure or two of each other on every metric. For context, the IDA 2016 challenge's
top-three published costs on this test set are 9,920 / 10,900 / 11,480.

**Baseline vs optimized on test.** AI4I: tuning barely moved the ensembles. GB is identical
at its operating point (F1 0.885, precision 1.000 for both baseline and tuned; PR-AUC 0.901 →
0.905). The gains are in the unstable learners: Decision Tree PR-AUC 0.642 → 0.870 and KNN
0.531 → 0.685. APS: GB cost 48,790 (baseline at 0.5) → 11,100 (tuned + ROS + cost threshold). Per-model rows: `results/full/*_test_results.csv`.

### 9.5 Training and inference cost (final fit on the full training set, idle machine)

| | AI4I fit (s) | AI4I ms / 1k rows | APS fit (s) | APS ms / 1k rows |
|---|---|---|---|---|
| Logistic Regression | 0.03 | 3.8 | 3.2 | 6.0 |
| KNN | 0.02 | 67.9 | 2.2 | 3,000 |
| Decision Tree | 0.04 | 4.1 | 9.6 | 5.6 |
| Random Forest | 1.5 | 38.4 | 53.2 | 11.5 |
| Gradient Boosting | 4.1 | 6.2 | 16.4 | 10.1 |
| GB classic (APS reference) | — | — | 186.6 | 6.6 |

KNN trains instantly but is by far the slowest at inference on APS (distance-weighted Manhattan
search over 60k × 172). Search wall-clock times are in `results/full/*_tuned_cv.csv`. The APS RF and
GB search times are inflated: an orphaned worker process competed for CPU during part of
those searches.

## 10. Explainability

Methods: impurity importance (Random Forest; Gradient Boosting, using the classic GB reference
for APS because HistGB exposes none), permutation importance of the raw inputs on the test set
(drop in average precision), and TreeSHAP on the selected Gradient Boosting model. Outputs:
`results/full/*_feature_importance.csv`, `results/full/*_shap_mean_abs.csv`, `results/full/figures/*/18–20_*.png`.

**AI4I.** By mean |SHAP| the drivers are rotational speed (0.349), tool wear (0.347) and ΔT
(0.343), then power (0.137) and strain (0.115). Raw air and process temperature are near zero
on their own; they matter through their difference. Permutation importance of the raw sensors
ranks torque, speed, air temperature and tool wear highest, because permuting a sensor
also corrupts every feature derived from it. The SHAP patterns match the documented mechanisms:
low ΔT and low speed push toward failure, and high tool wear, extreme power and high strain
push toward failure.

**Error analysis (AI4I test, selected model).** Recall by failure mode: OSF 15/15, PWF 12/12,
HDF 25/28, **TWF 0/9**, and 0/2 for failures with no subtype flag. By construction, tool-wear
failures occur at a *random* wear time between 200 and 240 min, so sensor readings say a
machine is in the risk window but not when it will fail. This sets a recall ceiling of about
0.80–0.85 that better models cannot remove.

**APS.** The same small set of anonymized features leads under every method: `aa_000`,
`ck_000`, `ci_000`, `ag_002`, `bs_000`, `aq_000` and `bj_000`. The engineered `n_missing`
ranks 46th of 171 by SHAP and the sentinel flag 69th: useful, but secondary.

**Association, not causation.** All importances describe how the trained models use
correlated inputs inside these datasets. They don't show that changing a feature would prevent
a failure. On AI4I, the importances recover the simulation's generating rules. On APS, the
features are anonymized, so even a consistent ranking can't be tied to a physical component
without Scania's documentation.

## 11. Comparative analysis

| Aspect | AI4I 2020 | APS Scania |
|---|---|---|
| Application | single machine process, failure-mode level | fleet component attribution |
| Size / dimensionality | 10k × 6 | 76k × 170 |
| Data quality | clean; synthetic; small label quirks | 8% missing, informative missingness, sentinel codes, extreme skew |
| Imbalance | 28:1 | 59:1 (train) |
| What drives performance | engineered physics features (+0.13 PR-AUC for GB) | many strong raw counters; a cost-aware threshold |
| Best linear model | weak (PR-AUC 0.46 test): failure rules are interactions | competitive (0.87): log-scaled counters separate almost monotonically |
| Best model | Gradient Boosting (RF essentially tied) | Gradient Boosting with oversampling (RF essentially tied) |
| Effect of resampling | none positive | helps only GB (+0.025 PR-AUC) |
| Effect of the threshold | modest (F1-optimal 0.19–0.58) | large (cost-optimal ≈ 0.003–0.04) |
| Main error source | intrinsically random failure modes (TWF, RNF) | overlap between APS and other faults; cost asymmetry forces many false alarms |
| Compute | seconds | minutes per model; KNN inference and classic GB training become bottlenecks |

What transfers between the two studies is **methodology, not models**. In both, ensembles of
trees led, tuning mattered most for unstable learners, imbalance was best handled by explicit
threshold selection, and the error metric had to match the business cost. Nothing here
suggests a model trained on one dataset would work on the other. Their inputs, labels and
generating processes are unrelated.

## 12. Limitations

1. **AI4I is synthetic and rule-generated.** Its high scores show that the models recover the
   simulation's rules, not that they would perform this well on a real machine.
2. **APS labels are fault attribution among failed trucks**, not failure vs health. Features
   are anonymized, so findings can't be mapped to physical components.
3. **Test-set size limits precision of comparisons.** With 68 (AI4I) and 375 (APS) positives,
   differences of a few missed failures are within noise, so the top two models are effectively tied.
4. **Search budgets were constrained** by a 2-core machine. APS RF and KNN used 3-fold search
   with 8 configurations. The APS RF space excluded `max_features=0.3` after it exceeded 10 min
   per fit. Classic GB was not tuned on APS. Larger searches might change the RF/GB ordering.
5. **Thresholds** are chosen on OOF predictions from five fold models and applied to a model
   refit on all training data, a standard approximation. Probabilities were not calibrated.
6. **HistGB early stopping with oversampling:** the internal 10% validation split is drawn from
   the oversampled training fold, so it can contain duplicates of training positives. This only
   affects when boosting stops; it never touches CV validation or test rows.
7. **Timing figures** come from single runs on a shared 2-core VM and should be read as relative.
8. **Temporal aspects are ignored.** Both datasets are treated as i.i.d. snapshots. Real
   deployment would need time-aware validation and drift monitoring.

## 13. Conclusion

Across two very different predictive-maintenance problems, tree ensembles gave the best
failure detection, and the largest gains came from domain-aware decisions rather than model
complexity. On AI4I, three physically motivated features lifted Gradient Boosting's PR-AUC
from 0.78 to 0.91. The selected model reached test F1 0.885 with zero false alarms. Its
remaining misses are dominated by tool-wear failures that are random by design. On APS,
aligning the decision threshold with the official cost cut the selected model's test cost from
27,460 at the default threshold to 11,100, at 96.5% recall. That is within the range of the
published challenge results, achieved with standard scikit-learn models and no test-set tuning.
Class-imbalance techniques mattered less than expected: threshold selection on out-of-fold
predictions delivered the recall gains more transparently, and resampling improved ranking
quality in only one of ten cases.

## 14. References

1. S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications,"
   *Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020.
2. AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository, 2020.
   https://doi.org/10.24432/C5HS5C
3. Scania CV AB, "APS Failure at Scania Trucks," UCI Machine Learning Repository, dataset #421,
   2016 (also listed as IDA2016Challenge, https://doi.org/10.24432/C5V60Q). Industrial Challenge,
   15th International Symposium on Intelligent Data Analysis (IDA 2016).
4. N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE: Synthetic Minority
   Over-sampling Technique," *Journal of Artificial Intelligence Research* 16, 321–357, 2002.
5. S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions,"
   *Advances in Neural Information Processing Systems* 30, 2017.
6. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR* 12, 2825–2830, 2011.
7. G. Lemaître, F. Nogueira, C. K. Aridas, "Imbalanced-learn: A Python Toolbox to Tackle the
   Curse of Imbalanced Datasets in Machine Learning," *JMLR* 18(17), 1–5, 2017.
8. L. Breiman, "Random Forests," *Machine Learning* 45, 5–32, 2001.
9. J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine,"
   *Annals of Statistics* 29(5), 1189–1232, 2001.
