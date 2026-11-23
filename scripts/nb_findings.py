"""Findings cells, written after inspecting the executed outputs (all numbers come from results/full/*.csv)."""

FINDINGS = {
"ai4i": """## 10. Findings

**Model ranking (5-fold CV, OOF).** Gradient Boosting (PR-AUC 0.909) ≈ Random Forest (0.901)
≈ tuned Decision Tree (0.894) ≫ KNN (0.691) ≫ Logistic Regression (0.496). The three tree
models are within about one fold standard deviation of each other (≈0.02–0.03). The failure
modes are thresholds and interactions (ΔT *and* speed; torque × wear above a type-specific
limit), which axis-aligned splits capture directly and a linear model can't.

**Selected model:** tuned Gradient Boosting (500 shallow trees, depth 2, lr ≈ 0.019),
threshold 0.58 chosen on OOF F1. **Test set:** precision 1.000, recall 0.794, F1 0.885,
PR-AUC 0.905, ROC-AUC 0.968. That is 0 false alarms and 14 missed failures out of 68.

**Imbalance handling did not improve ranking quality.** For every algorithm, class weighting,
random oversampling and SMOTE kept or *lowered* OOF PR-AUC. They raise recall at the default
0.5 threshold at a large cost in precision (e.g. GB with class weights: recall 0.83 → 0.89,
precision 0.96 → 0.56). SMOTE hurt the most: interpolating between the few, rule-generated
failure points creates synthetic rows that don't follow the rules. Choosing the decision
threshold on OOF predictions achieves the same trade-off more transparently, so no tuned
model ended up using resampling.

**Where the recall ceiling comes from.** The best model catches 100% of power (PWF) and
overstrain (OSF) failures and 25/28 heat-dissipation failures, but **0/9 tool-wear failures**
and neither of the 2 failures with no subtype flagged. By construction, TWF happens at a
*random* tool-wear time between 200 and 240 min, and RNF is random. The sensors only say
the machine is *in* the risk window, not when the failure will happen. That is a property of
the data, not a modeling defect.

**Feature engineering matters most.** Adding ΔT, power and strain raised OOF PR-AUC from
0.782 → 0.909 (GB), 0.760 → 0.895 (RF) and 0.483 → 0.688 (untuned DT). SHAP, impurity and
permutation importance all point to the same drivers: rotational speed, tool wear, ΔT,
power and strain. Raw air and process temperature matter mainly through their difference.

**Caveat.** AI4I is synthetic, and its failure labels are deterministic rules on these same
quantities. High scores show that the models recover the generating rules; they are not
evidence of comparable performance on a real milling machine. All importances are
statistical associations inside this dataset, not causal effects.""",
"aps": """## 10. Findings

**Preprocessing.** Mean, median, median + flags, dropping >70%-missing columns, adding
histogram totals and HistGB's native NaN handling all landed within ~0.01 OOF PR-AUC of each
other (HistGB 0.879–0.888; LR 0.762–0.766), inside the fold-to-fold spread. Median imputation
+ sentinel flag + missing count was kept: robust to the heavy skew, it keeps the documented
`ac_000` sentinel, and it gives every algorithm the same complete input (SMOTE requires one).
The engineered `n_missing` feature ranks 46th of 171 by mean |SHAP|; the sentinel flag ranks 69th.

**Model ranking (CV).** Tuned OOF PR-AUC: Gradient Boosting (HistGB) 0.899 > Random Forest
0.876 > Logistic Regression 0.821 ≈ KNN 0.817 > Decision Tree 0.731. Logistic Regression is
much stronger here than on AI4I. After log-scaling, many counters separate the classes almost
monotonically (34 features have a single-feature |AUC − 0.5| above 0.4; see notebook 03).

**Imbalance handling.** Only Gradient Boosting benefited: random oversampling raised its OOF
PR-AUC from 0.871 to 0.896 (SMOTE 0.889). For LR, KNN, DT and RF, every strategy lowered
PR-AUC, and for DT and RF it also lowered F1 at 0.5. As with AI4I, most of the useful
"imbalance handling" comes from moving the decision threshold.

**Cost-based thresholds are the single biggest lever.** At the default 0.5 threshold, the
tuned GB costs 27,460 on the test set (54 missed failures). At the OOF-selected threshold
of 0.0025, it costs **11,100** (13 missed, 460 false alarms). A missed failure costs 50× a
false alarm, so the cost-optimal model flags about 5% of test trucks (822 of 16,000) to catch 96.5% of APS failures.

**Test results (scored once).** Selected model, chosen on OOF cost: Gradient Boosting with
random oversampling. Test cost 11,100; recall 0.965; precision 0.440; PR-AUC 0.928; ROC-AUC 0.993.
Tuned Random Forest scored a lower test cost (10,440: 8 missed, 644 false alarms), but it
was not selected because its OOF cost was higher (43,610 vs 39,360). Picking it after seeing
test results would be test-set selection. For reference, the official IDA 2016 challenge
results on this test set were 9,920 / 10,900 / 11,480 for the top three entries.

**Explainability.** SHAP, permutation and impurity importance agree on a small core set:
`aa_000`, `ck_000`, `ci_000`, `ag_002` (a histogram bin), `bs_000`, `aq_000` and `bj_000`.
The features are anonymized, so these can't be mapped to physical components, and they
are statistical associations, not causes.

**Cost of the model zoo.** Classic `GradientBoostingClassifier` (untuned) took 187 s to fit
vs 16 s for the tuned HistGB, and scored a lower test PR-AUC (0.870 vs 0.928). Tuned KNN
(distance-weighted, Manhattan) needs ~3 s per 1,000 predictions. That is impractical at
fleet scale, though its accuracy is reasonable.

**Caveats.** The negative class is trucks with *other* failures, not healthy trucks. The test
set has a higher positive rate (2.34%) than training (1.67%). OOF costs cover 60,000 rows and
test costs 16,000, so the two are not directly comparable. Search budgets were reduced for
RF and KNN on this 2-core machine (see the technical report).""",
}
