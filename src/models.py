"""Algorithm registry, imbalance strategies and hyperparameter search spaces."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config as C
from .preprocessing import APS_DEFAULTS, ai4i_preprocessor, aps_preprocessor

SEED = C.RANDOM_STATE
ALGOS = ["logreg", "knn", "dtree", "rf", "gb"]
ALGO_NAMES = {"logreg": "Logistic Regression", "knn": "KNN", "dtree": "Decision Tree",
              "rf": "Random Forest", "gb": "Gradient Boosting"}
STRATEGIES = ["none", "class_weight", "ros", "smote"]
STRATEGY_NAMES = {"none": "None", "class_weight": "Class weight",
                  "ros": "Random oversampling", "smote": "SMOTE"}


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """GradientBoostingClassifier has no `class_weight`; this applies balanced
    sample weights computed from the labels passed to `fit` (i.e. the training fold)."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def supports(algo: str, strategy: str) -> bool:
    # KNN has no notion of class weights.
    return not (algo == "knn" and strategy == "class_weight")


def make_estimator(dataset: str, algo: str, strategy: str = "none", n_jobs: int = -1,
                   classic_gb: bool = False):
    cw = "balanced" if strategy == "class_weight" else None
    if algo == "logreg":
        return LogisticRegression(max_iter=3000, class_weight=cw, random_state=SEED)
    if algo == "knn":
        return KNeighborsClassifier(n_jobs=n_jobs)
    if algo == "dtree":
        return DecisionTreeClassifier(class_weight=cw, random_state=SEED)
    if algo == "rf":
        return RandomForestClassifier(n_estimators=100, class_weight=cw, n_jobs=n_jobs,
                                      random_state=SEED)
    if algo == "gb":
        if dataset == "ai4i" or classic_gb:
            cls = BalancedGradientBoostingClassifier if cw else GradientBoostingClassifier
            return cls(random_state=SEED)
        # APS: histogram-based GB (measured ~60x faster than the classic version here)
        # Early stopping holds out 10% of the *training* fold internally; max_iter is a cap.
        return HistGradientBoostingClassifier(class_weight=cw, max_iter=300, early_stopping=True,
                                              validation_fraction=0.1, n_iter_no_change=15,
                                              scoring="loss", random_state=SEED)
    raise ValueError(algo)


def make_sampler(dataset: str, strategy: str):
    if strategy == "ros":
        return RandomOverSampler(random_state=SEED)
    if strategy == "smote":
        if dataset == "ai4i":  # raw features incl. categorical 'Type'
            return SMOTENC(categorical_features=C.AI4I_CATEGORICAL, random_state=SEED)
        return SMOTE(random_state=SEED)
    return None


def build_pipeline(dataset: str, algo: str, strategy: str = "none", params: dict | None = None,
                   n_jobs: int = -1, classic_gb: bool = False) -> ImbPipeline:
    """Resampling lives inside the pipeline, so it only ever touches the data that
    the pipeline is fit on (a CV training fold or the full training set)."""
    sampler = make_sampler(dataset, strategy)
    clf = make_estimator(dataset, algo, strategy, n_jobs, classic_gb)
    if dataset == "ai4i":
        # Oversample raw rows first (SMOTENC handles 'Type'), then derive features,
        # so engineered features stay consistent with their physical formulas.
        steps = ([("sampler", sampler)] if sampler else []) + \
                list(ai4i_preprocessor().steps) + [("clf", clf)]
    else:
        # SMOTE needs complete numeric data -> sample after imputation/scaling.
        steps = list(aps_preprocessor(**APS_DEFAULTS).steps) + \
                ([("sampler", sampler)] if sampler else []) + [("clf", clf)]
    # imblearn forbids nested Pipelines as intermediate steps, so preprocessing
    # steps are flattened into one chain.
    pipe = ImbPipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe


def transform_for_model(pipe, X):
    """Apply the fitted preprocessing steps (skipping samplers) -> model-input DataFrame."""
    for name, step in pipe.steps[:-1]:
        if name == "sampler":
            continue
        X = step.transform(X)
    return X


# --------------------------------------------------------------- search spaces
def search_space(dataset: str, algo: str) -> dict:
    if algo == "logreg":
        return {"clf__C": loguniform(1e-3, 1e2)}
    if algo == "knn":
        return {"clf__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
                "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}
    if algo == "dtree":
        return {"clf__max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20, None],
                "clf__min_samples_leaf": [1, 2, 5, 10, 20, 50],
                "clf__criterion": ["gini", "entropy"]}
    if algo == "rf" and dataset == "aps":
        # Reduced for compute: max_features=0.3 with 300 trees took >10 min per single-core
        # fit on 40k x 172 (search aborted after 22 min); see technical report, Limitations.
        return {"clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 15, 25],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__max_features": ["sqrt", "log2"]}
    if algo == "rf":
        return {"clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 10, 20, 30],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__max_features": ["sqrt", "log2", 0.3]}
    if algo == "gb" and dataset == "ai4i":
        return {"clf__n_estimators": [100, 200, 300, 500],
                "clf__learning_rate": loguniform(0.01, 0.3),
                "clf__max_depth": [2, 3, 4, 5],
                "clf__subsample": [0.7, 0.85, 1.0],
                "clf__min_samples_leaf": [1, 5, 20]}
    if algo == "gb":
        return {"clf__max_iter": [200, 400, 800],
                "clf__learning_rate": loguniform(0.02, 0.3),
                "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [10, 20, 50],
                "clf__l2_regularization": loguniform(1e-3, 10)}
    raise ValueError(algo)
