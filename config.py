"""Project configuration: seeds, data/split settings, CV and search budgets, paths.

Two run modes share one pipeline:
  * sample (default for `python main.py`): bundled data in data/sample/, reduced CV folds and
    search budgets so the end-to-end run finishes in a few minutes on a CPU.
  * full (`python main.py --full`): official full datasets in data/full/ (fetched by
    data/download.py) with the budgets used for the reported results in results/full/.
Library code reads these module attributes at call time; `use_mode()` switches them.
Notebooks and the Streamlit app use the full-mode defaults set at the bottom of this file.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- randomness
RANDOM_STATE = 42          # used for every split, CV shuffle, sampler, estimator and search

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
FULL_DATA_DIR = DATA_DIR / "full"
REPORTS_DIR = ROOT / "reports"

AI4I_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
APS_URL = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"

# ----------------------------------------------------------------------------- AI4I 2020
AI4I_ID_COLS = ["UDI", "Product ID"]
AI4I_TARGET = "Machine failure"
# Failure-mode subtypes define the target, so they are never model inputs (leakage).
AI4I_LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
AI4I_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
AI4I_CATEGORICAL = ["Type"]
AI4I_TEST_SIZE = 0.20      # stratified hold-out (AI4I has no official split)

# ----------------------------------------------------------------------------- APS Scania
APS_TARGET = "class"
APS_HEADER_LINES = 20      # GPL license text preceding the CSV header
APS_NA_TOKEN = "na"
APS_COST_FP = 10           # official IDA 2016 challenge cost of a false alarm
APS_COST_FN = 500          # ... and of a missed APS failure
# ac_000 holds codes around 0x7F000000 (2130706432..2130706438) that act as sentinels.
APS_SENTINEL_COL = "ac_000"
APS_SENTINEL_MIN = 2_130_706_432

# Bundled APS sample: stratified random subsample of the official splits (see data/sample/README.md)
APS_SAMPLE_FRACTION = {"train": 0.05, "test": 0.10}

# ----------------------------------------------------------------------------- per-mode settings
# Search budgets: algorithm -> (n_iter, cv folds) for RandomizedSearchCV.
# Full-mode APS budgets for KNN and RF are reduced for compute (see reports/technical_report_draft.md).
_FULL_BUDGET = {
    "ai4i": {"logreg": (15, 5), "knn": (20, 5), "dtree": (30, 5), "rf": (20, 5), "gb": (25, 5)},
    "aps": {"logreg": (10, 5), "knn": (8, 3), "dtree": (20, 5), "rf": (8, 3), "gb": (25, 5)},
}
_SAMPLE_BUDGET = {ds: {a: (4, 3) for a in ["logreg", "knn", "dtree", "rf", "gb"]}
                  for ds in ["ai4i", "aps"]}

MODES = {
    "full": {
        "AI4I_RAW": FULL_DATA_DIR / "ai4i2020.csv",
        "APS_TRAIN_RAW": FULL_DATA_DIR / "aps_failure_training_set.csv",
        "APS_TEST_RAW": FULL_DATA_DIR / "aps_failure_test_set.csv",
        "RESULTS_DIR": ROOT / "results" / "full",
        "MODELS_DIR": ROOT / "models" / "full",
        "CV_FOLDS": 5,
        "SEARCH_BUDGET": _FULL_BUDGET,
    },
    "sample": {
        "AI4I_RAW": SAMPLE_DATA_DIR / "ai4i2020.csv",
        "APS_TRAIN_RAW": SAMPLE_DATA_DIR / "aps_failure_training_set_sample.csv",
        "APS_TEST_RAW": SAMPLE_DATA_DIR / "aps_failure_test_set_sample.csv",
        "RESULTS_DIR": ROOT / "results" / "sample",
        "MODELS_DIR": ROOT / "models" / "sample",
        "CV_FOLDS": 3,
        "SEARCH_BUDGET": _SAMPLE_BUDGET,
    },
}

MODE = None
AI4I_RAW = APS_TRAIN_RAW = APS_TEST_RAW = RESULTS_DIR = MODELS_DIR = FIGURES_DIR = None
CV_FOLDS = SEARCH_BUDGET = None


def use_mode(mode: str) -> None:
    """Switch data paths, output folders and budgets between 'sample' and 'full'."""
    g = globals()
    g.update(MODES[mode])
    g["MODE"] = mode
    g["FIGURES_DIR"] = g["RESULTS_DIR"] / "figures"


use_mode("full")
