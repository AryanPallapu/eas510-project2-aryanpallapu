# Bundled sample data

Used by the default run (`python main.py`). It needs no download or network access.

| File | Rows | Positives | Selection | License |
|---|---|---|---|---|
| `ai4i2020.csv` | 10,000 | 339 | **Complete AI4I 2020 dataset**, unchanged (small and redistributable) | CC BY 4.0 |
| `aps_failure_training_set_sample.csv` | 3,000 | 50 | Stratified 5% random subsample of the official APS training set (60,000 rows) | GPL v3 |
| `aps_failure_test_set_sample.csv` | 1,600 | 38 | Stratified 10% random subsample of the official APS test set (16,000 rows) | GPL v3 |

**How it was made:** `python data/download.py && python scripts/make_sample.py`, using
`sklearn.model_selection.train_test_split(stratify=class, random_state=42)` on each official
split separately. The train and test samples therefore never mix. The APS files keep the original
20-line GPL v3 license header and the `"na"` missing-value token, so the same loader reads the
sample and the full files.

**Differences from the full data:**
- **AI4I:** none. The sample is the full dataset, though `main.py` runs it with smaller
  CV/search budgets (see `config.py`).
- **APS:** about 1/20 of the training data and 1/10 of the test data. Class ratios match
  the official splits (1.67% / 2.4% positive), and missingness is similar (8.2% of cells).
  With only 50 training positives and 38 test positives, sample metrics are noisy. A single
  test failure moves recall by 2.6 points and cost by 500. Sample results check that the
  pipeline works; the reported findings come from the full data (`results/full/`).

**Sources:**
- AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository, https://doi.org/10.24432/C5HS5C
- APS Failure at Scania Trucks, Scania CV AB (2016), UCI ML Repository #421,
  https://archive.ics.uci.edu/dataset/421

Both were retrieved on 2026-11-16.
