"""Build the bundled sample in data/sample/ from the full data in data/full/.

    python data/download.py && python scripts/make_sample.py

* AI4I 2020: copied whole (10,000 rows, ~0.5 MB, CC BY 4.0), so the sample *is* the full dataset.
* APS Scania: stratified random subsample of each official split
  (train 5% -> 3,000 rows, test 10% -> 1,600 rows; seed 42), written with the original
  GPL v3 license header so the file format and loader are identical to the full data.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import config as C  # noqa: E402

if __name__ == "__main__":
    full, sample = C.MODES["full"], C.MODES["sample"]
    C.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(full["AI4I_RAW"], sample["AI4I_RAW"])
    for split, src_key, dst_key in [("train", "APS_TRAIN_RAW", "APS_TRAIN_RAW"),
                                    ("test", "APS_TEST_RAW", "APS_TEST_RAW")]:
        lines = full[src_key].read_text().splitlines(keepends=True)
        header = "".join(lines[:C.APS_HEADER_LINES])
        df = pd.read_csv(full[src_key], skiprows=C.APS_HEADER_LINES, dtype=str, keep_default_na=False)
        part, _ = train_test_split(df, train_size=C.APS_SAMPLE_FRACTION[split], stratify=df["class"],
                                   random_state=C.RANDOM_STATE)
        part = part.sort_index()
        with open(sample[dst_key], "w", newline="") as f:
            f.write(header)
            part.to_csv(f, index=False)
        print(f"APS {split}: {len(part)} rows, {int((part['class'] == 'pos').sum())} positives -> {sample[dst_key]}")
