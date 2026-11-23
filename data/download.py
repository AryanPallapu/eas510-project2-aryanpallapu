"""Download the full datasets from the official UCI repository into data/full/ and verify them.

    python data/download.py

Needed only for the full experiment (`python main.py --full`); the default sample run
uses the bundled files in data/sample/ and needs no network access.
Sources (retrieved 2026-11-16):
  AI4I 2020  https://archive.ics.uci.edu/dataset/601  (CC BY 4.0)
  APS Scania https://archive.ics.uci.edu/dataset/421  (GPL v3)
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def fetch(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)


def verify() -> None:
    from src.data_loader import load_ai4i, load_aps
    C.use_mode("full")
    a = load_ai4i()
    assert a.shape == (10_000, 14), a.shape
    assert int(a[C.AI4I_TARGET].sum()) == 339
    tr, te = load_aps()
    assert tr.shape == (60_000, 171) and te.shape == (16_000, 171)
    assert int(tr[C.APS_TARGET].sum()) == 1_000 and int(te[C.APS_TARGET].sum()) == 375
    print("Verified: AI4I 10000x14 (339 failures); APS train 60000 (1000 pos), test 16000 (375 pos)")


if __name__ == "__main__":
    full = C.MODES["full"]
    if not full["AI4I_RAW"].exists():
        fetch(C.AI4I_URL, C.FULL_DATA_DIR)
    if not (full["APS_TRAIN_RAW"].exists() and full["APS_TEST_RAW"].exists()):
        fetch(C.APS_URL, C.FULL_DATA_DIR)
    verify()
