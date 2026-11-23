import pytest

import config as C
from src.data_loader import load_ai4i, load_aps


@pytest.fixture(autouse=True)
def sample_mode():
    C.use_mode("sample")
    yield
    C.use_mode("full")


def test_ai4i_schema():
    df = load_ai4i()
    assert df.columns[0] == "UDI"  # BOM stripped
    assert df.shape == (10_000, 14)
    assert df.isna().sum().sum() == 0
    assert set(C.AI4I_LEAKAGE_COLS) <= set(df.columns)


def test_aps_sample_keeps_official_split_structure():
    tr, te = load_aps()
    assert tr.shape == (3_000, 171) and te.shape == (1_600, 171)
    assert list(tr.columns) == list(te.columns)
    assert tr[C.APS_TARGET].sum() == 50 and te[C.APS_TARGET].sum() == 38
    assert tr.drop(columns=C.APS_TARGET).dtypes.map(lambda t: t.kind in "fi").all()
