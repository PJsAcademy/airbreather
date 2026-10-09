"""Phase 2 invariants (6)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture(scope="module")
def trends() -> pd.DataFrame:
    path = CHECKPOINTS / "trends.parquet"
    if not path.exists():
        pytest.skip("trends.parquet missing — run `python src/phase2_analyze.py` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def covid() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "covid_effect.parquet")


def test_trends_nonempty(trends):
    assert len(trends) > 0


def test_trends_schema(trends):
    assert {"indicator_code", "Geo_Place_Name", "slope", "p_value", "r_squared", "n"}.issubset(trends.columns)


def test_pvalues_in_range(trends):
    valid = trends["p_value"].dropna()
    assert ((valid >= 0) & (valid <= 1)).all()


def test_r_squared_in_range(trends):
    valid = trends["r_squared"].dropna()
    assert ((valid >= 0) & (valid <= 1)).all()


def test_covid_has_all_indicators(covid):
    assert set(covid["indicator_code"]) >= {"PM2.5", "NO2", "O3"}


def test_covid_pct_change_reasonable(covid):
    assert (covid["pct_change"].abs() < 100).all(), \
        "COVID effect >100% is almost certainly a bug (division by near-zero mean)"
