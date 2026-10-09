"""Phase 1 invariants (7)."""
from pathlib import Path
import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    path = CHECKPOINTS / "clean.parquet"
    if not path.exists():
        pytest.skip("clean.parquet missing — run `python src/phase1_clean.py` first")
    return pd.read_parquet(path)


def test_clean_nonempty(clean):
    assert len(clean) > 0


def test_required_columns(clean):
    required = {"Indicator_ID", "Name", "indicator_code", "Geo_Place_Name",
                "pickup_borough", "year", "Start_Date", "Data_Value"}
    assert required.issubset(set(clean.columns))


def test_no_negatives(clean):
    assert (clean["Data_Value"] >= 0).all()


def test_no_future_dates(clean):
    assert (clean["Start_Date"] <= pd.Timestamp.now()).all()


def test_indicator_codes_valid(clean):
    assert set(clean["indicator_code"].unique()).issubset({"PM2.5", "NO2", "O3", "OTHER"})


def test_boroughs_valid(clean):
    assert set(clean["pickup_borough"].unique()).issubset(
        {"Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"}
    )


def test_year_range(clean):
    assert clean["year"].min() >= 2015
    assert clean["year"].max() <= 2100
