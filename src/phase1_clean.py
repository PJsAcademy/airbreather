"""Phase 1 — AirBreather clean.

Reads NYC Air Quality (c3uy-2p5r) raw CSV if present, else synthesises a
deterministic 10-year × 30-neighborhood × 3-indicator panel. Writes
checkpoints/clean.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "nyc_air_quality.csv"
CHECKPOINTS = HERE / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

INDICATORS = [
    ("PM2.5", "Fine particles (PM 2.5)", "mcg/m3"),
    ("NO2",   "Nitrogen dioxide (NO2)", "ppb"),
    ("O3",    "Ozone (O3)",             "ppb"),
]
NEIGHBORHOODS = [
    ("Chelsea - Clinton", "Manhattan"), ("Washington Heights", "Manhattan"),
    ("Lower Manhattan", "Manhattan"), ("Upper East Side", "Manhattan"),
    ("Upper West Side", "Manhattan"), ("East Harlem", "Manhattan"),
    ("Central Harlem", "Manhattan"), ("Greenwich Village", "Manhattan"),
    ("Fordham - Bronx Pk", "Bronx"), ("South Bronx", "Bronx"),
    ("Hunts Point - Mott Haven", "Bronx"), ("Pelham - Throgs Neck", "Bronx"),
    ("Kingsbridge - Riverdale", "Bronx"), ("Northeast Bronx", "Bronx"),
    ("Williamsburg - Bushwick", "Brooklyn"), ("Downtown Brooklyn", "Brooklyn"),
    ("Flatbush", "Brooklyn"), ("Bedford Stuyvesant", "Brooklyn"),
    ("East New York", "Brooklyn"), ("Sunset Park", "Brooklyn"),
    ("Long Island City - Astoria", "Queens"), ("Flushing - Clearview", "Queens"),
    ("Jamaica", "Queens"), ("Ridgewood - Forest Hills", "Queens"),
    ("Rockaway", "Queens"), ("Southwest Queens", "Queens"),
    ("Port Richmond", "Staten Island"), ("Stapleton - St. George", "Staten Island"),
    ("South Beach - Tottenville", "Staten Island"), ("Willowbrook", "Staten Island"),
]
YEARS = list(range(2015, 2025))


def synthesise() -> pd.DataFrame:
    """Deterministic 10-yr × 30-neighborhood × 3-indicator panel with realistic
    base levels, a per-neighborhood effect, and a mild downtrend over time."""
    rng = np.random.default_rng(42)
    rows = []
    base_levels = {"PM2.5": 9.5, "NO2": 20.0, "O3": 32.0}
    for ind_code, ind_name, unit in INDICATORS:
        base = base_levels[ind_code]
        for i, (place, borough) in enumerate(NEIGHBORHOODS):
            place_effect = (i % 6 - 2.5) * 1.5
            for y in YEARS:
                trend = (y - 2015) * -0.18 if ind_code != "O3" else (y - 2015) * 0.08
                covid = (-1.5 if y in (2020, 2021) and ind_code != "O3" else 0)
                noise = rng.normal(0, 0.8)
                val = max(0.5, base + place_effect + trend + covid + noise)
                rows.append({
                    "Indicator_ID": {"PM2.5": 365, "NO2": 383, "O3": 386}[ind_code],
                    "Name": ind_name, "indicator_code": ind_code,
                    "Measure_Info": unit,
                    "Geo_Type_Name": "UHF42",
                    "Geo_Place_Name": place, "pickup_borough": borough,
                    "Time_Period": f"Annual Average {y}", "year": y,
                    "Start_Date": pd.Timestamp(f"{y}-01-01"),
                    "Data_Value": round(float(val), 2),
                })
    return pd.DataFrame(rows)


def load_bronze() -> pd.DataFrame:
    if RAW.exists():
        print(f"[bronze] reading {RAW}")
        return pd.read_csv(RAW)
    print(f"[bronze] {RAW} missing — synthesising 10yr x 30nbhd x 3ind panel (seed=42)")
    return synthesise()


def clean_to_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = bronze.copy()
    rejects = {
        "R1_missing_value": int(df["Data_Value"].isna().sum()),
        "R2_negative":      int((df["Data_Value"] < 0).sum()),
        "R3_future_dated":  int((pd.to_datetime(df["Start_Date"], errors="coerce")
                                 > pd.Timestamp.now()).sum()),
    }
    df = df.dropna(subset=["Data_Value"])
    df = df[df["Data_Value"] >= 0]
    df["Start_Date"] = pd.to_datetime(df["Start_Date"], errors="coerce")
    df = df[df["Start_Date"] <= pd.Timestamp.now()]
    if "year" not in df.columns:
        df["year"] = df["Start_Date"].dt.year
    if "indicator_code" not in df.columns:
        name_to_code = {ind[1]: ind[0] for ind in INDICATORS}
        df["indicator_code"] = df["Name"].map(name_to_code).fillna("OTHER")
    return df.reset_index(drop=True), rejects


def main() -> None:
    bronze = load_bronze()
    print(f"[bronze] {len(bronze):,} rows")
    silver, rejects = clean_to_silver(bronze)
    print(f"[silver] {len(silver):,} rows after cleaning")
    for rule, n in rejects.items():
        print(f"  rejected {n:>4}  {rule}")
    silver.to_parquet(CHECKPOINTS / "clean.parquet", index=False)
    print(f"[silver] wrote {CHECKPOINTS / 'clean.parquet'}")


if __name__ == "__main__":
    main()
