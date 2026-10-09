"""Phase 2 — AirBreather statistical analysis.

Reads checkpoints/clean.parquet and produces:
  - checkpoints/trends.parquet       : per (indicator, neighborhood) slope + p-value
  - checkpoints/covid_effect.parquet : pre vs during COVID (2020-21) mean diff + p
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"


def _trend(group: pd.DataFrame) -> pd.Series:
    """OLS slope and two-sided p-value of Data_Value vs year."""
    g = group.dropna(subset=["Data_Value", "year"])
    if len(g) < 3:
        return pd.Series({"slope": np.nan, "p_value": np.nan, "r_squared": np.nan, "n": len(g)})
    r = stats.linregress(g["year"].astype(float), g["Data_Value"].astype(float))
    return pd.Series({
        "slope": r.slope, "p_value": r.pvalue, "r_squared": r.rvalue ** 2, "n": len(g),
    })


def compute_trends(clean: pd.DataFrame) -> pd.DataFrame:
    """Yearly trend per (indicator, neighborhood)."""
    out = (clean.groupby(["indicator_code", "Geo_Place_Name", "pickup_borough"],
                         group_keys=False, as_index=False)
                .apply(_trend))
    return out.reset_index(drop=True)


def compute_covid_effect(clean: pd.DataFrame) -> pd.DataFrame:
    """Pre-COVID (2015-2019) vs COVID (2020-2021) mean per indicator + Welch t-test."""
    rows = []
    for code, df in clean.groupby("indicator_code"):
        pre  = df[df["year"].between(2015, 2019)]["Data_Value"]
        covid = df[df["year"].between(2020, 2021)]["Data_Value"]
        if len(pre) < 5 or len(covid) < 5:
            continue
        t, p = stats.ttest_ind(pre, covid, equal_var=False)
        rows.append({
            "indicator_code": code, "pre_mean": float(pre.mean()),
            "covid_mean": float(covid.mean()),
            "pct_change": float((covid.mean() - pre.mean()) / pre.mean() * 100),
            "t_stat": float(t), "p_value": float(p),
            "significant": bool(p < 0.05),
        })
    return pd.DataFrame(rows)


def main() -> None:
    clean = pd.read_parquet(CHECKPOINTS / "clean.parquet")
    print(f"[phase2] loaded clean: {len(clean):,} rows")

    trends = compute_trends(clean)
    print(f"[phase2] trends computed for {len(trends):,} (indicator, neighborhood) pairs")
    sig_down = trends[(trends["slope"] < 0) & (trends["p_value"] < 0.05)]
    sig_up   = trends[(trends["slope"] > 0) & (trends["p_value"] < 0.05)]
    print(f"  significantly improving: {len(sig_down):,}")
    print(f"  significantly worsening: {len(sig_up):,}")
    trends.to_parquet(CHECKPOINTS / "trends.parquet", index=False)

    covid = compute_covid_effect(clean)
    print(f"[phase2] COVID effect per indicator:")
    print(covid.to_string(index=False))
    covid.to_parquet(CHECKPOINTS / "covid_effect.parquet", index=False)


if __name__ == "__main__":
    main()
