"""Phase 3 — AirBreather explanatory story outputs.

Reads trends.parquet + covid_effect.parquet and writes:
  - checkpoints/leaderboards.parquet : top-5 improved + top-5 worsened per indicator
  - checkpoints/story.json           : one-paragraph plain-English summary + the 3
                                       most interview-quotable numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"


def leaderboards(trends: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-k improving and worsening neighborhoods per indicator."""
    sig = trends[trends["p_value"] < 0.05].copy()
    rows = []
    for code, group in sig.groupby("indicator_code"):
        best = group.nsmallest(k, "slope").assign(direction="improving")
        worst = group.nlargest(k, "slope").assign(direction="worsening")
        rows.append(pd.concat([best, worst], ignore_index=True))
    if not rows:
        return pd.DataFrame(columns=["indicator_code", "Geo_Place_Name", "slope",
                                     "p_value", "direction"])
    return pd.concat(rows, ignore_index=True)


def one_paragraph_summary(trends: pd.DataFrame, covid: pd.DataFrame) -> dict:
    sig = trends[trends["p_value"] < 0.05]
    improving = sig[sig["slope"] < 0]
    worsening = sig[sig["slope"] > 0]

    headline_covid = ""
    if len(covid):
        c = covid.iloc[covid["pct_change"].abs().argmax()]
        headline_covid = (
            f"{c['indicator_code']} fell {abs(c['pct_change']):.1f}% during COVID "
            f"(2020-21) vs the 2015-19 baseline (p={c['p_value']:.3g})."
        )

    paragraph = (
        f"Across the 10-year panel, {len(improving):,} (indicator, neighborhood) pairs "
        f"show statistically significant improvement (p<0.05) and {len(worsening):,} "
        f"show significant worsening. "
        + headline_covid
    )

    quotables = []
    if len(improving):
        best = improving.nsmallest(1, "slope").iloc[0]
        quotables.append(
            f"{best['Geo_Place_Name']} saw {best['indicator_code']} drop "
            f"{abs(best['slope']):.2f} units/year (p={best['p_value']:.3g})"
        )
    if len(worsening):
        worst = worsening.nlargest(1, "slope").iloc[0]
        quotables.append(
            f"{worst['Geo_Place_Name']} saw {worst['indicator_code']} rise "
            f"{abs(worst['slope']):.2f} units/year (p={worst['p_value']:.3g})"
        )
    for _, c in covid.iterrows():
        quotables.append(
            f"{c['indicator_code']} {'fell' if c['pct_change']<0 else 'rose'} "
            f"{abs(c['pct_change']):.1f}% during COVID "
            f"({'significant' if c['significant'] else 'not significant'}, p={c['p_value']:.3g})"
        )

    return {"paragraph": paragraph, "quotables": quotables[:5]}


def main() -> None:
    trends = pd.read_parquet(CHECKPOINTS / "trends.parquet")
    covid  = pd.read_parquet(CHECKPOINTS / "covid_effect.parquet")
    print(f"[phase3] loaded trends ({len(trends):,}) + covid ({len(covid):,})")

    lb = leaderboards(trends)
    lb.to_parquet(CHECKPOINTS / "leaderboards.parquet", index=False)
    print(f"[phase3] leaderboards: {len(lb):,} rows (top-5 up + top-5 down per indicator)")

    story = one_paragraph_summary(trends, covid)
    (CHECKPOINTS / "story.json").write_text(json.dumps(story, indent=2))
    print(f"[phase3] wrote story.json")
    print(f"  paragraph: {story['paragraph']}")
    for q in story["quotables"]:
        print(f"  quotable:  {q}")


if __name__ == "__main__":
    main()
