# AirBreather

NYC air-quality trends across 30 UHF42 neighborhoods × 10 years × 3 pollutants.
Data Analysis capstone of [Bits to Builds](https://bitstobuilds.com), built on
the [NYC Open Data Air Quality](https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r)
dataset (public domain).

**Live demo:** (fill in after `publish.sh`)
**Source:** <https://github.com/PJsAcademy/airbreather>

---

## What it does

| Phase | Deliverable |
|-------|-------------|
| 1. Clean   | parse + validate NYC Air Quality raw CSV (or synthesise if missing); document rejects per rule |
| 2. Analyze | per-neighborhood OLS trend (slope + p-value); per-pollutant COVID pre-vs-during Welch's t-test |
| 3. Story   | top-5 improving + top-5 worsening leaderboards per pollutant; plain-English headline + quotable numbers |

Streamlit UI with Dashboard, Explorer (pick-your-own trend viewer with
regression overlay), Leaderboards, Methodology (5 decisions + "what a staff
engineer would flag"), About.

## Run locally

```bash
pip install -r requirements.txt
python src/phase1_clean.py    # ~2s on synth; writes clean.parquet
python src/phase2_analyze.py  # ~2s; writes trends.parquet + covid_effect.parquet
python src/phase3_story.py    # ~1s; writes leaderboards.parquet + story.json
pytest tests/ -q              # 18 invariants should pass
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud (free)

```bash
REPO=airbreather ../portfolio/publish.sh
```

Then at <https://share.streamlit.io>: Create app → pick this repo → main file
`streamlit_app.py` → Deploy.

## Honest limits

- Synthesises a deterministic 10yr × 30nbhd × 3-pollutant panel when the raw
  Socrata CSV is absent. Findings on the deployed version are illustrative, not
  factual claims about NYC.
- OLS linear trend ignores seasonality. The honest upgrade is monthly data +
  STL decomposition.
- No multiple-comparisons correction on the trend p-values (~5 of 90 would pass
  by chance at p<0.05). A BH-corrected toggle is in the roadmap.
- COVID effect is a naive pre-vs-during Welch's t-test with no DiD; it conflates
  the ongoing cleanup trend with the pandemic shock.

## Credits

Dataset: NYC DOHMH Environment & Health, Air Quality indicators. Built from the
[Bits to Builds](https://bitstobuilds.com) curriculum.
