"""AirBreather — NYC air-quality analytics for the Bits to Builds DA capstone.

Tabs:
  1. Dashboard  — hero KPIs, borough trends chart, pollutant heatmap, COVID effect
  2. Explorer   — pick an indicator + neighborhood, see its 10-year trajectory and
                  linear-regression trend line with p-value
  3. Leaderboards — top-5 most-improved + top-5 most-worsened neighborhoods per
                    indicator (stat-significance filtered)
  4. Methodology — 5 decisions + "what a staff engineer would flag that I left in"
  5. About
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
CHECKPOINTS = HERE / "checkpoints"

BRAND_YELLOW = "#FFC72C"
BRAND_RED = "#E63946"
BRAND_GREEN = "#50C878"
BRAND_BLUE = "#4A90E2"
BRAND_INK = "#0E1117"
BRAND_INK2 = "#171B22"
BRAND_INK3 = "#232833"
BRAND_FG = "#E7E9EC"
BRAND_FG_DIM = "#9AA3B2"

sys.path.insert(0, str(HERE / "src"))

st.set_page_config(
    page_title="AirBreather — NYC Air Quality",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================= bootstrap

@st.cache_resource(show_spinner="First-time setup: building clean + trends + story (~15s)...")
def bootstrap_checkpoints() -> None:
    required = [
        CHECKPOINTS / "clean.parquet",
        CHECKPOINTS / "trends.parquet",
        CHECKPOINTS / "covid_effect.parquet",
        CHECKPOINTS / "leaderboards.parquet",
        CHECKPOINTS / "story.json",
    ]
    if all(p.exists() for p in required):
        try:
            pd.read_parquet(CHECKPOINTS / "clean.parquet")
            return
        except Exception:
            pass
    CHECKPOINTS.mkdir(exist_ok=True)
    import phase1_clean, phase2_analyze, phase3_story
    phase1_clean.main()
    phase2_analyze.main()
    phase3_story.main()


bootstrap_checkpoints()


# ======================================================= loaders

@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "clean.parquet")


@st.cache_data
def load_trends() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "trends.parquet")


@st.cache_data
def load_covid() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "covid_effect.parquet")


@st.cache_data
def load_leaderboards() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "leaderboards.parquet")


@st.cache_data
def load_story() -> dict:
    return json.loads((CHECKPOINTS / "story.json").read_text())


# ======================================================= Global CSS (shared with other B2B capstones)

st.markdown(
    f"""
    <style>
    #MainMenu, footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}

    .kpi-card {{
        background: linear-gradient(135deg, {BRAND_INK2} 0%, {BRAND_INK3} 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 4px solid var(--accent, {BRAND_YELLOW});
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        height: 100%;
    }}
    .kpi-card .kpi-label {{color: {BRAND_FG_DIM}; font-size: 12px; font-weight: 500;
                          text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;}}
    .kpi-card .kpi-value {{color: {BRAND_FG}; font-size: clamp(16px, 1.9vw, 30px);
                          font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums;
                          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
    .kpi-card .kpi-delta {{display: inline-block; margin-top: 6px; color: {BRAND_FG_DIM};
                          font-size: 12px;}}
    .kpi-icon {{float: right; font-size: 20px; opacity: 0.35; margin-left: 4px;}}
    @media (max-width: 1100px) {{ .kpi-icon {{display: none;}} }}

    .insight {{background: {BRAND_INK2}; border: 1px solid rgba(255,199,44,0.15);
              border-radius: 10px; padding: 10px 14px; font-size: 13px;
              color: {BRAND_FG}; line-height: 1.45;}}
    .insight .insight-tag {{color: {BRAND_YELLOW}; font-weight: 600; font-size: 11px;
                           text-transform: uppercase; letter-spacing: 0.07em; margin-right: 6px;}}

    .hero {{background: radial-gradient(circle at top left, rgba(80,200,120,0.14) 0%, rgba(14,17,23,0) 55%);
           padding: 10px 0 16px; margin-bottom: 10px;}}
    .hero h1 {{font-size: 36px !important; font-weight: 800 !important; margin-bottom: 4px !important;}}
    .hero .tagline {{color: {BRAND_FG_DIM}; font-size: 15px;}}

    section[data-testid="stSidebar"] {{background: {BRAND_INK};}}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, delta: str = "", icon: str = "", accent: str = BRAND_YELLOW):
    st.markdown(
        f"""<div class="kpi-card" style="--accent:{accent};">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ======================================================= load data

clean = load_clean()
trends = load_trends()
covid = load_covid()
leaders = load_leaderboards()
story = load_story()

indicators_available = sorted(clean["indicator_code"].unique())
boroughs_available = sorted(clean["pickup_borough"].unique())


# ======================================================= Sidebar

with st.sidebar:
    st.markdown("### 🫁 AirBreather")
    st.caption("Bits to Builds · DA capstone")
    st.divider()
    st.markdown("##### Filters")
    sel_indicators = st.multiselect("Pollutants", indicators_available, default=indicators_available)
    sel_boroughs = st.multiselect("Boroughs", boroughs_available, default=boroughs_available)
    year_min, year_max = int(clean["year"].min()), int(clean["year"].max())
    sel_years = st.slider("Years", year_min, year_max, (year_min, year_max))

    st.divider()
    st.markdown("##### Data")
    st.caption(f"**{len(clean):,}** observations")
    st.caption(f"**{year_min}-{year_max}**, 30 UHF42 neighborhoods × 3 pollutants")
    st.download_button(
        "⬇ Download clean panel (CSV)",
        clean.to_csv(index=False).encode(), "nyc_air_clean.csv", "text/csv",
        use_container_width=True,
    )
    st.download_button(
        "⬇ Download trends", trends.to_csv(index=False).encode(),
        "trends.csv", "text/csv", use_container_width=True,
    )

    st.divider()
    st.markdown("##### Links")
    st.markdown("[💻 Source on GitHub](https://github.com/PJsAcademy/airbreather)")
    st.markdown("[📚 Bits to Builds](https://bitstobuilds.com)")
    st.markdown("[🫁 NYC Air Quality dataset](https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r)")


# Apply filters
mask = (clean["indicator_code"].isin(sel_indicators)
        & clean["pickup_borough"].isin(sel_boroughs)
        & clean["year"].between(*sel_years))
clean_f = clean[mask] if mask.any() else clean


# ======================================================= Hero + KPIs

st.markdown(
    """<div class="hero">
      <h1>🫁 AirBreather</h1>
      <div class="tagline">NYC air-quality trends across 30 neighborhoods × 10 years × 3 pollutants · DA capstone of
      <a href="https://bitstobuilds.com" style="color:#FFC72C;">Bits to Builds</a></div>
    </div>""",
    unsafe_allow_html=True,
)

n_obs = len(clean_f)
n_nbhds = clean_f["Geo_Place_Name"].nunique()
avg_pm = float(clean_f[clean_f["indicator_code"] == "PM2.5"]["Data_Value"].mean()) if "PM2.5" in sel_indicators else float("nan")
sig_improving = int(((trends["indicator_code"].isin(sel_indicators))
                     & (trends["slope"] < 0) & (trends["p_value"] < 0.05)).sum())
sig_worsening = int(((trends["indicator_code"].isin(sel_indicators))
                     & (trends["slope"] > 0) & (trends["p_value"] < 0.05)).sum())

k1, k2, k3, k4, k5 = st.columns(5)
with k1: kpi_card("Observations", f"{n_obs:,}", f"{sel_years[0]}-{sel_years[1]}", "📊")
with k2: kpi_card("Neighborhoods", f"{n_nbhds}", "UHF42 health districts", "📍", BRAND_YELLOW)
with k3: kpi_card("Avg PM2.5", f"{avg_pm:.2f}" if not np.isnan(avg_pm) else "—",
                   "mcg/m³ (WHO limit: 10)", "🌫️", BRAND_BLUE)
with k4: kpi_card("Improving (p<0.05)", f"{sig_improving:,}", "neighborhood × pollutant", "📉", BRAND_GREEN)
with k5: kpi_card("Worsening (p<0.05)", f"{sig_worsening:,}", "neighborhood × pollutant", "📈", BRAND_RED)


# Insights band from phase 3 story
st.markdown("")
cols = st.columns(min(4, max(1, len(story["quotables"]))))
tags = ["Headline", "Best", "Worst", "COVID"]
for col, tag, body in zip(cols, tags, story["quotables"]):
    with col:
        st.markdown(
            f'<div class="insight"><span class="insight-tag">{tag}</span>{body}</div>',
            unsafe_allow_html=True,
        )

st.markdown("")
st.divider()

tab_dash, tab_explore, tab_leaders, tab_method, tab_about = st.tabs(
    ["📊 Dashboard", "🔎 Explorer", "🏆 Leaderboards", "🛠 Methodology", "ℹ️ About"]
)


# ------- Dashboard tab -------
with tab_dash:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Pollutant levels by year — median per borough")
        by_bor = (clean_f.groupby(["year", "pickup_borough", "indicator_code"])["Data_Value"]
                        .median().reset_index())
        line = (
            alt.Chart(by_bor).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("Data_Value:Q", title="Median value"),
                color=alt.Color("pickup_borough:N", title=None,
                                scale=alt.Scale(scheme="category10")),
                strokeDash="indicator_code:N",
                tooltip=["year", "pickup_borough", "indicator_code",
                         alt.Tooltip("Data_Value:Q", format=".2f")],
            ).properties(height=320)
        )
        st.altair_chart(line, use_container_width=True)
        st.caption("Lines by borough, dash pattern by pollutant. Lower is cleaner air.")

    with c2:
        st.subheader("COVID effect per pollutant")
        if len(covid):
            covid_filtered = covid[covid["indicator_code"].isin(sel_indicators)]
            bar = (
                alt.Chart(covid_filtered).mark_bar(cornerRadius=4).encode(
                    y=alt.Y("indicator_code:N", title=None),
                    x=alt.X("pct_change:Q", title="% change: 2020-21 vs 2015-19"),
                    color=alt.condition(
                        "datum.pct_change < 0",
                        alt.value(BRAND_GREEN), alt.value(BRAND_RED),
                    ),
                    tooltip=["indicator_code",
                             alt.Tooltip("pct_change:Q", format="+.2f"),
                             alt.Tooltip("p_value:Q", format=".3g"),
                             "significant"],
                ).properties(height=240)
            )
            st.altair_chart(bar, use_container_width=True)
            sig_count = int(covid_filtered["significant"].sum())
            st.caption(f"**{sig_count} of {len(covid_filtered)} pollutants** show a statistically "
                       "significant COVID-era shift (Welch's t-test, p<0.05).")
        else:
            st.info("No COVID-era data in current filter.")

    st.divider()
    st.subheader("Neighborhood × year heatmap — median per pollutant")
    sel_single = st.selectbox("Pick a pollutant for the heatmap",
                              [i for i in sel_indicators] or indicators_available)
    heat_data = (clean_f[clean_f["indicator_code"] == sel_single]
                 .groupby(["Geo_Place_Name", "year"])["Data_Value"].median().reset_index())
    nbhd_sort = (heat_data.groupby("Geo_Place_Name")["Data_Value"].mean()
                          .sort_values(ascending=False).index.tolist())
    heat = (
        alt.Chart(heat_data).mark_rect(stroke=BRAND_INK, strokeWidth=1).encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Geo_Place_Name:N", sort=nbhd_sort, title=None),
            color=alt.Color("Data_Value:Q", scale=alt.Scale(scheme="yelloworangered"),
                            title=sel_single, legend=alt.Legend(orient="right")),
            tooltip=["Geo_Place_Name", "year",
                     alt.Tooltip("Data_Value:Q", format=".2f")],
        ).properties(height=min(460, max(240, len(nbhd_sort) * 14)))
    )
    st.altair_chart(heat, use_container_width=True)


# ------- Explorer tab -------
with tab_explore:
    st.subheader("Trajectory + OLS trend for one (pollutant, neighborhood)")
    c1, c2 = st.columns(2)
    pick_ind = c1.selectbox("Pollutant", indicators_available, index=0)
    nbhds_this = sorted(clean[clean["indicator_code"] == pick_ind]["Geo_Place_Name"].unique())
    pick_nbhd = c2.selectbox("Neighborhood", nbhds_this,
                             index=min(0, len(nbhds_this)-1) if nbhds_this else 0)
    sub = clean[(clean["indicator_code"] == pick_ind) & (clean["Geo_Place_Name"] == pick_nbhd)]
    trend_row = trends[(trends["indicator_code"] == pick_ind)
                       & (trends["Geo_Place_Name"] == pick_nbhd)]

    pts = alt.Chart(sub).mark_point(size=120, filled=True, color=BRAND_YELLOW).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("Data_Value:Q", title=pick_ind),
        tooltip=["year", alt.Tooltip("Data_Value:Q", format=".2f")],
    )
    reg_line = alt.Chart(sub).transform_regression("year", "Data_Value").mark_line(
        color=BRAND_RED, strokeWidth=2, strokeDash=[4, 4],
    ).encode(x="year:Q", y="Data_Value:Q")
    st.altair_chart((pts + reg_line).properties(height=380), use_container_width=True)

    if len(trend_row):
        r = trend_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Slope", f"{r['slope']:+.3f}/yr",
                           "negative = improving",
                           "📉" if r['slope'] < 0 else "📈",
                           BRAND_GREEN if r['slope'] < 0 else BRAND_RED)
        with c2: kpi_card("p-value", f"{r['p_value']:.3g}",
                           "significant" if r['p_value'] < 0.05 else "not significant",
                           "🧪",
                           BRAND_GREEN if r['p_value'] < 0.05 else BRAND_FG_DIM)
        with c3: kpi_card("R²", f"{r['r_squared']:.3f}",
                           f"n={int(r['n'])}", "📏", BRAND_BLUE)


# ------- Leaderboards tab -------
with tab_leaders:
    st.subheader("Most-improved and most-worsened neighborhoods per pollutant")
    st.caption("Filtered to statistically significant trends only (p<0.05). "
               "Slope sign tells direction: negative = cleaner air.")
    for ind in indicators_available:
        this = leaders[leaders["indicator_code"] == ind]
        if this.empty:
            continue
        st.markdown(f"#### {ind}")
        c1, c2 = st.columns(2)
        improving = this[this["direction"] == "improving"].copy()
        worsening = this[this["direction"] == "worsening"].copy()
        with c1:
            st.markdown("**📉 Improving**")
            if len(improving):
                show = improving[["Geo_Place_Name", "slope", "p_value"]].rename(
                    columns={"Geo_Place_Name": "Neighborhood",
                             "slope": "Slope (/yr)", "p_value": "p"})
                show = show.sort_values("Slope (/yr)")
                st.dataframe(show, use_container_width=True, hide_index=True,
                             column_config={
                                 "Slope (/yr)": st.column_config.NumberColumn(format="%+.3f"),
                                 "p": st.column_config.NumberColumn(format="%.3g"),
                             })
            else:
                st.caption("No significantly-improving neighborhoods.")
        with c2:
            st.markdown("**📈 Worsening**")
            if len(worsening):
                show = worsening[["Geo_Place_Name", "slope", "p_value"]].rename(
                    columns={"Geo_Place_Name": "Neighborhood",
                             "slope": "Slope (/yr)", "p_value": "p"})
                show = show.sort_values("Slope (/yr)", ascending=False)
                st.dataframe(show, use_container_width=True, hide_index=True,
                             column_config={
                                 "Slope (/yr)": st.column_config.NumberColumn(format="%+.3f"),
                                 "p": st.column_config.NumberColumn(format="%.3g"),
                             })
            else:
                st.caption("No significantly-worsening neighborhoods.")


# ------- Methodology tab -------
with tab_method:
    st.markdown(
        """
        ## Methodology — decisions, tradeoffs, honest shortcuts

        ---
        ### Decision 1 — Why linear regression for trend, not something fancier?

        **Chose:** OLS slope + two-sided p-value per (indicator, neighborhood).

        **Why:** 10 annual observations per group is too few for a non-linear fit without
        overfitting. A linear trend tests the right question ("is this getting better or
        worse year on year") and gives a p-value out of the box. Fancier (Mann-Kendall,
        STL) buys ~0 interpretability over OLS at this n.

        **What I'd change with 10× the time:** monthly observations instead of annual
        (TLC pattern), then STL seasonal decomposition to separate seasonal cycles from
        the trend.

        ---
        ### Decision 2 — Why Welch's t-test for COVID effect, not paired or Wilcoxon?

        **Chose:** Welch's two-sample t-test between pre-COVID (2015-19) and COVID (2020-21).

        **Why:** The two periods are not paired (different years, mostly different
        observations). Welch's drops the equal-variance assumption that Student's t makes
        and is robust under moderate sample sizes. Non-parametric (Mann-Whitney) would be
        the honest upgrade; here the sample is large enough that normality is fine.

        **What I'd change with 10× the time:** add a difference-in-differences design with
        each neighborhood as its own control to isolate the local COVID effect from the
        broader national trend.

        ---
        ### Decision 3 — Why synthesise the dataset instead of hitting the live API?

        **Chose:** 10-year × 30-neighborhood × 3-indicator deterministic sample (seed=42).

        **Why (honest):** Streamlit Cloud free tier has no network access to the Socrata
        API at cold-start. Downloading the full raw dataset at build time is also a hard
        dependency. The synthetic sample preserves the schema and shows the full pipeline;
        pointing phase1.py at the real CSV is a one-line swap.

        **What I'd change with 10× the time:** nightly cron that pulls the live Socrata
        API, snapshots to parquet, and commits via GitHub Actions. The deployed app then
        reads the latest snapshot.

        ---
        ### Decision 4 — Why UHF42 neighborhoods, not community districts or zip codes?

        **Chose:** UHF42 (United Hospital Fund 42-neighborhood scheme) — NYC DOHMH's
        health-reporting default.

        **Why:** These are the geographies the actual air-quality monitoring stations map
        to. Community districts (CDs) are smaller and most don't have a monitor; aggregating
        to CDs would require interpolation that pretends to precision we don't have. ZIPs
        are a US Postal Service invention, not a public-health one — they stretch across
        very different exposures.

        **What I'd change with 10× the time:** add a CD-level overlay with explicit
        interpolation + uncertainty bands, so the user sees that the reported CD value is
        a smoothed estimate, not a measurement.

        ---
        ### Decision 5 — Why p<0.05 for the significance filter?

        **Chose:** Standard 5% threshold, no multiple-comparisons correction.

        **Why (honest shortcut):** With 30 neighborhoods × 3 pollutants = 90 tests per
        trend scan, we're testing enough hypotheses that some pass by chance (~4-5 false
        positives at p=0.05 under the null). Correcting (Benjamini-Hochberg) would be the
        right move. I didn't, because the alternative — showing 0 significant results after
        correction — is less informative than showing the uncorrected count with this
        caveat visible.

        **What I'd change with 10× the time:** add a toggle for "show BH-corrected results"
        and explain the drop in count honestly.

        ---

        ## What a staff engineer would flag that I left in

        - **No multiple-comparisons correction** (above).
        - **No stations metadata.** Each neighborhood may have 1 or 10 monitoring stations;
          averaging equally weights them. The honest fix is to use station-day as the unit
          of observation and group-by station density.
        - **COVID effect ignores pre-trend.** A naive pre-vs-during comparison attributes
          the ongoing cleanup trend to COVID. The DiD design in Decision 2's note fixes this.
        - **Synthetic data is still synthetic.** The relationships are plausible but
          data-set-specific findings should not be quoted as factual claims about NYC.
        """
    )


# ------- About tab -------
with tab_about:
    st.markdown(
        f"""
        ## About

        **AirBreather** is the Data Analysis capstone of the
        [Bits to Builds](https://bitstobuilds.com) course. It's a 3-phase pipeline:

        | Phase | Deliverable |
        |-------|-------------|
        | 1. Clean   | parse + validate NYC Air Quality raw CSV, document rejects |
        | 2. Analyze | per-neighborhood linear-trend OLS + per-pollutant COVID t-test |
        | 3. Story   | top-5 improving + worsening leaderboards + plain-English summary |

        ## Data
        [NYC Open Data: Air Quality (c3uy-2p5r)](https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r),
        public domain. When the raw CSV is absent, the pipeline falls back to a
        deterministic 10yr × 30-neighborhood × 3-indicator synthetic sample.

        ## Honest limits
        - **Synthetic sample** when raw CSV not present — findings on this deploy are
          data-set-illustrative only.
        - **No multiple-comparisons correction** on the trend p-values.
        - **UHF42 neighborhoods** are coarse; some have 1 monitoring station, some have 10.
        - **COVID effect** is a naive pre-vs-during comparison, no DiD.

        ## Headline from Phase 3

        > {story['paragraph']}

        ## Source
        [github.com/PJsAcademy/airbreather](https://github.com/PJsAcademy/airbreather)
        """
    )
