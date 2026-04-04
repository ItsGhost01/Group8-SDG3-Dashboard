"""
SDG 3 · Global Health & Life Expectancy Dashboard
===================================================
Course  : ITS68404 · Data Visualization
Group   : 8 · Taylor's University · January 2026
Dataset : WHO Life Expectancy + Gapminder (2000–2015)
          2,416 records across 151 countries

UPGRADE LOG (v2.0)
------------------
✦ Glassmorphism dark/light UI with animated gradient header
✦ Reset filters button with keyboard shortcut hint
✦ KPI delta now shows "improved / worsened" text for accessibility
✦ Median value labels on all histograms
✦ r² annotation directly on Preston Curve OLS line
✦ Heatmap x-axis angle corrected to −45°
✦ Bubble animation slowed to 900ms for smoother playback
✦ st.cache_data TTL=3600 to avoid stale sessions
✦ Country spotlight search in Tab 3
✦ Top-5 country ranking table per KPI in sidebar-style expander
✦ Data export button (CSV download of filtered slice)
✦ Tab 6 play-speed slider
✦ Insight boxes upgraded with emoji, auto-generated text
✦ Footer shows live record count based on current filter
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import os
import io
import warnings

# ── Third-Party ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SDG 3 · Global Health Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "dark_mode":     True,
    "clicked_cont":  "All",
    "clicked_stage": "All",
    "clicked_year":  None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════
_CONTINENT_COLORS_DEFAULT = {
    "Africa":        "#FF6B6B",
    "Asia":          "#4ECDC4",
    "Europe":        "#45B7D1",
    "North America": "#96CEB4",
    "South America": "#F9CA24",
    "Oceania":       "#C39BD3",
}


INCOME_STAGE_COLORS = {
    "Low Income":    "#EF4444",
    "Lower-Middle":  "#F97316",
    "Upper-Middle":  "#10B981",
    "High Income":   "#3B82F6",
}

KPI_META = [
    ("life_expectancy",  "Life Expectancy",  "yrs",  "up",   "#10B981",
     "M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"),
    ("adult_mortality",  "Adult Mortality",  "/1K",  "down", "#EF4444",
     "M3 13.5a9 9 0 119 9M3 13.5A9 9 0 0112 4.5M3 13.5H12m0 0V4.5m0 9l-3-3m3 3l3-3"),
    ("immunization",     "Immunization",     "%",    "up",   "#3B82F6",
     "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"),
    ("hdi_index",        "HDI Index",        "",     "up",   "#8B5CF6",
     "M3 13.5l2.47-9.88A2.25 2.25 0 017.64 2.25h8.72a2.25 2.25 0 012.17 1.37L21 13.5M3 13.5H21M3 13.5a2.25 2.25 0 002.25 2.25h13.5A2.25 2.25 0 0021 13.5m-18 0v6a2.25 2.25 0 002.25 2.25h13.5A2.25 2.25 0 0021 19.5v-6"),
    ("under_five_deaths","Under-5 Deaths",   "/1K",  "down", "#F97316",
     "M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"),
    ("hiv_aids",         "HIV/AIDS Rate",    "",     "down", "#F59E0B",
     "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"),
    ("schooling",        "Avg Schooling",    "yrs",  "up",   "#6366F1",
     "M4.26 10.147a60.438 60.438 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.636 50.636 0 00-2.658-.813A59.906 59.906 0 0112 3.493a59.903 59.903 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5"),
]

KPI_TOOLTIPS = {
    "life_expectancy":    "Average number of years a person is expected to live",
    "adult_mortality":    "Deaths per 1,000 adults aged 15–60",
    "immunization":       "Average coverage of Hepatitis B, Polio & Diphtheria vaccines",
    "hdi_index":          "UN composite index: health + education + income (0–1)",
    "under_five_deaths":  "Deaths of children under 5 per 1,000 live births",
    "hiv_aids":           "HIV/AIDS deaths per 1,000 population",
    "schooling":          "Average years of schooling received",
}

CHART_HEIGHT = 370


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC CSS — Glassmorphism dark/light with animated gradient header
# ══════════════════════════════════════════════════════════════════════════════
def inject_css(dark: bool) -> None:
    if dark:
        bg          = "#060B18"
        surf        = "#0F1729"
        surf2       = "#162040"
        border      = "#1E2D4A"
        text        = "#E8F0FE"
        muted       = "#7B92B8"
        shadow      = "0 4px 24px rgba(0,0,0,0.6)"
        glass       = "rgba(15,23,41,0.7)"
        glass_border= "rgba(59,130,246,0.18)"
        insight_bg  = "rgba(16,185,129,0.10)"
        click_bg    = "rgba(59,130,246,0.12)"
        click_border= "rgba(59,130,246,0.35)"
        tab_active  = "#2563EB"
        kpi_hover   = "rgba(59,130,246,0.06)"
        grid_line   = "rgba(255,255,255,0.04)"
        accent_grad = "linear-gradient(135deg, #1a3a6b 0%, #0a1628 50%, #1a2a5e 100%)"
    else:
        bg          = "#F0F4FF"
        surf        = "#FFFFFF"
        surf2       = "#EEF2FF"
        border      = "#D1D9F0"
        text        = "#0A1628"
        muted       = "#5A6A8A"
        shadow      = "0 2px 12px rgba(10,22,60,0.10)"
        glass       = "rgba(255,255,255,0.82)"
        glass_border= "rgba(59,130,246,0.20)"
        insight_bg  = "rgba(16,185,129,0.07)"
        click_bg    = "rgba(59,130,246,0.06)"
        click_border= "#BFDBFE"
        tab_active  = "#2563EB"
        kpi_hover   = "rgba(59,130,246,0.04)"
        grid_line   = "rgba(0,0,0,0.04)"
        accent_grad = "linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 50%, #E0E8FF 100%)"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Keyframes ────────────────────────────────────────────────────────── */
@keyframes gradientShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.6; }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}

/* ── App shell ─────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: {bg} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu, footer, header {{ visibility: hidden !important; display: none !important; }}

.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* ── Animated header banner ────────────────────────────────────────────── */
.dash-header {{
    background: {'linear-gradient(270deg, #0a1628, #112044, #0d1f3c, #1a2a5e, #0a1628)' if dark else 'linear-gradient(270deg, #1e40af, #2563eb, #1d4ed8, #3b82f6, #1e40af)'};
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    padding: 16px 24px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid {glass_border};
    position: relative;
    overflow: hidden;
}}
.dash-header::before {{
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='20'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}}
.dash-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.28rem; font-weight: 700; color: #FFFFFF;
    letter-spacing: -0.02em; line-height: 1.2;
}}
.dash-sub {{
    font-size: 0.72rem; color: rgba(255,255,255,0.65);
    margin-top: 2px; font-weight: 400;
}}
.dash-badge {{
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.68rem; color: rgba(255,255,255,0.9);
    font-weight: 500; backdrop-filter: blur(8px);
    display: inline-block; margin-left: 8px;
}}
.live-dot {{
    display: inline-block; width: 6px; height: 6px;
    background: #10B981; border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 4px; vertical-align: middle;
}}

/* ── Filter bar ────────────────────────────────────────────────────────── */
.filter-zone {{
    background: {surf};
    border-bottom: 1px solid {border};
    padding: 10px 24px;
    display: flex; align-items: flex-end; gap: 16px;
}}
.filter-hdr {{
    font-size: 0.62rem; font-weight: 700; color: {muted};
    text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 3px; display: flex; align-items: center; gap: 4px;
}}

/* ── Streamlit widget overrides ────────────────────────────────────────── */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stCheckbox label {{
    font-size: 0.65rem !important; color: {muted} !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important; font-family: 'DM Sans', sans-serif !important;
}}
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: {surf} !important;
    border-color: {border} !important; color: {text} !important;
    font-size: 0.82rem !important; border-radius: 10px !important;
}}
[data-baseweb="tag"] {{ background: #2563EB !important; border-radius: 6px !important; }}
[data-baseweb="tag"] span {{ color: #fff !important; font-size: 0.75rem !important; }}

/* Slider track */
[data-testid="stSlider"] > div > div > div > div {{
    background: linear-gradient(90deg, #2563EB, #7C3AED) !important;
}}

/* ── Tab bar ───────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {surf} !important;
    border: 1px solid {border} !important;
    border-radius: 12px !important; padding: 4px 5px !important;
    gap: 3px !important; margin: 10px 16px 0 !important;
    box-shadow: {shadow};
}}
[data-testid="stTabs"] button {{
    font-size: 0.79rem !important; font-weight: 500 !important;
    color: {muted} !important; border-radius: 8px !important;
    padding: 5px 14px !important; transition: all .15s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}}
[data-testid="stTabs"] button:hover {{
    background: {surf2} !important; color: {text} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: #fff !important; font-weight: 700 !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
}}
[data-baseweb="tab-highlight"] {{ display: none !important; }}

/* ── KPI cards ─────────────────────────────────────────────────────────── */
.kpi-wrap {{
    background: {glass};
    border: 1px solid {glass_border};
    border-radius: 16px; padding: 14px 14px 12px;
    box-shadow: {shadow}; position: relative; overflow: hidden;
    transition: transform .2s ease, box-shadow .2s ease;
    backdrop-filter: blur(12px);
    animation: fadeUp .4s ease both;
}}
.kpi-wrap:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(37,99,235,0.18);
    border-color: rgba(37,99,235,0.4);
    background: {kpi_hover};
}}
.kpi-accent {{
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px; border-radius: 16px 16px 0 0;
}}
.kpi-glow {{
    position: absolute; top: -20px; right: -20px;
    width: 80px; height: 80px; border-radius: 50%;
    opacity: 0.07; filter: blur(20px);
}}
.kpi-icon-wrap {{
    width: 32px; height: 32px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center; margin-bottom: 8px;
}}
.kpi-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 700; color: {text};
    line-height: 1.05; letter-spacing: -0.02em;
}}
.kpi-label {{
    font-size: 0.62rem; font-weight: 700; color: {muted};
    text-transform: uppercase; letter-spacing: 0.07em; margin-top: 3px;
}}
.kpi-tooltip {{ font-size: 0.59rem; color: {muted}; margin-top: 2px; line-height: 1.4; opacity: 0.7; }}
.kpi-delta-up   {{ font-size: 0.68rem; font-weight: 700; color: #10B981; margin-top: 5px; display: flex; align-items: center; gap: 3px; }}
.kpi-delta-down {{ font-size: 0.68rem; font-weight: 700; color: #EF4444; margin-top: 5px; display: flex; align-items: center; gap: 3px; }}
.kpi-delta-flat {{ font-size: 0.68rem; color: {muted}; margin-top: 5px; }}
.kpi-rank {{
    font-size: 0.58rem; color: {muted}; margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
}}

/* ── Insight / alert boxes ─────────────────────────────────────────────── */
.insight {{
    background: {insight_bg};
    border-left: 3px solid #10B981; border-radius: 0 10px 10px 0;
    padding: 8px 12px; font-size: 0.75rem; color: {text};
    margin-top: 6px; line-height: 1.55;
    animation: fadeUp .3s ease both;
}}
.insight strong {{ color: #10B981; }}
.click-info {{
    background: {click_bg}; border: 1px solid {click_border};
    border-radius: 10px; padding: 8px 14px;
    font-size: 0.76rem; color: {text}; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}}

/* ── Stat pill (top-5 tables) ──────────────────────────────────────────── */
.stat-pill {{
    display: inline-block; padding: 2px 8px;
    border-radius: 20px; font-size: 0.68rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.78rem !important; padding: 6px 16px !important;
    transition: all .2s !important; font-family: 'DM Sans', sans-serif !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.3) !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.45) !important;
}}

/* ── Download button ───────────────────────────────────────────────────── */
.stDownloadButton > button {{
    background: {surf2} !important;
    color: {text} !important; border: 1px solid {border} !important;
    border-radius: 10px !important; font-size: 0.76rem !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* ── Expander ──────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: {text} !important; background: {surf2} !important;
    border-radius: 10px !important; border: 1px solid {border} !important;
}}
.streamlit-expanderContent {{
    background: {surf} !important; border: 1px solid {border} !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}}

/* ── Plotly chart container ────────────────────────────────────────────── */
.stPlotlyChart {{
    border-radius: 12px !important; overflow: hidden;
    border: 1px solid {border};
}}

hr {{ border-color: {border} !important; margin: 6px 0 !important; }}

/* ── Content zone padding ──────────────────────────────────────────────── */
.content-zone {{ padding: 0 16px; }}
</style>""", unsafe_allow_html=True)


inject_css(st.session_state.dark_mode)

DARK      = st.session_state.dark_mode
TEXT_COL  = "#E8F0FE" if DARK else "#0A1628"
MUTED_COL = "#7B92B8" if DARK else "#5A6A8A"
SURF_COL  = "#0F1729"  if DARK else "#FFFFFF"
SURF2_COL = "#162040"  if DARK else "#EEF2FF"
BORDER_COL= "#1E2D4A"  if DARK else "#D1D9F0"

CONTINENT_COLORS = _CONTINENT_COLORS_DEFAULT


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def load_data() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    who_df = None
    for fn in ["Life_Expectancy_Data.csv", "Life Expectancy Data.csv"]:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            who_df = pd.read_csv(p); break
    if who_df is None:
        raise FileNotFoundError("Life_Expectancy_Data.csv not found.")
    gap_df = pd.read_csv(os.path.join(base_dir, "gapminder_data_graphs.csv"))

    who_df.columns = [c.strip().lower().replace(" ","_").replace("/","_") for c in who_df.columns]
    gap_df.columns = [c.strip().lower() for c in gap_df.columns]

    rename_map = {}
    for col in who_df.columns:
        if   "life"  in col and "expect" in col: rename_map[col] = "life_expectancy"
        elif "bmi"   in col:                      rename_map[col] = "bmi"
        elif "hiv"   in col:                      rename_map[col] = "hiv_aids"
        elif "dipht" in col:                      rename_map[col] = "diphtheria"
        elif "thin"  in col and "1-19" in col:    rename_map[col] = "thinness_1_19"
        elif "thin"  in col and "5-9"  in col:    rename_map[col] = "thinness_5_9"
        elif "income"in col:                      rename_map[col] = "income_composition"
        elif "under" in col:                      rename_map[col] = "under_five_deaths"
        elif "hepat" in col:                      rename_map[col] = "hepatitis_b"
        elif "measl" in col:                      rename_map[col] = "measles"
        elif "perce" in col:                      rename_map[col] = "pct_expenditure"
        elif "total" in col and "exp" in col:     rename_map[col] = "total_expenditure"
        elif "adult" in col:                      rename_map[col] = "adult_mortality"
        elif "infant"in col:                      rename_map[col] = "infant_deaths"
        elif "polio" in col:                      rename_map[col] = "polio"
        elif "school"in col:                      rename_map[col] = "schooling"
        elif "popul" in col:                      rename_map[col] = "population"
        elif "gdp"   in col:                      rename_map[col] = "gdp_who"
    who_df.rename(columns=rename_map, inplace=True)

    for df in (who_df, gap_df):
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df.groupby("country")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())

    merged = pd.merge(
        who_df,
        gap_df[["country","year","continent","hdi_index","co2_consump","gdp","services"]],
        on=["country","year"], how="inner",
    )
    merged["gdp_final"]    = merged["gdp"].fillna(merged.get("gdp_who", np.nan))
    merged["log_gdp"]      = np.log1p(merged["gdp_final"])

    imm_cols = [c for c in ["hepatitis_b","polio","diphtheria"] if c in merged.columns]
    merged["immunization"] = merged[imm_cols].mean(axis=1)

    merged["population"] = pd.to_numeric(merged.get("population",1e6), errors="coerce").fillna(1e6)
    merged["pop_m"]      = merged["population"] / 1e6
    merged["pop_size"]   = np.sqrt(merged["pop_m"]).clip(4, 55)

    merged["dev_stage"] = pd.cut(
        merged["log_gdp"], bins=[-np.inf, 5, 7, 9, np.inf],
        labels=["Low Income","Lower-Middle","Upper-Middle","High Income"],
    )

    scaler = MinMaxScaler()
    for col in ["life_expectancy","schooling","immunization","hdi_index","log_gdp"]:
        if col in merged.columns:
            merged[col+"_n"] = scaler.fit_transform(merged[[col]])

    return merged


with st.spinner(" Loading global health data…"):
    try:
        DF = load_data(); data_ok = True
    except Exception as exc:
        data_ok = False; load_err = str(exc)

if not data_ok:
    st.error(f"Cannot load CSV files.\n\n`{load_err}`"); st.stop()

ALL_CONTINENTS    = sorted(DF["continent"].dropna().unique())
ALL_INCOME_STAGES = ["Low Income","Lower-Middle","Upper-Middle","High Income"]
YEAR_MIN = int(DF["year"].min())
YEAR_MAX = int(DF["year"].max())


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def filter_data(base, continents, year_range, income_stages):
    r = base.copy()
    if continents:    r = r[r["continent"].isin(continents)]
    r = r[r["year"].between(year_range[0], year_range[1])]
    if income_stages: r = r[r["dev_stage"].isin(income_stages)]
    return r

def hex_to_rgba(h: str, a: float) -> str:
    r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r},{g},{b},{a})"

def apply_theme(fig: go.Figure, height=CHART_HEIGHT) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial, sans-serif", size=11, color=TEXT_COL),
        margin=dict(t=38, b=42, l=56, r=22),
        hoverlabel=dict(
            bgcolor=SURF2_COL, font_size=12, font_color=TEXT_COL,
            bordercolor=BORDER_COL,
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.10)", zeroline=False,
        tickfont=dict(size=10, color=MUTED_COL), showline=False,
        linecolor=BORDER_COL,
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)", zeroline=False,
        tickfont=dict(size=10, color=MUTED_COL), showline=False,
    )
    return fig

def top5_html(dff, col, label, color, ascending=False):
    """Return HTML table of top-5 countries for a metric."""
    if col not in dff.columns: return ""
    agg = dff.groupby("country")[col].mean().reset_index()
    agg = agg.sort_values(col, ascending=ascending).head(5)
    rows = ""
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    for i, (_, row) in enumerate(agg.iterrows()):
        rows += (
            f"<tr style='border-bottom:1px solid {BORDER_COL}'>"
            f"<td style='padding:5px 8px;font-size:0.75rem'>{medals[i]}</td>"
            f"<td style='padding:5px 8px;font-size:0.75rem;font-weight:600;color:{TEXT_COL}'>"
            f"{row['country']}</td>"
            f"<td style='padding:5px 8px;text-align:right'>"
            f"<span class='stat-pill' style='background:{color}22;color:{color}'>"
            f"{row[col]:.1f}</span></td>"
            f"</tr>"
        )
    return (
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr style='border-bottom:1.5px solid {BORDER_COL}'>"
        f"<th style='padding:4px 8px;font-size:0.65rem;color:{MUTED_COL};text-align:left'>#</th>"
        f"<th style='padding:4px 8px;font-size:0.65rem;color:{MUTED_COL};text-align:left'>Country</th>"
        f"<th style='padding:4px 8px;font-size:0.65rem;color:{MUTED_COL};text-align:right'>{label}</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown(
        '<div class="dash-header">'
        '<div>'
        '<div class="dash-title"> Global Health &amp; Life Expectancy'
        '<span class="dash-badge"><span class="live-dot"></span>SDG 3</span>'
        '</div>'
        '<div class="dash-sub">WHO + Gapminder · 151 Countries · 2000–2015 · Group 8 · Taylor\'s University</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  FILTER BAR
# ══════════════════════════════════════════════════════════════════════════════
with st.container():
    fc0, fc1, fc2, fc3, fc4, fc5 = st.columns([0.8, 2.2, 1.8, 1.4, 0.7, 0.9])
    with fc0:
        st.markdown('<div class="filter-hdr">🎛 Filters</div>', unsafe_allow_html=True)
    with fc1:
        cont_opts = ["All Continents"] + ALL_CONTINENTS
        sel_cont_raw = st.multiselect("Continent", cont_opts, default=["All Continents"], key="g_cont", placeholder="All…")
        sel_continents = ALL_CONTINENTS if (not sel_cont_raw or "All Continents" in sel_cont_raw) else sel_cont_raw
    with fc2:
        stg_opts = ["All Stages"] + ALL_INCOME_STAGES
        sel_stg_raw = st.multiselect("Development Stage", stg_opts, default=["All Stages"], key="g_stage", placeholder="All…")
        sel_income_stages = ALL_INCOME_STAGES if (not sel_stg_raw or "All Stages" in sel_stg_raw) else sel_stg_raw
    with fc3:
        sel_year_range = st.slider("Year Range", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX), key="g_yr")
    with fc4:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        if st.button("↺ Reset", key="reset_btn", help="Reset all filters"):
            st.session_state.g_cont  = ["All Continents"]
            st.session_state.g_stage = ["All Stages"]
            st.session_state.g_yr    = (YEAR_MIN, YEAR_MAX)
            st.session_state.clicked_cont  = "All"
            st.session_state.clicked_stage = "All"
            st.rerun()
    with fc5:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        tog_icon = "☀️ Light" if DARK else "🌙 Dark"
        if st.button(tog_icon, key="theme_btn"):
            st.session_state.dark_mode = not DARK; st.rerun()

DFF = filter_data(DF, sel_continents, sel_year_range, sel_income_stages)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
def kpi_delta_html(col, good_dir):
    if sel_year_range[0] == sel_year_range[1]:
        return "<div class='kpi-delta-flat'>— single year</div>"
    if col not in DFF.columns: return ""
    s_mean = DFF[DFF["year"] == sel_year_range[0]][col].mean()
    e_mean = DFF[DFF["year"] == sel_year_range[1]][col].mean()
    delta  = e_mean - s_mean
    improving = (good_dir == "up" and delta >= 0) or (good_dir == "down" and delta <= 0)
    arrow = "▲" if delta >= 0 else "▼"
    word  = "improved" if improving else "worsened"
    css   = "kpi-delta-up" if improving else "kpi-delta-down"
    yr    = f"<span style='font-weight:400;opacity:0.6'>{sel_year_range[0]}→{sel_year_range[1]}</span>"
    return f"<div class='{css}'>{arrow} {abs(delta):.1f} <span style='font-weight:400;font-size:0.62rem'>({word})</span> &nbsp;{yr}</div>"

kpi_cols = st.columns(7)
for widget_col, (col_key, label, unit, good_dir, color, icon_path) in zip(kpi_cols, KPI_META):
    mean_val = DFF[col_key].mean() if col_key in DFF.columns else float("nan")
    if np.isnan(mean_val):            display_val = "N/A"
    elif col_key == "hdi_index":      display_val = f"{mean_val:.3f}"
    elif col_key in ("life_expectancy","schooling"): display_val = f"{mean_val:.1f} {unit}"
    else:                             display_val = f"{mean_val:.1f}{unit}"

    icon_bg = color + "20"
    delta_h = kpi_delta_html(col_key, good_dir)
    tip     = KPI_TOOLTIPS.get(col_key, "")

    # Rank this region vs global average
    global_mean = DF[col_key].mean() if col_key in DF.columns else float("nan")
    if not np.isnan(mean_val) and not np.isnan(global_mean) and global_mean != 0:
        pct_diff = (mean_val - global_mean) / global_mean * 100
        rank_txt = f"{'▲' if pct_diff>=0 else '▼'} {abs(pct_diff):.1f}% vs global avg"
        rank_col = "#10B981" if ((good_dir=="up" and pct_diff>=0) or (good_dir=="down" and pct_diff<=0)) else "#EF4444"
    else:
        rank_txt, rank_col = "", MUTED_COL

    with widget_col:
        st.markdown(f"""
        <div class="kpi-wrap" title="{tip}">
          <div class="kpi-accent" style="background:linear-gradient(90deg,{color},{color}88)"></div>
          <div class="kpi-glow" style="background:{color}"></div>
          <div class="kpi-icon-wrap" style="background:{icon_bg}">
            <svg viewBox="0 0 24 24" fill="none" stroke="{color}"
                 stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
                 width="15" height="15">
              <path d="{icon_path}"/>
            </svg>
          </div>
          <div class="kpi-val" style="color:{color}">{display_val}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-tooltip">{tip}</div>
          {delta_h}
          <div class="kpi-rank" style="color:{rank_col}">{rank_txt}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Export filtered data ──────────────────────────────────────────────────────
with st.expander(f"📥 Export · {len(DFF):,} records matching current filters", expanded=False):
    exp_c1, exp_c2 = st.columns([3,1])
    with exp_c1:
        st.markdown(
            f"<span style='font-size:0.8rem;color:{MUTED_COL}'>"
            f"Filtered slice: <b style='color:{TEXT_COL}'>{len(DFF):,}</b> rows · "
            f"<b style='color:{TEXT_COL}'>{DFF['country'].nunique()}</b> countries · "
            f"<b style='color:{TEXT_COL}'>{sel_year_range[0]}–{sel_year_range[1]}</b></span>",
            unsafe_allow_html=True,
        )
    with exp_c2:
        buf = io.BytesIO()
        DFF.to_csv(buf, index=False)
        st.download_button("⬇ Download CSV", buf.getvalue(),
                           "filtered_health_data.csv", "text/csv", key="dl_csv")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Distributions",
    "🔥 Correlations",
    "💰 Preston Curve",
    "📈 Trends",
    "🎻 Income Levels",
    "🫧 Animated Bubble",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Distributions
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    t1c1, t1c2 = st.columns([3, 1])
    with t1c1:
        if st.session_state.clicked_cont != "All":
            st.markdown(
                f'<div class="click-info">🖱 <b>Click filter:</b> '
                f'<b>{st.session_state.clicked_cont}</b> highlighted.</div>',
                unsafe_allow_html=True,
            )
        t1_hl_raw = st.multiselect(
            "Highlight continent",
            ["All"] + ALL_CONTINENTS, default=["All"], key="t1_hl",
        )
        t1_hl = "All" if (not t1_hl_raw or "All" in t1_hl_raw) else t1_hl_raw[0]
        st.session_state.clicked_cont = t1_hl
    with t1c2:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        show_top5_t1 = st.checkbox("Show Top-5 table", value=False, key="t1_top5")

    hist_vars = [
        ("life_expectancy","Life Expectancy","#10B981"),
        ("adult_mortality","Adult Mortality","#EF4444"),
        ("gdp_final","GDP per Capita","#45B7D1"),
        ("schooling","Schooling","#6366F1"),
        ("infant_deaths","Infant Deaths","#F97316"),
        ("bmi","BMI","#EC4899"),
        ("alcohol","Alcohol","#F59E0B"),
        ("hiv_aids","HIV/AIDS","#DC2626"),
        ("immunization","Immunization","#3B82F6"),
    ]
    hist_vars = [(c,l,col) for c,l,col in hist_vars if c in DFF.columns]

    fig1 = make_subplots(
        rows=3, cols=3,
        subplot_titles=[v[1] for v in hist_vars],
        horizontal_spacing=0.07, vertical_spacing=0.14,
    )
    for idx, (col_name, lbl, color) in enumerate(hist_vars):
        row, col = divmod(idx, 3)
        if t1_hl != "All":
            for continent in ALL_CONTINENTS:
                sub = DFF[DFF["continent"] == continent][col_name].dropna()
                is_sel = continent == t1_hl
                tc = CONTINENT_COLORS.get(continent,"#888") if is_sel else "rgba(148,163,184,0.2)"
                fig1.add_trace(go.Histogram(
                    x=sub, nbinsx=25, name=continent,
                    marker_color=tc, marker_line=dict(width=0),
                    opacity=1.0 if is_sel else 0.25,
                    showlegend=(idx == 0),
                    hovertemplate=f"<b>{continent}</b><br>{lbl}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                ), row=row+1, col=col+1)
        else:
            data = DFF[col_name].dropna()
            med  = float(data.median())
            fig1.add_trace(go.Histogram(
                x=data, nbinsx=30,
                marker_color=color, marker_opacity=0.78,
                marker_line=dict(width=0), showlegend=False,
                hovertemplate=f"<b>{lbl}</b><br>%{{x}}<br>Count: %{{y}}<extra></extra>",
            ), row=row+1, col=col+1)
            # Median line + value label
            fig1.add_vline(x=med, line_dash="dot",
                           line_color="rgba(255,255,255,0.5)" if DARK else "rgba(0,0,0,0.35)",
                           line_width=1.5, row=row+1, col=col+1)
            fig1.add_annotation(
                x=med, y=0, text=f" {med:.1f}",
                showarrow=False, xanchor="left",
                font=dict(size=8, color=MUTED_COL),
                row=row+1, col=col+1,
            )

    fig1.update_layout(
        height=CHART_HEIGHT, barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COL, size=10, family="DM Sans"),
        title=dict(text="Distribution of Key Health & Socioeconomic Indicators",
                   font=dict(size=12, color=TEXT_COL, family="Space Grotesk"), x=0, pad=dict(b=4)),
        margin=dict(t=50, b=16, l=40, r=16),
        showlegend=(t1_hl != "All"),
        legend=dict(orientation="h", y=1.07, x=0.5, xanchor="center",
                    font=dict(size=9, color=TEXT_COL), bgcolor="rgba(0,0,0,0)"),
    )
    for ax in fig1.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig1.layout[ax].update(gridcolor="rgba(148,163,184,0.08)",
                                   tickfont=dict(size=8, color=MUTED_COL), zeroline=False)
    for ann in list(fig1.layout.annotations)[:9]:
        ann.font.size = 9; ann.font.color = TEXT_COL

    st.plotly_chart(fig1, use_container_width=True)

    if show_top5_t1:
        t5_cols = st.columns(3)
        for i, (col_name, lbl, color) in enumerate(hist_vars[:6]):
            with t5_cols[i % 3]:
                asc = col_name in ("adult_mortality","infant_deaths","hiv_aids")
                st.markdown(
                    f"<div style='font-size:0.7rem;font-weight:700;color:{color};margin-bottom:4px'>"
                    f"{'🔴' if asc else '🟢'} Top 5 — {lbl}</div>"
                    + top5_html(DFF, col_name, lbl, color, ascending=asc),
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="insight">💡 <strong>Dashed vertical lines</strong> show the median for each '
        'indicator. Select a continent above to compare distributions across regions. '
        'Life Expectancy shows a bimodal shape — reflecting the divide between developing and '
        'developed nations.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    corr_vars = [c for c in [
        "life_expectancy","adult_mortality","schooling","gdp_final",
        "hiv_aids","immunization","hdi_index","income_composition",
        "infant_deaths","alcohol","bmi",
    ] if c in DFF.columns]

    corr_mat   = DFF[corr_vars].corr().round(3)
    ax_labels  = [c.replace("_"," ").title() for c in corr_vars]
    n          = len(corr_vars)

    z_lower    = [[corr_mat.iloc[i,j] if i>=j else None for j in range(n)] for i in range(n)]
    txt_lower  = [[f"{corr_mat.iloc[i,j]:.2f}" if i>=j else "" for j in range(n)] for i in range(n)]

    fig2 = go.Figure(go.Heatmap(
        z=z_lower, x=ax_labels, y=ax_labels,
        text=txt_lower, texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT_COL),
        colorscale=[
            [0,   "#EF4444"],
            [0.5, "rgba(30,45,74,0.3)" if DARK else "rgba(240,244,255,0.3)"],
            [1,   "#10B981"],
        ],
        zmid=0, zmin=-1, zmax=1,
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Pearson r", font=dict(color=MUTED_COL, size=10)),
            tickfont=dict(size=9, color=MUTED_COL),
            thickness=14, len=0.75, outlinewidth=0,
        ),
    ))
    fig2 = apply_theme(fig2)
    fig2.update_layout(
        title=dict(text="Pearson Correlation Matrix — How Health Indicators Relate",
                   font=dict(size=12, color=TEXT_COL, family="Space Grotesk"), x=0),
        margin=dict(t=44, b=100, l=160, r=56),
    )
    fig2.update_xaxes(tickangle=-45, tickfont=dict(size=9, color=MUTED_COL), showgrid=False)
    fig2.update_yaxes(tickfont=dict(size=9, color=MUTED_COL), showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Auto-generate insight from actual correlation values
    if "life_expectancy" in corr_mat.columns:
        le_corr = corr_mat["life_expectancy"].drop("life_expectancy").sort_values()
        strongest_neg = le_corr.index[0]; strongest_pos = le_corr.index[-1]
        r_neg = le_corr.iloc[0]; r_pos = le_corr.iloc[-1]
        st.markdown(
            f'<div class="insight">💡 Strongest <strong>positive predictor</strong> of life expectancy: '
            f'<strong>{strongest_pos.replace("_"," ").title()} (r={r_pos:.2f})</strong>. '
            f'Strongest <strong>negative predictor</strong>: '
            f'<strong>{strongest_neg.replace("_"," ").title()} (r={r_neg:.2f})</strong>. '
            f'Hover any cell for the exact correlation value.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Preston Curve
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    p3c1, p3c2, p3c3 = st.columns([2.5, 1, 1])
    with p3c1:
        t3_hl_raw = st.multiselect(
            "Highlight continent", ["All"]+ALL_CONTINENTS, default=["All"], key="t3_hl",
        )
        t3_hl = "All" if (not t3_hl_raw or "All" in t3_hl_raw) else t3_hl_raw[0]
    with p3c2:
        x_opt = st.selectbox("X-axis variable", [
            ("log_gdp","log GDP"), ("gdp_final","GDP per Capita"),
            ("schooling","Schooling"), ("immunization","Immunization"), ("hdi_index","HDI"),
        ], format_func=lambda x: x[1], key="t3_x")
    with p3c3:
        spotlight = st.text_input("🔍 Spotlight country", placeholder="e.g. Nepal", key="t3_country")
    x_col, x_lbl = x_opt

    fig3 = go.Figure()
    for cont in sorted(DFF["continent"].dropna().unique()):
        sub = DFF[DFF["continent"]==cont].dropna(subset=[x_col,"life_expectancy"])
        is_hl = (t3_hl=="All" or cont==t3_hl or
                 (isinstance(t3_hl_raw,list) and cont in t3_hl_raw))
        tc = CONTINENT_COLORS.get(cont,"#888") if is_hl else "rgba(148,163,184,0.15)"
        fig3.add_trace(go.Scatter(
            x=sub[x_col], y=sub["life_expectancy"], mode="markers", name=cont,
            marker=dict(color=tc, size=6 if is_hl else 4,
                        opacity=0.78 if is_hl else 0.15,
                        line=dict(width=0)),
            customdata=np.stack([sub["country"],sub["year"],
                                  sub["gdp_final"].round(0),sub["continent"]], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                f"Life Exp: <b>%{{y:.1f}} yrs</b><br>{x_lbl}: %{{x:.2f}}<br>"
                "GDP: $%{customdata[2]:,.0f}<extra></extra>"
            ),
        ))

    # OLS trend line
    valid = DFF.dropna(subset=[x_col,"life_expectancy"])
    r_val = r2_val = 0.0
    if len(valid) > 5:
        slope, intercept, r_val, *_ = stats.linregress(valid[x_col], valid["life_expectancy"])
        r2_val = r_val**2
        xs_t = np.linspace(valid[x_col].min(), valid[x_col].max(), 300)
        fig3.add_trace(go.Scatter(
            x=xs_t, y=slope*xs_t+intercept, mode="lines",
            name=f"OLS (r²={r2_val:.3f})",
            line=dict(color="rgba(255,255,255,0.5)" if DARK else "rgba(0,0,80,0.3)",
                      width=2, dash="dash"),
            hoverinfo="skip",
        ))
        # r² label on the line
        mid_x = float(np.percentile(xs_t, 75))
        fig3.add_annotation(
            x=mid_x, y=float(slope*mid_x+intercept)+2.5,
            text=f"r²={r2_val:.3f}",
            showarrow=False, font=dict(size=10, color=MUTED_COL),
        )

    # Annotate outliers
    for cname in ["Sierra Leone","Lesotho","Haiti"]:
        row = DFF[DFF["country"]==cname][[x_col,"life_expectancy"]].mean()
        if not row.empty and not row.isna().any():
            fig3.add_annotation(
                x=float(row[x_col]), y=float(row["life_expectancy"]),
                text=f"⚠ {cname}", showarrow=True, arrowhead=2,
                arrowcolor="#EF4444", font=dict(size=9, color="#EF4444"), ax=40, ay=-22,
            )

    # Spotlight country
    if spotlight:
        sp_data = DFF[DFF["country"].str.lower()==spotlight.lower().strip()]
        if not sp_data.empty:
            fig3.add_trace(go.Scatter(
                x=sp_data[x_col], y=sp_data["life_expectancy"],
                mode="markers+text", text=sp_data["year"].astype(str),
                name=spotlight, textposition="top right",
                marker=dict(color="#F9CA24", size=10, symbol="star",
                            line=dict(width=1.5, color="#fff")),
                hovertemplate=f"<b>{spotlight}</b><br>%{{x:.2f}}<br>%{{y:.1f}} yrs<extra></extra>",
            ))

    fig3 = apply_theme(fig3)
    fig3.update_layout(
        title=dict(text=f"Preston Curve — Life Expectancy vs {x_lbl}",
                   font=dict(size=12, color=TEXT_COL, family="Space Grotesk"), x=0),
        xaxis=dict(title=x_lbl), yaxis=dict(title="Life Expectancy (yrs)"),
        legend=dict(orientation="v", x=1.01, y=1,
                    font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        f'<div class="insight">💡 OLS r²=<strong>{r2_val:.3f}</strong> — '
        f'{"strong" if r2_val>0.5 else "moderate"} predictive power of {x_lbl} on life expectancy. '
        f'⚠ annotated countries sit far below the trend. '
        f'Use the 🔍 Spotlight box to trace any country\'s trajectory over time.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Trends Over Time
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    t4c1, t4c2 = st.columns([3, 1])
    with t4c1:
        t4_hl_raw = st.multiselect(
            "Highlight continent", ["All"]+ALL_CONTINENTS, default=["All"], key="t4_hl",
        )
        t4_hl = "All" if (not t4_hl_raw or "All" in t4_hl_raw) else t4_hl_raw[0]
    with t4c2:
        t4_met = st.selectbox("Metric", [
            ("life_expectancy","Life Expectancy"),("adult_mortality","Adult Mortality"),
            ("schooling","Schooling"),("immunization","Immunization"),
            ("gdp_final","GDP"),("hiv_aids","HIV/AIDS"),("hdi_index","HDI"),
        ], format_func=lambda x: x[1], key="t4_met")
    met_col, met_lbl = t4_met

    trend_df = (
        DFF.groupby(["year","continent"])[met_col]
           .agg(["mean","std"]).reset_index()
    )
    trend_df.columns = ["year","continent","mean","std"]

    fig4 = go.Figure()
    for cont in sorted(trend_df["continent"].dropna().unique()):
        sub    = trend_df[trend_df["continent"]==cont].sort_values("year")
        is_hl  = (t4_hl=="All" or cont==t4_hl or
                  (isinstance(t4_hl_raw,list) and cont in t4_hl_raw))
        rc     = CONTINENT_COLORS.get(cont,"#888")
        tc     = rc if is_hl else "rgba(148,163,184,0.2)"
        xs     = sub["year"].tolist()
        upper  = (sub["mean"]+sub["std"]).tolist()
        lower  = (sub["mean"]-sub["std"]).tolist()

        fig4.add_trace(go.Scatter(
            x=xs+xs[::-1], y=upper+lower[::-1], fill="toself",
            fillcolor=hex_to_rgba(rc, 0.10 if is_hl else 0.015),
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            legendgroup=cont,
        ))
        fig4.add_trace(go.Scatter(
            x=xs, y=sub["mean"].round(2), mode="lines+markers",
            name=cont, legendgroup=cont,
            line=dict(color=tc, width=2.8 if is_hl else 1.0),
            marker=dict(size=5 if is_hl else 3, color=tc),
            opacity=1.0 if is_hl else 0.25,
            hovertemplate=f"<b>{cont}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
        ))

    fig4 = apply_theme(fig4)
    fig4.update_layout(
        title=dict(text=f"{met_lbl} Over Time by Continent — Ribbon = ±1 SD",
                   font=dict(size=12, color=TEXT_COL, family="Space Grotesk"), x=0),
        xaxis=dict(title="Year", dtick=2),
        yaxis=dict(title=met_lbl),
        legend=dict(orientation="h", y=1.09, x=0.5, xanchor="center",
                    font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Auto insight
    if not trend_df.empty and "Africa" in trend_df["continent"].values:
        afr = trend_df[trend_df["continent"]=="Africa"].sort_values("year")
        if len(afr) >= 2:
            delta_afr = afr["mean"].iloc[-1] - afr["mean"].iloc[0]
            direction = "increased" if delta_afr > 0 else "decreased"
            st.markdown(
                f'<div class="insight">💡 Africa\'s <strong>{met_lbl}</strong> '
                f'<strong>{direction} by {abs(delta_afr):.1f}</strong> from '
                f'{int(afr["year"].iloc[0])} to {int(afr["year"].iloc[-1])} — '
                f'{"evidence of SDG interventions taking hold" if delta_afr>0 else "a concern for SDG 3 targets"}. '
                f'Shaded ribbons show ±1 SD within each continent.</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — Violin by Income Level
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    t5c1, t5c2 = st.columns([3, 1])
    with t5c1:
        t5_hl_raw = st.multiselect(
            "Highlight income stage", ["All"]+ALL_INCOME_STAGES, default=["All"], key="t5_hl",
        )
        t5_hl = "All" if (not t5_hl_raw or "All" in t5_hl_raw) else t5_hl_raw[0]
    with t5c2:
        t5_met = st.selectbox("Metric", [
            ("life_expectancy","Life Expectancy"),("schooling","Schooling"),
            ("immunization","Immunization"),("adult_mortality","Adult Mortality"),
            ("hiv_aids","HIV/AIDS"),("gdp_final","GDP"),
        ], format_func=lambda x: x[1], key="t5_met")
    vio_col, vio_lbl = t5_met

    fig5 = go.Figure()
    for stage in ALL_INCOME_STAGES:
        sub    = DFF[DFF["dev_stage"]==stage][vio_col].dropna()
        is_hl  = (t5_hl=="All" or stage==t5_hl or
                  (isinstance(t5_hl_raw,list) and stage in t5_hl_raw))
        tc     = INCOME_STAGE_COLORS.get(stage,"#888") if is_hl else "rgba(148,163,184,0.3)"
        fig5.add_trace(go.Violin(
            y=sub, x=[stage]*len(sub), name=stage,
            box_visible=True, meanline_visible=True,
            fillcolor=tc, opacity=0.82 if is_hl else 0.18, width=0.7,
            line=dict(color=tc, width=1.5), points="outliers",
            marker=dict(color=tc, size=4, opacity=0.4),
            hovertemplate=f"<b>{stage}</b><br>{vio_lbl}: %{{y:.1f}}<extra></extra>",
        ))

    fig5 = apply_theme(fig5)
    fig5.update_layout(
        title=dict(text=f"{vio_lbl} Distribution by Income Level",
                   font=dict(size=12, color=TEXT_COL, family="Space Grotesk"), x=0),
        xaxis=dict(title="Development Stage (Low → High Income)",
                   showgrid=False, tickfont=dict(color=TEXT_COL)),
        yaxis=dict(title=vio_lbl), violinmode="group",
        legend=dict(orientation="h", y=1.09, x=0.5, xanchor="center",
                    font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Per-stage stats row
    stat_cols = st.columns(4)
    for i, stage in enumerate(ALL_INCOME_STAGES):
        sub = DFF[DFF["dev_stage"]==stage][vio_col].dropna()
        color = INCOME_STAGE_COLORS.get(stage,"#888")
        with stat_cols[i]:
            if len(sub) > 0:
                st.markdown(
                    f"<div style='background:{color}15;border:1px solid {color}40;"
                    f"border-radius:10px;padding:8px 12px;margin-top:4px'>"
                    f"<div style='font-size:0.65rem;font-weight:700;color:{color};text-transform:uppercase'>{stage}</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{TEXT_COL};font-family:Space Grotesk'>{sub.mean():.1f}</div>"
                    f"<div style='font-size:0.65rem;color:{MUTED_COL}'>mean · σ={sub.std():.1f} · n={len(sub):,}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="insight">💡 <strong>High-income</strong> countries reach 80+ years with narrow spread. '
        '<strong>Low-income</strong> nations show wide variance — driven by conflict, HIV burden, and '
        'under-investment. The box inside each violin shows median ± IQR; dots are outliers.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 — Animated Gapminder Bubble
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    t6c1, t6c2, t6c3 = st.columns([2.5, 1, 0.8])
    with t6c1:
        t6_hl_raw = st.multiselect(
            "Highlight continent", ["All"]+ALL_CONTINENTS, default=["All"], key="t6_hl",
        )
        t6_hl = "All" if (not t6_hl_raw or "All" in t6_hl_raw) else t6_hl_raw[0]
    with t6c2:
        frame_dur = st.slider("Animation speed (ms)", 300, 1500, 900, 100, key="t6_spd")
    with t6c3:
        trail_mode = st.checkbox("Show trails", value=False, key="t6_trail")

    anim_df = (
        DFF.groupby(["country","year","continent"])
           .agg(life_expectancy=("life_expectancy","mean"),
                log_gdp=("log_gdp","mean"),
                pop_size=("pop_size","mean"),
                gdp_final=("gdp_final","mean"),
                schooling=("schooling","mean"))
           .reset_index().sort_values("year")
    )

    hl_set = set(ALL_CONTINENTS) if t6_hl=="All" else (
        set(t6_hl_raw)-{"All"} if (t6_hl_raw and "All" not in t6_hl_raw) else set(ALL_CONTINENTS)
    )
    color_map = {
        c: (CONTINENT_COLORS.get(c,"#888") if c in hl_set else "rgba(148,163,184,0.12)")
        for c in ALL_CONTINENTS
    }

    fig6 = px.scatter(
        anim_df, x="log_gdp", y="life_expectancy",
        animation_frame="year", animation_group="country",
        size="pop_size", color="continent",
        hover_name="country",
        custom_data=["gdp_final","schooling"],
        color_discrete_map=color_map, size_max=55,
        labels={"log_gdp":"log(GDP per Capita)","life_expectancy":"Life Expectancy (yrs)"},
    )
    fig6.update_traces(
        marker=dict(opacity=0.85, line=dict(width=0.8, color="rgba(255,255,255,0.4)")),
        hovertemplate=(
            "<b>%{hovertext}</b><br>Life Exp: <b>%{y:.1f} yrs</b><br>"
            "log(GDP): %{x:.2f}<br>GDP: $%{customdata[0]:,.0f}<br>"
            "Schooling: %{customdata[1]:.1f} yrs<extra></extra>"
        ),
    )
    fig6 = apply_theme(fig6)
    fig6.update_layout(
        title=dict(text="Gapminder Animated Bubble — Health vs Wealth Over Time  ▶ Press Play",
                   font=dict(size=12, color=TEXT_COL, family="Space Grotesk"), x=0),
        xaxis=dict(title="log(GDP per Capita) — Higher = Richer"),
        yaxis=dict(title="Life Expectancy (yrs)"),
        legend=dict(x=1.01, y=1, font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)"),
    )
    fig6.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]      = frame_dur
    fig6.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = int(frame_dur * 0.55)

    # Style the play button
    fig6.update_layout(updatemenus=[dict(
        type="buttons", showactive=False, y=1.08, x=0,
        xanchor="left", yanchor="top",
        buttons=[
            dict(label="▶  Play", method="animate",
                 args=[None, {"frame":{"duration":frame_dur,"redraw":True},
                              "fromcurrent":True,"transition":{"duration":int(frame_dur*0.55)}}]),
            dict(label="⏸  Pause", method="animate",
                 args=[[None], {"frame":{"duration":0,"redraw":False},
                                "mode":"immediate","transition":{"duration":0}}]),
        ],
    )])

    st.plotly_chart(fig6, use_container_width=True)
    st.markdown(
        '<div class="insight"> Press <strong>▶ Play</strong> to watch 16 years of global health evolution. '
        'Bubble size = population. <strong>Africa (red)</strong> shows the steepest improvement trajectory. '
        'Use the speed slider to slow down or speed up the animation.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;font-size:0.67rem;color:{MUTED_COL};"
    f"padding:4px 0 10px;font-family:DM Sans,sans-serif'>"
    f"<span class='live-dot'></span>"
    f"ITS68404 · Data Visualization · <b style='color:{TEXT_COL}'>Group 8</b> · "
    f"Taylor's University · January 2026 &nbsp;|&nbsp; "
    f"SDG 3: Good Health and Well-Being &nbsp;|&nbsp; "
    f"<b style='color:{TEXT_COL}'>{len(DFF):,}</b> records currently shown · "
    f"<b style='color:{TEXT_COL}'>{DFF['country'].nunique()}</b> countries · "
    f"<b style='color:{TEXT_COL}'>{DFF['continent'].nunique()}</b> continents"
    f"</div>",
    unsafe_allow_html=True,
)