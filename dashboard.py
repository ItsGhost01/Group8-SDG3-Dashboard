"""
SDG 3 · Global Health & Life Expectancy Dashboard
===================================================
Course  : ITS68404 · Data Visualization
Group   : 8 · Taylor's University · January 2026
Dataset : WHO Life Expectancy + Gapminder (2000–2015)
          2,416 records across 151 countries

INTERACTIVITY OVERVIEW
----------------------
This dashboard is fully interactive. Every control drives the entire page:

  Global Filters (top bar)
  ├── Continent multi-select  → filters ALL KPI cards and ALL tab charts
  ├── Development Stage       → same cross-filtering effect
  └── Year Range slider       → same cross-filtering effect

  Per-Tab Controls (inside each tab)
  ├── "Highlight continent / income stage" dropdowns visually dim non-selected
  │   groups without changing the underlying data filter, allowing comparison.
  └── Metric / X-axis selectors change what variable is plotted.

  KPI Cards
  └── Values update instantly whenever any global filter changes, showing the
      mean (and start→end delta) for the filtered subset.

CROSS-FILTERING MECHANISM
--------------------------
Streamlit reruns the entire script top-to-bottom on every widget interaction.
All charts and KPIs share the single filtered DataFrame `DFF`, which is
rebuilt from the global filter widgets at the top of the layout.  There is no
separate callback wiring needed: changing any filter automatically propagates
to every visual because they all read from `DFF`.

The "clicked_cont / clicked_stage" session-state keys allow a chart-selection
(e.g. choosing a continent inside Tab 1) to be remembered across reruns and
pre-populate the highlight dropdown in that tab — simulating click-to-filter.
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import warnings

# ── Third-Party ───────────────────────────────────────────────────────────────
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
#  PAGE CONFIGURATION
#  Must be the first Streamlit call in the script.
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SDG 3 · Global Health Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
#  THEME STATE
#  Persisted in session_state so the dark/light toggle survives reruns.
# ══════════════════════════════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# ══════════════════════════════════════════════════════════════════════════════
#  COLOR PALETTES
#  These palettes are shared between KPI cards and chart traces so that the
#  same continent / income stage always renders in the same colour everywhere.
# ══════════════════════════════════════════════════════════════════════════════
CONTINENT_COLORS: dict[str, str] = {
    "Africa":        "#FF6B6B",
    "Asia":          "#4ECDC4",
    "Europe":        "#45B7D1",
    "North America": "#96CEB4",
    "South America": "#F9CA24",
    "Oceania":       "#C39BD3",
}

INCOME_STAGE_COLORS: dict[str, str] = {
    "Low Income":    "#EF4444",
    "Lower-Middle":  "#F97316",
    "Upper-Middle":  "#10B981",
    "High Income":   "#3B82F6",
}


# ══════════════════════════════════════════════════════════════════════════════
#  KPI METADATA
#  Each tuple defines one KPI card:
#    (dataframe_column, display_label, unit_suffix, good_direction, hex_color,
#     heroicon_svg_path)
#
#  Colors deliberately match the chart color for that metric so users can
#  visually connect a KPI card to the corresponding chart trace:
#    Life Expectancy  → teal   #10B981  (positive health outcome)
#    Adult Mortality  → red    #EF4444  (negative health outcome)
#    Immunization     → blue   #3B82F6  (matches Preston curve scatter)
#    HDI Index        → purple #8B5CF6  (composite development index)
#    Under-5 Deaths   → orange #F97316  (child health risk)
#    HIV/AIDS         → amber  #F59E0B  (epidemic warning)
#    Avg Schooling    → indigo #6366F1  (education proxy)
# ══════════════════════════════════════════════════════════════════════════════
KPI_META: list[tuple] = [
    (
        "life_expectancy", "Life Expectancy", "yrs", "up", "#10B981",
        "M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364"
        "L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z",
    ),
    (
        "adult_mortality", "Adult Mortality", "/1K", "down", "#EF4444",
        "M3 13.5a9 9 0 119 9M3 13.5A9 9 0 0112 4.5M3 13.5H12m0 0V4.5m0 9l-3-3m3 3l3-3",
    ),
    (
        "immunization", "Immunization", "%", "up", "#3B82F6",
        "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
    ),
    (
        "hdi_index", "HDI Index", "", "up", "#8B5CF6",
        "M3 13.5l2.47-9.88A2.25 2.25 0 017.64 2.25h8.72a2.25 2.25 0 012.17 1.37L21 13.5"
        "M3 13.5H21M3 13.5a2.25 2.25 0 002.25 2.25h13.5A2.25 2.25 0 0021 13.5m-18 0v6a2.25"
        " 2.25 0 002.25 2.25h13.5A2.25 2.25 0 0021 19.5v-6",
    ),
    (
        "under_five_deaths", "Under-5 Deaths", "/1K", "down", "#F97316",
        "M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z"
        "M4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z",
    ),
    (
        "hiv_aids", "HIV/AIDS Rate", "", "down", "#F59E0B",
        "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874"
        " 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z",
    ),
    (
        "schooling", "Avg Schooling", "yrs", "up", "#6366F1",
        "M4.26 10.147a60.438 60.438 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627"
        " 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.636 50.636 0 00-2.658-.813"
        "A59.906 59.906 0 0112 3.493a59.903 59.903 0 0110.399 5.84c-.896.248-1.783.52-2.658.814"
        "m-15.482 0A50.717 50.717 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0"
        " 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981"
        " 5.981 0 006.75 15.75v-1.5",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC CSS
#  Rebuilt on every render so dark/light colours update immediately.
#
#  Layout targets 1920×1080 with no vertical scroll:
#    - Header + filters : ~58 px
#    - KPI row          : ~95 px
#    - Tab bar          : ~38 px
#    - Per-tab controls : ~44 px
#    - Chart area       : ~410–420 px  (see CHART_HEIGHT constant below)
#    - Insight bar      : ~34 px
#    - Footer           : ~26 px
#    - Paddings / gaps  : ~50 px
#    ─────────────────────────────
#    Total              : ~755 px   (leaves headroom for browser chrome)
# ══════════════════════════════════════════════════════════════════════════════
def inject_css(dark: bool) -> None:
    """Inject theme-aware CSS into the Streamlit app."""
    if dark:
        bg     = "#0F172A"
        surf   = "#1E293B"
        border = "#334155"
        text   = "#F1F5F9"
        muted  = "#94A3B8"
        shadow       = "0 1px 6px rgba(0,0,0,0.5)"
        tab_hover_bg = "#0F172A"
        toggle_bg    = "#1E293B"
        toggle_border = "#334155"
        insight_bg   = "rgba(16,185,129,0.12)"
        click_bg     = "rgba(59,130,246,0.15)"
        click_border = "rgba(59,130,246,0.4)"
    else:
        bg     = "#F1F5F9"
        surf   = "#FFFFFF"
        border = "#E2E8F0"
        text   = "#0F172A"
        muted  = "#64748B"
        shadow       = "0 1px 4px rgba(0,0,0,0.08)"
        tab_hover_bg = "#F1F5F9"
        toggle_bg    = "#FFFFFF"
        toggle_border = "#E2E8F0"
        insight_bg   = "rgba(16,185,129,0.08)"
        click_bg     = "rgba(59,130,246,0.08)"
        click_border = "#BFDBFE"

    st.markdown(f"""
<style>
/* ── App shell ──────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {{ background:{bg} !important; }}
[data-testid="stMain"]             {{ background:{bg} !important; }}
section[data-testid="stSidebar"]   {{ display:none !important; }}
[data-testid="collapsedControl"]   {{ display:none !important; }}
#MainMenu, footer, header          {{ visibility:hidden !important; }}

/* Tight padding keeps everything on one screen at 1080 px height */
.block-container {{
    padding:0.55rem 1.2rem 0.3rem !important;
    max-width:100% !important;
}}

/* ── Title ──────────────────────────────────────────────────────────────── */
.dash-title {{ font-size:1.15rem; font-weight:700; color:{text}; }}
.dash-sub   {{ font-size:0.74rem; color:{muted}; margin-top:1px; }}

/* ── Dark / Light toggle ─────────────────────────────────────────────────── */
.theme-btn {{
    background:{toggle_bg}; border:1px solid {toggle_border};
    border-radius:8px; padding:5px 12px; cursor:pointer;
    font-size:0.78rem; font-weight:600; color:{text};
    display:inline-flex; align-items:center; gap:6px;
    box-shadow:{shadow};
}}

/* ── Filter bar label ────────────────────────────────────────────────────── */
.filter-hdr {{
    font-size:0.65rem; font-weight:700; color:{muted};
    text-transform:uppercase; letter-spacing:0.07em;
    display:flex; align-items:center; gap:5px; margin-bottom:4px;
}}
.filter-hdr svg {{ width:11px; height:11px; stroke:{muted}; }}

/* ── Streamlit widget overrides ──────────────────────────────────────────── */
.stSelectbox label, .stMultiSelect label, .stSlider label {{
    font-size:0.68rem !important; color:{muted} !important;
    font-weight:600 !important; text-transform:uppercase !important;
    letter-spacing:0.05em !important;
}}
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background:{surf} !important;
    border-color:{border} !important;
    color:{text} !important;
    font-size:0.82rem !important;
}}
[data-baseweb="tag"]      {{ background:#3B82F6 !important; }}
[data-baseweb="tag"] span {{ color:#fff !important; }}
[data-baseweb="tab-highlight"] {{ display:none !important; }}

/* ── Tab bar ─────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background:{surf} !important;
    border:0.5px solid {border} !important;
    border-radius:10px !important;
    padding:3px 4px !important;
    gap:2px !important;
    margin-bottom:6px !important;
}}
[data-testid="stTabs"] button {{
    font-size:0.78rem !important; font-weight:500 !important;
    color:{muted} !important; border-radius:7px !important;
    padding:4px 11px !important; transition:all .12s !important;
}}
[data-testid="stTabs"] button:hover {{
    background:{tab_hover_bg} !important; color:{text} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    background:#3B82F6 !important; color:#fff !important; font-weight:600 !important;
}}

/* ── KPI cards ───────────────────────────────────────────────────────────── */
.kpi-wrap {{
    background:{surf};
    border:0.5px solid {border};
    border-radius:12px;
    padding:10px 12px;
    box-shadow:{shadow};
    position:relative;
    overflow:hidden;
    transition:transform .15s, box-shadow .15s;
}}
.kpi-wrap:hover {{ transform:translateY(-2px); box-shadow:0 4px 14px rgba(0,0,0,0.12); }}
.kpi-accent {{
    position:absolute; top:0; left:0;
    width:100%; height:3px; border-radius:12px 12px 0 0;
}}
.kpi-icon-wrap {{
    width:30px; height:30px; border-radius:8px;
    display:flex; align-items:center; justify-content:center; margin-bottom:6px;
}}
.kpi-icon-wrap svg {{ width:15px; height:15px; }}
.kpi-val   {{ font-size:1.3rem; font-weight:700; color:{text}; line-height:1.05; }}
.kpi-label {{
    font-size:0.63rem; font-weight:600; color:{muted};
    text-transform:uppercase; letter-spacing:0.05em; margin-top:2px;
}}
.kpi-delta-up   {{ font-size:0.67rem; font-weight:700; color:#10B981; margin-top:4px; }}
.kpi-delta-down {{ font-size:0.67rem; font-weight:700; color:#EF4444; margin-top:4px; }}
.kpi-delta-flat {{ font-size:0.67rem; color:{muted}; margin-top:4px; }}
.kpi-tooltip {{ font-size:0.6rem; color:{muted}; margin-top:2px; line-height:1.3;
                opacity:0.75; white-space:normal; }}

/* ── Insight box (below each chart) ──────────────────────────────────────── */
.insight {{
    background:{insight_bg};
    border-left:3px solid #10B981;
    border-radius:0 8px 8px 0;
    padding:6px 10px;
    font-size:0.74rem; color:{text};
    margin-top:4px; line-height:1.5;
}}

/* ── Active click-filter notice ──────────────────────────────────────────── */
.click-info {{
    background:{click_bg};
    border:0.5px solid {click_border};
    border-radius:9px; padding:7px 12px;
    font-size:0.76rem; color:{text};
    margin-bottom:5px;
    display:flex; align-items:center; gap:8px;
}}

hr {{ border-color:{border} !important; margin:4px 0 !important; }}
</style>""", unsafe_allow_html=True)


inject_css(st.session_state.dark_mode)

# Convenience colour variables — rebuilt after each theme injection
DARK      = st.session_state.dark_mode
TEXT_COL  = "#F1F5F9" if DARK else "#0F172A"
MUTED_COL = "#94A3B8" if DARK else "#64748B"
SURF_COL  = "#1E293B" if DARK else "#FFFFFF"

# Target chart height (px) sized so the full page fits in 1080 px tall windows.
# Reduce this value if you need taller tab-control areas.
CHART_HEIGHT = 320


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
#  Cached so the heavy merge/impute step only runs once per session.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load, merge, and preprocess the WHO and Gapminder source files.

    Returns
    -------
    pd.DataFrame
        Cleaned, merged dataset with derived columns:
        - gdp_final   : best available GDP per capita
        - log_gdp     : natural log of gdp_final (for Preston curve)
        - immunization: mean of hepatitis_b, polio, diphtheria coverage
        - pop_size    : sqrt-scaled population (for bubble chart sizing)
        - dev_stage   : income quartile label based on log_gdp
        - *_n         : min-max normalised versions of key indicators
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Load WHO file (try both common filename variants) ──────────────────
    who_df = None
    for filename in ["Life_Expectancy_Data.csv", "Life Expectancy Data.csv"]:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            who_df = pd.read_csv(path)
            break
    if who_df is None:
        raise FileNotFoundError(
            "Life_Expectancy_Data.csv not found. "
            "Place it in the same folder as this script."
        )

    gap_df = pd.read_csv(os.path.join(base_dir, "gapminder_data_graphs.csv"))

    # ── Normalise column names ─────────────────────────────────────────────
    who_df.columns = [
        c.strip().lower().replace(" ", "_").replace("/", "_")
        for c in who_df.columns
    ]
    gap_df.columns = [c.strip().lower() for c in gap_df.columns]

    # ── Rename WHO columns to stable keys ─────────────────────────────────
    rename_map: dict[str, str] = {}
    for col in who_df.columns:
        if   "life"   in col and "expect" in col: rename_map[col] = "life_expectancy"
        elif "bmi"    in col:                      rename_map[col] = "bmi"
        elif "hiv"    in col:                      rename_map[col] = "hiv_aids"
        elif "dipht"  in col:                      rename_map[col] = "diphtheria"
        elif "thin"   in col and "1-19"  in col:   rename_map[col] = "thinness_1_19"
        elif "thin"   in col and "5-9"   in col:   rename_map[col] = "thinness_5_9"
        elif "income" in col:                      rename_map[col] = "income_composition"
        elif "under"  in col:                      rename_map[col] = "under_five_deaths"
        elif "hepat"  in col:                      rename_map[col] = "hepatitis_b"
        elif "measl"  in col:                      rename_map[col] = "measles"
        elif "perce"  in col:                      rename_map[col] = "pct_expenditure"
        elif "total"  in col and "exp"   in col:   rename_map[col] = "total_expenditure"
        elif "adult"  in col:                      rename_map[col] = "adult_mortality"
        elif "infant" in col:                      rename_map[col] = "infant_deaths"
        elif "polio"  in col:                      rename_map[col] = "polio"
        elif "school" in col:                      rename_map[col] = "schooling"
        elif "popul"  in col:                      rename_map[col] = "population"
        elif "gdp"    in col:                      rename_map[col] = "gdp_who"
    who_df.rename(columns=rename_map, inplace=True)

    # ── Impute missing numeric values with per-country median ──────────────
    for df in (who_df, gap_df):
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = (
                df.groupby("country")[col]
                  .transform(lambda s: s.fillna(s.median()))
            )
            df[col] = df[col].fillna(df[col].median())

    # ── Merge WHO + Gapminder on (country, year) ───────────────────────────
    merged = pd.merge(
        who_df,
        gap_df[["country", "year", "continent", "hdi_index", "co2_consump", "gdp", "services"]],
        on=["country", "year"],
        how="inner",
    )

    # ── Derive helper columns ──────────────────────────────────────────────
    merged["gdp_final"]    = merged["gdp"].fillna(merged.get("gdp_who", np.nan))
    merged["log_gdp"]      = np.log1p(merged["gdp_final"])

    immunization_cols      = [c for c in ["hepatitis_b", "polio", "diphtheria"] if c in merged.columns]
    merged["immunization"] = merged[immunization_cols].mean(axis=1)

    merged["population"]   = pd.to_numeric(
        merged.get("population", 1e6), errors="coerce"
    ).fillna(1e6)
    merged["pop_m"]        = merged["population"] / 1e6
    merged["pop_size"]     = np.sqrt(merged["pop_m"]).clip(4, 55)  # bubble chart size

    # Income-stage quartile labels derived from log GDP
    merged["dev_stage"] = pd.cut(
        merged["log_gdp"],
        bins=[-np.inf, 5, 7, 9, np.inf],
        labels=["Low Income", "Lower-Middle", "Upper-Middle", "High Income"],
    )

    # Min-max normalise key indicators (used in radar / composite views)
    scaler = MinMaxScaler()
    normalise_cols = ["life_expectancy", "schooling", "immunization", "hdi_index", "log_gdp"]
    for col in normalise_cols:
        if col in merged.columns:
            merged[col + "_n"] = scaler.fit_transform(merged[[col]])

    return merged


# ── Load data — show a spinner while the cache is cold ────────────────────
with st.spinner("Loading data…"):
    try:
        DF   = load_data()
        data_ok = True
    except Exception as exc:
        data_ok  = False
        load_err = str(exc)

if not data_ok:
    st.error(
        f"Cannot load CSV files. Place them in the same folder as this script.\n\n`{load_err}`"
    )
    st.stop()

# ── Derive global filter options from the loaded data ─────────────────────
ALL_CONTINENTS    = sorted(DF["continent"].dropna().unique())
ALL_INCOME_STAGES = ["Low Income", "Lower-Middle", "Upper-Middle", "High Income"]
YEAR_MIN          = int(DF["year"].min())
YEAR_MAX          = int(DF["year"].max())


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def filter_data(
    base: pd.DataFrame,
    continents: list[str],
    year_range: tuple[int, int],
    income_stages: list[str],
) -> pd.DataFrame:
    """
    Return a filtered slice of the dataset.

    This is the core cross-filtering function.  Every chart and every KPI
    card calls this with the same global widget values, so they all reflect
    the same selection simultaneously.

    Parameters
    ----------
    base          : Full source DataFrame.
    continents    : Selected continent names (empty list = all).
    year_range    : (start_year, end_year) inclusive.
    income_stages : Selected development-stage labels (empty list = all).
    """
    result = base.copy()
    if continents:
        result = result[result["continent"].isin(continents)]
    result = result[result["year"].between(year_range[0], year_range[1])]
    if income_stages:
        result = result[result["dev_stage"].isin(income_stages)]
    return result


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#RRGGBB' hex string to an 'rgba(r,g,b,a)' CSS string."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def apply_chart_theme(fig: go.Figure, height: int = CHART_HEIGHT) -> go.Figure:
    """
    Apply the shared dark/light theme to a Plotly figure.

    Sets transparent backgrounds, consistent font, muted grid lines, and
    a uniform hover-label style.  All charts call this so the theme is
    defined in one place and updates automatically when the user toggles
    dark/light mode.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", size=11, color=TEXT_COL),
        margin=dict(t=28, b=38, l=50, r=18),
        hoverlabel=dict(bgcolor=SURF_COL, font_size=12, font_color=TEXT_COL),
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.15)",
        zeroline=False,
        tickfont=dict(size=10, color=MUTED_COL),
        showline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.15)",
        zeroline=False,
        tickfont=dict(size=10, color=MUTED_COL),
        showline=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER ROW — title | global filters | theme toggle
#  Laid out in three columns proportioned to fill the full width.
# ══════════════════════════════════════════════════════════════════════════════
header_col_title, header_col_filters, header_col_toggle = st.columns([1.2, 3.5, 0.5])

with header_col_title:
    st.markdown(
        '<div class="dash-title">Global Health & Life Expectancy</div>'
        '<div class="dash-sub">SDG 3 · WHO + Gapminder · 2000–2015</div>',
        unsafe_allow_html=True,
    )

with header_col_filters:
    # ── Filter bar label ──────────────────────────────────────────────────
    # These three widgets are the "global filters" that cross-filter every
    # KPI card and every chart on the page simultaneously.
    st.markdown(
        """<div class="filter-hdr">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2.5"
               stroke-linecap="round" stroke-linejoin="round">
            <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
          </svg>
          Global Filters — updates all KPIs and charts
        </div>""",
        unsafe_allow_html=True,
    )
    fc1, fc2, fc3 = st.columns([2.5, 2, 1.5])
    with fc1:
        # Multi-select: choose one, many, or "All Continents"
        cont_options = ["All Continents"] + ALL_CONTINENTS
        sel_cont_raw = st.multiselect(
            "Continent",
            cont_options,
            default=["All Continents"],
            key="g_cont",
            placeholder="Select continents…",
        )
        # If "All Continents" is chosen (or nothing selected), use every continent
        if not sel_cont_raw or "All Continents" in sel_cont_raw:
            sel_continents = ALL_CONTINENTS
        else:
            sel_continents = sel_cont_raw
    with fc2:
        # Multi-select: choose one, many, or "All Stages"
        stage_options = ["All Stages"] + ALL_INCOME_STAGES
        sel_stage_raw = st.multiselect(
            "Development Stage",
            stage_options,
            default=["All Stages"],
            key="g_stage",
            placeholder="Select stages…",
        )
        # If "All Stages" is chosen (or nothing selected), use every stage
        if not sel_stage_raw or "All Stages" in sel_stage_raw:
            sel_income_stages = ALL_INCOME_STAGES
        else:
            sel_income_stages = sel_stage_raw
    with fc3:
        sel_year_range = st.slider(
            "Year Range",
            YEAR_MIN, YEAR_MAX,
            (YEAR_MIN, YEAR_MAX),
            key="g_yr",
        )

with header_col_toggle:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    toggle_icon  = "🌙" if not DARK else "☀️"
    toggle_label = "Dark"  if not DARK else "Light"
    if st.button(f"{toggle_icon} {toggle_label}", key="theme_btn", help="Toggle dark / light mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Apply all three global filters to produce the shared filtered DataFrame ──
# Every component below this line reads from DFF, so changing any filter
# above automatically propagates to all KPIs and all six tab charts.
DFF = filter_data(DF, sel_continents, sel_year_range, sel_income_stages)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  KPI CARDS
#  Seven metric cards in a single responsive row.
#  Each card shows: icon · current mean value · trend delta (start→end year).
#  Colors match the chart trace color for that metric (see KPI_META comment).
# ══════════════════════════════════════════════════════════════════════════════
def compute_kpi_delta_html(col: str, good_direction: str) -> str:
    """
    Return an HTML snippet showing the change in `col` from the start to the
    end of the selected year range.

    The arrow colour is green when the direction of change is "good" (e.g.
    life expectancy rising, adult mortality falling) and red otherwise.
    """
    if sel_year_range[0] == sel_year_range[1]:
        return "<div class='kpi-delta-flat'>— single year</div>"
    if col not in DFF.columns:
        return ""

    start_mean = DFF[DFF["year"] == sel_year_range[0]][col].mean()
    end_mean   = DFF[DFF["year"] == sel_year_range[1]][col].mean()
    delta      = end_mean - start_mean
    improving  = (good_direction == "up" and delta >= 0) or \
                 (good_direction == "down" and delta <= 0)

    arrow = "▲" if delta >= 0 else "▼"
    css   = "kpi-delta-up" if improving else "kpi-delta-down"
    yr    = (
        f"<span style='font-weight:400;opacity:0.65'>"
        f"{sel_year_range[0]}→{sel_year_range[1]}</span>"
    )
    return f"<div class='{css}'>{arrow} {abs(delta):.1f} &nbsp;{yr}</div>"


kpi_columns = st.columns(7)
for kpi_col_widget, (col_key, label, unit, good_dir, color, icon_path) in zip(kpi_columns, KPI_META):
    mean_val = DFF[col_key].mean() if col_key in DFF.columns else float("nan")

    # Format the displayed value according to the metric's precision needs
    if np.isnan(mean_val):
        display_val = "N/A"
    elif col_key == "hdi_index":
        display_val = f"{mean_val:.3f}"
    elif col_key in ("life_expectancy", "schooling"):
        display_val = f"{mean_val:.1f} {unit}"
    else:
        display_val = f"{mean_val:.1f}{unit}"

    icon_bg   = color + "22"   # 13 % opacity tint of the metric color
    delta_html = compute_kpi_delta_html(col_key, good_dir)

    # Short tooltip for each KPI explaining what it means
    KPI_TOOLTIPS = {
        "life_expectancy":   "Average number of years a person is expected to live",
        "adult_mortality":   "Deaths per 1,000 adults aged 15–60",
        "immunization":      "Average coverage of Hepatitis B, Polio & Diphtheria vaccines",
        "hdi_index":         "UN composite index: health + education + income (0–1)",
        "under_five_deaths": "Deaths of children under 5 per 1,000 live births",
        "hiv_aids":          "HIV/AIDS deaths per 1,000 population",
        "schooling":         "Average years of schooling received",
    }
    tooltip = KPI_TOOLTIPS.get(col_key, "")

    with kpi_col_widget:
        st.markdown(f"""
        <div class="kpi-wrap" title="{tooltip}">
          <div class="kpi-accent" style="background:{color}"></div>
          <div class="kpi-icon-wrap" style="background:{icon_bg}">
            <svg viewBox="0 0 24 24" fill="none" stroke="{color}"
                 stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="{icon_path}"/>
            </svg>
          </div>
          <div class="kpi-val" style="color:{color}">{display_val}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-tooltip">{tooltip}</div>
          {delta_html}
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLICK-TO-FILTER SESSION STATE
#  These keys let individual tab controls remember which continent/stage was
#  last highlighted, so selecting a group inside one tab pre-selects the same
#  group when the user switches to another tab.
# ══════════════════════════════════════════════════════════════════════════════
if "clicked_cont"  not in st.session_state: st.session_state.clicked_cont  = "All"
if "clicked_stage" not in st.session_state: st.session_state.clicked_stage = "All"
if "clicked_year"  not in st.session_state: st.session_state.clicked_year  = None


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
#  Six analysis views, each in its own tab.  Switching tabs does NOT re-filter
#  the data — all tabs share DFF.  Per-tab controls only change what is
#  visually highlighted, not the underlying data slice.
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
#  TAB 1 — Distribution Histograms
#  Shows nine variable distributions in a 3×3 grid.
#  Selecting a continent dims all other continents, showing their
#  distribution shape in context (overlay mode) — a form of visual filtering.
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    t1_left, _ = st.columns([3, 1])
    with t1_left:
        # Show a banner when a click-filter from another tab is active
        if st.session_state.clicked_cont != "All":
            st.markdown(
                f'<div class="click-info">🖱️ <b>Click filter active:</b> '
                f'showing <b>{st.session_state.clicked_cont}</b> highlighted. '
                f'Change the selector below to reset.</div>',
                unsafe_allow_html=True,
            )
        # Highlight selector — synced with clicked_cont session state so
        # clicking a continent in another tab pre-selects it here too.
        t1_hl_raw = st.multiselect(
            "🖱️ Highlight continent — select one or more (or keep All to see all)",
            ["All"] + ALL_CONTINENTS,
            default=[st.session_state.clicked_cont]
                     if st.session_state.clicked_cont != "All" else ["All"],
            key="t1_hl",
        )
        t1_highlight = "All" if (not t1_hl_raw or "All" in t1_hl_raw) else t1_hl_raw[0]
        st.session_state.clicked_cont = t1_highlight   # write back for cross-tab sync

    # Variables to plot — only include columns present in the filtered data
    histogram_vars = [
        ("life_expectancy", "Life Expectancy", "#10B981"),
        ("adult_mortality",  "Adult Mortality",  "#EF4444"),
        ("gdp_final",        "GDP per Capita",   "#45B7D1"),
        ("schooling",        "Schooling",         "#6366F1"),
        ("infant_deaths",    "Infant Deaths",     "#F97316"),
        ("bmi",              "BMI",               "#EC4899"),
        ("alcohol",          "Alcohol",           "#F59E0B"),
        ("hiv_aids",         "HIV/AIDS",          "#DC2626"),
        ("immunization",     "Immunization",      "#3B82F6"),
    ]
    histogram_vars = [(c, l, col) for c, l, col in histogram_vars if c in DFF.columns]

    fig1 = make_subplots(
        rows=3, cols=3,
        subplot_titles=[v[1] for v in histogram_vars],
        horizontal_spacing=0.06,
        vertical_spacing=0.13,
    )

    for idx, (col_name, label, color) in enumerate(histogram_vars):
        row, col = divmod(idx, 3)

        if t1_highlight != "All":
            # Overlay all continents; dim non-selected ones
            for continent in ALL_CONTINENTS:
                subset       = DFF[DFF["continent"] == continent][col_name].dropna()
                is_selected  = (continent == t1_highlight or
                                (isinstance(t1_hl_raw, list) and continent in t1_hl_raw
                                 and "All" not in t1_hl_raw))
                trace_color  = (
                    CONTINENT_COLORS.get(continent, "#888")
                    if is_selected else "rgba(148,163,184,0.25)"
                )
                fig1.add_trace(
                    go.Histogram(
                        x=subset, nbinsx=25, name=continent,
                        marker_color=trace_color,
                        marker_line=dict(width=0),
                        opacity=1.0 if is_selected else 0.3,
                        showlegend=(idx == 0),
                        hovertemplate=f"<b>{continent}</b><br>{label}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                    ),
                    row=row + 1, col=col + 1,
                )
        else:
            # Single color; add median reference line
            data = DFF[col_name].dropna()
            fig1.add_trace(
                go.Histogram(
                    x=data, nbinsx=30,
                    marker_color=color, marker_opacity=0.75,
                    marker_line=dict(width=0), showlegend=False,
                    hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=row + 1, col=col + 1,
            )
            fig1.add_vline(
                x=float(data.median()),
                line_dash="dot", line_color="rgba(148,163,184,0.7)", line_width=1.5,
                row=row + 1, col=col + 1,
            )

    fig1.update_layout(
        height=CHART_HEIGHT,
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COL, size=10),
        title=dict(
            text="Distribution of Key Health & Socioeconomic Indicators",
            font=dict(size=12, color=TEXT_COL), x=0, pad=dict(b=4),
        ),
        margin=dict(t=48, b=16, l=36, r=16),
        showlegend=(t1_highlight != "All"),
        legend=dict(
            orientation="h", y=1.06, x=0.5, xanchor="center",
            font=dict(size=9, color=TEXT_COL), bgcolor="rgba(0,0,0,0)",
        ),
    )
    for axis_key in fig1.layout:
        if axis_key.startswith("xaxis") or axis_key.startswith("yaxis"):
            fig1.layout[axis_key].update(
                gridcolor="rgba(148,163,184,0.12)",
                tickfont=dict(size=8, color=MUTED_COL),
                zeroline=False,
            )
    for annotation in fig1.layout.annotations[:9]:
        annotation.font.size  = 9
        annotation.font.color = TEXT_COL

    st.plotly_chart(fig1, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Correlation Heatmap
#  Lower-triangle Pearson correlation matrix.
#  Hovering a cell shows the exact r value.
#  The heatmap itself does not have per-tab filters; it always reflects the
#  global filter (DFF).
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    corr_vars = [
        "life_expectancy", "adult_mortality", "schooling", "gdp_final",
        "hiv_aids", "immunization", "hdi_index", "income_composition",
        "infant_deaths", "alcohol", "bmi",
    ]
    corr_vars  = [c for c in corr_vars if c in DFF.columns]
    corr_matrix = DFF[corr_vars].corr().round(3)
    axis_labels = [c.replace("_", " ").title() for c in corr_vars]

    # Build lower-triangle only (upper half left as None = invisible)
    n = len(corr_vars)
    z_lower   = [
        [corr_matrix.iloc[i, j] if i >= j else None for j in range(n)]
        for i in range(n)
    ]
    text_lower = [
        [f"{corr_matrix.iloc[i, j]:.2f}" if i >= j else "" for j in range(n)]
        for i in range(n)
    ]

    fig2 = go.Figure(go.Heatmap(
        z=z_lower, x=axis_labels, y=axis_labels,
        text=text_lower, texttemplate="%{text}",
        textfont=dict(size=9, color=TEXT_COL),
        colorscale=[
            [0,   "#EF4444"],
            [0.5, "rgba(148,163,184,0.12)"],
            [1,   "#10B981"],
        ],
        zmid=0, zmin=-1, zmax=1,
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="r",
            tickfont=dict(size=9, color=MUTED_COL),
            thickness=13, len=0.75,
        ),
    ))
    fig2 = apply_chart_theme(fig2)
    fig2.update_layout(
        title=dict(
            text="Pearson Correlation Matrix — How Health Indicators Relate to Each Other",
            font=dict(size=12, color=TEXT_COL), x=0,
        ),
        margin=dict(t=44, b=88, l=148, r=48),
    )
    fig2.update_xaxes(tickangle=-38, tickfont=dict(size=9, color=MUTED_COL), showgrid=False)
    fig2.update_yaxes(tickfont=dict(size=9, color=MUTED_COL), showgrid=False)

    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        '<div class="insight">💡 <b>HDI (r≈0.85)</b> and <b>schooling (r≈0.75)</b> '
        "are the strongest positive predictors of life expectancy. "
        "<b>HIV/AIDS (r≈−0.58)</b> and <b>adult mortality (r≈−0.97)</b> are the "
        "strongest negative predictors. Hover a cell for the exact value.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Preston Curve
#  Scatter of life expectancy vs a chosen X-axis variable.
#  Clicking a continent in the dropdown highlights its points and dims others —
#  this is cross-filtering between the dropdown and the scatter traces.
#  An OLS trend line shows the global relationship; annotated outliers sit
#  far below it.
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    p3_left, p3_right = st.columns([3, 1])
    with p3_left:
        t3_hl_raw = st.multiselect(
            "Highlight continent (select one or more to compare)",
            ["All"] + ALL_CONTINENTS,
            default=["All"], key="t3_hl",
        )
        t3_highlight = "All" if (not t3_hl_raw or "All" in t3_hl_raw) else t3_hl_raw[0]
    with p3_right:
        x_axis_option = st.selectbox(
            "X-axis variable",
            [
                ("log_gdp",      "log GDP"),
                ("gdp_final",    "GDP per Capita"),
                ("schooling",    "Schooling"),
                ("immunization", "Immunization"),
                ("hdi_index",    "HDI"),
            ],
            format_func=lambda x: x[1],
            key="t3_x",
        )
    x_col, x_label = x_axis_option

    fig3 = go.Figure()
    for continent in sorted(DFF["continent"].dropna().unique()):
        subset       = DFF[DFF["continent"] == continent].dropna(subset=[x_col, "life_expectancy"])
        is_highlighted = (t3_highlight == "All" or continent == t3_highlight or
                          (isinstance(t3_hl_raw, list) and continent in t3_hl_raw))
        trace_color  = (
            CONTINENT_COLORS.get(continent, "#888")
            if is_highlighted else "rgba(148,163,184,0.2)"
        )
        fig3.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset["life_expectancy"],
            mode="markers",
            name=continent,
            marker=dict(
                color=trace_color,
                size=6 if is_highlighted else 4,
                opacity=0.75 if is_highlighted else 0.2,
                line=dict(width=0),
            ),
            customdata=np.stack(
                [subset["country"], subset["year"], subset["gdp_final"].round(0), subset["continent"]],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Life Exp: <b>%{y:.1f} yrs</b><br>"
                f"{x_label}: %{{x:.2f}}<br>GDP: $%{{customdata[2]:,.0f}}<extra></extra>"
            ),
        ))

    # OLS trend line across all filtered data
    valid = DFF.dropna(subset=[x_col, "life_expectancy"])
    r_val = 0.0
    if len(valid) > 5:
        slope, intercept, r_val, *_ = stats.linregress(valid[x_col], valid["life_expectancy"])
        xs_trend = np.linspace(valid[x_col].min(), valid[x_col].max(), 300)
        fig3.add_trace(go.Scatter(
            x=xs_trend, y=slope * xs_trend + intercept,
            mode="lines",
            name=f"OLS trend (r={r_val:.2f})",
            line=dict(color="rgba(148,163,184,0.7)", width=2, dash="dash"),
            hoverinfo="skip",
        ))

    # Annotate notable outliers that sit far below the trend line
    for country_name in ["Sierra Leone", "Lesotho"]:
        row = DFF[DFF["country"] == country_name][[x_col, "life_expectancy"]].mean()
        if not row.empty and not row.isna().any():
            fig3.add_annotation(
                x=float(row[x_col]), y=float(row["life_expectancy"]),
                text=f"⚠ {country_name}",
                showarrow=True, arrowhead=2,
                arrowcolor="#EF4444",
                font=dict(size=10, color="#EF4444"),
                ax=38, ay=-22,
            )

    fig3 = apply_chart_theme(fig3)
    fig3.update_layout(
        title=dict(
            text=f"Preston Curve — Life Expectancy vs {x_label} by Continent",
            font=dict(size=12, color=TEXT_COL), x=0,
        ),
        xaxis=dict(title=x_label, gridcolor="rgba(148,163,184,0.12)"),
        yaxis=dict(title="Life Expectancy (yrs)"),
        legend=dict(
            orientation="v", x=1.01, y=1,
            font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        f'<div class="insight">💡 Select a continent above to isolate it. '
        f"The OLS trend line (r={r_val:.2f}) shows the global wealth–health relationship. "
        f"⚠ annotated countries sit far below the trend line.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Trends Over Time
#  Line chart with ±1 SD ribbon per continent.
#  Selecting a continent fades all others, spotlighting its trajectory.
#  This is linked interaction: the highlight here shares the same
#  clicked_cont session state key used in Tab 1.
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    t4_left, t4_right = st.columns([3, 1])
    with t4_left:
        t4_hl_raw = st.multiselect(
            "Highlight continent (select one or more)",
            ["All"] + ALL_CONTINENTS,
            default=["All"], key="t4_hl",
        )
        t4_highlight = "All" if (not t4_hl_raw or "All" in t4_hl_raw) else t4_hl_raw[0]
    with t4_right:
        t4_metric = st.selectbox(
            "Metric",
            [
                ("life_expectancy", "Life Expectancy"),
                ("adult_mortality", "Adult Mortality"),
                ("schooling",       "Schooling"),
                ("immunization",    "Immunization"),
                ("gdp_final",       "GDP"),
                ("hiv_aids",        "HIV/AIDS"),
                ("hdi_index",       "HDI"),
            ],
            format_func=lambda x: x[1],
            key="t4_met",
        )
    metric_col, metric_label = t4_metric

    # Build per-continent mean ± std time series
    trend_df = (
        DFF.groupby(["year", "continent"])[metric_col]
           .agg(["mean", "std"])
           .reset_index()
    )
    trend_df.columns = ["year", "continent", "mean", "std"]

    # Map metric column → KPI card color for consistent color encoding
    metric_color_map = {k[0]: k[4] for k in KPI_META}

    fig4 = go.Figure()
    for continent in sorted(trend_df["continent"].dropna().unique()):
        subset         = trend_df[trend_df["continent"] == continent].sort_values("year")
        is_highlighted = (t4_highlight == "All" or continent == t4_highlight or
                          (isinstance(t4_hl_raw, list) and continent in t4_hl_raw))
        raw_color      = CONTINENT_COLORS.get(continent, "#888")
        trace_color    = raw_color if is_highlighted else "rgba(148,163,184,0.25)"
        xs             = subset["year"].tolist()
        upper          = (subset["mean"] + subset["std"]).tolist()
        lower          = (subset["mean"] - subset["std"]).tolist()

        # ±1 SD ribbon (filled area)
        fig4.add_trace(go.Scatter(
            x=xs + xs[::-1], y=upper + lower[::-1],
            fill="toself",
            fillcolor=hex_to_rgba(raw_color, 0.08 if is_highlighted else 0.015),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
            legendgroup=continent,
        ))
        # Mean line
        fig4.add_trace(go.Scatter(
            x=xs, y=subset["mean"].round(2),
            mode="lines+markers", name=continent, legendgroup=continent,
            line=dict(color=trace_color, width=2.5 if is_highlighted else 1.1),
            marker=dict(size=5 if is_highlighted else 3, color=trace_color),
            opacity=1.0 if is_highlighted else 0.3,
            hovertemplate=f"<b>{continent}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
        ))

    fig4 = apply_chart_theme(fig4)
    fig4.update_layout(
        title=dict(
            text=f"{metric_label} Trend Over Time by Continent (2000–2015) — Shaded Area = ±1 SD",
            font=dict(size=12, color=TEXT_COL), x=0,
        ),
        xaxis=dict(title="Year", dtick=2),
        yaxis=dict(title=metric_label),
        legend=dict(
            orientation="h", y=1.08, x=0.5, xanchor="center",
            font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        '<div class="insight">💡 Shaded ribbons = ±1 SD. '
        "Africa shows the steepest positive life expectancy trajectory from 2000–2015 — "
        "evidence that SDG interventions are working but from a low base.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — Violin Plot by Income Level
#  Compares the full distribution of a metric across development stages.
#  Selecting an income stage highlights that violin and dims others — another
#  form of visual cross-filtering within the tab.
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    t5_left, t5_right = st.columns([3, 1])
    with t5_left:
        t5_hl_raw = st.multiselect(
            "Highlight income stage (select one or more)",
            ["All"] + ALL_INCOME_STAGES,
            default=["All"], key="t5_hl",
        )
        t5_highlight = "All" if (not t5_hl_raw or "All" in t5_hl_raw) else t5_hl_raw[0]
    with t5_right:
        t5_metric = st.selectbox(
            "Metric",
            [
                ("life_expectancy", "Life Expectancy"),
                ("schooling",       "Schooling"),
                ("immunization",    "Immunization"),
                ("adult_mortality", "Adult Mortality"),
                ("hiv_aids",        "HIV/AIDS"),
                ("gdp_final",       "GDP"),
            ],
            format_func=lambda x: x[1],
            key="t5_met",
        )
    violin_col, violin_label = t5_metric

    fig5 = go.Figure()
    for stage in ALL_INCOME_STAGES:
        subset         = DFF[DFF["dev_stage"] == stage][violin_col].dropna()
        is_highlighted = (t5_highlight == "All" or stage == t5_highlight or
                          (isinstance(t5_hl_raw, list) and stage in t5_hl_raw))
        trace_color    = (
            INCOME_STAGE_COLORS.get(stage, "#888")
            if is_highlighted else "rgba(148,163,184,0.3)"
        )
        fig5.add_trace(go.Violin(
            y=subset, x=[stage] * len(subset), name=stage,
            box_visible=True, meanline_visible=True,
            fillcolor=trace_color,
            opacity=0.82 if is_highlighted else 0.2,
            width=0.7,
            line=dict(color=trace_color, width=1.5),
            points="outliers",
            marker=dict(color=trace_color, size=4, opacity=0.4),
            hovertemplate=f"<b>{stage}</b><br>{violin_label}: %{{y:.1f}}<extra></extra>",
        ))

    fig5 = apply_chart_theme(fig5)
    fig5.update_layout(
        title=dict(
            text=f"{violin_label} Distribution by Income Level — Box shows median & IQR, dots = outliers",
            font=dict(size=12, color=TEXT_COL), x=0,
        ),
        xaxis=dict(title="Development Stage (Low → High Income)",
                   showgrid=False, tickfont=dict(color=TEXT_COL)),
        yaxis=dict(title=violin_label),
        violinmode="group",
        legend=dict(
            orientation="h", y=1.08, x=0.5, xanchor="center",
            font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        '<div class="insight">💡 High-income countries reach 80+ years with narrow spread. '
        "Low-income nations show wide variance — high within-group inequality driven by "
        "conflict, HIV burden, and under-investment.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 — Animated Gapminder Bubble Chart
#  Replicates the famous Gapminder visualisation: GDP (x) vs life expectancy
#  (y), bubble size = population, colour = continent.
#  Press ▶ to animate across all years.  Selecting a continent highlights it
#  and fades all others — cross-filtering in motion.
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    t6_hl_raw = st.multiselect(
        "Highlight continent (select one or more to focus)",
        ["All"] + ALL_CONTINENTS,
        default=["All"], key="t6_hl",
    )
    t6_highlight = "All" if (not t6_hl_raw or "All" in t6_hl_raw) else t6_hl_raw[0]

    # Aggregate to one point per (country, year) for the animation
    anim_df = (
        DFF.groupby(["country", "year", "continent"])
           .agg(
               life_expectancy=("life_expectancy", "mean"),
               log_gdp=("log_gdp",          "mean"),
               pop_size=("pop_size",         "mean"),
               gdp_final=("gdp_final",       "mean"),
               schooling=("schooling",       "mean"),
           )
           .reset_index()
           .sort_values("year")
    )

    # Build color map: highlighted continent keeps its palette color;
    # all others are dimmed to near-invisible grey.
    highlighted_set = set(ALL_CONTINENTS) if t6_highlight == "All" else (
        set(t6_hl_raw) - {"All"} if t6_hl_raw and "All" not in t6_hl_raw else set(ALL_CONTINENTS)
    )
    color_map = {
        c: (CONTINENT_COLORS.get(c, "#888") if c in highlighted_set else "rgba(148,163,184,0.18)")
        for c in ALL_CONTINENTS
    }

    fig6 = px.scatter(
        anim_df,
        x="log_gdp", y="life_expectancy",
        animation_frame="year",
        animation_group="country",
        size="pop_size", color="continent",
        hover_name="country",
        custom_data=["gdp_final", "schooling"],
        color_discrete_map=color_map,
        size_max=55,
        labels={
            "log_gdp":         "log(GDP per Capita)",
            "life_expectancy": "Life Expectancy (yrs)",
        },
    )
    fig6.update_traces(
        marker=dict(opacity=0.82, line=dict(width=0.5, color="rgba(255,255,255,0.35)")),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Life Exp: <b>%{y:.1f} yrs</b><br>"
            "log(GDP): %{x:.2f}<br>"
            "GDP: $%{customdata[0]:,.0f}<br>"
            "Schooling: %{customdata[1]:.1f} yrs<extra></extra>"
        ),
    )
    fig6 = apply_chart_theme(fig6)
    fig6.update_layout(
        title=dict(
            text="Gapminder Animated Bubble — Health vs Wealth Over Time (Press ▶ to Play)",
            font=dict(size=12, color=TEXT_COL), x=0,
        ),
        xaxis=dict(title="log(GDP per Capita) — Higher = Richer Country"),
        yaxis=dict(title="Life Expectancy (yrs)"),
        legend=dict(x=1.01, y=1, font=dict(size=10, color=TEXT_COL), bgcolor="rgba(0,0,0,0)"),
    )
    # Slow the animation slightly for smoother playback
    fig6.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]      = 700
    fig6.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400

    st.plotly_chart(fig6, use_container_width=True)
    st.markdown(
        '<div class="insight">💡 Press ▶ to watch 16 years of global health evolution. '
        "Bubble size = population. Select a continent above to highlight it. "
        "Africa (red) shows the steepest improvement trajectory.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;font-size:0.67rem;color:{MUTED_COL};padding:2px 0'>"
    "ITS68404 · Data Visualization · Group 8 · Taylor's University · January 2026 &nbsp;|&nbsp;"
    " SDG 3: Good Health and Well-Being &nbsp;|&nbsp;"
    " 2,416 records · 151 countries · 2000–2015"
    "</div>",
    unsafe_allow_html=True,
)
