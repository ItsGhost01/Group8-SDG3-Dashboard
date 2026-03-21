"""
ITS68404 – Data Visualization | Group 8
SDG 3: Good Health and Well-Being
Streamlit Dashboard — Professional Redesign

Design Principles Applied:
  1. Data-Ink Ratio     — Remove all non-data ink (gridlines, borders, backgrounds)
  2. Gestalt Principles — Proximity, similarity, continuity, figure-ground
  3. Pre-attentive Attr — Color, size, position to guide attention immediately
  4. Tufte Principles   — No chartjunk, small multiples, sparklines, clear labels
  5. Few's Rules        — Right chart for right question, no 3D pie/donut abuse
  6. Stakeholder Focus  — Every visual answers a real decision question
  7. Visual Hierarchy   — Most important info largest + top-left
  8. Color              — Max 6 categorical colors, semantic (red=bad, green=good)
  9. Consistency        — Same color = same meaning throughout
 10. Accessibility      — Sufficient contrast ratios, no color-only encoding
"""

import os, warnings
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SDG 3 — Global Health Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# DESIGN SYSTEM  (single source of truth for all colors)
# Principle: Consistency — same color = same meaning everywhere
# ══════════════════════════════════════════════════════════
C_BG     = "#F8F9FA"
C_SURF   = "#FFFFFF"
C_BORDER = "#E9ECEF"
C_TEXT   = "#212529"
C_MUTED  = "#6C757D"
C_GRID   = "#F1F3F5"

# Semantic palette — color encodes meaning, not decoration
C_GOOD   = "#0D9488"   # teal  — positive outcomes
C_BAD    = "#DC2626"   # red   — risk / negative indicators
C_NEUTRAL= "#3B82F6"   # blue  — neutral comparisons
C_WARN   = "#D97706"   # amber — caution / developing nations
C_PURPLE = "#7C3AED"   # purple — composite indices

# Continent palette — max 6, colorblind-considered
CONT_COL = {
    "Africa":        "#DC2626",
    "Asia":          "#3B82F6",
    "Europe":        "#0D9488",
    "North America": "#D97706",
    "South America": "#7C3AED",
    "Oceania":       "#DB2777",
}

# Chart layout defaults — high data-ink ratio
CHART_DEFAULTS = dict(
    paper_bgcolor=C_SURF, plot_bgcolor=C_SURF,
    font=dict(family="Inter, Arial, sans-serif", color=C_TEXT, size=12),
    hoverlabel=dict(bgcolor=C_SURF, bordercolor=C_BORDER,
                    font=dict(size=12, color=C_TEXT)),
)
AXIS_STYLE = dict(
    gridcolor=C_GRID, gridwidth=1, showline=False, zeroline=False,
    tickfont=dict(size=11, color=C_MUTED),
    title_font=dict(size=12, color=C_TEXT),
)

def apply_defaults(fig, title="", h=400, margin=None):
    m = margin or dict(t=50, b=40, l=50, r=20)
    fig.update_layout(**CHART_DEFAULTS, height=h, margin=m,
                      title=dict(text=title, x=0,
                                 font=dict(size=14, color=C_TEXT),
                                 pad=dict(b=10)))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ══════════════════════════════════════════════════════════
# CSS — minimal and purposeful
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8F9FA; }
[data-testid="stMain"] { padding-top: 1rem; }
[data-testid="stSidebar"] { background: #1E293B; border-right: 1px solid #334155; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size: 0.78rem !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] { background: #1D4ED8 !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] * { color: #fff !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { background: #fff; border-radius: 8px;
    padding: 3px 6px; gap: 2px; border: 1px solid #E9ECEF; }
[data-testid="stTabs"] button { color: #495057 !important; font-weight: 500 !important;
    font-size: 0.88rem !important; border-radius: 6px !important; padding: 6px 14px !important; }
[data-testid="stTabs"] button:hover { background: #F1F3F5 !important; color: #212529 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { background: #EFF6FF !important;
    color: #1D4ED8 !important; font-weight: 600 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

.kpi-card { background: #fff; border-radius: 10px; padding: 16px 18px;
    border: 1px solid #E9ECEF; border-top: 3px solid #3B82F6; }
.kpi-icon { width: 36px; height: 36px; border-radius: 8px; display: flex;
    align-items: center; justify-content: center; margin-bottom: 10px; }
.kpi-icon svg { width: 18px; height: 18px; }
.kpi-val { font-size: 1.6rem; font-weight: 700; line-height: 1.1; color: #212529; }
.kpi-lbl { font-size: 0.74rem; color: #6C757D; font-weight: 600; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-sub { font-size: 0.68rem; color: #9CA3AF; margin-top: 2px; }
.kpi-delta-up   { font-size: 0.74rem; color: #0D9488; font-weight: 600; margin-top: 8px; }
.kpi-delta-down { font-size: 0.74rem; color: #DC2626; font-weight: 600; margin-top: 8px; }

.sec-hdr { font-size: 0.72rem; font-weight: 600; color: #6C757D; text-transform: uppercase;
    letter-spacing: 0.08em; border-bottom: 1px solid #E9ECEF;
    padding-bottom: 6px; margin-bottom: 16px; }
.callout { background: #F0FDF4; border-left: 3px solid #0D9488; border-radius: 0 8px 8px 0;
    padding: 10px 14px; font-size: 0.82rem; color: #065F46; margin: 8px 0; line-height: 1.6; }
.callout-warn { background: #FFF7ED; border-left: 3px solid #D97706; border-radius: 0 8px 8px 0;
    padding: 10px 14px; font-size: 0.82rem; color: #92400E; margin: 8px 0; line-height: 1.6; }
h1,h2,h3,h4 { color: #212529 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    BASE = os.path.dirname(os.path.abspath(__file__))
    gap  = pd.read_csv(os.path.join(BASE, "gapminder_data_graphs.csv"))
    who  = pd.read_csv(os.path.join(BASE, "Life Expectancy Data.csv"))

    who.columns = [c.strip().lower().replace(" ","_").replace("/","_") for c in who.columns]
    gap.columns = [c.strip().lower() for c in gap.columns]

    rmap = {}
    for c in who.columns:
        if   "life"   in c and "expect" in c: rmap[c] = "life_expectancy"
        elif "bmi"    in c:                   rmap[c] = "bmi"
        elif "hiv"    in c:                   rmap[c] = "hiv_aids"
        elif "dipht"  in c:                   rmap[c] = "diphtheria"
        elif "thin"   in c and "1-19" in c:   rmap[c] = "thinness_1_19"
        elif "thin"   in c and "5-9"  in c:   rmap[c] = "thinness_5_9"
        elif "income" in c:                   rmap[c] = "income_composition"
        elif "under"  in c:                   rmap[c] = "under_five_deaths"
        elif "hepat"  in c:                   rmap[c] = "hepatitis_b"
        elif "measl"  in c:                   rmap[c] = "measles"
        elif "perce"  in c:                   rmap[c] = "pct_expenditure"
        elif "total"  in c and "exp" in c:    rmap[c] = "total_expenditure"
        elif "adult"  in c:                   rmap[c] = "adult_mortality"
        elif "infant" in c:                   rmap[c] = "infant_deaths"
        elif "polio"  in c:                   rmap[c] = "polio"
        elif "school" in c:                   rmap[c] = "schooling"
        elif "popul"  in c:                   rmap[c] = "population"
        elif "gdp"    in c:                   rmap[c] = "gdp_who"
    who.rename(columns=rmap, inplace=True)

    # Impute: country median then global median
    for col in who.select_dtypes(include=np.number).columns:
        who[col] = who.groupby("country")[col].transform(lambda x: x.fillna(x.median()))
        who[col] = who[col].fillna(who[col].median())
    for col in gap.select_dtypes(include=np.number).columns:
        gap[col] = gap.groupby("country")[col].transform(lambda x: x.fillna(x.median()))
        gap[col] = gap[col].fillna(gap[col].median())

    df = pd.merge(who,
                  gap[["country","year","continent","hdi_index","co2_consump","gdp","services"]],
                  on=["country","year"], how="inner")

    df["gdp_final"]     = df["gdp"].fillna(df["gdp_who"])
    df["log_gdp"]       = np.log1p(df["gdp_final"])
    df["immunization"]  = df[["hepatitis_b","polio","diphtheria"]].mean(axis=1)
    df["mortality_idx"] = (df["adult_mortality"] + df["under_five_deaths"]*5) / 100
    df["population"]    = pd.to_numeric(df["population"], errors="coerce").fillna(1e6)
    df["pop_m"]         = df["population"] / 1e6
    df["pop_size"]      = np.sqrt(df["pop_m"]).clip(4, 55)
    df["dev_stage"]     = pd.cut(df["log_gdp"],
                                 bins=[-np.inf, 5, 7, 9, np.inf],
                                 labels=["Low Income","Lower-Middle","Upper-Middle","High Income"])
    sc = MinMaxScaler()
    for c in ["bmi","schooling","immunization","hdi_index","life_expectancy","log_gdp"]:
        df[c+"_n"] = sc.fit_transform(df[[c]])
    return df

df = load_data()


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌍 SDG 3 Dashboard")
    st.markdown("<p style='font-size:0.78rem;color:#64748B;margin-top:-8px'>Global Health & Life Expectancy</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    all_cont  = sorted(df["continent"].dropna().unique())
    sel_cont  = st.multiselect("Continent", all_cont, default=all_cont)
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_year  = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
    dev_stages = df["dev_stage"].cat.categories.tolist()
    sel_stage = st.multiselect("Development stage", dev_stages, default=dev_stages)

    st.markdown("---")
    st.markdown("<p style='font-size:0.7rem;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:0.06em'>Dataset</p>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem'>📊 {df.shape[0]:,} records</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem'>🌐 {df['country'].nunique()} countries</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem'>📅 {yr_min}–{yr_max}</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("⚖️ Ethics & Limitations"):
        st.markdown("""
**Reporting bias** — Developing nations have more missing values. Median imputation may understate severity.

**Ecological fallacy** — Country averages hide urban/rural and gender disparities.

**GDP note** — Gapminder GDP primary; WHO GDP as fallback. Current USD, PPP not adjusted.

**Temporal gap** — Dataset ends 2015. COVID-19 impact not captured.

**Ethical use** — Rankings must be contextualised with historical inequities.
        """)

    st.markdown("---")
    st.markdown("<p style='font-size:0.72rem;color:#475569;text-align:center'>ITS68404 · Group 8 · Taylor's</p>",
                unsafe_allow_html=True)

# Apply filters
mask = (
    df["continent"].isin(sel_cont) &
    df["year"].between(sel_year[0], sel_year[1]) &
    df["dev_stage"].isin(sel_stage)
)
dff = df[mask].copy()
if dff.empty:
    st.warning("No data matches your filters. Please adjust the sidebar.")
    st.stop()


# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("## 🌍 Global Health & Life Expectancy — SDG 3")
st.markdown(
    "<p style='color:#6C757D;font-size:0.85rem;margin-top:-10px'>"
    "ITS68404 Data Visualization · Group 8 &nbsp;|&nbsp; "
    "Stakeholders: WHO · UNICEF · Government Ministries · NGOs · World Bank &nbsp;|&nbsp; "
    "SDG Targets: 3.1 Maternal · 3.2 Child Survival · 3.3 Epidemics · 3.8 UHC"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ══════════════════════════════════════════════════════════
# KPI CARDS
# Principle: Pre-attentive — biggest numbers first (visual hierarchy)
# Principle: Semantic color — teal=good, red=bad, amber=caution
# Principle: Delta shows change over selected period
# ══════════════════════════════════════════════════════════
def kpi_delta(col, good="up"):
    y0 = dff[dff["year"]==sel_year[0]][col].mean()
    y1 = dff[dff["year"]==sel_year[1]][col].mean()
    d  = y1 - y0
    if sel_year[0] == sel_year[1]:
        return "<span style='font-size:0.72rem;color:#9CA3AF'>Single year selected</span>"
    if (good=="up" and d>=0) or (good=="down" and d<=0):
        cls, arrow = "kpi-delta-up", "▲"
    else:
        cls, arrow = "kpi-delta-down", "▼"
    return f"<div class='{cls}'>{arrow} {abs(d):.1f} &nbsp;({sel_year[0]}→{sel_year[1]})</div>"

# SVG icons — stroke-based, single color, scales with card
ICONS = {
    "life_expectancy":  '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>',
    "adult_mortality":  '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
    "immunization":     '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 12 2 2 4-4"/><path d="M12 3a9 9 0 1 0 9 9"/><path d="M15 3h6v6"/></svg>',
    "hdi_index":        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "under_five_deaths":'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "hiv_aids":         '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 2a2 2 0 0 0-2 2v5H4a2 2 0 0 0-2 2v2c0 1.1.9 2 2 2h5v5a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-5h5a2 2 0 0 0 2-2v-2a2 2 0 0 0-2-2h-5V4a2 2 0 0 0-2-2h-2z"/></svg>',
}

kpis = [
    (C_GOOD,    "life_expectancy",    f"{dff['life_expectancy'].mean():.1f} yrs",
     "Life Expectancy", "SDG 3 — core health outcome", "up"),
    (C_BAD,     "adult_mortality",    f"{dff['adult_mortality'].mean():.0f}",
     "Adult Mortality /1K", "Policy priority — ministries", "down"),
    (C_NEUTRAL, "immunization",       f"{dff['immunization'].mean():.1f}%",
     "Immunization Rate", "WHO/UNICEF vaccination KPI", "up"),
    (C_PURPLE,  "hdi_index",          f"{dff['hdi_index'].mean():.3f}",
     "HDI Index", "UNDP — SDG progress composite", "up"),
    (C_BAD,     "under_five_deaths",  f"{dff['under_five_deaths'].mean():.0f}",
     "Under-5 Deaths /1K", "SDG 3.2 — child survival target", "down"),
    (C_WARN,    "hiv_aids",           f"{dff['hiv_aids'].mean():.2f}",
     "HIV/AIDS Rate", "SDG 3.3 — epidemic control", "down"),
]

cols = st.columns(6)
for col_w, (color, col_key, val, lbl, sub, good) in zip(cols, kpis):
    with col_w:
        icon_svg = ICONS.get(col_key, "")
        # icon bg is 15% opacity of the card color
        icon_bg = color + "22"
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{color}">
            <div class="kpi-icon" style="background:{icon_bg};color:{color}">{icon_svg}</div>
            <div class="kpi-val" style="color:{color}">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-sub">{sub}</div>
            {kpi_delta(col_key, good)}
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview & EDA",
    "🌐 Global Trends",
    "🔬 Advanced Analysis",
    "⚖️ Scenario Comparison",
    "🗺️ Geospatial",
    "⚠️ Ethical Bias",
    "📉 Uncertainty",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-hdr">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Variable distributions by continent**")
        var_opts = [
            ("life_expectancy","Life Expectancy (yrs)"),
            ("adult_mortality","Adult Mortality /1K"),
            ("immunization","Immunization Rate (%)"),
            ("schooling","Schooling (yrs)"),
            ("hiv_aids","HIV/AIDS Rate"),
            ("hdi_index","HDI Index"),
            ("gdp_final","GDP per Capita ($)"),
        ]
        var_choice = st.selectbox("Indicator", var_opts,
                                  format_func=lambda x: x[1],
                                  label_visibility="collapsed")
        vcol, vlabel = var_choice

        fig_dist = go.Figure()
        for cont in sel_cont:
            sub = dff[dff["continent"]==cont][vcol].dropna()
            if len(sub) < 3: continue
            fig_dist.add_trace(go.Histogram(
                x=sub, name=cont, nbinsx=25,
                marker_color=CONT_COL.get(cont,"#888"),
                opacity=0.55, marker_line=dict(width=0),
                hovertemplate=f"<b>{cont}</b><br>{vlabel}: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))
        med = float(dff[vcol].median())
        fig_dist.add_vline(x=med, line_dash="dot", line_color=C_MUTED, line_width=1.5,
                           annotation_text=f"Median {med:.1f}",
                           annotation_font=dict(size=10, color=C_MUTED),
                           annotation_position="top right")
        fig_dist.update_layout(
            barmode="overlay", **CHART_DEFAULTS, height=320,
            title=dict(text=vlabel, font=dict(size=13,color=C_TEXT), x=0),
            showlegend=True,
            legend=dict(orientation="h", y=1.15, x=0,
                        font=dict(size=10,color=C_TEXT), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**AXIS_STYLE, title=""),
            yaxis=dict(**AXIS_STYLE, title="Count")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.markdown("**Correlation matrix — lower triangle**")
        cv = ["life_expectancy","adult_mortality","schooling","gdp_final",
              "hiv_aids","immunization","hdi_index","income_composition"]
        cv = [c for c in cv if c in dff.columns]
        corr = dff[cv].corr().round(2)
        labels = [c.replace("_"," ").title() for c in cv]
        z    = [[corr.iloc[i,j] if i>=j else None for j in range(len(cv))] for i in range(len(cv))]
        text = [[f"{corr.iloc[i,j]:.2f}" if i>=j else "" for j in range(len(cv))] for i in range(len(cv))]
        fig_corr = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels, text=text,
            texttemplate="%{text}", textfont=dict(size=9,color=C_TEXT),
            colorscale=[[0,C_BAD],[0.5,C_BG],[1,C_GOOD]],
            zmid=0, zmin=-1, zmax=1,
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(title="r", tickfont=dict(size=9,color=C_MUTED), len=0.7, thickness=12)
        ))
        fig_corr.update_layout(**CHART_DEFAULTS, height=320,
                               xaxis=dict(tickangle=-35, tickfont=dict(size=9,color=C_MUTED), showgrid=False),
                               yaxis=dict(tickfont=dict(size=9,color=C_MUTED), showgrid=False),
                               margin=dict(t=10,b=70,l=120,r=30))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<div class="callout">💡 <b>Key correlations:</b> HDI (r≈0.85) and schooling (r≈0.75) are the strongest positive predictors of life expectancy. HIV/AIDS (r≈−0.58) and adult mortality (r≈−0.97) are the strongest negative predictors.</div>', unsafe_allow_html=True)
    st.markdown("---")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Life expectancy by continent — sorted box plots**")
        cont_order = (dff.groupby("continent")["life_expectancy"].median()
                      .sort_values().index.tolist())
        fig_box = go.Figure()
        for cont in cont_order:
            sub = dff[dff["continent"]==cont]["life_expectancy"].dropna()
            if len(sub) < 3: continue
            fig_box.add_trace(go.Box(
                y=sub, name=cont,
                marker_color=CONT_COL.get(cont,"#888"),
                line=dict(width=1.5),
                fillcolor=CONT_COL.get(cont,"#888"),
                opacity=0.7, boxmean="sd", notched=True,
                hovertemplate=f"<b>{cont}</b><br>Life Exp: %{{y:.1f}} yrs<extra></extra>"
            ))
        fig_box = apply_defaults(fig_box, h=340)
        fig_box.update_layout(
            showlegend=False,
            yaxis=dict(**AXIS_STYLE, title="Life Expectancy (yrs)"),
            xaxis=dict(tickfont=dict(size=11,color=C_TEXT), showgrid=False)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c4:
        st.markdown("**Life expectancy trend over time**")
        trend = dff.groupby(["year","continent"])["life_expectancy"].mean().reset_index()
        fig_trend = go.Figure()
        for cont in sel_cont:
            sub = trend[trend["continent"]==cont]
            if sub.empty: continue
            fig_trend.add_trace(go.Scatter(
                x=sub["year"], y=sub["life_expectancy"],
                mode="lines+markers", name=cont,
                line=dict(color=CONT_COL.get(cont,"#888"), width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{cont}</b><br>%{{x}}: %{{y:.1f}} yrs<extra></extra>"
            ))
        fig_trend = apply_defaults(fig_trend, h=340)
        fig_trend.update_layout(
            xaxis=dict(**AXIS_STYLE, title="Year", dtick=2),
            yaxis=dict(**AXIS_STYLE, title="Life Expectancy (yrs)"),
            legend=dict(orientation="h", y=1.15, x=0,
                        font=dict(size=10,color=C_TEXT), bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    c5, c6 = st.columns([3,2])

    with c5:
        st.markdown("**Preston Curve — wealth vs life expectancy**")
        fig_pres = go.Figure()
        for cont in sel_cont:
            sub = dff[dff["continent"]==cont].dropna(subset=["log_gdp","life_expectancy"])
            if len(sub) < 5: continue
            fig_pres.add_trace(go.Scatter(
                x=sub["log_gdp"], y=sub["life_expectancy"],
                mode="markers", name=cont,
                marker=dict(color=CONT_COL.get(cont,"#888"),
                            size=5, opacity=0.5, line=dict(width=0)),
                customdata=np.stack([sub["country"],sub["year"],sub["gdp_final"].round(0)],axis=1),
                hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>log(GDP): %{x:.2f}<br>Life Exp: %{y:.1f} yrs<br>GDP: $%{customdata[2]:,.0f}<extra></extra>"
            ))
        v2 = dff[["log_gdp","life_expectancy"]].dropna()
        if len(v2) > 10:
            sl,ic,r,*_ = stats.linregress(v2["log_gdp"],v2["life_expectancy"])
            xs = np.linspace(v2["log_gdp"].min(), v2["log_gdp"].max(), 200)
            fig_pres.add_trace(go.Scatter(
                x=xs, y=sl*xs+ic, mode="lines",
                name=f"Global OLS (r={r:.2f})",
                line=dict(color=C_TEXT, width=2, dash="dash"), hoverinfo="skip"
            ))
        # Annotate outliers — explicitly flag anomalies
        for cname, xoff, yoff in [("Sierra Leone",0.3,-3),("Lesotho",0.3,-3)]:
            row = dff[dff["country"]==cname][["log_gdp","life_expectancy"]].mean()
            if not row.empty and not row.isna().any():
                fig_pres.add_annotation(
                    x=float(row["log_gdp"]), y=float(row["life_expectancy"]),
                    text=f"⚠ {cname}", showarrow=True,
                    arrowhead=2, arrowcolor=C_BAD,
                    font=dict(size=10,color=C_BAD), ax=40, ay=-25
                )
        fig_pres = apply_defaults(fig_pres, h=360)
        fig_pres.update_layout(
            xaxis=dict(**AXIS_STYLE, title="log(GDP per Capita)"),
            yaxis=dict(**AXIS_STYLE, title="Life Expectancy (yrs)"),
            legend=dict(orientation="h", y=1.15, x=0,
                        font=dict(size=10,color=C_TEXT), bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_pres, use_container_width=True)
        st.markdown('<div class="callout-warn">⚠️ <b>Anomalies:</b> Sierra Leone & Lesotho show life expectancy far below GDP peers — driven by HIV/AIDS burden and post-conflict fragility. Require targeted SDG 3.3 intervention.</div>', unsafe_allow_html=True)

    with c6:
        st.markdown("**Immunization by development stage**")
        imm_agg = (dff.groupby("dev_stage")["immunization"].mean()
                   .reset_index().sort_values("immunization"))
        stage_colors = {"Low Income":C_BAD,"Lower-Middle":C_WARN,
                        "Upper-Middle":C_NEUTRAL,"High Income":C_GOOD}
        fig_imm = go.Figure(go.Bar(
            x=imm_agg["immunization"].round(1),
            y=imm_agg["dev_stage"], orientation="h",
            marker_color=[stage_colors.get(s,C_NEUTRAL) for s in imm_agg["dev_stage"]],
            text=imm_agg["immunization"].round(1),
            textposition="outside", textfont=dict(size=11,color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>Immunization: %{x:.1f}%<extra></extra>"
        ))
        fig_imm = apply_defaults(fig_imm, h=360)
        fig_imm.update_layout(
            xaxis=dict(**AXIS_STYLE, title="Avg Immunization (%)", range=[0,105]),
            yaxis=dict(tickfont=dict(size=11,color=C_TEXT), showgrid=False),
        )
        st.plotly_chart(fig_imm, use_container_width=True)
        st.markdown('<div class="callout">💡 High-income countries achieve >90% immunization. The gap at Low Income stage is a key UNICEF intervention target.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — GLOBAL TRENDS
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-hdr">Global Health vs Wealth Dynamics</div>', unsafe_allow_html=True)
    st.markdown("**Animated bubble chart — press ▶ to see evolution 2000–2015**")
    st.caption("Bubble size = population · Color = continent · Hover for country details")

    anim_df = (dff.groupby(["country","year","continent"])
               .agg(life_expectancy=("life_expectancy","mean"),
                    log_gdp=("log_gdp","mean"), pop_size=("pop_size","mean"),
                    gdp_final=("gdp_final","mean"), schooling=("schooling","mean"))
               .reset_index().sort_values("year"))

    fig_anim = px.scatter(
        anim_df, x="log_gdp", y="life_expectancy",
        animation_frame="year", animation_group="country",
        size="pop_size", color="continent",
        hover_name="country", color_discrete_map=CONT_COL, size_max=50,
        labels={"log_gdp":"log(GDP per Capita)","life_expectancy":"Life Expectancy (yrs)"},
    )
    fig_anim.update_traces(
        marker=dict(opacity=0.7, line=dict(width=0.5,color="white")),
        hovertemplate="<b>%{hovertext}</b><br>Life Exp: %{y:.1f} yrs<br>log(GDP): %{x:.2f}<extra></extra>"
    )
    fig_anim.update_layout(**CHART_DEFAULTS, height=520,
                           xaxis=dict(**AXIS_STYLE, title="log(GDP per Capita)"),
                           yaxis=dict(**AXIS_STYLE, title="Life Expectancy (yrs)"),
                           legend=dict(x=1.02, y=1, font=dict(size=11,color=C_TEXT),
                                       bgcolor="rgba(0,0,0,0)"),
                           margin=dict(t=30,b=60,l=60,r=40))
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400
    st.plotly_chart(fig_anim, use_container_width=True)

    st.markdown("---")
    st.markdown("**Continent-level metric over time**")
    met_opts = [
        ("life_expectancy","Life Expectancy (yrs)"),
        ("adult_mortality","Adult Mortality /1K"),
        ("immunization","Immunization Rate (%)"),
        ("hdi_index","HDI Index"),
        ("schooling","Schooling (yrs)"),
        ("hiv_aids","HIV/AIDS Rate"),
    ]
    met_choice = st.selectbox("Metric", met_opts,
                              format_func=lambda x: x[1], key="tab2_met")
    mcol, mlabel = met_choice
    time_agg = dff.groupby(["year","continent"])[mcol].mean().reset_index()
    fig_t2 = go.Figure()
    for cont in sel_cont:
        sub = time_agg[time_agg["continent"]==cont]
        if sub.empty: continue
        fig_t2.add_trace(go.Scatter(
            x=sub["year"], y=sub[mcol].round(2),
            mode="lines+markers", name=cont,
            line=dict(color=CONT_COL.get(cont,"#888"), width=2.5),
            marker=dict(size=6),
            hovertemplate=f"<b>{cont}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>"
        ))
    fig_t2 = apply_defaults(fig_t2, h=380)
    fig_t2.update_layout(
        xaxis=dict(**AXIS_STYLE, title="Year", dtick=2),
        yaxis=dict(**AXIS_STYLE, title=mlabel),
        legend=dict(orientation="h", y=1.1, x=0,
                    font=dict(size=11,color=C_TEXT), bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_t2, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — ADVANCED ANALYSIS
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-hdr">Advanced Visualization & Insight</div>', unsafe_allow_html=True)
    adv1, adv2, adv3 = st.tabs(["🌐 3D Scatter","📐 Parallel Coordinates","🎯 Multi-Layer"])

    with adv1:
        st.markdown("**GDP × Schooling × Life Expectancy — 4th variable: HIV/AIDS burden (color)**")
        sc3 = dff.dropna(subset=["log_gdp","schooling","life_expectancy"]).copy()
        ck  = (sc3.groupby("continent")["country"].unique()
                  .apply(lambda x: np.random.choice(x, min(len(x),12), replace=False)))
        sc3 = sc3[sc3["country"].isin(np.concatenate(ck.values))]
        fig_3d = go.Figure()
        conts3 = sorted(sc3["continent"].dropna().unique())
        for cont in conts3:
            sub = sc3[sc3["continent"]==cont]
            fig_3d.add_trace(go.Scatter3d(
                x=sub["log_gdp"], y=sub["schooling"], z=sub["life_expectancy"],
                mode="markers", name=cont,
                marker=dict(size=4, opacity=0.8,
                            color=sub["hiv_aids"],
                            colorscale=[[0,C_GOOD],[0.5,C_WARN],[1,C_BAD]],
                            cmin=sc3["hiv_aids"].min(), cmax=sc3["hiv_aids"].max(),
                            showscale=(cont==conts3[-1]),
                            colorbar=dict(title="HIV/AIDS",x=1.05,thickness=12,len=0.6)),
                customdata=np.stack([sub["country"],sub["year"],sub["hiv_aids"].round(2)],axis=1),
                hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>log(GDP): %{x:.2f}<br>Schooling: %{y:.1f} yrs<br>Life Exp: %{z:.1f} yrs<br>HIV/AIDS: %{customdata[2]}<extra>"+cont+"</extra>"
            ))
        fig_3d.update_layout(
            height=560, paper_bgcolor=C_SURF,
            title=dict(text="3D: GDP, Education & Life Expectancy (HIV rate = color)",
                       font=dict(size=14,color=C_TEXT),x=0),
            scene=dict(bgcolor=C_SURF,
                       xaxis=dict(title="log(GDP)",gridcolor=C_GRID),
                       yaxis=dict(title="Schooling (yrs)",gridcolor=C_GRID),
                       zaxis=dict(title="Life Exp (yrs)",gridcolor=C_GRID),
                       camera=dict(eye=dict(x=1.6,y=1.6,z=0.9))),
            legend=dict(font=dict(size=10,color=C_TEXT),bgcolor="rgba(0,0,0,0)",x=0.01,y=0.99),
            margin=dict(t=50,b=20,l=20,r=40)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('<div class="callout">💡 Countries with high GDP AND schooling cluster in the top-right with long life expectancy. African nations (high HIV/AIDS = red) occupy the low-GDP, low-schooling corner — showing compounding disadvantage where multiple deprivations coexist.</div>', unsafe_allow_html=True)

    with adv2:
        st.markdown("**Multi-dimensional health profile by income level**")
        st.caption("Drag axis labels to reorder · Drag on axes to filter country groups")
        pc_vars = ["life_expectancy","schooling","log_gdp","immunization","adult_mortality","hdi_index","hiv_aids"]
        pc = dff[pc_vars+["dev_stage"]].dropna().copy()
        pc["stage_num"] = pd.Categorical(pc["dev_stage"],
                                          categories=["Low Income","Lower-Middle","Upper-Middle","High Income"],
                                          ordered=True).codes
        sc2 = MinMaxScaler()
        pcs = pc.copy()
        pcs[pc_vars] = sc2.fit_transform(pc[pc_vars])
        dims = [dict(label=v.replace("_"," ").title(), values=pcs[v],
                     range=[0,1], tickvals=[0,0.5,1], ticktext=["Low","Mid","High"])
                for v in pc_vars]
        fig_pc = go.Figure(go.Parcoords(
            line=dict(color=pcs["stage_num"],
                      colorscale=[[0,C_BAD],[0.33,C_WARN],[0.66,C_NEUTRAL],[1,C_GOOD]],
                      cmin=0, cmax=3, showscale=True,
                      colorbar=dict(title=dict(text="Income Stage",font=dict(size=11,color=C_TEXT)),
                                    tickvals=[0.4,1.1,1.9,2.6],
                                    ticktext=["Low","Lower-Mid","Upper-Mid","High"],
                                    thickness=12, len=0.5)),
            dimensions=dims,
            labelfont=dict(size=11,color=C_TEXT),
            tickfont=dict(size=9,color=C_MUTED),
        ))
        fig_pc.update_layout(height=480, paper_bgcolor=C_SURF, plot_bgcolor=C_SURF,
                             title=dict(text="Multi-dimensional health profile by income level",
                                        font=dict(size=14,color=C_TEXT),x=0),
                             margin=dict(t=60,b=60,l=90,r=130))
        st.plotly_chart(fig_pc, use_container_width=True)
        st.markdown('<div class="callout">💡 High-income countries (teal) consistently score high on life expectancy, schooling, and immunization while maintaining low adult mortality and HIV/AIDS rates. Low-income (red) show the exact inverse profile.</div>', unsafe_allow_html=True)

    with adv3:
        st.markdown("**Education × Mortality × Life Expectancy — multi-layer composite**")
        ml = dff.dropna(subset=["schooling","adult_mortality","life_expectancy","continent"]).copy()
        fig_ml = make_subplots(rows=1, cols=2, column_widths=[0.68,0.32],
                               horizontal_spacing=0.05)
        conts_ml = sorted(ml["continent"].dropna().unique())
        for cont in conts_ml:
            sub = ml[ml["continent"]==cont]
            fig_ml.add_trace(go.Scatter(
                x=sub["schooling"], y=sub["adult_mortality"],
                mode="markers", name=cont, legendgroup=cont,
                marker=dict(color=sub["life_expectancy"], colorscale="RdYlGn",
                            size=5, opacity=0.6, line=dict(width=0),
                            showscale=(cont==conts_ml[-1]),
                            colorbar=dict(title="Life Exp",x=0.72,thickness=10,len=0.6)),
                customdata=np.stack([sub["country"],sub["year"],sub["life_expectancy"].round(1)],axis=1),
                hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Schooling: %{x:.1f} yrs<br>Mortality: %{y:.0f}<br>Life Exp: %{customdata[2]} yrs<extra></extra>"
            ), row=1, col=1)
        v3 = ml[["schooling","adult_mortality"]].dropna()
        sl,ic,r,*_ = stats.linregress(v3["schooling"],v3["adult_mortality"])
        xs = np.linspace(v3["schooling"].min(),v3["schooling"].max(),200)
        fig_ml.add_trace(go.Scatter(x=xs,y=sl*xs+ic,mode="lines",
                                    name=f"OLS (r={r:.2f})",
                                    line=dict(color=C_TEXT,width=2,dash="dash"),
                                    hoverinfo="skip"), row=1, col=1)
        cont_ord = (ml.groupby("continent")["life_expectancy"].median()
                    .sort_values().index.tolist())
        for cont in cont_ord:
            sub = ml[ml["continent"]==cont]["life_expectancy"].dropna()
            fig_ml.add_trace(go.Box(x=sub, name=cont, legendgroup=cont,
                                    orientation="h", showlegend=False,
                                    marker_color=CONT_COL.get(cont,"#888"),
                                    opacity=0.75, boxmean=True, notched=True,
                                    line=dict(width=1.5)), row=1, col=2)
        fig_ml.update_layout(**CHART_DEFAULTS, height=460, showlegend=False,
                             title=dict(text="Schooling vs mortality (color=life expectancy) + continental distribution",
                                        font=dict(size=13,color=C_TEXT),x=0))
        fig_ml.update_xaxes(gridcolor=C_GRID, tickfont=dict(size=10,color=C_MUTED))
        fig_ml.update_yaxes(gridcolor=C_GRID, tickfont=dict(size=10,color=C_MUTED))
        fig_ml.update_yaxes(tickfont=dict(size=10,color=C_TEXT), showgrid=False, row=1, col=2)
        st.plotly_chart(fig_ml, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-hdr">Scenario Comparison — Top 10 vs Bottom 10 Countries</div>',
                unsafe_allow_html=True)

    cmp_opts = [
        ("life_expectancy","Life Expectancy"),
        ("hdi_index","HDI Index"),
        ("immunization","Immunization Rate"),
        ("schooling","Schooling"),
        ("adult_mortality","Adult Mortality"),
    ]
    cmp_met = st.selectbox("Rank countries by", cmp_opts,
                           format_func=lambda x: x[1])
    cmp_col, cmp_lbl = cmp_met

    latest_yr = dff["year"].max()
    latest = (dff[dff["year"]==latest_yr].groupby("country")
              .agg(life_expectancy=("life_expectancy","mean"),
                   hdi_index=("hdi_index","mean"),
                   immunization=("immunization","mean"),
                   schooling=("schooling","mean"),
                   adult_mortality=("adult_mortality","mean"),
                   gdp_final=("gdp_final","mean"),
                   hiv_aids=("hiv_aids","mean"),
                   continent=("continent","first"))
              .reset_index().dropna(subset=[cmp_col]))

    top10    = latest.nlargest(10, cmp_col)
    bottom10 = latest.nsmallest(10, cmp_col)

    # Radar
    rv = ["life_expectancy_n","schooling_n","immunization_n","hdi_index_n","log_gdp_n"]
    rl = ["Life Exp","Schooling","Immunization","HDI","Log GDP"]
    ta = df[df["country"].isin(top10["country"])].groupby("country")[rv].mean().mean()
    ba = df[df["country"].isin(bottom10["country"])].groupby("country")[rv].mean().mean()
    fig_rad = go.Figure()
    for grp, vals, col in [("Top 10", ta.tolist()+[ta.iloc[0]], C_NEUTRAL),
                            ("Bottom 10", ba.tolist()+[ba.iloc[0]], C_BAD)]:
        fig_rad.add_trace(go.Scatterpolar(
            r=vals, theta=rl+[rl[0]], fill="toself", name=grp,
            line_color=col, fillcolor=col, opacity=0.25,
            hovertemplate=f"<b>{grp}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"
        ))
    fig_rad.update_layout(
        height=360, paper_bgcolor=C_SURF,
        polar=dict(bgcolor=C_SURF,
                   radialaxis=dict(visible=True,range=[0,1],
                                   tickfont=dict(color=C_MUTED),gridcolor=C_GRID),
                   angularaxis=dict(tickfont=dict(size=12,color=C_TEXT),gridcolor=C_GRID)),
        title=dict(text=f"Normalized health profile — Top 10 vs Bottom 10 by {cmp_lbl}",
                   font=dict(size=13,color=C_TEXT),x=0),
        legend=dict(font=dict(color=C_TEXT),bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50,b=30,l=50,r=50)
    )
    st.plotly_chart(fig_rad, use_container_width=True)

    col_t, col_b = st.columns(2)
    with col_t:
        ts = top10.sort_values(cmp_col)
        fig_top = go.Figure(go.Bar(
            y=ts["country"], x=ts[cmp_col].round(1), orientation="h",
            marker_color=C_NEUTRAL,
            text=ts[cmp_col].round(1), textposition="outside",
            textfont=dict(size=10,color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>"+cmp_lbl+": %{x:.2f}<extra></extra>"
        ))
        fig_top = apply_defaults(fig_top, h=340)
        fig_top.update_layout(
            title=dict(text=f"Top 10 — {cmp_lbl}", font=dict(size=13,color=C_TEXT),x=0),
            xaxis=dict(**AXIS_STYLE, title=cmp_lbl),
            yaxis=dict(tickfont=dict(size=10,color=C_TEXT), showgrid=False),
            margin=dict(t=50,b=30,l=140,r=60)
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        bs = bottom10.sort_values(cmp_col)
        fig_bot = go.Figure(go.Bar(
            y=bs["country"], x=bs[cmp_col].round(1), orientation="h",
            marker_color=C_BAD,
            text=bs[cmp_col].round(1), textposition="outside",
            textfont=dict(size=10,color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>"+cmp_lbl+": %{x:.2f}<extra></extra>"
        ))
        fig_bot = apply_defaults(fig_bot, h=340)
        fig_bot.update_layout(
            title=dict(text=f"Bottom 10 — {cmp_lbl}", font=dict(size=13,color=C_TEXT),x=0),
            xaxis=dict(**AXIS_STYLE, title=cmp_lbl),
            yaxis=dict(tickfont=dict(size=10,color=C_TEXT), showgrid=False),
            margin=dict(t=50,b=30,l=170,r=60)
        )
        st.plotly_chart(fig_bot, use_container_width=True)

    # Diverging gap bar
    st.markdown("**Health indicator gap: Top 10 minus Bottom 10**")
    gvars  = ["life_expectancy","schooling","immunization","hdi_index","adult_mortality","hiv_aids","gdp_final"]
    glbls  = ["Life Expectancy","Schooling","Immunization","HDI Index","Adult Mortality","HIV/AIDS","GDP/Capita"]
    tm = df[df["country"].isin(top10["country"])][gvars].mean()
    bm = df[df["country"].isin(bottom10["country"])][gvars].mean()
    gv = tm - bm
    fig_gap = go.Figure(go.Bar(
        x=glbls, y=gv.values.round(2),
        marker_color=[C_GOOD if g>=0 else C_BAD for g in gv],
        text=[f"{v:+.1f}" for v in gv.values],
        textposition="outside", textfont=dict(size=11,color=C_TEXT),
        hovertemplate="<b>%{x}</b><br>Gap (Top−Bottom): %{y:+.2f}<extra></extra>"
    ))
    fig_gap.add_hline(y=0, line_color=C_TEXT, line_width=1)
    fig_gap = apply_defaults(fig_gap, h=340)
    fig_gap.update_layout(
        title=dict(text="Mean difference: Top 10 − Bottom 10 countries",
                   font=dict(size=13,color=C_TEXT),x=0),
        xaxis=dict(tickfont=dict(size=11,color=C_TEXT), showgrid=False),
        yaxis=dict(**AXIS_STYLE, title="Difference (Top 10 − Bottom 10)")
    )
    st.plotly_chart(fig_gap, use_container_width=True)
    st.markdown('<div class="callout">💡 Top 10 countries enjoy ~25 more years of life expectancy, 9 more years of schooling, and over $30,000 higher GDP per capita. These compounding inequalities demand targeted SDG 3 funding allocation in Sub-Saharan Africa.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 5 — GEOSPATIAL
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-hdr">Geospatial & Faceted Analysis</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns([2,1])
    with mc1:
        map_opts = [
            ("life_expectancy","Life Expectancy","Viridis"),
            ("adult_mortality","Adult Mortality","Reds"),
            ("hdi_index","HDI Index","Blues"),
            ("immunization","Immunization Rate","Greens"),
            ("hiv_aids","HIV/AIDS Rate","OrRd"),
            ("schooling","Avg Schooling (yrs)","Purples"),
        ]
        map_met = st.selectbox("Map indicator", map_opts,
                               format_func=lambda x: x[1], key="map_met")
    with mc2:
        map_yr = st.slider("Year", yr_min, yr_max, yr_max, key="map_yr")

    mdf = df[df["year"]==map_yr].groupby("country")[map_met[0]].mean().reset_index()
    fig_map = px.choropleth(
        mdf, locations="country", locationmode="country names",
        color=map_met[0], hover_name="country",
        color_continuous_scale=map_met[2],
        labels={map_met[0]: map_met[1]},
    )
    fig_map.update_layout(
        height=440, paper_bgcolor=C_SURF,
        title=dict(text=f"{map_met[1]} by country ({map_yr})",
                   font=dict(size=14,color=C_TEXT),x=0),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor=C_BORDER,
                 showland=True, landcolor="#F1F5F9",
                 showocean=True, oceancolor="#DBEAFE"),
        margin=dict(t=50,b=0,l=0,r=0),
        coloraxis_colorbar=dict(tickfont=dict(color=C_MUTED),
                                title_font=dict(color=C_TEXT))
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.markdown("**Faceted small multiples — life expectancy by continent & development status**")
    st.caption("Tufte principle: small multiples share the same scale for fair comparison")

    fdf = dff.groupby(["year","continent","status"])["life_expectancy"].mean().reset_index()
    fig_fac = px.line(
        fdf, x="year", y="life_expectancy",
        color="status", facet_col="continent", facet_col_wrap=3,
        color_discrete_map={"Developed": C_NEUTRAL, "Developing": C_WARN},
        markers=True,
        labels={"life_expectancy":"Life Expectancy (yrs)","year":"Year"},
    )
    fig_fac.update_layout(**CHART_DEFAULTS, height=460,
                          title=dict(text="Life expectancy trends — developed vs developing by continent",
                                     font=dict(size=14,color=C_TEXT),x=0),
                          legend=dict(orientation="h", y=1.08, x=0,
                                      font=dict(size=11,color=C_TEXT),
                                      bgcolor="rgba(0,0,0,0)"),
                          margin=dict(t=80,b=50,l=60,r=20))
    fig_fac.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1], font=dict(size=12,color=C_TEXT)))
    for ax in fig_fac.layout:
        if "xaxis" in ax: fig_fac.layout[ax].update(gridcolor=C_GRID, tickfont=dict(color=C_MUTED))
        if "yaxis" in ax: fig_fac.layout[ax].update(gridcolor=C_GRID, tickfont=dict(color=C_MUTED))
    st.plotly_chart(fig_fac, use_container_width=True)
    st.markdown('<div class="callout">💡 Africa shows the steepest improvement (+8 yrs, 2000–2015). Europe remains flat near its ceiling. The developed/developing gap is largest in Africa and Asia — targets for SDG 3.8 Universal Health Coverage.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 6 — ETHICAL BIAS VISUALIZATION
# ══════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-hdr">Ethical Bias — Data Completeness & Reporting Gaps</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="callout-warn">
    ⚠️ <b>Why this matters:</b> Under-reporting is highest in Africa and Oceania — the very regions
    with the worst health outcomes. Visualizing WHERE data is missing is an ethical obligation,
    not just a technical note. Median imputation masks these gaps, so results for under-reported
    regions must be interpreted with caution.
    </div>
    """, unsafe_allow_html=True)

    # Load RAW (pre-imputed) data to measure true missing rates
    @st.cache_data(show_spinner="Calculating missing data rates…")
    def load_raw_missing():
        import os
        BASE = os.path.dirname(os.path.abspath(__file__))
        who_raw = pd.read_csv(os.path.join(BASE, "Life Expectancy Data.csv"))
        who_raw.columns = [c.strip().lower().replace(" ","_").replace("/","_") for c in who_raw.columns]
        continent_map = df[["country","continent"]].drop_duplicates().set_index("country")["continent"]
        who_raw["continent"] = who_raw["country"].map(continent_map)
        key_cols = [c for c in who_raw.columns if any(k in c for k in
                    ["life","adult","hepatit","bmi","hiv","gdp","population","income","school"])]
        miss_cont = (
            who_raw.groupby("continent")[key_cols]
            .apply(lambda x: x.isnull().mean() * 100)
            .mean(axis=1).reset_index()
        )
        miss_cont.columns = ["continent","missing_pct"]
        heat_cols = key_cols[:8]
        heat_data = (
            who_raw.groupby("continent")[heat_cols]
            .apply(lambda x: x.isnull().mean() * 100)
            .round(1)
        )
        heat_labels = [c.replace("_"," ").title()[:16] for c in heat_cols]
        return miss_cont.dropna(), heat_data, heat_labels

    miss_cont, heat_data, heat_labels = load_raw_missing()

    b1, b2 = st.columns([1, 1.6])

    with b1:
        st.markdown("**Average missing data % by continent**")
        ms = miss_cont.sort_values("missing_pct")
        fig_bias = go.Figure(go.Bar(
            x=ms["missing_pct"].round(1),
            y=ms["continent"],
            orientation="h",
            marker_color=[CONT_COL.get(c, "#888") for c in ms["continent"]],
            marker_line=dict(width=0),
            text=ms["missing_pct"].round(1).astype(str) + "%",
            textposition="outside",
            textfont=dict(size=11, color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>"
        ))
        fig_bias = apply_defaults(fig_bias, h=320,
                                  margin=dict(t=30, b=40, l=120, r=60))
        fig_bias.update_xaxes(**AXIS_STYLE, title="Avg missing data (%)", range=[0, 40])
        fig_bias.update_yaxes(tickfont=dict(size=11, color=C_TEXT), showgrid=False)
        st.plotly_chart(fig_bias, use_container_width=True)

    with b2:
        st.markdown("**Missing data heatmap — variable × continent**")
        fig_heat = go.Figure(go.Heatmap(
            z=heat_data.values,
            x=heat_labels,
            y=heat_data.index.tolist(),
            colorscale=[[0,"#F0FDF4"],[0.3,"#FEF9C3"],[0.6,"#FED7AA"],[1.0,C_BAD]],
            zmin=0, zmax=50,
            text=heat_data.values.round(1),
            texttemplate="%{text}",
            textfont=dict(size=9, color=C_TEXT),
            hovertemplate="<b>%{y}</b> — %{x}<br>Missing: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Missing %", title_font=dict(size=11, color=C_TEXT),
                          tickfont=dict(size=9, color=C_MUTED), len=0.7, thickness=12)
        ))
        fig_heat.update_layout(**CHART_DEFAULTS, height=320,
                               margin=dict(t=30, b=80, l=110, r=80),
                               xaxis=dict(tickangle=-30, tickfont=dict(size=9, color=C_MUTED),
                                          showgrid=False),
                               yaxis=dict(tickfont=dict(color=C_TEXT), showgrid=False))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("**Missing data % by development stage — who is silenced?**")

    @st.cache_data(show_spinner=False)
    def load_stage_missing():
        import os
        BASE = os.path.dirname(os.path.abspath(__file__))
        who_raw = pd.read_csv(os.path.join(BASE, "Life Expectancy Data.csv"))
        who_raw.columns = [c.strip().lower().replace(" ","_").replace("/","_") for c in who_raw.columns]
        # Use plain dict — avoids any Series index uniqueness issues entirely
        stage_dict = (df[["country","dev_stage"]]
                      .drop_duplicates(subset="country")
                      .set_index("country")["dev_stage"]
                      .astype(str).to_dict())
        who_raw["dev_stage"] = who_raw["country"].map(stage_dict)
        key_cols = [c for c in who_raw.columns if any(k in c for k in
                    ["life","adult","hepatit","bmi","hiv","gdp","population","income","school"])]
        rows = []
        for stage in ["Low Income","Lower-Middle","Upper-Middle","High Income"]:
            sub = who_raw[who_raw["dev_stage"]==stage][key_cols]
            if sub.empty: continue
            rows.append({"dev_stage": stage,
                         "missing_pct": sub.isnull().mean().mean() * 100})
        return pd.DataFrame(rows)

    miss_stage = load_stage_missing()
    stage_order = ["Low Income","Lower-Middle","Upper-Middle","High Income"]
    miss_stage["dev_stage"] = pd.Categorical(miss_stage["dev_stage"],
                                              categories=stage_order, ordered=True)
    miss_stage = miss_stage.sort_values("dev_stage")

    stage_cols = {"Low Income": C_BAD, "Lower-Middle": C_WARN,
                  "Upper-Middle": C_NEUTRAL, "High Income": C_GOOD}

    fig_stage = go.Figure(go.Bar(
        x=miss_stage["dev_stage"].astype(str),
        y=miss_stage["missing_pct"].round(1),
        marker_color=[stage_cols.get(str(s), C_NEUTRAL) for s in miss_stage["dev_stage"]],
        marker_line=dict(width=0),
        text=miss_stage["missing_pct"].round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(size=12, color=C_TEXT),
        hovertemplate="<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>"
    ))
    fig_stage = apply_defaults(fig_stage, h=320, margin=dict(t=30, b=60, l=60, r=30))
    fig_stage.update_xaxes(tickfont=dict(size=12, color=C_TEXT), showgrid=False)
    fig_stage.update_yaxes(**AXIS_STYLE, title="Avg missing data (%)")
    st.plotly_chart(fig_stage, use_container_width=True)

    st.markdown("""
    <div class="callout-warn">
    ⚠️ <b>Ethical finding:</b> Low-income countries have the highest missing data rates — meaning
    the nations most in need of health interventions are also the least visible in the data.
    This creates a systematic bias: analyses will underestimate health burdens in poorer regions
    and overstate global progress. Policymakers must account for these reporting gaps when
    allocating SDG 3 resources.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 7 — UNCERTAINTY VISUALIZATION
# ══════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="sec-hdr">Uncertainty Visualization — Confidence Intervals & Statistical Spread</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="callout">
    💡 <b>Why uncertainty matters:</b> Showing only mean trends can mislead policymakers.
    The 95% CI ribbon shows how confident we are in each estimate — wider ribbons mean
    fewer data points and less reliable conclusions for that region.
    </div>
    """, unsafe_allow_html=True)

    def hex_rgba(h, a):
        return f"rgba({int(h[1:3],16)},{int(h[3:5],16)},{int(h[5:7],16)},{a})"

    unc_met = st.selectbox("Select indicator", [
        ("life_expectancy","Life Expectancy (yrs)"),
        ("adult_mortality","Adult Mortality /1K"),
        ("immunization","Immunization Rate (%)"),
        ("hdi_index","HDI Index"),
        ("schooling","Schooling (yrs)"),
    ], format_func=lambda x: x[1], key="unc_met")
    ucol, ulabel = unc_met

    trend = (
        dff.groupby(["year","continent"])[ucol]
        .agg(["mean","std","count"])
        .reset_index()
    )
    trend.columns = ["year","continent","mean","std","n"]
    trend["se"]   = trend["std"] / np.sqrt(trend["n"].clip(lower=1))
    trend["ci95"] = trend["se"] * 1.96

    fig_unc = go.Figure()
    for cont in sel_cont:
        sub = trend[trend["continent"]==cont].sort_values("year")
        if sub.empty: continue
        col = CONT_COL.get(cont, "#888")
        xs  = sub["year"].tolist()
        hi2 = (sub["mean"] + sub["std"]).tolist()
        lo2 = (sub["mean"] - sub["std"]).tolist()
        hi  = (sub["mean"] + sub["ci95"]).tolist()
        lo  = (sub["mean"] - sub["ci95"]).tolist()

        # ±1 SD ribbon
        fig_unc.add_trace(go.Scatter(
            x=xs+xs[::-1], y=hi2+lo2[::-1],
            fill="toself", fillcolor=hex_rgba(col, 0.07),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, legendgroup=cont, hoverinfo="skip",
            name=f"{cont} ±1 SD"
        ))
        # 95% CI ribbon
        fig_unc.add_trace(go.Scatter(
            x=xs+xs[::-1], y=hi+lo[::-1],
            fill="toself", fillcolor=hex_rgba(col, 0.18),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, legendgroup=cont, hoverinfo="skip",
            name=f"{cont} 95% CI"
        ))
        # Mean line
        fig_unc.add_trace(go.Scatter(
            x=xs, y=sub["mean"].round(2),
            mode="lines+markers", name=cont, legendgroup=cont,
            line=dict(color=col, width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{cont}</b><br>Year: %{{x}}<br>Mean: %{{y:.2f}}<br>±95% CI: %{{customdata:.2f}}",
            customdata=sub["ci95"].round(2)
        ))

    fig_unc.add_annotation(
        x=0.98, y=0.04, xref="paper", yref="paper",
        text="Dark band = 95% CI · Light band = ±1 SD",
        showarrow=False, font=dict(size=11, color=C_MUTED),
        bgcolor=C_SURF, borderpad=4, xanchor="right"
    )

    fig_unc = apply_defaults(fig_unc, h=460, margin=dict(t=40, b=60, l=65, r=140))
    fig_unc.update_layout(
        xaxis=dict(**AXIS_STYLE, title="Year", dtick=2),
        yaxis=dict(**AXIS_STYLE, title=ulabel),
        legend=dict(orientation="v", x=1.02, y=1,
                    font=dict(size=11, color=C_TEXT), bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_unc, use_container_width=True)

    # CI width comparison bar
    st.markdown("**Which region has the most uncertain estimates? (2015, 95% CI width)**")
    ci_rows = []
    yr_data = dff[dff["year"]==dff["year"].max()]
    for cont in sel_cont:
        sub = yr_data[yr_data["continent"]==cont][ucol].dropna()
        if len(sub) < 3: continue
        ci = sub.sem() * 1.96
        ci_rows.append({"continent": cont, "ci_width": ci*2,
                        "n": len(sub), "mean": sub.mean()})
    ci_df = pd.DataFrame(ci_rows).sort_values("ci_width", ascending=False)

    if not ci_df.empty:
        fig_ci = go.Figure(go.Bar(
            x=ci_df["continent"], y=ci_df["ci_width"].round(2),
            marker_color=[CONT_COL.get(c, "#888") for c in ci_df["continent"]],
            marker_line=dict(width=0),
            text=ci_df["ci_width"].round(2), textposition="outside",
            textfont=dict(size=11, color=C_TEXT),
            hovertemplate="<b>%{x}</b><br>95% CI width: ±%{y:.2f}<br>n=%{customdata}",
            customdata=ci_df["n"]
        ))
        fig_ci = apply_defaults(fig_ci, h=300, margin=dict(t=30, b=60, l=60, r=20))
        fig_ci.update_xaxes(tickfont=dict(size=11, color=C_TEXT), showgrid=False)
        fig_ci.update_yaxes(**AXIS_STYLE, title=f"95% CI full width ({ulabel})")
        st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown("""
    <div class="callout">
    💡 <b>Uncertainty insight:</b> Africa and Oceania have the widest confidence intervals —
    meaning our estimates for these regions are the least statistically reliable, directly
    linked to the data gaps shown in the Ethical Bias tab. Wider CI = fewer reporting countries
    = less trustworthy regional averages. Policy decisions for these regions must account for
    this uncertainty.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9CA3AF;font-size:0.72rem;padding:8px 0'>"
    "ITS68404 – Data Visualization · Group 8 · January 2026 · Taylor's University &nbsp;|&nbsp; "
    "SDG 3: Good Health and Well-Being &nbsp;|&nbsp; "
    "WHO Life Expectancy + Gapminder (2000–2015) · 2,416 records · 151 countries"
    "</div>",
    unsafe_allow_html=True
)