# 🌍 Global Health & Life Expectancy — SDG 3 Dashboard

**ITS68404 – Data Visualization | Group 8 | Taylor's University | January 2026**

> An interactive Streamlit dashboard and Jupyter notebook analyzing global health indicators across 151 countries (2000–2015), aligned to **UN Sustainable Development Goal 3: Good Health and Well-Being**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Setup on a New PC](#setup-on-a-new-pc)
- [Project Structure](#project-structure)
- [Dashboard Features](#dashboard-features)
- [Datasets](#datasets)
- [Running the Notebook](#running-the-notebook)
- [Group Members](#group-members)

---

## 📌 Project Overview

This project addresses the research question:

> *How do socioeconomic and healthcare factors influence life expectancy across nations, and what actionable insights can guide SDG 3 policy decisions for 2030?*

### SDG Targets Covered

| Target | Description | Indicator |
|--------|-------------|-----------|
| SDG 3.1 | Maternal health | Adult mortality rate |
| SDG 3.2 | Child survival | Under-5 deaths |
| SDG 3.3 | Epidemic control | HIV/AIDS rate |
| SDG 3.8 | Universal Health Coverage | Immunization rate |

### Key Findings

- Top 10 countries have **~25 years** more life expectancy than Bottom 10
- **HDI** is the strongest predictor of life expectancy (r ≈ 0.85)
- **Africa** shows the steepest improvement (+8 yrs, 2000–2015)
- **Sierra Leone & Lesotho** are statistical outliers — HIV/AIDS burden drives extreme under-performance relative to GDP peers
- Low-income countries have up to **35% missing data** — systematic under-reporting in the most vulnerable regions

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/ITS68404-Group8-SDG3-Dashboard.git
cd ITS68404-Group8-SDG3-Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run dashboard.py
```

Dashboard opens automatically at **http://localhost:8501**

---

## 🖥️ Setup on a New PC

Follow these steps carefully if setting up on a new machine.

### Prerequisites

- Python **3.9 or higher** — download from [python.org](https://www.python.org/downloads/)
- Git — download from [git-scm.com](https://git-scm.com/downloads)

> ⚠️ During Python installation on Windows, tick **"Add Python to PATH"**

---

### Step 1 — Clone the repository

Open **Command Prompt** or **PowerShell** and run:

```bash
git clone https://github.com/YourUsername/ITS68404-Group8-SDG3-Dashboard.git
cd ITS68404-Group8-SDG3-Dashboard
```

---

### Step 2 — Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs everything needed — Streamlit, Plotly, Pandas, NumPy, scikit-learn, and SciPy.

If you don't have a `requirements.txt`, install manually:

```bash
pip install streamlit plotly pandas numpy scikit-learn scipy
```

---

### Step 4 — Verify your folder structure

Make sure all these files are present:

```
📁 ITS68404-Group8-SDG3-Dashboard/
  ├── dashboard.py                          ← Streamlit app
  ├── ITS68404_Group8_GROUPASGNMT_FINAL.ipynb  ← Jupyter notebook
  ├── gapminder_data_graphs.csv             ← Dataset 1
  ├── Life_Expectancy_Data.csv              ← Dataset 2
  ├── requirements.txt                      ← Dependencies
  └── README.md                             ← This file
```

> ⚠️ Both CSV files **must be in the same folder** as `dashboard.py`

---

### Step 5 — Run the dashboard

```bash
streamlit run dashboard.py
```

Your browser will open automatically at `http://localhost:8501`

If it does not open automatically, copy and paste that URL into your browser.

---

### Step 6 — Run the Jupyter Notebook (optional)

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch notebook
jupyter notebook ITS68404_Group8_GROUPASGNMT_FINAL.ipynb
```

Or open it directly in **VS Code** with the Jupyter extension installed.

---

### Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install streamlit` |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| `ModuleNotFoundError: No module named 'plotly'` | Run `pip install plotly` |
| `FileNotFoundError: Life_Expectancy_Data.csv` | Make sure CSV files are in the same folder as `dashboard.py` |
| `streamlit: command not found` | Run `python -m streamlit run dashboard.py` instead |
| Port 8501 already in use | Run `streamlit run dashboard.py --server.port 8502` |

---

## 📁 Project Structure

```
ITS68404-Group8-SDG3-Dashboard/
│
├── dashboard.py                          # Main Streamlit dashboard
│   ├── Tab 1: Overview & EDA             # 6+ chart types, distributions, correlations
│   ├── Tab 2: Global Trends              # Animated Gapminder bubble chart
│   ├── Tab 3: Advanced Analysis          # 3D scatter, parallel coordinates, multi-layer
│   ├── Tab 4: Scenario Comparison        # Top 10 vs Bottom 10 countries
│   ├── Tab 5: Geospatial                 # Choropleth map + faceted small multiples
│   ├── Tab 6: Ethical Bias               # Missing data heatmap + bias visualization
│   └── Tab 7: Uncertainty                # 95% CI ribbons + uncertainty comparison
│
├── ITS68404_Group8_GROUPASGNMT_FINAL.ipynb
│   ├── Task 1: Problem Identification    # Stakeholders, SDG alignment
│   ├── Task 2: Data Engineering          # Loading, cleaning, outlier detection, features
│   ├── Task 3: EDA (6 viz types)         # Histograms, heatmap, Preston curve, violin, etc.
│   ├── Task 4: Advanced (7/7 criteria)   # 3D, parallel coords, scenario, bias, uncertainty
│   ├── Task 5: Reflection & Ethics       # Individual reflections, ethics table
│   └── Task 6: Summary & Presentation   # Key findings + presentation prep
│
├── gapminder_data_graphs.csv             # Gapminder: continent, HDI, GDP, CO2 (3,675 rows)
├── Life_Expectancy_Data.csv              # WHO: 22 health indicators (2,938 rows)
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 📊 Dashboard Features

### Task 4 — All 7 Advanced Criteria Met

| Criterion | Implementation |
|-----------|----------------|
| ✅ Multi-layer visualizations | Scatter + OLS trend + notched box plots in one panel |
| ✅ Faceted / small-multiple charts | Life expectancy by continent × development status (3×2 grid) |
| ✅ Interactive Streamlit dashboard | 7 tabs, live sidebar filters, 20+ interactive charts |
| ✅ Geospatial visualizations | Choropleth world map with year slider across 7 indicators |
| ✅ Scenario-based comparison | Radar + diverging bars + Top 10 vs Bottom 10 ranked charts |
| ✅ Ethical bias visualization | Missing data heatmap + bias bar chart by continent & income stage |
| ✅ Uncertainty visualization | 95% CI ribbon + ±1 SD band per continent, CI width comparison |

### KPI Cards (Stakeholder-Validated)

| KPI | SDG Target | Stakeholder |
|-----|-----------|-------------|
| Life Expectancy | SDG 3 core outcome | WHO / Ministries |
| Adult Mortality /1K | SDG 3.1 | Government policy |
| Immunization Rate | SDG 3.8 | UNICEF / WHO |
| HDI Index | SDG 3 composite | UNDP |
| Under-5 Deaths /1K | SDG 3.2 | UNICEF |
| HIV/AIDS Rate | SDG 3.3 | NGOs / UNAIDS |

---

## 📂 Datasets

### 1. WHO Life Expectancy Data
- **Source**: World Health Organization (WHO)
- **File**: `Life_Expectancy_Data.csv`
- **Size**: 2,938 rows × 22 columns
- **Coverage**: 193 countries, 2000–2015
- **Key columns**: Life expectancy, adult mortality, infant deaths, immunization rates (hepatitis B, polio, diphtheria), HIV/AIDS, GDP, schooling, BMI

### 2. Gapminder Data
- **Source**: Gapminder Foundation
- **File**: `gapminder_data_graphs.csv`
- **Size**: 3,675 rows × 8 columns
- **Coverage**: Multiple countries, 1998–2018
- **Key columns**: Continent, HDI index, CO2 consumption, GDP per capita, services

### Merged Dataset
After inner join on country + year:
- **2,416 records** × 40 columns
- **151 countries** across 6 continents
- **2000–2015** (16 years)
- **0 missing values** (post-imputation)

### Data Assumptions
- GDP: Gapminder GDP used as primary; WHO GDP as fallback — both in current USD, **PPP not adjusted**
- Missing values: imputed using **country-level median** first, then global median as fallback
- Outliers: **retained** — extreme cases (Sierra Leone, Lesotho) represent genuine health crises relevant to SDG 3

---

## 👥 Group Members

| # | Name | Student ID | Role |
|---|------|------------|------|
| 1 | | | Data Engineering & EDA |
| 2 | | | Advanced Visualizations |
| 3 | | | Dashboard & Scenario Comparison |
| 4 | | | Ethical Bias & Uncertainty |
| 5 | | | Report Writing & Task 1 |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Core language |
| Streamlit | Latest | Interactive dashboard |
| Plotly | Latest | All visualizations |
| Pandas | Latest | Data manipulation |
| NumPy | Latest | Numerical computing |
| scikit-learn | Latest | MinMaxScaler normalization |
| SciPy | Latest | Statistics (OLS, CI, Z-score) |

---

## 📄 License

This project was created for academic purposes as part of ITS68404 – Data Visualization at Taylor's University, January 2026 Semester.

---

*ITS68404 · Data Visualization · Group 8 · Taylor's University · January 2026*
