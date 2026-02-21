"""
MLOps Dashboard — Credit Risk Scoring & Data Drift Detection
=============================================================
Dashboard profesional de monitoreo con diseño visual de alta impacto.

Stack:  Streamlit · Plotly · scikit-learn · SciPy

Features:
- Dark UI con glassmorphism y neon gradients
- Detección de Data Drift (KS, PSI, JS Divergence, Chi²)
- Sistema de alertas multinivel con animaciones
- Análisis temporal y comparación de distribuciones
- Recomendaciones automáticas contextualizadas

Autor: Alexis Jacquet
Proyecto: Henry — M5 · Avance 3
Fecha: Febrero 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "mlops_pipeline", "src"))

from ft_engineering import load_and_prepare_data
from model_monitoring import DataDriftDetector
from sklearn.tree import DecisionTreeClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLOps Monitor | Credit Risk",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark Cyberpunk / Finance aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
  --bg-primary:    #020817;
  --bg-secondary:  #0c1425;
  --bg-card:       rgba(15, 23, 42, 0.80);
  --border-dim:    rgba(99, 179, 237, 0.12);
  --border-glow:   rgba(99, 179, 237, 0.50);
  --text-primary:  #e2e8f0;
  --text-muted:    #64748b;
  --accent-blue:   #38bdf8;
  --accent-purple: #a78bfa;
  --accent-teal:   #2dd4bf;
  --accent-gold:   #fbbf24;
  --ok:            #22d3ee;
  --warn:          #fbbf24;
  --danger:        #f87171;
  --critical:      #e11d48;
  --radius:        14px;
  --shadow:        0 8px 32px rgba(0,0,0,.55);
}

/* ── Base — forzar fondo oscuro en TODOS los contenedores Streamlit ── */
html, body { background: var(--bg-primary) !important; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="stHeader"],
[class*="css"] {
  font-family: 'Space Grotesk', sans-serif !important;
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

/* Sticky header barra superior */
[data-testid="stHeader"] {
  background: rgba(2,8,23,0.95) !important;
  border-bottom: 1px solid var(--border-dim) !important;
  backdrop-filter: blur(12px);
}

/* Toolbar (Deploy, etc.) */
[data-testid="stToolbar"] { background: transparent !important; }

.main { background: var(--bg-primary) !important; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1400px; background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── Headings ── */
h1, h2, h3, h4 { color: var(--text-primary) !important; font-family: 'Space Grotesk', sans-serif !important; }

/* ── Hero title gradient ── */
.hero-title {
  background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 50%, #2dd4bf 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.1;
  animation: gradientShift 4s ease infinite;
  background-size: 200% 200%;
}

@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.hero-subtitle {
  color: var(--text-muted);
  font-size: 1.05rem;
  font-weight: 400;
  letter-spacing: 0.04em;
  margin-top: 0.4rem;
}

.hero-badge {
  display: inline-block;
  background: rgba(56,189,248,0.12);
  border: 1px solid rgba(56,189,248,0.30);
  color: var(--accent-blue);
  padding: 0.2rem 0.75rem;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-right: 0.5rem;
}

/* ── Glassmorphism card ── */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border-dim);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(56,189,248,0.40), transparent);
}
.glass-card:hover {
  border-color: var(--border-glow);
  box-shadow: 0 8px 40px rgba(56,189,248,0.10);
}

/* ── Metric card variants ── */
.metric-card {
  padding: 1.2rem 1.5rem;
  border-radius: var(--radius);
  background: var(--bg-card);
  border: 1px solid var(--border-dim);
  backdrop-filter: blur(12px);
}
.metric-card .mc-label {
  color: var(--text-muted);
  font-size: 0.78rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.metric-card .mc-value {
  font-size: 2rem;
  font-weight: 700;
  line-height: 1.1;
  margin: 0.2rem 0 0.1rem;
}
.metric-card .mc-delta {
  font-size: 0.82rem;
  font-weight: 500;
}
.mc-blue   .mc-value { color: var(--accent-blue); }
.mc-purple .mc-value { color: var(--accent-purple); }
.mc-teal   .mc-value { color: var(--accent-teal); }
.mc-gold   .mc-value { color: var(--accent-gold); }

/* ── Alert banners ── */
.alert-box {
  padding: 1.2rem 1.6rem;
  border-radius: var(--radius);
  margin: 1rem 0;
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  backdrop-filter: blur(10px);
}
.alert-icon  { font-size: 2rem; line-height: 1; flex-shrink: 0; }
.alert-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.25rem; }
.alert-body  { font-size: 0.90rem; color: var(--text-muted); line-height: 1.6; }

.alert-green  { background: rgba(34,211,238,0.08);  border: 1px solid rgba(34,211,238,0.35); }
.alert-green  .alert-title { color: #22d3ee; }
.alert-yellow { background: rgba(251,191,36,0.08);  border: 1px solid rgba(251,191,36,0.35); }
.alert-yellow .alert-title { color: #fbbf24; }
.alert-orange { background: rgba(251,146,60,0.08);  border: 1px solid rgba(251,146,60,0.35); }
.alert-orange .alert-title { color: #fb923c; }
.alert-red    { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.35); animation: pulseRed 2.5s ease infinite; }
.alert-red    .alert-title { color: #f87171; }

@keyframes pulseRed {
  0%, 100% { box-shadow: 0 0 0   0 rgba(248,113,113,0.00); }
  50%       { box-shadow: 0 0 20px 4px rgba(248,113,113,0.25); }
}

/* ── Status dot ── */
.status-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 6px;
  animation: blink 2s ease infinite;
}
.dot-green  { background: #22d3ee; }
.dot-yellow { background: #fbbf24; animation: none; }
.dot-orange { background: #fb923c; animation: none; }
.dot-red    { background: #f87171; }

@keyframes blink {
  0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Section divider ── */
.section-sep {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
  margin: 1.8rem 0;
  border: none;
}

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-secondary) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  gap: 0 !important;
  border: 1px solid var(--border-dim) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: 9px !important;
  padding: 8px 18px !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  border: none !important;
  transition: all 0.25s !important;
  white-space: nowrap;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(56,189,248,0.20), rgba(167,139,250,0.20)) !important;
  color: var(--accent-blue) !important;
  border: 1px solid rgba(56,189,248,0.30) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.55rem 1.4rem !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 12px rgba(56,189,248,0.20) !important;
  transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider p { color: var(--text-primary) !important; }

/* ── Slider track ── */
[data-testid="stSlider"] > div > div > div > div {
  background: linear-gradient(90deg, #38bdf8, #a78bfa) !important;
}

/* ── Metric component ── */
[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.2rem !important;
  backdrop-filter: blur(8px);
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrameContainer"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 10px !important;
}

/* ── Info / warning / error boxes ── */
div[data-testid="stInfo"]    { background: rgba(56,189,248,0.08)  !important; border: 1px solid rgba(56,189,248,0.25)  !important; }
div[data-testid="stSuccess"] { background: rgba(34,197,94,0.08)   !important; }
div[data-testid="stWarning"] { background: rgba(251,191,36,0.08)  !important; }
div[data-testid="stError"]   { background: rgba(248,113,113,0.08) !important; }

/* ── Footer ── */
.custom-footer {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  background: rgba(2,8,23,0.95);
  border-top: 1px solid var(--border-dim);
  padding: 0.55rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.78rem;
  color: var(--text-muted);
  backdrop-filter: blur(12px);
  z-index: 9999;
}
.custom-footer span { color: var(--accent-blue); font-weight: 500; }

/* ── Stat row ── */
.stat-row { display: flex; justify-content: space-between; align-items: center; margin: 0.4rem 0; }
.stat-row .label { color: var(--text-muted); font-size: 0.82rem; }
.stat-row .value { font-size: 0.92rem; font-weight: 600; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; }

/* ── Badge ── */
.feature-tag {
  display: inline-block;
  background: rgba(56,189,248,0.10);
  border: 1px solid rgba(56,189,248,0.25);
  color: var(--accent-blue);
  padding: 0.15rem 0.6rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 500;
  font-family: 'JetBrains Mono', monospace;
}

.sidebar-title {
  font-size: 1.1rem; font-weight: 700;
  background: linear-gradient(135deg, #38bdf8, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-align: center;
}
.sidebar-version { text-align: center; color: var(--text-muted); font-size: 0.78rem; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY DARK THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Space Grotesk", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="rgba(51,65,85,0.6)", linecolor="#334155", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(51,65,85,0.6)", linecolor="#334155", tickfont=dict(size=10)),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#0c1425", bordercolor="#334155", font=dict(family="Space Grotesk", size=12)),
)

COLOR_SCALE = [
    [0.00, "#0c1425"], [0.25, "#1e3a5f"],
    [0.50, "#0ea5e9"], [0.75, "#7c3aed"],
    [1.00, "#f87171"],
]

PALETTE = {
    "baseline": "#38bdf8",
    "current":  "#a78bfa",
    "green":    "#22d3ee",
    "yellow":   "#fbbf24",
    "orange":   "#fb923c",
    "red":      "#f87171",
    "psi":      "#38bdf8",
    "ks":       "#a78bfa",
    "js":       "#2dd4bf",
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADERS (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    DATA_PATH = os.path.join(ROOT, "data", "Base_de_datos.csv")
    data = load_and_prepare_data(DATA_PATH, test_size=0.2, random_state=42)
    if data is not None and "feature_names" in data:
        cols = list(data["feature_names"])
        if not isinstance(data["X_train"], pd.DataFrame):
            data["X_train"] = pd.DataFrame(data["X_train"], columns=cols)
        if not isinstance(data["X_test"], pd.DataFrame):
            data["X_test"] = pd.DataFrame(data["X_test"], columns=cols)
    return data


@st.cache_resource(show_spinner=False)
def load_model():
    data = load_data()
    if data is None:
        return None
    clf = DecisionTreeClassifier(
        max_depth=None, min_samples_split=2,
        class_weight="balanced", random_state=42,
    )
    clf.fit(data["X_train"], data["y_train"])
    return clf


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def gauge_chart(value: float, title: str, lo: float, hi: float, height=260) -> go.Figure:
    """Velocimetro neon."""
    if value < lo:
        color, status = PALETTE["green"],  "OK"
    elif value < hi:
        color, status = PALETTE["yellow"], "WARN"
    else:
        color, status = PALETTE["red"],    "ALERT"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 4),
        number=dict(font=dict(size=28, color=color, family="JetBrains Mono")),
        title=dict(
            text=f"{title}<br><span style='font-size:0.7em;color:#64748b'>{status}</span>",
            font=dict(size=13, color="#94a3b8"),
        ),
        gauge=dict(
            axis=dict(range=[0, max(1, value * 1.5)],
                      tickfont=dict(size=9, color="#64748b"),
                      nticks=5,
                      tickcolor="#334155"),
            bar=dict(color=color, thickness=0.70),
            bgcolor="#0c1425",
            borderwidth=0,
            steps=[
                dict(range=[0,            lo],            color="rgba(34,211,238,0.06)"),
                dict(range=[lo,           hi],            color="rgba(251,191,36,0.06)"),
                dict(range=[hi, max(1, value*1.5)],       color="rgba(248,113,113,0.06)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=10),
        font=dict(family="Space Grotesk", color="#94a3b8"),
    )
    return fig


def distribution_chart(baseline: pd.Series, current: pd.Series, name: str) -> go.Figure:
    """Overlapping histograms con KDE overlay."""
    from scipy.stats import gaussian_kde

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=baseline, name="Baseline (Train)",
        opacity=0.55, marker_color=PALETTE["baseline"],
        nbinsx=60, histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=current, name="Current (Test)",
        opacity=0.55, marker_color=PALETTE["current"],
        nbinsx=60, histnorm="probability density",
    ))

    for series, color, label in [
        (baseline, PALETTE["baseline"], "KDE Baseline"),
        (current,  PALETTE["current"],  "KDE Current"),
    ]:
        vals = series.dropna().values
        if len(vals) < 5:
            continue
        x_kde = np.linspace(vals.min(), vals.max(), 200)
        try:
            kde = gaussian_kde(vals, bw_method="silverman")
            fig.add_trace(go.Scatter(
                x=x_kde, y=kde(x_kde),
                name=label, mode="lines",
                line=dict(color=color, width=2.5),
            ))
        except Exception:
            pass

    fig.update_layout(
        title=dict(text=f"<b>{name}</b>", font=dict(size=14)),
        barmode="overlay",
        xaxis_title=name,
        yaxis_title="Density",
        height=380,
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=10)),
        **PLOTLY_DARK,
    )
    return fig


def drift_bar_chart(drift_df: pd.DataFrame) -> go.Figure:
    """Grouped bar PSI / KS / JS por feature."""
    metrics = [m for m in ["psi", "ks_statistic", "js_divergence"] if m in drift_df.columns]
    if not metrics:
        return None

    top = drift_df.sort_values("drift_detected", ascending=False).head(20)
    features = top["feature"].tolist()

    fig = go.Figure()
    colors = [PALETTE["psi"], PALETTE["ks"], PALETTE["js"]]
    labels = {"psi": "PSI", "ks_statistic": "KS Statistic", "js_divergence": "JS Divergence"}

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=labels.get(metric, metric),
            x=features,
            y=top[metric].fillna(0),
            marker_color=color,
            marker_opacity=0.80,
            text=top[metric].fillna(0).round(3),
            textposition="auto",
            textfont=dict(size=9),
        ))

    fig.update_layout(
        title=dict(text="<b>Top Features — Drift Metrics</b>", font=dict(size=14)),
        barmode="group",
        yaxis_title="Metric Value",
        height=440,
        legend=dict(orientation="h", y=1.05, x=0),
        **PLOTLY_DARK,
    )
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    return fig


def drift_heatmap(drift_df: pd.DataFrame) -> go.Figure:
    """Heatmap de metricas de drift."""
    metrics = [m for m in ["psi", "ks_statistic", "js_divergence"] if m in drift_df.columns]
    if not metrics:
        return None

    top = drift_df.sort_values("drift_detected", ascending=False).head(25)
    z = top[metrics].fillna(0).values.T

    fig = go.Figure(go.Heatmap(
        z=z,
        x=top["feature"].tolist(),
        y=[m.upper() for m in metrics],
        colorscale=COLOR_SCALE,
        showscale=True,
        colorbar=dict(thickness=14, tickfont=dict(size=9), outlinewidth=0),
        text=np.round(z, 3),
        texttemplate="%{text}",
        textfont=dict(size=8.5),
    ))
    fig.update_layout(
        title=dict(text="<b>Drift Heatmap — Top 25 Features</b>", font=dict(size=14)),
        height=300,
        **PLOTLY_DARK,
    )
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    return fig


def timeline_chart(history: list) -> go.Figure:
    """Linea de tiempo de drift con area rellena."""
    if not history:
        return None
    ts  = [h["timestamp"] for h in history]
    pct = [h["summary"]["drift_percentage"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=pct,
        mode="lines+markers",
        name="Drift %",
        line=dict(color=PALETTE["baseline"], width=2.5, shape="spline"),
        marker=dict(size=7, color=PALETTE["baseline"],
                    line=dict(color="#020817", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.08)",
    ))

    fig.add_hline(y=10, line_dash="dash", line_color=PALETTE["yellow"],
                  annotation_text="10% warn", annotation_font_size=10,
                  annotation_font_color=PALETTE["yellow"])
    fig.add_hline(y=30, line_dash="dash", line_color=PALETTE["red"],
                  annotation_text="30% critical", annotation_font_size=10,
                  annotation_font_color=PALETTE["red"])

    fig.update_layout(
        title=dict(text="<b>Drift % — Temporal Evolution</b>", font=dict(size=14)),
        xaxis_title="Date",
        yaxis_title="Drift Percentage (%)",
        height=380,
        legend=dict(orientation="h", y=1.05),
        **PLOTLY_DARK,
    )
    return fig


def radar_chart(drift_df: pd.DataFrame) -> go.Figure:
    """Radar chart con top features con drift."""
    metrics = [m for m in ["psi", "ks_statistic", "js_divergence"] if m in drift_df.columns]
    if not metrics:
        return None
    top = drift_df.sort_values("drift_detected", ascending=False).head(8)

    fig = go.Figure()
    for _, row in top.iterrows():
        vals = [row.get(m, 0) or 0 for m in metrics]
        vals += [vals[0]]
        cats = [m.upper() for m in metrics] + [metrics[0].upper()]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself",
            name=row["feature"][:18],
            opacity=0.55,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(12,20,37,0.8)",
            radialaxis=dict(visible=True, gridcolor="#334155", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="#334155"),
        ),
        showlegend=True,
        legend=dict(font=dict(size=9), x=1.05),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#94a3b8"),
        title=dict(text="<b>Radar — Drift per Feature</b>", font=dict(size=14)),
        margin=dict(l=40, r=120, t=50, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def alert_banner(level: str, title: str, body: str = ""):
    icons = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}
    st.markdown(f"""
    <div class="alert-box alert-{level}">
      <div class="alert-icon">{icons.get(level, '⚪')}</div>
      <div>
        <div class="alert-title">{title}</div>
        {"<div class='alert-body'>" + body + "</div>" if body else ""}
      </div>
    </div>""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", color: str = "blue"):
    st.markdown(f"""
    <div class="metric-card mc-{color}">
      <div class="mc-label">{label}</div>
      <div class="mc-value">{value}</div>
      {"<div class='mc-delta'>" + delta + "</div>" if delta else ""}
    </div>""", unsafe_allow_html=True)


def stat_row(label: str, value: str):
    st.markdown(f"""
    <div class="stat-row">
      <span class="label">{label}</span>
      <span class="value">{value}</span>
    </div>""", unsafe_allow_html=True)


def section_sep():
    st.markdown('<hr class="section-sep">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem;font-size:2rem;'>📡</div>
    <div class='sidebar-title'>MLOps Monitor</div>
    <div class='sidebar-version'>v3.0.0 · Credit Risk · Henry M5</div>
    """, unsafe_allow_html=True)
    section_sep()

    st.markdown("**⚙️ Drift Thresholds**")
    ks_threshold  = st.slider("KS p-value cutoff", 0.01, 0.10, 0.05, 0.01)
    psi_threshold = st.slider("PSI threshold",     0.05, 0.30, 0.10, 0.05)
    js_threshold  = st.slider("JS Divergence",     0.05, 0.30, 0.10, 0.05)
    section_sep()

    st.markdown("**📐 Display Options**")
    show_distributions = st.checkbox("Distribuciones con KDE", value=True)
    show_radar         = st.checkbox("Radar chart de features", value=True)
    show_temporal      = st.checkbox("Analisis temporal",       value=True)
    section_sep()

    if st.button("🔄  Clear cache & rerun", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style='margin-top:1rem;text-align:center;color:#475569;font-size:0.75rem;'>
    Alexis Jacquet · Feb 2026<br>Henry Data Science Bootcamp
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & RUN DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("⚡ Inicializando pipeline…"):
    data  = load_data()
    model = load_model()

if data is None or model is None:
    st.error("❌ Error al cargar datos o modelo. Revisa la configuracion.")
    st.stop()

with st.spinner("🔬 Analizando distribuciones…"):
    detector = DataDriftDetector(
        ks_threshold=ks_threshold,
        psi_threshold=psi_threshold,
        js_threshold=js_threshold,
    )
    detector.fit(data["X_train"], data["feature_names"])
    drift_results = detector.detect_drift(data["X_test"])
    drift_df      = detector.get_drift_summary_dataframe(drift_results)

summary     = drift_results["summary"]
alert_level = summary["alert_level"]


# ══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_hero, col_status = st.columns([3, 1])
with col_hero:
    st.markdown("""
    <div style='padding:0.5rem 0 0.3rem;'>
      <div class='hero-title'>MLOps Dashboard</div>
      <div class='hero-subtitle'>Credit Risk Scoring &amp; Real-Time Data Drift Detection</div>
      <div style='margin-top:0.8rem;'>
        <span class='hero-badge'>KS Test</span>
        <span class='hero-badge'>PSI</span>
        <span class='hero-badge'>Jensen–Shannon</span>
        <span class='hero-badge'>Chi²</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_status:
    dot = "dot-" + alert_level
    level_labels = {
        "green":  "OPERATIVO",
        "yellow": "MONITOREO",
        "orange": "INVESTIGAR",
        "red":    "CRITICO",
    }
    st.markdown(f"""
    <div class='glass-card' style='text-align:center;padding:1.5rem;'>
      <div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>
        Estado del Sistema</div>
      <span class='status-dot {dot}'></span>
      <span style='font-size:1.1rem;font-weight:700;color:var(--text-primary);'>
        {level_labels.get(alert_level, "UNKNOWN")}</span>
      <div style='font-size:0.78rem;color:#64748b;margin-top:0.4rem;'>
        {datetime.now().strftime("%H:%M:%S")}</div>
    </div>
    """, unsafe_allow_html=True)

section_sep()

# ══════════════════════════════════════════════════════════════════════════════
#  TOP KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("📊 Features Analizadas", summary["total_features"])
with k2:
    st.metric(
        "⚠️ Features con Drift",
        summary["features_with_drift"],
        delta=f"{summary['drift_percentage']:.1f}% del total",
        delta_color="inverse",
    )
with k3:
    score = model.score(data["X_test"], data["y_test"])
    st.metric("🎯 Accuracy (Test)", f"{score*100:.2f}%")
with k4:
    st.metric("🗄️ Registros procesados", "10,763")

section_sep()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡  Dashboard",
    "🔬  Feature Analysis",
    "📊  Distributions",
    "⏱   Timeline",
    "💡  Recommendations",
])


# ─── TAB 1 — DASHBOARD ───────────────────────────────────────────────────────
with tab1:
    alert_msgs = {
        "green":  ("All Clear — No Significant Drift Detected",
                   "Las distribuciones se mantienen estables respecto al baseline."),
        "yellow": ("Minor Drift Detected — Increased Monitoring Required",
                   "Un porcentaje pequeño de features muestra desviaciones. Monitoreo mas frecuente recomendado."),
        "orange": ("Moderate Drift Alert — Investigation Recommended",
                   "Drift significativo en multiples features. Revisar fuentes de datos y planificar reentrenamiento."),
        "red":    ("CRITICAL DRIFT — Immediate Action Required",
                   "El modelo puede estar produciendo predicciones degradadas. Evaluar pausar inferencias."),
    }
    title, body = alert_msgs.get(alert_level, ("Unknown", ""))
    alert_banner(alert_level, title, body)

    section_sep()

    # Gauges
    st.markdown("#### 🎚 Indicadores de Drift Estadistico")
    g1, g2, g3 = st.columns(3)

    avg_psi = float(drift_df["psi"].mean())           if "psi"           in drift_df.columns else 0.0
    avg_ks  = float(drift_df["ks_statistic"].mean())  if "ks_statistic"  in drift_df.columns else 0.0
    avg_js  = float(drift_df["js_divergence"].mean()) if "js_divergence" in drift_df.columns else 0.0

    with g1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(avg_psi, "PSI (avg)",      0.10, 0.20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with g2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(avg_ks,  "KS Stat (avg)",  0.10, 0.30), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with g3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(avg_js,  "JS Div (avg)",   0.10, 0.20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    section_sep()

    # Dataset overview
    st.markdown("#### 🗃 Dataset Overview")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:0.78rem;color:#38bdf8;text-transform:uppercase;
            letter-spacing:0.1em;margin-bottom:0.8rem;'>🔵 Baseline — Training Set</div>""",
            unsafe_allow_html=True)
        stat_row("Muestras",         f"{len(data['X_train']):,}")
        stat_row("Features",         str(len(data["feature_names"])))
        stat_row("% Clase 0 (mora)", f"{(data['y_train']==0).sum()/len(data['y_train'])*100:.1f}%")
        stat_row("% Clase 1 (pago)", f"{(data['y_train']==1).sum()/len(data['y_train'])*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:0.78rem;color:#a78bfa;text-transform:uppercase;
            letter-spacing:0.1em;margin-bottom:0.8rem;'>🟣 Current — Test Set</div>""",
            unsafe_allow_html=True)
        stat_row("Muestras",         f"{len(data['X_test']):,}")
        stat_row("Features",         str(len(data["feature_names"])))
        stat_row("% Clase 0 (mora)", f"{(data['y_test']==0).sum()/len(data['y_test'])*100:.1f}%")
        stat_row("% Clase 1 (pago)", f"{(data['y_test']==1).sum()/len(data['y_test'])*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    section_sep()

    # Feature importance
    st.markdown("#### 🌲 Feature Importance — Decision Tree")
    importances = model.feature_importances_
    feat_names  = list(data["feature_names"])
    fi_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(20)
    )
    fig_fi = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(
            color=fi_df["importance"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#0ea5e9"], [1, "#a78bfa"]],
            showscale=False,
        ),
        text=fi_df["importance"].round(4),
        textposition="auto",
        textfont=dict(size=9),
    ))
    fig_fi.update_layout(
        title=dict(text="<b>Top-20 Features por Importancia</b>", font=dict(size=14)),
        xaxis_title="Importance",
        height=520,
        **PLOTLY_DARK,
    )
    fig_fi.update_layout(yaxis=dict(tickfont=dict(size=9)))
    st.plotly_chart(fig_fi, use_container_width=True)


# ─── TAB 2 — FEATURE ANALYSIS ────────────────────────────────────────────────
with tab2:
    st.markdown("#### 🔬 Analisis por Feature")

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        filter_opt = st.radio(
            "Filtrar:",
            ["Todas", "Solo Drift", "Numericas", "Categoricas"],
            horizontal=True,
        )
    with fc2:
        sort_col = st.selectbox("Ordenar por",
                                ["drift_detected", "psi", "ks_statistic", "js_divergence"])

    fdf = drift_df.copy()
    if filter_opt == "Solo Drift":
        fdf = fdf[fdf["drift_detected"] == True]
    elif filter_opt == "Numericas":
        fdf = fdf[fdf["type"] == "numeric"]
    elif filter_opt == "Categoricas":
        fdf = fdf[fdf["type"] == "categorical"]

    if sort_col in fdf.columns:
        fdf = fdf.sort_values(sort_col, ascending=False)

    n_drift = int(fdf["drift_detected"].sum()) if "drift_detected" in fdf.columns else 0
    st.markdown(
        f"Mostrando **{len(fdf)}** features · "
        f"<span style='color:#f87171;font-weight:600;'>{n_drift} con drift detectado</span>",
        unsafe_allow_html=True,
    )

    numeric_cols = [c for c in ["psi", "ks_statistic", "ks_pvalue", "js_divergence"] if c in fdf.columns]
    fmt = {c: "{:.4f}" for c in numeric_cols}
    styled = fdf.style.background_gradient(
        subset=["psi"] if "psi" in fdf.columns else [],
        cmap="Blues", vmin=0, vmax=0.5,
    ).format(fmt)
    st.dataframe(styled, use_container_width=True, height=360)

    section_sep()

    fig_bar = drift_bar_chart(fdf)
    if fig_bar:
        st.plotly_chart(fig_bar, use_container_width=True)

    fig_heat = drift_heatmap(fdf)
    if fig_heat:
        st.plotly_chart(fig_heat, use_container_width=True)

    if show_radar:
        fig_radar = radar_chart(fdf)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)


# ─── TAB 3 — DISTRIBUTIONS ───────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📊 Comparacion de Distribuciones — Baseline vs Current")

    feature_names_list = list(data["feature_names"])
    if "feature" in drift_df.columns and "drift_detected" in drift_df.columns:
        drifted_first = drift_df.sort_values("drift_detected", ascending=False)["feature"].tolist()
        others = [f for f in feature_names_list if f not in drifted_first]
        order  = drifted_first + others
    else:
        order = feature_names_list

    selected_feat = st.selectbox("Feature:", options=order, index=0)

    if show_distributions:
        dist_fig = distribution_chart(
            data["X_train"][selected_feat],
            data["X_test"][selected_feat],
            selected_feat,
        )
        st.plotly_chart(dist_fig, use_container_width=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#38bdf8;font-weight:600;font-size:0.88rem;margin-bottom:0.6rem;'>📐 Baseline — {selected_feat}</div>", unsafe_allow_html=True)
            st.dataframe(data["X_train"][selected_feat].describe().to_frame(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with s2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#a78bfa;font-weight:600;font-size:0.88rem;margin-bottom:0.6rem;'>📐 Current — {selected_feat}</div>", unsafe_allow_html=True)
            st.dataframe(data["X_test"][selected_feat].describe().to_frame(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    section_sep()

    st.markdown("#### 🪟 Multi-Feature Grid — Top Drift Features")
    top_feats = []
    if "drift_detected" in drift_df.columns:
        top_feats = drift_df[drift_df["drift_detected"] == True]["feature"].head(6).tolist()

    if top_feats:
        for i in range(0, len(top_feats), 2):
            cc1, cc2 = st.columns(2)
            for col_w, feat in zip([cc1, cc2], top_feats[i:i+2]):
                with col_w:
                    mini = distribution_chart(data["X_train"][feat], data["X_test"][feat], feat)
                    mini.update_layout(height=280, margin=dict(l=30, r=10, t=40, b=30))
                    st.plotly_chart(mini, use_container_width=True)
    else:
        st.info("✅ No hay features con drift significativo para mostrar en la grilla.")


# ─── TAB 4 — TIMELINE ────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### ⏱ Evolucion Temporal del Data Drift")

    if show_temporal:
        st.info("📌 Simulacion historica para demostracion. En produccion se almacenarian resultados reales por fecha.")

        rng   = np.random.default_rng(0)
        dates = pd.date_range(end=datetime.now(), periods=14, freq="D")
        base  = summary["drift_percentage"]
        simulated = []
        for i, d in enumerate(dates):
            pct = max(0, min(100, base + rng.normal(0, 3) + i * 0.5))
            simulated.append({
                "timestamp": d.strftime("%Y-%m-%d"),
                "summary":   {"drift_percentage": round(pct, 2)},
            })

        fig_tl = timeline_chart(simulated)
        if fig_tl:
            st.plotly_chart(fig_tl, use_container_width=True)

        section_sep()

        hist_pcts = [h["summary"]["drift_percentage"] for h in simulated]
        tm1, tm2, tm3 = st.columns(3)
        trend = "↗️  Creciente" if hist_pcts[-1] > hist_pcts[0] else "↘️  Decreciente"
        with tm1:
            st.metric("Tendencia",      trend)
        with tm2:
            st.metric("Drift Promedio", f"{np.mean(hist_pcts):.1f}%")
        with tm3:
            st.metric("Drift Maximo",   f"{max(hist_pcts):.1f}%")

        section_sep()

        st.markdown("#### 📋 Registro de Analisis")
        hist_df = pd.DataFrame([{
            "Fecha":   h["timestamp"],
            "Drift %": h["summary"]["drift_percentage"],
            "Estado":  "⚠️" if h["summary"]["drift_percentage"] > 10 else "✅",
        } for h in simulated])
        st.dataframe(hist_df, use_container_width=True, height=320)


# ─── TAB 5 — RECOMMENDATIONS ─────────────────────────────────────────────────
with tab5:
    st.markdown("#### 💡 Recomendaciones & Plan de Accion")

    raw_msg = detector.generate_alert_message(drift_results)
    alert_banner(
        alert_level,
        {"green": "All Clear", "yellow": "Alerta Menor",
         "orange": "Alerta Moderada", "red": "ALERTA CRITICA"}.get(alert_level, ""),
        raw_msg.replace("\n", " · "),
    )

    section_sep()

    action_plans = {
        "green": {
            "short":  ["Mantener monitoreo semanal programado",
                       "Documentar estado actual como referencia",
                       "Revisar logs de calidad de datos entrantes"],
            "medium": ["Preparar plan de reentrenamiento trimestral",
                       "Evaluar nuevos datos para enriquecer el baseline"],
            "long":   ["Revisar arquitectura del pipeline de features",
                       "Explorar tecnicas de drift detection adicionales"],
        },
        "yellow": {
            "short":  ["Incrementar frecuencia de monitoreo a diario",
                       "Analizar features con mayor PSI individualmente",
                       "Revisar integridad del pipeline de datos"],
            "medium": ["Evaluar si el drift es estacional o permanente",
                       "Preparar dataset actualizado para re-training",
                       "Comunicar hallazgos al equipo de riesgo"],
            "long":   ["Planificar ciclo de reentrenamiento anticipado",
                       "Revisar thresholds de alerta segun contexto"],
        },
        "orange": {
            "short":  ["🚨 Investigacion urgente de fuentes de datos",
                       "Validar calidad del pipeline de ingesta",
                       "Ejecutar validacion cruzada con datos recientes"],
            "medium": ["Iniciar proceso de reentrenamiento del modelo",
                       "Preparar A/B test con modelo actualizado",
                       "Notificar a stakeholders sobre el estado"],
            "long":   ["Implementar reentrenamiento automatico continuo",
                       "Mejorar cobertura de monitoreo con mas metricas"],
        },
        "red": {
            "short":  ["🛑 EVALUAR pausar predicciones del modelo",
                       "Notificar INMEDIATAMENTE a todos los stakeholders",
                       "Activar plan de contingencia de credito"],
            "medium": ["Reentrenar con datos mas recientes HOY",
                       "Validar nuevo modelo (CV + hold-out)",
                       "Documentar incidente con fecha y causa raiz"],
            "long":   ["Implementar reentrenamiento continuo en produccion",
                       "Revisar SLAs y alertas automaticas del sistema",
                       "Post-mortem y mejora del proceso de MLOps"],
        },
    }

    plan = action_plans.get(alert_level, action_plans["green"])

    c1, c2, c3 = st.columns(3)
    for col, title_txt, items, color in [
        (c1, "⚡ Accion Inmediata",  plan["short"],  "#22d3ee"),
        (c2, "📅 Mediano Plazo",     plan["medium"], "#a78bfa"),
        (c3, "🔭 Largo Plazo",       plan["long"],   "#2dd4bf"),
    ]:
        with col:
            bullet = "".join(
                f"<li style='margin:0.4rem 0;font-size:0.88rem;'>{i}</li>" for i in items
            )
            col.markdown(f"""
            <div class='glass-card'>
              <div style='font-size:0.85rem;font-weight:700;color:{color};margin-bottom:0.8rem;'>{title_txt}</div>
              <ul style='padding-left:1.1rem;color:#94a3b8;line-height:1.6;'>{bullet}</ul>
            </div>""", unsafe_allow_html=True)

    section_sep()

    # Model performance
    st.markdown("#### 📈 Performance del Modelo en Produccion")
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

    y_pred  = model.predict(data["X_test"])
    y_proba = model.predict_proba(data["X_test"])[:, 1]

    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    for col, label, val, color in zip(
        [pm1, pm2, pm3, pm4, pm5],
        ["ROC-AUC", "F1-Score", "Precision", "Recall", "Accuracy"],
        [
            f"{roc_auc_score(data['y_test'], y_proba):.4f}",
            f"{f1_score(data['y_test'], y_pred):.4f}",
            f"{precision_score(data['y_test'], y_pred):.4f}",
            f"{recall_score(data['y_test'], y_pred):.4f}",
            f"{accuracy_score(data['y_test'], y_pred):.4f}",
        ],
        ["blue", "purple", "teal", "gold", "blue"],
    ):
        with col:
            kpi_card(label, val, color=color)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="custom-footer">
  <span>📡 MLOps Dashboard</span> &nbsp;·&nbsp; Credit Risk Scoring &nbsp;·&nbsp; Data Drift Detection
  <div>Alexis Jacquet &nbsp;·&nbsp; <span>Henry M5</span> &nbsp;·&nbsp; Febrero 2026 &nbsp;·&nbsp; v3.0.0</div>
</div>
""", unsafe_allow_html=True)
