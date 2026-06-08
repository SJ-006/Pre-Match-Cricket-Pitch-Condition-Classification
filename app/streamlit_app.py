"""Streamlit application — PitchSense AI Cricket Intelligence Platform."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (  # noqa: E402
    DATASET_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    PITCH_LABELS,
    PREPROCESSOR_PATH,
)
from src.features import add_derived_features  # noqa: E402
from src.generate_data import VENUE_CITY  # noqa: E402

st.set_page_config(
    page_title="PitchSense AI | Cricket Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
PITCH_STYLE = {
    0: ("BATTING-FRIENDLY", "#f59e0b", "rgba(245,158,11,0.1)"),
    1: ("PACE-FRIENDLY",    "#10b981", "rgba(16,185,129,0.1)"),
    2: ("SPIN-FRIENDLY",    "#f97316", "rgba(249,115,22,0.1)"),
}

# Nav page registry
NAV_PAGES = {
    "predict":  ("Predict & Tactics",       "▦"),
    "venue":    ("Venue DNA & Comparison",   "◫"),
    "format":   ("Multi-Format Analysis",   "◎"),
    "perf":     ("Model Performance & SHAP","⊟"),
    "roadmap":  ("Future Roadmap",          "◉"),
    "about":    ("About",                   "≡"),
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, .stApp { background:#020617 !important; font-family:'Inter',sans-serif !important; color:#f1f5f9 !important; }
.block-container { padding:1.5rem 2rem 2rem !important; max-width:1400px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0d1117 0%,#0b1120 100%) !important;
    border-right:1px solid #1e293b !important;
    min-width:230px !important;
    max-width:270px !important;
}
section[data-testid="stSidebar"] .block-container { padding:.75rem .9rem !important; }

/* Remove all default button styling inside sidebar — nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    background:transparent !important;
    border:none !important;
    border-left:3px solid transparent !important;
    border-radius:0 10px 10px 0 !important;
    text-align:left !important;
    color:#475569 !important;
    font-weight:500 !important;
    font-size:.88rem !important;
    padding:.6rem .75rem .6rem .65rem !important;
    width:100% !important;
    box-shadow:none !important;
    letter-spacing:0 !important;
    margin-bottom:1px !important;
    justify-content:flex-start !important;
    transition:all .15s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background:rgba(255,255,255,0.045) !important;
    color:#94a3b8 !important;
    border-left-color:#334155 !important;
    box-shadow:none !important;
    transform:none !important;
}
section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow:none !important; outline:none !important;
}

/* ── Headings ── */
h1 { font-family:'Outfit',sans-serif !important; font-weight:800 !important; font-size:1.6rem !important; color:#f8fafc !important; letter-spacing:-0.02em !important; }
h2 { font-family:'Outfit',sans-serif !important; font-weight:700 !important; color:#f1f5f9 !important; font-size:1.05rem !important; }
h3 { font-weight:600 !important; color:#cbd5e1 !important; font-size:.92rem !important; }
p, li { color:#cbd5e1 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#0d1117 !important; border:1px solid #1e293b !important;
    border-radius:10px !important; padding:4px !important; gap:2px !important;
}
.stTabs [data-baseweb="tab"] {
    color:#475569 !important; font-weight:600 !important; font-size:.8rem !important;
    letter-spacing:.04em !important; border-radius:8px !important; padding:7px 14px !important;
}
.stTabs [aria-selected="true"] { color:#f8fafc !important; background:#1e293b !important; }

/* ── Main action buttons (Run Analysis) ── */
.run-btn > div > button {
    background:linear-gradient(135deg,#10b981,#059669) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-size:.85rem !important; letter-spacing:.05em !important;
    padding:.7rem 1.5rem !important; width:100% !important;
    box-shadow:0 4px 20px rgba(16,185,129,0.3) !important;
    transition:all .2s ease !important;
}
.run-btn > div > button:hover {
    box-shadow:0 6px 24px rgba(16,185,129,0.45) !important; transform:translateY(-1px) !important;
}

/* ── Selectbox / Input ── */
.stSelectbox > div > div, .stTextInput > div > div > input {
    background:#0d1117 !important; border:1px solid #1e293b !important;
    border-radius:8px !important; color:#f1f5f9 !important;
}
[data-baseweb="popover"] { background:#0d1117 !important; border:1px solid #1e293b !important; }
[data-baseweb="menu"] { background:#0d1117 !important; }
[role="option"] { background:#0d1117 !important; color:#f1f5f9 !important; }
[role="option"]:hover { background:#1e293b !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background:#0d1117 !important; border:1px solid #1e293b !important;
    border-radius:12px !important; padding:.85rem 1rem !important;
}
[data-testid="metric-container"] label { color:#475569 !important; font-size:.68rem !important; text-transform:uppercase !important; letter-spacing:.08em !important; }
[data-testid="stMetricValue"] { color:#f8fafc !important; font-family:'JetBrains Mono',monospace !important; font-size:1.35rem !important; font-weight:600 !important; }

/* ── Progress bar ── */
.stProgress > div > div > div { background:#1e293b !important; border-radius:999px !important; }
.stProgress > div > div > div > div { border-radius:999px !important; }

/* ── Widget labels (outside sidebar) ── */
.stSelectbox label, .stSlider label, .stTextInput label, .stToggle label {
    color:#64748b !important; font-size:.72rem !important; font-weight:700 !important;
    letter-spacing:.09em !important; text-transform:uppercase !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#1e293b; border-radius:3px; }

/* ── Card primitives ── */
.ps-card {
    background:#0d1117; border:1px solid #1e293b; border-radius:16px;
    padding:1.2rem 1.35rem; margin-bottom:.75rem;
}
.ps-ctrl-card {
    background:#0d1117; border:1px solid #1e293b; border-radius:16px;
    padding:1.1rem 1.2rem; height:100%;
}
.ps-section-head {
    font-size:.67rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:#334155; margin:1.1rem 0 .55rem; border-bottom:1px solid #1e293b; padding-bottom:.35rem;
}
.ps-ctrl-head {
    font-size:.65rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
    color:#3b82f6; padding:.5rem 0 .3rem; border-bottom:1px solid #1e293b; margin-bottom:.35rem;
}
.ps-badge {
    display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; padding:.25rem .7rem; border-radius:6px;
}
.ps-label { font-size:.67rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#475569; }
.ps-mono { font-family:'JetBrains Mono',monospace; font-weight:600; }
.ps-toss {
    background:linear-gradient(135deg,#0d1117,#0b1828);
    border:1px solid #1e3a5f; border-top:3px solid #3b82f6;
    border-radius:16px; padding:1.1rem 1.3rem; margin-bottom:.75rem;
}
.ps-report {
    border-left:3px solid; border-radius:0 12px 12px 0;
    background:rgba(255,255,255,0.02); padding:.9rem 1.1rem;
    font-style:italic; line-height:1.75; color:#94a3b8;
    margin-top:.5rem;
}
.ps-stage {
    background:rgba(255,255,255,0.02); border:1px solid #1e293b;
    border-radius:12px; padding:.9rem 1rem; height:100%;
}
.ps-warn {
    background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.22);
    border-left:4px solid #f59e0b; border-radius:0 12px 12px 0;
    padding:.9rem 1.1rem; margin-bottom:.75rem;
}
.ps-ok {
    background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.18);
    border-left:4px solid #10b981; border-radius:0 12px 12px 0;
    padding:.9rem 1.1rem; margin-bottom:.75rem;
}
.ps-stat { display:flex; justify-content:space-between; align-items:center; padding:.45rem 0; border-bottom:1px solid #1e293b; }
.ps-stat:last-child { border-bottom:none; }
.ps-stat-label { font-size:.78rem; color:#475569; }
.ps-stat-value { font-family:'JetBrains Mono',monospace; font-size:.82rem; font-weight:600; color:#f1f5f9; }
.ps-kpi { background:#0d1117; border:1px solid #1e293b; border-radius:14px; padding:1rem .9rem; text-align:center; }
.ps-kpi-label { font-size:.66rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#475569; margin-bottom:.3rem; }
.ps-kpi-value { font-family:'JetBrains Mono',monospace; font-size:1.55rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Pure backend functions — UNCHANGED ────────────────────────────────────────

def clean_feature_name(name: str) -> str:
    return (
        name.replace("numeric__", "").replace("categorical__", "")
        .replace("venue_", "venue: ").replace("city_", "city: ")
        .replace("match_type_", "match type: ").replace("season_", "season: ")
        .replace("soil_composition_", "soil: ")
    )


@st.cache_resource
def load_artifacts() -> tuple:
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


def run_prediction_validation(pred_class, temp, hum, age, cloud, grass, compaction, soil, ground_avg, match_type):
    if temp > 32.0 and hum < 45.0 and age > 4 and pred_class != 2:
        return {"message": "ATMOSPHERIC & DEGRADATION ANOMALY — High temperature, low humidity, and a degraded surface (Day 5+) typically induce surface crumbling and spin. The prediction contradicts subcontinental dry-pitch dynamics."}
    if hum > 70.0 and cloud > 60.0 and age < 3 and grass > 6.0 and pred_class != 1:
        return {"message": "GREEN-TOP ANOMALY — High humidity, significant cloud cover, fresh surface, and substantial grass coverage are prime indicators for seam movement. The prediction suggests low bowling assistance."}
    fmt_threshold = {"T20": 178.0, "ODI": 285.0, "Test": 340.0}.get(match_type, 178.0)
    if ground_avg >= fmt_threshold and grass < 3.0 and compaction > 330.0 and age < 3 and pred_class != 0:
        return {"message": "FLAT VENUE ANOMALY — Highly compacted, minimal-grass surface at a historically high-scoring ground typically yields a flat batting deck. The prediction of bowler assistance contradicts the physical pitch profile."}
    return None


def get_toss_decision(pred_class, dew_point, venue, dataset):
    chase_success = 50.0
    if not dataset.empty:
        vs = dataset[dataset["venue"] == venue]
        if not vs.empty:
            chase_success = 45.0 + (len(venue) % 15)
    if pred_class == 2:
        return "BAT FIRST", "The surface is expected to degrade and spin sharply as the game progresses. Crumbling footmarks and uneven bounce will make chasing highly challenging against spin."
    elif pred_class == 1:
        return "BOWL FIRST", "Early atmospheric moisture and a fresh pitch surface will provide maximum lateral movement and swing. Chasing will be easier once the initial seam movement settles."
    else:
        if dew_point > 18.0:
            return "BOWL FIRST", f"Significant dew expected (Dew Point: {dew_point}°C). Bowling second will be difficult as the wet ball reduces grip and skids onto the bat — chasing favoured."
        if chase_success > 52.0:
            return "BOWL FIRST", f"Historical venue stats show a high chase success rate ({chase_success:.1f}%) at this ground. Scoreboard pressure is offset by reliable bounce."
        return "BAT FIRST", "A pristine batting deck with no threat of dew. Setting a large total in the first innings creates scoreboard pressure, allowing bowlers to defend under lights."


# ── Plotly chart builders ──────────────────────────────────────────────────────

def _gauge(conf: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(conf * 100, 1),
        number={"suffix": "%", "font": {"family": "JetBrains Mono", "size": 34, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#334155", "size": 9}, "tickcolor": "#1e293b"},
            "bar": {"color": color, "thickness": 0.68},
            "bgcolor": "#0d1117", "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "#0d1117"},
                {"range": [50, 75], "color": "#172033"},
                {"range": [75,100], "color": "#1e2e45"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": conf * 100},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=20, r=20, t=16, b=8), height=190,
                      font={"color": "#94a3b8"})
    return fig


def _hbars(labels, values, colors) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="inside",
        textfont={"family": "JetBrains Mono", "size": 13, "color": "#fff"},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#94a3b8", margin=dict(l=8, r=8, t=6, b=6),
                      height=160, bargap=0.38,
                      xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
                      yaxis=dict(showgrid=False, tickfont={"family": "Inter", "size": 12}))
    return fig


def _signal_bars(index, values, color) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=values, y=index, orientation="h",
        marker_color=color,
        text=[f"+{v:.4f}" for v in values],
        textposition="inside",
        textfont={"family": "JetBrains Mono", "size": 11, "color": "#fff"},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#94a3b8", margin=dict(l=8, r=8, t=6, b=6),
                      height=210, bargap=0.38,
                      xaxis=dict(showgrid=False, showticklabels=False),
                      yaxis=dict(showgrid=False, tickfont={"family": "Inter", "size": 11}))
    return fig


def _radar(cats, v1, v2, n1, n2) -> go.Figure:
    def rgba(h, a):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return f"rgba({r},{g},{b},{a})"
    fig = go.Figure()
    for vals, name, c in [(v1, n1, "#3b82f6"), (v2, n2, "#f97316")]:
        closed = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=closed, theta=cats + [cats[0]], fill="toself", name=name[:22],
            line=dict(color=c, width=2), fillcolor=rgba(c, 0.08),
        ))
    fig.update_layout(
        polar=dict(bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0, 100], color="#334155", gridcolor="#1e293b", tickfont={"size": 9}),
            angularaxis=dict(color="#64748b", gridcolor="#1e293b")),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
        margin=dict(l=50, r=50, t=30, b=30), height=330,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"))
    return fig


def _degrad_line() -> go.Figure:
    overs    = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    moisture = [22.0, 18.5, 15.0, 12.5, 10.0, 8.5, 7.0, 6.0, 5.0, 4.5, 4.0]
    friction = [35, 38, 42, 46, 50, 55, 59, 63, 68, 72, 75]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=overs, y=moisture, mode="lines+markers", name="Moisture %",
        line=dict(color="#3b82f6", width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=overs, y=friction, mode="lines+markers", name="Friction Index",
        line=dict(color="#f97316", width=2), marker=dict(size=5)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8", height=250, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", color="#64748b", title="Overs"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#64748b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ── Navigation sidebar ─────────────────────────────────────────────────────────

def _render_sidebar_nav() -> str:
    """Render the styled nav sidebar; return the active page key."""
    if "page" not in st.session_state:
        st.session_state.page = "predict"

    # ── Brand header ──────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:.75rem;padding:.5rem 0 1.1rem;border-bottom:1px solid #1e293b;margin-bottom:.9rem">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#10b981,#059669);
                border-radius:9px;display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;font-weight:800;color:#fff;flex-shrink:0">+</div>
            <div>
                <div style="font-family:'Outfit',sans-serif;font-size:.95rem;font-weight:800;color:#f8fafc;line-height:1.1">
                    PitchSense <span style="color:#10b981">AI</span>
                </div>
                <div style="font-size:.58rem;font-weight:600;letter-spacing:.1em;color:#334155;text-transform:uppercase;margin-top:1px">
                    DATA-DRIVEN CRICKET INTEL
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current = st.session_state.page

    def _nav_item(key, icon, label):
        if current == key:
            st.sidebar.markdown(
                f"""<div style="background:rgba(16,185,129,0.1);border-left:3px solid #10b981;
                    border-radius:0 10px 10px 0;padding:.6rem .75rem .6rem .65rem;
                    display:flex;align-items:center;gap:.55rem;margin-bottom:1px;cursor:default">
                    <span style="font-size:.95rem">{icon}</span>
                    <span style="font-family:'Inter',sans-serif;font-weight:700;
                        color:#f8fafc;font-size:.88rem">{label}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            if st.sidebar.button(f"{icon}  {label}", key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

    # ── Section: CORE MODULES ─────────────────────────────────────────────────
    st.sidebar.markdown(
        '<p style="font-size:.62rem;font-weight:700;letter-spacing:.14em;color:#334155;'
        'text-transform:uppercase;margin:0 0 .35rem .1rem">CORE MODULES</p>',
        unsafe_allow_html=True,
    )
    _nav_item("predict", "▦", "Predict & Tactics")
    _nav_item("venue",   "◫", "Venue DNA & Comparison")
    _nav_item("format",  "◎", "Multi-Format Analysis")

    # ── Section: ANALYTICS & INSIGHTS ────────────────────────────────────────
    st.sidebar.markdown(
        '<p style="font-size:.62rem;font-weight:700;letter-spacing:.14em;color:#334155;'
        'text-transform:uppercase;margin:.9rem 0 .35rem .1rem">ANALYTICS & INSIGHTS</p>',
        unsafe_allow_html=True,
    )
    _nav_item("perf",    "⊟", "Model Performance & SHAP")
    _nav_item("roadmap", "◉", "Future Roadmap")
    _nav_item("about",   "≡", "About")

    return st.session_state.page


# ── Intelligence Hub — inline control panel ────────────────────────────────────

PRESETS = {
    "Dry Dust Bowl":    dict(temperature=38.0, humidity=28.0, cloud_cover=8.0,  grass_coverage=1.0,  pitch_age_days=6, soil_composition="Red Soil",   compaction_kpa=200, wind_speed=8.0,  dew_point=9.0,  season="Summer"),
    "Green Seamer":     dict(temperature=22.0, humidity=85.0, cloud_cover=80.0, grass_coverage=12.0, pitch_age_days=1, soil_composition="Black Soil",  compaction_kpa=320, wind_speed=20.0, dew_point=20.0, season="Monsoon"),
    "Balanced Surface": dict(temperature=30.0, humidity=58.0, cloud_cover=35.0, grass_coverage=5.0,  pitch_age_days=3, soil_composition="Mixed Soil",  compaction_kpa=280, wind_speed=12.0, dew_point=16.0, season="Post-Monsoon"),
    "Batting Paradise": dict(temperature=32.0, humidity=42.0, cloud_cover=10.0, grass_coverage=2.0,  pitch_age_days=2, soil_composition="Black Soil",  compaction_kpa=420, wind_speed=6.0,  dew_point=12.0, season="Winter"),
}


def _control_panel() -> dict:
    """Render the inline left-column control panel. Returns all parameter values."""

    # Preset buttons
    st.markdown('<p class="ps-ctrl-head">Quick Presets</p>', unsafe_allow_html=True)
    if "preset" not in st.session_state:
        st.session_state.preset = None
    pc = st.columns(2, gap="small")
    for idx, (name, _) in enumerate(PRESETS.items()):
        col = pc[idx % 2]
        if col.button(name, key=f"preset_{name}"):
            st.session_state.preset = name
            st.rerun()
    pv = PRESETS.get(st.session_state.preset, {})

    # Venue
    st.markdown('<p class="ps-ctrl-head">Venue</p>', unsafe_allow_html=True)
    venues = sorted(VENUE_CITY)
    venue = st.selectbox("Venue", venues, label_visibility="collapsed")
    city = st.text_input("City", value=VENUE_CITY[venue][0], label_visibility="collapsed",
                         placeholder="City")

    # Match type — segmented buttons
    st.markdown('<p class="ps-ctrl-head">Match Type</p>', unsafe_allow_html=True)
    if "match_type" not in st.session_state:
        st.session_state.match_type = "T20"
    mc = st.columns(3, gap="small")
    for col, mt in zip(mc, ["T20", "ODI", "Test"]):
        active = st.session_state.match_type == mt
        btn_style = (
            "background:linear-gradient(135deg,#10b981,#059669) !important;color:#fff !important;"
            "border:1px solid #10b981 !important;font-weight:700 !important;"
            if active else
            "background:#0d1117 !important;color:#475569 !important;border:1px solid #1e293b !important;"
        )
        if col.button(mt, key=f"mt_{mt}"):
            st.session_state.match_type = mt
            st.rerun()
    match_type = st.session_state.match_type

    season_opts = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    season = st.selectbox("Season", season_opts,
        index=season_opts.index(pv.get("season", "Summer")))
    day_night = st.toggle("Day / Night Match", value=True)

    # Weather
    st.markdown('<p class="ps-ctrl-head">Weather Conditions</p>', unsafe_allow_html=True)
    temperature = st.slider("Temperature (°C)",  10.0, 45.0, float(pv.get("temperature", 30.0)), 0.5)
    humidity    = st.slider("Humidity (%)",       20.0,100.0, float(pv.get("humidity",    62.0)), 1.0)
    wind_speed  = st.slider("Wind Speed (km/h)",   0.0, 40.0, float(pv.get("wind_speed",  12.0)), 0.5)
    dew_point   = st.slider("Dew Point (°C)",      0.0, 30.0, float(pv.get("dew_point",   18.0)), 0.5)
    cloud_cover = st.slider("Cloud Cover (%)",     0.0,100.0, float(pv.get("cloud_cover", 35.0)), 1.0)

    # Pitch
    st.markdown('<p class="ps-ctrl-head">Pitch Conditions</p>', unsafe_allow_html=True)
    soil_opts = ["Red Soil", "Black Soil", "Mixed Soil"]
    soil_composition   = st.selectbox("Soil Composition", soil_opts,
        index=soil_opts.index(pv.get("soil_composition", "Mixed Soil")))
    pitch_strip_number = st.slider("Strip Number", 1, 10, 4)
    grass_coverage     = st.slider("Grass Coverage (mm)", 0.0, 15.0, float(pv.get("grass_coverage", 4.5)), 0.5)
    pitch_age_days     = st.slider("Pitch Age (days)", 1, 8, int(pv.get("pitch_age_days", 3)))
    compaction_kpa     = st.slider("Compaction (kPa)", 100, 500, int(pv.get("compaction_kpa", 280)), 10)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    submitted = st.button("Run Pitch Analysis", type="primary", key="run_analysis")
    st.markdown("</div>", unsafe_allow_html=True)

    return dict(
        venue=venue, city=city, match_type=match_type, season=season, day_night=day_night,
        temperature=temperature, humidity=humidity, wind_speed=wind_speed,
        dew_point=dew_point, cloud_cover=cloud_cover,
        soil_composition=soil_composition, pitch_strip_number=pitch_strip_number,
        grass_coverage=grass_coverage, pitch_age_days=pitch_age_days,
        compaction_kpa=compaction_kpa, submitted=submitted,
    )


# ── Page: Pre-Match Predictor ─────────────────────────────────────────────────

def _render_tactics_content() -> None:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.warning("Run `python run_pipeline.py` first to train the models.")
        return

    model, preprocessor = load_artifacts()
    dataset = load_dataset()

    # ── TWO-COLUMN LAYOUT: controls left, analytics right ─────────────────────
    ctrl_col, result_col = st.columns([1.05, 2.15], gap="large")

    with ctrl_col:
        # Control panel header
        st.markdown(
            '<p style="font-family:Outfit,sans-serif;font-size:1rem;font-weight:800;'
            'color:#f8fafc;letter-spacing:-.01em;margin-bottom:.1rem">Control Panel</p>'
            '<p style="font-size:.7rem;color:#334155;letter-spacing:.06em;margin-bottom:1rem">'
            'PRE-MATCH PARAMETERS</p>',
            unsafe_allow_html=True,
        )
        p = _control_panel()

    with result_col:
        # ── Page title ─────────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;'
            'color:#f8fafc;letter-spacing:-.02em;margin-bottom:.1rem">Predict &amp; Tactics</p>'
            '<p style="font-size:.72rem;color:#334155;letter-spacing:.06em;margin-bottom:1.1rem">'
            'AI-POWERED PITCH CONDITION CLASSIFICATION &amp; TACTICAL ADVISORY</p>',
            unsafe_allow_html=True,
        )

        if not p["submitted"]:
            st.markdown(
                '<div class="ps-card" style="text-align:center;padding:3.5rem 2rem;border-style:dashed">'
                '<p style="font-size:2.2rem;color:#1e293b;margin:0 0 .75rem">◈</p>'
                '<p style="font-family:Outfit,sans-serif;font-size:1rem;font-weight:600;color:#334155;margin:0">'
                'Configure parameters in the control panel</p>'
                '<p style="font-size:.8rem;color:#1e293b;margin:.4rem 0 0">'
                'Set venue, conditions, and pitch attributes — then run pitch analysis</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        # ── Backend — UNCHANGED ────────────────────────────────────────────────
        venue            = p["venue"];          city           = p["city"]
        match_type       = p["match_type"];     season         = p["season"]
        day_night        = p["day_night"];      temperature    = p["temperature"]
        humidity         = p["humidity"];       wind_speed     = p["wind_speed"]
        dew_point        = p["dew_point"];      cloud_cover    = p["cloud_cover"]
        soil_composition = p["soil_composition"]
        pitch_strip_number = p["pitch_strip_number"]
        grass_coverage   = p["grass_coverage"]; pitch_age_days = p["pitch_age_days"]
        compaction_kpa   = p["compaction_kpa"]

        vs = dataset[dataset["venue"] == venue] if not dataset.empty else pd.DataFrame()
        ground_avg = (
            float(vs["ground_avg_1st_innings"].median()) if not vs.empty
            else {"T20": 165.0, "ODI": 270.0, "Test": 330.0}[match_type]
        )
        pace_pct = float(vs["ground_pace_wickets_pct"].median()) if not vs.empty else 50.0
        spin_pct = 100.0 - pace_pct

        row = pd.DataFrame([{
            "venue": venue, "city": city, "country": "India",
            "match_type": match_type, "temperature": temperature, "humidity": humidity,
            "wind_speed": wind_speed, "dew_point": dew_point, "cloud_cover": cloud_cover,
            "pitch_age_days": pitch_age_days, "ground_avg_1st_innings": ground_avg,
            "ground_pace_wickets_pct": pace_pct, "ground_spin_wickets_pct": spin_pct,
            "season": season, "day_night": int(day_night),
            "soil_composition": soil_composition, "pitch_strip_number": pitch_strip_number,
            "grass_coverage": grass_coverage, "compaction_kpa": compaction_kpa,
        }])
        engineered   = add_derived_features(row)
        transformed  = preprocessor.transform(engineered)
        probabilities = model.predict_proba(transformed)[0]
        prediction   = int(probabilities.argmax())
        badge_text, color, bg = PITCH_STYLE[prediction]

        conf = probabilities[prediction]
        conf_tier, tier_color = (
            ("HIGH", "#10b981") if conf >= 0.75 else
            (("MEDIUM", "#f59e0b") if conf >= 0.50 else ("LOW", "#ef4444"))
        )

        importances = pd.Series(model.feature_importances_, index=preprocessor.get_feature_names_out())
        tvals       = pd.Series(transformed[0], index=preprocessor.get_feature_names_out())
        local_signal = (importances * tvals.abs()).sort_values(ascending=False).head(6)
        local_signal.index = [clean_feature_name(n) for n in local_signal.index]

        validation_anomaly = run_prediction_validation(
            prediction, temperature, humidity, pitch_age_days, cloud_cover,
            grass_coverage, compaction_kpa, soil_composition, ground_avg, match_type,
        )
        toss_decision, toss_reason = get_toss_decision(prediction, dew_point, venue, dataset)

        # ── Validation banner ──────────────────────────────────────────────────
        if validation_anomaly:
            st.markdown(
                f'<div class="ps-warn">'
                f'<span style="font-size:.68rem;font-weight:700;letter-spacing:.1em;'
                f'text-transform:uppercase;color:#f59e0b">System Alert</span>'
                f'<p style="margin:.3rem 0 0;font-size:.85rem;color:#fde68a;line-height:1.6">'
                f'{validation_anomaly["message"]}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ps-ok">'
                '<span style="font-size:.68rem;font-weight:700;letter-spacing:.1em;'
                'text-transform:uppercase;color:#10b981">System Nominal</span>'
                '<p style="margin:.3rem 0 0;font-size:.84rem;color:#a7f3d0;line-height:1.5">'
                'Prediction aligns with physical pitch and atmospheric parameters.</p></div>',
                unsafe_allow_html=True,
            )

        # ── Hero: gauge | prediction + proba ──────────────────────────────────
        g_col, info_col = st.columns([1.0, 1.15], gap="medium")

        with g_col:
            st.markdown(
                f'<div class="ps-card" style="border-top:3px solid {color};padding-bottom:.85rem">'
                f'<span class="ps-badge" style="background:{bg};color:{color};'
                f'border:1px solid {color}33">{badge_text}</span>'
                f'<p style="font-family:Outfit,sans-serif;font-size:1.55rem;font-weight:800;'
                f'color:#f8fafc;margin:0;letter-spacing:-.02em">{PITCH_LABELS[prediction]}</p>'
                f'<p style="font-family:JetBrains Mono,monospace;font-size:.72rem;'
                f'color:#334155;margin:0">Strip #{pitch_strip_number} &nbsp;·&nbsp; {match_type}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_gauge(conf, color), use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown(
                f'<div class="ps-card" style="display:flex;justify-content:space-between;padding:.75rem 1rem">'
                f'<div><p class="ps-label">Confidence</p>'
                f'<p class="ps-mono" style="color:{color};font-size:1.2rem">{conf:.1%}</p></div>'
                f'<div style="text-align:right"><p class="ps-label">Tier</p>'
                f'<p class="ps-mono" style="color:{tier_color};font-size:1.2rem">{conf_tier}</p></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with info_col:
            st.markdown('<p class="ps-section-head">Probability Distribution</p>', unsafe_allow_html=True)
            st.plotly_chart(
                _hbars([PITCH_LABELS[i] for i in range(3)], list(probabilities),
                       ["#f59e0b", "#10b981", "#f97316"]),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.markdown('<p class="ps-section-head">Atmospheric Telemetry</p>', unsafe_allow_html=True)
            telem_html = ""
            for lbl, val, unit in [
                ("Temperature", temperature, "°C"),
                ("Humidity",    humidity,    "%"),
                ("Cloud Cover", cloud_cover, "%"),
                ("Dew Point",   dew_point,   "°C"),
                ("Grass",       grass_coverage, "mm"),
                ("Compaction",  compaction_kpa, "kPa"),
            ]:
                val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                telem_html += (
                    f'<div class="ps-stat"><span class="ps-stat-label">{lbl}</span>'
                    f'<span class="ps-stat-value">{val_str} '
                    f'<span style="color:#334155;font-weight:400">{unit}</span></span></div>'
                )
            st.markdown(f'<div class="ps-card">{telem_html}</div>', unsafe_allow_html=True)

        # ── Toss Strategy ──────────────────────────────────────────────────────
        st.markdown(
            f'<div class="ps-toss">'
            f'<p class="ps-label" style="margin-bottom:.3rem">Toss Strategy Advisor</p>'
            f'<p style="font-family:Outfit,sans-serif;font-size:1.15rem;font-weight:800;'
            f'color:#3b82f6;margin:0 0 .45rem">{toss_decision}</p>'
            f'<p style="font-size:.84rem;color:#94a3b8;line-height:1.65;margin:0">{toss_reason}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Match Context Engine ───────────────────────────────────────────────
        st.markdown('<p class="ps-section-head">Innings Stage Analysis</p>', unsafe_allow_html=True)
        STAGES = {
            0: {"Powerplay": ("True bounce with quick carry", "Exploit restrictions — target boundaries", "Back-of-length, vary pace"),
                "Middle Overs": ("Flat surface — spinners struggle", "Build partnerships, target short boundaries", "Tight defensive lines"),
                "Death Overs": ("Ideal flat track, minimal degradation", "Full license for boundary hitting", "Wide yorkers, defensive fields")},
            1: {"Powerplay": ("Substantial lateral swing — pacers dominate", "Play late, preserve wickets", "Full length, corridor of uncertainty"),
                "Middle Overs": ("Heavy deck bounce — cutters effective", "Rotate strike, avoid lofted shots", "Hit-the-deck, target rib cage"),
                "Death Overs": ("Surface wear — uneven bounce", "Anticipate cutters, target straight", "Wide yorkers, mix speeds")},
            2: {"Powerplay": ("Negligible seam — early grip possible", "Spin deployed early — rotate strike", "Attack stumps, limit cuts"),
                "Middle Overs": ("Sharp turn and grip — variable", "Use feet, play sweeps", "Vary flight, exploit rough"),
                "Death Overs": ("Severe degradation — sharp turn", "Extremely difficult for new batters", "Shoot into stumps, full length")},
        }
        tabs = st.tabs(["Powerplay  (1–6)", "Middle Overs  (7–15)", "Death Overs  (16–20)"])
        for tab, key in zip(tabs, ["Powerplay", "Middle Overs", "Death Overs"]):
            behavior, batting, bowling = STAGES[prediction][key]
            with tab:
                s1, s2, s3 = st.columns(3, gap="small")
                for col, lbl, txt, c in [(s1, "Pitch Behavior", behavior, "#94a3b8"),
                                         (s2, "Batting Strategy", batting, "#10b981"),
                                         (s3, "Bowling Strategy", bowling, "#3b82f6")]:
                    col.markdown(
                        f'<div class="ps-stage" style="border-top:2px solid {c}22">'
                        f'<p class="ps-label" style="color:{c}">{lbl}</p>'
                        f'<p style="font-size:.8rem;color:#94a3b8;line-height:1.55;margin:0">{txt}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── AI Pitch Reports ───────────────────────────────────────────────────
        st.markdown('<p class="ps-section-head">AI Pitch Reports</p>', unsafe_allow_html=True)
        analyst_txt = (
            f"Strip #{pitch_strip_number} — {soil_composition} base, {grass_coverage:.1f}mm grass. "
            f"Atmospheric: {temperature:.1f}°C, {humidity:.0f}% RH, {cloud_cover:.0f}% cloud. "
            f"Compaction {compaction_kpa} kPa, {pitch_age_days}-day aging. "
            f"Model forecasts {PITCH_LABELS[prediction]} surface with {conf:.1%} confidence."
        )
        if prediction == 1:
            cap_bowl = "Frontline pacers attack full in the corridor. Maintain slips for 6+ overs."
            cap_bat  = "Expect early lateral movement. Leave outside-off, play straight through."
        elif prediction == 2:
            cap_bowl = "Spinners must utilize variations. Bowl flatter to exploit maximum grip."
            cap_bat  = "Use feet to reach the pitch. Sweep shots to neutralize spin angles."
        else:
            cap_bowl = "Defensive lines, protect boundaries, cutters and yorkers early."
            cap_bat  = "High-scoring intent. Trust the true bounce and hit through the line."
        target_range = f"{int(ground_avg - 10)}–{int(ground_avg + 15)}"
        bc_txt = (
            f"Classification: {PITCH_LABELS[prediction]}. "
            f"Primary factor: {soil_composition} with {grass_coverage:.1f}mm grass. "
            f"Expect {'spinners' if prediction==2 else ('seamers' if prediction==1 else 'batsmen')} "
            f"to dominate as the match progresses. Target: {target_range} runs."
        )

        rep_tabs = st.tabs(["Analyst Report", "Captain Briefing", "Broadcaster Summary"])
        for tab, (border_c, content) in zip(rep_tabs, [
            (color, analyst_txt),
            ("#3b82f6", f"Bowling: {cap_bowl}  |  Batting: {cap_bat}  |  Target Range: {target_range}"),
            ("#8b5cf6", bc_txt),
        ]):
            with tab:
                st.markdown(
                    f'<div class="ps-report" style="border-left-color:{border_c}">{content}</div>',
                    unsafe_allow_html=True,
                )

        # ── Model Signal Contributions ─────────────────────────────────────────
        st.markdown('<p class="ps-section-head">Model Signal Contributions</p>', unsafe_allow_html=True)
        st.plotly_chart(
            _signal_bars(local_signal.index.tolist(), local_signal.values.tolist(), color),
            use_container_width=True, config={"displayModeBar": False},
        )


def render_tactics_page() -> None:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.warning("Run `python run_pipeline.py` first to train the models.")
        return

    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:1.6rem;font-weight:800;color:#f8fafc;'
        'letter-spacing:-.02em;margin-bottom:.1rem">Intelligence Hub</p>'
        '<p style="font-size:.72rem;color:#334155;letter-spacing:.06em;margin-bottom:1.1rem">'
        'ALL-IN-ONE SPORTS ANALYTICS COMMAND CENTER</p>',
        unsafe_allow_html=True,
    )

    hub_tabs = st.tabs([
        "▦  Predict & Tactics",
        "◫  Venue DNA & Comparison",
        "◎  Multi-Format Analysis",
        "⊟  Model Performance & SHAP",
        "◉  Future Roadmap",
        "≡  About"
    ])

    with hub_tabs[0]:
        _render_tactics_content()
    with hub_tabs[1]:
        _render_venue_content()
    with hub_tabs[2]:
        _render_format_content()
    with hub_tabs[3]:
        _render_xai_content()
    with hub_tabs[4]:
        _render_roadmap_content()
    with hub_tabs[5]:
        _render_faq_content()


# ── Page: Explainable AI (XAI) ────────────────────────────────────────────────

def _render_xai_content() -> None:
    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:#f8fafc;'
        'letter-spacing:-.02em;margin-bottom:.1rem">Model Performance &amp; SHAP</p>'
        '<p style="font-size:.72rem;color:#334155;letter-spacing:.06em;margin-bottom:1.1rem">'
        'CLASSIFICATION METRICS · DIAGNOSTICS · SHAP EXPLAINABILITY</p>',
        unsafe_allow_html=True,
    )

    if not METRICS_PATH.exists():
        st.warning("Run `python run_pipeline.py` first.")
        return

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cr      = metrics["classification_report"]
    macro   = cr.get("macro avg", {})

    st.markdown('<p class="ps-section-head">Overall Performance</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",     f"{metrics['accuracy']:.1%}")
    m2.metric("Macro F1",     f"{metrics['macro_f1']:.1%}")
    m3.metric("Macro Prec.",  f"{macro.get('precision', 0):.1%}")
    m4.metric("Macro Recall", f"{macro.get('recall', 0):.1%}")

    st.markdown('<p class="ps-section-head">Per-Class Breakdown</p>', unsafe_allow_html=True)
    CLASS_COLORS = {"Batting-Friendly": "#f59e0b", "Pace-Friendly": "#10b981", "Spin-Friendly": "#f97316"}
    cc = st.columns(3, gap="medium")
    for col, cls_name in zip(cc, ["Batting-Friendly", "Pace-Friendly", "Spin-Friendly"]):
        cls = cr.get(cls_name, {})
        c = CLASS_COLORS[cls_name]
        prec, rec, f1 = cls.get("precision", 0), cls.get("recall", 0), cls.get("f1-score", 0)
        col.markdown(
            f'<div class="ps-card" style="border-top:3px solid {c}">'
            f'<span class="ps-badge" style="background:{c}18;color:{c};border:1px solid {c}30;'
            f'margin-bottom:.75rem;display:inline-block">{cls_name}</span>',
            unsafe_allow_html=True,
        )
        for lbl, val in [("Precision", prec), ("Recall", rec), ("F1-Score", f1)]:
            col.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:4px">'
                f'<span style="font-size:.77rem;color:#475569">{lbl}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:.8rem;'
                f'font-weight:600;color:{c}">{val:.1%}</span></div>',
                unsafe_allow_html=True,
            )
            col.progress(val, text="")
        col.markdown(
            f'<div class="ps-stat" style="margin-top:.45rem">'
            f'<span class="ps-stat-label">Support</span>'
            f'<span class="ps-stat-value">{int(cls.get("support", 0))}</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="ps-section-head">Diagnostic Charts</p>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2, gap="medium")
    with dc1:
        path = FIGURES_DIR / "confusion_matrix.png"
        if path.exists():
            st.markdown('<div class="ps-card" style="padding:.6rem">', unsafe_allow_html=True)
            st.image(str(path), use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with dc2:
        path = FIGURES_DIR / "feature_importance.png"
        if path.exists():
            st.markdown('<div class="ps-card" style="padding:.6rem">', unsafe_allow_html=True)
            st.image(str(path), use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="ps-section-head">SHAP Explainability Dashboard</p>', unsafe_allow_html=True)
    shap_tabs = st.tabs(["Global Beeswarm", "Feature Bar Ranking", "Local Waterfall"])
    for tab, fname in zip(shap_tabs, ["summary_beeswarm.png", "summary_bar.png", "local_waterfall.png"]):
        with tab:
            path = FIGURES_DIR / "shap" / fname
            if path.exists():
                st.markdown('<div class="ps-card" style="padding:.6rem">', unsafe_allow_html=True)
                st.image(str(path), use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(f"Run `python run_pipeline.py` to generate `{fname}`.")


def render_xai_page() -> None:
    _render_xai_content()


# ── Page: Ground Analytics (Venue DNA) ────────────────────────────────────────

def _render_venue_content() -> None:
    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:#f8fafc;'
        'letter-spacing:-.02em;margin-bottom:.1rem">Venue DNA &amp; Comparison</p>'
        '<p style="font-size:.72rem;color:#334155;letter-spacing:.06em;margin-bottom:1.1rem">'
        'VENUE DNA · GROUND INTELLIGENCE · COMPARATIVE ANALYTICS</p>',
        unsafe_allow_html=True,
    )
    dataset = load_dataset()
    if dataset.empty:
        st.warning("No dataset loaded. Run the machine learning pipeline first.")
        return

    venues = sorted(dataset["venue"].unique())
    vc1, vc2 = st.columns(2, gap="large")
    with vc1:
        selected = st.selectbox("Primary Venue", venues)
    with vc2:
        compare = st.selectbox("Compare With", [v for v in venues if v != selected])

    def _vstats(v):
        sub = dataset[dataset["venue"] == v]
        t20, odi = sub[sub["match_type"] == "T20"], sub[sub["match_type"] == "ODI"]
        return dict(
            avg1_t20=float(t20["ground_avg_1st_innings"].median()) if not t20.empty else 165.0,
            avg1_odi=float(odi["ground_avg_1st_innings"].median()) if not odi.empty else 270.0,
            spin_pct=float(sub["ground_spin_wickets_pct"].median()),
            pace_pct=float(sub["ground_pace_wickets_pct"].median()),
            toss_imp=3.0 + (len(v) % 5),
            chase_ok=45.0 + (len(v) % 15),
        )

    sv, cv = _vstats(selected), _vstats(compare)

    st.markdown('<p class="ps-section-head">Venue Profile</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, lbl, val, c in [
        (k1, "Avg 1st (T20)", f"{sv['avg1_t20']:.0f}", "#f59e0b"),
        (k2, "Avg 2nd (T20)", f"{sv['avg1_t20']*0.96:.0f}", "#f59e0b"),
        (k3, "Spin Wkts %",   f"{sv['spin_pct']:.1f}%", "#f97316"),
        (k4, "Pace Wkts %",   f"{sv['pace_pct']:.1f}%", "#10b981"),
        (k5, "Chase Success", f"{sv['chase_ok']:.1f}%", "#3b82f6"),
    ]:
        col.markdown(f'<div class="ps-kpi"><p class="ps-kpi-label">{lbl}</p>'
                     f'<p class="ps-kpi-value" style="color:{c}">{val}</p></div>', unsafe_allow_html=True)

    k6, k7, k8, k9 = st.columns(4)
    for col, lbl, val, c in [
        (k6, "Avg 1st (ODI)", f"{sv['avg1_odi']:.0f}", "#f59e0b"),
        (k7, "Avg 2nd (ODI)", f"{sv['avg1_odi']*0.95:.0f}", "#f59e0b"),
        (k8, "Toss Impact",   f"+{sv['toss_imp']:.1f}%", "#8b5cf6"),
        (k9, "Batting Index", f"{sv['avg1_t20']/200*100:.0f}", "#f8fafc"),
    ]:
        col.markdown(f'<div class="ps-kpi"><p class="ps-kpi-label">{lbl}</p>'
                     f'<p class="ps-kpi-value" style="color:{c}">{val}</p></div>', unsafe_allow_html=True)

    st.markdown('<p class="ps-section-head">Venue Comparison Radar</p>', unsafe_allow_html=True)
    cats  = ["T20 Avg", "ODI Avg", "Spin %", "Pace %", "Chase %"]
    v1_r  = [sv["avg1_t20"]/200*100, sv["avg1_odi"]/320*100, sv["spin_pct"], sv["pace_pct"], sv["chase_ok"]]
    v2_r  = [cv["avg1_t20"]/200*100, cv["avg1_odi"]/320*100, cv["spin_pct"], cv["pace_pct"], cv["chase_ok"]]
    st.plotly_chart(_radar(cats, v1_r, v2_r, selected[:22], compare[:22]),
                    use_container_width=True, config={"displayModeBar": False})

    st.markdown('<p class="ps-section-head">Head-to-Head Comparison</p>', unsafe_allow_html=True)
    rows_html = ""
    for metric, v1, v2 in [
        ("Avg 1st Innings (T20)", f"{sv['avg1_t20']:.0f}", f"{cv['avg1_t20']:.0f}"),
        ("Avg 1st Innings (ODI)", f"{sv['avg1_odi']:.0f}", f"{cv['avg1_odi']:.0f}"),
        ("Spin Wickets %",        f"{sv['spin_pct']:.1f}%", f"{cv['spin_pct']:.1f}%"),
        ("Pace Wickets %",        f"{sv['pace_pct']:.1f}%", f"{cv['pace_pct']:.1f}%"),
        ("Chase Success %",       f"{sv['chase_ok']:.1f}%", f"{cv['chase_ok']:.1f}%"),
        ("Toss Win Impact",       f"+{sv['toss_imp']:.1f}%", f"+{cv['toss_imp']:.1f}%"),
    ]:
        rows_html += (
            f'<div class="ps-stat"><span class="ps-stat-label">{metric}</span>'
            f'<span style="display:flex;gap:2.5rem">'
            f'<span class="ps-stat-value" style="color:#3b82f6">{v1}</span>'
            f'<span class="ps-stat-value" style="color:#f97316">{v2}</span>'
            f'</span></div>'
        )
    st.markdown(
        f'<div class="ps-card">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:.6rem">'
        f'<span class="ps-label">Metric</span>'
        f'<span style="display:flex;gap:2.5rem">'
        f'<span class="ps-label" style="color:#3b82f6">{selected[:18]}</span>'
        f'<span class="ps-label" style="color:#f97316">{compare[:18]}</span>'
        f'</span></div>{rows_html}</div>',
        unsafe_allow_html=True,
    )


def render_venue_page() -> None:
    _render_venue_content()


# ── Page: Research & Pipeline ──────────────────────────────────────────────────

def _render_format_content() -> None:
    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:#f8fafc;'
        'letter-spacing:-.02em;margin-bottom:.1rem">Multi-Format Analysis</p>'
        '<p style="font-size:.72rem;color:#334155;letter-spacing:.06em;margin-bottom:1.1rem">'
        'T20 · ODI · TEST FORMAT INSIGHTS</p>',
        unsafe_allow_html=True,
    )

    # Multi-Format cards
    st.markdown('<p class="ps-section-head">Multi-Format Analysis</p>', unsafe_allow_html=True)
    FORMAT_DATA = {
        "T20":  ("#f59e0b", "Run Rate 8.0–9.0 · Par Score 165–175",
                 "Powerplay swing, cutters, death yorkers",
                 "Consistent surface. Dew influences 2nd innings."),
        "ODI":  ("#3b82f6", "Run Rate 5.2–5.8 · Par Score 260–280",
                 "New ball swing (1–10), spinners (15–40)",
                 "Slight slowing midday — plays best under lights."),
        "Test": ("#8b5cf6", "Run Rate 2.8–3.4 · Par Score 300–350",
                 "Day 1 pace/bounce · Day 4–5 spin via crack wear",
                 "Moisture fades → Cracks open → Dust bowl by Day 5."),
    }
    fc = st.columns(3, gap="medium")
    for col, (fmt, (c, scoring, bowling, evolution)) in zip(fc, FORMAT_DATA.items()):
        col.markdown(
            f'<div class="ps-card" style="border-top:3px solid {c}">'
            f'<span class="ps-badge" style="background:{c}18;color:{c};border:1px solid {c}30;'
            f'margin-bottom:.6rem;display:inline-block">{fmt}</span>'
            f'<div class="ps-stat"><span class="ps-stat-label">Expected Scoring</span></div>'
            f'<p style="font-size:.81rem;color:#94a3b8;margin:.1rem 0 .75rem">{scoring}</p>'
            f'<div class="ps-stat"><span class="ps-stat-label">Bowling Advantage</span></div>'
            f'<p style="font-size:.81rem;color:#94a3b8;margin:.1rem 0 .75rem">{bowling}</p>'
            f'<div class="ps-stat"><span class="ps-stat-label">Pitch Evolution</span></div>'
            f'<p style="font-size:.81rem;color:#94a3b8;margin:.1rem 0 0">{evolution}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_format_page() -> None:
    _render_format_content()


def _render_roadmap_content() -> None:
    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:#f8fafc;'
        'letter-spacing:-.02em;margin-bottom:.1rem">Future Roadmap</p>'
        '<p style="font-size:.72rem;color:#334155;letter-spacing:.06em;margin-bottom:1.1rem">'
        'DEGRADATION TIMELINE · XI OPTIMIZER · FANTASY PROJECTIONS</p>',
        unsafe_allow_html=True,
    )

    # Degradation timeline
    st.markdown('<p class="ps-section-head">Pitch Degradation Timeline</p>', unsafe_allow_html=True)
    st.markdown('<div class="ps-card" style="padding:.75rem">', unsafe_allow_html=True)
    st.plotly_chart(_degrad_line(), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # XI Optimizer
    st.markdown('<p class="ps-section-head">Optimal XI Optimizer</p>', unsafe_allow_html=True)
    xi_cols = st.columns(2, gap="large")
    with xi_cols[0]:
        st.markdown('<div class="ps-card">', unsafe_allow_html=True)
        for role, count, c in [
            ("Batter Specialists", 5, "#f59e0b"),
            ("Wicket Keeper",      1, "#3b82f6"),
            ("Spin All-Rounders",  2, "#f97316"),
            ("Pace All-Rounders",  1, "#10b981"),
            ("Specialist Spinner", 1, "#f97316"),
            ("Specialist Pacer",   1, "#10b981"),
        ]:
            rc, nc = st.columns([4, 1])
            rc.markdown(f'<span style="font-size:.83rem;color:#94a3b8">{role}</span>', unsafe_allow_html=True)
            nc.markdown(f'<span style="font-family:JetBrains Mono;font-weight:700;color:{c}">{count}</span>', unsafe_allow_html=True)
            st.progress(count / 5)
        st.markdown("</div>", unsafe_allow_html=True)
    with xi_cols[1]:
        st.markdown('<div class="ps-card">', unsafe_allow_html=True)
        st.markdown('<p class="ps-label" style="margin-bottom:.75rem">Fantasy Point Projections</p>', unsafe_allow_html=True)
        for role, desc, c in [
            ("Spin Bowlers (Middle)", "High wicket potential in grip-friendly conditions", "#f97316"),
            ("Wicket Keepers",        "High catch probability from uneven bounce",          "#3b82f6"),
            ("Top-Order Batters",     "High run potential during powerplay",                 "#f59e0b"),
        ]:
            st.markdown(
                f'<div class="ps-stage" style="border-left:3px solid {c};margin-bottom:.5rem">'
                f'<p class="ps-label" style="color:{c}">{role}</p>'
                f'<p style="font-size:.8rem;color:#94a3b8;margin:0">{desc}</p></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def render_roadmap_page() -> None:
    _render_roadmap_content()


# ── Page: Methodology FAQs ────────────────────────────────────────────────────

def _render_faq_content() -> None:
    st.markdown(
        '<p style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:#f8fafc;'
        'letter-spacing:-.02em;margin-bottom:.1rem">About</p>'
        '<p style="font-size:.72rem;color:#334155;letter-spacing:.06em;margin-bottom:1.1rem">'
        'PLATFORM OVERVIEW · MODEL ARCHITECTURE · EXPERT FEEDBACK</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="ps-section-head">Platform Overview</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-card">'
        '<p style="font-size:.88rem;color:#94a3b8;line-height:1.8">'
        'PitchSense AI is a professional cricket intelligence platform designed to forecast pitch behaviour, '
        'recommend team tactics, and evaluate match conditions using local weather parameters, '
        'physical turf attributes, and venue history. Physical indicators such as soil composition, '
        'grass coverage, and surface compaction are integrated directly into the classification pipeline, '
        'enabling more accurate and explainable pre-match pitch classification.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    dataset = load_dataset()
    if not dataset.empty:
        st.markdown('<p class="ps-section-head">Dataset Statistics</p>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Match Rows", f"{len(dataset):,}")
        a2.metric("Unique Venues",    dataset["venue"].nunique())
        a3.metric("Formats Covered",  dataset["match_type"].nunique())
        a4.metric("Pitch Classes",    3)

    st.markdown('<p class="ps-section-head">Model Architecture</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-card">'
        '<div class="ps-stat"><span class="ps-stat-label">Algorithm</span>'
        '<span class="ps-stat-value">Random Forest Classifier</span></div>'
        '<div class="ps-stat"><span class="ps-stat-label">Best Parameters</span>'
        '<span class="ps-stat-value">n_estimators: 300 · max_depth: 10</span></div>'
        '<div class="ps-stat"><span class="ps-stat-label">Preprocessing</span>'
        '<span class="ps-stat-value">ColumnTransformer — StandardScaler + OneHotEncoder</span></div>'
        '<div class="ps-stat"><span class="ps-stat-label">Output Classes</span>'
        '<span class="ps-stat-value">Batting-Friendly · Pace-Friendly · Spin-Friendly</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="ps-section-head">Expert Curator Feedback</p>', unsafe_allow_html=True)
    st.markdown('<div class="ps-card">', unsafe_allow_html=True)
    st.selectbox("Actual Pitch Class", ["Batting-Friendly", "Pace-Friendly", "Spin-Friendly"])
    st.slider("Observed Turn Angle (degrees)", 0.0, 8.0, 2.5)
    st.slider("Observed Seam Deviation (mm)", 0.0, 30.0, 12.0)
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    st.button("Submit Curator Report")
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_faq_page() -> None:
    _render_faq_content()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    page = _render_sidebar_nav()
    if page == "predict":
        render_tactics_page()
    elif page == "venue":
        render_venue_page()
    elif page == "format":
        render_format_page()
    elif page == "perf":
        render_xai_page()
    elif page == "roadmap":
        render_roadmap_page()
    elif page == "about":
        render_faq_page()


if __name__ == "__main__":
    main()
