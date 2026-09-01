"""
🌬️ Sindh Air Quality Index — 3-Day ML Forecast Dashboard
===========================================================
Ultra-modern dark-mode 72-hour AQI forecasting dashboard
for 5 cities in Sindh, Pakistan.

Features:
  - Permanent Sleek Dark / Cyberpunk Aesthetic
  - Single 3-Day Average Hero Gauge + Daily Forecast Breakdown (Day 1, 2, 3)
  - Pure Machine Learning Forecasting (RandomForest Model v5)
  - Interactive Dark Cartography & SHAP Model Telemetry

Run with: streamlit run app.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

from utils import calculate_epa_aqi, get_aqi_category

warnings.filterwarnings('ignore')
load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sindh AQI — 3-Day ML Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CONSTANTS & DARK PALETTES
# ─────────────────────────────────────────────
CITIES = {
    "Karachi":   {"lat": 24.8607, "lon": 67.0011, "emoji": "🏙️"},
    "Hyderabad": {"lat": 25.3960, "lon": 68.3578, "emoji": "🌇"},
    "Jamshoro":  {"lat": 25.4300, "lon": 68.2800, "emoji": "🏛️"},
    "Nawabshah": {"lat": 26.2442, "lon": 68.4100, "emoji": "🌾"},
    "Sukkur":    {"lat": 27.7052, "lon": 68.8574, "emoji": "🌅"},
}

POLLUTANTS = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3', 'no']
ROLLING_POLLUTANTS = ['pm2_5', 'pm10', 'co', 'o3']
LAG_HOURS = [1, 6, 12, 24]

# Pure Dark-Mode Color Palette (Glowing High-Contrast)
AQI_THEME = {
    "Good": {"color": "#10b981", "bg": "rgba(16,185,129,0.18)", "border": "rgba(16,185,129,0.5)", "text": "#34d399"},
    "Moderate": {"color": "#f59e0b", "bg": "rgba(245,158,11,0.18)", "border": "rgba(245,158,11,0.5)", "text": "#fbbf24"},
    "Unhealthy for Sensitive Groups": {"color": "#f97316", "bg": "rgba(249,115,22,0.18)", "border": "rgba(249,115,22,0.5)", "text": "#fb923c"},
    "Unhealthy": {"color": "#ef4444", "bg": "rgba(239,68,68,0.18)", "border": "rgba(239,68,68,0.5)", "text": "#f87171"},
    "Very Unhealthy": {"color": "#a855f7", "bg": "rgba(168,85,247,0.18)", "border": "rgba(168,85,247,0.5)", "text": "#c084fc"},
    "Hazardous": {"color": "#e11d48", "bg": "rgba(225,29,72,0.22)", "border": "rgba(225,29,72,0.6)", "text": "#fda4af"},
}

AQI_HEALTH_MESSAGES = {
    "Good": "Atmospheric conditions nominal. Ideal for all outdoor operations. 🌳",
    "Moderate": "Acceptable air composition. Sensitive personnel should monitor prolonged exposure. 😷",
    "Unhealthy for Sensitive Groups": "Elevated particulate density. Sensitive groups should reduce outdoor activity. ⚠️",
    "Unhealthy": "Unhealthy atmospheric envelope. Filtration masks advised, restrict outdoor transit. 🚨",
    "Very Unhealthy": "Hazardous particulate saturation. Limit environmental exposure immediately. 🔴",
    "Hazardous": "CRITICAL ENVIRONMENTAL ALERT: Extreme toxicity levels. Maintain indoor containment. 🆘",
}

# ─────────────────────────────────────────────
# PURE DARK CSS (Sleek, High-Tech, Flawless)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean layout */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    body, .stApp {
        background-color: #0b1120 !important;
        background-image: 
            radial-gradient(at 15% 15%, rgba(2, 132, 199, 0.08) 0px, transparent 50%),
            radial-gradient(at 85% 85%, rgba(168, 85, 247, 0.08) 0px, transparent 50%) !important;
        color: #f8fafc !important;
    }

    .main .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Top HUD Bar */
    .top-hud-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    .hud-status-badge {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.74rem;
        letter-spacing: 1.2px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #38bdf8;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse-glow 2s infinite ease-in-out;
    }
    @keyframes pulse-glow {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.6; }
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        padding: 1.8rem 2rem;
        border-radius: 22px;
        margin-bottom: 1.4rem;
        text-align: center;
    }
    .hero-header h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    .hero-header p {
        margin: 0.35rem 0 0 0;
        font-size: 1.02rem;
        font-weight: 400;
        color: #94a3b8 !important;
    }
    .hud-chips-row {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 0.85rem;
    }
    .hud-chip {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.8px;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #38bdf8;
    }

    /* City Selector Container */
    .city-selector-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.4rem;
    }
    div[data-testid="stPills"] {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
    }
    div[data-testid="stPills"] button {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.6rem !important;
        border-radius: 35px !important;
        background: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stPills"] button:hover {
        background: #334155 !important;
        color: #ffffff !important;
    }
    div[data-testid="stPills"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.45) !important;
        transform: scale(1.04);
    }

    /* Clean Card Boxes */
    .card-box {
        background: #111827;
        border: 1px solid #1f2937;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        border-radius: 22px;
        padding: 1.5rem 1.4rem;
        height: 100%;
    }
    .hero-top-badge {
        font-family: 'SF Mono', 'Fira Code', monospace;
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }

    /* Daily Breakdown Telemetry Cards */
    .daily-breakdown-container {
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
        height: 100%;
        justify-content: space-between;
    }
    .day-row-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 1.1rem 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    .day-row-card:hover {
        background: #273549;
        border-color: #475569;
        transform: translateX(4px);
    }
    .day-label {
        font-size: 1.05rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }
    .day-date {
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .day-aqi-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 1.15rem;
        border-radius: 25px;
        font-weight: 800;
        font-size: 1.15rem;
    }

    /* Health Alert Banner */
    .health-alert {
        border-radius: 18px;
        padding: 1.2rem 1.6rem;
        margin: 1.4rem 0;
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .health-alert-icon {
        font-size: 2.2rem;
        flex-shrink: 0;
    }

    /* Stat Pill Row */
    .stat-pill-row {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin-top: 0.6rem;
    }
    .stat-pill {
        background: #1e293b;
        border: 1px solid #334155;
        color: #f1f5f9;
        padding: 0.45rem 0.95rem;
        border-radius: 16px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    /* Section Header */
    .section-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 1.8rem 0 0.9rem 0;
        padding-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 9px 22px;
        font-size: 0.95rem;
        color: #94a3b8 !important;
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        background: #1e293b !important;
        border-color: #0284c7 !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Hopsworks & initializing neural weights...")
def load_model():
    """Load the trained RandomForest model and feature names."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "aqi_model_dir")

    try:
        import hopsworks
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
            mr = project.get_model_registry()
            hw_model = mr.get_model("aqi_prediction_model", version=None)  # latest
            hw_model_dir = hw_model.download()
            model = joblib.load(os.path.join(hw_model_dir, "aqi_model.pkl"))
            feature_names = joblib.load(os.path.join(hw_model_dir, "feature_names.pkl"))
            return model, feature_names, hw_model.version
    except Exception:
        pass

    if os.path.exists(os.path.join(model_dir, "aqi_model.pkl")):
        model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
        return model, feature_names, "local"

    return None, None, None


@st.cache_data(ttl=3600, show_spinner="Synchronizing atmospheric telemetry from Hopsworks Cloud...")
def load_data():
    """Load historical baseline + latest cloud features."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dfs = []

    hist_path = os.path.join(base_dir, "feature_store", "aqi_historical.parquet")
    if os.path.exists(hist_path):
        hist_df = pd.read_parquet(hist_path)
        dfs.append(hist_df)

    try:
        import hopsworks
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
            dataset_api = project.get_dataset_api()
            latest_path = dataset_api.download(
                "Resources/latest_features.parquet", overwrite=True
            )
            if os.path.exists(latest_path):
                live_df = pd.read_parquet(latest_path)
                dfs.append(live_df)
    except Exception:
        pass

    local_latest = os.path.join(base_dir, "feature_store", "latest_features.parquet")
    if os.path.exists(local_latest):
        local_live = pd.read_parquet(local_latest)
        dfs.append(local_live)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="last")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING & 3-DAY MULTI-STEP FORECASTING
# ─────────────────────────────────────────────
def engineer_features(df):
    """Engineer features identical to training pipeline for inference."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(by=['city', 'timestamp'])

    for pol in POLLUTANTS:
        if pol in df.columns:
            for lag in LAG_HOURS:
                df[f'{pol}_lag_{lag}h'] = df.groupby('city')[pol].shift(lag)

    for pol in ROLLING_POLLUTANTS:
        if pol in df.columns:
            grouped = df.groupby('city')[pol]
            df[f'{pol}_roll_mean_24h'] = grouped.transform(
                lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
            )
            df[f'{pol}_roll_std_24h'] = grouped.transform(
                lambda x: x.shift(1).rolling(window=24, min_periods=1).std()
            )

    for pol in ROLLING_POLLUTANTS:
        lag_col = f'{pol}_lag_24h'
        if lag_col in df.columns:
            df[f'{pol}_roc_24h'] = (df[pol] - df[lag_col]) / (df[lag_col].abs() + 1e-6)

    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.fillna(0)
    df = pd.get_dummies(df, columns=['city'], drop_first=False)
    return df


def predict_3_days_for_city(model, feature_names, df_engineered, city_name):
    """Generate 3-day (+24h, +48h, +72h) PM2.5 and EPA AQI predictions."""
    city_col = f"city_{city_name}"
    if city_col not in df_engineered.columns:
        return None

    city_data = df_engineered[df_engineered[city_col] == True].copy()
    if city_data.empty:
        city_data = df_engineered[df_engineered[city_col] == 1].copy()
    if city_data.empty:
        return None

    latest_row = city_data.iloc[-1:].copy()
    
    base_time = pd.to_datetime(latest_row['timestamp'].values[0])
    if pd.isna(base_time):
        base_time = datetime.now()

    forecasts = []
    current_features = latest_row.copy()
    
    for day in range(1, 4):
        target_date = base_time + timedelta(days=day)
        date_str = target_date.strftime("%a, %b %d")
        
        available = [f for f in feature_names if f in current_features.columns]
        missing = [f for f in feature_names if f not in current_features.columns]

        X = current_features[available].copy()
        for col in missing:
            X[col] = 0
        X = X[feature_names]

        pred_pm25 = max(0.0, float(model.predict(X)[0]))
        epa_aqi, _ = calculate_epa_aqi(pm2_5=pred_pm25, pm10=0, co=0, no2=0, o3=0, so2=0)
        category, _ = get_aqi_category(epa_aqi)
        theme_cfg = AQI_THEME.get(category, AQI_THEME["Moderate"])

        forecasts.append({
            "day": day,
            "day_name": "Tomorrow" if day == 1 else f"Day {day}",
            "step_label": f"+{day*24}h",
            "date_str": date_str,
            "pm2_5": pred_pm25,
            "epa_aqi": epa_aqi,
            "category": category,
            "theme": theme_cfg,
            "health_msg": AQI_HEALTH_MESSAGES.get(category, "")
        })

        # Autoregressive update
        current_features['pm2_5_lag_24h'] = current_features['pm2_5']
        current_features['pm2_5'] = pred_pm25
        current_features['pm2_5_lag_1h'] = pred_pm25
        current_features['pm2_5_lag_6h'] = pred_pm25
        current_features['pm2_5_lag_12h'] = pred_pm25
        
        if 'pm2_5_roll_mean_24h' in current_features.columns:
            current_features['pm2_5_roll_mean_24h'] = (current_features['pm2_5_roll_mean_24h'] * 0.7) + (pred_pm25 * 0.3)
        
        next_hour = target_date.hour
        next_month = target_date.month
        current_features['hour_sin'] = np.sin(2 * np.pi * next_hour / 24)
        current_features['hour_cos'] = np.cos(2 * np.pi * next_hour / 24)
        current_features['month_sin'] = np.sin(2 * np.pi * next_month / 12)
        current_features['month_cos'] = np.cos(2 * np.pi * next_month / 12)

    return forecasts


# ─────────────────────────────────────────────
# UI & PLOTLY HELPER FUNCTIONS
# ─────────────────────────────────────────────
def render_hero_gauge(value, category_text, theme_cfg):
    """Render a high-contrast dark semicircular AQI gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 60, "color": theme_cfg["color"], "family": "Inter, sans-serif"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1.5, "tickcolor": "#94a3b8",
                     "tickvals": [0, 50, 100, 150, 200, 300, 500],
                     "ticktext": ["0", "50", "100", "150", "200", "300", "500"],
                     "tickfont": {"color": "#94a3b8", "size": 11}},
            "bar": {"color": theme_cfg["color"], "thickness": 0.38},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [50, 100], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [100, 150], "color": "rgba(249, 115, 22, 0.2)"},
                {"range": [150, 200], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [200, 300], "color": "rgba(168, 85, 247, 0.2)"},
                {"range": [300, 500], "color": "rgba(225, 29, 72, 0.2)"},
            ],
            "threshold": {
                "line": {"color": theme_cfg["color"], "width": 4},
                "thickness": 0.88,
                "value": value
            }
        }
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=15, r=15, t=15, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"}
    )
    return fig


def render_3day_trajectory(forecasts, city_name):
    """Render dark-mode 3-day AQI prediction trajectory area chart."""
    days = [f"{f['day_name']} ({f['date_str']})" for f in forecasts]
    aqis = [f["epa_aqi"] for f in forecasts]
    pm25s = [f["pm2_5"] for f in forecasts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days,
        y=aqis,
        mode='lines+markers+text',
        name='Predicted AQI',
        text=[f"AQI {a}" for a in aqis],
        textposition="top center",
        textfont=dict(size=13, weight="bold", color="#f8fafc"),
        line=dict(color="#38bdf8", width=3.5, shape='spline'),
        marker=dict(size=12, color="#38bdf8", line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor="rgba(56, 189, 248, 0.15)",
        hovertemplate="<b>%{x}</b><br>Predicted EPA AQI: %{y}<br>Predicted PM2.5: %{customdata:.1f} µg/m³<extra></extra>",
        customdata=pm25s
    ))

    # EPA Threshold lines
    fig.add_hline(y=50, line_dash="dot", line_color="#10b981", annotation_text="Good (50)", annotation_position="bottom right")
    fig.add_hline(y=100, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate (100)", annotation_position="bottom right")
    fig.add_hline(y=150, line_dash="dash", line_color="#f97316", annotation_text="Unhealthy (150)", annotation_position="bottom right")

    fig.update_layout(
        title=dict(text=f"72-Hour Air Quality Forecast Curve — {city_name}", font=dict(size=16, color="#f8fafc")),
        yaxis_title="Predicted EPA AQI",
        height=320,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor="#334155", range=[0, max(max(aqis) * 1.35, 170)], tickfont=dict(color="#f8fafc")),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#f8fafc")),
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    return fig


def render_city_forecast_map(city_forecasts_dict):
    """Create an interactive dark cartographic map showing 3-day forecast markers for all cities."""
    lats, lons, names, avg_aqis, colors, sizes, texts = [], [], [], [], [], [], []

    for city_name, info in CITIES.items():
        fc = city_forecasts_dict.get(city_name, [])
        if fc:
            avg_aqi = round(sum(f['epa_aqi'] for f in fc) / len(fc))
            avg_cat, _ = get_aqi_category(avg_aqi)
            theme_cfg = AQI_THEME.get(avg_cat, AQI_THEME["Moderate"])
            
            lats.append(info["lat"])
            lons.append(info["lon"])
            names.append(city_name)
            avg_aqis.append(avg_aqi)
            colors.append(theme_cfg["color"])
            sizes.append(max(26, min(avg_aqi / 4.0, 60)))
            
            text = (
                f"<b>{city_name} [3-DAY AI FORECAST]</b><br>"
                f"• <b>3-Day Avg AQI: {avg_aqi} ({avg_cat})</b><br>"
                f"• {fc[0]['day_name']}: AQI {fc[0]['epa_aqi']} ({fc[0]['category']})<br>"
                f"• {fc[1]['day_name']}: AQI {fc[1]['epa_aqi']} ({fc[1]['category']})<br>"
                f"• {fc[2]['day_name']}: AQI {fc[2]['epa_aqi']} ({fc[2]['category']})"
            )
            texts.append(text)

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, opacity=0.92, sizemode='diameter'),
        text=names,
        textposition="top center",
        textfont=dict(size=12, family="Inter, sans-serif", color="white"),
        hovertext=texts,
        hoverinfo='text',
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=26.0, lon=68.0),
            zoom=6.2,
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    # ── Top HUD Status Bar ──
    st.markdown("""
    <div class="top-hud-bar">
        <div class="hud-status-badge">
            <span class="live-dot"></span>
            <span>NEURAL INFERENCE ACTIVE • SINDH REGION • 72H WINDOW</span>
        </div>
        <div class="hud-status-badge" style="color: #94a3b8;">
            <span>HOPSWORKS MODEL REGISTRY (RF-v5)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🌬️ Sindh Air Quality — 3-Day ML Forecast</h1>
        <p>72-hour machine learning predictions across Sindh, powered by RandomForest AI (Model v5)</p>
        <div class="hud-chips-row">
            <span class="hud-chip">MODEL: RF-V5 (300 TREES)</span>
            <span class="hud-chip">ACCURACY: R² 74.7%</span>
            <span class="hud-chip">HOPSWORKS FEATURE STORE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data & Model ──
    model, feature_names, model_version = load_model()
    raw_df = load_data()

    if raw_df is None or raw_df.empty:
        st.error("❌ No air quality data found. Please run the feature pipeline first.")
        st.stop()

    if model is None or feature_names is None:
        st.error("❌ Model not loaded from Hopsworks Model Registry. Please run the training pipeline first.")
        st.stop()

    # ── Engineer features for inference ──
    df_eng = engineer_features(raw_df)

    # ── City Selector (Pill Buttons) ──
    city_options = list(CITIES.keys())
    
    st.markdown('<div class="city-selector-container">', unsafe_allow_html=True)
    selected_city = st.pills(
        "Select City",
        city_options,
        format_func=lambda c: f"{CITIES[c]['emoji']}  {c}",
        default="Karachi",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_city:
        selected_city = "Karachi"

    # ── Compute 3-Day Predictions for All Cities ──
    all_city_forecasts = {}
    for city_name in city_options:
        fc = predict_3_days_for_city(model, feature_names, df_eng, city_name)
        if fc:
            all_city_forecasts[city_name] = fc

    selected_forecasts = all_city_forecasts.get(selected_city)
    if not selected_forecasts or len(selected_forecasts) < 3:
        st.error(f"❌ Unable to generate 3-day forecast for {selected_city}.")
        st.stop()

    # Calculate 3-Day Averages
    avg_aqi = round(sum(f['epa_aqi'] for f in selected_forecasts) / len(selected_forecasts))
    avg_pm25 = sum(f['pm2_5'] for f in selected_forecasts) / len(selected_forecasts)
    avg_category, _ = get_aqi_category(avg_aqi)
    avg_theme = AQI_THEME.get(avg_category, AQI_THEME["Moderate"])

    # Trend calculation (Day 3 vs Day 1)
    diff_aqi = selected_forecasts[2]['epa_aqi'] - selected_forecasts[0]['epa_aqi']
    if diff_aqi <= -5:
        trend_label = f"📉 Improving (↓ {abs(diff_aqi)} AQI)"
        trend_color = "#10b981"
    elif diff_aqi >= 5:
        trend_label = f"📈 Increasing (↑ {diff_aqi} AQI)"
        trend_color = "#ef4444"
    else:
        trend_label = "➡️ Stable (±3 AQI)"
        trend_color = "#38bdf8"

    # ── Top Row: Single 3-Day Average Hero Gauge + Daily Breakdown Cards ──
    col_left, col_right = st.columns([1.15, 1.25])

    with col_left:
        st.markdown(f"""
        <div class="card-box" style="border-top: 5px solid {avg_theme['color']}; text-align: center; box-shadow: 0 0 35px {avg_theme['color']}25;">
            <div>
                <span class="hero-top-badge" style="background: {avg_theme['bg']}; color: {avg_theme['text']}; border: 1.5px solid {avg_theme['border']};">
                    ⚡ 3-DAY AVERAGE PREDICTION
                </span>
                <div class="card-title" style="font-size: 1.35rem; font-weight: 800; margin-top: 0.3rem;">
                    {CITIES[selected_city]['emoji']} {selected_city}, Sindh
                </div>
            </div>
        """, unsafe_allow_html=True)

        fig_hero = render_hero_gauge(avg_aqi, avg_category, avg_theme)
        st.plotly_chart(fig_hero, use_container_width=True)

        st.markdown(f"""
            <div style="margin-top: -10px;">
                <div style="font-size: 1.5rem; font-weight: 800; color: {avg_theme['text']}; margin-bottom: 0.4rem;">
                    {avg_category}
                </div>
                <div class="stat-pill-row">
                    <span class="stat-pill">💨 Avg PM2.5: <b>{avg_pm25:.1f} µg/m³</b></span>
                    <span class="stat-pill" style="color: {trend_color};">Trend: <b>{trend_label}</b></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="daily-breakdown-container">', unsafe_allow_html=True)
        
        for fc in selected_forecasts:
            t = fc['theme']
            st.markdown(f"""
            <div class="day-row-card" style="border-left: 6px solid {t['color']};">
                <div>
                    <div class="day-label">{fc['day_name']} ({fc['step_label']})</div>
                    <div class="day-date">{fc['date_str']}</div>
                </div>
                <div style="display: flex; align-items: center; gap: 18px;">
                    <div style="text-align: right;">
                        <span style="font-size: 0.78rem; color: #94a3b8;">Predicted PM2.5</span><br>
                        <span style="font-size: 1.05rem; font-weight: 800; color: #f8fafc;">{fc['pm2_5']:.1f} µg/m³</span>
                    </div>
                    <div class="day-aqi-pill" style="background: {t['bg']}; color: {t['text']}; border: 1.5px solid {t['border']};">
                        AQI {fc['epa_aqi']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 3-Day Health Advisory Banner ──
    highest_risk_day = max(selected_forecasts, key=lambda x: x['epa_aqi'])
    risk_t = highest_risk_day['theme']
    risk_cat = highest_risk_day['category']
    risk_msg = highest_risk_day['health_msg']
    alert_icon = "✅" if highest_risk_day['epa_aqi'] <= 50 else "⚠️" if highest_risk_day['epa_aqi'] <= 150 else "🚨" if highest_risk_day['epa_aqi'] <= 300 else "🆘"

    st.markdown(f"""
    <div class="health-alert" style="background: {risk_t['bg']}; border: 1.5px solid {risk_t['border']}; border-left: 6px solid {risk_t['color']}; color: {risk_t['text']};">
        <span class="health-alert-icon">{alert_icon}</span>
        <div>
            <strong style="font-size: 1.05rem;">3-Day Health Advisory for {selected_city}:</strong> Peak risk predicted on <b>{highest_risk_day['day_name']} ({highest_risk_day['date_str']})</b> with an AQI of <b>{highest_risk_day['epa_aqi']} ({risk_cat})</b>.<br>
            <span style="opacity: 0.95; font-size: 0.95rem;">{risk_msg}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: 3-Day Trajectory | Regional Map | All Cities Comparison | Model Info ──
    st.markdown("")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 3-Day Trajectory", "🗺️ 5-City Forecast Map", "🏙️ All Cities Comparison", "🤖 AI Model & SHAP"
    ])

    with tab1:
        st.markdown('<div class="section-header">📈 3-Day Forecast Trajectory & Trends</div>', unsafe_allow_html=True)
        col_traj, col_hist = st.columns([1.2, 1])

        with col_traj:
            fig_traj = render_3day_trajectory(selected_forecasts, selected_city)
            st.plotly_chart(fig_traj, use_container_width=True)

        with col_hist:
            city_ts = raw_df[raw_df['city'] == selected_city].copy()
            city_ts['timestamp'] = pd.to_datetime(city_ts['timestamp'], unit='ms', errors='coerce')
            city_ts = city_ts.sort_values('timestamp')
            cutoff_7 = city_ts['timestamp'].max() - timedelta(days=7)
            city_ts_7 = city_ts[city_ts['timestamp'] >= cutoff_7]

            if not city_ts_7.empty and 'pm2_5' in city_ts_7.columns:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=city_ts_7['timestamp'], y=city_ts_7['pm2_5'],
                    mode='lines', name='PM2.5',
                    line=dict(color="#38bdf8", width=2.5),
                    fill='tozeroy', fillcolor="rgba(56, 189, 248, 0.15)",
                    hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
                ))
                fig_hist.update_layout(
                    title=dict(text=f"Historical PM2.5 Context (Past 7 Days)", font=dict(size=16, color="#f8fafc")),
                    yaxis_title="PM2.5 (µg/m³)",
                    height=320,
                    margin=dict(l=40, r=20, t=50, b=30),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                    xaxis=dict(gridcolor="#334155", tickfont=dict(color="#f8fafc")),
                    font=dict(family="Inter, sans-serif"),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">🗺️ Sindh Regional 3-Day Forecast Map</div>', unsafe_allow_html=True)
        st.caption("Pins represent 3-day average forecasted AQI. Hover over each city pin for full Day 1, Day 2, and Day 3 breakdown.")
        fig_map = render_city_forecast_map(all_city_forecasts)
        st.plotly_chart(fig_map, use_container_width=True)

        # 5-city forecast cards below map
        map_cols = st.columns(5)
        for i, city_name in enumerate(city_options):
            fc = all_city_forecasts.get(city_name, [])
            if fc:
                c_avg = round(sum(f['epa_aqi'] for f in fc) / len(fc))
                c_cat, _ = get_aqi_category(c_avg)
                t = AQI_THEME.get(c_cat, AQI_THEME["Moderate"])
                with map_cols[i]:
                    st.markdown(f"""
                    <div style="text-align:center; padding: 1rem 0.6rem; border-radius: 18px;
                                background: {t['bg']}; border: 1.5px solid {t['border']}; box-shadow: 0 4px 15px rgba(0,0,0,0.25);">
                        <div style="font-size: 1.3rem;">{CITIES[city_name]['emoji']}</div>
                        <div style="font-weight: 800; font-size: 1rem; color: #f8fafc; margin: 0.2rem 0;">{city_name}</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {t['text']};">Avg {c_avg}</div>
                        <div style="font-size: 0.76rem; color: {t['text']}; font-weight: 700;">{c_cat}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">🏙️ All 5 Cities — 3-Day Forecast Matrix</div>', unsafe_allow_html=True)
        
        matrix_data = []
        for city_name in city_options:
            fc = all_city_forecasts.get(city_name, [])
            if fc:
                c_avg = round(sum(f['epa_aqi'] for f in fc) / len(fc))
                matrix_data.append({
                    "City": f"{CITIES[city_name]['emoji']} {city_name}",
                    "3-Day Avg AQI": f"{c_avg} ({get_aqi_category(c_avg)[0]})",
                    "Day 1 Tomorrow (+24h)": f"{fc[0]['epa_aqi']} ({fc[0]['category']}) • {fc[0]['pm2_5']:.1f} µg/m³",
                    "Day 2 (+48h)": f"{fc[1]['epa_aqi']} ({fc[1]['category']}) • {fc[1]['pm2_5']:.1f} µg/m³",
                    "Day 3 (+72h)": f"{fc[2]['epa_aqi']} ({fc[2]['category']}) • {fc[2]['pm2_5']:.1f} µg/m³",
                })
        
        matrix_df = pd.DataFrame(matrix_data)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown('<div class="section-header">🤖 AI Model & Interpretability (SHAP)</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.success(f"✅ Active Model: RandomForest (Version {model_version})")
            st.markdown(f"""
            | Architecture Parameter | Value |
            |---|---|
            | **Model Type** | RandomForest Regressor |
            | **Ensemble Trees** | 300 Estimators |
            | **Max Depth** | 20 Levels |
            | **Feature Dimensions** | {len(feature_names)} Lagged & Rolling Features |
            | **Target Variable** | PM2.5 (24h, 48h, 72h Ahead) |
            | **Training Source** | Hopsworks Model Registry (Auto-Retrained Daily) |
            | **PM2.5 Accuracy** | **R² = 74.7%** (RMSE = 23.9 µg/m³) |
            """)

        with col_m2:
            st.markdown('**SHAP Feature Importance Summary**')
            shap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "images", "shap_summary.png")
            if os.path.exists(shap_path):
                st.image(shap_path, use_container_width=True)
            else:
                st.info("SHAP explanation plot is generated during the daily training pipeline.")

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.84rem; padding: 0.8rem;">
        <strong>Sindh Air Quality Index System</strong> • 72-Hour Predictions Generated by RandomForest AI •
        Powered by Hopsworks Cloud & GitHub Actions CI/CD
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
