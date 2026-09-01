"""
🌬️ Sindh Air Quality Index — 3-Day ML Forecast Dashboard
===========================================================
Ultra-modern 72-hour AQI forecasting dashboard for 5 cities in Sindh, Pakistan.

Features:
  - Light Mode (Default) & Dark / Night Mode Toggle
  - Single 3-Day Average Hero Gauge + Daily Forecast Breakdown (Day 1, 2, 3)
  - Pure Machine Learning Forecasting (RandomForest Model v5)
  - Apple Weather / Vercel-inspired UI Design
  - Interactive Sindh Regional Map & Comparison Matrix
  - SHAP Feature Importance & Model Interpretability

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
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# THEME STATE INITIALIZATION
# ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ─────────────────────────────────────────────
# CONSTANTS
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

AQI_COLORS = {
    "Good": "#10b981",                          # Emerald
    "Moderate": "#f59e0b",                      # Amber
    "Unhealthy for Sensitive Groups": "#f97316", # Orange
    "Unhealthy": "#ef4444",                     # Red
    "Very Unhealthy": "#8b5cf6",                # Purple
    "Hazardous": "#7f1d1d",                     # Maroon
}

AQI_HEALTH_MESSAGES = {
    "Good": "Air quality is predicted to be ideal. Perfect for outdoor activities! 🌳",
    "Moderate": "Air quality is acceptable. Sensitive individuals should take precautions during prolonged outdoor exertion. 😷",
    "Unhealthy for Sensitive Groups": "Children, elderly, and those with respiratory issues should reduce outdoor exposure. ⚠️",
    "Unhealthy": "Air quality is unhealthy for everyone. Wear masks and keep windows closed. 🚨",
    "Very Unhealthy": "Serious health risk! Avoid outdoor exertion and use indoor air purifiers if possible. 🔴",
    "Hazardous": "EMERGENCY HEALTH HAZARD: Severe pollution forecasted. Stay indoors! 🆘",
}

# ─────────────────────────────────────────────
# DYNAMIC THEME CSS
# ─────────────────────────────────────────────
is_dark = (st.session_state.theme == "dark")

if is_dark:
    theme_css = """
    /* Dark / Night Theme */
    body, .stApp {
        background-color: #0b1329 !important;
        color: #f1f5f9 !important;
    }
    .hero-header {
        background: linear-gradient(135deg, #07172b 0%, #0d2744 50%, #153e6b 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
    }
    .hero-header h1 {
        background: linear-gradient(135deg, #ffffff 0%, #b0d5ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-header p {
        color: #94a3b8;
    }
    .hero-gauge-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1.5px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .day-row-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1.5px solid rgba(255, 255, 255, 0.08);
    }
    .day-row-card:hover {
        background: rgba(30, 41, 59, 0.85);
        border-color: rgba(255, 255, 255, 0.18);
    }
    .day-label { color: #f8fafc; }
    .day-date { color: #94a3b8; }
    .day-pm25 { color: #cbd5e1; }
    .stat-pill {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #f1f5f9;
    }
    .health-alert {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stPills"] button {
        background: rgba(30, 41, 59, 0.7) !important;
        color: #e2e8f0 !important;
        border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
    }
    div[data-testid="stPills"] button:hover {
        background: rgba(51, 65, 85, 0.9) !important;
    }
    div[data-testid="stPills"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%) !important;
        color: white !important;
        border-color: #06b6d4 !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45) !important;
    }
    .section-header { color: #f8fafc; }
    """
else:
    theme_css = """
    /* Light Theme (Clean, Crisp, Apple Weather aesthetic) */
    body, .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    .hero-header {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #3b82f6 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
    }
    .hero-header h1 {
        color: #ffffff;
    }
    .hero-header p {
        color: #e0f2fe;
    }
    .hero-gauge-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }
    .day-row-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
    }
    .day-row-card:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
    }
    .day-label { color: #0f172a; }
    .day-date { color: #64748b; }
    .day-pm25 { color: #475569; }
    .stat-pill {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }
    .health-alert {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
        border: 1.5px solid #e2e8f0;
    }
    div[data-testid="stPills"] button {
        background: #ffffff !important;
        color: #334155 !important;
        border: 1.5px solid #cbd5e1 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stPills"] button:hover {
        background: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
    div[data-testid="stPills"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
    }
    .section-header { color: #0f172a; }
    """

st.markdown(f"""
<style>
    /* Clean layout */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}

    .main .block-container {{
        padding-top: 0.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* Top bar for Theme Toggle */
    .top-bar-container {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-bottom: 0.6rem;
    }}

    /* Hero Header */
    .hero-header {{
        padding: 1.8rem 2rem;
        border-radius: 24px;
        margin-bottom: 1.3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    .hero-header h1 {{
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    .hero-header p {{
        margin: 0.4rem 0 0 0;
        font-size: 1.05rem;
        font-weight: 400;
    }}

    /* City Selector Container */
    .city-selector-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 1.4rem;
    }}

    /* Modern Streamlit Pills styling */
    div[data-testid="stPills"] {{
        display: flex;
        justify-content: center;
        gap: 0.75rem;
    }}
    div[data-testid="stPills"] button {{
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.6rem !important;
        border-radius: 35px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    div[data-testid="stPills"] button[aria-selected="true"] {{
        transform: scale(1.05);
    }}

    /* Hero Average Card */
    .hero-gauge-card {{
        border-radius: 24px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }}
    .hero-badge {{
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 0.4rem 1.1rem;
        border-radius: 25px;
        margin-bottom: 0.6rem;
    }}

    /* Daily Breakdown Cards */
    .daily-breakdown-container {{
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
        height: 100%;
        justify-content: space-between;
    }}
    .day-row-card {{
        border-radius: 20px;
        padding: 1.1rem 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }}
    .day-row-card:hover {{
        transform: translateX(4px);
    }}
    .day-label {{
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }}
    .day-date {{
        font-size: 0.82rem;
    }}
    .day-aqi-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.45rem 1.1rem;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.15rem;
    }}
    .day-pm25 {{
        font-size: 0.82rem;
        text-align: right;
    }}

    /* Health Alert Banner */
    .health-alert {{
        border-radius: 20px;
        padding: 1.2rem 1.8rem;
        margin: 1.4rem 0;
        display: flex;
        align-items: center;
        gap: 18px;
        font-size: 1.02rem;
    }}
    .health-alert-icon {{
        font-size: 2.4rem;
        flex-shrink: 0;
    }}

    /* Metric pill summary */
    .stat-pill-row {{
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 0.4rem;
    }}
    .stat-pill {{
        padding: 0.45rem 0.95rem;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    /* Section Header */
    .section-header {{
        font-size: 1.3rem;
        font-weight: 800;
        margin: 1.8rem 0 0.9rem 0;
        padding-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 700;
    }}

    {theme_css}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Hopsworks & loading ML model...")
def load_model():
    """Load the trained RandomForest model and feature names."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "aqi_model_dir")

    # Try loading from Hopsworks Model Registry first
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

    # Fallback to local model
    if os.path.exists(os.path.join(model_dir, "aqi_model.pkl")):
        model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
        return model, feature_names, "local"

    return None, None, None


@st.cache_data(ttl=3600, show_spinner="Loading historical & cloud features...")
def load_data():
    """Load historical baseline + latest cloud features."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dfs = []

    # 1. Load bundled historical data
    hist_path = os.path.join(base_dir, "feature_store", "aqi_historical.parquet")
    if os.path.exists(hist_path):
        hist_df = pd.read_parquet(hist_path)
        dfs.append(hist_df)

    # 2. Try downloading latest live features from Hopsworks
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

    # 3. Check local latest_features
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

    # Lagged features
    for pol in POLLUTANTS:
        if pol in df.columns:
            for lag in LAG_HOURS:
                df[f'{pol}_lag_{lag}h'] = df.groupby('city')[pol].shift(lag)

    # Rolling statistics (24h)
    for pol in ROLLING_POLLUTANTS:
        if pol in df.columns:
            grouped = df.groupby('city')[pol]
            df[f'{pol}_roll_mean_24h'] = grouped.transform(
                lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
            )
            df[f'{pol}_roll_std_24h'] = grouped.transform(
                lambda x: x.shift(1).rolling(window=24, min_periods=1).std()
            )

    # Rate of change
    for pol in ROLLING_POLLUTANTS:
        lag_col = f'{pol}_lag_24h'
        if lag_col in df.columns:
            df[f'{pol}_roc_24h'] = (df[pol] - df[lag_col]) / (df[lag_col].abs() + 1e-6)

    # Cyclical time encoding
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.fillna(0)

    # One-hot encode city
    df = pd.get_dummies(df, columns=['city'], drop_first=False)

    return df


def predict_3_days_for_city(model, feature_names, df_engineered, city_name):
    """
    Generate 3-day (+24h, +48h, +72h) PM2.5 and EPA AQI predictions
    using autoregressive multi-step ML inference.
    """
    city_col = f"city_{city_name}"
    if city_col not in df_engineered.columns:
        return None

    city_data = df_engineered[df_engineered[city_col] == True].copy()
    if city_data.empty:
        city_data = df_engineered[df_engineered[city_col] == 1].copy()
    if city_data.empty:
        return None

    latest_row = city_data.iloc[-1:].copy()
    
    # Base timestamp
    base_time = pd.to_datetime(latest_row['timestamp'].values[0])
    if pd.isna(base_time):
        base_time = datetime.now()

    forecasts = []
    current_features = latest_row.copy()
    
    for day in range(1, 4):
        target_date = base_time + timedelta(days=day)
        date_str = target_date.strftime("%a, %b %d")
        
        # Build feature vector matching model training signature
        available = [f for f in feature_names if f in current_features.columns]
        missing = [f for f in feature_names if f not in current_features.columns]

        X = current_features[available].copy()
        for col in missing:
            X[col] = 0
        X = X[feature_names]

        # Predict PM2.5
        pred_pm25 = max(0.0, float(model.predict(X)[0]))
        
        # Calculate EPA AQI from predicted PM2.5
        epa_aqi, _ = calculate_epa_aqi(pm2_5=pred_pm25, pm10=0, co=0, no2=0, o3=0, so2=0)
        category, _ = get_aqi_category(epa_aqi)

        forecasts.append({
            "day": day,
            "day_name": "Tomorrow" if day == 1 else f"Day {day}",
            "step_label": f"+{day*24}h",
            "date_str": date_str,
            "pm2_5": pred_pm25,
            "epa_aqi": epa_aqi,
            "category": category,
            "color": AQI_COLORS.get(category, "#95a5a6"),
            "health_msg": AQI_HEALTH_MESSAGES.get(category, "")
        })

        # Update lag features for next day prediction autoregressively
        current_features['pm2_5_lag_24h'] = current_features['pm2_5']
        current_features['pm2_5'] = pred_pm25
        current_features['pm2_5_lag_1h'] = pred_pm25
        current_features['pm2_5_lag_6h'] = pred_pm25
        current_features['pm2_5_lag_12h'] = pred_pm25
        
        # Update rolling mean
        if 'pm2_5_roll_mean_24h' in current_features.columns:
            current_features['pm2_5_roll_mean_24h'] = (current_features['pm2_5_roll_mean_24h'] * 0.7) + (pred_pm25 * 0.3)
        
        # Advance cyclical time encoding for the target day
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
def render_hero_gauge(value, category, color, is_dark_mode=False):
    """Create a stunning primary semicircular AQI gauge."""
    tick_color = "rgba(255,255,255,0.4)" if is_dark_mode else "#64748b"
    bg_color = "rgba(255,255,255,0.04)" if is_dark_mode else "rgba(0,0,0,0.04)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 58, "color": color, "family": "Inter, sans-serif"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1.5, "tickcolor": tick_color,
                     "tickvals": [0, 50, 100, 150, 200, 300, 500],
                     "ticktext": ["0", "50", "100", "150", "200", "300", "500"]},
            "bar": {"color": color, "thickness": 0.36},
            "bgcolor": bg_color,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(16, 185, 129, 0.18)"},
                {"range": [50, 100], "color": "rgba(245, 158, 11, 0.18)"},
                {"range": [100, 150], "color": "rgba(249, 115, 22, 0.18)"},
                {"range": [150, 200], "color": "rgba(239, 68, 68, 0.18)"},
                {"range": [200, 300], "color": "rgba(139, 92, 246, 0.18)"},
                {"range": [300, 500], "color": "rgba(127, 29, 29, 0.18)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3.5},
                "thickness": 0.85,
                "value": value
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"}
    )
    return fig


def render_3day_trajectory(forecasts, city_name, is_dark_mode=False):
    """Render a smooth 3-day AQI prediction trajectory area chart."""
    days = [f"{f['day_name']} ({f['date_str']})" for f in forecasts]
    aqis = [f["epa_aqi"] for f in forecasts]
    pm25s = [f["pm2_5"] for f in forecasts]

    line_color = "#38bdf8" if is_dark_mode else "#2563eb"
    fill_color = "rgba(56, 189, 248, 0.15)" if is_dark_mode else "rgba(37, 99, 235, 0.12)"
    grid_color = "rgba(255, 255, 255, 0.08)" if is_dark_mode else "#f1f5f9"
    text_color = "#f8fafc" if is_dark_mode else "#0f172a"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days,
        y=aqis,
        mode='lines+markers+text',
        name='Predicted AQI',
        text=[f"AQI {a}" for a in aqis],
        textposition="top center",
        textfont=dict(size=13, weight="bold", color=text_color),
        line=dict(color=line_color, width=3.5, shape='spline'),
        marker=dict(size=12, color=line_color, line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate="<b>%{x}</b><br>Predicted EPA AQI: %{y}<br>Predicted PM2.5: %{customdata:.1f} µg/m³<extra></extra>",
        customdata=pm25s
    ))

    # EPA Thresholds
    fig.add_hline(y=50, line_dash="dot", line_color="#10b981", annotation_text="Good (50)", annotation_position="bottom right")
    fig.add_hline(y=100, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate (100)", annotation_position="bottom right")
    fig.add_hline(y=150, line_dash="dash", line_color="#f97316", annotation_text="Unhealthy (150)", annotation_position="bottom right")

    fig.update_layout(
        title=dict(text=f"72-Hour Air Quality Forecast Curve — {city_name}", font=dict(size=16, color=text_color)),
        yaxis_title="Predicted EPA AQI",
        height=320,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor=grid_color, range=[0, max(max(aqis) * 1.35, 170)], tickfont=dict(color=text_color)),
        xaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color)),
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    return fig


def render_city_forecast_map(city_forecasts_dict, is_dark_mode=False):
    """Create an interactive map showing 3-day forecast markers for all cities."""
    lats, lons, names, avg_aqis, colors, sizes, texts = [], [], [], [], [], [], []

    for city_name, info in CITIES.items():
        fc = city_forecasts_dict.get(city_name, [])
        if fc:
            avg_aqi = round(sum(f['epa_aqi'] for f in fc) / len(fc))
            avg_cat, _ = get_aqi_category(avg_aqi)
            clr = AQI_COLORS.get(avg_cat, "#95a5a6")
            
            lats.append(info["lat"])
            lons.append(info["lon"])
            names.append(city_name)
            avg_aqis.append(avg_aqi)
            colors.append(clr)
            sizes.append(max(24, min(avg_aqi / 4.2, 58)))
            
            text = (
                f"<b>{city_name} (3-Day AI Forecast)</b><br>"
                f"• <b>3-Day Avg AQI: {avg_aqi} ({avg_cat})</b><br>"
                f"• {fc[0]['day_name']}: AQI {fc[0]['epa_aqi']} ({fc[0]['category']})<br>"
                f"• {fc[1]['day_name']}: AQI {fc[1]['epa_aqi']} ({fc[1]['category']})<br>"
                f"• {fc[2]['day_name']}: AQI {fc[2]['epa_aqi']} ({fc[2]['category']})"
            )
            texts.append(text)

    map_style = "carto-darkmatter" if is_dark_mode else "open-street-map"

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, opacity=0.92, sizemode='diameter'),
        text=names,
        textposition="top center",
        textfont=dict(size=12, family="Inter, sans-serif", color="white" if is_dark_mode else "#0f172a"),
        hovertext=texts,
        hoverinfo='text',
    ))

    fig.update_layout(
        mapbox=dict(
            style=map_style,
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
    # ── Top Bar: Theme Switcher Toggle ──
    col_empty, col_theme = st.columns([5.5, 1.5])
    with col_theme:
        theme_choice = st.segmented_control(
            "Theme Mode",
            options=["☀️ Light", "🌙 Night"],
            default="🌙 Night" if is_dark else "☀️ Light",
            label_visibility="collapsed"
        )
        new_theme = "dark" if "Night" in str(theme_choice) else "light"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🌬️ Sindh Air Quality — 3-Day ML Forecast</h1>
        <p>72-hour machine learning predictions for 5 cities in Sindh, powered by RandomForest AI (Model v5)</p>
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
    avg_color = AQI_COLORS.get(avg_category, "#95a5a6")

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
        trend_color = "#3b82f6"

    # ── Top Row: Single 3-Day Average Hero Gauge + Daily Breakdown Cards ──
    col_left, col_right = st.columns([1.15, 1.25])

    with col_left:
        glow_style = f"box-shadow: 0 0 30px {avg_color}30;" if is_dark else f"box-shadow: 0 10px 30px {avg_color}18;"
        st.markdown(f"""
        <div class="hero-gauge-card" style="border-color: {avg_color}55; {glow_style}">
            <div>
                <span class="hero-badge" style="background: {avg_color}22; color: {avg_color}; border: 1.5px solid {avg_color}60;">
                    🤖 3-DAY AVERAGE AI FORECAST
                </span>
                <div style="font-size: 1.2rem; font-weight: 800; margin-top: 0.2rem;">
                    {CITIES[selected_city]['emoji']} {selected_city}, Sindh
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        fig_hero = render_hero_gauge(avg_aqi, avg_category, avg_color, is_dark_mode=is_dark)
        st.plotly_chart(fig_hero, use_container_width=True)

        st.markdown(f"""
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: {avg_color}; margin-bottom: 0.35rem;">
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
            st.markdown(f"""
            <div class="day-row-card" style="border-left: 6px solid {fc['color']};">
                <div>
                    <div class="day-label">{fc['day_name']} ({fc['step_label']})</div>
                    <div class="day-date">{fc['date_str']}</div>
                </div>
                <div style="display: flex; align-items: center; gap: 18px;">
                    <div class="day-pm25">
                        Predicted PM2.5<br><b>{fc['pm2_5']:.1f} µg/m³</b>
                    </div>
                    <div class="day-aqi-pill" style="background: {fc['color']}22; color: {fc['color']}; border: 1.5px solid {fc['color']}60;">
                        AQI {fc['epa_aqi']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 3-Day Health Advisory Banner ──
    highest_risk_day = max(selected_forecasts, key=lambda x: x['epa_aqi'])
    risk_color = highest_risk_day['color']
    risk_cat = highest_risk_day['category']
    risk_msg = highest_risk_day['health_msg']
    alert_icon = "✅" if highest_risk_day['epa_aqi'] <= 50 else "⚠️" if highest_risk_day['epa_aqi'] <= 150 else "🚨" if highest_risk_day['epa_aqi'] <= 300 else "🆘"

    st.markdown(f"""
    <div class="health-alert" style="background: {risk_color}16; border-left: 6px solid {risk_color};">
        <span class="health-alert-icon">{alert_icon}</span>
        <div>
            <strong>3-Day Health Advisory for {selected_city}:</strong> Peak pollution is expected on <b>{highest_risk_day['day_name']} ({highest_risk_day['date_str']})</b> with an AQI of <b>{highest_risk_day['epa_aqi']} ({risk_cat})</b>. {risk_msg}
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
            fig_traj = render_3day_trajectory(selected_forecasts, selected_city, is_dark_mode=is_dark)
            st.plotly_chart(fig_traj, use_container_width=True)

        with col_hist:
            city_ts = raw_df[raw_df['city'] == selected_city].copy()
            city_ts['timestamp'] = pd.to_datetime(city_ts['timestamp'], unit='ms', errors='coerce')
            city_ts = city_ts.sort_values('timestamp')
            cutoff_7 = city_ts['timestamp'].max() - timedelta(days=7)
            city_ts_7 = city_ts[city_ts['timestamp'] >= cutoff_7]

            if not city_ts_7.empty and 'pm2_5' in city_ts_7.columns:
                fig_hist = go.Figure()
                line_col = "#38bdf8" if is_dark else "#2563eb"
                fill_col = "rgba(56, 189, 248, 0.15)" if is_dark else "rgba(37, 99, 235, 0.1)"
                grid_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#f1f5f9"
                txt_col = "#f8fafc" if is_dark else "#0f172a"

                fig_hist.add_trace(go.Scatter(
                    x=city_ts_7['timestamp'], y=city_ts_7['pm2_5'],
                    mode='lines', name='PM2.5',
                    line=dict(color=line_col, width=2.5),
                    fill='tozeroy', fillcolor=fill_col,
                    hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
                ))
                fig_hist.update_layout(
                    title=dict(text=f"Historical PM2.5 Context (Past 7 Days)", font=dict(size=16, color=txt_col)),
                    yaxis_title="PM2.5 (µg/m³)",
                    height=320,
                    margin=dict(l=40, r=20, t=50, b=30),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor=grid_col, tickfont=dict(color=txt_col)),
                    xaxis=dict(gridcolor=grid_col, tickfont=dict(color=txt_col)),
                    font=dict(family="Inter, sans-serif"),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">🗺️ Sindh Regional 3-Day Forecast Map</div>', unsafe_allow_html=True)
        st.caption("Pins represent 3-day average forecasted AQI. Hover over each city pin for full Day 1, Day 2, and Day 3 breakdown.")
        fig_map = render_city_forecast_map(all_city_forecasts, is_dark_mode=is_dark)
        st.plotly_chart(fig_map, use_container_width=True)

        # 5-city forecast cards below map
        map_cols = st.columns(5)
        for i, city_name in enumerate(city_options):
            fc = all_city_forecasts.get(city_name, [])
            if fc:
                c_avg = round(sum(f['epa_aqi'] for f in fc) / len(fc))
                c_cat, _ = get_aqi_category(c_avg)
                c_clr = AQI_COLORS.get(c_cat, "#95a5a6")
                with map_cols[i]:
                    card_bg = f"{c_clr}15" if is_dark else "#ffffff"
                    card_border = f"{c_clr}40" if is_dark else "#e2e8f0"
                    st.markdown(f"""
                    <div style="text-align:center; padding: 0.9rem 0.5rem; border-radius: 18px;
                                background: {card_bg}; border: 1.5px solid {card_border}; box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
                        <div style="font-size: 1.3rem;">{CITIES[city_name]['emoji']}</div>
                        <div style="font-weight: 800; font-size: 0.98rem; margin: 0.2rem 0;">{city_name}</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: {c_clr};">Avg {c_avg}</div>
                        <div style="font-size: 0.75rem; color: {c_clr}; font-weight: 700;">{c_cat}</div>
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
    footer_color = "#94a3b8" if is_dark else "#64748b"
    st.markdown(f"""
    <div style="text-align: center; color: {footer_color}; font-size: 0.84rem; padding: 0.8rem;">
        <strong>Sindh Air Quality Index System</strong> • 72-Hour Predictions Generated by RandomForest AI •
        Powered by Hopsworks Cloud & GitHub Actions CI/CD
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
