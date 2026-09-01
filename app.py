"""
🌬️ Sindh Air Quality Index — 3-Day ML Forecast Dashboard
===========================================================
Modern, production-grade 72-hour AQI forecasting dashboard
for 5 cities in Sindh, Pakistan.

Inspired by Apple Weather, Linear, and Vercel design aesthetics.

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
    page_title="Sindh Air Quality Forecast",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CONSTANTS & COLOR PALETTE
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

# High-contrast production color tokens
AQI_THEME = {
    "Good": {
        "color": "#10b981",
        "bg": "rgba(16, 185, 129, 0.15)",
        "border": "rgba(16, 185, 129, 0.35)",
        "text": "#34d399",
        "label": "Good"
    },
    "Moderate": {
        "color": "#f59e0b",
        "bg": "rgba(245, 158, 11, 0.15)",
        "border": "rgba(245, 158, 11, 0.35)",
        "text": "#fbbf24",
        "label": "Moderate"
    },
    "Unhealthy for Sensitive Groups": {
        "color": "#f97316",
        "bg": "rgba(249, 115, 22, 0.15)",
        "border": "rgba(249, 115, 22, 0.35)",
        "text": "#fb923c",
        "label": "Unhealthy (Sensitive)"
    },
    "Unhealthy": {
        "color": "#ef4444",
        "bg": "rgba(239, 68, 68, 0.15)",
        "border": "rgba(239, 68, 68, 0.35)",
        "text": "#f87171",
        "label": "Unhealthy"
    },
    "Very Unhealthy": {
        "color": "#a855f7",
        "bg": "rgba(168, 85, 247, 0.15)",
        "border": "rgba(168, 85, 247, 0.35)",
        "text": "#c084fc",
        "label": "Very Unhealthy"
    },
    "Hazardous": {
        "color": "#e11d48",
        "bg": "rgba(225, 29, 72, 0.2)",
        "border": "rgba(225, 29, 72, 0.4)",
        "text": "#fda4af",
        "label": "Hazardous"
    },
}

AQI_HEALTH_MESSAGES = {
    "Good": "Air quality is expected to be clean and safe for all outdoor activities. 🌳",
    "Moderate": "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion. 😷",
    "Unhealthy for Sensitive Groups": "Particulate levels are elevated. Children, elderly, and those with respiratory conditions should reduce outdoor exposure. ⚠️",
    "Unhealthy": "Air quality is unhealthy. Everyone should reduce prolonged outdoor exertion and consider wearing masks. 🚨",
    "Very Unhealthy": "Health warning: High pollution levels. Avoid outdoor physical activities and keep windows closed. 🔴",
    "Hazardous": "Emergency health advisory: Severe pollution expected. Everyone should remain indoors. 🆘",
}

# ─────────────────────────────────────────────
# PRODUCTION CSS (Apple Weather / Vercel Aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Reset */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    body, .stApp {
        background-color: #0b0f19 !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.08) 0, transparent 40%),
            radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.06) 0, transparent 40%) !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1140px;
    }

    /* Top Brand Navigation */
    .brand-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-badge {
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        background: rgba(37, 99, 235, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(37, 99, 235, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background: #10b981;
        border-radius: 50%;
    }

    /* City Selector Pills */
    .city-nav-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: 1.6rem;
    }
    div[data-testid="stPills"] {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
    }
    div[data-testid="stPills"] button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.4rem !important;
        border-radius: 25px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stPills"] button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    div[data-testid="stPills"] button[aria-selected="true"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }

    /* Weather Hero Cards */
    .weather-card {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.6rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Hero Big Number */
    .hero-card-header {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .hero-location {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
    }
    .hero-num-row {
        display: flex;
        align-items: baseline;
        gap: 16px;
        margin-bottom: 0.6rem;
    }
    .hero-aqi-number {
        font-size: 4.8rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -2px;
    }
    .hero-category-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.95rem;
        font-weight: 700;
        padding: 0.35rem 0.95rem;
        border-radius: 20px;
        margin-bottom: 1.2rem;
    }

    /* Apple Weather Style Horizontal Slider Bar */
    .aqi-slider-wrap {
        margin: 1rem 0 1.4rem 0;
    }
    .aqi-slider-track {
        height: 7px;
        border-radius: 10px;
        background: linear-gradient(to right, 
            #10b981 0%, #10b981 10%, 
            #f59e0b 10%, #f59e0b 20%, 
            #f97316 20%, #f97316 30%, 
            #ef4444 30%, #ef4444 40%, 
            #a855f7 40%, #a855f7 60%, 
            #e11d48 60%, #e11d48 100%);
        position: relative;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.3);
    }
    .aqi-slider-thumb {
        position: absolute;
        top: -4.5px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #ffffff;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.6), 0 0 0 2px rgba(255, 255, 255, 0.4);
        transform: translateX(-50%);
    }
    .aqi-slider-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Quick Stat Chips */
    .meta-chip-row {
        display: flex;
        gap: 0.75rem;
    }
    .meta-chip {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.5rem 0.9rem;
        font-size: 0.82rem;
        color: #cbd5e1;
    }
    .meta-chip b {
        color: #ffffff;
    }

    /* 3-Day Daily Rows */
    .daily-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    .day-row {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 0.95rem 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    .day-row:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateX(3px);
    }
    .day-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .day-sub {
        font-size: 0.78rem;
        color: #64748b;
    }
    .day-aqi-tag {
        font-size: 1.05rem;
        font-weight: 800;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Health Banner */
    .health-banner {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 1.4rem 0;
        display: flex;
        align-items: center;
        gap: 16px;
        backdrop-filter: blur(20px);
    }
    .health-icon {
        font-size: 1.8rem;
        flex-shrink: 0;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8 !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(37, 99, 235, 0.2) !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hopsworks Model Registry...")
def load_model():
    """Load trained RandomForest model and feature metadata."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "aqi_model_dir")

    try:
        import hopsworks
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if api_key:
            project = hopsworks.login(api_key_value=api_key)
            mr = project.get_model_registry()
            hw_model = mr.get_model("aqi_prediction_model", version=None)
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


@st.cache_data(ttl=3600, show_spinner="Loading air quality data...")
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
# FEATURE ENGINEERING & 3-DAY ML FORECASTING
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
    """Generate 3-day (+24h, +48h, +72h) predictions using autoregressive ML."""
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

        # Autoregressive lag updates
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
# PLOTLY VISUALIZATIONS
# ─────────────────────────────────────────────
def render_trajectory_chart(forecasts, city_name):
    """Render a clean, production-grade 72h trajectory curve."""
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
        textfont=dict(size=12, weight="bold", color="#f8fafc"),
        line=dict(color="#3b82f6", width=3, shape='spline'),
        marker=dict(size=10, color="#3b82f6", line=dict(width=2, color='#ffffff')),
        fill='tozeroy',
        fillcolor="rgba(59, 130, 246, 0.12)",
        hovertemplate="<b>%{x}</b><br>Predicted EPA AQI: %{y}<br>Predicted PM2.5: %{customdata:.1f} µg/m³<extra></extra>",
        customdata=pm25s
    ))

    # Reference levels
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(16, 185, 129, 0.6)", annotation_text="Good (50)", annotation_position="bottom right", annotation_font=dict(color="#10b981", size=10))
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(245, 158, 11, 0.6)", annotation_text="Moderate (100)", annotation_position="bottom right", annotation_font=dict(color="#f59e0b", size=10))
    fig.add_hline(y=150, line_dash="dash", line_color="rgba(249, 115, 22, 0.6)", annotation_text="Unhealthy (150)", annotation_position="bottom right", annotation_font=dict(color="#f97316", size=10))

    fig.update_layout(
        title=dict(text=f"72-Hour AQI Trend Curve — {city_name}", font=dict(size=15, color="#f8fafc")),
        yaxis_title="Predicted EPA AQI",
        height=300,
        margin=dict(l=35, r=20, t=45, b=25),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", range=[0, max(max(aqis) * 1.35, 160)], tickfont=dict(color="#94a3b8", size=11)),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#94a3b8", size=11)),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
        showlegend=False
    )
    return fig


def render_regional_map(city_forecasts_dict):
    """Create clean dark-mode OpenStreetMap with color-coded AQI pins."""
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
            sizes.append(max(22, min(avg_aqi / 4.2, 50)))
            
            text = (
                f"<b>{city_name}</b><br>"
                f"• 3-Day Avg AQI: <b>{avg_aqi} ({avg_cat})</b><br>"
                f"• Tomorrow: AQI {fc[0]['epa_aqi']}<br>"
                f"• Day 2: AQI {fc[1]['epa_aqi']}<br>"
                f"• Day 3: AQI {fc[2]['epa_aqi']}"
            )
            texts.append(text)

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, opacity=0.9, sizemode='diameter'),
        text=names,
        textposition="top center",
        textfont=dict(size=12, color="#ffffff", family="-apple-system, sans-serif"),
        hovertext=texts,
        hoverinfo='text',
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=26.0, lon=68.0),
            zoom=6.2,
        ),
        height=380,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    # ── Top Navigation Bar ──
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">
            <span>🌬️</span>
            <span>Sindh Air Quality Forecast</span>
        </div>
        <div class="brand-badge">
            <span class="status-dot"></span>
            <span>RandomForest Model v5 • 72-Hour Prediction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Model & Features ──
    model, feature_names, model_version = load_model()
    raw_df = load_data()

    if raw_df is None or raw_df.empty:
        st.error("❌ No air quality data available. Please run the feature pipeline first.")
        st.stop()

    if model is None or feature_names is None:
        st.error("❌ Model not loaded from Hopsworks Model Registry. Please run the training pipeline first.")
        st.stop()

    df_eng = engineer_features(raw_df)

    # ── City Selector (Clean Floating Pills) ──
    city_options = list(CITIES.keys())
    
    st.markdown('<div class="city-nav-wrap">', unsafe_allow_html=True)
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

    # ── Compute 3-Day ML Predictions ──
    all_city_forecasts = {}
    for city_name in city_options:
        fc = predict_3_days_for_city(model, feature_names, df_eng, city_name)
        if fc:
            all_city_forecasts[city_name] = fc

    selected_forecasts = all_city_forecasts.get(selected_city)
    if not selected_forecasts or len(selected_forecasts) < 3:
        st.error(f"❌ Unable to generate 3-day forecast for {selected_city}.")
        st.stop()

    # 3-Day Summary Metrics
    avg_aqi = round(sum(f['epa_aqi'] for f in selected_forecasts) / len(selected_forecasts))
    avg_pm25 = sum(f['pm2_5'] for f in selected_forecasts) / len(selected_forecasts)
    avg_category, _ = get_aqi_category(avg_aqi)
    avg_theme = AQI_THEME.get(avg_category, AQI_THEME["Moderate"])

    # Trend calculation
    diff_aqi = selected_forecasts[2]['epa_aqi'] - selected_forecasts[0]['epa_aqi']
    if diff_aqi <= -4:
        trend_text = f"Improving (↓ {abs(diff_aqi)} AQI)"
        trend_color = "#10b981"
    elif diff_aqi >= 4:
        trend_text = f"Increasing (↑ {diff_aqi} AQI)"
        trend_color = "#ef4444"
    else:
        trend_text = "Stable (±2 AQI)"
        trend_color = "#60a5fa"

    # Slider thumb position on 0-500 scale
    thumb_pct = min(max((avg_aqi / 500) * 100, 2), 98)

    # ── Hero Section (Left: Apple Weather Style Card | Right: Daily Strip) ──
    col_left, col_right = st.columns([1.15, 1.25])

    with col_left:
        st.markdown(f"""
        <div class="weather-card">
            <div>
                <div class="hero-card-header">3-Day Average Outlook</div>
                <div class="hero-location">{CITIES[selected_city]['emoji']} {selected_city}, Sindh</div>
                
                <div class="hero-num-row">
                    <div class="hero-aqi-number" style="color: {avg_theme['text']};">{avg_aqi}</div>
                    <div style="color: #64748b; font-size: 1.1rem; font-weight: 600;">AQI</div>
                </div>

                <div class="hero-category-pill" style="background: {avg_theme['bg']}; color: {avg_theme['text']}; border: 1px solid {avg_theme['border']};">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: {avg_theme['color']};"></span>
                    <span>{avg_category}</span>
                </div>

                <!-- Apple Weather Horizontal AQI Spectrum Slider -->
                <div class="aqi-slider-wrap">
                    <div class="aqi-slider-track">
                        <div class="aqi-slider-thumb" style="left: {thumb_pct}%;"></div>
                    </div>
                    <div class="aqi-slider-labels">
                        <span>0 Good</span>
                        <span>100 Mod</span>
                        <span>200 Unhealthy</span>
                        <span>500+</span>
                    </div>
                </div>
            </div>

            <div class="meta-chip-row">
                <div class="meta-chip">Avg PM2.5: <b>{avg_pm25:.1f} µg/m³</b></div>
                <div class="meta-chip" style="color: {trend_color};">Trend: <b>{trend_text}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="weather-card"><div class="hero-card-header" style="margin-bottom: 0.8rem;">Daily Forecast Breakdown</div><div class="daily-list">', unsafe_allow_html=True)
        
        for fc in selected_forecasts:
            t = fc['theme']
            st.markdown(f"""
            <div class="day-row">
                <div>
                    <div class="day-title">{fc['day_name']} <span style="font-size: 0.8rem; color: #64748b; font-weight: 500;">({fc['step_label']})</span></div>
                    <div class="day-sub">{fc['date_str']}</div>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #64748b;">PM2.5</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">{fc['pm2_5']:.1f} µg/m³</div>
                    </div>
                    <div class="day-aqi-tag" style="background: {t['bg']}; color: {t['text']}; border: 1px solid {t['border']};">
                        AQI {fc['epa_aqi']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Health Advisory Banner ──
    highest_risk_day = max(selected_forecasts, key=lambda x: x['epa_aqi'])
    risk_t = highest_risk_day['theme']
    risk_cat = highest_risk_day['category']
    risk_msg = highest_risk_day['health_msg']
    alert_icon = "🌳" if highest_risk_day['epa_aqi'] <= 50 else "😷" if highest_risk_day['epa_aqi'] <= 100 else "⚠️" if highest_risk_day['epa_aqi'] <= 150 else "🚨"

    st.markdown(f"""
    <div class="health-banner" style="border-left: 4px solid {risk_t['color']};">
        <div class="health-icon">{alert_icon}</div>
        <div>
            <div style="font-size: 0.98rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">
                Health Advisory • Peak risk on {highest_risk_day['day_name']} ({highest_risk_day['date_str']}) — AQI {highest_risk_day['epa_aqi']} ({risk_cat})
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8;">
                {risk_msg}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs Section ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast Curve", "🗺️ Regional Map", "🏙️ City Comparison", "🤖 Model Info"
    ])

    with tab1:
        col_traj, col_hist = st.columns([1.2, 1])

        with col_traj:
            fig_traj = render_trajectory_chart(selected_forecasts, selected_city)
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
                    fill='tozeroy', fillcolor="rgba(56, 189, 248, 0.1)",
                    hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
                ))
                fig_hist.update_layout(
                    title=dict(text=f"Historical Baseline (Last 7 Days)", font=dict(size=15, color="#f8fafc")),
                    yaxis_title="PM2.5 (µg/m³)",
                    height=300,
                    margin=dict(l=35, r=20, t=45, b=25),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#94a3b8", size=11)),
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#94a3b8", size=11)),
                    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.caption("3-day average forecasted AQI across Sindh. Click or hover on any marker for daily predictions.")
        fig_map = render_regional_map(all_city_forecasts)
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
                    <div style="text-align:center; padding: 0.9rem 0.5rem; border-radius: 14px;
                                background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);">
                        <div style="font-size: 1.2rem;">{CITIES[city_name]['emoji']}</div>
                        <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc; margin: 0.2rem 0;">{city_name}</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: {t['text']};">Avg {c_avg}</div>
                        <div style="font-size: 0.75rem; color: {t['text']}; font-weight: 600;">{c_cat}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        matrix_data = []
        for city_name in city_options:
            fc = all_city_forecasts.get(city_name, [])
            if fc:
                c_avg = round(sum(f['epa_aqi'] for f in fc) / len(fc))
                matrix_data.append({
                    "City": f"{CITIES[city_name]['emoji']} {city_name}",
                    "3-Day Avg AQI": f"{c_avg} ({get_aqi_category(c_avg)[0]})",
                    "Tomorrow (+24h)": f"{fc[0]['epa_aqi']} ({fc[0]['category']}) • {fc[0]['pm2_5']:.1f} µg/m³",
                    "Day 2 (+48h)": f"{fc[1]['epa_aqi']} ({fc[1]['category']}) • {fc[1]['pm2_5']:.1f} µg/m³",
                    "Day 3 (+72h)": f"{fc[2]['epa_aqi']} ({fc[2]['category']}) • {fc[2]['pm2_5']:.1f} µg/m³",
                })
        
        matrix_df = pd.DataFrame(matrix_data)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    with tab4:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1.2rem;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.8rem;">
                    Active Model: RandomForest (Version {model_version})
                </div>
                <table style="width: 100%; font-size: 0.88rem; color: #cbd5e1;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);"><td style="padding: 6px 0; color: #94a3b8;">Algorithm</td><td style="font-weight: 600; text-align: right;">RandomForest Regressor</td></tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);"><td style="padding: 6px 0; color: #94a3b8;">Trees / Depth</td><td style="font-weight: 600; text-align: right;">300 Trees / Max Depth 20</td></tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);"><td style="padding: 6px 0; color: #94a3b8;">Features</td><td style="font-weight: 600; text-align: right;">{len(feature_names)} Lagged & Rolling</td></tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);"><td style="padding: 6px 0; color: #94a3b8;">PM2.5 Performance</td><td style="font-weight: 600; text-align: right; color: #10b981;">R² = 74.7% (RMSE = 23.9)</td></tr>
                    <tr><td style="padding: 6px 0; color: #94a3b8;">Registry</td><td style="font-weight: 600; text-align: right;">Hopsworks Model Registry</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            shap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "images", "shap_summary.png")
            if os.path.exists(shap_path):
                st.image(shap_path, use_container_width=True, caption="SHAP Feature Importance Summary")
            else:
                st.info("SHAP feature importance plot is generated during the daily training pipeline.")

    # ── Footer ──
    st.markdown("""
    <div style="text-align: center; color: #475569; font-size: 0.8rem; margin-top: 2rem; padding: 1rem 0; border-top: 1px solid rgba(255,255,255,0.06);">
        Sindh Air Quality Forecast • Continuous ML pipeline powered by Hopsworks Cloud & GitHub Actions
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
