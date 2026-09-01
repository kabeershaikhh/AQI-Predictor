"""
🌬️ Sindh Air Quality Index — 3-Day ML Forecast Dashboard
===========================================================
Interactive 72-hour AQI forecasting dashboard for 5 cities in Sindh, Pakistan.

Features:
  - 3-Day Forward Predictions (Day 1: +24h, Day 2: +48h, Day 3: +72h)
  - Pure Machine Learning Forecasting (RandomForest Model v5)
  - Color-coded EPA AQI Gauges & Health Risk Advisories
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
    "Good": "#2ecc71",
    "Moderate": "#f1c40f",
    "Unhealthy for Sensitive Groups": "#e67e22",
    "Unhealthy": "#e74c3c",
    "Very Unhealthy": "#8e44ad",
    "Hazardous": "#7f1d1d",
}

AQI_HEALTH_MESSAGES = {
    "Good": "Air quality is expected to be ideal. Great conditions for all outdoor activities! 🌳",
    "Moderate": "Air quality is acceptable. Sensitive individuals should consider taking precautions during prolonged outdoor exertion. 😷",
    "Unhealthy for Sensitive Groups": "Children, elderly, and those with respiratory issues should limit outdoor activities. ⚠️",
    "Unhealthy": "Air quality is unhealthy for everyone. Wear masks and keep windows closed. 🚨",
    "Very Unhealthy": "Serious health risk! Avoid outdoor exertion and use indoor air purifiers if possible. 🔴",
    "Hazardous": "EMERGENCY HEALTH HAZARD: Severe pollution expected. Remain indoors! 🆘",
}

POLLUTANT_INFO = {
    "pm2_5": {"name": "PM2.5", "unit": "µg/m³", "icon": "🔴", "desc": "Fine particulate matter"},
    "pm10":  {"name": "PM10",  "unit": "µg/m³", "icon": "🟠", "desc": "Coarse dust particles"},
    "co":    {"name": "CO",    "unit": "µg/m³", "icon": "🟤", "desc": "Carbon monoxide"},
    "no2":   {"name": "NO₂",   "unit": "µg/m³", "icon": "🟡", "desc": "Nitrogen dioxide"},
    "o3":    {"name": "O₃",    "unit": "µg/m³", "icon": "🔵", "desc": "Ground-level Ozone"},
    "so2":   {"name": "SO₂",   "unit": "µg/m³", "icon": "🟣", "desc": "Sulfur dioxide"},
    "nh3":   {"name": "NH₃",   "unit": "µg/m³", "icon": "⚪", "desc": "Ammonia"},
    "no":    {"name": "NO",    "unit": "µg/m³", "icon": "🟢", "desc": "Nitric oxide"},
}


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean layout */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #0a192f 0%, #172a45 50%, #1e3c72 100%);
        padding: 1.8rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.3rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 400;
    }

    /* City Selector Container */
    .city-selector-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.4rem;
    }

    /* Modern Streamlit Pills styling */
    div[data-testid="stPills"] {
        display: flex;
        justify-content: center;
        gap: 0.7rem;
    }
    div[data-testid="stPills"] button {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        border-radius: 30px !important;
        border: 1.5px solid rgba(255, 255, 255, 0.2) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stPills"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%) !important;
        color: white !important;
        border-color: #00c6ff !important;
        box-shadow: 0 4px 18px rgba(0, 198, 255, 0.45) !important;
        transform: scale(1.04);
    }

    /* Forecast Card Container */
    .forecast-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1.5px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.2rem 1rem 0.8rem 1rem;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .forecast-card:hover {
        transform: translateY(-4px);
    }
    .forecast-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        margin-bottom: 0.4rem;
    }
    .forecast-date {
        font-size: 0.95rem;
        font-weight: 600;
        opacity: 0.85;
        margin-bottom: 0.2rem;
    }
    .forecast-meta {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: -0.5rem;
        padding-bottom: 0.4rem;
    }

    /* Health Alert Banner */
    .health-alert {
        border-radius: 16px;
        padding: 1.2rem 1.6rem;
        margin: 1.3rem 0;
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 1.02rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.12);
    }
    .health-alert-icon {
        font-size: 2.2rem;
        flex-shrink: 0;
    }

    /* Section Header */
    .section-header {
        font-size: 1.25rem;
        font-weight: 800;
        margin: 1.8rem 0 0.9rem 0;
        padding-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 700;
    }
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
    except Exception as e:
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
        date_str = target_date.strftime("%A, %b %d")
        
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
            "step_label": f"Day {day} (+{day*24}h)",
            "badge_title": "TOMORROW (+24h)" if day == 1 else f"DAY {day} (+{day*24}h)",
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
def render_aqi_gauge(value, title, subtitle="", color=None):
    """Create a sleek semicircular AQI gauge using Plotly."""
    category, _ = get_aqi_category(value)
    gauge_color = color or AQI_COLORS.get(category, "#95a5a6")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 48, "color": gauge_color, "family": "Inter, sans-serif"}, "suffix": ""},
        title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{subtitle}</span>",
               "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#ddd",
                     "tickvals": [0, 50, 100, 150, 200, 300, 500],
                     "ticktext": ["0", "50", "100", "150", "200", "300", "500"]},
            "bar": {"color": gauge_color, "thickness": 0.32},
            "bgcolor": "rgba(255, 255, 255, 0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(46,204,113,0.2)"},
                {"range": [50, 100], "color": "rgba(241,196,15,0.2)"},
                {"range": [100, 150], "color": "rgba(230,126,34,0.2)"},
                {"range": [150, 200], "color": "rgba(231,76,60,0.2)"},
                {"range": [200, 300], "color": "rgba(142,68,173,0.2)"},
                {"range": [300, 500], "color": "rgba(127,29,29,0.2)"},
            ],
            "threshold": {
                "line": {"color": gauge_color, "width": 3},
                "thickness": 0.8,
                "value": value
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=15, r=15, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"}
    )
    return fig


def render_3day_trajectory(forecasts, city_name):
    """Render a clean 3-day AQI prediction trajectory bar/line chart."""
    days = [f["date_str"] for f in forecasts]
    aqis = [f["epa_aqi"] for f in forecasts]
    pm25s = [f["pm2_5"] for f in forecasts]
    colors = [f["color"] for f in forecasts]

    fig = go.Figure()

    # AQI Bars
    fig.add_trace(go.Bar(
        x=days,
        y=aqis,
        marker_color=colors,
        text=[f"AQI: {a}<br>({f['category']})" for a, f in zip(aqis, forecasts)],
        textposition="auto",
        name="Predicted AQI",
        hovertemplate="<b>%{x}</b><br>Predicted EPA AQI: %{y}<br>Predicted PM2.5: %{customdata:.1f} µg/m³<extra></extra>",
        customdata=pm25s
    ))

    # Threshold guidelines
    fig.add_hline(y=50, line_dash="dot", line_color="#2ecc71", annotation_text="Good (50)")
    fig.add_hline(y=100, line_dash="dash", line_color="#f1c40f", annotation_text="Moderate (100)")
    fig.add_hline(y=150, line_dash="dash", line_color="#e67e22", annotation_text="Unhealthy (150)")

    fig.update_layout(
        title=dict(text=f"3-Day Air Quality Forecast Outlook — {city_name}", font=dict(size=16)),
        yaxis_title="Predicted EPA AQI",
        height=320,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor='rgba(255, 255, 255, 0.02)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', range=[0, max(max(aqis) * 1.25, 160)]),
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    return fig


def render_city_forecast_map(city_forecasts_dict):
    """Create an interactive map showing 3-day forecast markers for all cities."""
    lats, lons, names, day1_aqis, colors, sizes, texts = [], [], [], [], [], [], []

    for city_name, info in CITIES.items():
        fc = city_forecasts_dict.get(city_name, [])
        if fc:
            d1 = fc[0]
            aqi = d1["epa_aqi"]
            cat = d1["category"]
            clr = d1["color"]
            
            lats.append(info["lat"])
            lons.append(info["lon"])
            names.append(city_name)
            day1_aqis.append(aqi)
            colors.append(clr)
            sizes.append(max(22, min(aqi / 4.5, 55)))
            
            # Hover text showing all 3 days
            d2 = fc[1]
            d3 = fc[2]
            text = (
                f"<b>{city_name} (3-Day Forecast)</b><br>"
                f"• Day 1 ({d1['date_str']}): AQI {d1['epa_aqi']} ({d1['category']})<br>"
                f"• Day 2 ({d2['date_str']}): AQI {d2['epa_aqi']} ({d2['category']})<br>"
                f"• Day 3 ({d3['date_str']}): AQI {d3['epa_aqi']} ({d3['category']})"
            )
            texts.append(text)

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, opacity=0.88, sizemode='diameter'),
        text=names,
        textposition="top center",
        textfont=dict(size=12, family="Inter, sans-serif"),
        hovertext=texts,
        hoverinfo='text',
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
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
    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🌬️ Sindh Air Quality — 3-Day ML Forecast</h1>
        <p>72-hour air quality predictions for 5 cities in Sindh, powered by RandomForest AI (Model v5)</p>
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

    # ── Top Row: 3-Day Forecast Cards (Day 1, Day 2, Day 3) ──
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    badges = ["badge-ai", "badge-ai", "badge-ai"]

    for idx, (col, fc) in enumerate(zip(cols, selected_forecasts)):
        with col:
            st.markdown(f"""
            <div class="forecast-card" style="border-color: {fc['color']}; box-shadow: 0 0 25px {fc['color']}22;">
                <span class="forecast-badge" style="background: {fc['color']}25; color: {fc['color']}; border: 1px solid {fc['color']}50;">
                    {fc['badge_title']}
                </span>
                <div class="forecast-date">{fc['date_str']}</div>
            """, unsafe_allow_html=True)
            
            fig_gauge = render_aqi_gauge(
                fc['epa_aqi'],
                fc['category'],
                f"PM2.5: {fc['pm2_5']:.1f} µg/m³",
                color=fc['color']
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown(f"""
                <div class="forecast-meta">
                    {CITIES[selected_city]['emoji']} {selected_city} • <b>{fc['category']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 3-Day Health Advisory Banner ──
    # Pick the most critical risk day across the 3 days
    highest_risk_day = max(selected_forecasts, key=lambda x: x['epa_aqi'])
    risk_color = highest_risk_day['color']
    risk_cat = highest_risk_day['category']
    risk_msg = highest_risk_day['health_msg']
    alert_icon = "✅" if highest_risk_day['epa_aqi'] <= 50 else "⚠️" if highest_risk_day['epa_aqi'] <= 150 else "🚨" if highest_risk_day['epa_aqi'] <= 300 else "🆘"

    st.markdown(f"""
    <div class="health-alert" style="background: {risk_color}18; border: 1.5px solid {risk_color}45;">
        <span class="health-alert-icon">{alert_icon}</span>
        <div>
            <strong>3-Day Health Advisory for {selected_city}:</strong> Peak level expected on <b>{highest_risk_day['date_str']}</b> ({risk_cat}, AQI {highest_risk_day['epa_aqi']}). {risk_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: 3-Day Outlook | Regional Map | All Cities Comparison | Model Info ──
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
            # Historical 7-day trend for context
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
                    line=dict(color='#00c6ff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0,198,255,0.1)',
                    hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
                ))
                fig_hist.update_layout(
                    title=dict(text=f"Historical PM2.5 Context (Past 7 Days)", font=dict(size=16)),
                    yaxis_title="PM2.5 (µg/m³)",
                    height=320,
                    margin=dict(l=40, r=20, t=50, b=30),
                    plot_bgcolor='rgba(255, 255, 255, 0.02)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                    font=dict(family="Inter, sans-serif"),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">🗺️ Sindh Regional 3-Day Forecast Map</div>', unsafe_allow_html=True)
        st.caption("Hover over each city pin to view its full Day 1, Day 2, and Day 3 AQI predictions.")
        fig_map = render_city_forecast_map(all_city_forecasts)
        st.plotly_chart(fig_map, use_container_width=True)

        # 5-city forecast cards below map
        map_cols = st.columns(5)
        for i, city_name in enumerate(city_options):
            fc = all_city_forecasts.get(city_name, [])
            if fc:
                d1 = fc[0]
                with map_cols[i]:
                    st.markdown(f"""
                    <div style="text-align:center; padding: 0.7rem 0.5rem; border-radius: 14px;
                                background: {d1['color']}18; border: 1.5px solid {d1['color']}45;">
                        <div style="font-size: 1.2rem;">{CITIES[city_name]['emoji']}</div>
                        <div style="font-weight: 700; font-size: 0.95rem;">{city_name}</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: {d1['color']};">AQI {d1['epa_aqi']}</div>
                        <div style="font-size: 0.72rem; color: {d1['color']}; font-weight: 600;">{d1['category']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">🏙️ All 5 Cities — 3-Day Forecast Matrix</div>', unsafe_allow_html=True)
        
        matrix_data = []
        for city_name in city_options:
            fc = all_city_forecasts.get(city_name, [])
            if fc:
                matrix_data.append({
                    "City": f"{CITIES[city_name]['emoji']} {city_name}",
                    "Day 1 AQI (+24h)": f"{fc[0]['epa_aqi']} ({fc[0]['category']})",
                    "Day 1 PM2.5": f"{fc[0]['pm2_5']:.1f} µg/m³",
                    "Day 2 AQI (+48h)": f"{fc[1]['epa_aqi']} ({fc[1]['category']})",
                    "Day 2 PM2.5": f"{fc[1]['pm2_5']:.1f} µg/m³",
                    "Day 3 AQI (+72h)": f"{fc[2]['epa_aqi']} ({fc[2]['category']})",
                    "Day 3 PM2.5": f"{fc[2]['pm2_5']:.1f} µg/m³",
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
            | **Target Variable** | PM2.5 24h/48h/72h Ahead |
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
    <div style="text-align: center; color: #888; font-size: 0.82rem; padding: 0.8rem;">
        <strong>Sindh Air Quality Index System</strong> • 72-Hour Predictions Generated by RandomForest AI •
        Powered by Hopsworks Cloud & GitHub Actions CI/CD
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
