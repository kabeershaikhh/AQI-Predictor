"""
🌬️ Sindh Air Quality Index Dashboard
======================================
Interactive real-time AQI monitoring and 24-hour forecasting dashboard
for 5 cities in Sindh, Pakistan.

Powered by:
  - OpenWeather API (live pollution data)
  - Hopsworks Feature Store & Model Registry
  - RandomForest ML Model (retrained daily via GitHub Actions)

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
    page_title="Sindh AQI Dashboard",
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
    "Nawabshah":  {"lat": 26.2442, "lon": 68.4100, "emoji": "🌾"},
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
    "Good": "Air quality is satisfactory. Enjoy your outdoor activities! 🌳",
    "Moderate": "Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion. 😷",
    "Unhealthy for Sensitive Groups": "Children, elderly, and people with respiratory conditions should limit outdoor activity. ⚠️",
    "Unhealthy": "Everyone should reduce outdoor exertion. Keep windows closed. 🚨",
    "Very Unhealthy": "Health alert! Everyone may experience health effects. Avoid outdoor activities. 🔴",
    "Hazardous": "EMERGENCY: Serious health effects for entire population. Stay indoors! 🆘",
}

POLLUTANT_INFO = {
    "pm2_5": {"name": "PM2.5", "unit": "µg/m³", "icon": "🔴", "desc": "Fine particles"},
    "pm10":  {"name": "PM10",  "unit": "µg/m³", "icon": "🟠", "desc": "Coarse particles"},
    "co":    {"name": "CO",    "unit": "µg/m³", "icon": "🟤", "desc": "Carbon monoxide"},
    "no2":   {"name": "NO₂",   "unit": "µg/m³", "icon": "🟡", "desc": "Nitrogen dioxide"},
    "o3":    {"name": "O₃",    "unit": "µg/m³", "icon": "🔵", "desc": "Ozone"},
    "so2":   {"name": "SO₂",   "unit": "µg/m³", "icon": "🟣", "desc": "Sulfur dioxide"},
    "nh3":   {"name": "NH₃",   "unit": "µg/m³", "icon": "⚪", "desc": "Ammonia"},
    "no":    {"name": "NO",    "unit": "µg/m³", "icon": "🟢", "desc": "Nitric oxide"},
}


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1a5276 0%, #2e86c1 50%, #3498db 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .hero-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* AQI main card */
    .aqi-card {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .aqi-card:hover {
        transform: translateY(-2px);
    }
    .aqi-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .aqi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
    }
    .aqi-category {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Health alert banner */
    .health-alert {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.95rem;
    }
    .health-alert-icon {
        font-size: 1.8rem;
    }

    /* Pollutant cards */
    .pollutant-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .pollutant-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .pollutant-name {
        font-size: 0.75rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #3498db;
        display: inline-block;
    }

    /* City selector pills */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        gap: 0.5rem;
    }
    div[data-testid="stRadio"] label {
        background: #f0f2f6;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.2s;
        border: 2px solid transparent;
    }
    div[data-testid="stRadio"] label:hover {
        background: #e1e5eb;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #3498db;
        color: white;
        border-color: #2980b9;
    }

    /* Metric delta styling */
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model...")
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
        st.sidebar.warning(f"Hopsworks model load: {e}")

    # Fallback to local model
    if os.path.exists(os.path.join(model_dir, "aqi_model.pkl")):
        model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
        return model, feature_names, "local"

    return None, None, None


@st.cache_data(ttl=3600, show_spinner="Loading air quality data...")
def load_data():
    """Load historical + latest live data."""
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

    # 3. Also check local latest_features
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
# FEATURE ENGINEERING (matches training pipeline)
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


def predict_for_city(model, feature_names, df_engineered, city_name):
    """Get 24h PM2.5 prediction for a specific city."""
    city_col = f"city_{city_name}"
    if city_col not in df_engineered.columns:
        return None

    city_data = df_engineered[df_engineered[city_col] == True].copy()
    if city_data.empty:
        # If one-hot didn't work with True/1, try numeric
        city_data = df_engineered[df_engineered[city_col] == 1].copy()
    if city_data.empty:
        return None

    latest = city_data.iloc[-1:]

    # Build feature vector matching training
    available = [f for f in feature_names if f in latest.columns]
    missing = [f for f in feature_names if f not in latest.columns]

    X = latest[available].copy()
    for col in missing:
        X[col] = 0
    X = X[feature_names]

    pred = model.predict(X)[0]
    return max(0, pred)  # PM2.5 can't be negative


# ─────────────────────────────────────────────
# UI HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_aqi_color(aqi_value):
    """Return hex color for AQI value."""
    category, _ = get_aqi_category(aqi_value)
    return AQI_COLORS.get(category, "#95a5a6")


def render_aqi_gauge(value, title, subtitle=""):
    """Create a beautiful semicircular AQI gauge using Plotly."""
    category, _ = get_aqi_category(value)
    color = AQI_COLORS.get(category, "#95a5a6")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 52, "color": color}, "suffix": ""},
        title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{subtitle}</span>",
               "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#ddd",
                     "tickvals": [0, 50, 100, 150, 200, 300, 500],
                     "ticktext": ["0", "50", "100", "150", "200", "300", "500"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#f8f9fa",
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
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": value
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"}
    )
    return fig


def render_trend_chart(df, city_name, days=7):
    """Create an interactive PM2.5 trend line chart."""
    df_plot = df.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], unit='ms', errors='coerce')
    df_plot = df_plot[df_plot['city'] == city_name].sort_values('timestamp')

    cutoff = df_plot['timestamp'].max() - timedelta(days=days)
    df_plot = df_plot[df_plot['timestamp'] >= cutoff]

    if df_plot.empty:
        return None

    fig = go.Figure()

    # PM2.5 line
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['pm2_5'],
        mode='lines',
        name='PM2.5',
        line=dict(color='#e74c3c', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(231,76,60,0.1)',
        hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
    ))

    # AQI category bands
    fig.add_hrect(y0=0, y1=12, fillcolor="#2ecc71", opacity=0.05,
                  annotation_text="Good", annotation_position="top left")
    fig.add_hrect(y0=12, y1=35.4, fillcolor="#f1c40f", opacity=0.05,
                  annotation_text="Moderate", annotation_position="top left")
    fig.add_hrect(y0=35.4, y1=55.4, fillcolor="#e67e22", opacity=0.05,
                  annotation_text="Unhealthy (Sensitive)", annotation_position="top left")

    fig.update_layout(
        title=dict(text=f"PM2.5 Concentration — {city_name} (Last {days} Days)",
                   font=dict(size=16)),
        xaxis_title="",
        yaxis_title="PM2.5 (µg/m³)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=30),
        hovermode='x unified',
        plot_bgcolor='#fafbfc',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#eee', showgrid=True),
        yaxis=dict(gridcolor='#eee', showgrid=True, rangemode='tozero'),
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    return fig


def render_city_map(city_data_dict):
    """Create an interactive map with AQI-colored markers for all cities."""
    lats, lons, names, aqis, colors, sizes, texts = [], [], [], [], [], [], []

    for city_name, info in CITIES.items():
        data = city_data_dict.get(city_name, {})
        aqi = data.get("epa_aqi", 0)
        category = data.get("category", "Good")
        color = AQI_COLORS.get(category, "#95a5a6")

        lats.append(info["lat"])
        lons.append(info["lon"])
        names.append(city_name)
        aqis.append(aqi)
        colors.append(color)
        sizes.append(max(20, min(aqi / 5, 50)))
        texts.append(f"<b>{city_name}</b><br>AQI: {aqi}<br>{category}")

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, opacity=0.85,
                    sizemode='diameter'),
        text=names,
        textposition="top center",
        textfont=dict(size=12, color="#2c3e50", family="Inter, sans-serif"),
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


def render_pollutant_comparison(city_data_dict):
    """Create a grouped bar chart comparing pollutants across all cities."""
    cities_list = list(city_data_dict.keys())
    key_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']

    fig = go.Figure()
    colors = ['#e74c3c', '#e67e22', '#3498db', '#f1c40f', '#8e44ad', '#95a5a6']

    for i, pol in enumerate(key_pollutants):
        values = [city_data_dict.get(c, {}).get(pol, 0) for c in cities_list]
        fig.add_trace(go.Bar(
            name=POLLUTANT_INFO.get(pol, {}).get("name", pol),
            x=cities_list, y=values,
            marker_color=colors[i],
            hovertemplate='<b>%{x}</b><br>' + POLLUTANT_INFO.get(pol, {}).get("name", pol) +
                          ': %{y:.1f} µg/m³<extra></extra>'
        ))

    fig.update_layout(
        barmode='group',
        title=dict(text="Pollutant Levels Across Cities", font=dict(size=16)),
        yaxis_title="Concentration (µg/m³)",
        height=350,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor='#fafbfc',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    # ── Hero Header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🌬️ Sindh Air Quality Index Dashboard</h1>
        <p>Real-time AQI monitoring & 24-hour ML forecasting for 5 cities in Sindh, Pakistan</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data & Model ──
    model, feature_names, model_version = load_model()
    raw_df = load_data()

    if raw_df is None or raw_df.empty:
        st.error("❌ No data available. Please run the feature pipeline first.")
        st.stop()

    # ── City Selector (Pill Buttons) ──
    city_options = list(CITIES.keys())
    selected_city = st.radio(
        "Select City",
        city_options,
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )

    # ── Get Latest Data for Each City ──
    city_data_dict = {}
    for city_name in city_options:
        city_df = raw_df[raw_df['city'] == city_name].copy()
        if not city_df.empty:
            latest = city_df.sort_values('timestamp').iloc[-1]
            epa_aqi = int(latest.get('epa_aqi', 0))
            category, _ = get_aqi_category(epa_aqi)
            city_data_dict[city_name] = {
                "epa_aqi": epa_aqi,
                "category": category,
                "pm2_5": latest.get('pm2_5', 0),
                "pm10": latest.get('pm10', 0),
                "co": latest.get('co', 0),
                "no2": latest.get('no2', 0),
                "o3": latest.get('o3', 0),
                "so2": latest.get('so2', 0),
                "nh3": latest.get('nh3', 0),
                "no": latest.get('no', 0),
                "timestamp": latest.get('timestamp', 0),
                "dominant_pollutant": latest.get('dominant_pollutant', 'PM2.5'),
            }

    # ── Selected City Data ──
    current = city_data_dict.get(selected_city, {})
    current_aqi = current.get("epa_aqi", 0)
    current_category = current.get("category", "Good")
    current_color = AQI_COLORS.get(current_category, "#95a5a6")

    # ── 24h Prediction ──
    predicted_pm25 = None
    predicted_aqi = None
    predicted_category = None
    if model is not None and feature_names is not None:
        df_eng = engineer_features(raw_df)
        predicted_pm25 = predict_for_city(model, feature_names, df_eng, selected_city)
        if predicted_pm25 is not None:
            predicted_aqi, _ = calculate_epa_aqi(
                pm2_5=predicted_pm25, pm10=0, co=0, no2=0, o3=0, so2=0
            )
            predicted_category, _ = get_aqi_category(predicted_aqi)

    # ── Top Row: AQI Gauges + Health Status ──
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        fig_current = render_aqi_gauge(
            current_aqi,
            "Current EPA AQI",
            f"Dominant: {current.get('dominant_pollutant', 'PM2.5')}"
        )
        st.plotly_chart(fig_current, use_container_width=True)

    with col2:
        if predicted_aqi is not None:
            fig_forecast = render_aqi_gauge(
                predicted_aqi,
                "24h Forecast",
                f"Predicted PM2.5: {predicted_pm25:.1f} µg/m³"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.info("Model not loaded — prediction unavailable")

    with col3:
        st.markdown(f"""
        <div class="aqi-card" style="background: linear-gradient(135deg, {current_color}22, {current_color}44); border: 2px solid {current_color};">
            <div class="aqi-label">Health Status</div>
            <div class="aqi-value" style="color: {current_color};">{CITIES[selected_city]['emoji']}</div>
            <div class="aqi-category" style="color: {current_color};">{current_category}</div>
            <div style="margin-top: 0.8rem; font-size: 0.8rem; color: #555;">
                {selected_city}, Sindh
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Health Alert Banner ──
    alert_msg = AQI_HEALTH_MESSAGES.get(current_category, "")
    alert_icon = "✅" if current_aqi <= 50 else "⚠️" if current_aqi <= 150 else "🚨" if current_aqi <= 300 else "🆘"
    alert_bg = f"{current_color}15"
    alert_border = f"{current_color}40"

    st.markdown(f"""
    <div class="health-alert" style="background: {alert_bg}; border: 1px solid {alert_border};">
        <span class="health-alert-icon">{alert_icon}</span>
        <div>
            <strong>{current_category}</strong> — {alert_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pollutant Breakdown ──
    st.markdown('<div class="section-header">📊 Pollutant Concentrations</div>', unsafe_allow_html=True)

    pol_cols = st.columns(len(POLLUTANT_INFO))
    for i, (pol_key, pol_info) in enumerate(POLLUTANT_INFO.items()):
        with pol_cols[i]:
            value = current.get(pol_key, 0)
            st.markdown(f"""
            <div class="pollutant-card">
                <div style="font-size: 1.2rem;">{pol_info['icon']}</div>
                <div class="pollutant-value">{value:.1f}</div>
                <div class="pollutant-name">{pol_info['name']}</div>
                <div style="font-size: 0.65rem; color: #aaa;">{pol_info['unit']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabs: Trends | Map | All Cities | Model Info ──
    st.markdown("")  # spacer
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trends", "🗺️ City Map", "🏙️ All Cities Comparison", "🤖 Model Info"
    ])

    with tab1:
        col_trend1, col_trend2 = st.columns(2)
        with col_trend1:
            fig_7d = render_trend_chart(raw_df, selected_city, days=7)
            if fig_7d:
                st.plotly_chart(fig_7d, use_container_width=True)
            else:
                st.info("Not enough data for 7-day trend")

        with col_trend2:
            fig_30d = render_trend_chart(raw_df, selected_city, days=30)
            if fig_30d:
                fig_30d.update_layout(
                    title=dict(text=f"PM2.5 Concentration — {selected_city} (Last 30 Days)")
                )
                st.plotly_chart(fig_30d, use_container_width=True)
            else:
                st.info("Not enough data for 30-day trend")

        # EPA AQI trend
        city_ts = raw_df[raw_df['city'] == selected_city].copy()
        city_ts['timestamp'] = pd.to_datetime(city_ts['timestamp'], unit='ms', errors='coerce')
        city_ts = city_ts.sort_values('timestamp')
        cutoff_7 = city_ts['timestamp'].max() - timedelta(days=7)
        city_ts_7 = city_ts[city_ts['timestamp'] >= cutoff_7]

        if not city_ts_7.empty and 'epa_aqi' in city_ts_7.columns:
            fig_aqi = go.Figure()
            fig_aqi.add_trace(go.Scatter(
                x=city_ts_7['timestamp'], y=city_ts_7['epa_aqi'],
                mode='lines', name='EPA AQI',
                line=dict(color='#2e86c1', width=2.5),
                fill='tozeroy', fillcolor='rgba(46,134,193,0.1)',
                hovertemplate='<b>%{x}</b><br>EPA AQI: %{y}<extra></extra>'
            ))
            fig_aqi.add_hline(y=100, line_dash="dash", line_color="#e67e22",
                             annotation_text="Unhealthy Threshold")
            fig_aqi.update_layout(
                title=dict(text=f"EPA AQI Trend — {selected_city} (Last 7 Days)",
                           font=dict(size=16)),
                yaxis_title="EPA AQI",
                height=300,
                margin=dict(l=40, r=20, t=50, b=30),
                plot_bgcolor='#fafbfc', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif"),
                showlegend=False
            )
            st.plotly_chart(fig_aqi, use_container_width=True)

    with tab2:
        fig_map = render_city_map(city_data_dict)
        st.plotly_chart(fig_map, use_container_width=True)

        # City AQI summary below map
        map_cols = st.columns(5)
        for i, (city_name, info) in enumerate(CITIES.items()):
            data = city_data_dict.get(city_name, {})
            aqi = data.get("epa_aqi", 0)
            cat = data.get("category", "Good")
            clr = AQI_COLORS.get(cat, "#95a5a6")
            with map_cols[i]:
                st.markdown(f"""
                <div style="text-align:center; padding: 0.5rem; border-radius: 10px;
                            background: {clr}15; border: 1px solid {clr}40;">
                    <div style="font-weight: 700; color: {clr};">{aqi}</div>
                    <div style="font-size: 0.75rem; color: #555;">{city_name}</div>
                    <div style="font-size: 0.65rem; color: {clr};">{cat}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        fig_compare = render_pollutant_comparison(city_data_dict)
        st.plotly_chart(fig_compare, use_container_width=True)

        # AQI ranking table
        st.markdown('<div class="section-header">🏆 City AQI Ranking</div>', unsafe_allow_html=True)
        ranking = []
        for city_name in city_options:
            data = city_data_dict.get(city_name, {})
            ranking.append({
                "City": city_name,
                "EPA AQI": data.get("epa_aqi", 0),
                "Category": data.get("category", "—"),
                "PM2.5 (µg/m³)": round(data.get("pm2_5", 0), 1),
                "PM10 (µg/m³)": round(data.get("pm10", 0), 1),
                "Dominant": data.get("dominant_pollutant", "—"),
            })
        ranking_df = pd.DataFrame(ranking).sort_values("EPA AQI")
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    with tab4:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown('<div class="section-header">🤖 Model Details</div>', unsafe_allow_html=True)
            if model is not None:
                st.success(f"✅ Model loaded (Version: {model_version})")
                st.markdown(f"""
                | Property | Value |
                |---|---|
                | **Algorithm** | RandomForest Regressor |
                | **Trees** | 300 |
                | **Max Depth** | 20 |
                | **Features** | {len(feature_names) if feature_names else '—'} |
                | **Target** | PM2.5 (24h ahead) |
                | **Registry** | Hopsworks Model Registry |
                """)
            else:
                st.warning("⚠️ Model not loaded. Place model files in `aqi_model_dir/`.")

        with col_m2:
            st.markdown('<div class="section-header">📊 SHAP Feature Importance</div>', unsafe_allow_html=True)
            shap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "images", "shap_summary.png")
            if os.path.exists(shap_path):
                st.image(shap_path, use_container_width=True)
            else:
                st.info("SHAP plot will appear after the training pipeline runs.")

    # ── Footer ──
    st.markdown("---")
    last_update = current.get("timestamp", 0)
    if last_update:
        try:
            update_time = datetime.fromtimestamp(last_update / 1000).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            update_time = "Unknown"
    else:
        update_time = "Unknown"

    st.markdown(f"""
    <div style="text-align: center; color: #aaa; font-size: 0.8rem; padding: 1rem;">
        <strong>Sindh AQI Dashboard</strong> • Data refreshed hourly via GitHub Actions •
        Last reading: {update_time} •
        Built with Streamlit, Hopsworks & RandomForest ML
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
