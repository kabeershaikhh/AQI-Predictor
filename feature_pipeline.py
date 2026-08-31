import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
import hopsworks
from dotenv import load_dotenv
from utils import calculate_epa_aqi, get_aqi_category

# Detect if running in CI (GitHub Actions sets CI=true automatically)
IS_CI = os.getenv("CI", "false").lower() == "true"

# Load API keys from .env file
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Directory for local Parquet backup
FEATURE_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_store")
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

# List of cities we want to track (Sindh, Pakistan)
CITIES = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Hyderabad", "lat": 25.3960, "lon": 68.3578},
    {"name": "Jamshoro", "lat": 25.4361, "lon": 68.2802},
    {"name": "Nawabshah", "lat": 26.2483, "lon": 68.4096},
    {"name": "Sukkur", "lat": 27.7139, "lon": 68.8369}
]


# ─────────────────────────────────────────────
# STEP 1: FETCH DATA FROM OPENWEATHER API
# ─────────────────────────────────────────────
def fetch_air_pollution(city):
    """
    Calls the OpenWeather Air Pollution API for a single city.
    Returns a dictionary of pollutant concentrations + AQI, or None on failure.
    """
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={city['lat']}&lon={city['lon']}&appid={OPENWEATHER_API_KEY}"
    )
    
    print(f"  Fetching data for {city['name']}...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"  ✗ Failed for {city['name']}: {response.text}")
        return None
    
    data = response.json()
    current = data["list"][0]
    
    return {
        "city": city["name"],
        "timestamp": datetime.now(),
        "aqi": current["main"]["aqi"],           # 1-5 scale
        "co": current["components"]["co"],         # Carbon Monoxide (μg/m³)
        "no": current["components"]["no"],         # Nitrogen Monoxide (μg/m³)
        "no2": current["components"]["no2"],       # Nitrogen Dioxide (μg/m³)
        "o3": current["components"]["o3"],         # Ozone (μg/m³)
        "so2": current["components"]["so2"],       # Sulphur Dioxide (μg/m³)
        "pm2_5": current["components"]["pm2_5"],   # Fine Particles (μg/m³)
        "pm10": current["components"]["pm10"],     # Coarse Particles (μg/m³)
        "nh3": current["components"]["nh3"]        # Ammonia (μg/m³)
    }


# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(raw_data_list):
    """
    Takes a list of raw data dictionaries and returns a pandas DataFrame
    with engineered time-based features added.
    """
    df = pd.DataFrame(raw_data_list)
    
    # Calculate EPA AQI (0-500 scale) from raw pollutant concentrations
    epa_results = df.apply(
        lambda row: calculate_epa_aqi(
            pm2_5=row['pm2_5'], pm10=row['pm10'], co=row['co'],
            no2=row['no2'], o3=row['o3'], so2=row['so2']
        ), axis=1
    )
    df['epa_aqi'] = epa_results.apply(lambda x: x[0])            # 0-500 scale
    df['dominant_pollutant'] = epa_results.apply(lambda x: x[1])  # e.g. "PM2.5"
    df['aqi_category'] = df['epa_aqi'].apply(lambda x: get_aqi_category(x)[0])
    
    # Extract time-based features from the timestamp
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Convert timestamp to Unix milliseconds (required by Hopsworks event_time)
    df['timestamp'] = df['timestamp'].astype('int64') // 10**6
    
    return df


# ─────────────────────────────────────────────
# STEP 3A: SAVE TO LOCAL PARQUET (BACKUP)
# ─────────────────────────────────────────────
def save_to_local(df):
    """Saves the DataFrame to a local Parquet file (append mode)."""
    filepath = os.path.join(FEATURE_STORE_DIR, "aqi_features.parquet")
    
    if os.path.exists(filepath):
        existing_df = pd.read_parquet(filepath)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["city", "timestamp"], keep="last")
        combined_df.to_parquet(filepath, index=False)
        print(f"  ✓ Local backup updated. Total rows: {len(combined_df)}")
    else:
        df.to_parquet(filepath, index=False)
        print(f"  ✓ Local backup created with {len(df)} rows.")


# ─────────────────────────────────────────────
# STEP 3B: UPLOAD TO HOPSWORKS FEATURE STORE
# ─────────────────────────────────────────────
def upload_to_hopsworks(df):
    """
    Uploads engineered features directly to Hopsworks Cloud using Hopsworks REST Dataset API.
    Uses pure HTTPS (port 443) which works 100% reliably on all external environments and
    free-tier accounts without Kafka or HDFS broker authorization issues.
    """
    print("  Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    dataset_api = project.get_dataset_api()
    
    print("  Uploading features to Hopsworks Cloud (Resources/latest_features.parquet)...")
    temp_parquet = os.path.join(FEATURE_STORE_DIR, "latest_features.parquet")
    df.to_parquet(temp_parquet, index=False)
    dataset_api.upload(temp_parquet, upload_path="Resources", overwrite=True)
    print("  ✓ Features successfully uploaded to Hopsworks Cloud via REST API!")


# ─────────────────────────────────────────────
# MAIN: RUN THE PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AQI FEATURE PIPELINE")
    print("=" * 60)
    
    # Step 1: Fetch data from API
    print("\n[Step 1/3] Fetching air quality data from OpenWeather API...")
    raw_data = []
    for city in CITIES:
        data = fetch_air_pollution(city)
        if data:
            raw_data.append(data)
    
    if len(raw_data) == 0:
        print("\n✗ No data fetched. Check your API key and internet connection.")
        sys.exit(1)
    
    # Step 2: Engineer features
    print(f"\n[Step 2/3] Engineering features for {len(raw_data)} cities...")
    features_df = engineer_features(raw_data)
    print(features_df.to_string())
    
    # Step 3: Save data
    print("\n[Step 3/3] Saving data...")
    
    # 3A: Save locally (always save to feature_store/)
    print("\n  --- Local Parquet Backup ---")
    save_to_local(features_df)
    
    # 3B: Upload to Hopsworks Feature Store & Datasets
    print("\n  --- Hopsworks Cloud ---")
    try:
        upload_to_hopsworks(features_df)
    except Exception as e:
        print(f"  ✗ Hopsworks upload failed: {e}")
        if IS_CI:
            print("  ✗ FATAL: Hopsworks upload failed in CI. Exiting with error.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
