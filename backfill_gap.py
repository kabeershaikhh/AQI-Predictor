

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
from utils import calculate_epa_aqi, get_aqi_category

# Load API keys
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Local storage directory
FEATURE_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_store")
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

# Cities to track (Sindh, Pakistan)
CITIES = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Hyderabad", "lat": 25.3960, "lon": 68.3578},
    {"name": "Jamshoro", "lat": 25.4361, "lon": 68.2802},
    {"name": "Nawabshah", "lat": 26.2483, "lon": 68.4096},
    {"name": "Sukkur", "lat": 27.7139, "lon": 68.8369}
]


GAP_START = datetime(2026, 8, 10, 19, 0, 0)
GAP_END = datetime.utcnow()

# Chunk size: fetch 1 day at a time (gap is small)
CHUNK_DAYS = 1


def fetch_historical_chunk(city, start_unix, end_unix):
    """Fetches historical air pollution data for a city between two Unix timestamps."""
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={city['lat']}&lon={city['lon']}"
        f"&start={start_unix}&end={end_unix}"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print(f"    ✗ API error: {response.status_code} - {response.text[:100]}")
        return []

    data = response.json()
    rows = []

    for entry in data.get("list", []):
        dt = datetime.utcfromtimestamp(entry["dt"])
        rows.append({
            "city": city["name"],
            "timestamp": dt,
            "aqi": entry["main"]["aqi"],
            "co": entry["components"]["co"],
            "no": entry["components"]["no"],
            "no2": entry["components"]["no2"],
            "o3": entry["components"]["o3"],
            "so2": entry["components"]["so2"],
            "pm2_5": entry["components"]["pm2_5"],
            "pm10": entry["components"]["pm10"],
            "nh3": entry["components"]["nh3"]
        })

    return rows


def engineer_features(df):
    """Adds EPA AQI, time-based features, and converts timestamp to Unix ms."""
    epa_results = df.apply(
        lambda row: calculate_epa_aqi(
            pm2_5=row['pm2_5'], pm10=row['pm10'], co=row['co'],
            no2=row['no2'], o3=row['o3'], so2=row['so2']
        ), axis=1
    )
    df['epa_aqi'] = epa_results.apply(lambda x: x[0])
    df['dominant_pollutant'] = epa_results.apply(lambda x: x[1])
    df['aqi_category'] = df['epa_aqi'].apply(lambda x: get_aqi_category(x)[0])

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Convert to Unix milliseconds 
    df['timestamp'] = df['timestamp'].astype('int64') // 10**6

    return df


def upload_to_hopsworks(df):
    """Uploads the complete dataset to Hopsworks Feature Store."""
    print("\n  Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    print("  Getting or creating 'aqi_features' Feature Group...")
    try:
        aqi_fg = fs.get_feature_group("aqi_features", version=1)
        if not aqi_fg.online_enabled:
            print("  ⚠️ Existing feature group has online_enabled=False. Recreating with online_enabled=True...")
            aqi_fg.delete()
            aqi_fg = fs.create_feature_group(
                name="aqi_features",
                version=1,
                primary_key=["city", "timestamp"],
                event_time="timestamp",
                description="Air Quality Index data with pollution components and time features for 5 cities in Sindh, Pakistan",
                online_enabled=True
            )
    except Exception:
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            description="Air Quality Index data with pollution components and time features for 5 cities in Sindh, Pakistan",
            online_enabled=True
        )

    # Upload in batches to avoid timeouts
    batch_size = 5000
    total = len(df)
    print(f"  Uploading {total} rows in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = df.iloc[i:i+batch_size]
        aqi_fg.insert(
            batch,
            storage="online",
            write_options={"start_offline_materialization": True, "wait_for_job": False}
        )
        print(f"    ✓ Uploaded rows {i} - {min(i+batch_size, total)} of {total}")

    print("  ✓ All data uploaded to Hopsworks Feature Store!")


if __name__ == "__main__":
    print("=" * 60)
    print("  AQI GAP BACKFILL (Aug 10 → Aug 26)")
    print("=" * 60)
    print(f"\n  Gap range: {GAP_START.strftime('%Y-%m-%d %H:%M')} → {GAP_END.strftime('%Y-%m-%d %H:%M')} UTC")
    gap_hours = int((GAP_END - GAP_START).total_seconds() / 3600)
    print(f"  Expected new rows: ~{gap_hours * len(CITIES)} ({gap_hours}h × {len(CITIES)} cities)")

    #  Step 1: Fetch gap data 
    all_data = []
    for i, city in enumerate(CITIES):
        print(f"\n[{i+1}/{len(CITIES)}] Fetching gap data for {city['name']}...")
        current = GAP_START
        city_rows = []

        while current < GAP_END:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), GAP_END)
            start_unix = int(current.timestamp())
            end_unix = int(chunk_end.timestamp())

            print(f"    {current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}", end="")
            rows = fetch_historical_chunk(city, start_unix, end_unix)
            city_rows.extend(rows)
            print(f"  ({len(rows)} records)")

            current = chunk_end
            time.sleep(1)  # Respect API rate limits

        all_data.extend(city_rows)
        print(f"  ✓ {city['name']}: {len(city_rows)} records fetched")

    if len(all_data) == 0:
        print("\n✗ No gap data fetched! Check your API key.")
        exit(1)

    #  Step 2: Engineer features for gap data 
    print(f"\n[Processing] Engineering features for {len(all_data)} new rows...")
    gap_df = pd.DataFrame(all_data)
    gap_df = engineer_features(gap_df)

    #  Step 3: Merge with existing historical data 
    filepath = os.path.join(FEATURE_STORE_DIR, "aqi_historical.parquet")
    print(f"\n[Merging] Loading existing data from {filepath}...")
    existing_df = pd.read_parquet(filepath)
    print(f"  Existing rows: {len(existing_df)}")

    combined_df = pd.concat([existing_df, gap_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["city", "timestamp"], keep="last")
    combined_df = combined_df.sort_values(by=["city", "timestamp"]).reset_index(drop=True)
    print(f"  Combined rows: {len(combined_df)} (after dedup)")

    #  Step 4: Save updated parquet 
    combined_df.to_parquet(filepath, index=False)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n  ✓ Updated parquet saved: {filepath}")
    print(f"  ✓ File size: {file_size_mb:.1f} MB")

    # Show date range
    ts_min = pd.to_datetime(combined_df['timestamp'].min(), unit='ms')
    ts_max = pd.to_datetime(combined_df['timestamp'].max(), unit='ms')
    print(f"  ✓ Date range: {ts_min} → {ts_max}")

    #  Step 5: Upload complete dataset to Hopsworks 
    print("\n[Uploading] Uploading complete dataset to Hopsworks...")
    try:
        upload_to_hopsworks(combined_df)
    except Exception as e:
        print(f"\n  ✗ Hopsworks upload failed: {e}")
        print("  ! Data is safe in local parquet. You can retry later.")
        print("  ! If free tier issue, upload via Hopsworks UI terminal instead.")

    print("\n" + "=" * 60)
    print("  GAP BACKFILL COMPLETE!")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Date range: {ts_min} → {ts_max}")
    print("=" * 60)
