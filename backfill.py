
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils import calculate_epa_aqi, get_aqi_category

# Load API keys
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

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

# HOW LONG TO BACKFILL

# Fetch the last 1 year of data (enough for seasonal patterns)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)

# Chunk size: fetch 1 week at a time to keep API responses manageable
CHUNK_DAYS = 7


def fetch_historical_chunk(city, start_unix, end_unix):
    """
    Fetches historical air pollution data for a city between two Unix timestamps.
    
    The OpenWeather Historical Air Pollution API returns hourly data points
    for the requested time range in a single response.
    
    API docs: https://openweathermap.org/api/air-pollution
    Endpoint: /air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}
    """
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
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Convert to Unix milliseconds (matches feature_pipeline.py format)
    df['timestamp'] = df['timestamp'].astype('int64') // 10**6
    
    return df


def backfill_city(city, start_date, end_date):
    """
    Fetches all historical data for a single city in weekly chunks.
    Returns a list of row dictionaries.
    """
    all_rows = []
    current = start_date
    chunk_num = 0
    total_chunks = (end_date - start_date).days // CHUNK_DAYS + 1
    
    while current < end_date:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end_date)
        start_unix = int(current.timestamp())
        end_unix = int(chunk_end.timestamp())
        
        chunk_num += 1
        print(f"    Chunk {chunk_num}/{total_chunks}: "
              f"{current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}", end="")
        
        rows = fetch_historical_chunk(city, start_unix, end_unix)
        all_rows.extend(rows)
        print(f"  ({len(rows)} records)")
        
        current = chunk_end
        
        # Small delay to respect API rate limits (60 calls/min)
        time.sleep(1)
    
    return all_rows


if __name__ == "__main__":
    print("=" * 60)
    print("  AQI HISTORICAL BACKFILL")
    print("=" * 60)
    print(f"\n  Date range: {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Cities: {', '.join(c['name'] for c in CITIES)}")
    print(f"  Estimated API calls: {len(CITIES) * ((END_DATE - START_DATE).days // CHUNK_DAYS + 1)}")
    print(f"  Estimated time: ~{len(CITIES) * ((END_DATE - START_DATE).days // CHUNK_DAYS + 1)} seconds\n")
    
    #  Fetch historical data for all cities 
    all_data = []
    
    for i, city in enumerate(CITIES):
        print(f"\n[{i+1}/{len(CITIES)}] Fetching historical data for {city['name']}...")
        rows = backfill_city(city, START_DATE, END_DATE)
        all_data.extend(rows)
        print(f"  ✓ {city['name']}: {len(rows)} total records fetched")
    
    if len(all_data) == 0:
        print("\n✗ No data fetched! Check your API key.")
        exit(1)
    
    #  Create DataFrame and engineer features 
    print(f"\n[Processing] Creating DataFrame with {len(all_data)} rows...")
    df = pd.DataFrame(all_data)
    df = engineer_features(df)
    
    # Remove any duplicates (same city + same timestamp)
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="last")
    print(f"  After deduplication: {len(df)} rows")
    
    #  Show summary statistics 
    print(f"\n{'─' * 60}")
    print("  DATA SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Total rows:    {len(df)}")
    print(f"  Cities:        {df['city'].nunique()}")
    print(f"  Date range:    {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  AQI range:     {df['aqi'].min()} - {df['aqi'].max()} (OpenWeather 1-5)")
    print(f"  EPA AQI range: {df['epa_aqi'].min()} - {df['epa_aqi'].max()} (EPA 0-500)")
    print(f"  PM2.5 range:   {df['pm2_5'].min():.1f} - {df['pm2_5'].max():.1f}")
    print(f"\n  Per-city breakdown:")
    for city_name in df['city'].unique():
        city_df = df[df['city'] == city_name]
        print(f"    {city_name:12s}: {len(city_df):6d} rows | "
              f"Avg EPA AQI: {city_df['epa_aqi'].mean():.0f} | "
              f"Avg PM2.5: {city_df['pm2_5'].mean():.1f}")
    
    #  Save to local Parquet file 
    filepath = os.path.join(FEATURE_STORE_DIR, "aqi_historical.parquet")
    df.to_parquet(filepath, index=False)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n  ✓ Saved to: {filepath}")
    print(f"  ✓ File size: {file_size_mb:.1f} MB")
    
    #  Print next steps 
    print(f"\n{'=' * 60}")
    print("  BACKFILL COMPLETE!")
    print(f"{'=' * 60}")
    print(f"""
  Next step: Upload this data to Hopsworks Feature Store.
  
  1. Go to Hopsworks UI → Files → Upload 'aqi_historical.parquet'
     (from: {filepath})
  
  2. Then run the upload script from the Hopsworks Terminal.
     (We will create this script next)
""")
