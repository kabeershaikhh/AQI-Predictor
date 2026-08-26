import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # Create directory for images
    os.makedirs('images', exist_ok=True)
    
    try:
        df = pd.read_parquet('feature_store/aqi_historical.parquet')
        print(f"Loaded {len(df)} rows from historical data.")
    except Exception as e:
        print("Could not load historical data. Make sure backfill.py ran successfully.")
        return

    # 1. EPA AQI Distribution
    print("Generating AQI Distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['epa_aqi'], bins=50, kde=True, color='purple')
    plt.title('Distribution of EPA AQI (0-500 scale)')
    plt.xlabel('EPA AQI')
    plt.ylabel('Frequency')
    plt.savefig('images/eda_aqi_distribution.png')
    plt.close()
    
    # 2. AQI over time for each city
    print("Generating AQI Over Time plot...")
    plt.figure(figsize=(15, 7))
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Resample to daily average for a cleaner plot (the raw hourly data is too noisy)
    daily_avg = df.groupby(['city', pd.Grouper(key='timestamp_dt', freq='D')])['epa_aqi'].mean().reset_index()
    sns.lineplot(data=daily_avg, x='timestamp_dt', y='epa_aqi', hue='city', alpha=0.8)
    plt.title('Daily Average EPA AQI Over Time (Past Year)')
    plt.xlabel('Date')
    plt.ylabel('EPA AQI')
    plt.savefig('images/eda_aqi_over_time.png')
    plt.close()
    
    # 3. Correlation Heatmap
    print("Generating Feature Correlation Heatmap...")
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32']).drop(columns=['timestamp'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('images/eda_correlation_heatmap.png')
    plt.close()
    
    # 4. Average AQI by Day of Week
    print("Generating AQI by Day of Week plot...")
    plt.figure(figsize=(10, 6))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_by_day = df.groupby('day_of_week')['epa_aqi'].mean().reset_index()
    avg_by_day['day_name'] = avg_by_day['day_of_week'].apply(lambda x: days[x])
    sns.barplot(data=avg_by_day, x='day_name', y='epa_aqi', palette='viridis')
    plt.title('Average EPA AQI by Day of the Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average EPA AQI')
    plt.savefig('images/eda_aqi_by_day.png')
    plt.close()

    print("\n✓ EDA plots successfully saved in 'images/' directory!")

if __name__ == '__main__':
    run_eda()
