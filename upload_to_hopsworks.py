
import pandas as pd
import hopsworks

print('Connecting...')
project = hopsworks.login()
fs = project.get_feature_store()

print('Reading data...')
df = pd.read_parquet('/hopsfs/Users/shaikhka/aqi_historical.parquet')
print(f'Loaded {len(df)} rows, {len(df.columns)} columns')

print('Creating Feature Group...')
fg = fs.get_or_create_feature_group(
    name='aqi_features', version=1,
    primary_key=['city', 'timestamp'],
    event_time='timestamp',
    description='AQI data for 5 cities in Sindh Pakistan with EPA AQI',
    online_enabled=False
)

print('Inserting data in batches...')
batch_size = 5000
total = len(df)
for i in range(0, total, batch_size):
    batch = df.iloc[i:i+batch_size]
    fg.insert(batch)
    print(f'  Inserted rows {i} - {min(i+batch_size, total)} of {total}')

print('DONE! All historical data uploaded!')
