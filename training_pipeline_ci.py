

import os
import sys
import warnings
import pandas as pd
import numpy as np
import hopsworks
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import calculate_epa_aqi

# Silence warnings for cleaner CI logs
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# ─────────────────────────────────────────────
# POLLUTANTS AND CONFIGURATION
# ─────────────────────────────────────────────
POLLUTANTS = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3', 'no']
ROLLING_POLLUTANTS = ['pm2_5', 'pm10', 'co', 'o3']
LAG_HOURS = [1, 6, 12, 24]

# RandomForest hyperparameters (tuned from initial comparison)
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'min_samples_leaf': 5,
    'n_jobs': -1,
    'random_state': 42
}


def create_features_and_target(df):
    """
    Engineer features and create PM2.5 target (24h ahead).
    Same logic as training_pipeline.py to ensure consistency.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(by=['city', 'timestamp']).copy()

    # Target: raw PM2.5 concentration 24 hours from now
    df['target_pm25'] = df.groupby('city')['pm2_5'].shift(-24)

    # Lagged features for every pollutant
    for pol in POLLUTANTS:
        for lag in LAG_HOURS:
            df[f'{pol}_lag_{lag}h'] = df.groupby('city')[pol].shift(lag)

    # Rolling statistics (24h window) for key pollutants
    for pol in ROLLING_POLLUTANTS:
        grouped = df.groupby('city')[pol]
        df[f'{pol}_roll_mean_24h'] = grouped.transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
        )
        df[f'{pol}_roll_std_24h'] = grouped.transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1).std()
        )

    # Rate of change (current value vs 24h ago)
    for pol in ROLLING_POLLUTANTS:
        lag_col = f'{pol}_lag_24h'
        df[f'{pol}_roc_24h'] = (df[pol] - df[lag_col]) / (df[lag_col].abs() + 1e-6)

    # Cyclical time encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Drop incomplete rows
    required_cols = ['target_pm25'] + [f'{pol}_lag_24h' for pol in POLLUTANTS]
    df = df.dropna(subset=required_cols).copy()
    df = df.fillna(0)

    return df


def build_feature_columns(df):
    """Build feature columns and return (X, y, feature_names)."""
    feature_cols = list(POLLUTANTS)

    # Lagged features
    for pol in POLLUTANTS:
        for lag in LAG_HOURS:
            feature_cols.append(f'{pol}_lag_{lag}h')

    # Rolling stats
    for pol in ROLLING_POLLUTANTS:
        feature_cols.append(f'{pol}_roll_mean_24h')
        feature_cols.append(f'{pol}_roll_std_24h')

    # Rate of change
    for pol in ROLLING_POLLUTANTS:
        feature_cols.append(f'{pol}_roc_24h')

    # Cyclical time
    feature_cols += ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']

    # EPA AQI and day of week
    feature_cols += ['epa_aqi', 'day_of_week']

    # One-hot encode city
    df_encoded = pd.get_dummies(df, columns=['city'], drop_first=False)
    city_cols = [col for col in df_encoded.columns if col.startswith('city_')]
    feature_cols.extend(city_cols)

    X = df_encoded[feature_cols]
    y = df_encoded['target_pm25']

    return X, y, feature_cols


def pm25_to_epa_aqi(pm25_value):
    """Convert PM2.5 prediction to EPA AQI using official breakpoints."""
    aqi, _ = calculate_epa_aqi(pm2_5=pm25_value, pm10=0, co=0, no2=0, o3=0, so2=0)
    return aqi



def main():
    print("=" * 60)
    print("  AQI CI/CD TRAINING PIPELINE (RandomForest)")
    print("=" * 60)

    # ── Step 1: Fetch data ──
    print("\n[Step 1/5] Fetching data from Hopsworks & local repository...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    df = None
    # 1A. Try downloading training data via Feature View
    try:
        fg = fs.get_feature_group("aqi_features", version=1)
        try:
            feature_view = fs.get_feature_view("aqi_model_fv", version=1)
            print("  ✓ Found existing Feature View 'aqi_model_fv'.")
        except Exception:
            print("  Creating new Feature View 'aqi_model_fv'...")
            feature_view = fs.create_feature_view(
                name="aqi_model_fv",
                version=1,
                query=fg.select_all()
            )

        print("  Downloading data from Feature View...")
        fv_df, _ = feature_view.get_training_data(1)
        if fv_df is not None and len(fv_df) > 1000:
            df = fv_df
            print(f"  ✓ Fetched {len(df)} rows from Hopsworks Feature View.")
    except Exception as e:
        print(f"  ⚠️ Hopsworks Feature View download notice: {e}")

    # 1B. If Feature View didn't return full data, load the bundled historical dataset
    if df is None or len(df) < 1000:
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "feature_store", "aqi_historical.parquet"
        )
        if os.path.exists(local_path):
            df = pd.read_parquet(local_path)
            print(f"  ✓ Loaded {len(df)} historical rows from repository backup.")

        # 1C. Also download the latest live features uploaded by feature pipeline
        try:
            dataset_api = project.get_dataset_api()
            latest_path = dataset_api.download("Resources/latest_features.parquet", overwrite=True)
            if os.path.exists(latest_path):
                latest_df = pd.read_parquet(latest_path)
                print(f"  ✓ Fetched {len(latest_df)} latest live rows from Hopsworks Cloud.")
                df = pd.concat([df, latest_df], ignore_index=True)
                df = df.drop_duplicates(subset=["city", "timestamp"], keep="last")
        except Exception as e:
            print(f"  (Latest cloud features check: {e})")

    if df is None or len(df) < 100:
        print(f"  ✗ Not enough data to train. Exiting.")
        sys.exit(1)

    print(f"  ✓ Total dataset for training: {len(df)} rows.")

    # ── Step 2: Feature Engineering ──
    print("\n[Step 2/5] Engineering features and creating PM2.5 target...")
    df_engineered = create_features_and_target(df)
    X, y, feature_names = build_feature_columns(df_engineered)

    print(f"  Total features: {len(feature_names)}")
    print(f"  Total samples:  {len(X)}")

    if len(X) < 50:
        print(f"  ✗ Not enough samples after feature engineering ({len(X)}). Exiting.")
        sys.exit(1)

    # Chronological train-test split (last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Training set: {len(X_train)} rows")
    print(f"  Testing set:  {len(X_test)} rows")

    # ── Step 3: Train RandomForest ──
    print("\n[Step 3/5] Training RandomForest (300 trees)...")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # Evaluate
    preds = rf.predict(X_test)
    preds = np.clip(preds, 0, None)  # PM2.5 can't be negative

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n  ── RandomForest Results ──")
    print(f"  PM2.5 RMSE: {rmse:.2f} µg/m³")
    print(f"  PM2.5 MAE:  {mae:.2f} µg/m³")
    print(f"  PM2.5 R²:   {r2:.4f}")

    # EPA AQI equivalent metrics
    pred_aqi = np.array([pm25_to_epa_aqi(p) for p in preds])
    true_aqi = np.array([pm25_to_epa_aqi(t) for t in y_test.values])
    aqi_rmse = np.sqrt(mean_squared_error(true_aqi, pred_aqi))
    aqi_r2 = r2_score(true_aqi, pred_aqi)
    print(f"\n  ── EPA AQI Equivalent ──")
    print(f"  EPA AQI RMSE: {aqi_rmse:.2f}")
    print(f"  EPA AQI R²:   {aqi_r2:.4f}")

    # ── Step 4: SHAP Feature Importance ──
    print("\n[Step 4/5] Generating SHAP Feature Importance...")
    os.makedirs('images', exist_ok=True)

    try:
        explainer = shap.TreeExplainer(rf)
        background = shap.sample(X_train, 100)
        shap_values = explainer.shap_values(background)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, background, show=False)
        plt.tight_layout()
        plt.savefig('images/shap_summary.png', dpi=150)
        plt.close()
        print("  ✓ SHAP summary plot saved to 'images/shap_summary.png'")
    except Exception as e:
        print(f"  ✗ SHAP generation failed: {e}")

    # ── Step 5: Save model and upload to Hopsworks ──
    print("\n[Step 5/5] Saving model and uploading to Hopsworks...")
    model_dir = "aqi_model_dir"
    os.makedirs(model_dir, exist_ok=True)

    # Save model artifacts locally
    joblib.dump(rf, os.path.join(model_dir, "aqi_model.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
    # Save a scaler for compatibility (even though RF doesn't need scaling,
    # the inference pipeline may use it for other model types)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"  ✓ Model saved locally in '{model_dir}/'")

    # Always upload the latest model to Hopsworks.
    # Why? We retrain daily on NEW data (more hours of pollution readings).
    # The latest model reflects current pollution patterns — that's the whole
    # point of daily retraining. Comparison only matters when switching model
    # architectures (RF vs XGBoost), not when retraining the same one on more data.
    try:
        mr = project.get_model_registry()
        aqi_model = mr.python.create_model(
            name="aqi_prediction_model",
            metrics={"pm25_rmse": rmse, "pm25_r2": r2, "pm25_mae": mae},
            description=f"RandomForest predicting PM2.5 24h ahead. RMSE={rmse:.2f}, R²={r2:.4f}. Trained on {len(X)} samples."
        )
        aqi_model.save(model_dir)
        print("  ✓ Model registered in Hopsworks Model Registry!")
    except Exception as e:
        print(f"  ✗ Hopsworks upload failed: {e}")
        print("  ! Model is saved locally. Will retry on next run.")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  CI/CD TRAINING PIPELINE COMPLETE!")
    print(f"  Model:      RandomForest ({RF_PARAMS['n_estimators']} trees)")
    print(f"  PM2.5 RMSE: {rmse:.2f} µg/m³")
    print(f"  PM2.5 R²:   {r2:.4f}")
    print(f"  Trained on: {len(X)} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
