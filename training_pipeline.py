import os
import warnings
import pandas as pd
import numpy as np
import hopsworks
import joblib
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import calculate_epa_aqi

# Silence TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# ─────────────────────────────────────────────
# POLLUTANTS TO CREATE LAGGED FEATURES FOR
# ─────────────────────────────────────────────
POLLUTANTS = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3', 'no']


def create_features_and_target(df):
    """
    Engineer a rich feature set from the raw data.
    
    Target: raw PM2.5 concentration 24 hours from now (smooth, continuous).
    
    Features engineered:
      - Lagged values (1h, 6h, 12h, 24h) for every pollutant
      - Rolling mean & std (24h window) for key pollutants
      - Rate of change (current vs 24h ago) for key pollutants
      - Cyclical encoding of hour and month (sin/cos)
      - One-hot encoded city
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(by=['city', 'timestamp']).copy()
    
    # ── TARGET: raw PM2.5 concentration 24 hours from now ──
    df['target_pm25'] = df.groupby('city')['pm2_5'].shift(-24)
    
    # ── LAGGED FEATURES for every pollutant ──
    lag_hours = [1, 6, 12, 24]
    for pol in POLLUTANTS:
        for lag in lag_hours:
            df[f'{pol}_lag_{lag}h'] = df.groupby('city')[pol].shift(lag)
    
    # ── ROLLING STATISTICS (24h window) for key pollutants ──
    rolling_pollutants = ['pm2_5', 'pm10', 'co', 'o3']
    for pol in rolling_pollutants:
        grouped = df.groupby('city')[pol]
        # shift(1) to avoid data leakage — only use strictly past data
        df[f'{pol}_roll_mean_24h'] = grouped.transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
        )
        df[f'{pol}_roll_std_24h'] = grouped.transform(
            lambda x: x.shift(1).rolling(window=24, min_periods=1).std()
        )
    
    # ── RATE OF CHANGE (current value vs 24h ago) ──
    for pol in rolling_pollutants:
        lag_col = f'{pol}_lag_24h'
        # Avoid division by zero
        df[f'{pol}_roc_24h'] = (df[pol] - df[lag_col]) / (df[lag_col].abs() + 1e-6)
    
    # ── CYCLICAL TIME ENCODING ──
    # Encode hour and month as sin/cos so the model knows 23:00 is close to 00:00
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ── DROP INCOMPLETE ROWS ──
    # Need at least 24h of past data for lag features, and 24h of future for target
    required_cols = ['target_pm25'] + [f'{pol}_lag_24h' for pol in POLLUTANTS]
    df = df.dropna(subset=required_cols).copy()
    
    # Fill any remaining NaN in rolling std columns (can be NaN if all values identical)
    df = df.fillna(0)
    
    return df


def build_feature_columns(df):
    """
    Build the list of feature columns from the engineered DataFrame.
    Returns (X, y, feature_names).
    """
    # Raw pollutant features
    feature_cols = list(POLLUTANTS)
    
    # Lagged features
    lag_hours = [1, 6, 12, 24]
    for pol in POLLUTANTS:
        for lag in lag_hours:
            feature_cols.append(f'{pol}_lag_{lag}h')
    
    # Rolling stats
    rolling_pollutants = ['pm2_5', 'pm10', 'co', 'o3']
    for pol in rolling_pollutants:
        feature_cols.append(f'{pol}_roll_mean_24h')
        feature_cols.append(f'{pol}_roll_std_24h')
    
    # Rate of change
    for pol in rolling_pollutants:
        feature_cols.append(f'{pol}_roc_24h')
    
    # Cyclical time
    feature_cols += ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    
    # EPA AQI and time features (still useful context)
    feature_cols += ['epa_aqi', 'day_of_week']
    
    # One-hot encode city
    df_encoded = pd.get_dummies(df, columns=['city'], drop_first=False)
    city_cols = [col for col in df_encoded.columns if col.startswith('city_')]
    feature_cols.extend(city_cols)
    
    X = df_encoded[feature_cols]
    y = df_encoded['target_pm25']
    
    return X, y, feature_cols


def build_tf_model(input_dim):
    """Builds a deeper Neural Network with batch normalization and learning rate scheduling."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def evaluate_model(name, y_true, y_pred):
    """Evaluate model on raw PM2.5 predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"  PM2.5 RMSE: {rmse:.2f} µg/m³")
    print(f"  PM2.5 MAE:  {mae:.2f} µg/m³")
    print(f"  PM2.5 R²:   {r2:.4f}")
    return rmse, mae, r2


def pm25_to_epa_aqi(pm25_value):
    """Convert a single PM2.5 prediction to EPA AQI using the official breakpoints."""
    # Use our existing utility — PM2.5 is the dominant pollutant in Sindh,
    # so we use it as the primary driver and set others to 0.
    aqi, _ = calculate_epa_aqi(pm2_5=pm25_value, pm10=0, co=0, no2=0, o3=0, so2=0)
    return aqi


def main():
    print("=" * 60)
    print("  AQI MODEL TRAINING PIPELINE (v2 — PM2.5 Target)")
    print("=" * 60)
    
    # 1. Fetch historical data
    print("\n[Step 1/6] Fetching historical data from Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    try:
        fg = fs.get_feature_group("aqi_features", version=1)
        try:
            feature_view = fs.get_feature_view("aqi_model_fv", version=1)
            print("  ✓ Found existing Feature View 'aqi_model_fv'.")
        except:
            print("  Creating new Feature View 'aqi_model_fv'...")
            feature_view = fs.create_feature_view(
                name="aqi_model_fv",
                version=1,
                query=fg.select_all()
            )
            
        print("  Downloading data from Feature View (this may take a minute)...")
        df, _ = feature_view.get_training_data(1)
        print(f"  ✓ Fetched {len(df)} rows from Hopsworks.")
    except Exception as e:
        print(f"  ✗ Hopsworks download failed (common on external free tier): {e}")
        print("  ! Falling back to local historical parquet file...")
        df = pd.read_parquet("feature_store/aqi_historical.parquet")
        print(f"  ✓ Loaded {len(df)} rows from local backup.")

    # 2. Feature Engineering
    print("\n[Step 2/6] Engineering features and creating PM2.5 target...")
    df_engineered = create_features_and_target(df)
    X, y, feature_names = build_feature_columns(df_engineered)
    
    print(f"  Total features: {len(feature_names)}")
    print(f"  Total samples:  {len(X)}")
    
    # Chronological train-test split (last 20% of time as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features for models that require it (Ridge, NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Keep as DataFrames for SHAP
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"  Training set: {len(X_train)} rows")
    print(f"  Testing set:  {len(X_test)} rows")

    # 3. Model Training
    print("\n[Step 3/6] Training Models (Ridge, Random Forest, XGBoost, TensorFlow)...")
    models = {}
    
    # A) Ridge Regression (Baseline)
    print("  Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    models['Ridge'] = ridge
    
    # B) Random Forest (300 trees, tuned)
    print("  Training Random Forest (300 trees)...")
    rf = RandomForestRegressor(
        n_estimators=300, 
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1, 
        random_state=42
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    # C) XGBoost (500 trees, regularized, with early stopping)
    print("  Training XGBoost (500 trees with early stopping)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric='rmse'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    models['XGBoost'] = xgb_model
    
    # D) TensorFlow Neural Network (deeper, with callbacks)
    print("  Training TensorFlow NN (deeper architecture)...")
    tf_model = build_tf_model(X_train_scaled.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    tf_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=64,
        verbose=0,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr]
    )
    models['TensorFlow'] = tf_model

    # 4. Model Evaluation
    print("\n[Step 4/6] Evaluating Models...")
    print("\n  ── PM2.5 Prediction Accuracy (Primary Metric) ──")
    best_model_name = None
    best_rmse = float('inf')
    best_r2 = -float('inf')
    all_preds = {}
    
    for name, model in models.items():
        if name in ['Ridge', 'TensorFlow']:
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test)
        
        # Flatten predictions if needed
        preds = np.array(preds).flatten()
        # Clip negative predictions to 0 (PM2.5 can't be negative)
        preds = np.clip(preds, 0, None)
        
        all_preds[name] = preds
        rmse, mae, r2 = evaluate_model(name, y_test, preds)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_model_name = name

    print(f"\n  ★ BEST MODEL: {best_model_name} (R²: {best_r2:.4f}, RMSE: {best_rmse:.2f})")
    best_model = models[best_model_name]
    
    # Also show the EPA AQI equivalent accuracy for the best model
    print("\n  ── EPA AQI Equivalent (for reference) ──")
    best_preds = all_preds[best_model_name]
    pred_aqi = np.array([pm25_to_epa_aqi(p) for p in best_preds])
    true_aqi = np.array([pm25_to_epa_aqi(t) for t in y_test.values])
    aqi_rmse = np.sqrt(mean_squared_error(true_aqi, pred_aqi))
    aqi_mae = mean_absolute_error(true_aqi, pred_aqi)
    aqi_r2 = r2_score(true_aqi, pred_aqi)
    print(f"  EPA AQI RMSE: {aqi_rmse:.2f}")
    print(f"  EPA AQI MAE:  {aqi_mae:.2f}")
    print(f"  EPA AQI R²:   {aqi_r2:.4f}")

    # 5. SHAP Feature Importance
    print("\n[Step 5/6] Generating SHAP Feature Importance on Best Model...")
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    try:
        is_scaled = best_model_name in ['Ridge', 'TensorFlow']
        background_data = X_train_scaled if is_scaled else X_train
        background = shap.sample(background_data, 100)
        
        if best_model_name in ['RandomForest', 'XGBoost']:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(background)
            shap.summary_plot(shap_values, background, show=False)
        else:
            if best_model_name == 'Ridge':
                explainer = shap.LinearExplainer(best_model, background)
            else:
                explainer = shap.KernelExplainer(best_model.predict, background)
            
            shap_values = explainer.shap_values(background)
            shap.summary_plot(shap_values, background, show=False)
            
        plt.tight_layout()
        plt.savefig('images/shap_summary.png', dpi=150)
        print("  ✓ SHAP summary plot saved to 'images/shap_summary.png'")
    except Exception as e:
        print(f"  ✗ SHAP generation failed: {e}")

    # 5.5 Save ALL models locally
    print("\n[Step 5.5/6] Saving ALL models locally...")
    local_archive_dir = "models_archive"
    os.makedirs(local_archive_dir, exist_ok=True)
    
    for model_key, trained_model in models.items():
        if model_key == 'TensorFlow':
            trained_model.save(os.path.join(local_archive_dir, f"{model_key.lower()}.keras"))
        else:
            joblib.dump(trained_model, os.path.join(local_archive_dir, f"{model_key.lower()}.pkl"))
    
    # Save the scaler and feature names for inference
    joblib.dump(scaler, os.path.join(local_archive_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(local_archive_dir, "feature_names.pkl"))
    print(f"  ✓ All {len(models)} models saved locally in '{local_archive_dir}/'")

    # 6. Save and Register BEST model in Hopsworks
    print("\n[Step 6/6] Saving BEST model to Hopsworks Model Registry...")
    model_dir = "aqi_model_dir"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model locally first (always succeeds)
    if best_model_name == 'TensorFlow':
        best_model.save(os.path.join(model_dir, "aqi_model.keras"))
    else:
        joblib.dump(best_model, os.path.join(model_dir, "aqi_model.pkl"))
        
    # Save the scaler and feature names (needed for inference pipeline)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
    print(f"  ✓ Best model ({best_model_name}) saved locally in '{model_dir}/'")
    
    # Try to upload to Hopsworks (may fail on free tier)
    try:
        mr = project.get_model_registry()
        aqi_hopsworks_model = mr.python.create_model(
            name="aqi_prediction_model",
            metrics={"pm25_rmse": best_rmse, "pm25_r2": best_r2},
            description=f"Predicts PM2.5 for next 24h (converted to EPA AQI at display). Best: {best_model_name}"
        )
        aqi_hopsworks_model.save(model_dir)
        print(f"  ✓ Model successfully registered in Hopsworks Model Registry!")
    except Exception as e:
        print(f"  ✗ Hopsworks upload failed (free tier limit): {e}")
        print(f"  ! Model is safely saved locally. GitHub Actions will upload it later.")
    
    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETE!")
    print(f"  Best Model: {best_model_name}")
    print(f"  PM2.5 R²:   {best_r2:.4f}")
    print(f"  PM2.5 RMSE: {best_rmse:.2f} µg/m³")
    print("=" * 60)

if __name__ == "__main__":
    main()
