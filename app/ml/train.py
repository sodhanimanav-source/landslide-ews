import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def generate_geological_dataset(n_samples=2500):
    np.random.seed(42)
    
    # 1. Slope (in degrees: 0 to 65) - Steep slopes (>25 deg) have higher shear stress
    slope = np.random.beta(a=2, b=3, size=n_samples) * 60
    
    # 2. Elevation (meters above sea level: 200m to 3500m in NE Region)
    elevation = np.random.uniform(200, 3500, size=n_samples)
    
    # 3. 24-Hour Immediate Rainfall (mm: 0 to 250mm)
    rain_24h = np.random.exponential(scale=35, size=n_samples)
    rain_24h = np.clip(rain_24h, 0, 300)
    
    # 4. 7-Day Antecedent Precipitation Index (API in mm: Soil saturation decay index)
    # API = P0 + 0.85*P1 + (0.85^2)*P2 ...
    rain_7d_antecedent = np.random.exponential(scale=110, size=n_samples)
    rain_7d_antecedent = np.clip(rain_7d_antecedent, 0, 500)
    
    # 5. Soil Moisture Index (0.0 to 1.0 volumetric saturation)
    soil_moisture = np.clip((rain_7d_antecedent / 400.0) + np.random.normal(0.1, 0.05, n_samples), 0.1, 0.95)
    
    # 6. Distance to Road / Cut Slope (meters: 5m to 1000m)
    dist_road = np.random.exponential(scale=200, size=n_samples)
    dist_road = np.clip(dist_road, 5, 1000)
    
    # Physics-informed Empirical Geotechnical Susceptibility Equation
    # Safety Factor approximation: Lower FS = Landslide trigger
    critical_shear = (
        (slope / 45.0) * 0.35 +
        (rain_24h / 120.0) * 0.30 +
        (rain_7d_antecedent / 250.0) * 0.20 +
        (soil_moisture) * 0.15 -
        (dist_road / 800.0) * 0.10
    )
    
    noise = np.random.normal(0, 0.05, n_samples)
    risk_metric = critical_shear + noise
    
    # Threshold for binary event trigger (1 = Landslide Incident, 0 = Stable)
    labels = (risk_metric > 0.48).astype(int)
    
    df = pd.DataFrame({
        "slope": np.round(slope, 2),
        "elevation": np.round(elevation, 1),
        "rain_24h": np.round(rain_24h, 2),
        "rain_7d_antecedent": np.round(rain_7d_antecedent, 2),
        "soil_moisture": np.round(soil_moisture, 3),
        "dist_road": np.round(dist_road, 1),
        "landslide_occurred": labels
    })
    
    return df

def train_and_export_model():
    print("Ingesting Geological & Hydrological Feature Vectors...")
    df = generate_geological_dataset(n_samples=3000)
    
    X = df[["slope", "elevation", "rain_24h", "rain_7d_antecedent", "soil_moisture", "dist_road"]]
    y = df["landslide_occurred"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    roc = roc_auc_score(y_test, y_prob)
    print("\n--- Model Evaluation ---")
    print(f"ROC-AUC Score: {roc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stable", "Hazard Triggered"]))
    
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    train_and_export_model()