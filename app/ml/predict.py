import os
import joblib
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), "model.joblib")

if not os.path.exists(model_path):
    from app.ml.train import train_and_export_model
    train_and_export_model()

model = joblib.load(model_path)

def predict_landslide_risk(
    slope: float,
    elevation: float,
    rain_24h: float,
    rain_7d_antecedent: float = 40.0,
    soil_moisture: float = 0.45,
    dist_road: float = 150.0
) -> dict:
    features = np.array([[slope, elevation, rain_24h, rain_7d_antecedent, soil_moisture, dist_road]])
    prob = model.predict_proba(features)[0][1]
    score = int(round(prob * 100))
    
    if score >= 70:
        severity = "High"
        action = "Issue immediate Red Alert; evacuate vulnerable road stretches."
    elif score >= 40:
        severity = "Medium"
        action = "Issue Amber Advisory; monitor slope displacement and drainage."
    else:
        severity = "Low"
        action = "Green Zone; normal monitoring."
        
    return {
        "risk_score": score,
        "severity": severity,
        "recommended_action": action,
        "factors": {
            "slope_deg": slope,
            "elevation_m": elevation,
            "rain_24h_mm": rain_24h,
            "rain_7d_api_mm": rain_7d_antecedent,
            "soil_saturation": soil_moisture
        }
    }