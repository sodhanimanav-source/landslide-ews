import joblib
import os
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

def predict_landslide_risk(slope: float, elevation: float, rainfall: float, dist_road: float = 100.0) -> dict:
    features = np.array([[slope, elevation, rainfall, dist_road]])
    prob = model.predict_proba(features)[0][1] # Probability of landslide
    
    score = int(prob * 100)
    if score >= 70:
        severity = "High"
    elif score >= 40:
        severity = "Medium"
    else:
        severity = "Low"
        
    return {
        "risk_score": score,
        "severity": severity
    }