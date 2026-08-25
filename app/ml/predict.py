import joblib
import os
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), "model.joblib")

# Agar model file nahi hai, toh auto-train karke generate karein
if not os.path.exists(model_path):
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    np.random.seed(42)
    n = 500
    slope = np.random.uniform(5, 60, n)
    elevation = np.random.uniform(200, 3000, n)
    rainfall = np.random.uniform(0, 250, n)
    dist_road = np.random.uniform(10, 500, n)
    label = ((slope * 0.4) + (rainfall * 0.5) - (dist_road * 0.05) > 45).astype(int)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(np.column_stack([slope, elevation, rainfall, dist_road]), label)
    joblib.dump(clf, model_path)

model = joblib.load(model_path)

def predict_landslide_risk(slope: float, elevation: float, rainfall: float, dist_road: float = 100.0) -> dict:
    features = np.array([[slope, elevation, rainfall, dist_road]])
    prob = model.predict_proba(features)[0][1]
    score = int(prob * 100)
    
    severity = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
    return {"risk_score": score, "severity": severity}