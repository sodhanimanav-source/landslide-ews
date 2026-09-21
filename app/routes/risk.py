from fastapi import APIRouter
import requests
from app.ml.predict import predict_landslide_risk

router = APIRouter(prefix="/api/risk-zones", tags=["Risk Zones"])

DEMO_ZONES = [
    {"id": 1, "name": "Aizawl North", "state": "Mizoram", "lat": 23.7271, "lng": 92.7176, "slope": 38.0},
    {"id": 2, "name": "Gangtok East", "state": "Sikkim", "lat": 27.3389, "lng": 88.6065, "slope": 42.0},
    {"id": 3, "name": "Kohima Ridge", "state": "Nagaland", "lat": 25.6751, "lng": 94.1086, "slope": 32.0},
    {"id": 4, "name": "Shillong Peak Road", "state": "Meghalaya", "lat": 25.5788, "lng": 91.8933, "slope": 20.0},
    {"id": 5, "name": "Itanagar Hills", "state": "Arunachal Pradesh", "lat": 27.0844, "lng": 93.6053, "slope": 35.0}
]

@router.get("/")
def get_risk_zones():
    computed_zones = []
    for zone in DEMO_ZONES:
        # 1. Fetch live rainfall from Open-Meteo
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={zone['lat']}&longitude={zone['lng']}&current=rain"
            res = requests.get(url, timeout=4).json()
            rainfall = res.get("current", {}).get("rain", 0.0)
            elevation = res.get("elevation", 1200.0)
        except Exception:
            rainfall = 25.0
            elevation = 1200.0

        # 2. Run inference using trained ML model
        # NOTE: the parameter is `rain_24h`, not `rainfall`. Passing
        # `rainfall=` raised TypeError on every request to this endpoint.
        prediction = predict_landslide_risk(
            slope=zone["slope"],
            elevation=elevation,
            rain_24h=rainfall,
        )

        computed_zones.append({
            "id": zone["id"],
            "name": zone["name"],
            "state": zone["state"],
            "lat": zone["lat"],
            "lng": zone["lng"],
            "risk_score": prediction["risk_score"],
            "severity": prediction["severity"],
            "live_rainfall_mm": rainfall,
            "elevation_m": elevation
        })
    return computed_zones