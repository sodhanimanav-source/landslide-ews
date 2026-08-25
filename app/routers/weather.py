from fastapi import APIRouter, HTTPException
import requests

router = APIRouter(prefix="/api/weather", tags=["Weather"])

@router.get("/{lat}/{lng}")
def get_live_weather(lat: float, lng: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=precipitation,rain&hourly=precipitation_probability"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Weather service unavailable")
    
    data = response.json()
    return {
        "latitude": lat,
        "longitude": lng,
        "current_rainfall_mm": data.get("current", {}).get("rain", 0.0),
        "elevation": data.get("elevation", 0.0)
    }