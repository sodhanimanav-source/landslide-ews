from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Alert
from pydantic import BaseModel

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

class AlertCreate(BaseModel):
    location_name: str
    severity: str
    message: str

@router.post("/")
def create_alert(alert_data: AlertCreate, db: Session = Depends(get_db)):
    new_alert = Alert(
        location_name=alert_data.location_name,
        severity=alert_data.severity,
        message=alert_data.message
    )
    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)
    return {"message": "Alert issued successfully", "alert": new_alert}

@router.get("/")
def get_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).order_by(Alert.created_at.desc()).all()