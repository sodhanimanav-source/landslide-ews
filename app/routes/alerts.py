from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Alert
from pydantic import BaseModel

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

class AlertCreate(BaseModel):
    # The Alert table stores `corridor_name` and requires `risk_score`.
    # `location_name` is kept as an alias so existing clients keep working.
    corridor_name: Optional[str] = None
    location_name: Optional[str] = None
    severity: str
    message: Optional[str] = None
    risk_score: int = 0

    def resolved_corridor(self) -> str:
        name = self.corridor_name or self.location_name
        if not name:
            raise ValueError("corridor_name (or location_name) is required")
        return name


@router.post("/")
def create_alert(alert_data: AlertCreate, db: Session = Depends(get_db)):
    try:
        corridor = alert_data.resolved_corridor()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    new_alert = Alert(
        corridor_name=corridor,
        severity=alert_data.severity,
        risk_score=alert_data.risk_score,
        message=alert_data.message,
    )
    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)
    return {"message": "Alert issued successfully", "alert": new_alert}

@router.get("/")
def get_alerts(db: Session = Depends(get_db)):
    return db.query(Alert).order_by(Alert.created_at.desc()).all()