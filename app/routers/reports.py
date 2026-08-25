from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Report
from pydantic import BaseModel

router = APIRouter(prefix="/api/reports", tags=["Reports"])

class ReportCreate(BaseModel):
    lat: float
    lng: float
    description: str

@router.post("/")
def create_report(report_data: ReportCreate, db: Session = Depends(get_db)):
    new_report = Report(
        lat=report_data.lat,
        lng=report_data.lng,
        description=report_data.description
    )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    return {"message": "Report logged successfully", "report": new_report}

@router.get("/")
def get_reports(db: Session = Depends(get_db)):
    return db.query(Report).all()