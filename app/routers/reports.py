from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import HazardReport
from app.schemas import HazardReportCreate, HazardReportResponse
from typing import List
import math

router = APIRouter(prefix="/api/reports", tags=["Crowdsourced Reports"])

def haversine_distance(lat1, lon1, lat2, lon2):
    # Returns approximate distance in kilometers
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@router.post("/", response_model=HazardReportResponse)
def create_report(report: HazardReportCreate, db: Session = Depends(get_db)):
    # Spatial Clustering Check: If another report exists within 500m (0.5km), group them
    existing_reports = db.query(HazardReport).all()
    cluster_found = False
    
    for r in existing_reports:
        dist = haversine_distance(report.latitude, report.longitude, r.latitude, r.longitude)
        if dist < 0.5:
            cluster_found = True
            break
            
    db_report = HazardReport(**report.dict())
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

@router.get("/", response_model=List[HazardReportResponse])
def get_reports(db: Session = Depends(get_db)):
    return db.query(HazardReport).order_by(HazardReport.id.desc()).all()